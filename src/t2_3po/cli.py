"""Command line entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from pydantic import ValidationError
from pydantic_ai.exceptions import AgentRunError, ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.models import Model

from .agents import build_council, build_model, build_review_agents, profile_vehicle
from .config import Settings
from .market import MarketData, MarketDataError
from .output import export_approved, render_report, write_outputs, write_proposal
from .pipeline import build_proposal, evaluate, gather_context
from .portfolio import Limits, PortfolioSnapshot, fetch_snapshot
from .review import (
    StockReview,
    build_vehicle_proposal,
    gather_profiles,
    overview_or_none,
    render_review,
    stock_thesis,
    write_review,
)
from .stocks import Stock, StocksError, load_stocks, select_stocks
from .t212 import ReadOnlyTrading212Client, Trading212Error
from .thesis import ThesisError, discover_theses, parse_thesis, read_document

logger = logging.getLogger("t2_3po")

MAX_CONCURRENT_REVIEWS = 5

_EVALUATION_ERRORS = (
    ThesisError,
    MarketDataError,
    AgentRunError,
    ModelAPIError,
    UnexpectedModelBehavior,
)


async def _load_portfolio(settings: Settings, market: MarketData) -> PortfolioSnapshot:
    if settings.t212_api_key is None or settings.t212_api_secret is None:
        raise Trading212Error(
            "T212_API_KEY and T212_API_SECRET must be set, or pass --no-portfolio"
        )
    async with ReadOnlyTrading212Client(
        settings.t212_api_key.get_secret_value(),
        settings.t212_api_secret.get_secret_value(),
        settings.t212_base_url,
    ) as client:
        return await fetch_snapshot(client, market)


async def _setup(
    args: argparse.Namespace, settings: Settings
) -> tuple[Model, MarketData, Limits, PortfolioSnapshot | None] | None:
    """Build what every evaluation needs, or print the problem and return ``None``."""
    try:
        model = build_model(settings)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return None

    market = MarketData()
    limits = Limits(
        max_sector_weight=settings.max_sector_weight,
        max_position_weight=settings.max_position_weight,
    )
    portfolio = None
    if not args.no_portfolio:
        try:
            portfolio = await _load_portfolio(settings, market)
        except Trading212Error as exc:
            print(f"Portfolio error: {exc}", file=sys.stderr)
            return None
    return model, market, limits, portfolio


async def _evaluate(args: argparse.Namespace, settings: Settings) -> int:
    paths = [Path(p) for p in args.theses] or discover_theses(settings.theses_dir)
    if not paths:
        print(f"No theses found in {settings.theses_dir}", file=sys.stderr)
        return 1

    setup = await _setup(args, settings)
    if setup is None:
        return 1
    model, market, limits, portfolio = setup
    council = build_council(model, settings.digest_cache_ttl)

    failures = 0
    for path in paths:
        try:
            thesis = parse_thesis(path)
            context = await gather_context(thesis, market, portfolio, limits)
            result = await evaluate(context, council)
            proposal = build_proposal(context, result)
            if args.dry_run:
                print(render_report(context, result, proposal))
                continue
            report, machine = write_outputs(
                context, result, proposal, settings.reports_dir, settings.proposals_dir
            )
        except _EVALUATION_ERRORS as exc:
            # No proposal is written for a failed evaluation.
            logger.error("evaluation_failed thesis=%s error=%s", path, exc)
            failures += 1
            continue
        print(f"{proposal.ticker}: {proposal.thesis_status}\n  {report}\n  {machine}")
    return 1 if failures else 0


async def _review(args: argparse.Namespace, settings: Settings) -> int:
    thesis_path = Path(args.thesis) if args.thesis else settings.macro_thesis_file
    stocks_path = Path(args.stocks) if args.stocks else settings.stocks_file
    try:
        macro_thesis = read_document(thesis_path)
        stocks = select_stocks(load_stocks(stocks_path), args.only)
    except (ThesisError, StocksError) as exc:
        print(f"Review error: {exc}", file=sys.stderr)
        return 1

    setup = await _setup(args, settings)
    if setup is None:
        return 1
    model, market, limits, portfolio = setup
    agents = build_review_agents(model)
    print(
        f"Reviewing {len(stocks)} stocks against {thesis_path}: one Claude call each, "
        "then one across all of them."
    )

    async def profile(stock: Stock) -> StockReview:
        context = await gather_context(
            stock_thesis(thesis_path, macro_thesis, stock), market, portfolio, limits
        )
        vehicle = await profile_vehicle(agents, context.render())
        return StockReview(stock, context, vehicle, build_vehicle_proposal(stock, context, vehicle))

    reviews, failed = await gather_profiles(
        stocks, profile, _EVALUATION_ERRORS, MAX_CONCURRENT_REVIEWS
    )
    if not reviews:
        print("Every review failed, so there is nothing to report.", file=sys.stderr)
        return 1
    overview = await overview_or_none(agents, macro_thesis, reviews, _EVALUATION_ERRORS)
    report = render_review(thesis_path.name, reviews, overview, failed, portfolio is not None)
    if args.dry_run:
        print(report)
    else:
        print(write_review(report, reviews[0].proposal.timestamp, settings.reports_dir))
        for review in reviews:
            path = write_proposal(review.proposal, settings.proposals_dir)
            verdict = f"{review.profile.thesis_status}, {review.profile.role}"
            print(f"{review.stock.symbol}: {verdict}\n  {path}")
    return 1 if failed or overview is None else 0


def _export(args: argparse.Namespace, settings: Settings) -> int:
    if not args.approve:
        print(
            "Refusing to export without --approve. Review the report, then pass --approve.",
            file=sys.stderr,
        )
        return 2
    try:
        print(export_approved(Path(args.proposal), settings.export_dir))
    except (OSError, ValidationError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="t2-3po", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("evaluate", help="Run the advisory council on investment theses")
    run.add_argument("theses", nargs="*", help="Thesis files (default: every file in THESES_DIR)")
    run.add_argument(
        "--no-portfolio",
        action="store_true",
        help="Skip Trading212; concentration is not checked and the proposal says so",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full evaluation, with paid model calls, but print the report, not files",
    )

    review = commands.add_parser(
        "review", help="Review every stock in t2-d2's stocks file against your macro thesis"
    )
    review.add_argument("--thesis", help="Macro thesis file (default: MACRO_THESIS_FILE)")
    review.add_argument("--stocks", help="t2-d2 stocks file (default: STOCKS_FILE)")
    review.add_argument(
        "--only", nargs="+", default=[], metavar="SYMBOL", help="Review only these stocks"
    )
    review.add_argument(
        "--no-portfolio",
        action="store_true",
        help="Skip Trading212; concentration is not checked and the proposals say so",
    )
    review.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every review, with paid model calls, but print the reports, not files",
    )

    export = commands.add_parser("export", help="Export a human-approved proposal for t2-d2")
    export.add_argument("proposal", help="Path to a proposal JSON file")
    export.add_argument("--approve", action="store_true", help="Confirm you reviewed the proposal")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its exit code."""
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    args = _parser().parse_args(argv)
    settings = Settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s %(message)s")
    # HTTP clients log every request at INFO, and yfinance logs noisily on failed lookups.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    if args.command == "export":
        return _export(args, settings)
    if args.command == "review":
        return asyncio.run(_review(args, settings))
    return asyncio.run(_evaluate(args, settings))


if __name__ == "__main__":
    sys.exit(main())
