"""Command line entry point.

Exit codes follow the T2 droid contract: 0 success, 1 an evaluation or review failed (or the
portfolio couldn't be fetched), 2 configuration or usage error, 3 skipped (a lock is held). Every
run writes a JSON run report.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

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
from .t2contract import Exit, RunReport, Skipped, run_lock, t212_lock
from .t212 import ReadOnlyTrading212Client, Trading212Error
from .thesis import ThesisError, discover_theses, parse_thesis, read_document

logger = logging.getLogger("t2_3po")

DROID = "t2-3po"

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


class SetupError(Exception):
    """Something every evaluation needs is missing. ``code`` is the exit code to use."""

    def __init__(self, code: Exit, message: str) -> None:
        self.code = code
        super().__init__(message)


async def _setup(
    args: argparse.Namespace, settings: Settings
) -> tuple[Model, MarketData, Limits, PortfolioSnapshot | None]:
    """Build what every evaluation needs.

    Raises:
        SetupError: With ``Exit.CONFIG`` for configuration, or ``Exit.FAILED`` if the portfolio
            couldn't be fetched.
        Skipped: If another droid held the Trading212 lock for too long.
    """
    try:
        model = build_model(settings)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SetupError(Exit.CONFIG, f"configuration error: {exc}") from exc

    market = MarketData()
    limits = Limits(
        max_sector_weight=settings.max_sector_weight,
        max_position_weight=settings.max_position_weight,
    )
    portfolio = None
    if not args.no_portfolio:
        missing_keys = settings.t212_api_key is None or settings.t212_api_secret is None
        try:
            if missing_keys:
                portfolio = await _load_portfolio(settings, market)
            else:
                # Trading212 rate limits are per account, so take turns with the other droids.
                with t212_lock():
                    portfolio = await _load_portfolio(settings, market)
        except Trading212Error as exc:
            print(f"Portfolio error: {exc}", file=sys.stderr)
            code = Exit.CONFIG if missing_keys else Exit.FAILED
            raise SetupError(code, f"portfolio error: {exc}") from exc
    return model, market, limits, portfolio


async def _evaluate(args: argparse.Namespace, settings: Settings, report: RunReport) -> Exit:
    paths = [Path(p) for p in args.theses] or discover_theses(settings.theses_dir)
    if not paths:
        print(f"No theses found in {settings.theses_dir}", file=sys.stderr)
        return report.finish(Exit.CONFIG, f"no theses found in {settings.theses_dir}")

    model, market, limits, portfolio = await _setup(args, settings)
    council = build_council(model)

    results: list[dict[str, Any]] = []
    failed: list[str] = []
    for path in paths:
        try:
            thesis = parse_thesis(path)
            context = await gather_context(thesis, market, portfolio, limits)
            result = await evaluate(context, council)
            proposal = build_proposal(context, result)
            if args.dry_run:
                print(render_report(context, result, proposal))
                results.append(
                    {
                        "thesis": str(path),
                        "ticker": proposal.ticker,
                        "status": str(proposal.thesis_status),
                    }
                )
                continue
            written, machine = write_outputs(
                context, result, proposal, settings.reports_dir, settings.proposals_dir
            )
        except _EVALUATION_ERRORS as exc:
            # No proposal is written for a failed evaluation.
            logger.error("evaluation_failed thesis=%s error=%s", path, exc)
            failed.append(str(path))
            continue
        print(f"{proposal.ticker}: {proposal.thesis_status}\n  {written}\n  {machine}")
        results.append(
            {
                "thesis": str(path),
                "ticker": proposal.ticker,
                "status": str(proposal.thesis_status),
                "report": str(written),
                "proposal": str(machine),
            }
        )
    summary = f"{len(results)} of {len(paths)} theses evaluated"
    return report.finish(
        Exit.FAILED if failed else Exit.OK,
        summary,
        {"evaluations": results, "failed": failed, "portfolio": portfolio is not None},
        errors=[f"evaluation failed: {path}" for path in failed],
    )


async def _review(args: argparse.Namespace, settings: Settings, report: RunReport) -> Exit:
    thesis_path = Path(args.thesis) if args.thesis else settings.macro_thesis_file
    stocks_path = Path(args.stocks) if args.stocks else settings.stocks_file
    try:
        macro_thesis = read_document(thesis_path)
        stocks = select_stocks(load_stocks(stocks_path), args.only)
    except (ThesisError, StocksError) as exc:
        print(f"Review error: {exc}", file=sys.stderr)
        return report.finish(Exit.CONFIG, f"review error: {exc}")

    model, market, limits, portfolio = await _setup(args, settings)
    agents = build_review_agents(model)
    print(
        f"Reviewing {len(stocks)} stocks against {thesis_path}: one model call each, "
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
        return report.finish(
            Exit.FAILED, "every review failed", {"failed": failed}, errors=["every review failed"]
        )
    overview = await overview_or_none(agents, macro_thesis, reviews, _EVALUATION_ERRORS)
    rendered = render_review(thesis_path.name, reviews, overview, failed, portfolio is not None)
    stocks_result: list[dict[str, Any]] = []
    review_path: str | None = None
    if args.dry_run:
        print(rendered)
    else:
        review_path = str(
            write_review(rendered, reviews[0].proposal.timestamp, settings.reports_dir)
        )
        print(review_path)
    for review in reviews:
        entry: dict[str, Any] = {
            "symbol": review.stock.symbol,
            "status": str(review.profile.thesis_status),
            "role": str(review.profile.role),
        }
        if not args.dry_run:
            path = write_proposal(review.proposal, settings.proposals_dir)
            verdict = f"{review.profile.thesis_status}, {review.profile.role}"
            print(f"{review.stock.symbol}: {verdict}\n  {path}")
            entry["proposal"] = str(path)
        stocks_result.append(entry)
    errors = [f"review failed: {symbol}" for symbol in failed]
    if overview is None:
        errors.append("the portfolio overview failed")
    return report.finish(
        Exit.FAILED if errors else Exit.OK,
        f"{len(reviews)} of {len(stocks)} stocks reviewed against {thesis_path.name}",
        {
            "thesis": str(thesis_path),
            "stocks": stocks_result,
            "failed": failed,
            "overview": overview is not None,
            "report": review_path,
            "portfolio": portfolio is not None,
        },
        errors=errors,
    )


def _export(args: argparse.Namespace, settings: Settings, report: RunReport) -> Exit:
    if not args.approve:
        print(
            "Refusing to export without --approve. Review the report, then pass --approve.",
            file=sys.stderr,
        )
        return report.finish(Exit.CONFIG, "refused to export without --approve")
    try:
        exported = export_approved(Path(args.proposal), settings.export_dir)
    except (OSError, ValidationError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return report.finish(Exit.FAILED, "export failed", errors=[str(exc)])
    print(exported)
    return report.finish(Exit.OK, f"exported {args.proposal}", {"export": str(exported)})


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
        help="Run the full evaluation, with real model calls, but print the report, not write it",
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
        help="Run every review, with real model calls, but print the reports, not write them",
    )

    export = commands.add_parser("export", help="Export a human-approved proposal for t2-d2")
    export.add_argument("proposal", help="Path to a proposal JSON file")
    export.add_argument("--approve", action="store_true", help="Confirm you reviewed the proposal")
    for sub in (run, review, export):
        sub.add_argument(
            "--report",
            type=Path,
            help="Write the JSON run report here (default: $T2_RUNS_DIR or var/runs)",
        )
    return parser


def _version() -> str:
    try:
        return version("t2-3po")
    except PackageNotFoundError:
        return "unknown"


async def _locked(
    command: str, args: argparse.Namespace, settings: Settings, report: RunReport
) -> Exit:
    try:
        with run_lock(DROID):
            if command == "review":
                return await _review(args, settings, report)
            return await _evaluate(args, settings, report)
    except SetupError as exc:
        return report.finish(exc.code, str(exc), errors=[str(exc)])
    except Skipped as exc:
        print(f"Skipped: {exc.reason}", file=sys.stderr)
        return report.finish(Exit.SKIPPED, exc.reason, {"skipped": "lock_busy"})


def main(argv: list[str] | None = None) -> int:
    """Run the CLI, write the run report, and return the exit code."""
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    args = _parser().parse_args(argv)
    settings = Settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s %(message)s")
    # HTTP clients log every request at INFO, and yfinance logs noisily on failed lookups.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    report = RunReport(
        droid=DROID,
        version=_version(),
        command=args.command,
        argv=list(sys.argv[1:] if argv is None else argv),
    )
    if args.command == "export":
        code = _export(args, settings, report)
    else:
        code = asyncio.run(_locked(args.command, args, settings, report))
    try:
        path = report.write(args.report)
    except OSError as exc:
        logger.error("run_report_failed error=%s", exc)
    else:
        logger.info("run_report_written path=%s status=%s", path, report.to_dict()["status"])
    return int(code)


if __name__ == "__main__":
    sys.exit(main())
