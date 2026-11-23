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
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from pydantic_ai.exceptions import AgentRunError, ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.models import Model

from .advise import (
    AdviceContext,
    SuggestionStore,
    check_draft,
    finalize,
    render_advice,
    render_context,
    reviewer_prompt,
    since,
)
from .agents import (
    build_advise_agents,
    build_council,
    build_model,
    build_review_agents,
    draft_advice,
    profile_vehicle,
    review_advice,
)
from .config import Settings
from .configs import ConfigReadError, load_d2, load_k2
from .instruments import Instruments, InstrumentsError
from .market import MarketData, MarketDataError
from .models import Droid, Status
from .output import export_approved, render_report, write_new, write_outputs, write_proposal
from .pipeline import build_proposal, evaluate, gather_context
from .portfolio import Limits, PortfolioSnapshot, fetch_snapshot
from .prices import PriceHistory, StockStats, dip_days
from .review import (
    StockReview,
    build_vehicle_proposal,
    gather_profiles,
    overview_or_none,
    render_review,
    stock_thesis,
    write_review,
)
from .runlog import D2Activity, d2_activity, k2_latest, last_reviewed, load_reports
from .stocks import Stock, StocksError, due_stocks, load_stocks, select_stocks
from .t2contract import Exit, RunReport, Skipped, run_lock, runs_dir, t212_lock
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
    if args.batch is not None:
        # Spread the stocks over several days to stay inside a free tier's daily request quota.
        history = load_reports(runs_dir(), DROID, datetime.min.replace(tzinfo=UTC))
        stocks = due_stocks(stocks, last_reviewed(history), args.batch)

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


_DROIDS: dict[str, Droid] = {"d2": "t2-d2", "k2": "t2-k2"}


async def _advice_context(
    args: argparse.Namespace,
    settings: Settings,
    portfolio: PortfolioSnapshot | None,
    prices: PriceHistory,
    now: datetime,
) -> AdviceContext:
    """Gather the facts for the advice. Raises ConfigReadError or ThesisError on bad input."""
    droids: list[Droid] = [_DROIDS[args.only]] if args.only else ["t2-d2", "t2-k2"]
    notes: list[str] = []
    d2 = load_d2(settings.d2_dir) if "t2-d2" in droids else None
    k2 = load_k2(settings.k2_dir) if "t2-k2" in droids else None
    thesis = read_document(settings.macro_thesis_file)

    instruments = None
    if d2 is not None:
        try:
            instruments = Instruments.load(settings.bb8_dir)
        except InstrumentsError as exc:
            notes.append(f"New holdings can't be checked, so none are suggested: {exc}")

    stats: dict[str, StockStats | None] = {}
    days: dict[str, dict[str, int | None]] = {}
    if d2 is not None:
        symbols = {t.stock.symbol: t.stock.yahoo_symbol for t in d2.stocks}
        closes = await prices.many(s for s in symbols.values() if s)
        for symbol, yahoo in symbols.items():
            series = closes.get(yahoo) if yahoo else None
            stats[symbol] = StockStats.from_closes(series) if series else None
            if series:
                days[symbol] = {mode: dip_days(series, cfg) for mode, cfg in d2.modes.items()}
        missing = sorted(symbol for symbol, value in stats.items() if value is None)
        if missing:
            notes.append(f"No usable price history for: {', '.join(missing)}")

    start = since(now, settings.advise_lookback_days)
    activity = (
        d2_activity(load_reports(settings.d2_dir / "var" / "runs", "t2-d2", start))
        if d2 is not None
        else D2Activity()
    )
    if d2 is not None and activity.sessions == 0:
        notes.append(f"No t2-d2 run reports in the last {settings.advise_lookback_days} days.")
    k2_decisions = (
        k2_latest(load_reports(settings.k2_dir / "var" / "runs", "t2-k2", start))
        if k2 is not None
        else []
    )
    if k2 is not None and not k2_decisions:
        notes.append(f"No t2-k2 run reports in the last {settings.advise_lookback_days} days.")

    weights = None
    cash_weight = None
    if portfolio is not None:
        weights = {h.t212_ticker: h.weight for h in portfolio.holdings}
        cash_weight = portfolio.cash_weight
    else:
        notes.append("No portfolio: position weights were not considered.")

    store = SuggestionStore(settings.suggestions_dir)
    return AdviceContext(
        droids=droids,
        thesis=thesis,
        d2=d2,
        k2=k2,
        instruments=instruments,
        stats=stats,
        dip_days=days,
        activity=activity,
        k2_decisions=list(k2_decisions),
        weights=weights,
        cash_weight=cash_weight,
        past=store.past(since(now, settings.rejected_memory_days)),
        lookback_days=settings.advise_lookback_days,
        notes=notes,
    )


async def _advise(args: argparse.Namespace, settings: Settings, report: RunReport) -> Exit:
    now = datetime.now(UTC)
    model, _market, _limits, portfolio = await _setup(args, settings)
    prices = PriceHistory()
    try:
        ctx = await _advice_context(args, settings, portfolio, prices, now)
    except (ConfigReadError, ThesisError) as exc:
        print(f"Advice error: {exc}", file=sys.stderr)
        return report.finish(Exit.CONFIG, f"advice error: {exc}")

    agents = build_advise_agents(model)
    context = render_context(ctx)
    try:
        drafts = await draft_advice(agents, context)
        verdicts = await review_advice(agents, reviewer_prompt(context, drafts))
    except _EVALUATION_ERRORS as exc:
        logger.error("advice_failed error=%s", exc)
        return report.finish(Exit.FAILED, "the advice agents failed", errors=[str(exc)])

    # Price history for proposed new holdings, so code can check Yahoo knows them.
    new_symbols = []
    for draft in drafts.suggestions:
        if draft.kind == "add_holding" and ctx.instruments is not None and draft.symbol:
            checked = check_draft(draft, ctx, now)
            if not isinstance(checked, str) and checked.yahoo_symbol:
                new_symbols.append(checked.yahoo_symbol)
    new_closes = await prices.many(new_symbols)
    new_stats = {s: StockStats.from_closes(c) if c else None for s, c in new_closes.items()}

    run = finalize(
        drafts,
        verdicts,
        ctx,
        new_stats,
        now,
        settings.max_suggestions,
        settings.rejected_memory_days,
    )
    rendered = render_advice(run, ctx)
    files: dict[str, str] = {}
    if args.dry_run:
        print(rendered)
    else:
        store = SuggestionStore(settings.suggestions_dir)
        files["suggestions_file"] = str(store.write_run(run))
        files["report_file"] = str(
            write_new(settings.reports_dir, f"{now:%Y-%m-%d}_advice", ".md", rendered)
        )
        print(f"{len(run.suggestions)} suggestions")
        print(f"  {files['report_file']}\n  {files['suggestions_file']}")
        for s in run.suggestions:
            print(f"  {s.id}  {s.confidence:<6}  {s.title}")
    return report.finish(
        Exit.OK,
        f"{len(run.suggestions)} suggestions for {', '.join(ctx.droids)}"
        f" ({len(run.dropped)} drafts dropped)",
        {
            "suggestions": [
                {
                    "id": s.id,
                    "kind": s.kind,
                    "droid": s.droid,
                    "target": s.target,
                    "current": s.current,
                    "proposed": s.proposed,
                    "confidence": s.confidence,
                    "repeat": s.repeat,
                }
                for s in run.suggestions
            ],
            "dropped": [{"kind": d.draft.kind, "reason": d.reason} for d in run.dropped],
            "notes": run.notes,
            **files,
        },
    )


def _suggestions(args: argparse.Namespace, settings: Settings, report: RunReport) -> Exit:
    store = SuggestionStore(settings.suggestions_dir)
    if args.action in ("accept", "reject"):
        if not args.id:
            print(f"Give the id of the suggestion to {args.action}.", file=sys.stderr)
            return report.finish(Exit.CONFIG, f"no id to {args.action}")
        status: Status = "accepted" if args.action == "accept" else "rejected"
        try:
            suggestion = store.decide(args.id, status, args.note or "", datetime.now(UTC))
        except KeyError:
            print(f"No suggestion with id {args.id}.", file=sys.stderr)
            return report.finish(Exit.CONFIG, f"no suggestion {args.id}")
        print(f"{args.id} {status}: {suggestion.title}")
        return report.finish(Exit.OK, f"{args.id} {status}", {"id": args.id, "status": status})

    decisions = store.decisions()
    rows = []
    for suggestion in sorted(store.suggestions().values(), key=lambda s: s.created_at):
        decision = decisions.get(suggestion.id)
        status = decision.status if decision else "open"
        if status != "open" and not args.all:
            continue
        note = f" ({decision.note})" if decision and decision.note else ""
        rows.append(
            f"{suggestion.id}  {status:<8}  {suggestion.created_at:%Y-%m-%d}  "
            f"{suggestion.confidence:<6}  {suggestion.title}{note}"
        )
    print("\n".join(rows) if rows else "No open suggestions.")
    return report.finish(Exit.OK, f"{len(rows)} suggestions listed", {"count": len(rows)})


def list_gemini_models(api_key: str) -> list[str]:
    """Flash models the key can call with ``generateContent``, without the ``models/`` prefix."""
    from google import genai

    client = genai.Client(api_key=api_key)
    names = [
        model.name.removeprefix("models/")
        for model in client.models.list()
        if model.name
        and "flash" in model.name
        and "generateContent" in (model.supported_actions or [])
    ]
    return sorted(names)


def _models(settings: Settings, report: RunReport) -> Exit:
    if settings.google_api_key is None:
        print("Configuration error: GOOGLE_API_KEY is not set", file=sys.stderr)
        return report.finish(Exit.CONFIG, "GOOGLE_API_KEY is not set")
    try:
        names = list_gemini_models(settings.google_api_key.get_secret_value())
    except Exception as exc:  # the SDK raises several unrelated error types
        print(f"Could not list models: {type(exc).__name__}", file=sys.stderr)
        return report.finish(Exit.FAILED, "could not list models", errors=[type(exc).__name__])
    chosen = settings.gemini_models
    for name in names:
        marker = f"  (GEMINI_MODEL #{chosen.index(name) + 1})" if name in chosen else ""
        print(f"{name}{marker}")
    missing = [name for name in chosen if name not in names]
    if missing:
        print(f"In GEMINI_MODEL but not in this list: {', '.join(missing)}", file=sys.stderr)
    return report.finish(Exit.OK, f"{len(names)} Flash models available", {"models": names})


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


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
        "--batch",
        type=_positive_int,
        metavar="N",
        help="Review only the N stocks reviewed least recently, by earlier run reports",
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
    models = commands.add_parser(
        "models", help="List the Gemini Flash models your GOOGLE_API_KEY can use"
    )

    advise = commands.add_parser(
        "advise", help="Suggest changes to t2-d2's and t2-k2's strategy and configuration"
    )
    advise.add_argument(
        "--only", choices=sorted(_DROIDS), help="Advise on one droid only: d2 or k2"
    )
    advise.add_argument(
        "--no-portfolio",
        action="store_true",
        help="Skip Trading212; position weights are not considered",
    )
    advise.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the advice, with real model calls, but print it and record nothing",
    )

    suggestions = commands.add_parser(
        "suggestions", help="List suggestions, or record that you accepted or rejected one"
    )
    suggestions.add_argument(
        "action", nargs="?", default="list", choices=["list", "accept", "reject"]
    )
    suggestions.add_argument("id", nargs="?", help="Suggestion id, for accept and reject")
    suggestions.add_argument("--note", help="Why, recorded with your answer")
    suggestions.add_argument(
        "--all", action="store_true", help="List accepted and rejected suggestions too"
    )
    for sub in (run, review, export, models, advise, suggestions):
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
            if command == "advise":
                return await _advise(args, settings, report)
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
    elif args.command == "models":
        code = _models(settings, report)
    elif args.command == "suggestions":
        code = _suggestions(args, settings, report)
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
