"""The macro review: profile every stock as a vehicle for one thesis, then look across them all."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .agents import ReviewAgents, review_overview
from .market import Fundamentals
from .models import (
    ConfigUpdate,
    Proposal,
    ReviewOverview,
    ThesisStatus,
    VehicleProfile,
    VehicleRole,
)
from .output import bullets, metric, write_new
from .pipeline import PORTFOLIO_SKIPPED_WARNING, EvaluationContext
from .portfolio import decision_warnings
from .stocks import Stock
from .thesis import Thesis, ThesisMeta

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StockReview:
    """One stock's finished profile and the proposal built from it."""

    stock: Stock
    context: EvaluationContext
    profile: VehicleProfile
    proposal: Proposal


def stock_thesis(path: Path, macro_thesis: str, stock: Stock) -> Thesis:
    """Pair the macro thesis with one stock, so the normal data stage can gather its context."""
    return Thesis(
        path=path,
        meta=ThesisMeta(
            ticker=stock.symbol,
            t212_ticker=stock.t212_ticker,
            yahoo_ticker=stock.yahoo_symbol,
        ),
        body=(
            f"{macro_thesis.strip()}\n\n## Stock under review\n\n{stock.name} ({stock.symbol}), "
            f"currently on the `{stock.strategy}` strategy in t2-d2."
        ),
    )


def build_vehicle_proposal(
    stock: Stock, context: EvaluationContext, profile: VehicleProfile, now: datetime | None = None
) -> Proposal:
    """Turn a profile into a proposal, checking its allocation against the limits."""
    # The model is asked about one stock, so its allocation is pinned to that stock's symbol.
    allocation = profile.proposed_allocation.model_copy(update={"ticker": stock.symbol})
    computed = decision_warnings(
        [allocation],
        context.portfolio,
        context.limits,
        context.reviewed_tickers(),
        context.fundamentals.sector,
    )
    updates = []
    if profile.suggested_strategy and profile.suggested_strategy != stock.strategy:
        reasoning = profile.strategy_reasoning or "No reasoning given."
        updates.append(
            ConfigUpdate(
                ticker=stock.symbol, strategy=profile.suggested_strategy, reasoning=reasoning
            )
        )
    return Proposal(
        timestamp=(now or datetime.now(UTC)).replace(microsecond=0),
        thesis_status=profile.thesis_status,
        portfolio_warnings=list(dict.fromkeys([*context.warnings, *computed])),
        proposed_allocations=[allocation],
        suggested_config_updates=updates,
        ticker=stock.symbol,
        thesis_file=context.thesis.path.name,
    )


def overview_prompt(macro_thesis: str, reviews: list[StockReview]) -> str:
    """Render the thesis and every profile for the Portfolio Reviewer."""
    profiles = [
        {
            "symbol": r.stock.symbol,
            "name": r.stock.name,
            "sector": r.context.fundamentals.sector,
            "industry": r.context.fundamentals.industry,
            "t2d2_strategy": r.stock.strategy,
            **r.profile.model_dump(
                mode="json", exclude={"suggested_strategy", "strategy_reasoning"}
            ),
        }
        for r in reviews
    ]
    return (
        f"<thesis>\n{macro_thesis.strip()}\n</thesis>\n\n"
        f"<profiles>\n{json.dumps(profiles, indent=2)}\n</profiles>"
    )


async def overview_or_none(
    agents: ReviewAgents,
    macro_thesis: str,
    reviews: list[StockReview],
    errors: tuple[type[Exception], ...],
) -> ReviewOverview | None:
    """Run the Portfolio Reviewer. A failure is logged, as the profiles are still worth writing."""
    try:
        return await review_overview(agents, overview_prompt(macro_thesis, reviews))
    except errors as exc:
        logger.error("overview_failed error=%s", exc)
        return None


_ROLE_ORDER = list(VehicleRole)
_STATUS_ORDER = list(ThesisStatus)


def _order(reviews: list[StockReview]) -> list[StockReview]:
    """Best fit first, then growth before stability before neither."""
    return sorted(
        reviews,
        key=lambda r: (
            _STATUS_ORDER.index(r.profile.thesis_status),
            _ROLE_ORDER.index(r.profile.role),
        ),
    )


def _money(value: float | None) -> str:
    if value is None:
        return "n/a"
    for size, suffix in ((1e9, "bn"), (1e6, "m")):
        if abs(value) >= size:
            return f"{value / size:,.2f}{suffix}"
    return f"{value:,.0f}"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _history(data: Fundamentals) -> str:
    if not data.annual:
        return "No annual history was available.\n"
    rows = "\n".join(
        f"| {a.year_end} | {_money(a.revenue)} | {_money(a.net_income)} "
        f"| {_money(a.free_cash_flow)} |"
        for a in data.annual
    )
    return (
        f"| Year end | Revenue | Net income | Free cash flow |\n|---|---|---|---|\n{rows}\n\n"
        f"Revenue CAGR {metric(data.revenue_cagr, percent=True)}, profitable "
        f"{data.profitable_years or 'n/a'} years, free cash flow positive "
        f"{data.positive_fcf_years or 'n/a'} years. Figures in "
        f"{data.financial_currency or 'currency n/a'}.\n"
    )


def _stock_section(review: StockReview) -> str:
    profile, stock, data = review.profile, review.stock, review.context.fundamentals
    allocation = review.proposal.proposed_allocations[0]
    metrics = "\n".join(
        f"| {_cell(m.name)} | {_cell(m.value)} | {_cell(m.reading)} |" for m in profile.key_metrics
    )
    pillars = ", ".join(profile.thesis_pillars) or "None"
    strategy = (
        f"Suggest `strategy: {update.strategy}` (now `{stock.strategy}`). {update.reasoning}"
        if (update := next(iter(review.proposal.suggested_config_updates), None))
        else f"No change from `{stock.strategy}`."
    )
    warnings = [w for w in review.proposal.portfolio_warnings if w != PORTFOLIO_SKIPPED_WARNING]
    return f"""## {stock.name} ({stock.symbol}): {profile.thesis_status}, {profile.role}

**Pillars:** {pillars}. {profile.pillar_link}

**Role:** {profile.role_case}

| Metric | Value | Reading |
|---|---|---|
{metrics}

**Outlook:** {profile.outlook}

**Key risks:**

{bullets(profile.key_risks)}

**Proposal:** {allocation.action} at {allocation.target_weight:.1%}. {allocation.reasoning}

**t2-d2 strategy:** {strategy}

**Portfolio warnings:**

{bullets(warnings)}

**History ({data.symbol}):**

{_history(data)}"""


def _overview_section(overview: ReviewOverview | None) -> str:
    if overview is None:
        return "## Overview\n\nThe overview failed, so only the stock profiles are below.\n"
    coverage = "\n".join(
        f"| {_cell(c.pillar)} | {', '.join(c.stocks) or 'None'} | {_cell(c.assessment)} |"
        for c in overview.pillar_coverage
    )
    return f"""## Overview

{overview.summary}

| Pillar | Stocks | Coverage |
|---|---|---|
{coverage}

**Growth and stability:** {overview.growth_stability_balance}

**Overlaps:**

{bullets(overview.overlaps)}

**Gaps:**

{bullets(overview.gaps)}
"""


def render_review(
    thesis_name: str,
    reviews: list[StockReview],
    overview: ReviewOverview | None,
    failed: list[str],
    portfolio_evaluated: bool,
) -> str:
    """Render the whole review as one markdown report."""
    ordered = _order(reviews)
    rows = []
    for r in ordered:
        allocation = r.proposal.proposed_allocations[0]
        suggested = next(
            (u.strategy for u in r.proposal.suggested_config_updates), r.stock.strategy
        )
        rows.append(
            f"| {r.stock.name} ({r.stock.symbol}) | {r.profile.thesis_status} "
            f"| {', '.join(r.profile.thesis_pillars) or 'None'} | {r.profile.role} "
            f"| {allocation.action} | {allocation.target_weight:.1%} "
            f"| {r.stock.strategy} | {suggested} |"
        )
    total = sum(r.proposal.proposed_allocations[0].target_weight for r in reviews)
    # Each stock is sized alone, so nothing else stops the targets adding up to too much.
    warnings = [f"Target weights add up to {total:.1%} of the ISA. Each stock was sized alone."]
    if not portfolio_evaluated:
        warnings.append(PORTFOLIO_SKIPPED_WARNING)
    warnings += [f"Review failed, so this stock isn't included: {symbol}." for symbol in failed]
    timestamp = reviews[0].proposal.timestamp if reviews else datetime.now(UTC)
    table = "\n".join(rows)
    sections = "\n".join(_stock_section(r) for r in ordered)
    return f"""# Macro review against `{thesis_name}`

{timestamp:%Y-%m-%d %H:%M} UTC. {len(reviews)} stocks reviewed. Advisory output for human \
review; nothing has been executed.

{bullets(warnings)}

{_overview_section(overview)}
## Stocks

| Stock | Fit | Pillars | Role | Action | Target weight | Strategy now | Suggested |
|---|---|---|---|---|---|---|---|
{table}

{sections}"""


def write_review(report: str, timestamp: datetime, reports_dir: Path) -> Path:
    """Write ``YYYY-MM-DD_macro_review.md``."""
    return write_new(reports_dir, f"{timestamp:%Y-%m-%d}_macro_review", ".md", report)


async def gather_profiles(
    stocks: list[Stock],
    profile: Callable[[Stock], Awaitable[StockReview]],
    errors: tuple[type[Exception], ...],
    concurrency: int,
) -> tuple[list[StockReview], list[str]]:
    """Profile every stock a few at a time. One failure doesn't lose the others."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(stock: Stock) -> StockReview | None:
        async with semaphore:
            try:
                return await profile(stock)
            except errors as exc:
                logger.error("review_failed stock=%s error=%s", stock.symbol, exc)
                return None

    outcomes = await asyncio.gather(*(one(stock) for stock in stocks))
    reviews = [outcome for outcome in outcomes if outcome is not None]
    failed = [s.symbol for s, outcome in zip(stocks, outcomes, strict=True) if outcome is None]
    return reviews, failed
