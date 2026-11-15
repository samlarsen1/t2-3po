"""Strategy and configuration advice for the trading droids.

The flow keeps the model's job narrow:

1. Code gathers the facts: t2-d2's and t2-k2's configuration, a year of prices per stock, what
   the droids did (their run reports), portfolio weights, and past suggestions with their answers.
2. The Strategy Analyst drafts suggestions from those facts; the Risk Reviewer argues against each.
3. Code validates every surviving draft against the configuration and t2-bb8's instrument list,
   drops repeats of rejected advice, ranks what's left and caps it.

Nothing here edits another droid's files. Suggestions are written for the operator to act on.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from .configs import (
    D2_STOCKS_FILE,
    D2_STRATEGY_FILE,
    K2_STRATEGY_FILE,
    D2Config,
    K2Config,
)
from .instruments import Instruments, ResolveError
from .models import (
    AdvisorOutput,
    Confidence,
    Droid,
    Kind,
    ReviewerOutput,
    Status,
    SuggestionDraft,
)
from .output import write_new
from .prices import StockStats
from .runlog import D2Activity
from .stocks import Stock

_CONFIDENCE_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
_WHOLE_1_TO_3 = frozenset({"overbought.min_signals"})
_UNIT_INTERVAL = frozenset({"harvest.gain_fraction"})
_MAX_RELATIVE_CHANGE = 0.5


# --- What code produces ---------------------------------------------------------------------


class Suggestion(BaseModel):
    """A validated suggestion. ``id`` is stable, so the same advice keeps the same id."""

    id: str
    created_at: datetime
    kind: Kind
    droid: Droid
    file: str
    target: str
    current: str | None
    proposed: str
    details: dict[str, str] = Field(default_factory=dict)
    rationale: str
    evidence: list[str]
    expected_effect: str
    risk: str
    confidence: Confidence
    reviewer_note: str = ""
    repeat: bool = False

    @property
    def title(self) -> str:
        target = self.details.get("symbol", self.target)
        if self.kind == "change_mode":
            return f"Move {target} from `{self.current}` to `{self.proposed}`"
        if self.kind == "add_holding":
            return f"Add {target} ({self.details.get('name', '')}) as `{self.proposed}`"
        if self.kind == "remove_holding":
            return f"Move {target} from `{self.current}` to the watchlist"
        return f"Change {self.droid} `{self.target}` from {self.current} to {self.proposed}"


class Dropped(BaseModel):
    draft: SuggestionDraft
    reason: str


class AdviceRun(BaseModel):
    created_at: datetime
    droids: list[Droid]
    suggestions: list[Suggestion]
    dropped: list[Dropped]
    no_change: str
    notes: list[str] = Field(default_factory=list)


def suggestion_id(kind: str, droid: str, target: str, proposed: str) -> str:
    text = f"{kind}|{droid}|{target.upper()}|{proposed.lower()}"
    return hashlib.sha256(text.encode()).hexdigest()[:8]


# --- The facts ------------------------------------------------------------------------------


@dataclass
class PastAdvice:
    suggestion: Suggestion
    status: Status
    note: str
    decided_at: datetime | None


@dataclass
class AdviceContext:
    """Everything the advice is based on, gathered by code."""

    droids: list[Droid]
    thesis: str
    d2: D2Config | None
    k2: K2Config | None
    instruments: Instruments | None
    stats: dict[str, StockStats | None]
    dip_days: dict[str, dict[str, int | None]]
    activity: D2Activity
    k2_decisions: list[dict[str, object]]
    weights: dict[str, float] | None
    cash_weight: float | None
    past: list[PastAdvice]
    lookback_days: int
    notes: list[str] = field(default_factory=list)

    def tunables(self, droid: str) -> dict[str, float]:
        if droid == "t2-d2" and self.d2 is not None:
            return self.d2.tunables()
        if droid == "t2-k2" and self.k2 is not None:
            return self.k2.tunables()
        return {}


def _pct(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1f}%" if signed else f"{value:.1f}%"


def _stock_line(ctx: AdviceContext, stock: Stock, market: str) -> str:
    parts = [f"{stock.symbol} ({stock.t212_ticker}, {market}) mode={stock.strategy}"]
    if ctx.weights is not None:
        parts.append(f"weight={_pct(ctx.weights.get(stock.t212_ticker, 0.0) * 100, signed=False)}")
    stats = ctx.stats.get(stock.symbol)
    if stats is None:
        parts.append("prices: unavailable")
    else:
        sma = {True: "above", False: "below", None: "n/a"}[stats.above_sma200]
        parts.append(
            f"1y {_pct(stats.return_1y_pct)}, volatility {_pct(stats.volatility_pct, False)}, "
            f"{_pct(stats.drawdown_pct, False)} below 1y high, {sma} 200-day SMA"
        )
    days = ctx.dip_days.get(stock.symbol, {})
    if days:
        counts = ", ".join(f"{mode}={'n/a' if n is None else n}" for mode, n in days.items())
        parts.append(f"dip days in the last year by mode: {counts}")
    act = ctx.activity.by_symbol.get(stock.symbol)
    if act is not None and act.evaluated:
        line = (
            f"last {ctx.lookback_days}d: evaluated {act.evaluated}x, buy signal {act.signalled}x, "
            f"bought {act.bought}x (£{act.deployed_gbp:.2f})"
        )
        top = "; ".join(f"'{reason}' x{n}" for reason, n in act.top_reasons())
        parts.append(line + (f", top skip reasons: {top}" if top else ""))
    else:
        parts.append(f"last {ctx.lookback_days}d: no t2-d2 decisions recorded")
    return " | ".join(parts)


def render_context(ctx: AdviceContext) -> str:
    """The facts, as tagged data for the model."""
    sections: list[str] = [f"<macro_thesis>\n{ctx.thesis.strip()}\n</macro_thesis>"]
    if ctx.d2 is not None:
        modes = []
        for name, settings in ctx.d2.modes.items():
            description = " ".join(str(settings.get("description", "")).split())
            values = ", ".join(
                f"{key.split('.')[-1]}={value:g}"
                for key, value in ctx.d2.tunables().items()
                if key.startswith(f"modes.{name}.")
            )
            modes.append(f"- {name}: {description} [{values}]")
        sections.append("<t2_d2_modes>\n" + "\n".join(modes) + "\n</t2_d2_modes>")
        lines = [_stock_line(ctx, t.stock, t.market) for t in ctx.d2.stocks]
        sections.append(
            f"<t2_d2_stocks sessions_recorded={ctx.activity.sessions}>\n"
            + "\n".join(f"- {line}" for line in lines)
            + f"\nwatchlist (not traded): {', '.join(ctx.d2.watchlist) or 'none'}"
            + "\n</t2_d2_stocks>"
        )
    if ctx.k2 is not None:
        values = ", ".join(f"{k}={v:g}" for k, v in ctx.k2.tunables().items())
        rows = []
        for d in ctx.k2_decisions:
            rows.append(
                f"- {d.get('ticker')}: {d.get('action')} gain={d.get('gainPct')}% "
                f"rsi={d.get('rsi')} above_sma={d.get('pctAboveSma')}% "
                f"run_up={d.get('runUpPct')}% signals={d.get('signals')}"
            )
        last_run = "\n".join(rows) or "no t2-k2 run recorded"
        sections.append(
            f"<t2_k2_settings>\n{values}\n</t2_k2_settings>\n"
            f"<t2_k2_last_run>\n{last_run}\n</t2_k2_last_run>"
        )
    if ctx.weights is not None and ctx.cash_weight is not None:
        sections.append(f"<portfolio>cash weight {_pct(ctx.cash_weight * 100, False)}</portfolio>")
    else:
        sections.append("<portfolio>unavailable: weights unknown</portfolio>")
    past = [
        f"- {p.suggestion.kind} {p.suggestion.droid} {p.suggestion.target}: "
        f"{p.suggestion.current} -> {p.suggestion.proposed} [{p.status}"
        + (f": {p.note}" if p.note else "")
        + "]"
        for p in ctx.past
    ]
    sections.append("<past_suggestions>\n" + ("\n".join(past) or "none") + "\n</past_suggestions>")
    allowed = [f"droids under review: {', '.join(ctx.droids)}"]
    if ctx.d2 is not None:
        allowed.append(f"t2-d2 modes: {', '.join(ctx.d2.modes)}; markets: uk, us, europe")
    for droid in ctx.droids:
        keys = ", ".join(f"{k} (now {v:g})" for k, v in ctx.tunables(droid).items())
        allowed.append(f"{droid} adjustable keys: {keys or 'none'}")
    sections.append("<allowed_changes>\n" + "\n".join(allowed) + "\n</allowed_changes>")
    return "\n\n".join(sections)


def reviewer_prompt(context: str, drafts: AdvisorOutput) -> str:
    numbered = [
        {"index": i, **draft.model_dump(exclude_none=True)}
        for i, draft in enumerate(drafts.suggestions, start=1)
    ]
    drafted = json.dumps(numbered, indent=2)
    return f"{context}\n\n<draft_suggestions>\n{drafted}\n</draft_suggestions>"


# --- Validation -----------------------------------------------------------------------------


def threshold_problem(key: str, current: float, proposed: float) -> str | None:
    """Why a proposed setting is outside its sane range, or ``None`` if it's acceptable."""
    if not math.isfinite(proposed):
        return "not a finite number"
    if proposed == current:
        return "same as the current value"
    if key in _WHOLE_1_TO_3:
        return None if proposed.is_integer() and 1 <= proposed <= 3 else "must be 1, 2 or 3"
    if key in _UNIT_INTERVAL:
        return None if 0 < proposed <= 1 else "must be above 0 and at most 1"
    if current == 0:
        return None if abs(proposed) <= 100 else "out of range"
    if (current > 0) != (proposed > 0):
        return "would change sign"
    if abs(proposed - current) > _MAX_RELATIVE_CHANGE * abs(current):
        return f"changes by more than {_MAX_RELATIVE_CHANGE:.0%} in one step"
    return None


@dataclass
class Checked:
    """A draft that passed the configuration checks, before review and dedupe."""

    suggestion: Suggestion
    yahoo_symbol: str | None = None


def check_draft(draft: SuggestionDraft, ctx: AdviceContext, now: datetime) -> Checked | str:
    """Validate one draft against the configuration. Returns the reason if it fails."""
    if draft.droid not in ctx.droids:
        return f"{draft.droid} is not under review"

    def make(
        target: str,
        file: str,
        current: str | None,
        proposed: str,
        details: dict[str, str] | None = None,
    ) -> Suggestion:
        return Suggestion(
            id=suggestion_id(draft.kind, draft.droid, target, proposed),
            created_at=now,
            kind=draft.kind,
            droid=draft.droid,
            file=file,
            target=target,
            current=current,
            proposed=proposed,
            details=details or {},
            rationale=draft.rationale,
            evidence=draft.evidence,
            expected_effect=draft.expected_effect,
            risk=draft.risk,
            confidence=draft.confidence,
        )

    if draft.kind == "change_threshold":
        tunables = ctx.tunables(draft.droid)
        if not draft.key or draft.key not in tunables:
            return f"{draft.key!r} is not an adjustable {draft.droid} setting"
        try:
            proposed = float(draft.proposed)
        except ValueError:
            return f"{draft.proposed!r} is not a number"
        current = tunables[draft.key]
        problem = threshold_problem(draft.key, current, proposed)
        if problem:
            return f"{draft.key} = {proposed:g}: {problem}"
        file = D2_STRATEGY_FILE if draft.droid == "t2-d2" else K2_STRATEGY_FILE
        return Checked(make(draft.key, file, f"{current:g}", f"{proposed:g}"))

    if draft.droid != "t2-d2" or ctx.d2 is None:
        return f"{draft.kind} applies to t2-d2's stocks only"
    d2 = ctx.d2

    if draft.kind in ("change_mode", "remove_holding"):
        traded = d2.find(draft.symbol)
        if traded is None:
            return f"{draft.symbol!r} is not a traded stock in {D2_STOCKS_FILE}"
        stock = traded.stock
        if draft.kind == "change_mode":
            if draft.proposed not in d2.modes:
                return f"{draft.proposed!r} is not a t2-d2 mode"
            if draft.proposed == stock.strategy:
                return f"{stock.symbol} already uses {stock.strategy}"
            new_mode = draft.proposed
        else:
            new_mode = "watchlist"
        details = {"symbol": stock.symbol, "t212_ticker": stock.t212_ticker}
        return Checked(make(stock.symbol, D2_STOCKS_FILE, stock.strategy, new_mode, details))

    # add_holding
    if not draft.symbol or draft.market is None:
        return "a new holding needs a symbol and a market"
    if draft.proposed not in d2.modes:
        return f"{draft.proposed!r} is not a t2-d2 mode"
    if ctx.instruments is None:
        return "t2-bb8's instrument list is unavailable, so a new holding can't be checked"
    try:
        instrument = ctx.instruments.resolve(draft.symbol, draft.market, draft.t212_ticker)
    except ResolveError as exc:
        return str(exc)
    symbol = (instrument.shortName or draft.symbol).upper()
    if d2.lists(symbol, instrument.ticker):
        return f"{symbol} is already in {D2_STOCKS_FILE}"
    stock = Stock(
        symbol=symbol,
        t212_ticker=instrument.ticker,
        name=instrument.name or symbol,
        strategy=draft.proposed,
    )
    details = {
        "symbol": symbol,
        "t212_ticker": instrument.ticker,
        "name": stock.name,
        "market": draft.market,
        "currency": instrument.currencyCode or "",
    }
    suggestion = make(instrument.ticker, D2_STOCKS_FILE, None, draft.proposed, details)
    return Checked(suggestion, yahoo_symbol=stock.yahoo_symbol)


def finalize(
    drafts: AdvisorOutput,
    review: ReviewerOutput,
    ctx: AdviceContext,
    new_holding_stats: dict[str, StockStats | None],
    now: datetime,
    max_suggestions: int,
    memory_days: int,
) -> AdviceRun:
    """Apply the reviewer's verdicts and the code checks, drop repeats, rank and cap."""
    verdicts = {v.index: v for v in review.verdicts}
    past = {p.suggestion.id: p for p in ctx.past}
    kept: list[tuple[int, Suggestion]] = []
    dropped: list[Dropped] = []
    seen: set[str] = set()
    for index, draft in enumerate(drafts.suggestions, start=1):
        verdict = verdicts.get(index)
        if verdict is None:
            dropped.append(Dropped(draft=draft, reason="the reviewer gave no verdict"))
            continue
        if not verdict.keep:
            dropped.append(Dropped(draft=draft, reason=f"reviewer: {verdict.note}"))
            continue
        checked = check_draft(draft, ctx, now)
        if isinstance(checked, str):
            dropped.append(Dropped(draft=draft, reason=checked))
            continue
        suggestion = checked.suggestion
        if suggestion.kind == "add_holding":
            stats = new_holding_stats.get(checked.yahoo_symbol or "")
            if stats is None:
                dropped.append(Dropped(draft=draft, reason="no usable Yahoo price history"))
                continue
        if suggestion.id in seen:
            dropped.append(Dropped(draft=draft, reason="duplicate of another draft"))
            continue
        previous = past.get(suggestion.id)
        if previous is not None and previous.status == "accepted":
            dropped.append(Dropped(draft=draft, reason="already accepted"))
            continue
        if previous is not None and previous.status == "rejected":
            decided = previous.decided_at
            if decided is None or now - decided <= timedelta(days=memory_days):
                when = "" if decided is None else f" on {decided:%Y-%m-%d}"
                dropped.append(Dropped(draft=draft, reason=f"rejected{when}"))
                continue
        seen.add(suggestion.id)
        suggestion.confidence = verdict.confidence
        suggestion.reviewer_note = verdict.note
        suggestion.repeat = previous is not None and previous.status == "open"
        kept.append((index, suggestion))

    kept.sort(key=lambda item: (_CONFIDENCE_RANK[item[1].confidence], item[0]))
    for index, _extra in kept[max_suggestions:]:
        reason = f"over the cap of {max_suggestions}"
        dropped.append(Dropped(draft=drafts.suggestions[index - 1], reason=reason))
    return AdviceRun(
        created_at=now,
        droids=ctx.droids,
        suggestions=[s for _, s in kept[:max_suggestions]],
        dropped=dropped,
        no_change=drafts.no_change,
        notes=ctx.notes,
    )


# --- Memory ---------------------------------------------------------------------------------


class Decision(BaseModel):
    status: Status
    decided_at: datetime
    note: str = ""


class SuggestionStore:
    """Advice runs and the operator's answers, under ``var/suggestions``."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.decisions_path = directory / "decisions.json"

    def runs(self) -> list[AdviceRun]:
        if not self.directory.is_dir():
            return []
        runs = []
        for path in sorted(self.directory.glob("*_advice*.json")):
            try:
                runs.append(AdviceRun.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValidationError):
                continue
        return runs

    def decisions(self) -> dict[str, Decision]:
        try:
            raw = json.loads(self.decisions_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        decisions: dict[str, Decision] = {}
        for key, value in raw.items():
            try:
                decisions[key] = Decision.model_validate(value)
            except ValidationError:
                continue
        return decisions

    def suggestions(self) -> dict[str, Suggestion]:
        """Every suggestion ever made, newest version of each id."""
        latest: dict[str, Suggestion] = {}
        for run in self.runs():
            for suggestion in run.suggestions:
                latest[suggestion.id] = suggestion
        return latest

    def past(self, since: datetime) -> list[PastAdvice]:
        """Suggestions made or decided since ``since``, with their status."""
        decisions = self.decisions()
        past = []
        for suggestion in self.suggestions().values():
            decision = decisions.get(suggestion.id)
            recent = suggestion.created_at >= since or (
                decision is not None and decision.decided_at >= since
            )
            if recent:
                past.append(
                    PastAdvice(
                        suggestion=suggestion,
                        status=decision.status if decision else "open",
                        note=decision.note if decision else "",
                        decided_at=decision.decided_at if decision else None,
                    )
                )
        return past

    def decide(self, suggestion_id: str, status: Status, note: str, now: datetime) -> Suggestion:
        """Record the operator's answer.

        Raises:
            KeyError: If no suggestion has that id.
        """
        suggestion = self.suggestions().get(suggestion_id)
        if suggestion is None:
            raise KeyError(suggestion_id)
        decisions = self.decisions()
        decisions[suggestion_id] = Decision(status=status, decided_at=now, note=note)
        self.directory.mkdir(parents=True, exist_ok=True)
        data = {key: value.model_dump(mode="json") for key, value in decisions.items()}
        tmp = self.decisions_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.decisions_path)
        return suggestion

    def write_run(self, run: AdviceRun) -> Path:
        return write_new(
            self.directory,
            f"{run.created_at:%Y-%m-%d}_advice",
            ".json",
            run.model_dump_json(indent=2) + "\n",
        )


def since(now: datetime, days: int) -> datetime:
    return (now - timedelta(days=days)).astimezone(UTC)


# --- Report ---------------------------------------------------------------------------------


def _change_line(s: Suggestion) -> str:
    if s.kind == "add_holding":
        d = s.details
        return (
            f"`{s.droid}` `{s.file}`, under `markets.{d.get('market')}.tickers`, add: "
            f"`symbol: {d.get('symbol')}`, `t212_ticker: {d.get('t212_ticker')}`, "
            f"`name: {d.get('name')}`, `strategy: {s.proposed}`"
        )
    if s.kind == "remove_holding":
        return (
            f"`{s.droid}` `{s.file}`: remove {s.target} from its market's tickers and add it to "
            "`watchlist`"
        )
    if s.kind == "change_mode":
        return f"`{s.droid}` `{s.file}`: {s.target} `strategy: {s.current}` → `{s.proposed}`"
    return f"`{s.droid}` `{s.file}`: `{s.target}`: {s.current} → {s.proposed}"


def render_advice(run: AdviceRun, ctx: AdviceContext) -> str:
    """The advice as a markdown report for the operator."""
    out = [
        f"# Strategy advice, {run.created_at:%Y-%m-%d}",
        "",
        f"Reviewed {', '.join(run.droids)} over the last {ctx.lookback_days} days of run reports"
        + ("; portfolio weights included." if ctx.weights is not None else "; no portfolio."),
        "",
        f"## Suggestions ({len(run.suggestions)})",
        "",
    ]
    if not run.suggestions:
        out += ["None this time.", ""]
    for n, s in enumerate(run.suggestions, start=1):
        out += [
            f"### {n}. {s.title}",
            "",
            f"- **Id:** `{s.id}`" + (" (suggested before, still open)" if s.repeat else ""),
            f"- **Confidence:** {s.confidence}",
            f"- **Change:** {_change_line(s)}",
            f"- **Why:** {s.rationale}",
            "- **Evidence:**",
            *[f"  - {item}" for item in s.evidence],
            f"- **Expected effect:** {s.expected_effect}",
            f"- **Risk:** {s.risk}",
            f"- **Reviewer:** {s.reviewer_note}",
            "",
        ]
    if run.suggestions:
        out += [
            "Record your answer so the advice isn't repeated:",
            "",
            "```bash",
            "t2-3po suggestions accept <id> --note 'applied'",
            "t2-3po suggestions reject <id> --note 'why not'",
            "```",
            "",
        ]
    out += ["## What should stay as it is", "", run.no_change or "No comment.", ""]
    if run.dropped:
        out += ["## Dropped by the checks", ""]
        for item in run.dropped:
            d = item.draft
            what = d.key or d.symbol or d.t212_ticker or "?"
            out.append(f"- {d.kind} {d.droid} {what} → {d.proposed}: {item.reason}")
        out.append("")
    if ctx.d2 is not None:
        out += ["## Figures the advice was based on", ""]
        out += [f"- {_stock_line(ctx, t.stock, t.market)}" for t in ctx.d2.stocks]
        out.append("")
    if run.notes:
        out += ["## Notes", "", *[f"- {note}" for note in run.notes], ""]
    return "\n".join(out)
