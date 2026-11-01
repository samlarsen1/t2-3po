"""Writes the human report and machine proposal, and exports approved proposals for t2-d2."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .models import AnalystView, CouncilResult, Proposal
from .pipeline import EvaluationContext


def _write_new(directory: Path, stem: str, suffix: str, text: str) -> Path:
    """Write to ``stem.suffix``, adding ``_2``, ``_3``... rather than overwriting an earlier run."""
    directory.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        path = directory / f"{stem if attempt == 1 else f'{stem}_{attempt}'}{suffix}"
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(text)
            return path
        except FileExistsError:
            attempt += 1


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "- None"


def _analyst_section(title: str, view: AnalystView) -> str:
    return (
        f"## {title} (conviction: {view.conviction})\n\n{view.summary}\n\n"
        f"{_bullets(view.key_points)}\n"
    )


def _metric(value: float | None, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}" if percent else f"{value:,.2f}"


def render_report(context: EvaluationContext, result: CouncilResult, proposal: Proposal) -> str:
    """Render the human review report as markdown."""
    decision, data = result.decision, context.fundamentals
    allocations = "\n".join(
        f"| {a.ticker} | {a.action} | {a.target_weight:.1%} | {a.reasoning} |"
        for a in proposal.proposed_allocations
    )
    config_updates = _bullets(
        [
            f"`{u.ticker}`: set `strategy: {u.strategy}`. {u.reasoning}"
            for u in proposal.suggested_config_updates
        ]
    )
    claims = "\n".join(
        f"| {c.claim} | {c.verdict} | {c.evidence} |" for c in result.quant.claim_checks
    )
    return f"""# {proposal.ticker}: {proposal.thesis_status}

{proposal.timestamp:%Y-%m-%d %H:%M} UTC, thesis `{proposal.thesis_file}`. Advisory output for \
human review; nothing has been executed.

## Summary

{decision.executive_summary}

## Critique

{decision.critique}

## Key risks

{_bullets(decision.key_risks)}

## Portfolio warnings

{_bullets(proposal.portfolio_warnings)}

## Proposed allocations

| Ticker | Action | Target weight | Reasoning |
|---|---|---|---|
{allocations}

## Suggested t2-d2 config updates

{config_updates}

{_analyst_section("Bull case", result.bull)}
{_analyst_section("Bear case", result.bear)}
## Quantitative review

{result.quant.summary}

**Valuation:** {result.quant.valuation_assessment}

**Balance sheet:** {result.quant.balance_sheet_assessment}

| Claim | Verdict | Evidence |
|---|---|---|
{claims}

## Market data ({data.symbol}, {data.currency or "currency n/a"})

| Metric | Value |
|---|---|
| Price | {_metric(data.price)} |
| Trailing P/E | {_metric(data.trailing_pe)} |
| Forward P/E | {_metric(data.forward_pe)} |
| Debt/Equity | {_metric(data.debt_to_equity)} |
| FCF yield | {_metric(data.fcf_yield, percent=True)} |
| Consensus target | {_metric(data.consensus_target_price)} |
"""


def write_outputs(
    context: EvaluationContext,
    result: CouncilResult,
    proposal: Proposal,
    reports_dir: Path,
    proposals_dir: Path,
) -> tuple[Path, Path]:
    """Write ``YYYY-MM-DD_ticker.md`` and ``proposal_YYYYMMDD.json``."""
    report_stem = f"{proposal.timestamp:%Y-%m-%d}_{proposal.ticker.rstrip('.')}"
    report = _write_new(reports_dir, report_stem, ".md", render_report(context, result, proposal))
    machine = _write_new(
        proposals_dir,
        f"proposal_{proposal.timestamp:%Y%m%d}",
        ".json",
        serialise_proposal(proposal),
    )
    return report, machine


def serialise_proposal(proposal: Proposal) -> str:
    """Serialise a proposal, omitting the approval stamp until there is one."""
    return proposal.model_dump_json(indent=2, exclude_none=True) + "\n"


def export_approved(proposal_path: Path, export_dir: Path, now: datetime | None = None) -> Path:
    """Stamp a reviewed proposal as approved and write it to the export directory.

    The source file is left unchanged. t2-d2 is not invoked and its config is not edited.

    Raises:
        OSError: If the file can't be read or written.
        pydantic.ValidationError: If the file isn't a valid proposal.
    """
    proposal = Proposal.model_validate_json(proposal_path.read_text(encoding="utf-8"))
    approved = proposal.model_copy(
        update={"approved_at": (now or datetime.now(UTC)).replace(microsecond=0)}
    )
    return _write_new(export_dir, proposal_path.stem, ".json", serialise_proposal(approved))
