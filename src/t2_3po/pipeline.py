"""The three-stage pipeline: gather data, run the council, build the proposal."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .agents import Council, digest_document, run_council
from .market import Fundamentals, MarketData, to_yahoo_symbol
from .models import CouncilResult, Proposal
from .portfolio import (
    Limits,
    PortfolioSnapshot,
    concentration_warnings,
    decision_warnings,
)
from .thesis import Thesis, read_document

PORTFOLIO_SKIPPED_WARNING = (
    "Portfolio was not evaluated: concentration limits were not checked and proposed weights "
    "are provisional."
)


@dataclass
class EvaluationContext:
    """Everything the council sees, gathered before any model is called."""

    thesis: Thesis
    fundamentals: Fundamentals
    portfolio: PortfolioSnapshot | None
    limits: Limits
    warnings: list[str]
    digests: dict[str, str] = field(default_factory=dict)

    def reviewed_tickers(self) -> set[str]:
        meta = self.thesis.meta
        tickers = {meta.ticker, self.fundamentals.symbol}
        if meta.t212_ticker:
            tickers.add(meta.t212_ticker)
        return tickers

    def render(self) -> str:
        """Render the context as tagged data for the agents."""
        meta = self.thesis.meta
        sections = [
            f'<thesis ticker="{meta.ticker}">\n{self.thesis.body}\n</thesis>',
            f"<market_data>\n{self.fundamentals.model_dump_json(indent=2)}\n</market_data>",
        ]
        if self.portfolio is None:
            sections.append("<portfolio>unavailable</portfolio>")
        else:
            held = self.portfolio.holding_for(*self.reviewed_tickers())
            current_weight = held.weight if held else 0.0
            sections.append(
                f"<portfolio>\n{self.portfolio.model_dump_json(indent=2)}\n</portfolio>\n\n"
                f"<current_weight_of_ticker_under_review>{current_weight:.4f}"
                "</current_weight_of_ticker_under_review>"
            )
        findings = {"limits": self.limits.model_dump(), "warnings": self.warnings}
        sections.append(
            f"<concentration_findings>\n{json.dumps(findings, indent=2)}\n</concentration_findings>"
        )
        sections += [
            f'<document_digest name="{name}">\n{digest}\n</document_digest>'
            for name, digest in self.digests.items()
        ]
        return "\n\n".join(sections)


async def gather_context(
    thesis: Thesis,
    market: MarketData,
    portfolio: PortfolioSnapshot | None,
    limits: Limits,
) -> EvaluationContext:
    """Stage 1: fetch fundamentals and run the concentration checks.

    Raises:
        MarketDataError: If there are no fundamentals, since the council would have nothing to
            cross-check the thesis against.
    """
    meta = thesis.meta
    symbol = (
        meta.yahoo_ticker
        or (to_yahoo_symbol(meta.t212_ticker) if meta.t212_ticker else None)
        or meta.ticker
    )
    fundamentals = await market.fundamentals(symbol)
    warnings = (
        concentration_warnings(portfolio, limits)
        if portfolio is not None
        else [PORTFOLIO_SKIPPED_WARNING]
    )
    return EvaluationContext(thesis, fundamentals, portfolio, limits, warnings)


async def evaluate(context: EvaluationContext, council: Council) -> CouncilResult:
    """Stages 2 and 3: digest any documents, then run the debate and synthesis."""
    for path in context.thesis.meta.documents:
        context.digests[path.name] = await digest_document(
            council, context.thesis.body, path.name, read_document(path)
        )
    return await run_council(council, context.render())


def build_proposal(
    context: EvaluationContext, result: CouncilResult, now: datetime | None = None
) -> Proposal:
    """Combine computed warnings with the model's, then add checks on its allocations."""
    decision = result.decision
    computed = decision_warnings(
        decision,
        context.portfolio,
        context.limits,
        context.reviewed_tickers(),
        context.fundamentals.sector,
    )
    warnings = list(dict.fromkeys([*context.warnings, *computed, *decision.portfolio_warnings]))
    return Proposal(
        timestamp=(now or datetime.now(UTC)).replace(microsecond=0),
        thesis_status=decision.thesis_status,
        portfolio_warnings=warnings,
        proposed_allocations=decision.proposed_allocations,
        suggested_config_updates=decision.suggested_config_updates,
        ticker=context.thesis.meta.ticker,
        thesis_file=context.thesis.path.name,
    )
