"""Shared data models: agent outputs and the proposal contract consumed by t2-d2."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ThesisStatus(StrEnum):
    VALIDATED = "VALIDATED"
    VALIDATED_WITH_RISKS = "VALIDATED_WITH_RISKS"
    NEEDS_REVISION = "NEEDS_REVISION"
    REJECTED = "REJECTED"


class Action(StrEnum):
    BUY = "BUY"
    HOLD = "HOLD"
    TRIM = "TRIM"
    SELL = "SELL"


class ClaimVerdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"


class AnalystView(BaseModel):
    """A Bull or Bear analyst's case."""

    summary: str = Field(description="The analyst's case in two or three sentences.")
    key_points: list[str] = Field(description="The specific arguments, most important first.")
    conviction: Literal["LOW", "MEDIUM", "HIGH"]


class ClaimCheck(BaseModel):
    claim: str
    verdict: ClaimVerdict
    evidence: str = Field(description="The supplied figures behind the verdict, or what's missing.")


class QuantView(BaseModel):
    """The Quantitative Analyst's cross-check of the thesis and both analysts."""

    summary: str
    valuation_assessment: str
    balance_sheet_assessment: str
    claim_checks: list[ClaimCheck]


class ProposedAllocation(BaseModel):
    ticker: str
    target_weight: float = Field(ge=0, le=1, description="Target share of total ISA value, 0 to 1.")
    action: Action
    reasoning: str


class ConfigUpdate(BaseModel):
    """A suggested change to a ticker's ``strategy`` in t2-d2's ``config/stocks.yaml``."""

    ticker: str
    strategy: Literal["mild", "moderate", "aggressive"]
    reasoning: str


class OrchestratorDecision(BaseModel):
    """The Protocol Officer's synthesis of the council's debate."""

    thesis_status: ThesisStatus
    executive_summary: str
    critique: str = Field(description="Where the thesis holds, where it fails, and why.")
    key_risks: list[str]
    portfolio_warnings: list[str] = Field(
        description="Only portfolio risks not already in the concentration findings you were given."
    )
    proposed_allocations: list[ProposedAllocation]
    suggested_config_updates: list[ConfigUpdate]


class CouncilResult(BaseModel):
    bull: AnalystView
    bear: AnalystView
    quant: QuantView
    decision: OrchestratorDecision


class VehicleRole(StrEnum):
    """How a stock expresses the macro thesis."""

    GROWTH = "GROWTH"
    STABILITY = "STABILITY"
    NEITHER = "NEITHER"


class KeyMetric(BaseModel):
    name: str
    value: str = Field(
        description="The figure with its units, as supplied or by simple arithmetic shown here."
    )
    reading: str = Field(description="What it says about the stock's role, in one sentence.")


class VehicleProfile(BaseModel):
    """The Vehicle Analyst's view of one stock as a way to express the macro thesis."""

    thesis_status: ThesisStatus
    thesis_pillars: list[str] = Field(
        description="The pillars it serves, named as in the thesis. Empty if it serves none."
    )
    pillar_link: str = Field(
        description="How the business serves those pillars, and how much of it does."
    )
    role: VehicleRole
    role_case: str = Field(description="Why it fits that role, or why it fits neither.")
    key_metrics: list[KeyMetric] = Field(description="The four to eight metrics that decide it.")
    outlook: str = Field(
        description="Over the thesis horizon: what has to happen for the stock to do its job, "
        "and what would show it isn't."
    )
    key_risks: list[str]
    proposed_allocation: ProposedAllocation
    suggested_strategy: Literal["mild", "moderate", "aggressive"] | None = Field(
        description="A new t2-d2 strategy, only if the verdict justifies a change. Otherwise null."
    )
    strategy_reasoning: str | None = None


class PillarCoverage(BaseModel):
    pillar: str
    stocks: list[str] = Field(description="Symbols of the stocks that serve it.")
    assessment: str = Field(description="How well it is covered, and by what kind of vehicle.")


class ReviewOverview(BaseModel):
    """The Portfolio Reviewer's view across every stock."""

    summary: str
    pillar_coverage: list[PillarCoverage]
    growth_stability_balance: str
    overlaps: list[str] = Field(description="Stocks that depend on the same driver, and which.")
    gaps: list[str] = Field(description="Pillars or kinds of exposure with no viable vehicle.")


class Proposal(BaseModel):
    """Machine-readable proposal written to ``proposals/``.

    ``timestamp`` through ``proposed_allocations`` are the core contract. The remaining fields
    identify the thesis and carry the human approval stamp.
    """

    timestamp: datetime
    thesis_status: ThesisStatus
    portfolio_warnings: list[str]
    proposed_allocations: list[ProposedAllocation]
    suggested_config_updates: list[ConfigUpdate] = Field(default_factory=list)
    ticker: str
    thesis_file: str
    approved_at: datetime | None = None
