from typing import Any

import httpx
import pytest

from t2_3po.market import MarketData
from t2_3po.models import Action, OrchestratorDecision, ProposedAllocation, ThesisStatus
from t2_3po.portfolio import (
    Holding,
    Limits,
    PortfolioSnapshot,
    concentration_warnings,
    decision_warnings,
    fetch_snapshot,
)
from t2_3po.t212 import ReadOnlyTrading212Client

LIMITS = Limits(max_sector_weight=0.30, max_position_weight=0.175)
REVIEWED = {"NVDA", "NVDA_US_EQ"}


def _holding(ticker: str, symbol: str | None, weight: float, sector: str) -> Holding:
    return Holding(
        t212_ticker=ticker,
        yahoo_symbol=symbol,
        name=None,
        value=weight * 1000,
        weight=weight,
        sector=sector,
    )


def _snapshot() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        currency="GBP",
        total_value=1000,
        cash_available=100,
        cash_weight=0.1,
        holdings=[
            _holding("NVDA_US_EQ", "NVDA", 0.20, "Technology"),
            _holding("AMD_US_EQ", "AMD", 0.15, "Technology"),
            _holding("VWRPl_EQ", "VWRP.L", 0.10, "Unknown"),
        ],
        sector_weights={"Technology": 0.35, "Unknown": 0.10},
    )


def _decision(*allocations: ProposedAllocation) -> OrchestratorDecision:
    return OrchestratorDecision(
        thesis_status=ThesisStatus.VALIDATED,
        executive_summary="s",
        critique="c",
        key_risks=[],
        portfolio_warnings=[],
        proposed_allocations=list(allocations),
        suggested_config_updates=[],
    )


def _allocation(ticker: str, weight: float, action: Action) -> ProposedAllocation:
    return ProposedAllocation(ticker=ticker, target_weight=weight, action=action, reasoning="r")


async def test_fetch_snapshot_computes_weights_and_sector_totals() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("summary"):
            return httpx.Response(
                200,
                json={"currency": "GBP", "cash": {"availableToTrade": 100}, "totalValue": 1000},
            )

        def position(ticker: str, value: float) -> dict[str, Any]:
            return {
                "instrument": {"ticker": ticker},
                "walletImpact": {"currentValue": value},
            }

        return httpx.Response(
            200,
            json=[
                position("AMD_US_EQ", 100),
                position("NVDA_US_EQ", 300),
                position("ODDz_EQ", 50),
            ],
        )

    market = MarketData(fetcher=lambda _: {"sector": "Technology", "marketCap": 1})
    async with ReadOnlyTrading212Client(
        "k", "s", "https://x.test", transport=httpx.MockTransport(handler)
    ) as client:
        snapshot = await fetch_snapshot(client, market)

    assert [h.t212_ticker for h in snapshot.holdings] == ["NVDA_US_EQ", "AMD_US_EQ", "ODDz_EQ"]
    assert snapshot.holdings[0].weight == pytest.approx(0.3)
    assert snapshot.sector_weights == pytest.approx({"Technology": 0.4, "Unknown": 0.05})
    assert snapshot.cash_weight == pytest.approx(0.1)


def test_concentration_warnings_cover_sector_position_and_unknown() -> None:
    warnings = concentration_warnings(_snapshot(), LIMITS)
    assert warnings == [
        "Technology sector concentration 35.0% exceeds 30% ISA limit threshold.",
        "NVDA_US_EQ position 20.0% exceeds 17.5% single-position limit.",
        "10.0% of the ISA has no sector classification (ETFs or failed lookups) "
        "and is excluded from sector limit checks.",
    ]


def test_no_warnings_when_within_limits() -> None:
    snapshot = PortfolioSnapshot(
        currency="GBP",
        total_value=1000,
        cash_available=900,
        cash_weight=0.9,
        holdings=[_holding("AMD_US_EQ", "AMD", 0.1, "Technology")],
        sector_weights={"Technology": 0.1},
    )
    assert concentration_warnings(snapshot, LIMITS) == []


def test_consistent_trim_has_no_decision_warnings() -> None:
    decision = _decision(_allocation("NVDA", 0.08, Action.TRIM))
    assert (
        decision_warnings(
            decision.proposed_allocations, _snapshot(), LIMITS, REVIEWED, "Technology"
        )
        == []
    )


@pytest.mark.parametrize(
    ("allocation", "expected"),
    [
        (_allocation("NVDA", 0.25, Action.TRIM), "exceeds 17.5%"),
        (_allocation("NVDA", 0.25, Action.TRIM), "inconsistent"),
        (_allocation("NVDA", 0.10, Action.BUY), "inconsistent"),
        (_allocation("NVDA", 0.05, Action.SELL), "inconsistent"),
        (_allocation("TSLA_US_EQ", 0.05, Action.BUY), "neither under review nor currently held"),
        (_allocation("AMD_US_EQ", 0.17, Action.BUY), "takes Technology sector to 37.0%"),
    ],
)
def test_decision_warnings_flag_bad_proposals(
    allocation: ProposedAllocation, expected: str
) -> None:
    warnings = decision_warnings(
        _decision(allocation).proposed_allocations, _snapshot(), LIMITS, REVIEWED, "Technology"
    )
    assert any(expected in warning for warning in warnings)


def test_new_position_uses_reviewed_sector_for_projection() -> None:
    decision = _decision(_allocation("MU", 0.05, Action.BUY))
    warnings = decision_warnings(
        decision.proposed_allocations, _snapshot(), LIMITS, {"MU"}, "Technology"
    )
    assert warnings == [
        "Proposed BUY of MU takes Technology sector to 40.0%, above the 30% ISA limit threshold."
    ]


def test_position_cap_still_checked_without_portfolio() -> None:
    decision = _decision(_allocation("NVDA", 0.5, Action.BUY))
    warnings = decision_warnings(
        decision.proposed_allocations, None, LIMITS, REVIEWED, "Technology"
    )
    assert warnings == ["Proposed NVDA target weight 50.0% exceeds 17.5% single-position limit."]
