"""Portfolio snapshot and concentration checks.

These are plain code rather than LLM judgement, so a limit breach is always reported and the
figures in a warning are always computed.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel

from .market import MarketData, to_yahoo_symbol
from .models import Action, ProposedAllocation
from .t212 import ReadOnlyTrading212Client

UNKNOWN_SECTOR = "Unknown"


class Limits(BaseModel):
    max_sector_weight: float
    max_position_weight: float


class Holding(BaseModel):
    t212_ticker: str
    yahoo_symbol: str | None
    name: str | None
    value: float
    weight: float
    sector: str


class PortfolioSnapshot(BaseModel):
    currency: str
    total_value: float
    cash_available: float
    cash_weight: float
    holdings: list[Holding]
    sector_weights: dict[str, float]

    def holding_for(self, *tickers: str | None) -> Holding | None:
        """Find a holding by Trading212 code or Yahoo symbol."""
        wanted = {ticker for ticker in tickers if ticker}
        for holding in self.holdings:
            if holding.t212_ticker in wanted or holding.yahoo_symbol in wanted:
                return holding
        return None


async def fetch_snapshot(client: ReadOnlyTrading212Client, market: MarketData) -> PortfolioSnapshot:
    """Build a snapshot from live positions. Weights are shares of total ISA value."""
    summary = await client.get_account_summary()
    positions = await client.get_positions()

    symbols = {p.instrument.ticker: to_yahoo_symbol(p.instrument.ticker) for p in positions}
    sectors = await market.sectors(symbol for symbol in symbols.values() if symbol)

    total = summary.total_value
    holdings: list[Holding] = []
    sector_weights: dict[str, float] = {}
    for position in sorted(positions, key=lambda p: -p.wallet_impact.current_value):
        ticker = position.instrument.ticker
        symbol = symbols[ticker]
        value = position.wallet_impact.current_value
        weight = value / total if total > 0 else 0.0
        sector = (sectors.get(symbol) if symbol else None) or UNKNOWN_SECTOR
        holdings.append(
            Holding(
                t212_ticker=ticker,
                yahoo_symbol=symbol,
                name=position.instrument.name,
                value=value,
                weight=weight,
                sector=sector,
            )
        )
        sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

    cash = summary.cash.available_to_trade
    return PortfolioSnapshot(
        currency=summary.currency,
        total_value=total,
        cash_available=cash,
        cash_weight=cash / total if total > 0 else 0.0,
        holdings=holdings,
        sector_weights=sector_weights,
    )


def concentration_warnings(snapshot: PortfolioSnapshot, limits: Limits) -> list[str]:
    """Report every sector and position above its limit, and any unclassified holdings."""
    warnings = [
        f"{sector} sector concentration {weight:.1%} exceeds "
        f"{limits.max_sector_weight:.0%} ISA limit threshold."
        for sector, weight in sorted(snapshot.sector_weights.items(), key=lambda kv: -kv[1])
        if sector != UNKNOWN_SECTOR and weight > limits.max_sector_weight
    ]
    warnings += [
        f"{holding.t212_ticker} position {holding.weight:.1%} exceeds "
        f"{limits.max_position_weight:.1%} single-position limit."
        for holding in snapshot.holdings
        if holding.weight > limits.max_position_weight
    ]
    unknown = snapshot.sector_weights.get(UNKNOWN_SECTOR, 0.0)
    if unknown > 0:
        warnings.append(
            f"{unknown:.1%} of the ISA has no sector classification (ETFs or failed lookups) "
            "and is excluded from sector limit checks."
        )
    return warnings


def decision_warnings(
    allocations: Iterable[ProposedAllocation],
    snapshot: PortfolioSnapshot | None,
    limits: Limits,
    reviewed_tickers: set[str],
    reviewed_sector: str | None,
) -> list[str]:
    """Check the model's proposals against the limits and the actual portfolio.

    Args:
        allocations: The proposed allocations.
        snapshot: Current portfolio, or ``None`` if it wasn't evaluated.
        limits: Concentration limits.
        reviewed_tickers: Every code for the ticker under review.
        reviewed_sector: Sector of the ticker under review, if known.

    Returns:
        Warnings for the human reviewer. Proposals are flagged, never silently altered.
    """
    warnings: list[str] = []
    for allocation in allocations:
        ticker = allocation.ticker
        if allocation.target_weight > limits.max_position_weight:
            warnings.append(
                f"Proposed {ticker} target weight {allocation.target_weight:.1%} exceeds "
                f"{limits.max_position_weight:.1%} single-position limit."
            )
        if snapshot is None:
            continue

        is_reviewed = ticker in reviewed_tickers
        holding = snapshot.holding_for(*(reviewed_tickers if is_reviewed else {ticker}))
        if holding is None and not is_reviewed:
            warnings.append(f"Proposed {ticker} is neither under review nor currently held.")
            continue

        current = holding.weight if holding else 0.0
        target = allocation.target_weight
        if (
            (allocation.action == Action.BUY and target <= current)
            or (allocation.action == Action.TRIM and target >= current)
            or (allocation.action == Action.SELL and target > 0)
        ):
            warnings.append(
                f"Proposed {allocation.action} of {ticker} is inconsistent with moving from "
                f"{current:.1%} to {target:.1%}."
            )

        sector = holding.sector if holding else (reviewed_sector if is_reviewed else None)
        if allocation.action == Action.BUY and sector:
            projected = snapshot.sector_weights.get(sector, 0.0) + max(target - current, 0.0)
            if sector != UNKNOWN_SECTOR and projected > limits.max_sector_weight:
                warnings.append(
                    f"Proposed BUY of {ticker} takes {sector} sector to {projected:.1%}, above "
                    f"the {limits.max_sector_weight:.0%} ISA limit threshold."
                )
    return warnings
