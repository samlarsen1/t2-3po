"""A year of daily closes, and the statistics the strategy advice is based on.

Every figure the advisor sees comes from here or from the droids' run reports, not from a model.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)

CloseFetcher = Callable[[str], list[float]]

TRADING_DAYS = 252
WEEK = 5
_MIN_CLOSES = 30


def _fetch_closes(symbol: str) -> list[float]:
    history = yf.Ticker(symbol).history(period="1y", auto_adjust=True)
    return [float(value) for value in history["Close"].dropna()]


class PriceHistory:
    """Async wrapper around yfinance's daily history, which is synchronous."""

    def __init__(self, fetcher: CloseFetcher = _fetch_closes, concurrency: int = 5) -> None:
        self._fetcher = fetcher
        self._semaphore = asyncio.Semaphore(concurrency)

    async def closes(self, symbol: str) -> list[float] | None:
        """A year of daily closes, oldest first, or ``None`` if the lookup fails."""
        async with self._semaphore:
            try:
                return await asyncio.to_thread(self._fetcher, symbol)
            # yfinance raises many exception types, including from its HTTP and parsing layers.
            except Exception as exc:
                error = type(exc).__name__
                logger.warning("price_history_failed symbol=%s error=%s", symbol, error)
                return None

    async def many(self, symbols: Iterable[str]) -> dict[str, list[float] | None]:
        unique = sorted(set(symbols))
        found = await asyncio.gather(*(self.closes(symbol) for symbol in unique))
        return dict(zip(unique, found, strict=True))


@dataclass(frozen=True)
class StockStats:
    """Price behaviour over the last year."""

    closes: int
    return_1y_pct: float
    volatility_pct: float
    drawdown_pct: float
    above_sma200: bool | None

    @classmethod
    def from_closes(cls, closes: list[float]) -> StockStats | None:
        """Compute the stats, or ``None`` if there are too few usable closes."""
        if len(closes) < _MIN_CLOSES or any(c <= 0 or not math.isfinite(c) for c in closes):
            return None
        daily = _changes(closes, 1)
        mean = sum(daily) / len(daily)
        variance = sum((r - mean) ** 2 for r in daily) / max(1, len(daily) - 1)
        sma200 = sum(closes[-200:]) / 200 if len(closes) >= 200 else None
        return cls(
            closes=len(closes),
            return_1y_pct=(closes[-1] / closes[0] - 1) * 100,
            volatility_pct=math.sqrt(variance) * math.sqrt(TRADING_DAYS) * 100,
            drawdown_pct=(1 - closes[-1] / max(closes)) * 100,
            above_sma200=None if sma200 is None else closes[-1] > sma200,
        )


def _changes(closes: list[float], step: int) -> list[float]:
    """Fractional change over ``step`` days, for each day that has that much history."""
    return [closes[i] / closes[i - step] - 1 for i in range(step, len(closes))]


def dip_days(closes: list[float], mode: dict[str, Any]) -> int | None:
    """Days in the history when a daily-dip mode's price rule would have fired.

    Approximates t2-d2's rule: the daily fall reaches ``min_daily_dip_pct`` and, when the mode
    requires both timeframes, the five-day fall reaches ``min_weekly_dip_pct``. It ignores the
    RSI, trend and cash checks, so real buys are rarer. ``None`` for modes without a daily rule,
    such as ``deep_value``.
    """
    daily_limit = mode.get("min_daily_dip_pct")
    if not isinstance(daily_limit, int | float) or len(closes) <= WEEK:
        return None
    raw_weekly = mode.get("min_weekly_dip_pct")
    weekly_limit = float(raw_weekly) if isinstance(raw_weekly, int | float) else None
    if not mode.get("require_both_timeframes"):
        weekly_limit = None
    count = 0
    for i in range(WEEK, len(closes)):
        day = (closes[i] / closes[i - 1] - 1) * 100
        week = (closes[i] / closes[i - WEEK] - 1) * 100
        if day <= daily_limit and (weekly_limit is None or week <= weekly_limit):
            count += 1
    return count
