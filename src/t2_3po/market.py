"""Yahoo Finance fundamentals via yfinance. No API key is needed."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Iterable
from typing import Any

import yfinance as yf
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Exchange letter at the end of a Trading212 code (RIOl_EQ) -> Yahoo suffix (RIO.L)
_T212_EXCHANGE_SUFFIX = {"l": ".L", "d": ".DE", "a": ".AS", "p": ".PA"}
_T212_EXCHANGE_CODE = re.compile(r"^(?P<symbol>[A-Z0-9.]+?)(?P<exchange>[a-z])_EQ$")
_MINOR_UNIT_CURRENCIES = {"GBp": "GBP", "GBX": "GBP"}

InfoFetcher = Callable[[str], dict[str, Any]]


class MarketDataError(Exception):
    """Yahoo Finance returned no usable data for a symbol."""


class Fundamentals(BaseModel):
    """Fundamentals for one symbol. ``None`` means Yahoo didn't report that figure.

    Per-share and valuation figures are in ``currency``. Cash, debt, cash flow and net income are
    in ``financial_currency``, which can differ, e.g. RIO.L is quoted in GBp and reports in USD.
    """

    symbol: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    currency: str | None = None
    financial_currency: str | None = None
    price: float | None = None
    market_cap: float | None = None
    trailing_pe: float | None = None
    forward_pe: float | None = None
    price_to_book: float | None = None
    debt_to_equity: float | None = None
    fcf_yield: float | None = None
    current_ratio: float | None = None
    return_on_equity: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    profit_margin: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    trailing_eps: float | None = None
    forward_eps: float | None = None
    total_cash: float | None = None
    total_debt: float | None = None
    operating_cash_flow: float | None = None
    free_cash_flow: float | None = None
    net_income: float | None = None
    consensus_target_price: float | None = None
    consensus_recommendation: str | None = None
    analyst_count: int | None = None
    week52_high: float | None = None
    week52_low: float | None = None
    sma_200d: float | None = None


def to_yahoo_symbol(t212_ticker: str) -> str | None:
    """Map a Trading212 instrument code to a Yahoo symbol.

    Args:
        t212_ticker: A code such as ``NVDA_US_EQ``, ``BRK_B_US_EQ`` or ``RIOl_EQ``.

    Returns:
        The Yahoo symbol, or ``None`` for a format or exchange this mapping doesn't know.
    """
    if t212_ticker.endswith("_US_EQ"):
        return t212_ticker.removesuffix("_US_EQ").replace("_", "-")
    match = _T212_EXCHANGE_CODE.match(t212_ticker)
    if match is None or match["exchange"] not in _T212_EXCHANGE_SUFFIX:
        return None
    return match["symbol"].rstrip(".") + _T212_EXCHANGE_SUFFIX[match["exchange"]]


def _fcf_yield(info: dict[str, Any]) -> float | None:
    free_cash_flow, market_cap = info.get("freeCashflow"), info.get("marketCap")
    if not free_cash_flow or not market_cap:
        return None
    # Companies can report in a different currency to their listing, e.g. RIO.L reports in USD.
    quote = str(info.get("currency"))
    if info.get("financialCurrency") != _MINOR_UNIT_CURRENCIES.get(quote, quote):
        return None
    return float(free_cash_flow) / float(market_cap)


def fundamentals_from_info(symbol: str, info: dict[str, Any]) -> Fundamentals:
    """Convert a yfinance ``info`` dict, normalising debt/equity from a percentage to a ratio.

    Raises:
        MarketDataError: If the dict has neither a price nor a market cap.
    """
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if price is None and info.get("marketCap") is None:
        raise MarketDataError(f"Yahoo Finance returned no usable data for {symbol}")
    debt_to_equity = info.get("debtToEquity")
    return Fundamentals(
        symbol=symbol,
        name=info.get("longName") or info.get("shortName"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        currency=info.get("currency"),
        financial_currency=info.get("financialCurrency"),
        price=price,
        market_cap=info.get("marketCap"),
        trailing_pe=info.get("trailingPE"),
        forward_pe=info.get("forwardPE"),
        price_to_book=info.get("priceToBook"),
        debt_to_equity=None if debt_to_equity is None else debt_to_equity / 100,
        fcf_yield=_fcf_yield(info),
        current_ratio=info.get("currentRatio"),
        return_on_equity=info.get("returnOnEquity"),
        gross_margin=info.get("grossMargins"),
        operating_margin=info.get("operatingMargins"),
        profit_margin=info.get("profitMargins"),
        revenue_growth=info.get("revenueGrowth"),
        earnings_growth=info.get("earningsGrowth"),
        trailing_eps=info.get("trailingEps"),
        forward_eps=info.get("forwardEps"),
        total_cash=info.get("totalCash"),
        total_debt=info.get("totalDebt"),
        operating_cash_flow=info.get("operatingCashflow"),
        free_cash_flow=info.get("freeCashflow"),
        net_income=info.get("netIncomeToCommon"),
        consensus_target_price=info.get("targetMeanPrice"),
        consensus_recommendation=info.get("recommendationKey"),
        analyst_count=info.get("numberOfAnalystOpinions"),
        week52_high=info.get("fiftyTwoWeekHigh"),
        week52_low=info.get("fiftyTwoWeekLow"),
        sma_200d=info.get("twoHundredDayAverage"),
    )


def _fetch_yahoo_info(symbol: str) -> dict[str, Any]:
    return dict(yf.Ticker(symbol).info)


class MarketData:
    """Async wrapper around yfinance, which is synchronous."""

    def __init__(self, fetcher: InfoFetcher = _fetch_yahoo_info, concurrency: int = 5) -> None:
        self._fetcher = fetcher
        self._semaphore = asyncio.Semaphore(concurrency)

    async def fundamentals(self, symbol: str) -> Fundamentals:
        """Fetch fundamentals for one symbol.

        Raises:
            MarketDataError: If the lookup fails or returns nothing usable.
        """
        async with self._semaphore:
            try:
                info = await asyncio.to_thread(self._fetcher, symbol)
            # yfinance raises many exception types, including from its HTTP and parsing layers.
            except Exception as exc:
                raise MarketDataError(
                    f"Yahoo Finance lookup failed for {symbol} ({type(exc).__name__})"
                ) from exc
        return fundamentals_from_info(symbol, info)

    async def sectors(self, symbols: Iterable[str]) -> dict[str, str | None]:
        """Look up sectors. A failed lookup is logged and maps to ``None``."""

        async def lookup(symbol: str) -> tuple[str, str | None]:
            try:
                return symbol, (await self.fundamentals(symbol)).sector
            except MarketDataError as exc:
                logger.warning("sector_lookup_failed symbol=%s error=%s", symbol, exc)
                return symbol, None

        return dict(await asyncio.gather(*(lookup(symbol) for symbol in set(symbols))))
