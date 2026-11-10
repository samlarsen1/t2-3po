"""Yahoo Finance fundamentals via yfinance. No API key is needed."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, computed_field

logger = logging.getLogger(__name__)

# Exchange letter at the end of a Trading212 code (RIOl_EQ) -> Yahoo suffix (RIO.L)
_T212_EXCHANGE_SUFFIX = {"l": ".L", "d": ".DE", "a": ".AS", "p": ".PA"}
_T212_EXCHANGE_CODE = re.compile(r"^(?P<symbol>[A-Z0-9.]+?)(?P<exchange>[a-z])_EQ$")
_MINOR_UNIT_CURRENCIES = {"GBp": "GBP", "GBX": "GBP"}

InfoFetcher = Callable[[str], dict[str, Any]]


class MarketDataError(Exception):
    """Yahoo Finance returned no usable data for a symbol."""


# Key under which the default fetcher adds annual history to yfinance's ``info`` dict.
ANNUAL_KEY = "_annual"


class AnnualFigures(BaseModel):
    """One financial year, in ``financial_currency``. ``None`` means Yahoo didn't report it."""

    year_end: str
    revenue: float | None = None
    net_income: float | None = None
    free_cash_flow: float | None = None


class Fundamentals(BaseModel):
    """Fundamentals for one symbol. ``None`` means Yahoo didn't report that figure.

    Per-share and valuation figures are in ``currency``. Cash, debt, cash flow, net income and the
    annual history are in ``financial_currency``, which can differ, e.g. RIO.L is quoted in GBp and
    reports in USD.
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
    dividend_yield: float | None = None
    payout_ratio: float | None = None
    annual: list[AnnualFigures] = []

    # Computed here rather than by the model, which makes arithmetic slips.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def revenue_cagr(self) -> float | None:
        """Compound annual revenue growth from the oldest to the latest year in ``annual``.

        ``annual`` runs latest first, so the list index counts years back from the latest.
        """
        revenues = [(i, a.revenue) for i, a in enumerate(self.annual) if a.revenue is not None]
        if len(revenues) < 2:
            return None
        (latest_i, latest), (oldest_i, oldest) = revenues[0], revenues[-1]
        if oldest <= 0 or latest <= 0:
            return None
        return float((latest / oldest) ** (1 / (oldest_i - latest_i)) - 1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def profitable_years(self) -> str | None:
        """How many reported years had positive net income, e.g. ``"3 of 4"``."""
        return _count_positive([a.net_income for a in self.annual])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def positive_fcf_years(self) -> str | None:
        """How many reported years had positive free cash flow, e.g. ``"4 of 4"``."""
        return _count_positive([a.free_cash_flow for a in self.annual])


def _count_positive(values: list[float | None]) -> str | None:
    known = [v for v in values if v is not None]
    if not known:
        return None
    return f"{sum(v > 0 for v in known)} of {len(known)}"


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
        dividend_yield=_dividend_yield(info),
        payout_ratio=_payout_ratio(info),
        annual=[AnnualFigures.model_validate(row) for row in info.get(ANNUAL_KEY, [])],
    )


def _dividend_yield(info: dict[str, Any]) -> float | None:
    # Yahoo reports dividendYield as a percentage. trailingAnnualDividendYield is a ratio, but is
    # wrong by 100x for stocks quoted in pence, so it isn't used.
    value = info.get("dividendYield")
    return None if value is None else float(value) / 100


def _payout_ratio(info: dict[str, Any]) -> float | None:
    # A payout ratio on a loss is meaningless: Yahoo reports 35.5 for TRIG.L, for example.
    value, eps = info.get("payoutRatio"), info.get("trailingEps")
    if value is None or eps is None or eps <= 0:
        return None
    return float(value)


def _row(frame: pd.DataFrame, label: str, column: Any) -> float | None:
    if label not in frame.index or column not in frame.columns:
        return None
    value = frame.at[label, column]
    return None if pd.isna(value) else float(value)


def annual_figures(income: pd.DataFrame, cashflow: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert yfinance's annual income and cash flow statements, latest year first.

    Yahoo pads the oldest column with blanks for some companies, so years with no figures at all
    are dropped.
    """
    years = sorted(set(income.columns) | set(cashflow.columns), reverse=True)
    rows = []
    for year in years:
        row = {
            "revenue": _row(income, "Total Revenue", year),
            "net_income": _row(income, "Net Income", year),
            "free_cash_flow": _row(cashflow, "Free Cash Flow", year),
        }
        if any(value is not None for value in row.values()):
            rows.append({"year_end": pd.Timestamp(year).date().isoformat(), **row})
    return rows


def _fetch_yahoo_info(symbol: str) -> dict[str, Any]:
    ticker = yf.Ticker(symbol)
    info = dict(ticker.info)
    try:
        info[ANNUAL_KEY] = annual_figures(ticker.income_stmt, ticker.cashflow)
    # History helps but isn't essential, so a failure here shouldn't lose the snapshot.
    except Exception as exc:
        logger.warning("annual_history_failed symbol=%s error=%s", symbol, type(exc).__name__)
    return info


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
