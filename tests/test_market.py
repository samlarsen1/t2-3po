import math
from typing import Any

import pandas as pd
import pytest

from t2_3po.market import (
    ANNUAL_KEY,
    MarketData,
    MarketDataError,
    annual_figures,
    fundamentals_from_info,
    to_yahoo_symbol,
)

NVDA_INFO: dict[str, Any] = {
    "longName": "NVIDIA Corporation",
    "sector": "Technology",
    "currency": "USD",
    "financialCurrency": "USD",
    "currentPrice": 200.0,
    "marketCap": 1000.0,
    "freeCashflow": 50.0,
    "debtToEquity": 16.9,
    "forwardPE": 25.0,
}


@pytest.mark.parametrize(
    ("t212", "yahoo"),
    [
        ("NVDA_US_EQ", "NVDA"),
        ("BRK_B_US_EQ", "BRK-B"),
        ("RIOl_EQ", "RIO.L"),
        ("NGl_EQ", "NG.L"),
        ("ASMLa_EQ", "ASML.AS"),
        ("SAPd_EQ", "SAP.DE"),
        ("MCp_EQ", "MC.PA"),
        ("XYZq_EQ", None),
        ("garbage", None),
    ],
)
def test_ticker_mapping(t212: str, yahoo: str | None) -> None:
    assert to_yahoo_symbol(t212) == yahoo


def test_normalises_debt_to_equity_and_computes_fcf_yield() -> None:
    fundamentals = fundamentals_from_info("NVDA", NVDA_INFO)
    assert fundamentals.debt_to_equity == pytest.approx(0.169)
    assert fundamentals.fcf_yield == pytest.approx(0.05)
    assert fundamentals.trailing_pe is None


def test_passes_through_margins_and_financial_statement_figures() -> None:
    info = {
        **NVDA_INFO,
        "grossMargins": 0.7467,
        "operatingMargins": 0.6624,
        "trailingEps": 7.91,
        "totalCash": 62.0,
        "totalDebt": 39.0,
        "operatingCashflow": 134.0,
        "netIncomeToCommon": 193.0,
    }
    fundamentals = fundamentals_from_info("NVDA", info)
    assert fundamentals.gross_margin == 0.7467
    assert fundamentals.operating_margin == 0.6624
    assert fundamentals.trailing_eps == 7.91
    assert fundamentals.total_cash == 62.0
    assert fundamentals.total_debt == 39.0
    assert fundamentals.operating_cash_flow == 134.0
    assert fundamentals.free_cash_flow == 50.0
    assert fundamentals.net_income == 193.0
    assert fundamentals.financial_currency == "USD"


def test_fcf_yield_handles_pence_quotes() -> None:
    info = {**NVDA_INFO, "currency": "GBp", "financialCurrency": "GBP"}
    assert fundamentals_from_info("NG.L", info).fcf_yield == pytest.approx(0.05)


def test_fcf_yield_omitted_on_currency_mismatch() -> None:
    info = {**NVDA_INFO, "currency": "GBp", "financialCurrency": "USD"}
    assert fundamentals_from_info("RIO.L", info).fcf_yield is None


def test_empty_info_raises() -> None:
    with pytest.raises(MarketDataError, match="no usable data"):
        fundamentals_from_info("NOPE", {"trailingPegRatio": None})


async def test_fetch_failure_raises_market_data_error() -> None:
    def fail(_: str) -> dict[str, Any]:
        raise RuntimeError("HTTP 500")

    with pytest.raises(MarketDataError, match="RuntimeError"):
        await MarketData(fetcher=fail).fundamentals("NVDA")


async def test_sectors_tolerates_individual_failures() -> None:
    def fetch(symbol: str) -> dict[str, Any]:
        if symbol == "BAD":
            raise RuntimeError("boom")
        return NVDA_INFO

    assert await MarketData(fetcher=fetch).sectors(["NVDA", "BAD"]) == {
        "NVDA": "Technology",
        "BAD": None,
    }


def _frame(rows: dict[str, list[float]], years: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=pd.to_datetime(years)).T


def test_annual_figures_run_latest_first_and_drop_blank_years() -> None:
    years = ["2025-12-31", "2024-12-31", "2023-12-31"]
    income = _frame(
        {"Total Revenue": [121.0, 110.0, math.nan], "Net Income": [10.0, -5.0, math.nan]}, years
    )
    cashflow = _frame({"Free Cash Flow": [8.0, 3.0, math.nan]}, years)

    rows = annual_figures(income, cashflow)

    assert rows == [
        {"year_end": "2025-12-31", "revenue": 121.0, "net_income": 10.0, "free_cash_flow": 8.0},
        {"year_end": "2024-12-31", "revenue": 110.0, "net_income": -5.0, "free_cash_flow": 3.0},
    ]


def test_history_stats_are_computed_in_code() -> None:
    info = {
        **NVDA_INFO,
        ANNUAL_KEY: [
            {"year_end": "2025", "revenue": 121.0, "net_income": 10.0, "free_cash_flow": 8.0},
            {"year_end": "2024", "revenue": 110.0, "net_income": -5.0, "free_cash_flow": None},
            {"year_end": "2023", "revenue": 100.0, "net_income": 4.0, "free_cash_flow": -1.0},
        ],
    }

    fundamentals = fundamentals_from_info("NVDA", info)

    assert fundamentals.revenue_cagr == pytest.approx(0.1)
    assert fundamentals.profitable_years == "2 of 3"
    assert fundamentals.positive_fcf_years == "1 of 2"
    assert '"revenue_cagr"' in fundamentals.model_dump_json()


def test_no_history_means_no_stats() -> None:
    fundamentals = fundamentals_from_info("NVDA", NVDA_INFO)

    assert fundamentals.annual == []
    assert fundamentals.revenue_cagr is None
    assert fundamentals.profitable_years is None


def test_revenue_cagr_needs_positive_endpoints() -> None:
    # Investment trusts can report negative revenue from falling asset values.
    info = {
        **NVDA_INFO,
        ANNUAL_KEY: [{"year_end": "2025", "revenue": -12.0}, {"year_end": "2024", "revenue": 50.0}],
    }

    assert fundamentals_from_info("TRIG.L", info).revenue_cagr is None


def test_dividend_yield_is_a_ratio_and_payout_is_dropped_on_a_loss() -> None:
    profitable = {**NVDA_INFO, "dividendYield": 3.88, "payoutRatio": 0.72, "trailingEps": 0.65}
    loss = {**NVDA_INFO, "dividendYield": 9.38, "payoutRatio": 35.5, "trailingEps": -0.05}

    assert fundamentals_from_info("NG.L", profitable).dividend_yield == pytest.approx(0.0388)
    assert fundamentals_from_info("NG.L", profitable).payout_ratio == pytest.approx(0.72)
    assert fundamentals_from_info("TRIG.L", loss).payout_ratio is None
