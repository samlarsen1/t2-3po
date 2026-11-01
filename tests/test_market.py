from typing import Any

import pytest

from t2_3po.market import MarketData, MarketDataError, fundamentals_from_info, to_yahoo_symbol

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
