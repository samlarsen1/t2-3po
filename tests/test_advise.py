"""Strategy advice: config readers, figures, validation, memory and the advise command."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile

from t2_3po import cli
from t2_3po.advise import (
    AdviceContext,
    SuggestionStore,
    check_draft,
    finalize,
    render_advice,
    render_context,
    suggestion_id,
    threshold_problem,
)
from t2_3po.configs import ConfigReadError, load_d2, load_k2
from t2_3po.instruments import Instrument, Instruments, InstrumentsError, ResolveError
from t2_3po.models import AdvisorOutput, ReviewerOutput, SuggestionDraft
from t2_3po.prices import PriceHistory, StockStats, dip_days
from t2_3po.runlog import d2_activity, k2_latest, load_reports

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)

STOCKS_YAML = """
markets:
  uk:
    currency: GBP
    tickers:
      - {symbol: RIO, t212_ticker: RIOl_EQ, name: Rio Tinto, strategy: mild}
  us:
    currency: USD
    tickers:
      - {symbol: NVDA, t212_ticker: NVDA_US_EQ, name: Nvidia, strategy: mild}
  europe:
    currency: EUR
    tickers:
      - {symbol: ASML, t212_ticker: ASMLa_EQ, name: ASML, strategy: deep_value}
watchlist:
  - {symbol: SPY}
"""

STRATEGY_YAML = """
modes:
  deep_value:
    description: Buys long slides only.
    trigger: long_term_downtrend
    max_pct_above_52w_low: 2.0
    max_portfolio_weight_pct: 7.5
  mild:
    description: >
      Conservative DCA.
    min_daily_dip_pct: -3.0
    min_weekly_dip_pct: -4.0
    require_both_timeframes: true
    max_portfolio_weight_pct: 10.0
    order_type: limit
  moderate:
    description: Balanced DCA.
    min_daily_dip_pct: -2.0
    max_portfolio_weight_pct: 12.5
global_limits:
  max_trades_per_run: 3
"""

K2_YAML = """
overbought:
  rsi_period: 14
  rsi_min: 70.0
  min_signals: 2
harvest:
  min_gain_pct: 25.0
  gain_fraction: 0.5
  order_type: limit
"""

INSTRUMENTS = [
    {
        "ticker": "RIOl_EQ",
        "shortName": "RIO",
        "name": "Rio Tinto",
        "type": "STOCK",
        "currencyCode": "GBX",
        "exchangeName": "London Stock Exchange",
    },
    {
        "ticker": "SHELl_EQ",
        "shortName": "SHEL",
        "name": "Shell",
        "type": "STOCK",
        "currencyCode": "GBX",
        "exchangeName": "London Stock Exchange",
    },
    {
        "ticker": "SHEL_US_EQ",
        "shortName": "SHEL",
        "name": "Shell ADR",
        "type": "STOCK",
        "currencyCode": "USD",
        "exchangeName": "NYSE",
    },
    {
        "ticker": "NVDA_US_EQ",
        "shortName": "NVDA",
        "name": "Nvidia",
        "type": "STOCK",
        "currencyCode": "USD",
        "exchangeName": "NASDAQ",
    },
    {
        "ticker": "NVDd_EQ",
        "shortName": "NVDA",
        "name": "Nvidia",
        "type": "STOCK",
        "currencyCode": "EUR",
        "exchangeName": "Deutsche Börse Xetra",
    },
    {
        "ticker": "ASMLa_EQ",
        "shortName": "ASML",
        "name": "ASML",
        "type": "STOCK",
        "currencyCode": "EUR",
        "exchangeName": "Euronext Amsterdam",
    },
    {
        "ticker": "SAPd_EQ",
        "shortName": "SAP",
        "name": "SAP",
        "type": "STOCK",
        "currencyCode": "EUR",
        "exchangeName": "Deutsche Börse Xetra",
    },
    {
        "ticker": "SAPa_EQ",
        "shortName": "SAP",
        "name": "SAP",
        "type": "STOCK",
        "currencyCode": "EUR",
        "exchangeName": "Euronext Amsterdam",
    },
    {
        "ticker": "WARNl_EQ",
        "shortName": "WARN",
        "name": "A warrant",
        "type": "WARRANT",
        "currencyCode": "GBX",
        "exchangeName": "London Stock Exchange",
    },
]


def closes(n: int = 260, dip_every: int = 20) -> list[float]:
    """A rising series with a 4% one-day fall every ``dip_every`` days."""
    series = [100.0]
    for i in range(1, n):
        change = -0.04 if i % dip_every == 0 else 0.003
        series.append(series[-1] * (1 + change))
    return series


def run_report(droid: str, when: datetime, result: dict[str, Any], status: str = "ok") -> str:
    return json.dumps(
        {
            "schema": "t2.run-report/1",
            "droid": droid,
            "version": "0.2.0",
            "command": "x",
            "argv": [],
            "started_at": when.isoformat(),
            "finished_at": when.isoformat(),
            "duration_seconds": 1,
            "dry_run": True,
            "status": status,
            "exit_code": 0,
            "summary": "",
            "result": result,
            "errors": [],
        }
    )


def decision(ticker: str, buy: bool, reason: str, value: float = 0.0) -> dict[str, Any]:
    return {"ticker": ticker, "should_trade": buy, "reason": reason, "trade_value_gbp": value}


@pytest.fixture
def droids(tmp_path: Path) -> Path:
    """t2-d2, t2-k2 and t2-bb8 checkouts side by side, with recent run reports."""
    (tmp_path / "t2-d2/config").mkdir(parents=True)
    (tmp_path / "t2-d2/config/stocks.yaml").write_text(STOCKS_YAML)
    (tmp_path / "t2-d2/config/strategy.yaml").write_text(STRATEGY_YAML)
    (tmp_path / "t2-k2/config").mkdir(parents=True)
    (tmp_path / "t2-k2/config/strategy.yaml").write_text(K2_YAML)
    (tmp_path / "t2-bb8/data").mkdir(parents=True)
    (tmp_path / "t2-bb8/data/instruments.json").write_text(json.dumps(INSTRUMENTS))

    d2_runs = tmp_path / "t2-d2/var/runs"
    d2_runs.mkdir(parents=True)
    recent = NOW - timedelta(days=2)
    (d2_runs / "a.json").write_text(
        run_report(
            "t2-d2",
            recent,
            {
                "decisions": [
                    decision("RIO", False, "RSI 61.2 above max 55"),
                    decision("NVDA", True, "dip"),
                ],
                "trades": [decision("NVDA", True, "dip", 25.0)],
            },
        )
    )
    (d2_runs / "b.json").write_text(
        run_report(
            "t2-d2",
            recent,
            {
                "decisions": [decision("RIO", False, "RSI 58.0 above max 55")],
                "trades": [],
            },
        )
    )
    (d2_runs / "old.json").write_text(
        run_report(
            "t2-d2",
            NOW - timedelta(days=90),
            {
                "decisions": [decision("RIO", True, "dip")],
                "trades": [],
            },
        )
    )
    k2_runs = tmp_path / "t2-k2/var/runs"
    k2_runs.mkdir(parents=True)
    (k2_runs / "k.json").write_text(
        run_report(
            "t2-k2",
            recent,
            {
                "decisions": [
                    {
                        "ticker": "NVDA_US_EQ",
                        "action": "HARVEST",
                        "gainPct": "60",
                        "rsi": 81.0,
                        "pctAboveSma": 22.0,
                        "runUpPct": 30.0,
                        "signals": ["rsi", "sma"],
                    },
                ]
            },
        )
    )
    (tmp_path / "theses/macro").mkdir(parents=True)
    (tmp_path / "theses/macro/core.md").write_text("Pillar 1: Suppliers. Pillar 2: Compute.")
    return tmp_path


# --- Config readers ------------------------------------------------------------------------


def test_load_d2_reads_stocks_markets_watchlist_and_tunables(droids: Path) -> None:
    d2 = load_d2(droids / "t2-d2")
    assert [(t.stock.symbol, t.market) for t in d2.stocks] == [
        ("RIO", "uk"),
        ("NVDA", "us"),
        ("ASML", "europe"),
    ]
    assert d2.watchlist == ["SPY"]
    assert d2.find("rio") is not None and d2.find("NVDA_US_EQ") is not None
    assert d2.lists("SPY", "SPY_US_EQ") and not d2.lists("SHEL", "SHELl_EQ")
    tunables = d2.tunables()
    assert tunables["modes.mild.min_daily_dip_pct"] == -3.0
    assert tunables["modes.deep_value.max_pct_above_52w_low"] == 2.0
    assert "modes.mild.order_type" not in tunables
    assert "modes.mild.require_both_timeframes" not in tunables


def test_load_k2_tunables_skip_non_numeric_settings(droids: Path) -> None:
    assert load_k2(droids / "t2-k2").tunables() == {
        "overbought.rsi_min": 70.0,
        "overbought.min_signals": 2.0,
        "harvest.min_gain_pct": 25.0,
        "harvest.gain_fraction": 0.5,
    }


def test_missing_config_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigReadError):
        load_d2(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/strategy.yaml").write_text("- not a mapping")
    with pytest.raises(ConfigReadError):
        load_k2(tmp_path)


# --- Instruments ---------------------------------------------------------------------------


@pytest.fixture
def instruments() -> Instruments:
    return Instruments([Instrument.model_validate(item) for item in INSTRUMENTS])


@pytest.mark.parametrize(
    ("symbol", "market", "t212", "expected"),
    [
        ("SHEL", "uk", None, "SHELl_EQ"),
        ("shel", "us", None, "SHEL_US_EQ"),
        ("NVDA", "us", None, "NVDA_US_EQ"),
        ("ASML", "europe", None, "ASMLa_EQ"),
        ("whatever", "uk", "SHELl_EQ", "SHELl_EQ"),
    ],
)
def test_resolve_finds_the_one_listing(
    instruments: Instruments, symbol: str, market: str, t212: str | None, expected: str
) -> None:
    assert instruments.resolve(symbol, market, t212).ticker == expected


@pytest.mark.parametrize(
    ("symbol", "market", "t212", "message"),
    [
        ("SAP", "europe", None, "several instruments"),
        ("WARN", "uk", None, "no Trading212 instrument"),
        ("ASML", "us", None, "no Trading212 instrument"),
        ("X", "uk", "NOPE_EQ", "not on Trading212"),
        ("NVDA", "uk", "NVDA_US_EQ", "doesn't trade in the uk market"),
        ("RIO", "asia", None, "unknown market"),
    ],
)
def test_resolve_rejects_unsuitable_or_ambiguous_listings(
    instruments: Instruments, symbol: str, market: str, t212: str | None, message: str
) -> None:
    with pytest.raises(ResolveError, match=message):
        instruments.resolve(symbol, market, t212)


def test_instruments_load_prefers_the_deployed_data_dir(droids: Path) -> None:
    deployed = droids / "t2-bb8/var/data"
    deployed.mkdir(parents=True)
    (deployed / "instruments.json").write_text(json.dumps(INSTRUMENTS[:1]))
    assert len(Instruments.load(droids / "t2-bb8")) == 1
    with pytest.raises(InstrumentsError):
        Instruments.load(droids / "nowhere")


# --- Figures -------------------------------------------------------------------------------


def test_stock_stats() -> None:
    stats = StockStats.from_closes(closes())
    assert stats is not None
    assert stats.closes == 260
    assert stats.above_sma200 is True
    # Annualised and in percent: about 14% for this series, not a fraction such as 0.14.
    assert 5 < stats.volatility_pct < 50
    assert stats.return_1y_pct == pytest.approx((closes()[-1] / 100 - 1) * 100)
    assert StockStats.from_closes([1.0] * 10) is None
    assert StockStats.from_closes([*closes(), math.nan]) is None


def test_dip_days_follow_each_modes_rule() -> None:
    series = closes()  # 12 four-percent falls, each in a week that is otherwise rising
    assert dip_days(series, {"min_daily_dip_pct": -3.0}) == 12
    assert dip_days(series, {"min_daily_dip_pct": -5.0}) == 0
    both = {"min_daily_dip_pct": -3.0, "min_weekly_dip_pct": -4.0, "require_both_timeframes": True}
    assert dip_days(series, both) == 0  # the week as a whole falls less than 4%
    assert dip_days(series, {"trigger": "long_term_downtrend"}) is None


def test_activity_counts_recent_ok_runs_only(droids: Path) -> None:
    reports = load_reports(droids / "t2-d2/var/runs", "t2-d2", NOW - timedelta(days=30))
    activity = d2_activity(reports)
    assert activity.sessions == 2
    rio = activity.by_symbol["RIO"]
    assert (rio.evaluated, rio.signalled, rio.bought) == (2, 0, 0)
    assert rio.top_reasons() == [("RSI above max", 2)]
    nvda = activity.by_symbol["NVDA"]
    assert (nvda.signalled, nvda.bought, nvda.deployed_gbp) == (1, 1, 25.0)
    k2 = k2_latest(load_reports(droids / "t2-k2/var/runs", "t2-k2", NOW - timedelta(days=30)))
    assert k2[0]["action"] == "HARVEST"
    assert load_reports(droids / "missing", "t2-d2", NOW) == []


# --- Validation ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "current", "proposed", "problem"),
    [
        ("modes.mild.min_daily_dip_pct", -3.0, -2.5, None),
        ("modes.mild.min_daily_dip_pct", -3.0, 2.0, "would change sign"),
        ("modes.mild.min_daily_dip_pct", -3.0, -5.0, "more than 50%"),
        ("modes.mild.min_daily_dip_pct", -3.0, -3.0, "same as the current value"),
        ("overbought.min_signals", 2.0, 3.0, None),
        ("overbought.min_signals", 2.0, 2.5, "must be 1, 2 or 3"),
        ("harvest.gain_fraction", 0.5, 1.0, None),
        ("harvest.gain_fraction", 0.5, 1.2, "at most 1"),
        ("harvest.min_gain_pct", 25.0, math.inf, "not a finite number"),
    ],
)
def test_threshold_problem(key: str, current: float, proposed: float, problem: str | None) -> None:
    found = threshold_problem(key, current, proposed)
    if problem is None:
        assert found is None
    else:
        assert found is not None and problem in found


def draft(**fields: Any) -> SuggestionDraft:
    base = {
        "droid": "t2-d2",
        "rationale": "Why.",
        "evidence": ["RIO: 12 dip days"],
        "expected_effect": "Buys more often.",
        "risk": "More cash deployed.",
        "confidence": "MEDIUM",
    }
    return SuggestionDraft.model_validate({**base, **fields})


def context(droids: Path, instruments: Instruments | None = None) -> AdviceContext:
    return AdviceContext(
        droids=["t2-d2", "t2-k2"],
        thesis="Suppliers and compute.",
        d2=load_d2(droids / "t2-d2"),
        k2=load_k2(droids / "t2-k2"),
        instruments=instruments,
        stats={"RIO": StockStats.from_closes(closes()), "NVDA": None},
        dip_days={"RIO": {"mild": 0, "moderate": 12, "deep_value": None}},
        activity=d2_activity(
            load_reports(droids / "t2-d2/var/runs", "t2-d2", NOW - timedelta(days=30))
        ),
        k2_decisions=[],
        weights={"RIOl_EQ": 0.04},
        cash_weight=0.1,
        past=[],
        lookback_days=30,
    )


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"kind": "change_mode", "symbol": "RIO", "proposed": "turbo"}, "not a t2-d2 mode"),
        ({"kind": "change_mode", "symbol": "RIO", "proposed": "mild"}, "already uses"),
        ({"kind": "change_mode", "symbol": "TSLA", "proposed": "mild"}, "not a traded stock"),
        ({"kind": "remove_holding", "symbol": "X", "proposed": "watchlist"}, "not a traded stock"),
        (
            {"kind": "change_mode", "droid": "t2-k2", "symbol": "RIO", "proposed": "moderate"},
            "t2-d2's stocks only",
        ),
        (
            {
                "kind": "change_threshold",
                "key": "global_limits.max_trades_per_run",
                "proposed": "5",
            },
            "not an adjustable t2-d2 setting",
        ),
        (
            {"kind": "change_threshold", "key": "modes.mild.min_daily_dip_pct", "proposed": "lots"},
            "not a number",
        ),
        (
            {"kind": "add_holding", "symbol": "SHEL", "market": "uk", "proposed": "mild"},
            "instrument list is unavailable",
        ),
        (
            {"kind": "add_holding", "symbol": "SHEL", "proposed": "mild"},
            "needs a symbol and a market",
        ),
    ],
)
def test_check_draft_rejects(droids: Path, fields: dict[str, Any], reason: str) -> None:
    result = check_draft(draft(**fields), context(droids), NOW)
    assert isinstance(result, str) and reason in result


def test_check_draft_accepts_each_kind(droids: Path, instruments: Instruments) -> None:
    ctx = context(droids, instruments)

    mode = check_draft(draft(kind="change_mode", symbol="rio", proposed="moderate"), ctx, NOW)
    assert not isinstance(mode, str)
    s = mode.suggestion
    assert (s.file, s.target, s.current, s.proposed) == (
        "config/stocks.yaml",
        "RIO",
        "mild",
        "moderate",
    )
    assert s.id == suggestion_id("change_mode", "t2-d2", "RIO", "moderate")

    added = check_draft(
        draft(kind="add_holding", symbol="SHEL", market="uk", proposed="mild"), ctx, NOW
    )
    assert not isinstance(added, str)
    assert added.suggestion.details["t212_ticker"] == "SHELl_EQ"
    assert added.yahoo_symbol == "SHEL.L"

    listed = check_draft(
        draft(kind="add_holding", symbol="NVDA", market="us", proposed="mild"), ctx, NOW
    )
    assert isinstance(listed, str) and "already in" in listed

    threshold = check_draft(
        draft(kind="change_threshold", droid="t2-k2", key="overbought.rsi_min", proposed="75"),
        ctx,
        NOW,
    )
    assert not isinstance(threshold, str)
    assert (threshold.suggestion.current, threshold.suggestion.proposed) == ("70", "75")
    assert threshold.suggestion.file == "config/strategy.yaml"


def test_render_context_holds_code_figures_but_no_account_values(droids: Path) -> None:
    text = render_context(context(droids))
    assert "RIO (RIOl_EQ, uk) mode=mild | weight=4.0%" in text
    assert "dip days in the last year by mode: mild=0, moderate=12, deep_value=n/a" in text
    assert "bought 0x" in text and "'RSI above max' x2" in text
    assert "NVDA (NVDA_US_EQ, us) mode=mild | weight=0.0% | prices: unavailable" in text
    assert "t2-k2 adjustable keys: overbought.rsi_min (now 70)" in text
    assert "£" not in text.split("<portfolio>")[1].split("</portfolio>")[0]


# --- Review, dedupe, ranking ---------------------------------------------------------------


def verdicts(*keeps: tuple[bool, str]) -> ReviewerOutput:
    return ReviewerOutput.model_validate(
        {
            "verdicts": [
                {"index": i, "keep": keep, "confidence": conf, "note": f"note {i}"}
                for i, (keep, conf) in enumerate(keeps, start=1)
            ]
        }
    )


def test_finalize_applies_review_checks_rank_and_cap(
    droids: Path, instruments: Instruments
) -> None:
    drafts = AdvisorOutput(
        no_change="Keep the rest.",
        suggestions=[
            draft(kind="change_mode", symbol="RIO", proposed="moderate"),
            draft(kind="change_threshold", droid="t2-k2", key="overbought.rsi_min", proposed="75"),
            draft(kind="change_mode", symbol="NVDA", proposed="turbo"),
            draft(kind="remove_holding", symbol="ASML", proposed="watchlist"),
            draft(kind="add_holding", symbol="SHEL", market="uk", proposed="mild"),
        ],
    )
    review = verdicts(
        (True, "LOW"), (True, "HIGH"), (True, "HIGH"), (False, "LOW"), (True, "MEDIUM")
    )
    run = finalize(
        drafts,
        review,
        context(droids, instruments),
        {"SHEL.L": StockStats.from_closes(closes())},
        NOW,
        2,
        90,
    )

    assert [s.target for s in run.suggestions] == ["overbought.rsi_min", "SHELl_EQ"]
    assert run.suggestions[0].confidence == "HIGH"
    assert run.suggestions[0].reviewer_note == "note 2"
    reasons = {item.draft.symbol or item.draft.key: item.reason for item in run.dropped}
    assert "not a t2-d2 mode" in reasons["NVDA"]
    assert reasons["ASML"] == "reviewer: note 4"
    assert reasons["RIO"] == "over the cap of 2"


def test_finalize_drops_new_holdings_yahoo_doesnt_know(
    droids: Path, instruments: Instruments
) -> None:
    drafts = AdvisorOutput(
        no_change="",
        suggestions=[
            draft(kind="add_holding", symbol="SHEL", market="uk", proposed="mild"),
        ],
    )
    run = finalize(
        drafts, verdicts((True, "HIGH")), context(droids, instruments), {"SHEL.L": None}, NOW, 5, 90
    )
    assert run.suggestions == []
    assert run.dropped[0].reason == "no usable Yahoo price history"


def test_store_remembers_answers(droids: Path, tmp_path: Path) -> None:
    ctx = context(droids)
    drafts = AdvisorOutput(
        no_change="",
        suggestions=[
            draft(kind="change_mode", symbol="RIO", proposed="moderate"),
            draft(kind="change_threshold", droid="t2-k2", key="overbought.rsi_min", proposed="75"),
        ],
    )
    store = SuggestionStore(tmp_path / "suggestions")
    first = finalize(drafts, verdicts((True, "HIGH"), (True, "HIGH")), ctx, {}, NOW, 5, 90)
    store.write_run(first)
    rio, k2 = (s.id for s in first.suggestions)
    store.decide(k2, "rejected", "too few harvests already", NOW)
    with pytest.raises(KeyError):
        store.decide("deadbeef", "accepted", "", NOW)

    later = NOW + timedelta(days=7)
    ctx.past = store.past(later - timedelta(days=90))
    second = finalize(drafts, verdicts((True, "HIGH"), (True, "HIGH")), ctx, {}, later, 5, 90)
    assert [s.id for s in second.suggestions] == [rio]
    assert second.suggestions[0].repeat is True
    assert second.dropped[0].reason == "rejected on 2026-11-16"

    much_later = NOW + timedelta(days=120)
    ctx.past = store.past(much_later - timedelta(days=200))
    third = finalize(drafts, verdicts((True, "HIGH"), (True, "HIGH")), ctx, {}, much_later, 5, 90)
    assert {s.id for s in third.suggestions} == {rio, k2}  # the rejection has expired

    report = render_advice(second, ctx)
    assert "Move RIO from `mild` to `moderate`" in report
    assert "suggested before, still open" in report
    assert "rejected on 2026-11-16" in report


# --- The advise and suggestions commands ---------------------------------------------------


ADVISOR = {
    "no_change": "ASML stays on deep_value.",
    "suggestions": [
        {
            "kind": "change_mode",
            "droid": "t2-d2",
            "symbol": "RIO",
            "proposed": "moderate",
            "rationale": "mild never fires.",
            "evidence": ["RIO: mild=0, moderate=12 dip days"],
            "expected_effect": "Buys RIO on 2% dips.",
            "risk": "More RIO.",
            "confidence": "HIGH",
        },
        {
            "kind": "add_holding",
            "droid": "t2-d2",
            "symbol": "SHEL",
            "market": "uk",
            "proposed": "mild",
            "rationale": "Energy gap.",
            "evidence": ["No energy holding"],
            "expected_effect": "Covers energy.",
            "risk": "Oil price.",
            "confidence": "MEDIUM",
        },
        {
            "kind": "change_threshold",
            "droid": "t2-d2",
            "key": "modes.mild.min_daily_dip_pct",
            "proposed": "1.0",
            "rationale": "x",
            "evidence": [],
            "expected_effect": "x",
            "risk": "x",
            "confidence": "LOW",
        },
    ],
}
REVIEWER = {
    "verdicts": [
        {"index": 1, "keep": True, "confidence": "HIGH", "note": "Supported."},
        {"index": 2, "keep": True, "confidence": "MEDIUM", "note": "Fills a gap."},
        {"index": 3, "keep": True, "confidence": "LOW", "note": "Weak."},
    ]
}


class Agents:
    def __init__(self) -> None:
        self.prompts: dict[str, str] = {}
        self.model = FunctionModel(
            self._respond, profile=ModelProfile(supports_json_schema_output=True)
        )

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = str(messages[-1].parts[0].content)  # type: ignore[union-attr]
        name = (
            "advisor" if "You are the Strategy Analyst" in (info.instructions or "") else "reviewer"
        )
        self.prompts[name] = prompt
        output = ADVISOR if name == "advisor" else REVIEWER
        return ModelResponse(parts=[TextPart(json.dumps(output))])


def run_cli(droids: Path, monkeypatch: pytest.MonkeyPatch, agents: Agents, *args: str) -> int:
    def fetch(symbol: str) -> list[float]:
        if symbol not in {"RIO.L", "NVDA", "ASML.AS", "SHEL.L"}:
            raise RuntimeError("unknown")
        return closes()

    app = droids / "t2-3po"
    app.mkdir(exist_ok=True)
    monkeypatch.chdir(app)
    for name, value in {
        "D2_DIR": "../t2-d2",
        "K2_DIR": "../t2-k2",
        "BB8_DIR": "../t2-bb8",
        "MACRO_THESIS_FILE": "../theses/macro/core.md",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(cli, "build_model", lambda _: agents.model)
    monkeypatch.setattr(cli, "PriceHistory", lambda: PriceHistory(fetcher=fetch))
    return cli.main([*args])


def test_advise_writes_suggestions_and_a_report(
    droids: Path,
    monkeypatch: pytest.MonkeyPatch,
    t2_contract_dirs: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agents = Agents()
    assert run_cli(droids, monkeypatch, agents, "advise", "--no-portfolio") == 0

    assert "<t2_d2_stocks sessions_recorded=2>" in agents.prompts["advisor"]
    assert "<draft_suggestions>" in agents.prompts["reviewer"]
    app = droids / "t2-3po"
    [saved] = (app / "var/suggestions").glob("*_advice.json")
    run = json.loads(saved.read_text())
    assert [s["target"] for s in run["suggestions"]] == ["RIO", "SHELl_EQ"]
    assert "would change sign" in run["dropped"][0]["reason"]
    [report] = (app / "reports").glob("*_advice.md")
    assert "Add SHEL (Shell) as `mild`" in report.read_text()

    [record] = (t2_contract_dirs / "runs").glob("*-advise.json")
    data = json.loads(record.read_text())
    assert data["status"] == "ok"
    assert [s["kind"] for s in data["result"]["suggestions"]] == ["change_mode", "add_holding"]
    assert data["result"]["suggestions_file"].endswith("_advice.json")

    rio = run["suggestions"][0]["id"]
    capsys.readouterr()
    assert run_cli(droids, monkeypatch, agents, "suggestions") == 0
    assert rio in capsys.readouterr().out
    assert run_cli(droids, monkeypatch, agents, "suggestions", "accept", rio, "--note", "done") == 0
    assert run_cli(droids, monkeypatch, agents, "suggestions") == 0
    assert rio not in capsys.readouterr().out.split("accepted:")[-1]
    assert run_cli(droids, monkeypatch, agents, "suggestions", "--all") == 0
    assert "(done)" in capsys.readouterr().out
    assert run_cli(droids, monkeypatch, agents, "suggestions", "reject", "nope") == 2


def test_advise_dry_run_records_nothing(
    droids: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(droids, monkeypatch, Agents(), "advise", "--no-portfolio", "--dry-run") == 0
    assert "# Strategy advice" in capsys.readouterr().out
    assert not (droids / "t2-3po/var/suggestions").exists()
    assert not (droids / "t2-3po/reports").exists()


def test_advise_only_k2_skips_d2(droids: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agents = Agents()
    run_cli(droids, monkeypatch, agents, "advise", "--no-portfolio", "--only", "k2", "--dry-run")
    assert "<t2_d2_stocks" not in agents.prompts["advisor"]
    assert "<t2_k2_last_run>" in agents.prompts["advisor"]


def test_advise_without_d2_config_is_a_config_error(
    droids: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (droids / "t2-d2/config/stocks.yaml").unlink()
    assert run_cli(droids, monkeypatch, Agents(), "advise", "--no-portfolio") == 2
