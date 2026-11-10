"""The macro review: every stock in t2-d2's stocks file profiled as a vehicle for one thesis."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile

from t2_3po import cli
from t2_3po.agents import VEHICLE_PROMPT
from t2_3po.market import MarketData
from t2_3po.models import Proposal
from t2_3po.stocks import Stock, StocksError, load_stocks, select_stocks

STOCKS_YAML = """
markets:
  uk:
    currency: GBP
    tickers:
      - {symbol: NG., t212_ticker: NGl_EQ, name: National Grid, strategy: mild}
  us:
    currency: USD
    tickers:
      - {symbol: RKLB, t212_ticker: VACQ_US_EQ, name: RocketLab, strategy: moderate}
      - {symbol: HOOD, t212_ticker: HOOD_US_EQ, name: Robinhood, strategy: mild}
watchlist:
  - {symbol: SPY, name: S&P 500 ETF}
"""


def _profile(
    ticker: str, status: str, role: str, pillars: list[str], weight: float, strategy: str | None
) -> dict[str, Any]:
    return {
        "thesis_status": status,
        "thesis_pillars": pillars,
        "pillar_link": f"{ticker} link.",
        "role": role,
        "role_case": f"{ticker} role case.",
        "key_metrics": [{"name": "Revenue CAGR", "value": "12%", "reading": "Steady growth."}],
        "outlook": f"{ticker} outlook.",
        "key_risks": [f"{ticker} risk."],
        # Without the trailing dot, to check the allocation is pinned to the stock's own symbol.
        "proposed_allocation": {
            "ticker": ticker.rstrip("."),
            "target_weight": weight,
            "action": "HOLD",
            "reasoning": "Why.",
        },
        "suggested_strategy": strategy,
        "strategy_reasoning": "Because." if strategy else None,
    }


PROFILES = {
    "NG.": _profile("NG.", "VALIDATED", "STABILITY", ["Suppliers"], 0.06, None),
    "RKLB": _profile("RKLB", "VALIDATED", "GROWTH", ["Space"], 0.05, "aggressive"),
    # Robinhood is already on mild, so suggesting mild isn't a change.
    "HOOD": _profile("HOOD", "REJECTED", "NEITHER", [], 0.0, "mild"),
}

OVERVIEW = {
    "summary": "Space and suppliers are covered.",
    "pillar_coverage": [
        {"pillar": "Space", "stocks": ["RKLB"], "assessment": "One growth vehicle."},
        {"pillar": "Fintech", "stocks": [], "assessment": "Not covered."},
    ],
    "growth_stability_balance": "Balanced.",
    "overlaps": ["None found."],
    "gaps": ["No fintech vehicle."],
}


class Recorder:
    """Scripted review agents that record every agent's instructions and prompt."""

    def __init__(self, overview_fails: bool = False) -> None:
        self.overview_fails = overview_fails
        self.instructions: dict[str, str] = {}
        self.prompts: dict[str, list[str]] = {"vehicle": [], "overview": [], "council": []}
        self.model = FunctionModel(
            self._respond, profile=ModelProfile(supports_json_schema_output=True)
        )

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = str(messages[-1].parts[0].content)  # type: ignore[union-attr]
        text = info.instructions or ""
        if "You are the Vehicle Analyst" in text:
            name, ticker = "vehicle", prompt.split('<thesis ticker="')[1].split('"')[0]
            output: Any = PROFILES[ticker]
        elif "You are the Portfolio Reviewer" in text:
            name, output = "overview", ("not json" if self.overview_fails else OVERVIEW)
        else:
            name, output = "council", COUNCIL[text.split("You are the ")[1].split(" ")[0]]
        self.instructions[name] = text
        self.prompts[name].append(prompt)
        return ModelResponse(
            parts=[TextPart(output if isinstance(output, str) else json.dumps(output))]
        )


ANALYST = {"summary": "A case.", "key_points": ["Point"], "conviction": "MEDIUM"}
COUNCIL: dict[str, Any] = {
    "Bull": ANALYST,
    "Bear": ANALYST,
    "Quantitative": {
        "summary": "Mixed.",
        "valuation_assessment": "Fair.",
        "balance_sheet_assessment": "Strong.",
        "claim_checks": [],
    },
    "Lead": {
        "thesis_status": "VALIDATED",
        "executive_summary": "Summary.",
        "critique": "Critique.",
        "key_risks": [],
        "portfolio_warnings": [],
        "proposed_allocations": [],
        "suggested_config_updates": [],
    },
}


INFO = {"sector": "Utilities", "currentPrice": 100.0, "currency": "USD"}


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorder: Recorder,
    *flags: str,
    fetched: list[str] | None = None,
    failing: str | None = None,
) -> int:
    (tmp_path / "core.md").write_text("Bullish on space and suppliers. Five years plus.")
    (tmp_path / "stocks.yaml").write_text(STOCKS_YAML)

    def fetch(symbol: str) -> dict[str, Any]:
        if fetched is not None:
            fetched.append(symbol)
        if symbol == failing:
            raise RuntimeError("boom")
        return INFO

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "build_model", lambda _: recorder.model)
    monkeypatch.setattr(cli, "MarketData", lambda: MarketData(fetcher=fetch))
    return cli.main(
        ["review", "--thesis", "core.md", "--stocks", "stocks.yaml", "--no-portfolio", *flags]
    )


def _stock(symbol: str, t212_ticker: str) -> Stock:
    return Stock(symbol=symbol, t212_ticker=t212_ticker, name=symbol, strategy="mild")


def test_load_stocks_reads_every_market_and_skips_the_watchlist(tmp_path: Path) -> None:
    path = tmp_path / "stocks.yaml"
    path.write_text(STOCKS_YAML)

    assert [stock.symbol for stock in load_stocks(path)] == ["NG.", "RKLB", "HOOD"]


@pytest.mark.parametrize("content", ["markets: {}", "not: [valid", "markets: {uk: {tickers: [1]}}"])
def test_load_stocks_rejects_unusable_files(tmp_path: Path, content: str) -> None:
    path = tmp_path / "stocks.yaml"
    path.write_text(content)

    with pytest.raises(StocksError):
        load_stocks(path)


@pytest.mark.parametrize(
    ("symbol", "t212_ticker", "yahoo"),
    [
        ("RKLB", "VACQ_US_EQ", "RKLB"),
        ("NVDA", "NVDA_US_EQ", "NVDA"),
        ("NG.", "NGl_EQ", "NG.L"),
        ("ASML", "ASMLa_EQ", "ASML.AS"),
    ],
)
def test_yahoo_symbol(symbol: str, t212_ticker: str, yahoo: str) -> None:
    assert _stock(symbol, t212_ticker).yahoo_symbol == yahoo


def test_select_ignores_case_and_trailing_dot() -> None:
    stocks = [_stock("NG.", "NGl_EQ"), _stock("HOOD", "HOOD_US_EQ")]

    assert select_stocks(stocks, []) == stocks
    assert select_stocks(stocks, ["ng"]) == stocks[:1]


def test_select_rejects_a_typo() -> None:
    with pytest.raises(StocksError, match="HOOOD"):
        select_stocks([_stock("HOOD", "HOOD_US_EQ")], ["HOOOD"])


def test_every_stock_is_profiled_then_reviewed_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder = Recorder()
    fetched: list[str] = []

    assert _run(tmp_path, monkeypatch, recorder, "--dry-run", fetched=fetched) == 0

    assert sorted(fetched) == ["HOOD", "NG.L", "RKLB"]
    assert len(recorder.prompts["vehicle"]) == 3
    assert all("Bullish on space and suppliers." in p for p in recorder.prompts["vehicle"])
    assert any(
        "RocketLab (RKLB), currently on the `moderate` strategy" in p
        for p in recorder.prompts["vehicle"]
    )
    assert recorder.prompts["council"] == []

    [overview_prompt] = recorder.prompts["overview"]
    assert "Bullish on space and suppliers." in overview_prompt
    assert all(f'"symbol": "{s}"' in overview_prompt for s in ("NG.", "RKLB", "HOOD"))

    output = capsys.readouterr().out
    assert "No fintech vehicle." in output
    assert "| Fintech | None | Not covered. |" in output
    table = output[output.index("| Stock | Fit |") :]
    # Same status, so growth comes before stability; the rejected stock comes last.
    assert table.index("RocketLab") < table.index("National Grid") < table.index("Robinhood")
    assert (
        "| RocketLab (RKLB) | VALIDATED | Space | GROWTH | HOLD | 5.0% | moderate | aggressive |"
        in table
    )
    assert "| Robinhood (HOOD) | REJECTED | None | NEITHER | HOLD | 0.0% | mild | mild |" in table
    assert "## National Grid (NG.): VALIDATED, STABILITY" in output
    assert "Target weights add up to 11.0% of the ISA" in output
    assert sorted(p.name for p in tmp_path.iterdir()) == ["core.md", "stocks.yaml"]


def test_vehicle_analyst_is_told_not_to_sell_only_for_serving_no_pillar() -> None:
    assert "Serving no pillar is not by itself a reason to sell" in VEHICLE_PROMPT


def test_a_real_run_writes_one_report_and_a_proposal_per_stock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run(tmp_path, monkeypatch, Recorder()) == 0

    reports = [p.name.split("_", 1)[1] for p in (tmp_path / "reports").iterdir()]
    assert reports == ["macro_review.md"]
    proposals = [
        Proposal.model_validate_json(p.read_text())
        for p in (tmp_path / "proposals").glob("proposal_*.json")
    ]
    by_ticker = {p.ticker: p for p in proposals}
    assert sorted(by_ticker) == ["HOOD", "NG.", "RKLB"]
    assert by_ticker["NG."].proposed_allocations[0].ticker == "NG."
    assert [u.strategy for u in by_ticker["RKLB"].suggested_config_updates] == ["aggressive"]
    assert by_ticker["HOOD"].suggested_config_updates == []
    assert by_ticker["NG."].thesis_file == "core.md"


def test_only_limits_the_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[str] = []

    assert (
        _run(tmp_path, monkeypatch, Recorder(), "--dry-run", "--only", "HOOD", fetched=fetched) == 0
    )

    assert fetched == ["HOOD"]


def test_one_failed_stock_keeps_the_others_and_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder()

    assert _run(tmp_path, monkeypatch, recorder, failing="RKLB") == 1

    [report] = (tmp_path / "reports").iterdir()
    assert "Review failed, so this stock isn't included: RKLB." in report.read_text()
    assert '"symbol": "RKLB"' not in recorder.prompts["overview"][0]
    assert len(list((tmp_path / "proposals").iterdir())) == 2


def test_a_failed_overview_still_writes_the_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run(tmp_path, monkeypatch, Recorder(overview_fails=True)) == 1

    [report] = (tmp_path / "reports").iterdir()
    text = report.read_text()
    assert "The overview failed" in text
    assert "## RocketLab (RKLB): VALIDATED, GROWTH" in text
    assert len(list((tmp_path / "proposals").iterdir())) == 3


def test_a_typo_in_only_fails_before_any_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder = Recorder()

    assert _run(tmp_path, monkeypatch, recorder, "--only", "HOOOD") == 1

    assert "HOOOD" in capsys.readouterr().err
    assert recorder.prompts == {"vehicle": [], "overview": [], "council": []}


def test_evaluate_still_runs_the_full_council(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder()
    thesis = tmp_path / "NG.md"
    thesis.write_text("---\nticker: NG.\n---\nBody")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "build_model", lambda _: recorder.model)
    monkeypatch.setattr(cli, "MarketData", lambda: MarketData(fetcher=lambda _: INFO))

    assert cli.main(["evaluate", str(thesis), "--no-portfolio", "--dry-run"]) == 0

    assert len(recorder.prompts["council"]) == 4
    assert recorder.prompts["vehicle"] == []
