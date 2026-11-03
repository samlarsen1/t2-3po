"""The macro review: one thesis checked against every stock in t2-d2's stocks file."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile

from t2_3po import cli
from t2_3po.agents import MACRO_ORCHESTRATOR_PROMPT, MACRO_PROMPT
from t2_3po.market import MarketData
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

ANALYST = {"summary": "A case.", "key_points": ["Point"], "conviction": "MEDIUM"}
QUANT = {
    "summary": "Mixed.",
    "valuation_assessment": "Fair.",
    "balance_sheet_assessment": "Strong.",
    "claim_checks": [],
}


def _decision(ticker: str, status: str, pillars: list[str], weight: float) -> dict[str, Any]:
    return {
        "thesis_status": status,
        "thesis_pillars": pillars,
        "executive_summary": f"{ticker} summary.",
        "critique": "Critique.",
        "key_risks": [],
        "portfolio_warnings": [],
        "proposed_allocations": [
            {"ticker": ticker, "target_weight": weight, "action": "HOLD", "reasoning": "Why."}
        ],
        "suggested_config_updates": [],
    }


DECISIONS = {
    "NG.": _decision("NG.", "VALIDATED", ["Suppliers"], 0.6),
    "RKLB": _decision("RKLB", "VALIDATED_WITH_RISKS", ["Space"], 0.5),
    "HOOD": _decision("HOOD", "REJECTED", [], 0.0),
}


class Recorder:
    """Scripted council that records every agent's instructions and prompt."""

    def __init__(self) -> None:
        self.instructions: dict[str, str] = {}
        self.prompts: list[str] = []
        self.model = FunctionModel(
            self._respond, profile=ModelProfile(supports_json_schema_output=True)
        )

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = str(messages[-1].parts[0].content)  # type: ignore[union-attr]
        self.prompts.append(prompt)
        text = info.instructions or ""
        ticker = prompt.split('<thesis ticker="')[1].split('"')[0]
        if "You are the Lead Orchestrator" in text:
            self.instructions["orchestrator"] = text
            output: Any = DECISIONS[ticker]
        elif "You are the Quantitative Analyst" in text:
            self.instructions["quant"] = text
            output = QUANT
        else:
            self.instructions["analyst"] = text
            output = ANALYST
        return ModelResponse(parts=[TextPart(json.dumps(output))])


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


def test_every_stock_is_reviewed_against_the_macro_thesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder = Recorder()
    fetched: list[str] = []

    assert _run(tmp_path, monkeypatch, recorder, "--dry-run", fetched=fetched) == 0

    assert sorted(fetched) == ["HOOD", "NG.L", "RKLB"]
    assert all("Bullish on space and suppliers." in p for p in recorder.prompts)
    assert any(
        "RocketLab (RKLB), currently on the `moderate` strategy" in p for p in recorder.prompts
    )
    assert MACRO_PROMPT in recorder.instructions["analyst"]
    assert MACRO_PROMPT in recorder.instructions["quant"]
    assert MACRO_ORCHESTRATOR_PROMPT in recorder.instructions["orchestrator"]
    assert MACRO_ORCHESTRATOR_PROMPT not in recorder.instructions["analyst"]

    output = capsys.readouterr().out
    assert "Thesis pillars: Suppliers." in output
    table = output[output.index("| Stock | Fit |") :]
    assert table.index("National Grid") < table.index("RocketLab") < table.index("Robinhood")
    assert "| National Grid (NG.) | VALIDATED | Suppliers | HOLD | 60.0% | mild | mild |" in table
    assert "| Robinhood (HOOD) | REJECTED | None | HOLD | 0.0% | mild | mild |" in table
    assert "Target weights add up to 110.0% of the ISA" in output
    assert sorted(p.name for p in tmp_path.iterdir()) == ["core.md", "stocks.yaml"]


def test_orchestrator_is_told_not_to_sell_only_for_serving_no_pillar() -> None:
    assert "Serving no pillar is not by itself a reason to sell" in MACRO_ORCHESTRATOR_PROMPT


def test_a_real_run_writes_each_report_and_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run(tmp_path, monkeypatch, Recorder()) == 0

    reports = sorted(p.name.split("_", 1)[1] for p in (tmp_path / "reports").iterdir())
    assert reports == ["HOOD.md", "NG.md", "RKLB.md", "macro_review.md"]
    assert len(list((tmp_path / "proposals").glob("proposal_*.json"))) == 3


def test_only_limits_the_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[str] = []

    assert (
        _run(tmp_path, monkeypatch, Recorder(), "--dry-run", "--only", "HOOD", fetched=fetched) == 0
    )

    assert fetched == ["HOOD"]


def test_one_failed_stock_keeps_the_others_and_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(tmp_path, monkeypatch, Recorder(), failing="RKLB") == 1

    assert "Review failed, so this stock isn't included: RKLB." in capsys.readouterr().out
    reports = sorted(p.name.split("_", 1)[1] for p in (tmp_path / "reports").iterdir())
    assert reports == ["HOOD.md", "NG.md", "macro_review.md"]


def test_a_typo_in_only_fails_before_any_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder = Recorder()

    assert _run(tmp_path, monkeypatch, recorder, "--only", "HOOOD") == 1

    assert "HOOOD" in capsys.readouterr().err
    assert recorder.prompts == []


def test_evaluate_council_has_no_macro_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder()
    thesis = tmp_path / "NG.md"
    thesis.write_text("---\nticker: NG.\n---\nBody")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "build_model", lambda _: recorder.model)
    monkeypatch.setattr(cli, "MarketData", lambda: MarketData(fetcher=lambda _: INFO))

    assert cli.main(["evaluate", str(thesis), "--no-portfolio", "--dry-run"]) == 0

    assert MACRO_PROMPT not in recorder.instructions["analyst"]
    assert MACRO_ORCHESTRATOR_PROMPT not in recorder.instructions["orchestrator"]
