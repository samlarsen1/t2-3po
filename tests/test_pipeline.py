"""End-to-end pipeline tests with scripted models: no network and no API spend."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile

from t2_3po import cli
from t2_3po.agents import build_council
from t2_3po.market import MarketData, MarketDataError
from t2_3po.output import export_approved, render_report, write_outputs
from t2_3po.pipeline import (
    PORTFOLIO_SKIPPED_WARNING,
    build_proposal,
    evaluate,
    gather_context,
)
from t2_3po.portfolio import Holding, Limits, PortfolioSnapshot
from t2_3po.thesis import Thesis, ThesisError, ThesisMeta

LIMITS = Limits(max_sector_weight=0.30, max_position_weight=0.175)
NOW = datetime(2026, 11, 1, 20, 0, tzinfo=UTC)
INFO = {
    "sector": "Technology",
    "currentPrice": 200.0,
    "currency": "USD",
    "forwardPE": 25.0,
    "grossMargins": 0.7467,
}

ANALYST = {"summary": "A case.", "key_points": ["Point"], "conviction": "MEDIUM"}
QUANT = {
    "summary": "Mixed.",
    "valuation_assessment": "Rich.",
    "balance_sheet_assessment": "Strong.",
    "claim_checks": [{"claim": "Cheap", "verdict": "CONTRADICTED", "evidence": "P/E 25"}],
}
DECISION = {
    "thesis_status": "VALIDATED_WITH_RISKS",
    "executive_summary": "Hold the thesis, trim the size.",
    "critique": "Sound but overweight.",
    "key_risks": ["Capex slowdown"],
    "portfolio_warnings": ["Model-found warning."],
    "proposed_allocations": [
        {"ticker": "NVDA", "target_weight": 0.08, "action": "TRIM", "reasoning": "Too large."}
    ],
    "suggested_config_updates": [{"ticker": "NVDA", "strategy": "mild", "reasoning": "Slow down."}],
}


class ScriptedModels:
    """Routes each agent, identified by its instructions, to a canned response."""

    def __init__(self) -> None:
        self.prompts: dict[str, str] = {}
        self.council = FunctionModel(
            self._respond, profile=ModelProfile(supports_json_schema_output=True)
        )

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1].parts[0]
        assert isinstance(request, UserPromptPart)
        content = request.content
        prompt = content if isinstance(content, str) else "\n".join(map(str, content))
        instructions = info.instructions or ""
        routes: list[tuple[str, str, Any]] = [
            ("You are the Bull Analyst", "bull", ANALYST),
            ("You are the Bear Analyst", "bear", ANALYST),
            ("You are the Quantitative Analyst", "quant", QUANT),
            ("You are the Lead Orchestrator", "orchestrator", DECISION),
        ]
        for marker, name, output in routes:
            if marker in instructions:
                self.prompts[name] = prompt
                return ModelResponse(parts=[TextPart(json.dumps(output))])
        self.prompts["digest"] = prompt
        return ModelResponse(parts=[TextPart("Digest: revenue rose 50%.")])


def _thesis(tmp_path: Path, documents: list[Path] | None = None) -> Thesis:
    return Thesis(
        path=tmp_path / "NVDA.md",
        meta=ThesisMeta(ticker="NVDA", t212_ticker="NVDA_US_EQ", documents=documents or []),
        body="AI capex grows.",
    )


def _snapshot() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        currency="GBP",
        total_value=1000,
        cash_available=100,
        cash_weight=0.1,
        holdings=[
            Holding(
                t212_ticker="NVDA_US_EQ",
                yahoo_symbol="NVDA",
                name="NVIDIA",
                value=400,
                weight=0.4,
                sector="Technology",
            )
        ],
        sector_weights={"Technology": 0.4},
    )


async def test_full_run_writes_report_and_plan_shaped_proposal(tmp_path: Path) -> None:
    models = ScriptedModels()
    council = build_council(models.council)
    market = MarketData(fetcher=lambda _: INFO)

    context = await gather_context(_thesis(tmp_path), market, _snapshot(), LIMITS)
    result = await evaluate(context, council)
    proposal = build_proposal(context, result, now=NOW)
    report, machine = write_outputs(
        context, result, proposal, tmp_path / "reports", tmp_path / "proposals"
    )

    # Later agents see earlier agents' work, and everyone sees computed findings.
    assert "<bull_case>" not in models.prompts["bull"]
    assert "<bear_case>" in models.prompts["quant"]
    assert "<quant_review>" in models.prompts["orchestrator"]
    assert "exceeds 30% ISA limit" in models.prompts["bull"]
    assert "<current_weight_of_ticker_under_review>0.4000" in models.prompts["bull"]
    assert '"gross_margin": 0.7467' in models.prompts["bull"]

    assert report.name == "2026-11-01_NVDA.md"
    assert machine.name == "proposal_20261101.json"
    payload = json.loads(machine.read_text())
    assert payload["timestamp"] == "2026-11-01T20:00:00Z"
    assert payload["thesis_status"] == "VALIDATED_WITH_RISKS"
    assert payload["proposed_allocations"] == DECISION["proposed_allocations"]
    assert "approved_at" not in payload
    # Computed warnings are kept alongside the model's own.
    assert any("Technology sector" in w for w in payload["portfolio_warnings"])
    assert "Model-found warning." in payload["portfolio_warnings"]

    text = report.read_text()
    assert "NVDA: VALIDATED_WITH_RISKS" in text
    assert "| NVDA | TRIM | 8.0% | Too large. |" in text
    assert "set `strategy: mild`" in text
    assert "| Cheap | CONTRADICTED | P/E 25 |" in text
    assert "| Gross margin | 74.67% |" in text


async def test_bull_and_bear_run_in_parallel(tmp_path: Path) -> None:
    scripted = ScriptedModels()
    active = peak = 0

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return scripted._respond(messages, info)

    model = FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))
    context = await gather_context(
        _thesis(tmp_path), MarketData(fetcher=lambda _: INFO), None, LIMITS
    )

    await evaluate(context, build_council(model))

    assert peak == 2
    assert set(scripted.prompts) == {"bull", "bear", "quant", "orchestrator"}


async def test_no_portfolio_is_stated_in_the_proposal(tmp_path: Path) -> None:
    models = ScriptedModels()
    context = await gather_context(
        _thesis(tmp_path), MarketData(fetcher=lambda _: INFO), None, LIMITS
    )
    result = await evaluate(context, build_council(models.council))
    proposal = build_proposal(context, result, now=NOW)

    assert "<portfolio>unavailable</portfolio>" in models.prompts["orchestrator"]
    assert proposal.portfolio_warnings[0] == PORTFOLIO_SKIPPED_WARNING


async def test_documents_reach_digester_in_full_and_digest_reaches_council(tmp_path: Path) -> None:
    document = tmp_path / "10k.txt"
    full_text = "RISK FACTOR " * 50_000
    document.write_text(full_text)
    models = ScriptedModels()
    context = await gather_context(
        _thesis(tmp_path, [document]), MarketData(fetcher=lambda _: INFO), None, LIMITS
    )

    await evaluate(context, build_council(models.council))

    assert full_text in models.prompts["digest"]
    assert '<document_digest name="10k.txt">' in models.prompts["bear"]
    assert "Digest: revenue rose 50%." in models.prompts["bear"]


async def test_missing_document_fails_before_the_council_runs(tmp_path: Path) -> None:
    models = ScriptedModels()
    context = await gather_context(
        _thesis(tmp_path, [tmp_path / "missing.txt"]),
        MarketData(fetcher=lambda _: INFO),
        None,
        LIMITS,
    )
    with pytest.raises(ThesisError):
        await evaluate(context, build_council(models.council))
    assert models.prompts == {}


async def test_no_market_data_fails_rather_than_debating_without_figures(tmp_path: Path) -> None:
    with pytest.raises(MarketDataError):
        await gather_context(_thesis(tmp_path), MarketData(fetcher=lambda _: {}), None, LIMITS)


async def test_second_run_same_day_does_not_overwrite(tmp_path: Path) -> None:
    models = ScriptedModels()
    context = await gather_context(
        _thesis(tmp_path), MarketData(fetcher=lambda _: INFO), None, LIMITS
    )
    result = await evaluate(context, build_council(models.council))
    proposal = build_proposal(context, result, now=NOW)

    first = write_outputs(context, result, proposal, tmp_path / "r", tmp_path / "p")
    second = write_outputs(context, result, proposal, tmp_path / "r", tmp_path / "p")

    assert first[1].name == "proposal_20261101.json"
    assert second[0].name == "2026-11-01_NVDA_2.md"
    assert second[1].name == "proposal_20261101_2.json"
    assert render_report(context, result, proposal) == first[0].read_text()


async def _proposal_file(tmp_path: Path) -> Path:
    models = ScriptedModels()
    context = await gather_context(
        _thesis(tmp_path), MarketData(fetcher=lambda _: INFO), None, LIMITS
    )
    result = await evaluate(context, build_council(models.council))
    proposal = build_proposal(context, result, now=NOW)
    return write_outputs(context, result, proposal, tmp_path / "r", tmp_path / "p")[1]


async def test_export_stamps_approval_and_leaves_source_unchanged(tmp_path: Path) -> None:
    source = await _proposal_file(tmp_path)
    original = source.read_text()

    exported = export_approved(source, tmp_path / "exports", now=NOW)

    assert json.loads(exported.read_text())["approved_at"] == "2026-11-01T20:00:00Z"
    assert source.read_text() == original


async def test_cli_export_refuses_without_approve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = await _proposal_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EXPORT_DIR", str(tmp_path / "exports"))

    assert cli.main(["export", str(source)]) == 2
    assert not (tmp_path / "exports").exists()
    assert cli.main(["export", str(source), "--approve"]) == 0
    assert (tmp_path / "exports" / source.name).exists()


def test_cli_export_rejects_invalid_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"thesis_status": "NOPE"}')
    monkeypatch.chdir(tmp_path)
    assert cli.main(["export", str(bad), "--approve"]) == 1


def test_cli_evaluate_requires_anthropic_key_before_any_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    thesis = tmp_path / "NVDA.md"
    thesis.write_text("---\nticker: NVDA\n---\nBody")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert cli.main(["evaluate", str(thesis), "--no-portfolio"]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def _run_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, models: ScriptedModels, *flags: str
) -> int:
    thesis = tmp_path / "NVDA.md"
    thesis.write_text("---\nticker: NVDA\n---\nBody")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "build_model", lambda _: models.council)
    monkeypatch.setattr(cli, "MarketData", lambda: MarketData(fetcher=lambda _: INFO))
    return cli.main(["evaluate", str(thesis), "--no-portfolio", *flags])


def test_cli_dry_run_calls_every_agent_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    models = ScriptedModels()

    assert _run_cli(tmp_path, monkeypatch, models, "--dry-run") == 0

    assert set(models.prompts) == {"bull", "bear", "quant", "orchestrator"}
    output = capsys.readouterr().out
    assert "# NVDA: VALIDATED_WITH_RISKS" in output
    assert "| NVDA | TRIM | 8.0% | Too large. |" in output
    assert sorted(p.name for p in tmp_path.iterdir()) == ["NVDA.md"]


def test_cli_without_dry_run_writes_report_and_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run_cli(tmp_path, monkeypatch, ScriptedModels()) == 0

    assert len(list((tmp_path / "reports").glob("*_NVDA.md"))) == 1
    assert len(list((tmp_path / "proposals").glob("proposal_*.json"))) == 1


def test_cli_dry_run_requires_anthropic_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    thesis = tmp_path / "NVDA.md"
    thesis.write_text("---\nticker: NVDA\n---\nBody")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert cli.main(["evaluate", str(thesis), "--no-portfolio", "--dry-run"]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
