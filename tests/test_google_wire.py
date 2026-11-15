"""Checks the request sent to the Gemini API, using a mock transport."""

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from t2_3po.agents import GEMINI_RETRY, build_council, build_model
from t2_3po.config import Settings
from t2_3po.models import AnalystView
from t2_3po.pacing import PacedModel

VIEW = {"summary": "Case.", "key_points": ["Point"], "conviction": "HIGH"}


def _response(text: str) -> dict[str, Any]:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        "modelVersion": "gemini-3.7-flash",
    }


def _model(requests: list[httpx2.Request], text: str) -> GoogleModel:
    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=_response(text))

    provider = GoogleProvider(
        api_key="test-key",  # pragma: allowlist secret
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return GoogleModel("gemini-3.7-flash", provider=provider)


def test_build_model_paces_gemini_and_retries_rate_limits() -> None:
    settings = Settings(
        model_provider="google",
        google_api_key=SecretStr("test-key"),  # pragma: allowlist secret
        gemini_min_interval_seconds=9.0,
    )
    model = build_model(settings)

    assert isinstance(model, PacedModel)
    assert model.min_interval == 9.0
    assert isinstance(model.wrapped, GoogleModel)
    assert model.wrapped.model_name == "gemini-3.7-flash"
    assert GEMINI_RETRY.http_status_codes is not None
    assert 429 in GEMINI_RETRY.http_status_codes


def test_build_model_requires_a_google_key() -> None:
    settings = Settings(model_provider="google", google_api_key=None)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        build_model(settings)


async def test_request_asks_for_json_matching_the_output_schema() -> None:
    requests: list[httpx2.Request] = []
    model = PacedModel(_model(requests, json.dumps(VIEW)), 0)

    result = await build_council(model).bull.run("<thesis>x</thesis>")

    assert result.output == AnalystView.model_validate(VIEW)
    assert "gemini-3.7-flash:generateContent" in str(requests[0].url)
    config = json.loads(requests[0].content)["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert "conviction" in json.dumps(config)
    assert config["maxOutputTokens"] == 16000


def test_models_lists_flash_models_and_flags_a_missing_setting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from t2_3po import cli

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")  # pragma: allowlist secret
    monkeypatch.setenv("GEMINI_MODEL", "gemini-9-flash")
    monkeypatch.setattr(
        cli, "list_gemini_models", lambda key: ["gemini-3.5-flash", "gemini-3.7-flash"]
    )

    assert cli.main(["models"]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["gemini-3.5-flash", "gemini-3.7-flash"]
    assert "GEMINI_MODEL=gemini-9-flash is not in this list" in captured.err


def test_models_needs_a_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from t2_3po import cli

    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert cli.main(["models"]) == 2
