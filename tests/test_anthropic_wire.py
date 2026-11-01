"""Checks the actual request sent to the Claude API, using a mock transport.

Claude Sonnet 5.5 returns a 400 for forced ``tool_choice``, sampling parameters and
``budget_tokens``, so a framework upgrade that reintroduces any of them should fail here.
"""

import json
from typing import Any

import httpx2
import pytest
from anthropic import AsyncAnthropic
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from t2_3po.agents import build_council
from t2_3po.models import AnalystView

VIEW = {"summary": "Case.", "key_points": ["Point"], "conviction": "HIGH"}


def _message(text: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _model(transport: httpx2.MockTransport) -> AnthropicModel:
    client = AsyncAnthropic(
        api_key="test-key",
        http_client=httpx2.AsyncClient(transport=transport),
        max_retries=0,
    )
    return AnthropicModel("claude-sonnet-5-5", provider=AnthropicProvider(anthropic_client=client))


async def test_request_uses_native_structured_output() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=_message(json.dumps(VIEW)))

    model = _model(httpx2.MockTransport(handler))
    result = await build_council(model).bull.run("<thesis>x</thesis>")

    assert result.output == AnalystView.model_validate(VIEW)
    body = bodies[0]
    assert body["model"] == "claude-sonnet-5-5"
    assert body["output_config"]["effort"] == "high"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "tool_choice" not in body
    assert "tools" not in body
    assert not {"temperature", "top_p", "top_k"} & body.keys()
    assert body.get("thinking", {}).get("type", "adaptive") == "adaptive"
    assert "Bull Analyst" in json.dumps(body["system"])


async def test_bounds_are_enforced_after_schema_transformation() -> None:
    """Numeric bounds may be stripped from the wire schema, so validation must still apply."""
    bad = {
        "thesis_status": "VALIDATED",
        "executive_summary": "s",
        "critique": "c",
        "key_risks": [],
        "portfolio_warnings": [],
        "proposed_allocations": [
            {"ticker": "NVDA", "target_weight": 8, "action": "BUY", "reasoning": "r"}
        ],
        "suggested_config_updates": [],
    }
    good = {**bad, "proposed_allocations": []}
    responses = [bad, good]
    requests = 0

    def handler(_: httpx2.Request) -> httpx2.Response:
        nonlocal requests
        requests += 1
        return httpx2.Response(200, json=_message(json.dumps(responses.pop(0))))

    model = _model(httpx2.MockTransport(handler))
    result = await build_council(model).orchestrator.run("<thesis>x</thesis>")

    assert requests == 2
    assert result.output.proposed_allocations == []


async def test_api_error_surfaces_as_model_http_error() -> None:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            400,
            json={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "bad request"},
            },
        )

    model = _model(httpx2.MockTransport(handler))
    with pytest.raises(ModelHTTPError):
        await build_council(model).bull.run("<thesis>x</thesis>")
