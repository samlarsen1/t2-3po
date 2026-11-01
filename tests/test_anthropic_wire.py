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

from t2_3po.agents import build_council, digest_document
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


def _cache_control(block: dict[str, Any]) -> dict[str, Any] | None:
    return block.get("cache_control")


def _sse(text: str) -> httpx2.Response:
    """Build a streamed response, which the digester gets because it has no output cap."""
    message = {**_message(""), "content": [], "stop_reason": None}
    events: list[dict[str, Any]] = [
        {"type": "message_start", "message": message},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]
    payload = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, text=payload)


def _request_recorder(
    bodies: list[dict[str, Any]],
) -> httpx2.MockTransport:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("stream"):
            return _sse("Digest.")
        return httpx2.Response(200, json=_message(json.dumps(VIEW)))

    return httpx2.MockTransport(handler)


async def test_council_requests_are_not_cached() -> None:
    bodies: list[dict[str, Any]] = []

    await build_council(_model(_request_recorder(bodies))).bull.run("<thesis>x</thesis>")

    assert "cache_control" not in json.dumps(bodies[0])


async def test_digest_caches_the_document_and_keeps_the_thesis_after_it() -> None:
    bodies: list[dict[str, Any]] = []
    council = build_council(_model(_request_recorder(bodies)), "1h")

    assert await digest_document(council, "My thesis", "10k.txt", "Filing text") == "Digest."

    first, second = bodies[0]["messages"][0]["content"]
    assert "Filing text" in first["text"]
    assert _cache_control(first) == {"type": "ephemeral", "ttl": "1h"}
    assert "My thesis" in second["text"]
    assert _cache_control(second) is None


async def test_off_sends_no_cache_control() -> None:
    bodies: list[dict[str, Any]] = []
    council = build_council(_model(_request_recorder(bodies)), "off")

    await digest_document(council, "My thesis", "10k.txt", "Filing text")

    assert "cache_control" not in json.dumps(bodies[0])


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
