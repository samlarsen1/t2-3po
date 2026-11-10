"""Checks the request sent to a local Ollama server, using a mock transport."""

import json
from typing import Any

import httpx2
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from t2_3po.agents import build_council, build_model, digest_document
from t2_3po.config import Settings
from t2_3po.models import AnalystView

VIEW = {"summary": "Case.", "key_points": ["Point"], "conviction": "HIGH"}


def _completion(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "qwen3:4b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _model(bodies: list[dict[str, Any]], text: str) -> OpenAIChatModel:
    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=_completion(text))

    provider = OllamaProvider(
        base_url="http://localhost:11434/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return OpenAIChatModel("qwen3:4b", provider=provider)


def test_build_model_uses_ollama_without_an_anthropic_key() -> None:
    settings = Settings(model_provider="ollama", anthropic_api_key=None)
    assert isinstance(build_model(settings), OpenAIChatModel)


async def test_request_uses_json_schema_and_no_anthropic_settings() -> None:
    bodies: list[dict[str, Any]] = []
    result = await build_council(_model(bodies, json.dumps(VIEW))).bull.run("<thesis>x</thesis>")

    assert result.output == AnalystView.model_validate(VIEW)
    body = bodies[0]
    assert body["model"] == "qwen3:4b"
    assert body["response_format"]["type"] == "json_schema"
    assert not any("anthropic" in key or "effort" in key for key in body)


async def test_digest_skips_prompt_caching() -> None:
    bodies: list[dict[str, Any]] = []
    council = build_council(_model(bodies, "Digest."), "1h")
    assert council.digest_cache_ttl == "off"
    await digest_document(council, "thesis", "10-K", "text")
    assert "prompt_cache_breakpoint" not in json.dumps(bodies[0])
