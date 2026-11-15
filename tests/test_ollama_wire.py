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


def test_build_model_uses_ollama_without_a_google_key() -> None:
    settings = Settings(model_provider="ollama", google_api_key=None)
    assert isinstance(build_model(settings), OpenAIChatModel)


async def test_request_uses_json_schema() -> None:
    bodies: list[dict[str, Any]] = []
    result = await build_council(_model(bodies, json.dumps(VIEW))).bull.run("<thesis>x</thesis>")

    assert result.output == AnalystView.model_validate(VIEW)
    body = bodies[0]
    assert body["model"] == "qwen3:4b"
    assert body["response_format"]["type"] == "json_schema"
    assert body["temperature"] == 0.3


async def test_digest_sends_the_document_then_the_thesis() -> None:
    bodies: list[dict[str, Any]] = []
    council = build_council(_model(bodies, "Digest."))
    assert await digest_document(council, "my thesis", "10-K", "the text") == "Digest."
    sent = json.dumps(bodies[0]["messages"])
    assert sent.index("the text") < sent.index("my thesis")
