"""Space out model requests to stay under a free tier's per-minute limit."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings


@dataclass(init=False)
class PacedModel(WrapperModel):
    """A model whose requests start at least ``min_interval`` seconds apart.

    Concurrent agents share one schedule, so the council's parallel calls and the review's
    concurrent profiles still reach the provider one at a time. Requests wait their turn in order.

    Args:
        wrapped: The model to pace.
        min_interval: Minimum seconds between the starts of two requests. ``0`` disables pacing.
        clock: Monotonic clock, injectable for tests.
        sleep: Async sleep, injectable for tests.
    """

    min_interval: float

    def __init__(
        self,
        wrapped: Model,
        min_interval: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if min_interval < 0:
            raise ValueError("min_interval must not be negative")
        super().__init__(wrapped)
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._next_start: float | None = None

    async def _wait_turn(self) -> None:
        async with self._lock:
            now = self._clock()
            if self._next_start is not None and now < self._next_start:
                await self._sleep(self._next_start - now)
                now = self._clock()
            self._next_start = now + self.min_interval

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        await self._wait_turn()
        return await self.wrapped.request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        await self._wait_turn()
        async with self.wrapped.request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as response:
            yield response
