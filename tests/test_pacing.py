"""PacedModel spaces request starts, including for agents that run concurrently."""

import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from t2_3po.pacing import PacedModel


class FakeTime:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _paced(interval: float, time: FakeTime) -> PacedModel:
    return PacedModel(TestModel(), interval, clock=time.clock, sleep=time.sleep)


async def test_first_request_is_not_delayed() -> None:
    time = FakeTime()
    await Agent(_paced(7.0, time)).run("hi")
    assert time.sleeps == []


async def test_sequential_requests_wait_out_the_remaining_interval() -> None:
    time = FakeTime()
    agent = Agent(_paced(7.0, time))
    await agent.run("one")
    time.now += 2.0  # the caller did other work in between
    await agent.run("two")
    time.now += 10.0  # longer than the interval: no wait needed
    await agent.run("three")
    assert time.sleeps == [5.0]


async def test_concurrent_requests_are_spaced_out() -> None:
    time = FakeTime()
    model = _paced(7.0, time)
    await asyncio.gather(*(Agent(model).run(f"q{i}") for i in range(4)))
    assert time.sleeps == [7.0, 7.0, 7.0]


async def test_zero_interval_never_waits() -> None:
    time = FakeTime()
    agent = Agent(_paced(0.0, time))
    for _ in range(3):
        await agent.run("hi")
    assert time.sleeps == []


def test_negative_interval_is_rejected() -> None:
    with pytest.raises(ValueError):
        PacedModel(TestModel(), -1.0)
