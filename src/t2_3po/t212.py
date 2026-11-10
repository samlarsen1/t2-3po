"""Read-only Trading212 client.

This is deliberately not t2-d2's client, which can place orders. Every request goes through
``_get``, so t2-3po has no code path that can change the account.
"""

from __future__ import annotations

import base64
import logging
from types import TracebackType
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

logger = logging.getLogger(__name__)


class Trading212Error(Exception):
    """Trading212 returned an error or couldn't be reached."""


class _RateLimitedError(Exception):
    """Internal signal that makes tenacity retry an HTTP 429."""


class Cash(BaseModel):
    available_to_trade: float = Field(alias="availableToTrade")


class AccountSummary(BaseModel):
    currency: str
    cash: Cash
    total_value: float = Field(alias="totalValue")


class Instrument(BaseModel):
    ticker: str
    name: str | None = None


class WalletImpact(BaseModel):
    current_value: float = Field(alias="currentValue")


class Position(BaseModel):
    model_config = ConfigDict(extra="ignore")

    instrument: Instrument
    wallet_impact: WalletImpact = Field(alias="walletImpact")


class ReadOnlyTrading212Client:
    """Async read-only Trading212 client. Use as an async context manager."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        retry_wait_multiplier: float = 2.0,
    ) -> None:
        encoded = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Basic {encoded}"},
            timeout=timeout,
            transport=transport,
        )
        self._retry_wait_multiplier = retry_wait_multiplier

    async def __aenter__(self) -> ReadOnlyTrading212Client:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    async def _get_once(self, path: str) -> Any:
        response = await self._client.get(path)
        if response.status_code == 429:
            logger.warning("t212_rate_limited path=%s", path)
            raise _RateLimitedError
        if not response.is_success:
            raise Trading212Error(f"GET {path} returned HTTP {response.status_code}")
        return response.json()

    async def _get(self, path: str) -> Any:
        retrying = retry(
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.NetworkError, _RateLimitedError)
            ),
            wait=wait_exponential(multiplier=self._retry_wait_multiplier, max=60)
            + wait_random(0, self._retry_wait_multiplier),
            stop=stop_after_attempt(4),
            reraise=True,
        )
        try:
            return await retrying(self._get_once)(path)
        except (httpx.HTTPError, _RateLimitedError) as exc:
            raise Trading212Error(f"GET {path} failed ({type(exc).__name__})") from exc

    async def get_account_summary(self) -> AccountSummary:
        """Fetch cash and total account value, in the account currency."""
        return AccountSummary.model_validate(await self._get("/api/v0/equity/account/summary"))

    async def get_positions(self) -> list[Position]:
        """Fetch all open positions."""
        data = await self._get("/api/v0/equity/positions")
        return [Position.model_validate(position) for position in data]
