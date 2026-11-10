from collections.abc import Callable

import httpx
import pytest

from t2_3po.t212 import ReadOnlyTrading212Client, Trading212Error

SUMMARY = {
    "id": 1,
    "currency": "GBP",
    "cash": {"availableToTrade": 500.5, "inPies": 0, "reservedForOrders": 10},
    "investments": {"currentValue": 9500, "totalCost": 8000},
    "totalValue": 10000.5,
}
POSITION = {
    "instrument": {"ticker": "NVDA_US_EQ", "name": "NVIDIA", "currency": "USD"},
    "quantity": 3,
    "averagePricePaid": 100,
    "currentPrice": 200,
    "walletImpact": {
        "currency": "GBP",
        "currentValue": 450.25,
        "totalCost": 225,
        "unrealizedProfitLoss": 225.25,
    },
}

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> ReadOnlyTrading212Client:
    return ReadOnlyTrading212Client(
        "key",
        "secret",
        "https://demo.trading212.com",
        transport=httpx.MockTransport(handler),
        retry_wait_multiplier=0,
    )


async def test_reads_summary_and_positions_using_only_get() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = SUMMARY if request.url.path.endswith("summary") else [POSITION]
        return httpx.Response(200, json=body)

    async with _client(handler) as client:
        summary = await client.get_account_summary()
        positions = await client.get_positions()

    assert summary.total_value == 10000.5
    assert summary.cash.available_to_trade == 500.5
    assert positions[0].instrument.ticker == "NVDA_US_EQ"
    assert positions[0].wallet_impact.current_value == 450.25
    assert {request.method for request in requests} == {"GET"}
    # base64("key:secret")
    assert requests[0].headers["Authorization"] == "Basic a2V5OnNlY3JldA=="


def test_client_exposes_no_order_methods() -> None:
    public = {name for name in dir(ReadOnlyTrading212Client) if not name.startswith("_")}
    assert public == {"get_account_summary", "get_positions"}


async def test_retries_rate_limit_then_succeeds() -> None:
    responses = [httpx.Response(429), httpx.Response(200, json=[POSITION])]

    async with _client(lambda _: responses.pop(0)) as client:
        assert len(await client.get_positions()) == 1


async def test_persistent_rate_limit_raises() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429)

    async with _client(handler) as client:
        with pytest.raises(Trading212Error, match="RateLimited"):
            await client.get_positions()
    assert calls == 4


async def test_timeout_is_retried_then_raises_without_exposing_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async with _client(handler) as client:
        with pytest.raises(Trading212Error, match="ReadTimeout"):
            await client.get_positions()


@pytest.mark.parametrize("status", [401, 403, 500, 503])
async def test_http_errors_raise_without_retry(status: int) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"secret-looking": "detail"})

    async with _client(handler) as client:
        with pytest.raises(Trading212Error, match=f"HTTP {status}") as caught:
            await client.get_account_summary()
    assert "secret-looking" not in str(caught.value)
    assert calls == 1
