"""Reads the traded stocks from t2-d2's ``config/stocks.yaml``."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from .market import to_yahoo_symbol


class StocksError(Exception):
    """The stocks file can't be used, or a requested stock isn't in it."""


class Stock(BaseModel):
    symbol: str
    t212_ticker: str
    name: str
    strategy: str

    @property
    def yahoo_symbol(self) -> str | None:
        """Yahoo symbol for this stock.

        US symbols are used as written, because a Trading212 code can outlive a rename: Rocket Lab
        is still ``VACQ_US_EQ``, which Yahoo doesn't recognise.
        """
        if self.t212_ticker.endswith("_US_EQ"):
            return self.symbol
        return to_yahoo_symbol(self.t212_ticker)


class _Market(BaseModel):
    tickers: list[Stock]


class _StocksFile(BaseModel):
    markets: dict[str, _Market]


def load_stocks(path: Path) -> list[Stock]:
    """Load every traded stock, in file order. The informational watchlist is left out.

    Raises:
        StocksError: If the file is unreadable, malformed, or lists no stocks.
    """
    try:
        parsed = _StocksFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValidationError) as exc:
        raise StocksError(f"{path}: can't load stocks ({exc})") from exc
    stocks = [stock for market in parsed.markets.values() for stock in market.tickers]
    if not stocks:
        raise StocksError(f"{path}: lists no stocks")
    return stocks


def _key(symbol: str) -> str:
    return symbol.rstrip(".").upper()


def select_stocks(stocks: list[Stock], only: Iterable[str]) -> list[Stock]:
    """Keep the named stocks, ignoring case and the trailing dot in symbols such as ``NG.``.

    Raises:
        StocksError: If a name matches no stock, so a typo can't silently shrink a review.
    """
    wanted = {_key(symbol) for symbol in only}
    if not wanted:
        return stocks
    unknown = wanted - {_key(stock.symbol) for stock in stocks}
    if unknown:
        raise StocksError(f"Not in the stocks file: {', '.join(sorted(unknown))}")
    return [stock for stock in stocks if _key(stock.symbol) in wanted]
