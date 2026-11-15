"""The Trading212 instrument list harvested by t2-bb8, used to check suggested holdings."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from .configs import MARKET_CURRENCIES

# Where t2-bb8 writes, deployed first, then a development checkout.
_DATA_DIRS = ("var/data", "data")
_TRADABLE_TYPES = frozenset({"STOCK", "ETF"})
_UK_EXCHANGE = "London Stock Exchange"


class InstrumentsError(Exception):
    """The instrument list can't be read."""


class ResolveError(Exception):
    """A suggested holding doesn't match exactly one suitable Trading212 instrument."""


class Instrument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ticker: str
    shortName: str | None = None  # noqa: N815 - Trading212's field name
    name: str | None = None
    type: str | None = None
    currencyCode: str | None = None  # noqa: N815 - Trading212's field name
    exchangeName: str | None = None  # noqa: N815 - Trading212's field name


def _norm(symbol: str) -> str:
    return symbol.strip().rstrip(".").upper()


class Instruments:
    """Every instrument Trading212 offers, indexed by code and by exchange symbol."""

    def __init__(self, items: list[Instrument]) -> None:
        self._by_ticker = {item.ticker: item for item in items}
        self._by_symbol: dict[str, list[Instrument]] = defaultdict(list)
        for item in items:
            if item.shortName:
                self._by_symbol[_norm(item.shortName)].append(item)

    def __len__(self) -> int:
        return len(self._by_ticker)

    @classmethod
    def load(cls, bb8_dir: Path) -> Instruments:
        """Load ``instruments.json`` from t2-bb8's data directory.

        Raises:
            InstrumentsError: If no readable instrument list is found.
        """
        for sub in _DATA_DIRS:
            path = bb8_dir / sub / "instruments.json"
            if path.is_file():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    return cls([Instrument.model_validate(item) for item in raw])
                except (OSError, ValueError, TypeError, ValidationError) as exc:
                    raise InstrumentsError(f"{path}: can't read ({type(exc).__name__})") from exc
        raise InstrumentsError(f"no instruments.json under {bb8_dir} (run t2-bb8 first)")

    def get(self, t212_ticker: str) -> Instrument | None:
        return self._by_ticker.get(t212_ticker)

    def resolve(self, symbol: str, market: str, t212_ticker: str | None = None) -> Instrument:
        """Find the one tradable Trading212 instrument for a symbol in a t2-d2 market.

        Raises:
            ResolveError: If there is no match, or more than one.
        """
        currencies = MARKET_CURRENCIES.get(market)
        if currencies is None:
            raise ResolveError(f"unknown market {market!r}")

        def suitable(item: Instrument) -> bool:
            return (
                item.type in _TRADABLE_TYPES
                and item.currencyCode in currencies
                and (market != "us" or item.ticker.endswith("_US_EQ"))
            )

        if t212_ticker:
            found = self.get(t212_ticker)
            if found is None:
                raise ResolveError(f"{t212_ticker} is not on Trading212")
            if not suitable(found):
                raise ResolveError(f"{t212_ticker} doesn't trade in the {market} market")
            return found

        matches = [item for item in self._by_symbol.get(_norm(symbol), []) if suitable(item)]
        if market == "uk" and len(matches) > 1:
            matches = [item for item in matches if item.exchangeName == _UK_EXCHANGE] or matches
        if not matches:
            raise ResolveError(f"no Trading212 instrument for {symbol} in the {market} market")
        if len(matches) > 1:
            codes = ", ".join(sorted(item.ticker for item in matches))
            raise ResolveError(f"{symbol} matches several instruments ({codes})")
        return matches[0]
