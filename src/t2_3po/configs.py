"""Read the other droids' configuration: t2-d2's stocks and modes, and t2-k2's strategy.

t2-3po only reads these files. Changes reach them as suggestions that the operator applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

import yaml
from pydantic import ValidationError

from .stocks import Stock

# Instrument currencies that belong to each t2-d2 market.
MARKET_CURRENCIES: dict[str, frozenset[str]] = {
    "uk": frozenset({"GBX", "GBP"}),
    "us": frozenset({"USD"}),
    "europe": frozenset({"EUR"}),
}

D2_STOCKS_FILE = "config/stocks.yaml"
D2_STRATEGY_FILE = "config/strategy.yaml"
K2_STRATEGY_FILE = "config/strategy.yaml"

# Settings a suggestion may change. Anything else (order types, schedules, global limits,
# analysis windows) is out of scope for automated advice.
_D2_MODE_KEYS = (
    "min_daily_dip_pct",
    "min_weekly_dip_pct",
    "max_position_increase_pct",
    "max_cash_deploy_pct",
    "limit_discount_pct",
    "max_portfolio_weight_pct",
    "max_pct_above_52w_low",
    "min_quarter_change_pct",
    "min_half_year_change_pct",
)
_K2_KEYS = (
    "overbought.rsi_min",
    "overbought.min_pct_above_sma",
    "overbought.min_run_up_pct",
    "overbought.min_signals",
    "harvest.min_gain_pct",
    "harvest.gain_fraction",
    "harvest.max_position_sell_pct",
)


class ConfigReadError(Exception):
    """A droid's configuration file is missing or malformed."""


@dataclass(frozen=True)
class TradedStock:
    """A stock t2-d2 trades, with the market it is listed under."""

    stock: Stock
    market: str


@dataclass(frozen=True)
class D2Config:
    stocks: list[TradedStock]
    watchlist: list[str]
    modes: dict[str, dict[str, Any]]

    def find(self, name: str | None) -> TradedStock | None:
        """Find a traded stock by symbol or Trading212 code, ignoring case and a trailing dot."""
        if not name:
            return None
        wanted = name.rstrip(".").upper()
        for traded in self.stocks:
            stock = traded.stock
            if wanted in (stock.symbol.rstrip(".").upper(), stock.t212_ticker.upper()):
                return traded
        return None

    def lists(self, symbol: str, t212_ticker: str) -> bool:
        """Whether the stock is already traded or on the watchlist."""
        watch = {s.rstrip(".").upper() for s in self.watchlist}
        return (
            self.find(t212_ticker) is not None
            or self.find(symbol) is not None
            or symbol.rstrip(".").upper() in watch
        )

    def tunables(self) -> dict[str, float]:
        """Adjustable numeric settings, as ``modes.<mode>.<key>``, with current values."""
        values: dict[str, float] = {}
        for mode, settings in self.modes.items():
            for key in _D2_MODE_KEYS:
                value = settings.get(key)
                if _is_number(value):
                    values[f"modes.{mode}.{key}"] = float(value)
        return values


@dataclass(frozen=True)
class K2Config:
    settings: dict[str, Any]

    def tunables(self) -> dict[str, float]:
        """Adjustable numeric settings, as ``<section>.<key>``, with current values."""
        values: dict[str, float] = {}
        for dotted in _K2_KEYS:
            section, key = dotted.split(".")
            value = (self.settings.get(section) or {}).get(key)
            if _is_number(value):
                values[dotted] = float(value)
        return values


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigReadError(f"{path}: can't read ({type(exc).__name__})") from exc
    if not isinstance(data, dict):
        raise ConfigReadError(f"{path}: expected a mapping at the top level")
    return data


def load_d2(d2_dir: Path) -> D2Config:
    """Read t2-d2's traded stocks, watchlist and strategy modes.

    Raises:
        ConfigReadError: If either file is missing or malformed.
    """
    stocks_data = _load_yaml(d2_dir / D2_STOCKS_FILE)
    strategy_data = _load_yaml(d2_dir / D2_STRATEGY_FILE)
    stocks: list[TradedStock] = []
    try:
        for market, section in (stocks_data.get("markets") or {}).items():
            for item in (section or {}).get("tickers") or []:
                stocks.append(TradedStock(Stock.model_validate(item), market))
    except (AttributeError, ValidationError) as exc:
        raise ConfigReadError(f"{d2_dir / D2_STOCKS_FILE}: malformed stock entry") from exc
    watchlist = [
        str(item["symbol"])
        for item in stocks_data.get("watchlist") or []
        if isinstance(item, dict) and "symbol" in item
    ]
    modes = strategy_data.get("modes")
    if not isinstance(modes, dict) or not modes:
        raise ConfigReadError(f"{d2_dir / D2_STRATEGY_FILE}: no strategy modes")
    return D2Config(stocks=stocks, watchlist=watchlist, modes=modes)


def load_k2(k2_dir: Path) -> K2Config:
    """Read t2-k2's strategy.

    Raises:
        ConfigReadError: If the file is missing or malformed.
    """
    return K2Config(_load_yaml(k2_dir / K2_STRATEGY_FILE))
