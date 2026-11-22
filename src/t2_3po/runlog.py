"""What the trading droids actually did, from their T2 run reports."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = "t2.run-report/1"


def load_reports(runs_dir: Path, droid: str, since: datetime) -> list[dict[str, Any]]:
    """Run reports from ``droid`` that started at or after ``since``, oldest first.

    Unreadable files and reports from other droids are skipped.
    """
    if not runs_dir.is_dir():
        return []
    reports: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            started = datetime.fromisoformat(report["started_at"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if report.get("schema") == SCHEMA and report.get("droid") == droid and started >= since:
            reports.append(report)
    return reports


@dataclass
class TickerActivity:
    """One stock's t2-d2 history over the lookback."""

    evaluated: int = 0
    signalled: int = 0
    bought: int = 0
    deployed_gbp: float = 0.0
    reasons: Counter[str] = field(default_factory=Counter)

    def top_reasons(self, n: int = 2) -> list[tuple[str, int]]:
        return self.reasons.most_common(n)


@dataclass
class D2Activity:
    sessions: int = 0
    by_symbol: dict[str, TickerActivity] = field(
        default_factory=lambda: defaultdict(TickerActivity)
    )


def _reason_kind(reason: str) -> str:
    """Group reasons that differ only in their figures, such as 'RSI 61.2 > 55'."""
    words = [w for w in reason.split() if not any(ch.isdigit() for ch in w)]
    return " ".join(words)[:80] or reason[:80]


def d2_activity(reports: list[dict[str, Any]]) -> D2Activity:
    """Tally evaluations, buy signals, trades and skip reasons per stock."""
    activity = D2Activity()
    for report in reports:
        if report.get("status") != "ok":
            continue
        result = report.get("result") or {}
        decisions = result.get("decisions") or []
        if not decisions and not result.get("trades"):
            continue
        activity.sessions += 1
        for decision in decisions:
            symbol = str(decision.get("ticker"))
            stats = activity.by_symbol[symbol]
            stats.evaluated += 1
            if decision.get("should_trade"):
                stats.signalled += 1
            else:
                stats.reasons[_reason_kind(str(decision.get("reason", "")))] += 1
        for trade in result.get("trades") or []:
            stats = activity.by_symbol[str(trade.get("ticker"))]
            stats.bought += 1
            stats.deployed_gbp += float(trade.get("trade_value_gbp") or 0.0)
    return activity


def k2_latest(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The holding decisions from the newest successful t2-k2 run, or an empty list."""
    for report in reversed(reports):
        if report.get("status") == "ok":
            decisions = (report.get("result") or {}).get("decisions")
            return list(decisions) if isinstance(decisions, list) else []
    return []


def last_reviewed(reports: list[dict[str, Any]]) -> dict[str, datetime]:
    """When each stock was last profiled by a real ``review``, keyed by symbol.

    Dry runs are skipped. A stock that failed isn't listed in ``result.stocks``, so it stays due.
    """
    latest: dict[str, datetime] = {}
    for report in reports:
        if report.get("command") != "review" or report.get("dry_run"):
            continue
        try:
            started = datetime.fromisoformat(report["started_at"])
            symbols = [str(entry["symbol"]) for entry in report["result"]["stocks"]]
        except (KeyError, TypeError, ValueError):
            continue
        for symbol in symbols:
            if symbol not in latest or started > latest[symbol]:
                latest[symbol] = started
    return latest
