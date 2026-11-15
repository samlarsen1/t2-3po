"""T2 droid contract: exit codes, JSON run reports and locks.

Vendored, unchanged, into every droid. The reference copy lives in t2-hq at
``contract/t2contract.py``; edit it there and run ``bin/t2-contract sync``. See t2-hq's
CONTRACT.md for the rules.

Standard library only, so every droid can carry it without new dependencies.
"""

from __future__ import annotations

import enum
import errno
import fcntl
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

CONTRACT_VERSION = "1"
SCHEMA = "t2.run-report/1"

# Seconds to wait for another droid to finish with the Trading212 account.
DEFAULT_T212_LOCK_TIMEOUT = 600.0


class Exit(enum.IntEnum):
    """Process exit codes shared by every droid."""

    OK = 0
    FAILED = 1
    CONFIG = 2
    SKIPPED = 3


STATUS = {
    Exit.OK: "ok",
    Exit.FAILED: "failed",
    Exit.CONFIG: "config_error",
    Exit.SKIPPED: "skipped",
}


class Skipped(Exception):  # noqa: N818 - it is a control-flow signal, not an error
    """The run had nothing to do or must not run now. Exits with ``Exit.SKIPPED``."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class LockBusy(Skipped):
    """Another process holds a lock this run needs."""


def lock_dir() -> Path:
    """Directory shared by every droid run by the same user, created if missing.

    ``T2_LOCK_DIR`` overrides it. Otherwise it is derived from the user ID, not from
    ``XDG_RUNTIME_DIR``, so system services that run as the user find the same directory.
    """
    override = os.environ.get("T2_LOCK_DIR")
    if override:
        path = Path(override)
    else:
        runtime = Path(f"/run/user/{os.getuid()}")
        base = runtime if runtime.is_dir() else Path(tempfile.gettempdir())
        path = base / ("t2" if base == runtime else f"t2-{os.getuid()}")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _try_lock(handle: IO[str]) -> bool:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EACCES):
            return False
        raise
    return True


@contextmanager
def _file_lock(name: str, timeout: float, what: str) -> Iterator[None]:
    path = lock_dir() / f"{name}.lock"
    with path.open("a+", encoding="utf-8") as handle:
        deadline = time.monotonic() + timeout
        while not _try_lock(handle):
            if time.monotonic() >= deadline:
                raise LockBusy(f"{what} is held by another process ({path})")
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def run_lock(droid: str) -> Iterator[None]:
    """Allow one run of ``droid`` at a time. Raises ``LockBusy`` at once if one is running."""
    with _file_lock(f"{droid}.run", 0.0, f"another {droid} run"):
        yield


def t212_lock_timeout() -> float:
    """``T2_T212_LOCK_TIMEOUT`` in seconds, or the default."""
    raw = os.environ.get("T2_T212_LOCK_TIMEOUT")
    try:
        return DEFAULT_T212_LOCK_TIMEOUT if raw is None else max(0.0, float(raw))
    except ValueError:
        return DEFAULT_T212_LOCK_TIMEOUT


@contextmanager
def t212_lock(timeout: float | None = None) -> Iterator[None]:
    """Hold the Trading212 account for this run.

    Rate limits apply per account, so droids on independent timers take turns. Waits up to
    ``timeout`` seconds (default ``t212_lock_timeout()``), then raises ``LockBusy``.
    """
    wait = t212_lock_timeout() if timeout is None else timeout
    with _file_lock("t212", wait, "the Trading212 account lock"):
        yield


def runs_dir() -> Path:
    """Where run reports go: ``T2_RUNS_DIR``, else ``var/runs`` under the working directory."""
    return Path(os.environ.get("T2_RUNS_DIR") or "var/runs")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "run"


@dataclass
class RunReport:
    """One droid run, written as JSON when the run ends.

    Create it when the run starts, call ``finish`` exactly once, then ``write``.
    """

    droid: str
    version: str
    command: str
    dry_run: bool | None = None
    argv: list[str] = field(default_factory=lambda: list(sys.argv[1:]))
    started_at: datetime = field(default_factory=_utc_now)
    finished_at: datetime | None = None
    exit_code: Exit = Exit.FAILED
    summary: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def finish(
        self,
        code: Exit,
        summary: str,
        result: dict[str, Any] | None = None,
        errors: list[str] | None = None,
    ) -> Exit:
        """Record how the run ended. Returns ``code``, so ``return report.finish(...)`` works."""
        self.finished_at = _utc_now()
        self.exit_code = Exit(code)
        self.summary = summary
        if result is not None:
            self.result = result
        if errors:
            self.errors.extend(errors)
        return self.exit_code

    def to_dict(self) -> dict[str, Any]:
        """The report as JSON-serialisable data, in the shape of ``run-report.schema.json``."""
        finished = self.finished_at or _utc_now()
        return {
            "schema": SCHEMA,
            "droid": self.droid,
            "version": self.version,
            "command": self.command,
            "argv": self.argv,
            "started_at": self.started_at.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_seconds": round((finished - self.started_at).total_seconds(), 3),
            "dry_run": self.dry_run,
            "status": STATUS[self.exit_code],
            "exit_code": int(self.exit_code),
            "summary": self.summary,
            "result": self.result,
            "errors": self.errors,
        }

    def default_path(self) -> Path:
        stamp = self.started_at.strftime("%Y%m%dT%H%M%SZ")
        return runs_dir() / f"{stamp}-{_slug(self.command)}.json"

    def write(self, path: Path | None = None) -> Path:
        """Write the report atomically to ``path`` (default: ``default_path()``)."""
        target = path or self.default_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, default=str) + "\n"
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return target
