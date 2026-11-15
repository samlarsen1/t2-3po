"""Keep every test's run reports and locks out of the working tree and the user's lock dir."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def t2_contract_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    base = tmp_path_factory.mktemp("t2")
    monkeypatch.setenv("T2_RUNS_DIR", str(base / "runs"))
    monkeypatch.setenv("T2_LOCK_DIR", str(base / "locks"))
    return base
