"""Environment-driven configuration. Secrets live in ``.env``, which is gitignored."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, loaded from the environment and then ``.env``."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    t212_api_key: SecretStr | None = None
    t212_api_secret: SecretStr | None = None
    t212_base_url: str = "https://live.trading212.com"

    anthropic_api_key: SecretStr | None = None
    claude_model: str = "claude-sonnet-5-5"

    max_sector_weight: float = Field(default=0.30, gt=0, le=1)
    # Matches the largest max_portfolio_weight_pct in t2-d2's strategy.yaml (aggressive mode).
    max_position_weight: float = Field(default=0.175, gt=0, le=1)

    theses_dir: Path = Path("theses")
    reports_dir: Path = Path("reports")
    proposals_dir: Path = Path("proposals")
    export_dir: Path = Path("exports")

    log_level: str = "INFO"
