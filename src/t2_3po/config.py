"""Environment-driven configuration. Secrets live in ``.env``, which is gitignored."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ModelProvider = Literal["google", "ollama"]


class Settings(BaseSettings):
    """Runtime settings, loaded from the environment and then ``.env``."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    t212_api_key: SecretStr | None = None
    t212_api_secret: SecretStr | None = None
    t212_base_url: str = "https://live.trading212.com"

    # Both choices are free: Gemini's free API tier, or a local model served by Ollama.
    model_provider: ModelProvider = "google"

    google_api_key: SecretStr | None = None
    # A Flash model on the free tier. Google retires model names, so check them with
    # `t2-3po models` and override with GEMINI_MODEL.
    gemini_model: str = "gemini-3.7-flash"
    # The free tier allows only a few requests a minute, so requests are spaced at least this far
    # apart. 7s keeps a run under 10 requests a minute.
    gemini_min_interval_seconds: float = Field(default=7.0, ge=0)

    # A local model served by Ollama, used when MODEL_PROVIDER=ollama. Private, but much weaker.
    ollama_model: str = "qwen3:4b"
    ollama_base_url: str = "http://localhost:11434/v1"

    max_sector_weight: float = Field(default=0.30, gt=0, le=1)
    # Matches the largest max_portfolio_weight_pct in t2-d2's strategy.yaml (aggressive mode).
    max_position_weight: float = Field(default=0.175, gt=0, le=1)

    macro_thesis_file: Path = Path("theses/macro/core.md")
    stocks_file: Path = Path("../t2-d2/config/stocks.yaml")

    # The other droids, side by side with this one (deployed or in development).
    d2_dir: Path = Path("../t2-d2")
    k2_dir: Path = Path("../t2-k2")
    bb8_dir: Path = Path("../t2-bb8")
    suggestions_dir: Path = Path("var/suggestions")
    # Run reports from the last this-many days feed the advice.
    advise_lookback_days: int = Field(default=30, ge=1, le=365)
    max_suggestions: int = Field(default=5, ge=1, le=10)
    # A rejected suggestion isn't made again for this long.
    rejected_memory_days: int = Field(default=90, ge=1)

    theses_dir: Path = Path("theses")
    reports_dir: Path = Path("reports")
    proposals_dir: Path = Path("proposals")
    export_dir: Path = Path("exports")

    log_level: str = "INFO"
