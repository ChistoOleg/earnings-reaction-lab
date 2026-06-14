from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ERL_",
        case_sensitive=False,
        extra="ignore",
    )

    fmp_api_key: str = ""
    data_dir: str = "data"
    start_date: str = "2015-01-01"
    request_interval_seconds: float = 0.25
    max_retries: int = 3
    http_timeout_seconds: float = 30.0
    benchmark_symbol: str = "^GSPC"
    extra_benchmarks: str = "^VIX,^TNX"
    sector_etfs: str = "XLK,XLF,XLV,XLY,XLP,XLE,XLI,XLB,XLU,XLRE,XLC"

    @field_validator("start_date")
    @classmethod
    def _validate_start(cls, value: str) -> str:
        parts = value.split("-")
        if len(parts) != 3 or len(parts[0]) != 4:
            raise ValueError(f"start_date must be YYYY-MM-DD, got {value!r}")
        int(parts[0]), int(parts[1]), int(parts[2])
        return value

    @property
    def raw_dir(self) -> Path:
        return Path(self.data_dir) / "raw"

    @property
    def interim_dir(self) -> Path:
        return Path(self.data_dir) / "interim"

    @property
    def processed_dir(self) -> Path:
        return Path(self.data_dir) / "processed"

    @property
    def benchmark_symbols(self) -> list[str]:
        extras = [s.strip() for s in self.extra_benchmarks.split(",") if s.strip()]
        etfs = [s.strip() for s in self.sector_etfs.split(",") if s.strip()]
        return [self.benchmark_symbol, *extras, *etfs]

    def ensure_dirs(self) -> None:
        for path in (self.raw_dir, self.interim_dir, self.processed_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
