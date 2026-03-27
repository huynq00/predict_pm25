from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"

DEFAULT_FALLBACK_CSV = DATA_DIR / "fallback_pm25.csv"
FORECAST_CACHE_FILE = ARTIFACT_DIR / "latest_forecast.csv"
FORECAST_CACHE_META = ARTIFACT_DIR / "latest_forecast_meta.json"

