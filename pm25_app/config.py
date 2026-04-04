from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

FORECAST_CACHE_FILE = ARTIFACT_DIR / "latest_forecast.csv"
FORECAST_CACHE_META = ARTIFACT_DIR / "latest_forecast_meta.json"

# Chỉ ghi khi chạy `python run_eval_metrics.py` (tách khỏi precompute).
EVAL_METRICS_JSON = ARTIFACT_DIR / "eval_metrics.json"
EVAL_METRICS_REPORT_TXT = ARTIFACT_DIR / "eval_metrics_report.txt"
EVAL_METRICS_RUN_LOG = ARTIFACT_DIR / "eval_metrics_run.log"

