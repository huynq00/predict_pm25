from __future__ import annotations

import json
from typing import Literal

import pandas as pd

from pm25_app.config import ARTIFACT_DIR, FORECAST_CACHE_META
from pm25_app.precompute_main import run_once

LockOutcome = Literal["ok", "busy", "error"]


def cache_age_minutes() -> float | None:
    if not FORECAST_CACHE_META.is_file():
        return None
    try:
        meta = json.loads(FORECAST_CACHE_META.read_text(encoding="utf-8"))
        gen = pd.to_datetime(meta.get("generated_at_utc"), utc=True)
        now = pd.Timestamp.now(tz="UTC")
        return (now - gen).total_seconds() / 60.0
    except Exception:
        return None


def should_trigger_precompute(interval_minutes: int) -> bool:
    """Chạy lại khi chưa có meta hoặc cache đã cũ hơn interval (tránh precompute lặp mỗi lần reload khi cache còn mới)."""
    age = cache_age_minutes()
    if age is None:
        return True
    return age >= float(interval_minutes)


def run_precompute_locked() -> tuple[LockOutcome, str | None]:
    """
    Chạy run_once với file lock để tránh hai phiên Streamlit/tab chạy trùng.
    Trả về ("ok", None) | ("busy", None) | ("error", message).
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = ARTIFACT_DIR / ".precompute.lock"
    try:
        import fcntl
    except ImportError:
        try:
            run_once()
            return "ok", None
        except Exception as e:
            return "error", str(e)

    lock_f = open(lock_path, "w")
    try:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_f.close()
        return "busy", None
    try:
        try:
            run_once()
            return "ok", None
        except Exception as e:
            return "error", str(e)
    finally:
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock_f.close()
