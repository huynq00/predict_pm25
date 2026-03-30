from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from pm25_app.config import ARTIFACT_DIR, BASE_DIR, FORECAST_CACHE_FILE, FORECAST_CACHE_META
from pm25_app.env_utils import load_env_file
from timemoe_pm25_pipeline import (
    NOTEBOOK_BEST_ALPHA,
    NOTEBOOK_BEST_FEATURES,
    NOTEBOOK_BEST_THRESHOLD,
    OPEN_METEO_HISTORY_START,
    fetch_open_meteo_hcmc,
    fit_calibration_quick,
    forecast_next_hours,
    load_timemoe_model,
    resolve_default_timemoe_dir,
)


def dataset_data_mode() -> str:
    m = os.getenv("PM25_DATA_MODE", "historical").strip().lower()
    return "realtime" if m in ("realtime", "live") else "historical"


def write_cache(fc_df: pd.DataFrame, raw: np.ndarray, meta: dict) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "Thời gian": pd.to_datetime(fc_df["Thời gian"]),
            "Calibrated": fc_df["PM2.5 (μg/m³)"].astype(float).to_numpy(),
            "Raw": np.asarray(raw, dtype=float),
        }
    )
    out.to_csv(FORECAST_CACHE_FILE, index=False)
    FORECAST_CACHE_META.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def pm25_aqi_band_vn(ug_m3: float) -> str:
    x = float(ug_m3)
    if x <= 12:
        return "Tốt"
    if x <= 35:
        return "Trung bình"
    if x <= 55:
        return "Kém"
    if x <= 150:
        return "Xấu"
    return "Rất xấu"


def generate_llm_alert_from_forecast(fc_df: pd.DataFrame) -> str:
    vals = fc_df["PM2.5 (μg/m³)"].to_numpy(dtype=float)
    peak_idx = int(np.argmax(vals))
    level = pm25_aqi_band_vn(float(np.mean(vals)))
    prompt = (
        "Bạn là chuyên gia môi trường không khí. Viết ngắn gọn, thực tế, tiếng Việt.\n\n"
        "Dữ liệu dự báo PM2.5 trong 24h tới:\n"
        f"- trung bình: {float(np.mean(vals)):.1f} ug/m3\n"
        f"- cao nhất: {float(vals[peak_idx]):.1f} ug/m3 tại {fc_df['Thời gian'].iloc[peak_idx]}\n"
        f"- mức cảnh báo hiện tại: {level}\n\n"
        "Hãy đưa ra 4-6 gạch đầu dòng khuyến nghị cho người dân TP.HCM: "
        "nhóm bình thường, nhóm nhạy cảm, khung giờ nên tránh, và 1 lưu ý y tế."
    )
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    try:
        if provider == "gemini":
            gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
            if not gemini_key:
                raise RuntimeError(
                    "Thiếu GEMINI_API_KEY trong môi trường / .env — không sinh khuyến nghị LLM."
                )

            from google import genai

            client = genai.Client(api_key=gemini_key)
            model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
            response = client.models.generate_content(model=model, contents=prompt)
            text = (response.text or "").strip()
            if not text:
                raise RuntimeError("Gemini trả về nội dung rỗng.")
            return text

        import requests

        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Thiếu LLM_API_KEY trong môi trường / .env — không sinh khuyến nghị LLM (provider OpenAI-compatible)."
            )
        api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        payload = {
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "Bạn là chuyên gia môi trường không khí. Viết ngắn gọn, thực tế, tiếng Việt."},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        r = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        if not text:
            raise RuntimeError("LLM trả về nội dung rỗng.")
        return text
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Lỗi gọi LLM khi precompute: {e}") from e


def load_dataset() -> pd.DataFrame:
    """Open-Meteo: mặc định lịch sử 2021-01-01 → 2025-12-31; với PM25_DATA_MODE=realtime thì end_date=hôm nay và cắt tại now−1h."""
    if dataset_data_mode() == "realtime":
        end_live = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").strftime("%Y-%m-%d")
        return fetch_open_meteo_hcmc(
            start_date=OPEN_METEO_HISTORY_START,
            end_date=end_live,
            use_realtime_last_observation=True,
        )
    return fetch_open_meteo_hcmc()


def run_once(max_val_windows_quick: int = 160) -> None:
    load_env_file(BASE_DIR / ".env", override_all=True)
    df = load_dataset()
    model_path = os.getenv("MODEL_CHECKPOINT", str(resolve_default_timemoe_dir(BASE_DIR)))
    model, model_id, device = load_timemoe_model(
        device="cpu", local_model_path=model_path, local_files_only=True
    )
    ev = fit_calibration_quick(
        df,
        model,
        device,
        alpha_pm=NOTEBOOK_BEST_ALPHA,
        corr_threshold=NOTEBOOK_BEST_THRESHOLD,
        max_val_windows=max_val_windows_quick,
        forced_features=NOTEBOOK_BEST_FEATURES,
    )
    idx, cal, raw = forecast_next_hours(
        df,
        model,
        device,
        alpha_pm=NOTEBOOK_BEST_ALPHA,
        corr_threshold=NOTEBOOK_BEST_THRESHOLD,
        cal_a=ev["calibration_a"],
        cal_b=ev["calibration_b"],
        forced_features=NOTEBOOK_BEST_FEATURES,
    )
    cal = np.nan_to_num(cal, nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    fc_df = pd.DataFrame({"Thời gian": idx, "PM2.5 (μg/m³)": cal})
    llm_text = generate_llm_alert_from_forecast(fc_df)
    now_ts = pd.Timestamp.utcnow()
    write_cache(
        fc_df,
        raw,
        {
            "generated_at_utc": str(now_ts),
            "dataset_last_time": str(df["date"].iloc[-1]),
            "data_mode": dataset_data_mode(),
            "model_id": str(model_id),
            "max_val_windows_quick": int(max_val_windows_quick),
            "n_val_windows_used": ev.get("n_val_windows_used"),
            "llm_text": llm_text,
        },
    )
    print(
        f"[OK] cache updated at {now_ts} UTC | mean={float(np.mean(cal)):.2f} | "
        f"min={float(np.min(cal)):.2f} | max={float(np.max(cal)):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one precompute cycle and exit.")
    parser.add_argument(
        "--loop-minutes", type=int, default=60, help="Loop interval in minutes (default: 60)."
    )
    parser.add_argument(
        "--max-val-windows-quick", type=int, default=160, help="Validation windows for calibration."
    )
    args = parser.parse_args()

    if args.once:
        run_once(max_val_windows_quick=args.max_val_windows_quick)
        return

    while True:
        try:
            run_once(max_val_windows_quick=args.max_val_windows_quick)
        except Exception as e:
            print(f"[ERROR] precompute failed: {e}")
        sleep_sec = max(1, int(args.loop_minutes)) * 60
        time.sleep(sleep_sec)


if __name__ == "__main__":
    main()

