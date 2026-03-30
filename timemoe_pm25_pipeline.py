"""
Pipeline PM2.5 Time-MoE zero-shot — khớp logic HCMC_PM25_TimeMoE_ZeroShot_FullPipeline.ipynb
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM

CONTEXT_LEN = 24 * 7
PRED_LEN = 24
TRAIN_RATIO = 0.8
VAL_RATIO_IN_EVAL = 0.35

# Dữ liệu huấn luyện / hiệu chỉnh chỉ lấy từ Open-Meteo Archive & Air Quality API (không đọc CSV).
OPEN_METEO_HISTORY_START = "2021-01-01"
OPEN_METEO_HISTORY_END = "2025-12-31"

CANDIDATE_MODELS = ["Maple728/TimeMoE-200M", "Maple728/TimeMoE-50M"]
NOTEBOOK_BEST_ALPHA = 0.85


def resolve_default_timemoe_dir(base_dir: Path) -> Path:
    """
    Checkpoint TimeMoE-200M mặc định: ưu tiên base_dir/models (folder streamlit_app độc lập),
    nếu không có thì dùng models/ ở thư mục cha (vd. Code/models/ khi chạy trong repo).
    """
    under_base = base_dir / "models" / "TimeMoE-200M"
    repo_sibling = base_dir.parent / "models" / "TimeMoE-200M"
    if under_base.is_dir():
        return under_base
    if repo_sibling.is_dir():
        return repo_sibling
    return under_base
NOTEBOOK_BEST_THRESHOLD = 0.15
NOTEBOOK_BEST_FEATURES = ["wind", "hum", "temp", "solar"]


def ensure_local_hf_cache(cache_dir: str | Path | None = None) -> Path:
    """
    Ép HuggingFace cache vào thư mục project để tránh lỗi quyền trong ~/.cache.
    """
    default_dir = Path(__file__).resolve().parent / "models" / ".hf_cache"
    cdir = Path(cache_dir).expanduser() if cache_dir else default_dir
    cdir.mkdir(parents=True, exist_ok=True)

    # Hub cache + dynamic modules cache (trust_remote_code)
    os.environ["HF_HOME"] = str(cdir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cdir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cdir)
    os.environ["HF_MODULES_CACHE"] = str(cdir / "modules")
    Path(os.environ["HUGGINGFACE_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_MODULES_CACHE"]).mkdir(parents=True, exist_ok=True)
    return cdir


def pm25_aqi_band_vn(ug_m3: float) -> tuple[str, str]:
    """Nhãn mức độ gợi ý (theo ngưỡng tham chiếu phổ biến)."""
    x = float(ug_m3)
    if x <= 12:
        return "Tốt", "#00a854"
    if x <= 35:
        return "Trung bình", "#ffeb3b"
    if x <= 55:
        return "Kém", "#ff9800"
    if x <= 150:
        return "Xấu", "#f44336"
    return "Rất xấu", "#7b1fa2"


def fetch_open_meteo_hcmc(
    start_date: str = OPEN_METEO_HISTORY_START,
    end_date: str = OPEN_METEO_HISTORY_END,
    lat: float = 10.7756,
    lon: float = 106.7019,
    *,
    use_realtime_last_observation: bool = False,
) -> pd.DataFrame:
    """Tải hourly weather + PM2.5 từ Open-Meteo (archive + air-quality), khoảng [start_date, end_date].

    - ``use_realtime_last_observation=False`` (mặc định): cắt tại 23:00 ngày ``end_date`` (chuỗi lịch sử cố định).
    - ``use_realtime_last_observation=True``: cắt tại **giờ hiện tại (Asia/Ho_Chi_Minh) sàn xuống 1 giờ** để tránh giờ chưa khóa sổ trên API; dùng khi ``end_date`` là hôm nay.
    """
    weather_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "pressure_msl",
        "cloud_cover",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
        "shortwave_radiation",
    ]
    weather_url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly={','.join(weather_vars)}&timezone=Asia%2FBangkok"
    )
    aqi_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=pm2_5&timezone=Asia%2FBangkok"
    )
    # Tắt hoàn toàn proxy hệ thống để tránh lỗi tunnel 403 trên một số mạng.
    sess = requests.Session()
    sess.trust_env = False
    try:
        weather_res = sess.get(weather_url, timeout=30)
        weather_res.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Lỗi gọi API thời tiết Open-Meteo: {e}") from e

    try:
        aqi_res = sess.get(aqi_url, timeout=30)
        aqi_res.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Lỗi gọi API chất lượng không khí Open-Meteo: {e}") from e

    try:
        w_json = weather_res.json()
        a_json = aqi_res.json()
    except ValueError as e:
        raise RuntimeError(f"API Open-Meteo trả về JSON không hợp lệ: {e}") from e
    w_data = w_json.get("hourly", {})
    a_data = a_json.get("hourly", {})
    if "time" not in w_data or "pm2_5" not in a_data:
        raise RuntimeError("Không nhận được dữ liệu hợp lệ từ Open-Meteo.")
    df = pd.DataFrame(w_data)
    df["pm2_5"] = a_data["pm2_5"]
    df.rename(
        columns={
            "time": "date",
            "temperature_2m": "temp",
            "relative_humidity_2m": "hum",
            "dew_point_2m": "dew",
            "apparent_temperature": "apparent_temp",
            "pressure_msl": "pressure",
            "cloud_cover": "cloud",
            "wind_speed_10m": "wind",
            "wind_direction_10m": "wind_dir",
            "precipitation": "rain",
            "shortwave_radiation": "solar",
        },
        inplace=True,
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if use_realtime_last_observation:
        now_local_naive = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").tz_localize(None)
        cutoff = now_local_naive.floor("h") - pd.Timedelta(hours=1)
        df = df[df["date"] <= cutoff].reset_index(drop=True)
        if df.empty:
            raise RuntimeError(
                "Open-Meteo chưa có đủ dữ liệu tới mốc hiện tại − 1 giờ (chế độ realtime)."
            )
    else:
        last_hist_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23)
        df = df[df["date"] <= last_hist_ts].reset_index(drop=True)
        if df.empty:
            raise RuntimeError(
                f"Không còn dữ liệu sau khi giới hạn tới cuối ngày {end_date} (23:00)."
            )
    num_cols = [c for c in df.columns if c != "date"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df[num_cols] = df[num_cols].interpolate(limit_direction="both").bfill().ffill()
    return df


def load_timemoe_model(
    device: str | None = None,
    local_model_path: str | None = None,
    local_files_only: bool = False,
) -> tuple[Any, str, str]:
    ensure_local_hf_cache()
    import transformers

    if transformers.__version__ != "4.40.1":
        raise RuntimeError(
            f"Cần transformers==4.40.1 (hiện {transformers.__version__}) để tránh lỗi Time-MoE."
        )
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    last_err = None
    if local_model_path:
        p = Path(local_model_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Không thấy thư mục model local: {p}")
        try:
            m = AutoModelForCausalLM.from_pretrained(
                str(p),
                trust_remote_code=True,
                local_files_only=True,
            )
            m = m.to(dev)
            return m, f"local:{p}", dev
        except Exception as e:
            raise RuntimeError(f"Không load được model local từ {p}: {e}") from e

    for model_id in CANDIDATE_MODELS:
        try:
            m = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            m = m.to(dev)
            return m, model_id, dev
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Không tải được Time-MoE: {last_err}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mape = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-6))) * 100)
    r2 = float(r2_score(yt, yp))
    return {"MAE": float(mae), "RMSE": rmse, "MAPE_percent": mape, "R2": r2}


def prepare_arrays(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, float, float, pd.Series]:
    numeric_cols = [c for c in df.columns if c != "date"]
    corr = df[numeric_cols].corr(method="pearson")
    pm_corr = corr["pm2_5"].drop("pm2_5").sort_values(key=lambda s: s.abs(), ascending=False)
    BASE_FEATURES = pm_corr.index.tolist()
    TARGET_COL = "pm2_5"
    all_cols = [TARGET_COL] + BASE_FEATURES
    all_values = df[all_cols].values.astype(np.float32)
    all_values = np.nan_to_num(all_values, nan=0.0, posinf=0.0, neginf=0.0)
    split_idx = int(len(all_values) * TRAIN_RATIO)
    train_vals = all_values[:split_idx]
    eval_vals = all_values[split_idx - CONTEXT_LEN :]
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_vals)
    eval_scaled = scaler.transform(eval_vals)
    pm_eval = eval_scaled[:, 0]
    mu_y = scaler.mean_[0]
    sigma_y = scaler.scale_[0] + 1e-8
    return eval_scaled, pm_eval, scaler, mu_y, sigma_y, pm_corr


def build_signal(
    eval_scaled: np.ndarray,
    pm_eval: np.ndarray,
    pm_corr: pd.Series,
    alpha_pm: float,
    corr_threshold: float,
    forced_features: list[str] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    BASE_FEATURES = pm_corr.index.tolist()
    if forced_features:
        selected = [f for f in forced_features if f in BASE_FEATURES]
        if len(selected) == 0:
            selected = BASE_FEATURES[:5]
    else:
        selected = [f for f in BASE_FEATURES if abs(pm_corr.loc[f]) >= corr_threshold]
        if len(selected) < 3:
            selected = BASE_FEATURES[:5]
    idx_in_base = [BASE_FEATURES.index(f) for f in selected]
    feat_eval = eval_scaled[:, 1:][:, idx_in_base] if len(selected) > 0 else np.zeros((len(eval_scaled), 1), dtype=np.float32)
    w_raw = np.abs(pm_corr.loc[selected].values.astype(np.float64))
    weights = w_raw / (w_raw.sum() + 1e-12)
    impact = np.dot(feat_eval, weights.astype(np.float32)) if len(selected) > 0 else np.zeros(len(eval_scaled), dtype=np.float32)
    signal = alpha_pm * pm_eval + (1.0 - alpha_pm) * impact
    return signal, selected, weights


def make_windows_1d(
    arr_signal: np.ndarray,
    arr_pm_norm: np.ndarray,
    context_len: int,
    pred_len: int,
    max_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trượt cửa sổ ngữ cảnh. Nếu `max_windows` được set (vd. hiệu chỉnh nhanh),
    chỉ tạo đủ cửa sổ đầu tiên — tránh tạo hàng chục nghìn mảng khi chỉ cần ~160.
    """
    X, y = [], []
    max_i = len(arr_signal) - context_len - pred_len + 1
    if max_i <= 0:
        return np.empty((0, context_len), dtype=np.float32), np.empty((0, pred_len), dtype=np.float32)
    n_loop = max_i if max_windows is None else min(max_i, max_windows)
    for i in range(n_loop):
        X.append(arr_signal[i : i + context_len])
        y.append(arr_pm_norm[i + context_len : i + context_len + pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def run_predict_windows(
    model: Any,
    X_ctx: np.ndarray,
    device: str,
    pred_len: int = PRED_LEN,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_ctx), batch_size):
            xb = torch.tensor(X_ctx[i : i + batch_size], dtype=torch.float32, device=device)
            out = model.generate(xb, max_new_tokens=pred_len)
            out_np = out.detach().cpu().numpy()
            if out_np.ndim != 2:
                raise RuntimeError(f"Unexpected output shape: {out_np.shape}, expected [B, T].")
            preds.append(out_np[:, -pred_len:])
    return np.concatenate(preds, axis=0)


def run_eval_pipeline(
    df: pd.DataFrame,
    model: Any,
    device: str,
    alpha_pm: float,
    corr_threshold: float,
    forced_features: list[str] | None = None,
) -> dict[str, Any]:
    eval_scaled, pm_eval, scaler, mu_y, sigma_y, pm_corr = prepare_arrays(df)
    signal, selected_features, _ = build_signal(
        eval_scaled,
        pm_eval,
        pm_corr,
        alpha_pm=alpha_pm,
        corr_threshold=corr_threshold,
        forced_features=forced_features,
    )
    X_all, y_all_norm = make_windows_1d(signal, pm_eval, CONTEXT_LEN, PRED_LEN)
    n_total = len(X_all)
    n_val = int(n_total * VAL_RATIO_IN_EVAL)
    n_val = max(80, min(n_val, n_total - 80))
    X_val, X_test = X_all[:n_val], X_all[n_val:]
    y_val_norm, y_test_norm = y_all_norm[:n_val], y_all_norm[n_val:]
    pred_val_norm = run_predict_windows(model, X_val, device)
    pred_test_norm = run_predict_windows(model, X_test, device)
    y_val = y_val_norm * sigma_y + mu_y
    y_test = y_test_norm * sigma_y + mu_y
    pred_val = pred_val_norm * sigma_y + mu_y
    pred_test = pred_test_norm * sigma_y + mu_y
    lr = LinearRegression()
    lr.fit(pred_val.reshape(-1, 1), y_val.reshape(-1))
    a = float(lr.coef_[0])
    b = float(lr.intercept_)
    pred_test_cal = a * pred_test + b
    raw_m = compute_metrics(y_test, pred_test)
    cal_m = compute_metrics(y_test, pred_test_cal)
    return {
        "pm_corr": pm_corr,
        "selected_features": selected_features,
        "mu_y": mu_y,
        "sigma_y": sigma_y,
        "calibration_a": a,
        "calibration_b": b,
        "y_test": y_test,
        "pred_test_raw": pred_test,
        "pred_test_cal": pred_test_cal,
        "metrics_raw": raw_m,
        "metrics_cal": cal_m,
        "n_test_windows": len(y_test),
    }


def fit_calibration_quick(
    df: pd.DataFrame,
    model: Any,
    device: str,
    alpha_pm: float,
    corr_threshold: float,
    max_val_windows: int = 160,
    forced_features: list[str] | None = None,
) -> dict[str, Any]:
    """
    Bản nhanh cho suy luận 24h:
    - chỉ dùng validation để fit y = a*y_hat + b
    - giới hạn số cửa sổ để giảm thời gian chờ
    """
    eval_scaled, pm_eval, _, mu_y, sigma_y, pm_corr = prepare_arrays(df)
    signal, selected_features, _ = build_signal(
        eval_scaled,
        pm_eval,
        pm_corr,
        alpha_pm=alpha_pm,
        corr_threshold=corr_threshold,
        forced_features=forced_features,
    )
    # Chỉ dựng cửa sổ cần cho hiệu chỉnh (mặc định 160), không dựng toàn bộ ~30k+ cửa.
    X_val, y_val_norm = make_windows_1d(
        signal, pm_eval, CONTEXT_LEN, PRED_LEN, max_windows=max_val_windows
    )

    pred_val_norm = run_predict_windows(model, X_val, device)
    y_val = y_val_norm * sigma_y + mu_y
    pred_val = pred_val_norm * sigma_y + mu_y

    lr = LinearRegression()
    lr.fit(pred_val.reshape(-1, 1), y_val.reshape(-1))
    a = float(lr.coef_[0])
    b = float(lr.intercept_)
    return {
        "calibration_a": a,
        "calibration_b": b,
        "selected_features": selected_features,
        "n_val_windows_used": len(X_val),
    }


def forecast_next_hours(
    df: pd.DataFrame,
    model: Any,
    device: str,
    alpha_pm: float,
    corr_threshold: float,
    cal_a: float,
    cal_b: float,
    forced_features: list[str] | None = None,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Dự báo PRED_LEN giờ tiếp theo sau quan sát cuối: ngữ cảnh = CONTEXT_LEN giờ cuối của tín hiệu.
    Trả về: mốc thời gian dự báo, giá trị đã hiệu chỉnh, giá trị thô.
    """
    eval_scaled, pm_eval, _, mu_y, sigma_y, pm_corr = prepare_arrays(df)
    signal, _, _ = build_signal(
        eval_scaled,
        pm_eval,
        pm_corr,
        alpha_pm=alpha_pm,
        corr_threshold=corr_threshold,
        forced_features=forced_features,
    )
    if len(signal) < CONTEXT_LEN:
        raise ValueError("Chuỗi quá ngắn so với CONTEXT_LEN.")
    ctx = signal[-CONTEXT_LEN:][np.newaxis, :].astype(np.float32)
    pred_norm = run_predict_windows(model, ctx, device, pred_len=PRED_LEN, batch_size=1)
    raw = pred_norm[0] * sigma_y + mu_y
    cal = cal_a * raw + cal_b
    last_t = df["date"].iloc[-1]
    future_idx = pd.date_range(last_t + pd.Timedelta(hours=1), periods=PRED_LEN, freq="h")
    return future_idx, cal, raw
