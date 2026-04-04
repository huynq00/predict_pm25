"""
Đánh giá MAE / RMSE / MAPE / R² trên dự báo **thô** — tách khỏi precompute.
Ghi ``artifacts/eval_metrics*.json|txt|log``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from pm25_app.config import (
    ARTIFACT_DIR,
    BASE_DIR,
    EVAL_METRICS_JSON,
    EVAL_METRICS_REPORT_TXT,
    EVAL_METRICS_RUN_LOG,
)
from pm25_app.env_utils import load_env_file
from pm25_app.precompute_main import ensure_pm25_logging, load_dataset
from timemoe_pm25_pipeline import (
    NOTEBOOK_BEST_ALPHA,
    NOTEBOOK_BEST_FEATURES,
    NOTEBOOK_BEST_THRESHOLD,
    compute_test_metrics_on_test_split,
    load_timemoe_model,
    resolve_default_timemoe_dir,
    run_eval_pipeline,
)

_LOG = logging.getLogger("pm25.eval_metrics")


def _setup_file_log() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(EVAL_METRICS_RUN_LOG, encoding="utf-8", mode="a")
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger("pm25")
    if not any(getattr(h, "baseFilename", None) == str(EVAL_METRICS_RUN_LOG) for h in root.handlers):
        root.addHandler(fh)


def _metrics_row(m: dict[str, float]) -> str:
    return (
        f"  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
        f"MAPE={m['MAPE_percent']:.4f}%  R2={m['R2']:.6f}"
    )


def build_report_text(
    *,
    mode: str,
    generated_utc: str,
    dataset_last: str,
    data_mode: str,
    model_id: str,
    max_test_windows: int | None,
    n_test_full: int | None,
    n_test_used: int | None,
    selected_features: list[str],
    metrics: dict[str, float],
    duration_s: float,
) -> str:
    lines = [
        "=" * 72,
        "KẾT QUẢ ĐÁNH GIÁ DỰ BÁO PM2.5 (Time-MoE, zero-shot, dự báo thô)",
        "=" * 72,
        "",
        f"Thời điểm chạy (UTC):     {generated_utc}",
        f"Chế độ dữ liệu:           {data_mode}",
        f"Quan sát cuối (dataset):  {dataset_last}",
        f"Model:                    {model_id}",
        f"Chế độ đánh giá:          {mode}",
        f"Thời gian chạy:           {duration_s:.1f} s",
        "",
        "Cấu hình pipeline:",
        f"  alpha_pm={NOTEBOOK_BEST_ALPHA}  corr_threshold={NOTEBOOK_BEST_THRESHOLD}",
        f"  forced_features={NOTEBOOK_BEST_FEATURES}",
        "",
    ]
    if n_test_full is not None:
        lines.append(f"Số cửa sổ test (toàn phần):       {n_test_full}")
    if n_test_used is not None:
        if max_test_windows is not None:
            lines.append(f"Số cửa sổ test (suy luận):        {n_test_used} (giới hạn {max_test_windows})")
        else:
            lines.append(f"Số cửa sổ test (suy luận):        {n_test_used} (toàn bộ tập test)")
    lines += [
        f"Đặc trưng thời tiết chọn: {', '.join(selected_features)}",
        "",
        "-" * 72,
        "ĐỘ ĐO (µg/m³ cho MAE/RMSE)",
        "-" * 72,
        _metrics_row(metrics),
        "",
        "Gợi ý Markdown:",
        "",
        "| MAE | RMSE | MAPE (%) | R² |",
        "|-----|------|----------|-----|",
        f"| {metrics['MAE']:.4f} | {metrics['RMSE']:.4f} | {metrics['MAPE_percent']:.4f} | {metrics['R2']:.6f} |",
        "",
        "=" * 72,
    ]
    return "\n".join(lines)


def _json_safe_pm_corr(pm_corr: Any) -> dict[str, float]:
    return {str(k): float(v) for k, v in pm_corr.items()}


def run_default_test_split(
    df: Any,
    model: Any,
    device: str,
    model_id: str,
    data_mode: str,
    max_test_windows: int | None,
) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    em = compute_test_metrics_on_test_split(
        df,
        model,
        device,
        alpha_pm=NOTEBOOK_BEST_ALPHA,
        corr_threshold=NOTEBOOK_BEST_THRESHOLD,
        forced_features=NOTEBOOK_BEST_FEATURES,
        max_test_windows=max_test_windows,
    )
    duration = time.perf_counter() - t0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload: dict[str, Any] = {
        "eval_mode": "test_split_raw",
        "generated_at_utc": now,
        "dataset_last_time": str(df["date"].iloc[-1]),
        "data_mode": data_mode,
        "model_id": model_id,
        "max_test_windows_cap": max_test_windows,
        **em,
    }
    return payload, duration


def run_full_notebook_style(df: Any, model: Any, device: str, model_id: str, data_mode: str) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    out = run_eval_pipeline(
        df,
        model,
        device,
        alpha_pm=NOTEBOOK_BEST_ALPHA,
        corr_threshold=NOTEBOOK_BEST_THRESHOLD,
        forced_features=NOTEBOOK_BEST_FEATURES,
    )
    duration = time.perf_counter() - t0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload: dict[str, Any] = {
        "eval_mode": "full_run_eval_pipeline_raw",
        "generated_at_utc": now,
        "dataset_last_time": str(df["date"].iloc[-1]),
        "data_mode": data_mode,
        "model_id": model_id,
        "selected_features": out["selected_features"],
        "metrics": out["metrics"],
        "n_test_windows": out["n_test_windows"],
        "n_test_windows_used": out["n_test_windows"],
        "n_test_windows_full": out["n_test_windows"],
        "pm_corr_pearson": _json_safe_pm_corr(out["pm_corr"]),
        "mu_y_pm25": float(out["mu_y"]),
        "sigma_y_pm25": float(out["sigma_y"]),
    }
    return payload, duration


def write_artifacts(payload: dict[str, Any], report: str) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_METRICS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    EVAL_METRICS_REPORT_TXT.write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Độ đo trên dự báo thô PM2.5, ghi vào artifacts/."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="run_eval_pipeline đầy đủ (chỉ tập test, rất lâu).",
    )
    parser.add_argument(
        "--max-test-windows",
        type=int,
        default=None,
        help="Giới hạn cửa sổ test gần nhất; bỏ qua = toàn bộ test.",
    )
    args = parser.parse_args(argv)

    ensure_pm25_logging()
    _setup_file_log()
    load_env_file(BASE_DIR / ".env", override_all=True)

    mt: int | None = args.max_test_windows
    if mt is not None:
        mt = max(1, min(mt, 50_000))

    dm = os.getenv("PM25_DATA_MODE", "historical").strip().lower()
    data_mode = "realtime" if dm in ("realtime", "live") else "historical"

    _LOG.info("eval_metrics: tải dataset (%s)...", data_mode)
    df = load_dataset()
    _LOG.info("eval_metrics: %d dòng, cuối=%s", len(df), df["date"].iloc[-1])

    raw_ckpt = os.getenv("MODEL_CHECKPOINT", "").strip()
    model_path = raw_ckpt or str(resolve_default_timemoe_dir(BASE_DIR))
    model, model_id, device = load_timemoe_model(device="cpu", local_model_path=model_path)

    if args.full:
        _LOG.info("eval_metrics: chế độ --full")
        payload, duration = run_full_notebook_style(df, model, device, model_id, data_mode)
        report = build_report_text(
            mode="run_eval_pipeline (toàn bộ cửa sổ test)",
            generated_utc=payload["generated_at_utc"],
            dataset_last=payload["dataset_last_time"],
            data_mode=payload["data_mode"],
            model_id=payload["model_id"],
            max_test_windows=None,
            n_test_full=payload["n_test_windows_full"],
            n_test_used=payload["n_test_windows_used"],
            selected_features=payload["selected_features"],
            metrics=payload["metrics"],
            duration_s=duration,
        )
    else:
        test_desc = f"giới hạn {mt}" if mt is not None else "toàn bộ"
        _LOG.info("eval_metrics: test split (%s)", test_desc)
        payload, duration = run_default_test_split(df, model, device, model_id, data_mode, mt)
        report = build_report_text(
            mode=f"compute_test_metrics_on_test_split ({test_desc})",
            generated_utc=payload["generated_at_utc"],
            dataset_last=payload["dataset_last_time"],
            data_mode=payload["data_mode"],
            model_id=payload["model_id"],
            max_test_windows=mt,
            n_test_full=payload.get("n_test_windows_full"),
            n_test_used=payload.get("n_test_windows_used"),
            selected_features=payload["selected_features"],
            metrics=payload["metrics"],
            duration_s=duration,
        )

    write_artifacts(payload, report)
    _LOG.info("Đã ghi %s và %s", EVAL_METRICS_JSON, EVAL_METRICS_REPORT_TXT)
    print(report)
    print(f"\n[OK] {EVAL_METRICS_JSON}")
    print(f"[OK] {EVAL_METRICS_REPORT_TXT}")
    print(f"[OK] log append: {EVAL_METRICS_RUN_LOG}")


if __name__ == "__main__":
    main()
