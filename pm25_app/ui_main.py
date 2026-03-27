from __future__ import annotations

import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from pm25_app.config import BASE_DIR, FORECAST_CACHE_FILE, FORECAST_CACHE_META
from pm25_app.env_utils import load_env_file
from timemoe_pm25_pipeline import CONTEXT_LEN, PRED_LEN, pm25_aqi_band_vn


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .app-card {
            background: linear-gradient(135deg, rgba(71, 85, 105, 0.20), rgba(30, 41, 59, 0.35));
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 14px;
            padding: 14px 16px;
            margin: 8px 0 14px 0;
        }
        .app-card h4 { margin: 0 0 6px 0; }
        .small-muted { opacity: 0.86; font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def schedule_auto_refresh(minutes: int = 60) -> None:
    ms = max(1, int(minutes)) * 60 * 1000
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {ms});
        </script>
        """,
        height=0,
    )


def load_forecast_cache() -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict | None]:
    if not FORECAST_CACHE_FILE.is_file() or not FORECAST_CACHE_META.is_file():
        return None, None, None

    try:
        raw_df = pd.read_csv(FORECAST_CACHE_FILE)
        raw_df["Thời gian"] = pd.to_datetime(raw_df["Thời gian"])
        fc_df = raw_df.rename(columns={"Calibrated": "PM2.5 (μg/m³)"})[
            ["Thời gian", "PM2.5 (μg/m³)"]
        ]
        meta = json.loads(FORECAST_CACHE_META.read_text(encoding="utf-8"))
        return fc_df, raw_df, meta
    except Exception:
        return None, None, None


def render_forecast_chart(fc_df: pd.DataFrame) -> None:
    vals = fc_df["PM2.5 (μg/m³)"].to_numpy(dtype=float)
    plot_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(fc_df["Thời gian"]),
            "pm25": vals,
        }
    )

    q_low = float(np.quantile(vals, 0.05))
    q_high = float(np.quantile(vals, 0.95))
    if np.isclose(q_low, q_high):
        q_pad = max(0.8, abs(q_low) * 0.04)
    else:
        q_pad = max(0.8, (q_high - q_low) * 0.3)
    zoom_domain = [q_low - q_pad, q_high + q_pad]

    zoom_chart = (
        alt.Chart(plot_df)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=50), strokeWidth=3)
        .encode(
            x=alt.X("timestamp:T", title="Thời gian"),
            y=alt.Y(
                "pm25:Q",
                title="PM2.5 (μg/m³) — zoom",
                scale=alt.Scale(domain=zoom_domain),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Thời gian"),
                alt.Tooltip("pm25:Q", title="PM2.5 (μg/m³)", format=".3f"),
            ],
        )
        .properties(height=260)
    )
    st.caption("Biểu đồ dự báo 24h (đã tính sẵn).")
    st.altair_chart(zoom_chart, width="stretch")


def render_warning_recommendations(fc_df: pd.DataFrame, llm_text: str | None = None) -> None:
    vals = fc_df["PM2.5 (μg/m³)"].to_numpy(dtype=float)
    times = pd.to_datetime(fc_df["Thời gian"])
    avg24 = float(np.mean(vals))
    peak_idx = int(np.argmax(vals))
    peak_val = float(vals[peak_idx])
    peak_time = times.iloc[peak_idx]
    band, color = pm25_aqi_band_vn(avg24)

    st.subheader("Cảnh báo & khuyến nghị")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mức trung bình 24h", f"{avg24:.1f} μg/m³")
    c2.metric("Đỉnh dự báo", f"{peak_val:.1f} μg/m³")
    c3.metric("Khung giờ đỉnh", peak_time.strftime("%d-%m %H:%M"))

    st.markdown(
        (
            "<div class='app-card'>"
            "<h4>Mức cảnh báo hiện tại</h4>"
            f"<div><span style='color:{color};font-weight:700'>{band}</span></div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    if peak_val > 55:
        st.error(
            f"Khung giờ rủi ro cao quanh {peak_time.strftime('%H:%M')} "
            f"(PM2.5 ~ {peak_val:.1f} μg/m³). Nên hạn chế hoạt động ngoài trời."
        )
    elif peak_val > 35:
        st.warning(
            f"Khung giờ cần lưu ý quanh {peak_time.strftime('%H:%M')} "
            f"(PM2.5 ~ {peak_val:.1f} μg/m³)."
        )
    else:
        st.success("Mức dự báo không quá cao trong 24h tới.")

    if llm_text and not llm_text.startswith("(LLM precompute lỗi:"):
        st.markdown(
            "<div class='app-card'><h4>Khuyến nghị từ LLM (precompute)</h4>"
            "<div class='small-muted'>Sinh trước trên server/cron; trang web chỉ hiển thị.</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(llm_text)
    elif llm_text and llm_text.startswith("(LLM precompute lỗi:"):
        st.error(llm_text)
    else:
        st.info(
            "Chưa có khuyến nghị LLM trong cache. "
            "Chạy `precompute_forecast.py` với API key trong `.env` để sinh trước."
        )

    high_df = fc_df[fc_df["PM2.5 (μg/m³)"] >= max(35.0, avg24)]
    if len(high_df) > 0:
        st.caption("Khung giờ nên hạn chế ra ngoài:")
        st.dataframe(
            high_df[["Thời gian", "PM2.5 (μg/m³)"]].reset_index(drop=True),
            width="stretch",
        )


def main() -> None:
    st.set_page_config(
        page_title="Dự báo PM2.5 TP.HCM (Time-MoE)",
        page_icon="🌫️",
        layout="wide",
    )
    load_env_file(BASE_DIR / ".env")
    inject_custom_css()

    st.title("Dự báo chất lượng không khí — PM2.5 (TP.HCM)")
    st.caption(
        f"Chỉ hiển thị kết quả đã tính trước (ngữ cảnh {CONTEXT_LEN} giờ, dự báo {PRED_LEN} giờ). "
        "Không chạy mô hình trên trình duyệt."
    )

    fc_df, cache_raw_df, cache_meta = load_forecast_cache()
    if fc_df is None or cache_raw_df is None or cache_meta is None:
        st.error(
            "Chưa có dữ liệu dự báo trong `artifacts/`. "
            "Trên máy chủ, chạy: `python precompute_forecast.py --once` trong thư mục `streamlit_app`."
        )
        st.stop()

    with st.sidebar:
        st.header("Trang xem nhanh")
        refresh_minutes = st.number_input(
            "Tự làm mới trang (phút)",
            min_value=15,
            max_value=180,
            value=60,
            step=15,
            help="Chỉ tải lại trang để lấy file cache mới sau khi precompute cập nhật.",
        )
        st.caption(
            "Dự báo được cập nhật bởi tiến trình `precompute_forecast` (cron/systemd), không phải bởi mỗi lần mở web."
        )

    schedule_auto_refresh(int(refresh_minutes))
    st.caption(f"Tự động làm mới trang mỗi {int(refresh_minutes)} phút (để thấy cache mới nếu có).")

    now_ts = pd.Timestamp.utcnow()
    gen_at = cache_meta.get("generated_at_utc", "unknown")
    try:
        generated_at = pd.to_datetime(cache_meta.get("generated_at_utc"))
        age_sec = (now_ts - generated_at).total_seconds()
        fresh = age_sec >= 0 and age_sec < int(refresh_minutes) * 60
    except Exception:
        fresh = False

    if fresh:
        st.success(f"Cache còn mới (generated_at={gen_at} UTC).")
    else:
        st.warning(
            f"Đang hiển thị cache đã lưu (generated_at={gen_at} UTC). "
            "Nếu cần số mới, chạy lại precompute trên server."
        )

    st.subheader("Thông tin lần tính gần nhất")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Quan sát cuối (dataset)", str(cache_meta.get("dataset_last_time", "—")))
    model_id = str(cache_meta.get("model_id", "—"))
    m2.metric("Model", model_id[:32] + ("…" if len(model_id) > 32 else ""))
    m3.metric("Cửa sổ val (precompute)", str(cache_meta.get("n_val_windows_used", "—")))
    m4.metric("max_val_windows", str(cache_meta.get("max_val_windows_quick", "—")))

    st.divider()
    st.header("Dự báo 24 giờ tới (đã tính sẵn)")
    st.caption(
        "Toàn bộ đường cong và bảng bên dưới đọc từ `artifacts/latest_forecast*.csv/json` — không suy luận trên web."
    )

    llm_text_cached = cache_meta.get("llm_text", "")
    raw = cache_raw_df["Raw"].to_numpy(dtype=float)
    idx = fc_df["Thời gian"].to_numpy()
    cal = fc_df["PM2.5 (μg/m³)"].to_numpy(dtype=float)

    render_forecast_chart(fc_df)
    band, color = pm25_aqi_band_vn(float(np.mean(cal)))
    st.markdown(
        f"**Trung bình 24h (gợi ý):** {np.mean(cal):.1f} μg/m³ — "
        f'<span style="color:{color}">**{band}**</span>',
        unsafe_allow_html=True,
    )
    render_warning_recommendations(fc_df, llm_text=llm_text_cached)
    with st.expander("Bảng số & bản thô (chưa hiệu chỉnh)"):
        st.dataframe(
            pd.DataFrame({"Thời gian": idx, "Calibrated": cal, "Raw": raw}),
            width="stretch",
        )

    st.divider()
    st.markdown(
        "**Ghi chú:** Kết quả do pipeline Time-MoE zero-shot + hiệu chỉnh tuyến tính (xem notebook). "
        "Tham chiếu sức khỏe cần đối chiếu quy chuẩn Việt Nam / WHO."
    )


if __name__ == "__main__":
    main()

