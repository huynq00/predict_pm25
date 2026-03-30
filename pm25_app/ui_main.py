from __future__ import annotations

import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from pm25_app.config import BASE_DIR, FORECAST_CACHE_FILE, FORECAST_CACHE_META
from pm25_app.env_utils import load_env_file
from pm25_app.health_recommendations_card import parse_llm_bullets, render_health_recommendations_card
from pm25_app.hourly_strip import render_hourly_forecast_strip
from pm25_app.precompute_trigger import run_precompute_locked, should_trigger_precompute
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


def load_forecast_cache() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not FORECAST_CACHE_FILE.is_file():
        raise FileNotFoundError(
            f"Thiếu {FORECAST_CACHE_FILE}. Chạy `python precompute_forecast.py --once` trong thư mục streamlit_app."
        )
    if not FORECAST_CACHE_META.is_file():
        raise FileNotFoundError(
            f"Thiếu {FORECAST_CACHE_META}. Chạy `python precompute_forecast.py --once` trong thư mục streamlit_app."
        )
    try:
        raw_df = pd.read_csv(FORECAST_CACHE_FILE)
        raw_df["Thời gian"] = pd.to_datetime(raw_df["Thời gian"])
        fc_df = raw_df.rename(columns={"Calibrated": "PM2.5 (μg/m³)"})[
            ["Thời gian", "PM2.5 (μg/m³)"]
        ]
        meta = json.loads(FORECAST_CACHE_META.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Không đọc hoặc không parse được cache dự báo: {e}") from e
    return fc_df, raw_df, meta


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

    llm_err_old = bool(llm_text and llm_text.startswith("(LLM precompute lỗi:"))
    if llm_err_old:
        st.error(
            f"Cache chứa lỗi LLM cũ. Xóa `artifacts/` và chạy lại precompute. {llm_text}"
        )
    elif not (llm_text and llm_text.strip()):
        st.error(
            "Cache không có khuyến nghị LLM (`llm_text` rỗng). "
            "Cấu hình GEMINI_API_KEY / LLM_API_KEY trong `.env` rồi chạy lại precompute."
        )

    render_health_recommendations_card(
        peak_pm25=peak_val,
        llm_text=llm_text if not llm_err_old else None,
        llm_error=llm_err_old,
    )

    if llm_text and llm_text.strip() and not llm_err_old and not parse_llm_bullets(llm_text):
        with st.expander("Văn bản đầy đủ từ LLM"):
            st.markdown(llm_text)

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
        f"Ngữ cảnh {CONTEXT_LEN} giờ, dự báo {PRED_LEN} giờ. "
        "Precompute (Open-Meteo + Time-MoE + LLM) chạy **trên máy chủ Streamlit**, không chạy trong trình duyệt."
    )

    with st.sidebar:
        st.header("Trang xem nhanh")
        refresh_minutes = st.number_input(
            "Chu kỳ làm mới dự báo + tải lại trang (phút)",
            min_value=15,
            max_value=180,
            value=60,
            step=15,
            key="pm25_refresh_minutes",
            help="Mỗi lần tải trang: nếu chưa có cache hoặc cache cũ hơn số phút này thì chạy precompute trên server. Trình duyệt tự reload sau cùng khoảng thời gian để kiểm tra lại.",
        )
        st.caption(
            "Có thể vẫn chạy `python precompute_forecast.py` từ terminal hoặc cron nếu muốn cập nhật khi không ai mở web."
        )

    if should_trigger_precompute(int(refresh_minutes)):
        with st.spinner(
            "Đang cập nhật dự báo trên server (Open-Meteo, Time-MoE, LLM) — có thể vài phút..."
        ):
            outcome, err = run_precompute_locked(160)
        if outcome == "error" and err:
            st.error(f"Precompute thất bại: {err}")
        elif outcome == "busy":
            st.info("Đang có tiến trình precompute khác (tab hoặc user khác) — tạm hiển thị cache hiện có.")

    try:
        fc_df, cache_raw_df, cache_meta = load_forecast_cache()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(str(e))
        st.stop()

    schedule_auto_refresh(int(refresh_minutes))
    st.caption(
        f"Tự động tải lại trang mỗi {int(refresh_minutes)} phút; sau mỗi lần tải lại, server sẽ precompute lại nếu cache đã cũ hơn cùng số phút."
    )

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
            f"Cache đã lưu (generated_at={gen_at} UTC). Lần tải trang tiếp theo sẽ precompute lại khi đã quá {int(refresh_minutes)} phút (hoặc dùng nút rerun / reload trình duyệt sau khi đủ thời gian)."
        )

    st.subheader("Thông tin lần tính gần nhất")
    st.caption(
        f"**Chế độ dữ liệu (cache):** `{cache_meta.get('data_mode', 'historical')}` — "
        "dự báo realtime: đặt `PM25_DATA_MODE=realtime` trong `.env`, rồi để trang precompute lại (hoặc `precompute_forecast.py --once`)."
    )
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
    st.caption(
        "Kết quả đọc từ `artifacts/latest_forecast*.csv/json` sau mỗi lần precompute (tự chạy khi mở tab hoặc khi cache cũ). "
        "Vẫn có thể chạy tay: `python precompute_forecast.py --once`."
    )
    _dm = str(cache_meta.get("data_mode", "historical")).lower()
    if _dm == "realtime":
        st.caption(
            "Chế độ **realtime** (`PM25_DATA_MODE=realtime` trong `.env`): quan sát cuối gần **hiện tại** (API cắt tại giờ đã khóa, thường now−1h); "
            "24 mốc trên biểu đồ là **24 giờ tiếp theo** sau quan sát đó. `generated_at` là lúc chạy precompute (UTC)."
        )
    else:
        st.caption(
            "Chế độ **historical** (mặc định): dataset Open-Meteo tới **31/12/2025**; trục dự báo là **24 giờ liên tục** ngay sau quan sát cuối "
            "(vd. quan sát cuối 31/12/2025 23:00 → các giờ 01/01/2026). `generated_at` là lúc chạy precompute (UTC)."
        )

    llm_text_cached = cache_meta.get("llm_text", "")
    raw = cache_raw_df["Raw"].to_numpy(dtype=float)
    idx = fc_df["Thời gian"].to_numpy()
    cal = fc_df["PM2.5 (μg/m³)"].to_numpy(dtype=float)

    render_hourly_forecast_strip(fc_df)
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

