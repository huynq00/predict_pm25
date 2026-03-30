from __future__ import annotations

import html

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from timemoe_pm25_pipeline import pm25_aqi_band_vn


def contrasting_text_color(hex_bg: str) -> str:
    h = hex_bg.lstrip("#")
    if len(h) != 6:
        return "#111111"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#111111" if lum > 145 else "#ffffff"


HOURLY_STRIP_CSS_RULES = """
    .pm25-hourly-wrap {
        background: #ffffff;
        border-radius: 16px;
        box-shadow: 0 1px 8px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 18px 16px 20px 16px;
        margin: 12px 0 20px 0;
    }
    .pm25-hourly-wrap h3 {
        margin: 0 0 4px 0;
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
    }
    .pm25-hourly-sub {
        margin: 0 0 16px 0;
        font-size: 0.88rem;
        color: #64748b;
        line-height: 1.35;
    }
    .pm25-hourly-scroll {
        display: flex;
        overflow-x: auto;
        gap: 0;
        padding-bottom: 6px;
        -webkit-overflow-scrolling: touch;
    }
    .pm25-hourly-col {
        flex: 0 0 auto;
        min-width: 72px;
        text-align: center;
        padding: 0 10px;
        border-right: 1px dashed #e2e8f0;
        box-sizing: border-box;
    }
    .pm25-hourly-col:last-child { border-right: none; }
    .pm25-hourly-time {
        font-size: 0.78rem;
        color: #64748b;
        margin-bottom: 8px;
        font-weight: 500;
    }
    .pm25-hourly-val {
        display: inline-block;
        min-width: 52px;
        padding: 8px 10px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.05rem;
        line-height: 1.15;
    }
    .pm25-hourly-val-sub {
        font-size: 0.62rem;
        font-weight: 600;
        opacity: 0.9;
        margin-top: 4px;
        letter-spacing: 0.02em;
    }
"""


def build_hourly_strip_html(fc_df: pd.DataFrame) -> str:
    now_hcm = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh")
    cols_html: list[str] = []

    for i in range(len(fc_df)):
        row = fc_df.iloc[i]
        ts = row["Thời gian"]
        if getattr(ts, "tzinfo", None) is None:
            ts_hcm = ts.tz_localize("Asia/Bangkok")
        else:
            ts_hcm = ts.tz_convert("Asia/Bangkok")
        if i == 0 and abs((ts_hcm - now_hcm).total_seconds()) <= 5400:
            time_lbl = "Bây giờ"
        else:
            time_lbl = ts_hcm.strftime("%H:%M")

        pm = float(row["PM2.5 (μg/m³)"])
        _, band_hex = pm25_aqi_band_vn(pm)
        txt_col = contrasting_text_color(band_hex)
        pm_txt = f"{pm:.1f}"

        cols_html.append(
            "<div class=\"pm25-hourly-col\">"
            f"<div class=\"pm25-hourly-time\">{html.escape(time_lbl)}</div>"
            f"<div class=\"pm25-hourly-val\" style=\"background:{band_hex};color:{txt_col};\">"
            f"{html.escape(pm_txt)}"
            "<div class=\"pm25-hourly-val-sub\">μg/m³</div>"
            "</div>"
            "</div>"
        )

    body = "".join(cols_html)
    return (
        "<div class=\"pm25-hourly-wrap\">"
        "<h3>Dự báo theo giờ</h3>"
        "<p class=\"pm25-hourly-sub\">PM2.5 dự báo (Time-MoE, đã hiệu chỉnh) — TP.HCM.</p>"
        f"<div class=\"pm25-hourly-scroll\">{body}</div>"
        "</div>"
    )


def render_hourly_forecast_strip(fc_df: pd.DataFrame) -> None:
    """Thanh ngang: giờ + PM2.5 (μg/m³). Dùng iframe để tránh Markdown coi HTML là code."""
    try:
        inner = build_hourly_strip_html(fc_df)
        doc = (
            "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/>"
            f"<style>{HOURLY_STRIP_CSS_RULES}</style></head><body>"
            f"{inner}</body></html>"
        )
        components.html(doc, height=260, scrolling=True)
    except Exception as e:
        st.warning(f"Không vẽ được thanh dự báo theo giờ: {e}")
