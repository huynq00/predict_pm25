from __future__ import annotations

import html
import re

import streamlit as st
import streamlit.components.v1 as components

HEALTH_CARD_CSS = """
#pm25-health-root { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.pm25-health-card {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 2px 14px rgba(15, 23, 42, 0.08);
    border: 1px solid rgba(226, 232, 240, 0.95);
    padding: 20px 22px 22px 22px;
    margin: 10px 0 16px 0;
}
.pm25-health-title {
    margin: 0 0 18px 0;
    font-size: 1.12rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
}
.pm25-health-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 12px 0;
    border-bottom: 1px solid #f1f5f9;
}
.pm25-health-row:last-child { border-bottom: none; padding-bottom: 0; }
.pm25-health-ico {
    flex: 0 0 auto;
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: #fff9e6;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.35rem;
    line-height: 1;
}
.pm25-health-body { flex: 1; min-width: 0; }
.pm25-health-text {
    margin: 0;
    font-size: 0.92rem;
    line-height: 1.45;
    color: #334155;
    font-weight: 500;
}
.pm25-health-link {
    display: inline-block;
    margin-top: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #2563eb;
    text-decoration: none;
}
.pm25-health-link:hover { text-decoration: underline; color: #1d4ed8; }
.pm25-health-foot {
    margin-top: 14px;
    font-size: 0.78rem;
    color: #94a3b8;
    line-height: 1.4;
}
"""


def parse_llm_bullets(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s[:2] in ("- ", "* "):
            out.append(s[2:].strip())
        elif s.startswith("• "):
            out.append(s[2:].strip())
        elif re.match(r"^\d+[\).\]]\s+", s):
            out.append(re.sub(r"^\d+[\).\]]\s+", "", s).strip())
    return [x for x in out if x]


def _icon_for_line(line: str) -> str:
    t = line.lower()
    if any(k in t for k in ("mặt nạ", "khẩu trang", "mask")):
        return "😷"
    if any(k in t for k in ("cửa sổ", "cửa kính", "đóng cửa")):
        return "🪟"
    if any(k in t for k in ("tập", "thể dục", "vận động", "ngoài trời", "chạy bộ")):
        return "🚴"
    if any(k in t for k in ("lọc không khí", "máy lọc", "purifier", "hepa")):
        return "🌬️"
    if any(k in t for k in ("trẻ em", "người già", "nhạy cảm", "hen", "tim mạch")):
        return "🫁"
    return "✓"


def _default_rows(peak_pm25: float) -> list[tuple[str, str, str | None, str | None]]:
    hi = peak_pm25 > 55
    mid = peak_pm25 > 35
    sens = "Các nhóm nhạy cảm " if mid else "Mọi người "
    sens2 = "Nhóm nhạy cảm " if mid else ""
    rows: list[tuple[str, str, str | None, str | None]] = [
        (
            "🚴",
            (
                f"{sens}nên giảm tập thể dục gắng sức ngoài trời khi PM2.5 cao."
                if hi or mid
                else "Có thể duy trì hoạt động ngoài trời vừa phải; theo dõi khi chỉ số tăng."
            ),
            None,
            None,
        ),
        (
            "🪟",
            "Đóng cửa sổ khi không khí ngoài trời kém để hạn chế bụi mịn vào nhà.",
            "WHO — không khí & sức khỏe",
            "https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health",
        ),
        (
            "😷",
            (
                f"{sens2}nên đeo khẩu trang chuẩn (N95/KF94) khi phải ra ngoài lúc ô nhiễm."
                if mid
                else "Khi PM2.5 tăng cao, cân nhắc khẩu trang lọc bụi mịn khi ra đường."
            ),
            None,
            None,
        ),
        (
            "🌬️",
            (
                f"{sens2}nên bật máy lọc không khí trong nhà (nếu có) ở chế độ phù hợp."
                if mid
                else "Máy lọc không khí giúp giảm PM2.5 trong nhà khi chỉ số ngoài trời cao."
            ),
            None,
            None,
        ),
    ]
    return rows


def _rows_from_llm(bullets: list[str]) -> list[tuple[str, str, str | None, str | None]]:
    return [(_icon_for_line(b), b, None, None) for b in bullets[:8]]


def build_health_card_html(
    rows: list[tuple[str, str, str | None, str | None]],
    footnote: str | None = None,
) -> str:
    parts: list[str] = [
        '<div id="pm25-health-root">',
        '<div class="pm25-health-card">',
        '<h2 class="pm25-health-title">Khuyến nghị về sức khỏe</h2>',
    ]
    for emoji, text, link_lbl, link_href in rows:
        safe_t = html.escape(text)
        row = (
            '<div class="pm25-health-row">'
            f'<div class="pm25-health-ico">{emoji}</div>'
            '<div class="pm25-health-body">'
            f'<p class="pm25-health-text">{safe_t}</p>'
        )
        if link_lbl and link_href:
            row += (
                f'<a class="pm25-health-link" href="{html.escape(link_href, quote=True)}" '
                f'target="_blank" rel="noopener noreferrer">{html.escape(link_lbl)}</a>'
            )
        row += "</div></div>"
        parts.append(row)
    if footnote:
        parts.append(f'<p class="pm25-health-foot">{html.escape(footnote)}</p>')
    parts.append("</div></div>")
    return "".join(parts)


def render_health_recommendations_card(
    peak_pm25: float,
    llm_text: str | None,
    *,
    llm_error: bool = False,
) -> None:
    """Card khuyến nghị: ưu tiên gạch đầu dòng từ LLM, không thì mẫu cố định theo PM2.5."""
    foot: str | None = None
    if llm_error:
        rows = _default_rows(peak_pm25)
        foot = "Không tải được khuyến nghị LLM; hiển thị gợi ý chung."
    elif llm_text and llm_text.strip() and not llm_text.startswith("(LLM precompute lỗi:"):
        bullets = parse_llm_bullets(llm_text)
        if bullets:
            rows = _rows_from_llm(bullets)
            foot = "Nội dung sinh từ LLM khi precompute; tham khảo thêm bác sĩ / quy định địa phương."
        else:
            rows = _default_rows(peak_pm25)
            foot = "LLM không trả về gạch đầu dòng rõ ràng; hiển thị gợi ý chuẩn."
    else:
        rows = _default_rows(peak_pm25)
        if not llm_text or not llm_text.strip():
            foot = "Chưa có văn bản LLM trong cache — đây là khuyến nghị tham khảo chung."

    inner = build_health_card_html(rows, footnote=foot)
    doc = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/>"
        f"<style>{HEALTH_CARD_CSS}</style></head><body>{inner}</body></html>"
    )
    h = min(620, 140 + len(rows) * 76 + (40 if foot else 0))
    components.html(doc, height=h, scrolling=False)
