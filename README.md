# PM2.5 Streamlit App (portable)

Folder này chạy độc lập: precompute dữ liệu ở backend, web chỉ đọc cache và hiển thị.

## Cấu trúc source

- `app_pm25_timemoe.py`: entrypoint chạy Streamlit.
- `precompute_forecast.py`: entrypoint chạy precompute theo chu kỳ/1 lần.
- `pm25_app/`: module chính, đã tách theo vai trò:
  - `ui_main.py`: giao diện và hiển thị biểu đồ/cảnh báo.
  - `precompute_main.py`: quy trình tạo cache dự báo.
  - `config.py`: đường dẫn và hằng số cấp app.
  - `env_utils.py`: nạp biến môi trường từ `.env`.
- `timemoe_pm25_pipeline.py`: pipeline mô hình Time-MoE.
- `artifacts/`: cache kết quả mới nhất để web đọc.
- `models/`: checkpoint model local.

## Chạy web

```bash
cd streamlit_app
streamlit run app_pm25_timemoe.py
```

## Chạy precompute cache

Precompute **bắt buộc** tải dữ liệu từ Open-Meteo (Archive + Air Quality). Cần cấu hình LLM trong `.env` (`GEMINI_API_KEY` mặc định với `LLM_PROVIDER=gemini`, hoặc `LLM_API_KEY` cho API tương thích OpenAI); nếu thiếu hoặc gọi LLM lỗi, precompute sẽ dừng và **không** ghi cache.

```bash
cd streamlit_app
python precompute_forecast.py --once
```

### Historical vs realtime (`PM25_DATA_MODE`)

| Giá trị `.env` | Ý nghĩa |
|----------------|---------|
| *(không đặt)* hoặc `historical` | Dataset **2021-01-01 → 2025-12-31**; quan sát cuối là giờ cuối trong khoảng đó; 24h dự báo là các giờ **ngay sau** mốc đó (thích hợp bài tập / tái lập cố định). |
| `realtime` hoặc `live` | `end_date` API = **hôm nay** (Asia/Ho_Chi_Minh); cắt quan sát tại **giờ hiện tại sàn xuống trừ 1 giờ** (tránh giờ chưa khóa); 24h dự báo là **24 giờ tiếp theo** trong thực tế — cần precompute định kỳ (web hoặc cron) để luôn gần “bây giờ”. |

Open-Meteo vẫn có **độ trễ vài giờ** tùy biến; đó là giới hạn nguồn miễn phí, không phải độ trễ 0 giây.

### Cache (`artifacts/`) hoạt động thế nào

1. **Precompute** tải Open-Meteo, chạy Time-MoE, ghi **`latest_forecast.csv`** (bảng 24 mốc dự báo) và **`latest_forecast_meta.json`** (ví dụ `dataset_last_time`, `generated_at_utc`, `llm_text`).
2. **Streamlit** mỗi lần tải trang sẽ **gọi precompute trên server** nếu chưa có cache hoặc `generated_at` đã cũ hơn số phút bạn chọn trong sidebar (mặc định 60), kèm file lock để tránh hai tab chạy trùng. Trình duyệt cũng tự reload sau cùng khoảng phút đó để kiểm tra lại. Vẫn có thể chạy tay `python precompute_forecast.py` hoặc cron khi không ai mở web.
