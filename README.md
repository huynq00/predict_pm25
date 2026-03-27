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
- `data/`: dữ liệu fallback khi API lỗi.
- `models/`: checkpoint model local.

## Chạy web

```bash
cd streamlit_app
streamlit run app_pm25_timemoe.py
```

## Chạy precompute cache

```bash
cd streamlit_app
python precompute_forecast.py --once
```

## Fallback CSV khi API lỗi

- Mặc định dùng `data/fallback_pm25.csv`.
- Có thể đổi qua biến môi trường `DATASET_FALLBACK_CSV`.
- CSV hỗ trợ 2 dạng:
  - chuẩn Open-Meteo: có `date`, `pm2_5`
  - chuẩn cũ: `Date_Time,Temperature_C,Humidity_pct,Wind_Speed_ms,Precipitation_mm,Pressure_hPa,Cloud_Cover_pct,Radiation_W,PM2_5`
