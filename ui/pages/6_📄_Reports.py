
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
"""Reports — генерация и скачивание PDF/DOCX отчётов."""

from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

from ui.api_client import get_client

st.set_page_config(page_title="Отчёты", page_icon="📄", layout="wide")

if not st.session_state.get("token"):
    st.warning("Войдите в систему через главную страницу")
    st.stop()

st.title("📄 Отчёты")
st.caption("Генерация и скачивание аналитических отчётов в форматах PDF и DOCX.")

client = get_client()
client.token = st.session_state.token

routes = client.list_routes()

# --- Generation form ---
st.subheader("1. Сгенерировать новый отчёт")
with st.form("report_form"):
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        route_id = st.selectbox(
            "Маршрут",
            options=[r["route_id"] for r in routes],
            format_func=lambda rid: next((r["name"] for r in routes if r["route_id"] == rid), str(rid)),
        )
    with c2:
        model_type = st.selectbox("Модель", ["sarima", "xgboost", "prophet", "lstm"])
    with c3:
        horizon = st.slider("Горизонт", 3, 24, 12)
    with c4:
        fmt = st.selectbox("Формат", ["pdf", "docx"])

    submitted = st.form_submit_button("📝 Сгенерировать", type="primary", use_container_width=True)

if submitted:
    with st.spinner("Генерация отчёта (может занять до 30 с)…"):
        result = client.generate_report(route_id, model_type, horizon, fmt)
    if result.get("status") in ("generated", "completed"):
        st.success(f"✅ Отчёт создан: {result['report_id']}")
        st.json(result)
    else:
        st.error(f"Ошибка генерации: {result}")

st.divider()

# --- Report list ---
st.subheader("2. Существующие отчёты")
reports = client.list_reports()
if reports:
    df = pd.DataFrame(reports)
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
    df["size_kb"] = (df["size_bytes"] / 1024).round(1)
    df_display = df[["title", "format", "size_kb", "created_at"]].rename(columns={
        "title": "Название", "format": "Формат", "size_kb": "Размер, КБ", "created_at": "Создан",
    })
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Download buttons for each report
    reports_dir = Path(__file__).parent.parent.parent / "reports" / "generated"
    for r in reports:
        filename = f"{r['title']}.{r['format']}"
        filepath = reports_dir / filename
        if filepath.exists():
            with open(filepath, "rb") as f:
                data = f.read()
            mime = "application/pdf" if r["format"] == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            st.download_button(
                f"⬇️ Скачать {filename} ({r['size_bytes'] // 1024} КБ)",
                data=data, file_name=filename, mime=mime,
                key=f"dl_{r['report_id']}"
            )
else:
    st.info("Отчётов пока нет. Сгенерируйте первый через форму выше.")
