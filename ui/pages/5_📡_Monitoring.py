
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
"""Monitoring — мониторинг последних рейсов и состояния системы."""

import pandas as pd
import plotly.express as px
import streamlit as st

from ui.api_client import get_client, get_full_dataset

st.set_page_config(page_title="Мониторинг", page_icon="📡", layout="wide")

if not st.session_state.get("token"):
    st.warning("Войдите в систему через главную страницу")
    st.stop()

st.title("📡 Мониторинг текущих рейсов")
st.caption("Агрегированное представление пассажиропотока за последние 30 дней.")

client = get_client()
client.token = st.session_state.token

df = get_full_dataset()

# --- System status ---
c1, c2, c3 = st.columns(3)
c1.metric("Состояние API", "🟢 online" if client._api_up() else "🟡 local fallback")
c2.metric("Маршрутов в мониторинге", df["route_id"].nunique())
c3.metric("Последнее обновление", df["date"].max().strftime("%Y-%m-%d"))

st.divider()

# --- Last month data ---
last_month = df[df["date"] == df["date"].max()]
st.subheader("Сводка за последний отчётный месяц")
st.dataframe(last_month[["route_id", "passengers"]].rename(
    columns={"route_id": "Маршрут", "passengers": "Пассажиропоток"}
), use_container_width=True, hide_index=True)

# --- Last 12 months trend per route ---
st.subheader("Динамика за последние 12 месяцев (все маршруты)")
cut = df["date"].max() - pd.Timedelta(days=370)
last_year = df[df["date"] >= cut]
fig = px.line(last_year, x="date", y="passengers", color="route_id",
              title="Пассажиропоток за год")
fig.update_layout(height=440)
st.plotly_chart(fig, use_container_width=True)

# --- Anomalies ---
st.subheader("Аномалии (отклонение >2σ от скользящего среднего)")
anomalies = []
for rid in df["route_id"].unique():
    s = df[df["route_id"] == rid].set_index("date")["passengers"].sort_index()
    rolling_mean = s.rolling(window=6, min_periods=3).mean()
    rolling_std  = s.rolling(window=6, min_periods=3).std()
    z = (s - rolling_mean) / rolling_std
    found = z[z.abs() > 2]
    for date, z_val in found.items():
        anomalies.append({
            "Маршрут": rid, "Дата": date.strftime("%Y-%m"),
            "Факт": int(s.loc[date]),
            "Среднее": int(rolling_mean.loc[date]),
            "Z-score": round(float(z_val), 2),
        })

if anomalies:
    adf = pd.DataFrame(anomalies).sort_values("Z-score", key=lambda s: s.abs(), ascending=False)
    st.dataframe(adf.head(20), use_container_width=True, hide_index=True)
else:
    st.info("Аномалий не обнаружено.")
