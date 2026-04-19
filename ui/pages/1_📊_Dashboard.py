
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
"""Dashboard — сводка по маршрутам, KPI, карта, топ маршрутов."""

import pandas as pd
import plotly.express as px
import streamlit as st

from ui.api_client import get_client, get_full_dataset

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

if not st.session_state.get("token"):
    st.warning("Войдите в систему через главную страницу")
    st.stop()

st.title("📊 Панель управления")
st.caption(f"Роль: {st.session_state.role} · пользователь: {st.session_state.username}")

client = get_client()
client.token = st.session_state.token

# --- KPI row ---
df = get_full_dataset()
routes = client.list_routes()
total_routes = len(routes)
active_routes = sum(1 for r in routes if r.get("status") == "active")
recent_passengers = int(df[df["date"] == df["date"].max()]["passengers"].sum())
avg_passengers = int(df.groupby("date")["passengers"].sum().mean())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Активных маршрутов", f"{active_routes} / {total_routes}")
c2.metric("Последний месяц, пасс.", f"{recent_passengers:,}".replace(",", " "))
c3.metric("Среднемесячный трафик",   f"{avg_passengers:,}".replace(",", " "))
c4.metric("Моделей доступно", "2", "SARIMA, XGBoost")

st.divider()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🗺️ Маршруты и карта", "📈 Исторические тренды", "🏆 Топ маршрутов"])

with tab1:
    routes_df = pd.DataFrame(routes)
    st.subheader("Реестр маршрутов")
    st.dataframe(routes_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Помесячная динамика пассажиропотока (все маршруты)")
    monthly = df.groupby("date")["passengers"].sum().reset_index()
    fig = px.line(monthly, x="date", y="passengers",
                  title="Совокупный пассажиропоток 5 сельских маршрутов",
                  labels={"date": "Месяц", "passengers": "Пассажиров"})
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Разрез по маршрутам")
    fig2 = px.line(df, x="date", y="passengers", color="route_id",
                   title="Индивидуальная динамика маршрутов")
    fig2.update_layout(height=420)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Топ-5 маршрутов по суммарному пассажиропотоку за 12 мес.")
    last_year = df[df["date"] >= df["date"].max() - pd.Timedelta(days=365)]
    top = (last_year.groupby("route_id")["passengers"].sum()
           .sort_values(ascending=False).head(10).reset_index())
    top.columns = ["Маршрут", "Пассажиров за год"]
    st.dataframe(top, use_container_width=True, hide_index=True)

    fig3 = px.bar(top, x="Маршрут", y="Пассажиров за год",
                  color_discrete_sequence=["#2980b9"], text_auto=True)
    fig3.update_traces(marker_line_color="#1a5276", marker_line_width=1.2,
                        textposition="outside")
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
