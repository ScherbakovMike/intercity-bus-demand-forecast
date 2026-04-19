
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
"""Route Analysis — детальный анализ одного маршрута: история + прогноз + CI."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.api_client import get_client, get_full_dataset

st.set_page_config(page_title="Route Analysis", page_icon="🗺️", layout="wide")

if not st.session_state.get("token"):
    st.warning("Войдите в систему через главную страницу")
    st.stop()

st.title("🗺️ Анализ маршрута")

client = get_client()
client.token = st.session_state.token

routes = client.list_routes()
routes_df = pd.DataFrame(routes)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    selected_id = st.selectbox(
        "Выберите маршрут",
        options=routes_df["route_id"].tolist(),
        format_func=lambda rid: next((r["name"] for r in routes if r["route_id"] == rid), str(rid)),
        key="route_id_select",
    )
with col2:
    model_type = st.selectbox(
        "Модель",
        ["sarima", "xgboost", "prophet", "lstm"],
        key="model_select",
        help="Prophet и LSTM требуют установленных библиотек prophet/tensorflow-cpu",
    )
with col3:
    horizon = st.slider("Горизонт (мес.)", 3, 24, 12, key="horizon_slider")

# Data
df = get_full_dataset()
route_code = f"RU-RURAL-{selected_id:03d}"
series = df[df["route_id"] == route_code].set_index("date")["passengers"].sort_index() \
    if route_code in df["route_id"].unique() else None

if series is None or series.empty:
    st.warning(f"Данных по маршруту {route_code} нет в синтетическом датасете. "
               f"Попробуйте маршруты 1–5 (RU-RURAL-001 … RU-RURAL-005).")
    st.stop()

st.subheader("1. Исторические данные маршрута")
hist_fig = go.Figure()
hist_fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines+markers",
                               name="Факт", line=dict(color="#2980b9", width=2)))
hist_fig.update_layout(title=f"Пассажиропоток: {routes_df.query('route_id == @selected_id')['name'].iloc[0]}",
                        xaxis_title="Месяц", yaxis_title="Пассажиров", height=420)
st.plotly_chart(hist_fig, use_container_width=True)

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Средний поток, пасс./мес.", f"{int(series.mean()):,}".replace(",", " "))
col_m2.metric("Минимум (COVID)", f"{int(series.min()):,}".replace(",", " "),
              delta=f"{(series.min() - series.mean()) / series.mean() * 100:.0f}%")
col_m3.metric("Максимум", f"{int(series.max()):,}".replace(",", " "),
              delta=f"+{(series.max() - series.mean()) / series.mean() * 100:.0f}%")

st.divider()

st.subheader("2. Прогноз и доверительный интервал")
if st.button("🔮 Построить прогноз", type="primary", key="btn_forecast"):
    with st.spinner(f"Обучение {model_type.upper()} и прогнозирование на {horizon} мес…"):
        result = client.forecast(selected_id, model_type, horizon)

    if "error" in result:
        st.error(result["error"])
    else:
        points = pd.DataFrame(result["points"])
        last_date = series.index[-1]
        future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")
        points["date"] = future_dates

        fc_fig = go.Figure()
        # History
        fc_fig.add_trace(go.Scatter(x=series.index, y=series.values,
                                      mode="lines", name="Факт",
                                      line=dict(color="#2980b9", width=2)))
        # Forecast
        fc_fig.add_trace(go.Scatter(x=points["date"], y=points["point"],
                                      mode="lines+markers", name="Прогноз",
                                      line=dict(color="#e74c3c", width=2, dash="dash")))
        # CI band
        fc_fig.add_trace(go.Scatter(
            x=list(points["date"]) + list(reversed(points["date"])),
            y=list(points["upper"]) + list(reversed(points["lower"])),
            fill="toself", fillcolor="rgba(231,76,60,0.15)",
            line=dict(color="rgba(255,255,255,0)"), name="95% ДИ"))
        fc_fig.update_layout(title=f"Прогноз ({model_type.upper()}, H={horizon})",
                              xaxis_title="Месяц", yaxis_title="Пассажиров",
                              height=460)
        st.plotly_chart(fc_fig, use_container_width=True)

        st.subheader("3. Табличные значения прогноза")
        points_display = points[["date", "point", "lower", "upper"]].copy()
        points_display.columns = ["Месяц", "Прогноз", "Нижняя ДИ", "Верхняя ДИ"]
        points_display["Месяц"] = points_display["Месяц"].dt.strftime("%Y-%m")
        st.dataframe(points_display, use_container_width=True, hide_index=True)
else:
    st.info("Нажмите «Построить прогноз», чтобы получить результаты от модели.")
