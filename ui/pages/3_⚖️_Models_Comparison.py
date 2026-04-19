
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
"""Models Comparison — сравнительный анализ SARIMA vs XGBoost по метрикам."""

import pandas as pd
import plotly.express as px
import streamlit as st

from ui.api_client import get_client

st.set_page_config(page_title="Models Comparison", page_icon="⚖️", layout="wide")

if not st.session_state.get("token"):
    st.warning("Войдите в систему через главную страницу")
    st.stop()

st.title("⚖️ Сравнительный анализ моделей")
st.caption("Сравнение SARIMA и XGBoost на hold-out выборке по метрикам MAE, RMSE, MAPE, R².")

client = get_client()
client.token = st.session_state.token

routes = client.list_routes()
routes_df = pd.DataFrame(routes)

col1, col2 = st.columns([2, 1])
with col1:
    selected_id = st.selectbox(
        "Маршрут",
        options=routes_df["route_id"].tolist(),
        format_func=lambda rid: next((r["name"] for r in routes if r["route_id"] == rid), str(rid)),
        key="cmp_route_id",
    )
with col2:
    test_size = st.slider("Тест, мес.", 6, 24, 12, key="cmp_test_size")

if st.button("📊 Запустить сравнение", type="primary", key="btn_cmp"):
    with st.spinner("Обучение двух моделей и расчёт метрик…"):
        metrics = client.metrics(selected_id, model_type=None, test_size=test_size)

    if not metrics:
        st.error("Не удалось вычислить метрики (нет данных для выбранного маршрута).")
    else:
        df_m = pd.DataFrame(metrics)
        df_m["model_type"] = df_m["model_type"].str.upper()

        # --- Ranked table ---
        df_m = df_m.sort_values("mape").reset_index(drop=True)
        df_m.insert(0, "rank", df_m.index + 1)
        df_display = df_m.rename(columns={
            "rank": "№", "model_type": "Модель",
            "mape": "MAPE, %", "mae": "MAE, чел.", "rmse": "RMSE, чел.",
            "r_squared": "R²", "test_period_months": "Тест, мес.",
        })[["№", "Модель", "MAPE, %", "MAE, чел.", "RMSE, чел.", "R²", "Тест, мес."]]

        st.subheader("Таблица метрик (сортировка по MAPE)")
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # --- Bar charts ---
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.bar(df_m, x="model_type", y="mape", color="model_type",
                         title="MAPE, %", text_auto=".2f",
                         color_discrete_map={"SARIMA": "#2980b9", "XGBOOST": "#e67e22"})
            fig.update_layout(showlegend=False, height=340)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig2 = px.bar(df_m, x="model_type", y="rmse", color="model_type",
                          title="RMSE, чел.", text_auto=".0f",
                          color_discrete_map={"SARIMA": "#2980b9", "XGBOOST": "#e67e22"})
            fig2.update_layout(showlegend=False, height=340)
            st.plotly_chart(fig2, use_container_width=True)

        best = df_m.iloc[0]
        st.success(
            f"🏆 Лучшая модель по MAPE на маршруте {selected_id}: "
            f"**{best['model_type']}** (MAPE = {best['mape']:.2f}%)"
        )
else:
    st.info("Выберите параметры и нажмите «Запустить сравнение».")
