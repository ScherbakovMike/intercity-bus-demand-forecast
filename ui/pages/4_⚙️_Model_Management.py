"""Model Management — переобучение моделей с настройкой гиперпараметров."""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from ui.api_client import get_client

st.set_page_config(page_title="Model Management", page_icon="⚙️", layout="wide")

if not st.session_state.get("token"):
    st.warning("Войдите в систему через главную страницу")
    st.stop()

role = st.session_state.get("role")
if role not in ("admin", "analyst"):
    st.error(f"Доступ запрещён. Для роли '{role}' запуск переобучения недоступен. "
             f"Требуется роль 'admin' или 'analyst'.")
    st.stop()

st.title("⚙️ Управление моделями")
st.caption("Переобучение моделей прогнозирования с настраиваемыми гиперпараметрами.")

client = get_client()
client.token = st.session_state.token

routes = client.list_routes()
col1, col2 = st.columns(2)
with col1:
    selected_id = st.selectbox(
        "Маршрут",
        options=[r["route_id"] for r in routes],
        format_func=lambda rid: next((r["name"] for r in routes if r["route_id"] == rid), str(rid)),
        key="mgr_route",
    )
with col2:
    model_type = st.selectbox(
        "Модель", ["sarima", "xgboost", "prophet", "lstm"], key="mgr_model",
        help="Требует prophet/tensorflow-cpu для Prophet/LSTM",
    )

st.subheader("Гиперпараметры")
params = {}
if model_type == "xgboost":
    c1, c2, c3 = st.columns(3)
    with c1:
        params["n_estimators"] = st.number_input("n_estimators", 50, 1000, 200, step=50)
    with c2:
        params["max_depth"] = st.number_input("max_depth", 2, 15, 6)
    with c3:
        params["n_splits"] = st.number_input("n_splits (CV)", 2, 10, 3)
elif model_type == "sarima":
    st.info("SARIMA использует автоматический подбор порядка (auto_arima).")

col_btn1, col_btn2 = st.columns([1, 3])
with col_btn1:
    train_clicked = st.button("▶️ Запустить обучение", type="primary", key="btn_train")

if train_clicked:
    with st.spinner(f"Обучение {model_type.upper()} на маршруте {selected_id}…"):
        result = client.train(selected_id, model_type, params=params)

    if result.get("status") == "completed":
        st.success(f"✅ {result['message']}")
        st.json(result)
    else:
        st.error(f"❌ Обучение не удалось: {result.get('message')}")
        st.json(result)

st.divider()
st.subheader("Текущие гиперпараметры в config.py")
st.code("""
# src/config.py
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)

XGBOOST_N_ESTIMATORS = 500
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.05
XGBOOST_N_SPLITS = 5

LSTM_UNITS = [64, 32]
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 100

PROPHET_CHANGEPOINT_PRIOR = 0.05
""", language="python")
