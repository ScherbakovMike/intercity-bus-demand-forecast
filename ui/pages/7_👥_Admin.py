
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
"""Admin — управление пользователями и ролями (только для роли admin)."""

import pandas as pd
import streamlit as st

from ui.api_client import get_client

st.set_page_config(page_title="Admin", page_icon="👥", layout="wide")

if not st.session_state.get("token"):
    st.warning("Войдите в систему через главную страницу")
    st.stop()

if st.session_state.get("role") != "admin":
    st.error("Доступ запрещён. Эта страница доступна только администраторам.")
    st.stop()

st.title("👥 Администрирование")
st.caption("Список пользователей, создание новых учётных записей, мониторинг ролей.")

client = get_client()
client.token = st.session_state.token

# --- User list ---
st.subheader("1. Текущие пользователи")
users = [
    {"Логин": "admin",      "Роль": "admin",      "ФИО": "Администратор системы"},
    {"Логин": "planner",    "Роль": "planner",    "ФИО": "Иван Планировщик"},
    {"Логин": "analyst",    "Роль": "analyst",    "ФИО": "Анна Аналитик"},
    {"Логин": "dispatcher", "Роль": "dispatcher", "ФИО": "Олег Диспетчер"},
]
st.dataframe(pd.DataFrame(users), use_container_width=True, hide_index=True)

# --- Role matrix ---
st.subheader("2. Матрица прав доступа к экранам")
role_matrix = pd.DataFrame([
    {"Экран": "Dashboard",            "admin": "✅", "planner": "✅", "analyst": "✅", "dispatcher": "✅"},
    {"Экран": "Анализ маршрута",      "admin": "✅", "planner": "✅", "analyst": "✅", "dispatcher": "✅"},
    {"Экран": "Сравнение моделей",    "admin": "✅", "planner": "✅", "analyst": "✅", "dispatcher": "❌"},
    {"Экран": "Управление моделями",  "admin": "✅", "planner": "❌", "analyst": "✅", "dispatcher": "❌"},
    {"Экран": "Мониторинг",           "admin": "✅", "planner": "✅", "analyst": "✅", "dispatcher": "✅"},
    {"Экран": "Отчёты",               "admin": "✅", "planner": "✅", "analyst": "✅", "dispatcher": "❌"},
    {"Экран": "Администрирование",    "admin": "✅", "planner": "❌", "analyst": "❌", "dispatcher": "❌"},
])
st.dataframe(role_matrix, use_container_width=True, hide_index=True)

# --- New user form ---
st.subheader("3. Создать нового пользователя")
with st.form("new_user_form"):
    c1, c2 = st.columns(2)
    with c1:
        new_username = st.text_input("Логин")
        new_password = st.text_input("Пароль", type="password")
    with c2:
        new_role = st.selectbox("Роль", ["analyst", "planner", "dispatcher", "admin"])
        new_fullname = st.text_input("ФИО")
        new_email = st.text_input("Email")

    submitted = st.form_submit_button("➕ Создать", type="primary")
    if submitted:
        if not new_username or not new_password:
            st.error("Заполните логин и пароль")
        else:
            st.success(f"Пользователь {new_username} ({new_role}) создан")
            st.info("Примечание: в демо-режиме изменения не сохраняются между сеансами.")

# --- Settings ---
st.subheader("4. Системные настройки")
st.json({
    "database": {
        "host": "postgres",
        "port": 5432,
        "name": "passenger_forecast",
        "pool_size": 5,
        "max_overflow": 10,
    },
    "api": {"host": "0.0.0.0", "port": 8000, "cors": "*"},
    "jwt": {"algorithm": "HS256", "expire_minutes": 60},
    "models": {
        "sarima": {"order": [1, 1, 1], "seasonal_order": [1, 1, 1, 12]},
        "xgboost": {"n_estimators": 500, "max_depth": 6, "n_splits": 5},
        "lstm": {"units": [64, 32], "dropout": 0.2, "epochs": 100},
        "prophet": {"changepoint_prior": 0.05},
    },
})
