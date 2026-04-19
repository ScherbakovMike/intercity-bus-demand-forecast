"""Streamlit entry point — login + navigation hub.

Запуск:  streamlit run ui/app.py
"""

from __future__ import annotations

import os
import sys

# Allow imports from project root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from ui.api_client import get_client

st.set_page_config(
    page_title="ИС Прогнозирование пассажиропотока",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session state init ---
for key, default in [("token", None), ("role", None), ("username", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


def _login_form():
    st.title("🚌 Вход в систему прогнозирования пассажиропотока")
    st.caption("ИС прогнозирования пассажиропотока на междугородних рейсах. ВКР МТИ 2026.")

    with st.form("login_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Логин", value="planner", key="login_username")
        with col2:
            password = st.text_input("Пароль", type="password", value="planner123", key="login_password")
        submitted = st.form_submit_button("Войти", type="primary", use_container_width=True)

        if submitted:
            client = get_client()
            result = client.login(username, password)
            if result:
                st.session_state.token = result["access_token"]
                st.session_state.role = result["role"]
                st.session_state.username = username
                st.success(f"Успешный вход как {username} (роль: {result['role']})")
                st.rerun()
            else:
                st.error("Неверный логин или пароль")

    with st.expander("Демо-учётные записи"):
        st.markdown("""
| Логин | Пароль | Роль |
|---|---|---|
| `admin` | `admin123` | Администратор |
| `planner` | `planner123` | Планировщик |
| `analyst` | `analyst123` | Аналитик |
| `dispatcher` | `dispatch123` | Диспетчер |
""")


def _main_dashboard():
    st.sidebar.title("🚌 Навигация")
    st.sidebar.caption(f"👤 **{st.session_state.username}** · {st.session_state.role}")
    st.sidebar.divider()
    st.sidebar.markdown(
        "Используйте меню слева (раздел **Pages**) для перехода между экранами:"
        "\n\n- 📊 Dashboard"
        "\n- 🗺️ Анализ маршрута"
        "\n- ⚖️ Сравнение моделей"
        "\n- ⚙️ Управление моделями"
        "\n- 📡 Мониторинг"
        "\n- 📄 Отчёты"
        "\n- 👥 Администрирование"
    )
    st.sidebar.divider()
    if st.sidebar.button("Выйти", type="secondary", use_container_width=True):
        for k in ["token", "role", "username"]:
            st.session_state[k] = None
        st.rerun()

    st.title("🚌 Система прогнозирования пассажиропотока")
    st.caption("Панель управления (выберите экран в левом меню Pages)")

    # Landing KPI summary
    client = get_client()
    client.token = st.session_state.token
    routes = client.list_routes()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Активных маршрутов", sum(1 for r in routes if r.get("status") == "active"))
    c2.metric("Всего маршрутов", len(routes))
    c3.metric("Моделей в системе", "4", "SARIMA, Prophet, LSTM, XGBoost")
    c4.metric("Статус API", "🟢 online" if client._api_up() else "🟡 local fallback")

    st.divider()
    st.subheader("Маршруты")
    import pandas as pd
    st.dataframe(pd.DataFrame(routes), use_container_width=True, hide_index=True)


# --- Routing ---
if not st.session_state.token:
    _login_form()
else:
    _main_dashboard()
