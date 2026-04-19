"""UI tests via streamlit.testing.v1 — verify pages load and buttons work.

Note: these tests use Streamlit's built-in AppTest framework (not a browser),
which exercises the full Streamlit runtime without requiring a live server.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


UI_DIR = Path(__file__).parent.parent / "ui"
PAGES_DIR = UI_DIR / "pages"


# ---------- Main app (login) ----------

def test_app_login_page_renders():
    """Unauthenticated session shows login form with demo creds."""
    at = AppTest.from_file(str(UI_DIR / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Exception on load: {at.exception}"
    # Title should appear
    all_texts = list(at.title) + list(at.markdown)
    assert any("Вход" in t.value or "прогнозирования" in t.value.lower() for t in all_texts)


def test_app_login_success():
    """Successful login populates session state and rerenders."""
    at = AppTest.from_file(str(UI_DIR / "app.py"), default_timeout=30)
    at.run()
    # Fill login form
    at.text_input(key="login_username").set_value("planner")
    at.text_input(key="login_password").set_value("planner123")
    at.button[0].click().run()

    assert not at.exception, f"Login flow raised: {at.exception}"
    assert at.session_state["token"] if "token" in at.session_state else None is not None
    assert at.session_state["role"] if "role" in at.session_state else None == "planner"


def test_app_login_wrong_password():
    at = AppTest.from_file(str(UI_DIR / "app.py"), default_timeout=30)
    at.run()
    at.text_input(key="login_username").set_value("planner")
    at.text_input(key="login_password").set_value("wrongpass")
    at.button[0].click().run()

    token_value = at.session_state["token"] if "token" in at.session_state else None
    assert token_value is None, f"Expected no token after failed login, got {token_value!r}"
    # Should have an error message
    assert any("Неверный" in e.value for e in at.error)


# ---------- Authenticated helper ----------

def _authed_apptest(script_path: Path, role: str = "planner"):
    at = AppTest.from_file(str(script_path), default_timeout=60)
    at.session_state["token"] = "test-token-" + role
    at.session_state["role"] = role
    at.session_state["username"] = role
    at.run()
    return at


# ---------- Per-page load tests ----------

PAGES_TO_TEST = [
    "1_📊_Dashboard.py",
    "2_🗺️_Route_Analysis.py",
    "3_⚖️_Models_Comparison.py",
    "5_📡_Monitoring.py",
    "6_📄_Reports.py",
]


@pytest.mark.parametrize("page_file", PAGES_TO_TEST)
def test_page_loads_authenticated(page_file):
    """Each page should render without exceptions when authenticated."""
    at = _authed_apptest(PAGES_DIR / page_file, role="planner")
    assert not at.exception, f"Page {page_file} raised: {at.exception}"
    # Page has at least a title or header
    assert len(list(at.title)) + len(list(at.header)) >= 1


def test_model_management_requires_analyst_role():
    """4_Model_Management requires admin or analyst."""
    # As planner: access denied
    at_p = _authed_apptest(PAGES_DIR / "4_⚙️_Model_Management.py", role="planner")
    assert any("Доступ запрещён" in e.value for e in at_p.error)

    # As analyst: access granted
    at_a = _authed_apptest(PAGES_DIR / "4_⚙️_Model_Management.py", role="analyst")
    assert not any("Доступ запрещён" in e.value for e in at_a.error)


def test_admin_page_requires_admin_role():
    at_p = _authed_apptest(PAGES_DIR / "7_👥_Admin.py", role="planner")
    assert any("запрещён" in e.value for e in at_p.error)

    at_a = _authed_apptest(PAGES_DIR / "7_👥_Admin.py", role="admin")
    assert not any("запрещён" in e.value for e in at_a.error)


def test_pages_blocked_when_unauthenticated():
    """Pages should show a warning if user is not logged in."""
    for page_file in PAGES_TO_TEST:
        at = AppTest.from_file(str(PAGES_DIR / page_file), default_timeout=15)
        at.run()
        assert any("Войдите" in w.value for w in at.warning), \
            f"{page_file}: expected a login warning for unauthenticated user"


# ---------- Forecast button on Route Analysis page ----------

def test_route_analysis_forecast_button_works():
    at = _authed_apptest(PAGES_DIR / "2_🗺️_Route_Analysis.py", role="planner")
    # Button "Построить прогноз"
    forecast_btn = at.button(key="btn_forecast")
    forecast_btn.click().run()
    assert not at.exception, f"Forecast button failed: {at.exception}"


# ---------- Comparison button ----------

def test_models_comparison_run_button():
    at = _authed_apptest(PAGES_DIR / "3_⚖️_Models_Comparison.py", role="planner")
    at.button(key="btn_cmp").click().run()
    assert not at.exception, f"Comparison button failed: {at.exception}"
