"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def auth_headers(client):
    """Log in as planner and return Authorization header."""
    r = client.post("/api/auth/login", data={"username": "planner", "password": "planner123"})
    assert r.status_code == 200
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def admin_headers(client):
    r = client.post("/api/auth/login", data={"username": "admin", "password": "admin123"})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ---------- Health ----------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------- Auth ----------

def test_login_success(client):
    r = client.post("/api/auth/login", data={"username": "admin", "password": "admin123"})
    assert r.status_code == 200
    body = r.json()
    assert "access_token" in body
    assert body["role"] == "admin"


def test_login_failure(client):
    r = client.post("/api/auth/login", data={"username": "admin", "password": "wrong"})
    assert r.status_code == 401


def test_me_requires_auth(client):
    r = client.get("/api/auth/me")
    assert r.status_code == 401


def test_me_returns_user_info(client, auth_headers):
    r = client.get("/api/auth/me", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["username"] == "planner"
    assert body["role"] == "planner"


# ---------- Routes ----------

def test_list_routes(client, auth_headers):
    r = client.get("/api/routes/", headers=auth_headers)
    assert r.status_code == 200
    routes = r.json()
    assert isinstance(routes, list)
    assert len(routes) == 5
    assert all("route_id" in r and "name" in r for r in routes)


def test_list_routes_filter_by_status(client, auth_headers):
    r = client.get("/api/routes/?status=active", headers=auth_headers)
    assert r.status_code == 200
    assert all(r["status"] == "active" for r in r.json())


def test_get_route_by_id(client, auth_headers):
    r = client.get("/api/routes/1", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["route_id"] == 1


def test_get_nonexistent_route(client, auth_headers):
    r = client.get("/api/routes/999", headers=auth_headers)
    assert r.status_code == 404


# ---------- Forecast ----------

def test_forecast_sarima(client, auth_headers):
    r = client.post("/api/forecast/",
                    json={"route_id": 1, "model_type": "sarima", "horizon": 6},
                    headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["route_id"] == 1
    assert body["model_type"] == "sarima"
    assert body["horizon"] == 6
    assert len(body["points"]) == 6
    for p in body["points"]:
        assert p["point"] >= 0
        assert p["lower"] <= p["upper"]


def test_forecast_xgboost(client, auth_headers):
    r = client.post("/api/forecast/",
                    json={"route_id": 2, "model_type": "xgboost", "horizon": 3},
                    headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert len(body["points"]) == 3


def test_forecast_prophet_or_missing(client, auth_headers):
    """Prophet returns 200 if installed, 503 otherwise."""
    r = client.post("/api/forecast/",
                    json={"route_id": 1, "model_type": "prophet", "horizon": 6},
                    headers=auth_headers)
    assert r.status_code in (200, 503), f"Unexpected status: {r.status_code}"
    if r.status_code == 200:
        body = r.json()
        assert len(body["points"]) == 6
        assert all(p["point"] >= 0 for p in body["points"])


def test_forecast_unauthorized(client):
    r = client.post("/api/forecast/",
                    json={"route_id": 1, "model_type": "sarima", "horizon": 6})
    assert r.status_code == 401


# ---------- Metrics ----------

def test_metrics_all_models(client, auth_headers):
    r = client.get("/api/metrics/?route_id=1", headers=auth_headers)
    assert r.status_code == 200
    results = r.json()
    # 2 if only sarima+xgboost available, 3 with prophet, 4 with lstm
    assert 2 <= len(results) <= 4, f"Expected 2-4 models, got {len(results)}"
    model_types = {m["model_type"] for m in results}
    assert "sarima" in model_types and "xgboost" in model_types
    for m in results:
        assert m["mape"] >= 0
        assert m["mae"] >= 0
        assert m["rmse"] >= 0


def test_metrics_single_model(client, auth_headers):
    r = client.get("/api/metrics/?route_id=1&model_type=sarima", headers=auth_headers)
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 1
    assert results[0]["model_type"] == "sarima"


# ---------- Training ----------

def test_train_requires_privileged_role(client, auth_headers):
    """Planner shouldn't have access to train endpoint."""
    r = client.post("/api/models/train",
                    json={"route_id": 1, "model_type": "sarima", "params": {}},
                    headers=auth_headers)
    assert r.status_code == 403


def test_train_admin_success(client, admin_headers):
    r = client.post("/api/models/train",
                    json={"route_id": 1, "model_type": "sarima", "params": {}},
                    headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["status"] == "completed"


# ---------- Reports ----------

def test_generate_pdf_report(client, auth_headers):
    r = client.post("/api/reports/generate?route_id=1&model_type=sarima&horizon=6&format=pdf",
                    headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "generated"
    assert body["size_bytes"] > 500  # PDF with real content
    assert body["report_id"].endswith(".pdf")


def test_generate_docx_report(client, auth_headers):
    r = client.post("/api/reports/generate?route_id=2&model_type=xgboost&horizon=6&format=docx",
                    headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["report_id"].endswith(".docx")


def test_list_reports(client, auth_headers):
    r = client.get("/api/reports/", headers=auth_headers)
    assert r.status_code == 200
    # Should have at least the reports we just created
    assert isinstance(r.json(), list)


def test_download_nonexistent_report(client, auth_headers):
    r = client.get("/api/reports/nonexistent.pdf", headers=auth_headers)
    assert r.status_code == 404
