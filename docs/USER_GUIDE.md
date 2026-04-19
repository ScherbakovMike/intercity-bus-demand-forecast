# User Guide — Intercity Bus Demand Forecast

Practical guide to installing, running and using the forecasting system.

---

## 1. System overview

The system consists of three tiers, deployable together via Docker Compose:

| Tier | Component | Port | Purpose |
|---|---|---|---|
| Presentation | **Streamlit UI** | 8501 | Web UI with 7 pages |
| Business logic | **FastAPI** | 8000 | REST API (5 endpoints) + JWT auth |
| Data | **PostgreSQL 15** | 5432 | 13-table schema, partitioned by year |

Forecasting engine (SARIMA, Prophet, LSTM, XGBoost) runs inside the API service.
Reports (PDF via ReportLab, DOCX via python-docx) are generated on demand.

---

## 2. Prerequisites

- Docker 24+ and Docker Compose v2, **or**
- Python 3.10+ for local (non-docker) run
- 8 GB RAM minimum, 16 GB recommended
- ~3 GB free disk (Postgres data + model cache)

---

## 3. Quick start with Docker

```bash
git clone https://github.com/ScherbakovMike/intercity-bus-demand-forecast
cd intercity-bus-demand-forecast
cp .env.example .env           # edit secrets as needed
docker compose up -d --build
```

Services will become available at:

- UI:    http://localhost:8501
- API:   http://localhost:8000/docs (Swagger)
- DB:    localhost:5432 (user/password from .env)

First start may take ~5 minutes (Postgres init + schema + dependency install).

---

## 4. Quick start without Docker (local)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — UI
streamlit run ui/app.py

# Terminal 3 — open browser
#   http://localhost:8501
```

PostgreSQL is optional: the demo uses in-memory synthetic data and works without a database.

---

## 5. Logging in

Open http://localhost:8501/ and sign in with one of the demo accounts:

| Login | Password | Role |
|---|---|---|
| `admin` | `admin123` | Administrator |
| `planner` | `planner123` | Route planner |
| `analyst` | `analyst123` | Analyst |
| `dispatcher` | `dispatch123` | Dispatcher |

Permissions are role-based — not every user sees every page.

---

## 6. Working with the system

### 6.1 Dashboard (📊)

Landing page with key performance indicators: active routes, monthly total passengers, average traffic, best model. Use tabs to browse the route registry, historical trends, and top routes.

### 6.2 Route analysis (🗺️)

1. Select a route from the dropdown (5 synthetic rural routes, Tver Oblast)
2. Pick a model: `sarima` or `xgboost`
3. Set the horizon (1–24 months)
4. Click **Построить прогноз** (Run forecast)

The page shows: history chart, 3 KPIs (mean/min/max), then after Run: history + forecast + 95% CI band, and a table of point values.

### 6.3 Models comparison (⚖️)

Side-by-side metrics of SARIMA vs XGBoost on hold-out test set. Pick a route + test size, click **Запустить сравнение**. You get the ranked table (MAPE, MAE, RMSE, R²), bar charts and the best-model verdict.

### 6.4 Model management (⚙️) — `admin` / `analyst` only

Retrain a model with custom hyperparameters. For XGBoost: `n_estimators`, `max_depth`, `n_splits`. For SARIMA: auto-arima. Click **Запустить обучение** and watch the log output.

### 6.5 Monitoring (📡)

System status banner (API state, last update, route count). Last month's snapshot, 12-month trend per route, and automatic anomaly detection by rolling Z-score (|z|>2).

### 6.6 Reports (📄)

Generate PDF or DOCX reports with forecast + metrics + confidence intervals. Pick route, model, horizon, format, click **Сгенерировать**. Existing reports appear below and can be downloaded.

### 6.7 Administration (👥) — `admin` only

User list, role-screen matrix, new user form (demo mode — not persisted), full system configuration view.

---

## 7. API for developers

Interactive API docs: http://localhost:8000/docs

### Authenticate

```bash
TOKEN=$(curl -s -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=planner&password=planner123" | jq -r .access_token)
```

### List routes

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/routes/
```

### Get a forecast

```bash
curl -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -X POST http://localhost:8000/api/forecast/ \
  -d '{"route_id": 1, "model_type": "sarima", "horizon": 12}'
```

### Get model metrics

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/metrics/?route_id=1"
```

### Generate a PDF report

```bash
curl -H "Authorization: Bearer $TOKEN" -X POST \
  "http://localhost:8000/api/reports/generate?route_id=1&model_type=sarima&horizon=12&format=pdf"
```

---

## 8. Database operations

With Docker running, apply Alembic migrations:

```bash
docker compose exec api alembic upgrade head
```

Or inspect data:

```bash
docker compose exec postgres psql -U postgres -d passenger_forecast -c "\dt"
```

---

## 9. Running tests

```bash
# Unit tests for the forecasting core (14 tests)
python -m pytest tests/test_synthetic.py tests/test_comparator.py -v

# API integration tests (21 tests)
python -m pytest tests/test_api.py -v

# UI tests via Streamlit testing framework (13 tests)
python -m pytest tests/test_ui.py -v
```

All 48 tests should pass.

---

## 10. Generating thesis-ready screenshots

With UI running on `:8501`:

```bash
python scripts/capture_screenshots.py
```

Headless Chromium walks through the 7 pages and saves PNGs to
`../output/figures/ui_*.png` (10 screenshots total).

---

## 11. Troubleshooting

**`ModuleNotFoundError: No module named 'ui'`**
Run Streamlit from the repo root: `streamlit run ui/app.py` (not from inside `ui/`).

**Streamlit pages show "Войдите в систему через главную страницу"**
You tried to open a page URL directly. Go to the root `/` first and log in.

**`ValueError: password cannot be longer than 72 bytes`**
Fixed in `api/auth.py` — passwords are truncated to 72 bytes before bcrypt.

**Docker Postgres fails to start**
Check port 5432 is free: `lsof -i :5432` (Linux/Mac) or `netstat -ano | findstr 5432` (Windows).

**Forecast takes >30 s**
Normal for XGBoost cold-start on first request (model training). Subsequent requests hit the LRU cache.

---

## 12. Next steps for production

- Replace in-memory user store with PostgreSQL `app_user` table
- Move synchronous training endpoint to a task queue (Celery or RQ)
- Add Prometheus metrics + Grafana dashboard
- Enable HTTPS via nginx reverse proxy + Let's Encrypt
- Increase `uvicorn` worker count for concurrent forecasts
- Set `JWT_SECRET` to a cryptographically strong value in `.env`
