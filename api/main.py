"""FastAPI application entry point.

Реализует 5 основных REST API эндпоинтов из Таблицы 9 ВКР (подраздел 2.3):
  GET  /api/routes            — список маршрутов
  POST /api/forecast          — прогноз с доверительным интервалом
  GET  /api/metrics           — метрики качества модели
  POST /api/models/train      — запуск обучения модели
  GET  /api/reports/{id}      — файл отчёта (PDF/DOCX)

Дополнительные служебные эндпоинты:
  POST /api/auth/register
  POST /api/auth/login
  GET  /health

В отсутствие работающего PostgreSQL API возвращает синтетические данные,
сгенерированные через SyntheticGenerator, чтобы обеспечить работоспособность
UI и тестов без поднятой БД.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import auth as auth_routes
from api.routes import routes as routes_routes
from api.routes import forecast as forecast_routes
from api.routes import metrics as metrics_routes
from api.routes import models as models_routes
from api.routes import reports as reports_routes


app = FastAPI(
    title="Intercity Bus Demand Forecast API",
    description=(
        "REST API информационной системы прогнозирования пассажиропотока "
        "на междугородних автобусных рейсах. Бакалаврская ВКР, МТИ 2026."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "service": "intercity-bus-demand-forecast-api"}


app.include_router(auth_routes.router,     prefix="/api/auth",    tags=["auth"])
app.include_router(routes_routes.router,   prefix="/api/routes",  tags=["routes"])
app.include_router(forecast_routes.router, prefix="/api/forecast",tags=["forecast"])
app.include_router(metrics_routes.router,  prefix="/api/metrics", tags=["metrics"])
app.include_router(models_routes.router,   prefix="/api/models",  tags=["models"])
app.include_router(reports_routes.router,  prefix="/api/reports", tags=["reports"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
