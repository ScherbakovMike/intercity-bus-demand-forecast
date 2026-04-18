"""Конфигурация системы прогнозирования пассажиропотока."""

import os
from pathlib import Path

# ── Пути ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved"
REPORTS_DIR = BASE_DIR / "reports"

for _d in (DATA_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── База данных ────────────────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "passenger_forecast")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# ── Горизонты прогноза (лаги в месяцах) ──────────────────────────────────────
# 1 — краткосрочный, 3 — квартальный, 6 — полугодовой, 12 — годовой цикл
FORECAST_HORIZONS = [1, 3, 6, 12]
DEFAULT_HORIZON = 3

# ── Гиперпараметры по умолчанию ───────────────────────────────────────────────
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)

PROPHET_CHANGEPOINT_PRIOR = 0.05
PROPHET_SEASONALITY_MODE = "multiplicative"

LSTM_UNITS = [64, 32]
LSTM_DROPOUT = 0.2
LSTM_RECURRENT_DROPOUT = 0.1
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32
LSTM_MC_PASSES = 50          # число стохастических прогонов MC Dropout

XGBOOST_N_ESTIMATORS = 500
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.05
XGBOOST_SUBSAMPLE = 0.8
XGBOOST_COLSAMPLE_BYTREE = 0.8
XGBOOST_N_SPLITS = 5        # TimeSeriesSplit folds

# ── Пороги качества ────────────────────────────────────────────────────────────
MAPE_WARNING_THRESHOLD = 25.0   # % — выше → предупреждение об аномальной ошибке
MAPE_CRITICAL_THRESHOLD = 50.0  # % — выше → рекомендовать расширить выборку

# ── Параметры синтетических данных ────────────────────────────────────────────
SYNTHETIC_SEED = 42
SYNTHETIC_N_ROUTES = 10
SYNTHETIC_N_YEARS = 5
SYNTHETIC_NOISE_LEVEL = 0.15

# ── Визуализация ──────────────────────────────────────────────────────────────
COLORS = {
    "sarima":   "#2980b9",
    "prophet":  "#27ae60",
    "lstm":     "#8e44ad",
    "xgboost":  "#e67e22",
    "actual":   "#2c3e50",
    "ci":       "#bdc3c7",
}
FIGURE_DPI = 150
FIGURE_SIZE = (12, 6)
