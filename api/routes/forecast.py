"""POST /api/forecast — прогноз пассажиропотока.

Использует реальные SARIMA/XGBoost модели из models/ на синтетических данных
из SyntheticGenerator (Тверская обл., 5 маршрутов). Без необходимости
запущенной PostgreSQL.
"""

from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.auth import get_current_user
from api.schemas import ForecastPoint, ForecastRequest, ForecastResponse
from data.synthetic import SyntheticGenerator
from models.sarima_model import SARIMAForecaster
from models.xgboost_model import XGBoostForecaster

router = APIRouter()


@lru_cache(maxsize=1)
def _cached_dataset():
    """Генерируем синтетический датасет один раз за жизнь процесса."""
    gen = SyntheticGenerator(n_routes=5, n_years=7, seed=42)
    df = gen.generate()
    return df


def _series_for(route_id: int) -> pd.Series:
    df = _cached_dataset()
    route_code = f"RU-RURAL-{route_id:03d}"
    sub = df[df["route_id"] == route_code]
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"Нет данных для маршрута {route_id}")
    return sub.set_index("date")["passengers"].sort_index()


@router.post("/", response_model=ForecastResponse)
def forecast(req: ForecastRequest, _user: dict = Depends(get_current_user)):
    series = _series_for(req.route_id)
    train = series.iloc[: -req.horizon]

    if req.model_type == "sarima":
        model = SARIMAForecaster()
    elif req.model_type == "xgboost":
        model = XGBoostForecaster(n_estimators=200, n_splits=3)
    elif req.model_type == "prophet":
        try:
            from models.prophet_model import ProphetForecaster
            model = ProphetForecaster()
        except ImportError as e:
            raise HTTPException(status_code=503,
                                detail=f"Prophet не установлен: {e}")
    elif req.model_type == "lstm":
        try:
            from models.lstm_model import LSTMForecaster
            model = LSTMForecaster()
        except ImportError as e:
            raise HTTPException(status_code=503,
                                detail=f"LSTM (tensorflow-cpu) не установлен: {e}")
    else:
        raise HTTPException(status_code=400, detail=f"Неизвестная модель {req.model_type}")

    model.fit(train)
    point = np.asarray(model.predict(horizon=req.horizon))
    lower, upper = model.get_confidence_intervals(horizon=req.horizon)
    lower, upper = np.asarray(lower), np.asarray(upper)

    points = [
        ForecastPoint(month_offset=i + 1,
                      point=float(round(point[i], 2)),
                      lower=float(round(lower[i], 2)),
                      upper=float(round(upper[i], 2)))
        for i in range(req.horizon)
    ]
    return ForecastResponse(
        route_id=req.route_id,
        model_type=req.model_type,
        horizon=req.horizon,
        points=points,
        generated_at=datetime.now(timezone.utc),
    )
