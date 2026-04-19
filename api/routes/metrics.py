"""GET /api/metrics — метрики качества модели на hold-out выборке."""

from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import get_current_user
from api.routes.forecast import _series_for
from api.schemas import MetricsOut, ModelType
from models.sarima_model import SARIMAForecaster
from models.xgboost_model import XGBoostForecaster

router = APIRouter()


@router.get("/", response_model=list[MetricsOut])
def get_metrics(
    route_id: int = Query(..., description="ID маршрута"),
    model_type: Optional[ModelType] = Query(None, description="Фильтр по типу модели"),
    test_size: int = Query(12, ge=3, le=24),
    _user: dict = Depends(get_current_user),
):
    series = _series_for(route_id)
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:].values.astype(float)

    results: list[MetricsOut] = []
    models_to_evaluate = [model_type] if model_type else ["sarima", "xgboost", "prophet", "lstm"]
    for mt in models_to_evaluate:
        if mt == "sarima":
            m = SARIMAForecaster()
        elif mt == "xgboost":
            m = XGBoostForecaster(n_estimators=200, n_splits=3)
        elif mt == "prophet":
            try:
                from models.prophet_model import ProphetForecaster
                m = ProphetForecaster()
            except ImportError:
                continue  # skip silently in multi-model comparison
        elif mt == "lstm":
            try:
                from models.lstm_model import LSTMForecaster
                m = LSTMForecaster()
            except ImportError:
                continue
        else:
            raise HTTPException(status_code=503, detail=f"Модель {mt} неизвестна")
        m.fit(train)
        pred = np.asarray(m.predict(horizon=test_size))
        metrics = m.evaluate(test, pred)
        ss_res = float(np.sum((test - pred) ** 2))
        ss_tot = float(np.sum((test - test.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-8 else None
        results.append(MetricsOut(
            route_id=route_id, model_type=mt,
            mape=float(round(metrics["mape"], 2)),
            rmse=float(round(metrics["rmse"], 2)),
            mae=float(round(metrics["mae"], 2)),
            r_squared=float(round(r2, 4)) if r2 is not None else None,
            test_period_months=test_size,
        ))
    return results
