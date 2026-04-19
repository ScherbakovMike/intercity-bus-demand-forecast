"""POST /api/models/train — запуск обучения модели (синхронная заглушка)."""

import uuid
from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_role
from api.routes.forecast import _series_for
from api.schemas import TrainingRequest, TrainingResponse
from models.sarima_model import SARIMAForecaster
from models.xgboost_model import XGBoostForecaster

router = APIRouter()


@router.post("/train", response_model=TrainingResponse)
def train_model(req: TrainingRequest, _user: dict = Depends(require_role("admin", "analyst"))):
    """Синхронно обучает модель и сохраняет гиперпараметры.
    В производственной версии — task-очередь (Celery/RQ)."""
    series = _series_for(req.route_id)
    train = series.iloc[:-12]

    try:
        if req.model_type == "sarima":
            m = SARIMAForecaster()
        elif req.model_type == "xgboost":
            m = XGBoostForecaster(
                n_estimators=int(req.params.get("n_estimators", 200)),
                max_depth=int(req.params.get("max_depth", 6)),
                n_splits=int(req.params.get("n_splits", 3)),
            )
        else:
            raise HTTPException(status_code=503, detail=f"Модель {req.model_type} не установлена")
        m.fit(train)
        status = "completed"
        message = f"Модель {req.model_type} обучена на {len(train)} наблюдениях"
    except Exception as e:
        status = "failed"
        message = f"Ошибка обучения: {e}"

    return TrainingResponse(
        task_id=str(uuid.uuid4()),
        status=status,
        route_id=req.route_id,
        model_type=req.model_type,
        message=message,
    )
