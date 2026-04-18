"""Сравнение моделей прогнозирования по метрикам на тестовой выборке."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .sarima_model import SARIMAForecaster
from .prophet_model import ProphetForecaster
from .lstm_model import LSTMForecaster
from .xgboost_model import XGBoostForecaster

logger = logging.getLogger(__name__)


class ModelComparator:
    """Обучает все модели, делает прогнозы и сравнивает метрики.

    Параметры
    ---------
    models : list[BaseForecaster], optional
        Список моделей для сравнения. По умолчанию — все четыре (SARIMA,
        Prophet, LSTM, XGBoost).
    test_size : int
        Число последних наблюдений, отложенных в тестовую выборку.
    """

    DEFAULT_MODELS: List[Type[BaseForecaster]] = [
        SARIMAForecaster,
        ProphetForecaster,
        XGBoostForecaster,
    ]

    def __init__(
        self,
        models: Optional[List[BaseForecaster]] = None,
        test_size: int = 12,
        include_lstm: bool = False,
    ):
        self.test_size = test_size
        self.include_lstm = include_lstm
        if models is not None:
            self.models = models
        else:
            self.models = [cls() for cls in self.DEFAULT_MODELS]
            if include_lstm:
                self.models.append(LSTMForecaster())
        self._results: Optional[pd.DataFrame] = None

    def compare(
        self,
        df: pd.DataFrame,
        target_col: str = "passengers",
        date_col: str = "date",
        route_id: Optional[str] = None,
        horizon: int = 3,
    ) -> pd.DataFrame:
        """Обучает модели и возвращает таблицу метрик.

        Параметры
        ---------
        df : pd.DataFrame
            DataFrame с колонками date, target_col. Если есть route_id, фильтрует.
        horizon : int
            Горизонт прогноза (шагов вперёд). Должен быть ≤ test_size.

        Возвращает
        ----------
        pd.DataFrame
            Строки = модели, колонки = MAE, RMSE, MAPE, rank.
        """
        if route_id:
            df = df[df["route_id"] == route_id]

        series = df.set_index(date_col)[target_col].sort_index()
        train = series.iloc[:-self.test_size]
        test = series.iloc[-self.test_size:]

        records = []
        for model in self.models:
            try:
                if isinstance(model, LSTMForecaster):
                    model.fit(train)
                    seed = train.values
                    predicted = model.predict(horizon, seed_sequence=seed)
                else:
                    model.fit(train)
                    predicted = model.predict(horizon)

                actual = test.values[:horizon]
                metrics = model.evaluate(actual, predicted)
                metrics["horizon"] = horizon
                records.append(metrics)
                logger.info("[Comparator] %s: MAPE=%.2f%%", model.name, metrics["mape"])
            except Exception as e:
                logger.error("[Comparator] %s завершилась с ошибкой: %s", model.name, e)
                records.append({"model": model.name, "mae": None, "rmse": None, "mape": None, "error": str(e)})

        result = pd.DataFrame(records)
        if "mape" in result.columns:
            result = result.sort_values("mape").reset_index(drop=True)
            result.insert(0, "rank", result.index + 1)

        self._results = result
        return result

    def best_model(self) -> Optional[BaseForecaster]:
        """Возвращает лучшую модель по MAPE (наименьшая ошибка)."""
        if self._results is None:
            logger.warning("Сначала вызовите compare().")
            return None
        best_name = self._results.dropna(subset=["mape"]).iloc[0]["model"]
        return next((m for m in self.models if m.name == best_name), None)

    def summary(self) -> str:
        """Текстовое резюме сравнения."""
        if self._results is None:
            return "Нет результатов. Вызовите compare()."
        return self._results[["rank", "model", "mae", "rmse", "mape"]].to_string(index=False)
