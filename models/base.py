"""Абстрактный базовый класс для всех моделей прогнозирования."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Единый интерфейс для всех прогнозных моделей системы.

    Все конкретные модели (SARIMA, Prophet, LSTM, XGBoost) наследуются
    от этого класса и реализуют методы fit() и predict().
    """

    def __init__(self, name: str):
        self.name = name
        self._fitted = False

    # ── Обязательные методы ────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, series: pd.Series, **kwargs) -> "BaseForecaster":
        """Обучает модель на временном ряду.

        Параметры
        ---------
        series : pd.Series
            Ряд с DatetimeIndex и числовыми значениями пассажиропотока.
        """

    @abstractmethod
    def predict(self, horizon: int, **kwargs) -> np.ndarray:
        """Возвращает точечный прогноз на horizon шагов вперёд."""

    # ── Дополнительные методы ──────────────────────────────────────────────────

    def get_confidence_intervals(
        self, horizon: int, alpha: float = 0.05, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает (lower, upper) доверительный интервал уровня (1-alpha).

        По умолчанию — симметричный интервал ±10% (заглушка).
        Переопределяется в конкретных моделях (MC Dropout, Prophet intervals и т.п.).
        """
        point = self.predict(horizon, **kwargs)
        margin = point * 0.10
        return point - margin, point + margin

    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> dict:
        """Вычисляет метрики качества: MAE, RMSE, MAPE."""
        from sys import path as syspath
        import os
        syspath.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from config import MAPE_WARNING_THRESHOLD

        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        nonzero = actual != 0
        mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100

        result = {"model": self.name, "mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}

        if mape > MAPE_WARNING_THRESHOLD:
            logger.warning(
                "[%s] Аномально высокая ошибка прогноза: MAPE=%.1f%% > %.0f%%. "
                "Рекомендуется расширить обучающую выборку.",
                self.name, mape, MAPE_WARNING_THRESHOLD,
            )
            result["warning"] = f"MAPE={mape:.1f}% превышает порог {MAPE_WARNING_THRESHOLD}%"

        return result

    def save(self, path: Path) -> None:
        """Сериализует модель на диск."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("[%s] Модель сохранена: %s", self.name, path)

    @classmethod
    def load(cls, path: Path) -> "BaseForecaster":
        """Загружает модель с диска."""
        import joblib
        model = joblib.load(path)
        logger.info("Модель загружена: %s", path)
        return model

    def __repr__(self) -> str:
        status = "обучена" if self._fitted else "не обучена"
        return f"{self.__class__.__name__}(name='{self.name}', status={status})"
