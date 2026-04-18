"""XGBoost модель прогнозирования пассажиропотока с TimeSeriesSplit."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from .base import BaseForecaster

logger = logging.getLogger(__name__)


class XGBoostForecaster(BaseForecaster):
    """XGBoost с TimeSeriesSplit и лаговыми признаками.

    Преобразует задачу прогнозирования временного ряда в задачу регрессии:
    для каждого наблюдения строятся лаговые признаки (1, 3, 6, 12 месяцев),
    а также циклические временные признаки (sin/cos месяца).

    Для оценки доверительных интервалов используются квантильные модели
    (objective='reg:quantileerror').

    Параметры
    ---------
    lags : list[int]
        Лаги для формирования признаков.
    n_estimators : int
        Число деревьев.
    max_depth : int
        Максимальная глубина дерева.
    learning_rate : float
        Скорость обучения (eta).
    subsample : float
        Доля наблюдений для каждого дерева.
    colsample_bytree : float
        Доля признаков для каждого дерева.
    n_splits : int
        Число фолдов TimeSeriesSplit для кросс-валидации.
    """

    def __init__(
        self,
        lags: list = None,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_splits: int = 5,
    ):
        super().__init__("XGBoost")
        self.lags = lags or [1, 3, 6, 12]
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_splits = n_splits
        self._model: Optional[XGBRegressor] = None
        self._model_lower: Optional[XGBRegressor] = None
        self._model_upper: Optional[XGBRegressor] = None
        self._last_values: Optional[np.ndarray] = None

    def fit(self, series: pd.Series, **kwargs) -> "XGBoostForecaster":
        df_feat = self._build_features(series)
        X = df_feat.drop(columns=["y"])
        y = df_feat["y"]

        # Кросс-валидация для логирования метрик
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            m = self._make_model()
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = m.predict(X.iloc[val_idx])
            mape = np.mean(np.abs((y.iloc[val_idx].values - pred) / (y.iloc[val_idx].values + 1e-8))) * 100
            cv_scores.append(mape)

        logger.info("[XGBoost] CV MAPE: %.2f ± %.2f%%", np.mean(cv_scores), np.std(cv_scores))

        # Финальное обучение на всех данных
        self._model = self._make_model()
        self._model.fit(X, y)

        # Квантильные модели для доверительных интервалов
        self._model_lower = self._make_model(quantile=0.025)
        self._model_lower.fit(X, y)
        self._model_upper = self._make_model(quantile=0.975)
        self._model_upper.fit(X, y)

        self._last_values = series.values
        self._fitted = True
        logger.info("[XGBoost] Обучен на %d наблюдениях.", len(y))
        return self

    def predict(self, horizon: int, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Модель не обучена.")
        return self._recursive_predict(horizon, self._model)

    def get_confidence_intervals(
        self, horizon: int, alpha: float = 0.05, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Модель не обучена.")
        lower = np.maximum(0, self._recursive_predict(horizon, self._model_lower))
        upper = np.maximum(0, self._recursive_predict(horizon, self._model_upper))
        return lower, upper

    def _build_features(self, series: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({"y": series.values})
        for lag in self.lags:
            df[f"lag_{lag}"] = df["y"].shift(lag)
        if hasattr(series.index, "month"):
            m = series.index.month
            df["month_sin"] = np.sin(2 * np.pi * m / 12)
            df["month_cos"] = np.cos(2 * np.pi * m / 12)
            df["quarter"] = (m - 1) // 3 + 1
        return df.dropna().reset_index(drop=True)

    def _recursive_predict(self, horizon: int, model: XGBRegressor) -> np.ndarray:
        history = list(self._last_values)
        preds = []
        for _ in range(horizon):
            row = {}
            for lag in self.lags:
                row[f"lag_{lag}"] = history[-lag] if len(history) >= lag else 0.0
            x = pd.DataFrame([row])
            pred = model.predict(x)[0]
            preds.append(max(0.0, pred))
            history.append(pred)
        return np.array(preds)

    def _make_model(self, quantile: Optional[float] = None) -> XGBRegressor:
        params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=42,
            n_jobs=-1,
        )
        if quantile is not None:
            params["objective"] = "reg:quantileerror"
            params["quantile_alpha"] = quantile
        return XGBRegressor(**params)
