"""SARIMA/SARIMAX модель прогнозирования пассажиропотока."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from .base import BaseForecaster

logger = logging.getLogger(__name__)


class SARIMAForecaster(BaseForecaster):
    """SARIMA(p,d,q)(P,D,Q)s — модель на основе statsmodels.

    При инициализации можно задать фиксированный порядок или включить
    автоматический подбор через pmdarima.auto_arima (auto=True).

    Параметры
    ---------
    order : tuple
        (p, d, q) — несезонные параметры.
    seasonal_order : tuple
        (P, D, Q, s) — сезонные параметры (s=12 для месячных данных).
    auto : bool
        Если True — использовать auto_arima для подбора порядка.
    exog_cols : list, optional
        Имена колонок внешних регрессоров (SARIMAX). DataFrame передаётся
        как exog= в fit().
    """

    def __init__(
        self,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (1, 1, 1, 12),
        auto: bool = False,
        exog_cols: Optional[list] = None,
    ):
        super().__init__("SARIMA")
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto = auto
        self.exog_cols = exog_cols or []
        self._model_fit = None

    def fit(self, series: pd.Series, exog: Optional[pd.DataFrame] = None, **kwargs) -> "SARIMAForecaster":
        self._check_stationarity(series)

        if self.auto:
            self.order, self.seasonal_order = self._auto_select_order(series)

        model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            exog=exog,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._model_fit = model.fit(disp=False, **kwargs)
        self._fitted = True
        logger.info(
            "[SARIMA] Обучена: order=%s, seasonal_order=%s, AIC=%.2f",
            self.order, self.seasonal_order, self._model_fit.aic,
        )
        return self

    def predict(self, horizon: int, exog: Optional[pd.DataFrame] = None, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Модель не обучена.")
        forecast = self._model_fit.forecast(steps=horizon, exog=exog)
        return np.maximum(0, forecast.values)

    def get_confidence_intervals(
        self, horizon: int, alpha: float = 0.05, exog: Optional[pd.DataFrame] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Модель не обучена.")
        pred = self._model_fit.get_forecast(steps=horizon, exog=exog)
        ci = pred.conf_int(alpha=alpha)
        lower = np.maximum(0, ci.iloc[:, 0].values)
        upper = np.maximum(0, ci.iloc[:, 1].values)
        return lower, upper

    def _check_stationarity(self, series: pd.Series) -> None:
        result = adfuller(series.dropna())
        p_value = result[1]
        if p_value > 0.05:
            logger.warning(
                "[SARIMA] ADF-тест: ряд нестационарен (p=%.3f). Рассмотрите d=1 или d=2.",
                p_value,
            )

    def _auto_select_order(self, series: pd.Series) -> Tuple[tuple, tuple]:
        try:
            import pmdarima as pm
            model = pm.auto_arima(
                series,
                seasonal=True,
                m=12,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
            )
            logger.info("[SARIMA] auto_arima выбрала: %s", model.order)
            return model.order, model.seasonal_order
        except ImportError:
            logger.warning("pmdarima не установлена. Используем порядок по умолчанию.")
            return self.order, self.seasonal_order
