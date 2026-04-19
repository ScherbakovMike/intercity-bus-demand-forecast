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
        # Cache train-series stats for CI sanitization
        self._train_max = float(series.max())
        self._train_std = float(series.std())
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
        raw_lower = ci.iloc[:, 0].values
        raw_upper = ci.iloc[:, 1].values
        point = np.maximum(0, pred.predicted_mean.values)

        # Sanity-check аналитического ДИ: если верхняя граница превышает
        # 5× максимума обучающей выборки — модель не сошлась (характерно
        # для длинных горизонтов при нестабильном MLE). Заменяем на
        # эмпирический ДИ point ± 2·σ_train, что устойчиво и интерпретируемо.
        if hasattr(self, '_train_max') and hasattr(self, '_train_std'):
            max_reasonable = self._train_max * 5.0
            if np.any(raw_upper > max_reasonable) or np.any(np.abs(raw_lower) > max_reasonable):
                # Fallback: эмпирический ДИ
                sigma = max(self._train_std, 1.0)
                lower = np.maximum(0, point - 2.0 * sigma)
                upper = point + 2.0 * sigma
                return lower, upper

        lower = np.maximum(0, raw_lower)
        upper = np.maximum(0, raw_upper)
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
