"""Prophet модель прогнозирования пассажиропотока."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseForecaster

logger = logging.getLogger(__name__)


class ProphetForecaster(BaseForecaster):
    """Meta Prophet — модель с явным разложением на тренд, сезонность и праздники.

    Параметры
    ---------
    changepoint_prior_scale : float
        Гибкость тренда. Меньше → сглаженный тренд. По умолчанию 0.05.
    seasonality_mode : str
        'additive' или 'multiplicative'. Для пассажиропотока обычно 'multiplicative'.
    add_russian_holidays : bool
        Добавить российские праздники как внешние регрессоры.
    """

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_mode: str = "multiplicative",
        add_russian_holidays: bool = True,
    ):
        super().__init__("Prophet")
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.add_russian_holidays = add_russian_holidays
        self._model = None
        self._last_df: Optional[pd.DataFrame] = None

    def fit(self, series: pd.Series, **kwargs) -> "ProphetForecaster":
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Установите prophet: pip install prophet")

        df = self._to_prophet_df(series)
        self._last_df = df

        self._model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )

        if self.add_russian_holidays:
            self._add_holidays()

        self._model.fit(df, **kwargs)
        self._fitted = True
        logger.info("[Prophet] Обучена на %d точках.", len(df))
        return self

    def predict(self, horizon: int, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Модель не обучена.")
        future = self._model.make_future_dataframe(periods=horizon, freq="MS")
        forecast = self._model.predict(future)
        result = forecast.tail(horizon)["yhat"].values
        return np.maximum(0, result)

    def get_confidence_intervals(
        self, horizon: int, alpha: float = 0.05, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Модель не обучена.")
        future = self._model.make_future_dataframe(periods=horizon, freq="MS")
        forecast = self._model.predict(future)
        tail = forecast.tail(horizon)
        lower = np.maximum(0, tail["yhat_lower"].values)
        upper = np.maximum(0, tail["yhat_upper"].values)
        return lower, upper

    def plot_components(self) -> None:
        """Визуализирует компоненты Prophet (тренд, сезонность)."""
        if not self._fitted:
            raise RuntimeError("Модель не обучена.")
        future = self._model.make_future_dataframe(periods=0, freq="MS")
        forecast = self._model.predict(future)
        self._model.plot_components(forecast)

    def _to_prophet_df(self, series: pd.Series) -> pd.DataFrame:
        df = series.reset_index()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])
        return df

    def _add_holidays(self) -> None:
        """Добавляет российские праздники как events."""
        import pandas as pd
        years = list(range(2019, 2030))
        holidays = []
        for year in years:
            holidays += [
                {"holiday": "Новый год", "ds": f"{year}-01-01", "lower_window": -1, "upper_window": 7},
                {"holiday": "Рождество", "ds": f"{year}-01-07", "lower_window": 0, "upper_window": 0},
                {"holiday": "День Победы", "ds": f"{year}-05-09", "lower_window": -1, "upper_window": 2},
                {"holiday": "Майские", "ds": f"{year}-05-01", "lower_window": -1, "upper_window": 3},
                {"holiday": "День России", "ds": f"{year}-06-12", "lower_window": 0, "upper_window": 1},
            ]
        self._model.holidays = pd.DataFrame(holidays)
