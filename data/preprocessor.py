"""Предобработка временных рядов: очистка, признаки, нормализация."""

from __future__ import annotations

import joblib
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL

logger = logging.getLogger(__name__)


class Preprocessor:
    """Конвейер предобработки временного ряда пассажиропотока.

    Этапы:
    1. Обнаружение и удаление выбросов (IQR, 3-sigma)
    2. Заполнение пропусков (STL-интерполяция или линейная)
    3. Генерация временных признаков (месяц, квартал, день недели и др.)
    4. Нормализация MinMaxScaler [0, 1]
    """

    def __init__(
        self,
        outlier_method: str = "iqr",
        fill_method: str = "stl",
        scale: bool = True,
        seasonal_period: int = 12,
    ):
        self.outlier_method = outlier_method
        self.fill_method = fill_method
        self.scale = scale
        self.seasonal_period = seasonal_period
        self._scaler: Optional[MinMaxScaler] = None
        self._fitted = False

    # ── Публичный API ──────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame, target_col: str = "passengers") -> pd.DataFrame:
        """Обучает препроцессор и трансформирует данные."""
        df = df.copy()
        df = self._remove_outliers(df, target_col)
        df = self._fill_missing(df, target_col)
        df = self._add_time_features(df)
        if self.scale:
            df[target_col] = self._fit_scale(df[[target_col]])
        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame, target_col: str = "passengers") -> pd.DataFrame:
        """Трансформирует новые данные с обученным препроцессором."""
        if not self._fitted:
            raise RuntimeError("Preprocessor не обучен. Вызовите fit_transform() сначала.")
        df = df.copy()
        df = self._fill_missing(df, target_col)
        df = self._add_time_features(df)
        if self.scale and self._scaler:
            df[target_col] = self._scaler.transform(df[[target_col]])
        return df

    def inverse_scale(self, values: np.ndarray) -> np.ndarray:
        """Обратная нормализация прогнозных значений."""
        if self._scaler is None:
            return values
        return self._scaler.inverse_transform(values.reshape(-1, 1)).ravel()

    def save_state(self, path: Path) -> None:
        joblib.dump({"scaler": self._scaler, "params": self._get_params()}, path)
        logger.info("Препроцессор сохранён: %s", path)

    def load_state(self, path: Path) -> None:
        state = joblib.load(path)
        self._scaler = state["scaler"]
        for k, v in state["params"].items():
            setattr(self, k, v)
        self._fitted = True
        logger.info("Препроцессор загружен: %s", path)

    # ── Внутренние методы ──────────────────────────────────────────────────────

    def _remove_outliers(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        s = df[col].copy()
        if self.outlier_method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
        else:  # 3-sigma
            mask = (s - s.mean()).abs() > 3 * s.std()

        n_outliers = mask.sum()
        if n_outliers > 0:
            logger.info("Обнаружено выбросов: %d (метод: %s)", n_outliers, self.outlier_method)
            df.loc[mask, col] = np.nan
        return df

    def _fill_missing(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            return df

        logger.info("Заполняем пропуски: %d значений (метод: %s)", n_missing, self.fill_method)

        if self.fill_method == "stl" and len(df) >= 2 * self.seasonal_period:
            try:
                filled = df[col].interpolate(method="linear")
                stl = STL(filled, period=self.seasonal_period, robust=True)
                result = stl.fit()
                trend_season = result.trend + result.seasonal
                df[col] = df[col].fillna(trend_season)
                return df
            except Exception as e:
                logger.warning("STL не удалась (%s), используем линейную интерполяцию.", e)

        # Fallback: если ряд слишком короткий или STL не сработала
        df[col] = df[col].interpolate(method="linear").ffill().bfill()
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "date" not in df.columns:
            return df
        dt = pd.to_datetime(df["date"])
        df["month"] = dt.dt.month
        df["quarter"] = dt.dt.quarter
        df["year"] = dt.dt.year
        df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
        df["is_summer"] = dt.dt.month.isin([6, 7, 8]).astype(int)
        df["is_holiday_season"] = dt.dt.month.isin([1, 5, 12]).astype(int)
        return df

    def _fit_scale(self, data: pd.DataFrame) -> np.ndarray:
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        return self._scaler.fit_transform(data).ravel()

    def _get_params(self) -> dict:
        return {
            "outlier_method": self.outlier_method,
            "fill_method": self.fill_method,
            "scale": self.scale,
            "seasonal_period": self.seasonal_period,
        }


def make_supervised(series: pd.Series, lags: list[int] = None) -> pd.DataFrame:
    """Преобразует ряд в датафрейм с лаговыми признаками для XGBoost.

    Параметры
    ---------
    series : pd.Series
        Временной ряд пассажиропотока.
    lags : list[int]
        Список лагов в месяцах (по умолчанию: 1, 3, 6, 12).
    """
    from config import FORECAST_HORIZONS
    lags = lags or FORECAST_HORIZONS
    df = pd.DataFrame({"y": series.values})
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    return df.dropna().reset_index(drop=True)
