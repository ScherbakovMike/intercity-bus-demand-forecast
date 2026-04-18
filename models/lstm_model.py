"""LSTM модель с Monte Carlo Dropout для прогнозирования пассажиропотока."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseForecaster

logger = logging.getLogger(__name__)

# Ленивый импорт TensorFlow — не падаем при отсутствии GPU
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class LSTMForecaster(BaseForecaster):
    """LSTM с MC Dropout (Monte Carlo Dropout) для оценки неопределённости.

    Архитектура: LSTM(units[0]) → Dropout → LSTM(units[1]) → Dropout → Dense(1)

    MC Dropout: для получения доверительных интервалов используется
    стохастичность при инференсе — вызов с training=True позволяет Dropout
    оставаться активным, генерируя распределение прогнозов:
        predictions = [model(X, training=True) for _ in range(mc_passes)]

    Параметры
    ---------
    units : list[int]
        Число нейронов в каждом LSTM-слое.
    dropout : float
        Доля отключаемых нейронов в Dropout-слоях.
    recurrent_dropout : float
        Dropout внутри LSTM-ячеек (рекуррентный).
    lookback : int
        Глубина окна (число прошлых месяцев на входе).
    epochs : int
        Максимальное число эпох обучения.
    batch_size : int
        Размер мини-батча.
    mc_passes : int
        Число стохастических прогонов при MC Dropout.
    """

    def __init__(
        self,
        units: list = None,
        dropout: float = 0.2,
        recurrent_dropout: float = 0.1,
        lookback: int = 12,
        epochs: int = 100,
        batch_size: int = 32,
        mc_passes: int = 50,
    ):
        super().__init__("LSTM")
        self.units = units or [64, 32]
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.mc_passes = mc_passes
        self._model = None
        self._history = None

    def fit(self, series: pd.Series, validation_split: float = 0.1, **kwargs) -> "LSTMForecaster":
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow не установлен: pip install tensorflow")

        X, y = self._make_sequences(series.values)
        self._model = self._build_model(X.shape[1], X.shape[2])

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )

        self._history = self._model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0,
            **kwargs,
        )
        self._fitted = True
        final_loss = self._history.history["loss"][-1]
        logger.info(
            "[LSTM] Обучена: %d эпох, loss=%.4f. units=%s, dropout=%.2f, lookback=%d",
            len(self._history.history["loss"]), final_loss,
            self.units, self.dropout, self.lookback,
        )
        return self

    def predict(self, horizon: int, seed_sequence: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Модель не обучена.")
        return self._recursive_forecast(horizon, seed_sequence, stochastic=False)

    def get_confidence_intervals(
        self, horizon: int, alpha: float = 0.05,
        seed_sequence: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """MC Dropout: N стохастических прогонов → перцентильные интервалы."""
        if not self._fitted:
            raise RuntimeError("Модель не обучена.")

        mc_samples = np.array([
            self._recursive_forecast(horizon, seed_sequence, stochastic=True)
            for _ in range(self.mc_passes)
        ])

        lower_pct = (alpha / 2) * 100
        upper_pct = (1 - alpha / 2) * 100
        lower = np.percentile(mc_samples, lower_pct, axis=0)
        upper = np.percentile(mc_samples, upper_pct, axis=0)
        return np.maximum(0, lower), np.maximum(0, upper)

    def _build_model(self, timesteps: int, features: int):
        inputs = keras.Input(shape=(timesteps, features))
        x = keras.layers.LSTM(
            self.units[0],
            return_sequences=len(self.units) > 1,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
        )(inputs)
        for i, u in enumerate(self.units[1:], 1):
            return_seq = i < len(self.units) - 1
            x = keras.layers.LSTM(u, return_sequences=return_seq,
                                   dropout=self.dropout,
                                   recurrent_dropout=self.recurrent_dropout)(x)
        x = keras.layers.Dense(1)(x)
        model = keras.Model(inputs, x)
        model.compile(optimizer="adam", loss="mse")
        return model

    def _make_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback: i])
            y.append(data[i])
        X = np.array(X).reshape(-1, self.lookback, 1)
        y = np.array(y)
        return X, y

    def _recursive_forecast(
        self, horizon: int, seed: Optional[np.ndarray], stochastic: bool
    ) -> np.ndarray:
        """Авторегрессионный (рекурсивный) прогноз на horizon шагов."""
        if seed is None:
            raise ValueError("Передайте seed_sequence — последние lookback значений ряда.")
        window = seed[-self.lookback:].copy()
        preds = []
        for _ in range(horizon):
            x = window.reshape(1, self.lookback, 1).astype("float32")
            # training=True активирует Dropout при инференсе (MC Dropout)
            pred = self._model(x, training=stochastic).numpy().ravel()[0]
            preds.append(pred)
            window = np.roll(window, -1)
            window[-1] = pred
        return np.maximum(0, np.array(preds))
