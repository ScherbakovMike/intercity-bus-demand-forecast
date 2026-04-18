"""Визуализация прогнозов и метрик."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

COLORS = {
    "sarima":  "#2980b9",
    "prophet": "#27ae60",
    "lstm":    "#8e44ad",
    "xgboost": "#e67e22",
    "actual":  "#2c3e50",
    "ci":      "#bdc3c7",
}
FIGURE_DPI = 150


class Visualizer:
    """Строит графики прогнозов и метрик (matplotlib + опционально plotly)."""

    def __init__(self, output_dir: Optional[Path] = None, dpi: int = FIGURE_DPI):
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def plot_forecast(
        self,
        actual: pd.Series,
        forecasts: Dict[str, np.ndarray],
        confidence_intervals: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        title: str = "Прогноз пассажиропотока",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """График фактических значений и прогнозов нескольких моделей.

        Параметры
        ---------
        actual : pd.Series
            Фактический ряд с DatetimeIndex.
        forecasts : dict
            {название_модели: array прогнозных значений}.
        confidence_intervals : dict, optional
            {название_модели: (lower_array, upper_array)}.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(actual.index, actual.values, color=COLORS["actual"],
                linewidth=2, label="Факт", zorder=5)

        last_date = actual.index[-1]
        horizon = max(len(v) for v in forecasts.values())
        future_dates = pd.date_range(last_date, periods=horizon + 1, freq="MS")[1:]

        for model_name, pred in forecasts.items():
            color = COLORS.get(model_name.lower(), "#95a5a6")
            ax.plot(future_dates[:len(pred)], pred, color=color,
                    linewidth=1.8, linestyle="--", label=model_name, marker="o", markersize=4)
            if confidence_intervals and model_name in confidence_intervals:
                lo, hi = confidence_intervals[model_name]
                ax.fill_between(future_dates[:len(lo)], lo, hi,
                                color=color, alpha=0.15)

        ax.axvline(x=last_date, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Пассажиры, чел.")
        ax.legend(loc="upper left", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_name:
            path = self.output_dir / save_name
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_metrics_comparison(
        self,
        metrics_df: pd.DataFrame,
        metric: str = "mape",
        title: str = "Сравнение моделей",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """Горизонтальная столбчатая диаграмма метрик по моделям."""
        df = metrics_df.dropna(subset=[metric]).sort_values(metric)
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(df["model"], df[metric],
                       color=[COLORS.get(m.lower(), "#95a5a6") for m in df["model"]])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
        ax.set_xlabel(metric.upper())
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.tight_layout()

        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_residuals(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        model_name: str = "Model",
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """График остатков прогноза."""
        residuals = actual - predicted
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(residuals, marker="o", markersize=4, linewidth=1,
                     color=COLORS.get(model_name.lower(), "#2c3e50"))
        axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
        axes[0].set_title(f"Остатки — {model_name}")
        axes[0].set_ylabel("Ошибка (факт − прогноз)")

        axes[1].hist(residuals, bins=20, color=COLORS.get(model_name.lower(), "#2c3e50"), alpha=0.7)
        axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
        axes[1].set_title("Распределение остатков")
        axes[1].set_xlabel("Ошибка")

        plt.tight_layout()
        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=self.dpi, bbox_inches="tight")
        return fig

    def plot_training_history(
        self,
        history,
        save_name: Optional[str] = None,
    ) -> plt.Figure:
        """График потерь обучения LSTM (train vs val loss)."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history.history["loss"], label="Train loss", color=COLORS["lstm"])
        if "val_loss" in history.history:
            ax.plot(history.history["val_loss"], label="Val loss",
                    color=COLORS["lstm"], linestyle="--", alpha=0.7)
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Кривые обучения LSTM")
        ax.legend()
        plt.tight_layout()
        if save_name:
            fig.savefig(self.output_dir / save_name, dpi=self.dpi, bbox_inches="tight")
        return fig
