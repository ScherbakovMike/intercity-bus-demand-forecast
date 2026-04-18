"""Генератор синтетических данных пассажиропотока для сельского кейса.

Модель: пассажиропоток = базовый тренд + годовая сезонность + летний пик
        + праздники + COVID-провал (2020–2021) + шум.
Соответствует параметрам Раздела 2.2.2 ВКР.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SyntheticGenerator:
    """Генерирует синтетические ряды пассажиропотока для нескольких маршрутов.

    Параметры
    ---------
    n_routes : int
        Число маршрутов.
    n_years : int
        Число лет (2019–2019+n_years).
    noise_level : float
        Стандартное отклонение мультипликативного шума (доля от значения).
    seed : int
        Зерно генератора для воспроизводимости.
    base_passengers : int
        Средний пассажиропоток на маршруте в месяц (чел.).
    """

    def __init__(
        self,
        n_routes: int = 10,
        n_years: int = 5,
        noise_level: float = 0.08,
        seed: int = 42,
        base_passengers: int = 800,
    ):
        self.n_routes = n_routes
        self.n_years = n_years
        self.noise_level = noise_level
        self.seed = seed
        self.base_passengers = base_passengers

    def generate(self, save_path: Optional[Path] = None) -> pd.DataFrame:
        """Генерирует датасет и опционально сохраняет в CSV.

        Возвращает
        ----------
        pd.DataFrame
            Колонки: date, route_id, passengers, trend, seasonal, holiday, noise
        """
        rng = np.random.default_rng(self.seed)
        start_date = pd.Timestamp("2019-01-01")
        dates = pd.date_range(start_date, periods=self.n_years * 12, freq="MS")
        records = []

        for route_idx in range(self.n_routes):
            route_id = f"RU-RURAL-{route_idx + 1:03d}"
            base = self.base_passengers * rng.uniform(0.5, 2.0)

            # Долгосрочный тренд: слабый рост ~2% в год
            trend_slope = base * 0.02 / 12
            trend = np.arange(len(dates)) * trend_slope

            # Годовая сезонность: максимум летом (июль), минимум зимой
            month_nums = np.array([d.month for d in dates])
            seasonal = base * 0.25 * np.sin(2 * np.pi * (month_nums - 1) / 12 - np.pi / 2)

            # Летний пик: дополнительный подъём в июле–августе
            summer_peak = base * 0.15 * np.exp(-0.5 * ((month_nums - 7.5) / 1.2) ** 2)

            # Праздничный эффект: январь и майские
            holiday = np.zeros(len(dates))
            holiday[month_nums == 1] = -base * 0.08
            holiday[month_nums == 5] = base * 0.06

            # COVID-провал 2020–2021
            covid_mask = np.array([d.year in (2020, 2021) for d in dates])
            covid_recovery = np.ones(len(dates))
            for i, d in enumerate(dates):
                if d.year == 2020:
                    # Спад с апреля 2020
                    months_since_start = max(0, (d.month - 4))
                    covid_recovery[i] = max(0.35, 1.0 - 0.12 * months_since_start)
                elif d.year == 2021:
                    # Постепенное восстановление
                    covid_recovery[i] = 0.55 + 0.04 * d.month

            signal = (base + trend + seasonal + summer_peak + holiday) * covid_recovery
            noise = signal * rng.normal(0, self.noise_level, len(dates))
            passengers = np.maximum(0, signal + noise).round().astype(int)

            for i, date in enumerate(dates):
                records.append({
                    "date": date,
                    "route_id": route_id,
                    "passengers": passengers[i],
                    "trend": round(base + trend[i], 1),
                    "seasonal": round(seasonal[i] + summer_peak[i], 1),
                    "holiday": round(holiday[i], 1),
                    "noise": round(noise[i], 1),
                })

        df = pd.DataFrame(records).sort_values(["route_id", "date"]).reset_index(drop=True)
        logger.info(
            "Синтетические данные: %d маршрутов × %d месяцев = %d записей",
            self.n_routes, self.n_years * 12, len(df),
        )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            logger.info("Сохранено: %s", save_path)

        return df

    def generate_single_route(self, route_id: str = "RU-RURAL-001") -> pd.Series:
        """Генерирует ряд для одного маршрута (удобно для быстрых тестов)."""
        df = self.generate()
        series = df[df["route_id"] == route_id].set_index("date")["passengers"]
        return series


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    gen = SyntheticGenerator()
    out_path = Path(__file__).parent.parent / "data" / "processed" / "synthetic_passengers.csv"
    df = gen.generate(save_path=out_path)
    print(df.head(10).to_string())
    print(f"\nВсего записей: {len(df)}")
    print(f"Маршруты: {df['route_id'].nunique()}")
    print(f"Период: {df['date'].min().date()} — {df['date'].max().date()}")
