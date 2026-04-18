"""Генератор синтетических данных пассажиропотока для сельского кейса.

Сезонная модель соответствует параметрам Раздела 2.2.2 ВКР:
  - Тренд: +3% в год
  - Сезонность: пики в марте и сентябре, минимум в январе и июле
  - COVID: снижение до 0.30 в апреле 2020, восстановление до 0.85 к октябрю 2021
  - Шум: ε ~ N(0, 0.15·μ), коэффициент вариации 15%
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Месячные коэффициенты сезонности (пики мар/сен, минимум янв/июл)
# Нормированы так, что среднее = 1.0
_SEASONAL_COEFF = np.array([
    0.62,  # январь   — минимум
    0.78,  # февраль
    1.22,  # март     — весенний пик
    1.28,  # апрель
    1.08,  # май
    0.92,  # июнь
    0.80,  # июль     — летний минимум
    0.86,  # август
    1.24,  # сентябрь — осенний пик
    1.18,  # октябрь
    0.94,  # ноябрь
    0.78,  # декабрь
])
# Нормализуем: среднее = 1.0
_SEASONAL_COEFF = _SEASONAL_COEFF / _SEASONAL_COEFF.mean()


class SyntheticGenerator:
    """Генерирует синтетические ряды пассажиропотока для нескольких маршрутов.

    Параметры
    ---------
    n_routes : int
        Число маршрутов.
    n_years : int
        Число лет истории (начиная с 2019 г.).
    noise_level : float
        Стандартное отклонение мультипликативного шума (доля от значения).
        По умолчанию 0.15, что даёт коэффициент вариации ~15% согласно ВКР.
    seed : int
        Зерно генератора для воспроизводимости.
    base_passengers : int
        Базовый среднемесячный пассажиропоток (чел.). При 550 чел/мес среднесуточный
        показатель составляет ~18 чел/день — центр диапазона 15–50 чел/день из ВКР.
    """

    def __init__(
        self,
        n_routes: int = 10,
        n_years: int = 5,
        noise_level: float = 0.15,
        seed: int = 42,
        base_passengers: int = 550,
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
            # Базовый пассажиропоток маршрута: варьируется от ~15 до ~50 чел/день
            # 450*0.9=405 до 450*3.0=1350 мес → 13.5–45 чел/день
            base = self.base_passengers * rng.uniform(0.9, 3.0)

            # Долгосрочный тренд: +3% в год (= +0.25% в месяц)
            trend_slope = base * 0.03 / 12
            trend = np.arange(len(dates)) * trend_slope

            # Годовая сезонность: пики март/сентябрь, спад январь/июль
            month_nums = np.array([d.month for d in dates])
            seasonal_factor = _SEASONAL_COEFF[month_nums - 1]  # коэффициент для каждой точки
            # Сезонное отклонение от нуля: (coeff - 1) * base
            seasonal = (seasonal_factor - 1.0) * base

            # Праздничный эффект: январь −8%, майские +6%
            holiday = np.zeros(len(dates))
            holiday[month_nums == 1] = -base * 0.08
            holiday[month_nums == 5] = base * 0.06

            # COVID-провал 2020–2021:
            # - апрель 2020: снижение до 0.30
            # - постепенный спад с января 2020 по апрель 2020
            # - постепенное восстановление: к октябрю 2021 = 0.85
            covid_recovery = np.ones(len(dates))
            for i, d in enumerate(dates):
                if d.year == 2020:
                    if d.month <= 3:
                        # до апреля — нарастающее снижение
                        covid_recovery[i] = 1.0 - (1.0 - 0.30) * (d.month - 1) / 3
                    elif d.month == 4:
                        covid_recovery[i] = 0.30  # минимум
                    else:
                        # частичное восстановление к концу 2020: апрель=0.30 → декабрь=0.55
                        covid_recovery[i] = 0.30 + (0.55 - 0.30) * (d.month - 4) / 8
                elif d.year == 2021:
                    # восстановление: январь=0.57 → октябрь=0.85 → декабрь=0.90
                    if d.month <= 10:
                        covid_recovery[i] = 0.57 + (0.85 - 0.57) * (d.month - 1) / 9
                    else:
                        covid_recovery[i] = 0.85 + (0.90 - 0.85) * (d.month - 10) / 2

            signal = (base + trend + seasonal + holiday) * covid_recovery
            noise = signal * rng.normal(0, self.noise_level, len(dates))
            passengers = np.maximum(0, signal + noise).round().astype(int)

            for i, date in enumerate(dates):
                records.append({
                    "date": date,
                    "route_id": route_id,
                    "passengers": int(passengers[i]),
                    "trend": round(base + trend[i], 1),
                    "seasonal": round(seasonal[i], 1),
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
