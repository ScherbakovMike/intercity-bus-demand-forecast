"""Тесты генератора синтетических данных."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.synthetic import SyntheticGenerator


@pytest.fixture
def gen():
    return SyntheticGenerator(n_routes=3, n_years=5, seed=42)


@pytest.fixture
def df(gen):
    return gen.generate()


def test_record_count(df):
    """Генератор возвращает n_routes × n_years × 12 записей."""
    assert len(df) == 3 * 5 * 12


def test_columns(df):
    """DataFrame содержит обязательные колонки."""
    required = {"date", "route_id", "passengers", "trend", "seasonal", "holiday", "noise"}
    assert required.issubset(df.columns)


def test_daily_range(df):
    """Среднесуточный пассажиропоток укладывается в диапазон 15–50 чел/день (ВКР п. 2.2.2)."""
    avg_daily = df["passengers"] / 30
    assert avg_daily.mean() >= 10, f"Среднее {avg_daily.mean():.1f} < 10"
    assert avg_daily.mean() <= 60, f"Среднее {avg_daily.mean():.1f} > 60"


def test_seasonal_peak_march(df):
    """Пик сезонности — март (ВКР: пики март и сентябрь)."""
    monthly_avg = df.groupby(df["date"].dt.month)["passengers"].mean()
    peak_month = monthly_avg.idxmax()
    assert peak_month in (3, 4, 9, 10), f"Пик в месяце {peak_month}, ожидалось 3/4/9/10"


def test_seasonal_trough_january(df):
    """Минимум сезонности — январь (ВКР: минимум январь)."""
    monthly_avg = df.groupby(df["date"].dt.month)["passengers"].mean()
    trough_month = monthly_avg.idxmin()
    assert trough_month in (1, 7), f"Минимум в месяце {trough_month}, ожидалось 1 или 7"


def test_covid_april_2020(df):
    """Апрель 2020 — ковидный минимум ~0.30 от базового (ВКР: снижение до 0.30)."""
    pre = df[df["date"].dt.year == 2019]["passengers"].mean()
    apr2020 = df[(df["date"].dt.year == 2020) & (df["date"].dt.month == 4)]["passengers"].mean()
    ratio = apr2020 / pre
    assert 0.20 <= ratio <= 0.45, f"COVID коэффициент апрель 2020 = {ratio:.2f}, ожидалось 0.20–0.45"


def test_noise_cv(df):
    """Коэффициент вариации шума ≈ 15% (ВКР п. 2.2.2)."""
    cv = df["noise"].std() / df["passengers"].mean()
    assert 0.08 <= cv <= 0.25, f"CV шума = {cv:.2%}, ожидалось 8–25%"


def test_no_negative_passengers(df):
    """Пассажиропоток неотрицателен."""
    assert (df["passengers"] >= 0).all()


def test_route_ids(df):
    """route_id содержит правильный префикс."""
    assert df["route_id"].str.startswith("RU-RURAL-").all()
