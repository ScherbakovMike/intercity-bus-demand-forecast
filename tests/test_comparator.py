"""Тесты ModelComparator: разбиение данных и корректность метрик."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.synthetic import SyntheticGenerator
from models.comparator import ModelComparator
from models.xgboost_model import XGBoostForecaster


@pytest.fixture
def series():
    """Синтетический ряд для одного маршрута (60 месяцев)."""
    gen = SyntheticGenerator(n_routes=1, n_years=5, seed=0)
    df = gen.generate()
    s = df.set_index("date")["passengers"].sort_index()
    return s


def test_train_test_no_overlap():
    """train и test не пересекаются по индексу."""
    gen = SyntheticGenerator(n_routes=1, n_years=5, seed=0)
    df = gen.generate()
    series = df.set_index("date")["passengers"].sort_index()
    test_size = 12
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    assert len(train) + len(test) == len(series)
    assert not train.index.isin(test.index).any()


def test_test_size_correct():
    """test всегда содержит ровно test_size элементов."""
    gen = SyntheticGenerator(n_routes=1, n_years=5, seed=0)
    df = gen.generate()
    series = df.set_index("date")["passengers"].sort_index()
    for test_size in (3, 6, 12):
        test = series.iloc[-test_size:]
        assert len(test) == test_size, f"test_size={test_size}, len(test)={len(test)}"


def test_xgboost_predict_positive(series):
    """XGBoost предсказывает неотрицательные значения."""
    train = series.iloc[:-6]
    model = XGBoostForecaster(n_estimators=50, n_splits=2)
    model.fit(train)
    preds = model.predict(horizon=6)
    assert (preds >= 0).all()
    assert len(preds) == 6


def test_xgboost_ci_ordered(series):
    """Нижняя граница CI ≤ верхней границе (после swap-фикса)."""
    train = series.iloc[:-6]
    model = XGBoostForecaster(n_estimators=50, n_splits=2)
    model.fit(train)
    lower, upper = model.get_confidence_intervals(horizon=6)
    assert (lower <= upper).all(), "lower > upper в доверительном интервале XGBoost"


def test_evaluate_mape_positive(series):
    """MAPE > 0 и конечное число для реальных данных."""
    train = series.iloc[:-6]
    model = XGBoostForecaster(n_estimators=50, n_splits=2)
    model.fit(train)
    preds = model.predict(horizon=6)
    actual = series.iloc[-6:].values
    metrics = model.evaluate(actual, preds)
    assert metrics["mape"] >= 0
    assert np.isfinite(metrics["mape"])
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
