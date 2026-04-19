"""Common forecast-computation helpers for the reporter."""

import numpy as np

from data.synthetic import SyntheticGenerator
from models.sarima_model import SARIMAForecaster
from models.xgboost_model import XGBoostForecaster


def compute_forecast(route_id: int, model_type: str, horizon: int):
    """Return (series_train, forecast_points, lower, upper, metrics)."""
    gen = SyntheticGenerator(n_routes=5, n_years=7, seed=42)
    df = gen.generate()
    route_code = f"RU-RURAL-{route_id:03d}"
    sub = df[df["route_id"] == route_code]
    if sub.empty:
        raise ValueError(f"No data for route {route_id}")
    series = sub.set_index("date")["passengers"].sort_index()
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:].values.astype(float)

    if model_type == "sarima":
        m = SARIMAForecaster()
    elif model_type == "xgboost":
        m = XGBoostForecaster(n_estimators=200, n_splits=3)
    else:
        raise ValueError(f"Unsupported model {model_type}")

    m.fit(train)
    forecast = np.asarray(m.predict(horizon=horizon))
    lower, upper = m.get_confidence_intervals(horizon=horizon)
    metrics = m.evaluate(test, forecast)
    return {
        "route_code": route_code,
        "history": train,
        "actual": test,
        "forecast": forecast,
        "lower": np.asarray(lower),
        "upper": np.asarray(upper),
        "metrics": metrics,
    }
