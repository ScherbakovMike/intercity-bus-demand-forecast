__all__ = [
    "SARIMAForecaster",
    "ProphetForecaster",
    "LSTMForecaster",
    "XGBoostForecaster",
    "ModelComparator",
]


def __getattr__(name):
    if name == "SARIMAForecaster":
        from .sarima_model import SARIMAForecaster
        return SARIMAForecaster
    if name == "ProphetForecaster":
        from .prophet_model import ProphetForecaster
        return ProphetForecaster
    if name == "LSTMForecaster":
        from .lstm_model import LSTMForecaster
        return LSTMForecaster
    if name == "XGBoostForecaster":
        from .xgboost_model import XGBoostForecaster
        return XGBoostForecaster
    if name == "ModelComparator":
        from .comparator import ModelComparator
        return ModelComparator
    raise AttributeError(f"module 'models' has no attribute {name!r}")
