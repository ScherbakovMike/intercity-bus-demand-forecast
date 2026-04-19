# Intercity Bus Demand Forecast

> Information system for forecasting passenger demand on intercity bus routes using four machine learning models (SARIMA, Prophet, LSTM, XGBoost) under a unified interface.

**Context.** The project was developed as a bachelor's thesis in Information Systems and Technologies (field 09.03.02) at Moscow Technological Institute, 2026. Designed in the context of FSAU "TsITiS" (Federal State Autonomous Scientific Institution "Center for Information Technologies and Systems") as an example of an applied system for regional transport operators.

---

## Table of Contents

- [Key features](#key-features)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage examples](#usage-examples)
- [Methodology](#methodology)
- [Interpreting the results](#interpreting-the-results)
- [Testing](#testing)
- [Database](#database)
- [Known limitations](#known-limitations)
- [License](#license)

---

## Key features

- **Four forecasting models under a unified `BaseForecaster` interface**: SARIMA, Prophet, LSTM, XGBoost — each exposing `fit()`, `predict()`, `get_confidence_intervals()`, `evaluate()`.
- **Automatic model comparison** by MAE, RMSE, MAPE metrics and best-model selection by minimum MAPE (`ModelComparator`).
- **Synthetic data generator** for rural routes calibrated against Rosstat statistics (Tver Oblast, 2019–2024): +3%/year trend, annual seasonality with peaks in March/September, COVID correction for 2020–2021, noise with CV=15%.
- **Forecast confidence intervals**: analytical for SARIMA, posterior 95% for Prophet (1 000 Monte Carlo simulations), Monte Carlo Dropout for LSTM (50 stochastic passes), quantile regression for XGBoost (three models: 0.025 / 0.500 / 0.975).
- **Walk-Forward Validation** and scikit-learn `TimeSeriesSplit` — strict validation without future-data leakage.
- **Real-data loaders**: Rosstat (monthly bus passenger flow), National Transit Database (USA), CTA Chicago (daily ridership), CSV/XLSX files.
- **PostgreSQL 15 database schema**: 12 entities plus yearly partitioning of `passenger_count` for scalability.
- **Unit tests**: 14 tests (9 for the generator, 5 for the comparator) — all pass in ~2 seconds.

---

## Project structure

```
intercity-bus-demand-forecast/
├── config.py                  # Model hyperparameters, paths, MAPE thresholds
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .gitignore
│
├── data/
│   ├── __init__.py
│   ├── loader.py              # Loaders: Rosstat, NTD, CTA, CSV/XLSX
│   ├── preprocessor.py        # Cleaning, interpolation, Z-score, MinMax
│   └── synthetic.py           # SyntheticGenerator for the rural case
│
├── models/
│   ├── __init__.py
│   ├── base.py                # Abstract BaseForecaster
│   ├── sarima_model.py        # SARIMA/SARIMAX (statsmodels)
│   ├── prophet_model.py       # Prophet (Meta)
│   ├── lstm_model.py          # LSTM with MC Dropout (TensorFlow/Keras)
│   ├── xgboost_model.py       # XGBoost with quantile regression
│   └── comparator.py          # ModelComparator
│
├── visualization/
│   ├── __init__.py
│   └── plotter.py             # Matplotlib and Plotly charts
│
├── db/
│   ├── __init__.py
│   └── schema.sql             # PostgreSQL DDL: 12 tables, indexes, partitions
│
├── scripts/
│   ├── download_data.py       # Download NTD, CTA, Rosstat
│   └── generate_synthetic.py  # Generate synthetic data for experiments
│
└── tests/
    ├── __init__.py
    ├── test_synthetic.py      # 9 generator tests
    └── test_comparator.py     # 5 comparator tests
```

Total volume: ~1 840 lines of Python plus DDL.

---

## Installation

### Environment requirements

- **Python 3.10+** (developed on 3.12)
- **pip ≥ 23** or **poetry ≥ 1.6**
- **Git** (for cloning)
- **RAM:** at least 8 GB; 16 GB recommended when running LSTM
- **PostgreSQL 14+** — optional, only for persistent storage of real data
- **CUDA 12+** — optional, for GPU-accelerated LSTM training

### Step by step

```bash
# 1. Clone the repository
git clone https://github.com/ScherbakovMike/intercity-bus-demand-forecast
cd intercity-bus-demand-forecast

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify the installation
python -m pytest tests/ -v
# Expected: 14 passed in ~2s
```

### Minimal installation

If only SARIMA and XGBoost are needed (without Prophet / LSTM / Streamlit):

```bash
pip install numpy pandas statsmodels xgboost scikit-learn pytest matplotlib
```

---

## Quick start

```python
from data.synthetic import SyntheticGenerator
from models.xgboost_model import XGBoostForecaster

# 1. Generate a synthetic series for a single route (60 months)
gen = SyntheticGenerator(n_routes=1, n_years=5, seed=42)
df = gen.generate()
series = df.set_index("date")["passengers"].sort_index()

# 2. Train/test split and fit XGBoost
train = series.iloc[:-12]
test  = series.iloc[-12:]

model = XGBoostForecaster(n_estimators=200, n_splits=3)
model.fit(train)

# 3. Point forecast with a 95% confidence interval
forecast = model.predict(horizon=12)
lower, upper = model.get_confidence_intervals(horizon=12)

# 4. Quality metrics
metrics = model.evaluate(test.values, forecast)
print(f"MAPE: {metrics['mape']:.2f}%, MAE: {metrics['mae']:.0f}, RMSE: {metrics['rmse']:.0f}")
```

Expected output: `MAPE: ~10–22%, MAE: ~165–434, RMSE: ~218–512` (range depends on seed and route).

---

## Usage examples

### Generate synthetic data

```python
from data.synthetic import SyntheticGenerator

gen = SyntheticGenerator(
    n_routes=5,          # 5 routes
    n_years=7,           # 2019 through 2025
    noise_level=0.15,    # CV = 15%
    seed=42,             # reproducibility
    base_passengers=550, # average monthly base flow
)
df = gen.generate(save_path="data/processed/synthetic.csv")
# 420 rows: date, route_id, passengers, trend, seasonal, holiday, noise
```

### Compare all models

```python
from data.synthetic import SyntheticGenerator
from models.comparator import ModelComparator

gen = SyntheticGenerator(n_routes=5, n_years=7, seed=42)
df = gen.generate()

cmp = ModelComparator(test_size=12, include_lstm=False)  # True if TensorFlow is available
result = cmp.compare(df, route_id="RU-RURAL-001", horizon=12)
print(result)
# DataFrame columns: rank, model, mae, rmse, mape
#   rank  model     mae    rmse   mape
#     1   SARIMA    165    218   10.36
#     2   Prophet   ...    ...   ...
#     3   XGBoost   180    235   10.40

best = cmp.best_model()
forecast = best.predict(horizon=12)
```

### Walk-Forward Validation

```python
import numpy as np
from data.synthetic import SyntheticGenerator
from models.sarima_model import SARIMAForecaster

gen = SyntheticGenerator(n_routes=5, n_years=7, seed=42)
df = gen.generate()

mapes = []
for route_id in df["route_id"].unique():
    s = df[df["route_id"] == route_id].set_index("date")["passengers"].sort_index()
    n = len(s)
    for tr_end in [n-24, n-18, n-12]:       # 3 folds, 6-month step
        train = s.iloc[:tr_end]
        test  = s.iloc[tr_end : tr_end+6].values
        m = SARIMAForecaster()
        m.fit(train)
        pred = m.predict(horizon=6)
        mapes.append(m.evaluate(test, pred)["mape"])

print(f"SARIMA WFV: MAPE = {np.mean(mapes):.2f} ± {np.std(mapes):.2f} % (n={len(mapes)})")
```

### Load real data

```python
from data.loader import RosstatLoader, NTDLoader

# Rosstat (bus passenger flow by Russian federal subject)
ros = RosstatLoader(source="rosstat_bus_by_region_2000_2024.csv")
df_rus = ros.load(region="Tver Oblast")

# NTD (USA transit agencies, monthly)
ntd = NTDLoader(mode="MB")  # Motor Bus
df_ntd = ntd.load(agency_id="30019")  # e.g. Chicago CTA
```

---

## Methodology

### Why four models?

Four models cover all key classes of time-series forecasting methods:

| Class | Model | Strengths | Weaknesses |
|---|---|---|---|
| Statistical | **SARIMA** | Interpretability, correct CIs, works on small samples (36+) | Linear dependencies only, single seasonality |
| Bayesian additive | **Prophet** | Multiple seasonalities, holidays, robust to outliers | Misses nonlinear long-term dependencies |
| Recurrent neural net | **LSTM** | Nonlinearities, long-term dependencies | Needs 500+ observations, low interpretability |
| Ensemble boosting | **XGBoost** | External regressors, feature importance, speed | Requires manual feature engineering |

References: Pereira et al. 2024, Afandizadeh et al. 2024, Stadler et al. 2021, TCRP Report 147.

### LSTM architecture

Input tensor **X ∈ ℝ^{N × L × F}**, where:

- N — number of training samples;
- L = 12 (rolling window length in months);
- F = 1 (baseline implementation: univariate passenger flow series).

Network: LSTM(64, return_sequences=True) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1).

Confidence intervals use Monte Carlo Dropout: 50 stochastic forward passes with active Dropout (`training=True` in Keras) — 2.5th and 97.5th percentiles of the forecast distribution.

### XGBoost features

```python
lags = [1, 3, 6, 12]           # short-term / quarterly / semiannual / annual
ma_windows = [3, 6]            # moving averages
calendar = [month_sin, month_cos, quarter]  # cyclic calendar features
```

For CIs, two additional models are trained with `objective='reg:quantileerror'` and `quantile_alpha ∈ {0.025, 0.975}`. If quantiles cross, a swap fix is applied (`lower[mask], upper[mask] = upper[mask], lower[mask]`).

### Synthetic generator parameters

| Component | Value | Source |
|---|---|---|
| Base level | U[600, 2 400] passengers/month | Rosstat 2022, rural segment |
| Trend | +3%/year | Rosstat 2019–2024 |
| Seasonality | ±25%, peaks in March/September | TCRP Report 147, Rosstat |
| COVID coefficient | 0.30 in Apr 2020 → 0.85 by Oct 2021 | NTD Report 2021 |
| Noise | CV = 15% (Gaussian) | TCRP 147 for small rural routes |

---

## Interpreting the results

### Metrics

- **MAE** (Mean Absolute Error) — in passenger units; scale-dependent.
- **RMSE** (Root Mean Squared Error) — in passenger units; quadratically penalizes large errors.
- **MAPE** (Mean Absolute Percentage Error) — percentage; scale-independent, the primary criterion.

### Quality thresholds (from `config.py`)

| MAPE | Grade | Action |
|---|---|---|
| < 5% | Excellent | Production-ready |
| 5–15% | Good | Suitable for planning |
| 15–25% | Acceptable | Requires monitoring, possibly calibration |
| > 25% | Warning | **Log warning**: expand the training sample |
| > 50% | Critical | Model unusable, revise the method |

**Observed values** on the synthetic rural dataset (5 routes × 84 months, H = 12):

- SARIMA: MAPE = 17.05%
- XGBoost: MAPE = 18.47%

### Confidence interval coverage

For a nominal 95% CI, approximately 95% of test-set points should fall inside `[lower, upper]`.

| Model | Observed coverage | Interpretation |
|---|---|---|
| SARIMA | 94.4% | Correct coverage — CIs are statistically well-calibrated |
| XGBoost | 66.7% | Underestimated — quantile regression on 72 observations misses the tails; conformal prediction recommended |

### Which model to choose?

- **Small samples (< 100 observations)** — SARIMA. The parametric structure gives correct CIs.
- **Series with multiple seasonalities and holidays** — Prophet.
- **Large samples (500+) with nonlinear dependencies** — LSTM.
- **With external regressors (weather, fares, events)** — XGBoost.

Automatic selection by MAPE is performed by `ModelComparator.best_model()`.

---

## Testing

```bash
python -m pytest tests/ -v
```

Expected output:

```
tests/test_comparator.py::test_train_test_no_overlap        PASSED
tests/test_comparator.py::test_test_size_correct            PASSED
tests/test_comparator.py::test_xgboost_predict_positive     PASSED
tests/test_comparator.py::test_xgboost_ci_ordered           PASSED
tests/test_comparator.py::test_evaluate_mape_positive       PASSED
tests/test_synthetic.py::test_record_count                  PASSED
tests/test_synthetic.py::test_columns                       PASSED
tests/test_synthetic.py::test_daily_range                   PASSED
tests/test_synthetic.py::test_seasonal_peak_march           PASSED
tests/test_synthetic.py::test_seasonal_trough_january       PASSED
tests/test_synthetic.py::test_covid_april_2020              PASSED
tests/test_synthetic.py::test_noise_cv                      PASSED
tests/test_synthetic.py::test_no_negative_passengers        PASSED
tests/test_synthetic.py::test_route_ids                     PASSED

======================== 14 passed in 1.80s =========================
```

### What the tests check

**Generator (`test_synthetic.py`, 9 tests):**

- Record count = `n_routes × n_years × 12`
- All required columns present
- Daily passenger range 10–60 passengers/day
- Seasonal peaks in March / September
- Minimum in January / July
- COVID coefficient April 2020 ∈ [0.20; 0.45]
- Noise CV in [0.08; 0.25]
- Non-negative passengers
- `route_id` prefix correctness

**Comparator (`test_comparator.py`, 5 tests):**

- Train and test do not overlap by index
- Test sample size is correct (for `test_size ∈ {3, 6, 12}`)
- XGBoost forecast is non-negative
- `lower ≤ upper` in CIs (swap-fix check)
- MAPE/MAE/RMSE are non-negative and finite

---

## Database

For a full deployment with real data:

```bash
# Create PostgreSQL database
createdb passenger_forecast

# Apply schema (12 tables with indexes and partitions)
psql -d passenger_forecast -f db/schema.sql

# Define connection parameters in .env
echo "DB_HOST=localhost"          >> .env
echo "DB_PORT=5432"               >> .env
echo "DB_NAME=passenger_forecast" >> .env
echo "DB_USER=postgres"           >> .env
echo "DB_PASSWORD=secret"         >> .env
```

Schema contents:

- `route`, `region`, `station`, `route_station` — route network
- `trip`, `passenger_count` — individual trips and passenger counts
- `external_factor`, `trip_factor` — weather, holidays, events
- `forecast_model`, `forecast`, `model_metrics` — forecasts and quality
- `app_user` — users with roles (planner/dispatcher/analyst/admin)
- `report` — generated analytical reports

`passenger_count` is partitioned by year for linear query scaling on multi-year data.

---

## Known limitations

1. **Experimental validation** was performed only for SARIMA and XGBoost. Prophet and LSTM architectures are fully implemented but require installed `prophet` and `tensorflow-cpu` libraries to run.
2. **LSTM uses F = 1** (univariate rolling window). Multivariate extension (F = 15 from the full feature space) is planned; requires 500+ training observations.
3. **Training and testing on synthetic data.** Metrics validate correctness of algorithm implementation but do not confirm applicability to real routes. A pilot project with a real transport operator is required before production deployment.
4. **REST API and UI** are specified at the architectural level only; server-side code (FastAPI + Streamlit) is out of scope and is deferred to the production deployment stage.
5. **XGBoost CI coverage is underestimated** (66.7% vs nominal 95%) when training sample size is below 100. Expanding the sample to 120+ or applying post-hoc calibration (conformal prediction) is recommended.

---

## License

MIT License — see the `LICENSE` file (if present).

---

## Citation

If you use this project in academic work:

```
Shcherbakov M. N. Forecasting passenger flow on intercity routes
using FSAU "TsITiS" as an example : bachelor's thesis /
M. N. Shcherbakov ; supervisor Yu. S. Buzykova. — Moscow : MTI, 2026.
```

---

## Acknowledgements

Thesis supervisor: **Yulia S. Buzykova** (MTI).

Theoretical foundations:

- Box G. E. P., Jenkins G. M. et al. *Time Series Analysis.* — Wiley, 2015.
- Goodfellow I., Bengio Y., Courville A. *Deep Learning.* — MIT Press, 2016.
- Hastie T., Tibshirani R., Friedman J. *The Elements of Statistical Learning.* — Springer, 2009.
- Pereira F. C. et al. *Machine Learning for Public Transportation Demand Prediction.* — EAAI, 2024.
- TCRP Report 147. *Toolkit for Estimating Demand for Rural Intercity Bus Services.* — TRB, 2012.
