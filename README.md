# Система прогнозирования пассажиропотока на междугородних рейсах

Информационная система для краткосрочного и среднесрочного прогнозирования пассажиропотока
на автобусных маршрутах, реализованная в рамках ВКР по направлению 09.03.02.

## Структура проекта

```
src/
├── config.py                  # Конфигурация (пути, гиперпараметры, пороги)
├── requirements.txt
├── data/
│   ├── loader.py              # Загрузчики данных (Росстат, NTD, CTA, файлы)
│   ├── preprocessor.py        # Очистка, признаки, нормализация
│   └── synthetic.py           # Генератор синтетических данных (сельский кейс)
├── models/
│   ├── base.py                # Абстрактный класс BaseForecaster
│   ├── sarima_model.py        # SARIMA/SARIMAX (statsmodels)
│   ├── prophet_model.py       # Prophet (Meta)
│   ├── lstm_model.py          # LSTM с MC Dropout (TensorFlow/Keras)
│   ├── xgboost_model.py       # XGBoost с TimeSeriesSplit
│   └── comparator.py          # Сравнение моделей по метрикам
├── visualization/
│   └── plotter.py             # Графики (matplotlib + plotly)
├── db/
│   └── schema.sql             # DDL-схема PostgreSQL (12 таблиц)
└── scripts/
    ├── download_data.py       # Загрузка открытых датасетов
    └── generate_synthetic.py  # Запуск генерации синтетических данных
```

## Быстрый старт

```bash
# Установка зависимостей
pip install -r requirements.txt

# Генерация синтетических данных для демонстрации
python scripts/generate_synthetic.py

# Запуск сравнения моделей
python -c "
from data.synthetic import SyntheticGenerator
from models.comparator import ModelComparator

gen = SyntheticGenerator()
df = gen.generate()
comp = ModelComparator()
results = comp.compare(df, target_col='passengers', horizon=3)
print(results)
"
```

## Модели

| Модель   | Класс               | Подходит для                        |
|----------|---------------------|-------------------------------------|
| SARIMA   | SARIMAForecaster    | Стационарные ряды с сезонностью     |
| Prophet  | ProphetForecaster   | Ряды с праздниками и трендом        |
| LSTM     | LSTMForecaster      | Нелинейные зависимости, длинные ряды|
| XGBoost  | XGBoostForecaster   | Ансамблевый метод, много признаков  |

## Требования

- Python 3.10+
- PostgreSQL 14+ (для полной функциональности БД)
- Минимальная длина ряда для SARIMA/Prophet: 24 месяца
- Для LSTM: рекомендуется GPU (CUDA 12+)
