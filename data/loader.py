"""Загрузчики данных из различных источников."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Базовый класс для всех загрузчиков данных."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load(self, **kwargs) -> pd.DataFrame:
        """Загружает данные и возвращает DataFrame с колонками [date, route_id, passengers]."""

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / name

    def _load_cached(self, name: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(name)
        if path.exists():
            logger.info("Загружаем из кэша: %s", path)
            return pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path, parse_dates=["date"])
        return None

    def _save_cache(self, df: pd.DataFrame, name: str) -> None:
        path = self._cache_path(name)
        if name.endswith(".parquet"):
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        logger.info("Сохранено в кэш: %s", path)


class RosstatLoader(BaseLoader):
    """Загружает данные Росстат по пассажирским перевозкам автомобильным транспортом.

    Источник: https://rosstat.gov.ru/statistics/transport
    Форматы: xlsx (форма 65-автотранс) и open-data CSV.
    """

    ROSSTAT_URL = "https://rosstat.gov.ru/storage/mediabank/avto_perevoz.xlsx"

    def load(self, file_path: Optional[Path] = None, **kwargs) -> pd.DataFrame:
        """
        Параметры
        ---------
        file_path : Path, optional
            Путь к локальному файлу Росстат (.xlsx). Если не указан — попытка
            загрузки из кэша, затем скачивание с сайта Росстат.
        """
        cache_name = "rosstat_passengers.csv"
        cached = self._load_cached(cache_name)
        if cached is not None:
            return cached

        if file_path and Path(file_path).exists():
            df = self._parse_rosstat_xlsx(Path(file_path))
        else:
            logger.warning(
                "Файл Росстат не найден локально. Укажите file_path= или скачайте вручную: %s",
                self.ROSSTAT_URL,
            )
            return pd.DataFrame(columns=["date", "region", "passengers"])

        self._save_cache(df, cache_name)
        return df

    def _parse_rosstat_xlsx(self, path: Path) -> pd.DataFrame:
        """Парсит файл форма 65-автотранс."""
        raw = pd.read_excel(path, header=None)
        records = []
        for _, row in raw.iterrows():
            try:
                date = pd.to_datetime(row.iloc[0])
                region = str(row.iloc[1]).strip()
                passengers = float(str(row.iloc[2]).replace(",", ".").replace(" ", ""))
                records.append({"date": date, "region": region, "passengers": passengers})
            except (ValueError, TypeError):
                continue
        return pd.DataFrame(records)


class NTDLoader(BaseLoader):
    """Загружает данные National Transit Database (США).

    Источник: https://www.transit.dot.gov/ntd
    Используется как прокси-данные для отработки алгоритмов.
    """

    def load(self, file_path: Optional[Path] = None, **kwargs) -> pd.DataFrame:
        cache_name = "ntd_monthly.csv"
        cached = self._load_cached(cache_name)
        if cached is not None:
            return cached

        if file_path and Path(file_path).exists():
            df = pd.read_csv(file_path, parse_dates=["Month"])
            df = df.rename(columns={"Month": "date", "UPT": "passengers"})
            df["passengers"] = pd.to_numeric(df["passengers"], errors="coerce")
            df = df.dropna(subset=["passengers"])
            self._save_cache(df, cache_name)
            return df

        logger.warning("NTD файл не указан. Скачайте с https://www.transit.dot.gov/ntd")
        return pd.DataFrame(columns=["date", "route_id", "passengers"])


class CTALoader(BaseLoader):
    """Загружает данные Chicago Transit Authority (CTA Bus Ridership).

    Источник: City of Chicago Open Data Portal.
    """

    def load(self, file_path: Optional[Path] = None, **kwargs) -> pd.DataFrame:
        cache_name = "cta_ridership.csv"
        cached = self._load_cached(cache_name)
        if cached is not None:
            return cached

        if file_path and Path(file_path).exists():
            df = pd.read_csv(file_path)
            df["date"] = pd.to_datetime(df.get("service_date") or df.get("date"))
            df = df.rename(columns={"rides": "passengers", "route": "route_id"})
            df["passengers"] = pd.to_numeric(df["passengers"], errors="coerce")
            df = df.dropna(subset=["date", "passengers"])
            self._save_cache(df, cache_name)
            return df

        logger.warning("CTA файл не указан.")
        return pd.DataFrame(columns=["date", "route_id", "passengers"])


class FileLoader(BaseLoader):
    """Универсальный загрузчик из локального CSV/Excel-файла.

    Ожидаемые колонки: date (или datetime), passengers (или count).
    Дополнительные колонки сохраняются как внешние факторы.
    """

    DATE_ALIASES = ["date", "datetime", "period", "month", "дата"]
    PASSENGERS_ALIASES = ["passengers", "count", "ridership", "passazhiry", "пассажиры"]

    def load(self, file_path: Path, date_col: str = None, passengers_col: str = None, **kwargs) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        if path.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path, **kwargs)
        else:
            df = pd.read_csv(path, **kwargs)

        df.columns = [c.lower().strip() for c in df.columns]

        if date_col is None:
            date_col = next((c for c in df.columns if c in self.DATE_ALIASES), df.columns[0])
        if passengers_col is None:
            passengers_col = next((c for c in df.columns if c in self.PASSENGERS_ALIASES), df.columns[1])

        df = df.rename(columns={date_col: "date", passengers_col: "passengers"})
        df["date"] = pd.to_datetime(df["date"])
        df["passengers"] = pd.to_numeric(df["passengers"], errors="coerce")
        return df.sort_values("date").reset_index(drop=True)
