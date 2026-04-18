"""Скрипт для загрузки открытых датасетов (NTD, CTA, Росстат).

Запуск:
    python scripts/download_data.py --source all
    python scripts/download_data.py --source ntd
    python scripts/download_data.py --source cta
"""

import argparse
import logging
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "ntd": {
        "description": "National Transit Database — Monthly Ridership (NTD)",
        "url": "https://www.transit.dot.gov/sites/fta.dot.gov/files/2024-01/January%202024%20Adjusted%20Database.xlsx",
        "filename": "ntd_monthly_ridership.xlsx",
        "note": "Прокси-данные США для отладки алгоритмов (см. processed/10_data_sources.md)",
    },
    "cta": {
        "description": "Chicago Transit Authority — Bus Ridership by Route Daily Totals",
        "url": "https://data.cityofchicago.org/api/views/jyb9-n7fm/rows.csv?accessType=DOWNLOAD",
        "filename": "cta_ridership_daily.csv",
        "note": "Открытые данные портала data.cityofchicago.org",
    },
}


def download_file(url: str, dest: Path, timeout: int = 60) -> bool:
    try:
        logger.info("Скачиваем: %s", url)
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        size_mb = dest.stat().st_size / 1024 / 1024
        logger.info("Сохранено: %s (%.1f MB)", dest, size_mb)
        return True
    except requests.RequestException as e:
        logger.error("Ошибка загрузки %s: %s", url, e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Загрузка открытых датасетов")
    parser.add_argument(
        "--source",
        choices=list(SOURCES.keys()) + ["all"],
        default="all",
        help="Источник данных для загрузки",
    )
    args = parser.parse_args()

    targets = list(SOURCES.keys()) if args.source == "all" else [args.source]

    for key in targets:
        info = SOURCES[key]
        dest = RAW_DIR / info["filename"]
        print(f"\n{'='*60}")
        print(f"Источник: {info['description']}")
        print(f"Примечание: {info['note']}")
        print(f"Файл: {dest}")
        if dest.exists():
            logger.info("Файл уже существует, пропускаем.")
            continue
        success = download_file(info["url"], dest)
        if not success:
            logger.warning("Не удалось скачать %s. Проверьте подключение.", key)

    print("\nЗавершено.")


if __name__ == "__main__":
    main()
