"""Запускает генерацию синтетических данных и сохраняет в data/processed/."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic import SyntheticGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = SyntheticGenerator(n_routes=10, n_years=5, seed=42)
    df = gen.generate(save_path=out_dir / "synthetic_passengers.csv")

    print(f"\nГенерация завершена.")
    print(f"Файл: {out_dir / 'synthetic_passengers.csv'}")
    print(f"Записей: {len(df)}")
    print(f"Маршруты: {df['route_id'].unique().tolist()}")
    print(f"\nПример данных:")
    print(df.groupby("route_id")["passengers"].describe().round(0).to_string())
