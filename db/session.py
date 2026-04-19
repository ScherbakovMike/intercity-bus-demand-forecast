"""Database session management."""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


def _build_db_url() -> str:
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "passenger_forecast")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


engine = create_engine(_build_db_url(), pool_size=5, max_overflow=10, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """FastAPI dependency. Yields a SQLAlchemy session that closes after request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
