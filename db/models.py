"""SQLAlchemy ORM models for PostgreSQL — mirrors db/schema.sql (13 entities)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, DateTime, ForeignKey, Integer,
    Numeric, String, Text, UniqueConstraint, text, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Region(Base):
    __tablename__ = 'region'
    region_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    okato_code: Mapped[Optional[str]] = mapped_column(String(20))
    population: Mapped[Optional[int]] = mapped_column(Integer)
    federal_district: Mapped[Optional[str]] = mapped_column(String(100))
    area_km2: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Station(Base):
    __tablename__ = 'station'
    station_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    city: Mapped[Optional[str]] = mapped_column(String(100))
    region_id: Mapped[int] = mapped_column(ForeignKey('region.region_id'))
    latitude: Mapped[Optional[Decimal]] = mapped_column(Numeric(9, 6))
    longitude: Mapped[Optional[Decimal]] = mapped_column(Numeric(9, 6))
    type: Mapped[str] = mapped_column(String(20), default='stop')
    __table_args__ = (CheckConstraint("type IN ('terminal', 'stop')"),)


class Route(Base):
    __tablename__ = 'route'
    route_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    route_code: Mapped[Optional[str]] = mapped_column(String(50), unique=True)
    origin_id: Mapped[Optional[int]] = mapped_column(ForeignKey('region.region_id'))
    destination_id: Mapped[Optional[int]] = mapped_column(ForeignKey('region.region_id'))
    distance_km: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2))
    base_fare: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))
    status: Mapped[str] = mapped_column(String(20), default='active')
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    __table_args__ = (CheckConstraint("status IN ('active', 'suspended', 'archived')"),)


class RouteStation(Base):
    __tablename__ = 'route_station'
    route_station_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    route_id: Mapped[int] = mapped_column(ForeignKey('route.route_id', ondelete='CASCADE'))
    station_id: Mapped[int] = mapped_column(ForeignKey('station.station_id'))
    stop_sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    arrival_offset_min: Mapped[Optional[int]] = mapped_column(Integer)
    __table_args__ = (UniqueConstraint('route_id', 'stop_sequence'),)


class Trip(Base):
    __tablename__ = 'trip'
    trip_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    route_id: Mapped[int] = mapped_column(ForeignKey('route.route_id'))
    departure_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    arrival_datetime: Mapped[Optional[datetime]] = mapped_column(DateTime)
    day_type: Mapped[Optional[str]] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default='planned')
    __table_args__ = (
        CheckConstraint("day_type IN ('weekday', 'saturday', 'sunday_holiday')"),
        CheckConstraint("status IN ('completed', 'cancelled', 'planned')"),
    )


class PassengerCount(Base):
    __tablename__ = 'passenger_count'
    count_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trip_id: Mapped[int] = mapped_column(ForeignKey('trip.trip_id'), unique=True)
    passenger_count: Mapped[int] = mapped_column(Integer, nullable=False)
    data_type: Mapped[str] = mapped_column(String(20), default='actual')
    data_source: Mapped[Optional[str]] = mapped_column(String(30))
    record_date: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    load_factor: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 2))
    __table_args__ = (
        CheckConstraint("data_type IN ('actual', 'synthetic', 'estimated')"),
        CheckConstraint("data_source IN ('rosstat', 'ntd', 'cta', 'synthetic', 'manual')"),
        CheckConstraint("passenger_count >= 0"),
    )


class ExternalFactor(Base):
    __tablename__ = 'external_factor'
    factor_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    factor_type: Mapped[str] = mapped_column(String(30), nullable=False)
    factor_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    region_id: Mapped[Optional[int]] = mapped_column(ForeignKey('region.region_id'))
    value: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4))
    description: Mapped[Optional[str]] = mapped_column(Text)
    __table_args__ = (
        CheckConstraint("factor_type IN ('weather', 'holiday', 'event', 'tariff')"),
    )


class TripFactor(Base):
    __tablename__ = 'trip_factor'
    tf_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trip_id: Mapped[int] = mapped_column(ForeignKey('trip.trip_id', ondelete='CASCADE'))
    factor_id: Mapped[int] = mapped_column(ForeignKey('external_factor.factor_id'))
    impact_weight: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    __table_args__ = (
        UniqueConstraint('trip_id', 'factor_id'),
        CheckConstraint("impact_weight BETWEEN 0 AND 1"),
    )


class ForecastModel(Base):
    __tablename__ = 'forecast_model'
    model_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(String(20), nullable=False)
    version: Mapped[str] = mapped_column(String(20), default='1.0')
    training_date: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    route_id: Mapped[Optional[int]] = mapped_column(ForeignKey('route.route_id'))
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSONB)
    model_path: Mapped[Optional[str]] = mapped_column(String(500))
    __table_args__ = (CheckConstraint("model_type IN ('sarima', 'prophet', 'lstm', 'xgboost')"),)


class Forecast(Base):
    __tablename__ = 'forecast'
    forecast_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(ForeignKey('forecast_model.model_id'))
    route_id: Mapped[int] = mapped_column(ForeignKey('route.route_id'))
    forecast_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    horizon_months: Mapped[Optional[int]] = mapped_column(Integer)
    forecast_value: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    lower_bound: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    upper_bound: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ModelMetrics(Base):
    __tablename__ = 'model_metrics'
    metric_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(ForeignKey('forecast_model.model_id', ondelete='CASCADE'))
    route_id: Mapped[Optional[int]] = mapped_column(ForeignKey('route.route_id'))
    mape: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))
    rmse: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))
    mae: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))
    r_squared: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    test_period_start: Mapped[Optional[datetime]] = mapped_column(DateTime)
    test_period_end: Mapped[Optional[datetime]] = mapped_column(DateTime)


class AppUser(Base):
    __tablename__ = 'app_user'
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default='analyst')
    full_name: Mapped[Optional[str]] = mapped_column(String(150))
    email: Mapped[Optional[str]] = mapped_column(String(150))
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    __table_args__ = (CheckConstraint("role IN ('planner', 'dispatcher', 'analyst', 'admin')"),)


class Report(Base):
    __tablename__ = 'report'
    report_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(250), nullable=False)
    created_by: Mapped[int] = mapped_column(ForeignKey('app_user.user_id'))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    format: Mapped[str] = mapped_column(String(10), default='pdf')
    file_path: Mapped[Optional[str]] = mapped_column(String(500))
    period_start: Mapped[Optional[datetime]] = mapped_column(DateTime)
    period_end: Mapped[Optional[datetime]] = mapped_column(DateTime)
    __table_args__ = (CheckConstraint("format IN ('pdf', 'docx', 'xlsx')"),)
