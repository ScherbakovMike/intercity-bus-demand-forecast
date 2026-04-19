"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


ModelType = Literal["sarima", "prophet", "lstm", "xgboost"]
RouteStatus = Literal["active", "suspended", "archived"]


# ---------- Routes ----------

class RouteOut(BaseModel):
    route_id: int
    name: str
    route_code: Optional[str] = None
    origin_id: Optional[int] = None
    destination_id: Optional[int] = None
    distance_km: Optional[float] = None
    base_fare: Optional[float] = None
    status: RouteStatus = "active"


# ---------- Forecast ----------

class ForecastRequest(BaseModel):
    route_id: int = Field(..., description="ID маршрута")
    model_type: ModelType = Field(..., description="Тип модели")
    horizon: int = Field(12, ge=1, le=24, description="Горизонт прогноза в месяцах")


class ForecastPoint(BaseModel):
    month_offset: int
    point: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    route_id: int
    model_type: ModelType
    horizon: int
    points: list[ForecastPoint]
    generated_at: datetime


# ---------- Metrics ----------

class MetricsOut(BaseModel):
    route_id: int
    model_type: ModelType
    mape: float
    rmse: float
    mae: float
    r_squared: Optional[float] = None
    test_period_months: int


# ---------- Training ----------

class TrainingRequest(BaseModel):
    route_id: int
    model_type: ModelType
    params: dict = Field(default_factory=dict)


class TrainingResponse(BaseModel):
    task_id: str
    status: Literal["queued", "running", "completed", "failed"]
    route_id: int
    model_type: ModelType
    message: str


# ---------- Reports ----------

class ReportInfo(BaseModel):
    report_id: int
    title: str
    created_at: datetime
    format: Literal["pdf", "docx", "xlsx"]
    size_bytes: int


# ---------- Auth ----------

class UserCreate(BaseModel):
    username: str
    password: str
    role: Literal["planner", "dispatcher", "analyst", "admin"] = "analyst"
    full_name: Optional[str] = None
    email: Optional[str] = None


class Token(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    role: str
