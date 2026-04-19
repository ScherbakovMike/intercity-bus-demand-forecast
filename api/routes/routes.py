"""GET /api/routes — список маршрутов (из таблицы 9 ВКР)."""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.auth import get_current_user
from api.schemas import RouteOut, RouteStatus

router = APIRouter()

# Демо-данные (5 сельских маршрутов Тверской обл., согласованы с SyntheticGenerator)
_DEMO_ROUTES: list[dict] = [
    {"route_id": 1, "name": "Тверь – Торжок",     "route_code": "RU-RURAL-001", "origin_id": 1, "destination_id": 2, "distance_km": 65.0,  "base_fare": 180.0, "status": "active"},
    {"route_id": 2, "name": "Тверь – Ржев",       "route_code": "RU-RURAL-002", "origin_id": 1, "destination_id": 3, "distance_km": 120.0, "base_fare": 320.0, "status": "active"},
    {"route_id": 3, "name": "Тверь – Бежецк",     "route_code": "RU-RURAL-003", "origin_id": 1, "destination_id": 4, "distance_km": 130.0, "base_fare": 350.0, "status": "active"},
    {"route_id": 4, "name": "Тверь – Весьегонск", "route_code": "RU-RURAL-004", "origin_id": 1, "destination_id": 5, "distance_km": 220.0, "base_fare": 550.0, "status": "active"},
    {"route_id": 5, "name": "Москва – Тверь",     "route_code": "RU-INTER-001", "origin_id": 6, "destination_id": 1, "distance_km": 170.0, "base_fare": 450.0, "status": "active"},
]


@router.get("/", response_model=list[RouteOut])
def list_routes(
    region_id: Optional[int] = Query(None, description="Фильтр по ID региона (origin или destination)"),
    status: Optional[RouteStatus] = Query(None),
    _user: dict = Depends(get_current_user),
):
    rows = _DEMO_ROUTES
    if region_id is not None:
        rows = [r for r in rows if r["origin_id"] == region_id or r["destination_id"] == region_id]
    if status is not None:
        rows = [r for r in rows if r["status"] == status]
    return rows


@router.get("/{route_id}", response_model=RouteOut)
def get_route(route_id: int, _user: dict = Depends(get_current_user)):
    for r in _DEMO_ROUTES:
        if r["route_id"] == route_id:
            return r
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail=f"Маршрут {route_id} не найден")
