"""HTTP client to the FastAPI backend with direct-module fallback.

If the API service is unreachable, calls fall back to direct invocation
of underlying modules so the UI remains functional for demos.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

# Настройка логов для UI: debug → консоль + файл logs/ui.log
_LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "logs"))
os.makedirs(_LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(_LOG_DIR, "ui.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ui.api_client")

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 120


class ApiClient:
    def __init__(self, base_url: str = API_URL, token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _api_up(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=2)
            return r.ok
        except Exception:
            return False

    # ---------- Auth ----------
    def login(self, username: str, password: str) -> Optional[dict]:
        try:
            r = requests.post(
                f"{self.base_url}/api/auth/login",
                data={"username": username, "password": password},
                timeout=TIMEOUT,
            )
            if r.ok:
                data = r.json()
                self.token = data["access_token"]
                return data
        except Exception:
            pass
        # Fallback: local validation against known demo users
        demo = {
            "admin":     ("admin123",     "admin"),
            "planner":   ("planner123",   "planner"),
            "analyst":   ("analyst123",   "analyst"),
            "dispatcher":("dispatch123",  "dispatcher"),
        }
        if username in demo and demo[username][0] == password:
            self.token = f"local-token-{username}"
            return {"access_token": self.token, "role": demo[username][1], "token_type": "bearer"}
        return None

    # ---------- Routes ----------
    def list_routes(self) -> list[dict]:
        if self._api_up():
            r = requests.get(f"{self.base_url}/api/routes/", headers=self._headers(), timeout=TIMEOUT)
            if r.ok:
                return r.json()
        # Fallback
        from api.routes.routes import _DEMO_ROUTES
        return _DEMO_ROUTES

    # ---------- Forecast ----------
    def forecast(self, route_id: int, model_type: str, horizon: int) -> dict:
        logger.info("[forecast] request: route_id=%d model=%s horizon=%d",
                    route_id, model_type, horizon)
        api_up = self._api_up()
        logger.info("[forecast] API up: %s (url=%s)", api_up, self.base_url)

        if api_up:
            r = requests.post(
                f"{self.base_url}/api/forecast/",
                json={"route_id": route_id, "model_type": model_type, "horizon": horizon},
                headers=self._headers(), timeout=TIMEOUT,
            )
            logger.info("[forecast] API response: status=%d", r.status_code)
            if r.ok:
                body = r.json()
                if body.get("points"):
                    pts = body["points"]
                    logger.info(
                        "[forecast] API returned %d points. "
                        "Point range [%.2f..%.2f], Upper range [%.2f..%.2e]",
                        len(pts),
                        min(p["point"] for p in pts), max(p["point"] for p in pts),
                        min(p["upper"] for p in pts), max(p["upper"] for p in pts),
                    )
                return body
            else:
                logger.warning("[forecast] API returned error: %s", r.text[:200])
        # Fallback to direct computation
        logger.info("[forecast] Falling back to direct module computation")
        from reporter.common import compute_forecast
        try:
            data = compute_forecast(route_id, model_type, horizon)
        except ValueError as e:
            logger.error("[forecast] compute_forecast raised: %s", e)
            return {"error": str(e)}
        logger.info(
            "[forecast] Direct computation: point [%.2f..%.2f], upper [%.2f..%.2e]",
            float(data["forecast"].min()), float(data["forecast"].max()),
            float(data["upper"].min()), float(data["upper"].max()),
        )
        points = [{
            "month_offset": i + 1,
            "point": float(round(data["forecast"][i], 2)),
            "lower": float(round(data["lower"][i], 2)),
            "upper": float(round(data["upper"][i], 2)),
        } for i in range(horizon)]
        return {
            "route_id": route_id, "model_type": model_type, "horizon": horizon,
            "points": points,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ---------- Metrics ----------
    def metrics(self, route_id: int, model_type: Optional[str] = None, test_size: int = 12) -> list[dict]:
        if self._api_up():
            params = {"route_id": route_id, "test_size": test_size}
            if model_type:
                params["model_type"] = model_type
            r = requests.get(f"{self.base_url}/api/metrics/", params=params,
                             headers=self._headers(), timeout=TIMEOUT)
            if r.ok:
                return r.json()
        # Fallback
        from reporter.common import compute_forecast
        results = []
        models_to_eval = [model_type] if model_type else ["sarima", "xgboost"]
        for mt in models_to_eval:
            try:
                d = compute_forecast(route_id, mt, test_size)
            except ValueError:
                continue
            m = d["metrics"]
            ss_res = float(np.sum((d["actual"] - d["forecast"]) ** 2))
            ss_tot = float(np.sum((d["actual"] - d["actual"].mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-8 else None
            results.append({
                "route_id": route_id, "model_type": mt,
                "mape": m["mape"], "rmse": m["rmse"], "mae": m["mae"],
                "r_squared": round(r2, 4) if r2 is not None else None,
                "test_period_months": test_size,
            })
        return results

    # ---------- Training ----------
    def train(self, route_id: int, model_type: str, params: Optional[dict] = None) -> dict:
        if self._api_up():
            r = requests.post(
                f"{self.base_url}/api/models/train",
                json={"route_id": route_id, "model_type": model_type, "params": params or {}},
                headers=self._headers(), timeout=TIMEOUT,
            )
            if r.ok:
                return r.json()
        # Fallback: direct training (blocking)
        from reporter.common import compute_forecast
        try:
            compute_forecast(route_id, model_type, 12)
            return {"task_id": "local", "status": "completed", "route_id": route_id,
                    "model_type": model_type, "message": "Модель обучена локально"}
        except Exception as e:
            return {"task_id": "local", "status": "failed", "route_id": route_id,
                    "model_type": model_type, "message": str(e)}

    # ---------- Reports ----------
    def list_reports(self) -> list[dict]:
        if self._api_up():
            r = requests.get(f"{self.base_url}/api/reports/", headers=self._headers(), timeout=TIMEOUT)
            if r.ok:
                return r.json()
        # Fallback: list local files
        from pathlib import Path
        p = Path(__file__).parent.parent / "reports" / "generated"
        p.mkdir(parents=True, exist_ok=True)
        items = []
        for i, f in enumerate(sorted(p.glob("*.pdf")) + sorted(p.glob("*.docx"))):
            items.append({
                "report_id": i + 1, "title": f.stem,
                "created_at": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
                "format": f.suffix.lstrip('.'),
                "size_bytes": f.stat().st_size,
            })
        return items

    def generate_report(self, route_id: int, model_type: str, horizon: int, fmt: str = "pdf") -> dict:
        if self._api_up():
            r = requests.post(
                f"{self.base_url}/api/reports/generate",
                params={"route_id": route_id, "model_type": model_type, "horizon": horizon, "format": fmt},
                headers=self._headers(), timeout=TIMEOUT,
            )
            if r.ok:
                return r.json()
        # Fallback
        from pathlib import Path
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        reports_dir = Path(__file__).parent.parent / "reports" / "generated"
        reports_dir.mkdir(parents=True, exist_ok=True)
        filename = f"forecast_route{route_id}_{model_type}_{ts}.{fmt}"
        filepath = reports_dir / filename
        if fmt == "pdf":
            from reporter.pdf_report import build_pdf_report
            build_pdf_report(str(filepath), route_id=route_id, model_type=model_type, horizon=horizon)
        else:
            from reporter.docx_report import build_docx_report
            build_docx_report(str(filepath), route_id=route_id, model_type=model_type, horizon=horizon)
        return {
            "report_id": filename,
            "file_path": str(filepath),
            "size_bytes": filepath.stat().st_size,
            "status": "generated",
        }


@lru_cache(maxsize=1)
def get_client() -> ApiClient:
    return ApiClient()


# --- Synthetic data access for charts (mock-less, uses real SyntheticGenerator) ---
@lru_cache(maxsize=1)
def get_full_dataset() -> pd.DataFrame:
    from data.synthetic import SyntheticGenerator
    gen = SyntheticGenerator(n_routes=5, n_years=7, seed=42)
    return gen.generate()
