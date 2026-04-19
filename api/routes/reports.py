"""GET /api/reports/{id} — генерация и выдача отчётов (PDF/DOCX)."""

from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from api.auth import get_current_user
from api.schemas import ReportInfo
from reporter.pdf_report import build_pdf_report
from reporter.docx_report import build_docx_report

router = APIRouter()

REPORTS_DIR = Path(__file__).parent.parent.parent / "reports" / "generated"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/", response_model=list[ReportInfo])
def list_reports(_user: dict = Depends(get_current_user)):
    items: list[ReportInfo] = []
    for i, f in enumerate(sorted(REPORTS_DIR.glob("*.pdf")) + sorted(REPORTS_DIR.glob("*.docx"))):
        items.append(ReportInfo(
            report_id=i + 1,
            title=f.stem,
            created_at=datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc),
            format=f.suffix.lstrip('.').lower() if f.suffix.lower() in {'.pdf', '.docx', '.xlsx'} else "pdf",
            size_bytes=f.stat().st_size,
        ))
    return items


@router.post("/generate")
def generate_report(
    route_id: int = Query(..., description="ID маршрута"),
    model_type: str = Query("sarima"),
    horizon: int = Query(12, ge=1, le=24),
    format: str = Query("pdf", pattern="^(pdf|docx)$"),
    _user: dict = Depends(get_current_user),
):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"forecast_route{route_id}_{model_type}_{timestamp}.{format}"
    filepath = REPORTS_DIR / filename

    if format == "pdf":
        build_pdf_report(str(filepath), route_id=route_id, model_type=model_type, horizon=horizon)
    else:
        build_docx_report(str(filepath), route_id=route_id, model_type=model_type, horizon=horizon)

    return {
        "report_id": filename,
        "file_path": str(filepath),
        "size_bytes": filepath.stat().st_size,
        "status": "generated",
    }


@router.get("/{report_id}")
def download_report(report_id: str, _user: dict = Depends(get_current_user)):
    filepath = REPORTS_DIR / report_id
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Отчёт {report_id} не найден")
    media = "application/pdf" if filepath.suffix == ".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return FileResponse(str(filepath), media_type=media, filename=report_id)
