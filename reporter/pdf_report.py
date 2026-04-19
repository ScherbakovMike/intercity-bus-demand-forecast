"""PDF report generator using ReportLab."""

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)

from reporter.common import compute_forecast


def _register_cyrillic_font() -> str:
    """Try to register a Cyrillic-capable font. Returns font name."""
    candidates = [
        ("DejaVuSans", r"C:\Windows\Fonts\DejaVuSans.ttf"),
        ("Arial",      r"C:\Windows\Fonts\arial.ttf"),
        ("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ("DejaVuSans", "/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for name, path in candidates:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception:
                continue
    return "Helvetica"  # fallback; non-Cyrillic


def build_pdf_report(filepath: str, route_id: int, model_type: str, horizon: int) -> str:
    data = compute_forecast(route_id, model_type, horizon)
    font = _register_cyrillic_font()

    doc = SimpleDocTemplate(
        filepath, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
        title=f"Forecast report {data['route_code']}",
        author="Intercity Bus Demand Forecast",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"], fontName=font, fontSize=16, alignment=1)
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontName=font, fontSize=12, textColor=colors.HexColor("#1a5276"))
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontName=font, fontSize=10, leading=14)
    small = ParagraphStyle(
        "Small", parent=styles["BodyText"], fontName=font, fontSize=8, textColor=colors.grey)

    elems = []
    elems.append(Paragraph("Отчёт о прогнозе пассажиропотока", title_style))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(f"Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M')}", small))
    elems.append(Spacer(1, 12))

    elems.append(Paragraph("1. Параметры прогноза", h2))
    params_table = Table([
        ["Маршрут", data["route_code"]],
        ["Модель", model_type.upper()],
        ["Горизонт, мес.", str(horizon)],
        ["Обучающих наблюдений", str(len(data["history"]))],
    ], colWidths=[60*mm, 90*mm])
    params_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ecf0f1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elems.append(params_table)
    elems.append(Spacer(1, 12))

    elems.append(Paragraph("2. Метрики качества на тестовой выборке", h2))
    metrics = data["metrics"]
    metrics_table = Table([
        ["Метрика", "Значение"],
        ["MAPE, %",    f"{metrics['mape']:.2f}"],
        ["MAE, пасс.", f"{metrics['mae']:.0f}"],
        ["RMSE, пасс.",f"{metrics['rmse']:.0f}"],
    ], colWidths=[60*mm, 90*mm])
    metrics_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ]))
    elems.append(metrics_table)
    elems.append(Spacer(1, 12))

    elems.append(Paragraph("3. Прогнозные значения с 95% ДИ", h2))
    forecast_rows = [["Месяц", "Прогноз", "Нижняя ДИ", "Верхняя ДИ", "Факт"]]
    for i, (p, l, u, a) in enumerate(zip(data["forecast"], data["lower"], data["upper"], data["actual"])):
        forecast_rows.append([
            f"+{i+1}", f"{p:,.0f}", f"{l:,.0f}", f"{u:,.0f}", f"{a:,.0f}",
        ])
    forecast_table = Table(forecast_rows, colWidths=[25*mm, 35*mm, 35*mm, 35*mm, 25*mm])
    forecast_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    elems.append(forecast_table)

    elems.append(Spacer(1, 18))
    elems.append(Paragraph(
        "Отчёт сформирован автоматически информационной системой прогнозирования "
        "пассажиропотока на междугородних автобусных рейсах (МТИ, 2026).", small))

    doc.build(elems)
    return filepath
