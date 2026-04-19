"""Reporter module: PDF (ReportLab) and DOCX (python-docx) generation."""

from .pdf_report import build_pdf_report
from .docx_report import build_docx_report

__all__ = ["build_pdf_report", "build_docx_report"]
