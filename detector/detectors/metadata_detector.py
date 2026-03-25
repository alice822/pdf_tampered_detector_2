"""
Metadata Detector  (v3 — reduced false positives)
--------------------------------------------------
Only flags things that are genuinely suspicious:
  - ONLINE EDITOR tools (ilovepdf, smallpdf, sejda, pdf24) — HIGH confidence tamper signal
  - Programmatic PDF libraries (reportlab, pypdf, fpdf) — LOW, informational only
  - Future/impossible dates — HIGH
  - Creation after modification — HIGH
  - XMP vs Info title mismatch — MED
  - Missing producer on a signed doc — LOW

NOT flagged as suspicious:
  - Ghostscript (used by printers, legitimate)
  - LibreOffice (legitimate office suite)
  - Microsoft Word / Adobe Acrobat (legitimate)
  - Standard timezone-aware dates
  - ASCII85 filter (normal in any PDF with images/fonts)
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
from typing import List
from ..engine import Finding

# ── Online editor tools — strong tampering signal ─────────────────────────
ONLINE_EDITOR_PRODUCERS = [
    (r"ilovepdf",       "iLovePDF"),
    (r"smallpdf",       "Smallpdf"),
    (r"sejda",          "Sejda"),
    (r"pdf24",          "PDF24"),
    (r"pdfescaper",     "PDFEscaper"),
    (r"pdf\.io\b",      "PDF.io"),
    (r"pdfresize",      "PDFResize"),
    (r"compress\s*pdf", "CompressPDF tool"),
    (r"split\s*pdf",    "SplitPDF tool"),
    (r"merge\s*pdf",    "MergePDF tool"),
]

# ── Programmatic libraries — LOW severity, informational ─────────────────
PROGRAMMATIC_PRODUCERS = [
    (r"reportlab",      "ReportLab"),
    (r"\bfpdf\b",       "FPDF"),
    (r"\bpypdf\b",      "pyPDF"),
    (r"\bpikepdf\b",    "pikePDF"),
    (r"wkhtmltopdf",    "wkhtmltopdf"),
    (r"pdfkit\b",       "PDFKit"),
    (r"weasyprint",     "WeasyPrint"),
    (r"prince\s*xml",   "PrinceXML"),
]

# ── Known AI/automation tool markers in raw bytes ─────────────────────────
TOOL_BYTE_MARKERS = [b"iLovePDF", b"Smallpdf", b"SEJDA", b"PDF24"]


def detect(reader, raw: bytes) -> List[Finding]:
    findings: List[Finding] = []
    meta = reader.metadata or {}

    _check_producer(meta, findings)
    _check_dates(meta, findings)
    _check_xmp_vs_info(reader, findings)
    _check_tool_artifacts(raw, findings)

    return findings


def _check_producer(meta: dict, findings: list):
    producer = str(meta.get("/Producer", "") or "").strip()

    if not producer:
        # Only flag missing producer if there are other suspicious signals
        # (don't add findings here — let corroboration handle it)
        return

    prod_lower = producer.lower()

    # Strong signal: online editor tool
    for pattern, name in ONLINE_EDITOR_PRODUCERS:
        if re.search(pattern, prod_lower):
            findings.append(Finding(
                category="Metadata Inconsistency",
                description=(
                    f"Producer field identifies an online PDF editor: '{producer[:80]}'. "
                    f"Online tools like {name} are commonly used to modify/replace content "
                    "in existing documents without leaving other traces."
                ),
                severity=0.55,
                evidence={"producer": producer[:80]},
            ))
            return

    # Weak signal: programmatic library
    for pattern, name in PROGRAMMATIC_PRODUCERS:
        if re.search(pattern, prod_lower):
            findings.append(Finding(
                category="Metadata Inconsistency",
                description=(
                    f"Producer '{producer[:60]}' is a programmatic PDF generation library ({name}). "
                    "This is normal for auto-generated documents but unusual for scanned originals."
                ),
                severity=0.20,   # very low — informational only
                evidence={"producer": producer[:60]},
            ))
            return


def _check_dates(meta: dict, findings: list):
    now = datetime.now(tz=timezone.utc)
    parsed_dates = {}

    for field_key, label in [("/CreationDate", "creation"), ("/ModDate", "modification")]:
        raw_date = str(meta.get(field_key, "") or "")
        if not raw_date:
            continue
        parsed = _parse_pdf_date(raw_date)
        if parsed is None:
            # Only flag truly unparseable dates (not just unusual formats)
            if len(raw_date) > 4 and not re.match(r'^D:\d{4}', raw_date):
                findings.append(Finding(
                    category="Metadata Inconsistency",
                    description=f"Completely malformed {label} date: '{raw_date[:40]}' — not a valid PDF date format.",
                    severity=0.30,
                    evidence={"date_field": field_key, "value": raw_date},
                ))
        else:
            parsed_dates[field_key] = parsed
            if parsed.year > now.year + 1:
                findings.append(Finding(
                    category="Metadata Inconsistency",
                    description=f"Future {label} date ({parsed.date()}) — year {parsed.year} is in the future. Metadata was likely forged.",
                    severity=0.70,
                    evidence={"date_field": field_key, "parsed": str(parsed.date())},
                ))
            elif parsed.year < 1993:
                findings.append(Finding(
                    category="Metadata Inconsistency",
                    description=f"Implausible {label} date ({parsed.date()}) — PDF format didn't exist before 1993.",
                    severity=0.55,
                    evidence={"date_field": field_key},
                ))

    # Modification before creation = forged
    c = parsed_dates.get("/CreationDate")
    m = parsed_dates.get("/ModDate")
    if c and m and m < c:
        findings.append(Finding(
            category="Metadata Inconsistency",
            description=f"Modification date ({m.date()}) is earlier than creation date ({c.date()}). This is impossible — metadata was manually altered.",
            severity=0.70,
        ))


def _check_xmp_vs_info(reader, findings: list):
    try:
        xmp = reader.xmp_metadata
        if xmp is None:
            return
        info = reader.metadata or {}
        xmp_title = _xmp_get(xmp, "dc:title") or _xmp_get(xmp, "title")
        info_title = str(info.get("/Title", "") or "")
        if xmp_title and info_title and xmp_title.strip() != info_title.strip():
            findings.append(Finding(
                category="Metadata Inconsistency",
                description=f"XMP title '{xmp_title[:50]}' differs from Info dict title '{info_title[:50]}'. Document metadata was edited after creation.",
                severity=0.55,
            ))
    except Exception:
        pass


def _check_tool_artifacts(raw: bytes, findings: list):
    sample = raw[:4096] + raw[-2048:]
    for marker in TOOL_BYTE_MARKERS:
        if marker.lower() in sample.lower():
            findings.append(Finding(
                category="Metadata Inconsistency",
                description=f"Online editor artifact '{marker.decode()}' found in file header/trailer bytes. Document was processed by this tool.",
                severity=0.45,
                evidence={"marker": marker.decode()},
            ))


def _parse_pdf_date(s: str):
    s = s.strip().strip("'")
    if s.startswith("D:"):
        s = s[2:]
    s_clean = re.sub(r"[Z+\-]\d{2}'?\d{0,2}'?$", "", s).strip().strip("'")
    for fmt, length in [("%Y%m%d%H%M%S", 14), ("%Y%m%d%H%M", 12), ("%Y%m%d", 8)]:
        try:
            return datetime.strptime(s_clean[:length], fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return None


def _xmp_get(xmp, tag: str):
    try:
        val = xmp.custom_properties.get(tag)
        return str(val) if val else None
    except Exception:
        return None