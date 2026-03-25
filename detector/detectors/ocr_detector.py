"""
OCR Detector  (v2)
------------------
Runs on ALL pages (not just scan pages) to detect:
  - Low OCR confidence regions (blurred/replaced text)
  - Text layer vs visible content divergence (searchable PDFs)
  - Embedded text that contradicts the visible image
"""
from __future__ import annotations
import io
from typing import List
from ..engine import Finding

try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

OCR_MIN_CONFIDENCE = 40
MAX_OCR_PAGES      = 6


def detect(reader) -> List[Finding]:
    findings: List[Finding] = []
    if not OCR_AVAILABLE:
        return findings

    try:
        import pypdf
        buf = io.BytesIO()
        writer = pypdf.PdfWriter()
        total = min(len(reader.pages), MAX_OCR_PAGES)
        for i in range(total):
            writer.add_page(reader.pages[i])
        writer.write(buf)

        images = convert_from_bytes(buf.getvalue(), dpi=150)

        for i, img in enumerate(images):
            page_num = i + 1
            try:
                _analyze_ocr_page(img, page_num, reader.pages[i], findings)
            except Exception:
                pass
    except Exception:
        pass

    return findings


def _analyze_ocr_page(img, page_num: int, page, findings: list):
    # Confidence analysis
    try:
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang="eng")
        confidences = [int(c) for c in ocr_data["conf"] if str(c).lstrip("-").isdigit() and int(c) > 0]
        if confidences:
            low_conf = sum(1 for c in confidences if c < OCR_MIN_CONFIDENCE)
            ratio    = low_conf / len(confidences)
            if ratio > 0.30:
                findings.append(Finding(
                    category="OCR Anomaly",
                    description=(
                        f"Page {page_num}: {ratio*100:.0f}% of OCR-detected words have "
                        f"low confidence (<{OCR_MIN_CONFIDENCE}%). May indicate blurred "
                        "text replacement or image manipulation."
                    ),
                    severity=min(0.30 + ratio * 0.5, 0.55),
                    page=page_num,
                    evidence={"low_conf_ratio": round(ratio, 2), "total_words": len(confidences)},
                ))
    except Exception:
        pass

    # Embedded text vs visible text divergence
    try:
        ocr_text      = pytesseract.image_to_string(img, lang="eng").strip()
        embedded_text = (page.extract_text() or "").strip()
        if ocr_text and embedded_text and len(embedded_text) > 30 and len(ocr_text) > 30:
            ocr_words      = set(ocr_text.lower().split())
            embedded_words = set(embedded_text.lower().split())
            overlap = len(ocr_words & embedded_words) / max(len(ocr_words), len(embedded_words))
            if overlap < 0.25:
                findings.append(Finding(
                    category="Invisible Text",
                    description=(
                        f"Page {page_num}: OCR text and embedded text layer differ "
                        f"significantly (overlap {overlap*100:.0f}%). "
                        "Hidden text layer may contain content different from the visible scan."
                    ),
                    severity=0.65,
                    page=page_num,
                    evidence={"text_overlap": round(overlap, 2)},
                ))
    except Exception:
        pass