"""
image_converter.py
==================
Converts uploaded image files (JPG, PNG, WEBP, BMP, TIFF) into a
temporary PDF so the full detector pipeline can run on them unchanged.

Why convert to PDF instead of running detectors on the raw image?
  - All 22+ detectors expect a pypdf PdfReader object
  - All heatmap/page coordinate logic uses PDF points
  - Conversion preserves original image bytes embedded in the PDF
  - No detector needs to be modified

Conversion strategy:
  - Image is embedded at NATIVE resolution (no resampling)
  - Page size matches image aspect ratio at 72 DPI (1px = 1pt)
  - Original image bytes are stored directly (no re-encoding for JPEG)
  - Metadata is injected: original filename, format, dimensions
  - Temp file is cleaned up after analysis

Usage:
    from detector.image_converter import maybe_convert_to_pdf, is_image_file

    # In app.py or cli.py:
    pdf_path, was_converted = maybe_convert_to_pdf(uploaded_path)
    result = detector.analyze(pdf_path)
    if was_converted:
        os.unlink(pdf_path)   # clean up temp file
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Tuple

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}

# Max dimension before we warn (not resize — just warn)
MAX_DIMENSION_WARNING = 8000


def is_image_file(path: str) -> bool:
    """Return True if the file extension is a supported image format."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def maybe_convert_to_pdf(path: str) -> Tuple[str, bool]:
    """
    If path is an image, convert it to a temp PDF and return (pdf_path, True).
    If path is already a PDF, return (path, False) unchanged.

    Caller is responsible for deleting the temp file when was_converted=True.
    """
    if not is_image_file(path):
        return path, False

    pdf_path = image_to_pdf(path)
    return pdf_path, True


def image_to_pdf(image_path: str) -> str:
    """
    Convert an image file to a PDF with the image embedded at native resolution.
    Returns path to a temporary PDF file.

    The PDF has:
      - One page sized to match the image dimensions (1 point per pixel)
      - The original image embedded as an XObject (no quality loss for JPEG)
      - Producer metadata set to the original filename for traceability
    """
    try:
        from PIL import Image
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError(f"PIL and PyMuPDF required for image conversion: {e}")

    image_path = str(image_path)

    # ── Open and validate image ────────────────────────────────────────
    try:
        img = Image.open(image_path)
        img.verify()  # Check for corruption
        img = Image.open(image_path)  # Re-open after verify (verify closes it)
    except Exception as e:
        raise ValueError(f"Cannot open image '{image_path}': {e}")

    # Convert to RGB if needed (PDF doesn't support all PIL modes directly)
    if img.mode in ("RGBA", "P", "LA"):
        # Preserve alpha by compositing on white background
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode in ("RGBA", "LA"):
            background.paste(img, mask=img.split()[-1])
        else:
            background.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    width_px, height_px = img.size

    if max(width_px, height_px) > MAX_DIMENSION_WARNING:
        import logging
        logging.getLogger(__name__).warning(
            f"Large image: {width_px}×{height_px}px — analysis may be slow"
        )

    # ── Create PDF with PyMuPDF ────────────────────────────────────────
    # Page size: 1 PDF point per pixel (so DPI calc in image_detector gives real DPI)
    # This is important: if we shrink the page, the DPI estimate will be wrong
    page_width_pt  = float(width_px)
    page_height_pt = float(height_px)

    pdf_doc = fitz.open()  # New empty PDF
    page = pdf_doc.new_page(width=page_width_pt, height=page_height_pt)

    # Embed the image
    # For JPEG: use original bytes directly (no re-encoding = no quality loss)
    # For PNG/others: let PyMuPDF handle it
    orig_ext = Path(image_path).suffix.lower()
    img_rect = fitz.Rect(0, 0, page_width_pt, page_height_pt)

    if orig_ext in (".jpg", ".jpeg"):
        # Embed original JPEG bytes directly — preserves all compression artifacts
        # This is critical for ELA and double-JPEG detection
        with open(image_path, "rb") as f:
            jpeg_bytes = f.read()
        page.insert_image(img_rect, stream=jpeg_bytes)
    else:
        # For PNG/WEBP/etc: insert from PIL image
        import io
        buf = io.BytesIO()
        img.save(buf, "PNG")
        buf.seek(0)
        page.insert_image(img_rect, stream=buf.read())

    # ── Add metadata so detectors know this was originally an image ────
    orig_filename = Path(image_path).name
    pdf_doc.set_metadata({
        "producer":   f"image_converter ({orig_filename})",
        "creator":    "PDFGuard image upload",
        "title":      orig_filename,
        "subject":    f"Converted from {orig_ext.upper()} for tamper analysis",
    })

    # ── Save to temp file ──────────────────────────────────────────────
    suffix = f"_converted_{Path(image_path).stem}.pdf"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)

    pdf_doc.save(tmp_path, garbage=4, deflate=True)
    pdf_doc.close()

    return tmp_path


def get_image_info(image_path: str) -> dict:
    """
    Return basic info about an image file for display purposes.
    Used by app.py to show file details before analysis.
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        w, h = img.size
        size_kb = os.path.getsize(image_path) // 1024
        return {
            "width": w,
            "height": h,
            "mode": img.mode,
            "format": img.format or Path(image_path).suffix.upper().lstrip("."),
            "size_kb": size_kb,
            "filename": Path(image_path).name,
        }
    except Exception:
        return {"filename": Path(image_path).name}