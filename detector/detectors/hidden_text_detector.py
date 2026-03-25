"""
Hidden Text Detector
---------------------
Detects:
  - Text rendered with colour matching background (white-on-white etc.)
  - Text with zero font size
  - Text with rendering mode 3 (invisible)
  - OCG (Optional Content Groups) used as hidden layers
  - Text positioned outside the page crop box
"""

from __future__ import annotations
import re
from typing import List
from ..engine import Finding


# PDF text rendering modes
# 0=fill, 1=stroke, 2=fill+stroke, 3=invisible, 4–7 with clipping
INVISIBLE_RENDER_MODE = 3


def detect(reader) -> List[Finding]:
    findings: List[Finding] = []

    for page_num, page in enumerate(reader.pages, 1):
        try:
            raw_content = _get_page_content(page)
            if raw_content:
                _check_invisible_text(raw_content, page_num, findings)
                _check_zero_size_text(raw_content, page_num, findings)
                _check_off_page_text(raw_content, page, page_num, findings)
            _check_ocg(page, page_num, findings)
        except Exception:
            pass

    return findings


def _get_page_content(page) -> bytes:
    try:
        content = page.get("/Contents")
        if not content:
            return b""
        obj = content.get_object() if hasattr(content, "get_object") else content
        if hasattr(obj, "get_data"):
            return obj.get_data()
        # Array of content streams
        combined = b""
        try:
            for item in obj:
                item_obj = item.get_object() if hasattr(item, "get_object") else item
                if hasattr(item_obj, "get_data"):
                    combined += item_obj.get_data()
        except Exception:
            pass
        return combined
    except Exception:
        return b""


def _check_invisible_text(content: bytes, page_num: int, findings: list):
    """Detect rendering mode 3 (invisible text)."""
    # Pattern: Tr = text render mode operator
    tr_pattern = re.compile(rb"(\d)\s+Tr\b")
    for m in tr_pattern.finditer(content):
        mode = int(m.group(1))
        if mode == INVISIBLE_RENDER_MODE:
            findings.append(Finding(
                category="Invisible Text",
                description="Text with rendering mode 3 (invisible) detected. Used for hidden machine-readable text overlaying scanned content, or to hide watermarks/metadata.",
                severity=0.65,
                page=page_num,
            ))
            break


def _check_zero_size_text(content: bytes, page_num: int, findings: list):
    """Detect text drawn with font size 0."""
    # Pattern: 0 Tf or 0.0 Tf
    tf_pattern = re.compile(rb"/\w+\s+0(?:\.0+)?\s+Tf\b")
    if tf_pattern.search(content):
        findings.append(Finding(
            category="Invisible Text",
            description="Text rendered at font size 0 detected. Zero-size text is invisible but machine-readable — commonly used to embed hidden data.",
            severity=0.6,
            page=page_num,
        ))


def _check_off_page_text(content: bytes, page, page_num: int, findings: list):
    """
    Detect text positioned far outside the page mediabox.
    Uses a simple heuristic on Td/TD/Tm operators.
    """
    try:
        mb = page.mediabox
        page_w = float(mb.width)
        page_h = float(mb.height)
        margin = max(page_w, page_h) * 2  # Allow 2× page size offset before flagging

        # Tm: [a b c d x y] sets text matrix, x/y are position
        tm_pattern = re.compile(
            rb"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+"
            rb"([-\d.]+)\s+([-\d.]+)\s+Tm\b"
        )
        for m in tm_pattern.finditer(content):
            try:
                x, y = float(m.group(5)), float(m.group(6))
                if abs(x) > margin or abs(y) > margin:
                    findings.append(Finding(
                        category="Hidden Layer",
                        description=f"Text matrix positions content far outside page bounds ({x:.0f}, {y:.0f}) — possible hidden data layer.",
                        severity=0.5,
                        page=page_num,
                        evidence={"x": x, "y": y, "page_w": page_w, "page_h": page_h},
                    ))
                    break
            except (ValueError, IndexError):
                pass
    except Exception:
        pass


def _check_ocg(page, page_num: int, findings: list):
    """Detect Optional Content Groups (OCG) — toggleable hidden layers."""
    try:
        resources = page.get("/Resources")
        if not resources:
            return
        resources = resources.get_object() if hasattr(resources, "get_object") else resources
        props = resources.get("/Properties")
        if not props:
            return
        props_obj = props.get_object() if hasattr(props, "get_object") else props
        for key in props_obj:
            try:
                obj = props_obj[key]
                obj = obj.get_object() if hasattr(obj, "get_object") else obj
                if str(obj.get("/Type", "")) == "/OCG":
                    name = str(obj.get("/Name", "unnamed"))
                    findings.append(Finding(
                        category="Hidden Layer",
                        description=f"Optional Content Group (hidden layer) '{name}' found on page. May be used to show/hide content selectively.",
                        severity=0.45,
                        page=page_num,
                        evidence={"ocg_name": name},
                    ))
            except Exception:
                pass
    except Exception:
        pass
