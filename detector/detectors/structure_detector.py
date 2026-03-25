"""
Structure Detector
------------------
Low-level binary analysis of the PDF structure:
  - Mismatched object offsets in xref table
  - Truncated or padded content after %%EOF
  - Binary junk before %PDF- header
  - Duplicate object IDs in different generations
  - Linearization hints vs actual structure mismatch
"""

from __future__ import annotations
import re
from typing import List
from ..engine import Finding


def detect(raw: bytes) -> List[Finding]:
    findings: List[Finding] = []

    _check_pre_header_junk(raw, findings)
    _check_post_eof_content(raw, findings)
    _check_xref_offsets(raw, findings)
    _check_linearization(raw, findings)
    _check_duplicate_objects(raw, findings)

    return findings


def _check_pre_header_junk(raw: bytes, findings: list):
    """Data before %PDF- header is suspicious."""
    header_pos = raw.find(b"%PDF-")
    if header_pos < 0:
        findings.append(Finding(
            category="Unusual Encoding",
            description="No %PDF- header found. File may be malformed or disguised.",
            severity=0.7,
        ))
    elif header_pos > 1024:
        findings.append(Finding(
            category="Unusual Encoding",
            description=f"%PDF- header starts at byte offset {header_pos} (expected near 0). Pre-header data may conceal a different payload.",
            severity=0.55,
            evidence={"header_offset": header_pos},
        ))
    elif header_pos > 0:
        pre_bytes = raw[:header_pos]
        if any(b > 127 for b in pre_bytes):
            findings.append(Finding(
                category="Unusual Encoding",
                description=f"Non-ASCII bytes found before PDF header (offset {header_pos}). May indicate file prepended with binary data.",
                severity=0.4,
                evidence={"pre_header_size": header_pos},
            ))


def _check_post_eof_content(raw: bytes, findings: list):
    """Significant data after last %%EOF is suspicious."""
    last_eof = raw.rfind(b"%%EOF")
    if last_eof == -1:
        return
    post_eof = raw[last_eof + 5:].strip()
    if len(post_eof) > 256:
        findings.append(Finding(
            category="Unusual Encoding",
            description=f"{len(post_eof)} bytes found after final %%EOF. May contain hidden payload or second embedded document.",
            severity=0.6,
            evidence={"post_eof_bytes": len(post_eof)},
        ))


def _check_xref_offsets(raw: bytes, findings: list):
    """
    Parse traditional xref table and verify a sample of offsets.
    Mismatches indicate the file was rebuilt without updating xref.
    """
    xref_pos = raw.rfind(b"xref")
    if xref_pos == -1:
        return

    try:
        xref_section = raw[xref_pos: xref_pos + 65536]
        # Parse "OFFSET GENERATION n" entries
        entry_re = re.compile(rb"(\d{10})\s+(\d{5})\s+([fn])\s*")
        mismatches = 0
        checked = 0

        for m in entry_re.finditer(xref_section):
            if checked >= 20:
                break
            offset = int(m.group(1))
            flag = m.group(3)
            if flag == b"n" and offset > 0:
                # Check that there's an "obj" keyword near this offset
                region = raw[max(0, offset - 2): offset + 30]
                if b"obj" not in region:
                    mismatches += 1
                checked += 1

        if checked > 0 and mismatches / checked > 0.3:
            findings.append(Finding(
                category="Unusual Encoding",
                description=f"XRef offset mismatch: {mismatches}/{checked} checked entries point to wrong locations. Document may have been manually edited or reassembled.",
                severity=0.65,
                evidence={"mismatches": mismatches, "checked": checked},
            ))
    except Exception:
        pass


def _check_linearization(raw: bytes, findings: list):
    """Check for /Linearized dict that doesn't match the actual structure."""
    if b"/Linearized" not in raw[:4096]:
        return
    # Linearized PDFs should not have incremental updates
    eof_count = raw.count(b"%%EOF")
    if eof_count > 1:
        findings.append(Finding(
            category="Unusual Encoding",
            description="Document claims to be linearized but has incremental updates — linearization hint is invalid, suggesting structural modification.",
            severity=0.45,
            evidence={"eof_count": eof_count},
        ))


def _check_duplicate_objects(raw: bytes, findings: list):
    """Detect objects with the same ID defined multiple times (outside incremental updates)."""
    try:
        # Find all "N M obj" definitions
        obj_defs = re.findall(rb"(\d+)\s+(\d+)\s+obj\b", raw)
        seen = {}
        duplicates = set()
        for obj_id, gen in obj_defs:
            key = (int(obj_id), int(gen))
            count = seen.get(key, 0) + 1
            seen[key] = count
            if count > 1:
                duplicates.add(key)

        if len(duplicates) > 2:
            findings.append(Finding(
                category="Unusual Encoding",
                description=f"{len(duplicates)} object IDs are defined multiple times in the same revision. This is a hallmark of manual hex-editing or object injection.",
                severity=0.6,
                evidence={"duplicate_count": len(duplicates)},
            ))
    except Exception:
        pass
