"""
Incremental Update Detector
----------------------------
PDF supports appending incremental updates (new cross-reference sections)
without modifying the original body. This is used by:
  - Legitimate tools for commenting/signing
  - Attackers to modify content while preserving original bytes

Detects:
  - Number of incremental update sections (%%EOF count)
  - Size of each update relative to original
  - New xref sections modifying content streams
  - Shadow attacks (object ID reuse)
"""

from __future__ import annotations
import re
from typing import List
from ..engine import Finding


def detect(raw: bytes) -> List[Finding]:
    findings: List[Finding] = []

    eof_positions = [m.start() for m in re.finditer(rb"%%EOF", raw)]
    update_count = len(eof_positions) - 1  # first is original

    if update_count <= 0:
        return findings

    if update_count == 1:
        # Single update — only suspicious with other evidence
        findings.append(Finding(
            category="Incremental Update",
            description="Document has 1 incremental update section. May be a legitimate annotation/signature, but also used to modify content post-signing.",
            severity=0.2,
            evidence={"update_count": 1},
        ))

    elif update_count == 2:
        findings.append(Finding(
            category="Incremental Update",
            description=f"Document has {update_count} incremental updates. Multiple revisions suggest repeated post-issuance modifications.",
            severity=0.45,
            evidence={"update_count": update_count},
        ))

    elif update_count >= 3:
        findings.append(Finding(
            category="Incremental Update",
            description=f"Document has {update_count} incremental update sections — unusually high. Possible iterative manipulation or automated patching.",
            severity=0.65,
            evidence={"update_count": update_count},
        ))

    # Measure sizes of each section
    _check_update_sizes(raw, eof_positions, update_count, findings)

    # Detect shadow attack: object numbers reused in later xref
    _check_shadow_attack(raw, findings)

    return findings


def _check_update_sizes(raw: bytes, eof_positions: list, update_count: int, findings: list):
    """Flag updates that are large relative to the original body."""
    if len(eof_positions) < 2:
        return
    original_size = eof_positions[0]
    if original_size == 0:
        return

    for i, pos in enumerate(eof_positions[1:], 1):
        prev = eof_positions[i - 1]
        update_size = pos - prev
        ratio = update_size / original_size
        if ratio > 0.3:
            findings.append(Finding(
                category="Incremental Update",
                description=f"Incremental update #{i} is {ratio*100:.0f}% the size of the original document — disproportionately large for a legitimate annotation.",
                severity=min(0.7, 0.3 + ratio * 0.5),
                evidence={"update_num": i, "update_size": update_size, "original_size": original_size},
            ))


def _check_shadow_attack(raw: bytes, findings: list):
    """
    Shadow attack: a malicious update redefines high-value objects.
    Heuristic: look for xref entries that assign new content to obj 1-5
    (document catalog, page tree root — high-value targets).
    """
    # Find all "startxref" sections after the first
    startxref_positions = [m.start() for m in re.finditer(rb"startxref\s+(\d+)", raw)]
    if len(startxref_positions) < 2:
        return

    # Look for xref sections in updates redefining object 1–5
    update_region = raw[startxref_positions[0]:]
    obj_redef_pattern = re.compile(rb"\b([1-5])\s+0\s+obj\b")
    redefined = set()
    for m in obj_redef_pattern.finditer(update_region):
        redefined.add(int(m.group(1)))

    if redefined:
        findings.append(Finding(
            category="Incremental Update",
            description=f"Critical PDF objects ({sorted(redefined)}) are redefined in incremental updates. This pattern is used in shadow attacks to replace signed content.",
            severity=0.8,
            evidence={"redefined_objects": sorted(redefined)},
        ))
