"""
Encoding Detector
--------------------------------------------------
ASCII85Decode and FlateDecode are NORMAL in any PDF with embedded fonts/images.
Only flag genuinely suspicious patterns.
"""
from __future__ import annotations
import re
from typing import List
from ..engine import Finding

VALID_VERSIONS = {b"1.0",b"1.1",b"1.2",b"1.3",b"1.4",b"1.5",b"1.6",b"1.7",b"2.0"}

# Only flag filters that are UNUSUAL — NOT ASCII85 or FlateDecode (those are normal)
SUSPICIOUS_FILTERS = [
    (b"/Crypt",          "/Crypt filter — custom encryption within a stream, can hide content", 0.55),
    (b"/RunLengthDecode","/RunLengthDecode — uncommon filter, sometimes used for evasion", 0.25),
]

NESTED_FILTER_RE = re.compile(rb"/Filter\s*\[\s*(?:/\w+\s*){3,}\]")


def detect(raw: bytes) -> List[Finding]:
    findings: List[Finding] = []
    _check_header_version(raw, findings)
    _check_suspicious_filters(raw, findings)
    _check_nested_filters(raw, findings)
    _check_objstm(raw, findings)
    _check_xref_integrity(raw, findings)
    return findings


def _check_header_version(raw: bytes, findings: list):
    header = raw[:16]
    m = re.match(rb"%PDF-(\d+\.\d+)", header)
    if not m:
        findings.append(Finding(
            category="Unusual Encoding",
            description="Missing or invalid %PDF- header. File may be disguised as a PDF.",
            severity=0.55,
        ))
        return
    version = m.group(1)
    if version not in VALID_VERSIONS:
        findings.append(Finding(
            category="Unusual Encoding",
            description=f"Non-standard PDF version '{version.decode()}' in header.",
            severity=0.25,
            evidence={"version": version.decode()},
        ))


def _check_suspicious_filters(raw: bytes, findings: list):
    for pattern, description, severity in SUSPICIOUS_FILTERS:
        count = raw.count(pattern)
        if count > 0:
            findings.append(Finding(
                category="Unusual Encoding",
                description=f"{description} (found {count}× in document).",
                severity=severity,
                evidence={"filter": pattern.decode(), "count": count},
            ))


def _check_nested_filters(raw: bytes, findings: list):
    """3+ chained filters on a single stream = obfuscation attempt."""
    matches = NESTED_FILTER_RE.findall(raw)
    if matches:
        findings.append(Finding(
            category="Unusual Encoding",
            description=f"Stream with 3+ chained filters ({len(matches)}× found). Deeply nested filters are a known obfuscation technique.",
            severity=0.50,
            evidence={"count": len(matches)},
        ))


def _check_objstm(raw: bytes, findings: list):
    count = raw.count(b"/ObjStm")
    if count > 8:   # a few ObjStm is normal; many is not
        findings.append(Finding(
            category="Unusual Encoding",
            description=f"Unusually high number of Object Streams (/ObjStm ×{count}). Can be used to hide objects from basic parsers.",
            severity=0.35,
            evidence={"count": count},
        ))


def _check_xref_integrity(raw: bytes, findings: list):
    try:
        defined = set(int(m.group(1)) for m in re.finditer(rb"(\d+)\s+\d+\s+obj\b", raw))
        xref_counts = [int(m.group(1)) for m in re.finditer(rb"^0\s+(\d+)\s*$", raw, re.MULTILINE)]
        if xref_counts and defined:
            expected = max(xref_counts)
            actual = len(defined)
            if actual > 0 and (actual > expected * 1.6 or actual < expected * 0.4):
                findings.append(Finding(
                    category="Unusual Encoding",
                    description=f"XRef table declares {expected} objects but {actual} definitions found. Suggests hidden or injected objects.",
                    severity=0.55,
                    evidence={"xref_count": expected, "actual_count": actual},
                ))
    except Exception:
        pass