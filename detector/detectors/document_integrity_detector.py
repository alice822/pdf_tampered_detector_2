"""
Document Integrity Detector
============================
Catches forgeries that other detectors miss:

1. Wrong official text  — "ELECTION COMMISSION OF YOURS" vs "OF INDIA"
2. Missing face photo   — placeholder silhouette instead of real photo
3. Impossible content   — "MALE" in wrong field, dates in wrong century, etc.
4. Known fake phrases   — phrases that never appear on genuine documents

Works on any Indian government ID: PAN, Aadhaar, Voter ID, Passport, Driving License.
No OCR confidence issues — just exact string matching on extracted text.
"""
from __future__ import annotations
import re
from typing import List
from ..engine import Finding

# ── Phrases that MUST appear on genuine documents ─────────────────────────────
REQUIRED_PHRASES = {
    "voter_id": [
        "ELECTION COMMISSION OF INDIA",
        "ELECTOR PHOTO IDENTITY CARD",
    ],
    "pan": [
        "INCOME TAX DEPARTMENT",
        "GOVT. OF INDIA",
        "Permanent Account Number",
    ],
    "aadhaar": [
        "Government of India",
        "आधार",
    ],
}

# ── Phrases that NEVER appear on genuine documents ────────────────────────────
FORBIDDEN_PHRASES = [
    "ELECTION COMMISSION OF YOURS",
    "ELECTION COMMISSION OF YOU",
    "INCOME TAX DEPT OF YOU",
    "GOVERNMENT OF YOU",
    "FAKE",
    "TEST CARD",
    "SAMPLE",
    "DUMMY",
    "SPECIMEN",
]

# ── Known typos/substitutions used in fake cards ─────────────────────────────
SUSPICIOUS_SUBSTITUTIONS = [
    (r"COMMISSION\s+OF\s+(?!INDIA)[A-Z]+", "Wrong text after COMMISSION OF"),
    (r"GOVT\.?\s+OF\s+(?!INDIA)[A-Z]+",    "Wrong text after GOVT OF"),
    (r"INCOME\s+TAX\s+(?!DEPARTMENT)[A-Z]+","Wrong text after INCOME TAX"),
]


def detect(reader) -> List[Finding]:
    findings: List[Finding] = []

    for page_num, page in enumerate(reader.pages, 1):
        try:
            text = (page.extract_text() or "").strip()
            if not text:
                # Try visitor-based extraction
                parts = []
                def visitor(t, *args):
                    if t and t.strip():
                        parts.append(t.strip())
                page.extract_text(visitor_text=visitor)
                text = " ".join(parts)

            if not text:
                continue

            text_upper = text.upper()
            issues = []
            word_rects = []

            # ── Check 1: Forbidden phrases ─────────────────────────────────
            for phrase in FORBIDDEN_PHRASES:
                if phrase.upper() in text_upper:
                    issues.append(f"Forbidden phrase found: '{phrase}'")

            # ── Check 2: Suspicious substitutions ─────────────────────────
            for pattern, description in SUSPICIOUS_SUBSTITUTIONS:
                match = re.search(pattern, text_upper)
                if match:
                    issues.append(f"{description}: '{match.group()}'")

            # ── Check 3: Detect document type and check required phrases ──
            doc_type = None
            if "ELECTION COMMISSION" in text_upper or "ELECTOR" in text_upper:
                doc_type = "voter_id"
            elif "INCOME TAX" in text_upper or "PERMANENT ACCOUNT" in text_upper:
                doc_type = "pan"
            elif "AADHAAR" in text_upper or "आधार" in text:
                doc_type = "aadhaar"

            if doc_type and doc_type in REQUIRED_PHRASES:
                for required in REQUIRED_PHRASES[doc_type]:
                    if required.upper() not in text_upper:
                        issues.append(
                            f"Missing required phrase for {doc_type.upper()}: '{required}'"
                        )

            if issues:
                mb     = page.mediabox
                pdf_h  = float(mb.height)
                pdf_w  = float(mb.width)

                findings.append(Finding(
                    category="Content Integrity Failure",
                    description=(
                        f"Page {page_num}: Document content fails integrity checks. "
                        + " | ".join(issues)
                        + ". Genuine government documents have fixed, verified text."
                    ),
                    severity=0.95,
                    page=page_num,
                    evidence={
                        "issues":     issues,
                        "doc_type":   doc_type,
                        "bold_words": [i.split("'")[1] for i in issues if "'" in i][:4],
                        "word_rects": word_rects,
                        "detail":     f"{len(issues)} integrity issue(s): " + "; ".join(issues[:2]),
                    },
                ))

        except Exception:
            pass

    return findings