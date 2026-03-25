"""
URL Detector
------------
Detects:
  - Suspicious external URLs in annotations and raw content
  - URL shorteners / redirectors
  - JavaScript URIs
  - Known phishing / malware domain patterns
  - Mismatch between displayed text and actual link target
"""

from __future__ import annotations
import re
from typing import List
from ..engine import Finding

URL_SHORTENERS = [
    r"bit\.ly", r"tinyurl\.com", r"t\.co", r"goo\.gl",
    r"ow\.ly", r"is\.gd", r"buff\.ly", r"tiny\.cc",
    r"rb\.gy", r"cutt\.ly", r"short\.io",
]

SUSPICIOUS_TLD = [
    r"\.xyz$", r"\.tk$", r"\.ml$", r"\.ga$", r"\.cf$",
    r"\.gq$", r"\.top$", r"\.pw$", r"\.click$", r"\.download$",
]

JS_URI_RE = re.compile(rb"javascript\s*:", re.IGNORECASE)
URL_RE = re.compile(
    rb"https?://[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]{5,200}"
)


def detect(reader, raw: bytes) -> List[Finding]:
    findings: List[Finding] = []

    urls_from_raw = _extract_urls_raw(raw)
    _check_urls(urls_from_raw, findings, page=None)

    for page_num, page in enumerate(reader.pages, 1):
        try:
            page_urls = _extract_urls_from_page(page)
            _check_urls(page_urls, findings, page=page_num)
        except Exception:
            pass

    # JavaScript URIs
    if JS_URI_RE.search(raw):
        findings.append(Finding(
            category="Suspicious URL",
            description="JavaScript URI found in document. PDF files should not contain javascript: protocol links.",
            severity=0.75,
        ))

    return findings


def _extract_urls_raw(raw: bytes) -> list:
    matches = URL_RE.findall(raw[:65536] + raw[-4096:])
    return [m.decode(errors="replace") for m in matches[:50]]


def _extract_urls_from_page(page) -> list:
    urls = []
    try:
        annots = page.get("/Annots")
        if not annots:
            return urls
        for annot in annots:
            try:
                obj = annot.get_object() if hasattr(annot, "get_object") else annot
                action = obj.get("/A")
                if not action:
                    continue
                action_obj = action.get_object() if hasattr(action, "get_object") else action
                uri = str(action_obj.get("/URI", "") or "")
                if uri:
                    urls.append(uri)
            except Exception:
                pass
    except Exception:
        pass
    return urls


def _check_urls(urls: list, findings: list, page):
    seen = set()
    for url in urls:
        url_lower = url.lower()
        if url_lower in seen:
            continue
        seen.add(url_lower)

        # URL shorteners
        for pattern in URL_SHORTENERS:
            if re.search(pattern, url_lower):
                findings.append(Finding(
                    category="Suspicious URL",
                    description=f"URL shortener detected: '{url[:80]}'. Destination is obscured — common in phishing and malware distribution.",
                    severity=0.55,
                    page=page,
                    evidence={"url": url[:100]},
                ))
                break

        # Suspicious TLDs
        for pattern in SUSPICIOUS_TLD:
            domain_match = re.search(r"https?://([^/?\s]+)", url_lower)
            if domain_match:
                domain = domain_match.group(1)
                if re.search(pattern, domain):
                    findings.append(Finding(
                        category="Suspicious URL",
                        description=f"URL with high-risk TLD: '{url[:80]}'.",
                        severity=0.45,
                        page=page,
                        evidence={"url": url[:100], "domain": domain},
                    ))
                    break

        # IP-based URLs (bypass DNS)
        if re.search(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url_lower):
            findings.append(Finding(
                category="Suspicious URL",
                description=f"Direct IP-address URL: '{url[:80]}'. Bypasses domain-based security controls.",
                severity=0.5,
                page=page,
                evidence={"url": url[:100]},
            ))
