"""
Script / Malware Detector
--------------------------
Detects:
  - Embedded JavaScript (/JS, /JavaScript)
  - OpenAction / AA auto-actions
  - Launch actions (run external programs)
  - Known shellcode byte patterns
  - Heap-spray NOP-sled patterns
  - PDF malware keyword signatures
"""

from __future__ import annotations
import re
from typing import List
from ..engine import Finding


MALWARE_PATTERNS = [
    (rb"eval\s*\(", "eval() call in JavaScript", 0.7),
    (rb"unescape\s*\(", "unescape() in JavaScript (common obfuscation)", 0.65),
    (rb"String\.fromCharCode", "String.fromCharCode obfuscation in JavaScript", 0.65),
    (rb"this\[.{1,20}\]\s*\(", "Dynamic method call in JavaScript (common obfuscation)", 0.6),
    (rb"%u0000", "Unicode escape sequence (heap spray indicator)", 0.6),
    (rb"\x90{16,}", "NOP-sled pattern (shellcode indicator)", 0.9),
    (rb"cmd\.exe", "cmd.exe reference in PDF content", 0.85),
    (rb"powershell", "PowerShell reference", 0.8),
    (rb"/Launch\b", "/Launch action — can run external executables", 0.9),
    (rb"/SubmitForm\b", "/SubmitForm action — sends data to remote server", 0.5),
    (rb"/ImportData\b", "/ImportData action — loads external data", 0.55),
    (rb"app\.openDoc\s*\(", "app.openDoc() JavaScript call", 0.65),
    (rb"app\.launchURL\s*\(", "app.launchURL() JavaScript call", 0.6),
    (rb"getAnnots\s*\(", "getAnnots() JavaScript (common in exploits)", 0.55),
    (rb"getPageNthWord\s*\(", "getPageNthWord() exploit pattern", 0.6),
    (rb"util\.printf\s*\(", "util.printf() buffer overflow exploit pattern", 0.75),
    (rb"collab\.getIcon\s*\(", "collab.getIcon() exploit pattern", 0.8),
    (rb"/EmbeddedFile\b", "/EmbeddedFile — file attachment (may contain malware)", 0.4),
    (rb"CVE-20[12]\d-\d+", "CVE identifier in document content", 0.5),
]

JS_KEY_PATTERNS = [rb"/JS\b", rb"/JavaScript\b", rb"/OpenAction\b", rb"/AA\b", rb"/AcroForm\b"]


def detect(reader, raw: bytes) -> List[Finding]:
    findings: List[Finding] = []

    has_js = any(pat in raw for pat in JS_KEY_PATTERNS[:2])
    has_open_action = b"/OpenAction" in raw
    has_aa = b"/AA" in raw

    if has_js:
        findings.append(Finding(
            category="Embedded Script",
            description="Document contains JavaScript (/JS or /JavaScript). PDF JavaScript is rarely legitimate and is commonly used in malicious documents.",
            severity=0.7,
        ))

    if has_open_action:
        findings.append(Finding(
            category="Embedded Script",
            description="/OpenAction detected — executes automatically when the PDF is opened.",
            severity=0.65,
        ))

    if has_aa:
        findings.append(Finding(
            category="Embedded Script",
            description="/AA (Additional Actions) found — automatic triggers on page open/close.",
            severity=0.5,
        ))

    # Scan for malware patterns (first 512 KB + last 32 KB for speed)
    scan_window = raw[:524288] + raw[-32768:]
    for pattern, description, severity in MALWARE_PATTERNS:
        if re.search(pattern, scan_window, re.IGNORECASE):
            findings.append(Finding(
                category="Embedded Script",
                description=description,
                severity=severity,
                evidence={"pattern": pattern.decode(errors="replace")[:40]},
            ))

    # Check for embedded files
    _check_embedded_files(reader, findings)

    return findings


def _check_embedded_files(reader, findings: list):
    try:
        catalog = reader.trailer.get("/Root")
        if not catalog:
            return
        catalog = catalog.get_object() if hasattr(catalog, "get_object") else catalog
        names = catalog.get("/Names")
        if not names:
            return
        names_obj = names.get_object() if hasattr(names, "get_object") else names
        ef = names_obj.get("/EmbeddedFiles")
        if ef:
            findings.append(Finding(
                category="Embedded Script",
                description="Embedded file attachment found. Attachments can contain malware and may be used to exfiltrate or deliver payloads.",
                severity=0.55,
            ))
    except Exception:
        pass
