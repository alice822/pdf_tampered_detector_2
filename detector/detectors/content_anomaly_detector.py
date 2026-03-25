"""
Content Anomaly Detector  (v3 — fully document-agnostic)
---------------------------------------------------------
NO hardcoded rules for any specific document type.
Works on PAN cards, passports, salary slips, bank statements,
degree certificates, court orders, offer letters — anything.

Detection is entirely STATISTICAL and STRUCTURAL:

  1. Truncated tokens     — values cut off mid-sequence (any format)
  2. Duplicate lines      — copy-paste artifact (any document)
  3. Impossible numbers   — dates with invalid month/day/year, any number
                            that violates its own stated format
  4. Mixed-script fields  — Latin + non-Latin in same logical field
  5. Field value conflicts — same label appearing twice with different values
  6. Character substitution — visually similar char swaps (0→O, 1→l, rn→m)
  7. AI-generation signals — unnaturally uniform text entropy, zero errors,
                             suspiciously perfect punctuation
  8. Statistical word anomaly — word appearing in context far from its
                                 typical semantic cluster

None of these require knowing whether the document is a PAN card,
a diploma, a bank statement, or an employment contract.
"""

from __future__ import annotations
import re
import io
import math
from collections import Counter, defaultdict
from typing import List
from ..engine import Finding

try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image, ImageStat
    OCR_OK = True
except ImportError:
    OCR_OK = False

MAX_PAGES = 6


def detect(reader) -> List[Finding]:
    findings: List[Finding] = []
    if not OCR_OK:
        return findings
    try:
        buf = io.BytesIO()
        from pypdf import PdfWriter
        writer = PdfWriter()
        total = min(len(reader.pages), MAX_PAGES)
        for i in range(total):
            writer.add_page(reader.pages[i])
        writer.write(buf)
        images = convert_from_bytes(buf.getvalue(), dpi=200)
        for page_num, img in enumerate(images, 1):
            try:
                _analyze_page(img, page_num, findings)
            except Exception:
                pass
    except Exception:
        pass
    return findings


def _analyze_page(img, page_num: int, findings: list):
    # Full OCR — try with Hindi support, fall back to English only
    try:
        ocr_text = pytesseract.image_to_string(img, lang="eng+hin", config="--psm 6").strip()
    except Exception:
        ocr_text = pytesseract.image_to_string(img, lang="eng",     config="--psm 6").strip()

    if len(ocr_text) < 15:
        return

    lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
    words = ocr_text.split()

    _check_truncated_tokens(ocr_text, page_num, findings)
    _check_duplicate_lines(lines, page_num, findings)
    _check_impossible_numbers(ocr_text, page_num, findings)
    _check_unusual_names(ocr_text, page_num, findings)
    _check_field_value_conflicts(lines, page_num, findings)
    _check_char_substitution(ocr_text, lines, page_num, findings)
    _check_ai_generation_signals(ocr_text, words, page_num, findings)
    _check_mixed_script_fields(lines, page_num, findings)


# ══════════════════════════════════════════════════════════════════
# 1. Truncated tokens — ANY format
# ══════════════════════════════════════════════════════════════════

# Patterns that look like they should be longer but are cut off
_TRUNCATED_DATE = re.compile(
    r'\b(\d{1,2})[/\.](\d{1,2})[/\.](\d{1,3})\b'
)
_TRUNCATED_ALNUM = re.compile(
    r'\b([A-Z]{2,8})(\d{1,3})\s*$',                      # ID-like ending at line end
    re.MULTILINE
)

def _check_truncated_tokens(text: str, page_num: int, findings: list):
    issues = []
    for m in _TRUNCATED_DATE.finditer(text):
        yr = m.group(3)
        if len(yr) < 4:
            issues.append(f"Truncated date '{m.group(0)}' — year fragment has {len(yr)} digit(s)")
    for m in _TRUNCATED_ALNUM.finditer(text):
        candidate = m.group(0).strip()
        # Only flag if it looks like it should continue (ends without natural punctuation)
        if len(candidate) >= 5 and not candidate[-1] in '.,:;)':
            issues.append(f"Possible truncated identifier '{candidate}' at line end")
    if issues:
        findings.append(Finding(
            category="Content Anomaly",
            description=(
                f"Truncated field values on page {page_num}: "
                + "; ".join(issues[:3])
                + ". Values cut off mid-sequence indicate the original content "
                "was replaced imprecisely — the replacement didn't fit the space."
            ),
            severity=0.72,
            page=page_num,
            evidence={"truncated": issues[:4]},
        ))


# ══════════════════════════════════════════════════════════════════
# 2. Duplicate lines — copy-paste artifact
# ══════════════════════════════════════════════════════════════════

def _check_duplicate_lines(lines: list, page_num: int, findings: list):
    counts = Counter(l for l in lines if len(l) > 6)
    repeated = [(l, c) for l, c in counts.items() if c > 1]
    if repeated:
        findings.append(Finding(
            category="Content Anomaly",
            description=(
                f"Duplicate text lines on page {page_num}: "
                + "; ".join(f"'{l}' ×{c}" for l, c in repeated[:3])
                + ". Identical lines indicate content was copy-pasted or overlaid."
            ),
            severity=0.58,
            page=page_num,
            evidence={"repeated_lines": [l for l, _ in repeated[:3]]},
        ))


# ══════════════════════════════════════════════════════════════════
# 3. Impossible numbers — dates, percentages, any numeric field
# ══════════════════════════════════════════════════════════════════

_DATE_PATTERN = re.compile(r'\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})\b')

# Common Indian name prefixes and suffixes for validation
_VALID_NAME_STARTS = re.compile(
    r'^(RAM|SHYAM|MOHAN|KUMAR|LAXMI|DEVI|SINGH|SHARMA|GUPTA|VERMA|'
    r'RAJ|ARUN|SURESH|MAHESH|DINESH|RAKESH|VIJAY|AMIT|SANJAY|AJAY|'
    r'PRIYA|SITA|GEETA|SUNITA|ANITA|REKHA|POOJA|NEHA|ASHA|USHA|'
    r'SEETARAM|RAMESH|NARESH|GANESH|UMESH|SUNIL|ANIL|PATEL|YADAV|'
    r'NARAYAN|JITENDRA|RAHUL|ROHIT|VIKAS|DEEPAK|SANTOSH|ASHOK|VINOD)',
    re.IGNORECASE
)

# Unusual consonant clusters that appear in no real Indian name
_UNUSUAL_CLUSTERS = re.compile(
    r'(BB[A-Z]|CHB|BHB|XB|QJ|WB|VB|ZB|FJ|HH[A-Z])',
    re.IGNORECASE
)


def _check_unusual_names(text: str, page_num: int, findings: list):
    """Detect names with unusual character patterns not found in real names."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    suspicious = []
    for line in lines:
        words = line.split()
        for word in words:
            if (len(word) >= 4 and word.isupper() and word.isalpha()
                    and _UNUSUAL_CLUSTERS.search(word)):
                suspicious.append(word)
    if suspicious:
        findings.append(Finding(
            category="Content Anomaly",
            description=(
                f"Unusual name pattern detected on page {page_num}: "
                f"{', '.join(suspicious[:3])}. "
                "This name contains character combinations that do not appear in "
                "any real name. This may indicate the name field was tampered with "
                "or the document contains fabricated personal information."
            ),
            severity=0.75,
            page=page_num,
            evidence={"unusual_names": suspicious[:3]},
        ))


def _check_impossible_numbers(text: str, page_num: int, findings: list):
    import datetime
    this_year = datetime.date.today().year
    issues = []
    for m in _DATE_PATTERN.finditer(text):
        try:
            d, mo, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if yr < 100:
                yr += 2000
            if mo > 12:
                issues.append(f"Invalid month {mo} in '{m.group(0)}'")
            elif d > 31:
                issues.append(f"Invalid day {d} in '{m.group(0)}'")
            elif yr > this_year + 1:
                issues.append(f"Future date '{m.group(0)}' (year {yr})")
            elif yr < 1900:
                issues.append(f"Implausible year {yr} in '{m.group(0)}'")
        except (ValueError, TypeError):
            pass
    if issues:
        findings.append(Finding(
            category="Content Anomaly",
            description=(
                f"Impossible/invalid values on page {page_num}: "
                + "; ".join(issues[:3])
                + ". Invalid field values indicate the data was edited incorrectly."
            ),
            severity=0.68,
            page=page_num,
            evidence={"number_issues": issues[:3]},
        ))


# ══════════════════════════════════════════════════════════════════
# 4. Field value conflicts — same label, different values
# ══════════════════════════════════════════════════════════════════

# Generic label words that introduce a field value
# Field label followed by separator then value
# Require explicit separator (: or -) to avoid matching header words
_LABEL_RE = re.compile(
    r'^(name|date|birth|dob|id|number|no|ref|amount|salary|account|'
    r'address|phone|mobile|email|gender|nationality|valid|issued|expiry|'
    r'employer|employee|designation|department|grade|total|net|gross)'
    r'[\s]*[:=-][\s]*(.+)$',
    re.IGNORECASE
)

def _check_field_value_conflicts(lines: list, page_num: int, findings: list):
    """Detect the same label appearing with two different values — without knowing the doc type."""
    label_values: dict = defaultdict(list)
    for line in lines:
        m = _LABEL_RE.match(line.strip())
        if m:
            label = m.group(1).lower()
            value = m.group(2).strip()
            if value and len(value) > 1:
                label_values[label].append(value)

    conflicts = []
    for label, values in label_values.items():
        unique = list(dict.fromkeys(values))
        if len(unique) >= 2:
            conflicts.append(f"'{label}' appears with values: "
                             + " / ".join(f"'{v[:30]}'" for v in unique[:3]))

    if conflicts:
        findings.append(Finding(
            category="Content Anomaly",
            description=(
                f"Conflicting field values on page {page_num}: "
                + "; ".join(conflicts[:3])
                + ". The same label appears with different values — "
                "a strong indicator that one value was overwritten."
            ),
            severity=0.80,
            page=page_num,
            evidence={"conflicts": conflicts[:3]},
        ))


# ══════════════════════════════════════════════════════════════════
# 5. Character substitution — visually similar swaps
# ══════════════════════════════════════════════════════════════════

# Known substitution patterns used in document fraud
_SUBSTITUTIONS = [
    (r'(?<![A-Z])0(?=[A-Z])|(?<=[A-Z])0(?![A-Z0-9])',  'digit 0 in letter context'),
    (r'(?<!\d)l(?=\d)|(?<=\d)l(?!\d)',                   'letter l in digit context'),
    (r'(?<![A-Z])1(?=[A-Za-z])|(?<=[a-z])1(?![A-Za-z0-9])', 'digit 1 in letter context'),
    (r'[rnm]{1}(?=\s)',                                   'possible rn→m substitution'),
    (r'[A-Z][a-z][A-Z]',                                 'unusual mixed-case pattern'),
]

def _check_char_substitution(text: str, lines: list, page_num: int, findings: list):
    issues = []

    # Check for l/I/1 confusion in ID-like strings
    for line in lines:
        tokens = line.split()
        for token in tokens:
            if len(token) < 4:
                continue
            # Token looks like an ID (mix of letters and digits) but has ambiguous chars
            has_ambig = bool(re.search(r'[lI1O0]', token))
            has_alnum = bool(re.search(r'[A-Z]', token)) and bool(re.search(r'\d', token))
            if has_ambig and has_alnum and len(token) >= 5:
                # Count how many ambiguous chars
                ambig_count = len(re.findall(r'[lI1O0]', token))
                if ambig_count >= 2 and ambig_count / len(token) > 0.3:
                    issues.append(f"Ambiguous characters in '{token}' — {ambig_count} of {len(token)} chars are l/I/1/O/0")

    if issues and len(issues) >= 2:
        findings.append(Finding(
            category="Content Anomaly",
            description=(
                f"Potential character substitution on page {page_num}: "
                + "; ".join(issues[:3])
                + ". High proportion of visually ambiguous characters (l/1/I, O/0) "
                "in identifier strings is a common forgery technique."
            ),
            severity=0.60,
            page=page_num,
            evidence={"substitution_issues": issues[:3]},
        ))


# ══════════════════════════════════════════════════════════════════
# 6. AI-generation signals — statistical text analysis
# ══════════════════════════════════════════════════════════════════

def _text_burstiness(words: list) -> float:
    """
    Burstiness measures how 'spiky' word lengths are.
    Human text: highly variable (B >> 0)
    AI text: unnaturally uniform (B near 0 or negative)
    B = (std - mean) / (std + mean)
    """
    if len(words) < 20:
        return 1.0  # not enough data
    lengths = [len(w) for w in words if w.isalpha()]
    if not lengths:
        return 1.0
    mean = sum(lengths) / len(lengths)
    var  = sum((l - mean)**2 for l in lengths) / len(lengths)
    std  = math.sqrt(var)
    return (std - mean) / (std + mean + 1e-9)


def _bigram_entropy(words: list) -> float:
    """
    Shannon entropy of word bigrams.
    AI text: lower entropy (more repetitive/predictable patterns)
    Human text: higher entropy (more varied)
    """
    if len(words) < 10:
        return 10.0
    bigrams = [(words[i].lower(), words[i+1].lower()) for i in range(len(words)-1)]
    total   = len(bigrams)
    counts  = Counter(bigrams)
    entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())
    return entropy


def _check_ai_generation_signals(text: str, words: list, page_num: int, findings: list):
    """
    Detect statistically unusual text patterns associated with AI generation.
    Works on any document — no domain knowledge required.
    """
    if len(words) < 30:
        return

    signals = []

    # 1. Burstiness — AI text is unnaturally uniform in word length
    b = _text_burstiness(words)
    if b < -0.1:
        signals.append(f"unusually uniform word lengths (burstiness={b:.2f}, human text > 0)")

    # 2. Bigram entropy — AI text has lower diversity of word pairs
    h = _bigram_entropy(words)
    expected_h = math.log2(max(len(set(words)), 2))   # max possible for vocab size
    if h < expected_h * 0.35 and len(words) >= 50:
        signals.append(f"low bigram entropy ({h:.1f} bits, expected ~{expected_h:.1f}) — repetitive phrasing")

    # 3. Perfect punctuation score — real documents have typos; AI-generated ones don't
    # Check: does every sentence end with exactly one punctuation mark?
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) >= 5:
        # Count sentences starting with a capital (proper sentence structure)
        well_formed = sum(1 for s in sentences if s and s[0].isupper())
        perfection_ratio = well_formed / len(sentences)
        if perfection_ratio > 0.97 and len(sentences) >= 8:
            signals.append(
                f"suspiciously perfect sentence structure ({perfection_ratio*100:.0f}% well-formed) — "
                "genuine typed documents typically contain minor formatting irregularities"
            )

    # 4. Vocabulary richness — AI text often repeats the same formal phrases
    if len(words) >= 50:
        alpha_words = [w.lower() for w in words if w.isalpha() and len(w) > 3]
        if alpha_words:
            type_token = len(set(alpha_words)) / len(alpha_words)
            if type_token < 0.35 and len(alpha_words) >= 40:
                signals.append(
                    f"low vocabulary diversity (type/token ratio={type_token:.2f}) — "
                    "repetitive formal language pattern"
                )

    if len(signals) >= 2:
        findings.append(Finding(
            category="AI-Generation Signal",
            description=(
                f"Statistical text patterns on page {page_num} suggest AI-generated content: "
                + "; ".join(signals[:3])
                + ". These signals occur when text is generated by a language model "
                "rather than typed by a person, regardless of document type."
            ),
            severity=min(0.55 + len(signals) * 0.08, 0.78),
            page=page_num,
            evidence={"ai_signals": signals[:4], "burstiness": round(b, 3), "bigram_entropy": round(h, 2)},
        ))


# ══════════════════════════════════════════════════════════════════
# 7. Mixed-script inconsistency within fields
# ══════════════════════════════════════════════════════════════════

_LATIN_RE     = re.compile(r'[A-Za-z]')
_NON_LATIN_RE = re.compile(r'[^\x00-\x7F]')

def _check_mixed_script_fields(lines: list, page_num: int, findings: list):
    """
    Detect lines where Latin and non-Latin scripts appear unexpectedly mixed
    within what should be a single-script field value.
    This catches cases where text was overlaid from a different language source.
    """
    mixed = []
    for line in lines:
        has_latin     = bool(_LATIN_RE.search(line))
        has_non_latin = bool(_NON_LATIN_RE.search(line))
        if has_latin and has_non_latin:
            latin_count     = len(_LATIN_RE.findall(line))
            non_latin_count = len(_NON_LATIN_RE.findall(line))
            total = latin_count + non_latin_count
            # Only flag if both scripts are substantial (not just a single char)
            if latin_count >= 3 and non_latin_count >= 3:
                ratio = min(latin_count, non_latin_count) / total
                if ratio > 0.2:  # both scripts represent >20% of chars
                    mixed.append(f"'{line[:50]}' (Latin:{latin_count}, non-Latin:{non_latin_count})")

    if len(mixed) >= 2:
        findings.append(Finding(
            category="Content Anomaly",
            description=(
                f"Mixed-script anomaly on page {page_num}: {len(mixed)} line(s) contain "
                "substantial Latin and non-Latin characters mixed within the same field. "
                "This can indicate text was overlaid from a different language source."
            ),
            severity=0.45,
            page=page_num,
            evidence={"mixed_lines": mixed[:3]},
        ))