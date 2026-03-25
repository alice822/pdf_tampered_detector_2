"""
Text Layer Detector — (v3 — foreign font + form field insertion aware)
-------------------------------------------------------------
FIXED: Previously missed tampered fields like "kshitij lama" / "gurgaon"
because dominant_families threshold was too loose (cnt >= 2) allowing
inserted words' fonts to be treated as dominant.

Now detects:
  CASE A: Scanned image page → any text object = inserted (original logic)
  CASE B: Native PDF page   → text objects that are spatially anomalous:
          - Font family is minority (<30% of page tokens)  ← KEY FIX
          - Unusual font size
          - Out-of-baseline position
          - Bold on normal line
          - Single foreign-font token is now enough to trigger a finding

No OCR. Deterministic. Works on PAN, Aadhaar, Medical certs, payslips.
"""
from __future__ import annotations
import re
import sys
from collections import Counter
from typing import List
from ..engine import Finding

# Bold/heavy font name keywords
BOLD_RE = re.compile(r'(?:^|[-])(?:Bold|Black|Heavy|ExtraBold|SemiBold|Demi)', re.I)

# Minimum characters for a "real" text token
MIN_TEXT_LEN = 2

# How far (in points) a text object's Y must deviate from its nearest
# form-line cluster to be considered "floating" (out-of-baseline)
BASELINE_DEVIATION_PT = 6.0

# A font whose size is > this multiple of the page median is a size outlier
SIZE_OUTLIER_RATIO = 1.8

# A font family must represent at least this fraction of page tokens
# to be considered "dominant" (KEY FIX — was cnt >= 2 before)
DOMINANT_FAMILY_RATIO = 0.30


def detect(reader) -> List[Finding]:
    findings: List[Finding] = []

    for page_num, page in enumerate(reader.pages, 1):
        try:
            resources = page.get("/Resources", {})
            if resources and hasattr(resources, "get_object"):
                resources = resources.get_object()

            has_image = bool(resources.get("/XObject") if resources else False)

            # Collect all text objects with position + font info
            page_texts = _extract_text_objects(page, page_num)
            real = [t for t in page_texts
                    if len(t["text"].strip()) >= MIN_TEXT_LEN
                    and any(c.isalnum() for c in t["text"])]

            if not real:
                continue

            # AFTER
            if has_image:
                _emit_image_page_finding(real, page_num, page, findings)
            _check_native_pdf_insertions(real, page_num, findings)  # always run

        except Exception:
            pass

    return findings


# ══════════════════════════════════════════════════════════════════════════
# CASE A helper — scanned image logic
# ══════════════════════════════════════════════════════════════════════════

def _emit_image_page_finding(real: list, page_num: int, page, findings: list):
    rects = []
    for obj in real[:12]:
        est_w = len(obj["text"]) * obj["size"] * 0.65
        est_h = obj["size"] * 1.3
        rects.append({
            "text": obj["text"],
            "x":    max(0, obj["x"] - 4),
            "y":    max(0, obj["y"] - 4),
            "w":    est_w + 8,
            "h":    est_h + 8,
        })

    findings.append(Finding(
        category="Text Layer Inserted",
        description=(
            f"Page {page_num}: Found {len(real)} text object(s) "
            f"embedded directly over a scanned image. "
            f"Genuine scanned documents contain no text layer — "
            f"text objects are only present when content has been "
            f"digitally added using a PDF editor. "
            f"Detected text: "
            + ", ".join(f"'{t['text']}'" for t in real[:6])
        ),
        severity=0.92,
        page=page_num,
        evidence={
            "text_objects": real[:12],
            "text_count":   len(real),
            "word_rects":   rects,
            "bold_words":   [t["text"] for t in real[:6]],
            "detail":       f"{len(real)} text objects over image: "
                            + ", ".join(f"'{t['text']}'" for t in real[:4]),
        },
    ))


# ══════════════════════════════════════════════════════════════════════════
# CASE B — native PDF anomaly detection
# ══════════════════════════════════════════════════════════════════════════

def _check_native_pdf_insertions(real: list, page_num: int, findings: list):
    """
    On a native PDF (no image XObject), flag tokens that are anomalous
    compared to the page baseline:

      Signal 1: Font-family outlier — font represents <30% of page tokens
      Signal 2: Size outlier        — much larger/smaller than page median
      Signal 3: Floating baseline   — Y doesn't cluster with other text lines
      Signal 4: Bold-on-normal-line — bold while surrounding line is not

    KEY FIXES vs v2:
      - dominant_families now requires >=30% share (not just cnt>=2)
      - A single foreign-font token is enough to emit a finding
      - Debug output includes dominant_families + family_counter
    """
    if len(real) < 3:
        return

    # ── Build page-level baselines ────────────────────────────────────────
    font_families = _font_families(real)
    family_counter = Counter(font_families[_key(t)] for t in real)
    total_tokens = len(real)

    # KEY FIX: dominant = font that covers >=30% of tokens, not just seen twice
    dominant_families = {
        fam for fam, cnt in family_counter.items()
        if cnt / total_tokens >= DOMINANT_FAMILY_RATIO
    }

    sizes = [t["size"] for t in real if t["size"] > 0]
    median_size = _median(sizes) if sizes else 10.0

    y_clusters = _cluster_ys([t["y"] for t in real], tol=4.0)

    # ── Score each token ──────────────────────────────────────────────────
    suspicious_tokens = []
    for t in real:
        signals = []
        fam = font_families.get(_key(t), "")

        # Signal 1: font family not dominant on page
        if fam and dominant_families and fam not in dominant_families:
            signals.append("foreign-font")

        # Signal 2: font size outlier
        if median_size > 0 and t["size"] > 0:
            ratio = t["size"] / median_size
            if ratio > SIZE_OUTLIER_RATIO or ratio < 0.5:
                signals.append(
                    f"size-outlier({t['size']:.1f}pt vs median {median_size:.1f}pt)"
                )

        # Signal 3: Y position doesn't align with any existing text line
        if y_clusters:
            nearest_dist = min(abs(t["y"] - cy) for cy in y_clusters)
            if nearest_dist > BASELINE_DEVIATION_PT and len(y_clusters) >= 3:
                signals.append(f"floating-baseline(off {nearest_dist:.1f}pt)")

        # Signal 4: bold token on a predominantly non-bold line
        same_line = [o for o in real if abs(o["y"] - t["y"]) < 4.0]
        if len(same_line) >= 2:
            tok_bold = bool(BOLD_RE.search(t.get("fontname", "")))
            line_bold_ratio = (
                sum(1 for o in same_line if BOLD_RE.search(o.get("fontname", "")))
                / len(same_line)
            )
            if tok_bold and line_bold_ratio < 0.4:
                signals.append("bold-on-normal-line")

        if signals:
            suspicious_tokens.append({**t, "signals": signals})

    # ── Debug output ──────────────────────────────────────────────────────
    words = [t["text"] for t in real]
    cands = [t["text"] for t in suspicious_tokens]
    print(
        f"[DEBUG] suspicious={len(cands)} words={len(words)} "
        f"cands={len(cands)} ratio={int(len(cands)/max(len(words),1)*100)}%",
        file=sys.stderr,
    )
    print(f"[DEBUG] suspicious_words={cands[:10]}", file=sys.stderr)
    print(f"[DEBUG] dominant_families={dominant_families}", file=sys.stderr)
    print(f"[DEBUG] family_counter={family_counter.most_common(8)}", file=sys.stderr)

    if not suspicious_tokens:
        return

    # ── Cluster suspicious tokens into insertion regions ──────────────────
    insertion_groups = _group_by_proximity(suspicious_tokens, x_tol=200, y_tol=20)

    for group in insertion_groups:
        strong = [t for t in group if len(t["signals"]) >= 2]
        has_foreign = any("foreign-font" in t["signals"] for t in group)

        # KEY FIX: a single foreign-font token is now enough to report
        if not (strong or len(group) >= 2 or has_foreign):
            continue

        texts = [t["text"] for t in group]
        all_signals = list({s for t in group for s in t["signals"]})
        severity = min(0.60 + 0.08 * len(strong), 0.88)
        # Boost severity if foreign-font is a signal
        if has_foreign:
            severity = min(severity + 0.05, 0.92)

        rects = []
        for t in group:
            est_w = len(t["text"]) * (t["size"] or 10) * 0.65
            est_h = (t["size"] or 10) * 1.3
            rects.append({
                "text": t["text"],
                "x":    max(0, t["x"] - 4),
                "y":    max(0, t["y"] - 4),
                "w":    est_w + 8,
                "h":    est_h + 8,
            })

        findings.append(Finding(
            category="Text Layer Inserted",
            description=(
                f"Page {page_num}: Text token(s) appear to have been inserted "
                f"into this native PDF — they are anomalous compared to the "
                f"rest of the document. "
                f"Inserted text: {', '.join(repr(t) for t in texts[:6])}. "
                f"Anomaly signals: {', '.join(all_signals[:4])}."
            ),
            severity=severity,
            page=page_num,
            evidence={
                "text_objects": group[:12],
                "text_count":   len(group),
                "word_rects":   rects,
                "bold_words":   texts[:6],
                "signals":      all_signals,
                "detail": (
                    f"Inserted tokens: {', '.join(repr(t) for t in texts[:4])} "
                    f"| Signals: {', '.join(all_signals[:4])}"
                ),
            },
        ))


# ══════════════════════════════════════════════════════════════════════════
# Text extraction (shared)
# ══════════════════════════════════════════════════════════════════════════

def _extract_text_objects(page, page_num: int) -> list:
    """Extract all text tokens with position, size, and font name."""
    page_texts = []

    def visitor(text, cm, tm, fontDict, fontSize):
        if not text or not text.strip():
            return
        t = text.strip()
        if len(t) < 1:
            return
        try:
            x  = float(tm[4])
            y  = float(tm[5])
            sz = float(fontSize) if fontSize else 0
            if sz < 4:
                return
            fname = ""
            if fontDict:
                try:
                    raw = fontDict.get("/BaseFont") or fontDict.get("/Name") or ""
                    fname = str(raw) if raw else ""
                    fname = re.sub(r"^[A-Z]{6}\+", "", fname)
                except Exception:
                    pass

            page_texts.append({
                "text":     t,
                "x":        round(x, 1),
                "y":        round(y, 1),
                "size":     round(sz, 1),
                "fontname": fname,
                "page":     page_num,
            })
        except Exception:
            pass

    try:
        page.extract_text(visitor_text=visitor)
    except Exception:
        pass

    return page_texts


# ══════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════

def _key(t: dict) -> str:
    return t["text"] + str(t["x"])


def _font_families(tokens: list) -> dict:
    result = {}
    for t in tokens:
        fname = t.get("fontname", "")
        family = re.split(r"[-,]", fname)[0].lower().strip() if fname else ""
        result[_key(t)] = family
    return result


def _median(values: list) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return (s[n // 2] + s[(n - 1) // 2]) / 2.0


def _cluster_ys(ys: list, tol: float = 4.0) -> list:
    if not ys:
        return []
    sorted_ys = sorted(set(ys))
    clusters = []
    current = [sorted_ys[0]]
    for y in sorted_ys[1:]:
        if y - current[-1] <= tol:
            current.append(y)
        else:
            clusters.append(sum(current) / len(current))
            current = [y]
    clusters.append(sum(current) / len(current))
    return clusters


def _group_by_proximity(tokens: list, x_tol: float, y_tol: float) -> list:
    if not tokens:
        return []
    groups = []
    used = [False] * len(tokens)
    for i, tok in enumerate(tokens):
        if used[i]:
            continue
        group = [tok]
        used[i] = True
        for j, other in enumerate(tokens):
            if used[j]:
                continue
            if (abs(other["x"] - tok["x"]) < x_tol
                    and abs(other["y"] - tok["y"]) < y_tol):
                group.append(other)
                used[j] = True
        groups.append(group)
    return groups