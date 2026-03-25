from __future__ import annotations
import re
import math
from collections import Counter, defaultdict
from typing import List
from ..engine import Finding

AI_SYNTHETIC_FONT_PATTERNS = [
    r"helvetica\+?neue", r"inter", r"roboto", r"noto",
    r"sourcesans", r"source\s*sans", r"opensans",
    r"arimo", r"tinos", r"cousine",
]

BOLD_KEYWORDS   = re.compile(r'(?:^|[-])(?:Bold|Black|Heavy|ExtraBold|SemiBold|Demi)', re.I)
NORMAL_KEYWORDS = re.compile(r'(?:^|[-])(?:Regular|Roman|Light|Thin|Medium)$|^(?:Helvetica|Arial|Times|Courier|Verdana|Georgia)$', re.I)

SIZE_OUTLIER_RATIO = 2.0
Y_BUCKET_PT        = 8


def detect(reader) -> List[Finding]:
    findings: List[Finding] = []

    plumber_ok = False
    try:
        import pdfplumber
        plumber_ok = True
    except ImportError:
        pass

    for page_num, page in enumerate(reader.pages, 1):
        try:
            fonts = _extract_page_fonts(page)
            if not fonts:
                continue
            _check_font_mixing(fonts, page_num, findings)
            _check_non_embedded(fonts, page_num, findings)
            _check_synthetic_fonts(fonts, page_num, findings)
            _check_unusual_encoding(fonts, page_num, findings)
            _check_bold_weight_descriptors(fonts, page_num, findings)
        except Exception:
            pass

    if plumber_ok:
        _plumber_weight_analysis(reader, findings)
        _plumber_spatial_font_analysis(reader, findings)  # ← NEW: catches "kshitij lama" style insertions

    return findings


# ══════════════════════════════════════════════════════════════════════════
# pypdf structural font checks
# ══════════════════════════════════════════════════════════════════════════

def _extract_page_fonts(page) -> list:
    fonts = []
    try:
        resources = page.get("/Resources")
        if not resources:
            return fonts
        resources = resources.get_object() if hasattr(resources, "get_object") else resources
        font_dict = resources.get("/Font")
        if not font_dict:
            return fonts
        font_dict = font_dict.get_object() if hasattr(font_dict, "get_object") else font_dict

        for key in font_dict:
            try:
                fobj = font_dict[key]
                fobj = fobj.get_object() if hasattr(fobj, "get_object") else fobj
                base_font = str(fobj.get("/BaseFont", key) or key)

                stem_v, font_weight, cap_height = 0.0, 0.0, 0.0
                fd_ref = fobj.get("/FontDescriptor")
                if fd_ref:
                    fd = fd_ref.get_object() if hasattr(fd_ref, "get_object") else fd_ref
                    try: stem_v      = float(fd.get("/StemV",     0) or 0)
                    except: pass
                    try: font_weight = float(fd.get("/FontWeight", 0) or 0)
                    except: pass
                    try: cap_height  = float(fd.get("/CapHeight",  0) or 0)
                    except: pass

                fonts.append({
                    "name":        base_font,
                    "subtype":     str(fobj.get("/Subtype", "") or ""),
                    "encoding":    str(fobj.get("/Encoding", "") or ""),
                    "embedded":    "/FontDescriptor" in fobj,
                    "subset":      bool(re.match(r"^[A-Z]{6}\+", base_font)),
                    "stem_v":      stem_v,
                    "font_weight": font_weight,
                    "cap_height":  cap_height,
                })
            except Exception:
                pass
    except Exception:
        pass
    return fonts


def _check_bold_weight_descriptors(fonts: list, page_num: int, findings: list):
    for f in fonts:
        sv = f["stem_v"]; fw = f["font_weight"]
        if sv > 160 or fw > 700:
            findings.append(Finding(
                category="Unnatural Text Weight",
                description=(
                    f"Font '{f['name']}' has heavy stroke weight "
                    f"(StemV={sv:.0f}, FontWeight={fw:.0f}). "
                    "Artificially bold text is common when individual fields "
                    "are replaced in identity documents (ID cards, certificates, payslips)."
                ),
                severity=0.55,
                page=page_num,
                evidence={"font": f["name"], "stem_v": sv, "font_weight": fw},
            ))


def _check_font_mixing(fonts: list, page_num: int, findings: list):
    families = set()
    for f in fonts:
        name = re.sub(r"^[A-Z]{6}\+", "", f["name"], flags=re.IGNORECASE).lower()
        families.add(re.split(r"[-,]", name)[0])
    if len(families) > 6:
        findings.append(Finding(
            category="AI-Pattern Font Substitution",
            description=f"Page uses {len(families)} distinct font families. "
                        "Unusually high mixing may indicate text replacement or AI-assisted editing.",
            severity=0.45,
            page=page_num,
            evidence={"font_families": list(families)[:10]},
        ))


def _check_non_embedded(fonts: list, page_num: int, findings: list):
    ne = [f["name"] for f in fonts if not f["embedded"]]
    if len(ne) > 3:
        findings.append(Finding(
            category="AI-Pattern Font Substitution",
            description=f"{len(ne)} fonts are not embedded. Rendering may differ across viewers, "
                        "which can conceal textual changes.",
            severity=0.3,
            page=page_num,
            evidence={"non_embedded_fonts": ne[:5]},
        ))


def _check_synthetic_fonts(fonts: list, page_num: int, findings: list):
    for f in fonts:
        name = f["name"].lower()
        for pattern in AI_SYNTHETIC_FONT_PATTERNS:
            if re.search(pattern, name):
                findings.append(Finding(
                    category="AI-Pattern Font Substitution",
                    description=f"Font '{f['name']}' is commonly used by AI/automated PDF generators.",
                    severity=0.35,
                    page=page_num,
                    evidence={"font": f["name"]},
                ))
                break


def _check_unusual_encoding(fonts: list, page_num: int, findings: list):
    encodings = [f["encoding"] for f in fonts if f["encoding"]]
    enc_count = Counter(encodings)
    if len(enc_count) > 3:
        findings.append(Finding(
            category="Unusual Encoding",
            description=f"Page uses {len(enc_count)} different font encodings. "
                        "Mixed encodings often appear when text has been selectively replaced.",
            severity=0.4,
            page=page_num,
            evidence={"encodings": dict(enc_count)},
        ))


# ══════════════════════════════════════════════════════════════════════════
# pdfplumber visual weight forensics
# ══════════════════════════════════════════════════════════════════════════

def _plumber_weight_analysis(reader, findings: list):
    try:
        import pdfplumber, io
        from pypdf import PdfWriter

        buf = io.BytesIO()
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        writer.write(buf)
        buf.seek(0)

        with pdfplumber.open(buf) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                chars = page.chars
                if not chars:
                    continue
                _check_field_replacement_pattern(chars, page_num, findings)
                _check_mixed_line_weights(chars, page_num, findings)
                _check_size_outliers(chars, page_num, findings)
    except Exception:
        pass


def _build_line_buckets(chars: list) -> dict:
    buckets: dict[int, list] = defaultdict(list)
    for c in chars:
        bucket = round(float(c.get("top", 0)) / Y_BUCKET_PT) * Y_BUCKET_PT
        buckets[bucket].append(c)
    return buckets


def _check_field_replacement_pattern(chars: list, page_num: int, findings: list):
    buckets = _build_line_buckets(chars)
    sorted_ys = sorted(buckets.keys())

    normal_buckets = set()
    bold_buckets   = set()

    for y in sorted_ys:
        line_chars = buckets[y]
        if len(line_chars) < 2:
            continue
        fns = set(c.get("fontname", "") for c in line_chars)
        has_bold   = any(BOLD_KEYWORDS.search(fn)   for fn in fns)
        has_normal = any(NORMAL_KEYWORDS.search(fn) for fn in fns)

        if has_bold and not has_normal:
            bold_buckets.add(y)
        elif has_normal and not has_bold:
            normal_buckets.add(y)

    alternating_pairs = 0
    bold_sample_texts = []
    for ny in sorted(normal_buckets):
        for by in bold_buckets:
            gap = abs(by - ny)
            if 0 < gap <= 20:
                alternating_pairs += 1
                sample = "".join(c.get("text", "") for c in buckets[by])[:30].strip()
                if sample:
                    bold_sample_texts.append(sample)
                break

    if alternating_pairs >= 2:
        sample_display = ", ".join(f"'{t}'" for t in bold_sample_texts[:3])
        findings.append(Finding(
            category="Unnatural Text Weight",
            description=(
                f"Field-replacement pattern detected on page {page_num}: "
                f"{alternating_pairs} label/value line pairs where labels are normal-weight "
                f"and values are bold. This alternating pattern strongly indicates individual "
                f"field values were overwritten (e.g. name, date, ID number). "
                f"Bold values: {sample_display}."
            ),
            severity=0.72,
            page=page_num,
            evidence={
                "alternating_pairs": alternating_pairs,
                "bold_value_samples": bold_sample_texts[:5],
                "sample_text": bold_sample_texts[0] if bold_sample_texts else "",
            },
        ))


def _check_mixed_line_weights(chars: list, page_num: int, findings: list):
    buckets  = _build_line_buckets(chars)
    mixed_lines = 0
    examples    = []

    for y, line_chars in sorted(buckets.items()):
        if len(line_chars) < 4:
            continue
        fns       = set(c.get("fontname", "") for c in line_chars)
        has_bold  = any(BOLD_KEYWORDS.search(fn)   for fn in fns)
        has_norm  = any(NORMAL_KEYWORDS.search(fn) for fn in fns)
        if has_bold and has_norm:
            mixed_lines += 1
            sample = "".join(c.get("text", "") for c in line_chars)[:40].strip()
            examples.append(sample)

    if mixed_lines >= 2:
        findings.append(Finding(
            category="Unnatural Text Weight",
            description=(
                f"{mixed_lines} lines on page {page_num} mix bold and normal-weight fonts "
                "within the same line. This is uncommon in genuine documents and strongly "
                "suggests selective character/word replacement — typical in tampered "
                "ID cards, certificates, salary slips, or bank statements. "
                f"Sample lines: {'; '.join(repr(e) for e in examples[:3])}."
            ),
            severity=0.70,
            page=page_num,
            evidence={
                "mixed_weight_lines": mixed_lines,
                "sample_text": examples[0] if examples else "",
            },
        ))


def _check_size_outliers(chars: list, page_num: int, findings: list):
    sizes = [c.get("size", 0) for c in chars if c.get("size", 0) > 0]
    if len(sizes) < 6:
        return

    sizes_sorted = sorted(sizes)
    median_size  = sizes_sorted[len(sizes_sorted) // 2]
    if median_size < 4:
        return

    threshold    = median_size * SIZE_OUTLIER_RATIO
    large_chars  = [c for c in chars if c.get("size", 0) > threshold]
    large_ratio  = len(large_chars) / len(chars)

    if len(large_chars) >= 3 and large_ratio < 0.35:
        large_bold = [c for c in large_chars if BOLD_KEYWORDS.search(c.get("fontname", ""))]
        large_text = "".join(c.get("text", "") for c in large_chars[:50]).strip()
        bold_bonus = 0.1 if len(large_bold) > len(large_chars) * 0.5 else 0.0

        findings.append(Finding(
            category="Unnatural Text Weight",
            description=(
                f"Font-size outliers on page {page_num}: {len(large_chars)} characters "
                f"rendered at >{threshold:.0f}pt vs page median {median_size:.0f}pt "
                f"({large_ratio*100:.0f}% of all chars). "
                + ("These outlier characters are also bold, doubling the suspicion. " if bold_bonus else "")
                + f"Oversized text: '{large_text[:60]}'. "
                "Selective font enlargement is typical when a name, number, or date "
                "is replaced with content in a different size."
            ),
            severity=min(0.68 + bold_bonus, 0.78),
            page=page_num,
            evidence={
                "median_size":   round(median_size, 1),
                "threshold":     round(threshold, 1),
                "outlier_count": len(large_chars),
                "large_bold":    len(large_bold),
                "sample_text":   large_text[:60],
            },
        ))


# ══════════════════════════════════════════════════════════════════════════
# NEW: Spatial font-family comparison (catches "kshitij lama" / "gurgaon")
# ══════════════════════════════════════════════════════════════════════════

def _plumber_spatial_font_analysis(reader, findings: list):
    """
    Compares font families of individual words/tokens against the dominant
    font family on the page.  Tokens whose font family is unique (appears
    only once or twice while all other text uses a different family) are
    flagged as likely insertions — even when they're not bold.

    This is the key fix for catching names like "kshitij lama" and "gurgaon"
    inserted into a Medical Certificate: the form itself uses one font
    (e.g. Times / Helvetica) but the typed-over names use a different one
    (e.g. Arial / Calibri).
    """
    try:
        import pdfplumber, io
        from pypdf import PdfWriter

        buf = io.BytesIO()
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        writer.write(buf)
        buf.seek(0)

        with pdfplumber.open(buf) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                chars = page.chars
                if not chars or len(chars) < 10:
                    continue

                try:
                    _detect_foreign_font_words(chars, page_num, findings)
                    _detect_form_field_insertions(chars, page_num, findings)  # 🔥 NEW LINE
                except Exception:
                    pass
    except Exception:
        pass


def _simplify_family(fontname: str) -> str:
    """Strip subset prefix and weight/style suffix → base family name."""
    name = re.sub(r"^[A-Z]{6}\+", "", fontname or "")
    name = re.split(r"[-,]", name)[0].lower().strip()
    return name


def _detect_foreign_font_words(chars: list, page_num: int, findings: list):
    """
    Group characters into words by X proximity, then find words whose
    font family differs from the page-dominant family.
    """
    # Step 1: assign simplified family to every char
    for c in chars:
        c["_family"] = _simplify_family(c.get("fontname", ""))

    # Step 2: find dominant family (most common non-empty family)
    family_counter: Counter = Counter(
        c["_family"] for c in chars if c["_family"]
    )
    if not family_counter:
        return
    dominant_family, dominant_count = family_counter.most_common(1)[0]
    total_chars = sum(family_counter.values())

    # Only proceed if dominant family is clear (≥50% of chars)
    if dominant_count / total_chars < 0.30:
        return

    # Step 3: group chars into word-level tokens by line (Y-bucket) + X gap
    words = _chars_to_words(chars)

    # Step 4: find words that are entirely in a non-dominant family
    foreign_words = []
    for word in words:
        wfamilies = Counter(c["_family"] for c in word["chars"] if c["_family"])
        if not wfamilies:
            continue
        word_family = wfamilies.most_common(1)[0][0]
        if word_family and word_family != dominant_family:
            foreign_words.append({
                "text":   word["text"],
                "family": word_family,
                "x":      word["x"],
                "y":      word["y"],
                "size":   word["size"],
            })

    # Step 5: filter out very short tokens (single letters, punctuation)
    foreign_words = [w for w in foreign_words if len(w["text"].strip()) >= 2
                     and any(c.isalnum() for c in w["text"])]

    if not foreign_words:
        return

    # Step 6: only flag if foreign words are a minority (< 25% of all words)
    all_words = _chars_to_words(chars)
    if len(foreign_words) / max(len(all_words), 1) > 0.40:
        return

    # Build heatmap rects
    rects = []
    for w in foreign_words[:12]:
        sz = w["size"] or 10
        rects.append({
            "text": w["text"],
            "x":    max(0, w["x"] - 4),
            "y":    max(0, w["y"] - 4),
            "w":    len(w["text"]) * sz * 0.65 + 8,
            "h":    sz * 1.3 + 8,
        })

    word_texts = [w["text"] for w in foreign_words]
    families_seen = list({w["family"] for w in foreign_words})

    severity = min(0.62 + 0.05 * len(foreign_words), 0.85)

    findings.append(Finding(
        category="Text Layer Inserted",
        description=(
            f"Page {page_num}: {len(foreign_words)} word(s) use a different font family "
            f"({', '.join(families_seen[:3])}) from the rest of the document "
            f"({dominant_family}). "
            f"This strongly suggests these words were typed or pasted in after the "
            f"original document was created. "
            f"Suspicious words: {', '.join(repr(t) for t in word_texts[:6])}."
        ),
        severity=severity,
        page=page_num,
        evidence={
            "word_rects":         rects,
            "bold_words":         word_texts[:6],
            "dominant_family":    dominant_family,
            "foreign_families":   families_seen[:4],
            "foreign_word_count": len(foreign_words),
            "detail": (
                f"Foreign font words: {', '.join(repr(t) for t in word_texts[:6])} "
                f"| Page dominant font: {dominant_family}"
            ),
        },
    ))

def _detect_form_field_insertions(chars: list, page_num: int, findings: list):
    """
    Detect words written over blank form lines (______).
    This catches perfectly matched font insertions.
    """

    words = _chars_to_words(chars)

    suspicious = []

    for w in words:
        text = w["text"].strip()

        # Skip small or useless tokens
        if len(text) < 3:
            continue

     

        y = w["y"]

        # Look for underline / blank-line structure nearby
        underline_detected = False

        for c in chars:
            same_line = abs(float(c.get("top", 0)) - y) < 4

            if same_line:
                t = c.get("text", "")

                # Detect underscores or long empty spacing
                if "_" in t or t.strip() == "":
                    underline_detected = True
                    break

        if underline_detected:
            suspicious.append(w)

    if not suspicious:
        return

    rects = []
    texts = []

    for w in suspicious[:12]:
        sz = w["size"] or 10
        rects.append({
            "text": w["text"],
            "x": max(0, w["x"] - 4),
            "y": max(0, w["y"] - 4),
            "w": len(w["text"]) * sz * 0.6 + 8,
            "h": sz * 1.3 + 8,
        })
        texts.append(w["text"])

    findings.append(Finding(
        category="Form Field Insertion",
        description=(
            f"Page {page_num}: Text appears on structured blank/underline fields. "
            f"This strongly indicates manually inserted values in a form. "
            f"Detected values: {', '.join(repr(t) for t in texts[:6])}."
        ),
        severity=0.82,
        page=page_num,
        evidence={
            "word_rects": rects,
            "inserted_words": texts[:10],
            "detail": f"Form-filled words: {texts[:6]}"
        },
    ))


def _chars_to_words(chars: list, x_gap_pt: float = 6.0, y_tol: float = 4.0) -> list:
    """
    Naively group pdfplumber chars into word tokens.
    Two chars belong to the same word if they're on the same Y-bucket and
    their X positions are within x_gap_pt of each other.
    """
    if not chars:
        return []

    # Sort by Y then X
    sorted_chars = sorted(chars, key=lambda c: (round(float(c.get("top", 0)) / y_tol), float(c.get("x0", 0))))

    words = []
    current_word_chars = [sorted_chars[0]]

    for c in sorted_chars[1:]:
        prev = current_word_chars[-1]
        same_line = abs(float(c.get("top", 0)) - float(prev.get("top", 0))) < y_tol * 2
        x_close   = float(c.get("x0", 0)) - float(prev.get("x1", prev.get("x0", 0))) < x_gap_pt

        if same_line and x_close:
            current_word_chars.append(c)
        else:
            words.append(_make_word(current_word_chars))
            current_word_chars = [c]

    if current_word_chars:
        words.append(_make_word(current_word_chars))

    return words


def _make_word(chars: list) -> dict:
    text = "".join(c.get("text", "") for c in chars).strip()
    sizes = [float(c.get("size", 0)) for c in chars if c.get("size", 0)]
    return {
        "text":  text,
        "chars": chars,
        "x":     float(chars[0].get("x0", 0)),
        "y":     float(chars[0].get("top", 0)),
        "size":  (sum(sizes) / len(sizes)) if sizes else 10.0,
    }