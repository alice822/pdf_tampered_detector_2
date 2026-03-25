"""
Image Text Detector  
---------------------------------------------------------------
Core insight: use HEADER and LABEL words as the reference baseline,
not all words. Tampered bold text is always injected as VALUE words,
while the original printed headers/labels remain at normal weight.

Algorithm:
  1. Run OCR on the rendered page image
  2. Exclude QR/barcode regions (max 8% image area, roughly square)
  3. Split words into:
       REFERENCE = headers + labels (known document structure text)
       CANDIDATES = value words (non-header, non-label, Latin, len>=3)
  4. Compute baseline = median stroke of REFERENCE words
  5. Flag candidates where stroke > baseline * 1.55
     OR height > baseline_height * 1.5
  6. Only fire if suspicious < 40% of candidates (minority check)

This separates original printed text (headers/labels) from
injected replacement text (names, dates, IDs in bold).
"""

#  (p25 baseline, real-word filter, dpi=250)
from __future__ import annotations
import io, math, re
from typing import List, Optional
from ..engine import Finding

try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image, ImageStat, ImageFilter
    OCR_OK = True
except ImportError:
    OCR_OK = False

MAX_PAGES = 4
DPI       = 300
MIN_CONF  = 40

# Words always in the header — document structure, printed at normal weight
HEADER_RE = re.compile(
    r'^(income|tax|department|govt\.?|government|india|permanent|account|number|'
    r'republic|ministry|authority|revenue|service|national|incom|deparment|'
    r'departement|goverment|indya|inida|inda)$',   # include common misspellings
    re.IGNORECASE
)

# Field label words — also normal weight in genuine documents
LABEL_RE = re.compile(
    r'^(name|date|birth|dob|signature|address|father|mother|gender|sex|'
    r'valid|issued|pan|serial|sr|no\.?|of|the|to|by|and|for|a|an|'
    r'हस्ताक्षर|जन्म|तिथि|नाम|स्थायी|लेखा|संख्या)$',
    re.IGNORECASE
)


def _preprocess_for_ocr(img):
    """
    Enhance image quality before OCR.
    Tries multiple preprocessing strategies and returns the best result
    (most words detected). This handles worn physical cards, low-contrast
    scans, and compressed photos.
    """
    from PIL import ImageOps, ImageFilter
    gray = img.convert("L")
    w, h = gray.size

    # Upscale if image is small
    if w < 1200 or h < 800:
        scale = max(1200/w, 800/h, 1.0)
        gray = gray.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    candidates = []

    # Strategy 1: autocontrast + unsharp (good for digital images)
    s1 = ImageOps.autocontrast(gray, cutoff=1)
    s1 = s1.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=2))
    candidates.append(s1)

    # Strategy 2: threshold at 120 (good for worn physical cards)
    # Separates dark text from light/worn background
    s2 = gray.point(lambda p: 255 if p > 120 else 0)
    candidates.append(s2)

    # Strategy 3: threshold at 140 (for very low contrast)
    s3 = gray.point(lambda p: 255 if p > 140 else 0)
    candidates.append(s3)

    import pytesseract
    best = s2  # default to threshold@120 — most reliable across OS versions
    best_count = 0
    for s in candidates:
        try:
            data = pytesseract.image_to_data(
                s, config='--psm 3 --oem 1 -l eng',
                output_type=pytesseract.Output.DICT
            )
            count = sum(
                1 for i in range(len(data['text']))
                if data['text'][i].strip()
                and str(data['conf'][i]).lstrip('-').isdigit()
                and int(data['conf'][i]) >= 40
                and len(data['text'][i].strip()) >= 3
            )
            if count > best_count:
                best_count = count
                best = s
        except Exception:
            pass

    return best


def detect(reader) -> List[Finding]:
    findings: List[Finding] = []
    if not OCR_OK:
        return findings

    # Run on ALL pages that contain image resources
    # Don't gate on embedded text count - iLovePDF and other tools often
    # add invisible text layers that would incorrectly skip OCR analysis
    image_pages = []
    for i, page in enumerate(reader.pages):
        try:
            resources = page.get("/Resources", {})
            has_image = bool(resources.get("/XObject"))
            embedded_words = len((page.extract_text() or "").split())
            is_scanned_image = has_image
            is_nearly_blank  = embedded_words < 8   # truly blank/image-only pages
            if is_scanned_image or is_nearly_blank:
                image_pages.append(i)
        except Exception:
            image_pages.append(i)

    if not image_pages:
        return findings

    try:
        from pypdf import PdfWriter
        buf = io.BytesIO()
        pw  = PdfWriter()
        for i in image_pages[:MAX_PAGES]:
            pw.add_page(reader.pages[i])
        pw.write(buf)
        images = convert_from_bytes(buf.getvalue(), dpi=DPI)

        for idx, img in enumerate(images):
            pn      = image_pages[idx] + 1
            mb      = reader.pages[image_pages[idx]].mediabox
            pdf_w   = float(mb.width)
            pdf_h   = float(mb.height)
            img_w, img_h = img.size
            sx, sy  = pdf_w / img_w, pdf_h / img_h
            try:
                _analyze_page(img, pn, pdf_w, pdf_h, sx, sy, findings)
            except Exception as e:
                pass
    except Exception:
        pass
    return findings


def _extract_native_image(reader, page_idx):
    """
    Extract the largest embedded image from a PDF page at native resolution.
    Returns PIL Image or None if extraction fails.
    This gives better OCR quality than rendering at a fixed DPI.
    """
    try:
        page = reader.pages[page_idx]
        resources = page.get('/Resources', {})
        xobjects = resources.get('/XObject', {}) if resources else {}
        best = None
        best_size = 0
        for name, ref in xobjects.items():
            try:
                obj = ref.get_object()
                if obj.get('/Subtype') != '/Image':
                    continue
                w = int(obj.get('/Width', 0))
                h = int(obj.get('/Height', 0))
                if w * h > best_size:
                    data = obj.get_data()
                    from PIL import Image as PILImage
                    img = PILImage.open(__import__('io').BytesIO(data))
                    img.load()  # verify it's valid
                    best = img.convert('RGB')
                    best_size = w * h
            except Exception:
                continue
        return best
    except Exception:
        return None


def _analyze_page(img, pn, pdf_w, pdf_h, sx, sy, findings):
    # Preprocess for better OCR on physical cards and worn documents
    img_proc = _preprocess_for_ocr(img)
    gray     = img_proc.convert("L") if img_proc.mode != "L" else img_proc

    # Adjust scale: OCR coords are in preprocessed image pixels
    # We need to convert them back to PDF points
    proc_sx = sx * (img.width  / img_proc.width)
    proc_sy = sy * (img.height / img_proc.height)

    qr = _detect_qr_regions(gray)

    data = pytesseract.image_to_data(
        img_proc, output_type=pytesseract.Output.DICT,
        config="--psm 3 --oem 1 -l eng"
    )
    words = _build_words(data, qr)
    if len(words) < 4:
        return

    # Compute stroke for every word
    for w in words:
        w["stroke"] = _mean_stroke(_crop(gray, w))

    # Split into reference (original doc text) and candidates (possibly tampered)
    # Clean text = alphanumeric only, for robust header matching
    # This catches OCR noise like '[AX' (clean='AX') or 'DEPARTMEN]' (clean='DEPARTMEN')
    def _clean(t):
        return re.sub(r'[^A-Za-z0-9]', '', t)

    def _is_header(w):
        txt = w["text"]
        clean = _clean(txt)
        # Direct match
        if HEADER_RE.match(txt) or LABEL_RE.match(txt):
            return True
        # Clean match (handles OCR noise like '[AX', 'DEPARTMEN]')
        if clean and (HEADER_RE.match(clean) or LABEL_RE.match(clean)):
            return True
        # Partial match: if clean text is substring of a known header word
        known = ['income','tax','department','permanent','account','number',
                 'government','india','signature','birth','name','date']
        cl = clean.lower()
        if len(cl) >= 4:
            for k in known:
                if cl in k or k in cl:
                    return True
        return False

    reference  = [w for w in words if _is_header(w) and w["stroke"] > 0]

    def _is_real_word(w):
        """Filter out OCR garbage like 'aor', 'mor', 'Chip', 'feta'"""
        txt = w["text"]
        clean = _clean(txt)
        if len(clean) < 3:
            return False
        # Must have reasonable confidence
        if w["conf"] < 45:
            return False
        # Real content words are either:
        # - ALL CAPS (PAN number, name in capitals): DMEPK4085C, RAHUL
        # - Title case (name): Narayan, Kumar
        # - Date/number pattern: 10/10/1980
        # - All digits or alphanumeric ID
        is_all_caps = clean == clean.upper() and any(c.isalpha() for c in clean) and len(clean) >= 4
        is_title    = txt[0].isupper() and len(txt) >= 4
        is_date_num = bool(re.match(r'^[\d/\-\.]+$', txt)) and len(txt) >= 6
        is_id_code  = bool(re.match(r'^[A-Z0-9]{5,}$', clean))
        if not (is_all_caps or is_title or is_date_num or is_id_code):
            return False
        # Skip known OCR noise patterns
        noise_patterns = re.compile(
            r'^(aor|mor|feta|Chip|chip|ane|wen|hed|uw|ue|oo|ii|ee|aa|ce|ae|'
            r'bi|az|gp|axa|pw|yx|ww|sa|re|da|ha|ma|na|ka|ga|ba|ta|la|pa|ja|'
            r'MOTORS|motors|eurd|terfea|Mates|ara|den|det|Nate|GOVE|ADM)$',
            re.IGNORECASE
        )
        if noise_patterns.match(txt):
            return False
        return True

    candidates = [w for w in words if
                  not _is_header(w)
                  and w["is_latin"]
                  and w["ww"] >= 10
                  and w["h"] >= 5
                  and _is_real_word(w)]

    if not candidates:
        return

    # Baseline from reference words; fall back to lower-half of all words
    import statistics as _stats
    if len(reference) >= 3:
        ref_strokes = sorted(w["stroke"] for w in reference if w["stroke"] > 0)
        ref_heights = sorted(w["h"]      for w in reference)
        import numpy as _np
        # Use 25th percentile — anchors to the THINNEST genuine words
        # (Date, Birth, Name labels) rather than thick ones (INCOME, TAX, DEPT)
        # This prevents baseline inflation that causes RAHUL/BYEPB to be missed
        baseline_sw = float(_np.percentile(ref_strokes, 25))
        baseline_h  = float(_np.percentile(ref_heights, 25))
        if baseline_sw < 0.5:
            baseline_sw = _stats.median(ref_strokes)  # fallback
    elif len(words) >= 6:
        # No clear headers — use lower half of all strokes as baseline
        all_sw = sorted(w["stroke"] for w in words if w["stroke"] > 0)
        half   = all_sw[:len(all_sw)//2]
        baseline_sw = _stats.median(half) if half else 0
        all_h  = sorted(w["h"] for w in words)
        baseline_h  = all_h[len(all_h)//4]
    else:
        return

    if baseline_sw < 0.5:
        return

    # Fewer reference words = less reliable baseline = use tighter multiplier
    thresh_mult = 1.40 if len(reference) <= 2 else 1.55
    thresh_sw = baseline_sw * thresh_mult
    thresh_h  = baseline_h  * 1.40  # height threshold slightly lower than stroke

    suspicious = []
    for w in candidates:
        sw_flag = w["stroke"] > thresh_sw
        h_flag  = w["h"] > thresh_h
        if sw_flag or h_flag:
            signals = []
            if sw_flag: signals.append(f"stroke {w['stroke']/baseline_sw:.1f}×")
            if h_flag:  signals.append(f"height {w['h']/baseline_h:.1f}×")
            score = w["stroke"] / baseline_sw
            suspicious.append((score, w, signals))

    if not suspicious:
        return

    # DEBUG — remove after fix
    import sys
    print(f"[DEBUG] suspicious={len(suspicious)} words={len(words)} cands={len(candidates)} ratio={len(suspicious)/max(len(words),1):.0%}", file=sys.stderr)
    print(f"[DEBUG] suspicious_words={[w['text'] for _,w,_ in suspicious]}", file=sys.stderr)

    # Minority check: suspicious must be minority of ALL words.
    # Exceptions:
    # 1. All candidates suspicious = strongest possible signal
    # 2. Very few words (≤10) = worn/degraded card, be lenient
    #    On such cards OCR misses many headers, ratio skews high
    # 3. Majority of candidates suspicious (≥75%) = strong signal
    all_candidates_suspicious = len(suspicious) >= len(candidates)
    most_candidates_suspicious = len(candidates) >= 2 and len(suspicious) >= len(candidates) * 0.75
    few_words_on_page = len(words) <= 15

    if len(suspicious) > len(words) * 0.55:
        if not (all_candidates_suspicious or most_candidates_suspicious or few_words_on_page):
            return

    suspicious.sort(reverse=True)
    texts  = [w["text"] for _, w, _ in suspicious[:6]]
    rects  = [_to_rect(w, pdf_w, pdf_h, proc_sx, proc_sy) for _, w, _ in suspicious[:6]]
    rects  = [r for r in rects if r]
    sample = " / ".join(f"'{t}'" for t in texts[:4])
    top_sig = suspicious[0][2]
    dual   = len(suspicious[0][2]) == 2

    findings.append(Finding(
        category="Unnatural Text Weight",
        description=(
            f"Visual text anomaly on page {pn}: "
            f"{len(suspicious)} word(s) have significantly heavier stroke "
            f"than the document baseline ({', '.join(top_sig)}). "
            f"Affected words: {sample}. "
            "In genuine documents all text shares the same print run — "
            "bold or heavy isolated words indicate inserted/replaced content."
        ),
        severity=min(0.68 + len(suspicious) * 0.04 + (0.05 if dual else 0), 0.88),
        page=pn,
        evidence={
            "bold_words":   texts,
            "sample_text":  texts[0] if texts else "",
            "word_rects":   rects,
            "baseline_sw":  round(baseline_sw, 2),
            "threshold_sw": round(thresh_sw, 2),
        },
    ))


# ─── QR / barcode region detection ────────────────────────────────

def _detect_qr_regions(gray):
    iw, ih   = gray.size
    max_area = iw * ih * 0.08   # real QR ≤ 8% of image
    min_area = 30000            # ~173×173 minimum — filters texture noise

    raw = []
    block = 50
    for gy in range(0, ih - block, block):
        for gx in range(0, iw - block, block):
            crop  = gray.crop((gx, gy, gx+block, gy+block))
            pxls  = list(crop.getdata())
            dark  = sum(1 for p in pxls if p < 80)
            density = dark / len(pxls)
            if 0.18 < density < 0.68:
                mean_p = sum(pxls) / len(pxls)
                if mean_p > 0:
                    var = sum((p-mean_p)**2 for p in pxls) / len(pxls)
                    if math.sqrt(var) / mean_p > 0.45:
                        raw.append((gx, gy, gx+block, gy+block))

    merged = _merge_capped(raw, max_area)
    return [(x0,y0,x1,y1) for x0,y0,x1,y1 in merged
            if min_area <= (x1-x0)*(y1-y0) <= max_area
            and 0.5 <= (x1-x0)/max(y1-y0,1) <= 2.0]  # tighter aspect ratio


def _merge_capped(regions, max_area, gap=30):
    if not regions:
        return []
    merged  = list(regions)
    changed = True
    while changed:
        changed = False
        out  = []
        used = [False] * len(merged)
        for i, (x0,y0,x1,y1) in enumerate(merged):
            if used[i]: continue
            for j, (ax0,ay0,ax1,ay1) in enumerate(merged):
                if i == j or used[j]: continue
                if ax0-gap<=x1 and ax1+gap>=x0 and ay0-gap<=y1 and ay1+gap>=y0:
                    nx0,ny0 = min(x0,ax0), min(y0,ay0)
                    nx1,ny1 = max(x1,ax1), max(y1,ay1)
                    if (nx1-nx0)*(ny1-ny0) <= max_area:
                        x0,y0,x1,y1 = nx0,ny0,nx1,ny1
                        used[j] = True; changed = True
            out.append((x0,y0,x1,y1)); used[i] = True
        merged = out
    return merged


def _word_in_qr(w, qr_regions, thresh=0.35):
    wx0=w["left"]; wy0=w["top"]
    wx1=wx0+w["ww"]; wy1=wy0+w["h"]
    area = max((wx1-wx0)*(wy1-wy0), 1)
    for (qx0,qy0,qx1,qy1) in qr_regions:
        ix0=max(wx0,qx0); iy0=max(wy0,qy0)
        ix1=min(wx1,qx1); iy1=min(wy1,qy1)
        if ix1>ix0 and iy1>iy0:
            if (ix1-ix0)*(iy1-iy0)/area > thresh:
                return True
    return False


# ─── Word list builder ─────────────────────────────────────────────

def _build_words(data, qr_regions):
    words = []
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        try:
            conf = int(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1
        if not txt or conf < MIN_CONF:
            continue
        w = {
            "text":     txt,
            "conf":     conf,
            "left":     data["left"][i],
            "top":      data["top"][i],
            "ww":       data["width"][i],
            "h":        data["height"][i],
            "block":    data["block_num"][i],
            "is_latin": bool(txt.encode("ascii","ignore").decode("ascii").strip()),
            "stroke":   0.0,
        }
        if not _word_in_qr(w, qr_regions):
            words.append(w)
    return words


# ─── Pixel helpers ─────────────────────────────────────────────────

def _crop(gray, w, pad=2):
    x0 = max(0, w["left"]-pad);        y0 = max(0, w["top"]-pad)
    x1 = min(gray.width,  w["left"]+w["ww"]+pad)
    y1 = min(gray.height, w["top"]+w["h"]+pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return gray.crop((x0, y0, x1, y1))


def _mean_stroke(crop) -> float:
    if crop is None or crop.width < 6 or crop.height < 4:
        return 0.0
    bw   = crop.point(lambda p: 0 if p < 128 else 255, "L")
    pxls = list(bw.getdata())
    ww   = bw.width
    runs = []
    for row in range(bw.height):
        run = 0
        for px in pxls[row*ww:(row+1)*ww]:
            if px < 128:
                run += 1
            elif run > 0:
                runs.append(run); run = 0
        if run > 0:
            runs.append(run)
    return sum(runs) / len(runs) if runs else 0.0


def _to_rect(w, pdf_w, pdf_h, sx, sy, pad=6):
    try:
        # Expand height by 1.4x centred on the word for better visual coverage
        visual_h = max(w["h"] * 1.4, w["h"] + 8)
        top_adj  = w["top"] - (visual_h - w["h"]) / 2

        x  = max(0.0, (w["left"] - pad) * sx)
        y  = max(0.0, pdf_h - (top_adj + visual_h + pad) * sy)
        rw = max((w["ww"] + pad*2) * sx, 8.0)
        rh = max((visual_h + pad*2) * sy, 8.0)
        if rw < 4 or rh < 4:
            return None
        return {"x": x, "y": y, "w": rw, "h": rh, "text": w["text"]}
    except Exception:
        return None