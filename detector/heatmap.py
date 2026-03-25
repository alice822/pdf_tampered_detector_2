"""
Heatmap v10 — three key fixes:
1. drawn_pages updated before inner try/except (strip suppressed correctly)
2. _stroke_detect only runs on IMAGE pages, not text-layer PDFs
3. Reportlab path has same fixes
"""
from __future__ import annotations
import io, re, statistics
from collections import defaultdict

try:
    import fitz
    FITZ_OK = True
except ImportError:
    FITZ_OK = False

try:
    from reportlab.pdfgen import canvas
    RL_OK = True
except ImportError:
    RL_OK = False

try:
    from pypdf import PdfReader, PdfWriter
    PY_OK = True
except ImportError:
    PY_OK = False

try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image, ImageOps, ImageFilter
    OCR_OK = True
except ImportError:
    OCR_OK = False

_HEADER_RE = re.compile(
    r'^(income|tax|department|dept|govt|government|india|permanent|account|number|'
    r'incom|deparment|name|date|birth|dob|signature|of|the|and|for|pan|'
    r'republic|ministry|revenue|national|authority|service)$', re.IGNORECASE)

_NOISE_RE = re.compile(
    r'^(wen|hed|rarer|uw|ue|oo|ii|ee|aa|ce|ae|bi|az|gp|axa|pw|yx|ww|'
    r'sa|re|da|ha|ma|na|ka|ga|ba|ta|la|pa|ja|wa|va)$', re.IGNORECASE)


def _rgb(sev):
    if sev >= 0.7: return (0.84, 0.10, 0.10)
    if sev >= 0.4: return (0.90, 0.42, 0.02)
    return (0.75, 0.60, 0.02)


def _label(cat):
    return {
        "Unnatural Text Weight":           "Bold/heavy text",
        "Font Type Change":                "Font changed",
        "Content Anomaly":                 "Suspicious content",
        "Pixel-Level Anomaly":             "Image anomaly",
        "Noise Inconsistency":             "Noise anomaly",
        "Image Consistency":               "Texture anomaly",
        "Visual Pattern Anomaly":          "Tamper signal",
        "Image Splice Detected":           "Splice detected",
        "Background Mismatch":             "Background issue",
        "Sharp Text on Blurry Background": "Added text",
        "Missing Security Feature":        "Missing feature",
        "Metadata Inconsistency":          "Metadata issue",
        "OCR Anomaly":                     "Text quality",
        "AI-Generation Signal":            "AI generated",
    }.get(cat, cat[:18])


def generate_heatmap_pdf(original_pdf_path, heatmap_data, output_path, findings=None):
    findings = findings or []
    if not findings:
        return False
    if FITZ_OK:
        return _generate_fitz(original_pdf_path, output_path, findings)
    if RL_OK and PY_OK:
        return _generate_reportlab(original_pdf_path, output_path, findings)
    return False


# ═══════════════════════════════════════════════════════════════
# Page type detection
# ═══════════════════════════════════════════════════════════════

def _is_image_page(pdf_path, page_num):
    """
    Returns True if the page is primarily an image (scanned card, photo),
    False if it's a text-layer PDF (bank statement, salary slip, etc.)
    
    Stroke detection ONLY runs on image pages — on text PDFs, bold headings
    are legitimate formatting and should NOT be flagged as tampering.
    """
    if not PY_OK:
        return True  # assume image if can't check
    try:
        reader = PdfReader(pdf_path)
        if page_num < 1 or page_num > len(reader.pages):
            return True
        page = reader.pages[page_num - 1]
        page_obj = reader.pages[page_num - 1]
        text = (page_obj.extract_text() or "").strip()
        word_count = len(text.split())
        # Check for embedded images (XObject resources)
        resources = page_obj.get("/Resources", {})
        has_xobject = bool(resources.get("/XObject") if resources else False)
        # Text-layer PDF: has substantial text AND no dominant image
        # Image page: few embedded words (text layer from OCR) OR has image
        if word_count >= 30 and not has_xobject:
            return False  # text-layer PDF, do NOT stroke-detect
        return True  # image page, stroke-detect OK
    except Exception:
        return True


# ═══════════════════════════════════════════════════════════════
# Stroke-based bold word detection (image pages only)
# ═══════════════════════════════════════════════════════════════

def _stroke_detect(pdf_path, page_num, pdf_w, pdf_h):
    """
    Run OCR + stroke analysis on an IMAGE page.
    Only called for scanned cards/photos, never for text-layer PDFs.
    """
    if not OCR_OK:
        return []
    # CRITICAL: Only run on image pages
    if not _is_image_page(pdf_path, page_num):
        return []
    try:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        images = convert_from_bytes(pdf_bytes, dpi=200,
                                    first_page=page_num, last_page=page_num)
        if not images:
            return []
        img = images[0]
        gray = img.convert('L')
        gray = ImageOps.autocontrast(gray, cutoff=1)
        gray = gray.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=2))

        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT,
                                          config='--psm 3 --oem 1 -l eng')
        sx = pdf_w / img.width
        sy = pdf_h / img.height

        def mean_stroke(L, T, W, H):
            pad = 2
            x0, y0 = max(0, L-pad), max(0, T-pad)
            x1, y1 = min(gray.width, L+W+pad), min(gray.height, T+H+pad)
            if x1 <= x0 or y1 <= y0: return 0.0
            crop = gray.crop((x0, y0, x1, y1))
            bw = crop.point(lambda p: 0 if p < 128 else 255)
            pxdata = list(bw.tobytes())
            ww = bw.width; runs = []
            for row in range(bw.height):
                run = 0
                for px in pxdata[row*ww:(row+1)*ww]:
                    if px == 0: run += 1
                    elif run > 0: runs.append(run); run = 0
                if run > 0: runs.append(run)
            return sum(runs)/len(runs) if runs else 0.0

        words = []
        for i in range(len(data['text'])):
            txt = data['text'][i].strip()
            try: conf = int(data['conf'][i])
            except: conf = -1
            if not txt or conf < 40: continue
            clean = re.sub(r'[^A-Za-z0-9]', '', txt)
            if len(clean) < 3: continue
            if _NOISE_RE.match(txt): continue
            if txt != txt.encode('ascii', 'ignore').decode(): continue
            L, T, W, H = (data['left'][i], data['top'][i],
                          data['width'][i], data['height'][i])
            sw = mean_stroke(L, T, W, H)
            words.append({
                'text': txt, 'left': L, 'top': T, 'w': W, 'h': H,
                'stroke': sw, 'is_header': bool(_HEADER_RE.match(txt))
            })

        if len(words) < 4: return []
        ref = [w for w in words if w['is_header'] and w['stroke'] > 0]
        if len(ref) < 2: return []
        baseline = statistics.median(w['stroke'] for w in ref)
        if baseline < 0.5: return []
        thresh = baseline * 1.55
        cands = [w for w in words if not w['is_header']]
        suspicious = [w for w in cands if w['stroke'] > thresh]
        if not suspicious or len(suspicious) > len(words) * 0.40:
            return []

        result = []
        for w in suspicious:
            x_pdf = w['left'] * sx
            y_bot = pdf_h - (w['top'] + w['h']) * sy
            y_top_pts = w['top'] * sy
            if y_top_pts < pdf_h * 0.08: continue
            pad = 3
            result.append({
                'text': w['text'],
                'x': max(0, x_pdf - pad), 'y': max(0, y_bot - pad),
                'w': w['w'] * sx + pad*2, 'h': w['h'] * sy + pad*2,
            })
        return result
    except Exception:
        return []


def _ocr_locate(pdf_path, page_num, targets, pdf_w, pdf_h):
    if not OCR_OK or not targets: return []
    try:
        with open(pdf_path, 'rb') as f: pdf_bytes = f.read()
        images = convert_from_bytes(pdf_bytes, dpi=200,
                                    first_page=page_num, last_page=page_num)
        if not images: return []
        img = images[0]
        gray = img.convert('L')
        gray = ImageOps.autocontrast(gray, cutoff=1)
        gray = gray.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=2))
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT,
                                          config='--psm 3 --oem 1 -l eng')
        sx = pdf_w / img.width; sy = pdf_h / img.height
        target_lower = [t.lower() for t in targets if t and len(t) >= 3]
        result = []
        for i in range(len(data['text'])):
            txt = data['text'][i].strip()
            if not txt: continue
            try: conf = int(data['conf'][i])
            except: conf = -1
            if conf < 30: continue
            if not any(t in txt.lower() or txt.lower() in t for t in target_lower): continue
            L, T, W, H = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            y_bot = pdf_h - (T+H)*sy
            if T*sy < pdf_h*0.08: continue
            if W*sx > 5 and H*sy > 4:
                result.append({'text': txt, 'x': L*sx, 'y': y_bot, 'w': W*sx, 'h': H*sy})
        return result
    except Exception:
        return []


def _get_targets(finding):
    ev = finding.evidence
    words = []
    for key in ('bold_words', 'large_words', 'dark_words', 'unusual_names', 'sample_text'):
        v = ev.get(key, [])
        if isinstance(v, list): words.extend(v)
        elif isinstance(v, str) and len(v) >= 3: words.append(v)
    return [w for w in dict.fromkeys(words) if w and len(w) >= 3]


def _get_word_rects(pdf_path, finding, pdf_w, pdf_h):
    # Method 1: stored word_rects
    rects = finding.evidence.get('word_rects', [])
    if rects: return rects

    pn = finding.page or 1

    # Method 2: OCR-locate named words
    targets = _get_targets(finding)
    if targets:
        located = _ocr_locate(pdf_path, pn, targets, pdf_w, pdf_h)
        if located: return located

    # Method 3: stroke detection — IMAGE pages only
    visual_cats = {
        'Unnatural Text Weight', 'Font Type Change',
        'Sharp Text on Blurry Background', 'Image Consistency',
        'Noise Inconsistency', 'Image Splice Detected',
    }
    if finding.category in visual_cats:
        return _stroke_detect(pdf_path, pn, pdf_w, pdf_h)

    return []


# ═══════════════════════════════════════════════════════════════
# Method 1: PyMuPDF
# ═══════════════════════════════════════════════════════════════

def _generate_fitz(pdf_path, output_path, findings):
    try:
        doc = fitz.open(pdf_path)
        drawn_pages = set()

        # First pass: collect word rects per page
        page_rects = defaultdict(list)  # pn → [(rects, rgb, label)]
        for finding in findings:
            pn = finding.page
            if pn is None or pn < 1 or pn > len(doc): continue
            page = doc[pn-1]
            mb = page.mediabox if hasattr(page, 'mediabox') else None
            pdf_w = float(mb.width)  if mb else page.rect.width
            pdf_h = float(mb.height) if mb else page.rect.height
            rects = _get_word_rects(pdf_path, finding, pdf_w, pdf_h)
            if rects:
                page_rects[pn].append((rects, _rgb(finding.severity), _label(finding.category)))

        # Second pass: draw boxes
        for pn, groups in page_rects.items():
            page = doc[pn-1]
            ph = page.rect.height
            mb = page.mediabox if hasattr(page, 'mediabox') else None
            drawn_keys = set()
            # Mark this page as having content BEFORE drawing (fix for exception swallow)
            drawn_pages.add(pn)

            for rects, rgb, lbl in groups:
                for wr in rects[:8]:
                    try:
                        txt = wr.get('text', '')
                        if len(re.sub(r'[^A-Za-z0-9]', '', txt)) < 3: continue
                        pdf_x = float(wr['x'])
                        pdf_y = float(wr.get('y', wr.get('y_bottom', 0)))
                        box_w = max(float(wr['w']), 10.0)
                        box_h = max(float(wr['h']),  8.0)
                        key = (round(pdf_x/15), round(pdf_y/15))
                        if key in drawn_keys: continue
                        drawn_keys.add(key)

                        # Convert PDF coords (y from bottom) → fitz coords (y from top)
                        fy0 = ph - pdf_y - box_h
                        fy1 = ph - pdf_y
                        if fy0 < ph * 0.08: continue
                        if fy1 > ph or fy0 < 0: continue

                        rect = fitz.Rect(pdf_x, fy0, pdf_x+box_w, fy1)
                        page.draw_rect(rect, color=rgb, fill=None, width=2.5)

                        bh = 9.0; bw = min(len(lbl)*5.2+8, box_w+70)
                        by0 = fy0 - bh
                        if by0 < 2: by0 = fy1
                        badge = fitz.Rect(pdf_x, by0, pdf_x+bw, by0+bh)
                        page.draw_rect(badge, color=None, fill=rgb, width=0)
                        page.insert_text(
                            fitz.Point(pdf_x+2, by0+bh-2),
                            lbl[:26], fontsize=6.5, color=(1., 1., 1.)
                        )
                    except Exception:
                        pass

        # Margin strip ONLY for pages with no word boxes
        added_strips = set()
        for finding in findings:
            pn = finding.page
            if pn is None or pn in drawn_pages or pn in added_strips: continue
            if pn < 1 or pn > len(doc): continue
            try:
                page = doc[pn-1]; ph = page.rect.height
                strip = fitz.Rect(2, ph*0.44, 8, ph*0.56)
                page.draw_rect(strip, color=_rgb(finding.severity),
                               fill=_rgb(finding.severity), width=0)
                added_strips.add(pn)
            except Exception:
                pass

        doc.save(output_path, garbage=4, deflate=True)
        doc.close()
        return True
    except Exception as e:
        print(f"[heatmap fitz] {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# Method 2: reportlab fallback
# ═══════════════════════════════════════════════════════════════

def _generate_reportlab(pdf_path, output_path, findings):
    try:
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        entries = _build_entries(pdf_path, findings, reader)

        for pn, page in enumerate(reader.pages, 1):
            elist = entries.get(pn, [])
            if elist:
                mb = page.mediabox
                pw, ph = float(mb.width), float(mb.height)
                ov = _render_overlay(pw, ph, elist)
                if ov:
                    ovr = PdfReader(io.BytesIO(ov))
                    page.merge_page(ovr.pages[0])
            writer.add_page(page)

        with open(output_path, 'wb') as f:
            writer.write(f)
        return True
    except Exception as e:
        print(f"[heatmap rl] {e}")
        return False


def _build_entries(pdf_path, findings, reader):
    entries = defaultdict(list)
    has_abs = set()

    for finding in findings:
        pn = finding.page
        if pn is None: continue
        mb = reader.pages[pn-1].mediabox
        pw = float(mb.width); ph = float(mb.height)
        rects = _get_word_rects(pdf_path, finding, pw, ph)
        for wr in rects[:8]:
            try:
                txt = wr.get('text', '')
                if len(re.sub(r'[^A-Za-z0-9]', '', txt)) < 3: continue
                x = float(wr['x']); y = float(wr.get('y', wr.get('y_bottom', 0)))
                w = max(float(wr['w']), 10); h = max(float(wr['h']), 8)
                if y + h > ph * 0.92: continue  # skip top 8%
                entries[pn].append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'sev': finding.severity,
                    'lbl': _label(finding.category),
                    'abs': True,
                })
                has_abs.add(pn)
            except Exception:
                pass

    added_strips = set()
    for f in findings:
        pn = f.page
        if pn is None or pn in has_abs or pn in added_strips: continue
        entries[pn].append({
            'x': 0.01, 'y': 0.42, 'w': 0.01, 'h': 0.16,
            'sev': f.severity * 0.7,
            'lbl': _label(f.category),
            'abs': False,
        })
        added_strips.add(pn)

    return entries


def _render_overlay(pw, ph, entries):
    try:
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=(pw, ph))
        drawn = set()
        for e in entries:
            r, g, b = _rgb(e['sev']); lbl = e['lbl']
            if e.get('abs'):
                x = e['x']; y = e['y']; bw = e['w']; bh = e['h']
                key = (round(x/15), round(y/15))
                if key in drawn: continue
                drawn.add(key)
                if y + bh > ph * 0.92: continue
            else:
                x = e['x']*pw; y = e['y']*ph; bw = e['w']*pw; bh = e['h']*ph

            c.setStrokeColorRGB(r, g, b, alpha=0.95)
            c.setLineWidth(2.0)
            c.rect(x, y, bw, bh, fill=0, stroke=1)
            lh = 9.5; lw = min(len(lbl)*5.5+8, bw+60)
            badge_y = y+bh
            if badge_y+lh > ph: badge_y = y-lh
            c.setFillColorRGB(r, g, b, alpha=1.0)
            c.rect(x, badge_y, lw, lh, fill=1, stroke=0)
            c.setFillColorRGB(1, 1, 1, alpha=1.0)
            c.setFont("Helvetica-Bold", 6.5)
            c.drawString(x+2.5, badge_y+2.5, lbl[:28])

        c.save()
        return buf.getvalue()
    except Exception:
        return None