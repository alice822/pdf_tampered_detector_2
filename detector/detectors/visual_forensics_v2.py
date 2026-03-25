"""
Image Forensics Detector (v3 — card-boundary-aware, multi-signal)
==================================================================
Detects tampering using 6 image forensic signals.
NO OCR dependency. Self-calibrating on each image.

Key improvements over v2:
- Card boundary detection: ignores browser chrome, white margins, UI elements
- All analysis confined to actual card region only
- Shorter badge labels that fit in heatmap boxes
- Noise/sharpness signals suppressed in top 25% (genuine header zone)
- Boxes positioned at detected anomaly locations, not at x=0
"""
from __future__ import annotations
import io
from typing import List
from ..engine import Finding

try:
    import cv2
    import numpy as np
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from pdf2image import convert_from_bytes
    PDF2IMG_OK = True
except ImportError:
    PDF2IMG_OK = False

DPI = 150


def detect(reader) -> List[Finding]:
    findings: List[Finding] = []
    if not CV2_OK or not PDF2IMG_OK:
        return findings
    try:
        import pypdf
        buf = io.BytesIO()
        w = pypdf.PdfWriter()
        for i in range(min(len(reader.pages), 3)):
            w.add_page(reader.pages[i])
        w.write(buf)
        images = convert_from_bytes(buf.getvalue(), dpi=DPI)
        for pn, img in enumerate(images, 1):
            try:
                mb    = reader.pages[pn-1].mediabox
                pdf_w = float(mb.width)
                pdf_h = float(mb.height)
                _analyze(img, pn, pdf_w, pdf_h, findings)
            except Exception:
                pass
    except Exception:
        pass
    return findings


def _find_card_bounds(gray):
    """
    Find the actual card/document within the rendered page.
    Returns (y0f, x0f, y1f, x1f) as fractions of image dimensions.
    Ignores white margins and browser UI chrome.
    """
    h, w = gray.shape
    is_content = gray < 240
    row_density = is_content.mean(axis=1)
    col_density = is_content.mean(axis=0)
    content_rows = np.where(row_density > 0.12)[0]
    content_cols = np.where(col_density > 0.08)[0]
    if len(content_rows) < 10 or len(content_cols) < 10:
        return 0.0, 0.0, 1.0, 1.0
    y0 = max(0, int(np.percentile(content_rows,  3)) - 5)
    y1 = min(h, int(np.percentile(content_rows, 97)) + 5)
    x0 = max(0, int(np.percentile(content_cols,  3)) - 5)
    x1 = min(w, int(np.percentile(content_cols, 97)) + 5)
    return y0/h, x0/w, y1/h, x1/w


def _analyze(img, page_num: int, pdf_w: float, pdf_h: float, findings: list):
    rgb  = np.array(img.convert("RGB")).astype(np.float32)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    H, W = gray.shape

    # Detect card boundaries — work ONLY inside the card
    cy0f, cx0f, cy1f, cx1f = _find_card_bounds(gray)
    cy0, cx0 = int(H*cy0f), int(W*cx0f)
    cy1, cx1 = int(H*cy1f), int(W*cx1f)
    ch = cy1 - cy0  # card height in pixels
    cw = cx1 - cx0  # card width in pixels

    if ch < 50 or cw < 50:
        return

    # Scale: pixel → PDF coords
    sx = pdf_w / W
    sy = pdf_h / H

    def to_rect(px, py, pw, ph_r, label):
        """Pixel coords (within full image) → PDF coord dict."""
        # Clamp to card bounds
        px  = max(cx0, px)
        py  = max(cy0, py)
        pw  = min(cw,  pw)
        ph_r= max(8,   ph_r)
        # PDF y=0 is bottom
        y_pdf = pdf_h - (py + ph_r) * sy
        return {"x": px*sx, "y": max(0,y_pdf),
                "w": pw*sx, "h": ph_r*sy, "text": label}

    signals    = []
    word_rects = []

    # Working regions (fractions of card height, not full image)
    # Header zone = top 28% of card, Body zone = 28-88% of card
    hdr_y1 = cy0 + int(ch * 0.28)
    body_y0 = hdr_y1
    body_y1 = cy0 + int(ch * 0.88)

    # ── Signal 1: Color inconsistency ─────────────────────────────────
    def rb(zone_rgb):
        dark = zone_rgb.mean(axis=2) < 110
        if dark.sum() < 30: return 1.0
        r = zone_rgb[:,:,0][dark].mean()
        b = zone_rgb[:,:,2][dark].mean()
        return float(r / (b + 1e-6))

    hdr_zone  = rgb[cy0:hdr_y1,  cx0:cx1]
    body_zone = rgb[body_y0:body_y1, cx0:cx0+int(cw*0.62)]
    hdr_rb  = rb(hdr_zone)
    body_rb = rb(body_zone)
    color_diff = body_rb - hdr_rb

    if color_diff > 0.30:
        signals.append(("color", 0.68))
        # Scan body rows for black text (lower R/B than header)
        for yi in range(body_y0, body_y1, 20):
            row = rgb[yi:yi+25, cx0:cx0+int(cw*0.60)]
            if row.size > 0 and rb(row) < hdr_rb - 0.20 and row.mean() < 210:
                word_rects.append(to_rect(cx0, yi, int(cw*0.58), 25, "inserted text"))

    # ── Signal 2: Local sharpness spike (body only, not header) ───────
    BAND = 28
    sharp_rows = []
    for y in range(body_y0, body_y1 - BAND, BAND):
        band = gray[y:y+BAND, cx0:cx1]
        if band.mean() > 238 or band.mean() < 5:
            continue
        lap_var = float(cv2.Laplacian(band.astype(np.float32), cv2.CV_32F).var())
        sharp_rows.append((y, lap_var))

    if len(sharp_rows) >= 4:
        vals  = [v for _, v in sharp_rows]
        med   = float(np.median(vals))
        std   = float(np.std(vals)) + 1.0
        spikes= [(y,v) for y,v in sharp_rows if (v-med)/std > 2.5]
        if spikes:
            signals.append(("sharpness", 0.58))
            for y, _ in spikes[:4]:
                word_rects.append(to_rect(cx0, y, int(cw*0.58), BAND, "sharp region"))

    # ── Signal 3: Noise inconsistency ─────────────────────────────────
    BLOCK = 48
    noise_map = []
    for y in range(cy0, cy1 - BLOCK, BLOCK):
        for x in range(cx0, cx1 - BLOCK, BLOCK):
            tile = gray[y:y+BLOCK, x:x+BLOCK]
            if tile.mean() > 238:
                continue
            blur  = cv2.GaussianBlur(tile, (5,5), 0)
            noise = float((tile.astype(np.float32) - blur).std())
            noise_map.append((y, x, noise))

    if noise_map:
        nv  = [v for _,_,v in noise_map]
        nm, ns = float(np.mean(nv)), float(np.std(nv)) + 1e-6
        anom   = [(y,x,v) for y,x,v in noise_map
                  if 2.0 < (v-nm)/ns < 4.5
                  and y > hdr_y1]  # skip header zone
        if len(anom) >= 3:
            signals.append(("noise", 0.55))

    # ── Signal 4: DCT frequency anomaly ───────────────────────────────
    PATCH = 64
    dct_vals = []
    for y in range(cy0, cy1 - PATCH, PATCH):
        for x in range(cx0, cx0 + int(cw*0.55) - PATCH, PATCH):
            tile = gray[y:y+PATCH, x:x+PATCH].astype(np.float32)
            if tile.mean() > 235:
                continue
            dct  = cv2.dct(tile)
            high = float(np.abs(dct[16:, 16:]).mean())
            low  = float(np.abs(dct[:8,  :8 ]).mean()) + 1e-6
            dct_vals.append((y, x, high/low))

    if dct_vals:
        rv = [r for _,_,r in dct_vals]
        rm, rs = float(np.mean(rv)), float(np.std(rv)) + 1e-6
        dct_anom = [(y,x,r) for y,x,r in dct_vals
                    if (r-rm)/rs > 2.5 and y > hdr_y1]
        if len(dct_anom) >= 2:
            signals.append(("dct", 0.50))

    # ── Signal 5: Text size bimodality (within card) ──────────────────
    card_gray  = gray[cy0:cy1, cx0:cx1]
    row_dark   = (card_gray < 100).sum(axis=1).astype(np.float32)
    text_rows  = row_dark[row_dark > cw * 0.01]
    if len(text_rows) >= 8:
        t25 = float(np.percentile(text_rows, 25))
        t75 = float(np.percentile(text_rows, 75))
        ratio = t75 / (t25 + 1e-6)
        if ratio > 3.5:
            signals.append(("bimodal", 0.52))

    # ── Signal 6: Blank hologram (within card, top-right zone) ────────
    holo_zone = gray[cy0:cy0+int(ch*0.50), cx0+int(cw*0.55):cx1]
    BSIZ = 22
    blank = 0
    for y in range(0, holo_zone.shape[0]-BSIZ, BSIZ):
        for x in range(0, holo_zone.shape[1]-BSIZ, BSIZ):
            cell = holo_zone[y:y+BSIZ, x:x+BSIZ]
            if cell.std() < 12 and 150 < float(cell.mean()) < 235:
                blank += 1
    if blank >= 4:
        signals.append(("hologram", 0.48))

    if not signals:
        return

    sig_names  = [s[0] for s in signals]
    max_sev    = max(s[1] for s in signals)
    severity   = min(max_sev + len(signals) * 0.06, 0.88)

    desc_map = {
        "color":    "Header text is maroon but body text is black — inserted content",
        "sharpness":"Isolated sharp regions on blurred background — digital insertion",
        "noise":    "Inconsistent image noise — regions from different image sources",
        "dct":      "DCT frequency anomaly — different JPEG compression history",
        "bimodal":  "Two distinct text size families — original print + inserted text",
        "hologram": "Missing/blank hologram area — security feature removed",
    }
    description = " | ".join(desc_map.get(s,"anomaly") for s in sig_names[:2])

    findings.append(Finding(
        category="Visual Pattern Anomaly",
        description=f"Page {page_num}: {description}.",
        severity=severity,
        page=page_num,
        evidence={
            "signals":      sig_names,
            "signal_count": len(signals),
            "word_rects":   word_rects[:8],
            "detail":       f"{len(signals)} signals: {', '.join(sig_names)}",
        },
    ))