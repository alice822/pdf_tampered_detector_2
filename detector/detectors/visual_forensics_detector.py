"""
Visual Forensics Detector  (v2 — plain language, physical card aware)
---------------------------------------------------------------------
Analyses rendered page images for signs of digital manipulation.
All findings use plain language a non-technical user can understand.

Physical card photos naturally trigger brightness/noise checks —
we suppress those findings unless corroborated by other signals.
"""

from __future__ import annotations
import io, math
from typing import List
from ..engine import Finding

try:
    from PIL import Image, ImageStat, ImageFilter, ImageChops
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

DPI       = 150
BLOCK     = 32
MAX_PAGES = 4


def detect(reader) -> List[Finding]:
    findings: List[Finding] = []
    if not PIL_AVAILABLE or not PDF2IMAGE_AVAILABLE:
        return findings
    try:
        buf = io.BytesIO()
        from pypdf import PdfWriter
        w = PdfWriter()
        for i in range(min(len(reader.pages), MAX_PAGES)):
            w.add_page(reader.pages[i])
        w.write(buf)
        images = convert_from_bytes(buf.getvalue(), dpi=DPI)
        for page_num, img in enumerate(images, 1):
            try:
                _analyze_visual(img, page_num, findings)
            except Exception:
                pass
    except Exception:
        pass
    return findings


def _analyze_visual(img, page_num, findings):
    gray = img.convert("L")
    w, h = gray.size
    _check_noise_inconsistency(gray, w, h, page_num, findings)
    _check_brightness_splice(img, w, h, page_num, findings)
    _check_background_discontinuity(img, w, h, page_num, findings)
    _check_compression_artifacts(gray, w, h, page_num, findings)
    _check_sharpness_islands(gray, w, h, page_num, findings)
    _check_security_features(gray, w, h, page_num, findings)


# ── 1. Noise inconsistency ────────────────────────────────────────────────

def _check_noise_inconsistency(gray, w, h, page_num, findings):
    """
    Uneven noise across the page = different image sources merged together.
    Physical card photos have high noise uniformly — that's normal.
    We only flag when noise VARIANCE is extreme (patchy = spliced regions).
    """
    if NUMPY_AVAILABLE:
        arr = np.array(gray, dtype=np.float32)
        stds = []
        for y in range(0, h - BLOCK, BLOCK):
            for x in range(0, w - BLOCK, BLOCK):
                stds.append(float(np.std(arr[y:y+BLOCK, x:x+BLOCK])))
    else:
        stds = []
        for y in range(0, h - BLOCK, BLOCK):
            for x in range(0, w - BLOCK, BLOCK):
                stds.append(ImageStat.Stat(gray.crop((x,y,x+BLOCK,y+BLOCK))).stddev[0])

    if not stds: return
    mean_std = sum(stds)/len(stds)
    if mean_std <= 0: return
    std_std  = math.sqrt(sum((s-mean_std)**2 for s in stds)/len(stds))
    cv = std_std / mean_std

    # High threshold — physical card texture already raises CV
    if cv > 3.2:
        findings.append(Finding(
            category="Image Consistency",
            description=(
                f"The document image on page {page_num} has uneven visual texture — "
                "some areas are noticeably grainier or cleaner than others. "
                "This can happen when parts of an image are copied from a different source "
                "and pasted onto the document. A genuine printed document should have "
                "consistent texture throughout."
            ),
            severity=min(0.35 + (cv - 3.5) * 0.08, 0.62),
            page=page_num,
            evidence={"detail": f"texture variation CV={cv:.2f}"},
        ))


# ── 2. Brightness splice (top vs bottom) ─────────────────────────────────

def _check_brightness_splice(img, w, h, page_num, findings):
    """
    Sharp brightness difference between top and bottom = two images joined.
    Physical card photos always have some gradient — only flag extreme cases.
    Use a MUCH higher threshold than before.
    """
    top    = img.crop((w//4, 0,      3*w//4, h//2))
    bottom = img.crop((w//4, h//2,   3*w//4, h))
    top_b    = ImageStat.Stat(top).mean[0]
    bottom_b = ImageStat.Stat(bottom).mean[0]
    delta = abs(top_b - bottom_b)

    # Only flag if difference is very large AND there's a sharp LINE (not gradual)
    if delta > 80:
        # Check if it's a sharp transition (not gradual gradient)
        strips = []
        for y_start in range(0, h, h//10):
            strip = img.crop((w//4, y_start, 3*w//4, min(y_start+h//10, h)))
            strips.append(ImageStat.Stat(strip).mean[0])
        # Calculate how sharp the transition is
        diffs = [abs(strips[i+1]-strips[i]) for i in range(len(strips)-1)]
        max_jump = max(diffs) if diffs else 0

        if max_jump > 40:  # sharp transition = likely spliced
            findings.append(Finding(
                category="Image Splice Detected",
                description=(
                    f"Page {page_num} shows a sharp brightness change at a specific "
                    "horizontal line across the document. This is a common sign of "
                    "two separate images being joined together — for example, a name "
                    "taken from one document and placed onto another."
                ),
                severity=0.65,
                page=page_num,
                evidence={"detail": f"brightness jump={max_jump:.0f}/255 across splice line"},
            ))


# ── 3. Background discontinuity ──────────────────────────────────────────

def _check_background_discontinuity(img, w, h, page_num, findings):
    """
    Bright border + dark centre = document image pasted onto different background.
    But: physical cards placed on white paper ALWAYS show this pattern.
    Only flag if the contrast is extreme (>120 difference).
    """
    if NUMPY_AVAILABLE:
        arr = np.array(img.convert("L"), dtype=np.float32)
        border = 15  # pixels
        corners = np.concatenate([
            arr[:border, :border].flatten(),
            arr[:border, -border:].flatten(),
            arr[-border:, :border].flatten(),
            arr[-border:, -border:].flatten()
        ])
        centre = arr[h//3:2*h//3, w//3:2*w//3].flatten()
        corners_avg = float(corners.mean())
        centre_avg  = float(centre.mean())
    else:
        b = 15
        c_crops = [img.crop((0,0,b,b)), img.crop((w-b,0,w,b)),
                   img.crop((0,h-b,b,h)), img.crop((w-b,h-b,w,h))]
        corners_avg = sum(ImageStat.Stat(c.convert("L")).mean[0] for c in c_crops)/4
        ctr = img.crop((w//3, h//3, 2*w//3, 2*h//3)).convert("L")
        centre_avg = ImageStat.Stat(ctr).mean[0]

    diff = abs(corners_avg - centre_avg)

    # Physical card on white paper gives ~75-90 diff. Flag >100 as suspicious.
    if diff > 100:
        findings.append(Finding(
            category="Background Mismatch",
            description=(
                f"Page {page_num}: The edges of the page are very different in brightness "
                "from the centre, suggesting the document image may have been placed onto "
                "a different background. This can indicate the document was digitally "
                "composited rather than scanned directly."
            ),
            severity=0.55,
            page=page_num,
            evidence={"detail": f"edge/centre brightness contrast={diff:.0f}/255"},
        ))


# ── 4. Compression artifacts ──────────────────────────────────────────────

def _check_compression_artifacts(gray, w, h, page_num, findings):
    """Double-JPEG compression = image was saved, edited, saved again."""
    if not NUMPY_AVAILABLE:
        return
    arr    = np.array(gray, dtype=np.float32)
    bs     = 8
    scores = []
    for y in range(0, h - bs, bs):
        for x in range(0, w - bs, bs):
            b1 = arr[y:y+bs, x:x+bs]
            b2 = arr[y:y+bs, x+bs:x+2*bs] if x+2*bs <= w else None
            if b2 is not None:
                scores.append(abs(float(np.mean(b1)) - float(np.mean(b2))))
    if not scores: return
    mean_s = sum(scores)/len(scores)
    std_s  = math.sqrt(sum((s-mean_s)**2 for s in scores)/len(scores))
    if mean_s > 0:
        cv = std_s / mean_s
        if cv > 1.8 and mean_s > 12:
            findings.append(Finding(
                category="Image Re-compression",
                description=(
                    f"Page {page_num} shows signs of the image being saved, edited, "
                    "then saved again (double-compression). When an image is edited and "
                    "re-saved as JPEG multiple times, it leaves a distinctive pattern. "
                    "This can indicate the document was digitally modified after its "
                    "original creation."
                ),
                severity=0.48,
                page=page_num,
                evidence={"detail": f"compression pattern CV={cv:.2f}"},
            ))


# ── 5. Sharpness islands ──────────────────────────────────────────────────

def _check_sharpness_islands(gray, w, h, page_num, findings):
    """
    Detect regions significantly sharper than the rest of the page.
    
    TRUE POSITIVE: Sharp text pasted onto a blurry scanned document.
    FALSE POSITIVE risk: Any blurry photo of a card where printed text is 
    naturally sharper than decorative backgrounds (rosettes, watermarks).
    
    Guard conditions to reduce false positives:
    - Require overall image to not be extremely low quality (mean_s > 8)
      because on a very blurry scan EVERYTHING is uniformly poor quality
    - Require higher cv threshold (2.0 not 1.3)
    - Require more extreme ratio (0.05 not 0.025)
    """
    if not NUMPY_AVAILABLE:
        return
    arr   = np.array(gray, dtype=np.float32)
    bsize = 48
    sharpness_map = []
    for y in range(0, h - bsize, bsize):
        for x in range(0, w - bsize, bsize):
            tile = arr[y:y+bsize, x:x+bsize]
            lap  = np.abs(np.diff(np.diff(tile, axis=0), axis=0))
            sharpness_map.append(float(lap.var()))

    if len(sharpness_map) < 8:
        return
    mean_s = sum(sharpness_map) / len(sharpness_map)
    std_s  = math.sqrt(sum((s-mean_s)**2 for s in sharpness_map) / len(sharpness_map))
    if mean_s <= 0:
        return

    # Guard: if overall image is very low quality (blurry scan/photo),
    # sharpness variation is expected — don't flag it
    # mean_s < 8 means the image is uniformly blurry (low DPI scan/photo)
    if mean_s < 8.0:
        return

    cv = std_s / mean_s
    sharp_tiles = sum(1 for s in sharpness_map if s > mean_s + 3*std_s)
    ratio = sharp_tiles / len(sharpness_map)

    # Require strong signal: high cv AND meaningful ratio of sharp tiles
    # cv > 2.0 (was 1.3) and ratio > 0.05 (was 0.025) to reduce false positives
    if ratio > 0.05 and ratio < 0.35 and cv > 2.0:
        findings.append(Finding(
            category="Sharp Text on Blurry Background",
            description=(
                f"Page {page_num}: {sharp_tiles} image areas are significantly sharper "
                f"than the document baseline. "
                "In a genuine scanned document everything should have similar clarity. "
                "Isolated sharp areas on a blurry background can indicate that "
                "text or images were added digitally after the original scan."
            ),
            severity=min(0.52 + ratio * 0.4, 0.68),
            page=page_num,
            evidence={"detail": f"{sharp_tiles}/{len(sharpness_map)} sharp regions, CV={cv:.2f}"},
        ))


# ── 6. Missing security feature (blank hologram area) ────────────────────

def _check_security_features(gray, w, h, page_num, findings):
    """
    Detect unusually blank/uniform rectangular areas that should contain
    security features (holograms, watermarks, stamps).
    In genuine ID cards, the hologram area has complex iridescent patterns.
    A blank or nearly uniform rectangle = hologram was removed/covered.
    """
    if not NUMPY_AVAILABLE:
        return
    arr = np.array(gray, dtype=np.float32)
    
    # Scan in blocks looking for suspiciously uniform bright regions
    block = 30
    uniform_regions = []
    for y in range(0, h - block*2, block):
        for x in range(0, w - block*2, block):
            # Check 2x2 block area (60x60px minimum)
            region = arr[y:y+block*2, x:x+block*2]
            std = float(np.std(region))
            mean = float(np.mean(region))
            # Very uniform AND bright = likely blank area where something was removed
            if std < 8.0 and mean > 200 and region.size >= 1800:
                uniform_regions.append((x, y, std, mean))
    
    # Only flag if count is small (3-50): real missing hologram area
    # Very high counts (>50) = white paper document, not ID card missing hologram
    if 3 <= len(uniform_regions) <= 50:
        findings.append(Finding(
            category="Missing Security Feature",
            description=(
                f"Page {page_num}: {len(uniform_regions)} unusually blank area(s) detected "
                "that may indicate security features were removed or covered. "
                "Genuine ID cards contain holograms, watermarks, and stamps "
                "which create complex visual patterns. "
                "A blank or uniformly white rectangle in these positions "
                "suggests the document security feature may have been removed."
            ),
            severity=0.62,
            page=page_num,
            evidence={"detail": f"{len(uniform_regions)} blank security-feature regions found"},
        ))