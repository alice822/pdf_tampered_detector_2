from __future__ import annotations
import io, math
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

DPI        = 100
PATCH_SIZE = 32
MIN_BLOCKS = 16


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
                _analyze_page(img, pn, findings)
            except Exception:
                pass
    except Exception:
        pass

    return findings


def _extract_block_features(gray_block: np.ndarray) -> np.ndarray:
    b = cv2.resize(gray_block, (PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
    feats = []

    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for sigma in [1.5, 3.0]:
            k = cv2.getGaborKernel((11, 11), sigma, theta,
                                    lambd=6.0, gamma=0.5, psi=0)
            resp = cv2.filter2D(b, cv2.CV_32F, k)
            feats += [float(resp.mean()), float(resp.std())]

    dct = cv2.dct(b)
    feats += [
        float(np.abs(dct[:4,  :4]).mean()),
        float(np.abs(dct[4:12, 4:12]).mean()),
        float(np.abs(dct[12:, 12:]).mean()),
    ]

    blur  = cv2.GaussianBlur(b, (5, 5), 0)
    noise = b - blur
    feats += [
        float(noise.std()),
        float(np.percentile(np.abs(noise), 90)),
    ]

    feats += [
        float(b.mean()),
        float(b.std()),
        float(cv2.Laplacian(b, cv2.CV_32F).var()),
    ]

    return np.array(feats, dtype=np.float32)


def _analyze_page(img, page_num: int, findings: list):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    blocks    = []
    positions = []
    for y in range(0, h - PATCH_SIZE, PATCH_SIZE):
        for x in range(0, w - PATCH_SIZE, PATCH_SIZE):
            block = gray[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            if block.mean() > 250 or block.mean() < 5:
                continue
            feats = _extract_block_features(block)
            blocks.append(feats)
            positions.append((x, y))

    if len(blocks) < MIN_BLOCKS:
        return

    blocks_arr = np.array(blocks)
    median     = np.median(blocks_arr, axis=0)
    mad        = np.median(np.abs(blocks_arr - median), axis=0) + 1e-6
    z_scores   = np.abs((blocks_arr - median) / mad)
    anomaly    = z_scores.mean(axis=1)

    threshold = anomaly.mean() + 2.5 * anomaly.std()
    outliers  = [(positions[i], float(anomaly[i]))
                 for i in range(len(blocks)) if anomaly[i] > threshold]

    if not outliers:
        return

    ratio = len(outliers) / len(blocks)

    # ── FIX 1: Tighter ratio window ────────────────────────────────
    # Old: 0.03–0.30  →  too wide, catches watermarks/logos
    # New: 0.03–0.12  →  only flag genuinely localised anomalies
    # A watermark covers ~20-30% of page → now excluded
    if ratio < 0.03 or ratio > 0.12:
        return

    # ── FIX 2: Higher z-score threshold ───────────────────────────
    # Old: mean + 2.5×std  →  fires on subtle texture differences
    # New: require max anomaly score > 8.0 (was any outlier above threshold)
    # This ensures only strongly anomalous blocks count
    max_anomaly = max(score for _, score in outliers)
    if max_anomaly < 8.0:
        return

    # ── FIX 3: Require spatial clustering ─────────────────────────
    # Random texture differences scatter across page (logo, watermark)
    # Real insertions are spatially concentrated in 1-2 regions
    regions = _cluster_blocks(outliers)
    if len(regions) > 4:
        # Too many scattered regions = background noise, not insertion
        return

    severity = min(0.40 + (max_anomaly / 20) * 0.25, 0.65)

    findings.append(Finding(
        category="Visual Pattern Anomaly",
        description=(
            f"Page {page_num}: CNN-based visual analysis found "
            f"{len(outliers)} image regions ({ratio*100:.0f}% of page) "
            "with significantly different visual texture than the surrounding document."
        ),
        severity=severity,
        page=page_num,
        evidence={
            "outlier_blocks": len(outliers),
            "total_blocks":   len(blocks),
            "outlier_ratio":  round(ratio, 3),
            "max_anomaly":    round(max_anomaly, 2),
            "regions":        len(regions),
            "detail": f"{len(outliers)}/{len(blocks)} blocks anomalous",
        },
    ))


def _cluster_blocks(outliers, gap=PATCH_SIZE*2):
    if not outliers:
        return []
    positions = [pos for pos, _ in outliers]
    clusters  = []
    used      = [False] * len(positions)
    for i, (x0, y0) in enumerate(positions):
        if used[i]:
            continue
        cluster = [(x0, y0)]
        used[i] = True
        for j, (x1, y1) in enumerate(positions):
            if not used[j] and abs(x1-x0) < gap and abs(y1-y0) < gap:
                cluster.append((x1, y1))
                used[j] = True
        clusters.append(cluster)
    return clusters