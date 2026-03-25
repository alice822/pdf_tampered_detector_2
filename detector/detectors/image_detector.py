"""
Image Detector
--------------
Analyzes embedded images for:
  - Abnormally low DPI (suggests blurred/replaced image)
  - JPEG compression artifacts inconsistent with surrounding quality
  - Unusual colour-space or bit-depth mix
  - Double-JPEG compression fingerprints (AI-edited images)
  - Images with alpha channel (common in composited fakes)
  - Pixel noise level anomalies (ELA-inspired)
"""

from __future__ import annotations
import io
import math
from typing import List
from ..engine import Finding

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Page resolution assumption for DPI estimation (pts per inch = 72)
PT_PER_INCH = 72.0
LOW_DPI_THRESHOLD = 72          # images below this are suspicious
VERY_LOW_DPI_THRESHOLD = 36


def detect(reader) -> List[Finding]:
    findings: List[Finding] = []
    if not PIL_AVAILABLE:
        return findings

    for page_num, page in enumerate(reader.pages, 1):
        try:
            _analyze_page_images(page, page_num, findings)
        except Exception:
            pass

    return findings


def _analyze_page_images(page, page_num: int, findings: list):
    images = page.images if hasattr(page, "images") else []
    for img_info in images:
        try:
            raw_data = img_info.data
            pil_img = Image.open(io.BytesIO(raw_data))

            width_px, height_px = pil_img.size
            # Estimate rendered size from page mediabox
            mb = page.mediabox
            page_w_pt = float(mb.width)
            page_h_pt = float(mb.height)

            # Approximate DPI (using page dimensions as proxy for image placement)
            dpi_x = (width_px / page_w_pt) * PT_PER_INCH if page_w_pt > 0 else 0
            dpi_y = (height_px / page_h_pt) * PT_PER_INCH if page_h_pt > 0 else 0
            avg_dpi = (dpi_x + dpi_y) / 2

            if 0 < avg_dpi < VERY_LOW_DPI_THRESHOLD:
                findings.append(Finding(
                    category="Image Compression Anomaly",
                    description=f"Very low DPI image (~{avg_dpi:.0f} DPI). Content may have been replaced with a low-resolution substitute.",
                    severity=0.6,
                    page=page_num,
                    evidence={"estimated_dpi": avg_dpi, "pixels": f"{width_px}x{height_px}"},
                ))
            elif 0 < avg_dpi < LOW_DPI_THRESHOLD:
                findings.append(Finding(
                    category="Image Compression Anomaly",
                    description=f"Low DPI image (~{avg_dpi:.0f} DPI) — may indicate replacement or downscaling.",
                    severity=0.35,
                    page=page_num,
                    evidence={"estimated_dpi": avg_dpi},
                ))

            # Check for alpha channel (composite fake indicator)
            if pil_img.mode in ("RGBA", "LA"):
                findings.append(Finding(
                    category="Image Compression Anomaly",
                    description="Image has transparency channel (alpha) — may indicate compositing or layered manipulation.",
                    severity=0.4,
                    page=page_num,
                ))

            # ELA-inspired noise analysis
            _ela_check(pil_img, page_num, img_info, findings)

            # JPEG double-compression check
            _double_jpeg_check(raw_data, pil_img, page_num, findings)

        except Exception:
            pass


def _ela_check(pil_img: "Image.Image", page_num: int, img_info, findings: list):
    """
    Error Level Analysis (ELA) approximation.
    Re-compress at known quality and compare noise variance.
    Regions with very different noise levels suggest manipulation.
    """
    try:
        if pil_img.mode not in ("RGB", "L"):
            pil_img = pil_img.convert("RGB")

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        original = pil_img.convert("RGB")
        if original.size != recompressed.size:
            return

        # Compare pixel-level differences in blocks
        orig_data = list(original.getdata())
        recomp_data = list(recompressed.getdata())

        diffs = [
            abs(o[0] - r[0]) + abs(o[1] - r[1]) + abs(o[2] - r[2])
            for o, r in zip(orig_data, recomp_data)
        ]

        if not diffs:
            return

        mean_diff = sum(diffs) / len(diffs)
        variance = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
        std_dev = math.sqrt(variance)

        # High variance relative to mean = uneven compression history
        if mean_diff > 0 and std_dev / mean_diff > 3.5:
            findings.append(Finding(
                category="Noise Inconsistency",
                description=f"ELA analysis shows high noise variance (σ/μ = {std_dev/mean_diff:.1f}). Image regions have inconsistent compression history — possible splice or replacement.",
                severity=0.65,
                page=page_num,
                evidence={"ela_ratio": round(std_dev / mean_diff, 2)},
            ))
    except Exception:
        pass


def _double_jpeg_check(raw_data: bytes, pil_img: "Image.Image", page_num: int, findings: list):
    """
    Detect double-JPEG compression.
    A JPEG re-saved as JPEG shows characteristic quantization table artifacts.
    """
    try:
        # JPEG files start with FFD8
        if not raw_data[:2] == b"\xff\xd8":
            return

        # Count quantization table markers (FFC4 = Huffman, FFDB = quant)
        quant_markers = raw_data.count(b"\xff\xdb")
        huffman_markers = raw_data.count(b"\xff\xc4")

        # Double JPEG typically has >2 quant tables or mismatched counts
        if quant_markers > 4:
            findings.append(Finding(
                category="Image Compression Anomaly",
                description=f"JPEG image has {quant_markers} quantization tables (expected ≤4). Possible double-compression indicating AI image substitution.",
                severity=0.5,
                page=page_num,
                evidence={"quant_tables": quant_markers},
            ))
    except Exception:
        pass
