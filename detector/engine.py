"""
PDF Tamper Detection Engine  (v3 — calibrated scoring + ML integration)
Now includes Gradient Boosting / CNN model prediction as an additional detector.
"""

import io
import time
import hashlib
import logging
import math
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import pickle
import os

logger = logging.getLogger(__name__)

COMMON_PASSWORDS = [
    "", "password", "123456", "admin", "user", "test",
    "pdf", "document", "owner", "1234", "12345", "qwerty"
]

CATEGORY_CAPS = {
    "Metadata Inconsistency":      15,
    "Unusual Encoding":            10,
    "AI-Pattern Font Substitution":15,
    "Noise Inconsistency":         25,
    "Image Text Anomaly":          30,
    "Pixel-Level Anomaly":         15,
    "Image Compression Anomaly":   20,
    "OCR Anomaly":                 15,
    "Content Anomaly":             35,
    "AI-Generation Signal":        30,
    "Unnatural Text Weight":       55,
    "Text Layer Inserted":         75,
    "Content Integrity Failure":   80,
    "Missing Face Photo":          60,
    "Font Type Change":            65,
    "Digital Signature":           40,
    "Incremental Update":          30,
    "Embedded Script":             50,
    "Annotation":                  20,
    "Hidden Layer":                25,
    "Suspicious URL":              30,
    "ML Model Prediction":         30,  # New cap for ML detector
}
DEFAULT_CAP = 25

PLAIN_CATEGORY = {
    "Unnatural Text Weight":    "Bold/heavy text inserted",
    "Font Type Change":         "Different font detected",
    "Content Anomaly":          "Suspicious field values",
    "Pixel-Level Anomaly":      "Image inconsistency",
    "Noise Inconsistency":      "Image texture anomaly",
    "Text Layer Inserted":      "Text digitally inserted over document image",
    "ML Model Prediction":      "ML model predicted tampering",
}

def plain_name(category: str) -> str:
    return PLAIN_CATEGORY.get(category, category)

@dataclass
class Finding:
    category: str
    description: str
    severity: float  # 0.0 – 1.0
    evidence: dict = field(default_factory=dict)
    page: Optional[int] = None

@dataclass
class AnalysisResult:
    tampered: bool = False
    risk_score: float = 0.0
    confidence: float = 0.0
    tamper_type: str = "Unknown"
    findings: list = field(default_factory=list)
    heatmap_data: list = field(default_factory=list)
    processing_time: float = 0.0
    total_pages: int = 0
    file_hash: str = ""
    encrypted: bool = False
    ocr_used: bool = False
    error: Optional[str] = None

    def top_findings(self, n: int = 10) -> list:
        return sorted(self.findings, key=lambda f: f.severity, reverse=True)[:n]

    def human_readable(self, max_reasons: int = 10) -> str:
        lines = [
            f"Tampered  : {'YES' if self.tampered else 'NO'}",
            f"Confidence: {self.confidence:.0f}%",
            f"Risk Score: {self.risk_score:.0f}/100",
            f"Type      : {self.tamper_type}",
            "",
            "Reasons:",
        ]
        for f in self.top_findings(max_reasons):
            sev_label = (
                "🔴 HIGH" if f.severity >= 0.7
                else "🟡 MED" if f.severity >= 0.4
                else "🟢 LOW"
            )
            page_tag = f" [page {f.page}]" if f.page is not None else ""
            lines.append(f"  [{sev_label}] {f.category}{page_tag}: {f.description}")
        if not self.findings:
            lines.append("  No suspicious indicators found.")
        if self.encrypted:
            lines.append("\n  ℹ️  Document was encrypted (attempted common passwords).")
        if self.ocr_used:
            lines.append("  ℹ️  OCR was used for page image analysis.")
        return "\n".join(lines)

# ── Tamper type classification ─────────────────────────────
AI_EDIT_CATEGORIES = {
    "Noise Inconsistency", "Image Compression Anomaly",
    "AI-Pattern Font Substitution", "Invisible Text",
    "Pixel-Level Anomaly", "Unnatural Text Weight",
    "Font Type Change", "Text Layer Inserted", 
}
MANUAL_EDIT_CATEGORIES = {
    "Metadata Inconsistency", "Incremental Update",
    "Digital Signature", "Annotation",
    "Embedded Script", "Unusual Encoding",
    "Hidden Layer", "Suspicious URL",
    "Content Anomaly", "OCR Anomaly", "AI-Generation Signal",
}

WEAK_ALONE_CATEGORIES = {
    "Unusual Encoding",
    "OCR Anomaly",
}

def _classify_tamper_type(findings: list, risk_score: float = 0) -> str:
    if risk_score < 40:
        return "No tampering detected"
    ai_score = sum(f.severity for f in findings if f.category in AI_EDIT_CATEGORIES and f.severity >= 0.4)
    manual_score = sum(f.severity for f in findings if f.category in MANUAL_EDIT_CATEGORIES and f.severity >= 0.4)
    if ai_score == 0 and manual_score == 0:
        return "Suspicious — low confidence"
    if ai_score > manual_score * 1.4:
        return "Likely AI-assisted edit"
    if manual_score > ai_score * 1.4:
        return "Likely manual tampering"
    return "Mixed (AI + manual) tampering"

def _score_to_confidence(score: float) -> float:
    return round(100 / (1 + math.exp(-0.10 * (score - 50))), 1)

def _compute_score(findings: list) -> float:
    if not findings:
        return 0.0

    strong_cats = {
        f.category for f in findings
        if f.category not in WEAK_ALONE_CATEGORIES and f.severity >= 0.4
    }

    visual_cats_on_page = {}
    for f in findings:
        if f.page is not None and f.severity >= 0.4:
            visual_cats_on_page.setdefault(f.page, set()).add(f.category)
    page_corroborated = any(len(cats) >= 2 for cats in visual_cats_on_page.values())

    corroborated = len(strong_cats) >= 1 or page_corroborated

    cat_scores: dict[str, float] = defaultdict(float)
    for f in findings:
        weight = _severity_weight(f)
        contribution = f.severity * 100 * weight
        if f.category in WEAK_ALONE_CATEGORIES and not corroborated:
            contribution *= 0.40
        cat_scores[f.category] += contribution

    total = 0.0
    for cat, score in cat_scores.items():
        cap = CATEGORY_CAPS.get(cat, DEFAULT_CAP)
        total += min(score, cap)

    return round(min(total, 100.0), 1)

def _severity_weight(finding: Finding) -> float:
    if finding.severity >= 0.7:
        return 1.0
    if finding.severity >= 0.4:
        return 0.65
    return 0.30

# ── Load ML Model ─────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_model.pkl")
try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    ML_MODEL = model_data['model']
    print(f"✅ Loaded CNN/GB model with {model_data['n_genuine']} genuine, {model_data['n_tampered']} tampered samples")
except Exception as e:
    ML_MODEL = None
    print(f"⚠️ Could not load ML model: {e}")

# ── PDF Tamper Detector ─────────────────────────────
class PDFTamperDetector:
    def __init__(self, enable_ocr: bool = True, enable_heatmap: bool = True):
        self.enable_ocr = enable_ocr
        self.enable_heatmap = enable_heatmap

    def analyze(self, pdf_path: str) -> AnalysisResult:
        t0 = time.time()
        result = AnalysisResult()
        path = Path(pdf_path)

        if not path.exists():
            result.error = f"File not found: {pdf_path}"
            return result

        raw = path.read_bytes()
        result.file_hash = hashlib.sha256(raw).hexdigest()

        reader, result.encrypted = self._open_reader(raw)
        if reader is None:
            result.error = "Could not open PDF (encrypted with unknown password?)"
            result.processing_time = time.time() - t0
            return result

        result.total_pages = len(reader.pages)

        from .detectors import (
            metadata_detector,
            signature_detector,
            annotation_detector,
            image_detector,
            font_detector,
            url_detector,
            script_detector,
            encoding_detector,
            hidden_text_detector,
            incremental_update_detector,
            structure_detector,
            ocr_detector,
            content_anomaly_detector,
            visual_forensics_detector,
            image_text_detector,
            cnn_detector,
            visual_forensics_v2,
            text_layer_detector,
            document_integrity_detector,
            pan_card_detector, 
        )

        def extract_ml_features(pdf_reader, pdf_bytes):
            from pdf2image import convert_from_bytes
            import numpy as np
            import cv2

            images = convert_from_bytes(pdf_bytes, dpi=100)
            img = images[0]
            rgb = np.array(img.convert('RGB')).astype(np.float32)
            gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            feats = []

            # Noise per zone
            for y0,y1 in [(.0,.25),(.25,.50),(.50,.75),(.75,1.)]:
                z = gray[int(h*y0):int(h*y1),:]
                bl = cv2.GaussianBlur(z,(5,5),0)
                feats.append(float((z.astype(float)-bl).std()))

            # Sharpness
            body = gray[int(h*.25):int(h*.85),:int(w*.60)]
            feats.append(float(cv2.Laplacian(body.astype(np.float32), cv2.CV_32F).var()))

            return np.array(feats).reshape(1, -1)

        def ml_detector():
            if ML_MODEL is None:
                return []
            features = extract_ml_features(reader, raw)
            tamper_prob = float(ML_MODEL.predict_proba(features)[0][1])
            severity = tamper_prob
            return [Finding(
                category="ML Model Prediction",
                description=f"Predicted tampered probability {tamper_prob:.2f}",
                severity=severity
            )]

        detector_fns = [
            lambda: metadata_detector.detect(reader, raw),
            lambda: signature_detector.detect(reader, raw),
            lambda: annotation_detector.detect(reader),
            lambda: image_detector.detect(reader),
            lambda: font_detector.detect(reader),
            lambda: url_detector.detect(reader, raw),
            lambda: script_detector.detect(reader, raw),
            lambda: encoding_detector.detect(raw),
            lambda: hidden_text_detector.detect(reader),
            lambda: incremental_update_detector.detect(raw),
            lambda: structure_detector.detect(raw),
            lambda: visual_forensics_detector.detect(reader),
            lambda: image_text_detector.detect(reader),
            lambda: cnn_detector.detect(reader),
            lambda: visual_forensics_v2.detect(reader),
            lambda: text_layer_detector.detect(reader),
            lambda: document_integrity_detector.detect(reader),
            lambda: pan_card_detector.detect(reader),
            ml_detector
        ]

        if self.enable_ocr:
            detector_fns.append(lambda: ocr_detector.detect(reader))
            detector_fns.append(lambda: content_anomaly_detector.detect(reader))

        for fn in detector_fns:
            try:
                findings = fn()
                result.findings.extend(findings)
                if any(f.category in ("OCR Anomaly", "Content Anomaly",
                                      "Unnatural Text Weight", "Font Type Change") for f in findings):
                    result.ocr_used = True
            except Exception as exc:
                logger.debug("Detector error: %s", exc, exc_info=True)

        if self.enable_heatmap:
            result.heatmap_data = self._build_heatmap(result.findings)

        result.risk_score  = _compute_score(result.findings)
        result.confidence  = _score_to_confidence(result.risk_score)
        result.tampered    = result.risk_score >= 30.0
        result.tamper_type = _classify_tamper_type(result.findings, result.risk_score)
        result.processing_time = round(time.time() - t0, 3)
        return result

    @staticmethod
    def _open_reader(raw: bytes):
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(raw))
            if reader.is_encrypted:
                for pwd in COMMON_PASSWORDS:
                    try:
                        if reader.decrypt(pwd):
                            return reader, True
                    except Exception:
                        pass
                return None, True
            return reader, False
        except Exception as exc:
            logger.debug("pypdf open failed: %s", exc)
            return None, False

    @staticmethod
    def _build_heatmap(findings: list) -> list:
        heatmap = []
        for f in findings:
            if f.page is None:
                continue
            evidence = f.evidence or {}
            word_rects = evidence.get("word_rects", [])
            if word_rects:
                for rect in word_rects:
                    if rect and isinstance(rect, dict):
                        heatmap.append({
                            "page":   f.page,
                            "x":      rect.get("x", 0),
                            "y":      rect.get("y", 0),
                            "w":      rect.get("w", 30),
                            "h":      rect.get("h", 12),
                            "weight": f.severity,
                            "label":  rect.get("text", f.category),
                        })
                continue
            rect = evidence.get("rect")
            if rect:
                heatmap.append({
                    "page":   f.page,
                    "x":      rect[0],
                    "y":      rect[1],
                    "w":      rect[2],
                    "h":      rect[3],
                    "weight": f.severity,
                    "label":  f.category,
                })
            else:
                heatmap.append({
                    "page":   f.page,
                    "x":      0,
                    "y":      0,
                    "w":      1.0,
                    "h":      1.0,
                    "weight": f.severity * 0.5,
                    "label":  f.category,
                })
        return heatmap