# PDF Tamper Detection System

A modular, production-ready system for detecting manual and AI-assisted tampering in PDF documents.

---

## Features

| Detector | What it catches |
|---|---|
| **Metadata** | Forged dates, suspicious producer tools, XMP vs Info mismatch |
| **Digital Signatures** | Broken byte-ranges, DocMDP violations, shadow attacks |
| **Annotations** | Redactions, white-fill overlays, auto-trigger actions |
| **Images** | Low-DPI replacements, ELA noise inconsistency, double-JPEG compression |
| **Fonts** | Mixed families, non-embedded fonts, AI-tool font signatures |
| **Hidden Text** | Invisible render mode, zero-size text, off-page content, OCG layers |
| **Incremental Updates** | Multiple revisions, disproportionate update sizes, object ID reuse |
| **Encoding/Structure** | Obfuscation filters, xref mismatches, pre-header junk, post-EOF data |
| **Scripts/Malware** | JavaScript, Launch actions, heap-spray patterns, embedded files |
| **URLs** | Shorteners, IP-based links, suspicious TLDs, javascript: URIs |
| **OCR Analysis** | Low-confidence regions, text layer vs visible content divergence |

---

## Output Format

```
Tampered  : YES
Confidence: 91%
Risk Score: 78/100
Type      : Likely AI-assisted edit

Reasons:
  [🔴 HIGH] Digital Signature: Signature ByteRange ends at 48320 but file size is 51200.
  [🔴 HIGH] Incremental Update: Critical PDF objects ([1, 2]) redefined in incremental updates.
  [🟡 MED]  Image Compression Anomaly: ELA analysis shows high noise variance (σ/μ = 4.2).
  [🟡 MED]  Metadata Inconsistency: Modification date is earlier than creation date.
  [🟢 LOW]  Annotation: Free-text annotation may overlay or replace original text.
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Install system dependencies (poppler, tesseract, libmagic) — see `requirements.txt` for OS-specific instructions.

### 2. Command-line usage

```bash
# Basic analysis
python cli.py document.pdf

# With heatmap overlay
python cli.py document.pdf --heatmap flagged.pdf

# JSON output (for integration)
python cli.py document.pdf --json

# Skip OCR (faster)
python cli.py document.pdf --no-ocr
```

Exit code: `1` = tampered, `0` = clean.

### 3. Web API

```bash
python app.py
# → http://localhost:5000
```

#### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/analyze` | Analyze PDF → JSON |
| `POST` | `/analyze/heatmap` | Analyze PDF → annotated PDF download |

#### API example

```bash
curl -X POST http://localhost:5000/analyze \
  -F "file=@document.pdf" | python -m json.tool
```

### 4. Python API

```python
from detector import PDFTamperDetector

det = PDFTamperDetector(enable_ocr=True, enable_heatmap=True)
result = det.analyze("document.pdf")

print(result.human_readable())
# result.tampered        → bool
# result.risk_score      → float 0–100
# result.confidence      → float 0–100
# result.tamper_type     → str
# result.findings        → list[Finding]
# result.heatmap_data    → list[dict]
```

---

## Architecture

```
pdf_tamper_detector/
├── detector/
│   ├── __init__.py          # Public API
│   ├── engine.py            # Orchestrator + AnalysisResult
│   ├── heatmap.py           # PDF heatmap overlay generator
│   └── detectors/
│       ├── metadata_detector.py
│       ├── signature_detector.py
│       ├── annotation_detector.py
│       ├── image_detector.py
│       ├── font_detector.py
│       ├── url_detector.py
│       ├── script_detector.py
│       ├── encoding_detector.py
│       ├── hidden_text_detector.py
│       ├── incremental_update_detector.py
│       ├── structure_detector.py
│       └── ocr_detector.py
├── app.py                   # Flask web API
├── cli.py                   # Command-line interface
├── index.html               # Frontend UI
└── requirements.txt
```

Each detector is **independent** — it receives the pypdf reader and/or raw bytes and returns a list of `Finding` objects. Adding a new detector requires only:
1. Create `detector/detectors/my_detector.py` with a `detect(...)` function
2. Import and call it in `engine.py`

---

## Scoring Model

- Each finding carries a `severity` from 0.0–1.0
- Findings are weighted by severity tier (high = 1.0×, med = 0.65×, low = 0.35×)
- Raw risk score = weighted sum, capped at 100
- Confidence = sigmoid mapping of risk score (0→5%, 40→50%, 90→98%)
- Tamper threshold: risk score ≥ 25

---

## Notes

- **No pre-training required** — all detectors are rule-based and statistical; they work on any PDF without prior ingestion.
- **Encrypted PDFs** — common passwords are tried automatically; if decryption fails the file is flagged.
- **OCR** requires Tesseract and poppler. Disable with `--no-ocr` if not installed.
- **Heatmap** requires reportlab. Regions are colour-coded: 🔴 red = high risk, 🟠 orange = medium, 🟡 yellow = low.
