"""
Flask Web API  (v4 — Windows-safe heatmap file handling)
"""
from __future__ import annotations
import os
import gc
import tempfile
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_file

from detector import PDFTamperDetector
from detector.heatmap import generate_heatmap_pdf
from detector.image_converter import maybe_convert_to_pdf, is_image_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

INDEX_HTML = Path(__file__).parent / "index.html"
detector   = PDFTamperDetector(enable_ocr=True, enable_heatmap=True)


@app.route("/")
def index():
    return send_file(INDEX_HTML)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    ALLOWED = {'pdf','jpg','jpeg','png','webp','bmp','tiff','tif'}
    ext = f.filename.rsplit('.', 1)[-1].lower() if '.' in f.filename else ''
    if not f.filename or ext not in ALLOWED:
       return jsonify({"error": "Please upload a PDF or image (JPG, PNG, WEBP)"}), 400
    suffix = '.' + ext
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp.name)
        upload_path = tmp.name

    pdf_path, was_converted = maybe_convert_to_pdf(upload_path)
    try:
        result = detector.analyze(pdf_path)
        if result.error:
            return jsonify({"error": result.error}), 422

        # Extract document metadata for general info panel
        from pypdf import PdfReader
        doc_info = {}
        try:
            rdr = PdfReader(pdf_path)
            meta = rdr.metadata or {}
            doc_info = {
                "producer":      str(meta.get("/Producer", "") or ""),
                "creator":       str(meta.get("/Creator", "")  or ""),
                "author":        str(meta.get("/Author", "")   or ""),
                "title":         str(meta.get("/Title", "")    or ""),
                "pages":         len(rdr.pages),
                "pdf_version":   getattr(rdr, "pdf_header", ""),
                "created":       str(meta.get("/CreationDate", "") or ""),
                "modified":      str(meta.get("/ModDate", "")      or ""),
                "encrypted":     rdr.is_encrypted,
            }
        except Exception:
            pass

        return jsonify({
            "tampered":         result.tampered,
            "risk_score":       result.risk_score,
            "confidence":       result.confidence,
            "tamper_type":      result.tamper_type,
            "total_pages":      result.total_pages,
            "processing_time":  result.processing_time,
            "file_hash":        result.file_hash,
            "encrypted":        result.encrypted,
            "ocr_used":         result.ocr_used,
            "human_readable":   result.human_readable(),
            "document_info":    doc_info,
            "findings": [
                {
                    "category":    fn.category,
                    "description": fn.description,
                    "severity":    round(fn.severity, 2),
                    "page":        fn.page,
                    "evidence":    fn.evidence,
                }
                for fn in result.top_findings(10)
            ],
            "heatmap_available": bool(result.heatmap_data),
        })
    finally:
        _safe_unlink(upload_path)
        if was_converted:
            _safe_unlink(pdf_path)


@app.route("/analyze/heatmap", methods=["POST"])
def analyze_heatmap():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    
    
    ALLOWED = {'pdf','jpg','jpeg','png','webp','bmp','tiff','tif'}
    ext = f.filename.rsplit('.', 1)[-1].lower() if '.' in f.filename else ''
    if not f.filename or ext not in ALLOWED:
        return jsonify({"error": "Please upload a PDF or image file"}), 400

    suffix = '.' + ext
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        f.save(tmp_in.name)
        in_path = tmp_in.name

    # Convert image to PDF if needed
    in_path_original = in_path
    in_path, was_converted = maybe_convert_to_pdf(in_path)

    # Use a separate temp file for output — never derive name from in_path
    out_fd, out_path = tempfile.mkstemp(suffix="_heatmap.pdf")
    os.close(out_fd)

    try:
        result = detector.analyze(in_path)
        if result.error:
            return jsonify({"error": result.error}), 422

        if not result.heatmap_data and not result.findings:
            return jsonify({"error": "No suspicious regions to highlight."}), 200

        success = generate_heatmap_pdf(
            in_path, result.heatmap_data, out_path, findings=result.findings
        )

        if not success or not Path(out_path).exists():
            return jsonify({"error": "Heatmap generation failed. Check reportlab is installed."}), 500

        # Read into memory first so we can delete the file before Flask streams it
        with open(out_path, "rb") as fh:
            pdf_bytes = fh.read()

        # Force GC + small delay for Windows file handle release
        gc.collect()
        time.sleep(0.05)

        from flask import Response
        resp = Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=forensic_heatmap.pdf",
                "Content-Length": str(len(pdf_bytes)),
            }
        )
        return resp

    finally:
        _safe_unlink(in_path_original)
        if was_converted:
            _safe_unlink(in_path)
        _safe_unlink(out_path)


def _safe_unlink(path: str, retries: int = 5, delay: float = 0.1):
    """Delete a file safely on Windows where files may still be open."""
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                os.unlink(path)
            return
        except PermissionError:
            if attempt < retries - 1:
                gc.collect()
                time.sleep(delay * (attempt + 1))
        except Exception:
            return


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)