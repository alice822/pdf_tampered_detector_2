#!/usr/bin/env python3
"""
PDF Tamper Detection – Command Line Interface
=============================================

Usage:
  python cli.py document.pdf
  python cli.py document.pdf --heatmap output_heatmap.pdf
  python cli.py document.pdf --json
  python cli.py document.pdf --no-ocr
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from detector import PDFTamperDetector
from detector.heatmap import generate_heatmap_pdf


def main():
    parser = argparse.ArgumentParser(
        description="PDF Tamper Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pdf", help="Path to PDF file to analyze")
    parser.add_argument("--heatmap", metavar="OUTPUT.PDF", help="Generate heatmap overlay PDF")
    parser.add_argument("--json", action="store_true", help="Output full JSON results")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR analysis (faster)")
    parser.add_argument("--max-findings", type=int, default=10, help="Max findings to display (default: 10)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[ERROR] File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    print(f"🔍 Analyzing: {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB)")
    print("   Please wait…\n")

    detector = PDFTamperDetector(
        enable_ocr=not args.no_ocr,
        enable_heatmap=True,
    )

    result = detector.analyze(str(pdf_path))

    if result.error:
        print(f"[ERROR] {result.error}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        output = {
            "tampered": result.tampered,
            "risk_score": result.risk_score,
            "confidence": result.confidence,
            "tamper_type": result.tamper_type,
            "total_pages": result.total_pages,
            "processing_time": result.processing_time,
            "file_hash": result.file_hash,
            "encrypted": result.encrypted,
            "ocr_used": result.ocr_used,
            "findings": [
                {
                    "category": f.category,
                    "description": f.description,
                    "severity": round(f.severity, 3),
                    "page": f.page,
                    "evidence": f.evidence,
                }
                for f in result.top_findings(args.max_findings)
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print("=" * 60)
        print(result.human_readable(max_reasons=args.max_findings))
        print("=" * 60)
        print(f"\n  SHA-256: {result.file_hash[:16]}…")
        print(f"  Processed in {result.processing_time}s | {result.total_pages} pages")
        if result.heatmap_data:
            print(f"  Suspicious regions found: {len(result.heatmap_data)}")

    # Heatmap output
    if args.heatmap:
        heatmap_path = args.heatmap
        if not result.heatmap_data:
            print("\n[INFO] No suspicious regions detected — heatmap not generated.")
        else:
            ok = generate_heatmap_pdf(str(pdf_path), result.heatmap_data, heatmap_path)
            if ok:
                print(f"\n✅ Heatmap saved: {heatmap_path}")
            else:
                print("\n[WARN] Heatmap generation failed. Install reportlab: pip install reportlab")

    # Exit code: 1 if tampered, 0 if clean
    sys.exit(1 if result.tampered else 0)


if __name__ == "__main__":
    main()
