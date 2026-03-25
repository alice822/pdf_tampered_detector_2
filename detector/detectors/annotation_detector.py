
from __future__ import annotations
from typing import List
from ..engine import Finding

SUSPICIOUS_SUBTYPES = {
    "/Redact": ("Redaction annotation found — may be used to hide original content.", 0.65),
    "/FreeText": ("Free-text annotation may overlay or replace original text.", 0.35),
    "/Stamp": ("Stamp annotation present — verify authenticity.", 0.3),
    "/Ink": ("Ink annotation detected — handwritten overlay possible.", 0.2),
}


def detect(reader) -> List[Finding]:
    findings: List[Finding] = []
    total_annots = 0
    page_count = len(reader.pages)

    for page_num, page in enumerate(reader.pages, 1):
        try:
            annots = page.get("/Annots")
            if not annots:
                continue
            annot_list = annots
            # resolve indirect references
            try:
                annot_list = list(annots)
            except Exception:
                continue

            total_annots += len(annot_list)

            for annot in annot_list:
                try:
                    obj = annot.get_object() if hasattr(annot, "get_object") else annot
                    subtype = str(obj.get("/Subtype", ""))
                    if subtype in SUSPICIOUS_SUBTYPES:
                        msg, sev = SUSPICIOUS_SUBTYPES[subtype]
                        rect = obj.get("/Rect")
                        evidence = {"subtype": subtype}
                        if rect:
                            evidence["rect"] = [float(x) for x in rect]
                        findings.append(Finding(
                            category="Annotation",
                            description=msg,
                            severity=sev,
                            page=page_num,
                            evidence=evidence,
                        ))

                    # White-fill rectangle covering content
                    _check_white_cover(obj, page_num, findings)

                except Exception:
                    pass
        except Exception:
            pass

    # High annotation density
    if page_count > 0 and total_annots / page_count > 10:
        findings.append(Finding(
            category="Annotation",
            description=f"Unusually high annotation density ({total_annots} annotations across {page_count} pages). May indicate automated overlay.",
            severity=0.4,
            evidence={"total_annots": total_annots, "page_count": page_count},
        ))

    return findings


def _check_white_cover(obj, page_num: int, findings: list):
    """Detect annotation that draws a white rectangle (common cover-up)."""
    try:
        ap = obj.get("/AP")
        if not ap:
            return
        n_stream = ap.get("/N")
        if n_stream is None:
            return
        stream_obj = n_stream.get_object() if hasattr(n_stream, "get_object") else n_stream
        data = stream_obj.get_data() if hasattr(stream_obj, "get_data") else b""
        if b"1 1 1 rg" in data or b"1 g" in data:
            # white fill used
            findings.append(Finding(
                category="Annotation",
                description="Annotation uses white fill — may be covering original content.",
                severity=0.6,
                page=page_num,
            ))
    except Exception:
        pass
