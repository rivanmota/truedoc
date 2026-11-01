"""Cross-page PDF structural anomaly checks without extra dependencies.

Uses the existing pypdfium2-backed pdf_utils page inspection to find pages that
stand out in dimensions, rotation, text density, raster-image usage, or object
composition. These are corroborating fraud signals, not proof of manipulation.
"""
from __future__ import annotations

from collections import Counter
from statistics import median

import pypdfium2 as pdfium

from app import pdf_utils


def _rotations(pdf_bytes: bytes, max_pages: int) -> list[int]:
    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        out: list[int] = []
        for i in range(min(len(doc), max_pages)):
            page = doc[i]
            try:
                out.append(int(page.get_rotation() or 0) % 4 * 90)
            except Exception:
                out.append(0)
            finally:
                page.close()
        return out
    finally:
        doc.close()


def analyze(pdf_bytes: bytes) -> dict:
    try:
        profiles = pdf_utils.inspect_pdf_pages(pdf_bytes)
        rotations = _rotations(pdf_bytes, len(profiles))
        for p, rotation in zip(profiles, rotations):
            p["rotation"] = rotation
    except Exception as exc:
        return {
            "score": 0.0,
            "confidence": "low",
            "page_count": 0,
            "anomalies": [],
            "note": f"structure analysis unavailable: {exc}",
        }

    anomalies: list[dict] = []
    if not profiles:
        return {"score": 0.0, "confidence": "low", "page_count": 0, "anomalies": []}

    size_keys = [(round(p["width_points"], 1), round(p["height_points"], 1)) for p in profiles]
    common_size, common_size_count = Counter(size_keys).most_common(1)[0]
    if common_size_count >= max(2, len(profiles) // 2):
        for p, size in zip(profiles, size_keys):
            if size != common_size:
                anomalies.append({
                    "code": "page_size_outlier",
                    "severity": "medium",
                    "page": p["page"],
                    "message": "Page dimensions differ from the document's dominant page size.",
                    "page_size": size,
                    "dominant_size": common_size,
                })

    rotation_counts = Counter(p.get("rotation", 0) for p in profiles)
    dominant_rotation, dominant_rotation_count = rotation_counts.most_common(1)[0]
    if dominant_rotation_count >= max(2, len(profiles) // 2):
        for p in profiles:
            if p.get("rotation", 0) != dominant_rotation:
                anomalies.append({
                    "code": "page_rotation_outlier",
                    "severity": "low",
                    "page": p["page"],
                    "message": "Page rotation differs from the document's dominant rotation.",
                    "rotation": p.get("rotation", 0),
                    "dominant_rotation": dominant_rotation,
                })

    if len(profiles) >= 3:
        image_counts = [p.get("image_objects", 0) for p in profiles]
        med_images = float(median(image_counts))
        for p in profiles:
            count = p.get("image_objects", 0)
            if count >= max(3, med_images + 3):
                anomalies.append({
                    "code": "image_resource_outlier",
                    "severity": "medium",
                    "page": p["page"],
                    "message": "Page contains substantially more raster image objects than typical pages in this PDF.",
                    "image_count": count,
                    "median_image_count": med_images,
                })

        text_counts = [p.get("text_chars", 0) for p in profiles]
        med_text = float(median(text_counts))
        if med_text >= 100:
            for p in profiles:
                chars = p.get("text_chars", 0)
                if chars > 0 and (chars > med_text * 4 or chars < med_text * 0.10):
                    anomalies.append({
                        "code": "text_density_outlier",
                        "severity": "low",
                        "page": p["page"],
                        "message": "Page text density differs sharply from the document median.",
                        "text_chars": chars,
                        "median_text_chars": round(med_text, 1),
                    })

        path_counts = [p.get("path_objects", 0) for p in profiles]
        med_paths = float(median(path_counts))
        for p in profiles:
            paths = p.get("path_objects", 0)
            if paths >= max(40, med_paths + 50):
                anomalies.append({
                    "code": "vector_resource_outlier",
                    "severity": "low",
                    "page": p["page"],
                    "message": "Page contains substantially more vector/path objects than typical pages.",
                    "path_objects": paths,
                    "median_path_objects": round(med_paths, 1),
                })

    severity_score = {"low": 0.12, "medium": 0.40, "high": 0.85}
    score = max((severity_score[a["severity"]] for a in anomalies), default=0.0)
    if len({a.get("page") for a in anomalies if a.get("page")}) >= 2:
        score = min(0.75, score + 0.08)

    compact_pages = [
        {
            "page": p["page"],
            "width": p["width_points"],
            "height": p["height_points"],
            "rotation": p.get("rotation", 0),
            "text_chars": p.get("text_chars", 0),
            "image_objects": p.get("image_objects", 0),
            "path_objects": p.get("path_objects", 0),
            "form_objects": p.get("form_objects", 0),
            "content_type": p.get("content_type"),
        }
        for p in profiles
    ]

    return {
        "score": round(score, 4),
        "confidence": "medium",
        "page_count": len(profiles),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "pages": compact_pages,
        "note": "structural outliers are corroborating fraud evidence and may be legitimate in mixed-format documents",
    }
