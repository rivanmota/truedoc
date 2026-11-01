"""
Local AI-generated image detector with a Sightengine-compatible
`/1.0/check.json` endpoint (see https://sightengine.com/docs/ai-generated-image-detection).

Accepts either a multipart file upload (`media`) or a `url`, plus a
`models` param (only `genai` is supported locally). Supported file
formats: PNG, JPEG, and PDF (each PDF page is rasterized and analyzed
individually — see `app/pdf_utils.py`). Format is sniffed from the
file's magic bytes, not trusted from the filename or an HTTP
Content-Type header.

By default the response is compact. Pass `details=true` as a query parameter to include full per-signal, per-page, and embedded-image diagnostics.

Response is shaped like Sightengine's:

    {
      "status": "success",
      "request": {"id": ..., "timestamp": ..., "operations": 1},
      "type": {
        "ai_generated": 0.73,
        "signals": { ...per-signal breakdown, not part of the real API... }
      },
      "media": {"id": ..., "uri": ...}
    }

For a PDF, `type` additionally carries per-page and embedded-raster
forensics. Native text/vector pages are not treated as camera images. The
top-level risk is driven by the highest-scoring reliable page render or
embedded raster image, so one suspicious element cannot be diluted by clean
pages.

`api_user` / `api_secret` are accepted and ignored — kept only so
existing Sightengine client code needs nothing but a base-URL change
to point at this locally.

This is signal-based (frequency spectrum, sensor-noise residual,
texture smoothness, lighting consistency, metadata), not a trained
classifier — see each analyzer module's docstring and the README for
accuracy caveats. It will not match Sightengine's accuracy or its
per-generator identification.
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.analyzers import (
    document,
    frequency,
    gradients,
    lighting,
    metadata,
    noise,
    pdf_metadata_consistency,
    pdf_revisions,
    pdf_security,
    pdf_signatures,
    pdf_structure,
)
from app import pdf_utils

app = FastAPI(title="Local AI Image Detector", version="0.3.4")

# Weights across the four visual signals (metadata is handled
# separately since a signature hit should dominate the score
# outright, not just average in).
WEIGHTS = {
    "frequency": 0.05,
    "noise": 0.45,
    "gradients": 0.35,
    "lighting": 0.15,
}

MAX_DIM = 1536  # downscale very large images for speed; doesn't affect the signals materially

SUPPORTED_FORMATS = "png, jpg/jpeg, pdf"


def _detect_format(data: bytes) -> str:
    """Sniffs the file format from magic bytes rather than trusting a
    filename or Content-Type header."""
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    return "unknown"


def _load_image(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Returns (bgr uint8 array, gray float32 array), resized if huge."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image (unsupported or corrupt file)")

    h, w = bgr.shape[:2]
    scale = min(1.0, MAX_DIM / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return bgr, gray


def _combine(visual_score: float, meta_result: dict) -> float:
    """A strong metadata signature (explicit generator fingerprint)
    overrides rather than just blends in. A camera-EXIF signal pulls
    the score down moderately. Otherwise metadata stays neutral and
    doesn't affect the visual-signal average."""
    meta_score = meta_result["score"]
    if meta_result.get("confidence") == "high":
        combined = max(visual_score, meta_score)
    elif meta_result.get("confidence") == "low-medium":
        combined = 0.85 * visual_score + 0.15 * meta_score
    else:
        combined = visual_score
    return float(np.clip(combined, 0, 1))


def _run_image_analysis(image_bytes: bytes) -> dict:
    """Runs all five signals on a single PNG/JPEG image."""
    bgr, gray = _load_image(image_bytes)

    freq_result = frequency.analyze(gray)
    noise_result = noise.analyze(gray)
    grad_result = gradients.analyze(bgr, gray)
    light_result = lighting.analyze(gray)
    meta_result = metadata.analyze(image_bytes)

    visual_score = (
        WEIGHTS["frequency"] * freq_result["score"]
        + WEIGHTS["noise"] * noise_result["score"]
        + WEIGHTS["gradients"] * grad_result["score"]
        + WEIGHTS["lighting"] * light_result["score"]
    )
    combined = _combine(visual_score, meta_result)

    return {
        "ai_generated": round(combined, 4),
        "signals": {
            "frequency": freq_result,
            "noise": noise_result,
            "gradients": grad_result,
            "lighting": light_result,
            "metadata": meta_result,
        },
    }


def _run_pdf_analysis(pdf_bytes: bytes) -> dict:
    """Document-aware, fraud-sensitive PDF analysis.

    Whole-page camera/photo forensics are used only for image-dominant pages.
    Every embedded raster image is also analyzed independently, regardless of
    how little page area it occupies. The document score is driven by the
    highest-scoring *reliable* evidence unit (page render or embedded image),
    so clean pages cannot camouflage one suspicious element.
    """
    rendered_pages = pdf_utils.render_pdf_pages_with_profiles(pdf_bytes)
    if not rendered_pages:
        raise ValueError("PDF has no renderable pages")

    pages = []
    for item in rendered_pages:
        page_bytes = item["png_bytes"]
        profile = item["profile"]

        # Whole-page analysis is valid only when raster content dominates.
        if profile.get("photo_forensics_applicable"):
            image_result = _run_image_analysis(page_bytes)
            page_signals = image_result["signals"]
        else:
            page_signals = document.not_applicable_signals(profile)

        page_render_scoring = document.score_pdf_page(page_signals, profile)

        embedded_results = []
        for embedded in item.get("embedded_images", []):
            embedded_profile = embedded["profile"]
            try:
                embedded_image_result = _run_image_analysis(embedded["png_bytes"])
                embedded_signals = embedded_image_result["signals"]
                embedded_scoring = document.score_embedded_image(
                    embedded_signals,
                    embedded_profile,
                )
            except Exception as exc:
                embedded_signals = document.unavailable_embedded_signals(str(exc))
                embedded_scoring = document.unusable_embedded_image_score(
                    embedded_profile,
                    reason=f"embedded image analysis failed: {exc}",
                )

            embedded_results.append({
                "image_index": embedded_profile["image_index"],
                "page_coverage": embedded_profile["page_coverage"],
                "width_px": embedded_profile["width_px"],
                "height_px": embedded_profile["height_px"],
                "filters": embedded_profile.get("filters", []),
                "ai_generated": embedded_scoring["score"],
                "confidence": embedded_scoring["confidence"],
                "usable": embedded_scoring["usable"],
                "scoring": embedded_scoring,
                "signals": embedded_signals,
            })

        page_risk = document.combine_page_evidence(
            page_number=item["page"],
            page_render_scoring=page_render_scoring,
            embedded_images=embedded_results,
        )

        # Keep the signal set aligned with the evidence unit that drove risk.
        if page_risk.get("source") == "embedded_image":
            idx = page_risk.get("image_index")
            driving = next(
                (x for x in embedded_results if x["image_index"] == idx),
                None,
            )
            driving_signals = driving["signals"] if driving else page_signals
        else:
            driving_signals = page_signals

        pages.append({
            "page": item["page"],
            "ai_generated": page_risk["score"],
            "confidence": page_risk["confidence"],
            "content_type": profile["content_type"],
            "risk_source": page_risk.get("source"),
            "risk_image_index": page_risk.get("image_index"),
            "page_profile": profile,
            "page_render_scoring": page_render_scoring,
            # Backward-compatible key consumed by aggregate_pdf_pages().
            "pdf_scoring": page_risk,
            "embedded_images": embedded_results,
            "signals": driving_signals,
        })

    aggregate = document.aggregate_pdf_pages(pages)

    # PDF-level fraud/tamper diagnostics are intentionally kept separate from
    # ai_generated. A document can be structurally suspicious without being
    # AI-generated.
    metadata_consistency = pdf_metadata_consistency.analyze(pdf_bytes)
    revisions_result = pdf_revisions.analyze(pdf_bytes)
    signatures_result = pdf_signatures.analyze(pdf_bytes)
    structure_result = pdf_structure.analyze(pdf_bytes)
    security_result = pdf_security.analyze(pdf_bytes)

    anomaly_components = {
        "metadata_consistency": metadata_consistency,
        "revisions": revisions_result,
        "signatures": signatures_result,
        "structure": structure_result,
        "security": security_result,
    }
    anomaly_score = max(
        (float(v.get("score", 0.0)) for v in anomaly_components.values()),
        default=0.0,
    )

    anomaly_flags = []
    for component_name, component in anomaly_components.items():
        for anomaly in component.get("anomalies", []):
            if anomaly.get("severity") in {"medium", "high"}:
                anomaly_flags.append({
                    "source": component_name,
                    "code": anomaly.get("code"),
                    "severity": anomaly.get("severity"),
                    **({"page": anomaly.get("page")} if anomaly.get("page") is not None else {}),
                })

    if anomaly_score >= 0.80:
        anomaly_classification = "high_document_anomaly"
    elif anomaly_score >= 0.45:
        anomaly_classification = "moderate_document_anomaly"
    elif anomaly_score >= 0.20:
        anomaly_classification = "low_document_anomaly"
    else:
        anomaly_classification = "no_significant_document_anomaly"

    document_anomalies = {
        "score": round(anomaly_score, 4),
        "classification": anomaly_classification,
        "flag_count": len(anomaly_flags),
        "flags": anomaly_flags,
        "components": anomaly_components,
    }

    doc_fields = pdf_utils.read_pdf_metadata(pdf_bytes)
    explicit_meta = document.analyze_pdf_metadata(doc_fields)

    if hasattr(metadata, "analyze_pdf_document"):
        legacy_doc_meta = metadata.analyze_pdf_document(doc_fields)
    else:
        legacy_doc_meta = {
            "score": 0.5,
            "confidence": "low",
            "pdf_fields": doc_fields,
            "note": "no legacy PDF metadata analyzer available",
        }

    explicit_metadata_hit = explicit_meta.get("confidence") == "high"
    legacy_metadata_hit = legacy_doc_meta.get("confidence") == "high"

    if explicit_metadata_hit:
        combined = float(explicit_meta["score"])
        overall_confidence = "high"
        score_applicable = True
    elif legacy_metadata_hit:
        combined = max(float(aggregate["score"]), float(legacy_doc_meta["score"]))
        overall_confidence = "high"
        score_applicable = True
    else:
        combined = float(aggregate["score"])
        overall_confidence = aggregate["confidence"]
        score_applicable = aggregate.get("worst_page") is not None

    combined = float(np.clip(combined, 0, 1))
    classification, description = document.classify_pdf(
        combined,
        overall_confidence,
        metadata_hit=explicit_metadata_hit or legacy_metadata_hit,
    )

    aggregate_worst_page = aggregate.get("worst_page")
    if aggregate_worst_page is not None:
        worst = next(p for p in pages if p["page"] == aggregate_worst_page)
    else:
        worst = max(pages, key=lambda p: float(p.get("ai_generated", 0.0)))

    flagged_embedded_images = []
    for page in pages:
        for img in page.get("embedded_images", []):
            if (
                img.get("usable")
                and img.get("confidence") in {"low-medium", "medium", "medium-high", "high"}
                and float(img.get("ai_generated", 0.0)) >= 0.40
            ):
                flagged_embedded_images.append({
                    "page": page["page"],
                    "image_index": img["image_index"],
                    "score": round(float(img["ai_generated"]), 4),
                    "confidence": img["confidence"],
                    "page_coverage": img["page_coverage"],
                    "width_px": img["width_px"],
                    "height_px": img["height_px"],
                })
    flagged_embedded_images.sort(key=lambda x: x["score"], reverse=True)

    return {
        "ai_generated": round(combined, 4),
        "score_applicable": score_applicable,
        "classification": classification,
        "description": description,
        "confidence": overall_confidence,
        "media_type": "pdf",
        "analysis_mode": "document_aware_embedded_images",
        "signals": {
            **{k: v for k, v in worst["signals"].items() if k != "metadata"},
            "metadata": legacy_doc_meta,
            "pdf_generator_metadata": explicit_meta,
            "document_aggregate": aggregate,
        },
        "page_count": len(pages),
        "worst_page": aggregate.get("worst_page"),
        "worst_page_confidence": (
            worst.get("pdf_scoring", {}).get("confidence")
            if aggregate.get("worst_page") is not None
            else "insufficient"
        ),
        "worst_evidence_source": (
            worst.get("pdf_scoring", {}).get("source")
            if aggregate.get("worst_page") is not None
            else None
        ),
        "worst_embedded_image": (
            worst.get("pdf_scoring", {}).get("image_index")
            if aggregate.get("worst_page") is not None
            else None
        ),
        "flagged_pages": aggregate.get("flagged_pages", []),
        "flagged_embedded_images": flagged_embedded_images,
        "document_mean": aggregate.get("mean_page_score", 0.0),
        "aggregation": aggregate.get("aggregation", "insufficient"),
        "document_anomalies": document_anomalies,
        "pages": pages,
    }


def _compact_result(result: dict, fmt: str) -> dict:
    """Return the production response. Full diagnostics are opt-in via details=true."""
    if fmt != "pdf":
        return {"ai_generated": result["ai_generated"]}

    score_applicable = bool(result.get("score_applicable"))
    compact = {
        "ai_generated": result.get("ai_generated") if score_applicable else None,
        "score_applicable": score_applicable,
        "classification": result.get("classification"),
        "description": result.get("description"),
        "confidence": result.get("confidence"),
        "media_type": "pdf",
        "page_count": result.get("page_count", 0),
        "flagged_pages": result.get("flagged_pages", []),
        "flagged_embedded_images": result.get("flagged_embedded_images", []),
    }

    anomalies = result.get("document_anomalies", {})
    compact_flags = []
    for flag in anomalies.get("flags", []):
        code = flag.get("code") if isinstance(flag, dict) else str(flag)
        if code and code not in compact_flags:
            compact_flags.append(code)

    compact["document_anomalies"] = {
        "score": anomalies.get("score", 0.0),
        "classification": anomalies.get("classification", "no_significant_document_anomaly"),
        "flags": compact_flags[:10],
    }

    if score_applicable and result.get("worst_page") is not None:
        compact["worst_evidence"] = {
            "page": result.get("worst_page"),
            "source": result.get("worst_evidence_source"),
            "image_index": result.get("worst_embedded_image"),
            "score": result.get("ai_generated"),
        }

    return compact

def _envelope(result: dict, uri: str, started: float) -> dict:
    return {
        "status": "success",
        "request": {
            "id": f"req_{uuid.uuid4().hex[:20]}",
            "timestamp": started,
            "operations": 1,
        },
        "type": result,
        "media": {
            "id": f"med_{uuid.uuid4().hex[:20]}",
            "uri": uri,
        },
    }


def _error(message: str, code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={
            "status": "failure",
            "error": {"type": "invalid_request", "message": message},
        },
    )


@app.post("/1.0/check.json")
async def check(
    models: str = Form(...),
    media: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    details: bool = False,
    api_user: Optional[str] = Form(None),
    api_secret: Optional[str] = Form(None),
):
    return await _handle_check(models, media, url, details)


@app.get("/1.0/check.json")
async def check_get(
    models: str,
    url: Optional[str] = None,
    details: bool = False,
    api_user: Optional[str] = None,
    api_secret: Optional[str] = None,
):
    return await _handle_check(models, None, url, details)


async def _handle_check(models: str, media: Optional[UploadFile], url: Optional[str], details: bool = False):
    started = time.time()

    if "genai" not in [m.strip() for m in models.split(",")]:
        return _error("only the 'genai' model is supported by this local server")

    if media is None and not url:
        return _error("provide either a 'media' file upload or a 'url' parameter")

    if media is not None:
        file_bytes = await media.read()
        uri = media.filename or "upload"
    else:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            file_bytes = resp.content
        except Exception as e:
            return _error(f"could not fetch url: {e}")
        uri = url

    fmt = _detect_format(file_bytes)
    if fmt == "unknown":
        return _error(f"unsupported file format — supported formats: {SUPPORTED_FORMATS}")

    try:
        if fmt == "pdf":
            result = _run_pdf_analysis(file_bytes)
        else:
            result = _run_image_analysis(file_bytes)
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"analysis failed: {e}")

    response_result = result if details else _compact_result(result, fmt)
    return _envelope(response_result, uri, started)


@app.get("/health")
async def health():
    return {"status": "ok"}
