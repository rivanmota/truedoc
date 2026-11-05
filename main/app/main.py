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
from app import context as image_context_module
from app import pdf_utils

app = FastAPI(title="Local AI Image Detector", version="0.3.4")

# The combiner lives in app/scoring.py so the PDF path can share it; these
# names are kept for existing callers (tests, eval/calibrate_weights.py).
from app.scoring import (  # noqa: E402,F401
    CONFIDENCE_FACTOR,
    GRADIENT_FALLBACK_SIGNALS,
    NO_EVIDENCE_SCORE,
    SINGLE_SIGNAL_CAP,
    SUBSTANTIVE_SIGNALS,
    WEIGHTS,
    _DEFAULT_MODEL,
)
from app.scoring import combine as _combine  # noqa: E402
from app.scoring import signal_applicable as _signal_applicable  # noqa: E402
from app.scoring import visual_score as _visual_score  # noqa: E402

_SCORING_REEXPORTS = (
    CONFIDENCE_FACTOR, GRADIENT_FALLBACK_SIGNALS, NO_EVIDENCE_SCORE, SINGLE_SIGNAL_CAP,
    SUBSTANTIVE_SIGNALS, WEIGHTS, _signal_applicable,
)

# Geometry-level signals (gradients, lighting) run on a copy downscaled to
# at most MAX_DIM on the long side; that only changes scale, not what they
# measure.
MAX_DIM = 1536
# Pixel-level signals (noise, frequency) must see the file's own pixels:
# downscaling averages grain away and pushes an upscaling knee past
# Nyquist. They run on a native-resolution centre crop of at most
# NATIVE_CROP x NATIVE_CROP to bound FFT cost.
NATIVE_CROP = 2048

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


def _load_image(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Returns (bgr uint8, gray float32, native_gray float32, (height, width)).

    ``bgr``/``gray`` are downscaled to MAX_DIM on the long side for the
    geometry-level signals. ``native_gray`` is a centre crop at the file's
    own resolution (at most NATIVE_CROP square) for the pixel-level ones.
    The returned size is the decoded image's native size.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image (unsupported or corrupt file)")

    h, w = bgr.shape[:2]
    y0 = max(0, (h - NATIVE_CROP) // 2)
    x0 = max(0, (w - NATIVE_CROP) // 2)
    native = bgr[y0:y0 + NATIVE_CROP, x0:x0 + NATIVE_CROP]
    native_gray = cv2.cvtColor(native, cv2.COLOR_BGR2GRAY).astype(np.float32)

    scale = min(1.0, MAX_DIM / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return bgr, gray, native_gray, (h, w)


def analyze_image_signals(image_bytes: bytes) -> dict:
    """Runs all five signals on a single PNG/JPEG image and records the
    observable context they were measured under. No weighting happens
    here, so a result can be re-scored under different weight models
    (``eval/calibrate_weights.py`` relies on this)."""
    bgr, gray, native_gray, (h, w) = _load_image(image_bytes)

    freq_result = frequency.analyze(native_gray)
    noise_result = noise.analyze(native_gray)
    grad_result = gradients.analyze(bgr, gray)
    light_result = lighting.analyze(gray)
    meta_result = metadata.analyze(image_bytes)

    context = image_context_module.image_context(
        image_bytes,
        fmt=_detect_format(image_bytes),
        width=w,
        height=h,
        bgr=bgr,
        native_gray=native_gray,
        frequency_result=freq_result,
    )
    return {
        "signals": {
            "frequency": freq_result,
            "noise": noise_result,
            "gradients": grad_result,
            "lighting": light_result,
        },
        "metadata": meta_result,
        "context": context,
    }


def score_image_signals(analysis: dict, model=_DEFAULT_MODEL) -> dict:
    """Combines an ``analyze_image_signals`` result into the response."""
    signals = analysis["signals"]
    context = analysis["context"]
    visual_score, scoring = _visual_score(signals, context["buckets"], model)
    combined = _combine(visual_score, analysis["metadata"])
    return {
        "ai_generated": round(combined, 4),
        "signals": {**signals, "metadata": analysis["metadata"]},
        "scoring": {**scoring, "context": context},
    }


def _run_image_analysis(image_bytes: bytes) -> dict:
    """Runs all five signals on a single PNG/JPEG image and scores them."""
    return score_image_signals(analyze_image_signals(image_bytes))


def _pdf_metadata_verdict(aggregate: dict, explicit_meta: dict, legacy_meta: dict) -> tuple[float, str, bool]:
    """Folds the PDF's document metadata into the visual aggregate.

    Uses the image path's rule (``app.scoring.combine``): a generator
    signature raises the score to at least its own value and never lowers
    stronger visual evidence. Returns (score, confidence, score_applicable).
    """
    visual = float(aggregate.get("score", 0.0))
    for meta in (explicit_meta, legacy_meta):
        if meta.get("confidence") == "high":
            return float(np.clip(_combine(visual, meta), 0, 1)), "high", True
    return visual, aggregate.get("confidence", "insufficient"), aggregate.get("worst_page") is not None


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
        # Page renders and embedded rasters are combined with the same
        # weights, context factors and cap as a direct upload (app.scoring);
        # document.py only adds the PDF gates.
        page_buckets = None
        if profile.get("photo_forensics_applicable"):
            image_result = _run_image_analysis(page_bytes)
            page_signals = image_result["signals"]
            page_buckets = image_result["scoring"]["context"]["buckets"]
            # A 1:1 render reproduces its dominant raster's pixels, so it
            # gets that raster's context too (e.g. the JPEG quality read from
            # its DCTDecode stream), exactly as the embedded image does.
            dominant = next(
                (img["profile"] for img in item.get("embedded_images", [])
                 if img["profile"].get("image_index") == profile.get("dominant_image_index")),
                None,
            )
            if dominant is not None and profile.get("render_resampled") is False:
                page_buckets = image_context_module.embedded_raster_buckets(page_buckets, dominant)
        else:
            page_signals = document.not_applicable_signals(profile)

        page_render_scoring = document.score_pdf_page(page_signals, profile, page_buckets)

        embedded_results = []
        for embedded in item.get("embedded_images", []):
            embedded_profile = embedded["profile"]
            try:
                embedded_image_result = _run_image_analysis(embedded["png_bytes"])
                embedded_signals = embedded_image_result["signals"]
                # The image's own file metadata (EXIF in a DCTDecode stream) is
                # folded in exactly as for an upload; the PNG re-encode used
                # for the pixel signals carries none.
                source = embedded.get("source_bytes")
                embedded_scoring = document.score_embedded_image(
                    embedded_signals,
                    embedded_profile,
                    image_context_module.embedded_raster_buckets(
                        embedded_image_result["scoring"]["context"]["buckets"],
                        embedded_profile,
                    ),
                    metadata_result=metadata.analyze(source) if source else None,
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
    combined, overall_confidence, score_applicable = _pdf_metadata_verdict(
        aggregate, explicit_meta, legacy_doc_meta
    )
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
