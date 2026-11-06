"""
Scan API: ``POST /api/v1/scans``.

The same scan the web UI runs, as a self-describing JSON document: one
request in (a multipart file upload or an image/PDF URL), one response out
in two sections:

- ``summary`` - the critical information, in reading order: HTTP status,
  thresholds, final score, classification, confidence, a one-line
  explanation, scan ID, timestamp and engine version. Error responses carry
  the same section (with the error instead of a score).
- ``details`` - file metadata, engine info, every forensic signal with its
  measurements and explanation, how the signals were combined, and (PDFs)
  per-page results. Omitted when the request sends ``details=false``.

This module is a presentation layer only. It calls the exact analysis
functions behind ``/1.0/check.json`` and the UI (``app.main``), so a file
gets the same score here as it does there. What it adds:

- **classification** - the UI's verdict bands (``web/lib/verdict.ts``):
  below 0.35 "likely_real", above 0.65 "likely_ai_generated", otherwise
  "inconclusive"; "insufficient_evidence" for a PDF with no scorable raster
  content (the UI's "not scorable" state).
- **overall confidence for images** - the engine only grades confidence for
  PDF elements. An image is graded with the same rule the PDF path applies
  to an embedded image (``document._tier``: how much of the evidence stack
  was measured), so one picture gets the same confidence whichever way it
  is submitted. An explicit generator signature in the file's metadata makes
  it "high", as in the PDF path.

Input is validated before any analysis (``app/scan_validation.py``): the
request body, the form fields, the URL (public hosts only, every redirect
re-checked) and the file content (format, readable header, pixel limit,
password-free PDF).

It is idempotent: the same input (fields + file bytes) on the same engine
version returns the same response apart from ``scan_id``, ``timestamp``,
``duration_ms`` and the engine version (see ``create_scan`` and
``tests/test_scan_idempotency.py``).

It is stateless: nothing is stored, the response is the result. The scan ID
identifies the response (for logs, support requests, client-side storage);
it cannot be used to fetch the scan again.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from urllib.parse import unquote, urljoin, urlparse

import requests
from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.datastructures import FormData
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import main as engine
from app import scan_validation as validation
from app import scoring, weighting
from app.analyzers import document
from app.scan_validation import ScanError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["scans"])

ENGINE_NAME = "truedoc-forensics"

# Verdict bands. Must match VERDICT_THRESHOLDS in web/lib/verdict.ts so the
# API and the UI classify a score identically (tests/test_scan_api.py checks).
REAL_MAX = 0.35
AI_MIN = 0.65

URL_FETCH_TIMEOUT_S = float(os.environ.get("TRUEDOC_URL_FETCH_TIMEOUT_S", "15"))
MAX_REDIRECTS = 5

SIGNAL_ORDER = ("frequency", "noise", "gradients", "lighting", "metadata")
VISUAL_SIGNALS = ("frequency", "noise", "gradients", "lighting")
# Same labels as web/lib/types.ts SIGNAL_META.
SIGNAL_LABELS = {
    "frequency": "Frequency spectrum",
    "noise": "Sensor noise floor",
    "gradients": "Texture smoothness",
    "lighting": "Lighting consistency",
    "metadata": "Metadata",
}
# Same labels as web/lib/verdict.ts VERDICT_LABEL.
CLASSIFICATION_LABELS = {
    "likely_real": "Likely a true doc",
    "inconclusive": "Inconclusive",
    "likely_ai_generated": "Likely not a true doc",
    "insufficient_evidence": "Not scorable",
}
MIME_TYPES = {"png": "image/png", "jpeg": "image/jpeg", "pdf": "application/pdf"}
# Keys every analyzer result carries; everything else is a measurement.
_SIGNAL_ENVELOPE_KEYS = {"score", "confidence", "applicable", "note"}
_RELIABLE_CONFIDENCES = {"low-medium", "medium", "medium-high", "high"}

Classification = Literal["likely_real", "inconclusive", "likely_ai_generated", "insufficient_evidence"]
SignalVerdict = Literal["likely_real", "inconclusive", "likely_ai", "neutral", "not_applicable"]


# --------------------------------------------------------------------------- #
# Response schema (also drives the OpenAPI docs at /docs)
# --------------------------------------------------------------------------- #
class EngineInfo(BaseModel):
    name: str
    version: str = Field(description="Scan engine version.")
    weighting: str = Field(description="Signal weighting model in use: calibrated context factors or static weights.")
    calibration_generated: Optional[str] = Field(None, description="When the weighting calibration was generated, if one is loaded.")


class FileInfo(BaseModel):
    source: Literal["upload", "url"]
    filename: Optional[str] = None
    url: Optional[str] = None
    size_bytes: int
    sha256: str
    format: Literal["png", "jpeg", "pdf"] = Field(description="Sniffed from the file's bytes, not its name or Content-Type.")
    mime_type: str
    width: Optional[int] = Field(None, description="Pixel width (images).")
    height: Optional[int] = Field(None, description="Pixel height (images).")
    page_count: Optional[int] = Field(None, description="Number of analysed pages (PDFs).")


class Thresholds(BaseModel):
    likely_real_below: float = Field(description="final_score below this is classified likely_real.")
    likely_ai_above: float = Field(description="final_score above this is classified likely_ai_generated.")


class OverallResult(BaseModel):
    """Internal: the overall verdict, before it is placed in the summary."""
    classification: Classification
    label: str
    score: Optional[float]
    confidence: str
    explanation: str


class SignalEvidence(BaseModel):
    signal: str
    label: str
    applicable: bool = Field(description="Whether the analyzer could take this measurement on this file.")
    used_in_score: bool = Field(description="Whether this signal influenced the overall score.")
    score: Optional[float] = Field(description="Signal score 0-1 (higher = more AI-like); null when not applicable.")
    verdict: SignalVerdict
    confidence: str
    weight: Optional[float] = Field(
        description="Share of the visual score carried by this signal (0-1). Null for metadata, "
                    "which adjusts the score instead of averaging in (see `effect`)."
    )
    effect: str = Field(description="weighted | not_used | override | pull_toward_real | neutral")
    explanation: Optional[str] = None
    measurements: dict[str, Any]


class Evidence(BaseModel):
    frequency: SignalEvidence
    noise: SignalEvidence
    gradients: SignalEvidence
    lighting: SignalEvidence
    metadata: Optional[SignalEvidence] = None


class EmbeddedImageSummary(BaseModel):
    image_index: int
    score: float
    confidence: str
    usable: bool
    page_coverage: Optional[float] = None
    width_px: Optional[int] = None
    height_px: Optional[int] = None
    note: Optional[str] = None


class PageResult(BaseModel):
    page: int
    classification: Classification
    score: Optional[float]
    confidence: str
    content_type: Optional[str] = None
    evidence_source: Optional[str] = Field(None, description="page_render | embedded_image | null")
    embedded_image_index: Optional[int] = None
    evidence: Evidence = Field(description="Visual signals of the element that drove this page's score.")
    embedded_images: list[EmbeddedImageSummary]


class DocumentInfo(BaseModel):
    page_count: int
    worst_page: Optional[int] = None
    evidence_source: Optional[str] = None
    embedded_image_index: Optional[int] = None
    engine_classification: Optional[str] = Field(None, description="The PDF engine's own finer-grained classification.")
    engine_description: Optional[str] = None
    flagged_pages: list[Any]
    flagged_embedded_images: list[Any]
    document_anomalies: dict[str, Any] = Field(description="Structural/tamper diagnostics, kept separate from the AI score.")
    pages: list[PageResult]


class ScanSummary(BaseModel):
    """The critical information. Field order is the reading order."""
    http_status: int = Field(description="HTTP status code of this response (200).")
    status: Literal["completed"]
    thresholds: Thresholds
    final_score: Optional[float] = Field(
        description="Likelihood (0-1) that the visual content is AI-generated or AI-manipulated. "
                    "Null when a PDF has no scorable raster content."
    )
    classification: Classification
    label: str = Field(description="Human-readable classification, as shown in the web UI.")
    confidence: str = Field(description="high | medium-high | medium | low-medium | low | insufficient")
    explanation: str = Field(description="One-paragraph plain-English reading of the result.")
    scan_id: str
    timestamp: str = Field(
        description="UTC time of the scan, ISO 8601. Also the reference time for the only "
                    "time-relative check (PDF metadata dated in the future)."
    )
    engine_version: str


class ScanDetails(BaseModel):
    """Everything behind the summary."""
    duration_ms: int
    engine: EngineInfo
    file: FileInfo
    evidence: Evidence = Field(description="The five forensic signals. For a PDF: those of the element that drove the score.")
    scoring: dict[str, Any] = Field(description="How the signals were combined (weights, caps, image context).")
    document: Optional[DocumentInfo] = Field(None, description="PDF only (omitted for images): per-page results and document anomalies.")


class ScanResponse(BaseModel):
    summary: ScanSummary
    details: Optional[ScanDetails] = Field(None, description="Omitted when the request sets details=false.")


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorSummary(BaseModel):
    http_status: int = Field(description="HTTP status code of this response.")
    status: Literal["failed"]
    error: ErrorBody
    scan_id: str
    timestamp: str
    engine_version: str


class ErrorResponse(BaseModel):
    summary: ErrorSummary


# --------------------------------------------------------------------------- #
# Classification helpers
# --------------------------------------------------------------------------- #
def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def classify_score(score: Optional[float]) -> Classification:
    """The UI's three verdict bands (web/lib/verdict.ts verdictFor)."""
    if score is None:
        return "insufficient_evidence"
    if score < REAL_MAX:
        return "likely_real"
    if score > AI_MIN:
        return "likely_ai_generated"
    return "inconclusive"


def _signal_verdict(score: Optional[float]) -> SignalVerdict:
    if score is None:
        return "not_applicable"
    return {"likely_real": "likely_real", "likely_ai_generated": "likely_ai"}.get(
        classify_score(score), "inconclusive"
    )


def image_confidence(signals: dict, scoring_info: dict, meta: dict) -> str:
    """Overall confidence for an image, by the PDF path's rule for an
    embedded image: the tier of the evidence mass (base weight x confidence
    factor over the applicable visual signals); "high" on an explicit
    generator signature; "insufficient" when no substantive signal applied."""
    if meta.get("confidence") == "high":
        return "high"
    if scoring_info.get("insufficient_evidence"):
        return "insufficient"
    mass = 0.0
    for name, base in scoring.WEIGHTS.items():
        result = signals.get(name) or {}
        if scoring.signal_applicable(result):
            mass += base * scoring.CONFIDENCE_FACTOR.get(
                str(result.get("confidence", "low")), scoring.UNKNOWN_CONFIDENCE_FACTOR
            )
    return document._tier(round(mass, 4))


def _metadata_effect(meta: dict) -> str:
    confidence = meta.get("confidence")
    if confidence == "high":
        return "override"
    if confidence == "low-medium":
        return "pull_toward_real"
    return "neutral"


# --------------------------------------------------------------------------- #
# Evidence builders
# --------------------------------------------------------------------------- #
def _measurements(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in _SIGNAL_ENVELOPE_KEYS}


def _visual_evidence(name: str, result: dict, used: bool, weight: float) -> SignalEvidence:
    result = result or {}
    applicable = result.get("applicable") is not False and _finite(result.get("score")) is not None
    score = _finite(result.get("score")) if applicable else None
    return SignalEvidence(
        signal=name,
        label=SIGNAL_LABELS[name],
        applicable=applicable,
        used_in_score=used,
        score=round(score, 4) if score is not None else None,
        verdict=_signal_verdict(score),
        confidence=str(result.get("confidence", "unavailable")),
        weight=round(weight, 4) if used else 0.0,
        effect="weighted" if used else "not_used",
        explanation=result.get("note"),
        measurements=_measurements(result),
    )


def _metadata_evidence(meta: dict, measurements: Optional[dict] = None) -> SignalEvidence:
    effect = _metadata_effect(meta)
    verdict: SignalVerdict = {"override": "likely_ai", "pull_toward_real": "likely_real"}.get(effect, "neutral")
    score = _finite(meta.get("score"))
    return SignalEvidence(
        signal="metadata",
        label=SIGNAL_LABELS["metadata"],
        applicable=True,
        used_in_score=effect != "neutral",
        score=round(score, 4) if score is not None else None,
        verdict=verdict,
        confidence=str(meta.get("confidence", "low")),
        weight=None,
        effect=effect,
        explanation=meta.get("note"),
        measurements=measurements if measurements is not None else _measurements(meta),
    )


def _evidence_from_contributions(signals: dict, contributions: Optional[dict]) -> dict:
    """Visual evidence for a PDF element, using its scoring contributions
    (which already reflect the PDF gates)."""
    contributions = contributions or {}
    out = {}
    for name in VISUAL_SIGNALS:
        c = contributions.get(name) or {}
        used = bool(c.get("used"))
        out[name] = _visual_evidence(name, signals.get(name) or {}, used, float(c.get("combined_weight", 0.0)))
    return out


# --------------------------------------------------------------------------- #
# Result mapping
# --------------------------------------------------------------------------- #
def _summary(classification: str, confidence: str, evidence: dict, notes: list[str]) -> str:
    used = [n for n in VISUAL_SIGNALS if evidence[n].used_in_score]
    unused = [n for n in VISUAL_SIGNALS if not evidence[n].used_in_score]
    if confidence == "insufficient" or classification == "insufficient_evidence":
        parts = [f"{CLASSIFICATION_LABELS[classification]}: insufficient evidence."]
    else:
        parts = [f"{CLASSIFICATION_LABELS[classification]} ({confidence} confidence)."]
    parts.append(f"Signals used: {', '.join(used) if used else 'none'}.")
    if unused:
        parts.append(f"Not applicable or not used: {', '.join(unused)}.")
    parts.extend(notes)
    return " ".join(parts)


def build_image_report(raw: dict) -> dict:
    signals = raw["signals"]
    scoring_info = raw["scoring"]
    meta = signals["metadata"]
    score = float(raw["ai_generated"])
    classification = classify_score(score)
    confidence = image_confidence(signals, scoring_info, meta)

    applicable = set(scoring_info.get("applicable_signals", []))
    weights = scoring_info.get("effective_weights", {})
    evidence = {
        name: _visual_evidence(
            name, signals[name],
            used=name in applicable and not scoring_info.get("insufficient_evidence") and weights.get(name, 0) > 0,
            weight=float(weights.get(name, 0.0)),
        )
        for name in VISUAL_SIGNALS
    }
    evidence["metadata"] = _metadata_evidence(meta)

    notes = []
    if scoring_info.get("single_signal_cap_applied"):
        notes.append(f"Only one substantive signal applied, so the score is capped at {scoring.SINGLE_SIGNAL_CAP:.2f}.")
    if scoring_info.get("insufficient_evidence"):
        notes.append("No substantive signal could be measured; the score is the neutral mid-point.")
    effect = evidence["metadata"].effect
    if effect == "override":
        notes.append("Explicit AI-generator metadata was found in the file.")
    elif effect == "pull_toward_real":
        notes.append("Plausible camera EXIF lowered the score slightly.")

    context = scoring_info.get("context", {})
    return {
        "file_extra": {"width": context.get("width"), "height": context.get("height")},
        "result": OverallResult(
            classification=classification,
            label=CLASSIFICATION_LABELS[classification],
            score=round(score, 4),
            confidence=confidence,
            explanation=_summary(classification, confidence, evidence, notes),
        ),
        "evidence": Evidence(**evidence),
        "scoring": {
            "visual_score": scoring_info.get("visual_score"),
            "metadata_effect": effect,
            "applicable_signals": scoring_info.get("applicable_signals", []),
            "insufficient_evidence": scoring_info.get("insufficient_evidence", False),
            "single_signal_cap_applied": scoring_info.get("single_signal_cap_applied", False),
            "gradients_weight_redistributed_to": scoring_info.get("gradients_weight_redistributed_to", []),
            "context_multipliers": scoring_info.get("context_multipliers", {}),
            "context_discount": scoring_info.get("context_discount", 0.0),
            "context": context,
        },
        "document": None,
    }


def _element_scoring(page: dict) -> dict:
    """The scoring record of the element (page render or embedded image)
    that drove a PDF page's score."""
    if page.get("risk_source") == "embedded_image":
        for img in page.get("embedded_images", []):
            if img.get("image_index") == page.get("risk_image_index"):
                return img.get("scoring") or {}
        return {}
    if page.get("risk_source") == "page_render":
        return page.get("page_render_scoring") or {}
    return {}


def _pdf_metadata(raw_signals: dict) -> tuple[dict, dict]:
    """Document-level metadata evidence: the stronger of the two PDF Info
    dictionary checks, with both sets of findings as measurements."""
    legacy = raw_signals.get("metadata") or {}
    explicit = raw_signals.get("pdf_generator_metadata") or {}
    meta = explicit if explicit.get("confidence") == "high" and legacy.get("confidence") != "high" else legacy
    measurements = {
        "pdf_fields": legacy.get("pdf_fields") or explicit.get("pdf_fields"),
        "signature_hits": legacy.get("signature_hits", []),
        "generator_references": explicit.get("generator_references", []),
    }
    return meta, measurements


def build_pdf_report(raw: dict) -> dict:
    applicable = bool(raw.get("score_applicable"))
    score = round(float(raw["ai_generated"]), 4) if applicable else None
    classification = classify_score(score)
    confidence = str(raw.get("confidence", "insufficient"))

    pages_raw = raw.get("pages", [])
    worst_page = raw.get("worst_page")
    driving = next((p for p in pages_raw if p["page"] == worst_page), None) if worst_page is not None else None
    driving_scoring = _element_scoring(driving) if driving else {}

    evidence = _evidence_from_contributions(raw["signals"], driving_scoring.get("contributions"))
    meta, meta_measurements = _pdf_metadata(raw["signals"])
    evidence["metadata"] = _metadata_evidence(meta, meta_measurements)

    pages = []
    for p in pages_raw:
        page_scoring = _element_scoring(p)
        page_conf = str(p.get("confidence", "insufficient"))
        page_score = round(float(p["ai_generated"]), 4) if page_conf != "insufficient" else None
        pages.append(PageResult(
            page=p["page"],
            classification=classify_score(page_score),
            score=page_score,
            confidence=page_conf,
            content_type=p.get("content_type"),
            evidence_source=p.get("risk_source"),
            embedded_image_index=p.get("risk_image_index"),
            evidence=Evidence(**_evidence_from_contributions(p.get("signals") or {}, page_scoring.get("contributions"))),
            embedded_images=[
                EmbeddedImageSummary(
                    image_index=img["image_index"],
                    score=round(float(img.get("ai_generated", 0.0)), 4),
                    confidence=str(img.get("confidence")),
                    usable=bool(img.get("usable")),
                    page_coverage=img.get("page_coverage"),
                    width_px=img.get("width_px"),
                    height_px=img.get("height_px"),
                    note=(img.get("scoring") or {}).get("note"),
                )
                for img in p.get("embedded_images", [])
            ],
        ))

    notes = []
    if driving is not None:
        source = "an embedded image" if driving.get("risk_source") == "embedded_image" else "the page render"
        notes.append(f"Score driven by {source} on page {worst_page} of {raw.get('page_count', len(pages))}.")
    elif not applicable:
        notes.append(raw.get("description") or "No page contained enough raster content to score.")
    if evidence["metadata"].effect == "override":
        notes.append("Explicit AI-generator reference found in the PDF metadata.")
    anomalies = raw.get("document_anomalies", {})
    if anomalies.get("flag_count"):
        notes.append(f"{anomalies['flag_count']} document anomaly flag(s): {anomalies.get('classification')}.")

    aggregate = raw["signals"].get("document_aggregate", {})
    combiner = driving_scoring.get("combiner", {})
    return {
        "file_extra": {"page_count": raw.get("page_count")},
        "result": OverallResult(
            classification=classification,
            label=CLASSIFICATION_LABELS[classification],
            score=score,
            confidence=confidence,
            explanation=_summary(classification, confidence, evidence, notes),
        ),
        "evidence": Evidence(**evidence),
        "scoring": {
            "visual_score": aggregate.get("score"),
            "metadata_effect": evidence["metadata"].effect,
            "aggregation": raw.get("aggregation"),
            "document_mean": raw.get("document_mean"),
            "usable_pages": aggregate.get("usable_pages"),
            "reliable_pages": aggregate.get("reliable_pages"),
            "single_signal_cap_applied": combiner.get("single_signal_cap_applied", False),
            "gradients_weight_redistributed_to": combiner.get("gradients_weight_redistributed_to", []),
            "context_multipliers": combiner.get("context_multipliers", {}),
            "context_discount": combiner.get("context_discount", 0.0),
            "context": combiner.get("context_buckets"),
        },
        "document": DocumentInfo(
            page_count=raw.get("page_count", len(pages)),
            worst_page=worst_page,
            evidence_source=raw.get("worst_evidence_source"),
            embedded_image_index=raw.get("worst_embedded_image"),
            engine_classification=raw.get("classification"),
            engine_description=raw.get("description"),
            flagged_pages=raw.get("flagged_pages", []),
            flagged_embedded_images=raw.get("flagged_embedded_images", []),
            document_anomalies={
                "score": anomalies.get("score", 0.0),
                "classification": anomalies.get("classification"),
                "flags": anomalies.get("flags", []),
            },
            pages=pages,
        ),
    }


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
async def _read_upload(upload) -> bytes:
    chunks, total = [], 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > validation.MAX_FILE_BYTES:
            raise validation.file_too_large()
        chunks.append(chunk)
    return b"".join(chunks)


def _fetch_url(url: str) -> bytes:
    """Downloads a validated URL. Redirects are followed by hand so every hop
    gets the same URL and host checks as the original (a public URL must not
    be able to redirect the server to a private one)."""
    deadline = time.monotonic() + URL_FETCH_TIMEOUT_S
    headers = {"User-Agent": f"{ENGINE_NAME}/{engine.app.version}"}
    current = url
    try:
        for _ in range(MAX_REDIRECTS + 1):
            validation.validate_url(current)
            validation.check_host_allowed(current)
            with requests.get(
                current, stream=True, timeout=URL_FETCH_TIMEOUT_S, headers=headers, allow_redirects=False,
            ) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("Location", "")
                    if not location:
                        raise ScanError(502, "url_fetch_failed", "redirect without a Location header")
                    current = urljoin(current, location)
                    continue
                resp.raise_for_status()
                return _read_body(resp, deadline)
        raise ScanError(502, "url_fetch_failed", f"more than {MAX_REDIRECTS} redirects")
    except requests.RequestException as exc:
        raise _fetch_error(exc, current) from exc


def _fetch_error(exc: requests.RequestException, url: str) -> ScanError:
    """A stable, readable error for a failed download. The exception's own
    text is not used: urllib3 embeds object addresses (``... object at
    0x7f...``), so the same failing URL would produce a different message on
    every request."""
    host = urlparse(url).hostname or url
    if isinstance(exc, requests.Timeout):
        return ScanError(504, "url_fetch_timeout", f"timed out connecting to or reading from '{host}'")
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        reason = exc.response.reason or ""
        return ScanError(502, "url_fetch_failed", f"'{host}' responded with HTTP {status} {reason}".rstrip())
    if isinstance(exc, requests.exceptions.SSLError):
        return ScanError(502, "url_fetch_failed", f"TLS/SSL error connecting to '{host}'")
    if isinstance(exc, requests.ConnectionError):
        return ScanError(502, "url_fetch_failed", f"could not connect to '{host}'")
    return ScanError(502, "url_fetch_failed", f"could not fetch url ({type(exc).__name__})")


def _read_body(resp: requests.Response, deadline: float) -> bytes:
    declared = resp.headers.get("Content-Length")
    if declared and declared.isdigit() and int(declared) > validation.MAX_FILE_BYTES:
        raise validation.file_too_large()
    chunks, total = [], 0
    for chunk in resp.iter_content(64 * 1024):
        total += len(chunk)
        if total > validation.MAX_FILE_BYTES:
            raise validation.file_too_large()
        if time.monotonic() > deadline:
            raise ScanError(504, "url_fetch_timeout", f"downloading the url took longer than {URL_FETCH_TIMEOUT_S:g}s")
        chunks.append(chunk)
    return b"".join(chunks)


def _engine_info() -> EngineInfo:
    model = weighting.load_model()
    return EngineInfo(
        name=ENGINE_NAME,
        version=engine.app.version,
        weighting=(model or {}).get("weighting", "calibrated") if model else "static",
        calibration_generated=(model or {}).get("generated"),
    )


def _clock() -> datetime:
    """The one place the scan API reads the wall clock (tests pin it)."""
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _error_response(scan_id: str, scanned_at: datetime, err: ScanError) -> JSONResponse:
    body = ErrorResponse(summary=ErrorSummary(
        http_status=err.status_code,
        status="failed",
        error=ErrorBody(code=err.code, message=err.message),
        scan_id=scan_id,
        timestamp=_iso(scanned_at),
        engine_version=engine.app.version,
    ))
    return JSONResponse(status_code=err.status_code, content=body.model_dump())


_ERROR_RESPONSES = {
    code: {"model": ErrorResponse, "description": desc}
    for code, desc in {
        400: "Invalid request: missing, duplicate, unknown or mistyped field; invalid URL or "
             "'details' value; URL on a non-public network; malformed body; empty file",
        413: "Request body or file larger than the configured limit",
        415: "Body is not a form, or the file is not a PNG, JPEG or PDF",
        422: "File is corrupt, an image is over the pixel limit, or a PDF is password-protected",
        500: "Analysis failed",
        502: "The URL could not be fetched",
        504: "Downloading the URL timed out",
    }.items()
}

_REQUEST_BODY_DOC = {
    "required": True,
    "content": {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "format": "binary", "description": "PNG, JPEG or PDF file."},
                    "url": {"type": "string", "maxLength": validation.MAX_URL_LENGTH,
                            "description": "Public http(s) URL of a PNG, JPEG or PDF. Send either file or url."},
                    "details": {"type": "boolean", "default": True,
                                "description": "false: return only the summary section."},
                },
                "additionalProperties": False,
            },
        },
        "application/x-www-form-urlencoded": {
            "schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "maxLength": validation.MAX_URL_LENGTH},
                    "details": {"type": "boolean", "default": True},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
}


async def _parse_request(request: Request, forms: list[FormData]) -> validation.ScanInput:
    """Request-level and form-level validation (see app/scan_validation.py).
    The parsed form is appended to ``forms`` so the caller can close its
    uploaded files, even when validation then fails."""
    content_type = request.headers.get("content-type")
    limit = validation.request_body_limit(content_type)
    length = validation.validate_content_length(request.headers.get("content-length"), limit)
    validation.validate_content_type(content_type, length)
    limited = Request(request.scope, receive=validation.limit_receive(request.receive, limit))
    try:
        form = await limited.form(
            max_files=validation.MAX_FORM_FILES,
            max_fields=validation.MAX_FORM_FIELDS,
            max_part_size=validation.MAX_FORM_PART_BYTES,
        )
    except validation.BodyTooLarge:
        raise validation.file_too_large() from None
    except StarletteHTTPException as exc:
        # starlette's own multipart limits/errors (e.g. too many fields).
        raise ScanError(400, "malformed_body", f"could not parse the form body: {exc.detail}") from exc
    except ValueError as exc:
        # python-multipart's parse errors (FormParserError is a ValueError),
        # which starlette lets through: a garbled body, a wrong boundary.
        raise ScanError(400, "malformed_body", "could not parse the form body (malformed multipart or form data)") from exc
    forms.append(form)
    return validation.validate_form(form.multi_items())


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #
@router.post(
    "/scans",
    response_model=ScanResponse,
    responses=_ERROR_RESPONSES,
    summary="Scan an image or PDF",
    openapi_extra={"requestBody": _REQUEST_BODY_DOC},
)
async def create_scan(request: Request):
    """Runs the full forensic scan synchronously and returns the result.

    Send exactly one of `file` (multipart upload) or `url`, and optionally
    `details=false`. No other fields are accepted. The response has two
    sections: `summary` (HTTP status, thresholds, final score,
    classification, confidence) and `details` (file metadata, every signal's
    measurements, scoring, PDF pages). Nothing is stored server-side; keep
    the response if you need it.
    """
    # Deterministic by design: for the same input (request fields and file
    # bytes) and engine version, the response is identical apart from
    # scan_id, timestamp, duration_ms and the engine version itself. The
    # clock is read once, here: it is the response timestamp and the
    # reference time for the only time-relative check (future-dated PDF
    # metadata), so no other part of the output depends on when it ran.
    scan_id = f"scn_{uuid.uuid4().hex}"
    scanned_at = _clock()
    started = time.monotonic()
    forms: list[FormData] = []
    try:
        return await _run_scan(request, scan_id, scanned_at, started, forms)
    except ScanError as err:
        return _error_response(scan_id, scanned_at, err)
    finally:
        for form in forms:
            await form.close()


async def _run_scan(
    request: Request, scan_id: str, scanned_at: datetime, started: float, forms: list[FormData],
) -> JSONResponse:
    scan_input = await _parse_request(request, forms)

    if scan_input.file is not None:
        data = await _read_upload(scan_input.file)
        file_info = {
            "source": "upload",
            "filename": validation.normalize_filename(scan_input.file.filename),
            "url": None,
        }
    else:
        url = validation.validate_url(scan_input.url)
        data = await run_in_threadpool(_fetch_url, url)
        name = validation.normalize_filename(unquote(urlparse(url).path))
        file_info = {"source": "url", "filename": name, "url": url}

    facts = await run_in_threadpool(validation.validate_file_content, data)
    fmt = facts.format

    try:
        if fmt == "pdf":
            raw = await run_in_threadpool(engine._run_pdf_analysis, data, now=scanned_at)
            report = build_pdf_report(raw)
        else:
            raw = await run_in_threadpool(engine._run_image_analysis, data)
            report = build_image_report(raw)
    except ValueError as exc:
        raise ScanError(422, "unprocessable_file", str(exc)) from exc
    except Exception as exc:
        logger.exception("scan %s failed", scan_id)
        raise ScanError(500, "analysis_failed", "analysis failed; the server log has details under this scan_id") from exc

    include_details = scan_input.details
    overall: OverallResult = report["result"]
    summary = ScanSummary(
        http_status=200,
        status="completed",
        thresholds=Thresholds(likely_real_below=REAL_MAX, likely_ai_above=AI_MIN),
        final_score=overall.score,
        classification=overall.classification,
        label=overall.label,
        confidence=overall.confidence,
        explanation=overall.explanation,
        scan_id=scan_id,
        timestamp=_iso(scanned_at),
        engine_version=engine.app.version,
    )
    content: dict[str, Any] = {"summary": summary.model_dump()}
    if include_details:
        scan_details = ScanDetails(
            duration_ms=int((time.monotonic() - started) * 1000),
            engine=_engine_info(),
            file=FileInfo(
                **file_info,
                size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                format=fmt,
                mime_type=MIME_TYPES[fmt],
                **report["file_extra"],
            ),
            evidence=report["evidence"],
            scoring=report["scoring"],
            document=report["document"],
        )
        # `document` is PDF-only: leave the key out for images rather than
        # sending a null section.
        content["details"] = scan_details.model_dump(exclude={"document"} if scan_details.document is None else None)
    # Returned directly so sections that were not requested are absent, not null.
    return JSONResponse(status_code=200, content=content)
