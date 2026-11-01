"""PDF/document-aware forensic scoring.

A rendered PDF page is not equivalent to a camera photograph. Native PDF text,
vector paths and fonts are rasterized by PDFium, which creates clean edges,
flat regions and frequency patterns that can look "AI-like" to photo-oriented
forensic analyzers.

This module decides when photo-forensic signals are applicable to a PDF page,
ignores unavailable/low-value signals, and aggregates page evidence without
letting a single weak page dominate the entire document.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

# Separate weights for image-dominant PDF pages. These do not change the
# normal PNG/JPEG scoring path.
PDF_SCAN_WEIGHTS = {
    "frequency": 0.05,
    "noise": 0.45,
    "gradients": 0.35,
    "lighting": 0.15,
}

CONFIDENCE_FACTOR = {
    "high": 1.0,
    "medium": 1.0,
    "low-medium": 0.60,
    "low": 0.20,
}

_UNAVAILABLE_PHRASES = (
    "not enough",
    "no clear",
    "too small",
    "could not",
    "unable to",
    "unavailable",
)

# Strong, explicit provenance terms only. Generic PDF producers such as
# OpenText, Acrobat, Preview, Chrome, wkhtmltopdf, etc. are intentionally not
# treated as AI evidence.
AI_GENERATOR_TERMS = (
    "chatgpt",
    "openai",
    "dall-e",
    "dalle",
    "midjourney",
    "stable diffusion",
    "comfyui",
    "automatic1111",
    "adobe firefly",
    "firefly",
    "leonardo.ai",
    "leonardo ai",
    "ideogram",
    "runwayml",
    "runway ai",
    "recraft",
    "flux",
    "nightcafe",
)


def analyze_pdf_metadata(fields: dict[str, str]) -> dict:
    """Looks only for explicit AI/generator references in PDF metadata.

    Absence of a signature is neutral/low evidence rather than proof the PDF is
    human-authored.
    """
    normalized = "\n".join(f"{k}: {v}" for k, v in fields.items()).lower()
    hits = [term for term in AI_GENERATOR_TERMS if term in normalized]
    if hits:
        return {
            "score": 0.98,
            "confidence": "high",
            "generator_references": sorted(set(hits)),
            "pdf_fields": fields,
            "note": "explicit AI/generator reference found in PDF document metadata",
        }
    return {
        "score": 0.0,
        "confidence": "low",
        "generator_references": [],
        "pdf_fields": fields,
        "note": "no explicit AI/generator reference found in PDF metadata; absence is not proof of human authorship",
    }


def _signal_available(name: str, result: dict, page_profile: dict) -> bool:
    score = result.get("score")
    if score is None:
        return False

    note = str(result.get("note", "")).lower()
    if any(phrase in note for phrase in _UNAVAILABLE_PHRASES):
        return False

    # Native/vector PDF rendering destroys the assumptions behind camera-photo
    # forensic signals. Apply them only when the page is substantially raster.
    content_type = page_profile.get("content_type")
    image_fraction = float(page_profile.get("image_area_fraction", 0.0))

    if content_type in {"native_document", "vector_document", "sparse_document"}:
        return False

    if content_type == "mixed_document":
        # Whole-page visual metrics are contaminated by rasterized text/vector
        # content unless images dominate most of the page.
        if image_fraction < 0.70:
            return False

    # Frequency/lighting are particularly sensitive to page rendering and text
    # edges, so require stronger raster dominance than noise/gradients.
    if name in {"frequency", "lighting"} and image_fraction < 0.80:
        return False

    return True


def score_pdf_page(signals: dict, page_profile: dict) -> dict:
    """Returns a PDF-aware score for one rasterized page.

    Unavailable signals contribute no weight. Low-confidence signals are
    strongly downweighted. The normalized score is additionally shrunk when
    only a small fraction of the expected evidence is usable, preventing a
    single weak signal from creating a high page score.
    """
    contributions = {}
    numerator = 0.0
    effective_weight_sum = 0.0
    total_base_weight = sum(PDF_SCAN_WEIGHTS.values())

    for name, base_weight in PDF_SCAN_WEIGHTS.items():
        result = signals.get(name) or {}
        available = _signal_available(name, result, page_profile)
        confidence = str(result.get("confidence", "low"))
        confidence_factor = CONFIDENCE_FACTOR.get(confidence, 0.0)

        if not available:
            contributions[name] = {
                "used": False,
                "reason": "not applicable or unavailable for this PDF page",
                "effective_weight": 0.0,
            }
            continue

        score = float(np.clip(result.get("score", 0.0), 0.0, 1.0))
        effective_weight = base_weight * confidence_factor
        numerator += score * effective_weight
        effective_weight_sum += effective_weight
        contributions[name] = {
            "used": True,
            "score": round(score, 4),
            "confidence": confidence,
            "effective_weight": round(effective_weight, 4),
        }

    if effective_weight_sum <= 1e-9:
        return {
            "score": 0.0,
            "confidence": "insufficient",
            "usable": False,
            "evidence_coverage": 0.0,
            "content_type": page_profile.get("content_type", "unknown"),
            "contributions": contributions,
            "note": "photo-forensic analysis is not applicable/reliable on this rendered PDF page",
        }

    normalized = numerator / effective_weight_sum

    # Evidence coverage measures how much of the intended photo-forensics stack
    # was actually available. Require roughly noise + one other useful signal
    # before allowing the page score to express full confidence.
    evidence_coverage = float(np.clip(effective_weight_sum / total_base_weight, 0.0, 1.0))
    reliability = float(np.clip(effective_weight_sum / 0.55, 0.0, 1.0))
    score = float(np.clip(normalized * reliability, 0.0, 1.0))

    if effective_weight_sum >= 0.65:
        confidence = "medium-high"
    elif effective_weight_sum >= 0.40:
        confidence = "medium"
    elif effective_weight_sum >= 0.20:
        confidence = "low-medium"
    else:
        confidence = "low"

    return {
        "score": round(score, 4),
        "raw_normalized_score": round(float(normalized), 4),
        "confidence": confidence,
        "usable": effective_weight_sum >= 0.15,
        "effective_weight_sum": round(effective_weight_sum, 4),
        "evidence_coverage": round(evidence_coverage, 4),
        "content_type": page_profile.get("content_type", "unknown"),
        "contributions": contributions,
        "note": "PDF-aware score: unavailable signals are ignored and native text/vector rendering is not treated as camera evidence",
    }




def unavailable_embedded_signals(reason: str) -> dict:
    """Return explicit unavailable signal objects after extraction failure."""
    unavailable = {"score": None, "confidence": "unavailable", "note": reason}
    return {
        "frequency": dict(unavailable),
        "noise": dict(unavailable),
        "gradients": dict(unavailable),
        "lighting": dict(unavailable),
        "metadata": dict(unavailable),
    }


def unusable_embedded_image_score(image_profile: dict, reason: str) -> dict:
    return {
        "score": 0.0,
        "confidence": "insufficient",
        "usable": False,
        "evidence_coverage": 0.0,
        "image_profile": image_profile,
        "contributions": {},
        "note": reason,
    }


def _embedded_signal_available(result: dict) -> bool:
    score = result.get("score")
    if score is None:
        return False
    note = str(result.get("note", "")).lower()
    return not any(phrase in note for phrase in _UNAVAILABLE_PHRASES)


def score_embedded_image(signals: dict, image_profile: dict) -> dict:
    """Score one raster image extracted from a PDF page.

    Page coverage is deliberately *not* used as a negative factor: a fraudulent
    signature/photo/receipt can occupy 1% of the page and still matter. Pixel
    dimensions do matter because these forensic heuristics become unstable on
    icons and very small raster assets.
    """
    width_px = int(image_profile.get("width_px", 0))
    height_px = int(image_profile.get("height_px", 0))
    pixel_count = int(image_profile.get("pixel_count", width_px * height_px))

    # Avoid claiming AI evidence on logos/icons where there are simply not
    # enough pixels for noise/texture forensics. Small page coverage alone does
    # not disqualify an image.
    if width_px < 64 or height_px < 64 or pixel_count < 16_384:
        return unusable_embedded_image_score(
            image_profile,
            reason="embedded raster is too small for reliable AI-image forensics",
        )

    contributions = {}
    numerator = 0.0
    effective_weight_sum = 0.0
    total_base_weight = sum(PDF_SCAN_WEIGHTS.values())

    for name, base_weight in PDF_SCAN_WEIGHTS.items():
        result = signals.get(name) or {}
        available = _embedded_signal_available(result)
        confidence = str(result.get("confidence", "low"))
        confidence_factor = CONFIDENCE_FACTOR.get(confidence, 0.0)

        if not available or confidence_factor <= 0:
            contributions[name] = {
                "used": False,
                "reason": "unavailable or insufficient-confidence signal",
                "effective_weight": 0.0,
            }
            continue

        score = float(np.clip(result.get("score", 0.0), 0.0, 1.0))
        effective_weight = base_weight * confidence_factor
        numerator += score * effective_weight
        effective_weight_sum += effective_weight
        contributions[name] = {
            "used": True,
            "score": round(score, 4),
            "confidence": confidence,
            "effective_weight": round(effective_weight, 4),
        }

    if effective_weight_sum <= 1e-9:
        return unusable_embedded_image_score(
            image_profile,
            reason="embedded raster had no usable forensic signals",
        )

    normalized = numerator / effective_weight_sum
    evidence_coverage = float(np.clip(effective_weight_sum / total_base_weight, 0.0, 1.0))

    # Require at least roughly noise plus one corroborating signal before the
    # extracted image can express its full score.
    reliability = float(np.clip(effective_weight_sum / 0.55, 0.0, 1.0))
    score = float(np.clip(normalized * reliability, 0.0, 1.0))

    if effective_weight_sum >= 0.65:
        confidence = "medium-high"
    elif effective_weight_sum >= 0.40:
        confidence = "medium"
    elif effective_weight_sum >= 0.20:
        confidence = "low-medium"
    else:
        confidence = "low"

    usable = effective_weight_sum >= 0.15
    return {
        "score": round(score, 4),
        "raw_normalized_score": round(float(normalized), 4),
        "confidence": confidence,
        "usable": usable,
        "effective_weight_sum": round(effective_weight_sum, 4),
        "evidence_coverage": round(evidence_coverage, 4),
        "image_profile": image_profile,
        "contributions": contributions,
        "note": "embedded-image score; page coverage does not dilute a reliable suspicious raster element",
    }


def combine_page_evidence(
    page_number: int,
    page_render_scoring: dict,
    embedded_images: list[dict],
) -> dict:
    """Worst reliable evidence unit on a page wins.

    Evidence units are the whole-page raster (when applicable) and each
    independently extracted embedded image. Low-confidence units remain in the
    response but cannot drive page/document fraud risk.
    """
    candidates = []

    if (
        page_render_scoring.get("usable")
        and page_render_scoring.get("confidence")
        in {"low-medium", "medium", "medium-high", "high"}
    ):
        candidates.append({
            "score": float(page_render_scoring.get("score", 0.0)),
            "confidence": page_render_scoring.get("confidence", "low-medium"),
            "source": "page_render",
            "image_index": None,
            "evidence_coverage": page_render_scoring.get("evidence_coverage", 0.0),
        })

    for img in embedded_images:
        if (
            img.get("usable")
            and img.get("confidence")
            in {"low-medium", "medium", "medium-high", "high"}
        ):
            candidates.append({
                "score": float(img.get("ai_generated", 0.0)),
                "confidence": img.get("confidence", "low-medium"),
                "source": "embedded_image",
                "image_index": img.get("image_index"),
                "evidence_coverage": img.get("scoring", {}).get("evidence_coverage", 0.0),
            })

    if not candidates:
        return {
            "score": 0.0,
            "confidence": "insufficient",
            "usable": False,
            "source": None,
            "image_index": None,
            "page": page_number,
            "evidence_coverage": 0.0,
            "note": "no reliable page-render or embedded-image evidence on this PDF page",
        }

    worst = max(candidates, key=lambda x: x["score"])
    return {
        "score": round(float(worst["score"]), 4),
        "confidence": str(worst["confidence"]),
        "usable": True,
        "source": worst["source"],
        "image_index": worst["image_index"],
        "page": page_number,
        "evidence_coverage": round(float(worst.get("evidence_coverage", 0.0)), 4),
        "candidate_count": len(candidates),
        "note": "fraud-sensitive page risk: highest-scoring reliable page render or embedded raster wins",
    }

def aggregate_pdf_pages(page_results: Iterable[dict]) -> dict:
    """Fraud-sensitive PDF aggregation: worst reliable page wins.

    A single manipulated page can be hidden inside an otherwise legitimate
    multi-page document, so clean pages must never dilute the strongest
    reliable page-level evidence. Pages with insufficient/very-low confidence
    are excluded from the max calculation to avoid one noisy render creating a
    document-level false positive.

    The reliability-weighted mean is retained only as diagnostic context.
    """
    pages = list(page_results)
    usable = [p for p in pages if p.get("pdf_scoring", {}).get("usable")]
    if not usable:
        return {
            "score": 0.0,
            "confidence": "insufficient",
            "usable_pages": 0,
            "reliable_pages": 0,
            "worst_page": None,
            "max_page_score": 0.0,
            "mean_page_score": 0.0,
            "flagged_pages": [],
            "note": "no PDF pages contained enough applicable raster evidence for AI-image scoring",
        }

    # Low-confidence pages can still be useful diagnostically, but they should
    # not be allowed to trigger the whole document. low-medium or better means
    # the page has at least a meaningful subset of the intended evidence stack.
    reliable = [
        p for p in usable
        if p.get("pdf_scoring", {}).get("confidence")
        in {"low-medium", "medium", "medium-high", "high"}
    ]

    # If every raster page is only low confidence, keep the result
    # inconclusive rather than promoting a weak maximum to document risk.
    if not reliable:
        scores = np.array([float(p["pdf_scoring"]["score"]) for p in usable], dtype=np.float64)
        weights = np.array([
            max(float(p["pdf_scoring"].get("evidence_coverage", 0.0)), 0.05)
            for p in usable
        ], dtype=np.float64)
        weights /= weights.sum()
        weighted_mean = float(np.sum(scores * weights))
        return {
            "score": 0.0,
            "confidence": "insufficient",
            "usable_pages": len(usable),
            "reliable_pages": 0,
            "worst_page": None,
            "max_page_score": round(float(np.max(scores)), 4),
            "mean_page_score": round(weighted_mean, 4),
            "flagged_pages": [],
            "note": "raster evidence exists, but no page is reliable enough to drive a fraud-sensitive document score",
        }

    scores = np.array([float(p["pdf_scoring"]["score"]) for p in usable], dtype=np.float64)
    weights = np.array([
        max(float(p["pdf_scoring"].get("evidence_coverage", 0.0)), 0.05)
        for p in usable
    ], dtype=np.float64)
    weights /= weights.sum()
    weighted_mean = float(np.sum(scores * weights))

    worst = max(reliable, key=lambda p: float(p["pdf_scoring"]["score"]))
    worst_score = float(worst["pdf_scoring"]["score"])
    worst_confidence = str(worst["pdf_scoring"].get("confidence", "low-medium"))

    # Flag every reliable page with at least moderate/suspicious evidence. This
    # makes the max decision auditable and surfaces multiple manipulated pages.
    flagged = [
        {
            "page": int(p["page"]),
            "score": round(float(p["pdf_scoring"]["score"]), 4),
            "confidence": str(p["pdf_scoring"].get("confidence", "low-medium")),
            "content_type": p.get("content_type", "unknown"),
            "risk_source": p.get("pdf_scoring", {}).get("source"),
            "image_index": p.get("pdf_scoring", {}).get("image_index"),
        }
        for p in reliable
        if float(p["pdf_scoring"]["score"]) >= 0.40
    ]
    flagged.sort(key=lambda x: x["score"], reverse=True)

    return {
        "score": round(worst_score, 4),
        "confidence": worst_confidence,
        "usable_pages": len(usable),
        "reliable_pages": len(reliable),
        "worst_page": int(worst["page"]),
        "max_page_score": round(worst_score, 4),
        "mean_page_score": round(weighted_mean, 4),
        "flagged_pages": flagged,
        "aggregation": "worst_reliable_evidence",
        "worst_evidence_source": worst.get("pdf_scoring", {}).get("source"),
        "worst_image_index": worst.get("pdf_scoring", {}).get("image_index"),
        "note": "fraud-sensitive aggregation: the highest-scoring reliable page render or embedded image determines document risk; clean pages never dilute it",
    }



def not_applicable_signals(page_profile: dict) -> dict:
    """Placeholder signal objects for native/vector PDF pages.

    Returning ``None`` instead of 0.5 is intentional: 0.5 is a numerical AI
    score while these signals are simply not applicable after PDF rendering.
    """
    reason = (
        f"not applicable to PDF page type '{page_profile.get('content_type', 'unknown')}'; "
        "native PDF rasterization does not preserve camera/sensor forensic assumptions"
    )
    unavailable = {"score": None, "confidence": "unavailable", "note": reason}
    return {
        "frequency": dict(unavailable),
        "noise": dict(unavailable),
        "gradients": dict(unavailable),
        "lighting": dict(unavailable),
        "metadata": {
            "score": None,
            "confidence": "unavailable",
            "note": "per-page raster metadata is not meaningful for native PDF content",
        },
    }

def classify_pdf(score: float, confidence: str, metadata_hit: bool = False) -> tuple[str, str]:
    if metadata_hit:
        return "likely_ai_generated", "Explicit AI/generator provenance was found in the PDF metadata."
    if confidence == "insufficient":
        return (
            "insufficient_visual_evidence",
            "Not enough reliable raster evidence is available to score AI-generated image content.",
        )
    if score < 0.30:
        return "low_ai_evidence", "Low AI-generated image evidence detected."
    if score < 0.50:
        return "possible_ai_evidence", "Possible AI-generated image evidence detected."
    if score < 0.70:
        return "suspicious_raster_content", "Moderate AI-generated image evidence detected."
    return "likely_ai_generated", "Strong AI-generated image evidence detected."
