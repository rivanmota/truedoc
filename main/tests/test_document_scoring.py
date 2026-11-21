"""Standalone tests for the PDF / embedded-image scoring logic in
``app/analyzers/document.py``.

These tests exercise the pure scoring functions directly with hand-built
signal dictionaries and page profiles. No PDF rendering, no PDFium, no image
analyzers, no FastAPI -- only ``numpy`` (a dependency of the module under
test) and ``pytest`` are required.

Run from the ``main/`` directory (or anywhere -- the path shim below makes the
``app`` package importable):

    pytest tests/test_document_scoring.py -v

Coverage map
------------
* ``analyze_pdf_metadata``        -- explicit AI-generator detection
* ``_signal_available``           -- page-type / raster-fraction gating
* ``score_pdf_page``              -- weights, confidence factors, reliability
                                     shrink, confidence tiers, usability
* ``score_embedded_image``        -- size gating, coverage-independence, same
                                     weighting rules as pages
* ``unavailable_embedded_signals`` / ``unusable_embedded_image_score`` /
  ``not_applicable_signals``      -- placeholder shapes
* ``combine_page_evidence``       -- worst reliable evidence unit on a page
* ``aggregate_pdf_pages``         -- worst reliable page wins, flagged-page
                                     threshold (exactly 0.40), diagnostics
* ``classify_pdf``                -- classification band boundaries
* End-to-end composition of the above for realistic multi-page documents
"""
from __future__ import annotations

import os
import sys
from typing import Iterable

import pytest

# --- path shim -------------------------------------------------------------
# Make ``app`` importable regardless of the working directory pytest is
# launched from, so this file is self-contained.
_MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

from app import scoring  # noqa: E402
from app.analyzers import document  # noqa: E402

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

SIGNAL_NAMES = ("frequency", "noise", "gradients", "lighting")
RELIABLE_CONFIDENCES = {"low-medium", "medium", "medium-high", "high"}

IMAGE_PAGE = {"content_type": "scanned_or_image_document", "image_area_fraction": 1.0}
NATIVE_PAGE = {"content_type": "native_document", "image_area_fraction": 0.0}
VECTOR_PAGE = {"content_type": "vector_document", "image_area_fraction": 0.10}
SPARSE_PAGE = {"content_type": "sparse_document", "image_area_fraction": 0.05}

# A comfortably-sized embedded raster: 400x300 = 120,000 px (>= 16,384).
EMBEDDED_PROFILE = {
    "image_index": 1,
    "width_px": 400,
    "height_px": 300,
    "pixel_count": 120_000,
    "page_coverage": 0.25,
    "filters": ["DCTDecode"],
}

# PDF elements are combined by the image path (app.scoring): same weights,
# confidence factors, no-portrait redistribution and single-signal cap.
W = {"frequency": 0.25, "noise": 0.40, "gradients": 0.25, "lighting": 0.10}
LOW_MEDIUM, LOW = 0.7, 0.5
CAP = 0.60
# Gradients not applicable: its 0.25 moves to noise and frequency, 0.40 : 0.25.
NP_F = W["frequency"] + W["gradients"] * W["frequency"] / (W["frequency"] + W["noise"])
NP_N = W["noise"] + W["gradients"] * W["noise"] / (W["frequency"] + W["noise"])


def sig(score, confidence: str = "medium", note: str = "signal computed normally") -> dict:
    """Build a single analyzer result dict."""
    return {"score": score, "confidence": confidence, "note": note}


def full_signals(score: float, confidence: str = "medium") -> dict:
    """All four visual signals at the same score/confidence."""
    return {name: sig(score, confidence) for name in SIGNAL_NAMES}


def page_entry(
    page: int,
    score: float,
    confidence: str = "medium",
    *,
    usable: bool = True,
    evidence_coverage: float = 1.0,
    content_type: str = "scanned_or_image_document",
    source: str | None = "page_render",
    image_index: int | None = None,
) -> dict:
    """Build a page dict in the shape ``main._run_pdf_analysis`` feeds to
    ``aggregate_pdf_pages`` -- i.e. with a ``pdf_scoring`` sub-dict that is
    the output of ``combine_page_evidence``."""
    return {
        "page": page,
        "ai_generated": score,
        "confidence": confidence,
        "content_type": content_type,
        "pdf_scoring": {
            "score": score,
            "confidence": confidence,
            "usable": usable,
            "source": source,
            "image_index": image_index,
            "page": page,
            "evidence_coverage": evidence_coverage,
        },
    }


def embedded_entry(index: int, scoring: dict) -> dict:
    """Wrap a ``score_embedded_image`` result the way ``main.py`` does before
    passing it to ``combine_page_evidence``."""
    return {
        "image_index": index,
        "ai_generated": scoring["score"],
        "confidence": scoring["confidence"],
        "usable": scoring["usable"],
        "scoring": scoring,
    }


def build_page(
    page_number: int,
    signals: dict,
    profile: dict,
    embedded: Iterable[tuple[dict, dict]] = (),
) -> dict:
    """Run the real per-page pipeline: score_pdf_page -> score_embedded_image
    (for each (signals, profile) pair) -> combine_page_evidence, and return a
    page dict suitable for ``aggregate_pdf_pages``."""
    render = document.score_pdf_page(signals, profile)
    embedded_results = []
    for idx, (emb_signals, emb_profile) in enumerate(embedded, start=1):
        emb_profile = {**emb_profile, "image_index": idx}
        scoring = document.score_embedded_image(emb_signals, emb_profile)
        embedded_results.append(embedded_entry(idx, scoring))
    risk = document.combine_page_evidence(page_number, render, embedded_results)
    return {
        "page": page_number,
        "ai_generated": risk["score"],
        "confidence": risk["confidence"],
        "content_type": profile.get("content_type", "unknown"),
        "page_render_scoring": render,
        "pdf_scoring": risk,
        "embedded_images": embedded_results,
    }


def used_signals(result: dict) -> set[str]:
    return {name for name, c in result["contributions"].items() if c.get("used")}


# ---------------------------------------------------------------------------
# Module-level constants -- guard against silent retuning
# ---------------------------------------------------------------------------


class TestConstants:
    def test_pdf_weights_are_the_image_path_weights(self):
        # One set of weights for every path: a picture inside a PDF must not
        # be weighted differently from the same picture uploaded directly.
        assert document.PDF_SCAN_WEIGHTS is scoring.WEIGHTS
        assert document.PDF_SCAN_WEIGHTS == W

    def test_pdf_weights_sum_to_one(self):
        assert sum(document.PDF_SCAN_WEIGHTS.values()) == pytest.approx(1.0)

    def test_pdf_weights_cover_exactly_the_four_visual_signals(self):
        assert set(document.PDF_SCAN_WEIGHTS) == set(SIGNAL_NAMES)

    def test_noise_is_the_dominant_signal_and_lighting_the_weakest(self):
        w = document.PDF_SCAN_WEIGHTS
        assert w["noise"] > w["gradients"] == w["frequency"] > w["lighting"]

    def test_confidence_factors_are_the_image_path_factors(self):
        f = document.CONFIDENCE_FACTOR
        assert f is scoring.CONFIDENCE_FACTOR
        assert f["high"] == f["medium"] == 1.0
        assert f["low-medium"] == LOW_MEDIUM and f["low"] == LOW
        assert 0 < f["low"] < f["low-medium"] < f["medium"]

    def test_tier_thresholds(self):
        assert (document.TIER_MEDIUM_HIGH, document.TIER_MEDIUM, document.TIER_LOW_MEDIUM,
                document.USABLE_MIN_WEIGHT) == (0.65, 0.40, 0.20, 0.15)

    def test_no_single_signal_reaches_medium_high_on_its_own(self):
        # Only corroborated evidence can carry the strongest tier.
        assert max(document.PDF_SCAN_WEIGHTS.values()) < document.TIER_MEDIUM_HIGH


# ---------------------------------------------------------------------------
# analyze_pdf_metadata
# ---------------------------------------------------------------------------


class TestAnalyzePdfMetadata:
    def test_no_fields_is_neutral_low_confidence(self):
        result = document.analyze_pdf_metadata({})
        assert result["score"] == 0.0
        assert result["confidence"] == "low"
        assert result["generator_references"] == []
        assert result["pdf_fields"] == {}
        assert "absence is not proof" in result["note"]

    @pytest.mark.parametrize(
        "producer",
        [
            "Adobe Acrobat Pro DC 23.1",
            "macOS Version 14.2 (Build 23C64) Quartz PDFContext",
            "Skia/PDF m120",  # Chrome
            "wkhtmltopdf 0.12.6",
            "OpenText Exstream",
            "Microsoft: Print To PDF",
            "LibreOffice 7.6",
        ],
    )
    def test_generic_producers_are_not_ai_evidence(self, producer):
        result = document.analyze_pdf_metadata({"Producer": producer, "Creator": "Word"})
        assert result["score"] == 0.0
        assert result["confidence"] == "low"
        assert result["generator_references"] == []

    @pytest.mark.parametrize("term", document.AI_GENERATOR_TERMS)
    def test_every_listed_generator_term_is_detected(self, term):
        result = document.analyze_pdf_metadata({"Creator": f"made with {term}"})
        assert result["score"] == 0.98
        assert result["confidence"] == "high"
        assert term in result["generator_references"]

    def test_detection_is_case_insensitive(self):
        result = document.analyze_pdf_metadata({"Author": "MidJourney Bot"})
        assert result["confidence"] == "high"
        assert result["generator_references"] == ["midjourney"]

    def test_detection_considers_field_names_and_values(self):
        # The key/value pair is flattened to "key: value" before matching, so
        # a hit in any field is enough.
        result = document.analyze_pdf_metadata({"Keywords": "draft", "Subject": "ComfyUI export"})
        assert result["generator_references"] == ["comfyui"]

    def test_multiple_hits_are_sorted_and_deduplicated(self):
        result = document.analyze_pdf_metadata({
            "Creator": "Stable Diffusion via ComfyUI",
            "Producer": "stable diffusion",  # duplicate term, different case
        })
        assert result["generator_references"] == ["comfyui", "stable diffusion"]

    def test_overlapping_terms_both_reported(self):
        # "dall-e" does not contain "dalle" and vice versa, but "openai" and
        # "chatgpt" can co-occur. Verify substrings report every match.
        result = document.analyze_pdf_metadata({"Producer": "ChatGPT (OpenAI)"})
        assert result["generator_references"] == ["chatgpt", "openai"]

    def test_original_fields_are_echoed_back(self):
        fields = {"Title": "Invoice", "Producer": "Adobe Firefly"}
        result = document.analyze_pdf_metadata(fields)
        assert result["pdf_fields"] is fields
        assert result["generator_references"] == ["adobe firefly"]

    def test_bare_common_words_are_not_generator_terms(self):
        # "flux", "dalle" and "firefly" are ordinary English/French/Italian
        # words; only the product forms ("FLUX.1", "DALL-E", "Adobe Firefly")
        # are provenance. (This test used to expect "Firefly" -> a hit, which
        # contradicted the documented term list.)
        for word in ("flux", "dalle", "firefly"):
            assert word not in document.AI_GENERATOR_TERMS
        result = document.analyze_pdf_metadata({"Title": "Invoice", "Producer": "Firefly Print Server"})
        assert result["confidence"] == "low"
        assert result["generator_references"] == []

    def test_generator_list_is_shared_with_image_metadata_analyzer(self):
        from app.analyzers import metadata
        assert metadata.KNOWN_GENERATOR_SOFTWARE is document.AI_GENERATOR_TERMS
        for fields in ({"Title": "reflux study"}, {"Producer": "InfluxData"}, {"Author": "Dallesandro"}):
            assert metadata.analyze_pdf_document(fields)["confidence"] == "low"
            assert document.analyze_pdf_metadata(fields)["confidence"] == "low"
        for fields in ({"Creator": "Midjourney"}, {"Creator": "InvokeAI"}, {"Creator": "ChatGPT"}):
            assert metadata.analyze_pdf_document(fields)["confidence"] == "high"
            assert document.analyze_pdf_metadata(fields)["confidence"] == "high"


# ---------------------------------------------------------------------------
# _signal_available (exercised through score_pdf_page contributions)
# ---------------------------------------------------------------------------


class TestSignalAvailabilityGating:
    def test_all_signals_used_on_fully_raster_page(self):
        result = document.score_pdf_page(full_signals(0.5), IMAGE_PAGE)
        assert used_signals(result) == set(SIGNAL_NAMES)

    def test_none_score_is_unavailable(self):
        signals = full_signals(0.9)
        signals["noise"] = {"score": None, "confidence": "unavailable", "note": "x"}
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        assert "noise" not in used_signals(result)
        assert result["contributions"]["noise"]["used"] is False
        assert result["contributions"]["noise"]["effective_weight"] == 0.0

    def test_missing_signal_key_is_unavailable(self):
        signals = full_signals(0.9)
        del signals["gradients"]
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        assert "gradients" not in used_signals(result)

    def test_signal_value_of_none_is_unavailable(self):
        # ``signals.get(name) or {}`` must tolerate an explicit None entry.
        signals = full_signals(0.9)
        signals["lighting"] = None
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        assert "lighting" not in used_signals(result)

    @pytest.mark.parametrize("phrase", document._UNAVAILABLE_PHRASES)
    def test_unavailable_note_phrases_disable_a_signal(self, phrase):
        signals = full_signals(0.9)
        signals["noise"] = sig(0.9, "high", note=f"Analyzer says: {phrase.upper()} flat regions")
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        assert "noise" not in used_signals(result)

    def test_unavailable_phrase_matching_is_case_insensitive(self):
        signals = full_signals(0.9)
        signals["gradients"] = sig(0.9, note="NOT ENOUGH skin pixels")
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        assert "gradients" not in used_signals(result)

    def test_benign_note_does_not_disable_signal(self):
        signals = full_signals(0.9)
        signals["noise"] = sig(0.9, note="noise floor estimated from 42 flat patches")
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        assert "noise" in used_signals(result)

    @pytest.mark.parametrize("profile", [NATIVE_PAGE, VECTOR_PAGE, SPARSE_PAGE])
    def test_native_vector_sparse_pages_reject_all_signals(self, profile):
        result = document.score_pdf_page(full_signals(1.0, "high"), profile)
        assert used_signals(result) == set()
        assert result["usable"] is False
        assert result["confidence"] == "insufficient"
        assert result["score"] == 0.0

    def test_native_page_with_high_image_fraction_still_rejected(self):
        # content_type wins over image_area_fraction for native/vector/sparse.
        profile = {"content_type": "native_document", "image_area_fraction": 0.99}
        result = document.score_pdf_page(full_signals(1.0, "high"), profile)
        assert used_signals(result) == set()

    @pytest.mark.parametrize("fraction", [0.0, 0.25, 0.50, 0.69, 0.6999])
    def test_mixed_page_below_070_rejects_all_signals(self, fraction):
        profile = {"content_type": "mixed_document", "image_area_fraction": fraction}
        result = document.score_pdf_page(full_signals(1.0, "high"), profile)
        assert used_signals(result) == set()
        assert result["confidence"] == "insufficient"

    @pytest.mark.parametrize("fraction", [0.70, 0.75, 0.7999])
    def test_mixed_page_between_070_and_080_allows_noise_and_gradients_only(self, fraction):
        profile = {"content_type": "mixed_document", "image_area_fraction": fraction}
        result = document.score_pdf_page(full_signals(1.0, "high"), profile)
        assert used_signals(result) == {"noise", "gradients"}
        assert result["effective_weight_sum"] == pytest.approx(W["noise"] + W["gradients"])

    @pytest.mark.parametrize("fraction", [0.80, 0.95, 1.0])
    def test_mixed_page_at_or_above_080_allows_all_signals(self, fraction):
        profile = {"content_type": "mixed_document", "image_area_fraction": fraction}
        result = document.score_pdf_page(full_signals(1.0, "high"), profile)
        assert used_signals(result) == set(SIGNAL_NAMES)

    @pytest.mark.parametrize("fraction", [0.65, 0.70, 0.79])
    def test_image_page_below_080_drops_frequency_and_lighting(self, fraction):
        profile = {"content_type": "scanned_or_image_document", "image_area_fraction": fraction}
        result = document.score_pdf_page(full_signals(1.0, "high"), profile)
        assert used_signals(result) == {"noise", "gradients"}

    @pytest.mark.parametrize("profile", [
        {},
        {"content_type": None, "image_area_fraction": 1.0},
        {"content_type": "unknown", "image_area_fraction": 1.0},
        {"content_type": "form_document", "image_area_fraction": 1.0},
        {"content_type": "SCANNED_OR_IMAGE_DOCUMENT", "image_area_fraction": 1.0},
        {"content_type": "photo", "image_area_fraction": 1.0},
    ])
    def test_unknown_or_missing_content_type_fails_closed(self, profile):
        # Page-type gating is an allow-list (RASTER_PAGE_TYPES): an empty
        # profile, None, a typo, a case mismatch or a future type must not be
        # scored as a camera image, whatever image_area_fraction says. (This
        # replaces a test expecting unknown types to be scored, which
        # contradicted the documented fail-closed design.)
        assert document.RASTER_PAGE_TYPES == {"scanned_or_image_document", "mixed_document"}
        result = document.score_pdf_page(full_signals(1.0, "high"), profile)
        assert used_signals(result) == set()
        assert result["confidence"] == "insufficient"
        assert result["usable"] is False

    def test_missing_content_type_is_reported_as_unknown(self):
        assert document.score_pdf_page(full_signals(1.0), {})["content_type"] == "unknown"

    def test_resampled_render_gates_pixel_level_signals(self):
        # A render our renderer could not draw 1:1 would make the frequency,
        # noise and gradients signals measure the renderer; only lighting is
        # left, which is never evidence on its own.
        profile = {**IMAGE_PAGE, "render_resampled": True}
        result = document.score_pdf_page(full_signals(1.0, "high"), profile)
        assert used_signals(result) == {"lighting"}
        assert result["confidence"] == "insufficient"
        assert result["score"] == 0.0

    def test_faithful_render_keeps_every_signal(self):
        profile = {**IMAGE_PAGE, "render_resampled": False}
        result = document.score_pdf_page(full_signals(0.7), profile)
        assert used_signals(result) == set(SIGNAL_NAMES)
        assert result["score"] == pytest.approx(0.7)

    def test_profiles_without_render_information_are_not_gated(self):
        # Hand-built / older profiles: the pipeline always sets the key.
        assert used_signals(document.score_pdf_page(full_signals(0.7), IMAGE_PAGE)) == set(SIGNAL_NAMES)

    def test_pixel_level_signal_set(self):
        assert document.PIXEL_LEVEL_SIGNALS == {"frequency", "noise", "gradients"}

    def test_image_area_fraction_accepts_string_numbers(self):
        profile = {"content_type": "scanned_or_image_document", "image_area_fraction": "0.9"}
        result = document.score_pdf_page(full_signals(1.0, "high"), profile)
        assert used_signals(result) == set(SIGNAL_NAMES)


# ---------------------------------------------------------------------------
# score_pdf_page
# ---------------------------------------------------------------------------


class TestScorePdfPage:
    def test_uniform_signals_give_that_score_when_fully_reliable(self):
        for s in (0.0, 0.25, 0.5, 0.75, 1.0):
            result = document.score_pdf_page(full_signals(s), IMAGE_PAGE)
            assert result["score"] == pytest.approx(s)
            assert result["raw_normalized_score"] == pytest.approx(s)
            assert result["effective_weight_sum"] == pytest.approx(1.0)
            assert result["evidence_coverage"] == pytest.approx(1.0)
            assert result["confidence"] == "medium-high"
            assert result["usable"] is True

    def test_score_is_weighted_by_the_image_path_weights(self):
        signals = {
            "frequency": sig(1.0),
            "noise": sig(0.0),
            "gradients": sig(1.0),
            "lighting": sig(0.0),
        }
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        expected = (0.25 * 1.0 + 0.40 * 0.0 + 0.25 * 1.0 + 0.10 * 0.0) / 1.0
        assert result["score"] == pytest.approx(expected, abs=1e-4)
        assert result["score"] == pytest.approx(0.50, abs=1e-4)

    def test_no_portrait_redistributes_gradients_weight(self):
        # Same rule as an uploaded non-portrait: gradients' weight moves to
        # noise and frequency in proportion 0.40 : 0.25.
        signals = {"frequency": sig(0.8), "noise": sig(0.4), "lighting": sig(0.2)}
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        expected = (NP_F * 0.8 + NP_N * 0.4 + W["lighting"] * 0.2) / (NP_F + NP_N + W["lighting"])
        assert result["score"] == pytest.approx(expected, abs=1e-4)
        assert result["combiner"]["gradients_weight_redistributed_to"] == ["noise", "frequency"]
        # ...but the evidence weight reports what was *measured*, before
        # redistribution: 0.25 + 0.40 + 0.10.
        assert result["effective_weight_sum"] == pytest.approx(0.75)

    def test_high_and_medium_confidence_are_equivalent(self):
        r_high = document.score_pdf_page(full_signals(0.7, "high"), IMAGE_PAGE)
        r_med = document.score_pdf_page(full_signals(0.7, "medium"), IMAGE_PAGE)
        assert r_high["score"] == r_med["score"]
        assert r_high["effective_weight_sum"] == r_med["effective_weight_sum"]

    def test_low_medium_confidence_scales_weight_by_0_7(self):
        signals = full_signals(0.5)
        signals["noise"] = sig(0.5, "low-medium")
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        assert result["contributions"]["noise"]["effective_weight"] == pytest.approx(0.40 * 0.7)
        assert result["effective_weight_sum"] == pytest.approx(0.25 + 0.28 + 0.25 + 0.10)

    def test_low_confidence_scales_weight_by_0_5(self):
        signals = full_signals(0.5)
        signals["gradients"] = sig(0.5, "low")
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        assert result["contributions"]["gradients"]["effective_weight"] == pytest.approx(0.25 * 0.5)

    def test_unknown_confidence_string_counts_as_low(self):
        # Changed with the shared combiner: an unrecognised confidence used
        # to zero a signal on PDFs only; it now counts as "low" (0.5) on
        # every path, as it always did for uploaded images.
        signals = full_signals(0.2)
        signals["noise"] = sig(1.0, "bogus")
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        assert result["contributions"]["noise"]["used"] is True
        assert result["contributions"]["noise"]["effective_weight"] == pytest.approx(0.40 * 0.5)
        assert result["effective_weight_sum"] == pytest.approx(0.25 + 0.20 + 0.25 + 0.10)
        expected = (0.25 * 0.2 + 0.20 * 1.0 + 0.25 * 0.2 + 0.10 * 0.2) / 0.80
        assert result["score"] == pytest.approx(expected, abs=1e-4)

    def test_only_unknown_confidence_signal_is_a_capped_single_signal(self):
        result = document.score_pdf_page({"noise": sig(1.0, "bogus")}, IMAGE_PAGE)
        assert result["effective_weight_sum"] == pytest.approx(0.20)
        assert result["confidence"] == "low-medium"
        assert result["score"] == CAP

    def test_missing_confidence_defaults_to_low(self):
        result = document.score_pdf_page({"noise": {"score": 1.0, "note": ""}}, IMAGE_PAGE)
        assert result["contributions"]["noise"]["confidence"] == "low"
        assert result["contributions"]["noise"]["effective_weight"] == pytest.approx(0.20)

    def test_low_confidence_evidence_gets_a_low_tier(self):
        # Every signal "screams AI" at low confidence. Confidence re-weights
        # signals against each other; it does not shrink a unanimous score
        # (the old PDF-only reliability shrink is gone). Thin evidence shows
        # in the tier instead: 0.5 of the stack -> "medium", not "medium-high".
        result = document.score_pdf_page(full_signals(1.0, "low"), IMAGE_PAGE)
        assert result["effective_weight_sum"] == pytest.approx(0.50)
        assert result["score"] == pytest.approx(1.0)
        assert result["confidence"] == "medium"

    def test_noise_only_is_capped_like_an_uploaded_image(self):
        # Replaces the old reliability shrink (1.0 -> 0.8182): one
        # substantive signal is capped at 0.60, as for a direct upload.
        result = document.score_pdf_page({"noise": sig(1.0)}, IMAGE_PAGE)
        assert result["raw_normalized_score"] == 1.0
        assert result["effective_weight_sum"] == pytest.approx(0.40)
        assert result["score"] == CAP
        assert result["combiner"]["single_signal_cap_applied"] is True
        assert result["confidence"] == "medium"
        assert result["usable"] is True

    def test_single_signal_below_the_cap_is_not_changed(self):
        result = document.score_pdf_page({"noise": sig(0.45)}, IMAGE_PAGE)
        assert result["score"] == pytest.approx(0.45)
        assert result["combiner"]["single_signal_cap_applied"] is False

    def test_lighting_is_not_corroboration_on_pdfs_either(self):
        result = document.score_pdf_page({"noise": sig(0.8), "lighting": sig(0.8)}, IMAGE_PAGE)
        assert result["score"] == CAP

    def test_two_substantive_signals_are_not_capped(self):
        signals = {"frequency": sig(0.8), "gradients": sig(0.8), "lighting": sig(0.8)}
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        assert result["effective_weight_sum"] == pytest.approx(0.60)
        assert result["score"] == pytest.approx(0.8)

    def test_lighting_only_page_is_insufficient_not_inconclusive(self):
        # A direct upload with no substantive signal reports 0.5
        # ("inconclusive"); inside a document it must add no risk at all.
        result = document.score_pdf_page({"lighting": sig(1.0, "high")}, IMAGE_PAGE)
        assert result["confidence"] == "insufficient"
        assert result["usable"] is False
        assert result["score"] == 0.0

    @pytest.mark.parametrize(
        "signals, expected_ews, expected_conf",
        [
            (full_signals(0.5), 1.00, "medium-high"),
            ({"noise": sig(0.5), "lighting": sig(0.5), "frequency": sig(0.5)}, 0.75, "medium-high"),
            ({"noise": sig(0.5), "frequency": sig(0.5)}, 0.65, "medium-high"),
            ({"noise": sig(0.5), "lighting": sig(0.5)}, 0.50, "medium"),
            ({"gradients": sig(0.5), "frequency": sig(0.5)}, 0.50, "medium"),
            ({"noise": sig(0.5)}, 0.40, "medium"),
            ({"frequency": sig(0.5), "lighting": sig(0.5)}, 0.35, "low-medium"),
            ({"noise": sig(0.5, "low-medium")}, 0.28, "low-medium"),
            ({"gradients": sig(0.5)}, 0.25, "low-medium"),
            ({"noise": sig(0.5, "low")}, 0.20, "low-medium"),
            ({"frequency": sig(0.5, "low-medium")}, 0.175, "low"),
            ({"gradients": sig(0.5, "low")}, 0.125, "low"),
        ],
    )
    def test_confidence_tiers_by_effective_weight(self, signals, expected_ews, expected_conf):
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        assert result["effective_weight_sum"] == pytest.approx(expected_ews)
        assert result["confidence"] == expected_conf

    @pytest.mark.parametrize(
        "signals, expected_usable",
        [
            ({"frequency": sig(0.9, "low-medium")}, True),  # 0.175
            ({"noise": sig(0.9, "low")}, True),             # 0.20
            ({"gradients": sig(0.9, "low")}, False),        # 0.125
            ({"frequency": sig(0.9, "low")}, False),        # 0.125
            ({"lighting": sig(0.9)}, False),                # no substantive signal
            ({"gradients": sig(0.9, "low"), "lighting": sig(0.9, "low")}, True),  # 0.175
        ],
    )
    def test_usable_requires_effective_weight_of_at_least_015(self, signals, expected_usable):
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        assert result["usable"] is expected_usable

    def test_scores_above_one_are_clipped(self):
        result = document.score_pdf_page({"noise": sig(5.0), "gradients": sig(5.0)}, IMAGE_PAGE)
        assert result["contributions"]["noise"]["score"] == 1.0
        assert result["raw_normalized_score"] == 1.0
        assert result["score"] == 1.0

    def test_scores_below_zero_are_clipped(self):
        result = document.score_pdf_page({"noise": sig(-3.0), "gradients": sig(-3.0)}, IMAGE_PAGE)
        assert result["contributions"]["noise"]["score"] == 0.0
        assert result["score"] == 0.0

    def test_insufficient_result_shape(self):
        result = document.score_pdf_page({}, IMAGE_PAGE)
        assert result == {
            "score": 0.0,
            "confidence": "insufficient",
            "usable": False,
            "evidence_coverage": 0.0,
            "content_type": "scanned_or_image_document",
            "contributions": {
                name: {
                    "used": False,
                    "reason": "not applicable or unavailable for this PDF element",
                    "effective_weight": 0.0,
                }
                for name in SIGNAL_NAMES
            },
            "note": "photo-forensic analysis is not applicable/reliable on this rendered PDF page",
        }

    def test_usable_result_contains_all_expected_keys(self):
        result = document.score_pdf_page(full_signals(0.3), IMAGE_PAGE)
        assert set(result) == {
            "score",
            "raw_normalized_score",
            "confidence",
            "usable",
            "effective_weight_sum",
            "evidence_coverage",
            "content_type",
            "contributions",
            "combiner",
            "note",
        }
        assert set(result["contributions"]) == set(SIGNAL_NAMES)
        for name in SIGNAL_NAMES:
            c = result["contributions"][name]
            assert c["used"] is True
            assert c["score"] == 0.3
            assert c["confidence"] == "medium"
            assert c["effective_weight"] == document.PDF_SCAN_WEIGHTS[name]
            assert c["combined_weight"] == pytest.approx(document.PDF_SCAN_WEIGHTS[name])

    def test_evidence_coverage_is_fraction_of_total_base_weight(self):
        result = document.score_pdf_page({"gradients": sig(0.5), "noise": sig(0.5, "low-medium")}, IMAGE_PAGE)
        assert result["evidence_coverage"] == pytest.approx(0.25 + 0.28)

    def test_scores_are_rounded_to_four_decimals(self):
        result = document.score_pdf_page({"noise": sig(1 / 3)}, IMAGE_PAGE)
        assert result["score"] == round(result["score"], 4)
        assert result["raw_normalized_score"] == 0.3333

    def test_all_signals_at_040_produce_exactly_040(self):
        # Fixture used by the flagged-page threshold test further below.
        result = document.score_pdf_page(full_signals(0.40), IMAGE_PAGE)
        assert result["score"] == 0.4
        assert result["usable"] is True
        assert result["confidence"] == "medium-high"

    def test_input_dicts_are_not_mutated(self):
        signals = full_signals(0.5)
        profile = dict(IMAGE_PAGE)
        snapshot_signals = {k: dict(v) for k, v in signals.items()}
        snapshot_profile = dict(profile)
        document.score_pdf_page(signals, profile)
        assert signals == snapshot_signals
        assert profile == snapshot_profile


# ---------------------------------------------------------------------------
# score_embedded_image
# ---------------------------------------------------------------------------


class TestScoreEmbeddedImage:
    @pytest.mark.parametrize(
        "profile",
        [
            {"width_px": 63, "height_px": 1000, "pixel_count": 63_000},
            {"width_px": 1000, "height_px": 63, "pixel_count": 63_000},
            {"width_px": 64, "height_px": 64, "pixel_count": 4096},
            {"width_px": 128, "height_px": 127, "pixel_count": 16_256},  # 16,256 < 16,384
            {"width_px": 0, "height_px": 0},
            {},
        ],
    )
    def test_too_small_rasters_are_unusable(self, profile):
        result = document.score_embedded_image(full_signals(1.0, "high"), profile)
        assert result["usable"] is False
        assert result["score"] == 0.0
        assert result["confidence"] == "insufficient"
        assert result["evidence_coverage"] == 0.0
        assert result["contributions"] == {}
        assert result["image_profile"] is profile
        assert "too small" in result["note"]

    def test_minimum_acceptable_size(self):
        profile = {"width_px": 128, "height_px": 128, "pixel_count": 16_384}
        result = document.score_embedded_image(full_signals(0.5), profile)
        assert result["usable"] is True
        assert result["score"] == pytest.approx(0.5)

    def test_pixel_count_defaults_to_width_times_height(self):
        profile = {"width_px": 200, "height_px": 100}  # 20,000 px, no pixel_count key
        result = document.score_embedded_image(full_signals(0.5), profile)
        assert result["usable"] is True

    def test_explicit_pixel_count_overrides_dimensions(self):
        # The module trusts pixel_count when present (even if inconsistent).
        profile = {"width_px": 200, "height_px": 200, "pixel_count": 100}
        result = document.score_embedded_image(full_signals(0.5), profile)
        assert result["usable"] is False
        assert "too small" in result["note"]

    @pytest.mark.parametrize("coverage", [1e-6, 0.001, 0.01, 0.5, 1.0])
    def test_page_coverage_never_dilutes_the_score(self, coverage):
        profile = {**EMBEDDED_PROFILE, "page_coverage": coverage}
        result = document.score_embedded_image(full_signals(0.9), profile)
        assert result["score"] == pytest.approx(0.9)
        assert result["confidence"] == "medium-high"
        assert result["usable"] is True

    def test_embedded_scoring_ignores_page_type_gating(self):
        # Embedded images are scored on their own pixels, so the *page* being
        # native text must not suppress them. score_embedded_image does not
        # even take a page profile -- assert that all four signals are used.
        result = document.score_embedded_image(full_signals(0.6), EMBEDDED_PROFILE)
        assert used_signals(result) == set(SIGNAL_NAMES)

    def test_none_score_is_unavailable(self):
        signals = full_signals(0.9)
        signals["noise"] = {"score": None, "confidence": "high", "note": ""}
        result = document.score_embedded_image(signals, EMBEDDED_PROFILE)
        assert result["contributions"]["noise"] == {
            "used": False,
            "reason": "not applicable or unavailable for this PDF element",
            "effective_weight": 0.0,
        }

    def test_applicable_false_is_unavailable(self):
        signals = full_signals(0.9)
        signals["gradients"] = {"score": 0.5, "confidence": "low", "applicable": False, "note": "x"}
        result = document.score_embedded_image(signals, EMBEDDED_PROFILE)
        assert "gradients" not in used_signals(result)

    @pytest.mark.parametrize("phrase", document._UNAVAILABLE_PHRASES)
    def test_unavailable_note_phrases_disable_a_signal(self, phrase):
        signals = full_signals(0.9)
        signals["gradients"] = sig(0.9, note=f"{phrase} skin")
        result = document.score_embedded_image(signals, EMBEDDED_PROFILE)
        assert "gradients" not in used_signals(result)

    def test_missing_and_none_signal_entries_are_tolerated(self):
        signals = {"noise": sig(0.5), "gradients": None}
        result = document.score_embedded_image(signals, EMBEDDED_PROFILE)
        assert used_signals(result) == {"noise"}

    def test_unknown_confidence_counts_as_low(self):
        signals = full_signals(0.2)
        signals["noise"] = sig(1.0, "definitely-not-a-tier")
        result = document.score_embedded_image(signals, EMBEDDED_PROFILE)
        assert result["contributions"]["noise"]["used"] is True
        assert result["effective_weight_sum"] == pytest.approx(0.80)
        page = document.score_pdf_page(signals, IMAGE_PAGE)
        assert result["score"] == page["score"]

    def test_no_usable_signals_is_unusable_with_distinct_reason(self):
        result = document.score_embedded_image({}, EMBEDDED_PROFILE)
        assert result["usable"] is False
        assert result["score"] == 0.0
        assert result["confidence"] == "insufficient"
        assert result["note"] == "embedded raster had no usable forensic signals"
        assert result["image_profile"] is EMBEDDED_PROFILE

    def test_noise_only_is_capped_like_an_uploaded_image(self):
        result = document.score_embedded_image({"noise": sig(1.0)}, EMBEDDED_PROFILE)
        assert result["raw_normalized_score"] == 1.0
        assert result["score"] == CAP
        assert result["confidence"] == "medium"
        assert result["usable"] is True

    def test_low_confidence_evidence_gets_a_low_tier(self):
        result = document.score_embedded_image(full_signals(1.0, "low"), EMBEDDED_PROFILE)
        assert result["effective_weight_sum"] == pytest.approx(0.50)
        assert result["score"] == pytest.approx(1.0)
        assert result["confidence"] == "medium"

    def test_lighting_only_raster_is_unusable(self):
        result = document.score_embedded_image({"lighting": sig(1.0, "high")}, EMBEDDED_PROFILE)
        assert result["usable"] is False
        assert result["note"] == "embedded raster had no usable forensic signals"

    @pytest.mark.parametrize(
        "signals, expected_conf, expected_usable",
        [
            (full_signals(0.5), "medium-high", True),
            ({"noise": sig(0.5), "lighting": sig(0.5)}, "medium", True),
            ({"gradients": sig(0.5)}, "low-medium", True),
            ({"frequency": sig(0.5, "low-medium")}, "low", True),  # 0.175
            ({"gradients": sig(0.5, "low")}, "low", False),        # 0.125
        ],
    )
    def test_confidence_and_usable_tiers_match_page_scoring(self, signals, expected_conf, expected_usable):
        result = document.score_embedded_image(signals, EMBEDDED_PROFILE)
        assert result["confidence"] == expected_conf
        assert result["usable"] is expected_usable

    def test_page_and_embedded_scoring_agree_on_identical_signals(self):
        # Both paths use the image-path combiner; for a fully-raster page they
        # must produce identical numbers.
        for score in (0.1, 0.4, 0.77):
            for conf in ("high", "medium", "low-medium", "low"):
                signals = full_signals(score, conf)
                page = document.score_pdf_page(signals, IMAGE_PAGE)
                emb = document.score_embedded_image(signals, EMBEDDED_PROFILE)
                assert page["score"] == emb["score"]
                assert page["confidence"] == emb["confidence"]
                assert page["usable"] == emb["usable"]
                assert page["effective_weight_sum"] == emb["effective_weight_sum"]

    @pytest.mark.parametrize(
        "signals, raw_sum, expected_conf",
        [
            # 0.175 + 0.175 + 0.05 -> 0.39999999999999997 raw
            ({"frequency": sig(0.5, "low-medium"), "gradients": sig(0.5, "low-medium"),
              "lighting": sig(0.5, "low")}, 0.39999999999999997, "medium"),
            # 0.175 + 0.28 + 0.125 + 0.07 -> 0.6499999999999999 raw
            ({"frequency": sig(0.5, "low-medium"), "noise": sig(0.5, "low-medium"),
              "gradients": sig(0.5, "low"), "lighting": sig(0.5, "low-medium")}, 0.6499999999999999, "medium-high"),
        ],
    )
    def test_tier_boundaries_use_rounded_weight_sum(self, signals, raw_sum, expected_conf):
        """Regression: tier comparison must not be demoted by IEEE-754
        accumulation error in the base weights. Both scoring paths round the
        effective weight sum to 4 decimals before comparing."""
        # Sanity-check the premise by replicating the module's accumulation
        # (0.0 += weight * factor, in PDF_SCAN_WEIGHTS order): the raw float
        # sum really is below the nominal boundary.
        naive = 0.0
        for name, base_weight in document.PDF_SCAN_WEIGHTS.items():
            if name in signals:
                naive += base_weight * document.CONFIDENCE_FACTOR[signals[name]["confidence"]]
        assert naive == raw_sum
        assert naive < round(naive, 4)

        for result in (
            document.score_pdf_page(signals, IMAGE_PAGE),
            document.score_embedded_image(signals, EMBEDDED_PROFILE),
        ):
            assert result["effective_weight_sum"] == round(raw_sum, 4)
            assert result["confidence"] == expected_conf

    def test_scores_are_clipped_to_unit_interval(self):
        hi = document.score_embedded_image({"noise": sig(9.0), "gradients": sig(9.0)}, EMBEDDED_PROFILE)
        lo = document.score_embedded_image({"noise": sig(-9.0), "gradients": sig(-9.0)}, EMBEDDED_PROFILE)
        assert hi["score"] == 1.0
        assert lo["score"] == 0.0

    def test_result_shape(self):
        result = document.score_embedded_image(full_signals(0.3), EMBEDDED_PROFILE)
        assert set(result) == {
            "score",
            "raw_normalized_score",
            "confidence",
            "usable",
            "effective_weight_sum",
            "evidence_coverage",
            "image_profile",
            "contributions",
            "combiner",
            "note",
        }
        assert result["image_profile"] is EMBEDDED_PROFILE
        assert "page coverage does not dilute" in result["note"]

    def test_all_signals_at_040_produce_exactly_040(self):
        result = document.score_embedded_image(full_signals(0.40), EMBEDDED_PROFILE)
        assert result["score"] == 0.4


# ---------------------------------------------------------------------------
# Parity with the image path
# ---------------------------------------------------------------------------

NEUTRAL_META = {"score": 0.5, "confidence": "low", "note": "neutral"}
CTX = {"compression": "jpeg_low", "size": "medium", "scale": "upscaled", "color": "color"}
CONTEXT_MODEL = {"multipliers": {
    "frequency": {"compression": {"jpeg_low": 0.5}},
    "noise": {"scale": {"upscaled": 1.5}},
    "gradients": {"scale": {"upscaled": 0.8}},
}}


def _random_signal_sets(n, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    confs = ["high", "medium", "low-medium", "low"]
    for _ in range(n):
        signals = {}
        for name in SIGNAL_NAMES:
            roll = rng.random()
            if roll < 0.25:
                signals[name] = {"score": 0.5, "confidence": "low", "applicable": False, "note": "n/a"}
            else:
                signals[name] = sig(float(rng.random()), confs[int(rng.integers(0, 4))])
        yield signals


def _upload_score(signals, buckets=None, model=None):
    """What the same signals score as a direct PNG/JPEG upload."""
    from app import main
    analysis = {"signals": signals, "metadata": NEUTRAL_META,
                "context": {"buckets": buckets or {}}}
    return main.score_image_signals(analysis, model)["ai_generated"]


class TestImagePathParity:
    def test_embedded_raster_scores_exactly_like_an_upload(self):
        checked = 0
        for signals in _random_signal_sets(400, seed=11):
            emb = document.score_embedded_image(signals, EMBEDDED_PROFILE, None, None)
            if not emb["usable"] and emb["confidence"] == "insufficient":
                continue
            assert emb["score"] == _upload_score(signals)
            checked += 1
        assert checked > 250

    def test_full_raster_page_scores_exactly_like_an_upload(self):
        for signals in _random_signal_sets(300, seed=12):
            page = document.score_pdf_page(signals, IMAGE_PAGE, None, None)
            if page["confidence"] == "insufficient":
                continue
            assert page["score"] == _upload_score(signals)

    def test_context_factors_apply_identically(self):
        for signals in _random_signal_sets(300, seed=13):
            emb = document.score_embedded_image(signals, EMBEDDED_PROFILE, CTX, CONTEXT_MODEL)
            if emb["confidence"] == "insufficient":
                continue
            assert emb["score"] == _upload_score(signals, CTX, CONTEXT_MODEL)
            assert emb["combiner"]["context_buckets"] == CTX

    def test_context_factors_change_the_pdf_score(self):
        signals = {"frequency": sig(0.9), "noise": sig(0.3), "lighting": sig(0.5)}
        plain = document.score_embedded_image(signals, EMBEDDED_PROFILE, None, CONTEXT_MODEL)
        ctx = document.score_embedded_image(signals, EMBEDDED_PROFILE, CTX, CONTEXT_MODEL)
        assert ctx["score"] < plain["score"]  # frequency halved, noise boosted
        assert ctx["combiner"]["context_weighting"] is True
        assert plain["combiner"]["context_weighting"] is False

    def test_gated_page_equals_an_upload_of_the_allowed_signals(self):
        # On a mixed page below 0.80 image area, frequency and lighting are
        # gated out: the page must score exactly what an upload would with
        # only noise + gradients measured (the gated signals not applicable).
        profile = {"content_type": "mixed_document", "image_area_fraction": 0.75}
        for signals in _random_signal_sets(200, seed=14):
            page = document.score_pdf_page(signals, profile, None, None)
            if page["confidence"] == "insufficient":
                continue
            allowed = dict(signals)
            for gated in ("frequency", "lighting"):
                allowed[gated] = {"score": None, "applicable": False, "note": "gated"}
            assert page["score"] == _upload_score(allowed)

    def test_realistic_non_portrait_raster(self):
        # Signals measured on fireplace-generated.png: the embedded raster
        # gets the upload's 0.731 (static weights), where the old PDF
        # weights gave it 0.66.
        signals = {
            "frequency": sig(0.870), "noise": sig(0.694),
            "gradients": {"score": 0.5, "confidence": "low", "applicable": False,
                          "note": "no clear portrait subject"},
            "lighting": sig(0.179, "low"),
        }
        emb = document.score_embedded_image(signals, EMBEDDED_PROFILE, None, None)
        assert emb["score"] == pytest.approx(0.7309, abs=1e-4)
        assert emb["score"] == _upload_score(signals)
        assert emb["confidence"] == "medium-high"  # 0.25 + 0.40 + 0.05

    def test_realistic_faceless_jpeg_is_now_usable_evidence(self):
        # a faceless JPEG like real-photo.jpeg: noise unavailable (JPEG grid), no face. Under the old
        # PDF weights (frequency 0.05) this was "insufficient"; with the
        # image-path weights it is low-medium evidence that reads as real.
        signals = {
            "frequency": sig(0.0),
            "noise": {"score": 0.5, "confidence": "low", "applicable": False,
                      "note": "noise floor unavailable as evidence"},
            "gradients": {"score": 0.5, "confidence": "low", "applicable": False,
                          "note": "no clear portrait subject"},
            "lighting": sig(0.502, "low"),
        }
        emb = document.score_embedded_image(signals, EMBEDDED_PROFILE, None, None)
        assert emb["usable"] is True
        assert emb["confidence"] == "low-medium"  # 0.25 + 0.05
        assert emb["score"] == _upload_score(signals) < 0.1


SIGNATURE_META = {"score": 0.97, "confidence": "high", "signature_hits": ["EXIF Software tag references 'midjourney'"]}
CAMERA_META = {"score": 0.15, "confidence": "low-medium", "camera_fields_present": ["Make", "Model", "FNumber"]}
NEUTRAL_FILE_META = {"score": 0.5, "confidence": "low"}
TINY = {"width_px": 48, "height_px": 48, "pixel_count": 2304}


class TestEmbeddedImageFileMetadata:
    """The embedded image's own file metadata is folded in with
    scoring.combine, exactly as for an upload."""

    def test_no_metadata_leaves_the_pixel_score(self):
        base = document.score_embedded_image(full_signals(0.4), EMBEDDED_PROFILE, None, None)
        assert document.score_embedded_image(full_signals(0.4), EMBEDDED_PROFILE, None, None,
                                             metadata_result=None) == base

    def test_neutral_metadata_does_not_change_the_score(self):
        result = document.score_embedded_image(full_signals(0.4), EMBEDDED_PROFILE, None, None,
                                               metadata_result=NEUTRAL_FILE_META)
        assert result["score"] == pytest.approx(0.4)
        assert result["metadata"] == NEUTRAL_FILE_META

    def test_generator_signature_raises_to_at_least_its_score(self):
        result = document.score_embedded_image(full_signals(0.2), EMBEDDED_PROFILE, None, None,
                                               metadata_result=SIGNATURE_META)
        assert result["score"] == 0.97
        assert result["visual_score"] == pytest.approx(0.2)
        assert result["confidence"] == "high"
        assert result["usable"] is True

    def test_generator_signature_never_lowers_stronger_pixels(self):
        result = document.score_embedded_image(full_signals(0.99), EMBEDDED_PROFILE, None, None,
                                               metadata_result=SIGNATURE_META)
        assert result["score"] == pytest.approx(0.99)

    def test_generator_signature_counts_even_on_an_icon_sized_raster(self):
        # The size gate protects pixel heuristics; a signature does not
        # depend on pixels.
        result = document.score_embedded_image(full_signals(0.1), TINY, None, None,
                                               metadata_result=SIGNATURE_META)
        assert result["usable"] is True and result["confidence"] == "high"
        assert result["score"] == 0.97
        assert result["visual_score"] == 0.0

    def test_camera_exif_nudges_like_an_upload(self):
        result = document.score_embedded_image(full_signals(0.6), EMBEDDED_PROFILE, None, None,
                                               metadata_result=CAMERA_META)
        assert result["score"] == pytest.approx(0.85 * 0.6 + 0.15 * 0.15, abs=1e-4)
        assert result["confidence"] == "medium-high"  # camera EXIF does not change the tier

    def test_camera_exif_is_not_evidence_on_its_own(self):
        result = document.score_embedded_image(full_signals(0.6), TINY, None, None,
                                               metadata_result=CAMERA_META)
        assert result["usable"] is False

    def test_signature_on_an_embedded_image_drives_the_document(self):
        emb = document.score_embedded_image(full_signals(0.1), EMBEDDED_PROFILE, None, None,
                                            metadata_result=SIGNATURE_META)
        page = document.combine_page_evidence(1, {"usable": False, "confidence": "insufficient"},
                                              [embedded_entry(1, emb)])
        doc = document.aggregate_pdf_pages([{"page": 1, "pdf_scoring": page}])
        assert doc["score"] == 0.97 and doc["confidence"] == "high"
        assert document.classify_pdf(doc["score"], doc["confidence"])[0] == "likely_ai_generated"


# ---------------------------------------------------------------------------
# Placeholder helpers
# ---------------------------------------------------------------------------


class TestPlaceholderHelpers:
    def test_unavailable_embedded_signals_shape(self):
        result = document.unavailable_embedded_signals("decoder crashed")
        assert set(result) == {"frequency", "noise", "gradients", "lighting", "metadata"}
        for value in result.values():
            assert value == {"score": None, "confidence": "unavailable", "note": "decoder crashed"}

    def test_unavailable_embedded_signals_entries_are_independent_dicts(self):
        result = document.unavailable_embedded_signals("x")
        result["noise"]["score"] = 0.9
        assert result["frequency"]["score"] is None

    def test_unavailable_embedded_signals_score_as_unusable(self):
        signals = document.unavailable_embedded_signals("decoder crashed")
        result = document.score_embedded_image(signals, EMBEDDED_PROFILE)
        assert result["usable"] is False
        assert result["confidence"] == "insufficient"

    def test_unusable_embedded_image_score_shape(self):
        profile = {"width_px": 10, "height_px": 10}
        result = document.unusable_embedded_image_score(profile, "because")
        assert result == {
            "score": 0.0,
            "confidence": "insufficient",
            "usable": False,
            "evidence_coverage": 0.0,
            "image_profile": profile,
            "contributions": {},
            "note": "because",
        }

    def test_not_applicable_signals_use_none_not_half(self):
        # 0.5 would be a numeric AI score; None means "not applicable".
        result = document.not_applicable_signals(NATIVE_PAGE)
        assert set(result) == {"frequency", "noise", "gradients", "lighting", "metadata"}
        for name, value in result.items():
            assert value["score"] is None
            assert value["confidence"] == "unavailable"
        assert "native_document" in result["noise"]["note"]
        assert "per-page raster metadata" in result["metadata"]["note"]

    def test_not_applicable_signals_entries_are_independent_dicts(self):
        result = document.not_applicable_signals(NATIVE_PAGE)
        result["noise"]["score"] = 0.9
        assert result["gradients"]["score"] is None

    def test_not_applicable_signals_handle_unknown_content_type(self):
        result = document.not_applicable_signals({})
        assert "'unknown'" in result["noise"]["note"]

    def test_not_applicable_signals_score_as_insufficient_page(self):
        signals = document.not_applicable_signals(NATIVE_PAGE)
        result = document.score_pdf_page(signals, NATIVE_PAGE)
        assert result["confidence"] == "insufficient"
        assert result["usable"] is False
        # Even on an image page these None-scored placeholders are unusable.
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        assert result["confidence"] == "insufficient"


# ---------------------------------------------------------------------------
# combine_page_evidence
# ---------------------------------------------------------------------------


class TestCombinePageEvidence:
    def _render(self, score, confidence="medium", usable=True, coverage=1.0):
        return {
            "score": score,
            "confidence": confidence,
            "usable": usable,
            "evidence_coverage": coverage,
        }

    def test_no_candidates_returns_insufficient(self):
        render = document.score_pdf_page({}, NATIVE_PAGE)
        result = document.combine_page_evidence(3, render, [])
        assert result == {
            "score": 0.0,
            "confidence": "insufficient",
            "usable": False,
            "source": None,
            "image_index": None,
            "page": 3,
            "evidence_coverage": 0.0,
            "note": "no reliable page-render or embedded-image evidence on this PDF page",
        }

    def test_page_render_only(self):
        result = document.combine_page_evidence(1, self._render(0.61, "medium-high", coverage=0.9), [])
        assert result["score"] == 0.61
        assert result["confidence"] == "medium-high"
        assert result["usable"] is True
        assert result["source"] == "page_render"
        assert result["image_index"] is None
        assert result["page"] == 1
        assert result["evidence_coverage"] == 0.9
        assert result["candidate_count"] == 1

    def test_embedded_only_on_native_page(self):
        render = document.score_pdf_page(document.not_applicable_signals(NATIVE_PAGE), NATIVE_PAGE)
        scoring = document.score_embedded_image(full_signals(0.83), EMBEDDED_PROFILE)
        result = document.combine_page_evidence(2, render, [embedded_entry(7, scoring)])
        assert result["score"] == 0.83
        assert result["source"] == "embedded_image"
        assert result["image_index"] == 7
        assert result["confidence"] == "medium-high"
        assert result["evidence_coverage"] == 1.0
        assert result["candidate_count"] == 1

    def test_worst_candidate_wins_regardless_of_source(self):
        render = self._render(0.30)
        embedded = [
            embedded_entry(1, document.score_embedded_image(full_signals(0.10), EMBEDDED_PROFILE)),
            embedded_entry(2, document.score_embedded_image(full_signals(0.95), EMBEDDED_PROFILE)),
            embedded_entry(3, document.score_embedded_image(full_signals(0.50), EMBEDDED_PROFILE)),
        ]
        result = document.combine_page_evidence(1, render, embedded)
        assert result["score"] == 0.95
        assert result["source"] == "embedded_image"
        assert result["image_index"] == 2
        assert result["candidate_count"] == 4

    def test_page_render_wins_when_it_is_worst(self):
        render = self._render(0.90)
        embedded = [embedded_entry(1, document.score_embedded_image(full_signals(0.40), EMBEDDED_PROFILE))]
        result = document.combine_page_evidence(1, render, embedded)
        assert result["source"] == "page_render"
        assert result["score"] == 0.9

    def test_tie_prefers_page_render(self):
        # ``max`` keeps the first maximal candidate; page render is appended
        # first. Pins the current ordering so a reshuffle is noticed.
        render = self._render(0.5)
        embedded = [embedded_entry(1, document.score_embedded_image(full_signals(0.5), EMBEDDED_PROFILE))]
        result = document.combine_page_evidence(1, render, embedded)
        assert result["score"] == 0.5
        assert result["source"] == "page_render"
        assert result["candidate_count"] == 2

    def test_unusable_page_render_is_excluded_even_with_reliable_confidence(self):
        render = self._render(0.99, "medium", usable=False)
        result = document.combine_page_evidence(1, render, [])
        assert result["usable"] is False
        assert result["confidence"] == "insufficient"

    @pytest.mark.parametrize("confidence", ["low", "insufficient", "unavailable", "", "bogus"])
    def test_unreliable_page_render_confidence_is_excluded(self, confidence):
        render = self._render(0.99, confidence)
        result = document.combine_page_evidence(1, render, [])
        assert result["usable"] is False
        assert result["score"] == 0.0

    @pytest.mark.parametrize("confidence", sorted(RELIABLE_CONFIDENCES))
    def test_reliable_page_render_confidences_are_included(self, confidence):
        render = self._render(0.42, confidence)
        result = document.combine_page_evidence(1, render, [])
        assert result["usable"] is True
        assert result["confidence"] == confidence

    def test_low_confidence_embedded_image_cannot_drive_page_risk(self):
        # Embedded image with a high score but only 'low' confidence (a lone
        # low-medium-confidence signal, 0.175 of the stack) is retained in
        # the response but not a candidate.
        scoring = document.score_embedded_image({"frequency": sig(1.0, "low-medium")}, EMBEDDED_PROFILE)
        assert scoring["usable"] is True and scoring["confidence"] == "low"
        result = document.combine_page_evidence(1, self._render(0.2), [embedded_entry(1, scoring)])
        assert result["source"] == "page_render"
        assert result["score"] == 0.2
        assert result["candidate_count"] == 1

    def test_unusable_embedded_image_is_excluded(self):
        tiny = document.score_embedded_image(full_signals(1.0, "high"), {"width_px": 10, "height_px": 10})
        result = document.combine_page_evidence(1, self._render(0.2), [embedded_entry(1, tiny)])
        assert result["score"] == 0.2
        assert result["candidate_count"] == 1

    def test_embedded_image_missing_optional_keys(self):
        # ai_generated / scoring may be absent -> defaults of 0.0 / coverage 0.
        img = {"usable": True, "confidence": "medium", "image_index": 4}
        result = document.combine_page_evidence(1, {}, [img])
        assert result["score"] == 0.0
        assert result["source"] == "embedded_image"
        assert result["image_index"] == 4
        assert result["evidence_coverage"] == 0.0

    def test_empty_page_render_dict_is_tolerated(self):
        result = document.combine_page_evidence(5, {}, [])
        assert result["usable"] is False
        assert result["page"] == 5

    def test_scores_are_rounded_and_cast(self):
        render = self._render("0.123456", "medium", coverage="0.98765")
        result = document.combine_page_evidence(1, render, [])
        assert result["score"] == 0.1235
        assert result["evidence_coverage"] == 0.9877
        assert isinstance(result["score"], float)

    def test_page_number_is_echoed(self):
        for n in (1, 7, 20):
            assert document.combine_page_evidence(n, self._render(0.1), [])["page"] == n


# ---------------------------------------------------------------------------
# aggregate_pdf_pages
# ---------------------------------------------------------------------------


class TestAggregatePdfPagesEmptyAndUnusable:
    def test_no_pages(self):
        result = document.aggregate_pdf_pages([])
        assert result == {
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

    def test_all_pages_unusable(self):
        pages = [page_entry(i, 0.9, "insufficient", usable=False) for i in range(1, 4)]
        result = document.aggregate_pdf_pages(pages)
        assert result["score"] == 0.0
        assert result["confidence"] == "insufficient"
        assert result["usable_pages"] == 0
        assert result["worst_page"] is None
        assert result["flagged_pages"] == []

    def test_pages_without_pdf_scoring_key_are_unusable(self):
        result = document.aggregate_pdf_pages([{"page": 1}, {"page": 2}])
        assert result["usable_pages"] == 0
        assert result["worst_page"] is None

    def test_only_low_confidence_pages_stay_inconclusive_but_report_diagnostics(self):
        pages = [
            page_entry(1, 0.90, "low", evidence_coverage=0.15),
            page_entry(2, 0.10, "low", evidence_coverage=0.15),
        ]
        result = document.aggregate_pdf_pages(pages)
        assert result["score"] == 0.0
        assert result["confidence"] == "insufficient"
        assert result["usable_pages"] == 2
        assert result["reliable_pages"] == 0
        assert result["worst_page"] is None
        assert result["flagged_pages"] == []
        # Diagnostics are still computed from the usable pages.
        assert result["max_page_score"] == 0.9
        assert result["mean_page_score"] == pytest.approx(0.5)
        assert "no page is reliable enough" in result["note"]

    def test_low_confidence_weighted_mean_uses_coverage_floor(self):
        # Coverage 0.0 is floored to 0.05 so a page is never weighted to zero.
        pages = [
            page_entry(1, 1.0, "low", evidence_coverage=0.0),
            page_entry(2, 0.0, "low", evidence_coverage=0.05),
        ]
        result = document.aggregate_pdf_pages(pages)
        assert result["mean_page_score"] == pytest.approx(0.5)


class TestAggregatePdfPagesWorstPageWins:
    def test_multi_page_document_one_page_clearly_worse(self):
        """Headline scenario: a five-page document where page 4 is blatant and
        every other page is clean. The document score must equal page 4's
        score, page 4 must be the only flagged page, and the clean pages must
        not dilute the verdict."""
        pages = [
            page_entry(1, 0.08, "medium-high"),
            page_entry(2, 0.12, "medium-high"),
            page_entry(3, 0.05, "medium"),
            page_entry(4, 0.91, "medium-high"),
            page_entry(5, 0.10, "medium-high"),
        ]
        result = document.aggregate_pdf_pages(pages)

        assert result["score"] == 0.91
        assert result["max_page_score"] == 0.91
        assert result["worst_page"] == 4
        assert result["confidence"] == "medium-high"
        assert result["usable_pages"] == 5
        assert result["reliable_pages"] == 5
        assert result["aggregation"] == "worst_reliable_evidence"
        assert result["worst_evidence_source"] == "page_render"
        assert result["worst_image_index"] is None

        # Diagnostic mean is far lower than the max -- proving the verdict is
        # not an average.
        expected_mean = (0.08 + 0.12 + 0.05 + 0.91 + 0.10) / 5
        assert result["mean_page_score"] == pytest.approx(expected_mean, abs=1e-4)
        assert result["mean_page_score"] < 0.30 < result["score"]

        assert result["flagged_pages"] == [{
            "page": 4,
            "score": 0.91,
            "confidence": "medium-high",
            "content_type": "scanned_or_image_document",
            "risk_source": "page_render",
            "image_index": None,
        }]

    def test_multi_page_document_one_page_clearly_worse_end_to_end(self):
        """Same scenario, but every page is built through score_pdf_page ->
        combine_page_evidence with realistic signal dicts, including a native
        text page and a text-heavy mixed page that must be ignored."""
        pages = [
            build_page(1, document.not_applicable_signals(NATIVE_PAGE), NATIVE_PAGE),
            build_page(2, full_signals(0.10), IMAGE_PAGE),
            build_page(3, full_signals(0.95, "high"),
                       {"content_type": "mixed_document", "image_area_fraction": 0.40}),
            build_page(4, full_signals(0.88), IMAGE_PAGE),
            build_page(5, full_signals(0.15), IMAGE_PAGE),
        ]
        result = document.aggregate_pdf_pages(pages)

        # Page 3 "screams AI" but is mostly text, so it is not even usable.
        assert pages[2]["pdf_scoring"]["usable"] is False
        assert pages[0]["pdf_scoring"]["usable"] is False

        assert result["worst_page"] == 4
        assert result["score"] == 0.88
        assert result["usable_pages"] == 3
        assert result["reliable_pages"] == 3
        assert [f["page"] for f in result["flagged_pages"]] == [4]
        assert result["mean_page_score"] == pytest.approx((0.10 + 0.88 + 0.15) / 3, abs=1e-4)

    def test_adding_clean_pages_never_lowers_the_document_score(self):
        base = [page_entry(1, 0.75, "medium")]
        base_score = document.aggregate_pdf_pages(base)["score"]
        padded = base + [page_entry(i, 0.02, "medium-high") for i in range(2, 21)]
        padded_result = document.aggregate_pdf_pages(padded)
        assert padded_result["score"] == base_score == 0.75
        assert padded_result["worst_page"] == 1
        assert padded_result["mean_page_score"] < 0.10

    def test_worst_page_is_argmax_not_last_or_first(self):
        pages = [page_entry(i, s, "medium") for i, s in enumerate([0.3, 0.7, 0.9, 0.6, 0.2], start=1)]
        assert document.aggregate_pdf_pages(pages)["worst_page"] == 3

    def test_worst_page_tie_keeps_first_occurrence(self):
        pages = [page_entry(1, 0.5, "medium"), page_entry(2, 0.5, "medium-high")]
        result = document.aggregate_pdf_pages(pages)
        assert result["worst_page"] == 1
        assert result["confidence"] == "medium"

    def test_document_confidence_is_that_of_the_worst_page(self):
        pages = [
            page_entry(1, 0.20, "medium-high"),
            page_entry(2, 0.55, "low-medium"),
        ]
        result = document.aggregate_pdf_pages(pages)
        assert result["worst_page"] == 2
        assert result["confidence"] == "low-medium"

    def test_low_confidence_page_with_higher_score_does_not_become_worst(self):
        pages = [
            page_entry(1, 0.35, "medium"),
            page_entry(2, 0.99, "low"),  # usable but not reliable
        ]
        result = document.aggregate_pdf_pages(pages)
        assert result["worst_page"] == 1
        assert result["score"] == 0.35
        assert result["usable_pages"] == 2
        assert result["reliable_pages"] == 1
        assert result["flagged_pages"] == []
        # ...but the diagnostics still see it.
        assert result["max_page_score"] == 0.35  # max over *reliable* pages
        assert result["mean_page_score"] > 0.35  # mean includes the low-confidence page

    def test_unusable_pages_are_excluded_from_everything(self):
        pages = [
            page_entry(1, 0.30, "medium"),
            page_entry(2, 0.99, "medium", usable=False),
        ]
        result = document.aggregate_pdf_pages(pages)
        assert result["usable_pages"] == 1
        assert result["worst_page"] == 1
        assert result["mean_page_score"] == 0.3

    def test_worst_page_driven_by_embedded_image_is_attributed(self):
        pages = [
            page_entry(1, 0.10, "medium-high"),
            page_entry(2, 0.86, "medium-high", content_type="native_document",
                       source="embedded_image", image_index=3),
        ]
        result = document.aggregate_pdf_pages(pages)
        assert result["worst_page"] == 2
        assert result["worst_evidence_source"] == "embedded_image"
        assert result["worst_image_index"] == 3
        assert result["flagged_pages"][0]["risk_source"] == "embedded_image"
        assert result["flagged_pages"][0]["image_index"] == 3
        assert result["flagged_pages"][0]["content_type"] == "native_document"

    def test_mean_page_score_weights_by_evidence_coverage(self):
        pages = [
            page_entry(1, 1.0, "medium", evidence_coverage=0.75),
            page_entry(2, 0.0, "medium", evidence_coverage=0.25),
        ]
        result = document.aggregate_pdf_pages(pages)
        assert result["mean_page_score"] == pytest.approx(0.75)

    def test_mean_page_score_floors_zero_coverage_at_005(self):
        pages = [
            page_entry(1, 1.0, "medium", evidence_coverage=0.0),   # -> 0.05
            page_entry(2, 0.0, "medium", evidence_coverage=0.95),
        ]
        result = document.aggregate_pdf_pages(pages)
        assert result["mean_page_score"] == pytest.approx(0.05)

    def test_accepts_any_iterable(self):
        gen = (page_entry(i, 0.1 * i, "medium") for i in range(1, 4))
        result = document.aggregate_pdf_pages(gen)
        assert result["worst_page"] == 3
        assert result["usable_pages"] == 3

    def test_page_numbers_are_cast_to_int(self):
        pages = [page_entry("2", 0.5, "medium")]
        result = document.aggregate_pdf_pages(pages)
        assert result["worst_page"] == 2
        assert result["flagged_pages"][0]["page"] == 2

    def test_result_shape_for_reliable_document(self):
        result = document.aggregate_pdf_pages([page_entry(1, 0.5, "medium")])
        assert set(result) == {
            "score",
            "confidence",
            "usable_pages",
            "reliable_pages",
            "worst_page",
            "max_page_score",
            "mean_page_score",
            "flagged_pages",
            "aggregation",
            "worst_evidence_source",
            "worst_image_index",
            "note",
        }


class TestAggregatePdfPagesFlaggedThreshold:
    """The flagged-page cutoff is ``score >= 0.40`` over *reliable* pages."""

    def test_page_scoring_exactly_040_is_flagged(self):
        pages = [
            page_entry(1, 0.10, "medium"),
            page_entry(2, 0.40, "medium"),
            page_entry(3, 0.20, "medium"),
        ]
        result = document.aggregate_pdf_pages(pages)
        assert result["worst_page"] == 2
        assert result["score"] == 0.4
        assert [f["page"] for f in result["flagged_pages"]] == [2]
        assert result["flagged_pages"][0]["score"] == 0.4

    def test_page_scoring_exactly_040_is_flagged_end_to_end(self):
        """Drive 0.40 through the real scoring chain rather than injecting it:
        four medium signals at 0.40 on a fully-raster page -> score_pdf_page
        -> combine_page_evidence -> aggregate_pdf_pages."""
        pages = [
            build_page(1, full_signals(0.10), IMAGE_PAGE),
            build_page(2, full_signals(0.40), IMAGE_PAGE),
            build_page(3, document.not_applicable_signals(NATIVE_PAGE), NATIVE_PAGE),
        ]
        assert pages[1]["page_render_scoring"]["score"] == 0.4
        assert pages[1]["pdf_scoring"]["score"] == 0.4

        result = document.aggregate_pdf_pages(pages)
        assert result["worst_page"] == 2
        assert result["score"] == 0.4
        assert result["flagged_pages"] == [{
            "page": 2,
            "score": 0.4,
            "confidence": "medium-high",
            "content_type": "scanned_or_image_document",
            "risk_source": "page_render",
            "image_index": None,
        }]

    def test_embedded_image_scoring_exactly_040_is_flagged_end_to_end(self):
        page = build_page(
            1,
            document.not_applicable_signals(NATIVE_PAGE),
            NATIVE_PAGE,
            embedded=[(full_signals(0.40), EMBEDDED_PROFILE)],
        )
        assert page["pdf_scoring"]["score"] == 0.4
        assert page["pdf_scoring"]["source"] == "embedded_image"
        result = document.aggregate_pdf_pages([page])
        assert [f["page"] for f in result["flagged_pages"]] == [1]
        assert result["flagged_pages"][0]["risk_source"] == "embedded_image"
        assert result["flagged_pages"][0]["image_index"] == 1

    @pytest.mark.parametrize("score", [0.3999, 0.39, 0.3, 0.0])
    def test_pages_just_below_040_are_not_flagged(self, score):
        pages = [page_entry(1, score, "medium")]
        result = document.aggregate_pdf_pages(pages)
        assert result["worst_page"] == 1  # still the worst (only) page...
        assert result["score"] == score
        assert result["flagged_pages"] == []  # ...but not flagged

    @pytest.mark.parametrize("score", [0.4, 0.4001, 0.41, 0.5, 1.0])
    def test_pages_at_or_above_040_are_flagged(self, score):
        result = document.aggregate_pdf_pages([page_entry(1, score, "medium")])
        assert [f["page"] for f in result["flagged_pages"]] == [1]

    def test_threshold_compares_the_unrounded_page_score(self):
        # KNOWN QUIRK: aggregate_pdf_pages reports round(score, 4) but applies
        # the >= 0.40 test to the *raw* value. A raw 0.39995 therefore displays
        # as 0.4 yet is not flagged. In the real pipeline this cannot occur
        # because combine_page_evidence has already rounded to 4 decimals
        # (see the end-to-end tests above); this test pins the raw behaviour.
        result = document.aggregate_pdf_pages([page_entry(1, 0.39995, "medium")])
        assert result["score"] == 0.4
        assert result["max_page_score"] == 0.4
        assert result["flagged_pages"] == []

    def test_039994_stays_below_and_is_not_flagged(self):
        result = document.aggregate_pdf_pages([page_entry(1, 0.39994, "medium")])
        assert result["score"] == 0.3999
        assert result["flagged_pages"] == []

    def test_threshold_is_applied_only_to_reliable_pages(self):
        pages = [
            page_entry(1, 0.40, "low"),         # usable but not reliable
            page_entry(2, 0.40, "low-medium"),  # reliable boundary
        ]
        result = document.aggregate_pdf_pages(pages)
        assert [f["page"] for f in result["flagged_pages"]] == [2]

    def test_multiple_flagged_pages_sorted_by_score_descending(self):
        pages = [
            page_entry(1, 0.45, "medium"),
            page_entry(2, 0.10, "medium"),
            page_entry(3, 0.95, "medium-high"),
            page_entry(4, 0.40, "low-medium"),
            page_entry(5, 0.70, "medium"),
        ]
        result = document.aggregate_pdf_pages(pages)
        assert [f["page"] for f in result["flagged_pages"]] == [3, 5, 1, 4]
        assert [f["score"] for f in result["flagged_pages"]] == [0.95, 0.7, 0.45, 0.4]
        assert result["worst_page"] == 3
        assert result["score"] == 0.95

    def test_flagged_page_entry_shape(self):
        result = document.aggregate_pdf_pages([page_entry(9, 0.5, "medium", content_type="mixed_document")])
        assert result["flagged_pages"] == [{
            "page": 9,
            "score": 0.5,
            "confidence": "medium",
            "content_type": "mixed_document",
            "risk_source": "page_render",
            "image_index": None,
        }]

    def test_flagged_page_missing_content_type_defaults_to_unknown(self):
        page = page_entry(1, 0.5, "medium")
        del page["content_type"]
        result = document.aggregate_pdf_pages([page])
        assert result["flagged_pages"][0]["content_type"] == "unknown"


# ---------------------------------------------------------------------------
# classify_pdf
# ---------------------------------------------------------------------------


class TestClassifyPdf:
    @pytest.mark.parametrize(
        "score, expected",
        [
            (0.0, "low_ai_evidence"),
            (0.2999, "low_ai_evidence"),
            (0.30, "possible_ai_evidence"),
            (0.40, "possible_ai_evidence"),
            (0.4999, "possible_ai_evidence"),
            (0.50, "suspicious_raster_content"),
            (0.6999, "suspicious_raster_content"),
            (0.70, "likely_ai_generated"),
            (1.0, "likely_ai_generated"),
        ],
    )
    def test_score_bands(self, score, expected):
        classification, description = document.classify_pdf(score, "medium")
        assert classification == expected
        assert isinstance(description, str) and description

    def test_band_descriptions(self):
        assert document.classify_pdf(0.1, "medium")[1] == "Low AI-generated image evidence detected."
        assert document.classify_pdf(0.4, "medium")[1] == "Possible AI-generated image evidence detected."
        assert document.classify_pdf(0.6, "medium")[1] == "Moderate AI-generated image evidence detected."
        assert document.classify_pdf(0.9, "medium")[1] == "Strong AI-generated image evidence detected."

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_insufficient_confidence_overrides_score(self, score):
        classification, description = document.classify_pdf(score, "insufficient")
        assert classification == "insufficient_visual_evidence"
        assert "Not enough reliable raster evidence" in description

    @pytest.mark.parametrize("confidence", ["low", "insufficient", "medium-high"])
    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_metadata_hit_overrides_everything(self, score, confidence):
        classification, description = document.classify_pdf(score, confidence, metadata_hit=True)
        assert classification == "likely_ai_generated"
        assert "provenance" in description

    @pytest.mark.parametrize("confidence", ["low", "low-medium", "medium", "medium-high", "high", "anything"])
    def test_non_insufficient_confidence_does_not_change_band(self, confidence):
        assert document.classify_pdf(0.6, confidence)[0] == "suspicious_raster_content"


# ---------------------------------------------------------------------------
# End-to-end composition of the pure functions
# ---------------------------------------------------------------------------


class TestStrongBandGuard:
    def test_low_medium_page_at_its_maximum_score_is_not_likely(self):
        # No reliability shrink any more: two weak but agreeing signals reach
        # 1.0 on a low-medium page (frequency low-medium 0.175 + noise low
        # 0.20 = 0.375). Only the tier guard keeps it out of "likely".
        signals = {"frequency": sig(1.0, "low-medium"), "noise": sig(1.0, "low")}
        page = document.score_pdf_page(signals, IMAGE_PAGE)
        assert page["confidence"] == "low-medium"
        assert page["score"] == 1.0
        assert document.classify_pdf(page["score"], page["confidence"])[0] == "suspicious_raster_content"


class TestEndToEndComposition:
    def test_tiny_suspicious_embedded_image_on_text_page_drives_document(self):
        """A 1%-coverage, 400x300 AI-looking raster on an otherwise native
        text page must set the document score, even though the page render
        itself is not scorable."""
        tiny_but_real = {**EMBEDDED_PROFILE, "page_coverage": 0.01}
        pages = [
            build_page(1, document.not_applicable_signals(NATIVE_PAGE), NATIVE_PAGE),
            build_page(
                2,
                document.not_applicable_signals(NATIVE_PAGE),
                NATIVE_PAGE,
                embedded=[(full_signals(0.05), EMBEDDED_PROFILE), (full_signals(0.92, "high"), tiny_but_real)],
            ),
            build_page(3, full_signals(0.12), IMAGE_PAGE),
        ]
        result = document.aggregate_pdf_pages(pages)
        assert result["worst_page"] == 2
        assert result["score"] == 0.92
        assert result["worst_evidence_source"] == "embedded_image"
        assert result["worst_image_index"] == 2
        assert [f["page"] for f in result["flagged_pages"]] == [2]
        assert document.classify_pdf(result["score"], result["confidence"])[0] == "likely_ai_generated"

    def test_icon_sized_embedded_images_cannot_trigger_document(self):
        icon = {"image_index": 1, "width_px": 48, "height_px": 48, "pixel_count": 2304, "page_coverage": 0.002}
        pages = [
            build_page(
                1,
                document.not_applicable_signals(NATIVE_PAGE),
                NATIVE_PAGE,
                embedded=[(full_signals(1.0, "high"), icon)],
            ),
        ]
        result = document.aggregate_pdf_pages(pages)
        assert result["usable_pages"] == 0
        assert result["worst_page"] is None
        assert document.classify_pdf(result["score"], result["confidence"])[0] == "insufficient_visual_evidence"

    def test_all_native_document_is_insufficient(self):
        pages = [build_page(i, document.not_applicable_signals(NATIVE_PAGE), NATIVE_PAGE) for i in range(1, 6)]
        result = document.aggregate_pdf_pages(pages)
        assert result["confidence"] == "insufficient"
        assert result["usable_pages"] == 0
        assert document.classify_pdf(result["score"], result["confidence"])[0] == "insufficient_visual_evidence"

    def test_noise_only_evidence_on_every_page_is_capped_consistently(self):
        # Pages where only the noise signal is valid: each is scored like a
        # single-signal upload (capped at 0.60), and the worst page still wins.
        pages = [
            build_page(1, {"noise": sig(0.30)}, IMAGE_PAGE),
            build_page(2, {"noise": sig(1.00)}, IMAGE_PAGE),
        ]
        result = document.aggregate_pdf_pages(pages)
        assert result["worst_page"] == 2
        assert result["score"] == CAP
        assert result["confidence"] == "medium"
        assert [f["page"] for f in result["flagged_pages"]] == [2]
        assert pages[0]["pdf_scoring"]["score"] == pytest.approx(0.30, abs=1e-4)
        # Single-signal evidence cannot reach the strongest band.
        assert document.classify_pdf(result["score"], result["confidence"])[0] == "suspicious_raster_content"

    def test_mixed_page_with_dominant_image_is_scored_with_reduced_signal_set(self):
        profile = {"content_type": "mixed_document", "image_area_fraction": 0.75}
        page = build_page(1, full_signals(0.9), profile)
        render = page["page_render_scoring"]
        assert used_signals(render) == {"noise", "gradients"}
        assert render["effective_weight_sum"] == pytest.approx(0.65)
        assert render["confidence"] == "medium-high"
        assert render["score"] == pytest.approx(0.9)
        assert page["pdf_scoring"]["usable"] is True
        result = document.aggregate_pdf_pages([page])
        assert result["score"] == 0.9
        assert result["flagged_pages"][0]["content_type"] == "mixed_document"

    def test_metadata_hit_classifies_even_when_visual_evidence_is_clean(self):
        pages = [build_page(1, full_signals(0.05), IMAGE_PAGE)]
        aggregate = document.aggregate_pdf_pages(pages)
        meta = document.analyze_pdf_metadata({"Creator": "Midjourney"})
        hit = meta["confidence"] == "high"
        assert hit
        classification, _ = document.classify_pdf(aggregate["score"], aggregate["confidence"], metadata_hit=hit)
        assert classification == "likely_ai_generated"
