"""Standalone verification suite for the PDF / embedded-image scoring logic
in ``app/analyzers/document.py``.

Uses only the Python standard library (``unittest``) plus ``numpy``, which is
a dependency of the module under test. No pytest, no PDF rendering, no image
analyzers, no FastAPI.

Run from the ``main/`` directory, any of:

    python -m app.analyzers.verify_document
    python -m unittest app.analyzers.verify_document -v
    python app/analyzers/verify_document.py

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
* ``classify_pdf``                -- classification band boundaries, tier cap
* Gap closures: whole-token generator matching shared with metadata.py,
  non-finite signal scores, page-type allow-list, flat/gradient-fill gate on
  both page and embedded paths, strong band needs solid confidence
* End-to-end composition of the above for realistic multi-page documents
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from typing import Iterable

# --- import shim -----------------------------------------------------------
# Works when run as a module (``python -m app.analyzers.verify_document``) and
# when executed directly as a script (``python app/analyzers/verify_document.py``),
# in which case ``app`` is not yet importable and must be added to sys.path.
try:
    from app.analyzers import document
except ImportError:  # executed as a script from an arbitrary cwd
    _MAIN_DIR = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
    )
    if _MAIN_DIR not in sys.path:
        sys.path.insert(0, _MAIN_DIR)
    from app.analyzers import document

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

SIGNAL_NAMES = ("frequency", "noise", "gradients", "lighting")
RELIABLE_CONFIDENCES = ("low-medium", "medium", "medium-high", "high")

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

# Reliability shrink is ews / 0.55 (clipped to 1). Noise alone at medium
# confidence has ews == 0.45, so its score is multiplied by 0.45 / 0.55.
NOISE_ONLY_RELIABILITY = 0.45 / 0.55


def sig(score, confidence: str = "medium", note: str = "signal computed normally") -> dict:
    """Build a single analyzer result dict."""
    return {"score": score, "confidence": confidence, "note": note}


def full_signals(score: float, confidence: str = "medium") -> dict:
    """All four visual signals at the same score/confidence."""
    return {name: sig(score, confidence) for name in SIGNAL_NAMES}


def page_entry(
    page,
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


def used_signals(result: dict) -> set:
    return {name for name, c in result["contributions"].items() if c.get("used")}


def render_stub(score, confidence="medium", usable=True, coverage=1.0) -> dict:
    """Minimal stand-in for a ``score_pdf_page`` result."""
    return {
        "score": score,
        "confidence": confidence,
        "usable": usable,
        "evidence_coverage": coverage,
    }


class DocumentTestCase(unittest.TestCase):
    """Base class adding a tolerance-based float assertion."""

    def assertClose(self, actual, expected, tol=1e-6, msg=None):
        self.assertTrue(
            abs(float(actual) - float(expected)) <= tol,
            msg or f"{actual!r} != {expected!r} within {tol}",
        )


# ---------------------------------------------------------------------------
# Module-level constants -- guard against silent retuning
# ---------------------------------------------------------------------------


class TestConstants(DocumentTestCase):
    def test_pdf_scan_weights_sum_to_one(self):
        self.assertClose(sum(document.PDF_SCAN_WEIGHTS.values()), 1.0)

    def test_pdf_scan_weights_cover_exactly_the_four_visual_signals(self):
        self.assertEqual(set(document.PDF_SCAN_WEIGHTS), set(SIGNAL_NAMES))

    def test_noise_is_the_dominant_pdf_signal(self):
        w = document.PDF_SCAN_WEIGHTS
        self.assertTrue(w["noise"] > w["gradients"] > w["lighting"] > w["frequency"])

    def test_confidence_factors_are_monotonic(self):
        f = document.CONFIDENCE_FACTOR
        self.assertEqual(f["high"], 1.0)
        self.assertEqual(f["medium"], 1.0)
        self.assertTrue(0 < f["low"] < f["low-medium"] < f["medium"])

    def test_noise_alone_at_full_confidence_cannot_reach_full_reliability(self):
        # Documented intent: roughly noise + one other useful signal (0.55)
        # is needed before a page may express its full score.
        self.assertLess(document.PDF_SCAN_WEIGHTS["noise"], 0.55)


# ---------------------------------------------------------------------------
# analyze_pdf_metadata
# ---------------------------------------------------------------------------


class TestAnalyzePdfMetadata(DocumentTestCase):
    def test_no_fields_is_neutral_low_confidence(self):
        result = document.analyze_pdf_metadata({})
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["confidence"], "low")
        self.assertEqual(result["generator_references"], [])
        self.assertEqual(result["pdf_fields"], {})
        self.assertIn("absence is not proof", result["note"])

    def test_generic_producers_are_not_ai_evidence(self):
        producers = [
            "Adobe Acrobat Pro DC 23.1",
            "macOS Version 14.2 (Build 23C64) Quartz PDFContext",
            "Skia/PDF m120",  # Chrome
            "wkhtmltopdf 0.12.6",
            "OpenText Exstream",
            "Microsoft: Print To PDF",
            "LibreOffice 7.6",
        ]
        for producer in producers:
            with self.subTest(producer=producer):
                result = document.analyze_pdf_metadata({"Producer": producer, "Creator": "Word"})
                self.assertEqual(result["score"], 0.0)
                self.assertEqual(result["confidence"], "low")
                self.assertEqual(result["generator_references"], [])

    def test_every_listed_generator_term_is_detected(self):
        for term in document.AI_GENERATOR_TERMS:
            with self.subTest(term=term):
                result = document.analyze_pdf_metadata({"Creator": f"made with {term}"})
                self.assertEqual(result["score"], 0.98)
                self.assertEqual(result["confidence"], "high")
                self.assertIn(term, result["generator_references"])

    def test_detection_is_case_insensitive(self):
        result = document.analyze_pdf_metadata({"Author": "MidJourney Bot"})
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["generator_references"], ["midjourney"])

    def test_detection_considers_field_names_and_values(self):
        # The key/value pair is flattened to "key: value" before matching, so
        # a hit in any field is enough.
        result = document.analyze_pdf_metadata({"Keywords": "draft", "Subject": "ComfyUI export"})
        self.assertEqual(result["generator_references"], ["comfyui"])

    def test_multiple_hits_are_sorted_and_deduplicated(self):
        result = document.analyze_pdf_metadata({
            "Creator": "Stable Diffusion via ComfyUI",
            "Producer": "stable diffusion",  # duplicate term, different case
        })
        self.assertEqual(result["generator_references"], ["comfyui", "stable diffusion"])

    def test_overlapping_terms_both_reported(self):
        result = document.analyze_pdf_metadata({"Producer": "ChatGPT (OpenAI)"})
        self.assertEqual(result["generator_references"], ["chatgpt", "openai"])

    def test_terms_must_match_as_whole_tokens(self):
        # Bare substring matching produced 0.98-confidence AI provenance for
        # ordinary documents. Each of these must now be neutral.
        false_positives = [
            {"Title": "Magnetic flux density measurements in superconductors"},
            {"Title": "Gastroesophageal reflux disease - patient report"},
            {"Producer": "InfluxData PDF Export 2.1"},
            {"Title": "OpenAir Festival 2024 - ticket"},
            {"Author": "Dallesandro, Maria"},
            {"Subject": "Recrafted kitchen cabinets quote"},
            {"Keywords": "ideogrammatic writing systems"},
            {"Creator": "xmidjourney"},
            {"Creator": "midjourneyx"},
            {"Creator": "2flux.1"},
            {"Creator": "influx.1"},
            {"Title": "Magnetic flux density measurements"},
        ]
        for fields in false_positives:
            with self.subTest(fields=fields):
                result = document.analyze_pdf_metadata(fields)
                self.assertEqual(result["generator_references"], [])
                self.assertEqual(result["confidence"], "low")
                self.assertEqual(result["score"], 0.0)

    def test_tokens_still_match_with_punctuation_digits_and_boundaries(self):
        # Real generator strings keep matching: version suffixes, punctuation,
        # brackets, start/end of string, multi-word terms.
        true_positives = [
            ({"Creator": "FLUX.1 [dev]"}, ["flux.1"]),
            ({"Creator": "black-forest-labs/FLUX.1-schnell"}, ["flux.1"]),
            ({"Creator": "flux1-dev.safetensors"}, ["flux1"]),
            ({"Producer": "dalle3"}, ["dalle3"]),
            ({"Producer": "DALL·E 3"}, ["dall·e"]),
            ({"Producer": "DALL-E 2"}, ["dall-e"]),
            ({"Producer": "ChatGPT (OpenAI)"}, ["chatgpt", "openai"]),
            ({"Creator": "Midjourney v6.1"}, ["midjourney"]),
            ({"Creator": "Adobe Firefly"}, ["adobe firefly"]),
            ({"Creator": "leonardo.ai"}, ["leonardo.ai"]),
            ({"Creator": "made with Stable Diffusion, via ComfyUI"}, ["comfyui", "stable diffusion"]),
            ({"Creator": "nano banana"}, ["nano banana"]),
            ({"Creator": "InvokeAI"}, ["invokeai"]),
        ]
        for fields, expected in true_positives:
            with self.subTest(fields=fields):
                result = document.analyze_pdf_metadata(fields)
                self.assertEqual(result["generator_references"], expected)
                self.assertEqual(result["confidence"], "high")

    def test_find_generator_references_helper(self):
        self.assertEqual(document.find_generator_references("no tools here"), [])
        self.assertEqual(document.find_generator_references("MIDJOURNEY and midjourney"), ["midjourney"])
        self.assertEqual(document.find_generator_references("reflux"), [])
        self.assertEqual(document.find_generator_references(12345), [])  # non-string tolerated

    def test_generator_list_is_shared_with_image_metadata_analyzer(self):
        # Both provenance paths (PDF Info dict here, PNG/EXIF + legacy PDF
        # check in metadata.py) must use the same terms and the same matcher,
        # otherwise the same document gets different verdicts per path.
        from app.analyzers import metadata
        self.assertIs(metadata.KNOWN_GENERATOR_SOFTWARE, document.AI_GENERATOR_TERMS)
        for fields in ({"Title": "reflux study"}, {"Producer": "InfluxData"}, {"Author": "Dallesandro"}):
            with self.subTest(fields=fields):
                self.assertEqual(metadata.analyze_pdf_document(fields)["confidence"], "low")
                self.assertEqual(document.analyze_pdf_metadata(fields)["confidence"], "low")
        for fields in ({"Creator": "Midjourney"}, {"Creator": "InvokeAI"}, {"Creator": "ChatGPT"}):
            with self.subTest(fields=fields):
                self.assertEqual(metadata.analyze_pdf_document(fields)["confidence"], "high")
                self.assertEqual(document.analyze_pdf_metadata(fields)["confidence"], "high")

    def test_original_fields_are_echoed_back(self):
        fields = {"Title": "Invoice", "Producer": "Adobe Firefly"}
        result = document.analyze_pdf_metadata(fields)
        self.assertIs(result["pdf_fields"], fields)
        self.assertEqual(result["generator_references"], ["adobe firefly"])

    def test_common_words_are_not_generator_terms(self):
        # "flux", "dalle" and "firefly" are ordinary words in English/French/
        # Italian; only the product forms are provenance.
        for word in ("flux", "dalle", "firefly"):
            self.assertNotIn(word, document.AI_GENERATOR_TERMS)
        for fields in (
            {"Title": "Flux de trésorerie 2024"},          # cash-flow statement (FR)
            {"Title": "Pose de dalles en terrasse"},        # tiling quote (FR)
            {"Subject": "Apertura dalle ore 9 alle 18"},    # opening hours (IT)
            {"Title": "Firefly Aerospace launch manifest"},
            {"Producer": "Firefly"},
        ):
            with self.subTest(fields=fields):
                self.assertEqual(document.analyze_pdf_metadata(fields)["generator_references"], [])


# ---------------------------------------------------------------------------
# _signal_available (exercised through score_pdf_page contributions)
# ---------------------------------------------------------------------------


class TestSignalAvailabilityGating(DocumentTestCase):
    def test_all_signals_used_on_fully_raster_page(self):
        result = document.score_pdf_page(full_signals(0.5), IMAGE_PAGE)
        self.assertEqual(used_signals(result), set(SIGNAL_NAMES))

    def test_none_score_is_unavailable(self):
        signals = full_signals(0.9)
        signals["noise"] = {"score": None, "confidence": "unavailable", "note": "x"}
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        self.assertNotIn("noise", used_signals(result))
        self.assertIs(result["contributions"]["noise"]["used"], False)
        self.assertEqual(result["contributions"]["noise"]["effective_weight"], 0.0)

    def test_missing_signal_key_is_unavailable(self):
        signals = full_signals(0.9)
        del signals["gradients"]
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        self.assertNotIn("gradients", used_signals(result))

    def test_signal_value_of_none_is_unavailable(self):
        # ``signals.get(name) or {}`` must tolerate an explicit None entry.
        signals = full_signals(0.9)
        signals["lighting"] = None
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        self.assertNotIn("lighting", used_signals(result))

    def test_unavailable_note_phrases_disable_a_signal(self):
        for phrase in document._UNAVAILABLE_PHRASES:
            with self.subTest(phrase=phrase):
                signals = full_signals(0.9)
                signals["noise"] = sig(0.9, "high", note=f"Analyzer says: {phrase.upper()} flat regions")
                result = document.score_pdf_page(signals, IMAGE_PAGE)
                self.assertNotIn("noise", used_signals(result))

    def test_unavailable_phrase_matching_is_case_insensitive(self):
        signals = full_signals(0.9)
        signals["gradients"] = sig(0.9, note="NOT ENOUGH skin pixels")
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        self.assertNotIn("gradients", used_signals(result))

    def test_benign_note_does_not_disable_signal(self):
        signals = full_signals(0.9)
        signals["noise"] = sig(0.9, note="noise floor estimated from 42 flat patches")
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        self.assertIn("noise", used_signals(result))

    def test_non_finite_or_non_numeric_scores_are_unavailable(self):
        # A NaN must never enter the weighted sum: it would propagate to the
        # page and document score and defeat every downstream threshold.
        for bad in (float("nan"), float("inf"), float("-inf"), "0.5", "abc", [], {}, True):
            with self.subTest(score=bad):
                signals = full_signals(0.2)
                signals["noise"] = {"score": bad, "confidence": "high", "note": "ok"}
                result = document.score_pdf_page(signals, IMAGE_PAGE)
                self.assertNotIn("noise", used_signals(result))
                # Remaining weights 0.05 + 0.35 + 0.15 = 0.55 -> exactly 0.2.
                self.assertClose(result["score"], 0.2)
                self.assertTrue(math.isfinite(result["score"]))
                emb = document.score_embedded_image(signals, EMBEDDED_PROFILE)
                self.assertNotIn("noise", used_signals(emb))
                self.assertClose(emb["score"], 0.2)

    def test_only_non_finite_signal_is_insufficient_not_nan(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(score=bad):
                page = document.score_pdf_page({"noise": sig(bad, "high")}, IMAGE_PAGE)
                emb = document.score_embedded_image({"noise": sig(bad, "high")}, EMBEDDED_PROFILE)
                for result in (page, emb):
                    self.assertEqual(result["confidence"], "insufficient")
                    self.assertEqual(result["score"], 0.0)
                    self.assertIs(result["usable"], False)

    def test_real_numbers_including_numpy_scalars_are_accepted(self):
        import numpy as np
        for value in (np.float32(0.75), np.float64(0.75), np.int64(1), 1, 0, 0.75):
            with self.subTest(value=value):
                result = document.score_pdf_page({"noise": sig(value), "gradients": sig(value)}, IMAGE_PAGE)
                self.assertClose(result["score"], float(value))

    def test_native_vector_sparse_pages_reject_all_signals(self):
        for profile in (NATIVE_PAGE, VECTOR_PAGE, SPARSE_PAGE):
            with self.subTest(content_type=profile["content_type"]):
                result = document.score_pdf_page(full_signals(1.0, "high"), profile)
                self.assertEqual(used_signals(result), set())
                self.assertIs(result["usable"], False)
                self.assertEqual(result["confidence"], "insufficient")
                self.assertEqual(result["score"], 0.0)

    def test_native_page_with_high_image_fraction_still_rejected(self):
        # content_type wins over image_area_fraction for native/vector/sparse.
        profile = {"content_type": "native_document", "image_area_fraction": 0.99}
        result = document.score_pdf_page(full_signals(1.0, "high"), profile)
        self.assertEqual(used_signals(result), set())

    def test_mixed_page_below_070_rejects_all_signals(self):
        for fraction in (0.0, 0.25, 0.50, 0.69, 0.6999):
            with self.subTest(fraction=fraction):
                profile = {"content_type": "mixed_document", "image_area_fraction": fraction}
                result = document.score_pdf_page(full_signals(1.0, "high"), profile)
                self.assertEqual(used_signals(result), set())
                self.assertEqual(result["confidence"], "insufficient")

    def test_mixed_page_between_070_and_080_allows_noise_and_gradients_only(self):
        for fraction in (0.70, 0.75, 0.7999):
            with self.subTest(fraction=fraction):
                profile = {"content_type": "mixed_document", "image_area_fraction": fraction}
                result = document.score_pdf_page(full_signals(1.0, "high"), profile)
                self.assertEqual(used_signals(result), {"noise", "gradients"})
                self.assertClose(result["effective_weight_sum"], 0.80)

    def test_mixed_page_at_or_above_080_allows_all_signals(self):
        for fraction in (0.80, 0.95, 1.0):
            with self.subTest(fraction=fraction):
                profile = {"content_type": "mixed_document", "image_area_fraction": fraction}
                result = document.score_pdf_page(full_signals(1.0, "high"), profile)
                self.assertEqual(used_signals(result), set(SIGNAL_NAMES))

    def test_image_page_below_080_drops_frequency_and_lighting(self):
        for fraction in (0.65, 0.70, 0.79):
            with self.subTest(fraction=fraction):
                profile = {"content_type": "scanned_or_image_document", "image_area_fraction": fraction}
                result = document.score_pdf_page(full_signals(1.0, "high"), profile)
                self.assertEqual(used_signals(result), {"noise", "gradients"})

    def test_unknown_or_missing_content_type_fails_closed(self):
        # Page-type gating is an allow-list (RASTER_PAGE_TYPES). An empty
        # profile, None, a typo, a case mismatch or a future type must not be
        # scored as a camera image, whatever image_area_fraction says.
        self.assertEqual(document.RASTER_PAGE_TYPES, {"scanned_or_image_document", "mixed_document"})
        for profile in (
            {},
            {"content_type": None, "image_area_fraction": 1.0},
            {"content_type": "unknown", "image_area_fraction": 1.0},
            {"content_type": "form_document", "image_area_fraction": 1.0},
            {"content_type": "SCANNED_OR_IMAGE_DOCUMENT", "image_area_fraction": 1.0},
            {"content_type": "photo", "image_area_fraction": 1.0},
        ):
            with self.subTest(profile=profile):
                result = document.score_pdf_page(full_signals(1.0, "high"), profile)
                self.assertEqual(used_signals(result), set())
                self.assertEqual(result["confidence"], "insufficient")
                self.assertIs(result["usable"], False)
        self.assertEqual(document.score_pdf_page(full_signals(1.0), {})["content_type"], "unknown")

    def test_flat_fill_page_raster_is_not_camera_evidence(self):
        # Few colours AND no edges = gradient/solid fill. Both are required.
        cases = [
            ({"unique_colors": 205, "edge_density": 0.0}, set(), "radial vignette"),
            ({"unique_colors": 1, "edge_density": 0.0}, set(), "solid block"),
            ({"unique_colors": 1023, "edge_density": 0.0099}, set(), "just inside both limits"),
            ({"unique_colors": 1024, "edge_density": 0.0}, set(SIGNAL_NAMES), "colour count at limit"),
            ({"unique_colors": 1, "edge_density": 0.01}, set(SIGNAL_NAMES), "edge density at limit"),
            ({"unique_colors": 248, "edge_density": 0.03}, set(SIGNAL_NAMES), "grayscale text scan"),
            ({"unique_colors": 26352, "edge_density": 0.0}, set(SIGNAL_NAMES), "heavily blurred colour photo"),
            ({"unique_colors": 84249, "edge_density": 0.24}, set(SIGNAL_NAMES), "photo"),
            ({}, set(SIGNAL_NAMES), "stats absent -> not gated (older callers)"),
            ({"unique_colors": None, "edge_density": 0.0}, set(SIGNAL_NAMES), "partial stats -> not gated"),
            ({"unique_colors": "abc", "edge_density": 0.0}, set(SIGNAL_NAMES), "garbage stats -> not gated"),
        ]
        for stats, expected_used, label in cases:
            with self.subTest(label=label):
                profile = {**IMAGE_PAGE, **stats}
                result = document.score_pdf_page(full_signals(1.0, "high"), profile)
                self.assertEqual(used_signals(result), expected_used)
                if not expected_used:
                    self.assertEqual(result["confidence"], "insufficient")

    def test_flat_fill_constants(self):
        self.assertEqual(document.FLAT_FILL_MAX_UNIQUE_COLORS, 1024)
        self.assertEqual(document.FLAT_FILL_MAX_EDGE_DENSITY, 0.01)

    def test_image_area_fraction_accepts_string_numbers(self):
        profile = {"content_type": "scanned_or_image_document", "image_area_fraction": "0.9"}
        result = document.score_pdf_page(full_signals(1.0, "high"), profile)
        self.assertEqual(used_signals(result), set(SIGNAL_NAMES))


# ---------------------------------------------------------------------------
# score_pdf_page
# ---------------------------------------------------------------------------


class TestScorePdfPage(DocumentTestCase):
    def test_uniform_signals_give_that_score_when_fully_reliable(self):
        for s in (0.0, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(score=s):
                result = document.score_pdf_page(full_signals(s), IMAGE_PAGE)
                self.assertClose(result["score"], s)
                self.assertClose(result["raw_normalized_score"], s)
                self.assertClose(result["effective_weight_sum"], 1.0)
                self.assertClose(result["evidence_coverage"], 1.0)
                self.assertEqual(result["confidence"], "medium-high")
                self.assertIs(result["usable"], True)

    def test_score_is_weighted_by_pdf_scan_weights(self):
        signals = {
            "frequency": sig(1.0),
            "noise": sig(0.0),
            "gradients": sig(1.0),
            "lighting": sig(0.0),
        }
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        expected = (0.05 * 1.0 + 0.45 * 0.0 + 0.35 * 1.0 + 0.15 * 0.0) / 1.0
        self.assertClose(result["score"], expected, tol=1e-4)
        self.assertClose(result["score"], 0.40, tol=1e-4)

    def test_high_and_medium_confidence_are_equivalent(self):
        r_high = document.score_pdf_page(full_signals(0.7, "high"), IMAGE_PAGE)
        r_med = document.score_pdf_page(full_signals(0.7, "medium"), IMAGE_PAGE)
        self.assertEqual(r_high["score"], r_med["score"])
        self.assertEqual(r_high["effective_weight_sum"], r_med["effective_weight_sum"])

    def test_low_medium_confidence_scales_weight_by_0_6(self):
        signals = full_signals(0.5)
        signals["noise"] = sig(0.5, "low-medium")
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        self.assertClose(result["contributions"]["noise"]["effective_weight"], 0.45 * 0.6)
        self.assertClose(result["effective_weight_sum"], 0.05 + 0.27 + 0.35 + 0.15)

    def test_low_confidence_scales_weight_by_0_2(self):
        signals = full_signals(0.5)
        signals["gradients"] = sig(0.5, "low")
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        self.assertClose(result["contributions"]["gradients"]["effective_weight"], 0.35 * 0.2)

    def test_unknown_confidence_string_contributes_zero_weight(self):
        # NOTE: unlike score_embedded_image, score_pdf_page still reports the
        # signal as ``used`` -- but with effective_weight 0 it cannot move the
        # score. This pins the current behaviour.
        signals = full_signals(0.2)
        signals["noise"] = sig(1.0, "bogus")
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        self.assertEqual(result["contributions"]["noise"]["effective_weight"], 0.0)
        # Remaining weights: 0.05 + 0.35 + 0.15 = 0.55 -> reliability exactly 1.
        self.assertClose(result["effective_weight_sum"], 0.55)
        self.assertClose(result["raw_normalized_score"], 0.2)
        self.assertClose(result["score"], 0.2)

    def test_only_unknown_confidence_signal_is_insufficient(self):
        result = document.score_pdf_page({"noise": sig(1.0, "bogus")}, IMAGE_PAGE)
        self.assertEqual(result["confidence"], "insufficient")
        self.assertIs(result["usable"], False)
        self.assertEqual(result["score"], 0.0)

    def test_missing_confidence_defaults_to_low(self):
        result = document.score_pdf_page({"noise": {"score": 1.0, "note": ""}}, IMAGE_PAGE)
        self.assertEqual(result["contributions"]["noise"]["confidence"], "low")
        self.assertClose(result["contributions"]["noise"]["effective_weight"], 0.09)

    def test_low_confidence_signals_cannot_create_a_high_page_score(self):
        # Every signal "screams AI" but at low confidence: ews = 0.20,
        # reliability = 0.20/0.55, so the score is strongly shrunk.
        result = document.score_pdf_page(full_signals(1.0, "low"), IMAGE_PAGE)
        self.assertClose(result["effective_weight_sum"], 0.20)
        self.assertClose(result["raw_normalized_score"], 1.0)
        self.assertClose(result["score"], 0.20 / 0.55, tol=1e-4)
        self.assertIs(result["usable"], True)
        # The unrounded sum 0.01+0.09+0.07+0.03 is 0.19999999999999998 in
        # IEEE-754; tiers are compared on the 4-decimal-rounded sum so this
        # lands on the nominal "low-medium" boundary, not below it.
        self.assertEqual(result["confidence"], "low-medium")

    def test_noise_only_is_shrunk_by_reliability(self):
        result = document.score_pdf_page({"noise": sig(1.0)}, IMAGE_PAGE)
        self.assertEqual(result["raw_normalized_score"], 1.0)
        self.assertClose(result["effective_weight_sum"], 0.45)
        self.assertClose(result["score"], NOISE_ONLY_RELIABILITY, tol=1e-4)
        self.assertEqual(result["score"], 0.8182)
        self.assertEqual(result["confidence"], "medium")
        self.assertIs(result["usable"], True)

    def test_reliability_saturates_at_055(self):
        # noise (0.45) + lighting (0.15) = 0.60 >= 0.55 -> no shrink.
        signals = {"noise": sig(0.8), "lighting": sig(0.8)}
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        self.assertClose(result["score"], 0.8)
        self.assertClose(result["raw_normalized_score"], 0.8)

    def test_reliability_exactly_at_055_is_not_shrunk(self):
        # frequency + gradients + lighting = 0.05 + 0.35 + 0.15 = 0.55
        signals = {"frequency": sig(0.6), "gradients": sig(0.6), "lighting": sig(0.6)}
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        self.assertClose(result["effective_weight_sum"], 0.55)
        self.assertClose(result["score"], 0.6)

    def test_confidence_tiers_by_effective_weight(self):
        cases = [
            (full_signals(0.5), 1.00, "medium-high"),
            ({"noise": sig(0.5), "lighting": sig(0.5), "frequency": sig(0.5)}, 0.65, "medium-high"),
            ({"noise": sig(0.5), "lighting": sig(0.5)}, 0.60, "medium"),
            ({"noise": sig(0.5)}, 0.45, "medium"),
            # 0.35 + 0.05 == 0.39999999999999997 in floating point; the
            # rounded comparison keeps this on the "medium" boundary.
            ({"gradients": sig(0.5), "frequency": sig(0.5)}, 0.40, "medium"),
            ({"gradients": sig(0.5)}, 0.35, "low-medium"),
            ({"noise": sig(0.5, "low-medium")}, 0.27, "low-medium"),
            ({"frequency": sig(0.5), "lighting": sig(0.5)}, 0.20, "low-medium"),
            ({"lighting": sig(0.5)}, 0.15, "low"),
            ({"noise": sig(0.5, "low")}, 0.09, "low"),
            ({"frequency": sig(0.5)}, 0.05, "low"),
        ]
        for signals, expected_ews, expected_conf in cases:
            with self.subTest(signals=sorted(signals), ews=expected_ews):
                result = document.score_pdf_page(signals, IMAGE_PAGE)
                self.assertClose(result["effective_weight_sum"], expected_ews)
                self.assertEqual(result["confidence"], expected_conf)

    def test_usable_requires_effective_weight_of_at_least_015(self):
        cases = [
            ({"lighting": sig(0.9)}, True),                 # 0.15 -> boundary, usable
            ({"lighting": sig(0.9, "low-medium")}, False),  # 0.09
            ({"noise": sig(0.9, "low")}, False),            # 0.09
            ({"frequency": sig(0.9), "noise": sig(0.9, "low")}, False),  # 0.14
            ({"gradients": sig(0.9, "low-medium")}, True),  # 0.21
        ]
        for signals, expected_usable in cases:
            with self.subTest(signals=sorted(signals), expected=expected_usable):
                result = document.score_pdf_page(signals, IMAGE_PAGE)
                self.assertIs(result["usable"], expected_usable)

    def test_scores_above_one_are_clipped(self):
        result = document.score_pdf_page({"noise": sig(5.0), "gradients": sig(5.0)}, IMAGE_PAGE)
        self.assertEqual(result["contributions"]["noise"]["score"], 1.0)
        self.assertEqual(result["raw_normalized_score"], 1.0)
        self.assertEqual(result["score"], 1.0)

    def test_scores_below_zero_are_clipped(self):
        result = document.score_pdf_page({"noise": sig(-3.0), "gradients": sig(-3.0)}, IMAGE_PAGE)
        self.assertEqual(result["contributions"]["noise"]["score"], 0.0)
        self.assertEqual(result["score"], 0.0)

    def test_insufficient_result_shape(self):
        result = document.score_pdf_page({}, IMAGE_PAGE)
        self.assertEqual(result, {
            "score": 0.0,
            "confidence": "insufficient",
            "usable": False,
            "evidence_coverage": 0.0,
            "content_type": "scanned_or_image_document",
            "contributions": {
                name: {
                    "used": False,
                    "reason": "not applicable or unavailable for this PDF page",
                    "effective_weight": 0.0,
                }
                for name in SIGNAL_NAMES
            },
            "note": "photo-forensic analysis is not applicable/reliable on this rendered PDF page",
        })

    def test_usable_result_contains_all_expected_keys(self):
        result = document.score_pdf_page(full_signals(0.3), IMAGE_PAGE)
        self.assertEqual(set(result), {
            "score",
            "raw_normalized_score",
            "confidence",
            "usable",
            "effective_weight_sum",
            "evidence_coverage",
            "content_type",
            "contributions",
            "note",
        })
        self.assertEqual(set(result["contributions"]), set(SIGNAL_NAMES))
        for name in SIGNAL_NAMES:
            c = result["contributions"][name]
            self.assertIs(c["used"], True)
            self.assertEqual(c["score"], 0.3)
            self.assertEqual(c["confidence"], "medium")
            self.assertEqual(c["effective_weight"], document.PDF_SCAN_WEIGHTS[name])

    def test_evidence_coverage_is_fraction_of_total_base_weight(self):
        result = document.score_pdf_page({"gradients": sig(0.5), "noise": sig(0.5, "low-medium")}, IMAGE_PAGE)
        self.assertClose(result["evidence_coverage"], 0.35 + 0.27)

    def test_scores_are_rounded_to_four_decimals(self):
        result = document.score_pdf_page({"noise": sig(1 / 3)}, IMAGE_PAGE)
        self.assertEqual(result["score"], round(result["score"], 4))
        self.assertEqual(result["raw_normalized_score"], 0.3333)

    def test_all_signals_at_040_produce_exactly_040(self):
        # Fixture used by the flagged-page threshold test further below.
        result = document.score_pdf_page(full_signals(0.40), IMAGE_PAGE)
        self.assertEqual(result["score"], 0.4)
        self.assertIs(result["usable"], True)
        self.assertEqual(result["confidence"], "medium-high")

    def test_input_dicts_are_not_mutated(self):
        signals = full_signals(0.5)
        profile = dict(IMAGE_PAGE)
        snapshot_signals = {k: dict(v) for k, v in signals.items()}
        snapshot_profile = dict(profile)
        document.score_pdf_page(signals, profile)
        self.assertEqual(signals, snapshot_signals)
        self.assertEqual(profile, snapshot_profile)


# ---------------------------------------------------------------------------
# score_embedded_image
# ---------------------------------------------------------------------------


class TestScoreEmbeddedImage(DocumentTestCase):
    def test_too_small_rasters_are_unusable(self):
        profiles = [
            {"width_px": 63, "height_px": 1000, "pixel_count": 63_000},
            {"width_px": 1000, "height_px": 63, "pixel_count": 63_000},
            {"width_px": 64, "height_px": 64, "pixel_count": 4096},
            {"width_px": 128, "height_px": 127, "pixel_count": 16_256},  # 16,256 < 16,384
            {"width_px": 0, "height_px": 0},
            {},
        ]
        for profile in profiles:
            with self.subTest(profile=profile):
                result = document.score_embedded_image(full_signals(1.0, "high"), profile)
                self.assertIs(result["usable"], False)
                self.assertEqual(result["score"], 0.0)
                self.assertEqual(result["confidence"], "insufficient")
                self.assertEqual(result["evidence_coverage"], 0.0)
                self.assertEqual(result["contributions"], {})
                self.assertIs(result["image_profile"], profile)
                self.assertIn("too small", result["note"])

    def test_flat_fill_embedded_raster_is_unusable_with_reason(self):
        for stats, label in [
            ({"unique_colors": 462, "edge_density": 0.0}, "linear gradient banner"),
            ({"unique_colors": 205, "edge_density": 0.0}, "radial vignette"),
            ({"unique_colors": 2, "edge_density": 0.002}, "two-tone block"),
        ]:
            with self.subTest(label=label):
                profile = {**EMBEDDED_PROFILE, **stats}
                result = document.score_embedded_image(full_signals(1.0, "high"), profile)
                self.assertIs(result["usable"], False)
                self.assertEqual(result["confidence"], "insufficient")
                self.assertEqual(result["score"], 0.0)
                self.assertIn("flat or gradient fill", result["note"])
                self.assertIn(str(stats["unique_colors"]), result["note"])
                self.assertIs(result["image_profile"], profile)

    def test_photo_like_embedded_raster_is_not_gated_by_stats(self):
        for stats, label in [
            ({"unique_colors": 38176, "edge_density": 0.0179}, "blurred AI photo"),
            ({"unique_colors": 256, "edge_density": 0.2379}, "grayscale photo"),
            ({"unique_colors": 118, "edge_density": 0.0785}, "128px scan thumbnail"),
            ({"unique_colors": 2, "edge_density": 0.0332}, "flat logo shapes with edges"),
            ({}, "stats absent"),
        ]:
            with self.subTest(label=label):
                profile = {**EMBEDDED_PROFILE, **stats}
                result = document.score_embedded_image(full_signals(0.9), profile)
                self.assertIs(result["usable"], True)
                self.assertClose(result["score"], 0.9)

    def test_size_gate_is_checked_before_flat_fill_gate(self):
        profile = {"width_px": 10, "height_px": 10, "unique_colors": 1, "edge_density": 0.0}
        result = document.score_embedded_image(full_signals(1.0, "high"), profile)
        self.assertIn("too small", result["note"])

    def test_flat_fill_cannot_drive_page_or_document_risk(self):
        # End-to-end: a vignette on a text page next to a clean photo page.
        vignette = {**EMBEDDED_PROFILE, "unique_colors": 205, "edge_density": 0.0}
        pages = [
            build_page(1, document.not_applicable_signals(NATIVE_PAGE), NATIVE_PAGE,
                       embedded=[(full_signals(1.0, "high"), vignette)]),
            build_page(2, full_signals(0.12), IMAGE_PAGE),
        ]
        self.assertIs(pages[0]["pdf_scoring"]["usable"], False)
        result = document.aggregate_pdf_pages(pages)
        self.assertEqual(result["worst_page"], 2)
        self.assertEqual(result["score"], 0.12)
        self.assertEqual(result["flagged_pages"], [])

    def test_minimum_acceptable_size(self):
        profile = {"width_px": 128, "height_px": 128, "pixel_count": 16_384}
        result = document.score_embedded_image(full_signals(0.5), profile)
        self.assertIs(result["usable"], True)
        self.assertClose(result["score"], 0.5)

    def test_pixel_count_defaults_to_width_times_height(self):
        profile = {"width_px": 200, "height_px": 100}  # 20,000 px, no pixel_count key
        result = document.score_embedded_image(full_signals(0.5), profile)
        self.assertIs(result["usable"], True)

    def test_explicit_pixel_count_overrides_dimensions(self):
        # The module trusts pixel_count when present (even if inconsistent).
        profile = {"width_px": 200, "height_px": 200, "pixel_count": 100}
        result = document.score_embedded_image(full_signals(0.5), profile)
        self.assertIs(result["usable"], False)
        self.assertIn("too small", result["note"])

    def test_page_coverage_never_dilutes_the_score(self):
        for coverage in (1e-6, 0.001, 0.01, 0.5, 1.0):
            with self.subTest(coverage=coverage):
                profile = {**EMBEDDED_PROFILE, "page_coverage": coverage}
                result = document.score_embedded_image(full_signals(0.9), profile)
                self.assertClose(result["score"], 0.9)
                self.assertEqual(result["confidence"], "medium-high")
                self.assertIs(result["usable"], True)

    def test_embedded_scoring_ignores_page_type_gating(self):
        # Embedded images are scored on their own pixels, so the *page* being
        # native text must not suppress them. score_embedded_image does not
        # even take a page profile -- assert that all four signals are used.
        result = document.score_embedded_image(full_signals(0.6), EMBEDDED_PROFILE)
        self.assertEqual(used_signals(result), set(SIGNAL_NAMES))

    def test_none_score_is_unavailable(self):
        signals = full_signals(0.9)
        signals["noise"] = {"score": None, "confidence": "high", "note": ""}
        result = document.score_embedded_image(signals, EMBEDDED_PROFILE)
        self.assertEqual(result["contributions"]["noise"], {
            "used": False,
            "reason": "unavailable or insufficient-confidence signal",
            "effective_weight": 0.0,
        })

    def test_unavailable_note_phrases_disable_a_signal(self):
        for phrase in document._UNAVAILABLE_PHRASES:
            with self.subTest(phrase=phrase):
                signals = full_signals(0.9)
                signals["gradients"] = sig(0.9, note=f"{phrase} skin")
                result = document.score_embedded_image(signals, EMBEDDED_PROFILE)
                self.assertNotIn("gradients", used_signals(result))

    def test_missing_and_none_signal_entries_are_tolerated(self):
        signals = {"noise": sig(0.5), "gradients": None}
        result = document.score_embedded_image(signals, EMBEDDED_PROFILE)
        self.assertEqual(used_signals(result), {"noise"})

    def test_unknown_confidence_is_marked_unused(self):
        signals = full_signals(0.2)
        signals["noise"] = sig(1.0, "definitely-not-a-tier")
        result = document.score_embedded_image(signals, EMBEDDED_PROFILE)
        self.assertIs(result["contributions"]["noise"]["used"], False)
        self.assertClose(result["effective_weight_sum"], 0.55)
        self.assertClose(result["score"], 0.2)

    def test_no_usable_signals_is_unusable_with_distinct_reason(self):
        result = document.score_embedded_image({}, EMBEDDED_PROFILE)
        self.assertIs(result["usable"], False)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["confidence"], "insufficient")
        self.assertEqual(result["note"], "embedded raster had no usable forensic signals")
        self.assertIs(result["image_profile"], EMBEDDED_PROFILE)

    def test_noise_only_is_shrunk_by_reliability(self):
        result = document.score_embedded_image({"noise": sig(1.0)}, EMBEDDED_PROFILE)
        self.assertEqual(result["raw_normalized_score"], 1.0)
        self.assertEqual(result["score"], 0.8182)
        self.assertEqual(result["confidence"], "medium")
        self.assertIs(result["usable"], True)

    def test_low_confidence_signals_cannot_create_a_high_score(self):
        result = document.score_embedded_image(full_signals(1.0, "low"), EMBEDDED_PROFILE)
        self.assertClose(result["effective_weight_sum"], 0.20)
        self.assertClose(result["score"], 0.20 / 0.55, tol=1e-4)
        self.assertIs(result["usable"], True)
        # Same float-accumulation case as the page path: rounded to 0.20.
        self.assertEqual(result["confidence"], "low-medium")

    def test_confidence_and_usable_tiers_match_page_scoring(self):
        cases = [
            (full_signals(0.5), "medium-high", True),
            ({"noise": sig(0.5), "lighting": sig(0.5)}, "medium", True),
            ({"gradients": sig(0.5)}, "low-medium", True),
            ({"lighting": sig(0.5)}, "low", True),     # 0.15 == usable threshold
            ({"noise": sig(0.5, "low")}, "low", False),  # 0.09
        ]
        for signals, expected_conf, expected_usable in cases:
            with self.subTest(signals=sorted(signals), conf=expected_conf):
                result = document.score_embedded_image(signals, EMBEDDED_PROFILE)
                self.assertEqual(result["confidence"], expected_conf)
                self.assertIs(result["usable"], expected_usable)

    def test_page_and_embedded_scoring_agree_on_identical_signals(self):
        # Both paths share PDF_SCAN_WEIGHTS / CONFIDENCE_FACTOR / reliability
        # rule; for a fully-raster page they must produce identical numbers.
        for score in (0.1, 0.4, 0.77):
            for conf in ("high", "medium", "low-medium", "low"):
                with self.subTest(score=score, confidence=conf):
                    signals = full_signals(score, conf)
                    page = document.score_pdf_page(signals, IMAGE_PAGE)
                    emb = document.score_embedded_image(signals, EMBEDDED_PROFILE)
                    self.assertEqual(page["score"], emb["score"])
                    self.assertEqual(page["confidence"], emb["confidence"])
                    self.assertEqual(page["usable"], emb["usable"])
                    self.assertEqual(page["effective_weight_sum"], emb["effective_weight_sum"])

    def test_tier_boundaries_use_rounded_weight_sum(self):
        """Regression: tier comparison must not be demoted by IEEE-754
        accumulation error in the base weights. Both scoring paths round the
        effective weight sum to 4 decimals before comparing."""
        cases = [
            # 0.01 + 0.09 + 0.07 + 0.03 -> 0.19999999999999998 raw
            (full_signals(0.5, "low"), 0.19999999999999998, "low-medium"),
            # 0.35 + 0.05 -> 0.39999999999999997 raw
            ({"gradients": sig(0.5), "frequency": sig(0.5)}, 0.39999999999999997, "medium"),
            # 0.05 + 0.35 + 0.15 -> 0.5499999999999999 raw (no tier edge, but
            # confirms the reported sum is the rounded one)
            ({"frequency": sig(0.5), "gradients": sig(0.5), "lighting": sig(0.5)}, 0.5499999999999999, "medium"),
        ]
        for signals, raw_sum, expected_conf in cases:
            with self.subTest(signals=sorted(signals), raw_sum=raw_sum):
                # Sanity-check the premise by replicating the module's
                # accumulation (0.0 += weight * factor, in PDF_SCAN_WEIGHTS
                # order): the raw float sum really is below the nominal
                # boundary.
                naive = 0.0
                for name, base_weight in document.PDF_SCAN_WEIGHTS.items():
                    if name in signals:
                        naive += base_weight * document.CONFIDENCE_FACTOR[signals[name]["confidence"]]
                self.assertEqual(naive, raw_sum)
                self.assertLess(naive, round(naive, 4))

                for result in (
                    document.score_pdf_page(signals, IMAGE_PAGE),
                    document.score_embedded_image(signals, EMBEDDED_PROFILE),
                ):
                    self.assertEqual(result["effective_weight_sum"], round(raw_sum, 4))
                    self.assertEqual(result["confidence"], expected_conf)

    def test_scores_are_clipped_to_unit_interval(self):
        hi = document.score_embedded_image({"noise": sig(9.0), "gradients": sig(9.0)}, EMBEDDED_PROFILE)
        lo = document.score_embedded_image({"noise": sig(-9.0), "gradients": sig(-9.0)}, EMBEDDED_PROFILE)
        self.assertEqual(hi["score"], 1.0)
        self.assertEqual(lo["score"], 0.0)

    def test_result_shape(self):
        result = document.score_embedded_image(full_signals(0.3), EMBEDDED_PROFILE)
        self.assertEqual(set(result), {
            "score",
            "raw_normalized_score",
            "confidence",
            "usable",
            "effective_weight_sum",
            "evidence_coverage",
            "image_profile",
            "contributions",
            "note",
        })
        self.assertIs(result["image_profile"], EMBEDDED_PROFILE)
        self.assertIn("page coverage does not dilute", result["note"])

    def test_all_signals_at_040_produce_exactly_040(self):
        result = document.score_embedded_image(full_signals(0.40), EMBEDDED_PROFILE)
        self.assertEqual(result["score"], 0.4)


# ---------------------------------------------------------------------------
# Placeholder helpers
# ---------------------------------------------------------------------------


class TestPlaceholderHelpers(DocumentTestCase):
    def test_unavailable_embedded_signals_shape(self):
        result = document.unavailable_embedded_signals("decoder crashed")
        self.assertEqual(set(result), {"frequency", "noise", "gradients", "lighting", "metadata"})
        for value in result.values():
            self.assertEqual(value, {"score": None, "confidence": "unavailable", "note": "decoder crashed"})

    def test_unavailable_embedded_signals_entries_are_independent_dicts(self):
        result = document.unavailable_embedded_signals("x")
        result["noise"]["score"] = 0.9
        self.assertIsNone(result["frequency"]["score"])

    def test_unavailable_embedded_signals_score_as_unusable(self):
        signals = document.unavailable_embedded_signals("decoder crashed")
        result = document.score_embedded_image(signals, EMBEDDED_PROFILE)
        self.assertIs(result["usable"], False)
        self.assertEqual(result["confidence"], "insufficient")

    def test_unusable_embedded_image_score_shape(self):
        profile = {"width_px": 10, "height_px": 10}
        result = document.unusable_embedded_image_score(profile, "because")
        self.assertEqual(result, {
            "score": 0.0,
            "confidence": "insufficient",
            "usable": False,
            "evidence_coverage": 0.0,
            "image_profile": profile,
            "contributions": {},
            "note": "because",
        })

    def test_not_applicable_signals_use_none_not_half(self):
        # 0.5 would be a numeric AI score; None means "not applicable".
        result = document.not_applicable_signals(NATIVE_PAGE)
        self.assertEqual(set(result), {"frequency", "noise", "gradients", "lighting", "metadata"})
        for name, value in result.items():
            with self.subTest(signal=name):
                self.assertIsNone(value["score"])
                self.assertEqual(value["confidence"], "unavailable")
        self.assertIn("native_document", result["noise"]["note"])
        self.assertIn("per-page raster metadata", result["metadata"]["note"])

    def test_not_applicable_signals_entries_are_independent_dicts(self):
        result = document.not_applicable_signals(NATIVE_PAGE)
        result["noise"]["score"] = 0.9
        self.assertIsNone(result["gradients"]["score"])

    def test_not_applicable_signals_handle_unknown_content_type(self):
        result = document.not_applicable_signals({})
        self.assertIn("'unknown'", result["noise"]["note"])

    def test_not_applicable_signals_score_as_insufficient_page(self):
        signals = document.not_applicable_signals(NATIVE_PAGE)
        result = document.score_pdf_page(signals, NATIVE_PAGE)
        self.assertEqual(result["confidence"], "insufficient")
        self.assertIs(result["usable"], False)
        # Even on an image page these None-scored placeholders are unusable.
        result = document.score_pdf_page(signals, IMAGE_PAGE)
        self.assertEqual(result["confidence"], "insufficient")


# ---------------------------------------------------------------------------
# combine_page_evidence
# ---------------------------------------------------------------------------


class TestCombinePageEvidence(DocumentTestCase):
    def test_no_candidates_returns_insufficient(self):
        render = document.score_pdf_page({}, NATIVE_PAGE)
        result = document.combine_page_evidence(3, render, [])
        self.assertEqual(result, {
            "score": 0.0,
            "confidence": "insufficient",
            "usable": False,
            "source": None,
            "image_index": None,
            "page": 3,
            "evidence_coverage": 0.0,
            "note": "no reliable page-render or embedded-image evidence on this PDF page",
        })

    def test_page_render_only(self):
        result = document.combine_page_evidence(1, render_stub(0.61, "medium-high", coverage=0.9), [])
        self.assertEqual(result["score"], 0.61)
        self.assertEqual(result["confidence"], "medium-high")
        self.assertIs(result["usable"], True)
        self.assertEqual(result["source"], "page_render")
        self.assertIsNone(result["image_index"])
        self.assertEqual(result["page"], 1)
        self.assertEqual(result["evidence_coverage"], 0.9)
        self.assertEqual(result["candidate_count"], 1)

    def test_embedded_only_on_native_page(self):
        render = document.score_pdf_page(document.not_applicable_signals(NATIVE_PAGE), NATIVE_PAGE)
        scoring = document.score_embedded_image(full_signals(0.83), EMBEDDED_PROFILE)
        result = document.combine_page_evidence(2, render, [embedded_entry(7, scoring)])
        self.assertEqual(result["score"], 0.83)
        self.assertEqual(result["source"], "embedded_image")
        self.assertEqual(result["image_index"], 7)
        self.assertEqual(result["confidence"], "medium-high")
        self.assertEqual(result["evidence_coverage"], 1.0)
        self.assertEqual(result["candidate_count"], 1)

    def test_worst_candidate_wins_regardless_of_source(self):
        render = render_stub(0.30)
        embedded = [
            embedded_entry(1, document.score_embedded_image(full_signals(0.10), EMBEDDED_PROFILE)),
            embedded_entry(2, document.score_embedded_image(full_signals(0.95), EMBEDDED_PROFILE)),
            embedded_entry(3, document.score_embedded_image(full_signals(0.50), EMBEDDED_PROFILE)),
        ]
        result = document.combine_page_evidence(1, render, embedded)
        self.assertEqual(result["score"], 0.95)
        self.assertEqual(result["source"], "embedded_image")
        self.assertEqual(result["image_index"], 2)
        self.assertEqual(result["candidate_count"], 4)

    def test_page_render_wins_when_it_is_worst(self):
        render = render_stub(0.90)
        embedded = [embedded_entry(1, document.score_embedded_image(full_signals(0.40), EMBEDDED_PROFILE))]
        result = document.combine_page_evidence(1, render, embedded)
        self.assertEqual(result["source"], "page_render")
        self.assertEqual(result["score"], 0.9)

    def test_tie_prefers_page_render(self):
        # ``max`` keeps the first maximal candidate; page render is appended
        # first. Pins the current ordering so a reshuffle is noticed.
        render = render_stub(0.5)
        embedded = [embedded_entry(1, document.score_embedded_image(full_signals(0.5), EMBEDDED_PROFILE))]
        result = document.combine_page_evidence(1, render, embedded)
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["source"], "page_render")
        self.assertEqual(result["candidate_count"], 2)

    def test_unusable_page_render_is_excluded_even_with_reliable_confidence(self):
        render = render_stub(0.99, "medium", usable=False)
        result = document.combine_page_evidence(1, render, [])
        self.assertIs(result["usable"], False)
        self.assertEqual(result["confidence"], "insufficient")

    def test_unreliable_page_render_confidence_is_excluded(self):
        for confidence in ("low", "insufficient", "unavailable", "", "bogus"):
            with self.subTest(confidence=confidence):
                render = render_stub(0.99, confidence)
                result = document.combine_page_evidence(1, render, [])
                self.assertIs(result["usable"], False)
                self.assertEqual(result["score"], 0.0)

    def test_reliable_page_render_confidences_are_included(self):
        for confidence in RELIABLE_CONFIDENCES:
            with self.subTest(confidence=confidence):
                render = render_stub(0.42, confidence)
                result = document.combine_page_evidence(1, render, [])
                self.assertIs(result["usable"], True)
                self.assertEqual(result["confidence"], confidence)

    def test_low_confidence_embedded_image_cannot_drive_page_risk(self):
        # Embedded image with a high score but only 'low' confidence
        # (e.g. lighting-only) is retained in the response but not a candidate.
        scoring = document.score_embedded_image({"lighting": sig(1.0)}, EMBEDDED_PROFILE)
        self.assertIs(scoring["usable"], True)
        self.assertEqual(scoring["confidence"], "low")
        result = document.combine_page_evidence(1, render_stub(0.2), [embedded_entry(1, scoring)])
        self.assertEqual(result["source"], "page_render")
        self.assertEqual(result["score"], 0.2)
        self.assertEqual(result["candidate_count"], 1)

    def test_unusable_embedded_image_is_excluded(self):
        tiny = document.score_embedded_image(full_signals(1.0, "high"), {"width_px": 10, "height_px": 10})
        result = document.combine_page_evidence(1, render_stub(0.2), [embedded_entry(1, tiny)])
        self.assertEqual(result["score"], 0.2)
        self.assertEqual(result["candidate_count"], 1)

    def test_embedded_image_missing_optional_keys(self):
        # ai_generated / scoring may be absent -> defaults of 0.0 / coverage 0.
        img = {"usable": True, "confidence": "medium", "image_index": 4}
        result = document.combine_page_evidence(1, {}, [img])
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["source"], "embedded_image")
        self.assertEqual(result["image_index"], 4)
        self.assertEqual(result["evidence_coverage"], 0.0)

    def test_empty_page_render_dict_is_tolerated(self):
        result = document.combine_page_evidence(5, {}, [])
        self.assertIs(result["usable"], False)
        self.assertEqual(result["page"], 5)

    def test_scores_are_rounded_and_cast(self):
        render = render_stub("0.123456", "medium", coverage="0.98765")
        result = document.combine_page_evidence(1, render, [])
        self.assertEqual(result["score"], 0.1235)
        self.assertEqual(result["evidence_coverage"], 0.9877)
        self.assertIsInstance(result["score"], float)

    def test_page_number_is_echoed(self):
        for n in (1, 7, 20):
            with self.subTest(page=n):
                self.assertEqual(document.combine_page_evidence(n, render_stub(0.1), [])["page"], n)


# ---------------------------------------------------------------------------
# aggregate_pdf_pages
# ---------------------------------------------------------------------------


class TestAggregatePdfPagesEmptyAndUnusable(DocumentTestCase):
    def test_no_pages(self):
        result = document.aggregate_pdf_pages([])
        self.assertEqual(result, {
            "score": 0.0,
            "confidence": "insufficient",
            "usable_pages": 0,
            "reliable_pages": 0,
            "worst_page": None,
            "max_page_score": 0.0,
            "mean_page_score": 0.0,
            "flagged_pages": [],
            "note": "no PDF pages contained enough applicable raster evidence for AI-image scoring",
        })

    def test_all_pages_unusable(self):
        pages = [page_entry(i, 0.9, "insufficient", usable=False) for i in range(1, 4)]
        result = document.aggregate_pdf_pages(pages)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["confidence"], "insufficient")
        self.assertEqual(result["usable_pages"], 0)
        self.assertIsNone(result["worst_page"])
        self.assertEqual(result["flagged_pages"], [])

    def test_pages_without_pdf_scoring_key_are_unusable(self):
        result = document.aggregate_pdf_pages([{"page": 1}, {"page": 2}])
        self.assertEqual(result["usable_pages"], 0)
        self.assertIsNone(result["worst_page"])

    def test_only_low_confidence_pages_stay_inconclusive_but_report_diagnostics(self):
        pages = [
            page_entry(1, 0.90, "low", evidence_coverage=0.15),
            page_entry(2, 0.10, "low", evidence_coverage=0.15),
        ]
        result = document.aggregate_pdf_pages(pages)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["confidence"], "insufficient")
        self.assertEqual(result["usable_pages"], 2)
        self.assertEqual(result["reliable_pages"], 0)
        self.assertIsNone(result["worst_page"])
        self.assertEqual(result["flagged_pages"], [])
        # Diagnostics are still computed from the usable pages.
        self.assertEqual(result["max_page_score"], 0.9)
        self.assertClose(result["mean_page_score"], 0.5)
        self.assertIn("no page is reliable enough", result["note"])

    def test_low_confidence_weighted_mean_uses_coverage_floor(self):
        # Coverage 0.0 is floored to 0.05 so a page is never weighted to zero.
        pages = [
            page_entry(1, 1.0, "low", evidence_coverage=0.0),
            page_entry(2, 0.0, "low", evidence_coverage=0.05),
        ]
        result = document.aggregate_pdf_pages(pages)
        self.assertClose(result["mean_page_score"], 0.5)


class TestAggregatePdfPagesWorstPageWins(DocumentTestCase):
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

        self.assertEqual(result["score"], 0.91)
        self.assertEqual(result["max_page_score"], 0.91)
        self.assertEqual(result["worst_page"], 4)
        self.assertEqual(result["confidence"], "medium-high")
        self.assertEqual(result["usable_pages"], 5)
        self.assertEqual(result["reliable_pages"], 5)
        self.assertEqual(result["aggregation"], "worst_reliable_evidence")
        self.assertEqual(result["worst_evidence_source"], "page_render")
        self.assertIsNone(result["worst_image_index"])

        # Diagnostic mean is far lower than the max -- proving the verdict is
        # not an average.
        expected_mean = (0.08 + 0.12 + 0.05 + 0.91 + 0.10) / 5
        self.assertClose(result["mean_page_score"], expected_mean, tol=1e-4)
        self.assertLess(result["mean_page_score"], 0.30)
        self.assertGreater(result["score"], 0.30)

        self.assertEqual(result["flagged_pages"], [{
            "page": 4,
            "score": 0.91,
            "confidence": "medium-high",
            "content_type": "scanned_or_image_document",
            "risk_source": "page_render",
            "image_index": None,
        }])

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
        self.assertIs(pages[2]["pdf_scoring"]["usable"], False)
        self.assertIs(pages[0]["pdf_scoring"]["usable"], False)

        self.assertEqual(result["worst_page"], 4)
        self.assertEqual(result["score"], 0.88)
        self.assertEqual(result["usable_pages"], 3)
        self.assertEqual(result["reliable_pages"], 3)
        self.assertEqual([f["page"] for f in result["flagged_pages"]], [4])
        self.assertClose(result["mean_page_score"], (0.10 + 0.88 + 0.15) / 3, tol=1e-4)

    def test_adding_clean_pages_never_lowers_the_document_score(self):
        base = [page_entry(1, 0.75, "medium")]
        base_score = document.aggregate_pdf_pages(base)["score"]
        padded = base + [page_entry(i, 0.02, "medium-high") for i in range(2, 21)]
        padded_result = document.aggregate_pdf_pages(padded)
        self.assertEqual(padded_result["score"], base_score)
        self.assertEqual(base_score, 0.75)
        self.assertEqual(padded_result["worst_page"], 1)
        self.assertLess(padded_result["mean_page_score"], 0.10)

    def test_worst_page_is_argmax_not_last_or_first(self):
        pages = [page_entry(i, s, "medium") for i, s in enumerate([0.3, 0.7, 0.9, 0.6, 0.2], start=1)]
        self.assertEqual(document.aggregate_pdf_pages(pages)["worst_page"], 3)

    def test_worst_page_tie_keeps_first_occurrence(self):
        pages = [page_entry(1, 0.5, "medium"), page_entry(2, 0.5, "medium-high")]
        result = document.aggregate_pdf_pages(pages)
        self.assertEqual(result["worst_page"], 1)
        self.assertEqual(result["confidence"], "medium")

    def test_document_confidence_is_that_of_the_worst_page(self):
        pages = [
            page_entry(1, 0.20, "medium-high"),
            page_entry(2, 0.55, "low-medium"),
        ]
        result = document.aggregate_pdf_pages(pages)
        self.assertEqual(result["worst_page"], 2)
        self.assertEqual(result["confidence"], "low-medium")

    def test_low_confidence_page_with_higher_score_does_not_become_worst(self):
        pages = [
            page_entry(1, 0.35, "medium"),
            page_entry(2, 0.99, "low"),  # usable but not reliable
        ]
        result = document.aggregate_pdf_pages(pages)
        self.assertEqual(result["worst_page"], 1)
        self.assertEqual(result["score"], 0.35)
        self.assertEqual(result["usable_pages"], 2)
        self.assertEqual(result["reliable_pages"], 1)
        self.assertEqual(result["flagged_pages"], [])
        # ...but the diagnostics still see it.
        self.assertEqual(result["max_page_score"], 0.35)  # max over *reliable* pages
        self.assertGreater(result["mean_page_score"], 0.35)  # mean includes the low-confidence page

    def test_unusable_pages_are_excluded_from_everything(self):
        pages = [
            page_entry(1, 0.30, "medium"),
            page_entry(2, 0.99, "medium", usable=False),
        ]
        result = document.aggregate_pdf_pages(pages)
        self.assertEqual(result["usable_pages"], 1)
        self.assertEqual(result["worst_page"], 1)
        self.assertEqual(result["mean_page_score"], 0.3)

    def test_worst_page_driven_by_embedded_image_is_attributed(self):
        pages = [
            page_entry(1, 0.10, "medium-high"),
            page_entry(2, 0.86, "medium-high", content_type="native_document",
                       source="embedded_image", image_index=3),
        ]
        result = document.aggregate_pdf_pages(pages)
        self.assertEqual(result["worst_page"], 2)
        self.assertEqual(result["worst_evidence_source"], "embedded_image")
        self.assertEqual(result["worst_image_index"], 3)
        self.assertEqual(result["flagged_pages"][0]["risk_source"], "embedded_image")
        self.assertEqual(result["flagged_pages"][0]["image_index"], 3)
        self.assertEqual(result["flagged_pages"][0]["content_type"], "native_document")

    def test_mean_page_score_weights_by_evidence_coverage(self):
        pages = [
            page_entry(1, 1.0, "medium", evidence_coverage=0.75),
            page_entry(2, 0.0, "medium", evidence_coverage=0.25),
        ]
        result = document.aggregate_pdf_pages(pages)
        self.assertClose(result["mean_page_score"], 0.75)

    def test_mean_page_score_floors_zero_coverage_at_005(self):
        pages = [
            page_entry(1, 1.0, "medium", evidence_coverage=0.0),   # -> 0.05
            page_entry(2, 0.0, "medium", evidence_coverage=0.95),
        ]
        result = document.aggregate_pdf_pages(pages)
        self.assertClose(result["mean_page_score"], 0.05)

    def test_accepts_any_iterable(self):
        gen = (page_entry(i, 0.1 * i, "medium") for i in range(1, 4))
        result = document.aggregate_pdf_pages(gen)
        self.assertEqual(result["worst_page"], 3)
        self.assertEqual(result["usable_pages"], 3)

    def test_page_numbers_are_cast_to_int(self):
        pages = [page_entry("2", 0.5, "medium")]
        result = document.aggregate_pdf_pages(pages)
        self.assertEqual(result["worst_page"], 2)
        self.assertEqual(result["flagged_pages"][0]["page"], 2)

    def test_result_shape_for_reliable_document(self):
        result = document.aggregate_pdf_pages([page_entry(1, 0.5, "medium")])
        self.assertEqual(set(result), {
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
        })


class TestAggregatePdfPagesFlaggedThreshold(DocumentTestCase):
    """The flagged-page cutoff is ``score >= 0.40`` over *reliable* pages."""

    def test_page_scoring_exactly_040_is_flagged(self):
        pages = [
            page_entry(1, 0.10, "medium"),
            page_entry(2, 0.40, "medium"),
            page_entry(3, 0.20, "medium"),
        ]
        result = document.aggregate_pdf_pages(pages)
        self.assertEqual(result["worst_page"], 2)
        self.assertEqual(result["score"], 0.4)
        self.assertEqual([f["page"] for f in result["flagged_pages"]], [2])
        self.assertEqual(result["flagged_pages"][0]["score"], 0.4)

    def test_page_scoring_exactly_040_is_flagged_end_to_end(self):
        """Drive 0.40 through the real scoring chain rather than injecting it:
        four medium signals at 0.40 on a fully-raster page -> score_pdf_page
        -> combine_page_evidence -> aggregate_pdf_pages."""
        pages = [
            build_page(1, full_signals(0.10), IMAGE_PAGE),
            build_page(2, full_signals(0.40), IMAGE_PAGE),
            build_page(3, document.not_applicable_signals(NATIVE_PAGE), NATIVE_PAGE),
        ]
        self.assertEqual(pages[1]["page_render_scoring"]["score"], 0.4)
        self.assertEqual(pages[1]["pdf_scoring"]["score"], 0.4)

        result = document.aggregate_pdf_pages(pages)
        self.assertEqual(result["worst_page"], 2)
        self.assertEqual(result["score"], 0.4)
        self.assertEqual(result["flagged_pages"], [{
            "page": 2,
            "score": 0.4,
            "confidence": "medium-high",
            "content_type": "scanned_or_image_document",
            "risk_source": "page_render",
            "image_index": None,
        }])

    def test_embedded_image_scoring_exactly_040_is_flagged_end_to_end(self):
        page = build_page(
            1,
            document.not_applicable_signals(NATIVE_PAGE),
            NATIVE_PAGE,
            embedded=[(full_signals(0.40), EMBEDDED_PROFILE)],
        )
        self.assertEqual(page["pdf_scoring"]["score"], 0.4)
        self.assertEqual(page["pdf_scoring"]["source"], "embedded_image")
        result = document.aggregate_pdf_pages([page])
        self.assertEqual([f["page"] for f in result["flagged_pages"]], [1])
        self.assertEqual(result["flagged_pages"][0]["risk_source"], "embedded_image")
        self.assertEqual(result["flagged_pages"][0]["image_index"], 1)

    def test_pages_just_below_040_are_not_flagged(self):
        for score in (0.3999, 0.39, 0.3, 0.0):
            with self.subTest(score=score):
                pages = [page_entry(1, score, "medium")]
                result = document.aggregate_pdf_pages(pages)
                self.assertEqual(result["worst_page"], 1)  # still the worst (only) page...
                self.assertEqual(result["score"], score)
                self.assertEqual(result["flagged_pages"], [])  # ...but not flagged

    def test_pages_at_or_above_040_are_flagged(self):
        for score in (0.4, 0.4001, 0.41, 0.5, 1.0):
            with self.subTest(score=score):
                result = document.aggregate_pdf_pages([page_entry(1, score, "medium")])
                self.assertEqual([f["page"] for f in result["flagged_pages"]], [1])

    def test_threshold_compares_the_unrounded_page_score(self):
        # KNOWN QUIRK: aggregate_pdf_pages reports round(score, 4) but applies
        # the >= 0.40 test to the *raw* value. A raw 0.39995 therefore displays
        # as 0.4 yet is not flagged. In the real pipeline this cannot occur
        # because combine_page_evidence has already rounded to 4 decimals
        # (see the end-to-end tests above); this test pins the raw behaviour.
        result = document.aggregate_pdf_pages([page_entry(1, 0.39995, "medium")])
        self.assertEqual(result["score"], 0.4)
        self.assertEqual(result["max_page_score"], 0.4)
        self.assertEqual(result["flagged_pages"], [])

    def test_039994_stays_below_and_is_not_flagged(self):
        result = document.aggregate_pdf_pages([page_entry(1, 0.39994, "medium")])
        self.assertEqual(result["score"], 0.3999)
        self.assertEqual(result["flagged_pages"], [])

    def test_threshold_is_applied_only_to_reliable_pages(self):
        pages = [
            page_entry(1, 0.40, "low"),         # usable but not reliable
            page_entry(2, 0.40, "low-medium"),  # reliable boundary
        ]
        result = document.aggregate_pdf_pages(pages)
        self.assertEqual([f["page"] for f in result["flagged_pages"]], [2])

    def test_multiple_flagged_pages_sorted_by_score_descending(self):
        pages = [
            page_entry(1, 0.45, "medium"),
            page_entry(2, 0.10, "medium"),
            page_entry(3, 0.95, "medium-high"),
            page_entry(4, 0.40, "low-medium"),
            page_entry(5, 0.70, "medium"),
        ]
        result = document.aggregate_pdf_pages(pages)
        self.assertEqual([f["page"] for f in result["flagged_pages"]], [3, 5, 1, 4])
        self.assertEqual([f["score"] for f in result["flagged_pages"]], [0.95, 0.7, 0.45, 0.4])
        self.assertEqual(result["worst_page"], 3)
        self.assertEqual(result["score"], 0.95)

    def test_flagged_page_entry_shape(self):
        result = document.aggregate_pdf_pages([page_entry(9, 0.5, "medium", content_type="mixed_document")])
        self.assertEqual(result["flagged_pages"], [{
            "page": 9,
            "score": 0.5,
            "confidence": "medium",
            "content_type": "mixed_document",
            "risk_source": "page_render",
            "image_index": None,
        }])

    def test_flagged_page_missing_content_type_defaults_to_unknown(self):
        page = page_entry(1, 0.5, "medium")
        del page["content_type"]
        result = document.aggregate_pdf_pages([page])
        self.assertEqual(result["flagged_pages"][0]["content_type"], "unknown")


# ---------------------------------------------------------------------------
# classify_pdf
# ---------------------------------------------------------------------------


class TestClassifyPdf(DocumentTestCase):
    def test_score_bands(self):
        cases = [
            (0.0, "low_ai_evidence"),
            (0.2999, "low_ai_evidence"),
            (0.30, "possible_ai_evidence"),
            (0.40, "possible_ai_evidence"),
            (0.4999, "possible_ai_evidence"),
            (0.50, "suspicious_raster_content"),
            (0.6999, "suspicious_raster_content"),
            (0.70, "likely_ai_generated"),
            (1.0, "likely_ai_generated"),
        ]
        for score, expected in cases:
            with self.subTest(score=score):
                classification, description = document.classify_pdf(score, "medium")
                self.assertEqual(classification, expected)
                self.assertIsInstance(description, str)
                self.assertTrue(description)

    def test_band_descriptions(self):
        self.assertEqual(document.classify_pdf(0.1, "medium")[1], "Low AI-generated image evidence detected.")
        self.assertEqual(document.classify_pdf(0.4, "medium")[1], "Possible AI-generated image evidence detected.")
        self.assertEqual(document.classify_pdf(0.6, "medium")[1], "Moderate AI-generated image evidence detected.")
        self.assertEqual(document.classify_pdf(0.9, "medium")[1], "Strong AI-generated image evidence detected.")

    def test_insufficient_confidence_overrides_score(self):
        for score in (0.0, 0.5, 1.0):
            with self.subTest(score=score):
                classification, description = document.classify_pdf(score, "insufficient")
                self.assertEqual(classification, "insufficient_visual_evidence")
                self.assertIn("Not enough reliable raster evidence", description)

    def test_metadata_hit_overrides_everything(self):
        for confidence in ("low", "insufficient", "medium-high"):
            for score in (0.0, 0.5, 1.0):
                with self.subTest(score=score, confidence=confidence):
                    classification, description = document.classify_pdf(score, confidence, metadata_hit=True)
                    self.assertEqual(classification, "likely_ai_generated")
                    self.assertIn("provenance", description)

    def test_non_insufficient_confidence_does_not_change_band(self):
        for confidence in ("low", "low-medium", "medium", "medium-high", "high", "anything"):
            with self.subTest(confidence=confidence):
                self.assertEqual(document.classify_pdf(0.6, confidence)[0], "suspicious_raster_content")

    def test_strong_band_requires_solid_confidence(self):
        # A low-medium page can reach ~0.71 after the reliability shrink; it
        # must read as suspicious, not likely. Medium or better may be likely.
        self.assertEqual(document.STRONG_BAND_CONFIDENCES, {"medium", "medium-high", "high"})
        for confidence in ("medium", "medium-high", "high"):
            with self.subTest(confidence=confidence):
                self.assertEqual(document.classify_pdf(0.70, confidence)[0], "likely_ai_generated")
                self.assertEqual(document.classify_pdf(1.0, confidence)[0], "likely_ai_generated")
        for confidence in ("low-medium", "low", "anything"):
            with self.subTest(confidence=confidence):
                self.assertEqual(document.classify_pdf(0.70, confidence)[0], "suspicious_raster_content")
                self.assertEqual(document.classify_pdf(1.0, confidence)[0], "suspicious_raster_content")
                # Lower bands are unaffected by the cap.
                self.assertEqual(document.classify_pdf(0.6, confidence)[0], "suspicious_raster_content")
                self.assertEqual(document.classify_pdf(0.4, confidence)[0], "possible_ai_evidence")
                self.assertEqual(document.classify_pdf(0.1, confidence)[0], "low_ai_evidence")

    def test_low_medium_page_at_its_maximum_score_is_not_likely(self):
        # Worst case found by exhaustive search: frequency high + noise
        # low-medium + gradients low, all scoring 1.0 -> ews 0.39 -> 0.7091.
        signals = {"frequency": sig(1.0, "high"), "noise": sig(1.0, "low-medium"), "gradients": sig(1.0, "low")}
        page = document.score_pdf_page(signals, IMAGE_PAGE)
        self.assertEqual(page["confidence"], "low-medium")
        self.assertGreaterEqual(page["score"], 0.70)
        self.assertEqual(document.classify_pdf(page["score"], page["confidence"])[0], "suspicious_raster_content")

    def test_non_finite_or_non_numeric_score_is_insufficient(self):
        for bad in (float("nan"), float("inf"), float("-inf"), None, "abc"):
            with self.subTest(score=bad):
                classification, description = document.classify_pdf(bad, "medium-high")
                self.assertEqual(classification, "insufficient_visual_evidence")
                self.assertIn("Not enough reliable raster evidence", description)
        # The metadata override still wins over a bad score.
        self.assertEqual(document.classify_pdf(float("nan"), "medium", metadata_hit=True)[0], "likely_ai_generated")


# ---------------------------------------------------------------------------
# End-to-end composition of the pure functions
# ---------------------------------------------------------------------------


class TestEndToEndComposition(DocumentTestCase):
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
        self.assertEqual(result["worst_page"], 2)
        self.assertEqual(result["score"], 0.92)
        self.assertEqual(result["worst_evidence_source"], "embedded_image")
        self.assertEqual(result["worst_image_index"], 2)
        self.assertEqual([f["page"] for f in result["flagged_pages"]], [2])
        self.assertEqual(
            document.classify_pdf(result["score"], result["confidence"])[0],
            "likely_ai_generated",
        )

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
        self.assertEqual(result["usable_pages"], 0)
        self.assertIsNone(result["worst_page"])
        self.assertEqual(
            document.classify_pdf(result["score"], result["confidence"])[0],
            "insufficient_visual_evidence",
        )

    def test_all_native_document_is_insufficient(self):
        pages = [build_page(i, document.not_applicable_signals(NATIVE_PAGE), NATIVE_PAGE) for i in range(1, 6)]
        result = document.aggregate_pdf_pages(pages)
        self.assertEqual(result["confidence"], "insufficient")
        self.assertEqual(result["usable_pages"], 0)
        self.assertEqual(
            document.classify_pdf(result["score"], result["confidence"])[0],
            "insufficient_visual_evidence",
        )

    def test_noise_only_evidence_on_every_page_is_shrunk_consistently(self):
        # Pages where only the noise signal is valid: each page is shrunk by
        # 0.45/0.55, and the worst page still wins.
        pages = [
            build_page(1, {"noise": sig(0.30)}, IMAGE_PAGE),
            build_page(2, {"noise": sig(1.00)}, IMAGE_PAGE),
        ]
        result = document.aggregate_pdf_pages(pages)
        self.assertEqual(result["worst_page"], 2)
        self.assertEqual(result["score"], 0.8182)
        self.assertEqual(result["confidence"], "medium")
        self.assertEqual([f["page"] for f in result["flagged_pages"]], [2])
        self.assertClose(pages[0]["pdf_scoring"]["score"], 0.30 * NOISE_ONLY_RELIABILITY, tol=1e-4)

    def test_mixed_page_with_dominant_image_is_scored_with_reduced_signal_set(self):
        profile = {"content_type": "mixed_document", "image_area_fraction": 0.75}
        page = build_page(1, full_signals(0.9), profile)
        render = page["page_render_scoring"]
        self.assertEqual(used_signals(render), {"noise", "gradients"})
        self.assertClose(render["effective_weight_sum"], 0.80)
        self.assertClose(render["score"], 0.9)
        self.assertIs(page["pdf_scoring"]["usable"], True)
        result = document.aggregate_pdf_pages([page])
        self.assertEqual(result["score"], 0.9)
        self.assertEqual(result["flagged_pages"][0]["content_type"], "mixed_document")

    def test_metadata_hit_classifies_even_when_visual_evidence_is_clean(self):
        pages = [build_page(1, full_signals(0.05), IMAGE_PAGE)]
        aggregate = document.aggregate_pdf_pages(pages)
        meta = document.analyze_pdf_metadata({"Creator": "Midjourney"})
        hit = meta["confidence"] == "high"
        self.assertTrue(hit)
        classification, _ = document.classify_pdf(aggregate["score"], aggregate["confidence"], metadata_hit=hit)
        self.assertEqual(classification, "likely_ai_generated")


if __name__ == "__main__":
    unittest.main(verbosity=2)
