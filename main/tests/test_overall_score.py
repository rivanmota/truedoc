"""
Scenario tests for the overall score (``type.ai_generated`` for PNG/JPEG).

The overall score is produced by ``app.main.score_image_signals``:

    signals + context --_visual_score--> visual score --_combine(metadata)--> ai_generated

Every test here feeds hand-built signal dicts straight into that function
(no image is decoded), so each scenario is deterministic, fast and states
exactly which rule it exercises. Expected values are written as explicit
arithmetic from the documented rules, not copied from program output:

  base weights          frequency 0.25, noise 0.40, gradients 0.25, lighting 0.10
  confidence factors    high/medium 1.0, low-medium 0.7, low 0.5 (unknown -> 0.5)
  not applicable        zero weight (``applicable: False`` or a non-finite score)
  no portrait           gradients' weight moves to noise + frequency, 0.40 : 0.25
  one substantive sig.  score capped at 0.60 (lighting never counts as corroboration)
  no substantive sig.   score 0.50 ("inconclusive": no evidence either way)
  context factors       weight x factor; factors < 1 discount absolutely
  metadata              generator signature -> max(visual, 0.97);
                        camera EXIF -> 0.85 * visual + 0.15 * 0.15; otherwise unchanged
  UI bands              < 0.35 "Likely real", > 0.65 "Likely AI", else "Inconclusive"

Scenario groups:
  1. realistic archetypes  - signal values measured on the baseline files and
                             controls, with the verdict each must get
  2. weighting arithmetic  - exact values for each weighting rule
  3. evidence sufficiency  - cap, no-evidence, corroboration
  4. robustness            - malformed inputs, invariants, property checks
  5. metadata              - how file metadata combines with the visual score
  6. context weighting     - calibrated factors, with explicit models
  7. response + contract   - output shape, rounding, UI threshold sync

Static-weight scenarios pass ``model=None`` so they don't depend on the
shipped calibration file; group 6 tests the shipped file separately.
"""
from __future__ import annotations

import itertools
import math
import re
from pathlib import Path

import numpy as np
import pytest

from app import main, weighting

# --------------------------------------------------------------------------- #
# Specification constants (independent of the implementation's constants;
# TestSpecification checks the two agree, so a deliberate change to a rule
# fails there first with a readable message).
# --------------------------------------------------------------------------- #
W_F, W_N, W_G, W_L = 0.25, 0.40, 0.25, 0.10
LOW = 0.5                      # confidence factor for "low"
LOW_MEDIUM = 0.7
L_EFF = W_L * LOW              # lighting always reports "low"
# No portrait: gradients' weight split between noise and frequency 0.40 : 0.25
NP_F = W_F + W_G * W_F / (W_F + W_N)
NP_N = W_N + W_G * W_N / (W_F + W_N)
CAP = 0.60
NO_EVIDENCE = 0.50
SIGNATURE = 0.97               # metadata.analyze() score for a generator signature
CAMERA_EXIF = 0.15             # metadata.analyze() score for plausible camera EXIF

REAL_BELOW = 0.35
AI_ABOVE = 0.65

SIGNALS = ("frequency", "noise", "gradients", "lighting")
DEFAULT_BUCKETS = {"compression": "lossless", "size": "medium", "scale": "native", "color": "color"}


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def S(score, confidence="medium"):
    """An applicable signal."""
    return {"score": score, "confidence": confidence, "applicable": True, "note": "scenario"}


def NA(placeholder=0.5):
    """A not-applicable signal (analyzers report a placeholder score)."""
    return {"score": placeholder, "confidence": "low", "applicable": False, "note": "not applicable"}


NEUTRAL_META = {"score": 0.5, "confidence": "low", "note": "no signature, no camera EXIF"}
SIGNATURE_META = {"score": SIGNATURE, "confidence": "high", "signature_hits": ["x"], "note": "generator"}
CAMERA_META = {"score": CAMERA_EXIF, "confidence": "low-medium", "note": "camera EXIF"}


def analysis(frequency=None, noise=None, gradients=None, lighting=None,
             metadata=NEUTRAL_META, buckets=None):
    """An ``analyze_image_signals``-shaped dict. Omitted signals are NA."""
    sigs = {"frequency": frequency, "noise": noise, "gradients": gradients, "lighting": lighting}
    return {
        "signals": {k: (v if v is not None else NA()) for k, v in sigs.items()},
        "metadata": metadata,
        "context": {"format": "png", "buckets": dict(buckets or DEFAULT_BUCKETS)},
    }


def overall(a, model=None) -> float:
    return main.score_image_signals(a, model)["ai_generated"]


def scoring(a, model=None) -> dict:
    return main.score_image_signals(a, model)["scoring"]


def verdict(score: float) -> str:
    """Mirror of web/lib/verdict.ts (kept in sync by TestResponseContract)."""
    if score < REAL_BELOW:
        return "real"
    if score > AI_ABOVE:
        return "ai"
    return "inconclusive"


def model_with(signal, dim, bucket, factor, base_weights=None):
    m = {"multipliers": {signal: {dim: {bucket: factor}}}}
    if base_weights:
        m["base_weights"] = base_weights
    return m


# =========================================================================== #
# 0. The specification this file is written against
# =========================================================================== #
class TestSpecification:
    def test_base_weights(self):
        assert main.WEIGHTS == {"frequency": W_F, "noise": W_N, "gradients": W_G, "lighting": W_L}

    def test_confidence_factors(self):
        assert main.CONFIDENCE_FACTOR == {"high": 1.0, "medium": 1.0, "low-medium": LOW_MEDIUM, "low": LOW}

    def test_cap_and_no_evidence_constants(self):
        assert main.SINGLE_SIGNAL_CAP == CAP
        assert main.NO_EVIDENCE_SCORE == NO_EVIDENCE

    def test_gradients_fallback(self):
        assert tuple(main.GRADIENT_FALLBACK_SIGNALS) == ("noise", "frequency")
        assert set(main.SUBSTANTIVE_SIGNALS) == {"noise", "frequency", "gradients"}


# =========================================================================== #
# 1. Realistic archetypes
#    Signal values were measured on an earlier set of baseline files and on
#    controls (the file names in the ids below refer to those earlier
#    pictures, which have been replaced; the current baseline pictures are
#    not flagged, see the README). The scenarios feed fixed signal values to
#    the combiner, so they are independent of the files now in baseline/.
#    Each pins the verdict and, under static weights, the exact value.
# =========================================================================== #
ARCHETYPES = [
    # id, analysis kwargs, expected static value, expected verdict, why
    (
        "generated non-portrait scene (fireplace-generated.png)",
        dict(frequency=S(0.870), noise=S(0.694), lighting=S(0.179, "low"),
             buckets={"compression": "lossless", "size": "large", "scale": "upscaled", "color": "color"}),
        (NP_F * 0.870 + NP_N * 0.694 + L_EFF * 0.179) / (NP_F + NP_N + L_EFF),
        "ai",
        "no face: noise + frequency carry the score and both fire",
    ),
    (
        "generated portrait (ai-generated.png)",
        dict(frequency=S(0.0), noise=S(0.903), gradients=S(1.0), lighting=S(0.950, "low"),
             buckets={"compression": "lossless", "size": "medium", "scale": "native", "color": "color"}),
        (W_F * 0.0 + W_N * 0.903 + W_G * 1.0 + L_EFF * 0.950) / (W_F + W_N + W_G + L_EFF),
        "ai",
        "airbrushed skin and synthetic grain corroborate each other",
    ),
    (
        "AI-retouched photo, noise is the only tell (ai-retouched.png)",
        dict(frequency=S(0.0), noise=S(0.919), lighting=S(0.873, "low"),
             buckets={"compression": "lossless", "size": "medium", "scale": "native", "color": "color"}),
        (NP_F * 0.0 + NP_N * 0.919 + L_EFF * 0.873) / (NP_F + NP_N + L_EFF),
        "inconclusive",
        "one strong tell, contradicted by a clean spectrum: flagged, not 'likely AI'",
    ),
    (
        "genuine phone JPEG (no face, noise N/A)",
        dict(frequency=S(0.0), lighting=S(0.502, "low"),
             buckets={"compression": "jpeg_mid", "size": "medium", "scale": "native", "color": "color"}),
        # noise N/A (JPEG grid), no face: frequency gets 0.25 + all of gradients' 0.25
        (0.50 * 0.0 + L_EFF * 0.502) / (0.50 + L_EFF),
        "real",
        "no tell fires; lighting alone cannot lift it",
    ),
    (
        "genuine photo with stationary film grain (astronaut.png) - KNOWN FALSE POSITIVE",
        dict(frequency=S(0.0), noise=S(1.0), gradients=S(0.079), lighting=S(0.892, "low"),
             buckets={"compression": "lossless", "size": "small", "scale": "native", "color": "color"}),
        (W_F * 0.0 + W_N * 1.0 + W_G * 0.079 + L_EFF * 0.892) / (W_F + W_N + W_G + L_EFF),
        "inconclusive",
        "film grain looks like synthetic grain; natural skin keeps it out of 'likely AI'",
    ),
    (
        "genuine phone photo upscaled 1.41x - KNOWN FALSE POSITIVE",
        dict(frequency=S(0.434), lighting=S(0.516, "low"),
             buckets={"compression": "lossless", "size": "medium", "scale": "upscaled", "color": "color"}),
        (0.50 * 0.434 + L_EFF * 0.516) / (0.50 + L_EFF),
        "inconclusive",
        "the frequency knee detects upscaling of any image",
    ),
    (
        "generated scene after JPEG 92 (noise erased, knee survives)",
        dict(frequency=S(0.736), lighting=S(0.178, "low"),
             buckets={"compression": "jpeg_high", "size": "large", "scale": "upscaled", "color": "color"}),
        CAP,  # (0.50*0.736 + 0.05*0.178)/0.55 = 0.685 > cap: frequency is the only substantive signal
        "inconclusive",
        "a single surviving tell is capped at 0.60",
    ),
    (
        "generated photo after JPEG 80 (every tell erased)",
        dict(frequency=S(0.0), lighting=S(0.858, "low"),
             buckets={"compression": "jpeg_mid", "size": "medium", "scale": "native", "color": "color"}),
        (0.50 * 0.0 + L_EFF * 0.858) / (0.50 + L_EFF),
        "real",
        "KNOWN MISS: compression destroyed the only evidence",
    ),
    (
        "icon-sized image (nothing measurable)",
        dict(lighting=S(0.9, "low"),
             buckets={"compression": "lossless", "size": "small", "scale": "native", "color": "color"}),
        NO_EVIDENCE,
        "inconclusive",
        "no substantive signal: no basis for either verdict",
    ),
]


class TestArchetypes:
    @pytest.mark.parametrize("name,kwargs,expected,band,why", ARCHETYPES, ids=[a[0] for a in ARCHETYPES])
    def test_static_value(self, name, kwargs, expected, band, why):
        assert overall(analysis(**kwargs)) == pytest.approx(expected, abs=1e-4), why

    @pytest.mark.parametrize("name,kwargs,expected,band,why", ARCHETYPES, ids=[a[0] for a in ARCHETYPES])
    def test_static_verdict(self, name, kwargs, expected, band, why):
        assert verdict(overall(analysis(**kwargs))) == band, why

    @pytest.mark.parametrize("name,kwargs,expected,band,why", ARCHETYPES, ids=[a[0] for a in ARCHETYPES])
    def test_shipped_calibration_keeps_the_verdict(self, name, kwargs, expected, band, why):
        model = weighting.load_model()
        if model is None:
            pytest.skip("no calibration file shipped")
        dynamic = overall(analysis(**kwargs), model)
        assert verdict(dynamic) == band, f"{why}: dynamic={dynamic}"

    @pytest.mark.parametrize("name,kwargs,expected,band,why", ARCHETYPES, ids=[a[0] for a in ARCHETYPES])
    def test_shipped_calibration_is_a_correction_not_an_override(self, name, kwargs, expected, band, why):
        # Context factors re-balance evidence; a large swing on a realistic
        # input means a recalibration changed behaviour materially and
        # should be reviewed (eval/REPORT.md), not silently shipped.
        model = weighting.load_model()
        if model is None:
            pytest.skip("no calibration file shipped")
        assert abs(overall(analysis(**kwargs), model) - overall(analysis(**kwargs))) <= 0.10

    def test_generated_archetypes_outrank_every_genuine_one(self):
        generated = [overall(analysis(**ARCHETYPES[i][1])) for i in (0, 1, 2)]
        genuine = [overall(analysis(**ARCHETYPES[i][1])) for i in (3, 4, 5)]
        assert min(generated) > max(genuine)


# =========================================================================== #
# 2. Weighting arithmetic
# =========================================================================== #
class TestWeightingArithmetic:
    def test_all_signals_applicable(self):
        a = analysis(S(0.8), S(0.6), S(0.4), S(0.2, "low"))
        expected = (W_F * 0.8 + W_N * 0.6 + W_G * 0.4 + L_EFF * 0.2) / (W_F + W_N + W_G + L_EFF)
        assert overall(a) == pytest.approx(expected, abs=1e-4)

    def test_no_portrait_moves_gradients_weight_to_noise_and_frequency(self):
        a = analysis(S(0.8), S(0.6), None, S(0.2, "low"))
        expected = (NP_F * 0.8 + NP_N * 0.6 + L_EFF * 0.2) / (NP_F + NP_N + L_EFF)
        assert overall(a) == pytest.approx(expected, abs=1e-4)
        sc = scoring(a)
        assert sc["gradients_weight_redistributed_to"] == ["noise", "frequency"]
        assert sc["effective_weights"]["noise"] / sc["effective_weights"]["frequency"] == pytest.approx(W_N / W_F, rel=1e-3)

    def test_no_portrait_and_no_noise_floor_gives_frequency_all_of_it(self):
        a = analysis(S(0.5), None, None, S(0.3, "low"))
        expected = ((W_F + W_G) * 0.5 + L_EFF * 0.3) / ((W_F + W_G) + L_EFF)
        assert overall(a) == pytest.approx(expected, abs=1e-4)
        assert scoring(a)["gradients_weight_redistributed_to"] == ["frequency"]

    def test_no_portrait_and_no_spectrum_gives_noise_all_of_it(self):
        a = analysis(None, S(0.5), None, S(0.3, "low"))
        expected = ((W_N + W_G) * 0.5 + L_EFF * 0.3) / ((W_N + W_G) + L_EFF)
        assert overall(a) == pytest.approx(expected, abs=1e-4)

    def test_portrait_weight_is_not_redistributed(self):
        a = analysis(S(0.5), S(0.5), S(0.5), S(0.5, "low"))
        assert scoring(a)["gradients_weight_redistributed_to"] == []

    def test_gradients_not_redistributed_to_lighting(self):
        # Only lighting left besides gradients: the weight is dropped, not
        # handed to the weakest signal.
        a = analysis(None, None, None, S(0.9, "low"))
        assert scoring(a)["gradients_weight_redistributed_to"] == []

    @pytest.mark.parametrize("confidence,factor", [("high", 1.0), ("medium", 1.0), ("low-medium", LOW_MEDIUM),
                                                   ("low", LOW), ("weird-value", LOW)])
    def test_confidence_scales_a_signals_weight(self, confidence, factor):
        a = analysis(S(1.0, confidence), S(0.0), S(0.0), S(0.0, "low"))
        expected = (W_F * factor * 1.0) / (W_F * factor + W_N + W_G + L_EFF)
        assert overall(a) == pytest.approx(expected, abs=1e-4)

    def test_missing_confidence_key_counts_as_low(self):
        freq = {"score": 1.0, "applicable": True, "note": "x"}
        a = analysis(freq, S(0.0), S(0.0), S(0.0, "low"))
        assert overall(a) == pytest.approx(W_F * LOW / (W_F * LOW + W_N + W_G + L_EFF), abs=1e-4)

    def test_not_applicable_placeholder_value_is_irrelevant(self):
        base = overall(analysis(S(0.7), S(0.3), NA(0.0), S(0.5, "low")))
        for placeholder in (0.5, 1.0, 0.123):
            assert overall(analysis(S(0.7), S(0.3), NA(placeholder), S(0.5, "low"))) == base

    def test_placeholder_half_would_not_be_neutral(self):
        # Documents *why* N/A signals are dropped: averaging a 0.5
        # placeholder in would lift a clean genuine photo toward the
        # 0.35 cut-off.
        clean = overall(analysis(S(0.0), S(0.0), NA(), S(0.0, "low")))
        averaged_in = overall(analysis(S(0.0), S(0.0), S(0.5), S(0.0, "low")))
        assert clean == 0.0 and averaged_in > 0.1

    def test_effective_weights_sum_to_one(self):
        for a in (analysis(S(0.1), S(0.2), S(0.3), S(0.4, "low")),
                  analysis(S(0.1), None, None, S(0.4, "low")),
                  analysis(None, S(0.2), S(0.3), None)):
            assert sum(scoring(a)["effective_weights"].values()) == pytest.approx(1.0, abs=1e-3)


# =========================================================================== #
# 3. Evidence sufficiency: cap, no-evidence, corroboration
# =========================================================================== #
class TestEvidenceSufficiency:
    @pytest.mark.parametrize("only", ["frequency", "noise", "gradients"])
    def test_single_substantive_signal_is_capped(self, only):
        kwargs = {only: S(1.0), "lighting": S(1.0, "low")}
        a = analysis(**kwargs)
        assert overall(a) == CAP
        assert scoring(a)["single_signal_cap_applied"] is True
        assert verdict(overall(a)) == "inconclusive"

    def test_cap_only_lowers(self):
        a = analysis(S(0.4), lighting=S(0.0, "low"))
        assert overall(a) == pytest.approx((0.5 * 0.4) / (0.5 + L_EFF), abs=1e-4)
        assert scoring(a)["single_signal_cap_applied"] is False

    def test_single_signal_can_still_say_real(self):
        assert verdict(overall(analysis(S(0.0), lighting=S(1.0, "low")))) == "real"

    def test_two_agreeing_signals_can_reach_likely_ai(self):
        a = analysis(S(1.0), S(1.0), None, S(0.5, "low"))
        assert verdict(overall(a)) == "ai"
        assert scoring(a)["single_signal_cap_applied"] is False

    def test_lighting_is_not_corroboration(self):
        assert overall(analysis(S(1.0), lighting=S(1.0, "low"))) == CAP

    @pytest.mark.parametrize("lighting", [None, S(0.0, "low"), S(1.0, "low")])
    def test_no_substantive_signal_is_inconclusive(self, lighting):
        a = analysis(lighting=lighting)
        assert overall(a) == NO_EVIDENCE
        assert scoring(a)["insufficient_evidence"] is True

    def test_disagreeing_evidence_lands_between(self):
        one_tell = overall(analysis(S(0.0), S(1.0), None, S(0.5, "low")))
        assert REAL_BELOW <= one_tell <= AI_ABOVE

    def test_adding_a_clean_signal_lowers_the_score(self):
        alone = overall(analysis(None, S(0.9), S(0.9), S(0.5, "low")))
        contradicted = overall(analysis(S(0.0), S(0.9), S(0.9), S(0.5, "low")))
        assert contradicted < alone

    def test_adding_an_agreeing_signal_lifts_a_capped_score(self):
        capped = overall(analysis(S(0.9), lighting=S(0.5, "low")))
        corroborated = overall(analysis(S(0.9), S(0.9), lighting=S(0.5, "low")))
        assert capped == CAP and corroborated > CAP


# =========================================================================== #
# 4. Robustness: malformed inputs, invariants, properties
# =========================================================================== #
class TestRobustness:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), None, "0.9", True, [0.9]])
    def test_malformed_score_is_not_applicable_and_never_leaks(self, bad):
        noise = {"score": bad, "confidence": "medium", "note": "x"}
        a = analysis(S(0.3), noise, S(0.3), S(0.3, "low"))
        score = overall(a)
        assert math.isfinite(score)
        assert "noise" not in scoring(a)["applicable_signals"]
        assert score == overall(analysis(S(0.3), None, S(0.3), S(0.3, "low")))

    def test_signal_without_applicable_key_counts_if_finite(self):
        legacy = {"score": 1.0, "confidence": "medium", "note": "pre-applicable analyzer"}
        a = analysis(S(0.0), legacy, None, None)
        assert "noise" in scoring(a)["applicable_signals"]

    @pytest.mark.parametrize("value", [-0.5, 1.7])
    def test_out_of_range_signal_scores_are_clipped(self, value):
        score = overall(analysis(S(value), S(value), S(value), S(value, "low")))
        assert 0.0 <= score <= 1.0

    def test_out_of_range_signal_cannot_outweigh_the_others(self):
        # Each signal is clipped to [0, 1] before weighting: a malformed 5.0
        # counts as 1.0, it does not drown three signals that read clean.
        a = analysis(S(0.0), S(5.0), S(0.0), S(0.0, "low"))
        assert overall(a) == pytest.approx(W_N * 1.0 / (W_F + W_N + W_G + L_EFF), abs=1e-4)
        b = analysis(S(1.0), S(-4.0), S(1.0), S(1.0, "low"))
        assert overall(b) == pytest.approx((W_F + W_G + L_EFF) / (W_F + W_N + W_G + L_EFF), abs=1e-4)

    def test_all_zero_and_all_one(self):
        assert overall(analysis(S(0.0), S(0.0), S(0.0), S(0.0, "low"))) == 0.0
        assert overall(analysis(S(1.0), S(1.0), S(1.0), S(1.0, "low"))) == 1.0

    def test_deterministic(self):
        a = analysis(S(0.37), S(0.61), None, S(0.22, "low"))
        assert len({overall(a) for _ in range(5)}) == 1

    def test_signal_order_does_not_matter(self):
        a = analysis(S(0.37), S(0.61), S(0.12), S(0.22, "low"))
        reordered = dict(a, signals={k: a["signals"][k] for k in reversed(SIGNALS)})
        assert overall(a) == overall(reordered)

    def test_input_is_not_mutated(self):
        a = analysis(S(0.37), S(0.61), None, S(0.22, "low"))
        snapshot = repr(a)
        main.score_image_signals(a, None)
        assert repr(a) == snapshot

    def test_random_inputs_stay_in_range_and_finite(self):
        rng = np.random.default_rng(0)
        confs = ["high", "medium", "low-medium", "low"]
        for _ in range(500):
            sigs = {}
            for name in SIGNALS:
                if rng.random() < 0.3:
                    sigs[name] = NA(float(rng.random()))
                else:
                    sigs[name] = S(float(rng.random()), confs[rng.integers(0, 4)])
            score = overall(analysis(**sigs))
            assert 0.0 <= score <= 1.0 and math.isfinite(score)

    @pytest.mark.parametrize("name", SIGNALS)
    def test_raising_any_signal_never_lowers_the_overall_score(self, name):
        # Monotonicity in each applicable signal's *value*, with applicability
        # held fixed (making a signal applicable is a different change: it
        # moves redistributed weight, see TestEvidenceSufficiency).
        rng = np.random.default_rng(SIGNALS.index(name))
        model = model_with("frequency", "compression", "lossless", 0.4)
        for _ in range(200):
            sigs = {n: S(float(rng.random()), "low" if n == "lighting" else "medium") for n in SIGNALS}
            if name != "gradients" and rng.random() < 0.5:
                sigs["gradients"] = NA()
            for m in (None, model):
                lower = overall(analysis(**sigs), m)
                bumped = dict(sigs)
                bumped[name] = S(min(1.0, sigs[name]["score"] + float(rng.random()) * 0.5),
                                 sigs[name]["confidence"])
                assert overall(analysis(**bumped), m) >= lower - 1e-9

    @pytest.mark.parametrize("value", [0.0, 0.3, 0.55, 1.0])
    def test_unanimous_signals_give_that_value(self, value):
        # A weighted mean of equal values is that value (unless capped).
        assert overall(analysis(S(value), S(value), S(value), S(value, "low"))) == pytest.approx(value, abs=1e-4)


# =========================================================================== #
# 5. Metadata
# =========================================================================== #
class TestMetadata:
    @pytest.mark.parametrize("visual_kwargs", [
        dict(frequency=S(0.0), noise=S(0.0)),                     # clean
        dict(frequency=S(0.5), noise=S(0.5)),                     # inconclusive
        dict(lighting=S(0.2, "low")),                             # no evidence
    ])
    def test_generator_signature_overrides_to_likely_ai(self, visual_kwargs):
        assert overall(analysis(metadata=SIGNATURE_META, **visual_kwargs)) == SIGNATURE

    def test_signature_never_lowers_a_higher_visual_score(self):
        a = analysis(S(1.0), S(1.0), S(1.0), S(1.0, "low"), metadata=SIGNATURE_META)
        assert overall(a) == 1.0

    @pytest.mark.parametrize("visual", [0.0, 0.3, 0.6, 0.9, 1.0])
    def test_camera_exif_blends(self, visual):
        a = analysis(S(visual), S(visual), S(visual), S(visual, "low"), metadata=CAMERA_META)
        assert overall(a) == pytest.approx(0.85 * visual + 0.15 * CAMERA_EXIF, abs=1e-4)

    def test_camera_exif_is_a_nudge_not_a_veto(self):
        strong = analysis(S(1.0), S(1.0), S(1.0), S(1.0, "low"), metadata=CAMERA_META)
        assert verdict(overall(strong)) == "ai"

    def test_camera_exif_on_no_evidence(self):
        a = analysis(lighting=S(0.9, "low"), metadata=CAMERA_META)
        assert overall(a) == pytest.approx(0.85 * NO_EVIDENCE + 0.15 * CAMERA_EXIF, abs=1e-4)

    def test_neutral_metadata_changes_nothing(self):
        sigs = dict(frequency=S(0.37), noise=S(0.61), lighting=S(0.22, "low"))
        visual = scoring(analysis(**sigs))["visual_score"]
        assert overall(analysis(**sigs)) == pytest.approx(visual, abs=1e-4)

    @pytest.mark.parametrize("confidence", ["low", "medium", "unavailable"])
    def test_only_high_and_low_medium_metadata_count(self, confidence):
        meta = {"score": 1.0, "confidence": confidence, "note": "x"}
        sigs = dict(frequency=S(0.2), noise=S(0.2))
        assert overall(analysis(metadata=meta, **sigs)) == overall(analysis(**sigs))


# =========================================================================== #
# 6. Context weighting (explicit models; the shipped file is tested above)
# =========================================================================== #
ALL_ONES = {"multipliers": {s: {d: {b: 1.0 for b in bs} for d, bs in {
    "compression": ("lossless", "jpeg_history", "jpeg_high", "jpeg_mid", "jpeg_low"),
    "size": ("small", "medium", "large"), "scale": ("native", "upscaled"),
    "color": ("color", "grayscale")}.items()} for s in SIGNALS}}


class TestContextWeighting:
    SIGS = dict(frequency=S(0.8), noise=S(0.4), lighting=S(0.2, "low"))

    def test_all_one_factors_equal_static(self):
        assert overall(analysis(**self.SIGS), ALL_ONES) == overall(analysis(**self.SIGS), None)

    def test_upweighting_a_signal(self):
        a = analysis(**self.SIGS)
        m = model_with("frequency", "compression", "lossless", 2.0)
        expected = (2 * NP_F * 0.8 + NP_N * 0.4 + L_EFF * 0.2) / (2 * NP_F + NP_N + L_EFF)
        assert overall(a, m) == pytest.approx(expected, abs=1e-4)
        assert scoring(a, m)["context_discount"] == 0.0

    def test_downweighting_is_an_absolute_discount(self):
        a = analysis(**self.SIGS)
        m = model_with("frequency", "compression", "lossless", 0.5)
        static_mass = NP_F + NP_N + L_EFF
        expected = (0.5 * NP_F * 0.8 + NP_N * 0.4 + L_EFF * 0.2) / static_mass
        assert overall(a, m) == pytest.approx(expected, abs=1e-4)
        assert scoring(a, m)["context_discount"] == pytest.approx(0.5 * NP_F / static_mass, abs=1e-3)

    def test_discount_can_move_a_lone_signal_out_of_inconclusive(self):
        a = analysis(frequency=S(0.8), lighting=None)
        assert verdict(overall(a)) == "inconclusive"
        m = model_with("frequency", "compression", "lossless", 0.25)
        assert overall(a, m) == pytest.approx(0.8 * 0.25, abs=1e-4)
        assert verdict(overall(a, m)) == "real"

    def test_factor_only_applies_in_its_bucket(self):
        m = model_with("frequency", "compression", "jpeg_low", 0.1)
        lossless = analysis(**self.SIGS)
        jpeg_low = analysis(**self.SIGS, buckets={**DEFAULT_BUCKETS, "compression": "jpeg_low"})
        assert overall(lossless, m) == overall(lossless, None)
        assert overall(jpeg_low, m) < overall(jpeg_low, None)

    def test_factors_multiply_across_dimensions(self):
        m = {"multipliers": {"frequency": {"compression": {"lossless": 2.0}, "scale": {"native": 0.5}}}}
        assert overall(analysis(**self.SIGS), m) == overall(analysis(**self.SIGS), None)

    def test_zeroed_only_signal_becomes_no_evidence(self):
        m = model_with("frequency", "compression", "lossless", 0.0)
        a = analysis(frequency=S(0.9), lighting=S(0.9, "low"))
        assert overall(a, m) == NO_EVIDENCE
        assert scoring(a, m)["insufficient_evidence"] is True

    def test_model_base_weights_are_used_and_normalised(self):
        m = {"multipliers": {}, "base_weights": {"frequency": 1, "noise": 1, "gradients": 1, "lighting": 1}}
        a = analysis(S(1.0), S(0.0), S(0.0), S(0.0, "low"))
        # equal base weights 0.25 each; lighting at "low" -> 0.125
        assert overall(a, m) == pytest.approx(0.25 / (0.25 * 3 + 0.125), abs=1e-4)

    def test_context_diagnostics(self):
        m = model_with("noise", "compression", "lossless", 1.5)
        sc = scoring(analysis(**self.SIGS), m)
        assert sc["context_weighting"] is True
        assert sc["context_multipliers"]["noise"] == 1.5
        assert sc["context"]["buckets"] == DEFAULT_BUCKETS
        assert scoring(analysis(**self.SIGS), None)["context_weighting"] is False

    def test_unknown_bucket_values_are_neutral(self):
        m = model_with("frequency", "compression", "lossless", 0.1)
        odd = analysis(**self.SIGS, buckets={"compression": "webp", "size": "?", "scale": None, "color": "cmyk"})
        assert overall(odd, m) == overall(odd, None)

    def test_default_model_is_the_shipped_file(self):
        a = analysis(**self.SIGS)
        assert main.score_image_signals(a)["ai_generated"] == overall(a, weighting.load_model())


# =========================================================================== #
# 7. Response shape and contract with the UI
# =========================================================================== #
class TestResponseContract:
    def test_response_shape(self):
        a = analysis(S(0.4), S(0.6), None, S(0.3, "low"))
        out = main.score_image_signals(a, None)
        assert set(out) == {"ai_generated", "signals", "scoring"}
        assert set(out["signals"]) == {*SIGNALS, "metadata"}
        assert out["scoring"]["context"] == a["context"]
        for key in ("visual_score", "insufficient_evidence", "single_signal_cap_applied",
                    "applicable_signals", "gradients_weight_redistributed_to",
                    "effective_weights", "context_multipliers", "context_discount"):
            assert key in out["scoring"]

    def test_rounded_to_four_decimals(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            v = overall(analysis(S(float(rng.random())), S(float(rng.random())), None, S(float(rng.random()), "low")))
            assert v == round(v, 4)

    def test_compact_image_response_is_only_the_score(self):
        out = main.score_image_signals(analysis(S(0.4), S(0.6)), None)
        assert main._compact_result(out, "png") == {"ai_generated": out["ai_generated"]}

    def test_band_constants_sit_in_the_inconclusive_band(self):
        assert REAL_BELOW <= CAP <= AI_ABOVE
        assert REAL_BELOW <= NO_EVIDENCE <= AI_ABOVE

    def test_ui_thresholds_match_this_spec(self):
        ts = Path(__file__).resolve().parents[1] / "web" / "lib" / "verdict.ts"
        if not ts.exists():
            pytest.skip("web UI not present")
        src = ts.read_text()
        real = re.search(r"REAL_MAX\s*:\s*([0-9.]+)", src)
        ai = re.search(r"AI_MIN\s*:\s*([0-9.]+)", src)
        assert real and ai, "verdict.ts no longer has the expected shape"
        assert float(real.group(1)) == REAL_BELOW
        assert float(ai.group(1)) == AI_ABOVE

    @pytest.mark.parametrize("score,band", [(0.0, "real"), (0.3499, "real"), (0.35, "inconclusive"),
                                            (0.65, "inconclusive"), (0.6501, "ai"), (1.0, "ai")])
    def test_band_boundaries(self, score, band):
        assert verdict(score) == band


# =========================================================================== #
# Exhaustive applicability grid: every combination of which signals apply
# =========================================================================== #
@pytest.mark.parametrize("mask", list(itertools.product([True, False], repeat=4)),
                         ids=lambda m: "+".join(s for s, on in zip(SIGNALS, m) if on) or "none")
def test_every_applicability_combination(mask):
    """For each of the 16 combinations: a valid score, the right evidence
    state, and a lighting-only or empty set never reading as a verdict."""
    on = dict(zip(SIGNALS, mask))
    sigs = {s: (S(0.9, "low" if s == "lighting" else "medium") if on[s] else None) for s in SIGNALS}
    a = analysis(**sigs)
    score = overall(a)
    sc = scoring(a)
    substantive = [s for s in ("noise", "frequency", "gradients") if on[s]]

    assert 0.0 <= score <= 1.0
    assert sc["insufficient_evidence"] is (not substantive)
    if not substantive:
        assert score == NO_EVIDENCE
    elif len(substantive) == 1:
        assert score == CAP  # 0.9 evidence from one signal is capped
    else:
        assert score == pytest.approx(0.9, abs=1e-4)
        assert verdict(score) == "ai"
    assert sc["gradients_weight_redistributed_to"] == (
        [] if on["gradients"] else [s for s in ("noise", "frequency") if on[s]]
    )
