"""
The visual-signal combiner shared by every scoring path.

A standalone PNG/JPEG (``app.main``), a whole rasterised PDF page and an
image embedded in a PDF (``app.analyzers.document``) are all scored by
``visual_score``, so the same picture gets the same number whichever way it
reaches the detector. The PDF path adds its own gates on top (which signals
may run on a rendered page, flat fills, icon-sized rasters, what counts as
reliable page evidence) but not its own weights.

Rules, in order:
  - a signal counts only if it is applicable and has a finite score;
  - no face (gradients not applicable): gradients' base weight moves to
    noise and frequency, 0.40 : 0.25;
  - each weight is scaled by the signal's confidence and by its calibrated
    context factor (``app.weighting``); factors < 1 discount absolutely;
  - one substantive signal alone is capped at SINGLE_SIGNAL_CAP;
  - no substantive signal: NO_EVIDENCE_SCORE (callers may treat that as
    "insufficient" instead, as the PDF path does);
  - ``combine`` then folds in file metadata.
"""
from __future__ import annotations

import numpy as np

from app import weighting

# Base weights across the four visual signals (metadata is handled
# separately since a signature hit should dominate the score outright,
# not just average in).
#
#   noise      - stationary/white synthetic grain vs a patchy camera floor
#   frequency  - high-band knee left by generator upscaling
#   gradients  - airbrushed facial skin / rendered backdrop (portraits only)
#   lighting   - shading-direction consistency; weakest, a tie-breaker
#
# A signal that reports itself not applicable (``applicable: False``: no
# face for gradients, a noise floor erased by JPEG, an image too small
# for an FFT) is dropped from the average instead of contributing a
# placeholder 0.5, which is not neutral against a 0.35 "likely real"
# cut-off. When gradients does not apply (every non-portrait photo) its
# weight is redistributed to noise and frequency in proportion to their
# base weights, so those two carry the score.
WEIGHTS = {
    "frequency": 0.25,
    "noise": 0.40,
    "gradients": 0.25,
    "lighting": 0.10,
}

# Signals that receive the gradients weight when it is not applicable.
GRADIENT_FALLBACK_SIGNALS = ("noise", "frequency")

# With only one substantive signal applicable (lighting is too weak to
# count as corroboration), the score is capped inside the UI's
# "inconclusive" band (0.35-0.65): one cue on its own, e.g. a frequency
# knee that an upscaled real photo also shows, cannot produce "likely AI".
SUBSTANTIVE_SIGNALS = ("noise", "frequency", "gradients")
SINGLE_SIGNAL_CAP = 0.60
# With none of them applicable there is no evidence either way: report
# the middle of the "inconclusive" band rather than a lighting-only score.
NO_EVIDENCE_SCORE = 0.50

# Per-signal confidence scales the base weight. Lighting always reports
# "low", so its effective weight is half its base weight.
# Factor for a missing or unrecognised confidence string.
UNKNOWN_CONFIDENCE_FACTOR = 0.5

CONFIDENCE_FACTOR = {
    "high": 1.0,
    "medium": 1.0,
    "low-medium": 0.7,
    "low": 0.5,
}


def signal_applicable(result: dict) -> bool:
    """A signal contributes unless it says it does not apply or has no
    finite score."""
    if result.get("applicable") is False:
        return False
    score = result.get("score")
    return isinstance(score, (int, float)) and not isinstance(score, bool) and bool(np.isfinite(score))


_DEFAULT_MODEL = object()


def visual_score(
    signals: dict,
    buckets: dict | None = None,
    model=_DEFAULT_MODEL,
) -> tuple[float, dict]:
    """Confidence- and context-weighted mean over the applicable visual
    signals.

    Returns (score, scoring diagnostics). Not-applicable signals get zero
    weight; if gradients is not applicable its base weight moves to the
    GRADIENT_FALLBACK_SIGNALS that are applicable. ``buckets`` is the
    image context (``app.context``); each signal's weight is then scaled
    by its calibrated multiplier for that context (``app.weighting``).
    With ``buckets=None`` every multiplier is 1 (static weighting).
    """
    if model is _DEFAULT_MODEL:
        model = weighting.load_model()
    weights = weighting.base_weights(model, WEIGHTS)
    multipliers = weighting.signal_multipliers(buckets, model if buckets else None)

    base = dict(weights)
    applicable = {name: signal_applicable(signals[name]) for name in base}

    redistributed_to: list[str] = []
    if not applicable["gradients"]:
        targets = [n for n in GRADIENT_FALLBACK_SIGNALS if applicable[n]]
        pool = sum(weights[n] for n in targets)
        if pool > 0:
            for n in targets:
                base[n] += weights["gradients"] * weights[n] / pool
            redistributed_to = targets
        base["gradients"] = 0.0

    effective = {}
    static_mass = 0.0
    for name, weight in base.items():
        if not applicable[name]:
            effective[name] = 0.0
            continue
        factor = CONFIDENCE_FACTOR.get(str(signals[name].get("confidence", "low")), UNKNOWN_CONFIDENCE_FACTOR)
        static_mass += weight * factor
        effective[name] = weight * factor * multipliers[name]["multiplier"]

    total = sum(effective.values())
    # Down-weighting is absolute, not just a re-mix: if the context says the
    # available signals are less trustworthy than usual (multipliers < 1),
    # the evidence mass they lost counts as "no evidence" (score 0) instead
    # of being renormalised away. Otherwise a discounted signal that is
    # alone would still speak at full strength. With every multiplier at 1
    # this is exactly the static weighted mean.
    denominator = max(total, static_mass)
    # A signal whose context multiplier is 0 is applicable but carries no
    # weight here; it cannot corroborate anything.
    substantive = [n for n in SUBSTANTIVE_SIGNALS if applicable[n] and effective[n] > 0]
    insufficient = not substantive or total <= 1e-9
    if insufficient:
        # No substantive evidence (e.g. an icon-sized image): lighting alone
        # must not decide, and there is no basis for "likely real" either.
        score = NO_EVIDENCE_SCORE
    else:
        # Skip zero-weight signals outright: 0 * NaN is still NaN. Each
        # signal is clipped to [0, 1] so an out-of-range value cannot
        # outweigh the others.
        score = sum(
            effective[n] * min(max(float(signals[n]["score"]), 0.0), 1.0)
            for n in effective if effective[n] > 0
        ) / denominator
    uncapped = score

    capped = not insufficient and len(substantive) == 1 and score > SINGLE_SIGNAL_CAP
    if capped:
        score = SINGLE_SIGNAL_CAP

    diagnostics = {
        "visual_score": round(float(score), 4),
        "uncapped_score": round(float(uncapped), 4),
        "insufficient_evidence": insufficient,
        "single_signal_cap_applied": capped,
        "applicable_signals": [n for n in base if applicable[n]],
        "gradients_weight_redistributed_to": redistributed_to,
        "context_weighting": bool(buckets) and model is not None,
        "context_multipliers": {n: multipliers[n]["multiplier"] for n in multipliers},
        "effective_weights": {
            n: round(v / denominator, 4) if total > 1e-9 else 0.0 for n, v in effective.items()
        },
        # Share of the static evidence mass discounted by the context (0 when
        # no signal was down-weighted).
        "context_discount": round(1.0 - total / denominator, 4) if total > 1e-9 else 0.0,
    }
    return float(np.clip(score, 0.0, 1.0)), diagnostics


def combine(visual: float, meta_result: dict) -> float:
    """A strong metadata signature (explicit generator fingerprint)
    overrides rather than just blends in. A camera-EXIF signal pulls
    the score down moderately. Otherwise metadata stays neutral and
    doesn't affect the visual-signal average."""
    meta_score = meta_result["score"]
    if meta_result.get("confidence") == "high":
        combined = max(visual, meta_score)
    elif meta_result.get("confidence") == "low-medium":
        combined = 0.85 * visual + 0.15 * meta_score
    else:
        combined = visual
    return float(np.clip(combined, 0, 1))
