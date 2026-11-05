"""
Context-dependent signal weights.

Each visual signal's base weight (``app.main.WEIGHTS``, or the calibrated
``base_weights`` below) is multiplied by one factor per context dimension
(``app.context.DIMENSIONS``: compression, size, scale, color):

    weight(signal, ctx) = base(signal) * prod_d factor(signal, d, bucket_d(ctx))

A factor above 1 means the signal separates generated from genuine images
*better* than its average in that bucket; below 1, worse; 0 means it did
not separate them at all there (or separated them in the wrong direction),
so it is ignored under those conditions.

The factors are not hand-set. ``eval/calibrate_weights.py`` builds a
synthetic batch of degraded variants (rescaled, JPEG-recompressed at several
qualities, cropped, grayscale, chained) from labeled sources, measures per
bucket how often each signal's tell survives on generated sources and how
often it false-alarms on genuine ones, and writes
``weighting_calibration.json`` next to this file. Dimensions confounded
with the label in the source set are not fitted. Dynamic factors are only
written if, under leave-one-source-out validation, they are non-inferior
to static weights and don't raise the false-positive rate; otherwise no
factors are written. See ``eval/REPORT.md`` for the last run.

Factors below 1 are applied as an absolute discount (see
``app.main._visual_score``): the discounted evidence counts as "no
evidence", it is not renormalised away.

With no calibration file every factor is 1, which reproduces the static
weighting exactly.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.context import DIMENSIONS

SIGNALS = ("frequency", "noise", "gradients", "lighting")
DEFAULT_MODEL_PATH = Path(__file__).with_name("weighting_calibration.json")

# Bounds on the product of all factors for one signal: no context may
# make a signal more than MAX_MULTIPLIER times its base weight.
MAX_MULTIPLIER = 2.5


@lru_cache(maxsize=8)
def _load(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with p.open() as fh:
        model = json.load(fh)
    if not isinstance(model, dict) or "multipliers" not in model:
        raise ValueError(f"{path}: not a weighting calibration file")
    return model


def load_model(path: str | Path | None = None) -> dict | None:
    """The calibration model, or None when no calibration file exists."""
    return _load(str(path or DEFAULT_MODEL_PATH))


def base_weights(model: dict | None, default: dict) -> dict:
    if model and isinstance(model.get("base_weights"), dict):
        weights = {s: float(model["base_weights"].get(s, default[s])) for s in SIGNALS}
        total = sum(weights.values())
        if total > 0:
            return {s: w / total for s, w in weights.items()}
    return dict(default)


def signal_multipliers(buckets: dict | None, model: dict | None) -> dict:
    """{signal: {"multiplier": float, "factors": {dimension: factor}}}.

    Unknown buckets or dimensions missing from the model contribute a
    factor of 1.
    """
    out = {}
    table = (model or {}).get("multipliers", {})
    for signal in SIGNALS:
        factors = {}
        product = 1.0
        for dim in DIMENSIONS:
            bucket = (buckets or {}).get(dim)
            factor = float(table.get(signal, {}).get(dim, {}).get(bucket, 1.0))
            factor = max(0.0, factor)
            factors[dim] = round(factor, 4)
            product *= factor
        out[signal] = {
            "multiplier": round(min(product, MAX_MULTIPLIER), 4),
            "factors": factors,
        }
    return out
