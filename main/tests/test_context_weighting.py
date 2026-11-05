"""
Tests for context-dependent weighting:

* ``app/context.py``   - observable context (format, JPEG quality, size,
                         upscaling, colour) and its buckets
* ``app/weighting.py`` - calibration model loading and per-signal factors
* ``app/main.py``      - how factors enter the combined score (absolute
                         down-weighting; all-1 factors == static weights)
* ``eval/calibrate_weights.py`` - the statistics the calibration decision
                         rests on (AUC, confound check, degradation-only fit,
                         leave-one-source-out isolation)
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app import context, main, weighting

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import calibrate_weights as cal  # noqa: E402

SIGNALS = ("frequency", "noise", "gradients", "lighting")


def sig(score, confidence="medium", applicable=True):
    return {"score": score, "confidence": confidence, "applicable": applicable, "note": "x"}


def na():
    return sig(0.5, "low", applicable=False)


def encode(arr: np.ndarray, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt, **kw)
    return buf.getvalue()


def model_with(signal: str, dim: str, bucket: str, factor: float) -> dict:
    return {"multipliers": {signal: {dim: {bucket: factor}}}}


BUCKETS_LOSSLESS = {"compression": "lossless", "size": "medium", "scale": "native", "color": "color"}


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
class TestContext:
    @pytest.mark.parametrize("quality", [50, 60, 75, 85, 90, 95])
    def test_jpeg_quality_is_read_from_the_quantisation_tables(self, quality):
        rng = np.random.default_rng(0)
        data = encode(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8), "JPEG", quality=quality)
        assert context.estimate_jpeg_quality(data) == quality

    def test_png_has_no_jpeg_quality(self):
        assert context.estimate_jpeg_quality(encode(np.zeros((8, 8, 3), np.uint8), "PNG")) is None

    @pytest.mark.parametrize("quality,bucket", [(95, "jpeg_high"), (90, "jpeg_high"), (89, "jpeg_mid"),
                                                (75, "jpeg_mid"), (74, "jpeg_low"), (None, "jpeg_mid")])
    def test_compression_bucket_for_jpeg(self, quality, bucket):
        assert context.compression_bucket("jpeg", quality, jpeg_grid=True) == bucket

    def test_png_bucket_depends_on_pixel_jpeg_history(self):
        assert context.compression_bucket("png", None, jpeg_grid=False) == "lossless"
        assert context.compression_bucket("png", None, jpeg_grid=True) == "jpeg_history"

    @pytest.mark.parametrize("mp,bucket", [(0.1, "small"), (0.34, "small"), (0.35, "medium"),
                                           (0.99, "medium"), (1.0, "large"), (12.0, "large")])
    def test_size_bucket(self, mp, bucket):
        assert context.size_bucket(mp) == bucket

    def test_scale_bucket_follows_the_frequency_knee(self):
        assert context.scale_bucket({"applicable": True, "high_band_knee_drop": 0.9}) == "upscaled"
        assert context.scale_bucket({"applicable": True, "high_band_knee_drop": 0.1}) == "native"
        assert context.scale_bucket({"applicable": False, "high_band_knee_drop": 0.9}) == "native"
        assert context.scale_bucket({}) == "native"

    def test_every_bucket_function_output_is_a_declared_bucket(self):
        for dim, fn_values in {
            "compression": [context.compression_bucket(f, q, g) for f in ("jpeg", "png")
                            for q in (None, 50, 80, 95) for g in (True, False)],
            "size": [context.size_bucket(m) for m in (0.01, 0.5, 5)],
            "scale": [context.scale_bucket({"high_band_knee_drop": k}) for k in (0.0, 1.0)],
        }.items():
            assert set(fn_values) <= set(context.BUCKETS[dim])

    def test_full_pipeline_reports_context(self):
        rng = np.random.default_rng(1)
        rgb = np.clip(128 + rng.normal(0, 20, (300, 400, 3)), 0, 255).astype(np.uint8)
        result = main._run_image_analysis(encode(rgb, "JPEG", quality=70))
        ctx = result["scoring"]["context"]
        assert ctx["format"] == "jpeg"
        assert ctx["jpeg_quality_estimate"] == 70
        assert (ctx["width"], ctx["height"]) == (400, 300)
        assert ctx["buckets"]["compression"] == "jpeg_low"
        assert ctx["buckets"]["size"] == "small"
        assert ctx["buckets"]["color"] == "color"

    def test_grayscale_is_detected(self):
        gray = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
        result = main._run_image_analysis(encode(gray, "PNG"))
        assert result["scoring"]["context"]["buckets"]["color"] == "grayscale"


# --------------------------------------------------------------------------- #
# Weighting module
# --------------------------------------------------------------------------- #
class TestWeighting:
    def test_no_model_means_all_factors_one(self):
        out = weighting.signal_multipliers(BUCKETS_LOSSLESS, None)
        assert all(v["multiplier"] == 1.0 for v in out.values())

    def test_factors_multiply_across_dimensions(self):
        model = {"multipliers": {"noise": {"compression": {"lossless": 1.5}, "size": {"medium": 0.5}}}}
        out = weighting.signal_multipliers(BUCKETS_LOSSLESS, model)
        assert out["noise"]["multiplier"] == pytest.approx(0.75)
        assert out["noise"]["factors"]["compression"] == 1.5
        assert out["frequency"]["multiplier"] == 1.0

    def test_product_is_capped(self):
        model = {"multipliers": {"noise": {d: {b: 2.0} for d, b in BUCKETS_LOSSLESS.items()}}}
        assert weighting.signal_multipliers(BUCKETS_LOSSLESS, model)["noise"]["multiplier"] == weighting.MAX_MULTIPLIER

    def test_negative_factor_is_clamped(self):
        model = model_with("noise", "compression", "lossless", -3.0)
        assert weighting.signal_multipliers(BUCKETS_LOSSLESS, model)["noise"]["multiplier"] == 0.0

    def test_missing_file_loads_none(self, tmp_path):
        assert weighting.load_model(tmp_path / "absent.json") is None

    def test_malformed_file_is_rejected(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"hello": 1}))
        with pytest.raises(ValueError):
            weighting.load_model(p)

    def test_shipped_calibration_is_well_formed(self):
        model = weighting.load_model()
        if model is None:
            pytest.skip("no calibration file shipped")
        assert set(model["multipliers"]) <= set(SIGNALS)
        for per_dim in model["multipliers"].values():
            for dim, buckets in per_dim.items():
                assert dim in context.DIMENSIONS
                assert set(buckets) <= set(context.BUCKETS[dim])
                assert all(cal.FACTOR_MIN <= f <= cal.FACTOR_MAX for f in buckets.values())
        # Only dimensions the calibrator judged unconfounded may be used.
        for dim in model.get("dimensions", []):
            assert model["confound_tv"][dim] <= cal.MAX_CONFOUND_TV

    def test_base_weights_from_model_are_normalised(self):
        w = weighting.base_weights({"base_weights": {s: 2.0 for s in SIGNALS}}, main.WEIGHTS)
        assert sum(w.values()) == pytest.approx(1.0)
        assert weighting.base_weights(None, main.WEIGHTS) == main.WEIGHTS


# --------------------------------------------------------------------------- #
# Combiner with context
# --------------------------------------------------------------------------- #
class TestContextCombiner:
    SIGS = {"frequency": sig(0.8), "noise": sig(0.6), "gradients": na(), "lighting": sig(0.3, "low")}

    def test_all_one_factors_equal_static(self):
        ones = {"multipliers": {s: {d: {b: 1.0 for b in context.BUCKETS[d]} for d in context.DIMENSIONS}
                                for s in SIGNALS}}
        dyn, _ = main._visual_score(self.SIGS, BUCKETS_LOSSLESS, ones)
        static, _ = main._visual_score(self.SIGS, None, None)
        assert dyn == pytest.approx(static)

    def test_upweighting_shifts_the_mix_toward_that_signal(self):
        static, _ = main._visual_score(self.SIGS, None, None)
        up, diag = main._visual_score(self.SIGS, BUCKETS_LOSSLESS, model_with("frequency", "compression", "lossless", 2.0))
        assert up > static  # frequency (0.8) is the highest score
        assert diag["context_discount"] == 0.0
        assert diag["context_weighting"] is True

    def test_downweighting_is_absolute_not_renormalised(self):
        lone = {"frequency": sig(0.8), "noise": na(), "gradients": na(), "lighting": na()}
        static, _ = main._visual_score(lone, None, None)
        down, diag = main._visual_score(lone, BUCKETS_LOSSLESS, model_with("frequency", "compression", "lossless", 0.25))
        # A renormalising mean would give the lone signal its full 0.8 (capped
        # at 0.6) regardless of the factor.
        assert down == pytest.approx(0.8 * 0.25)
        assert down < static
        assert diag["context_discount"] == pytest.approx(0.75)

    def test_context_is_ignored_without_buckets(self):
        m = model_with("frequency", "compression", "lossless", 0.1)
        a, diag = main._visual_score(self.SIGS, None, m)
        b, _ = main._visual_score(self.SIGS, None, None)
        assert a == pytest.approx(b)
        assert diag["context_weighting"] is False

    def test_score_stays_in_unit_interval(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            sigs = {s: sig(float(rng.random())) for s in SIGNALS}
            m = {"multipliers": {s: {"compression": {"lossless": float(rng.uniform(0, 2))}} for s in SIGNALS}}
            score, _ = main._visual_score(sigs, BUCKETS_LOSSLESS, m)
            assert 0.0 <= score <= 1.0


# --------------------------------------------------------------------------- #
# Calibrator statistics
# --------------------------------------------------------------------------- #
def record(source, label, scores, buckets=None, transform=None):
    return {
        "source": source,
        "label": label,
        "transform": transform or {"family": "grid", "scale": 1.0, "save": "png"},
        "signals": {s: sig(scores.get(s, 0.0)) for s in SIGNALS},
        "metadata": {"score": 0.5, "confidence": "low"},
        "context": {"buckets": buckets or dict(BUCKETS_LOSSLESS), "megapixels": 0.5},
    }


class TestCalibrator:
    def test_auc_matches_brute_force_with_ties(self):
        rng = np.random.default_rng(4)
        scores = rng.integers(0, 5, 60).astype(float)
        labels = rng.integers(0, 2, 60)
        pos, neg = scores[labels == 1], scores[labels == 0]
        brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
        assert cal.auc(scores, labels) == pytest.approx(brute)

    def test_auc_degenerate(self):
        assert cal.auc([1, 2, 3], [1, 1, 1]) == 0.5

    def test_confound_check(self):
        independent = [record("a", 1, {}, {**BUCKETS_LOSSLESS, "size": b}) for b in ("small", "large")] + \
                      [record("b", 0, {}, {**BUCKETS_LOSSLESS, "size": b}) for b in ("small", "large")]
        assert cal.confound_tv(independent)["size"] == pytest.approx(0.0)
        confounded = [record("a", 1, {}, {**BUCKETS_LOSSLESS, "size": "large"}),
                      record("b", 0, {}, {**BUCKETS_LOSSLESS, "size": "small"})]
        assert cal.confound_tv(confounded)["size"] == pytest.approx(1.0)

    def test_fit_learns_degradation_not_tells(self):
        # Generated source "g" fires frequency on its lossless reference and
        # keeps it under jpeg_high, loses it under jpeg_low. Genuine sources
        # false-alarm under jpeg_low only.
        recs = [record("g", 1, {"frequency": 0.9})]
        recs += [record("g", 1, {"frequency": 0.9}, {**BUCKETS_LOSSLESS, "compression": "jpeg_high"},
                        {"family": "grid", "scale": 1.0, "save": "jpeg95"}) for _ in range(10)]
        recs += [record("g", 1, {"frequency": 0.0}, {**BUCKETS_LOSSLESS, "compression": "jpeg_low"},
                        {"family": "grid", "scale": 1.0, "save": "jpeg60"}) for _ in range(10)]
        recs += [record(f"r{i}", 0, {"frequency": 0.0}, {**BUCKETS_LOSSLESS, "compression": "jpeg_high"}) for i in range(10)]
        recs += [record(f"r{i}", 0, {"frequency": 0.9}, {**BUCKETS_LOSSLESS, "compression": "jpeg_low"}) for i in range(10)]
        f = cal.fit_model(recs, ("compression",))["multipliers"]["frequency"]["compression"]
        assert f["jpeg_high"] > 1.0 > f["jpeg_low"]
        assert f["jpeg_low"] == pytest.approx(cal.FACTOR_MIN)

    def test_isotonic_fit(self):
        assert cal._isotonic_nonincreasing([1.0, 0.5, 0.8, 0.1], [1, 1, 1, 1]) == pytest.approx([1.0, 0.65, 0.65, 0.1])
        assert cal._isotonic_nonincreasing([0.2, 0.4], [3, 1]) == pytest.approx([0.25, 0.25])
        assert cal._isotonic_nonincreasing([0.9, 0.5, 0.1], [1, 1, 1]) == [0.9, 0.5, 0.1]

    def test_compression_factors_never_rise_as_quality_drops(self):
        # Noisy data where jpeg_mid looks better than jpeg_high.
        def recs_for(bucket, fa_hits):
            return [record(f"r{i}", 0, {"frequency": 0.9 if i < fa_hits else 0.0},
                           {**BUCKETS_LOSSLESS, "compression": bucket}) for i in range(20)]
        recs = [record("g", 1, {"frequency": 0.9})]
        recs += recs_for("lossless", 2) + recs_for("jpeg_high", 8) + recs_for("jpeg_mid", 1) + recs_for("jpeg_low", 6)
        f = cal.fit_model(recs, ("compression",))["multipliers"]["frequency"]["compression"]
        assert f["lossless"] >= f["jpeg_high"] >= f["jpeg_mid"] >= f["jpeg_low"]

    def test_signal_no_generated_source_fires_keeps_retention_prior(self):
        recs = [record("g", 1, {"noise": 0.0}), record("r", 0, {"noise": 0.0})]
        stats = cal.fit_model(recs, ("compression",))["_stats"]["noise"]["_all"]
        assert stats["firing_sources"] == []
        assert stats["retention"] == 1.0

    def test_frequency_gets_no_scale_factor(self):
        recs = [record("g", 1, {"frequency": 0.9}), record("r", 0, {"frequency": 0.0})]
        assert "scale" not in cal.fit_model(recs, ("scale",))["multipliers"]["frequency"]

    def test_loso_never_fits_on_the_held_out_source(self, monkeypatch):
        recs = [record(s, lab, {"noise": 0.5}) for s, lab in (("a", 1), ("b", 0), ("c", 0))]
        seen = []
        real_fit = cal.fit_model

        def spy(train, dims):
            seen.append({r["source"] for r in train})
            return real_fit(train, dims)

        monkeypatch.setattr(cal, "fit_model", spy)
        cal.loso_scores(recs, ("compression",))
        assert seen == [{"b", "c"}, {"a", "c"}, {"a", "b"}]

    def test_bootstrap_interval_contains_zero_for_identical_scores(self):
        recs = [record(s, lab, {}) for s, lab in (("a", 1), ("b", 1), ("c", 0), ("d", 0))]
        scores = np.array([0.9, 0.2, 0.1, 0.5])
        lo, hi = cal.bootstrap_diff(scores, scores.copy(), recs)
        assert lo == hi == 0.0

    def test_variants_cover_the_declared_grid(self, tmp_path):
        p = tmp_path / "src.png"
        Image.fromarray(np.full((200, 300, 3), 128, np.uint8)).save(p)
        variants = cal.variants_for(str(p))
        assert len(variants) == len(cal.SCALES) * len(cal.SAVES) + 6
        for t, data in variants:
            fmt = "png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "jpeg"
            assert fmt == ("png" if t["save"] == "png" else "jpeg")
