#!/usr/bin/env python3
"""
Calibrate context-dependent signal weights on a synthetic degradation batch.

    python eval/calibrate_weights.py                  # build, extract, fit, report
    python eval/calibrate_weights.py --reuse          # reuse extracted signals
    python eval/calibrate_weights.py --ai-dir DIR --real-dir DIR   # add your own sources
    python eval/calibrate_weights.py --dry-run        # report only, don't write the JSON

Pipeline
--------
1. Sources. Labeled images: generated (the baseline PNGs plus --ai-dir)
   and genuine (baseline/jpg/real-photo.jpeg, the sample photos bundled with
   scikit-image / scikit-learn / matplotlib when those packages are
   installed, plus --real-dir). No new generated images are synthesised:
   a fake "AI image" made by this script would only contain whatever
   artefacts the script put there, and the detector would be graded on
   its own assumptions.
2. Synthetic batch. Every source is put through the same grid of
   degradations a picture meets in the wild: rescaling (0.5x-2x),
   PNG / JPEG at quality 95-60, centre crops, grayscale conversion,
   resize-after-JPEG and double JPEG.
3. Extraction. Each variant runs through ``app.main.analyze_image_signals``
   once; signals and the observable context (``app.context``) are cached.
4. Confound check. A context dimension is only fitted if the batch assigns
   its buckets independently of the label (total-variation distance between
   the generated and genuine bucket distributions <= MAX_CONFOUND_TV).
   With the bundled sources, size and color fail this: the genuine library
   photos are natively small and partly grayscale while the generated ones
   are ~1 MP colour, so "bigger means AI" would be learned (megapixels
   alone reach AUC ~0.80 on the batch, better than the detector).
5. Fit. Only degradation behaviour is learned, never *which* signal
   indicates AI: with a handful of generators every generator has its own
   tell, and any model that learns tells from some generators inverts on
   the next one (an unconstrained logistic regression on these signals
   drops from AUC 0.81 in-sample to 0.05 held-out). Per signal and bucket:
     retention   = P(signal still fires | generated source whose untouched
                   original fires it, bucket)       -- within-source
     false alarm = P(signal fires | genuine, bucket)
     quality     = max(0, retention - false alarm)
   Bucket rates are shrunk toward the overall rate for sparse buckets.
   Along JPEG quality (lossless > high > mid > low) quality is constrained
   to be non-increasing (weighted isotonic regression): more quantisation
   never restores detail, so an apparent improvement at lower quality is
   sampling noise. The bucket factor is quality(bucket) / quality(all
   buckets), bounded to [FACTOR_MIN, FACTOR_MAX].
   "Fires" means score >= 0.35, the UI's "likely real" cut-off. The
   "scale" bucket is read off the frequency knee, so frequency gets no
   scale factor.
6. Validation. Leave-one-source-out: for every source, factors are fit on
   all *other* sources and the held-out source's variants are scored with
   them. Variants of one picture are near-duplicates; any split that lets
   a picture's variants sit on both sides would grade the fit on its own
   training data. Uncertainty: bootstrap over sources (stratified by
   label) of the balanced-accuracy difference against static weights.
7. Decision. A dynamic candidate (one per subset of the fitted dimensions)
   is adopted only if, on held-out sources, its balanced accuracy at the
   0.35 cut-off is not below static, the 95% bootstrap lower bound of the
   difference is above -NONINFERIORITY_MARGIN, and its false-positive
   rate is not higher. Otherwise all factors are written as 1 (static).
   ``app/weighting_calibration.json`` and ``eval/REPORT.md`` record the
   numbers either way.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from app import context as ctxmod  # noqa: E402
from app import main as app_main  # noqa: E402
from app import weighting  # noqa: E402

BASELINE_DIR = MAIN_DIR.parent / "baseline"
DEFAULT_WORKDIR = MAIN_DIR / "eval" / ".work"
CALIBRATION_PATH = weighting.DEFAULT_MODEL_PATH
REPORT_PATH = MAIN_DIR / "eval" / "REPORT.md"

REAL_CUTOFF = 0.35     # web/lib/verdict.ts
AI_CUTOFF = 0.65

SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)
SAVES = (("png", None), ("jpeg", 95), ("jpeg", 90), ("jpeg", 80), ("jpeg", 70), ("jpeg", 60))

# Fitting constants.
SHRINK_M = 8.0          # m-estimate pseudo-samples pulling bucket rates to the overall rate
FACTOR_MIN = 0.1        # a signal is down-weighted, never silently removed
FACTOR_MAX = 2.0
MAX_CONFOUND_TV = 0.15  # max label/bucket association for a dimension to be fitted
NONINFERIORITY_MARGIN = 0.01
# Buckets along which a signal's trustworthiness must be non-increasing
# (physics, not data: more quantisation never restores detail).
# jpeg_history (JPEG traces in a PNG, quality unknown) is left unordered.
MONOTONE_ORDER = {"compression": ("lossless", "jpeg_high", "jpeg_mid", "jpeg_low")}
BOOTSTRAP_ROUNDS = 2000
# Below this many genuine sources the fit is meaningless (leave-one-source-out
# with one or two real photos), and running it would overwrite the committed
# calibration with it. The sample-photo packages (scikit-image & co.) need
# internet to install and may have no wheel for the desktop's Python, so a
# machine without them ends up with only baseline/jpg/real-photo.jpeg.
MIN_GENUINE_SOURCES = 5

SIGNALS = weighting.SIGNALS
# The scale bucket is read off the frequency knee; fitting frequency's own
# weight to it would be circular.
EXCLUDED = {("frequency", "scale")}


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def _library_photos() -> list[Path]:
    """Genuine photographs bundled with common scientific Python packages."""
    wanted = {
        "skimage": ["data/astronaut.png", "data/brick.png", "data/camera.png", "data/chelsea.png",
                    "data/coins.png", "data/grass.png", "data/gravel.png",
                    "data/moon.png", "data/motorcycle_left.png", "data/motorcycle_right.png",
                    "data/rocket.jpg"],
        "sklearn": ["datasets/images/china.jpg", "datasets/images/flower.jpg"],
        "matplotlib": ["mpl-data/sample_data/grace_hopper.jpg"],
    }
    found = []
    for package, rels in wanted.items():
        try:
            mod = __import__(package)
        except Exception:
            continue
        root = Path(mod.__file__).parent
        found += [root / r for r in rels if (root / r).exists()]
    return found


def _images_in(directory: str | None) -> list[Path]:
    if not directory:
        return []
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in exts)


def collect_sources(args) -> list[dict]:
    ai = [BASELINE_DIR / "png" / n for n in ("fireplace-generated.png", "ai-generated.png", "ai-retouched.png")]
    real = [BASELINE_DIR / "jpg" / "real-photo.jpeg"]
    if not args.no_library:
        real += _library_photos()
    ai += _images_in(args.ai_dir)
    real += _images_in(args.real_dir)
    sources = [{"path": str(p), "label": 1} for p in ai if p.exists()]
    sources += [{"path": str(p), "label": 0} for p in real if p.exists()]
    seen = set()
    for s in sources:
        stem = Path(s["path"]).stem
        name, i = stem, 2
        while name in seen:
            name, i = f"{stem}_{i}", i + 1
        seen.add(name)
        s["name"] = name
    return sources


# --------------------------------------------------------------------------- #
# Synthetic batch
# --------------------------------------------------------------------------- #
def _resize(im: Image.Image, factor: float) -> Image.Image:
    if factor == 1.0:
        return im
    size = (max(1, round(im.width * factor)), max(1, round(im.height * factor)))
    return im.resize(size, Image.LANCZOS if factor < 1 else Image.BICUBIC)


def _save(im: Image.Image, fmt: str, quality: int | None) -> bytes:
    buf = io.BytesIO()
    if fmt == "png":
        im.save(buf, "PNG")
    else:
        im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _decode(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def variants_for(path: str) -> list[tuple[dict, bytes]]:
    """(transform description, encoded bytes) for one source image."""
    original = Image.open(path).convert("RGB")
    out = []
    for scale in SCALES:
        scaled = _resize(original, scale)
        for fmt, q in SAVES:
            out.append(({"family": "grid", "scale": scale, "save": fmt if q is None else f"jpeg{q}"},
                        _save(scaled, fmt, q)))

    side = min(320, original.width, original.height)
    left, top = (original.width - side) // 2, (original.height - side) // 2
    crop = original.crop((left, top, left + side, top + side))
    out.append(({"family": "crop320", "scale": 1.0, "save": "png"}, _save(crop, "png", None)))
    out.append(({"family": "crop320", "scale": 1.0, "save": "jpeg85"}, _save(crop, "jpeg", 85)))

    gray = original.convert("L").convert("RGB")
    out.append(({"family": "grayscale", "scale": 1.0, "save": "png"}, _save(gray, "png", None)))
    out.append(({"family": "grayscale", "scale": 1.0, "save": "jpeg85"}, _save(gray, "jpeg", 85)))

    resized_after = _resize(_decode(_save(original, "jpeg", 90)), 0.75)
    out.append(({"family": "jpeg90_then_resize0.75", "scale": 0.75, "save": "png"}, _save(resized_after, "png", None)))
    double = _decode(_save(original, "jpeg", 85))
    out.append(({"family": "double_jpeg85_75", "scale": 1.0, "save": "jpeg75"}, _save(double, "jpeg", 75)))
    return out


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _slim(result: dict) -> dict:
    keep = ("score", "confidence", "applicable", "note", "high_band_knee_drop", "floor_state")
    return {k: result.get(k) for k in keep if k in result}


def _extract(job: tuple[dict, dict, bytes]) -> dict:
    source, transform, data = job
    analysis = app_main.analyze_image_signals(data)
    return {
        "source": source["name"],
        "label": source["label"],
        "transform": transform,
        "signals": {k: _slim(v) for k, v in analysis["signals"].items()},
        "metadata": _slim(analysis["metadata"]),
        "context": analysis["context"],
    }


def extract(sources: list[dict], workdir: Path, workers: int) -> list[dict]:
    jobs = []
    for src in sources:
        for transform, data in variants_for(src["path"]):
            jobs.append((src, transform, data))
    print(f"synthetic batch: {len(jobs)} variants from {len(sources)} sources", flush=True)
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rec in enumerate(pool.map(_extract, jobs, chunksize=4), 1):
            records.append(rec)
            if i % 100 == 0:
                print(f"  extracted {i}/{len(jobs)}", flush=True)
    workdir.mkdir(parents=True, exist_ok=True)
    with (workdir / "signals.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return records


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def auc(scores, labels) -> float:
    """Mann-Whitney AUC with average ranks for ties; 0.5 if a class is empty."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    npos, nneg = int(labels.sum()), int((1 - labels).sum())
    if npos == 0 or nneg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def reliability(a: float) -> float:
    return max(0.0, 2.0 * a - 1.0)


def _applicable(rec: dict, signal: str) -> bool:
    return app_main._signal_applicable(rec["signals"][signal])


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def _fires(rec: dict, signal: str) -> bool:
    return float(rec["signals"][signal]["score"]) >= REAL_CUTOFF


def _is_reference(rec: dict) -> bool:
    t = rec["transform"]
    return t["family"] == "grid" and t["scale"] == 1.0 and t["save"] == "png"


def confound_tv(records: list[dict]) -> dict:
    """Total-variation distance between P(bucket | generated) and
    P(bucket | genuine), per dimension. 0 = assigned independently of the
    label; 1 = the bucket alone gives the label away."""
    out = {}
    for dim in ctxmod.DIMENSIONS:
        dist = {}
        for label in (0, 1):
            rows = [r for r in records if r["label"] == label]
            dist[label] = np.array([
                np.mean([r["context"]["buckets"][dim] == b for r in rows]) if rows else 0.0
                for b in ctxmod.BUCKETS[dim]
            ])
        out[dim] = float(0.5 * np.abs(dist[0] - dist[1]).sum())
    return out


def fit_model(records: list[dict], dims: tuple[str, ...]) -> dict:
    """Per-signal, per-bucket factors from degradation behaviour only."""
    multipliers: dict = {}
    stats: dict = {}
    for signal in SIGNALS:
        genuine = [r for r in records if r["label"] == 0 and _applicable(r, signal)]
        firing_sources = {r["source"] for r in records
                          if r["label"] == 1 and _is_reference(r) and _applicable(r, signal) and _fires(r, signal)}
        generated = [r for r in records
                     if r["label"] == 1 and r["source"] in firing_sources and _applicable(r, signal)]
        fa_all = float(np.mean([_fires(r, signal) for r in genuine])) if genuine else 0.0
        ret_all = float(np.mean([_fires(r, signal) for r in generated])) if generated else 1.0
        q_all = max(ret_all - fa_all, 1e-3)
        multipliers[signal], stats[signal] = {}, {
            "_all": {"retention": round(ret_all, 3), "false_alarm": round(fa_all, 3),
                     "n_generated": len(generated), "n_genuine": len(genuine),
                     "firing_sources": sorted(firing_sources)}}
        for dim in dims:
            if (signal, dim) in EXCLUDED:
                continue
            multipliers[signal][dim], stats[signal][dim] = {}, {}
            quality, counts = {}, {}
            for bucket in ctxmod.BUCKETS[dim]:
                g = [_fires(r, signal) for r in genuine if r["context"]["buckets"][dim] == bucket]
                a = [_fires(r, signal) for r in generated if r["context"]["buckets"][dim] == bucket]
                fa = (sum(g) + SHRINK_M * fa_all) / (len(g) + SHRINK_M)
                ret = (sum(a) + SHRINK_M * ret_all) / (len(a) + SHRINK_M)
                quality[bucket] = max(ret - fa, 0.0)
                counts[bucket] = len(g) + len(a)
                stats[signal][dim][bucket] = {"retention": round(ret, 3), "false_alarm": round(fa, 3),
                                              "n_generated": len(a), "n_genuine": len(g)}
            ordered = MONOTONE_ORDER.get(dim)
            if ordered:
                # Lower JPEG quality can only destroy more: a signal cannot
                # become more trustworthy as quality drops.
                fitted = _isotonic_nonincreasing([quality[b] for b in ordered], [counts[b] for b in ordered])
                quality.update(dict(zip(ordered, fitted)))
            for bucket, q in quality.items():
                factor = float(np.clip(q / q_all, FACTOR_MIN, FACTOR_MAX))
                multipliers[signal][dim][bucket] = round(factor, 4)
    return {"multipliers": multipliers, "_stats": stats}


def _isotonic_nonincreasing(values: list[float], weights: list[float]) -> list[float]:
    """Weighted pool-adjacent-violators fit of a non-increasing sequence."""
    blocks = [[v, max(w, 1e-9), 1] for v, w in zip(values, weights)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:  # violation: later bucket better than earlier
            v1, w1, n1 = blocks[i]
            v2, w2, n2 = blocks[i + 1]
            blocks[i] = [(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, _, n in blocks:
        out += [v] * n
    return out


STATIC = {"multipliers": {}}


def score_records(records: list[dict], model: dict | None) -> np.ndarray:
    out = []
    for r in records:
        visual, _ = app_main._visual_score(r["signals"], r["context"]["buckets"], model)
        out.append(app_main._combine(visual, r["metadata"]))
    return np.array(out)


def loso_scores(records: list[dict], dims: tuple[str, ...] | None) -> np.ndarray:
    """Held-out scores; dims=None means static weights."""
    if dims is None:
        return score_records(records, STATIC)
    scores = np.zeros(len(records))
    for held in sorted({r["source"] for r in records}):
        model = fit_model([r for r in records if r["source"] != held], dims)
        idx = [i for i, r in enumerate(records) if r["source"] == held]
        scores[idx] = score_records([records[i] for i in idx], model)
    return scores


def summarize(scores: np.ndarray, labels: np.ndarray) -> dict:
    ai, real = scores[labels == 1], scores[labels == 0]
    tpr = float(np.mean(ai >= REAL_CUTOFF)) if len(ai) else float("nan")
    fpr = float(np.mean(real >= REAL_CUTOFF)) if len(real) else float("nan")
    return {
        "auc": auc(scores, labels) if len(ai) and len(real) else float("nan"),
        "tpr": tpr,
        "fpr": fpr,
        "balanced_accuracy": (tpr + (1 - fpr)) / 2 if len(ai) and len(real) else float("nan"),
        "ai_likely_ai": float(np.mean(ai > AI_CUTOFF)) if len(ai) else float("nan"),
        "real_likely_ai": float(np.mean(real > AI_CUTOFF)) if len(real) else float("nan"),
        "n_ai": int(len(ai)),
        "n_real": int(len(real)),
    }


def _balanced(scores: np.ndarray, labels: np.ndarray) -> float:
    return (np.mean(scores[labels == 1] >= REAL_CUTOFF) + np.mean(scores[labels == 0] < REAL_CUTOFF)) / 2


def bootstrap_diff(candidate: np.ndarray, static: np.ndarray, records: list[dict], seed: int = 0) -> tuple[float, float]:
    """95% interval of balanced-accuracy(candidate) - balanced-accuracy(static),
    resampling whole sources within each label."""
    labels = np.array([r["label"] for r in records])
    sources = np.array([r["source"] for r in records])
    by_label = {lab: sorted({s for s, l in zip(sources, labels) if l == lab}) for lab in (0, 1)}
    index = {s: np.flatnonzero(sources == s) for s in set(sources)}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(BOOTSTRAP_ROUNDS):
        pick = [s for lab in (0, 1) for s in rng.choice(by_label[lab], len(by_label[lab]))]
        idx = np.concatenate([index[s] for s in pick])
        diffs.append(_balanced(candidate[idx], labels[idx]) - _balanced(static[idx], labels[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def _f(x: float) -> str:
    return "n/a" if x != x else f"{x:.3f}"


def _group_key(rec: dict, how: str) -> str:
    t = rec["transform"]
    if how == "transform":
        return t["family"] if t["family"] != "grid" else f"scale {t['scale']} / {t['save']}"
    if how == "save":
        return t["save"] if t["family"] == "grid" else f"{t['family']} ({t['save']})"
    if how == "scale":
        return f"{t['scale']}x" if t["family"] == "grid" else t["family"]
    return rec["context"]["buckets"][how]


def _cv_table(add, cv: dict, chosen: str) -> None:
    add("| weighting | AUC | generated flagged | genuine flagged (FPR) | balanced acc. | gain vs static [95% CI] "
        "| generated \"likely AI\" | genuine \"likely AI\" |")
    add("|---|---|---|---|---|---|---|---|")
    for name, res in cv.items():
        m = res["overall"]
        mark = " **(chosen)**" if name == chosen else ""
        ci = res.get("ci")
        gain = "—" if ci is None else f"{res['gain']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        add(f"| {name}{mark} | {_f(m['auc'])} | {_pct(m['tpr'])} | {_pct(m['fpr'])} | {_pct(m['balanced_accuracy'])} "
            f"| {gain} | {_pct(m['ai_likely_ai'])} | {_pct(m['real_likely_ai'])} |")


def write_report(path: Path, sources, records, cv: dict, chosen: str, final_model: dict,
                 decision: str, tv: dict, fitted_dims: tuple[str, ...]) -> None:
    labels = np.array([r["label"] for r in records])
    shown = {k: v for k, v in cv.items() if k in ("static", chosen)} if chosen != "static" else \
        {k: v for k, v in cv.items() if k == "static" or k == max(
            (c for c in cv if c != "static"), key=lambda c: cv[c]["overall"]["balanced_accuracy"], default="static")}
    lines: list[str] = []
    add = lines.append
    add("# Context-dependent weighting: calibration report")
    add("")
    add(f"Generated by `eval/calibrate_weights.py` on {_dt.date.today().isoformat()}. "
        "Re-run it after changing any analyzer or adding sources.")
    add("")
    add(f"**Decision:** {decision}")
    add("")
    add("## Batch")
    add("")
    n_ai = sum(s["label"] for s in sources)
    add(f"- {len(sources)} labeled sources: {n_ai} generated, {len(sources) - n_ai} genuine.")
    add(f"- {len(records)} synthetic variants ({len(records) // max(len(sources), 1)} per source): "
        f"rescale {', '.join(str(s) for s in SCALES)} x save PNG / JPEG {', '.join(str(q) for _, q in SAVES if q)}, "
        "plus 320px centre crops (PNG, JPEG 85), grayscale (PNG, JPEG 85), JPEG 90 then resize 0.75, "
        "double JPEG 85 -> 75.")
    add("- Generated sources: " + ", ".join(f"`{s['name']}`" for s in sources if s["label"] == 1) + ".")
    add("- Genuine sources: " + ", ".join(f"`{s['name']}`" for s in sources if s["label"] == 0) + ".")
    add("")
    add("## Which context dimensions can be learned from this batch")
    add("")
    add(f"Total-variation distance between the generated and genuine bucket distributions "
        f"(fitted only if <= {MAX_CONFOUND_TV}):")
    add("")
    add("| dimension | TV distance | fitted | generated buckets | genuine buckets |")
    add("|---|---|---|---|---|")
    for dim in ctxmod.DIMENSIONS:
        dist = {}
        for lab in (0, 1):
            rows = [r for r in records if r["label"] == lab]
            dist[lab] = ", ".join(f"{b} {np.mean([r['context']['buckets'][dim] == b for r in rows]):.0%}"
                                  for b in ctxmod.BUCKETS[dim])
        add(f"| {dim} | {tv[dim]:.3f} | {'yes' if dim in fitted_dims else '**no, confounded**'} | {dist[1]} | {dist[0]} |")
    mp_auc = auc([r["context"]["megapixels"] for r in records], labels)
    add("")
    add(f"Megapixels alone, used as an \"AI score\", reach AUC {mp_auc:.3f} on this batch. A size factor fitted "
        "here would learn the composition of the source set, not how size affects the signals.")
    add("")
    add("## Held-out results (leave-one-source-out)")
    add("")
    add(f"Flagged = score >= {REAL_CUTOFF} (not \"likely real\" in the UI); \"likely AI\" = score > {AI_CUTOFF}. "
        "The CI is a 95% bootstrap over sources.")
    add("")
    _cv_table(add, cv, chosen)
    add("")
    add("Caution on the static row: the analyzers' own thresholds were tuned while looking at these source "
        "images, which gives static weights an in-sample advantage no held-out fit can have.")
    add("")

    for key, title, observed in (("save", "By save format (applied)", False),
                                 ("scale", "By rescale factor (applied)", False),
                                 ("compression", "By observed compression bucket", True),
                                 ("scale", "By observed scale bucket", True),
                                 ("size", "By observed size bucket", True),
                                 ("color", "By observed color bucket", True)):
        keyf = (lambda r, k=key: r["context"]["buckets"][k]) if observed else (lambda r, k=key: _group_key(r, k))
        groups = sorted({keyf(r) for r in records})
        add(f"### {title}")
        add("")
        add("| group | n gen / real | " + " | ".join(f"{c}: gen flagged / real flagged" for c in shown) + " |")
        add("|---|---|" + "---|" * len(shown))
        for g in groups:
            idx = np.array([keyf(r) == g for r in records])
            cells = []
            for name, res in shown.items():
                m = summarize(res["scores"][idx], labels[idx])
                cells.append(f"{_pct(m['tpr'])} / {_pct(m['fpr'])}")
            add(f"| {g} | {int(labels[idx].sum())} / {int((1 - labels[idx]).sum())} | " + " | ".join(cells) + " |")
        add("")

    add("### Per source (mean held-out score over its variants, share flagged)")
    add("")
    add("| source | label | " + " | ".join(shown) + " |")
    add("|---|---|" + "---|" * len(shown))
    for s in sources:
        idx = np.array([r["source"] == s["name"] for r in records])
        add(f"| `{s['name']}` | {'generated' if s['label'] else 'genuine'} | "
            + " | ".join(f"{shown[c]['scores'][idx].mean():.3f} ({_pct(np.mean(shown[c]['scores'][idx] >= REAL_CUTOFF))})"
                         for c in shown) + " |")
    add("")

    add("## Signal behaviour by context (fit on all sources)")
    add("")
    add("Retention: share of variants where a signal still fires (score >= 0.35), among generated sources whose "
        "untouched original fires it. False alarm: share of genuine variants where it fires. "
        "Factor = (retention - false alarm) relative to the signal's overall value, shrunk and bounded "
        f"to [{FACTOR_MIN}, {FACTOR_MAX}].")
    add("")
    add("| signal | overall retention | overall false alarm | generated sources that fire it |")
    add("|---|---|---|---|")
    for sig in SIGNALS:
        a = final_model["_stats"][sig]["_all"]
        add(f"| {sig} | {_pct(a['retention'])} | {_pct(a['false_alarm'])} | "
            f"{', '.join(f'`{x}`' for x in a['firing_sources']) or 'none'} |")
    add("")
    add("| signal | dimension | bucket | retention | false alarm | n gen / genuine | factor |")
    add("|---|---|---|---|---|---|---|")
    for sig in SIGNALS:
        for dim, buckets in final_model["_stats"][sig].items():
            if dim == "_all":
                continue
            for b, st in buckets.items():
                add(f"| {sig} | {dim} | {b} | {_pct(st['retention'])} | {_pct(st['false_alarm'])} | "
                    f"{st['n_generated']} / {st['n_genuine']} | {final_model['multipliers'][sig][dim][b]:.2f} |")
    add("")
    add("## Caveats")
    add("")
    add(f"- Only {n_ai} generated sources, each caught by a different signal. That is why only degradation "
        "behaviour is fitted: a combiner that learns which signal means AI inverts on held-out generators. "
        "Adding generated images with `--ai-dir` (and large, colour genuine photos with `--real-dir`, which "
        "would make size and color learnable) is the most valuable improvement; everything here is re-derived.")
    add("- Degradations are synthetic stand-ins for real pipelines (social-media recompression, screenshots, "
        "CDN resizing). They cover the mechanisms, not every encoder.")
    add("- The batch's transform mix is itself an assumption about real traffic. For example, 40% of variants "
        "are upscaled, for generated and genuine sources alike, which is why the frequency knee (an upscaling "
        "detector) separates nothing across the batch as a whole.")
    add("- Weighting can only re-balance evidence the signals measured. Most misses in the tables above are "
        "variants where every tell was destroyed (downscaling, JPEG on a noise-only source); no weighting "
        "recovers those.")
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    ap.add_argument("--reuse", action="store_true", help="reuse extracted signals from --workdir")
    ap.add_argument("--ai-dir", help="extra generated images (png/jpg)")
    ap.add_argument("--real-dir", help="extra genuine images (png/jpg)")
    ap.add_argument("--no-library", action="store_true", help="don't use package-bundled sample photos")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--dry-run", action="store_true", help="write the report but not the calibration JSON")
    ap.add_argument("--allow-small", action="store_true",
                    help=f"run even with fewer than {MIN_GENUINE_SOURCES} genuine sources (results are not meaningful)")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    sources = collect_sources(args)
    genuine = sum(1 for s in sources if s["label"] == 0)
    if genuine < MIN_GENUINE_SOURCES and not args.allow_small:
        print(f"only {genuine} genuine source(s) found, need at least {MIN_GENUINE_SOURCES}; nothing was changed.\n"
              "The packaged sample photos come from scikit-image, scikit-learn and matplotlib, which are not "
              "installed here (or have no wheel for this Python). Supply your own with "
              "--real-dir DIR (and --ai-dir DIR), or pass --allow-small to run anyway.", file=sys.stderr)
        return 2
    cache = workdir / "signals.jsonl"
    if args.reuse and cache.exists():
        records = [json.loads(line) for line in cache.open()]
        names = {s["name"] for s in sources}
        records = [r for r in records if r["source"] in names]
        print(f"reusing {len(records)} extracted variants from {cache}")
    else:
        records = extract(sources, workdir, args.workers)

    labels = np.array([r["label"] for r in records])
    tv = confound_tv(records)
    fitted_dims = tuple(d for d in ctxmod.DIMENSIONS if tv[d] <= MAX_CONFOUND_TV)
    print("confound check (TV): " + ", ".join(f"{d}={tv[d]:.3f}" for d in ctxmod.DIMENSIONS)
          + f" -> fitted dimensions: {', '.join(fitted_dims) or 'none'}")

    static_scores = loso_scores(records, None)
    cv = {"static": {"scores": static_scores, "overall": summarize(static_scores, labels)}}
    static = cv["static"]["overall"]
    candidates = [tuple(c) for k in range(1, len(fitted_dims) + 1)
                  for c in itertools.combinations(fitted_dims, k)]
    for dims in candidates:
        name = "dynamic: " + "+".join(dims)
        scores = loso_scores(records, dims)
        overall = summarize(scores, labels)
        ci = bootstrap_diff(scores, static_scores, records)
        cv[name] = {"scores": scores, "overall": overall, "dims": dims, "ci": ci,
                    "gain": overall["balanced_accuracy"] - static["balanced_accuracy"]}
    for name, res in cv.items():
        m = res["overall"]
        extra = "" if "ci" not in res else f" gain={res['gain']:+.3f} CI[{res['ci'][0]:+.3f},{res['ci'][1]:+.3f}]"
        print(f"{name:32s} AUC={m['auc']:.3f} flagged gen={_pct(m['tpr'])} real={_pct(m['fpr'])} "
              f"balanced={_pct(m['balanced_accuracy'])}{extra}")

    passing = [n for n, r in cv.items() if n != "static"
               and r["gain"] >= 0
               and r["ci"][0] >= -NONINFERIORITY_MARGIN
               and r["overall"]["fpr"] <= static["fpr"] + 1e-9]
    if passing:
        chosen = max(passing, key=lambda n: (cv[n]["overall"]["balanced_accuracy"], -len(cv[n]["dims"])))
        r = cv[chosen]
        decision = (f"adopted **{chosen}**. On held-out sources: balanced accuracy "
                    f"{_pct(r['overall']['balanced_accuracy'])} vs {_pct(static['balanced_accuracy'])} static "
                    f"(difference {r['gain']:+.3f}, 95% CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]), "
                    f"false-positive rate {_pct(r['overall']['fpr'])} vs {_pct(static['fpr'])}, generated images "
                    f"flagged {_pct(r['overall']['tpr'])} vs {_pct(static['tpr'])}. "
                    + ("The improvement is **not** statistically significant: the interval includes 0. "
                       "Adopted as non-inferior and mechanistically motivated, not as a proven gain."
                       if r["ci"][0] <= 0 else "The improvement is statistically significant."))
    else:
        chosen = "static"
        decision = ("kept **static** weights: no dynamic candidate was non-inferior on held-out sources "
                    "without raising the false-positive rate.")

    chosen_dims = cv[chosen].get("dims", ())
    final = fit_model(records, chosen_dims if chosen != "static" else fitted_dims)
    write_report(REPORT_PATH, sources, records, cv, chosen, final, decision, tv, fitted_dims)
    print(f"report: {REPORT_PATH}")
    print(decision)

    if args.dry_run:
        return 0
    if chosen == "static":
        multipliers = {}
    else:
        multipliers = {s: {d: final["multipliers"][s][d] for d in chosen_dims if d in final["multipliers"][s]}
                       for s in SIGNALS}
    out = {
        "version": 2,
        "weighting": chosen,
        "dimensions": list(chosen_dims),
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "generated_by": "eval/calibrate_weights.py",
        "method": "factor = (retention - false alarm) in bucket / overall; static base weights; "
                  "absolute down-weighting (see app/main.py _visual_score)",
        "multipliers": multipliers,
        "confound_tv": {d: round(v, 4) for d, v in tv.items()},
        "held_out": {
            name: {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["overall"].items()},
                   **({"gain_vs_static": round(r["gain"], 4),
                       "gain_ci95": [round(r["ci"][0], 4), round(r["ci"][1], 4)]} if "ci" in r else {})}
            for name, r in cv.items()
        },
        "sources": {"generated": [s["name"] for s in sources if s["label"] == 1],
                    "genuine": [s["name"] for s in sources if s["label"] == 0]},
        "variants": len(records),
    }
    CALIBRATION_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"calibration: {CALIBRATION_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
