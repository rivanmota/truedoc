"""
Gradient (texture) smoothness analysis for portraits: facial skin and the
backdrop behind the subject.

Real facial skin has fine, irregular texture: pores, fine lines, freckles,
small colour variation. Generated faces are frequently airbrushed: the
generator reproduces the shading of a face but averages away its
micro-texture. Studio-style generated backdrops behave the same way: a
rendered backdrop is a mathematically smooth ramp (plus, at most, a
uniform grain), whereas a real wall or paper roll has paint texture,
uneven light and dust.

This signal only applies to portraits. Earlier versions located "skin" by
colour alone (YCrCb thresholds), which also matches fire, wood, brick,
sand, beige walls and floors: on a fireplace scene 81% of the frame was
treated as skin, and on a real store photo the wood floor was. Skin is now
only analysed *inside a detected face*, and a face only counts when two
independent Haar cascades agree on it (either cascade alone produced false
faces on fireplaces, coins and flowers).

When no face is found the analyzer reports itself not applicable and the
combined score redistributes its weight to the frequency and noise
signals instead of averaging in a placeholder value.

Units. Sobel responses are divided by 8 so gradients are in grey levels
per pixel. The previous thresholds (skin variance < 60, background
variance < 8) were written for those units but applied to raw 3x3 Sobel
output (64x larger in variance), so they could never fire: every photo
measured in the hundreds to thousands.

  - skin: each face is resampled to a canonical 128x128 crop so texture is
    compared at the same scale regardless of how large the face is in the
    frame. Gradient-magnitude variance over the interior skin pixels
    (eyes/brows/mouth edges excluded) is ~7-10 for airbrushed generated
    faces and >= 23 for real faces, including old scans.
  - backdrop: gradient-magnitude variance over flat, non-skin, unclipped
    pixels. ~0.9-1.1 for rendered backdrops; real backdrops measure >= 3.
"""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

CANONICAL_FACE = 128
MIN_FACE_PX = 64          # smaller faces would need upscaling, which blurs texture
MIN_DETECT_FRACTION = 0.035
MIN_DETECT_PX = 40
MAX_FACES = 3
DETECT_MAX_DIM = 800
FACE_AGREEMENT_IOU = 0.30

# Skin: gradient-magnitude variance on a 128px canonical face.
SKIN_VAR_FULL = 10.0      # at or below -> fully airbrushed
SKIN_VAR_NONE = 25.0      # at or above -> natural texture
MIN_SKIN_PIXELS = 800

# Backdrop: gradient-magnitude variance over flat non-skin regions.
BG_STRUCTURE_MAX = 1.0    # grey levels/pixel after sigma=2 smoothing
BG_VAR_FULL = 1.0
BG_VAR_NONE = 3.0
MIN_BG_FRACTION = 0.10

SKIN_WEIGHT = 0.6
BG_WEIGHT = 0.4

# One pair of classifiers per thread. cv2.CascadeClassifier keeps per-call
# state on the object, so concurrent detectMultiScale calls on a shared
# instance interfere: measured, they shift face boxes by a few pixels (and so
# the skin statistics) or fail inside OpenCV. The scan API analyses requests
# in a thread pool, so sharing one pair made results depend on what else was
# running. Loading is cheap and happens once per worker thread.
_CASCADES = threading.local()


def _load_cascades():
    """This thread's two frontal-face cascades shipped with opencv-python, or None."""
    cascades = getattr(_CASCADES, "pair", None)
    if cascades is None:
        try:
            base = cv2.data.haarcascades  # type: ignore[attr-defined]
            a = cv2.CascadeClassifier(os.path.join(base, "haarcascade_frontalface_default.xml"))
            b = cv2.CascadeClassifier(os.path.join(base, "haarcascade_frontalface_alt2.xml"))
            cascades = (a, b) if not (a.empty() or b.empty()) else ()
        except Exception:
            cascades = ()
        _CASCADES.pair = cascades
    return cascades or None


def _iou(p, q) -> float:
    x1, y1 = max(p[0], q[0]), max(p[1], q[1])
    x2, y2 = min(p[0] + p[2], q[0] + q[2]), min(p[1] + p[3], q[1] + q[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = p[2] * p[3] + q[2] * q[3] - inter
    return inter / union if union > 0 else 0.0


def detect_faces(gray_u8: np.ndarray) -> list[tuple[int, int, int, int]] | None:
    """Faces both cascades agree on, largest first, in ``gray_u8``
    coordinates. None if no detector is available.

    Detection runs on a copy capped at DETECT_MAX_DIM on the long side
    (faces that matter are >= MIN_DETECT_PX anyway) and boxes are scaled
    back, which keeps two cascade passes cheap on large images.
    """
    cascades = _load_cascades()
    if cascades is None:
        return None
    h, w = gray_u8.shape
    scale = min(1.0, DETECT_MAX_DIM / max(h, w))
    small = gray_u8 if scale >= 1.0 else cv2.resize(
        gray_u8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )
    eq = cv2.equalizeHist(small)
    min_side = max(24, int(max(MIN_DETECT_PX, MIN_DETECT_FRACTION * min(h, w)) * scale))
    a = cascades[0].detectMultiScale(eq, scaleFactor=1.1, minNeighbors=8, minSize=(min_side, min_side))
    b = cascades[1].detectMultiScale(eq, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
    agreed = [
        tuple(int(round(v / scale)) for v in p)
        for p in a
        if any(_iou(p, q) > FACE_AGREEMENT_IOU for q in b)
    ]
    agreed.sort(key=lambda r: r[2] * r[3], reverse=True)
    return agreed[:MAX_FACES]


def _skin_colour_mask(bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    return cv2.inRange(ycrcb, lower, upper) > 0


def _grad_mag(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude in grey levels per pixel."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.hypot(gx, gy)


def _face_skin_variance(bgr: np.ndarray, box) -> tuple[float, int] | None:
    x, y, w, h = box
    crop = bgr[y:y + h, x:x + w]
    interp = cv2.INTER_AREA if w >= CANONICAL_FACE else cv2.INTER_CUBIC
    crop = cv2.resize(crop, (CANONICAL_FACE, CANONICAL_FACE), interpolation=interp)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    c = CANONICAL_FACE
    interior = np.zeros((c, c), dtype=bool)
    interior[int(0.15 * c):int(0.90 * c), int(0.15 * c):int(0.85 * c)] = True
    mask = _skin_colour_mask(crop) & interior
    if mask.sum() < MIN_SKIN_PIXELS:
        return None

    # Drop the most structured 40% (eyes, brows, nostrils, lips) so the
    # statistic describes skin surface, not facial features.
    structure = _grad_mag(cv2.GaussianBlur(g, (0, 0), 2.0))
    mask &= structure < np.percentile(structure[mask], 60)
    if mask.sum() < MIN_SKIN_PIXELS:
        return None
    return float(np.var(_grad_mag(g)[mask])), int(mask.sum())


def _ramp_down(value: float, full: float, none: float) -> float:
    return float(np.clip((none - value) / (none - full), 0.0, 1.0))


def _not_applicable(note: str, **extra) -> dict:
    return {
        "score": 0.5,
        "confidence": "low",
        "applicable": False,
        "skin": None,
        "background": None,
        **extra,
        "note": note,
    }


def analyze(bgr: np.ndarray, gray: np.ndarray) -> dict:
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    faces = detect_faces(gray_u8)
    if faces is None:
        return _not_applicable("unavailable: face detector could not be loaded, portrait texture analysis skipped")
    if not faces:
        return _not_applicable(
            "no clear portrait subject (no reliably detected face); skin/backdrop texture analysis "
            "does not apply, its weight goes to the frequency and noise signals",
            faces_detected=0,
        )

    skin_results = []
    for box in faces:
        if min(box[2], box[3]) < MIN_FACE_PX:
            continue
        measured = _face_skin_variance(bgr, box)
        if measured is not None:
            skin_results.append((box, *measured))

    result: dict = {"faces_detected": len(faces), "skin": None, "background": None}
    signals: list[tuple[float, float]] = []

    if skin_results:
        # Largest face first; it has the most reliable texture estimate.
        box, skin_var, skin_px = skin_results[0]
        skin_signal = _ramp_down(skin_var, SKIN_VAR_FULL, SKIN_VAR_NONE)
        result["skin"] = {
            "face_box": list(box),
            "face_size_px": int(min(box[2], box[3])),
            "skin_pixels_canonical": skin_px,
            "gradient_variance": round(skin_var, 2),
            "smoothness_signal": round(skin_signal, 4),
        }
        signals.append((skin_signal, SKIN_WEIGHT))

    # Backdrop: flat, unclipped, non-skin pixels anywhere in the frame.
    g = gray.astype(np.float32)
    structure = _grad_mag(cv2.GaussianBlur(g, (0, 0), 2.0))
    bg_mask = (structure < BG_STRUCTURE_MAX) & (gray_u8 > 2) & (gray_u8 < 253)
    bg_mask &= ~_skin_colour_mask(bgr)
    bg_fraction = float(bg_mask.mean())
    if bg_fraction >= MIN_BG_FRACTION:
        bg_var = float(np.var(_grad_mag(g)[bg_mask]))
        bg_signal = _ramp_down(bg_var, BG_VAR_FULL, BG_VAR_NONE)
        result["background"] = {
            "coverage_fraction": round(bg_fraction, 3),
            "gradient_variance": round(bg_var, 3),
            "smoothness_signal": round(bg_signal, 4),
        }
        signals.append((bg_signal, BG_WEIGHT))

    if not signals:
        return _not_applicable(
            "no clear skin or flat-backdrop region large enough to evaluate texture on "
            f"(faces found: {len(faces)}, all smaller than {MIN_FACE_PX}px or without visible skin)",
            faces_detected=len(faces),
        )

    total_w = sum(wt for _, wt in signals)
    score = float(np.clip(sum(s * wt for s, wt in signals) / total_w, 0.0, 1.0))
    confidence = "medium" if (result["skin"] and result["background"]) else "low-medium"

    return {
        "score": round(score, 4),
        "confidence": confidence,
        "applicable": True,
        **result,
        "note": "airbrushed facial skin and a perfectly smooth backdrop (low gradient variance, measured in grey levels/pixel) suggest generated texture",
    }
