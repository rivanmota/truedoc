"""
Sensor-noise floor analysis.

Every real capture leaves a noise floor in flat, low-detail regions
(photon shot noise, read noise, film grain), and the camera pipeline
reshapes it: demosaicing and sharpening correlate it spatially, in-camera
denoising and JPEG quantisation wipe it out unevenly (some blocks keep
their grain, others are flattened to nothing), and its amplitude varies
with brightness and from surface to surface. A real noise floor is
therefore *patchy* and usually *spatially correlated*.

Generated images that carry a noise floor at all tend to carry a
synthetic one: modern generators and their upscalers lay a fine, uniform
grain over everything to avoid the "plastic" look. That grain is
*stationary* (same amplitude in every flat region, independent of
brightness or surface) and close to *white* (little pixel-to-pixel
correlation). That combination is what this analyzer looks for.

Note what it deliberately does NOT do any more: treat a very clean flat
region as AI evidence. Phone photos are routinely denoised and
JPEG-compressed until their flat regions carry no measurable noise, so
"too clean" mostly detected JPEG compression, not generation (re-saving
a generated PNG as JPEG used to *raise* its score). The signal now
reports itself not applicable when:
  - the pixels still carry an aligned 8x8 JPEG grid: quantisation zeroes
    grain in some blocks and not others, which makes synthetic grain look
    patchy (real) and can make real noise look uniform, so the grain's
    character cannot be judged in either direction; or
  - the flat regions carry no measurable noise at all (erased by
    compression, denoising or downscaling, or a clean render; these
    cannot be told apart from pixels alone).

Approach:
  1. split the image into 16x16 blocks and keep the flat, unclipped ones
     (mean gradient of the lightly-smoothed image < 1 grey level/pixel)
  2. per flat block, estimate the noise sigma with Immerkaer's
     second-difference operator (insensitive to linear shading ramps)
  3. stationarity: robust spread (IQR / median) of those block sigmas;
     synthetic grain sits around 0.15-0.28, real floors are >= 0.3 and
     often well above 1
  4. whiteness: lag-1 autocorrelation of the residual inside flat blocks;
     synthetic grain ~0.1-0.17, camera/JPEG floors mostly >= 0.25

Known false positives: film scans and some high-ISO captures also carry
stationary, white grain (skimage's ``astronaut.png`` is one). Stationary
grain is supporting evidence, not proof.

Any JPEG compression that leaves a detectable grid (in practice most
JPEGs that were not resized afterwards) makes this analyzer report
"unavailable" rather than guess; the other signals then carry the score.
"""
from __future__ import annotations

import cv2
import numpy as np

BLOCK = 16

# A block is "flat" when the mean gradient magnitude of the image, after a
# sigma=2 Gaussian to suppress the noise being measured, is below this many
# grey levels per pixel.
FLAT_STRUCTURE_MAX = 1.0
# Blocks with more than this fraction of clipped (0-2 / 253-255) pixels are
# skipped: clipping suppresses noise independently of the capture process.
CLIPPED_FRACTION_MAX = 0.05
MIN_FLAT_BLOCKS = 24
MIN_FLAT_FRACTION = 0.02

# Noise floor counted as erased (compression / heavy denoise) when the
# typical flat block has this fraction of its residual at exactly ~0, or
# its sigma is below ERASED_SIGMA grey levels.
ERASED_ZERO_FRACTION = 0.45
ERASED_SIGMA = 0.20

# Stationarity ramp on IQR/median of per-block sigma: full signal at or
# below STATIONARY_FULL, none at or above STATIONARY_NONE.
STATIONARY_FULL = 0.20
STATIONARY_NONE = 0.45

# Whiteness ramp on lag-1 residual autocorrelation (max of x and y).
WHITE_FULL = 0.15
WHITE_NONE = 0.30
# Share of the stationarity signal kept when the grain is NOT white.
# Correlated-but-uniform noise is also what optically soft or processed
# real photos show, so it only counts for a little.
NON_WHITE_FLOOR = 0.30

# JPEG 8x8 grid detection: mean |pixel difference| across the 7|8 block
# boundary relative to the largest mean difference at any other phase.
# Uncompressed and resampled images measure <= ~1.015; JPEGs whose grid
# is still pixel-aligned measure ~1.045-1.65.
JPEG_GRID_RATIO = 1.03
# ...and the boundary step must exceed the other phases by at least this
# many grey levels (real JPEG grids: >= ~0.14).
JPEG_GRID_MIN_EXCESS = 0.05

_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
_IMMERKAER_SCALE = float(np.sqrt(np.pi / 2.0) / 6.0)


def _block_view(a: np.ndarray, bs: int = BLOCK) -> np.ndarray:
    """(H, W) -> (nh, bs, nw, bs) view, dropping the ragged right/bottom edge."""
    h, w = a.shape
    nh, nw = h // bs, w // bs
    return a[: nh * bs, : nw * bs].reshape(nh, bs, nw, bs)


def _block_means(a: np.ndarray) -> np.ndarray:
    """Per-block mean, flattened row-major to (nh * nw,)."""
    return _block_view(a).mean(axis=(1, 3)).ravel()


def _select_blocks(a: np.ndarray, flat_idx: np.ndarray, nw: int) -> np.ndarray:
    """(n_selected, bs, bs) copies of just the selected blocks."""
    by, bx = np.divmod(flat_idx, nw)
    return _block_view(a)[by, :, bx, :]


def jpeg_grid_strength(gray: np.ndarray) -> float:
    """Blockiness at the JPEG 8x8 grid.

    For columns and rows separately: mean absolute difference across the
    7|8 boundary divided by the *largest* mean difference at the other
    seven phases, and the larger of the two axes is returned. ~1.0 (or
    below) means no grid. Comparing against the maximum rather than the
    median matters: 2x upscaling lifts every odd phase equally, which a
    median comparison mistakes for a JPEG grid. Resampling after
    compression destroys the alignment, so this only finds JPEG history
    that still shapes the pixels being measured, which is the history that
    matters here.
    """
    if gray.shape[0] < 64 or gray.shape[1] < 64:
        return 1.0
    # Mean |difference| per column boundary / per row boundary, then grouped
    # by phase mod 8 (every phase has the same number of rows, so the mean
    # of column means equals the mean over the strided slice).
    col_means = np.abs(np.diff(gray, axis=1)).mean(axis=0)
    row_means = np.abs(np.diff(gray, axis=0)).mean(axis=1)

    def ratio(per_phase: np.ndarray) -> float:
        others = float(per_phase[:7].max())
        # On near-textureless images every phase differs by hundredths of
        # a grey level and the ratio is noise; require a real step too.
        if per_phase[7] - others < JPEG_GRID_MIN_EXCESS:
            return float(min(per_phase[7] / max(others, 1e-3), 1.0))
        return float(min(per_phase[7] / max(others, 1e-3), 10.0))

    cols = np.array([col_means[p::8].mean() for p in range(8)])
    rows = np.array([row_means[p::8].mean() for p in range(8)])
    return max(ratio(cols), ratio(rows))


def _ramp_down(value: float, full: float, none: float) -> float:
    """1.0 at or below ``full``, 0.0 at or above ``none``, linear between."""
    return float(np.clip((none - value) / (none - full), 0.0, 1.0))


def analyze(gray: np.ndarray) -> dict:
    """
    gray: 2D float32 array, single channel, values 0-255, ideally at the
    file's native resolution (downscaling averages the noise floor away).
    """
    gray = gray.astype(np.float32, copy=False)
    h, w = gray.shape
    if h < 4 * BLOCK or w < 4 * BLOCK:
        return {
            "score": 0.5,
            "confidence": "low",
            "applicable": False,
            "note": "image too small to estimate a noise floor",
        }

    smooth = cv2.GaussianBlur(gray, (0, 0), 2.0)
    structure = np.hypot(
        cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) / 8.0,
        cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8.0,
    )
    residual = cv2.filter2D(gray, cv2.CV_32F, _IMMERKAER)
    # A 5x5 box-mean residual keeps the grain's own spatial correlation
    # (the Immerkaer kernel would impose its own) for the whiteness test.
    box_residual = gray - cv2.blur(gray, (5, 5))

    nw = w // BLOCK
    struct_mean = _block_means(structure)
    clipped = _block_means(((gray <= 2) | (gray >= 253)).astype(np.float32))

    flat = (struct_mean < FLAT_STRUCTURE_MAX) & (clipped < CLIPPED_FRACTION_MAX)
    n_flat = int(flat.sum())
    flat_fraction = n_flat / max(len(flat), 1)
    grid = jpeg_grid_strength(gray)
    jpeg_grid = grid >= JPEG_GRID_RATIO

    base = {
        "flat_block_count": n_flat,
        "flat_fraction": round(flat_fraction, 4),
        "jpeg_grid_strength": round(grid, 3),
        "jpeg_grid_detected": bool(jpeg_grid),
    }

    if n_flat < MIN_FLAT_BLOCKS or flat_fraction < MIN_FLAT_FRACTION:
        return {
            "score": 0.5,
            "confidence": "low",
            "applicable": False,
            **base,
            "note": "not enough flat/low-detail area to estimate a noise floor reliably",
        }

    flat_idx = np.flatnonzero(flat)
    abs_res = np.abs(_select_blocks(residual, flat_idx, nw))
    sigmas = abs_res.mean(axis=(1, 2)) * _IMMERKAER_SCALE
    zero_fraction = float(np.median((abs_res < 0.5).mean(axis=(1, 2))))
    sigma_median = float(np.median(sigmas))

    base.update({
        "noise_sigma_median": round(sigma_median, 3),
        "quantized_flat_fraction": round(zero_fraction, 3),
    })

    if jpeg_grid:
        return {
            "score": 0.5,
            "confidence": "low",
            "applicable": False,
            "floor_state": "reshaped_by_jpeg",
            **base,
            "note": "noise floor unavailable as evidence: an aligned JPEG grid shows quantisation has reshaped the grain, which hides synthetic grain and distorts real noise alike",
        }
    if zero_fraction > ERASED_ZERO_FRACTION or sigma_median < ERASED_SIGMA:
        return {
            "score": 0.5,
            "confidence": "low",
            "applicable": False,
            "floor_state": "erased",
            **base,
            "note": "noise floor unavailable: flat regions carry no measurable noise (compression, denoising, downscaling or a clean render cannot be told apart from pixels alone)",
        }

    q1, q2, q3 = np.percentile(sigmas, [25, 50, 75])
    spread = float((q3 - q1) / max(q2, 1e-3))

    g = _select_blocks(box_residual, flat_idx, nw)
    g = g - g.mean(axis=(1, 2), keepdims=True)
    var = (g ** 2).mean(axis=(1, 2)) + 1e-6
    ac_x = float(np.median((g[:, :, 1:] * g[:, :, :-1]).mean(axis=(1, 2)) / var))
    ac_y = float(np.median((g[:, 1:, :] * g[:, :-1, :]).mean(axis=(1, 2)) / var))
    autocorr = max(ac_x, ac_y)

    stationarity_signal = _ramp_down(spread, STATIONARY_FULL, STATIONARY_NONE)
    whiteness_signal = _ramp_down(autocorr, WHITE_FULL, WHITE_NONE)
    score = stationarity_signal * (NON_WHITE_FLOOR + (1.0 - NON_WHITE_FLOOR) * whiteness_signal)
    score = float(np.clip(score, 0.0, 1.0))

    confidence = "medium" if n_flat >= 3 * MIN_FLAT_BLOCKS else "low-medium"

    return {
        "score": round(score, 4),
        "confidence": confidence,
        "applicable": True,
        "floor_state": "observable",
        **base,
        "noise_level_spread": round(spread, 3),
        "residual_autocorrelation": round(autocorr, 3),
        "stationarity_signal": round(stationarity_signal, 4),
        "whiteness_signal": round(whiteness_signal, 4),
        "note": "uniform (stationary) and spatially white grain across flat regions suggests synthetic grain; real noise floors are patchy and correlated by the camera pipeline",
    }
