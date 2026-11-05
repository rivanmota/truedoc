"""
Frequency-domain analysis.

The radially-averaged power spectrum of a natural photo falls off close
to a power law (a straight line in log-log space) from the coarse
structure all the way out to the Nyquist limit; optics and sensor noise
bend the very top of it gently, not abruptly.

Generator pipelines commonly leave a different shape. The base model
renders at roughly one megapixel and the output is then enlarged by a
"hires fix" pass, a latent upscaler or a plain resampler. Enlargement
cannot create detail above the source image's own Nyquist limit, so the
spectrum follows the power law up to some fraction of the output
band and then *collapses*: a knee. VAE decoders that under-synthesise
fine texture produce the same kind of high-band deficit.

What this module measures:
  - a Hann-windowed 2D power spectrum, Welch-averaged over up to 16
    tiles of 512px (the window stops tile borders from painting a bright
    cross of fake high-frequency energy; averaging lowers the variance)
  - the radial profile on a normalised frequency axis (fraction of
    Nyquist, so non-square images compare correctly)
  - a power-law fit over the mid band (0.08-0.40 Nyquist)
  - how far the high band (0.78-0.95) falls below that fit *relative to*
    the band just above the fit (0.40-0.55). A large drop that starts
    only after the mid band is a knee; a deficit that is already present
    in the mid band is a gradual optical roll-off (defocus, soft lens,
    texture) and is discounted.

Read this signal as *resampling* evidence: it fires on any image that
was enlarged after capture or generation, including an upscaled real
photo. It cannot see anything in images that were downscaled afterwards
(downscaling pushes the knee past Nyquist), and it is blind to
generators that output at native resolution without upscaling.

The previous version of this module measured energy ratios on a
log-of-log-magnitude profile without windowing, which put every image
in the same narrow band; its "periodicity" feature saturated on every
input. Isolated spectral peaks were evaluated as a replacement and
dropped: JPEG's 8x8 grid and ordinary resampling produce the same peaks
on real photos.
"""
from __future__ import annotations

import numpy as np

N_BINS = 48
FIT_BAND = (0.08, 0.40)
MID_BAND = (0.40, 0.55)
HIGH_BAND = (0.78, 0.95)

# Knee ramp: drop (mid-band deviation minus high-band deviation, in log10
# power units) -> signal. 0.40 (a 2.5x power deficit) is within the range
# of unprocessed real photos; 0.80 (6.3x) is a hard cutoff.
KNEE_NONE = 0.40
KNEE_FULL = 0.80
# Gradual roll-off discount: if the mid band is already this far below the
# power-law fit, the high-band deficit started early and is optical.
ROLLOFF_OK = 0.10
ROLLOFF_SPAN = 0.20

MIN_SIDE = 128
# Welch estimate: square tiles of at most TILE px, up to
# MAX_TILES_PER_AXIS^2 of them spread evenly over the image.
TILE = 512
MAX_TILES_PER_AXIS = 4


def _fft_friendly(n: int) -> int:
    """Largest m <= n whose only prime factors are 2, 3 and 5 (fast FFT
    sizes; a dimension like 1448 = 8 * 181 is ~20x slower)."""
    for m in range(n, 0, -1):
        k = m
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
    return 1


def _tile_origins(length: int, tile: int, count: int) -> list[int]:
    if length <= tile:
        return [0]
    return sorted({int(round(v)) for v in np.linspace(0, length - tile, count)})


def _radial_power_profile(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Welch-averaged radial power profile.

    Returns (normalised radius bin centres, mean power per bin, tile size,
    tile count). Averaging Hann-windowed tiles lowers the variance of the
    estimate and keeps the FFT size small and fast regardless of the
    image's dimensions.
    """
    h, w = gray.shape
    tile = _fft_friendly(min(TILE, h, w))
    ys = _tile_origins(h, tile, MAX_TILES_PER_AXIS)
    xs = _tile_origins(w, tile, MAX_TILES_PER_AXIS)

    window = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32)
    power = np.zeros((tile, tile), dtype=np.float64)
    for y in ys:
        for x in xs:
            patch = gray[y:y + tile, x:x + tile].astype(np.float32)
            patch = (patch - patch.mean()) * window
            power += np.abs(np.fft.fft2(patch)) ** 2
    power = np.fft.fftshift(power / (len(ys) * len(xs)))

    yy, xx = np.indices((tile, tile))
    half = tile / 2.0
    radius = np.hypot((xx - tile // 2) / half, (yy - tile // 2) / half)
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    idx = np.digitize(radius.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < N_BINS)
    sums = np.bincount(idx[ok], power.ravel()[ok], minlength=N_BINS)
    counts = np.bincount(idx[ok], minlength=N_BINS)
    profile = sums / np.maximum(counts, 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return centres, profile, tile, len(ys) * len(xs)


def _band_mean(values: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> float:
    sel = (freqs >= band[0]) & (freqs <= band[1])
    return float(values[sel].mean())


def analyze(gray: np.ndarray) -> dict:
    """
    gray: 2D float32 array, single channel, values 0-255, ideally at the
    file's native resolution (downscaling hides an upscaling knee).
    Returns a dict with a 0-1 ai-likelihood score and diagnostics.
    """
    h, w = gray.shape
    if min(h, w) < MIN_SIDE:
        return {
            "score": 0.5,
            "confidence": "low",
            "applicable": False,
            "note": "image too small for reliable frequency analysis",
        }

    freqs, profile, tile, n_tiles = _radial_power_profile(gray)
    if not np.all(np.isfinite(profile)) or profile.max() <= 0:
        return {
            "score": 0.5,
            "confidence": "low",
            "applicable": False,
            "note": "unable to compute a usable power spectrum (flat image?)",
        }

    log_f = np.log10(freqs)
    log_p = np.log10(np.maximum(profile, 1e-12))
    fit_sel = (freqs >= FIT_BAND[0]) & (freqs <= FIT_BAND[1])
    slope, intercept = np.polyfit(log_f[fit_sel], log_p[fit_sel], 1)
    deviation = log_p - (slope * log_f + intercept)

    mid_dev = _band_mean(deviation, freqs, MID_BAND)
    high_dev = _band_mean(deviation, freqs, HIGH_BAND)
    knee_drop = mid_dev - high_dev
    early_rolloff = max(0.0, -mid_dev)

    knee_signal = float(np.clip((knee_drop - KNEE_NONE) / (KNEE_FULL - KNEE_NONE), 0.0, 1.0))
    knee_signal *= float(np.clip(1.0 - (early_rolloff - ROLLOFF_OK) / ROLLOFF_SPAN, 0.0, 1.0))
    score = float(np.clip(knee_signal, 0.0, 1.0))

    confidence = "medium" if min(h, w) >= 256 else "low-medium"

    return {
        "score": round(score, 4),
        "confidence": confidence,
        "applicable": True,
        "spectral_slope": round(float(slope), 3),
        "mid_band_deviation": round(mid_dev, 3),
        "high_band_deviation": round(high_dev, 3),
        "high_band_knee_drop": round(knee_drop, 3),
        "spectrum_tiles": f"{n_tiles} x {tile}px",
        "note": "a spectrum that follows the natural power law and then collapses in the top band (a knee) indicates the image was enlarged from a lower resolution, as generator upscaling pipelines do; upscaled real photos trigger it too",
    }
