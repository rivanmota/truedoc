"""
Observable image context: the conditions under which the forensic signals
were measured, as far as they can be read from the file itself.

The signals do not degrade uniformly. JPEG quantisation erodes skin
texture and attenuates the top of the spectrum; upscaling correlates the
noise floor and smooths texture; small images give the noise and spectral
estimators little to average over; grayscale input removes the colour the
skin mask needs. ``app/weighting.py`` uses this context to re-weight each
signal by how well it has been measured to separate generated from
genuine images *under these conditions* (see ``eval/``).

Only things a caller cannot lie about by renaming a file are used: the
container is sniffed from magic bytes, JPEG quality is estimated from the
quantisation tables actually stored in the file, and JPEG history in a
PNG is read from the pixel grid. What was done to an image before its
last save (a downscale, say) is generally *not* observable and is not
guessed at; only upscaling leaves a trace (the spectral knee).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.analyzers import noise as noise_analyzer

# Bucket edges. Kept here (not in the calibration JSON) because the
# evaluation buckets its samples with these same functions.
JPEG_HIGH_MIN_QUALITY = 90
JPEG_MID_MIN_QUALITY = 75
SMALL_MAX_MEGAPIXELS = 0.35
LARGE_MIN_MEGAPIXELS = 1.0
UPSCALE_KNEE_MIN = 0.40        # frequency.KNEE_NONE: start of the knee ramp
GRAYSCALE_MAX_CHANNEL_DIFF = 2  # grey levels

DIMENSIONS = ("compression", "size", "scale", "color")
BUCKETS = {
    "compression": ("lossless", "jpeg_history", "jpeg_high", "jpeg_mid", "jpeg_low"),
    "size": ("small", "medium", "large"),
    "scale": ("native", "upscaled"),
    "color": ("color", "grayscale"),
}

# IJG (libjpeg) standard luminance table, quality 50.
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)


def _ijg_table(quality: int) -> np.ndarray:
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    return np.clip(np.floor((_STD_LUMA * scale + 50.0) / 100.0), 1, 255)


_IJG_SUMS = {q: float(_ijg_table(q).sum()) for q in range(1, 101)}


def estimate_jpeg_quality(image_bytes: bytes) -> int | None:
    """IJG-equivalent quality of the luminance quantisation table, or None
    for non-JPEG input. Matches on the table sum, which is independent of
    coefficient order; encoders with custom tables (most phones) get the
    IJG quality with the same overall coarseness, which is what matters
    for how much detail was discarded."""
    try:
        im = Image.open(io.BytesIO(image_bytes))
        tables = getattr(im, "quantization", None)
    except Exception:
        return None
    if not tables:
        return None
    luma = np.asarray(tables.get(0, next(iter(tables.values()))), dtype=np.float64)
    total = float(luma.sum())
    return min(_IJG_SUMS, key=lambda q: abs(_IJG_SUMS[q] - total))


def _is_grayscale(bgr: np.ndarray) -> bool:
    if bgr.ndim == 2 or bgr.shape[2] == 1:
        return True
    sample = bgr[:: max(1, bgr.shape[0] // 256), :: max(1, bgr.shape[1] // 256)].astype(np.int16)
    spread = sample.max(axis=2) - sample.min(axis=2)
    return float(np.mean(spread <= GRAYSCALE_MAX_CHANNEL_DIFF)) > 0.999


def compression_bucket(fmt: str, jpeg_quality: int | None, jpeg_grid: bool) -> str:
    if fmt == "jpeg":
        if jpeg_quality is None:
            return "jpeg_mid"
        if jpeg_quality >= JPEG_HIGH_MIN_QUALITY:
            return "jpeg_high"
        if jpeg_quality >= JPEG_MID_MIN_QUALITY:
            return "jpeg_mid"
        return "jpeg_low"
    return "jpeg_history" if jpeg_grid else "lossless"


def size_bucket(megapixels: float) -> str:
    if megapixels < SMALL_MAX_MEGAPIXELS:
        return "small"
    if megapixels >= LARGE_MIN_MEGAPIXELS:
        return "large"
    return "medium"


def scale_bucket(frequency_result: dict) -> str:
    knee = frequency_result.get("high_band_knee_drop")
    if frequency_result.get("applicable") is False or knee is None:
        return "native"
    return "upscaled" if float(knee) >= UPSCALE_KNEE_MIN else "native"


def image_context(
    image_bytes: bytes,
    fmt: str,
    width: int,
    height: int,
    bgr: np.ndarray,
    native_gray: np.ndarray,
    frequency_result: dict,
) -> dict:
    """Context record for one decoded image (native width/height)."""
    jpeg_quality = estimate_jpeg_quality(image_bytes) if fmt == "jpeg" else None
    grid_strength = noise_analyzer.jpeg_grid_strength(native_gray)
    jpeg_grid = grid_strength >= noise_analyzer.JPEG_GRID_RATIO
    megapixels = width * height / 1e6
    grayscale = _is_grayscale(bgr)

    buckets = {
        "compression": compression_bucket(fmt, jpeg_quality, jpeg_grid),
        "size": size_bucket(megapixels),
        "scale": scale_bucket(frequency_result),
        "color": "grayscale" if grayscale else "color",
    }
    return {
        "format": fmt,
        "jpeg_quality_estimate": jpeg_quality,
        "jpeg_grid_strength": round(float(grid_strength), 3),
        "width": int(width),
        "height": int(height),
        "megapixels": round(megapixels, 3),
        "buckets": buckets,
    }


def embedded_raster_buckets(buckets: dict, profile: dict) -> dict:
    """Context buckets for a raster extracted from a PDF.

    Extraction re-encodes the pixels as PNG, so the pixel-derived buckets
    would call a JPEG-compressed raster "jpeg_history" (quality unknown).
    When the PDF stored it as a plain JPEG stream (DCTDecode) and the pixels
    were not resampled on extraction, use the quality read from that stream,
    so the raster is weighted exactly like the same JPEG uploaded directly.
    """
    quality = profile.get("jpeg_quality")
    if quality is None or profile.get("resampled_on_extraction"):
        return dict(buckets)
    return {**buckets, "compression": compression_bucket("jpeg", int(quality), jpeg_grid=True)}

