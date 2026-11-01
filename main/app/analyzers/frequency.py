"""
Frequency-domain analysis.

Real camera photos have a power spectrum that falls off smoothly and
roughly monotonically from the DC component (low frequency = the bulk
of the image's structure) out to high frequency (fine detail + sensor
noise). The falloff is close to a 1/f^2 power law for natural images.

AI decoders (especially anything with transposed-convolution or
pixel-shuffle upsampling in the generator/VAE) tend to leave one or
more of these fingerprints:
  - an unnaturally sharp cutoff at some frequency band, because the
    decoder doesn't synthesize genuine high-frequency sensor noise
  - periodic peaks in the radially-averaged spectrum (grid artifacts
    from checkerboard/upsampling patterns), which show up as local
    bumps that don't fit the smooth decay curve
  - a low ratio of energy in the outer (high-frequency) ring of the
    spectrum relative to total energy, since real sensor noise adds
    a noise floor that AI images usually lack

This module centers the DC component (fftshift), takes the log
magnitude spectrum, computes a radially-averaged power profile, fits
a decay curve to it, and scores how much the observed profile
deviates from what a natural photo usually looks like.
"""
from __future__ import annotations

import numpy as np


def _radial_profile(magnitude: np.ndarray) -> np.ndarray:
    """Radially-averaged magnitude from the center (DC) outward."""
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    r_max = r.max()
    tbin = np.bincount(r.ravel(), magnitude.ravel())
    nr = np.bincount(r.ravel())
    nr[nr == 0] = 1
    profile = tbin / nr
    return profile[: r_max + 1]


def analyze(gray: np.ndarray) -> dict:
    """
    gray: 2D float32 array, single channel, values 0-255.
    Returns a dict with a 0-1 ai-likelihood score and diagnostics.
    """
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)  # DC now at the center
    magnitude = np.log1p(np.abs(fshift))

    profile = _radial_profile(magnitude)
    # Drop the DC bin itself and a couple of neighbors — the DC region
    # is dominated by overall brightness, not structure.
    profile = profile[3:]
    if len(profile) < 20:
        return {
            "score": 0.5,
            "confidence": "low",
            "note": "image too small for reliable frequency analysis",
        }

    n = len(profile)
    outer_start = int(n * 0.85)
    outer_energy_ratio = float(profile[outer_start:].sum() / max(profile.sum(), 1e-6))

    # Fit a line to log(profile) vs log(radius) — natural images are
    # close to a power-law (a straight line in log-log space). Large
    # residuals from that fit indicate an unnatural spectral shape
    # (sharp cutoffs, plateaus, periodic bumps).
    radii = np.arange(1, n + 1)
    log_r = np.log(radii)
    log_p = np.log(np.clip(profile, 1e-6, None))
    slope, intercept = np.polyfit(log_r, log_p, 1)
    fit = slope * log_r + intercept
    residual = log_p - fit
    residual_std = float(np.std(residual))

    # Look for periodic bumps in the residual (grid/checkerboard
    # artifacts from upsampling) via the residual's own spectrum.
    residual_fft = np.abs(np.fft.rfft(residual - residual.mean()))
    if len(residual_fft) > 4:
        peak = float(np.max(residual_fft[2:]))
        baseline = float(np.median(residual_fft[2:]) + 1e-6)
        periodicity_ratio = peak / baseline
    else:
        periodicity_ratio = 1.0

    # --- Combine into a score ---
    # Low outer-ring energy (suppressed high frequencies) -> more AI-like.
    low_energy_signal = float(np.clip((0.05 - outer_energy_ratio) / 0.05, 0, 1))
    # High residual std (spectrum doesn't fit a clean power law) -> more AI-like.
    residual_signal = float(np.clip((residual_std - 0.15) / 0.5, 0, 1))
    # Strong periodic peak relative to baseline -> more AI-like (upsampling grid).
    periodicity_signal = float(np.clip((periodicity_ratio - 4.0) / 10.0, 0, 1))

    score = 0.45 * low_energy_signal + 0.35 * residual_signal + 0.20 * periodicity_signal
    score = float(np.clip(score, 0, 1))

    return {
        "score": round(score, 4),
        "confidence": "medium",
        "outer_ring_energy_ratio": round(outer_energy_ratio, 5),
        "spectral_fit_residual_std": round(residual_std, 4),
        "periodicity_ratio": round(periodicity_ratio, 3),
        "note": "low outer-ring energy or high spectral-fit residual suggests missing sensor noise / decoder upsampling artifacts",
    }
