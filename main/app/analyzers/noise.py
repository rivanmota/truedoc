"""
Sensor-noise residual analysis.

Every real camera sensor leaves a faint, roughly-random noise pattern
in flat/low-detail regions of a photo (photon shot noise, read noise,
PRNU). Generative decoders almost never reproduce this faithfully —
images either come out too clean (denoised look) or, if noise was
added synthetically in post, it tends to be too spatially uniform.

Approach: estimate a "clean" version of the image with a denoising
filter, take the residual (original - denoised), and measure how
that residual behaves in flat regions:
  - overall residual magnitude (too low -> suspiciously clean)
  - spatial autocorrelation of the residual (real sensor noise is
    close to spatially white / uncorrelated; synthetic uniform noise
    or over-smoothing shows up as either near-zero or unnaturally
    structured residuals)
"""
from __future__ import annotations

import cv2
import numpy as np


def _flat_region_mask(gray: np.ndarray) -> np.ndarray:
    """Rough mask of low-texture regions, where sensor noise is easiest
    to observe (edges/high-detail regions confound the estimate)."""
    grad = cv2.Laplacian(gray, cv2.CV_32F)
    grad_mag = np.abs(grad)
    thresh = np.percentile(grad_mag, 60)
    return grad_mag < thresh


def analyze(gray: np.ndarray) -> dict:
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

    denoised = cv2.medianBlur(gray_u8, 5)
    residual = gray_u8.astype(np.float32) - denoised.astype(np.float32)

    flat_mask = _flat_region_mask(gray_u8)
    if flat_mask.sum() < 200:
        return {
            "score": 0.5,
            "confidence": "low",
            "note": "not enough flat/low-detail area to estimate a noise floor reliably",
        }

    flat_residual = residual[flat_mask]
    noise_std = float(np.std(flat_residual))

    # Spatial autocorrelation at lag 1 within flat regions — real sensor
    # noise is close to white noise (autocorr near 0); an over-smoothed
    # AI image has near-zero residual everywhere (autocorr undefined /
    # meaningless at that point, handled by noise_std being tiny),
    # while some synthetic noise or compression blocking shows up as
    # noticeable structure at lag 1.
    res_img = residual * flat_mask
    shifted = np.roll(res_img, 1, axis=1)
    valid = flat_mask & np.roll(flat_mask, 1, axis=1)
    if valid.sum() > 200:
        a = res_img[valid]
        b = shifted[valid]
        if np.std(a) > 1e-3 and np.std(b) > 1e-3:
            autocorr = float(np.corrcoef(a, b)[0, 1])
        else:
            autocorr = 0.0
    else:
        autocorr = 0.0

    # --- Combine ---
    # Very low noise std -> suspiciously clean -> more AI-like.
    # Typical real-camera flat-region noise std lands roughly 1.5-6 for
    # 8-bit images depending on ISO/compression; below ~0.8 is a strong
    # "too clean" signal.
    clean_signal = float(np.clip((1.2 - noise_std) / 1.2, 0, 1))
    # High |autocorrelation| -> structured rather than white -> more AI-like.
    structure_signal = float(np.clip((abs(autocorr) - 0.15) / 0.5, 0, 1))

    score = 0.75 * clean_signal + 0.25 * structure_signal
    score = float(np.clip(score, 0, 1))

    return {
        "score": round(score, 4),
        "confidence": "medium",
        "flat_region_noise_std": round(noise_std, 3),
        "residual_autocorrelation": round(autocorr, 3),
        "note": "very low noise std in flat regions suggests no genuine sensor noise (over-clean / AI-decoded look)",
    }
