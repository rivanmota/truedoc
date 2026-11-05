"""
Lighting / shading consistency analysis.

Full shadow-geometry forensics (tracing cast shadows back to a
consistent light source position via vanishing-point analysis) is a
serious computer-vision project on its own and needs shadow/object
segmentation to do properly. What's implemented here is a much
cheaper proxy: under simple Lambertian shading, local intensity
gradients on shaded surfaces tend to point toward the light source.
We estimate a dominant "shading direction" per region and check how
consistent that is across the image.

This is the noisiest of the four signals — texture, albedo changes,
and multiple light sources all produce legitimate variation in real
photos too, and plenty of real images (studio product shots, overcast
outdoor light, multi-light scenes) have low or inconsistent gradient
coherence for reasons that have nothing to do with authenticity. It's
still useful when combined with the other three, so it's included,
but weighted low and always labeled with reduced confidence.
"""
from __future__ import annotations

import cv2
import numpy as np


def analyze(gray: np.ndarray, grid: int = 6) -> dict:
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    h, w = gray_u8.shape
    if h < grid * 8 or w < grid * 8:
        return {
            "score": 0.5,
            "confidence": "low",
            "applicable": False,
            "note": "image too small for a grid-based lighting estimate",
        }

    sobel_x = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=5)

    cell_h, cell_w = h // grid, w // grid
    angles = []
    weights = []

    for i in range(grid):
        for j in range(grid):
            gx = sobel_x[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            gy = sobel_y[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            mag = np.sqrt(gx ** 2 + gy ** 2)
            mean_mag = float(mag.mean())
            if mean_mag < 2.0:
                continue  # flat cell, no reliable gradient direction
            # Dominant gradient direction for the cell, magnitude-weighted.
            mean_gx = float(np.sum(gx * mag) / np.sum(mag))
            mean_gy = float(np.sum(gy * mag) / np.sum(mag))
            angle = np.arctan2(mean_gy, mean_gx)
            angles.append(angle)
            weights.append(mean_mag)

    if len(angles) < grid:  # need a reasonable number of textured cells
        return {
            "score": 0.5,
            "confidence": "low",
            "applicable": False,
            "note": "not enough textured regions to estimate lighting direction consistency",
        }

    angles = np.array(angles)
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Circular variance of the weighted gradient-direction distribution.
    # 0 = perfectly consistent single direction, 1 = fully scattered.
    sin_sum = float(np.sum(weights * np.sin(angles)))
    cos_sum = float(np.sum(weights * np.cos(angles)))
    resultant_length = float(np.sqrt(sin_sum ** 2 + cos_sum ** 2))
    circular_variance = float(1 - resultant_length)

    # High variance -> inconsistent shading cues -> mild AI-likelihood signal.
    # Kept deliberately gentle: this alone should never dominate the score.
    score = float(np.clip((circular_variance - 0.55) / 0.35, 0, 1))

    return {
        "score": round(score, 4),
        "confidence": "low",
        "applicable": True,
        "circular_variance": round(circular_variance, 4),
        "cells_used": len(angles),
        "note": "weakest signal of the four — treat as a minor corroborating factor only, not standalone evidence",
    }
