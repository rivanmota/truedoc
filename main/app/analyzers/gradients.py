"""
Gradient (texture) smoothness analysis, focused on skin and background.

Real skin has fine, irregular high-frequency texture: pores, peach
fuzz, small color variation. AI-generated skin is frequently
over-smoothed ("airbrushed") because the generator hasn't learned
that fine texture, or averages it away. Flat backgrounds (walls, sky,
studio backdrops) behave the opposite way: real ones often still show
subtle gradients (uneven lighting, paint texture, JPEG blocking);
AI backgrounds can be *unnaturally* perfectly flat OR show a subtly
wrong kind of noise.

This analyzer:
  1. finds a rough skin mask (HSV/YCrCb thresholding — cheap, no
     model needed)
  2. computes Sobel gradient magnitude over the image
  3. compares gradient variance inside the skin mask against a
     reference range for real photos
  4. does the same for large flat/background regions
"""
from __future__ import annotations

import cv2
import numpy as np


def _skin_mask(bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    return mask > 0


def _flat_background_mask(gray: np.ndarray) -> np.ndarray:
    grad = cv2.Laplacian(gray, cv2.CV_32F)
    grad_mag = np.abs(grad)
    thresh = np.percentile(grad_mag, 40)
    return grad_mag < thresh


def analyze(bgr: np.ndarray, gray: np.ndarray) -> dict:
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    sobel_x = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    result = {"skin": None, "background": None}

    skin_mask = _skin_mask(bgr)
    skin_fraction = float(skin_mask.mean())
    if skin_fraction > 0.02:
        skin_grad = grad_mag[skin_mask]
        skin_var = float(np.var(skin_grad))
        # Real skin macro/portrait texture typically keeps some grain;
        # heavily under a threshold suggests airbrushed/generated skin.
        skin_smooth_signal = float(np.clip((60.0 - skin_var) / 60.0, 0, 1))
        result["skin"] = {
            "coverage_fraction": round(skin_fraction, 3),
            "gradient_variance": round(skin_var, 2),
            "smoothness_signal": round(skin_smooth_signal, 4),
        }
    else:
        skin_smooth_signal = None

    bg_mask = _flat_background_mask(gray_u8)
    bg_fraction = float(bg_mask.mean())
    if bg_fraction > 0.05:
        bg_grad = grad_mag[bg_mask]
        bg_var = float(np.var(bg_grad))
        bg_smooth_signal = float(np.clip((8.0 - bg_var) / 8.0, 0, 1))
        result["background"] = {
            "coverage_fraction": round(bg_fraction, 3),
            "gradient_variance": round(bg_var, 2),
            "smoothness_signal": round(bg_smooth_signal, 4),
        }
    else:
        bg_smooth_signal = None

    signals = [s for s in (skin_smooth_signal, bg_smooth_signal) if s is not None]
    if not signals:
        return {
            "score": 0.5,
            "confidence": "low",
            "note": "no clear skin or flat-background region found to evaluate texture on",
            **result,
        }

    score = float(np.clip(np.mean(signals), 0, 1))
    confidence = "medium" if len(signals) == 2 else "low-medium"

    return {
        "score": round(score, 4),
        "confidence": confidence,
        "note": "unusually low gradient variance in skin/background regions suggests over-smoothed (airbrushed) generated texture",
        **result,
    }
