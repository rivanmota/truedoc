"""
Deterministic test pictures for pipeline mechanics.

``noise_flagged_image`` is a smooth random colour field plus independent
Gaussian grain. The noise and lighting signals read it as synthetic (even
heavy blurring does not change that), so the engine flags it (about 0.64),
also after the JPEG stream a PDF stores at quality 95.
It exists so tests can exercise "the engine flagged a picture" paths (PDF
flagging, the scan API's classification) without a flagged baseline file.

It is a mechanics fixture, not a measurement: nothing here says the
detector recognises real generated images. It does not (see README,
"Limitations"). The calibration script never uses images like this, for
the reason given in its docstring.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def noise_flagged_image(width: int = 1024, height: int = 768, seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    cells = 32
    base = rng.integers(0, 255, (height // cells + 2, width // cells + 2, 3)).astype(np.uint8)
    field = np.asarray(Image.fromarray(base).resize((width, height), Image.BICUBIC)).astype(float)
    grain = rng.normal(0, 4, (height, width, 1))
    return Image.fromarray(np.clip(field + grain, 0, 255).astype(np.uint8))
