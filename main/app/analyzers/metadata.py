"""
Metadata analysis.

Two very different kinds of evidence live here:

1. Explicit generator signatures. Several popular tools write their
   own metadata into the file and this is a near-certain signal when
   present (though trivially stripped by re-saving, so its absence
   proves nothing):
     - Stable Diffusion WebUI (AUTOMATIC1111 and forks) writes a
       "parameters" tEXt/PNG chunk with the full prompt + sampler info
     - ComfyUI writes "prompt" / "workflow" JSON chunks
     - Some tools set the EXIF Software tag to their own name
       (e.g. "Midjourney", "DALL-E", "Adobe Firefly", "NightCafe",
       "Leonardo.Ai")

2. Ordinary camera EXIF plausibility. A real photo straight off a
   phone or camera usually carries Make/Model/Exposure/FNumber/ISO.
   Its presence is a mild signal *toward* "real" — but far from
   proof, since it's easy to fake and many legitimately real images
   (screenshots, social re-uploads, edited exports) lose EXIF for
   unrelated reasons. So presence lowers the score a little;
   absence is treated as close to neutral, not as evidence of AI.
"""
from __future__ import annotations

import io
import re

from PIL import Image
from PIL.ExifTags import TAGS

KNOWN_GENERATOR_SOFTWARE = [
    "midjourney", "dall-e", "dalle", "adobe firefly", "firefly",
    "nightcafe", "leonardo.ai", "leonardo ai", "stable diffusion",
    "comfyui", "automatic1111", "invokeai", "runwayml", "ideogram",
    "playground ai", "flux", "recraft", "seedream", "nano banana",
]


def _get_exif(img: Image.Image) -> dict:
    try:
        raw = img.getexif()
        if not raw:
            return {}
        return {TAGS.get(k, k): v for k, v in raw.items()}
    except Exception:
        return {}


def _scan_png_text_chunks(img: Image.Image) -> list[str]:
    hits = []
    info = getattr(img, "text", {}) or {}
    for key, value in info.items():
        key_l = str(key).lower()
        val_str = str(value)
        if key_l in ("parameters", "prompt", "workflow"):
            hits.append(f"PNG text chunk '{key}' present (generator metadata)")
        for name in KNOWN_GENERATOR_SOFTWARE:
            if name in val_str.lower() or name in key_l:
                hits.append(f"PNG text chunk '{key}' references '{name}'")
    return hits


def analyze(image_bytes: bytes) -> dict:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:
        return {"score": 0.5, "confidence": "low", "note": f"could not parse image metadata: {e}"}

    signature_hits = []

    # PNG tEXt/iTXt chunks (Stable Diffusion WebUI, ComfyUI, etc.)
    if img.format == "PNG":
        signature_hits.extend(_scan_png_text_chunks(img))

    exif = _get_exif(img)
    software = str(exif.get("Software", ""))
    for name in KNOWN_GENERATOR_SOFTWARE:
        if name in software.lower():
            signature_hits.append(f"EXIF Software tag references '{name}'")

    if signature_hits:
        return {
            "score": 0.97,
            "confidence": "high",
            "signature_hits": signature_hits,
            "note": "explicit generator metadata found — strong signal when present, but trivially removed by re-saving/stripping, so its absence elsewhere carries little weight",
        }

    camera_fields = ["Make", "Model", "FNumber", "ExposureTime", "ISOSpeedRatings", "DateTimeOriginal"]
    present = [f for f in camera_fields if exif.get(f)]

    if len(present) >= 3:
        return {
            "score": 0.15,
            "confidence": "low-medium",
            "camera_fields_present": present,
            "note": "plausible camera EXIF present — mild signal toward a real capture, though fakeable",
        }

    return {
        "score": 0.5,
        "confidence": "low",
        "camera_fields_present": present,
        "note": "no generator signature and no/partial camera EXIF — this is common for real images too (screenshots, re-uploads, exports), so treated as neutral",
    }


def analyze_pdf_document(pdf_fields: dict[str, str]) -> dict:
    """
    Checks a PDF's document Info dict (Title/Author/Creator/Producer/
    Subject/Keywords) for a generator name. This is a much weaker
    signal than the PNG/EXIF checks above — most PDFs (scanned docs,
    exports from Word/Google Docs/LaTeX) will just show a mundane
    Producer like a print driver or "Skia/PDF", and rasterizing each
    page for the visual analyzers strips whatever metadata the
    original embedded image had. So this only ever raises the score
    (an explicit hit), never lowers it — absence here says nothing
    about the pages, that's what the per-page visual signals are for.
    """
    signature_hits = []
    for field, value in pdf_fields.items():
        value_l = str(value).lower()
        for name in KNOWN_GENERATOR_SOFTWARE:
            if name in value_l:
                signature_hits.append(f"PDF metadata field '{field}' references '{name}'")

    if signature_hits:
        return {
            "score": 0.9,
            "confidence": "high",
            "signature_hits": signature_hits,
            "pdf_fields": pdf_fields,
            "note": "explicit generator reference found in the PDF's document metadata",
        }

    return {
        "score": 0.5,
        "confidence": "low",
        "pdf_fields": pdf_fields or None,
        "note": "no generator reference in PDF document metadata — not meaningful either way, since rasterizing each page for analysis strips any metadata the original embedded image carried; treat the per-page visual signals as primary evidence for PDFs",
    }
