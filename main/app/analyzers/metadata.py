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

from PIL import Image
from PIL.ExifTags import TAGS

from app.analyzers.document import AI_GENERATOR_TERMS, find_generator_references

# One list and one matcher for every provenance check (PNG chunks, EXIF
# Software, the PDF Info dictionary here and in document.py), so the same
# file cannot get different verdicts per path. Terms match as whole tokens,
# case-insensitively: bare substrings turned "reflux", "InfluxData",
# "Dallesandro" and "Flux de tresorerie" into generator hits, and the bare
# words "flux", "dalle" and "firefly" are not listed at all (ordinary
# English/French/Italian words; the products are "FLUX.1", "DALL-E",
# "Adobe Firefly"). See document.AI_GENERATOR_TERMS.
KNOWN_GENERATOR_SOFTWARE = AI_GENERATOR_TERMS


EXIF_SUB_IFD_TAG = 0x8769  # "ExifOffset" — pointer from IFD0 to the Exif sub-IFD


def _get_exif(img: Image.Image) -> dict:
    """
    Flatten the image's EXIF into one ``{tag_name: value}`` dict.

    Pillow's ``getexif()`` only exposes IFD0 directly (Make, Model,
    Software, Orientation, ...). The exposure-related tags a camera
    writes — FNumber, ExposureTime, ISOSpeedRatings, DateTimeOriginal —
    live in the Exif sub-IFD (tag 0x8769), so that has to be merged in
    too, otherwise a genuine camera photo only ever surfaces Make/Model
    and never looks "plausible" to the camera-EXIF check below.

    IFD0 wins on a key collision (there shouldn't be one in a
    well-formed file). A corrupt sub-IFD is ignored rather than
    discarding the IFD0 data alongside it.
    """
    try:
        raw = img.getexif()
        if not raw:
            return {}
        merged = {TAGS.get(k, k): v for k, v in raw.items()}
    except Exception:
        return {}

    try:
        sub_ifd = raw.get_ifd(EXIF_SUB_IFD_TAG)
    except Exception:
        sub_ifd = {}
    for k, v in sub_ifd.items():
        merged.setdefault(TAGS.get(k, k), v)
    return merged


def _scan_png_text_chunks(img: Image.Image) -> list[str]:
    hits = []
    info = getattr(img, "text", {}) or {}
    for key, value in info.items():
        key_l = str(key).lower()
        val_str = str(value)
        if key_l in ("parameters", "prompt", "workflow"):
            hits.append(f"PNG text chunk '{key}' present (generator metadata)")
        for name in find_generator_references(f"{key}\n{val_str}"):
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
    for name in find_generator_references(software):
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
        # Same "field: value" text document.analyze_pdf_metadata matches.
        for name in find_generator_references(f"{field}: {value}"):
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
