#!/usr/bin/env python3
"""Run the baseline assets through the real analysis pipeline and validate
that the live output obeys every rule asserted by the unit tests in
``app/analyzers/verify_document.py``.

The unit tests exercise ``document.py`` with hand-built signal dicts. This
script closes the loop with real data:

  Step 1  Run the (unmodified) unit-test suite and require it to pass.
  Step 2  Run each baseline image through ``_run_image_analysis``.
  Step 3  Run each baseline PDF, untouched, through ``_run_pdf_analysis``.
  Step 4  Compose new PDFs *from the baseline images* that reproduce the
          scenarios the tests describe (multi-page with one clearly worse
          page, the 0.40 flag threshold, icon-sized rasters, metadata
          override, all-native documents, coverage independence) and run
          them through the same pipeline.
  Step 5  For every PDF result, independently recompute each stage with
          ``document.py`` and check the pipeline's numbers agree, plus
          check the structural invariants the tests assert (worst page
          wins, flagged == reliable pages >= 0.40, rounding, tiers, ...).

Only project dependencies are used (numpy, opencv, pillow, pypdfium2 via
``app``); PDFs are written by hand so nothing extra needs installing.

Usage (from ``main/``):

    python verify_baseline.py                 # full run
    python verify_baseline.py -v              # also print every passing check
    python verify_baseline.py --json out.json # dump all pipeline output
    python verify_baseline.py --save-pdfs DIR # keep the synthesized PDFs
    python verify_baseline.py --baseline PATH # baseline dir (default ../baseline)
    python verify_baseline.py --skip-unit-tests

Exit status is 0 when no check FAILed. WARN lines never affect the exit
status; they flag environment-dependent behaviour that could not be
exercised or that contradicts the README's stated expectations.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import unittest
import zlib
from typing import Any

_MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)
os.chdir(_MAIN_DIR)

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from app import main as app_main  # noqa: E402
from app import scoring as shared_scoring  # noqa: E402
from app.analyzers import document  # noqa: E402

SIGNAL_NAMES = ("frequency", "noise", "gradients", "lighting")
RELIABLE = {"low-medium", "medium", "medium-high", "high"}
FLAG_THRESHOLD = 0.40
PAGE_W, PAGE_H = 612, 792

BASELINE_FILES = {
    "human": "jpg/human.jpeg",
    "ai_generated": "png/ai-generated.png",
    "ai_retouched": "png/ai-retouched.png",
    "pdf_baseline": "pdf/baseline_sample.pdf",
    "pdf_anomaly": "pdf/anomaly_metadata_sample.pdf",
}

# README expectations for the three images (informational only).
README_EXPECTATION = {
    "human": "low",
    "ai_generated": "high",
    "ai_retouched": "in between",
}


# ---------------------------------------------------------------------------
# Check recorder
# ---------------------------------------------------------------------------


class Checks:
    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.rows: list[tuple[str, str, str, str]] = []  # (status, group, name, detail)

    def _add(self, status: str, group: str, name: str, detail: str = ""):
        self.rows.append((status, group, name, detail))
        if status != "PASS" or self.verbose:
            tag = {"PASS": "  ok  ", "FAIL": " FAIL ", "WARN": " warn ", "INFO": " info "}[status]
            line = f"[{tag}] {group} :: {name}"
            if detail:
                line += f"  --  {detail}"
            print(line)

    def check(self, group: str, name: str, condition: bool, detail: str = ""):
        self._add("PASS" if condition else "FAIL", group, name, detail)
        return condition

    def equal(self, group: str, name: str, actual, expected, detail: str = ""):
        ok = actual == expected
        if not ok:
            detail = (detail + " | " if detail else "") + f"actual={_short(actual)} expected={_short(expected)}"
        return self.check(group, name, ok, detail)

    def close(self, group: str, name: str, actual, expected, tol=1e-4, detail: str = ""):
        ok = abs(float(actual) - float(expected)) <= tol
        if not ok:
            detail = (detail + " | " if detail else "") + f"actual={actual} expected={expected} tol={tol}"
        return self.check(group, name, ok, detail)

    def warn(self, group: str, name: str, detail: str = ""):
        self._add("WARN", group, name, detail)

    def info(self, group: str, name: str, detail: str = ""):
        self._add("INFO", group, name, detail)

    def counts(self) -> dict[str, int]:
        out = {"PASS": 0, "FAIL": 0, "WARN": 0, "INFO": 0}
        for status, *_ in self.rows:
            out[status] += 1
        return out


def _pypdfium2_version() -> str:
    try:
        import importlib.metadata as md
        return md.version("pypdfium2")
    except Exception:
        return "?"


def _short(value, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Minimal PDF writer (text + raster images), no third-party PDF library
# ---------------------------------------------------------------------------


def _image_xobject(img_bytes: bytes) -> tuple[bytes, bytes]:
    """Return (dictionary, stream data) for an RGB image XObject. Baseline
    JPEGs are embedded byte-for-byte (DCTDecode) so their original
    compression is preserved; everything else is embedded as Flate RGB."""
    im = Image.open(io.BytesIO(img_bytes))
    if im.format == "JPEG" and im.mode == "RGB":
        data, filt = img_bytes, "/DCTDecode"
        w, h = im.size
    else:
        im = im.convert("RGB")
        w, h = im.size
        data, filt = zlib.compress(im.tobytes()), "/FlateDecode"
    d = (
        f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
        f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter {filt} /Length {len(data)} >>"
    )
    return d.encode("latin-1"), data


def _pdf_string(value: str) -> str:
    return "(" + value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"


def build_pdf(pages: list[dict], info: dict[str, str] | None = None) -> bytes:
    """Write a PDF 1.4 file.

    ``pages`` entries: ``{"text": [lines], "images": [(bytes, x, y, w, h)]}``
    where x/y/w/h are in points on a 612x792 page. ``info`` fills the
    document Info dictionary (Title/Author/Creator/Producer...).
    """
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    staged = []
    for page in pages:
        xobjects = []
        content = []
        for i, (img_bytes, x, y, w, h) in enumerate(page.get("images", [])):
            dictionary, data = _image_xobject(img_bytes)
            xid = add(dictionary + b"\nstream\n" + data + b"\nendstream")
            xobjects.append((f"/Im{i}", xid))
            content.append(f"q {w} 0 0 {h} {x} {y} cm /Im{i} Do Q")
        if page.get("text"):
            content.append("BT /F1 12 Tf 72 720 Td 14 TL")
            for line in page["text"]:
                content.append(f"{_pdf_string(line)} Tj T*")
            content.append("ET")
        stream = "\n".join(content).encode("latin-1")
        cid = add(f"<< /Length {len(stream)} >>".encode() + b"\nstream\n" + stream + b"\nendstream")
        staged.append((cid, xobjects))

    pages_id = len(objects) + len(staged) + 1
    page_ids = []
    for cid, xobjects in staged:
        xo = " ".join(f"{name} {oid} 0 R" for name, oid in xobjects)
        page_ids.append(add(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Contents {cid} 0 R /Resources << /Font << /F1 {font_id} 0 R >> "
            f"/XObject << {xo} >> >> >>".encode()
        ))
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    assert add(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()) == pages_id
    catalog_id = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())
    info_id = None
    if info:
        body = " ".join(f"/{k} {_pdf_string(v)}" for k, v in info.items())
        info_id = add(f"<< {body} >>".encode("latin-1"))

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for oid, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{oid} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R"
    if info_id:
        trailer += f" /Info {info_id} 0 R"
    trailer += f" >>\nstartxref\n{xref}\n%%EOF\n"
    out.write(trailer.encode())
    return out.getvalue()


# ---------------------------------------------------------------------------
# Baseline-derived rasters
# ---------------------------------------------------------------------------


def to_png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def center_crop_to_page_aspect(im: Image.Image) -> Image.Image:
    """Crop (never resample) so the raster has the page's aspect ratio. A
    raster placed on the page with its own aspect ratio is extracted by PDFium
    at native resolution, pixel-for-pixel; a stretched placement would be
    resampled on extraction and the forensic signals would no longer describe
    the baseline pixels."""
    target = PAGE_W / PAGE_H
    w, h = im.size
    if w / h > target:
        nw, nh = int(round(h * target)), h
    else:
        nw, nh = w, int(round(w / target))
    left, top = (w - nw) // 2, (h - nh) // 2
    return im.crop((left, top, left + nw, top + nh))


def derive_rasters(assets: dict[str, bytes]) -> dict[str, tuple[bytes, tuple[int, int]]]:
    """Every raster used in the synthesized PDFs comes from a baseline file.
    Returns ``name -> (png_bytes, (width_px, height_px))``."""
    out: dict[str, tuple[bytes, tuple[int, int]]] = {}

    def put(name: str, im: Image.Image):
        out[name] = (to_png(im), im.size)

    human = Image.open(io.BytesIO(assets["human"])).convert("RGB")
    ai = Image.open(io.BytesIO(assets["ai_generated"])).convert("RGB")
    ret = Image.open(io.BytesIO(assets["ai_retouched"])).convert("RGB")

    # Page-aspect center crops of the untouched baseline pixels.
    put("human", center_crop_to_page_aspect(human))
    put("ai_generated", center_crop_to_page_aspect(ai))
    put("ai_retouched", center_crop_to_page_aspect(ret))
    # Heavily smoothed copy of the AI image. Blur erases the grain and
    # texture the noise/gradients signals measure, so this raster carries
    # no usable evidence: it must come out insufficient, never flagged.
    # (Earlier detector versions read "too clean" as AI evidence and this
    # was the "clearly worse" page.)
    put("ai_generated_blur4", center_crop_to_page_aspect(ai.filter(ImageFilter.GaussianBlur(4))))
    # Icon-sized copy: below the 64px / 16,384px gate.
    put("ai_generated_icon48", ai.resize((48, 48), Image.LANCZOS))

    # Decorative fills, as found in letterheads/brochures. Not derived from the
    # baseline images: they exist to prove non-photographic rasters cannot
    # drive document risk (flat/gradient-fill gate).
    x = np.linspace(0.0, 1.0, 1200)[None, :, None]
    banner = np.tile(np.array([30, 60, 140])[None, None, :] * (1 - x) + 255 * x, (200, 1, 1))
    put("gradient_banner", Image.fromarray(banner.astype(np.uint8)))
    yy, xx = np.mgrid[0:800, 0:800]
    rr = np.sqrt((yy - 400) ** 2 + (xx - 400) ** 2) / 566.0
    vignette = np.tile((255 * (1 - 0.8 * rr))[:, :, None], (1, 1, 3))
    put("radial_vignette", Image.fromarray(vignette.astype(np.uint8)))
    return out


TEXT_LINES = [
    "Baseline validation document.",
    "This page is native PDF text rendered with the standard Helvetica font.",
    "It contains enough characters for the page profiler to classify it as a",
    "native document, so whole-page photo forensics must not be applied here.",
    "Any raster image placed on such a page is still extracted and analyzed",
    "independently, regardless of how little page area it occupies.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod",
    "tempor incididunt ut labore et dolore magna aliqua.",
]


def place(raster: tuple[bytes, tuple[int, int]], scale: float, x: float = 72, y: float = 72) -> tuple:
    """Place a raster at an exact fraction of its pixel size (1pt per px at
    scale 1). Dyadic scales (1, 0.5, 0.25, 0.125) extract pixel-exact from
    PDFium; arbitrary scales such as 0.1 round the extracted bitmap by 1px."""
    data, (w, h) = raster
    return (data, x, y, w * scale, h * scale)


def full_page(raster: tuple[bytes, tuple[int, int]]) -> dict:
    data, (w, h) = raster
    s = min(PAGE_W / w, PAGE_H / h)
    return {"images": [(data, 0, 0, w * s, h * s)], "expect_sizes": [(w, h)]}


def text_page(placements: list | None = None) -> dict:
    placements = placements or []
    return {"text": TEXT_LINES, "images": placements, "expect_sizes": None}


def text_page_with(raster: tuple[bytes, tuple[int, int]], scale: float) -> dict:
    page = text_page([place(raster, scale)])
    page["expect_sizes"] = [raster[1]]
    return page


def build_scenarios(rasters: dict[str, tuple[bytes, tuple[int, int]]]) -> list[dict]:
    generic_info = {
        "Title": "Synthesized from baseline assets",
        "Creator": "verify_baseline.py",
        "Producer": "Adobe Acrobat Pro DC 23.1",
    }
    return [
        {
            "name": "multi_page_one_clearly_worse",
            "description": (
                "5 pages: native text, human.jpeg, ai-generated.png (page 3, expected worst), "
                "BLURRED ai-generated.png (evidence erased, must not be flagged), native text"
            ),
            "pages": [
                text_page(),
                full_page(rasters["human"]),
                full_page(rasters["ai_generated"]),
                full_page(rasters["ai_generated_blur4"]),
                text_page(),
            ],
            "info": generic_info,
            "expect": {"worst_page": 3, "flagged_exactly": [3], "metadata_hit": False},
        },
        {
            "name": "every_ai_page_is_flagged",
            "description": "4 pages: native text, human.jpeg, ai-retouched.png, ai-generated.png; both AI pages must be flagged, the real photo must not",
            "pages": [
                text_page(),
                full_page(rasters["human"]),
                full_page(rasters["ai_retouched"]),
                full_page(rasters["ai_generated"]),
            ],
            "info": generic_info,
            "expect": {"flagged_exactly": [4, 3], "metadata_hit": False},
        },
        {
            "name": "small_embedded_on_text_page",
            "description": (
                "3 pages: native text, native text with the AI raster at 1/4 scale (~6% coverage), "
                "native text with human.jpeg at 1/4 scale"
            ),
            "pages": [
                text_page(),
                text_page_with(rasters["ai_generated"], 0.25),
                text_page_with(rasters["human"], 0.25),
            ],
            "info": generic_info,
            "expect": {"worst_page": 2, "worst_source": "embedded_image", "metadata_hit": False},
        },
        {
            "name": "coverage_independence",
            "description": "same AI raster full-page (p1) and at 1/8 scale (~1.4% coverage) on a text page (p2)",
            "pages": [
                full_page(rasters["ai_generated"]),
                text_page_with(rasters["ai_generated"], 0.125),
            ],
            "info": generic_info,
            "expect": {"same_embedded_score_pages": (1, 2), "metadata_hit": False},
        },
        {
            "name": "icon_sized_raster_only",
            "description": "native text page with a 48x48 icon derived from ai-generated.png",
            "pages": [text_page_with(rasters["ai_generated_icon48"], 1.0)],
            "info": generic_info,
            "expect": {"insufficient": True, "embedded_too_small": True, "metadata_hit": False},
        },
        {
            "name": "all_native_text",
            "description": "3 native text pages, no rasters",
            "pages": [text_page(), text_page(), text_page()],
            "info": generic_info,
            "expect": {"insufficient": True, "metadata_hit": False},
        },
        {
            "name": "generator_metadata_override",
            "description": "native text + human.jpeg page, Creator field names Midjourney",
            "pages": [text_page(), full_page(rasters["human"])],
            "info": {**generic_info, "Creator": "Midjourney Bot v6"},
            "expect": {"metadata_hit": True, "generator_references": ["midjourney"]},
        },
        {
            "name": "generic_producer_is_neutral",
            "description": "same document with only generic producer/creator strings",
            "pages": [text_page(), full_page(rasters["human"])],
            "info": {"Creator": "Microsoft Word", "Producer": "macOS Quartz PDFContext"},
            "expect": {"metadata_hit": False, "generator_references": []},
        },
        {
            "name": "common_words_are_not_provenance",
            "description": "French cash-flow statement: Title 'Flux de trésorerie', Subject mentions 'dalles', Author 'Dallesandro'",
            "pages": [text_page(), full_page(rasters["human"])],
            "info": {"Title": "Flux de tresorerie 2024", "Subject": "Pose de dalles - reflux - InfluxData export",
                     "Author": "Maria Dallesandro", "Producer": "Firefly Print Server"},
            "expect": {"metadata_hit": False, "generator_references": []},
        },
        {
            "name": "decorative_fills_cannot_drive_risk",
            "description": (
                "4 pages: text + gradient banner, text + radial vignette, solid full-page vignette, "
                "text + human.jpeg at 1/4 scale; fills must be gated out, the photo must win"
            ),
            "pages": [
                text_page([place(rasters["gradient_banner"], 0.5, 72, 680)]),
                text_page([place(rasters["radial_vignette"], 0.5, 72, 300)]),
                full_page(rasters["radial_vignette"]),
                text_page_with(rasters["human"], 0.25),
            ],
            "info": generic_info,
            # The fills are gated out; the real photo (scored with the image-path
            # weights: frequency + lighting, low-medium evidence) is the only
            # reliable element and reads as real, so nothing is flagged.
            "expect": {"metadata_hit": False, "worst_page": 4, "worst_source": "embedded_image",
                       "flat_fill_pages": [1, 2, 3], "flagged_exactly": []},
        },
    ]


# ---------------------------------------------------------------------------
# Validation of a pipeline PDF result against document.py
# ---------------------------------------------------------------------------


def tier_for(ews: float) -> str:
    ews = round(ews, 4)
    if ews >= 0.65:
        return "medium-high"
    if ews >= 0.40:
        return "medium"
    if ews >= 0.20:
        return "low-medium"
    return "low"


def is_rounded4(value) -> bool:
    return value == round(float(value), 4)


def validate_scoring_internals(c: Checks, group: str, label: str, scoring: dict):
    """Checks that apply to any score_pdf_page / score_embedded_image result."""
    c.check(group, f"{label}: score within [0,1] and 4dp", 0.0 <= scoring["score"] <= 1.0 and is_rounded4(scoring["score"]))
    if scoring.get("confidence") == "insufficient":
        c.equal(group, f"{label}: insufficient => score 0 / usable False", (scoring["score"], scoring["usable"]), (0.0, False))
        return
    contributions = scoring.get("contributions", {})
    ews = 0.0
    gated = {}
    for name, contrib in contributions.items():
        if contrib.get("used"):
            ews += contrib["effective_weight"]
            factor = document.CONFIDENCE_FACTOR.get(contrib["confidence"], shared_scoring.UNKNOWN_CONFIDENCE_FACTOR)
            c.check(group, f"{label}: {name} weight == image-path base * factor",
                    abs(contrib["effective_weight"] - round(shared_scoring.WEIGHTS[name] * factor, 4)) < 1e-9)
            gated[name] = {"score": contrib["score"], "confidence": contrib["confidence"], "applicable": True}
        else:
            c.equal(group, f"{label}: {name} unused => weight 0", contrib.get("effective_weight"), 0.0)
            gated[name] = {"score": None, "applicable": False}
    c.close(group, f"{label}: effective_weight_sum == sum of used weights", scoring["effective_weight_sum"], ews, tol=2e-4)
    c.equal(group, f"{label}: confidence tier matches rounded weight sum", scoring["confidence"], tier_for(scoring["effective_weight_sum"]))
    c.equal(group, f"{label}: usable == weight sum >= 0.15", scoring["usable"], scoring["effective_weight_sum"] >= 0.15)
    if ews > 1e-9:
        # Independent recomputation through the shared image-path combiner,
        # under the context buckets the element was scored with.
        buckets = scoring.get("combiner", {}).get("context_buckets")
        expected, diag = shared_scoring.visual_score(gated, buckets)
        pixel_score = scoring.get("visual_score", scoring["score"])
        c.close(group, f"{label}: pixel score == image-path visual_score of the used signals", pixel_score, expected, tol=2e-4)
        if "visual_score" in scoring:
            # The embedded image's own file metadata, folded in like an upload.
            c.close(group, f"{label}: score == scoring.combine(pixel score, file metadata)",
                    scoring["score"], shared_scoring.combine(pixel_score, scoring["metadata"]), tol=2e-4)
        c.close(group, f"{label}: raw_normalized_score == uncapped image-path score",
                scoring["raw_normalized_score"], diag["uncapped_score"], tol=2e-4)
        c.close(group, f"{label}: evidence_coverage == weight sum / 1.0", scoring["evidence_coverage"], ews, tol=2e-4)


def validate_pdf_result(c: Checks, group: str, result: dict):
    pages = result["pages"]
    aggregate = result["signals"]["document_aggregate"]
    explicit_meta = result["signals"]["pdf_generator_metadata"]
    legacy_meta = result["signals"]["metadata"]

    for page in pages:
        n = page["page"]
        profile = page["page_profile"]
        render = page["page_render_scoring"]
        risk = page["pdf_scoring"]
        embedded = page["embedded_images"]
        label = f"p{n}"

        # --- page render path -------------------------------------------
        c.check(group, f"{label}: page profile carries raster stats",
                isinstance(profile.get("unique_colors"), int) and isinstance(profile.get("edge_density"), float))
        if not profile.get("photo_forensics_applicable"):
            c.equal(group, f"{label}: non-applicable page render is insufficient", render["confidence"], "insufficient")
            placeholder = document.not_applicable_signals(profile)
            c.equal(group, f"{label}: non-applicable page render == score_pdf_page(not_applicable_signals)",
                    render, document.score_pdf_page(placeholder, profile))
            if risk.get("source") != "embedded_image":
                c.check(group, f"{label}: non-applicable page signals are None",
                        all(page["signals"][s]["score"] is None for s in SIGNAL_NAMES))
        else:
            if risk.get("source") != "embedded_image":
                c.equal(group, f"{label}: page render == score_pdf_page(signals, profile, buckets)",
                        render, document.score_pdf_page(page["signals"], profile,
                                                        render.get("combiner", {}).get("context_buckets")))
            else:
                c.info(group, f"{label}: applicable page driven by embedded image; render signals not recoverable from response")
        validate_scoring_internals(c, group, f"{label} render", render)

        # --- embedded images ---------------------------------------------
        for img in embedded:
            scoring = img["scoring"]
            il = f"{label} img{img['image_index']}"
            if "analysis failed" in scoring.get("note", ""):
                c.warn(group, f"{il}: embedded analysis failed upstream", scoring["note"])
                continue
            img_profile = scoring["image_profile"]
            c.check(group, f"{il}: raster stats present (unique_colors, edge_density)",
                    isinstance(img_profile.get("unique_colors"), int) and isinstance(img_profile.get("edge_density"), float))
            c.equal(group, f"{il}: scoring == score_embedded_image(signals, profile, buckets, metadata)",
                    scoring, document.score_embedded_image(img["signals"], img_profile,
                                                           scoring.get("combiner", {}).get("context_buckets"),
                                                           metadata_result=scoring.get("metadata")))
            if document._is_flat_fill(img_profile):
                c.check(group, f"{il}: flat-fill raster is unusable", not scoring["usable"] and "flat or gradient fill" in scoring["note"])
            too_small = (
                img_profile["width_px"] < 64 or img_profile["height_px"] < 64
                or img_profile.get("pixel_count", img_profile["width_px"] * img_profile["height_px"]) < 16_384
            )
            if too_small:
                c.check(group, f"{il}: too-small raster unusable with reason", (not scoring["usable"]) and "too small" in scoring["note"])
            c.equal(group, f"{il}: entry mirrors scoring", (img["ai_generated"], img["confidence"], img["usable"]),
                    (scoring["score"], scoring["confidence"], scoring["usable"]))
            validate_scoring_internals(c, group, il, scoring)

        # --- page risk ------------------------------------------------------
        c.equal(group, f"{label}: pdf_scoring == combine_page_evidence(...)",
                risk, document.combine_page_evidence(n, render, embedded))
        candidates = []
        if render.get("usable") and render.get("confidence") in RELIABLE:
            candidates.append(render["score"])
        candidates += [float(i["ai_generated"]) for i in embedded if i.get("usable") and i.get("confidence") in RELIABLE]
        if candidates:
            c.close(group, f"{label}: page risk == max reliable candidate", risk["score"], max(candidates))
            c.equal(group, f"{label}: candidate_count", risk.get("candidate_count"), len(candidates))
        else:
            c.equal(group, f"{label}: no candidates => insufficient", (risk["confidence"], risk["usable"]), ("insufficient", False))
        c.equal(group, f"{label}: page entry mirrors risk", (page["ai_generated"], page["confidence"], page["risk_source"]),
                (risk["score"], risk["confidence"], risk.get("source")))

    # --- aggregate ----------------------------------------------------------
    c.equal(group, "aggregate == aggregate_pdf_pages(pages)", aggregate, document.aggregate_pdf_pages(pages))
    usable = [p for p in pages if p["pdf_scoring"].get("usable")]
    reliable = [p for p in usable if p["pdf_scoring"].get("confidence") in RELIABLE]
    c.equal(group, "usable_pages", aggregate["usable_pages"], len(usable))
    c.equal(group, "reliable_pages", aggregate["reliable_pages"], len(reliable))
    if reliable:
        worst_score = max(float(p["pdf_scoring"]["score"]) for p in reliable)
        c.close(group, "document score == max reliable page score (never the mean)", aggregate["score"], worst_score)
        c.check(group, "worst_page carries the max score",
                float(next(p for p in pages if p["page"] == aggregate["worst_page"])["pdf_scoring"]["score"]) == worst_score)
        c.check(group, "mean_page_score <= max_page_score", aggregate["mean_page_score"] <= aggregate["max_page_score"] + 1e-9)
        expected_flagged = sorted(
            [p["page"] for p in reliable if float(p["pdf_scoring"]["score"]) >= FLAG_THRESHOLD],
            key=lambda n: -float(next(p for p in pages if p["page"] == n)["pdf_scoring"]["score"]),
        )
        c.equal(group, f"flagged_pages == reliable pages with score >= {FLAG_THRESHOLD}",
                [f["page"] for f in aggregate["flagged_pages"]], expected_flagged)
        c.check(group, "flagged_pages sorted by score desc",
                [f["score"] for f in aggregate["flagged_pages"]] == sorted((f["score"] for f in aggregate["flagged_pages"]), reverse=True))
        c.check(group, "no low-confidence page is flagged or worst",
                all(f["confidence"] in RELIABLE for f in aggregate["flagged_pages"]) and aggregate["confidence"] in RELIABLE)
        c.equal(group, "top-level flagged_pages mirrors aggregate", result["flagged_pages"], aggregate["flagged_pages"])
        c.equal(group, "worst evidence attribution",
                (result["worst_page"], result["worst_evidence_source"], result["worst_embedded_image"]),
                (aggregate["worst_page"], aggregate["worst_evidence_source"], aggregate["worst_image_index"]))
    else:
        c.equal(group, "no reliable page => aggregate insufficient", (aggregate["score"], aggregate["confidence"], aggregate["worst_page"]),
                (0.0, "insufficient", None))
        c.equal(group, "no reliable page => no flagged pages", aggregate["flagged_pages"], [])
        # main.py still reports a score (0.98) when the Info dict names a
        # generator, even with no reliable raster evidence.
        meta_hit = explicit_meta.get("confidence") == "high" or legacy_meta.get("confidence") == "high"
        c.equal(group, "no reliable page => score_applicable only via a metadata hit",
                result["score_applicable"], meta_hit)

    # --- flagged embedded images (main.py mirror of the 0.40 rule) ----------
    expected_emb = []
    for p in pages:
        for img in p["embedded_images"]:
            if img.get("usable") and img.get("confidence") in RELIABLE and float(img["ai_generated"]) >= FLAG_THRESHOLD:
                expected_emb.append((p["page"], img["image_index"]))
    c.equal(group, f"flagged_embedded_images == usable reliable rasters >= {FLAG_THRESHOLD}",
            sorted((f["page"], f["image_index"]) for f in result["flagged_embedded_images"]), sorted(expected_emb))

    # --- metadata + classification -------------------------------------------
    c.equal(group, "pdf_generator_metadata == analyze_pdf_metadata(fields)",
            explicit_meta, document.analyze_pdf_metadata(explicit_meta["pdf_fields"]))
    explicit_hit = explicit_meta["confidence"] == "high"
    legacy_hit = legacy_meta.get("confidence") == "high"
    if explicit_hit:
        # Image-path rule: the signature raises the score to >= 0.98 and never
        # lowers stronger visual evidence.
        c.equal(group, "explicit generator hit => ai_generated max(aggregate, 0.98) / confidence high",
                (result["ai_generated"], result["confidence"]),
                (round(max(float(aggregate["score"]), 0.98), 4), "high"))
    elif legacy_hit:
        c.close(group, "legacy metadata hit => max(aggregate, legacy)", result["ai_generated"],
                max(float(aggregate["score"]), float(legacy_meta["score"])))
    else:
        c.equal(group, "no metadata hit => ai_generated == aggregate score", (result["ai_generated"], result["confidence"]),
                (aggregate["score"], aggregate["confidence"]))
    c.equal(group, "classification == classify_pdf(score, confidence, hit)",
            (result["classification"], result["description"]),
            document.classify_pdf(result["ai_generated"], result["confidence"], metadata_hit=explicit_hit or legacy_hit))
    c.check(group, "ai_generated within [0,1] and 4dp", 0.0 <= result["ai_generated"] <= 1.0 and is_rounded4(result["ai_generated"]))
    c.equal(group, "page_count == len(pages)", result["page_count"], len(pages))
    return explicit_hit or legacy_hit


def check_profiler(c: Checks, group: str, result: dict) -> bool:
    """Environment diagnostic: does the page profiler see image objects?

    If a page holds an extracted raster covering >= 65% of the page but the
    profile reports image_area_fraction == 0, the profiler is blind (see
    pdf_utils._inspect_page); whole-page forensics then never run and every
    image page is scored through the embedded-image path instead. This is
    reported as WARN, not FAIL, because it is outside document.py.
    """
    blind = False
    for page in result["pages"]:
        big = [i for i in page["embedded_images"] if i["page_coverage"] >= 0.65]
        if big and page["page_profile"]["image_area_fraction"] == 0.0:
            blind = True
    if blind:
        ver = _pypdfium2_version()
        c.warn(group, "page profiler reports image_area_fraction 0 for full-page rasters",
               f"whole-page (score_pdf_page) path not exercised; image pages are scored via embedded extraction only. "
               f"Likely pdf_utils._inspect_page -> get_objects(textpage=...) raising on pypdfium2 {ver} (requirements pin 5.13.0)")
    return not blind


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step_unit_tests(c: Checks) -> None:
    group = "unit-tests"
    from app.analyzers import verify_document
    suite = unittest.defaultTestLoader.loadTestsFromModule(verify_document)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    c.check(group, f"verify_document.py test methods discovered ({result.testsRun})", result.testsRun >= 133)
    c.check(group, f"all {result.testsRun} tests pass", result.wasSuccessful(),
            "" if result.wasSuccessful() else f"failures={len(result.failures)} errors={len(result.errors)}\n{stream.getvalue()[-2000:]}")


def step_images(c: Checks, assets: dict[str, bytes], report: dict) -> None:
    group = "baseline-images"
    for key in ("human", "ai_generated", "ai_retouched"):
        res = app_main._run_image_analysis(assets[key])
        report["images"][key] = res
        score = res["ai_generated"]
        c.check(group, f"{key}: ai_generated within [0,1] and 4dp", 0.0 <= score <= 1.0 and is_rounded4(score))
        c.check(group, f"{key}: all five signals present with score/confidence/note",
                all(k in res["signals"] and {"score", "confidence", "note"} <= set(res["signals"][k]) for k in (*SIGNAL_NAMES, "metadata")))
        # The same signals scored through the PDF embedded-image rules.
        im = Image.open(io.BytesIO(assets[key]))
        profile = {"image_index": 1, "width_px": im.width, "height_px": im.height,
                   "pixel_count": im.width * im.height, "page_coverage": 1.0}
        emb = document.score_embedded_image(res["signals"], profile, res["scoring"]["context"]["buckets"])
        report["images"][key]["as_embedded"] = emb
        validate_scoring_internals(c, group, f"{key} as embedded raster", emb)
        # Real-data parity: a picture inside a PDF gets the same visual score
        # as the same picture uploaded directly (metadata aside, which PDF
        # extraction does not preserve).
        if emb["usable"]:
            c.close(group, f"{key}: embedded score == direct-upload visual score",
                    emb["score"], res["scoring"]["visual_score"], tol=1e-4)
        if key == "human":
            # A real photo must never be flagged.
            c.check(group, f"{key}: real photo is never flagged as embedded evidence",
                    not (emb["usable"] and emb["confidence"] in RELIABLE and emb["score"] >= FLAG_THRESHOLD),
                    f"usable={emb['usable']} confidence={emb['confidence']} score={emb['score']}")
            c.check(group, f"{key}: real photo is usable evidence that reads as real (image-path weights)",
                    emb["usable"] and emb["confidence"] in RELIABLE and emb["score"] < 0.35,
                    f"usable={emb['usable']} confidence={emb['confidence']} score={emb['score']}")
        else:
            c.check(group, f"{key}: full-size baseline raster is usable embedded evidence", emb["usable"] and emb["confidence"] in RELIABLE)
            c.check(group, f"{key}: AI baseline raster is flagged as embedded evidence (>= {FLAG_THRESHOLD})",
                    emb["score"] >= FLAG_THRESHOLD, f"score={emb['score']}")
        sig_summary = ", ".join(f"{s}={res['signals'][s]['score']:.3f}/{res['signals'][s]['confidence']}" for s in SIGNAL_NAMES)
        c.info(group, f"{key}: ai_generated={score:.4f} embedded={emb['score']:.4f} ({emb['confidence']})", sig_summary)

    from app import pdf_utils
    for key in ("human", "ai_generated", "ai_retouched"):
        stats = pdf_utils.raster_stats(Image.open(io.BytesIO(assets[key])))
        c.check(group, f"{key}: baseline raster is photo-like, not a flat fill",
                not document._is_flat_fill(stats), f"{stats}")
    for name, raster in derive_rasters(assets).items():
        if name in ("gradient_banner", "radial_vignette"):
            stats = pdf_utils.raster_stats(Image.open(io.BytesIO(raster[0])))
            res = app_main._run_image_analysis(raster[0])
            emb = document.score_embedded_image(res["signals"], {"width_px": raster[1][0], "height_px": raster[1][1], **stats})
            c.check(group, f"{name}: synthetic fill is rejected by the flat-fill gate with real signals",
                    document._is_flat_fill(stats) and not emb["usable"] and "flat or gradient fill" in emb["note"],
                    f"{stats} -> usable={emb['usable']} (would have scored {document.score_embedded_image(res['signals'], {'width_px': raster[1][0], 'height_px': raster[1][1]})['score']} ungated)")
        elif name != "ai_generated_icon48":
            stats = pdf_utils.raster_stats(Image.open(io.BytesIO(raster[0])))
            c.check(group, f"{name}: derived photo raster is not gated", not document._is_flat_fill(stats), f"{stats}")

    h, a, r = (report["images"][k]["ai_generated"] for k in ("human", "ai_generated", "ai_retouched"))
    if not (h < a):
        c.warn(group, "README expectation: ai-generated should score above human.jpeg",
               f"human={h} ai_generated={a} ai_retouched={r} (heuristic limitation, see README 'Limitations'; not a document.py issue)")
    else:
        c.info(group, "README ordering human < ai_generated holds", f"human={h} ai_generated={a} ai_retouched={r}")


def step_baseline_pdfs(c: Checks, assets: dict[str, bytes], report: dict) -> None:
    for key in ("pdf_baseline", "pdf_anomaly"):
        group = f"baseline-pdf:{key}"
        res = app_main._run_pdf_analysis(assets[key])
        report["pdfs"][key] = res
        validate_pdf_result(c, group, res)
        c.check(group, "every page is native text (no raster evidence)",
                all(p["content_type"] == "native_document" for p in res["pages"]))
        c.equal(group, "document classified insufficient_visual_evidence", res["classification"], "insufficient_visual_evidence")
        c.equal(group, "no flagged pages", res["flagged_pages"], [])
        c.equal(group, "no explicit generator reference in Info dictionary",
                res["signals"]["pdf_generator_metadata"]["generator_references"], [])
        fields = res["signals"]["pdf_generator_metadata"]["pdf_fields"]
        c.info(group, f"pages={res['page_count']} ai_generated={res['ai_generated']} anomalies={res['document_anomalies']['classification']}",
               f"Producer={fields.get('Producer')!r} Creator={fields.get('Creator')!r}")


def step_synthesized_pdfs(c: Checks, assets: dict[str, bytes], report: dict, save_dir: str | None) -> None:
    rasters = derive_rasters(assets)
    profiler_ok: bool | None = None
    for scenario in build_scenarios(rasters):
        name = scenario["name"]
        group = f"synth:{name}"
        pdf_bytes = build_pdf(scenario["pages"], scenario["info"])
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f"{name}.pdf"), "wb") as fh:
                fh.write(pdf_bytes)
        t0 = time.perf_counter()
        res = app_main._run_pdf_analysis(pdf_bytes)
        elapsed = time.perf_counter() - t0
        report["pdfs"][name] = res
        c.info(group, scenario["description"], f"{len(pdf_bytes)} bytes, {len(res['pages'])} pages, analyzed in {elapsed:.2f}s")

        hit = validate_pdf_result(c, group, res)
        if profiler_ok is None:
            profiler_ok = check_profiler(c, group, res)

        exp = scenario["expect"]
        agg = res["signals"]["document_aggregate"]
        per_page = {p["page"]: p for p in res["pages"]}

        c.equal(group, "expected metadata hit state", hit, exp["metadata_hit"])
        if "generator_references" in exp:
            c.equal(group, "generator_references", res["signals"]["pdf_generator_metadata"]["generator_references"], exp["generator_references"])
        if exp["metadata_hit"]:
            c.equal(group, "metadata hit => likely_ai_generated regardless of raster evidence",
                    (res["classification"], res["ai_generated"]), ("likely_ai_generated", 0.98))
            c.check(group, "aggregate itself stays low (override, not blend)", agg["score"] < FLAG_THRESHOLD,
                    f"aggregate={agg['score']}")

        if "worst_page" in exp:
            c.equal(group, f"worst page is page {exp['worst_page']}", agg["worst_page"], exp["worst_page"],
                    "scores: " + ", ".join(f"p{p['page']}={p['ai_generated']}/{p['confidence']}" for p in res["pages"]))
            others = [float(p["pdf_scoring"]["score"]) for p in res["pages"]
                      if p["page"] != exp["worst_page"] and p["pdf_scoring"].get("confidence") in RELIABLE]
            if others:
                margin = float(agg["score"]) - max(others)
                c.check(group, "worst page is *clearly* worse (margin >= 0.10 over next reliable page)", margin >= 0.10,
                        f"worst={agg['score']} next={max(others)} margin={margin:.4f}")
                c.check(group, "clean pages do not dilute: document score > mean of pages",
                        float(agg["score"]) > float(agg["mean_page_score"]), f"score={agg['score']} mean={agg['mean_page_score']}")
        if "worst_source" in exp:
            c.equal(group, f"worst evidence source is {exp['worst_source']}", agg.get("worst_evidence_source"), exp["worst_source"])
        if "flagged_exactly" in exp:
            c.equal(group, f"flagged pages exactly {exp['flagged_exactly']} (>= {FLAG_THRESHOLD})",
                    [f["page"] for f in agg["flagged_pages"]], exp["flagged_exactly"],
                    "scores: " + ", ".join(f"p{p['page']}={p['ai_generated']}" for p in res["pages"]))
            if exp["flagged_exactly"]:
                c.check(group, "worst page crosses the 0.40 flag threshold", float(agg["score"]) >= FLAG_THRESHOLD, f"score={agg['score']}")
            else:
                c.check(group, "no page reaches the 0.40 flag threshold", float(agg["score"]) < FLAG_THRESHOLD, f"score={agg['score']}")
            c.check(group, "every unflagged reliable page is below 0.40",
                    all(float(p["pdf_scoring"]["score"]) < FLAG_THRESHOLD for p in res["pages"]
                        if p["page"] not in exp["flagged_exactly"] and p["pdf_scoring"].get("confidence") in RELIABLE))
        if "same_embedded_score_pages" in exp:
            a, b = exp["same_embedded_score_pages"]
            ea = per_page[a]["embedded_images"][0]
            eb = per_page[b]["embedded_images"][0]
            c.check(group, "coverage differs materially between the two placements",
                    ea["page_coverage"] >= 0.9 and eb["page_coverage"] <= 0.05,
                    f"p{a} coverage={ea['page_coverage']} p{b} coverage={eb['page_coverage']}")
            c.equal(group, "identical raster scores identically at 100% and ~1% page coverage",
                    (ea["ai_generated"], ea["confidence"]), (eb["ai_generated"], eb["confidence"]))
            # Coverage must not dilute: each page carries at least the shared
            # raster's score. (The full-page copy may carry slightly more: its
            # faithful page render is a second evidence unit, and the worst
            # reliable unit wins.)
            for n in (a, b):
                c.check(group, f"p{n}: page risk is not diluted below the raster's own score",
                        per_page[n]["ai_generated"] >= per_page[n]["embedded_images"][0]["ai_generated"] - 1e-9,
                        f"page={per_page[n]['ai_generated']} raster={per_page[n]['embedded_images'][0]['ai_generated']}")
            c.equal(group, "the small placement's risk is exactly the raster's score",
                    per_page[b]["ai_generated"], eb["ai_generated"])
        if exp.get("insufficient"):
            c.equal(group, "document is insufficient_visual_evidence",
                    (res["classification"], res["score_applicable"], agg["worst_page"]),
                    ("insufficient_visual_evidence", False, None))
        if "flat_fill_pages" in exp:
            for n in exp["flat_fill_pages"]:
                imgs = per_page[n]["embedded_images"]
                c.check(group, f"p{n}: decorative fill extracted and rejected as flat/gradient fill",
                        len(imgs) == 1 and not imgs[0]["usable"] and "flat or gradient fill" in imgs[0]["scoring"]["note"],
                        f"{[(i['usable'], i['scoring']['note'][:60]) for i in imgs]}")
                c.equal(group, f"p{n}: page risk is insufficient", per_page[n]["pdf_scoring"]["confidence"], "insufficient")
        if exp.get("embedded_too_small"):
            imgs = [i for p in res["pages"] for i in p["embedded_images"]]
            c.equal(group, "exactly one embedded raster extracted", len(imgs), 1)
            if imgs:
                c.check(group, "48x48 icon rejected as too small",
                        (not imgs[0]["usable"]) and "too small" in imgs[0]["scoring"]["note"],
                        f"{imgs[0]['width_px']}x{imgs[0]['height_px']} -> {imgs[0]['scoring']['note']}")

        # The pixels the pipeline analyzed must be the baseline pixels: every
        # placed raster is extracted at its native size (no resampling).
        for n, pg in enumerate(scenario["pages"], start=1):
            if pg.get("expect_sizes"):
                got = [(i["width_px"], i["height_px"]) for i in per_page[n]["embedded_images"]]
                c.equal(group, f"p{n}: extracted raster size == placed source size", got, pg["expect_sizes"])

        # Page-type expectations that depend on the profiler.
        image_pages = [i for i, pg in enumerate(scenario["pages"], start=1)
                       if pg.get("images") and not pg.get("text")]
        for n in image_pages:
            ct = per_page[n]["content_type"]
            if profiler_ok:
                c.equal(group, f"p{n}: full-page raster classified scanned_or_image_document", ct, "scanned_or_image_document")
            else:
                c.info(group, f"p{n}: full-page raster classified {ct!r} (profiler blind; scored via embedded path)")
        for n, pg in enumerate(scenario["pages"], start=1):
            if pg.get("text"):
                c.equal(group, f"p{n}: text page classified native_document", per_page[n]["content_type"], "native_document")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_assets(baseline_dir: str, c: Checks) -> dict[str, bytes]:
    assets = {}
    for key, rel in BASELINE_FILES.items():
        path = os.path.join(baseline_dir, rel)
        if not os.path.isfile(path):
            c.check("environment", f"baseline file present: {rel}", False, path)
            continue
        with open(path, "rb") as fh:
            assets[key] = fh.read()
        fmt = app_main._detect_format(assets[key])
        c.check("environment", f"baseline file present & sniffed: {rel}", fmt != "unknown", f"{len(assets[key])} bytes, format={fmt}")
    return assets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--baseline", default=os.path.join(_MAIN_DIR, os.pardir, "baseline"), help="baseline directory")
    parser.add_argument("--json", metavar="PATH", help="write full pipeline output + check table as JSON")
    parser.add_argument("--save-pdfs", metavar="DIR", help="write the synthesized PDFs here for inspection")
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true", help="print passing checks too")
    args = parser.parse_args(argv)

    c = Checks(verbose=args.verbose)
    report: dict[str, Any] = {"images": {}, "pdfs": {}}

    try:
        import cv2
        c.info("environment", f"python={sys.version.split()[0]} numpy={np.__version__} opencv={cv2.__version__} "
                              f"pillow={Image.__version__} pypdfium2={_pypdfium2_version()}")
    except Exception as exc:  # pragma: no cover
        c.warn("environment", "could not report library versions", str(exc))

    print("=" * 78)
    print("Step 1  unit tests (app/analyzers/verify_document.py)")
    print("=" * 78)
    if args.skip_unit_tests:
        c.info("unit-tests", "skipped by flag")
    else:
        step_unit_tests(c)

    print("=" * 78)
    print("Step 2  baseline images -> _run_image_analysis")
    print("=" * 78)
    assets = load_assets(os.path.abspath(args.baseline), c)
    if len(assets) != len(BASELINE_FILES):
        print("missing baseline files; aborting")
        return 2
    step_images(c, assets, report)

    print("=" * 78)
    print("Step 3  baseline PDFs (untouched) -> _run_pdf_analysis")
    print("=" * 78)
    step_baseline_pdfs(c, assets, report)

    print("=" * 78)
    print("Step 4/5  PDFs synthesized from baseline images -> _run_pdf_analysis -> document.py cross-check")
    print("=" * 78)
    step_synthesized_pdfs(c, assets, report, args.save_pdfs)

    counts = c.counts()
    print("=" * 78)
    print(f"SUMMARY  pass={counts['PASS']}  fail={counts['FAIL']}  warn={counts['WARN']}  info={counts['INFO']}")
    print("=" * 78)
    if counts["FAIL"]:
        print("FAILED checks:")
        for status, group, name, detail in c.rows:
            if status == "FAIL":
                print(f"  - {group} :: {name}" + (f"  --  {detail}" if detail else ""))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"summary": counts, "checks": [dict(zip(("status", "group", "name", "detail"), r)) for r in c.rows],
                       "pipeline": report}, fh, indent=2, default=str)
        print(f"full report written to {args.json}")

    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
