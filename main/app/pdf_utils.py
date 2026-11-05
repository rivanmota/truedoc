"""PDF handling, structural inspection and embedded-image extraction.

Native text/vector PDF pages are not camera photographs. Whole-page photo
forensics therefore run only on image-dominant pages. Independently, every
embedded raster image with enough pixels is extracted and can be analyzed on
its own, even if it occupies a tiny fraction of a native PDF page.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image
import pypdfium2 as pdfium

from app.context import estimate_jpeg_quality

MAX_PAGES = 20
MAX_EMBEDDED_IMAGES_PER_PAGE = 64
MAX_EXTRACT_DIM = 4096

# Photo-forensic page renders are drawn at the dominant raster's own pixel
# density, so the render reproduces the file's pixels instead of our
# renderer's resampling (which the frequency, noise and gradients signals
# would otherwise measure). Bounds keep pathological PDFs cheap.
MIN_RENDER_DPI = 36.0
MAX_RENDER_DPI = 600.0
MAX_RENDER_DIM = 6000          # px, long side of a rendered page
# A render within this relative tolerance of the raster's density counts as
# 1:1; beyond it (or with non-uniform scaling) the render is "resampled".
RENDER_SCALE_TOLERANCE = 0.02
# Even a nominally 1:1 render is only used as pixel evidence if it is
# detail-faithful: its fine-scale energy (mean |pixel difference| along each
# axis) over the dominant raster's area is within this tolerance of the
# raster's own. Image smoothing or sub-pixel misplacement blurs the top band,
# which the frequency signal would read as an upscaling knee.
RENDER_DETAIL_TOLERANCE = 0.03
RENDER_SIZE_TOLERANCE_PX = 2


RASTER_STATS_MAX_PIXELS = 1_000_000


def raster_stats(pil_image, max_pixels: int = RASTER_STATS_MAX_PIXELS) -> dict:
    """Cheap content statistics used by the scoring layer to recognise flat
    or gradient fills (banners, vignettes, solid blocks) that are not
    photographs: ``unique_colors`` (distinct RGB triples) and
    ``edge_density`` (fraction of pixels whose neighbour differs by more
    than 8 gray levels). Large rasters are strided down to ~``max_pixels``.
    """
    arr = np.asarray(pil_image.convert("RGB"))
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return {"unique_colors": 0, "edge_density": 0.0}
    step = max(1, int(np.ceil(np.sqrt((h * w) / max_pixels))))
    sub = arr[::step, ::step]
    packed = (
        (sub[..., 0].astype(np.uint32) << 16)
        | (sub[..., 1].astype(np.uint32) << 8)
        | sub[..., 2].astype(np.uint32)
    )
    unique_colors = int(np.unique(packed).size)
    gray = sub.astype(np.float32).mean(axis=2)
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return {"unique_colors": unique_colors, "edge_density": 0.0}
    dx = np.abs(np.diff(gray, axis=1))[:-1, :]
    dy = np.abs(np.diff(gray, axis=0))[:, :-1]
    edge_density = float((np.maximum(dx, dy) > 8.0).mean())
    return {"unique_colors": unique_colors, "edge_density": round(edge_density, 4)}


def _safe_area(bounds, page_width: float, page_height: float) -> float:
    try:
        left, bottom, right, top = bounds
        width = max(0.0, min(float(right), page_width) - max(float(left), 0.0))
        height = max(0.0, min(float(top), page_height) - max(float(bottom), 0.0))
        return width * height
    except Exception:
        return 0.0


def _inspect_page(page) -> dict:
    width, height = page.get_size()
    page_area = max(float(width) * float(height), 1.0)

    text_chars = 0
    textpage = None
    try:
        textpage = page.get_textpage()
        text_chars = int(textpage.count_chars())
    except Exception:
        text_chars = 0

    counts = {
        "text_objects": 0,
        "image_objects": 0,
        "path_objects": 0,
        "form_objects": 0,
        "shading_objects": 0,
    }
    image_area = 0.0

    raw = pdfium.raw
    type_to_key = {
        raw.FPDF_PAGEOBJ_TEXT: "text_objects",
        raw.FPDF_PAGEOBJ_IMAGE: "image_objects",
        raw.FPDF_PAGEOBJ_PATH: "path_objects",
        raw.FPDF_PAGEOBJ_FORM: "form_objects",
        raw.FPDF_PAGEOBJ_SHADING: "shading_objects",
    }

    try:
        for obj in page.get_objects(max_depth=8):
            key = type_to_key.get(obj.type)
            if key:
                counts[key] += 1
            if obj.type == raw.FPDF_PAGEOBJ_IMAGE:
                try:
                    image_area += _safe_area(obj.get_bounds(), float(width), float(height))
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        if textpage is not None:
            try:
                textpage.close()
            except Exception:
                pass

    image_area_fraction = min(image_area / page_area, 1.0)
    has_native_text = text_chars >= 40 or counts["text_objects"] >= 3
    vector_heavy = counts["path_objects"] >= 20 and image_area_fraction < 0.25

    if has_native_text and image_area_fraction >= 0.50:
        content_type = "mixed_document"
    elif has_native_text:
        content_type = "native_document"
    elif image_area_fraction >= 0.65:
        content_type = "scanned_or_image_document"
    elif vector_heavy:
        content_type = "vector_document"
    elif image_area_fraction >= 0.25:
        content_type = "mixed_document"
    else:
        content_type = "sparse_document"

    photo_forensics_applicable = (
        content_type == "scanned_or_image_document"
        or (content_type == "mixed_document" and image_area_fraction >= 0.70)
    )

    return {
        "width_points": round(float(width), 2),
        "height_points": round(float(height), 2),
        "text_chars": text_chars,
        **counts,
        "image_area_fraction": round(float(image_area_fraction), 4),
        "content_type": content_type,
        "photo_forensics_applicable": photo_forensics_applicable,
    }


def _extract_embedded_images(page, page_number: int) -> list[dict]:
    """Extract raster image page objects as PNGs plus forensic context.

    Extraction is independent of page coverage. A small image can therefore be
    inspected for fraud even on a page that is otherwise 99% native PDF text.
    Very tiny images are still returned, but the scoring layer may mark them
    insufficient because AI-image heuristics are not reliable at icon sizes.
    """
    width_pt, height_pt = page.get_size()
    page_area = max(float(width_pt) * float(height_pt), 1.0)
    raw = pdfium.raw
    images: list[dict] = []

    try:
        objects = page.get_objects(max_depth=8)
        image_objects = [obj for obj in objects if obj.type == raw.FPDF_PAGEOBJ_IMAGE]
    except Exception:
        image_objects = []

    for image_index, obj in enumerate(image_objects[:MAX_EMBEDDED_IMAGES_PER_PAGE], start=1):
        bitmap = None
        try:
            bounds = obj.get_bounds()
            page_coverage = min(_safe_area(bounds, float(width_pt), float(height_pt)) / page_area, 1.0)

            filters = []
            try:
                filters = list(obj.get_filters())
            except Exception:
                pass

            # A DCTDecode-only stream *is* a JPEG file: read its quality from
            # the quantisation tables, exactly as for an uploaded JPEG, so the
            # raster gets the same context weighting (app.context).
            jpeg_quality = None
            source_bytes = None
            if filters == ["DCTDecode"]:
                try:
                    source_bytes = bytes(obj.get_data(decode_simple=False))
                    jpeg_quality = estimate_jpeg_quality(source_bytes)
                except Exception:
                    source_bytes, jpeg_quality = None, None

            # The image's own decoded pixels (render=False): exactly what a
            # direct upload of the same image would decode to. Drawing it
            # through its placement transform (render=True) resamples any
            # stretched/scaled placement, and pixel forensics would then
            # measure that resampling. Fall back to the drawn version only if
            # native decoding fails, and say so.
            extraction = "native"
            try:
                bitmap = obj.get_bitmap(render=False)
                pil_image = bitmap.to_pil().convert("RGB")
            except Exception:
                if bitmap is not None:
                    try:
                        bitmap.close()
                    except Exception:
                        pass
                bitmap = obj.get_bitmap(render=True, scale_to_original=True)
                pil_image = bitmap.to_pil().convert("RGB")
                extraction = "rendered"

            native_width, native_height = pil_image.size
            if max(native_width, native_height) > MAX_EXTRACT_DIM:
                scale = MAX_EXTRACT_DIM / max(native_width, native_height)
                pil_image = pil_image.resize(
                    (max(1, int(native_width * scale)), max(1, int(native_height * scale)))
                )

            width_px, height_px = pil_image.size
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")

            images.append({
                "page": page_number,
                "png_bytes": buf.getvalue(),
                # The image's own file (a DCTDecode stream is a complete JPEG,
                # EXIF included), for the metadata analyzer; None otherwise.
                "source_bytes": source_bytes,
                "profile": {
                    "image_index": image_index,
                    "width_px": int(width_px),
                    "height_px": int(height_px),
                    "native_width_px": int(native_width),
                    "native_height_px": int(native_height),
                    "pixel_count": int(width_px * height_px),
                    "page_coverage": round(float(page_coverage), 6),
                    "bounds": [round(float(v), 2) for v in bounds],
                    "filters": filters,
                    "jpeg_quality": jpeg_quality,
                    "extraction": extraction,
                    "resampled_on_extraction": (
                        extraction != "native"
                        or (int(width_px), int(height_px)) != (int(native_width), int(native_height))
                    ),
                    **raster_stats(pil_image),
                },
            })
        except Exception:
            # A malformed/unextractable object should not fail the entire PDF.
            continue
        finally:
            if bitmap is not None:
                try:
                    bitmap.close()
                except Exception:
                    pass

    return images


def _raster_dpi(image_profile: dict) -> tuple[float, float] | None:
    """(horizontal, vertical) pixel density of a placed raster, from its
    native pixel size and its bounds on the page (in points)."""
    bounds = image_profile.get("bounds")
    w_px = image_profile.get("native_width_px")
    h_px = image_profile.get("native_height_px")
    if not bounds or not w_px or not h_px:
        return None
    left, bottom, right, top = (float(v) for v in bounds)
    w_pt, h_pt = right - left, top - bottom
    if w_pt <= 0 or h_pt <= 0:
        return None
    return float(w_px) * 72.0 / w_pt, float(h_px) * 72.0 / h_pt


def plan_page_render(profile: dict, embedded_images: list[dict], default_dpi: float = 200.0) -> dict:
    """Render settings for one page.

    Pages that will be scored as photographs are rendered at the dominant
    (largest-coverage) raster's native density. ``render_resampled`` is
    True when that is impossible (no measurable raster, non-uniform
    scaling/rotation, or the density had to be clamped) -- the scoring
    layer then does not run pixel-level signals on the render.
    Non-photo pages keep ``default_dpi``; their render is not scored.
    """
    plan = {
        "render_dpi": float(default_dpi),
        "raster_native_dpi": None,
        "render_scale": None,
        "render_resampled": None,
        "dominant_image_index": None,
    }
    if not profile.get("photo_forensics_applicable"):
        return plan

    candidates = [img["profile"] for img in embedded_images if _raster_dpi(img["profile"])]
    if not candidates:
        plan["render_resampled"] = True
        return plan
    dominant = max(candidates, key=lambda p: float(p.get("page_coverage", 0.0)))
    dpi_x, dpi_y = _raster_dpi(dominant)
    plan["dominant_image_index"] = dominant.get("image_index")
    if abs(dpi_x / dpi_y - 1.0) > RENDER_SCALE_TOLERANCE:
        plan["raster_native_dpi"] = [round(dpi_x, 3), round(dpi_y, 3)]
        plan["render_resampled"] = True
        return plan

    native = (dpi_x + dpi_y) / 2.0
    page_long_pt = max(float(profile.get("width_points", 0.0)), float(profile.get("height_points", 0.0)), 1.0)
    render_dpi = min(max(native, MIN_RENDER_DPI), MAX_RENDER_DPI, MAX_RENDER_DIM * 72.0 / page_long_pt)
    scale = render_dpi / native
    plan.update({
        "render_dpi": round(render_dpi, 4),
        "raster_native_dpi": round(native, 4),
        "render_scale": round(scale, 4),
        "render_resampled": abs(scale - 1.0) > RENDER_SCALE_TOLERANCE,
    })
    return plan


def render_fidelity(render_rgb: Image.Image, raster_png: bytes, bounds, page_height_pt: float, dpi: float) -> dict:
    """Does the page render reproduce the raster's detail where it is placed?

    Crops the render to the raster's bounds and compares fine-scale energy
    (mean absolute difference of neighbouring pixels, per axis) with the
    extracted raster's own. Returns the measurements and ``faithful``.
    """
    scale = dpi / 72.0
    left, bottom, right, top = (float(v) for v in bounds)
    x0, x1 = int(round(left * scale)), int(round(right * scale))
    y0, y1 = int(round((page_height_pt - top) * scale)), int(round((page_height_pt - bottom) * scale))
    render = np.asarray(render_rgb.convert("L"), dtype=np.float32)[max(y0, 0):y1, max(x0, 0):x1]
    raster = np.asarray(Image.open(io.BytesIO(raster_png)).convert("L"), dtype=np.float32)

    size_ok = (
        render.size > 0
        and abs(render.shape[0] - raster.shape[0]) <= RENDER_SIZE_TOLERANCE_PX
        and abs(render.shape[1] - raster.shape[1]) <= RENDER_SIZE_TOLERANCE_PX
    )
    if not size_ok:
        return {"faithful": False, "size_match": False,
                "render_region_px": list(render.shape[:2]), "raster_px": list(raster.shape[:2])}

    def detail(a: np.ndarray, axis: int) -> float:
        return float(np.abs(np.diff(a, axis=axis)).mean())

    ratios = []
    for axis in (0, 1):
        base = detail(raster, axis)
        ratios.append(detail(render, axis) / base if base > 1e-6 else 1.0)
    faithful = all(abs(r - 1.0) <= RENDER_DETAIL_TOLERANCE for r in ratios)
    return {
        "faithful": bool(faithful),
        "size_match": True,
        "detail_ratio_vertical": round(ratios[0], 4),
        "detail_ratio_horizontal": round(ratios[1], 4),
    }


def render_pdf_pages_with_profiles(pdf_bytes: bytes, dpi: int = 200) -> list[dict]:
    """Return page PNGs, structural profiles and embedded raster images.

    ``dpi`` applies to pages that are not scored as photographs; photo pages
    are rendered at their dominant raster's native density (see
    ``plan_page_render``).
    """
    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        n = min(len(doc), MAX_PAGES)
        pages: list[dict] = []
        for i in range(n):
            page = doc[i]
            try:
                profile = _inspect_page(page)
                embedded_images = _extract_embedded_images(page, i + 1)
                profile.update(plan_page_render(profile, embedded_images, dpi))
                photo_page = bool(profile.get("photo_forensics_applicable"))
                # Photo pages are drawn without image smoothing so a 1:1 render
                # copies the raster's pixels instead of interpolating them.
                render_kwargs = {"no_smoothimage": True} if photo_page else {}
                bitmap = page.render(scale=profile["render_dpi"] / 72.0, **render_kwargs)
                try:
                    pil_image = bitmap.to_pil().convert("RGB")
                    buf = io.BytesIO()
                    pil_image.save(buf, format="PNG")
                    profile.update(raster_stats(pil_image))
                    if photo_page and profile.get("render_resampled") is False:
                        dominant = next(
                            img for img in embedded_images
                            if img["profile"].get("image_index") == profile["dominant_image_index"]
                        )
                        if dominant["profile"].get("resampled_on_extraction"):
                            fidelity = {"faithful": False, "note": "raster was resampled on extraction"}
                        else:
                            fidelity = render_fidelity(
                                pil_image, dominant["png_bytes"], dominant["profile"]["bounds"],
                                float(profile.get("height_points", 0.0)), profile["render_dpi"],
                            )
                        profile["render_fidelity"] = fidelity
                        profile["render_resampled"] = not fidelity["faithful"]
                finally:
                    try:
                        bitmap.close()
                    except Exception:
                        pass

                pages.append({
                    "page": i + 1,
                    "png_bytes": buf.getvalue(),
                    "profile": profile,
                    "embedded_images": embedded_images,
                })
            finally:
                page.close()
        return pages
    finally:
        doc.close()


def render_pdf_pages(pdf_bytes: bytes, dpi: int = 200) -> list[bytes]:
    """Backward-compatible wrapper returning PNG bytes only."""
    return [p["png_bytes"] for p in render_pdf_pages_with_profiles(pdf_bytes, dpi=dpi)]


def inspect_pdf_pages(pdf_bytes: bytes) -> list[dict]:
    """Return structural profiles without exposing rendered bytes."""
    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        n = min(len(doc), MAX_PAGES)
        profiles = []
        for i in range(n):
            page = doc[i]
            try:
                profile = _inspect_page(page)
                embedded = _extract_embedded_images(page, i + 1)
                profiles.append({
                    "page": i + 1,
                    **profile,
                    "embedded_images": [x["profile"] for x in embedded],
                })
            finally:
                page.close()
        return profiles
    finally:
        doc.close()


def read_pdf_metadata(pdf_bytes: bytes) -> dict[str, str]:
    """Document-level Info dict fields (Title/Author/Creator/etc.), if any."""
    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        fields = ["Title", "Author", "Subject", "Keywords", "Creator", "Producer"]
        values = {}
        for key in fields:
            try:
                v = doc.get_metadata_value(key)
            except Exception:
                v = None
            if v:
                values[key] = v
        return values
    finally:
        doc.close()
