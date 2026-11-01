"""PDF handling, structural inspection and embedded-image extraction.

Native text/vector PDF pages are not camera photographs. Whole-page photo
forensics therefore run only on image-dominant pages. Independently, every
embedded raster image with enough pixels is extracted and can be analyzed on
its own, even if it occupies a tiny fraction of a native PDF page.
"""
from __future__ import annotations

import io

import pypdfium2 as pdfium

MAX_PAGES = 20
MAX_EMBEDDED_IMAGES_PER_PAGE = 64
MAX_EXTRACT_DIM = 4096


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
        for obj in page.get_objects(max_depth=8, textpage=textpage):
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

            # render=True applies image transforms/masks while scale_to_original
            # keeps the raster close to its native pixel resolution.
            bitmap = obj.get_bitmap(render=True, scale_to_original=True)
            pil_image = bitmap.to_pil().convert("RGB")

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


def render_pdf_pages_with_profiles(pdf_bytes: bytes, dpi: int = 200) -> list[dict]:
    """Return page PNGs, structural profiles and embedded raster images."""
    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        n = min(len(doc), MAX_PAGES)
        scale = dpi / 72.0
        pages: list[dict] = []
        for i in range(n):
            page = doc[i]
            try:
                profile = _inspect_page(page)
                embedded_images = _extract_embedded_images(page, i + 1)
                bitmap = page.render(scale=scale)
                try:
                    pil_image = bitmap.to_pil().convert("RGB")
                    buf = io.BytesIO()
                    pil_image.save(buf, format="PNG")
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
