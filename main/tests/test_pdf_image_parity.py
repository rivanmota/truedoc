"""
PDFs are scored with the image path's weights.

Contract: a raster embedded in a PDF gets the same visual score as the same
picture uploaded directly (the combiner, weights, confidence factors,
no-portrait redistribution, single-signal cap and context factors are all
shared via ``app.scoring``). The PDF path adds only its own gates (page type,
flat fills, icon size) and treats "no evidence" as insufficient instead of
0.5.

These tests run the real pipeline (``main._run_pdf_analysis`` /
``main._run_image_analysis``) on PDFs built in memory with Pillow, which
stores RGB images as a plain JPEG stream (DCTDecode), the common case for
photos in PDFs.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image

from app import context, main, pdf_utils

BASELINE = Path(__file__).resolve().parents[2] / "baseline"


def pdf_with(image: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "PDF", quality=quality)
    return buf.getvalue()


def embedded_jpeg_stream(pdf_bytes: bytes) -> bytes:
    """The raw JPEG stream the PDF stores for its first image."""
    pdf = pdfium.PdfDocument(pdf_bytes)
    obj = next(o for o in pdf[0].get_objects(max_depth=8) if o.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE)
    return bytes(obj.get_data(decode_simple=False))


def synthetic_photo(seed: int = 0, size: int = 512) -> Image.Image:
    """Textured, photo-like test image: a colour 1/f (natural-image)
    spectrum, so it has edges and thousands of colours and is not mistaken
    for a decorative flat/gradient fill by the PDF gate."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(size)
    radius = np.hypot(*np.meshgrid(f, f))
    radius[0, 0] = 1.0
    channels = []
    for _ in range(3):
        spectrum = (rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))) / radius
        field = np.real(np.fft.ifft2(spectrum))
        channels.append((field - field.mean()) / field.std() * 40 + 128)
    rgb = np.stack(channels, axis=-1) + rng.normal(0, 1.5, (size, size, 3))
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# Context of an embedded raster
# --------------------------------------------------------------------------- #
class TestEmbeddedRasterBuckets:
    BUCKETS = {"compression": "jpeg_history", "size": "medium", "scale": "native", "color": "color"}

    @pytest.mark.parametrize("quality,bucket", [(95, "jpeg_high"), (80, "jpeg_mid"), (60, "jpeg_low")])
    def test_jpeg_stream_quality_sets_the_compression_bucket(self, quality, bucket):
        out = context.embedded_raster_buckets(self.BUCKETS, {"jpeg_quality": quality})
        assert out == {**self.BUCKETS, "compression": bucket}

    def test_unknown_quality_keeps_the_pixel_derived_buckets(self):
        assert context.embedded_raster_buckets(self.BUCKETS, {"jpeg_quality": None}) == self.BUCKETS
        assert context.embedded_raster_buckets(self.BUCKETS, {}) == self.BUCKETS

    def test_resampled_raster_keeps_the_pixel_derived_buckets(self):
        # Resampling on extraction erased the JPEG structure the factor describes.
        profile = {"jpeg_quality": 95, "resampled_on_extraction": True}
        assert context.embedded_raster_buckets(self.BUCKETS, profile) == self.BUCKETS

    def test_input_is_not_mutated(self):
        buckets = dict(self.BUCKETS)
        context.embedded_raster_buckets(buckets, {"jpeg_quality": 95})
        assert buckets == self.BUCKETS


class TestExtractionRecordsJpegQuality:
    @pytest.mark.parametrize("quality", [95, 75, 60])
    def test_dct_stream_quality_is_recorded(self, quality):
        pages = pdf_utils.render_pdf_pages_with_profiles(pdf_with(synthetic_photo(), quality))
        profile = pages[0]["embedded_images"][0]["profile"]
        assert profile["filters"] == ["DCTDecode"]
        assert profile["jpeg_quality"] == quality
        assert profile["resampled_on_extraction"] is False


# --------------------------------------------------------------------------- #
# End-to-end parity
# --------------------------------------------------------------------------- #
def _parity(pdf_bytes: bytes) -> tuple[dict, dict]:
    """(direct upload of the PDF's JPEG stream, the PDF's embedded raster)."""
    upload = main._run_image_analysis(embedded_jpeg_stream(pdf_bytes))
    result = main._run_pdf_analysis(pdf_bytes)
    return upload, result["pages"][0]["embedded_images"][0]


class TestSameJpegSameScore:
    @pytest.mark.parametrize("quality", [95, 80, 65])
    def test_synthetic_photo(self, quality):
        upload, embedded = _parity(pdf_with(synthetic_photo(seed=quality), quality))
        assert embedded["usable"] is True, embedded["scoring"]["note"]
        assert embedded["scoring"]["combiner"]["context_buckets"]["compression"] == \
            upload["scoring"]["context"]["buckets"]["compression"]
        assert embedded["ai_generated"] == pytest.approx(upload["scoring"]["visual_score"], abs=1e-4)

    def test_flat_fill_gate_still_applies(self):
        # PDF-only gate kept on top of the shared weights: a smooth gradient
        # banner is not a photograph, whatever its upload would score.
        yy, xx = np.mgrid[0:300, 0:900]
        banner = np.stack([60 + 150 * xx / 900] * 3, axis=-1).astype(np.uint8)
        result = main._run_pdf_analysis(pdf_with(Image.fromarray(banner), 80))
        img = result["pages"][0]["embedded_images"][0]
        assert img["usable"] is False
        assert "flat or gradient fill" in img["scoring"]["note"]

    @pytest.mark.parametrize("rel,band", [
        ("png/fireplace-generated.png", "flagged"),
        ("png/ai-generated.png", "flagged"),
        ("jpg/real-photo.jpeg", "real"),
    ])
    def test_baseline_pictures(self, rel, band):
        path = BASELINE / rel
        if not path.exists():
            pytest.skip(f"{rel} not present")
        upload, embedded = _parity(pdf_with(Image.open(path), 95))
        assert embedded["usable"] is True
        assert embedded["ai_generated"] == pytest.approx(upload["scoring"]["visual_score"], abs=1e-4)
        if band == "flagged":
            assert embedded["ai_generated"] >= 0.40  # the PDF flag threshold
        else:
            assert embedded["ai_generated"] < 0.35

    def test_faceless_generated_scene_is_now_flagged_in_a_pdf(self):
        # Under the old PDF-only weights (frequency 0.05, no redistribution)
        # this raster was "insufficient evidence"; it must now be flagged as
        # its upload is.
        path = BASELINE / "png/fireplace-generated.png"
        if not path.exists():
            pytest.skip("fireplace-generated.png not present")
        result = main._run_pdf_analysis(pdf_with(Image.open(path), 95))
        assert result["score_applicable"] is True
        assert result["flagged_pages"]
        # Either evidence unit may carry it: the embedded raster and the
        # (faithful) page render score the same.
        assert result["flagged_pages"][0]["risk_source"] in {"embedded_image", "page_render"}
        assert result["classification"] in {"suspicious_raster_content", "likely_ai_generated"}

    def test_genuine_photo_in_a_pdf_is_usable_and_reads_real(self):
        path = BASELINE / "jpg/real-photo.jpeg"
        if not path.exists():
            pytest.skip("real-photo.jpeg not present")
        result = main._run_pdf_analysis(pdf_with(Image.open(path), 95))
        assert result["score_applicable"] is True
        assert result["flagged_pages"] == []
        assert result["classification"] == "low_ai_evidence"


# --------------------------------------------------------------------------- #
# The whole-page render step
# --------------------------------------------------------------------------- #
def stretched_pdf(jpeg: bytes, px: tuple[int, int], placed_pt: tuple[float, float],
                  extra_content: bytes = b"") -> bytes:
    """Minimal one-page PDF drawing a JPEG (DCTDecode) into a box of
    ``placed_pt`` points: a box whose aspect differs from the image's
    stretches it non-uniformly. ``extra_content`` is appended to the page's
    content stream (e.g. vector drawing over the image)."""
    w_pt, h_pt = placed_pt
    content = f"q {w_pt} 0 0 {h_pt} 0 0 cm /Im0 Do Q".encode() + extra_content
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w_pt} {h_pt}] "
         f"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>").encode(),
        (f"<< /Type /XObject /Subtype /Image /Width {px[0]} /Height {px[1]} /ColorSpace /DeviceRGB "
         f"/BitsPerComponent 8 /Filter /DCTDecode /Length {len(jpeg)} >>\nstream\n").encode() + jpeg + b"\nendstream",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def jpeg_bytes(image: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


class TestPageProfiler:
    def test_image_pages_are_seen_by_the_profiler(self):
        # Regression: _inspect_page passed an unsupported ``textpage`` kwarg to
        # get_objects(); the TypeError was swallowed, every page reported 0%
        # image area, and whole-page photo scoring never ran.
        page = pdf_utils.render_pdf_pages_with_profiles(pdf_with(synthetic_photo(), 95))[0]
        profile = page["profile"]
        assert profile["image_objects"] == 1
        assert profile["image_area_fraction"] == pytest.approx(1.0, abs=0.01)
        assert profile["content_type"] == "scanned_or_image_document"
        assert profile["photo_forensics_applicable"] is True


class TestPlanPageRender:
    @staticmethod
    def img(coverage=1.0, native=(720, 360), bounds=(0.0, 0.0, 720.0, 360.0), index=1):
        return {"profile": {"image_index": index, "page_coverage": coverage, "native_width_px": native[0],
                            "native_height_px": native[1], "bounds": list(bounds)}}

    PHOTO = {"photo_forensics_applicable": True, "width_points": 720.0, "height_points": 360.0}

    def test_non_photo_page_keeps_the_default_dpi(self):
        plan = pdf_utils.plan_page_render({"photo_forensics_applicable": False}, [self.img()], 200)
        assert plan["render_dpi"] == 200.0
        assert plan["render_resampled"] is None

    def test_photo_page_renders_at_the_raster_density(self):
        # 720 px across 720 pt = 72 dpi; 1440 px across 720 pt = 144 dpi.
        assert pdf_utils.plan_page_render(self.PHOTO, [self.img()])["render_dpi"] == 72.0
        plan = pdf_utils.plan_page_render(self.PHOTO, [self.img(native=(1440, 720))])
        assert plan["render_dpi"] == 144.0
        assert plan["render_scale"] == 1.0
        assert plan["render_resampled"] is False

    def test_dominant_raster_is_the_largest_coverage(self):
        plan = pdf_utils.plan_page_render(self.PHOTO, [
            self.img(coverage=0.1, native=(100, 50), bounds=(0, 0, 100, 50), index=1),
            self.img(coverage=0.9, native=(1440, 720), index=2),
        ])
        assert plan["dominant_image_index"] == 2
        assert plan["render_dpi"] == 144.0

    def test_no_measurable_raster_fails_closed(self):
        assert pdf_utils.plan_page_render(self.PHOTO, [])["render_resampled"] is True

    def test_non_uniform_scaling_fails_closed(self):
        plan = pdf_utils.plan_page_render(self.PHOTO, [self.img(native=(720, 720))])  # 72 x 144 dpi
        assert plan["render_resampled"] is True

    def test_clamped_density_is_resampled(self):
        huge = self.img(native=(720 * 20, 360 * 20))  # 1440 dpi > MAX_RENDER_DPI
        plan = pdf_utils.plan_page_render(self.PHOTO, [huge])
        assert plan["render_dpi"] == pdf_utils.MAX_RENDER_DPI
        assert plan["render_resampled"] is True
        tiny = self.img(native=(72, 36))  # 7.2 dpi < MIN_RENDER_DPI
        assert pdf_utils.plan_page_render(self.PHOTO, [tiny])["render_resampled"] is True


class TestRenderFidelity:
    def test_identical_pixels_are_faithful(self):
        photo = synthetic_photo(size=256)
        buf = io.BytesIO()
        photo.save(buf, "PNG")
        result = pdf_utils.render_fidelity(photo, buf.getvalue(), (0, 0, 256, 256), 256.0, 72.0)
        assert result["faithful"] is True
        assert result["detail_ratio_horizontal"] == pytest.approx(1.0)

    def test_smoothed_render_is_not_faithful(self):
        from PIL import ImageFilter
        photo = synthetic_photo(size=256)
        buf = io.BytesIO()
        photo.save(buf, "PNG")
        blurred = photo.filter(ImageFilter.GaussianBlur(0.8))
        result = pdf_utils.render_fidelity(blurred, buf.getvalue(), (0, 0, 256, 256), 256.0, 72.0)
        assert result["faithful"] is False
        assert result["detail_ratio_horizontal"] < 1.0 - pdf_utils.RENDER_DETAIL_TOLERANCE

    def test_size_mismatch_is_not_faithful(self):
        photo = synthetic_photo(size=256)
        buf = io.BytesIO()
        photo.resize((200, 200)).save(buf, "PNG")
        result = pdf_utils.render_fidelity(photo, buf.getvalue(), (0, 0, 256, 256), 256.0, 72.0)
        assert result["faithful"] is False and result["size_match"] is False


class TestPageRenderStep:
    def test_page_render_equals_embedded_equals_upload(self):
        # Three-way parity: the whole-page step, the embedded-image step and a
        # direct upload of the same JPEG give the same score and context.
        pdf = pdf_with(synthetic_photo(seed=21), 90)
        upload = main._run_image_analysis(embedded_jpeg_stream(pdf))
        page = main._run_pdf_analysis(pdf)["pages"][0]
        render = page["page_render_scoring"]
        assert page["page_profile"]["render_resampled"] is False
        assert page["page_profile"]["render_fidelity"]["faithful"] is True
        assert render["usable"] is True
        assert render["score"] == pytest.approx(upload["scoring"]["visual_score"], abs=1e-4)
        assert render["score"] == pytest.approx(page["embedded_images"][0]["ai_generated"], abs=1e-4)
        assert render["combiner"]["context_buckets"] == upload["scoring"]["context"]["buckets"]
        assert render["combiner"]["context_buckets"]["compression"] == "jpeg_high"  # from the DCT stream

    def test_non_72_dpi_placement_is_rendered_1_to_1(self):
        buf = io.BytesIO()
        synthetic_photo(seed=22).save(buf, "PDF", quality=95, resolution=150.0)
        page = main._run_pdf_analysis(buf.getvalue())["pages"][0]
        assert page["page_profile"]["raster_native_dpi"] == pytest.approx(150.0, abs=0.01)
        assert page["page_profile"]["render_dpi"] == pytest.approx(150.0, abs=0.01)
        assert page["page_profile"]["render_resampled"] is False

    def test_stretched_image_render_is_not_pixel_evidence(self):
        # 400x400 px drawn into a 600x300 pt box: 48 dpi across, 96 dpi
        # down. No uniform render scale reproduces it 1:1.
        pdf = stretched_pdf(jpeg_bytes(synthetic_photo(seed=23, size=400)), (400, 400), (600.0, 300.0))
        result = main._run_pdf_analysis(pdf)
        page = result["pages"][0]
        assert page["page_profile"]["photo_forensics_applicable"] is True
        assert page["page_profile"]["raster_native_dpi"] == [48.0, 96.0]
        assert page["page_profile"]["render_resampled"] is True
        render = page["page_render_scoring"]
        assert render["confidence"] == "insufficient"
        assert all(not c["used"] for n, c in render["contributions"].items() if n != "lighting")
        # ...while the raster itself is still scored, on its own pixels.
        assert page["embedded_images"][0]["usable"] is True
        assert page["risk_source"] == "embedded_image"

    def test_stretched_image_is_extracted_at_its_own_pixels(self):
        # Regression: extraction drew images through their placement
        # transform (render=True), so a stretched image was scored as 800x400
        # resampled pixels. It must be the file's own 400x400 pixels, scored
        # exactly like an upload of the same JPEG.
        jpeg = jpeg_bytes(synthetic_photo(seed=26, size=400), 90)
        pdf = stretched_pdf(jpeg, (400, 400), (600.0, 300.0))
        embedded = pdf_utils.render_pdf_pages_with_profiles(pdf)[0]["embedded_images"][0]
        profile = embedded["profile"]
        assert (profile["native_width_px"], profile["native_height_px"]) == (400, 400)
        assert profile["extraction"] == "native"
        assert profile["resampled_on_extraction"] is False
        extracted = np.asarray(Image.open(io.BytesIO(embedded["png_bytes"])).convert("RGB"), dtype=np.int16)
        decoded = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"), dtype=np.int16)
        assert np.abs(extracted - decoded).max() == 0

        upload = main._run_image_analysis(jpeg)
        img = main._run_pdf_analysis(pdf)["pages"][0]["embedded_images"][0]
        assert img["ai_generated"] == pytest.approx(upload["scoring"]["visual_score"], abs=1e-4)

    def test_genuine_photo_page_is_not_flagged_by_its_render(self):
        # Regression for the false positive the render step produced before
        # renders were drawn 1:1 without smoothing: smoothing blurred the top
        # band and the frequency signal read it as an upscaling knee.
        path = BASELINE / "jpg/real-photo.jpeg"
        if not path.exists():
            pytest.skip("real-photo.jpeg not present")
        buf = io.BytesIO()
        Image.open(path).convert("RGB").save(buf, "PDF", quality=95, resolution=86.82)
        result = main._run_pdf_analysis(buf.getvalue())
        assert result["flagged_pages"] == []
        page = result["pages"][0]
        # Not just "not flagged": at this non-integer density the render must
        # be detail-faithful and actually used as pixel evidence (smoothing
        # would blur it and get it gated instead).
        assert page["page_profile"]["render_fidelity"]["faithful"] is True
        assert page["page_profile"]["render_resampled"] is False
        render = page["page_render_scoring"]
        assert render["usable"] is True
        assert render["contributions"]["frequency"]["used"] is True
        assert render["score"] < 0.35

    def test_render_that_differs_from_its_raster_is_not_pixel_evidence(self):
        # A nominally 1:1 placement with vector content drawn over the photo:
        # the render is not the raster, so the fidelity check must gate it.
        jpeg = jpeg_bytes(synthetic_photo(seed=27, size=400), 95)
        overlay = b" 0 0 0 RG 1 w" + b"".join(b" %d 0 m %d 400 l S" % (x, x) for x in range(5, 400, 6))
        pdf = stretched_pdf(jpeg, (400, 400), (400.0, 400.0), extra_content=overlay)
        page = main._run_pdf_analysis(pdf)["pages"][0]
        profile = page["page_profile"]
        assert profile["photo_forensics_applicable"] is True
        assert profile["render_scale"] == 1.0
        assert profile["render_fidelity"]["faithful"] is False
        assert profile["render_resampled"] is True
        assert page["page_render_scoring"]["confidence"] == "insufficient"


# --------------------------------------------------------------------------- #
# Metadata step
# --------------------------------------------------------------------------- #
class TestPdfMetadataVerdict:
    HIT = {"score": 0.98, "confidence": "high"}
    LEGACY_HIT = {"score": 0.9, "confidence": "high"}
    NONE = {"score": 0.0, "confidence": "low"}
    AGG = {"score": 0.2, "confidence": "medium", "worst_page": 1}

    def test_signature_raises_a_low_visual_score(self):
        assert main._pdf_metadata_verdict(self.AGG, self.HIT, self.NONE) == (0.98, "high", True)

    def test_signature_never_lowers_stronger_visual_evidence(self):
        # Image-path rule (scoring.combine): max, not override.
        strong = {**self.AGG, "score": 0.995}
        assert main._pdf_metadata_verdict(strong, self.HIT, self.NONE)[0] == 0.995

    def test_legacy_hit_uses_the_same_rule(self):
        assert main._pdf_metadata_verdict(self.AGG, self.NONE, self.LEGACY_HIT) == (0.9, "high", True)

    def test_signature_on_a_document_without_visual_evidence(self):
        empty = {"score": 0.0, "confidence": "insufficient", "worst_page": None}
        assert main._pdf_metadata_verdict(empty, self.HIT, self.NONE) == (0.98, "high", True)

    def test_no_signature_leaves_the_aggregate(self):
        assert main._pdf_metadata_verdict(self.AGG, self.NONE, self.NONE) == (0.2, "medium", True)
        empty = {"score": 0.0, "confidence": "insufficient", "worst_page": None}
        assert main._pdf_metadata_verdict(empty, self.NONE, self.NONE) == (0.0, "insufficient", False)

    def test_common_words_in_a_real_pdf_are_not_provenance(self):
        # End to end: the French cash-flow statement case from
        # verify_baseline_doc.py, through both metadata analyzers.
        buf = io.BytesIO()
        synthetic_photo(seed=24).save(buf, "PDF", quality=90, title="Flux de tresorerie 2024",
                                      author="Maria Dallesandro", subject="Pose de dalles - reflux - InfluxData",
                                      producer="Firefly Print Server")
        result = main._run_pdf_analysis(buf.getvalue())
        assert result["signals"]["pdf_generator_metadata"]["generator_references"] == []
        assert result["signals"]["metadata"]["confidence"] == "low"
        assert result["confidence"] != "high"

    def test_generator_in_a_real_pdf_is_provenance(self):
        buf = io.BytesIO()
        synthetic_photo(seed=25).save(buf, "PDF", quality=90, creator="Midjourney Bot v6")
        result = main._run_pdf_analysis(buf.getvalue())
        assert result["ai_generated"] >= 0.98
        assert result["classification"] == "likely_ai_generated"


# --------------------------------------------------------------------------- #
# The embedded image's own file metadata
# --------------------------------------------------------------------------- #
def jpeg_with_exif(image: Image.Image, exif: dict, quality: int = 92) -> bytes:
    ex = Image.Exif()
    for tag, value in exif.items():
        ex[tag] = value
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=quality, exif=ex)
    return buf.getvalue()


class TestEmbeddedFileMetadataParity:
    """A JPEG embedded as-is (DCTDecode keeps its EXIF) gets the same final
    score as an upload of the same file, metadata included."""

    def _both(self, jpeg: bytes):
        px = Image.open(io.BytesIO(jpeg)).size
        pdf = stretched_pdf(jpeg, px, (float(px[0]), float(px[1])))
        upload = main._run_image_analysis(jpeg)
        result = main._run_pdf_analysis(pdf)
        return upload, result, result["pages"][0]["embedded_images"][0]

    def test_generator_exif_is_caught_inside_a_pdf(self):
        from PIL.ExifTags import Base as Tag
        jpeg = jpeg_with_exif(synthetic_photo(seed=31), {Tag.Software: "Midjourney v6"})
        upload, result, embedded = self._both(jpeg)
        assert upload["ai_generated"] == 0.97
        assert embedded["ai_generated"] == upload["ai_generated"]
        assert embedded["confidence"] == "high"
        assert embedded["scoring"]["metadata"]["signature_hits"] == ["EXIF Software tag references 'midjourney'"]
        assert result["classification"] == "likely_ai_generated"
        # The document's own Info dictionary is clean: the hit came from the image.
        assert result["signals"]["pdf_generator_metadata"]["generator_references"] == []

    def test_plausible_camera_exif_blend_matches(self):
        from PIL.ExifTags import Base as Tag, IFD
        image = synthetic_photo(seed=33).convert("RGB")
        ex = image.getexif()
        ex[Tag.Make] = "Canon"
        ex[Tag.Model] = "Canon EOS R5"
        sub = ex.get_ifd(IFD.Exif)
        sub[Tag.FNumber] = 2.8
        sub[Tag.ISOSpeedRatings] = 200
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=92, exif=ex)
        upload, _, embedded = self._both(buf.getvalue())
        assert upload["signals"]["metadata"]["confidence"] == "low-medium"
        assert embedded["scoring"]["metadata"]["confidence"] == "low-medium"
        assert embedded["ai_generated"] == pytest.approx(upload["ai_generated"], abs=1e-4)

    def test_jpeg_without_exif_is_neutral(self):
        result = main._run_pdf_analysis(pdf_with(synthetic_photo(seed=34), 90))
        embedded = result["pages"][0]["embedded_images"][0]
        assert embedded["filters"] == ["DCTDecode"]
        assert embedded["scoring"]["metadata"]["confidence"] == "low"

    def test_flate_images_carry_no_file_metadata(self):
        # Only a DCTDecode stream is a complete image file; a Flate image is
        # raw pixels with nothing for the metadata analyzer to read.
        import zlib
        rgb = np.asarray(synthetic_photo(seed=35, size=256).convert("RGB"))
        data = zlib.compress(rgb.tobytes())
        pdf = stretched_pdf(data, (256, 256), (256.0, 256.0)).replace(b"/DCTDecode", b"/FlateDecode")
        pages = pdf_utils.render_pdf_pages_with_profiles(pdf)
        embedded = pages[0]["embedded_images"][0]
        assert embedded["profile"]["filters"] == ["FlateDecode"]
        assert embedded["source_bytes"] is None
        assert embedded["profile"]["jpeg_quality"] is None
        scoring = main._run_pdf_analysis(pdf)["pages"][0]["embedded_images"][0]["scoring"]
        assert "metadata" not in scoring
