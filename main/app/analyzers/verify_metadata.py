"""
Standalone tests for ``app/analyzers/metadata.py`` — standard library only.

Run directly from the project root (the directory containing ``app/``):

    python3 app/analyzers/verify_metadata.py

or through unittest's runner:

    python3 -m unittest app.analyzers.verify_metadata -v

Everything the analyzer is fed is synthesised in memory with Pillow —
no fixture files, no network, no FastAPI app, no third-party test
framework. The suite pins down:

* ``analyze()`` — the image path:
    - unparseable / truncated input degrades to a neutral result
    - EXIF ``Software`` tag naming a known generator  -> 0.97 / "high"
    - PNG tEXt / zTXt / iTXt chunks (SD WebUI, ComfyUI) -> 0.97 / "high"
    - >= 3 plausible camera EXIF fields                -> 0.15 / "low-medium"
      (counted across IFD0 *and* the Exif sub-IFD, where real cameras
      put FNumber/ExposureTime/ISO/DateTimeOriginal)
    - anything else                                     -> 0.50 / "low"
    - generator signatures take precedence over camera EXIF
* ``analyze_pdf_document()`` — the PDF Info-dict path:
    - generator name in any field -> 0.9 / "high", otherwise 0.5 / "low"
* the private helpers ``_get_exif`` / ``_scan_png_text_chunks`` on the
  defensive branches that are awkward to reach through real image bytes
* a couple of invariants the rest of the app leans on (``confidence ==
  "high"`` is what ``main.py`` keys off, list entries must be lowercase
  for matching to ever work, etc.)
"""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, PngImagePlugin
from PIL.ExifTags import Base as Tag

# When executed as a script, sys.path[0] is app/analyzers/, not the project
# root, so ``import app`` would fail. Put the root (two levels up) first.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.analyzers import metadata  # noqa: E402  (import after path bootstrap)
from app.analyzers.metadata import (  # noqa: E402
    KNOWN_GENERATOR_SOFTWARE,
    _get_exif,
    _scan_png_text_chunks,
    analyze,
    analyze_pdf_document,
)

# --------------------------------------------------------------------------- #
# Expected constants — spelled out once so a threshold tweak in the analyzer
# shows up as a single obvious diff here rather than a dozen scattered edits.
# --------------------------------------------------------------------------- #
IMAGE_SIGNATURE_SCORE = 0.97
PDF_SIGNATURE_SCORE = 0.9
CAMERA_EXIF_SCORE = 0.15
NEUTRAL_SCORE = 0.5
CAMERA_FIELD_THRESHOLD = 3

# Canonical order the analyzer reports camera fields in.
CAMERA_FIELDS = ["Make", "Model", "FNumber", "ExposureTime", "ISOSpeedRatings", "DateTimeOriginal"]

# A full, realistic set of camera EXIF values keyed by numeric tag id.
FULL_CAMERA_EXIF = {
    Tag.Make: "Canon",
    Tag.Model: "Canon EOS R5",
    Tag.FNumber: 2.8,
    Tag.ExposureTime: 1 / 250,
    Tag.ISOSpeedRatings: 400,
    Tag.DateTimeOriginal: "2024:03:15 14:22:08",
}
CAMERA_TAG_BY_NAME = {
    "Make": Tag.Make,
    "Model": Tag.Model,
    "FNumber": Tag.FNumber,
    "ExposureTime": Tag.ExposureTime,
    "ISOSpeedRatings": Tag.ISOSpeedRatings,
    "DateTimeOriginal": Tag.DateTimeOriginal,
}

EXPOSURE_FIELDS = ["FNumber", "ExposureTime", "ISOSpeedRatings", "DateTimeOriginal"]


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _base_image(size: tuple[int, int] = (16, 16)) -> Image.Image:
    # A flat colour is fine — the metadata analyzer never looks at pixels.
    return Image.new("RGB", size, (120, 90, 60))


def _build_exif(ifd0: dict | None = None, exif_ifd: dict | None = None) -> bytes | None:
    """Serialise EXIF. ``ifd0`` goes in the top-level IFD; ``exif_ifd`` goes
    in the Exif sub-IFD (tag 0x8769), which is where real cameras put the
    exposure fields."""
    if not ifd0 and not exif_ifd:
        return None
    exif = Image.Exif()
    for tag, value in (ifd0 or {}).items():
        exif[tag] = value
    if exif_ifd:
        sub = exif.get_ifd(0x8769)
        for tag, value in exif_ifd.items():
            sub[tag] = value
    return exif.tobytes()


def make_image(
    fmt: str = "JPEG",
    *,
    exif: dict | None = None,
    exif_sub_ifd: dict | None = None,
    png_text: dict[str, str] | None = None,
    png_itxt: dict[str, str] | None = None,
    png_ztxt: dict[str, str] | None = None,
    jpeg_comment: bytes | None = None,
    size: tuple[int, int] = (16, 16),
) -> bytes:
    """Encode a tiny image with exactly the metadata requested."""
    img = _base_image(size)
    save_kwargs: dict = {}

    exif_bytes = _build_exif(exif, exif_sub_ifd)
    if exif_bytes is not None:
        save_kwargs["exif"] = exif_bytes

    if fmt == "PNG" and (png_text or png_itxt or png_ztxt):
        info = PngImagePlugin.PngInfo()
        for k, v in (png_text or {}).items():
            info.add_text(k, v)  # tEXt
        for k, v in (png_ztxt or {}).items():
            info.add_text(k, v, zip=True)  # zTXt
        for k, v in (png_itxt or {}).items():
            info.add_itxt(k, v, lang="en")  # iTXt
        save_kwargs["pnginfo"] = info

    if fmt == "JPEG" and jpeg_comment is not None:
        save_kwargs["comment"] = jpeg_comment

    buf = io.BytesIO()
    img.save(buf, fmt, **save_kwargs)
    return buf.getvalue()


def camera_exif(*field_names: str) -> dict:
    """Subset of FULL_CAMERA_EXIF selected by tag *name*."""
    return {CAMERA_TAG_BY_NAME[n]: FULL_CAMERA_EXIF[CAMERA_TAG_BY_NAME[n]] for n in field_names}


class MetadataTestCase(unittest.TestCase):
    """Shared assertion helpers."""

    def assertNeutral(self, result: dict) -> None:
        self.assertEqual(result["score"], NEUTRAL_SCORE)
        self.assertEqual(result["confidence"], "low")
        self.assertNotIn("signature_hits", result)

    def assertImageSignature(self, result: dict) -> None:
        self.assertEqual(result["score"], IMAGE_SIGNATURE_SCORE)
        self.assertEqual(result["confidence"], "high")
        self.assertTrue(result["signature_hits"], "a signature verdict must say what it matched")
        self.assertNotIn("camera_fields_present", result)

    def assertCameraExif(self, result: dict) -> None:
        self.assertEqual(result["score"], CAMERA_EXIF_SCORE)
        self.assertEqual(result["confidence"], "low-medium")
        self.assertNotIn("signature_hits", result)


# =========================================================================== #
# KNOWN_GENERATOR_SOFTWARE — invariants the matching logic depends on
# =========================================================================== #
class TestKnownGeneratorList(MetadataTestCase):
    def test_every_entry_is_lowercase(self):
        # The analyzer compares ``name in value.lower()``. A mixed-case
        # entry would silently never match, so guard the list itself.
        for name in KNOWN_GENERATOR_SOFTWARE:
            with self.subTest(name=name):
                self.assertEqual(name, name.lower(), f"{name!r} must be lowercase to ever match")

    def test_entries_are_non_empty_and_unique(self):
        self.assertTrue(all(name.strip() for name in KNOWN_GENERATOR_SOFTWARE))
        self.assertEqual(len(set(KNOWN_GENERATOR_SOFTWARE)), len(KNOWN_GENERATOR_SOFTWARE))

    def test_headline_generators_from_docstring_are_listed(self):
        for expected in ["midjourney", "dall-e", "adobe firefly", "stable diffusion", "comfyui", "automatic1111"]:
            with self.subTest(expected=expected):
                self.assertIn(expected, KNOWN_GENERATOR_SOFTWARE)


# =========================================================================== #
# analyze() — unparseable input
# =========================================================================== #
class TestAnalyzeUnparseableInput(MetadataTestCase):
    def test_unreadable_bytes_degrade_to_neutral_low(self):
        cases = {
            "empty": b"",
            "garbage-text": b"definitely not an image",
            "png-magic-only": b"\x89PNG\r\n\x1a\n",
            "pdf-header": b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n",
        }
        for label, payload in cases.items():
            with self.subTest(label):
                result = analyze(payload)
                self.assertEqual(result["score"], NEUTRAL_SCORE)
                self.assertEqual(result["confidence"], "low")
                self.assertTrue(result["note"].startswith("could not parse image metadata"))
                self.assertNotIn("signature_hits", result)
                self.assertNotIn("camera_fields_present", result)

    def test_truncated_image_is_caught_by_load(self):
        # Image.open() happily parses the header; it's img.load() that fails.
        # The analyzer must call load() and survive the failure.
        full = make_image("JPEG", size=(256, 256))
        result = analyze(full[: len(full) // 2])
        self.assertEqual(result["score"], NEUTRAL_SCORE)
        self.assertEqual(result["confidence"], "low")
        self.assertIn("could not parse image metadata", result["note"])

    def test_never_raises(self):
        for payload in (b"", b"\x00" * 64, bytes(range(256))):
            with self.subTest(payload=payload[:8]):
                analyze(payload)  # must not raise


# =========================================================================== #
# analyze() — EXIF Software tag
# =========================================================================== #
class TestAnalyzeExifSoftwareTag(MetadataTestCase):
    def test_software_tag_matching_known_generator_scores_high(self):
        result = analyze(make_image("JPEG", exif={Tag.Software: "Midjourney"}))
        self.assertImageSignature(result)
        self.assertEqual(result["signature_hits"], ["EXIF Software tag references 'midjourney'"])
        self.assertIn("explicit generator metadata found", result["note"])

    def test_every_listed_generator_name_is_detected(self):
        for name in KNOWN_GENERATOR_SOFTWARE:
            with self.subTest(name=name):
                result = analyze(make_image("JPEG", exif={Tag.Software: name}))
                self.assertImageSignature(result)
                self.assertIn(f"EXIF Software tag references '{name}'", result["signature_hits"])

    def test_matching_is_case_insensitive_and_substring_based(self):
        cases = [
            ("MIDJOURNEY", "midjourney"),
            ("MidJourney", "midjourney"),
            ("DALL-E 3", "dall-e"),
            ("ComfyUI v0.3.10", "comfyui"),
            ("Ideogram 2.0 (web)", "ideogram"),
        ]
        for software, expected_name in cases:
            with self.subTest(software=software):
                result = analyze(make_image("JPEG", exif={Tag.Software: software}))
                self.assertImageSignature(result)
                self.assertIn(f"EXIF Software tag references '{expected_name}'", result["signature_hits"])

    def test_overlapping_list_entries_each_produce_a_hit(self):
        # "adobe firefly" and "firefly" are both listed; the substring check
        # fires once per entry rather than collapsing them. Pin that so a
        # future de-dup is a deliberate change, not an accident.
        result = analyze(make_image("JPEG", exif={Tag.Software: "Adobe Firefly"}))
        self.assertImageSignature(result)
        self.assertEqual(
            result["signature_hits"],
            [
                "EXIF Software tag references 'adobe firefly'",
                "EXIF Software tag references 'firefly'",
            ],
        )

    def test_software_tag_on_png_exif_chunk_is_detected(self):
        # PNG can carry EXIF via an eXIf chunk; the Software check is not
        # JPEG-only.
        result = analyze(make_image("PNG", exif={Tag.Software: "Leonardo.Ai"}))
        self.assertImageSignature(result)
        self.assertIn("EXIF Software tag references 'leonardo.ai'", result["signature_hits"])

    def test_mundane_software_strings_do_not_trigger(self):
        for software in [
            "Adobe Photoshop 25.0 (Windows)",
            "GIMP 2.10.36",
            "Ver.1.20",  # typical camera firmware string
            "Apple iOS 17.4",
            "darktable 4.6.0",
            "",
        ]:
            with self.subTest(software=software):
                result = analyze(make_image("JPEG", exif={Tag.Software: software}))
                self.assertNeutral(result)
                self.assertEqual(result["camera_fields_present"], [])

    def test_software_hit_beats_plausible_camera_exif(self):
        # A generator fingerprint is the stronger evidence; camera fields
        # must not pull a signature hit down to the 0.15 branch.
        exif = {**FULL_CAMERA_EXIF, Tag.Software: "Stable Diffusion"}
        result = analyze(make_image("JPEG", exif=exif))
        self.assertImageSignature(result)
        self.assertEqual(result["score"], IMAGE_SIGNATURE_SCORE)
        self.assertNotIn("camera_fields_present", result)


# =========================================================================== #
# analyze() — camera EXIF plausibility
# =========================================================================== #
class TestAnalyzeCameraExif(MetadataTestCase):
    def test_all_six_camera_fields_present(self):
        result = analyze(make_image("JPEG", exif=FULL_CAMERA_EXIF))
        self.assertCameraExif(result)
        self.assertEqual(result["camera_fields_present"], CAMERA_FIELDS)
        self.assertIn("plausible camera EXIF present", result["note"])

    def test_more_than_three_camera_fields_qualifies_for_low_medium(self):
        cases = {
            "4-fields": ["Make", "Model", "FNumber", "ExposureTime"],
            "5-fields": ["Make", "Model", "FNumber", "ExposureTime", "ISOSpeedRatings"],
            "5-fields-no-make": ["Model", "FNumber", "ExposureTime", "ISOSpeedRatings", "DateTimeOriginal"],
            "6-fields": CAMERA_FIELDS,
        }
        for label, fields in cases.items():
            with self.subTest(label):
                self.assertGreater(len(fields), CAMERA_FIELD_THRESHOLD)
                result = analyze(make_image("JPEG", exif=camera_exif(*fields)))
                self.assertCameraExif(result)
                self.assertEqual(sorted(result["camera_fields_present"]), sorted(fields))

    def test_exactly_three_fields_meets_the_threshold(self):
        # The analyzer uses ``>= 3`` — three is enough, and which three
        # doesn't matter.
        cases = {
            "make-model-fnumber": ["Make", "Model", "FNumber"],
            "exposure-iso-date": ["ExposureTime", "ISOSpeedRatings", "DateTimeOriginal"],
            "mixed": ["Make", "ISOSpeedRatings", "DateTimeOriginal"],
        }
        for label, fields in cases.items():
            with self.subTest(label):
                self.assertEqual(len(fields), CAMERA_FIELD_THRESHOLD)
                result = analyze(make_image("JPEG", exif=camera_exif(*fields)))
                self.assertCameraExif(result)
                self.assertEqual(sorted(result["camera_fields_present"]), sorted(fields))

    def test_fewer_than_three_fields_is_neutral(self):
        cases = {
            "0-fields": [],
            "1-field": ["Make"],
            "2-fields": ["Make", "Model"],
            "2-fields-exposure-only": ["FNumber", "ExposureTime"],
        }
        for label, fields in cases.items():
            with self.subTest(label):
                result = analyze(make_image("JPEG", exif=camera_exif(*fields)))
                self.assertNeutral(result)
                self.assertEqual(sorted(result["camera_fields_present"]), sorted(fields))
                self.assertIn("no/partial camera EXIF", result["note"])

    def test_reported_fields_follow_canonical_order_not_exif_order(self):
        # Build the EXIF in reverse so any dependence on insertion order
        # would show.
        reversed_exif = dict(reversed(list(FULL_CAMERA_EXIF.items())))
        result = analyze(make_image("JPEG", exif=reversed_exif))
        self.assertEqual(result["camera_fields_present"], CAMERA_FIELDS)

    def test_falsy_field_values_do_not_count_toward_threshold(self):
        # Empty Make and ISO=0 are "present" in the EXIF but the analyzer
        # checks truthiness, so they must not be counted. Only Model,
        # FNumber survive -> 2 -> neutral.
        exif = {
            Tag.Make: "",
            Tag.Model: "X100V",
            Tag.FNumber: 2.0,
            Tag.ISOSpeedRatings: 0,
        }
        result = analyze(make_image("JPEG", exif=exif))
        self.assertNeutral(result)
        self.assertEqual(result["camera_fields_present"], ["Model", "FNumber"])

    def test_non_camera_exif_tags_are_ignored(self):
        # Plenty of EXIF, none of it in the camera list -> neutral.
        exif = {
            Tag.ImageDescription: "holiday",
            Tag.Orientation: 1,
            Tag.XResolution: 72.0,
            Tag.YResolution: 72.0,
            Tag.ResolutionUnit: 2,
            Tag.Artist: "J. Doe",
            Tag.Copyright: "(c) 2024",
        }
        result = analyze(make_image("JPEG", exif=exif))
        self.assertNeutral(result)
        self.assertEqual(result["camera_fields_present"], [])

    def test_camera_exif_on_png_is_honoured(self):
        result = analyze(make_image("PNG", exif=FULL_CAMERA_EXIF))
        self.assertCameraExif(result)

    def test_image_with_no_exif_at_all_is_neutral(self):
        result = analyze(make_image("JPEG"))
        self.assertNeutral(result)
        self.assertEqual(result["camera_fields_present"], [])

    # -- Exif sub-IFD (0x8769) ------------------------------------------- #
    # Real cameras put Make/Model in IFD0 but FNumber/ExposureTime/ISO/
    # DateTimeOriginal in the Exif sub-IFD. The analyzer must see both.

    def test_realistic_camera_layout_with_exposure_fields_in_sub_ifd(self):
        payload = make_image(
            "JPEG",
            exif=camera_exif("Make", "Model"),
            exif_sub_ifd=camera_exif(*EXPOSURE_FIELDS),
        )
        result = analyze(payload)
        self.assertCameraExif(result)
        self.assertEqual(result["camera_fields_present"], CAMERA_FIELDS)

    def test_sub_ifd_fields_alone_can_meet_the_threshold(self):
        # No Make/Model at all (stripped by an editor, say) — three
        # exposure fields in the sub-IFD are still enough.
        payload = make_image(
            "JPEG",
            exif_sub_ifd=camera_exif("FNumber", "ExposureTime", "ISOSpeedRatings"),
        )
        result = analyze(payload)
        self.assertCameraExif(result)
        self.assertEqual(result["camera_fields_present"], ["FNumber", "ExposureTime", "ISOSpeedRatings"])

    def test_sub_ifd_fields_count_together_with_ifd0_fields(self):
        # 2 in IFD0 + 1 in the sub-IFD = 3 -> qualifies. Pins that the two
        # IFDs are merged rather than evaluated separately.
        payload = make_image(
            "JPEG",
            exif=camera_exif("Make", "Model"),
            exif_sub_ifd=camera_exif("DateTimeOriginal"),
        )
        result = analyze(payload)
        self.assertCameraExif(result)
        self.assertEqual(result["camera_fields_present"], ["Make", "Model", "DateTimeOriginal"])

    def test_sub_ifd_below_threshold_is_still_neutral(self):
        payload = make_image("JPEG", exif_sub_ifd=camera_exif("FNumber", "ExposureTime"))
        result = analyze(payload)
        self.assertNeutral(result)
        self.assertEqual(result["camera_fields_present"], ["FNumber", "ExposureTime"])

    def test_realistic_camera_layout_on_png(self):
        payload = make_image(
            "PNG",
            exif=camera_exif("Make", "Model"),
            exif_sub_ifd=camera_exif(*EXPOSURE_FIELDS),
        )
        result = analyze(payload)
        self.assertCameraExif(result)
        self.assertEqual(result["camera_fields_present"], CAMERA_FIELDS)

    def test_software_hit_in_ifd0_still_beats_realistic_camera_layout(self):
        payload = make_image(
            "JPEG",
            exif={**camera_exif("Make", "Model"), Tag.Software: "Midjourney"},
            exif_sub_ifd=camera_exif(*EXPOSURE_FIELDS),
        )
        result = analyze(payload)
        self.assertImageSignature(result)
        self.assertIn("EXIF Software tag references 'midjourney'", result["signature_hits"])


# =========================================================================== #
# analyze() — PNG text chunks
# =========================================================================== #
class TestAnalyzePngTextChunks(MetadataTestCase):
    SD_PARAMETERS = (
        "a cat wearing a top hat, photorealistic\n"
        "Negative prompt: blurry\n"
        "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 123, Size: 512x512"
    )

    def test_sd_webui_parameters_chunk(self):
        result = analyze(make_image("PNG", png_text={"parameters": self.SD_PARAMETERS}))
        self.assertImageSignature(result)
        self.assertIn("PNG text chunk 'parameters' present (generator metadata)", result["signature_hits"])

    def test_comfyui_json_chunks(self):
        for key in ["prompt", "workflow"]:
            with self.subTest(key=key):
                result = analyze(make_image("PNG", png_text={key: '{"3": {"class_type": "KSampler"}}'}))
                self.assertImageSignature(result)
                self.assertIn(f"PNG text chunk '{key}' present (generator metadata)", result["signature_hits"])

    def test_special_key_match_is_case_insensitive_but_reports_original_key(self):
        result = analyze(make_image("PNG", png_text={"Parameters": "Steps: 20"}))
        self.assertImageSignature(result)
        self.assertEqual(result["signature_hits"], ["PNG text chunk 'Parameters' present (generator metadata)"])

    def test_generator_name_inside_an_ordinary_chunk_value(self):
        result = analyze(make_image("PNG", png_text={"Comment": "Generated with Stable Diffusion XL"}))
        self.assertImageSignature(result)
        self.assertEqual(result["signature_hits"], ["PNG text chunk 'Comment' references 'stable diffusion'"])

    def test_generator_name_as_the_chunk_key(self):
        result = analyze(make_image("PNG", png_text={"ComfyUI": "some opaque payload"}))
        self.assertImageSignature(result)
        self.assertEqual(result["signature_hits"], ["PNG text chunk 'ComfyUI' references 'comfyui'"])

    def test_special_key_and_value_reference_both_reported(self):
        # "parameters" counts once for being present, and again for every
        # generator name mentioned in its value.
        result = analyze(
            make_image("PNG", png_text={"parameters": "Steps: 20, Model: stable diffusion 1.5 via automatic1111"})
        )
        self.assertImageSignature(result)
        self.assertEqual(
            result["signature_hits"],
            [
                "PNG text chunk 'parameters' present (generator metadata)",
                "PNG text chunk 'parameters' references 'stable diffusion'",
                "PNG text chunk 'parameters' references 'automatic1111'",
            ],
        )

    def test_compressed_ztxt_chunk_is_scanned(self):
        body = ("lorem ipsum " * 40) + "rendered by ComfyUI"
        result = analyze(make_image("PNG", png_ztxt={"Description": body}))
        self.assertImageSignature(result)
        self.assertIn("PNG text chunk 'Description' references 'comfyui'", result["signature_hits"])

    def test_international_itxt_chunk_is_scanned(self):
        result = analyze(make_image("PNG", png_itxt={"Description": "Généré avec Midjourney ✨"}))
        self.assertImageSignature(result)
        self.assertIn("PNG text chunk 'Description' references 'midjourney'", result["signature_hits"])

    def test_multiple_chunks_accumulate_hits(self):
        result = analyze(
            make_image(
                "PNG",
                png_text={"parameters": "Steps: 20", "Software": "made in NightCafe"},
                png_itxt={"Title": "a DALL-E render"},
            )
        )
        self.assertImageSignature(result)
        hits = result["signature_hits"]
        self.assertIn("PNG text chunk 'parameters' present (generator metadata)", hits)
        self.assertIn("PNG text chunk 'Software' references 'nightcafe'", hits)
        self.assertIn("PNG text chunk 'Title' references 'dall-e'", hits)
        self.assertEqual(len(hits), 3)

    def test_innocuous_chunks_fall_through_to_neutral(self):
        cases = {
            "author": {"Author": "Jane Doe"},
            "gimp": {"Software": "GIMP 2.10"},
            "figma": {"Comment": "exported from Figma"},
            "two-innocent": {"Description": "holiday photos 2024", "Copyright": "CC-BY"},
        }
        for label, chunks in cases.items():
            with self.subTest(label):
                result = analyze(make_image("PNG", png_text=chunks))
                self.assertNeutral(result)
                self.assertEqual(result["camera_fields_present"], [])

    def test_png_chunk_hit_combines_with_exif_software_hit(self):
        result = analyze(
            make_image("PNG", png_text={"parameters": "Steps: 20"}, exif={Tag.Software: "Midjourney"})
        )
        self.assertImageSignature(result)
        self.assertEqual(
            result["signature_hits"],
            [
                "PNG text chunk 'parameters' present (generator metadata)",
                "EXIF Software tag references 'midjourney'",
            ],
        )

    def test_jpeg_comment_segment_is_not_scanned(self):
        # The chunk scan is PNG-only by design; a JPEG COM marker carrying the
        # same text must not be picked up. Pins the scope so a future change
        # to scan JPEG comments is deliberate.
        result = analyze(make_image("JPEG", jpeg_comment=b"parameters: made with stable diffusion"))
        self.assertNeutral(result)


# =========================================================================== #
# analyze() — result contract
# =========================================================================== #
class TestAnalyzeResultContract(MetadataTestCase):
    def test_every_branch_returns_score_confidence_and_note(self):
        cases = {
            "unparseable": b"",
            "no-metadata": make_image("JPEG"),
            "software-hit": make_image("JPEG", exif={Tag.Software: "Midjourney"}),
            "png-chunk-hit": make_image("PNG", png_text={"parameters": "x"}),
            "camera-exif": make_image("JPEG", exif=FULL_CAMERA_EXIF),
        }
        for label, payload in cases.items():
            with self.subTest(label):
                result = analyze(payload)
                self.assertTrue(set(result) >= {"score", "confidence", "note"})
                self.assertIsInstance(result["score"], float)
                self.assertTrue(0.0 <= result["score"] <= 1.0)
                self.assertIn(result["confidence"], {"low", "low-medium", "medium", "high"})
                self.assertIsInstance(result["note"], str)
                self.assertTrue(result["note"])

    def test_signature_and_camera_branches_are_mutually_exclusive(self):
        hit = analyze(make_image("JPEG", exif={**FULL_CAMERA_EXIF, Tag.Software: "Midjourney"}))
        self.assertIn("signature_hits", hit)
        self.assertNotIn("camera_fields_present", hit)

        camera = analyze(make_image("JPEG", exif=FULL_CAMERA_EXIF))
        self.assertIn("camera_fields_present", camera)
        self.assertNotIn("signature_hits", camera)

    def test_only_high_confidence_is_produced_by_a_signature_hit(self):
        # main.py treats confidence == "high" as "metadata override"; make
        # sure nothing else can produce it.
        self.assertNotEqual(analyze(make_image("JPEG", exif=FULL_CAMERA_EXIF))["confidence"], "high")
        self.assertNotEqual(analyze(make_image("JPEG"))["confidence"], "high")
        self.assertNotEqual(analyze(b"")["confidence"], "high")
        self.assertEqual(analyze(make_image("JPEG", exif={Tag.Software: "Midjourney"}))["confidence"], "high")

    def test_analyze_does_not_consume_or_mutate_input(self):
        payload = make_image("JPEG", exif=FULL_CAMERA_EXIF)
        snapshot = bytes(payload)
        first = analyze(payload)
        second = analyze(payload)
        self.assertEqual(payload, snapshot)
        self.assertEqual(first, second)


# =========================================================================== #
# Private helpers — defensive branches
# =========================================================================== #
class TestGetExifHelper(MetadataTestCase):
    def test_image_without_exif_returns_empty_dict(self):
        img = Image.open(io.BytesIO(make_image("JPEG")))
        self.assertEqual(_get_exif(img), {})

    def test_tag_ids_are_translated_to_names(self):
        img = Image.open(io.BytesIO(make_image("JPEG", exif={Tag.Make: "Canon", Tag.Software: "x"})))
        exif = _get_exif(img)
        self.assertEqual(exif["Make"], "Canon")
        self.assertEqual(exif["Software"], "x")
        self.assertNotIn(Tag.Make, exif)  # numeric key must have been replaced

    def test_unknown_tag_ids_are_kept_as_integers(self):
        img = Image.open(io.BytesIO(make_image("JPEG", exif={0xFFFE: "custom"})))
        exif = _get_exif(img)
        self.assertEqual(exif.get(0xFFFE), "custom")

    def test_getexif_raising_is_swallowed(self):
        class Boom:
            def getexif(self):
                raise RuntimeError("corrupt APP1 segment")

        self.assertEqual(_get_exif(Boom()), {})

    def test_object_without_getexif_is_swallowed(self):
        self.assertEqual(_get_exif(SimpleNamespace()), {})

    # -- Exif sub-IFD merging ------------------------------------------- #

    def test_sub_ifd_tags_are_merged_and_named(self):
        img = Image.open(
            io.BytesIO(
                make_image(
                    "JPEG",
                    exif=camera_exif("Make", "Model"),
                    exif_sub_ifd=camera_exif(*EXPOSURE_FIELDS),
                )
            )
        )
        exif = _get_exif(img)
        self.assertEqual(exif["Make"], "Canon")
        self.assertEqual(exif["Model"], "Canon EOS R5")
        self.assertAlmostEqual(float(exif["FNumber"]), 2.8, places=6)
        self.assertAlmostEqual(float(exif["ExposureTime"]), 1 / 250, places=9)
        self.assertEqual(exif["ISOSpeedRatings"], 400)
        self.assertEqual(exif["DateTimeOriginal"], "2024:03:15 14:22:08")

    def test_image_without_sub_ifd_is_unaffected(self):
        img = Image.open(io.BytesIO(make_image("JPEG", exif={Tag.Make: "Canon"})))
        exif = _get_exif(img)
        self.assertEqual(exif["Make"], "Canon")
        self.assertNotIn("FNumber", exif)

    def test_ifd0_wins_on_key_collision(self):
        # Shouldn't happen in a well-formed file, but the merge order must
        # be deterministic: IFD0 is authoritative.
        test = self

        class FakeExif(dict):
            def get_ifd(self, tag):
                test.assertEqual(tag, metadata.EXIF_SUB_IFD_TAG)
                return {Tag.Make: "FromSubIFD", Tag.FNumber: 4.0}

        fake_img = SimpleNamespace(getexif=lambda: FakeExif({Tag.Make: "FromIFD0"}))
        exif = _get_exif(fake_img)
        self.assertEqual(exif["Make"], "FromIFD0")
        self.assertEqual(exif["FNumber"], 4.0)

    def test_corrupt_sub_ifd_does_not_discard_ifd0(self):
        class FakeExif(dict):
            def get_ifd(self, tag):
                raise OSError("bad IFD offset")

        fake_img = SimpleNamespace(getexif=lambda: FakeExif({Tag.Make: "Canon", Tag.Model: "R5"}))
        self.assertEqual(_get_exif(fake_img), {"Make": "Canon", "Model": "R5"})

    def test_exif_object_without_get_ifd_is_tolerated(self):
        # Defensive: a minimal mapping that lacks get_ifd() (e.g. a plugin
        # returning a plain dict) should still yield the IFD0 tags.
        fake_img = SimpleNamespace(getexif=lambda: {Tag.Software: "GIMP"})
        self.assertEqual(_get_exif(fake_img), {"Software": "GIMP"})


class TestScanPngTextChunksHelper(MetadataTestCase):
    def test_object_without_text_attribute(self):
        self.assertEqual(_scan_png_text_chunks(SimpleNamespace()), [])

    def test_text_attribute_none(self):
        self.assertEqual(_scan_png_text_chunks(SimpleNamespace(text=None)), [])

    def test_empty_text_mapping(self):
        self.assertEqual(_scan_png_text_chunks(SimpleNamespace(text={})), [])

    def test_non_string_keys_and_values_are_coerced(self):
        # Keys/values are str()-ed before matching, so odd types shouldn't
        # blow up, and a generator name inside them is still found.
        fake = SimpleNamespace(text={42: ["rendered via", "runwayml"], "n": 7})
        hits = _scan_png_text_chunks(fake)
        self.assertEqual(hits, ["PNG text chunk '42' references 'runwayml'"])

    def test_hits_preserve_chunk_iteration_order(self):
        fake = SimpleNamespace(text={"b": "comfyui", "a": "midjourney"})
        self.assertEqual(
            _scan_png_text_chunks(fake),
            [
                "PNG text chunk 'b' references 'comfyui'",
                "PNG text chunk 'a' references 'midjourney'",
            ],
        )


# =========================================================================== #
# analyze_pdf_document()
# =========================================================================== #
class TestAnalyzePdfDocument(MetadataTestCase):
    MUNDANE_FIELDS = {
        "Title": "Quarterly report",
        "Author": "Finance team",
        "Creator": "Microsoft® Word for Microsoft 365",
        "Producer": "Skia/PDF m120",
        "Subject": "",
        "Keywords": "finance, q3",
    }

    def test_empty_dict_is_neutral_and_echoes_none(self):
        result = analyze_pdf_document({})
        self.assertEqual(result["score"], NEUTRAL_SCORE)
        self.assertEqual(result["confidence"], "low")
        self.assertIsNone(result["pdf_fields"])  # ``{} or None`` -> None
        self.assertNotIn("signature_hits", result)
        self.assertIn("no generator reference", result["note"])

    def test_mundane_fields_are_neutral_and_echoed(self):
        result = analyze_pdf_document(dict(self.MUNDANE_FIELDS))
        self.assertEqual(result["score"], NEUTRAL_SCORE)
        self.assertEqual(result["confidence"], "low")
        self.assertEqual(result["pdf_fields"], self.MUNDANE_FIELDS)
        self.assertNotIn("signature_hits", result)

    def test_generator_in_producer_scores_high(self):
        fields = {**self.MUNDANE_FIELDS, "Producer": "Midjourney PDF export"}
        result = analyze_pdf_document(fields)
        self.assertEqual(result["score"], PDF_SIGNATURE_SCORE)
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["signature_hits"], ["PDF metadata field 'Producer' references 'midjourney'"])
        self.assertEqual(result["pdf_fields"], fields)
        self.assertIn("explicit generator reference", result["note"])

    def test_any_standard_field_can_carry_the_hit(self):
        for field in ["Title", "Author", "Creator", "Producer", "Subject", "Keywords"]:
            with self.subTest(field=field):
                fields = {**self.MUNDANE_FIELDS, field: "Made with DALL-E"}
                result = analyze_pdf_document(fields)
                self.assertEqual(result["confidence"], "high")
                self.assertEqual(result["signature_hits"], [f"PDF metadata field '{field}' references 'dall-e'"])

    def test_non_standard_field_names_are_scanned_too(self):
        # The function iterates whatever it's given — it isn't limited to
        # the six Info-dict keys.
        result = analyze_pdf_document({"CustomXmpTool": "Adobe Firefly"})
        self.assertEqual(result["confidence"], "high")
        self.assertIn("PDF metadata field 'CustomXmpTool' references 'adobe firefly'", result["signature_hits"])

    def test_every_listed_generator_name_is_detected(self):
        for name in KNOWN_GENERATOR_SOFTWARE:
            with self.subTest(name=name):
                result = analyze_pdf_document({"Creator": name})
                self.assertEqual(result["score"], PDF_SIGNATURE_SCORE)
                self.assertEqual(result["confidence"], "high")
                self.assertIn(f"PDF metadata field 'Creator' references '{name}'", result["signature_hits"])

    def test_matching_is_case_insensitive(self):
        result = analyze_pdf_document({"Creator": "STABLE DIFFUSION WebUI"})
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["signature_hits"], ["PDF metadata field 'Creator' references 'stable diffusion'"])

    def test_multiple_fields_each_report_their_own_hit(self):
        result = analyze_pdf_document({"Creator": "ComfyUI", "Producer": "Leonardo.Ai exporter", "Title": "cats"})
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(
            result["signature_hits"],
            [
                "PDF metadata field 'Creator' references 'comfyui'",
                "PDF metadata field 'Producer' references 'leonardo.ai'",
            ],
        )

    def test_one_field_mentioning_two_generators_reports_both(self):
        result = analyze_pdf_document({"Subject": "Midjourney vs Ideogram comparison"})
        self.assertEqual(
            result["signature_hits"],
            [
                "PDF metadata field 'Subject' references 'midjourney'",
                "PDF metadata field 'Subject' references 'ideogram'",
            ],
        )

    def test_non_string_values_are_coerced_not_crashed(self):
        # pdf_utils promises str values, but be defensive: None/ints must
        # not raise, and must not accidentally match.
        result = analyze_pdf_document({"Title": None, "Producer": 12345, "Author": 3.14})
        self.assertEqual(result["score"], NEUTRAL_SCORE)
        self.assertEqual(result["confidence"], "low")

    def test_does_not_mutate_input_mapping(self):
        fields = {"Creator": "Midjourney"}
        snapshot = dict(fields)
        analyze_pdf_document(fields)
        self.assertEqual(fields, snapshot)

    def test_result_contract(self):
        for fields in ({}, dict(self.MUNDANE_FIELDS), {"Creator": "Midjourney"}):
            with self.subTest(fields=fields):
                result = analyze_pdf_document(fields)
                self.assertTrue(set(result) >= {"score", "confidence", "note", "pdf_fields"})
                self.assertIsInstance(result["score"], float)
                self.assertTrue(0.0 <= result["score"] <= 1.0)
                self.assertIn(result["confidence"], {"low", "high"})


# =========================================================================== #
# Cross-checks between the image and PDF paths
# =========================================================================== #
class TestImageVsPdfWeighting(MetadataTestCase):
    def test_pdf_hit_is_scored_below_image_hit(self):
        # The module docstring calls the PDF Info-dict "a much weaker signal"
        # than PNG/EXIF evidence; the scores should reflect that ordering.
        image_hit = analyze(make_image("JPEG", exif={Tag.Software: "Midjourney"}))
        pdf_hit = analyze_pdf_document({"Creator": "Midjourney"})
        self.assertLess(pdf_hit["score"], image_hit["score"])
        self.assertEqual(pdf_hit["confidence"], "high")
        self.assertEqual(image_hit["confidence"], "high")

    def test_pdf_path_never_lowers_the_score(self):
        # analyze_pdf_document only raises on a hit; it has no equivalent of
        # the camera-EXIF 0.15 branch.
        neutral = analyze_pdf_document({"Producer": "Canon EOS R5 scanner software"})
        self.assertEqual(neutral["score"], NEUTRAL_SCORE)
        self.assertEqual(neutral["confidence"], "low")

    def test_both_paths_share_the_same_generator_list(self):
        # A name added for one path must be honoured by the other.
        for name in KNOWN_GENERATOR_SOFTWARE:
            with self.subTest(name=name):
                self.assertEqual(analyze(make_image("JPEG", exif={Tag.Software: name}))["confidence"], "high")
                self.assertEqual(analyze_pdf_document({"Title": name})["confidence"], "high")


if __name__ == "__main__":
    unittest.main(verbosity=2)
