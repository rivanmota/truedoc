"""
Standalone tests for ``app/analyzers/metadata.py``.

Everything the analyzer is fed is synthesised in memory with Pillow —
no fixture files, no network, no FastAPI app. The suite pins down:

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

Run with ``pytest tests/`` from the project root.
"""
from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from PIL import Image, PngImagePlugin
from PIL.ExifTags import Base as Tag

from app.analyzers import metadata
from app.analyzers.metadata import (
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


def assert_neutral(result: dict) -> None:
    assert result["score"] == NEUTRAL_SCORE
    assert result["confidence"] == "low"
    assert "signature_hits" not in result


def assert_image_signature(result: dict) -> None:
    assert result["score"] == IMAGE_SIGNATURE_SCORE
    assert result["confidence"] == "high"
    assert result["signature_hits"], "a signature verdict must say what it matched"
    assert "camera_fields_present" not in result


# =========================================================================== #
# KNOWN_GENERATOR_SOFTWARE — invariants the matching logic depends on
# =========================================================================== #
class TestKnownGeneratorList:
    def test_every_entry_is_lowercase(self):
        # The analyzer compares ``name in value.lower()``. A mixed-case
        # entry would silently never match, so guard the list itself.
        for name in KNOWN_GENERATOR_SOFTWARE:
            assert name == name.lower(), f"{name!r} must be lowercase to ever match"

    def test_entries_are_non_empty_and_unique(self):
        assert all(name.strip() for name in KNOWN_GENERATOR_SOFTWARE)
        assert len(set(KNOWN_GENERATOR_SOFTWARE)) == len(KNOWN_GENERATOR_SOFTWARE)

    @pytest.mark.parametrize(
        "expected",
        ["midjourney", "dall-e", "adobe firefly", "stable diffusion", "comfyui", "automatic1111"],
    )
    def test_headline_generators_from_docstring_are_listed(self, expected):
        assert expected in KNOWN_GENERATOR_SOFTWARE


# =========================================================================== #
# analyze() — unparseable input
# =========================================================================== #
class TestAnalyzeUnparseableInput:
    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param(b"", id="empty"),
            pytest.param(b"definitely not an image", id="garbage-text"),
            pytest.param(b"\x89PNG\r\n\x1a\n", id="png-magic-only"),
            pytest.param(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n", id="pdf-header"),
        ],
    )
    def test_unreadable_bytes_degrade_to_neutral_low(self, payload):
        result = analyze(payload)
        assert result["score"] == NEUTRAL_SCORE
        assert result["confidence"] == "low"
        assert result["note"].startswith("could not parse image metadata")
        assert "signature_hits" not in result
        assert "camera_fields_present" not in result

    def test_truncated_image_is_caught_by_load(self):
        # Image.open() happily parses the header; it's img.load() that fails.
        # The analyzer must call load() and survive the failure.
        full = make_image("JPEG", size=(256, 256))
        result = analyze(full[: len(full) // 2])
        assert result["score"] == NEUTRAL_SCORE
        assert result["confidence"] == "low"
        assert "could not parse image metadata" in result["note"]

    def test_never_raises(self):
        for payload in (b"", b"\x00" * 64, bytes(range(256))):
            analyze(payload)  # must not raise


# =========================================================================== #
# analyze() — EXIF Software tag
# =========================================================================== #
class TestAnalyzeExifSoftwareTag:
    def test_software_tag_matching_known_generator_scores_high(self):
        result = analyze(make_image("JPEG", exif={Tag.Software: "Midjourney"}))
        assert_image_signature(result)
        assert result["signature_hits"] == ["EXIF Software tag references 'midjourney'"]
        assert "explicit generator metadata found" in result["note"]

    @pytest.mark.parametrize("name", [n for n in KNOWN_GENERATOR_SOFTWARE if n.isascii()])
    def test_every_listed_generator_name_is_detected(self, name):
        # EXIF Software is an ASCII field; non-ASCII names ("dall·e") cannot
        # be stored in it and are covered through PNG text chunks below.
        result = analyze(make_image("JPEG", exif={Tag.Software: name}))
        assert_image_signature(result)
        assert f"EXIF Software tag references '{name}'" in result["signature_hits"]

    @pytest.mark.parametrize("name", [n for n in KNOWN_GENERATOR_SOFTWARE if not n.isascii()])
    def test_non_ascii_names_are_detected_in_png_text(self, name):
        result = analyze(make_image("PNG", png_text={"Software": name}))
        assert_image_signature(result)
        assert f"PNG text chunk 'Software' references '{name}'" in result["signature_hits"]

    @pytest.mark.parametrize("software", ["Reflux Imaging 2.1", "InfluxData Grafana export",
                                          "Firefly Print Server", "Dallesandro Photo", "Fluxus"])
    def test_ordinary_words_are_not_generator_names(self, software):
        # Whole-token matching shared with document.py: substrings inside
        # ordinary words, and the bare words "flux"/"dalle"/"firefly", are
        # not provenance.
        result = analyze(make_image("JPEG", exif={Tag.Software: software}))
        assert result["confidence"] != "high"
        assert "signature_hits" not in result

    @pytest.mark.parametrize(
        "software, expected_name",
        [
            ("MIDJOURNEY", "midjourney"),
            ("MidJourney", "midjourney"),
            ("DALL-E 3", "dall-e"),
            ("ComfyUI v0.3.10", "comfyui"),
            ("Ideogram 2.0 (web)", "ideogram"),
        ],
    )
    def test_matching_is_case_insensitive_and_whole_token_based(self, software, expected_name):
        result = analyze(make_image("JPEG", exif={Tag.Software: software}))
        assert_image_signature(result)
        assert f"EXIF Software tag references '{expected_name}'" in result["signature_hits"]

    def test_product_name_produces_exactly_one_hit(self):
        # Deliberate change (this test used to pin two hits, for "adobe
        # firefly" and the bare word "firefly"): bare "firefly" is no longer
        # a generator term, so the product name yields one hit.
        result = analyze(make_image("JPEG", exif={Tag.Software: "Adobe Firefly"}))
        assert_image_signature(result)
        assert result["signature_hits"] == ["EXIF Software tag references 'adobe firefly'"]

    def test_software_tag_on_png_exif_chunk_is_detected(self):
        # PNG can carry EXIF via an eXIf chunk; the Software check is not
        # JPEG-only.
        result = analyze(make_image("PNG", exif={Tag.Software: "Leonardo.Ai"}))
        assert_image_signature(result)
        assert "EXIF Software tag references 'leonardo.ai'" in result["signature_hits"]

    @pytest.mark.parametrize(
        "software",
        [
            "Adobe Photoshop 25.0 (Windows)",
            "GIMP 2.10.36",
            "Ver.1.20",  # typical camera firmware string
            "Apple iOS 17.4",
            "darktable 4.6.0",
            "",
        ],
    )
    def test_mundane_software_strings_do_not_trigger(self, software):
        result = analyze(make_image("JPEG", exif={Tag.Software: software}))
        assert_neutral(result)
        assert result["camera_fields_present"] == []

    def test_software_hit_beats_plausible_camera_exif(self):
        # A generator fingerprint is the stronger evidence; camera fields
        # must not pull a signature hit down to the 0.15 branch.
        exif = {**FULL_CAMERA_EXIF, Tag.Software: "Stable Diffusion"}
        result = analyze(make_image("JPEG", exif=exif))
        assert_image_signature(result)
        assert result["score"] == IMAGE_SIGNATURE_SCORE
        assert "camera_fields_present" not in result


# =========================================================================== #
# analyze() — camera EXIF plausibility
# =========================================================================== #
class TestAnalyzeCameraExif:
    def test_all_six_camera_fields_present(self):
        result = analyze(make_image("JPEG", exif=FULL_CAMERA_EXIF))
        assert result["score"] == CAMERA_EXIF_SCORE
        assert result["confidence"] == "low-medium"
        assert result["camera_fields_present"] == CAMERA_FIELDS
        assert "plausible camera EXIF present" in result["note"]
        assert "signature_hits" not in result

    @pytest.mark.parametrize(
        "fields",
        [
            pytest.param(["Make", "Model", "FNumber", "ExposureTime"], id="4-fields"),
            pytest.param(["Make", "Model", "FNumber", "ExposureTime", "ISOSpeedRatings"], id="5-fields"),
            pytest.param(["Model", "FNumber", "ExposureTime", "ISOSpeedRatings", "DateTimeOriginal"], id="5-fields-no-make"),
            pytest.param(CAMERA_FIELDS, id="6-fields"),
        ],
    )
    def test_more_than_three_camera_fields_qualifies_for_low_medium(self, fields):
        assert len(fields) > CAMERA_FIELD_THRESHOLD
        result = analyze(make_image("JPEG", exif=camera_exif(*fields)))
        assert result["score"] == CAMERA_EXIF_SCORE
        assert result["confidence"] == "low-medium"
        assert sorted(result["camera_fields_present"]) == sorted(fields)

    @pytest.mark.parametrize(
        "fields",
        [
            pytest.param(["Make", "Model", "FNumber"], id="make-model-fnumber"),
            pytest.param(["ExposureTime", "ISOSpeedRatings", "DateTimeOriginal"], id="exposure-iso-date"),
            pytest.param(["Make", "ISOSpeedRatings", "DateTimeOriginal"], id="mixed"),
        ],
    )
    def test_exactly_three_fields_meets_the_threshold(self, fields):
        # The analyzer uses ``>= 3`` — three is enough, and which three
        # doesn't matter.
        assert len(fields) == CAMERA_FIELD_THRESHOLD
        result = analyze(make_image("JPEG", exif=camera_exif(*fields)))
        assert result["score"] == CAMERA_EXIF_SCORE
        assert result["confidence"] == "low-medium"
        assert sorted(result["camera_fields_present"]) == sorted(fields)

    @pytest.mark.parametrize(
        "fields",
        [
            pytest.param([], id="0-fields"),
            pytest.param(["Make"], id="1-field"),
            pytest.param(["Make", "Model"], id="2-fields"),
            pytest.param(["FNumber", "ExposureTime"], id="2-fields-exposure-only"),
        ],
    )
    def test_fewer_than_three_fields_is_neutral(self, fields):
        result = analyze(make_image("JPEG", exif=camera_exif(*fields)))
        assert_neutral(result)
        assert sorted(result["camera_fields_present"]) == sorted(fields)
        assert "no/partial camera EXIF" in result["note"]

    def test_reported_fields_follow_canonical_order_not_exif_order(self):
        # Build the EXIF in reverse so any dependence on insertion order
        # would show.
        reversed_exif = dict(reversed(list(FULL_CAMERA_EXIF.items())))
        result = analyze(make_image("JPEG", exif=reversed_exif))
        assert result["camera_fields_present"] == CAMERA_FIELDS

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
        assert_neutral(result)
        assert result["camera_fields_present"] == ["Model", "FNumber"]

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
        assert_neutral(result)
        assert result["camera_fields_present"] == []

    def test_camera_exif_on_png_is_honoured(self):
        result = analyze(make_image("PNG", exif=FULL_CAMERA_EXIF))
        assert result["score"] == CAMERA_EXIF_SCORE
        assert result["confidence"] == "low-medium"

    def test_image_with_no_exif_at_all_is_neutral(self):
        result = analyze(make_image("JPEG"))
        assert_neutral(result)
        assert result["camera_fields_present"] == []

    # -- Exif sub-IFD (0x8769) ------------------------------------------- #
    # Real cameras put Make/Model in IFD0 but FNumber/ExposureTime/ISO/
    # DateTimeOriginal in the Exif sub-IFD. The analyzer must see both.

    def test_realistic_camera_layout_with_exposure_fields_in_sub_ifd(self):
        payload = make_image(
            "JPEG",
            exif=camera_exif("Make", "Model"),
            exif_sub_ifd=camera_exif("FNumber", "ExposureTime", "ISOSpeedRatings", "DateTimeOriginal"),
        )
        result = analyze(payload)
        assert result["score"] == CAMERA_EXIF_SCORE
        assert result["confidence"] == "low-medium"
        assert result["camera_fields_present"] == CAMERA_FIELDS

    def test_sub_ifd_fields_alone_can_meet_the_threshold(self):
        # No Make/Model at all (stripped by an editor, say) — three
        # exposure fields in the sub-IFD are still enough.
        payload = make_image(
            "JPEG",
            exif_sub_ifd=camera_exif("FNumber", "ExposureTime", "ISOSpeedRatings"),
        )
        result = analyze(payload)
        assert result["score"] == CAMERA_EXIF_SCORE
        assert result["confidence"] == "low-medium"
        assert result["camera_fields_present"] == ["FNumber", "ExposureTime", "ISOSpeedRatings"]

    def test_sub_ifd_fields_count_together_with_ifd0_fields(self):
        # 2 in IFD0 + 1 in the sub-IFD = 3 -> qualifies. Pins that the two
        # IFDs are merged rather than evaluated separately.
        payload = make_image(
            "JPEG",
            exif=camera_exif("Make", "Model"),
            exif_sub_ifd=camera_exif("DateTimeOriginal"),
        )
        result = analyze(payload)
        assert result["score"] == CAMERA_EXIF_SCORE
        assert result["camera_fields_present"] == ["Make", "Model", "DateTimeOriginal"]

    def test_sub_ifd_below_threshold_is_still_neutral(self):
        payload = make_image("JPEG", exif_sub_ifd=camera_exif("FNumber", "ExposureTime"))
        result = analyze(payload)
        assert_neutral(result)
        assert result["camera_fields_present"] == ["FNumber", "ExposureTime"]

    def test_realistic_camera_layout_on_png(self):
        payload = make_image(
            "PNG",
            exif=camera_exif("Make", "Model"),
            exif_sub_ifd=camera_exif("FNumber", "ExposureTime", "ISOSpeedRatings", "DateTimeOriginal"),
        )
        result = analyze(payload)
        assert result["score"] == CAMERA_EXIF_SCORE
        assert result["camera_fields_present"] == CAMERA_FIELDS

    def test_software_hit_in_ifd0_still_beats_realistic_camera_layout(self):
        payload = make_image(
            "JPEG",
            exif={**camera_exif("Make", "Model"), Tag.Software: "Midjourney"},
            exif_sub_ifd=camera_exif("FNumber", "ExposureTime", "ISOSpeedRatings", "DateTimeOriginal"),
        )
        result = analyze(payload)
        assert_image_signature(result)
        assert "EXIF Software tag references 'midjourney'" in result["signature_hits"]


# =========================================================================== #
# analyze() — PNG text chunks
# =========================================================================== #
class TestAnalyzePngTextChunks:
    SD_PARAMETERS = (
        "a cat wearing a top hat, photorealistic\n"
        "Negative prompt: blurry\n"
        "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 123, Size: 512x512"
    )

    def test_sd_webui_parameters_chunk(self):
        result = analyze(make_image("PNG", png_text={"parameters": self.SD_PARAMETERS}))
        assert_image_signature(result)
        assert "PNG text chunk 'parameters' present (generator metadata)" in result["signature_hits"]

    @pytest.mark.parametrize("key", ["prompt", "workflow"])
    def test_comfyui_json_chunks(self, key):
        result = analyze(make_image("PNG", png_text={key: '{"3": {"class_type": "KSampler"}}'}))
        assert_image_signature(result)
        assert f"PNG text chunk '{key}' present (generator metadata)" in result["signature_hits"]

    def test_special_key_match_is_case_insensitive_but_reports_original_key(self):
        result = analyze(make_image("PNG", png_text={"Parameters": "Steps: 20"}))
        assert_image_signature(result)
        assert result["signature_hits"] == ["PNG text chunk 'Parameters' present (generator metadata)"]

    def test_generator_name_inside_an_ordinary_chunk_value(self):
        result = analyze(make_image("PNG", png_text={"Comment": "Generated with Stable Diffusion XL"}))
        assert_image_signature(result)
        assert result["signature_hits"] == ["PNG text chunk 'Comment' references 'stable diffusion'"]

    def test_generator_name_as_the_chunk_key(self):
        result = analyze(make_image("PNG", png_text={"ComfyUI": "some opaque payload"}))
        assert_image_signature(result)
        assert result["signature_hits"] == ["PNG text chunk 'ComfyUI' references 'comfyui'"]

    def test_special_key_and_value_reference_both_reported(self):
        # "parameters" counts once for being present, and again for every
        # generator name mentioned in its value.
        result = analyze(
            make_image("PNG", png_text={"parameters": "Steps: 20, Model: stable diffusion 1.5 via automatic1111"})
        )
        assert_image_signature(result)
        # Generator names are reported in the shared matcher's sorted order.
        assert result["signature_hits"] == [
            "PNG text chunk 'parameters' present (generator metadata)",
            "PNG text chunk 'parameters' references 'automatic1111'",
            "PNG text chunk 'parameters' references 'stable diffusion'",
        ]

    def test_compressed_ztxt_chunk_is_scanned(self):
        body = ("lorem ipsum " * 40) + "rendered by ComfyUI"
        result = analyze(make_image("PNG", png_ztxt={"Description": body}))
        assert_image_signature(result)
        assert "PNG text chunk 'Description' references 'comfyui'" in result["signature_hits"]

    def test_international_itxt_chunk_is_scanned(self):
        result = analyze(make_image("PNG", png_itxt={"Description": "Généré avec Midjourney ✨"}))
        assert_image_signature(result)
        assert "PNG text chunk 'Description' references 'midjourney'" in result["signature_hits"]

    def test_multiple_chunks_accumulate_hits(self):
        result = analyze(
            make_image(
                "PNG",
                png_text={"parameters": "Steps: 20", "Software": "made in NightCafe"},
                png_itxt={"Title": "a DALL-E render"},
            )
        )
        assert_image_signature(result)
        hits = result["signature_hits"]
        assert "PNG text chunk 'parameters' present (generator metadata)" in hits
        assert "PNG text chunk 'Software' references 'nightcafe'" in hits
        assert "PNG text chunk 'Title' references 'dall-e'" in hits
        assert len(hits) == 3

    @pytest.mark.parametrize(
        "chunks",
        [
            pytest.param({"Author": "Jane Doe"}, id="author"),
            pytest.param({"Software": "GIMP 2.10"}, id="gimp"),
            pytest.param({"Comment": "exported from Figma"}, id="figma"),
            pytest.param({"Description": "holiday photos 2024", "Copyright": "CC-BY"}, id="two-innocent"),
        ],
    )
    def test_innocuous_chunks_fall_through_to_neutral(self, chunks):
        result = analyze(make_image("PNG", png_text=chunks))
        assert_neutral(result)
        assert result["camera_fields_present"] == []

    def test_png_chunk_hit_combines_with_exif_software_hit(self):
        result = analyze(
            make_image("PNG", png_text={"parameters": "Steps: 20"}, exif={Tag.Software: "Midjourney"})
        )
        assert_image_signature(result)
        assert result["signature_hits"] == [
            "PNG text chunk 'parameters' present (generator metadata)",
            "EXIF Software tag references 'midjourney'",
        ]

    def test_jpeg_comment_segment_is_not_scanned(self):
        # The chunk scan is PNG-only by design; a JPEG COM marker carrying the
        # same text must not be picked up. Pins the scope so a future change
        # to scan JPEG comments is deliberate.
        result = analyze(make_image("JPEG", jpeg_comment=b"parameters: made with stable diffusion"))
        assert_neutral(result)


# =========================================================================== #
# analyze() — result contract
# =========================================================================== #
class TestAnalyzeResultContract:
    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param(b"", id="unparseable"),
            pytest.param(make_image("JPEG"), id="no-metadata"),
            pytest.param(make_image("JPEG", exif={Tag.Software: "Midjourney"}), id="software-hit"),
            pytest.param(make_image("PNG", png_text={"parameters": "x"}), id="png-chunk-hit"),
            pytest.param(make_image("JPEG", exif=FULL_CAMERA_EXIF), id="camera-exif"),
        ],
    )
    def test_every_branch_returns_score_confidence_and_note(self, payload):
        result = analyze(payload)
        assert set(result) >= {"score", "confidence", "note"}
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0
        assert result["confidence"] in {"low", "low-medium", "medium", "high"}
        assert isinstance(result["note"], str) and result["note"]

    def test_signature_and_camera_branches_are_mutually_exclusive(self):
        hit = analyze(make_image("JPEG", exif={**FULL_CAMERA_EXIF, Tag.Software: "Midjourney"}))
        assert "signature_hits" in hit and "camera_fields_present" not in hit

        camera = analyze(make_image("JPEG", exif=FULL_CAMERA_EXIF))
        assert "camera_fields_present" in camera and "signature_hits" not in camera

    def test_only_high_confidence_is_produced_by_a_signature_hit(self):
        # main.py treats confidence == "high" as "metadata override"; make
        # sure nothing else can produce it.
        assert analyze(make_image("JPEG", exif=FULL_CAMERA_EXIF))["confidence"] != "high"
        assert analyze(make_image("JPEG"))["confidence"] != "high"
        assert analyze(b"")["confidence"] != "high"
        assert analyze(make_image("JPEG", exif={Tag.Software: "Midjourney"}))["confidence"] == "high"

    def test_analyze_does_not_consume_or_mutate_input(self):
        payload = make_image("JPEG", exif=FULL_CAMERA_EXIF)
        snapshot = bytes(payload)
        first = analyze(payload)
        second = analyze(payload)
        assert payload == snapshot
        assert first == second


# =========================================================================== #
# Private helpers — defensive branches
# =========================================================================== #
class TestGetExifHelper:
    def test_image_without_exif_returns_empty_dict(self):
        img = Image.open(io.BytesIO(make_image("JPEG")))
        assert _get_exif(img) == {}

    def test_tag_ids_are_translated_to_names(self):
        img = Image.open(io.BytesIO(make_image("JPEG", exif={Tag.Make: "Canon", Tag.Software: "x"})))
        exif = _get_exif(img)
        assert exif["Make"] == "Canon"
        assert exif["Software"] == "x"
        assert Tag.Make not in exif  # numeric key must have been replaced

    def test_unknown_tag_ids_are_kept_as_integers(self):
        img = Image.open(io.BytesIO(make_image("JPEG", exif={0xFFFE: "custom"})))
        exif = _get_exif(img)
        assert exif.get(0xFFFE) == "custom"

    def test_getexif_raising_is_swallowed(self):
        class Boom:
            def getexif(self):
                raise RuntimeError("corrupt APP1 segment")

        assert _get_exif(Boom()) == {}

    def test_object_without_getexif_is_swallowed(self):
        assert _get_exif(SimpleNamespace()) == {}

    # -- Exif sub-IFD merging ------------------------------------------- #

    def test_sub_ifd_tags_are_merged_and_named(self):
        img = Image.open(
            io.BytesIO(
                make_image(
                    "JPEG",
                    exif=camera_exif("Make", "Model"),
                    exif_sub_ifd=camera_exif("FNumber", "ExposureTime", "ISOSpeedRatings", "DateTimeOriginal"),
                )
            )
        )
        exif = _get_exif(img)
        assert exif["Make"] == "Canon"
        assert exif["Model"] == "Canon EOS R5"
        assert float(exif["FNumber"]) == pytest.approx(2.8)
        assert float(exif["ExposureTime"]) == pytest.approx(1 / 250)
        assert exif["ISOSpeedRatings"] == 400
        assert exif["DateTimeOriginal"] == "2024:03:15 14:22:08"

    def test_image_without_sub_ifd_is_unaffected(self):
        img = Image.open(io.BytesIO(make_image("JPEG", exif={Tag.Make: "Canon"})))
        exif = _get_exif(img)
        assert exif["Make"] == "Canon"
        assert "FNumber" not in exif

    def test_ifd0_wins_on_key_collision(self):
        # Shouldn't happen in a well-formed file, but the merge order must
        # be deterministic: IFD0 is authoritative.
        class FakeExif(dict):
            def get_ifd(self, tag):
                assert tag == metadata.EXIF_SUB_IFD_TAG
                return {Tag.Make: "FromSubIFD", Tag.FNumber: 4.0}

        fake_img = SimpleNamespace(getexif=lambda: FakeExif({Tag.Make: "FromIFD0"}))
        exif = _get_exif(fake_img)
        assert exif["Make"] == "FromIFD0"
        assert exif["FNumber"] == 4.0

    def test_corrupt_sub_ifd_does_not_discard_ifd0(self):
        class FakeExif(dict):
            def get_ifd(self, tag):
                raise OSError("bad IFD offset")

        fake_img = SimpleNamespace(getexif=lambda: FakeExif({Tag.Make: "Canon", Tag.Model: "R5"}))
        assert _get_exif(fake_img) == {"Make": "Canon", "Model": "R5"}

    def test_exif_object_without_get_ifd_is_tolerated(self):
        # Defensive: a minimal mapping that lacks get_ifd() (e.g. a plugin
        # returning a plain dict) should still yield the IFD0 tags.
        fake_img = SimpleNamespace(getexif=lambda: {Tag.Software: "GIMP"})
        assert _get_exif(fake_img) == {"Software": "GIMP"}


class TestScanPngTextChunksHelper:
    def test_object_without_text_attribute(self):
        assert _scan_png_text_chunks(SimpleNamespace()) == []

    def test_text_attribute_none(self):
        assert _scan_png_text_chunks(SimpleNamespace(text=None)) == []

    def test_empty_text_mapping(self):
        assert _scan_png_text_chunks(SimpleNamespace(text={})) == []

    def test_non_string_keys_and_values_are_coerced(self):
        # Keys/values are str()-ed before matching, so odd types shouldn't
        # blow up, and a generator name inside them is still found.
        fake = SimpleNamespace(text={42: ["rendered via", "runwayml"], "n": 7})
        hits = _scan_png_text_chunks(fake)
        assert hits == ["PNG text chunk '42' references 'runwayml'"]

    def test_hits_preserve_chunk_iteration_order(self):
        fake = SimpleNamespace(text={"b": "comfyui", "a": "midjourney"})
        assert _scan_png_text_chunks(fake) == [
            "PNG text chunk 'b' references 'comfyui'",
            "PNG text chunk 'a' references 'midjourney'",
        ]


# =========================================================================== #
# analyze_pdf_document()
# =========================================================================== #
class TestAnalyzePdfDocument:
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
        assert result["score"] == NEUTRAL_SCORE
        assert result["confidence"] == "low"
        assert result["pdf_fields"] is None  # ``{} or None`` -> None
        assert "signature_hits" not in result
        assert "no generator reference" in result["note"]

    def test_mundane_fields_are_neutral_and_echoed(self):
        result = analyze_pdf_document(dict(self.MUNDANE_FIELDS))
        assert result["score"] == NEUTRAL_SCORE
        assert result["confidence"] == "low"
        assert result["pdf_fields"] == self.MUNDANE_FIELDS
        assert "signature_hits" not in result

    def test_generator_in_producer_scores_high(self):
        fields = {**self.MUNDANE_FIELDS, "Producer": "Midjourney PDF export"}
        result = analyze_pdf_document(fields)
        assert result["score"] == PDF_SIGNATURE_SCORE
        assert result["confidence"] == "high"
        assert result["signature_hits"] == ["PDF metadata field 'Producer' references 'midjourney'"]
        assert result["pdf_fields"] == fields
        assert "explicit generator reference" in result["note"]

    @pytest.mark.parametrize("field", ["Title", "Author", "Creator", "Producer", "Subject", "Keywords"])
    def test_any_standard_field_can_carry_the_hit(self, field):
        fields = {**self.MUNDANE_FIELDS, field: "Made with DALL-E"}
        result = analyze_pdf_document(fields)
        assert result["confidence"] == "high"
        assert result["signature_hits"] == [f"PDF metadata field '{field}' references 'dall-e'"]

    def test_non_standard_field_names_are_scanned_too(self):
        # The function iterates whatever it's given — it isn't limited to
        # the six Info-dict keys.
        result = analyze_pdf_document({"CustomXmpTool": "Adobe Firefly"})
        assert result["confidence"] == "high"
        assert "PDF metadata field 'CustomXmpTool' references 'adobe firefly'" in result["signature_hits"]

    @pytest.mark.parametrize("name", KNOWN_GENERATOR_SOFTWARE)
    def test_every_listed_generator_name_is_detected(self, name):
        result = analyze_pdf_document({"Creator": name})
        assert result["score"] == PDF_SIGNATURE_SCORE
        assert result["confidence"] == "high"
        assert f"PDF metadata field 'Creator' references '{name}'" in result["signature_hits"]

    def test_matching_is_case_insensitive(self):
        result = analyze_pdf_document({"Creator": "STABLE DIFFUSION WebUI"})
        assert result["confidence"] == "high"
        assert result["signature_hits"] == ["PDF metadata field 'Creator' references 'stable diffusion'"]

    def test_multiple_fields_each_report_their_own_hit(self):
        result = analyze_pdf_document({"Creator": "ComfyUI", "Producer": "Leonardo.Ai exporter", "Title": "cats"})
        assert result["confidence"] == "high"
        assert result["signature_hits"] == [
            "PDF metadata field 'Creator' references 'comfyui'",
            "PDF metadata field 'Producer' references 'leonardo.ai'",
        ]

    def test_one_field_mentioning_two_generators_reports_both(self):
        result = analyze_pdf_document({"Subject": "Midjourney vs Ideogram comparison"})
        # Sorted order, as reported by the shared matcher.
        assert result["signature_hits"] == [
            "PDF metadata field 'Subject' references 'ideogram'",
            "PDF metadata field 'Subject' references 'midjourney'",
        ]

    @pytest.mark.parametrize("fields", [
        {"Title": "Flux de tresorerie 2024"},
        {"Subject": "Pose de dalles - reflux - InfluxData export"},
        {"Author": "Maria Dallesandro"},
        {"Producer": "Firefly Print Server"},
    ])
    def test_ordinary_words_are_not_provenance(self, fields):
        result = analyze_pdf_document(fields)
        assert result["confidence"] == "low"
        assert result["score"] == NEUTRAL_SCORE

    def test_non_string_values_are_coerced_not_crashed(self):
        # pdf_utils promises str values, but be defensive: None/ints must
        # not raise, and must not accidentally match.
        result = analyze_pdf_document({"Title": None, "Producer": 12345, "Author": 3.14})
        assert result["score"] == NEUTRAL_SCORE
        assert result["confidence"] == "low"

    def test_does_not_mutate_input_mapping(self):
        fields = {"Creator": "Midjourney"}
        snapshot = dict(fields)
        analyze_pdf_document(fields)
        assert fields == snapshot

    def test_result_contract(self):
        for fields in ({}, dict(self.MUNDANE_FIELDS), {"Creator": "Midjourney"}):
            result = analyze_pdf_document(fields)
            assert set(result) >= {"score", "confidence", "note", "pdf_fields"}
            assert isinstance(result["score"], float)
            assert 0.0 <= result["score"] <= 1.0
            assert result["confidence"] in {"low", "high"}


# =========================================================================== #
# Cross-checks between the image and PDF paths
# =========================================================================== #
class TestImageVsPdfWeighting:
    def test_pdf_hit_is_scored_below_image_hit(self):
        # The module docstring calls the PDF Info-dict "a much weaker signal"
        # than PNG/EXIF evidence; the scores should reflect that ordering.
        image_hit = analyze(make_image("JPEG", exif={Tag.Software: "Midjourney"}))
        pdf_hit = analyze_pdf_document({"Creator": "Midjourney"})
        assert pdf_hit["score"] < image_hit["score"]
        assert pdf_hit["confidence"] == image_hit["confidence"] == "high"

    def test_pdf_path_never_lowers_the_score(self):
        # analyze_pdf_document only raises on a hit; it has no equivalent of
        # the camera-EXIF 0.15 branch.
        neutral = analyze_pdf_document({"Producer": "Canon EOS R5 scanner software"})
        assert neutral["score"] == NEUTRAL_SCORE
        assert neutral["confidence"] == "low"

    def test_both_paths_share_the_same_generator_list(self):
        # A name added for one path must be honoured by the other. EXIF is
        # ASCII-only, so non-ASCII names go through a PNG text chunk.
        for name in KNOWN_GENERATOR_SOFTWARE:
            image = (make_image("JPEG", exif={Tag.Software: name}) if name.isascii()
                     else make_image("PNG", png_text={"Software": name}))
            assert analyze(image)["confidence"] == "high"
            assert analyze_pdf_document({"Title": name})["confidence"] == "high"

    def test_list_and_matcher_are_the_ones_document_py_uses(self):
        from app.analyzers import document
        assert KNOWN_GENERATOR_SOFTWARE is document.AI_GENERATOR_TERMS
        for fields in ({"Title": "reflux study"}, {"Producer": "InfluxData"}, {"Author": "Dallesandro"},
                       {"Creator": "Midjourney"}, {"Creator": "InvokeAI"}, {"Creator": "ChatGPT"},
                       {"Producer": "Adobe Firefly 3"}, {"Creator": "FLUX.1 [dev]"}, {"Title": "Flux de tresorerie"}):
            assert analyze_pdf_document(fields)["confidence"] == document.analyze_pdf_metadata(fields)["confidence"]
