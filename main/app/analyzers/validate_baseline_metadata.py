"""
Run the ``baseline/`` corpus against the metadata analyzer and use it to
validate the unit-test scenarios in ``verify_metadata.py`` on real files.

    python3 app/analyzers/validate_baseline.py            # from main/
    python3 app/analyzers/validate_baseline.py --verbose  # show analyzer output
    python3 app/analyzers/validate_baseline.py --baseline /path/to/baseline

Three stages, standard library + the project's own deps only (no pytest):

1. **Unit suite** — loads ``app.analyzers.verify_metadata`` through
   ``unittest`` and runs it unmodified. Nothing in that file is edited
   or monkey-patched; this stage just proves the 74 tests (and their
   subtests) still pass in this environment.

2. **Corpus as-is** — every file under ``baseline/`` goes through the
   same entry points production uses (``metadata.analyze`` for images,
   ``pdf_utils.read_pdf_metadata`` + ``metadata.analyze_pdf_document``
   for PDFs). Results are checked against the expectations declared in
   ``CORPUS_EXPECTATIONS`` *and* against the result-contract invariants
   the unit tests assert (keys, score range, confidence vocabulary,
   branch exclusivity, ``"high"`` only on a signature hit).

   Note: the shipped corpus is fully metadata-stripped — no EXIF, no
   PNG text chunks, mundane ReportLab Info dicts — so as-is it only
   reaches the neutral branch. Stage 3 is what covers the rest.

3. **Scenarios on real carriers** — the unit tests exercise the
   analyzer with 16x16 synthetic images. Here the same metadata
   scenarios (generator Software tag, camera EXIF with a proper Exif
   sub-IFD, SD-WebUI / ComfyUI chunks, innocuous chunks, truncation,
   ...) are spliced **byte-level** into the real baseline files in
   memory — an APP1/COM segment after the JPEG SOI, tEXt/eXIf chunks
   after the PNG IHDR — leaving the compressed pixel data untouched
   (which the script asserts by decoding and comparing). Each is then
   checked for the branch/score/hits the unit tests expect. For PDFs
   the real Info dict read from the file is overlaid with generator
   names and fed to ``analyze_pdf_document``.

   Expected scores, field orders and the EXIF builder are imported
   from ``verify_metadata`` so there is a single source of truth; this
   script never writes to ``baseline/`` or to the unit tests.

Exit status is 0 only if all three stages pass.
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import time
import unittest
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]  # main/
_REPO_ROOT = _HERE.parents[3]  # contains main/ and baseline/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PIL import Image  # noqa: E402
from PIL.ExifTags import Base as Tag  # noqa: E402

from app import pdf_utils  # noqa: E402
from app.analyzers import metadata  # noqa: E402
from app.analyzers import verify_metadata as vm  # noqa: E402  (read-only reuse)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
PDF_SUFFIXES = {".pdf"}

# --------------------------------------------------------------------------- #
# Stage 2 expectations for the shipped corpus (paths relative to baseline/).
# Files present on disk but absent here still get the contract checks and
# are reported as "unlisted"; files listed here but missing on disk fail.
# --------------------------------------------------------------------------- #
CORPUS_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "jpg/real-photo.jpeg": {
        "kind": "image",
        "score": vm.NEUTRAL_SCORE,
        "confidence": "low",
        "camera_fields_present": [],
        "why": "metadata-stripped JPEG; no EXIF at all",
    },
    "png/ai-generated.png": {
        "kind": "image",
        "score": vm.NEUTRAL_SCORE,
        "confidence": "low",
        "camera_fields_present": [],
        "why": "generator fingerprint stripped on export; neutral by design",
    },
    "png/ai-retouched.png": {
        "kind": "image",
        "score": vm.NEUTRAL_SCORE,
        "confidence": "low",
        "camera_fields_present": [],
        "why": "no EXIF / text chunks survive the retouch export",
    },
    "pdf/baseline_sample.pdf": {
        "kind": "pdf",
        "score": vm.NEUTRAL_SCORE,
        "confidence": "low",
        "field_contains": {"Producer": "ReportLab"},
        "why": "plain ReportLab document; no generator names",
    },
    "pdf/anomaly_metadata_sample.pdf": {
        "kind": "pdf",
        "score": vm.NEUTRAL_SCORE,
        "confidence": "low",
        "field_contains": {"Producer": "Malformed_Detector_Test"},
        "why": "odd Producer string, but not a known generator — this file "
        "targets the document-anomaly checks, not the metadata analyzer",
    },
}


# --------------------------------------------------------------------------- #
# Result bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class Check:
    stage: str
    name: str
    ok: bool
    detail: str = ""
    payload: Any = None


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, stage: str, name: str, problems: list[str], payload: Any = None) -> Check:
        c = Check(stage, name, not problems, "; ".join(problems), payload)
        self.checks.append(c)
        return c

    def failures(self, stage: str | None = None) -> list[Check]:
        return [c for c in self.checks if not c.ok and (stage is None or c.stage == stage)]

    def count(self, stage: str) -> tuple[int, int]:
        staged = [c for c in self.checks if c.stage == stage]
        return sum(c.ok for c in staged), len(staged)


def _fmt_result(r: dict) -> str:
    slim = {k: v for k, v in r.items() if k != "note"}
    return json.dumps(slim, ensure_ascii=False, default=str)


def _print_check(c: Check, verbose: bool) -> None:
    flag = "PASS" if c.ok else "FAIL"
    line = f"  {flag}  {c.name}"
    if not c.ok:
        line += f"\n        -> {c.detail}"
    print(line)
    if verbose and c.payload is not None:
        print(f"        {_fmt_result(c.payload)}")


# --------------------------------------------------------------------------- #
# Contract checks — mirror TestAnalyzeResultContract / PDF result_contract
# --------------------------------------------------------------------------- #
def image_contract_problems(result: dict) -> list[str]:
    p: list[str] = []
    if not {"score", "confidence", "note"} <= set(result):
        p.append(f"missing keys: {sorted({'score', 'confidence', 'note'} - set(result))}")
        return p
    if not isinstance(result["score"], float) or not 0.0 <= result["score"] <= 1.0:
        p.append(f"score {result['score']!r} not a float in [0, 1]")
    if result["confidence"] not in {"low", "low-medium", "medium", "high"}:
        p.append(f"unexpected confidence {result['confidence']!r}")
    if not isinstance(result["note"], str) or not result["note"]:
        p.append("note missing/empty")
    has_sig = "signature_hits" in result
    has_cam = "camera_fields_present" in result
    if has_sig and has_cam:
        p.append("signature_hits and camera_fields_present are mutually exclusive")
    if (result["confidence"] == "high") != has_sig:
        p.append("confidence 'high' must coincide exactly with a signature hit")
    if has_sig and not result["signature_hits"]:
        p.append("signature_hits present but empty")
    return p


def pdf_contract_problems(result: dict, fields: dict) -> list[str]:
    p: list[str] = []
    if not {"score", "confidence", "note", "pdf_fields"} <= set(result):
        p.append(f"missing keys: {sorted({'score', 'confidence', 'note', 'pdf_fields'} - set(result))}")
        return p
    if not isinstance(result["score"], float) or not 0.0 <= result["score"] <= 1.0:
        p.append(f"score {result['score']!r} not a float in [0, 1]")
    if result["confidence"] not in {"low", "high"}:
        p.append(f"unexpected confidence {result['confidence']!r}")
    if (result["confidence"] == "high") != ("signature_hits" in result):
        p.append("confidence 'high' must coincide exactly with a signature hit")
    expected_echo = fields or None
    if result["pdf_fields"] != expected_echo:
        p.append(f"pdf_fields not echoed: {result['pdf_fields']!r} != {expected_echo!r}")
    return p


def expectation_problems(result: dict, exp: dict[str, Any]) -> list[str]:
    """Declarative comparison used by stages 2 and 3."""
    p: list[str] = []
    if "score" in exp and result.get("score") != exp["score"]:
        p.append(f"score {result.get('score')} != {exp['score']}")
    if "confidence" in exp and result.get("confidence") != exp["confidence"]:
        p.append(f"confidence {result.get('confidence')!r} != {exp['confidence']!r}")
    if "camera_fields_present" in exp:
        got = result.get("camera_fields_present")
        want = exp["camera_fields_present"]
        if exp.get("unordered"):
            got, want = (sorted(got) if got is not None else None), sorted(want)
        if got != want:
            p.append(f"camera_fields_present {got!r} != {want!r}")
    for hit in exp.get("hits_include", []):
        if hit not in result.get("signature_hits", []):
            p.append(f"missing hit {hit!r} in {result.get('signature_hits')!r}")
    if "hits_exact" in exp and result.get("signature_hits") != exp["hits_exact"]:
        p.append(f"signature_hits {result.get('signature_hits')!r} != {exp['hits_exact']!r}")
    if "hit_count" in exp and len(result.get("signature_hits", [])) != exp["hit_count"]:
        p.append(f"hit count {len(result.get('signature_hits', []))} != {exp['hit_count']}")
    if "note_startswith" in exp and not str(result.get("note", "")).startswith(exp["note_startswith"]):
        p.append(f"note does not start with {exp['note_startswith']!r}")
    if exp.get("no_signature") and "signature_hits" in result:
        p.append(f"unexpected signature_hits {result['signature_hits']!r}")
    if exp.get("pdf_fields_is_none") and result.get("pdf_fields") is not None:
        p.append(f"pdf_fields should be None, got {result.get('pdf_fields')!r}")
    return p


# --------------------------------------------------------------------------- #
# Byte-level metadata injection (pixel data is left untouched)
# --------------------------------------------------------------------------- #
def exif_blob(ifd0: dict | None = None, sub_ifd: dict | None = None) -> bytes:
    """'Exif\\0\\0' + TIFF, via the same builder the unit tests use."""
    blob = vm._build_exif(ifd0, sub_ifd)
    if blob is None:
        raise ValueError("no EXIF tags given")
    assert blob.startswith(b"Exif\x00\x00"), "Pillow Exif.tobytes() should carry the APP1 prefix"
    return blob


def jpeg_inject(jpeg: bytes, *, exif: bytes | None = None, comment: bytes | None = None) -> bytes:
    """Insert an APP1 (EXIF) and/or COM segment directly after SOI.

    Placed first so Pillow sees it before any pre-existing APP1 segment;
    the baseline JPEG has none anyway.
    """
    if jpeg[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG (no SOI marker)")
    segments = b""
    if exif is not None:
        segments += b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
    if comment is not None:
        segments += b"\xff\xfe" + struct.pack(">H", len(comment) + 2) + comment
    return jpeg[:2] + segments + jpeg[2:]


def _png_chunk(ctype: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", crc)


def png_inject(png: bytes, *, text: dict[str, str] | None = None, exif: bytes | None = None) -> bytes:
    """Insert tEXt chunks and/or an eXIf chunk directly after IHDR."""
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG (bad signature)")
    ihdr_len = struct.unpack(">I", png[8:12])[0]
    cut = 8 + 12 + ihdr_len  # signature + IHDR(len, type, data, crc)
    chunks = b""
    for key, value in (text or {}).items():
        chunks += _png_chunk(b"tEXt", key.encode("latin-1") + b"\x00" + value.encode("latin-1"))
    if exif is not None:
        # The eXIf chunk carries the bare TIFF; drop the APP1 "Exif\0\0" prefix.
        chunks += _png_chunk(b"eXIf", exif[6:] if exif.startswith(b"Exif\x00\x00") else exif)
    return png[:cut] + chunks + png[cut:]


def pixels_identical(a: bytes, b: bytes) -> bool:
    with Image.open(io.BytesIO(a)) as ia, Image.open(io.BytesIO(b)) as ib:
        return ia.size == ib.size and ia.mode == ib.mode and ia.tobytes() == ib.tobytes()


# --------------------------------------------------------------------------- #
# Stage 3 scenario table — each mirrors a case in verify_metadata.py
# --------------------------------------------------------------------------- #
@dataclass
class ImageScenario:
    name: str
    applies_to: set[str]  # subset of {"JPEG", "PNG"}
    build: Callable[[bytes, str], bytes]  # (carrier_bytes, fmt) -> injected bytes
    expect: dict[str, Any]
    preserves_pixels: bool = True
    mirrors: str = ""  # unit test it corresponds to


def _inject(carrier: bytes, fmt: str, *, exif: bytes | None = None, text: dict | None = None,
            comment: bytes | None = None) -> bytes:
    if fmt == "JPEG":
        return jpeg_inject(carrier, exif=exif, comment=comment)
    if fmt == "PNG":
        return png_inject(carrier, text=text, exif=exif)
    raise ValueError(fmt)


SIG = vm.IMAGE_SIGNATURE_SCORE
CAM = vm.CAMERA_EXIF_SCORE
NEU = vm.NEUTRAL_SCORE
EXPOSURE = ["FNumber", "ExposureTime", "ISOSpeedRatings", "DateTimeOriginal"]

IMAGE_SCENARIOS: list[ImageScenario] = [
    ImageScenario(
        "exif-software-midjourney",
        {"JPEG", "PNG"},
        lambda c, f: _inject(c, f, exif=exif_blob({Tag.Software: "Midjourney"})),
        {"score": SIG, "confidence": "high",
         "hits_exact": ["EXIF Software tag references 'midjourney'"]},
        mirrors="TestAnalyzeExifSoftwareTag.test_software_tag_matching_known_generator_scores_high",
    ),
    ImageScenario(
        "exif-software-case-insensitive",
        {"JPEG", "PNG"},
        lambda c, f: _inject(c, f, exif=exif_blob({Tag.Software: "DALL-E 3"})),
        {"score": SIG, "confidence": "high", "hits_include": ["EXIF Software tag references 'dall-e'"]},
        mirrors="TestAnalyzeExifSoftwareTag.test_matching_is_case_insensitive_and_whole_token_based",
    ),
    ImageScenario(
        "exif-software-mundane-photoshop",
        {"JPEG", "PNG"},
        lambda c, f: _inject(c, f, exif=exif_blob({Tag.Software: "Adobe Photoshop 25.0 (Windows)"})),
        {"score": NEU, "confidence": "low", "no_signature": True, "camera_fields_present": []},
        mirrors="TestAnalyzeExifSoftwareTag.test_mundane_software_strings_do_not_trigger",
    ),
    ImageScenario(
        "camera-exif-realistic-layout-(IFD0+sub-IFD)",
        {"JPEG", "PNG"},
        lambda c, f: _inject(c, f, exif=exif_blob(vm.camera_exif("Make", "Model"), vm.camera_exif(*EXPOSURE))),
        {"score": CAM, "confidence": "low-medium", "camera_fields_present": vm.CAMERA_FIELDS, "no_signature": True},
        mirrors="TestAnalyzeCameraExif.test_realistic_camera_layout_with_exposure_fields_in_sub_ifd",
    ),
    ImageScenario(
        "camera-exif-all-six-in-IFD0",
        {"JPEG", "PNG"},
        lambda c, f: _inject(c, f, exif=exif_blob(vm.FULL_CAMERA_EXIF)),
        {"score": CAM, "confidence": "low-medium", "camera_fields_present": vm.CAMERA_FIELDS},
        mirrors="TestAnalyzeCameraExif.test_all_six_camera_fields_present",
    ),
    ImageScenario(
        "camera-exif-exactly-three-(threshold)",
        {"JPEG", "PNG"},
        lambda c, f: _inject(c, f, exif=exif_blob(vm.camera_exif("Make", "Model"), vm.camera_exif("DateTimeOriginal"))),
        {"score": CAM, "confidence": "low-medium", "camera_fields_present": ["Make", "Model", "DateTimeOriginal"]},
        mirrors="TestAnalyzeCameraExif.test_exactly_three_fields_meets_the_threshold / "
                "test_sub_ifd_fields_count_together_with_ifd0_fields",
    ),
    ImageScenario(
        "camera-exif-two-fields-(below-threshold)",
        {"JPEG", "PNG"},
        lambda c, f: _inject(c, f, exif=exif_blob(vm.camera_exif("Make", "Model"))),
        {"score": NEU, "confidence": "low", "camera_fields_present": ["Make", "Model"], "no_signature": True},
        mirrors="TestAnalyzeCameraExif.test_fewer_than_three_fields_is_neutral",
    ),
    ImageScenario(
        "camera-exif-falsy-values-ignored",
        {"JPEG", "PNG"},
        lambda c, f: _inject(c, f, exif=exif_blob({Tag.Make: "", Tag.Model: "X100V", Tag.FNumber: 2.0,
                                                    Tag.ISOSpeedRatings: 0})),
        {"score": NEU, "confidence": "low", "camera_fields_present": ["Model", "FNumber"]},
        mirrors="TestAnalyzeCameraExif.test_falsy_field_values_do_not_count_toward_threshold",
    ),
    ImageScenario(
        "software-hit-beats-camera-exif",
        {"JPEG", "PNG"},
        lambda c, f: _inject(c, f, exif=exif_blob({**vm.camera_exif("Make", "Model"), Tag.Software: "Stable Diffusion"},
                                                   vm.camera_exif(*EXPOSURE))),
        {"score": SIG, "confidence": "high", "hits_include": ["EXIF Software tag references 'stable diffusion'"]},
        mirrors="TestAnalyzeCameraExif.test_software_hit_in_ifd0_still_beats_realistic_camera_layout",
    ),
    ImageScenario(
        "png-sd-webui-parameters-chunk",
        {"PNG"},
        lambda c, f: _inject(c, f, text={"parameters": "a cat, photorealistic\nSteps: 20, Sampler: Euler a"}),
        {"score": SIG, "confidence": "high",
         "hits_exact": ["PNG text chunk 'parameters' present (generator metadata)"]},
        mirrors="TestAnalyzePngTextChunks.test_sd_webui_parameters_chunk",
    ),
    ImageScenario(
        "png-comfyui-workflow-chunk",
        {"PNG"},
        lambda c, f: _inject(c, f, text={"workflow": '{"3": {"class_type": "KSampler"}}'}),
        {"score": SIG, "confidence": "high",
         "hits_exact": ["PNG text chunk 'workflow' present (generator metadata)"]},
        mirrors="TestAnalyzePngTextChunks.test_comfyui_json_chunks",
    ),
    ImageScenario(
        "png-generator-name-in-comment-value",
        {"PNG"},
        lambda c, f: _inject(c, f, text={"Comment": "Generated with Stable Diffusion XL"}),
        {"score": SIG, "confidence": "high",
         "hits_exact": ["PNG text chunk 'Comment' references 'stable diffusion'"]},
        mirrors="TestAnalyzePngTextChunks.test_generator_name_inside_an_ordinary_chunk_value",
    ),
    ImageScenario(
        "png-chunk-plus-exif-software-accumulate",
        {"PNG"},
        lambda c, f: _inject(c, f, text={"parameters": "Steps: 20"}, exif=exif_blob({Tag.Software: "Midjourney"})),
        {"score": SIG, "confidence": "high", "hit_count": 2,
         "hits_exact": ["PNG text chunk 'parameters' present (generator metadata)",
                        "EXIF Software tag references 'midjourney'"]},
        mirrors="TestAnalyzePngTextChunks.test_png_chunk_hit_combines_with_exif_software_hit",
    ),
    ImageScenario(
        "png-innocuous-chunks-stay-neutral",
        {"PNG"},
        lambda c, f: _inject(c, f, text={"Author": "Jane Doe", "Software": "GIMP 2.10"}),
        {"score": NEU, "confidence": "low", "no_signature": True, "camera_fields_present": []},
        mirrors="TestAnalyzePngTextChunks.test_innocuous_chunks_fall_through_to_neutral",
    ),
    ImageScenario(
        "jpeg-COM-segment-not-scanned",
        {"JPEG"},
        lambda c, f: _inject(c, f, comment=b"parameters: made with stable diffusion"),
        {"score": NEU, "confidence": "low", "no_signature": True},
        mirrors="TestAnalyzePngTextChunks.test_jpeg_comment_segment_is_not_scanned",
    ),
    ImageScenario(
        "truncated-file",
        {"JPEG", "PNG"},
        lambda c, f: c[: len(c) // 2],
        {"score": NEU, "confidence": "low", "note_startswith": "could not parse image metadata"},
        preserves_pixels=False,
        mirrors="TestAnalyzeUnparseableInput.test_truncated_image_is_caught_by_load",
    ),
]


@dataclass
class PdfScenario:
    name: str
    build: Callable[[dict], dict]  # real Info dict -> fields to analyze
    expect: dict[str, Any]
    mirrors: str = ""


PDF_SCENARIOS: list[PdfScenario] = [
    PdfScenario(
        "info-dict-as-read",
        lambda fields: dict(fields),
        {"score": vm.NEUTRAL_SCORE, "confidence": "low", "no_signature": True},
        mirrors="TestAnalyzePdfDocument.test_mundane_fields_are_neutral_and_echoed",
    ),
    PdfScenario(
        "producer-overlaid-with-midjourney",
        lambda fields: {**fields, "Producer": "Midjourney PDF export"},
        {"score": vm.PDF_SIGNATURE_SCORE, "confidence": "high",
         "hits_exact": ["PDF metadata field 'Producer' references 'midjourney'"]},
        mirrors="TestAnalyzePdfDocument.test_generator_in_producer_scores_high",
    ),
    PdfScenario(
        "creator-case-insensitive",
        lambda fields: {**fields, "Creator": "STABLE DIFFUSION WebUI"},
        {"score": vm.PDF_SIGNATURE_SCORE, "confidence": "high",
         "hits_exact": ["PDF metadata field 'Creator' references 'stable diffusion'"]},
        mirrors="TestAnalyzePdfDocument.test_matching_is_case_insensitive",
    ),
    PdfScenario(
        "two-generators-in-subject",
        lambda fields: {**fields, "Subject": "Midjourney vs Ideogram comparison"},
        # Sorted order: the matcher shared with document.py reports terms
        # sorted and de-duplicated.
        {"confidence": "high",
         "hits_exact": ["PDF metadata field 'Subject' references 'ideogram'",
                        "PDF metadata field 'Subject' references 'midjourney'"]},
        mirrors="TestAnalyzePdfDocument.test_one_field_mentioning_two_generators_reports_both",
    ),
    PdfScenario(
        "common-words-are-not-provenance",
        lambda fields: {**fields, "Title": "Flux de tresorerie 2024",
                        "Subject": "Pose de dalles - reflux - InfluxData export",
                        "Author": "Maria Dallesandro", "Producer": "Firefly Print Server"},
        {"score": vm.NEUTRAL_SCORE, "confidence": "low", "no_signature": True},
        mirrors="TestAnalyzePdfDocument.test_ordinary_words_are_not_provenance",
    ),
    PdfScenario(
        "empty-info-dict",
        lambda fields: {},
        {"score": vm.NEUTRAL_SCORE, "confidence": "low", "pdf_fields_is_none": True},
        mirrors="TestAnalyzePdfDocument.test_empty_dict_is_neutral_and_echoes_none",
    ),
]


# --------------------------------------------------------------------------- #
# Stage runners
# --------------------------------------------------------------------------- #
class _CountingResult(unittest.TextTestResult):
    """TextTestResult that also counts subTest outcomes."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.subtests = 0

    def addSubTest(self, test, subtest, outcome):
        super().addSubTest(test, subtest, outcome)
        self.subtests += 1


def stage_unit_suite(report: Report, verbose: bool) -> None:
    print("\n[1/3] Unit suite: app/analyzers/verify_metadata.py (run unmodified)")
    suite = unittest.defaultTestLoader.loadTestsFromModule(vm)
    stream = sys.stdout if verbose else io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2 if verbose else 1, resultclass=_CountingResult)
    t0 = time.perf_counter()
    result = runner.run(suite)
    dt = time.perf_counter() - t0

    problems = []
    for test, tb in result.failures + result.errors:
        problems.append(f"{test.id()}: {tb.strip().splitlines()[-1]}")
    if result.testsRun == 0:
        problems.append("no tests were collected")
    check = report.add(
        "unit",
        f"{result.testsRun} test methods, {result.subtests} subtests, "
        f"{len(result.failures)} failures, {len(result.errors)} errors ({dt:.2f}s)",
        problems,
    )
    _print_check(check, verbose=False)
    if problems and not verbose:
        # Surface tracebacks the quiet runner swallowed.
        print(stream.getvalue())


def _discover_corpus(baseline: Path) -> list[Path]:
    return sorted(
        p for p in baseline.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES | PDF_SUFFIXES and not p.name.startswith(".")
    )


def stage_corpus_as_is(report: Report, baseline: Path, verbose: bool) -> dict[str, bytes]:
    print(f"\n[2/3] Corpus as-is: {baseline}")
    carriers: dict[str, bytes] = {}
    files = _discover_corpus(baseline)
    seen: set[str] = set()

    for path in files:
        rel = path.relative_to(baseline).as_posix()
        seen.add(rel)
        data = path.read_bytes()
        carriers[rel] = data
        exp = CORPUS_EXPECTATIONS.get(rel)
        label = rel if exp else f"{rel} (unlisted — contract checks only)"

        if path.suffix.lower() in IMAGE_SUFFIXES:
            result = metadata.analyze(data)
            problems = image_contract_problems(result)
            if exp:
                if exp["kind"] != "image":
                    problems.append("expectation says this should be a PDF")
                problems += expectation_problems(result, exp)
        else:
            try:
                fields = pdf_utils.read_pdf_metadata(data)
            except Exception as e:  # unreadable PDF is a hard failure for the corpus
                report.add("corpus", label, [f"read_pdf_metadata raised {e!r}"])
                continue
            result = metadata.analyze_pdf_document(fields)
            problems = pdf_contract_problems(result, fields)
            if exp:
                if exp["kind"] != "pdf":
                    problems.append("expectation says this should be an image")
                problems += expectation_problems(result, exp)
                for fname, needle in exp.get("field_contains", {}).items():
                    if needle not in str(fields.get(fname, "")):
                        problems.append(f"Info[{fname}] {fields.get(fname)!r} does not contain {needle!r}")
            result = {**result, "_info_dict": fields}
        _print_check(report.add("corpus", label, problems, result), verbose)

    for rel in sorted(set(CORPUS_EXPECTATIONS) - seen):
        _print_check(report.add("corpus", rel, [f"expected baseline file missing: {baseline / rel}"]), verbose)

    return carriers


def stage_scenarios_on_carriers(report: Report, carriers: dict[str, bytes], verbose: bool) -> None:
    print("\n[3/3] Unit-test scenarios spliced into the real baseline files (in memory)")
    images = {rel: data for rel, data in carriers.items() if Path(rel).suffix.lower() in IMAGE_SUFFIXES}
    pdfs = {rel: data for rel, data in carriers.items() if Path(rel).suffix.lower() in PDF_SUFFIXES}

    for rel, data in images.items():
        try:
            with Image.open(io.BytesIO(data)) as im:
                fmt = im.format
        except Exception as e:
            report.add("scenario", f"{rel}", [f"carrier unreadable: {e!r}"])
            continue
        for sc in IMAGE_SCENARIOS:
            if fmt not in sc.applies_to:
                continue
            name = f"{rel} :: {sc.name}"
            try:
                injected = sc.build(data, fmt)
            except Exception as e:
                _print_check(report.add("scenario", name, [f"injection failed: {e!r}"]), verbose)
                continue
            problems: list[str] = []
            if sc.preserves_pixels:
                if injected == data:
                    problems.append("injection produced identical bytes (nothing was added)")
                elif not pixels_identical(data, injected):
                    problems.append("pixel data changed — injection must only add metadata")
            result = metadata.analyze(injected)
            problems += image_contract_problems(result)
            problems += expectation_problems(result, sc.expect)
            _print_check(report.add("scenario", name, problems, result), verbose)

    for rel, data in pdfs.items():
        try:
            real_fields = pdf_utils.read_pdf_metadata(data)
        except Exception as e:
            report.add("scenario", rel, [f"read_pdf_metadata raised {e!r}"])
            continue
        for sc in PDF_SCENARIOS:
            name = f"{rel} :: {sc.name}"
            fields = sc.build(real_fields)
            result = metadata.analyze_pdf_document(fields)
            problems = pdf_contract_problems(result, fields) + expectation_problems(result, sc.expect)
            _print_check(report.add("scenario", name, problems, result), verbose)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline", type=Path, default=_REPO_ROOT / "baseline",
                    help="path to the baseline corpus (default: <repo>/baseline)")
    ap.add_argument("--skip-unit", action="store_true", help="skip stage 1 (the unittest run)")
    ap.add_argument("--verbose", "-v", action="store_true", help="print analyzer results and unittest detail")
    ap.add_argument("--report", type=Path, help="also write a machine-readable JSON report here")
    args = ap.parse_args(argv)

    if not args.baseline.is_dir():
        print(f"baseline directory not found: {args.baseline}", file=sys.stderr)
        return 2

    report = Report()
    print(f"metadata analyzer : {Path(metadata.__file__).relative_to(_PROJECT_ROOT)}")
    print(f"unit tests        : {Path(vm.__file__).relative_to(_PROJECT_ROOT)}  (not modified)")
    print(f"known generators  : {len(metadata.KNOWN_GENERATOR_SOFTWARE)}")

    if not args.skip_unit:
        stage_unit_suite(report, args.verbose)
    carriers = stage_corpus_as_is(report, args.baseline, args.verbose)
    if not carriers:
        report.add("corpus", "(no files)", ["no PNG/JPEG/PDF files found in the baseline directory"])
    stage_scenarios_on_carriers(report, carriers, args.verbose)

    print("\n" + "=" * 72)
    for stage, title in (("unit", "unit suite"), ("corpus", "corpus as-is"), ("scenario", "scenarios on carriers")):
        ok, total = report.count(stage)
        if total:
            print(f"  {title:<24} {ok:>3}/{total:<3} passed")
    failed = report.failures()
    print("=" * 72)
    if failed:
        print(f"RESULT: FAIL — {len(failed)} check(s) failed")
        for c in failed:
            print(f"  - [{c.stage}] {c.name}: {c.detail}")
    else:
        print("RESULT: OK — every unit test and every baseline check passed")

    if args.report:
        args.report.write_text(json.dumps(
            {"ok": not failed,
             "checks": [{"stage": c.stage, "name": c.name, "ok": c.ok, "detail": c.detail,
                         "result": c.payload} for c in report.checks]},
            indent=2, ensure_ascii=False, default=str))
        print(f"report written to {args.report}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
