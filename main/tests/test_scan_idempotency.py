"""
Idempotency of ``POST /api/v1/scans``.

Contract: the same input (request fields + file bytes) on the same engine
version gives the same response, byte for byte, except for

    summary.scan_id          a new UUID per scan
    summary.timestamp        when the scan ran
    details.duration_ms      how long it took
    summary.engine_version   (and details.engine.version) - outputs may change
                             between versions, never within one

"Same response" is checked on the canonical JSON of everything else, so a
new field that varies between runs fails these tests until it is made
deterministic or deliberately added to EXEMPT.

What could break it, and the test that guards each:

- repeated scans, success and error paths    TestRepeatedScans, TestErrors
- concurrent scans (thread-pool analysis):
  shared OpenCV face cascades and PDFium     TestConcurrency, TestPdfiumGuard
- process-level randomness (hash seeds)      TestAcrossProcesses
- the wall clock: read once per scan; only the future-dated-metadata check
  depends on it, via the (exempt) timestamp. That check is a deliberate
  exception: finding the anomaly now but not later is expected  TestClock
- unstable error text (object addresses)     TestErrors, TestFetchErrorMessages
"""
from __future__ import annotations

import concurrent.futures as cf
import copy
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pypdfium2
import pytest
import requests
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin

from app import main, pdfium_guard, scan_api
from app import scan_validation as validation
from app.analyzers import gradients

MAIN_DIR = Path(__file__).resolve().parents[1]
BASELINE = MAIN_DIR.parent / "baseline"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

pytestmark = pytest.mark.skipif(not BASELINE.exists(), reason="baseline files not present")

EXEMPT = (
    ("summary", "scan_id"),
    ("summary", "timestamp"),
    ("summary", "engine_version"),
    ("details", "duration_ms"),
    ("details", "engine", "version"),
)


def comparable(body: dict) -> dict:
    """The response without the exempt fields."""
    body = copy.deepcopy(body)
    for path in EXEMPT:
        node = body
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return body


def canonical(body: dict) -> str:
    return json.dumps(comparable(body), sort_keys=True, ensure_ascii=False)


def diff_paths(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b)) for p in diff_paths(a.get(k), b.get(k), f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in diff_paths(x, y, f"{path}[{i}]")]
    return [] if a == b else [path or "."]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _pdf(objects: list[bytes], info: int | None = None) -> bytes:
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref, size = len(out), len(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    info_ref = f" /Info {info} 0 R" if info else ""
    out += f"trailer\n<< /Size {size} /Root 1 0 R{info_ref} >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


# A one-page PDF whose metadata claims it was created in mid-2030.
FUTURE_DATED_PDF = _pdf([
    b"<< /Type /Catalog /Pages 2 0 R >>",
    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>",
    b"<< /CreationDate (D:20300615120000Z) /Producer (Test) >>",
], info=4)
BEFORE_2030 = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
AFTER_2030 = datetime(2031, 1, 1, tzinfo=timezone.utc)


def _photo_pdf() -> bytes:
    buf = io.BytesIO()
    Image.open(BASELINE / "png" / "ai-generated.png").convert("RGB").save(buf, format="PDF", resolution=72)
    return buf.getvalue()


def _generator_png() -> bytes:
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "portrait, Steps: 30, Sampler: Euler a")
    buf = io.BytesIO()
    Image.open(BASELINE / "jpg" / "real-photo.jpeg").save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


INPUTS = {
    "real-photo.jpeg": lambda: (BASELINE / "jpg" / "real-photo.jpeg").read_bytes(),
    "ai-generated.png": lambda: (BASELINE / "png" / "ai-generated.png").read_bytes(),
    "ai-retouched.png": lambda: (BASELINE / "png" / "ai-retouched.png").read_bytes(),
    "fireplace-generated.png": lambda: (BASELINE / "png" / "fireplace-generated.png").read_bytes(),
    "baseline_sample.pdf": lambda: (BASELINE / "pdf" / "baseline_sample.pdf").read_bytes(),
    "anomaly_metadata_sample.pdf": lambda: (BASELINE / "pdf" / "anomaly_metadata_sample.pdf").read_bytes(),
    "photo.pdf": _photo_pdf,
    "generator-metadata.png": _generator_png,
}


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


@pytest.fixture(scope="module")
def files() -> dict[str, bytes]:
    return {name: make() for name, make in INPUTS.items()}


def _scan(client, name: str, data: bytes, **form):
    resp = client.post("/api/v1/scans", files={"file": (name, data)}, data=form or None)
    return resp.status_code, resp.json()


@pytest.fixture(scope="module")
def reference(client, files) -> dict[str, str]:
    """One sequential scan per input: what every other run must reproduce."""
    out = {}
    for name, data in files.items():
        status, body = _scan(client, name, data)
        assert status == 200, body
        out[name] = canonical(body)
    return out


# --------------------------------------------------------------------------- #
# Repeated scans
# --------------------------------------------------------------------------- #
class TestRepeatedScans:
    @pytest.mark.parametrize("name", list(INPUTS))
    def test_same_input_same_output(self, client, files, reference, name):
        bodies = [_scan(client, name, files[name])[1] for _ in range(2)]
        for body in bodies:
            assert canonical(body) == reference[name], diff_paths(json.loads(reference[name]), comparable(body))

    @pytest.mark.parametrize("name", ["real-photo.jpeg", "photo.pdf"])
    def test_summary_only_is_idempotent_too(self, client, files, name):
        first, second = (_scan(client, name, files[name], details="false")[1] for _ in range(2))
        assert canonical(first) == canonical(second)

    def test_exempt_fields_do_vary(self, client, files):
        """Guards the comparison itself: the exemptions are real and present."""
        a = _scan(client, "real-photo.jpeg", files["real-photo.jpeg"])[1]
        b = _scan(client, "real-photo.jpeg", files["real-photo.jpeg"])[1]
        assert a["summary"]["scan_id"] != b["summary"]["scan_id"]
        for body in (a, b):
            for path in EXEMPT:
                node = body
                for key in path:
                    node = node[key]
                assert node is not None

    def test_only_exempt_fields_differ(self, client, files):
        a = _scan(client, "photo.pdf", files["photo.pdf"])[1]
        b = _scan(client, "photo.pdf", files["photo.pdf"])[1]
        allowed = {"." + ".".join(p) for p in EXEMPT}
        assert set(diff_paths(a, b)) <= allowed

    def test_filename_is_part_of_the_input(self, client, files):
        # Same bytes, different name: only the echoed filename changes.
        a = _scan(client, "one.jpeg", files["real-photo.jpeg"])[1]
        b = _scan(client, "two.jpeg", files["real-photo.jpeg"])[1]
        assert diff_paths(comparable(a), comparable(b)) == [".details.file.filename"]


# --------------------------------------------------------------------------- #
# Errors are deterministic too
# --------------------------------------------------------------------------- #
def _post(client, **kwargs):
    resp = client.post("/api/v1/scans", **kwargs)
    return resp.status_code, resp.json()


ERROR_REQUESTS = {
    "missing_input": lambda: {},
    "unknown_field": lambda: {"files": {"file": ("a.jpg", b"x")}, "data": {"detail": "false"}},
    "unsupported_media_type": lambda: {"content": b"{}", "headers": {"content-type": "application/json"}},
    "invalid_url": lambda: {"data": {"url": "ftp://example.com/a.png"}},
    "url_not_allowed": lambda: {"data": {"url": "http://169.254.169.254/latest/meta-data/"}},
    "unresolvable_host": lambda: {"data": {"url": "http://does-not-exist.invalid/a.png"}},
    "unsupported_format": lambda: {"files": {"file": ("a.gif", b"GIF89a" + b"\0" * 32)}},
    "corrupt_png": lambda: {"files": {"file": ("a.png", b"\x89PNG\r\n\x1a\ngarbage")}},
    "encrypted_pdf": lambda: {"files": {"file": ("e.pdf", (FIXTURES / "encrypted.pdf").read_bytes())}},
}


class TestErrors:
    @pytest.mark.parametrize("case", list(ERROR_REQUESTS))
    def test_same_bad_input_same_error(self, client, case):
        (s1, b1), (s2, b2) = (_post(client, **ERROR_REQUESTS[case]()) for _ in range(2))
        assert s1 == s2 and s1 >= 400
        assert canonical(b1) == canonical(b2)

    def test_connection_failure_message_is_stable(self, client, monkeypatch):
        monkeypatch.setattr(validation, "ALLOW_PRIVATE_URLS", True)
        (s1, b1), (s2, b2) = (_post(client, data={"url": "http://127.0.0.1:1/a.png"}) for _ in range(2))
        assert s1 == s2 == 502
        assert canonical(b1) == canonical(b2)
        assert b1["summary"]["error"]["message"] == "could not connect to '127.0.0.1'"

    def test_analysis_failure_message_is_stable(self, client, files, monkeypatch):
        def boom(_):
            raise RuntimeError(f"internal state {object()!r}")  # an address, different every call
        monkeypatch.setattr(main, "_run_image_analysis", boom)
        (s1, b1), (s2, b2) = (_scan(client, "real-photo.jpeg", files["real-photo.jpeg"]) for _ in range(2))
        assert s1 == s2 == 500
        assert canonical(b1) == canonical(b2)


class TestFetchErrorMessages:
    """requests/urllib3 exception text embeds object addresses; the API must
    not pass it on."""

    @staticmethod
    def _response(status, reason):
        resp = requests.Response()
        resp.status_code, resp.reason = status, reason
        return resp

    @pytest.mark.parametrize("exc, status, code, message", [
        (requests.ConnectTimeout("pool <object at 0x7f1>"), 504, "url_fetch_timeout",
         "timed out connecting to or reading from 'example.com'"),
        (requests.ReadTimeout("read <object at 0x7f2>"), 504, "url_fetch_timeout",
         "timed out connecting to or reading from 'example.com'"),
        (requests.exceptions.SSLError("<object at 0x7f3>"), 502, "url_fetch_failed",
         "TLS/SSL error connecting to 'example.com'"),
        (requests.ConnectionError("<urllib3 object at 0x7f4>"), 502, "url_fetch_failed",
         "could not connect to 'example.com'"),
        (requests.exceptions.ChunkedEncodingError("<object at 0x7f5>"), 502, "url_fetch_failed",
         "could not fetch url (ChunkedEncodingError)"),
    ])
    def test_mapping(self, exc, status, code, message):
        err = scan_api._fetch_error(exc, "https://example.com/a.png")
        assert (err.status_code, err.code, err.message) == (status, code, message)

    def test_http_status(self):
        exc = requests.HTTPError("404 for <object at 0x7f6>", response=self._response(404, "Not Found"))
        err = scan_api._fetch_error(exc, "https://example.com/a.png")
        assert (err.status_code, err.message) == (502, "'example.com' responded with HTTP 404 Not Found")


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #
class TestConcurrency:
    def test_concurrent_scans_match_sequential(self, client, files, reference):
        names = ["ai-generated.png", "photo.pdf", "real-photo.jpeg", "baseline_sample.pdf"] * 4
        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda n: (n, _scan(client, n, files[n])), names))
        for name, (status, body) in results:
            assert status == 200, body
            assert canonical(body) == reference[name], (name, diff_paths(json.loads(reference[name]), comparable(body)))

    def test_concurrent_face_detection_matches_single_threaded(self):
        """A shared cv2.CascadeClassifier shifted face boxes (or failed inside
        OpenCV) under concurrent use; each thread now has its own."""
        gray = cv2.imread(str(BASELINE / "png" / "ai-generated.png"), cv2.IMREAD_GRAYSCALE)
        images = [gray, cv2.resize(gray, None, fx=0.8, fy=0.8), cv2.resize(gray, None, fx=1.2, fy=1.2),
                  np.ascontiguousarray(gray[:, ::-1])]
        expected = [gradients.detect_faces(img) for img in images]
        assert all(expected), "the portrait must have a detectable face for this test to mean anything"
        jobs = [i for _ in range(3) for i in range(len(images)) for _ in range(4)]
        with cf.ThreadPoolExecutor(max_workers=16) as pool:
            got = list(pool.map(lambda i: (i, gradients.detect_faces(images[i])), jobs))
        assert all(result == expected[i] for i, result in got)

    def test_cascades_are_per_thread(self):
        mine = gradients._load_cascades()
        assert gradients._load_cascades() is mine
        with cf.ThreadPoolExecutor(max_workers=1) as pool:
            theirs = pool.submit(gradients._load_cascades).result()
        assert theirs is not mine and theirs[0] is not mine[0]


class TestPdfiumGuard:
    """PDFium is not thread-safe: every document must be opened (and so
    used and closed) while the process-wide guard is held."""

    def test_every_pdf_open_holds_the_guard(self, client, files, monkeypatch):
        real = pypdfium2.PdfDocument
        opens = []

        def checked(*args, **kwargs):
            opens.append(pdfium_guard.PDFIUM.held())
            return real(*args, **kwargs)

        monkeypatch.setattr(pypdfium2, "PdfDocument", checked)
        for name in ("photo.pdf", "baseline_sample.pdf", "anomaly_metadata_sample.pdf"):
            assert _scan(client, name, files[name])[0] == 200
        assert _post(client, files={"file": ("e.pdf", (FIXTURES / "encrypted.pdf").read_bytes())})[0] == 422
        # The legacy endpoint shares the engine and the guard.
        legacy = client.post("/1.0/check.json", files={"media": ("p.pdf", files["photo.pdf"])}, data={"models": "genai"})
        assert legacy.status_code == 200
        assert len(opens) >= 10
        assert all(opens), f"{opens.count(False)} of {len(opens)} PDFs opened without the guard"

    def test_guard_is_reentrant_and_per_thread(self):
        guard = pdfium_guard.PDFIUM
        assert not guard.held()
        with guard:
            with guard:
                assert guard.held()
            assert guard.held()
            with cf.ThreadPoolExecutor(max_workers=1) as pool:
                assert pool.submit(guard.held).result() is False
        assert not guard.held()


# --------------------------------------------------------------------------- #
# Across processes
# --------------------------------------------------------------------------- #
_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from fastapi.testclient import TestClient
from app import main
client = TestClient(main.app)
out = {}
for path in sys.argv[2:]:
    with open(path, "rb") as fh:
        out[path] = client.post("/api/v1/scans", files={"file": (path.rsplit("/", 1)[-1], fh.read())}).json()
print(json.dumps(out))
"""


class TestAcrossProcesses:
    @pytest.mark.parametrize("seed", ["1", "4242"])
    def test_fresh_process_with_other_hash_seed_agrees(self, reference, seed):
        paths = {
            "real-photo.jpeg": BASELINE / "jpg" / "real-photo.jpeg",
            "ai-generated.png": BASELINE / "png" / "ai-generated.png",
            "anomaly_metadata_sample.pdf": BASELINE / "pdf" / "anomaly_metadata_sample.pdf",
        }
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD, str(MAIN_DIR), *map(str, paths.values())],
            capture_output=True, text=True, env=env, timeout=300, check=True,
        )
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        for name, path in paths.items():
            assert canonical(out[str(path)]) == reference[name], name


# --------------------------------------------------------------------------- #
# The clock
# --------------------------------------------------------------------------- #
class TestClock:
    """The clock is read once per scan. It is the response timestamp and the
    reference for the one time-relative check: PDF metadata dated more than a
    day in the future. So the same PDF gets the same result for the same
    timestamp, and only that check can change as time passes."""

    def _scan_at(self, client, monkeypatch, moment, **form):
        monkeypatch.setattr(scan_api, "_clock", lambda: moment)
        status, body = _scan(client, "future.pdf", FUTURE_DATED_PDF, **form)
        assert status == 200, body
        return body

    def test_timestamp_is_the_single_clock_reading(self, client, monkeypatch):
        body = self._scan_at(client, monkeypatch, BEFORE_2030)
        assert body["summary"]["timestamp"] == "2026-11-05T12:00:00.000Z"

    def test_error_timestamp_uses_the_same_clock(self, client, monkeypatch):
        monkeypatch.setattr(scan_api, "_clock", lambda: BEFORE_2030)
        status, body = _post(client)
        assert status == 400 and body["summary"]["timestamp"] == "2026-11-05T12:00:00.000Z"

    def test_same_reference_time_same_result(self, client, monkeypatch):
        first = self._scan_at(client, monkeypatch, BEFORE_2030)
        second = self._scan_at(client, monkeypatch, BEFORE_2030)
        assert canonical(first) == canonical(second)
        codes = [f["code"] for f in first["details"]["document"]["document_anomalies"]["flags"]]
        assert "future_metadata_date" in codes

    def test_only_the_future_date_check_depends_on_time(self, client, monkeypatch):
        before = self._scan_at(client, monkeypatch, BEFORE_2030)
        after = self._scan_at(client, monkeypatch, AFTER_2030)
        codes = [f["code"] for f in after["details"]["document"]["document_anomalies"]["flags"]]
        assert "future_metadata_date" not in codes
        changed = diff_paths(comparable(before), comparable(after))
        assert changed, "the future-dated PDF should read differently once 2030 has passed"
        assert all(
            p.startswith(".details.document.document_anomalies") or p == ".summary.explanation" for p in changed
        ), changed

    def test_images_do_not_depend_on_time(self, client, files, monkeypatch):
        monkeypatch.setattr(scan_api, "_clock", lambda: BEFORE_2030)
        a = _scan(client, "real-photo.jpeg", files["real-photo.jpeg"])[1]
        monkeypatch.setattr(scan_api, "_clock", lambda: AFTER_2030)
        b = _scan(client, "real-photo.jpeg", files["real-photo.jpeg"])[1]
        assert canonical(a) == canonical(b)
