"""
Tests for the JSON scan API (``POST /api/v1/scans``, ``app/scan_api.py``).

End-to-end through FastAPI's TestClient on the baseline files, so the real
engine runs. Covers: the response contract, parity with ``/1.0/check.json``
(the API must score a file exactly as the UI's endpoint does), URL input via
a real local HTTP server, every error path, and the mapping rules
(classification bands in sync with the UI, image confidence tiers).
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import http.server
import io
import re
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin

from app import main, scan_api
from app import scan_validation as validation

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "baseline"
HUMAN = BASELINE / "jpg" / "human.jpeg"
AI = BASELINE / "png" / "ai-generated.png"
FIREPLACE = BASELINE / "png" / "fireplace-generated.png"
TEXT_PDF = BASELINE / "pdf" / "baseline_sample.pdf"
ENCRYPTED_PDF = Path(__file__).resolve().parent / "fixtures" / "encrypted.pdf"

SIGNALS = ("frequency", "noise", "gradients", "lighting", "metadata")

pytestmark = pytest.mark.skipif(not BASELINE.exists(), reason="baseline files not present")


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


def _scan(client, path: Path | None = None, data: bytes | None = None, name: str = "upload.bin", **form):
    files = None
    if path is not None:
        files = {"file": (path.name, path.read_bytes())}
    elif data is not None:
        files = {"file": (name, data)}
    return client.post("/api/v1/scans", files=files, data=form or None)


def _legacy_score(client, content: bytes, name: str):
    resp = client.post(
        "/1.0/check.json", files={"media": (name, content)}, data={"models": "genai"}
    )
    assert resp.status_code == 200
    return resp.json()["type"]["ai_generated"]


@pytest.fixture(scope="module")
def human(client):
    resp = _scan(client, HUMAN)
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.fixture(scope="module")
def ai(client):
    resp = _scan(client, AI)
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.fixture(scope="module")
def photo_pdf_bytes() -> bytes:
    """A single-page PDF whose page is the generated portrait (raster path)."""
    buf = io.BytesIO()
    Image.open(AI).convert("RGB").save(buf, format="PDF", resolution=72)
    return buf.getvalue()


@pytest.fixture(scope="module")
def photo_pdf(client, photo_pdf_bytes):
    resp = _scan(client, data=photo_pdf_bytes, name="photo.pdf")
    assert resp.status_code == 200, resp.text
    return resp.json()


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves baseline/, plus redirect routes:
    /redirect/<n>/<path>   n hops of 302, then /<path>
    /redirect-to?<url>     one 302 to <url>
    /not-an-image          a small HTML page"""

    def log_message(self, *args, **kwargs):
        pass

    def do_GET(self):
        if self.path.startswith("/redirect/"):
            _, _, hops, rest = self.path.split("/", 3)
            n = int(hops)
            target = f"/redirect/{n - 1}/{rest}" if n > 1 else f"/{rest}"
            return self._redirect(target)
        if self.path.startswith("/redirect-to?"):
            return self._redirect(self.path.split("?", 1)[1])
        if self.path == "/not-an-image":
            body = b"<html><body>not an image</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "image/png")  # lies; the bytes decide
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return None
        return super().do_GET()

    def _redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture(scope="module")
def file_server():
    """Serves baseline/ over HTTP on a free local port."""
    handler = functools.partial(_Handler, directory=str(BASELINE))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def allow_private_urls(monkeypatch):
    """The test file server is on loopback, which the API refuses by default."""
    monkeypatch.setattr(validation, "ALLOW_PRIVATE_URLS", True)


def assert_error(resp, status, code):
    """Every failure: the documented error shape, never FastAPI's default."""
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert list(body) == ["summary"], body
    s = body["summary"]
    assert list(s)[:3] == ["http_status", "status", "error"]
    assert s["http_status"] == status == resp.status_code
    assert s["status"] == "failed"
    assert s["error"]["code"] == code, s["error"]
    assert s["error"]["message"]
    assert s["scan_id"].startswith("scn_")
    assert s["engine_version"] == main.app.version
    return s["error"]


# --------------------------------------------------------------------------- #
# Response contract
# --------------------------------------------------------------------------- #
class TestContract:
    def test_two_sections_summary_first(self, human):
        assert list(human) == ["summary", "details"]

    def test_summary_reading_order(self, human):
        # The critical fields lead: HTTP status, thresholds, final score.
        assert list(human["summary"]) == [
            "http_status", "status", "thresholds", "final_score", "classification",
            "label", "confidence", "explanation", "scan_id", "timestamp", "engine_version",
        ]

    def test_summary_values(self, human):
        s = human["summary"]
        assert s["http_status"] == 200
        assert s["status"] == "completed"
        assert s["thresholds"] == {"likely_real_below": scan_api.REAL_MAX, "likely_ai_above": scan_api.AI_MIN}
        assert re.fullmatch(r"scn_[0-9a-f]{32}", s["scan_id"])
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", s["timestamp"])
        assert s["engine_version"] == main.app.version
        assert s["explanation"]

    def test_details_sections(self, human, photo_pdf):
        # `document` is PDF-only: absent (not null) for an image.
        assert list(human["details"]) == ["duration_ms", "engine", "file", "evidence", "scoring"]
        assert list(photo_pdf["details"]) == ["duration_ms", "engine", "file", "evidence", "scoring", "document"]
        assert human["details"]["duration_ms"] >= 0

    def test_summary_is_not_repeated_in_details(self, human):
        assert "thresholds" not in human["details"]["scoring"]
        assert "final_score" not in human["details"]["scoring"]

    def test_engine_info(self, human):
        assert human["details"]["engine"]["name"] == scan_api.ENGINE_NAME
        assert human["details"]["engine"]["version"] == main.app.version

    def test_file_metadata(self, human):
        content = HUMAN.read_bytes()
        f = human["details"]["file"]
        assert f["source"] == "upload"
        assert f["filename"] == "human.jpeg"
        assert f["url"] is None
        assert f["size_bytes"] == len(content)
        assert f["sha256"] == hashlib.sha256(content).hexdigest()
        assert f["format"] == "jpeg" and f["mime_type"] == "image/jpeg"
        with Image.open(HUMAN) as im:
            assert (f["width"], f["height"]) == im.size

    def test_all_five_signals_with_measurements_and_explanations(self, ai):
        assert set(ai["details"]["evidence"]) == set(SIGNALS)
        for name in SIGNALS:
            ev = ai["details"]["evidence"][name]
            assert ev["signal"] == name
            assert ev["label"] == scan_api.SIGNAL_LABELS[name]
            assert isinstance(ev["measurements"], dict)
            assert ev["explanation"]
            for envelope_key in ("score", "confidence", "applicable", "note"):
                assert envelope_key not in ev["measurements"]
        assert "noise_sigma_median" in ai["details"]["evidence"]["noise"]["measurements"]
        assert "spectral_slope" in ai["details"]["evidence"]["frequency"]["measurements"]
        assert "skin" in ai["details"]["evidence"]["gradients"]["measurements"]

    def test_scan_ids_are_unique(self, client):
        ids = {_scan(client, HUMAN).json()["summary"]["scan_id"] for _ in range(3)}
        assert len(ids) == 3

    def test_openapi_documents_the_endpoint(self, client):
        spec = client.get("/openapi.json").json()
        assert "/api/v1/scans" in spec["paths"]
        assert "ScanResponse" in spec["components"]["schemas"]


# --------------------------------------------------------------------------- #
# Same scan as the UI endpoint
# --------------------------------------------------------------------------- #
class TestParity:
    @pytest.mark.parametrize("path", [HUMAN, AI, FIREPLACE], ids=lambda p: p.name)
    def test_image_score_equals_check_json(self, client, path):
        api = _scan(client, path).json()
        assert api["summary"]["final_score"] == _legacy_score(client, path.read_bytes(), path.name)

    def test_pdf_score_equals_check_json(self, client, photo_pdf, photo_pdf_bytes):
        assert photo_pdf["summary"]["final_score"] == _legacy_score(client, photo_pdf_bytes, "photo.pdf")

    def test_signal_weights_match_engine_effective_weights(self, client, ai):
        raw = main._run_image_analysis(AI.read_bytes())
        for name, weight in raw["scoring"]["effective_weights"].items():
            assert ai["details"]["evidence"][name]["weight"] == pytest.approx(weight)


# --------------------------------------------------------------------------- #
# Baseline expectations
# --------------------------------------------------------------------------- #
class TestBaselines:
    def test_human_photo(self, human):
        r = human["summary"]
        assert r["classification"] == "likely_real"
        assert r["label"] == "Likely a true doc"
        # Faceless JPEG: frequency + lighting measured -> low-medium (README).
        assert r["confidence"] == "low-medium"
        assert human["details"]["evidence"]["noise"]["applicable"] is False
        assert human["details"]["evidence"]["noise"]["score"] is None
        assert human["details"]["evidence"]["noise"]["verdict"] == "not_applicable"
        assert human["details"]["evidence"]["gradients"]["used_in_score"] is False

    def test_generated_portrait(self, ai):
        r = ai["summary"]
        assert r["classification"] == "likely_ai_generated"
        assert r["confidence"] == "medium-high"
        assert ai["details"]["evidence"]["gradients"]["verdict"] == "likely_ai"
        used = [n for n in SIGNALS[:4] if ai["details"]["evidence"][n]["used_in_score"]]
        assert sum(ai["details"]["evidence"][n]["weight"] for n in used) == pytest.approx(1.0, abs=1e-3)

    def test_text_only_pdf_is_not_scorable(self, client):
        body = _scan(client, TEXT_PDF).json()
        assert body["details"]["file"]["format"] == "pdf" and body["details"]["file"]["page_count"] == 1
        assert body["summary"]["classification"] == "insufficient_evidence"
        assert body["summary"]["final_score"] is None
        assert body["summary"]["confidence"] == "insufficient"
        assert all(not body["details"]["evidence"][n]["used_in_score"] for n in SIGNALS[:4])
        doc = body["details"]["document"]
        assert doc["engine_classification"] == "insufficient_visual_evidence"
        assert doc["pages"][0]["classification"] == "insufficient_evidence"
        assert "classification" in doc["document_anomalies"]

    def test_raster_pdf(self, photo_pdf):
        doc = photo_pdf["details"]["document"]
        assert doc["worst_page"] == 1
        assert doc["evidence_source"] in {"page_render", "embedded_image"}
        page = doc["pages"][0]
        assert page["score"] == photo_pdf["summary"]["final_score"]
        assert page["embedded_images"] and page["embedded_images"][0]["usable"]
        assert photo_pdf["details"]["evidence"]["gradients"]["used_in_score"] is True
        assert photo_pdf["summary"]["confidence"] != "insufficient"


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
class TestMetadata:
    def test_generator_signature_overrides(self, client):
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "portrait, Steps: 30, Sampler: Euler a")
        buf = io.BytesIO()
        Image.open(HUMAN).save(buf, format="PNG", pnginfo=info)
        body = _scan(client, data=buf.getvalue(), name="sd.png").json()
        meta = body["details"]["evidence"]["metadata"]
        assert meta["effect"] == "override" and meta["verdict"] == "likely_ai"
        assert meta["used_in_score"] is True and meta["weight"] is None
        assert meta["measurements"]["signature_hits"]
        assert body["summary"]["classification"] == "likely_ai_generated"
        assert body["summary"]["confidence"] == "high"
        assert body["details"]["scoring"]["metadata_effect"] == "override"

    def test_no_metadata_is_neutral(self, human):
        meta = human["details"]["evidence"]["metadata"]
        assert meta["effect"] == "neutral" and meta["verdict"] == "neutral"
        assert meta["used_in_score"] is False


# --------------------------------------------------------------------------- #
# URL input
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("allow_private_urls")
class TestUrlInput:
    def test_scan_by_url(self, client, file_server):
        url = f"{file_server}/jpg/human.jpeg"
        body = _scan(client, url=url).json()
        assert body["details"]["file"]["source"] == "url"
        assert body["details"]["file"]["url"] == url
        assert body["details"]["file"]["filename"] == "human.jpeg"
        assert body["details"]["file"]["sha256"] == hashlib.sha256(HUMAN.read_bytes()).hexdigest()
        assert body["summary"]["final_score"] == _legacy_score(client, HUMAN.read_bytes(), "human.jpeg")

    def test_url_encoded_body(self, client, file_server):
        resp = client.post("/api/v1/scans", data={"url": f"{file_server}/jpg/human.jpeg", "details": "false"})
        assert resp.status_code == 200, resp.text
        assert list(resp.json()) == ["summary"]

    def test_http_error(self, client, file_server):
        assert_error(_scan(client, url=f"{file_server}/nope.png"), 502, "url_fetch_failed")

    def test_connection_refused(self, client):
        assert_error(_scan(client, url="http://127.0.0.1:1/x.png"), 502, "url_fetch_failed")

    @pytest.mark.parametrize("url", ["ftp://example.com/a.png", "file:///etc/passwd", "not a url", "http://",
                                     "http://u:p@example.com/a.png", "http://example.com:99999/a.png",
                                     "http://example.com/a b.png"])
    def test_invalid_urls(self, client, url):
        assert_error(_scan(client, url=url), 400, "invalid_url")

    def test_url_too_long(self, client):
        url = "http://example.com/" + "a" * validation.MAX_URL_LENGTH
        assert_error(_scan(client, url=url), 400, "invalid_url")

    def test_url_download_size_limit(self, client, file_server, monkeypatch):
        monkeypatch.setattr(validation, "MAX_FILE_BYTES", 1000)
        assert_error(_scan(client, url=f"{file_server}/jpg/human.jpeg"), 413, "file_too_large")

    def test_downloaded_file_is_validated_like_an_upload(self, client, file_server):
        # Same content checks as an upload: here, not a PNG/JPEG/PDF.
        assert_error(_scan(client, url=f"{file_server}/not-an-image"), 415, "unsupported_format")

    def test_redirects_are_followed(self, client, file_server):
        resp = _scan(client, url=f"{file_server}/redirect/3/jpg/human.jpeg", details="false")
        assert resp.status_code == 200, resp.text

    def test_too_many_redirects(self, client, file_server):
        err = assert_error(
            _scan(client, url=f"{file_server}/redirect/{scan_api.MAX_REDIRECTS + 1}/jpg/human.jpeg"),
            502, "url_fetch_failed",
        )
        assert "redirects" in err["message"]

    def test_redirect_target_is_validated(self, client, file_server):
        assert_error(_scan(client, url=f"{file_server}/redirect-to?ftp://example.com/a.png"), 400, "invalid_url")


class TestUrlHostPolicy:
    """Default policy: URLs may only reach public hosts."""

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:8008/health",
        "http://localhost/a.png",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/a.png",
        "http://[::1]/a.png",
    ])
    def test_non_public_hosts_refused(self, client, url):
        err = assert_error(_scan(client, url=url), 400, "url_not_allowed")
        assert "TRUEDOC_ALLOW_PRIVATE_URLS" in err["message"]

    def test_redirect_to_non_public_host_refused(self, client, file_server, monkeypatch):
        # Treat the test server (127.0.0.1) as public, everything else in
        # loopback as private; the redirect must be re-checked at each hop.
        monkeypatch.setattr(validation, "is_public_address", lambda a: a == "127.0.0.1")
        port = file_server.rsplit(":", 1)[1]
        resp = _scan(client, url=f"{file_server}/redirect-to?http://127.0.0.2:{port}/jpg/human.jpeg")
        assert_error(resp, 400, "url_not_allowed")

    def test_unresolvable_host(self, client):
        assert_error(_scan(client, url="http://does-not-exist.invalid/a.png"), 502, "url_fetch_failed")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class TestErrors:
    @staticmethod
    def _assert_error(resp, status, code):
        return assert_error(resp, status, code)

    @pytest.mark.parametrize("value", ["maybe", "2", "nope"])
    def test_invalid_details_flag(self, client, value):
        self._assert_error(_scan(client, HUMAN, details=value), 400, "invalid_parameter")

    def test_missing_input(self, client):
        self._assert_error(client.post("/api/v1/scans"), 400, "missing_input")

    def test_blank_url_is_missing_input(self, client):
        self._assert_error(_scan(client, url="   "), 400, "missing_input")

    def test_both_inputs(self, client):
        self._assert_error(_scan(client, HUMAN, url="http://example.com/a.png"), 400, "ambiguous_input")

    def test_empty_file(self, client):
        self._assert_error(_scan(client, data=b"", name="empty.png"), 400, "empty_file")

    def test_unsupported_format(self, client):
        self._assert_error(_scan(client, data=b"GIF89a" + b"\0" * 64, name="a.gif"), 415, "unsupported_format")

    def test_format_is_sniffed_not_taken_from_name(self, client):
        body = _scan(client, data=HUMAN.read_bytes(), name="renamed.png").json()
        assert body["details"]["file"]["format"] == "jpeg"

    @pytest.mark.parametrize("data", [b"\xff\xd8\xff" + b"\0" * 64, b"\x89PNG\r\n\x1a\ngarbage"], ids=["jpeg", "png"])
    def test_corrupt_image(self, client, data):
        self._assert_error(_scan(client, data=data, name="bad.img"), 422, "unprocessable_file")

    def test_upload_size_limit(self, client, monkeypatch):
        monkeypatch.setattr(validation, "MAX_FILE_BYTES", 1000)
        self._assert_error(_scan(client, HUMAN), 413, "file_too_large")

    def test_encrypted_pdf(self, client):
        self._assert_error(_scan(client, ENCRYPTED_PDF), 422, "encrypted_pdf")

    def test_image_over_pixel_limit_is_refused_before_decoding(self, client, monkeypatch):
        def must_not_run(_):
            raise AssertionError("engine ran on an over-limit image")
        monkeypatch.setattr(main, "_run_image_analysis", must_not_run)
        monkeypatch.setattr(validation, "MAX_IMAGE_PIXELS", 1000)
        err = self._assert_error(_scan(client, HUMAN), 422, "image_too_large")
        assert "738x1280" in err["message"]

    def test_unexpected_failure_is_500_without_internals(self, client, monkeypatch):
        def boom(_):
            raise RuntimeError("secret internal detail")
        monkeypatch.setattr(main, "_run_image_analysis", boom)
        resp = _scan(client, HUMAN)
        self._assert_error(resp, 500, "analysis_failed")
        assert "secret" not in resp.text


# --------------------------------------------------------------------------- #
# Request validation (HTTP level; the rules themselves are unit-tested in
# tests/test_scan_validation.py)
# --------------------------------------------------------------------------- #
def _multipart(parts: list[tuple[str, bytes, str | None]], boundary: str = "XYZ") -> tuple[bytes, dict]:
    """(name, value, filename-or-None) -> raw multipart body + headers."""
    body = b""
    for name, value, filename in parts:
        disposition = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        body += f"--{boundary}\r\nContent-Disposition: {disposition}\r\n\r\n".encode() + value + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, {"content-type": f"multipart/form-data; boundary={boundary}"}


class TestRequestValidation:
    @pytest.fixture(autouse=True)
    def engine_must_not_run(self, monkeypatch):
        """Invalid requests are refused before any analysis."""
        def fail(_):
            raise AssertionError("engine ran on an invalid request")
        monkeypatch.setattr(main, "_run_image_analysis", fail)
        monkeypatch.setattr(main, "_run_pdf_analysis", fail)

    def _post(self, client, parts, **kw):
        body, headers = _multipart(parts, **kw)
        return client.post("/api/v1/scans", content=body, headers=headers)

    def test_unknown_field(self, client):
        err = assert_error(_scan(client, HUMAN, detail="false"), 400, "unknown_field")
        assert "did you mean 'details'" in err["message"]

    def test_legacy_field_name(self, client):
        resp = client.post("/api/v1/scans", files={"media": ("a.jpg", HUMAN.read_bytes())}, data={"models": "genai"})
        err = assert_error(resp, 400, "unknown_field")
        assert "did you mean 'file'" in err["message"]

    def test_two_files(self, client):
        resp = self._post(client, [("file", b"a", "a.png"), ("file", b"b", "b.png")])
        assert_error(resp, 400, "duplicate_field")

    def test_file_sent_as_text(self, client):
        assert_error(self._post(client, [("file", b"hello", None)]), 400, "invalid_field_type")

    def test_url_sent_as_file(self, client):
        assert_error(self._post(client, [("url", b"http://e.com/a.png", "u.txt")]), 400, "invalid_field_type")

    @pytest.mark.parametrize("content_type, body", [
        ("application/json", b'{"url": "https://example.com/a.png"}'),
        ("text/plain", b"https://example.com/a.png"),
        ("image/png", b"\x89PNG\r\n\x1a\n"),
    ])
    def test_non_form_bodies(self, client, content_type, body):
        resp = client.post("/api/v1/scans", content=body, headers={"content-type": content_type})
        assert_error(resp, 415, "unsupported_media_type")

    def test_no_body(self, client):
        assert_error(client.post("/api/v1/scans"), 400, "missing_input")

    def test_multipart_without_boundary(self, client):
        resp = client.post("/api/v1/scans", content=b"x", headers={"content-type": "multipart/form-data"})
        assert_error(resp, 400, "malformed_body")

    @pytest.mark.parametrize("body", [
        b"this is not multipart",
        b"--XYZ\r\nContent-Disposition: form-data\r\n\r\nno name\r\n--XYZ--\r\n",
    ], ids=["garbage", "part-without-name"])
    def test_malformed_multipart(self, client, body):
        resp = client.post("/api/v1/scans", content=body, headers={"content-type": "multipart/form-data; boundary=XYZ"})
        assert_error(resp, 400, "malformed_body")

    def test_too_many_parts(self, client):
        parts = [(f"f{i}", b"1", None) for i in range(validation.MAX_FORM_FIELDS + 1)]
        assert_error(self._post(client, parts), 400, "malformed_body")

    def test_oversized_text_part(self, client):
        resp = self._post(client, [("url", b"a" * (validation.MAX_FORM_PART_BYTES + 1), None)])
        assert_error(resp, 400, "malformed_body")

    def test_declared_content_length_over_limit(self, client, monkeypatch):
        monkeypatch.setattr(validation, "MAX_FILE_BYTES", 1000)
        body, headers = _multipart([("file", b"x" * (1000 + validation.FORM_OVERHEAD_BYTES), "a.png")])
        assert_error(client.post("/api/v1/scans", content=body, headers=headers), 413, "file_too_large")

    def test_chunked_body_is_cut_off_while_streaming(self, monkeypatch):
        """No Content-Length: the server must stop reading at the limit, not
        read (and spool) the whole body first. TestClient hands the app the
        body in one piece, so this drives the ASGI app directly and counts
        how much of a 4 MB body it actually pulls."""
        monkeypatch.setattr(validation, "MAX_FILE_BYTES", 1000)
        limit = validation.max_request_bytes()
        body, headers = _multipart([("file", b"x" * (4 * 1024 * 1024), "a.png")])
        chunk = 64 * 1024
        pulled = 0
        sent = []

        async def receive():
            nonlocal pulled
            if pulled >= len(body):
                return {"type": "http.disconnect"}
            piece = body[pulled:pulled + chunk]
            pulled += len(piece)
            return {"type": "http.request", "body": piece, "more_body": pulled < len(body)}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": "/api/v1/scans", "raw_path": b"/api/v1/scans",
            "query_string": b"", "root_path": "", "server": ("test", 80), "client": ("test", 1),
            "headers": [(k.encode(), v.encode()) for k, v in headers.items()],  # no content-length
        }
        asyncio.run(main.app(scope, receive, send))

        start = next(m for m in sent if m["type"] == "http.response.start")
        payload = json.loads(b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"))
        assert start["status"] == 413
        assert payload["summary"]["error"]["code"] == "file_too_large"
        assert pulled <= limit + chunk, f"read {pulled} bytes of {len(body)}; limit {limit}"

    def test_url_encoded_body_has_small_limit(self, client):
        body = b"url=http://example.com/" + b"a" * validation.FORM_OVERHEAD_BYTES
        resp = client.post("/api/v1/scans", content=body,
                           headers={"content-type": "application/x-www-form-urlencoded"})
        assert_error(resp, 413, "file_too_large")

    def test_invalid_content_length(self, client):
        body, headers = _multipart([("url", b"https://example.com/a.png", None)])
        # httpx would correct a bad header; build the request by hand.
        request = client.build_request("POST", "/api/v1/scans", content=body, headers=headers)
        request.headers["content-length"] = "abc"
        assert_error(client.send(request), 400, "invalid_request")

    def test_filename_is_sanitised_in_response(self, client, monkeypatch):
        monkeypatch.undo()  # this one runs the engine
        body = _scan(client, data=HUMAN.read_bytes(), name="../../secret/dir/pho\x1bto.jpeg", details="true").json()
        assert body["details"]["file"]["filename"] == "photo.jpeg"

    def test_uploaded_files_are_closed(self, client, monkeypatch):
        from starlette.datastructures import FormData

        closed = []
        original = FormData.close

        async def spy(self):
            closed.append(True)
            await original(self)

        monkeypatch.setattr(FormData, "close", spy)
        assert_error(_scan(client, HUMAN, detail="x"), 400, "unknown_field")  # failure path
        assert closed


class TestOpenApi:
    def test_request_body_is_documented(self, client):
        op = client.get("/openapi.json").json()["paths"]["/api/v1/scans"]["post"]
        multipart = op["requestBody"]["content"]["multipart/form-data"]["schema"]
        assert set(multipart["properties"]) == set(validation.FIELDS)
        assert multipart["additionalProperties"] is False
        assert "application/x-www-form-urlencoded" in op["requestBody"]["content"]
        assert {"400", "413", "415", "422"} <= set(op["responses"])


# --------------------------------------------------------------------------- #
# details flag
# --------------------------------------------------------------------------- #
class TestDetailsFlag:
    @pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off"])
    def test_false_returns_summary_only(self, client, value, human):
        body = _scan(client, HUMAN, details=value).json()
        assert list(body) == ["summary"]
        assert body["summary"]["final_score"] == human["summary"]["final_score"]
        assert body["summary"]["classification"] == human["summary"]["classification"]

    @pytest.mark.parametrize("value", ["true", "1", "yes", "on", ""])
    def test_true_or_blank_includes_details(self, client, value):
        body = _scan(client, HUMAN, details=value).json()
        assert list(body) == ["summary", "details"]

    def test_default_includes_details(self, human):
        assert "details" in human

    def test_with_url_input(self, client, file_server, allow_private_urls):
        body = _scan(client, url=f"{file_server}/jpg/human.jpeg", details="false").json()
        assert list(body) == ["summary"]

    def test_pdf_summary_only(self, client, photo_pdf_bytes, photo_pdf):
        body = _scan(client, data=photo_pdf_bytes, name="photo.pdf", details="false").json()
        assert list(body) == ["summary"]
        assert body["summary"]["final_score"] == photo_pdf["summary"]["final_score"]


# --------------------------------------------------------------------------- #
# Mapping rules
# --------------------------------------------------------------------------- #
class TestMapping:
    @pytest.mark.parametrize("score, expected", [
        (0.0, "likely_real"), (0.3499, "likely_real"), (0.35, "inconclusive"),
        (0.5, "inconclusive"), (0.65, "inconclusive"), (0.6501, "likely_ai_generated"),
        (1.0, "likely_ai_generated"), (None, "insufficient_evidence"),
    ])
    def test_classification_bands(self, score, expected):
        assert scan_api.classify_score(score) == expected

    def test_bands_match_web_ui(self):
        ts = Path(__file__).resolve().parents[1] / "web" / "lib" / "verdict.ts"
        if not ts.exists():
            pytest.skip("web UI not present")
        src = ts.read_text()
        real = re.search(r"REAL_MAX:\s*([0-9.]+)", src)
        ai = re.search(r"AI_MIN:\s*([0-9.]+)", src)
        assert real and ai, "verdict.ts VERDICT_THRESHOLDS no longer has the expected shape"
        assert float(real.group(1)) == scan_api.REAL_MAX
        assert float(ai.group(1)) == scan_api.AI_MIN

    @staticmethod
    def _sig(score=0.5, confidence="medium", applicable=True):
        return {"score": score, "confidence": confidence, "applicable": applicable}

    @pytest.mark.parametrize("present, expected", [
        # mass = sum(base weight x confidence factor) over applicable signals
        ({"noise": "medium", "frequency": "medium"}, "medium-high"),   # 0.65
        ({"noise": "medium"}, "medium"),                               # 0.40
        ({"frequency": "medium", "lighting": "low"}, "low-medium"),    # 0.30
        ({"frequency": "low-medium"}, "low"),                          # 0.175
    ])
    def test_image_confidence_tiers(self, present, expected):
        signals = {
            n: self._sig(confidence=present[n]) if n in present else self._sig(applicable=False)
            for n in ("frequency", "noise", "gradients", "lighting")
        }
        meta = {"score": 0.5, "confidence": "low"}
        assert scan_api.image_confidence(signals, {"insufficient_evidence": False}, meta) == expected

    def test_image_confidence_insufficient_and_high(self):
        signals = {n: self._sig(applicable=False) for n in ("frequency", "noise", "gradients", "lighting")}
        neutral = {"score": 0.5, "confidence": "low"}
        assert scan_api.image_confidence(signals, {"insufficient_evidence": True}, neutral) == "insufficient"
        hit = {"score": 0.97, "confidence": "high"}
        assert scan_api.image_confidence(signals, {"insufficient_evidence": True}, hit) == "high"
