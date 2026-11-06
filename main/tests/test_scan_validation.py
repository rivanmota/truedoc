"""
Unit tests for the scan API's input validation (``app/scan_validation.py``).

Pure function tests: no HTTP, no forensic engine, no network (DNS is an
injected resolver). Inputs are built in memory. The end-to-end behaviour of
the same rules through the endpoint is in ``tests/test_scan_api.py``.
"""
from __future__ import annotations

import asyncio
import io
import struct
import zlib
from pathlib import Path

import pytest
from PIL import Image

from app import scan_validation as v
from app.scan_validation import ScanError

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _raises(fn, *args, status: int, code: str, **kwargs) -> ScanError:
    with pytest.raises(ScanError) as info:
        fn(*args, **kwargs)
    assert (info.value.status_code, info.value.code) == (status, code), info.value.message
    assert info.value.message
    return info.value


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
class FakeUpload:
    """Duck-typed stand-in for starlette's UploadFile."""

    def __init__(self, filename="a.png"):
        self.filename = filename

    async def read(self, n=-1):
        return b""


def _image_bytes(fmt="PNG", size=(32, 24), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, (120, 60, 30) if mode == "RGB" else 0).save(buf, format=fmt)
    return buf.getvalue()


def _png_header_only(width: int, height: int) -> bytes:
    """A valid PNG that declares width x height but holds almost no data:
    the shape of a decompression bomb, cheap to build at any size."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\0" * 64)) + chunk(b"IEND", b"")


def _pdf(objects: list[bytes]) -> bytes:
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


ONE_PAGE_PDF = _pdf([
    b"<< /Type /Catalog /Pages 2 0 R >>",
    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>",
])
ZERO_PAGE_PDF = _pdf([b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [] /Count 0 >>"])


# --------------------------------------------------------------------------- #
# 1. Request
# --------------------------------------------------------------------------- #
class TestContentLength:
    def test_absent_is_allowed(self):
        assert v.validate_content_length(None) is None

    @pytest.mark.parametrize("header, expected", [("0", 0), ("1234", 1234), (" 42 ", 42)])
    def test_valid(self, header, expected):
        assert v.validate_content_length(header) == expected

    @pytest.mark.parametrize("header", ["abc", "-1", "1.5", "", "12 34"])
    def test_not_a_length(self, header):
        _raises(v.validate_content_length, header, status=400, code="invalid_request")

    def test_at_limit_is_allowed(self):
        assert v.validate_content_length(str(v.max_request_bytes())) == v.max_request_bytes()

    def test_over_limit(self):
        _raises(v.validate_content_length, str(v.max_request_bytes() + 1), status=413, code="file_too_large")

    def test_explicit_limit(self):
        assert v.validate_content_length("100", limit=100) == 100
        _raises(v.validate_content_length, "101", limit=100, status=413, code="file_too_large")

    def test_limit_follows_max_file_bytes(self, monkeypatch):
        monkeypatch.setattr(v, "MAX_FILE_BYTES", 1000)
        assert v.max_request_bytes() == 1000 + v.FORM_OVERHEAD_BYTES
        _raises(v.validate_content_length, str(1001 + v.FORM_OVERHEAD_BYTES), status=413, code="file_too_large")


class TestRequestBodyLimit:
    def test_multipart_gets_file_allowance(self):
        assert v.request_body_limit("multipart/form-data; boundary=x") == v.max_request_bytes()

    @pytest.mark.parametrize("header", ["application/x-www-form-urlencoded", "Application/X-WWW-Form-Urlencoded; charset=utf-8"])
    def test_urlencoded_gets_small_allowance(self, header):
        # It can only carry url + details, never a file.
        assert v.request_body_limit(header) == v.FORM_OVERHEAD_BYTES

    @pytest.mark.parametrize("header", [None, "", "application/json"])
    def test_other_types_fall_back_to_multipart_limit(self, header):
        # Refused by validate_content_type anyway; the limit only bounds reading.
        assert v.request_body_limit(header) == v.max_request_bytes()


class TestContentType:
    @pytest.mark.parametrize("header", [
        "multipart/form-data; boundary=abc",
        "Multipart/Form-Data; Boundary=abc",
        "application/x-www-form-urlencoded",
        "application/x-www-form-urlencoded; charset=utf-8",
    ])
    def test_form_types_accepted(self, header):
        v.validate_content_type(header, 100)

    @pytest.mark.parametrize("header", ["application/json", "text/plain", "image/png", "application/pdf"])
    def test_other_types_rejected(self, header):
        err = _raises(v.validate_content_type, header, 100, status=415, code="unsupported_media_type")
        assert header in err.message

    @pytest.mark.parametrize("length", [None, 0])
    def test_no_type_and_no_body_is_missing_input(self, length):
        _raises(v.validate_content_type, None, length, status=400, code="missing_input")

    def test_no_type_with_body(self):
        _raises(v.validate_content_type, None, 10, status=415, code="unsupported_media_type")

    def test_multipart_without_boundary(self):
        _raises(v.validate_content_type, "multipart/form-data", 10, status=400, code="malformed_body")


class TestLimitReceive:
    @staticmethod
    def _drain(chunks: list[bytes], limit: int) -> int:
        messages = [{"type": "http.request", "body": c, "more_body": i < len(chunks) - 1} for i, c in enumerate(chunks)]

        async def receive():
            return messages.pop(0)

        async def run():
            wrapped = v.limit_receive(receive, limit)
            total = 0
            while True:
                message = await wrapped()
                total += len(message["body"])
                if not message["more_body"]:
                    return total

        return asyncio.run(run())

    def test_under_and_at_limit(self):
        assert self._drain([b"a" * 40, b"b" * 60], limit=100) == 100

    def test_over_limit_stops_reading(self):
        with pytest.raises(v.BodyTooLarge):
            self._drain([b"a" * 60, b"b" * 60, b"c" * 60], limit=100)

    def test_non_body_messages_pass_through(self):
        async def receive():
            return {"type": "http.disconnect"}

        assert asyncio.run(v.limit_receive(receive, 0)()) == {"type": "http.disconnect"}


# --------------------------------------------------------------------------- #
# 2. Form
# --------------------------------------------------------------------------- #
class TestValidateForm:
    def test_file_only(self):
        upload = FakeUpload()
        result = v.validate_form([("file", upload)])
        assert result.file is upload and result.url is None and result.details is True

    def test_url_only_is_stripped(self):
        result = v.validate_form([("url", "  https://example.com/a.png  ")])
        assert result.file is None and result.url == "https://example.com/a.png"

    @pytest.mark.parametrize("value, expected", [
        ("false", False), ("FALSE", False), ("0", False), ("no", False), ("off", False),
        ("true", True), ("1", True), ("Yes", True), ("on", True), ("", True), ("  ", True),
    ])
    def test_details_values(self, value, expected):
        assert v.validate_form([("file", FakeUpload()), ("details", value)]).details is expected

    @pytest.mark.parametrize("value", ["maybe", "2", "tru", "null"])
    def test_details_invalid(self, value):
        _raises(v.validate_form, [("file", FakeUpload()), ("details", value)], status=400, code="invalid_parameter")

    def test_missing_input(self):
        _raises(v.validate_form, [], status=400, code="missing_input")
        _raises(v.validate_form, [("details", "false")], status=400, code="missing_input")
        _raises(v.validate_form, [("url", "   ")], status=400, code="missing_input")

    def test_both_inputs(self):
        _raises(v.validate_form, [("file", FakeUpload()), ("url", "https://e.com/a.png")],
                status=400, code="ambiguous_input")

    def test_unknown_field_suggests_correction(self):
        err = _raises(v.validate_form, [("file", FakeUpload()), ("detail", "false")], status=400, code="unknown_field")
        assert "'detail' (did you mean 'details'?)" in err.message
        assert "allowed fields: file, url, details" in err.message

    def test_unknown_field_legacy_names(self):
        err = _raises(v.validate_form, [("media", FakeUpload()), ("models", "genai")], status=400, code="unknown_field")
        assert "'media' (did you mean 'file'?)" in err.message
        assert "'models' (only used by /1.0/check.json)" in err.message

    def test_unknown_field_no_suggestion(self):
        err = _raises(v.validate_form, [("file", FakeUpload()), ("zzz", "1")], status=400, code="unknown_field")
        assert "'zzz'" in err.message and "did you mean" not in err.message

    def test_unknown_field_list_is_capped(self):
        items = [(f"f{i}", "1") for i in range(20)]
        err = _raises(v.validate_form, items, status=400, code="unknown_field")
        assert "'f4'" in err.message and "'f5'" not in err.message
        assert "and 15 more" in err.message

    def test_unknown_fields_reported_before_other_problems(self):
        # One clear error for the likely root cause (a typo), not "missing input".
        _raises(v.validate_form, [("fil", FakeUpload())], status=400, code="unknown_field")

    def test_field_names_are_case_sensitive(self):
        _raises(v.validate_form, [("File", FakeUpload())], status=400, code="unknown_field")

    @pytest.mark.parametrize("items", [
        [("file", FakeUpload()), ("file", FakeUpload())],
        [("url", "https://e.com/a"), ("url", "https://e.com/b")],
        [("file", FakeUpload()), ("details", "true"), ("details", "false")],
    ])
    def test_duplicates(self, items):
        _raises(v.validate_form, items, status=400, code="duplicate_field")

    def test_file_must_be_an_upload(self):
        err = _raises(v.validate_form, [("file", "hello")], status=400, code="invalid_field_type")
        assert "file=@" in err.message

    @pytest.mark.parametrize("name", ["url", "details"])
    def test_text_fields_must_be_text(self, name):
        items = [("file", FakeUpload())] if name == "details" else []
        _raises(v.validate_form, items + [(name, FakeUpload())], status=400, code="invalid_field_type")


class TestNormalizeFilename:
    @pytest.mark.parametrize("name, expected", [
        ("photo.jpg", "photo.jpg"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\scan.pdf", "scan.pdf"),
        ("bad\x00na\x1bme\n.png", "badname.png"),
        ("  spaced.png  ", "spaced.png"),
        ("résumé 写真.png", "résumé 写真.png"),
        ("", None),
        (None, None),
        ("dir/", None),
        ("\x00\x01", None),
    ])
    def test_cases(self, name, expected):
        assert v.normalize_filename(name) == expected

    def test_length_is_capped(self):
        assert len(v.normalize_filename("a" * 1000 + ".png")) == v.MAX_FILENAME_LENGTH


# --------------------------------------------------------------------------- #
# 3. URL
# --------------------------------------------------------------------------- #
class TestValidateUrl:
    @pytest.mark.parametrize("url", [
        "https://example.com/a.png",
        "http://example.com:8080/path/a.jpg?x=1#frag",
        "HTTPS://EXAMPLE.COM/A.PNG",
        "http://[2001:db8::1]/a.png",
        "https://example.com/" + "a" * (v.MAX_URL_LENGTH - 20),
    ])
    def test_valid(self, url):
        assert v.validate_url(url) == url

    @pytest.mark.parametrize("url, fragment", [
        ("ftp://example.com/a.png", "http"),
        ("file:///etc/passwd", "http"),
        ("javascript:alert(1)", "http"),
        ("example.com/a.png", "http"),
        ("http://", "http"),
        ("//example.com/a.png", "http"),
        ("http://user:pw@example.com/a.png", "credentials"),
        ("http://user@example.com/a.png", "credentials"),
        ("http://example.com:99999/a.png", "port"),
        ("http://example.com:abc/a.png", "port"),
        ("http://:80/a.png", "host"),
        ("http://exa mple.com/a.png", "whitespace"),
        ("http://example.com/a\nb.png", "whitespace"),
        ("http://example.com/a\tb.png", "whitespace"),
        ("http://example.com/a\x00.png", "control"),
        ("http://example.com/" + "a" * v.MAX_URL_LENGTH, "longer"),
    ])
    def test_invalid(self, url, fragment):
        err = _raises(v.validate_url, url, status=400, code="invalid_url")
        assert fragment in err.message


class TestPublicAddress:
    @pytest.mark.parametrize("address", ["93.184.216.34", "8.8.8.8", "2606:4700:4700::1111", "::ffff:8.8.8.8"])
    def test_public(self, address):
        assert v.is_public_address(address)

    @pytest.mark.parametrize("address", [
        "127.0.0.1", "127.8.9.10",          # loopback
        "10.0.0.5", "172.16.0.1", "192.168.1.1",  # private
        "169.254.169.254",                  # link-local / cloud metadata
        "100.64.0.1",                       # carrier-grade NAT
        "0.0.0.0", "::",                    # unspecified
        "224.0.0.1", "ff02::1",             # multicast
        "::1", "fe80::1%eth0", "fd00::1",   # IPv6 loopback, link-local (scoped), ULA
        "::ffff:127.0.0.1", "::ffff:10.0.0.1",  # IPv4-mapped private
        # IPv4-mapped forms the stdlib alone misclassifies (it checks the
        # IPv6 form): carrier-grade NAT and multicast.
        "::ffff:100.64.0.1", "::ffff:224.0.0.1",
        "240.0.0.1",                        # reserved
    ])
    def test_not_public(self, address):
        assert not v.is_public_address(address)


class TestCheckHostAllowed:
    @staticmethod
    def resolver(mapping):
        def resolve(host, port):
            if host not in mapping:
                raise OSError("Name or service not known")
            return mapping[host]
        return resolve

    def test_public_host_allowed(self):
        v.check_host_allowed("https://example.com/a.png", self.resolver({"example.com": ["93.184.216.34"]}))

    @pytest.mark.parametrize("url, addresses", [
        ("http://localhost/a.png", ["127.0.0.1", "::1"]),
        ("http://internal.corp/a.png", ["10.1.2.3"]),
        ("http://metadata.example/a", ["169.254.169.254"]),
        # One private answer among public ones is enough to refuse.
        ("http://mixed.example/a.png", ["93.184.216.34", "192.168.0.10"]),
    ])
    def test_private_hosts_blocked(self, url, addresses):
        host = url.split("/")[2]
        err = _raises(v.check_host_allowed, url, self.resolver({host: addresses}), status=400, code="url_not_allowed")
        assert "TRUEDOC_ALLOW_PRIVATE_URLS" in err.message

    def test_ip_literal_blocked(self):
        _raises(v.check_host_allowed, "http://127.0.0.1:8008/health", self.resolver({"127.0.0.1": ["127.0.0.1"]}),
                status=400, code="url_not_allowed")

    def test_opt_out_allows_private(self, monkeypatch):
        monkeypatch.setattr(v, "ALLOW_PRIVATE_URLS", True)
        v.check_host_allowed("http://localhost/a.png", self.resolver({"localhost": ["127.0.0.1"]}))

    def test_unresolvable_host(self):
        _raises(v.check_host_allowed, "http://nope.invalid/a.png", self.resolver({}), status=502, code="url_fetch_failed")

    def test_no_addresses(self):
        _raises(v.check_host_allowed, "http://empty.example/a", self.resolver({"empty.example": []}),
                status=502, code="url_fetch_failed")

    def test_default_port_passed_to_resolver(self):
        seen = []

        def resolve(host, port):
            seen.append((host, port))
            return ["93.184.216.34"]

        v.check_host_allowed("https://example.com/a", resolve)
        v.check_host_allowed("http://example.com/a", resolve)
        v.check_host_allowed("http://example.com:8080/a", resolve)
        assert seen == [("example.com", 443), ("example.com", 80), ("example.com", 8080)]


# --------------------------------------------------------------------------- #
# 4. File content
# --------------------------------------------------------------------------- #
class TestValidateFileContent:
    def test_png(self):
        facts = v.validate_file_content(_image_bytes("PNG", (32, 24)))
        assert (facts.format, facts.width, facts.height, facts.page_count) == ("png", 32, 24, None)

    def test_jpeg(self):
        facts = v.validate_file_content(_image_bytes("JPEG", (40, 30)))
        assert (facts.format, facts.width, facts.height) == ("jpeg", 40, 30)

    def test_grayscale_png(self):
        assert v.validate_file_content(_image_bytes("PNG", (16, 16), mode="L")).format == "png"

    def test_pdf(self):
        facts = v.validate_file_content(ONE_PAGE_PDF)
        assert (facts.format, facts.page_count, facts.width) == ("pdf", 1, None)

    def test_empty(self):
        _raises(v.validate_file_content, b"", status=400, code="empty_file")

    def test_over_size_limit(self, monkeypatch):
        data = _image_bytes("PNG")
        monkeypatch.setattr(v, "MAX_FILE_BYTES", len(data) - 1)
        _raises(v.validate_file_content, data, status=413, code="file_too_large")

    @pytest.mark.parametrize("data", [
        _image_bytes("GIF"), _image_bytes("BMP"), _image_bytes("WEBP"),
        b"<html></html>", b"hello world", b"\x00" * 100,
    ], ids=["gif", "bmp", "webp", "html", "text", "zeros"])
    def test_unsupported_formats(self, data):
        _raises(v.validate_file_content, data, status=415, code="unsupported_format")

    def test_format_is_sniffed_not_named(self):
        # The filename plays no part; only the bytes do.
        assert v.validate_file_content(_image_bytes("JPEG")).format == "jpeg"

    @pytest.mark.parametrize("data", [
        b"\x89PNG\r\n\x1a\ngarbage",                    # bogus chunk after the signature
        b"\x89PNG\r\n\x1a\n",                           # signature only
        _image_bytes("PNG")[:60],                       # truncated mid-stream
        b"\xff\xd8\xff" + b"\x00" * 64,                 # JPEG marker, no image
    ], ids=["png-garbage", "png-signature-only", "png-truncated", "jpeg-garbage"])
    def test_corrupt_images(self, data):
        err = _raises(v.validate_file_content, data, status=422, code="unprocessable_file")
        assert "0x" not in err.message  # no object reprs leak to clients

    def test_png_with_bad_crc(self):
        data = bytearray(_image_bytes("PNG"))
        idat = data.find(b"IDAT")
        data[idat + 6] ^= 0xFF  # flip a byte inside the IDAT payload
        _raises(v.validate_file_content, bytes(data), status=422, code="unprocessable_file")

    def test_image_over_pixel_limit_reports_size(self):
        data = _png_header_only(12_000, 10_000)  # 120 MP, default limit 100 MP
        err = _raises(v.validate_file_content, data, status=422, code="image_too_large")
        assert "12000x10000" in err.message and "100 megapixels" in err.message

    def test_decompression_bomb(self):
        # Far past Pillow's own guard too: refused before any pixel is decoded.
        data = _png_header_only(20_000, 20_000)
        assert len(data) < 1000
        _raises(v.validate_file_content, data, status=422, code="image_too_large")

    def test_pixel_limit_is_configurable(self, monkeypatch):
        monkeypatch.setattr(v, "MAX_IMAGE_PIXELS", 32 * 24 - 1)
        _raises(v.validate_file_content, _image_bytes("PNG", (32, 24)), status=422, code="image_too_large")
        monkeypatch.setattr(v, "MAX_IMAGE_PIXELS", 32 * 24)
        assert v.validate_file_content(_image_bytes("PNG", (32, 24))).width == 32

    def test_encrypted_pdf(self):
        data = (FIXTURES / "encrypted.pdf").read_bytes()
        _raises(v.validate_file_content, data, status=422, code="encrypted_pdf")

    @pytest.mark.parametrize("data", [
        b"%PDF-1.4\ngarbage",
        b"%PDF-",
        ONE_PAGE_PDF[: len(ONE_PAGE_PDF) // 3],
        ZERO_PAGE_PDF,
    ], ids=["garbage", "header-only", "truncated", "zero-pages"])
    def test_unreadable_pdfs(self, data):
        err = _raises(v.validate_file_content, data, status=422, code="unprocessable_file")
        assert "PDFium" not in err.message

    def test_baseline_files_pass(self):
        baseline = Path(__file__).resolve().parents[2] / "baseline"
        if not baseline.exists():
            pytest.skip("baseline files not present")
        for path in sorted(baseline.rglob("*")):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}:
                assert v.validate_file_content(path.read_bytes()).format in {"png", "jpeg", "pdf"}, path
