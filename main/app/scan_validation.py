"""
Input validation for the scan API (``POST /api/v1/scans``).

Every check is a plain function that raises ``ScanError`` (HTTP status,
machine-readable code, message), so each rule can be unit-tested without
HTTP or the forensic engine. ``app/scan_api.py`` applies them in order:

1. request   - ``validate_content_length``, ``validate_content_type`` and
               ``limit_receive`` (the body is capped while it streams in,
               before it is parsed or spooled to disk);
2. form      - ``validate_form``: only known fields, each at most once and of
               the right kind (``file`` a file upload, ``url``/``details``
               text); exactly one of ``file``/``url``; ``details`` a boolean;
3. url       - ``validate_url`` (syntax) and ``check_host_allowed`` (the host
               must resolve to public addresses; applied to every redirect
               hop by the fetcher);
4. content   - ``validate_file_content``: non-empty, a PNG/JPEG/PDF by its
               bytes, an image header Pillow can read with a pixel count under
               the limit, a PDF that opens without a password and has pages.

Limits are read from the environment once, at import:

    TRUEDOC_MAX_FILE_MB              file size (upload or download), default 50
    TRUEDOC_MAX_IMAGE_MEGAPIXELS     decoded image size, default 100
    TRUEDOC_ALLOW_PRIVATE_URLS       "true" to allow URLs on loopback/private
                                     networks (off by default)
"""
from __future__ import annotations

import difflib
import io
import ipaddress
import os
import socket
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urlparse

from PIL import Image

from app.pdfium_guard import PDFIUM


class ScanError(Exception):
    """A request the API refuses, with the HTTP status and error code to send."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


MAX_FILE_BYTES = int(float(os.environ.get("TRUEDOC_MAX_FILE_MB", "50")) * 1024 * 1024)
# Multipart framing and the text fields on top of the file itself.
FORM_OVERHEAD_BYTES = 64 * 1024
MAX_IMAGE_PIXELS = int(float(os.environ.get("TRUEDOC_MAX_IMAGE_MEGAPIXELS", "100")) * 1_000_000)
ALLOW_PRIVATE_URLS = _env_bool("TRUEDOC_ALLOW_PRIVATE_URLS")
MAX_URL_LENGTH = 2048
MAX_FILENAME_LENGTH = 255
# Form parser limits: generous enough to reach validate_form (which names the
# offending field), small enough that a request cannot make it do real work.
MAX_FORM_FILES = 3
MAX_FORM_FIELDS = 10
MAX_FORM_PART_BYTES = 16 * 1024

FIELDS = ("file", "url", "details")
FORM_MEDIA_TYPES = ("multipart/form-data", "application/x-www-form-urlencoded")
# Field names clients are likely to carry over from /1.0/check.json.
_FIELD_HINTS = {"media": "file", "models": None, "api_user": None, "api_secret": None}

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


def max_request_bytes() -> int:
    return MAX_FILE_BYTES + FORM_OVERHEAD_BYTES


def _media_type(header: Optional[str]) -> str:
    return (header or "").split(";", 1)[0].strip().lower()


def request_body_limit(content_type: Optional[str]) -> int:
    """Largest body accepted for this Content-Type. Only multipart can carry
    a file; a url-encoded body holds just ``url`` and ``details`` (and
    starlette parses it into memory with no field limit), so it gets the
    small form allowance."""
    if _media_type(content_type) == "application/x-www-form-urlencoded":
        return FORM_OVERHEAD_BYTES
    return max_request_bytes()


def _mb(n: int) -> str:
    return f"{n / (1024 * 1024):g} MB"


def file_too_large() -> ScanError:
    return ScanError(413, "file_too_large", f"file exceeds the {_mb(MAX_FILE_BYTES)} limit")


# --------------------------------------------------------------------------- #
# 1. Request
# --------------------------------------------------------------------------- #
def validate_content_length(header: Optional[str], limit: Optional[int] = None) -> Optional[int]:
    """Rejects a declared body size over ``limit`` (default: the multipart
    limit) before anything is read. Returns the declared length (None when
    absent, e.g. chunked uploads, which ``limit_receive`` caps instead)."""
    if header is None:
        return None
    value = header.strip()
    if not value.isdigit():
        raise ScanError(400, "invalid_request", "Content-Length header is not a valid length")
    length = int(value)
    if length > (max_request_bytes() if limit is None else limit):
        raise file_too_large()
    return length


def validate_content_type(header: Optional[str], content_length: Optional[int]) -> None:
    """The body must be a form. A request with no body at all is reported as
    missing input rather than as the wrong media type."""
    media_type = _media_type(header)
    if not media_type:
        if not content_length:
            raise ScanError(400, "missing_input", "provide a 'file' upload or a 'url' field")
        raise ScanError(415, "unsupported_media_type", _media_type_message("no Content-Type"))
    if media_type not in FORM_MEDIA_TYPES:
        raise ScanError(415, "unsupported_media_type", _media_type_message(f"Content-Type '{media_type}'"))
    if media_type == "multipart/form-data" and "boundary=" not in (header or "").lower():
        raise ScanError(400, "malformed_body", "multipart/form-data Content-Type has no boundary")


def _media_type_message(got: str) -> str:
    return (
        f"request body must be multipart/form-data (file upload or url) or "
        f"application/x-www-form-urlencoded (url only); got {got}"
    )


class BodyTooLarge(Exception):
    pass


def limit_receive(receive: Callable, limit: int) -> Callable:
    """Wraps an ASGI ``receive`` so the request body stops being read, and
    ``BodyTooLarge`` is raised, as soon as it exceeds ``limit`` bytes. This
    holds for chunked bodies with no Content-Length and for clients whose
    Content-Length understates the body."""
    received = 0

    async def wrapped():
        nonlocal received
        message = await receive()
        if message.get("type") == "http.request":
            received += len(message.get("body", b""))
            if received > limit:
                raise BodyTooLarge()
        return message

    return wrapped


# --------------------------------------------------------------------------- #
# 2. Form
# --------------------------------------------------------------------------- #
@dataclass
class ScanInput:
    file: Any = None          # starlette UploadFile
    url: Optional[str] = None
    details: bool = True


def _is_upload(value: Any) -> bool:
    # Duck-typed: starlette and FastAPI each have an UploadFile class.
    return hasattr(value, "filename") and hasattr(value, "read") and not isinstance(value, str)


def parse_bool(name: str, value: str) -> bool:
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ScanError(400, "invalid_parameter", f"'{name}' must be true or false")


_MAX_REPORTED_FIELDS = 5


def _unknown_field_message(unknown: list[str]) -> str:
    parts = []
    for name in unknown[:_MAX_REPORTED_FIELDS]:
        key = name.lower()
        if key in _FIELD_HINTS:
            hint = _FIELD_HINTS[key]
            parts.append(f"'{name}' (did you mean '{hint}'?)" if hint else f"'{name}' (only used by /1.0/check.json)")
            continue
        close = difflib.get_close_matches(key, FIELDS, n=1, cutoff=0.6)
        parts.append(f"'{name}' (did you mean '{close[0]}'?)" if close else f"'{name}'")
    if len(unknown) > _MAX_REPORTED_FIELDS:
        parts.append(f"and {len(unknown) - _MAX_REPORTED_FIELDS} more")
    return f"unknown field(s): {', '.join(parts)}; allowed fields: {', '.join(FIELDS)}"


def validate_form(items: Iterable[tuple[str, Any]]) -> ScanInput:
    """Validates the parsed form (``FormData.multi_items()``)."""
    items = list(items)
    names = [name for name, _ in items]

    unknown = list(dict.fromkeys(n for n in names if n not in FIELDS))
    if unknown:
        raise ScanError(400, "unknown_field", _unknown_field_message(unknown))
    duplicated = [n for n in FIELDS if names.count(n) > 1]
    if duplicated:
        raise ScanError(
            400, "duplicate_field",
            f"field(s) sent more than once: {', '.join(duplicated)}; send one file or one url per scan",
        )

    values = dict(items)
    file, url, details = values.get("file"), values.get("url"), values.get("details")
    if file is not None and not _is_upload(file):
        raise ScanError(400, "invalid_field_type", "'file' must be a file upload (curl: -F 'file=@path/to/file')")
    for name, value in (("url", url), ("details", details)):
        if value is not None and not isinstance(value, str):
            raise ScanError(400, "invalid_field_type", f"'{name}' must be a text field, not a file upload")

    has_url = bool(url and url.strip())
    if file is not None and has_url:
        raise ScanError(400, "ambiguous_input", "provide either 'file' or 'url', not both")
    if file is None and not has_url:
        raise ScanError(400, "missing_input", "provide a 'file' upload or a 'url' field")

    return ScanInput(
        file=file,
        url=url.strip() if has_url else None,
        details=True if details is None or not details.strip() else parse_bool("details", details),
    )


def normalize_filename(name: Optional[str]) -> Optional[str]:
    """The client's filename as echoed in the response: directory parts and
    control characters removed, length capped. Never used as a path."""
    if not name:
        return None
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    clean = "".join(ch for ch in base if unicodedata.category(ch)[0] != "C").strip()
    return clean[:MAX_FILENAME_LENGTH] or None


# --------------------------------------------------------------------------- #
# 3. URL
# --------------------------------------------------------------------------- #
def validate_url(url: str) -> str:
    """Syntax checks only (no network). Returns the URL unchanged."""
    if len(url) > MAX_URL_LENGTH:
        raise ScanError(400, "invalid_url", f"url is longer than {MAX_URL_LENGTH} characters")
    if any(ch.isspace() or unicodedata.category(ch) == "Cc" for ch in url):
        raise ScanError(400, "invalid_url", "url must not contain whitespace or control characters")
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        raise ScanError(400, "invalid_url", "url must be an absolute http:// or https:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise ScanError(400, "invalid_url", "url must not contain credentials (user:password@)")
    if not parsed.hostname:
        raise ScanError(400, "invalid_url", "url has no host")
    try:
        parsed.port  # noqa: B018 - raises ValueError when out of range
    except ValueError:
        raise ScanError(400, "invalid_url", "url has an invalid port") from None
    return url


def is_public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast and not ip.is_unspecified


def _resolve(host: str, port: int) -> list[str]:
    return [info[4][0] for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


def check_host_allowed(url: str, resolver: Callable[[str, int], list[str]] = _resolve) -> None:
    """The URL's host must resolve, and only to public addresses, unless
    TRUEDOC_ALLOW_PRIVATE_URLS is set. This stops the server from being
    used to reach itself, the local network or cloud metadata endpoints.

    Every address the name resolves to is checked, so a name that mixes a
    public and a private address is refused. (The HTTP client resolves the
    name again when it connects; a DNS server that changes its answer in
    between is not defended against here.)"""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    try:
        addresses = resolver(host, port)
    except (socket.gaierror, UnicodeError, OSError) as exc:
        # The resolver's own text varies by platform and resolver state.
        raise ScanError(502, "url_fetch_failed", f"could not resolve host '{host}'") from exc
    if not addresses:
        raise ScanError(502, "url_fetch_failed", f"could not resolve host '{host}'")
    if ALLOW_PRIVATE_URLS:
        return
    blocked = sorted({a for a in addresses if not is_public_address(a)})
    if blocked:
        raise ScanError(
            400, "url_not_allowed",
            f"url host '{host}' resolves to a non-public address ({', '.join(blocked)}); "
            "set TRUEDOC_ALLOW_PRIVATE_URLS=true to allow local/private URLs",
        )


# --------------------------------------------------------------------------- #
# 4. File content
# --------------------------------------------------------------------------- #
# Pillow's names for each sniffed format. Multi-picture JPEGs from phones
# (MPO) are ordinary JPEG streams to every other decoder.
_PILLOW_FORMATS = {"png": {"PNG"}, "jpeg": {"JPEG", "MPO"}}


@dataclass
class FileFacts:
    format: str
    width: Optional[int] = None
    height: Optional[int] = None
    page_count: Optional[int] = None


def _validate_image(data: bytes, fmt: str) -> FileFacts:
    """Reads only the header (and, for PNG, walks the chunks) with Pillow,
    before the engine decodes pixels: a bogus header is refused in
    milliseconds instead of being handed to OpenCV, and the pixel count is
    known before any memory is committed to it."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:
                width, height = im.size
                pillow_format = im.format
                if width * height <= MAX_IMAGE_PIXELS:
                    im.verify()
    except Image.DecompressionBombError:
        raise _image_too_large(None, None) from None
    except Exception as exc:
        # Pillow's messages embed object reprs; the client gets a clean one.
        raise ScanError(422, "unprocessable_file", f"not a readable {fmt.upper()} image (corrupt or truncated)") from exc

    if pillow_format not in _PILLOW_FORMATS[fmt]:
        raise ScanError(422, "unprocessable_file", f"file starts like a {fmt.upper()} but decodes as {pillow_format}")
    if width < 1 or height < 1:
        raise ScanError(422, "unprocessable_file", "image has no pixels")
    if width * height > MAX_IMAGE_PIXELS:
        raise _image_too_large(width, height)
    return FileFacts(format=fmt, width=width, height=height)


def _image_too_large(width: Optional[int], height: Optional[int]) -> ScanError:
    limit = f"{MAX_IMAGE_PIXELS / 1e6:g} megapixels"
    if width and height:
        message = f"image is {width}x{height} ({width * height / 1e6:.1f} megapixels); the limit is {limit}"
    else:
        # Pillow's own bomb guard fired before the size could be read.
        message = f"image dimensions exceed the {limit} limit"
    return ScanError(422, "image_too_large", message)


def _validate_pdf(data: bytes) -> FileFacts:
    with PDFIUM:  # PDFium is not thread-safe (app/pdfium_guard.py)
        return _open_pdf(data)


def _open_pdf(data: bytes) -> FileFacts:
    import pypdfium2 as pdfium

    try:
        doc = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        if "password" in str(exc).lower():
            raise ScanError(422, "encrypted_pdf", "PDF is password-protected; remove the password and retry") from exc
        # pdfium refuses page-less documents at load time, with a misleading
        # "Success" status; neither its wording nor its codes are client-facing.
        raise ScanError(422, "unprocessable_file", "not a readable PDF (corrupt, truncated or without pages)") from exc
    try:
        pages = len(doc)
    finally:
        doc.close()
    if pages < 1:
        raise ScanError(422, "unprocessable_file", "PDF has no pages")
    return FileFacts(format="pdf", page_count=pages)


def validate_file_content(data: bytes) -> FileFacts:
    if not data:
        raise ScanError(400, "empty_file", "the file is empty")
    if len(data) > MAX_FILE_BYTES:
        raise file_too_large()
    # The engine's sniffer, so the API and the engine agree on the format.
    # Imported here: app.main imports this module (via app.scan_api).
    from app.main import SUPPORTED_FORMATS, _detect_format

    fmt = _detect_format(data)
    if fmt == "unknown":
        raise ScanError(415, "unsupported_format", f"unsupported file format; supported formats: {SUPPORTED_FORMATS}")
    if fmt == "pdf":
        return _validate_pdf(data)
    return _validate_image(data, fmt)
