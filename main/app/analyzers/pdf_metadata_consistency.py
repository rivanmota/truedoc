"""PDF metadata consistency checks for document forensics.

Uses pypdfium2 plus lightweight raw-XMP parsing so this module does not add a
new dependency. It looks for chronology and Info/XMP inconsistencies that can
support a fraud/tampering investigation. These signals are not AI-generation
proof by themselves.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
from typing import Any

import pypdfium2 as pdfium

from app.pdfium_guard import exclusive as _pdfium_exclusive


_PDF_DATE_RE = re.compile(
    r"^(?:D:)?(?P<y>\d{4})(?P<m>\d{2})?(?P<d>\d{2})?"
    r"(?P<h>\d{2})?(?P<mi>\d{2})?(?P<s>\d{2})?"
    r"(?P<tz>Z|[+\-]\d{2}'?\d{2}'?)?$"
)


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        return text or None
    except Exception:
        return None


def _parse_pdf_date(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    text = _safe_text(value)
    if not text:
        return None

    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    m = _PDF_DATE_RE.match(text)
    if not m:
        return None

    try:
        year = int(m.group("y"))
        month = int(m.group("m") or 1)
        day = int(m.group("d") or 1)
        hour = int(m.group("h") or 0)
        minute = int(m.group("mi") or 0)
        second = int(m.group("s") or 0)
        tz_text = m.group("tz")
        tz = timezone.utc
        if tz_text and tz_text != "Z":
            sign = 1 if tz_text[0] == "+" else -1
            digits = tz_text[1:].replace("'", "")
            hh = int(digits[:2])
            mm = int(digits[2:4])
            tz = timezone(sign * timedelta(hours=hh, minutes=mm))
        return datetime(year, month, day, hour, minute, second, tzinfo=tz).astimezone(timezone.utc)
    except Exception:
        return None


def _dt_iso(dt: datetime | None) -> str | None:
    return dt.isoformat().replace("+00:00", "Z") if dt else None


@_pdfium_exclusive
def _pdfium_metadata(pdf_bytes: bytes) -> dict[str, str]:
    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        fields = [
            "Title", "Author", "Subject", "Keywords", "Creator", "Producer",
            "CreationDate", "ModDate",
        ]
        out: dict[str, str] = {}
        for field in fields:
            try:
                value = doc.get_metadata_value(field)
            except Exception:
                value = None
            if value:
                out[field] = str(value).strip()
        return out
    finally:
        doc.close()


def _extract_xmp_value(text: str, tag: str) -> str | None:
    # Supports both element and attribute forms used by common XMP packets.
    patterns = [
        rf"<(?:[A-Za-z0-9_\-]+:)?{re.escape(tag)}\b[^>]*>(.*?)</(?:[A-Za-z0-9_\-]+:)?{re.escape(tag)}>",
        rf"(?:[A-Za-z0-9_\-]+:)?{re.escape(tag)}\s*=\s*[\"']([^\"']+)[\"']",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            value = re.sub(r"\s+", " ", m.group(1)).strip()
            if value:
                return value
    return None


def _raw_xmp(pdf_bytes: bytes) -> dict[str, str]:
    # XMP is normally UTF-8 XML. latin-1 fallback keeps byte positions harmless
    # and lets us inspect odd producer output without failing the request.
    text = pdf_bytes.decode("utf-8", errors="ignore")
    if "xmpmeta" not in text.lower() and "rdf:rdf" not in text.lower():
        text = pdf_bytes.decode("latin-1", errors="ignore")

    mapping = {
        "CreateDate": "creation",
        "ModifyDate": "modified",
        "MetadataDate": "metadata",
        "CreatorTool": "creator_tool",
        "Producer": "producer",
    }
    out: dict[str, str] = {}
    for xmp_tag, key in mapping.items():
        value = _extract_xmp_value(text, xmp_tag)
        if value:
            out[key] = value
    return out


def analyze(pdf_bytes: bytes, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    anomalies: list[dict] = []

    try:
        info = _pdfium_metadata(pdf_bytes)
    except Exception as exc:
        return {
            "score": 0.0,
            "confidence": "low",
            "anomalies": [],
            "note": f"metadata consistency analysis unavailable: {exc}",
        }

    xmp = _raw_xmp(pdf_bytes)

    info_creation = _parse_pdf_date(info.get("CreationDate"))
    info_mod = _parse_pdf_date(info.get("ModDate"))
    xmp_creation = _parse_pdf_date(xmp.get("creation"))
    xmp_mod = _parse_pdf_date(xmp.get("modified"))
    xmp_metadata_date = _parse_pdf_date(xmp.get("metadata"))

    def add(code: str, severity: str, message: str, **evidence: Any) -> None:
        anomalies.append({"code": code, "severity": severity, "message": message, **evidence})

    if info_creation and info_mod and info_mod + timedelta(minutes=1) < info_creation:
        add(
            "modification_before_creation", "high",
            "PDF modification date is earlier than its creation date.",
            creation=_dt_iso(info_creation), modified=_dt_iso(info_mod),
        )

    for label, dt in (
        ("info_creation", info_creation),
        ("info_modified", info_mod),
        ("xmp_creation", xmp_creation),
        ("xmp_modified", xmp_mod),
        ("xmp_metadata", xmp_metadata_date),
    ):
        if dt and dt > now + timedelta(days=1):
            add(
                "future_metadata_date", "high",
                f"{label} is more than one day in the future.",
                field=label, value=_dt_iso(dt),
            )
        if dt and dt.year < 1990:
            add(
                "implausibly_old_metadata_date", "medium",
                f"{label} predates common PDF-era document workflows.",
                field=label, value=_dt_iso(dt),
            )

    for label, a, b in (
        ("creation", info_creation, xmp_creation),
        ("modified", info_mod, xmp_mod),
    ):
        if a and b:
            delta = abs((a - b).total_seconds())
            if delta > 86400:
                add(
                    f"info_xmp_{label}_date_mismatch", "high",
                    f"Info and XMP {label} dates differ by more than 24 hours.",
                    info=_dt_iso(a), xmp=_dt_iso(b), difference_seconds=round(delta, 1),
                )
            elif delta > 300:
                add(
                    f"info_xmp_{label}_date_mismatch", "medium",
                    f"Info and XMP {label} dates differ by more than five minutes.",
                    info=_dt_iso(a), xmp=_dt_iso(b), difference_seconds=round(delta, 1),
                )

    creator = _safe_text(info.get("Creator"))
    producer = _safe_text(info.get("Producer"))
    xmp_creator = _safe_text(xmp.get("creator_tool"))
    xmp_producer = _safe_text(xmp.get("producer"))

    if creator and xmp_creator and creator.casefold() != xmp_creator.casefold():
        add(
            "creator_xmp_mismatch", "medium",
            "Info Creator and XMP CreatorTool disagree.",
            info_creator=creator, xmp_creator=xmp_creator,
        )
    if producer and xmp_producer and producer.casefold() != xmp_producer.casefold():
        add(
            "producer_xmp_mismatch", "medium",
            "Info Producer and XMP Producer disagree.",
            info_producer=producer, xmp_producer=xmp_producer,
        )

    severity_score = {"low": 0.15, "medium": 0.45, "high": 0.90}
    score = max((severity_score[a["severity"]] for a in anomalies), default=0.0)
    if len(anomalies) > 1:
        score = min(1.0, score + 0.04 * (len(anomalies) - 1))

    return {
        "score": round(score, 4),
        "confidence": "high" if anomalies else "medium",
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "dates": {
            "info_creation": _dt_iso(info_creation),
            "info_modified": _dt_iso(info_mod),
            "xmp_creation": _dt_iso(xmp_creation),
            "xmp_modified": _dt_iso(xmp_mod),
            "xmp_metadata": _dt_iso(xmp_metadata_date),
        },
        "software": {
            "creator": creator,
            "producer": producer,
            "xmp_creator_tool": xmp_creator,
            "xmp_producer": xmp_producer,
        },
        "note": "metadata anomalies are fraud/tamper evidence, not proof of AI generation",
    }
