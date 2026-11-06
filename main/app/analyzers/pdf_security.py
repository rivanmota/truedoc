"""Potentially risky PDF features and active-content checks.

Dependency-free beyond the project's existing pypdfium2 package. Most active
content is identified from PDF object syntax because these features are useful
security/fraud context even when the file cannot be fully opened.
"""
from __future__ import annotations

import re

import pypdfium2 as pdfium

from app.pdfium_guard import PDFIUM


_PATTERNS = {
    "javascript": re.compile(rb"/JavaScript\b|/JS\b"),
    "open_action": re.compile(rb"/OpenAction\b"),
    "additional_actions": re.compile(rb"/AA\b"),
    "launch_action": re.compile(rb"/Launch\b"),
    "embedded_files": re.compile(rb"/EmbeddedFiles\b|/Filespec\b"),
    "rich_media": re.compile(rb"/RichMedia\b"),
    "xfa": re.compile(rb"/XFA\b"),
    "uri_actions": re.compile(rb"/URI\b"),
}


def analyze(pdf_bytes: bytes) -> dict:
    features = {name: len(pattern.findall(pdf_bytes)) for name, pattern in _PATTERNS.items()}

    # C2PA Content Credentials are commonly embedded through the PDF
    # EmbeddedFiles name tree. Treat that known provenance payload as benign
    # rather than as a generic attachment anomaly.
    c2pa_refs = len(re.findall(rb"Content\s+Credentials|application/c2pa|c2pa", pdf_bytes, re.IGNORECASE))
    benign_c2pa_attachment = bool(features["embedded_files"] and c2pa_refs)

    # Raw syntax is the most robust lightweight indicator here. PDFium may
    # refuse encrypted PDFs without a password, so encryption is checked before
    # attempting to enumerate pages.
    encrypted = bool(re.search(rb"/Encrypt\b", pdf_bytes))
    acroform = bool(re.search(rb"/AcroForm\b", pdf_bytes))
    page_count = None

    try:
        with PDFIUM:
            doc = pdfium.PdfDocument(pdf_bytes)
            try:
                page_count = len(doc)
            finally:
                doc.close()
    except Exception:
        pass

    anomalies: list[dict] = []
    severity = {
        "javascript": "high",
        "open_action": "medium",
        "additional_actions": "medium",
        "launch_action": "high",
        "embedded_files": "medium",
        "rich_media": "high",
        "xfa": "medium",
        "uri_actions": "low",
    }
    messages = {
        "javascript": "PDF contains JavaScript-related objects.",
        "open_action": "PDF defines an OpenAction that can run when the document opens.",
        "additional_actions": "PDF defines additional automatic actions.",
        "launch_action": "PDF contains a Launch action.",
        "embedded_files": "PDF contains embedded-file/file-specification objects.",
        "rich_media": "PDF contains RichMedia objects.",
        "xfa": "PDF contains XFA form content.",
        "uri_actions": "PDF contains URI action(s).",
    }
    for name, count in features.items():
        if name == "embedded_files" and benign_c2pa_attachment:
            continue
        if count:
            anomalies.append({
                "code": name,
                "severity": severity[name],
                "count": count,
                "message": messages[name],
            })

    if encrypted:
        anomalies.append({
            "code": "encrypted_pdf",
            "severity": "low",
            "message": "PDF is encrypted; some structural checks may be limited without a password.",
        })

    severity_score = {"low": 0.10, "medium": 0.45, "high": 0.90}
    score = max((severity_score[a["severity"]] for a in anomalies), default=0.0)

    return {
        "score": round(score, 4),
        "confidence": "high",
        "encrypted": encrypted,
        "acroform_present": acroform,
        "page_count": page_count,
        "features": features,
        "content_credentials_present": benign_c2pa_attachment,
        "anomalies": anomalies,
        "note": "active content and embedded files are security/fraud context, not AI-generation evidence",
    }
