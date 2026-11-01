"""PDF digital-signature structure checks.

This is intentionally a structural verifier, not a full PKI validator. It
locates PDF signature dictionaries/ByteRange values and detects bytes appended
after the signed revision. Cryptographic certificate-chain validation should be
added separately if the application needs legal-grade signature validation.
"""
from __future__ import annotations

import re


_BYTERANGE_RE = re.compile(
    rb"/ByteRange\s*\[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\]",
    re.DOTALL,
)
_SIG_HINT_RE = re.compile(rb"/Type\s*/Sig\b|/FT\s*/Sig\b")


def _meaningful_tail(data: bytes) -> bool:
    # Ignore whitespace/NUL only. Any later xref/trailer/update is meaningful.
    return bool(data.strip(b"\x00\x09\x0a\x0c\x0d\x20"))


def analyze(pdf_bytes: bytes) -> dict:
    signatures = []
    anomalies = []

    for index, match in enumerate(_BYTERANGE_RE.finditer(pdf_bytes), start=1):
        a, b, c, d = [int(x) for x in match.groups()]
        covered_end = c + d
        valid_layout = (
            a == 0
            and b >= 0
            and c >= b
            and d >= 0
            and covered_end <= len(pdf_bytes)
        )
        appended = False
        appended_bytes = 0
        if valid_layout and covered_end < len(pdf_bytes):
            tail = pdf_bytes[covered_end:]
            appended = _meaningful_tail(tail)
            appended_bytes = len(tail) if appended else 0

        signatures.append({
            "index": index,
            "byte_range": [a, b, c, d],
            "covered_end": covered_end,
            "file_size": len(pdf_bytes),
            "byte_range_structurally_valid": valid_layout,
            "modified_after_signing": appended,
            "appended_bytes": appended_bytes,
        })

        if not valid_layout:
            anomalies.append({
                "code": "invalid_signature_byte_range",
                "severity": "high",
                "message": f"Signature {index} has an invalid ByteRange layout.",
            })
        elif appended:
            anomalies.append({
                "code": "content_appended_after_signature",
                "severity": "high",
                "message": f"PDF contains bytes appended after signature {index}'s signed revision.",
                "signature_index": index,
                "appended_bytes": appended_bytes,
            })

    signature_hint_count = len(_SIG_HINT_RE.findall(pdf_bytes))
    if signature_hint_count and not signatures:
        anomalies.append({
            "code": "signature_dictionary_without_byte_range",
            "severity": "medium",
            "message": "Signature-related objects were found but no parseable ByteRange was located.",
        })

    severity_score = {"low": 0.15, "medium": 0.50, "high": 0.97}
    score = max((severity_score[a["severity"]] for a in anomalies), default=0.0)

    return {
        "score": round(score, 4),
        "confidence": "high" if signatures else "medium",
        "signature_count": max(len(signatures), signature_hint_count),
        "byte_range_signature_count": len(signatures),
        "modified_after_signing": any(s["modified_after_signing"] for s in signatures),
        "cryptographic_validation_performed": False,
        "signatures": signatures,
        "anomalies": anomalies,
        "note": "ByteRange checks can detect post-signing append operations but do not validate certificate trust or cryptographic signature correctness",
    }
