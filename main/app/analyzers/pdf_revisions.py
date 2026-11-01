"""Incremental-update / revision heuristics for PDFs.

PDFs can be updated by appending a new xref/trailer and %%EOF marker rather than
rewriting the file. That is legitimate for signing and normal editing, but it
is also valuable forensic evidence because a later revision may introduce a
fraudulent page or image.
"""
from __future__ import annotations

import re


_EOF_RE = re.compile(rb"(?m)^[ \t]*%%EOF[ \t]*(?:\r?\n|$)")
_STARTXREF_RE = re.compile(rb"(?m)^[ \t]*startxref[ \t]*\r?\n[ \t]*(\d+)[ \t]*(?:\r?\n|$)")
_PREV_RE = re.compile(rb"/Prev\s+(\d+)")


def analyze(pdf_bytes: bytes) -> dict:
    eof_matches = list(_EOF_RE.finditer(pdf_bytes))
    startxrefs = [int(m.group(1)) for m in _STARTXREF_RE.finditer(pdf_bytes)]
    prev_offsets = [int(x) for x in _PREV_RE.findall(pdf_bytes)]

    # Multiple complete startxref/EOF sections are the most useful simple
    # indicator of incremental saves. Keep this conservative because some
    # legitimate PDF producers can leave unusual trailer structures.
    complete_sections = min(len(eof_matches), len(startxrefs))
    revision_count = max(1, complete_sections) if pdf_bytes.startswith(b"%PDF-") else complete_sections
    incremental_updates = max(0, revision_count - 1)

    anomalies = []
    if incremental_updates:
        # One incremental save is common for signing/forms and is only weak
        # evidence by itself. Multiple appended revisions are more notable.
        revision_severity = "medium" if incremental_updates >= 2 else "low"
        anomalies.append({
            "code": "incremental_updates_detected",
            "severity": revision_severity,
            "message": f"PDF contains {incremental_updates} appended incremental update(s).",
        })

    if len(eof_matches) != len(startxrefs):
        anomalies.append({
            "code": "xref_eof_count_mismatch",
            "severity": "medium",
            "message": "startxref and %%EOF counts do not match; the file may be malformed, repaired, or unusually constructed.",
            "startxref_count": len(startxrefs),
            "eof_count": len(eof_matches),
        })

    if startxrefs and any(x >= len(pdf_bytes) or x < 0 for x in startxrefs):
        anomalies.append({
            "code": "invalid_startxref_offset",
            "severity": "high",
            "message": "At least one startxref offset points outside the PDF byte range.",
        })

    severity_score = {"low": 0.12, "medium": 0.35, "high": 0.85}
    score = max((severity_score[a["severity"]] for a in anomalies), default=0.0)
    if incremental_updates >= 3:
        score = max(score, 0.45)

    return {
        "score": round(score, 4),
        "confidence": "medium-high",
        "revision_count": revision_count,
        "incremental_updates": incremental_updates,
        "startxref_count": len(startxrefs),
        "eof_count": len(eof_matches),
        "prev_reference_count": len(prev_offsets),
        "startxref_offsets": startxrefs[-10:],
        "prev_offsets": prev_offsets[-10:],
        "anomalies": anomalies,
        "note": "incremental updates are common in signed/edited PDFs; they become stronger fraud evidence when combined with changed signatures, metadata, pages, or images",
    }
