"""Dedicated maker-mark analysis helpers."""

from __future__ import annotations

import re
from typing import Any

MARK_TYPE_KEYWORDS = {
    "signature": "signature",
    "signed": "signature",
    "label": "label",
    "stamp": "stamp",
    "impressed": "impressed",
    "painted": "painted",
    "logo": "logo",
}

DEFAULT_MARK_REFERENCE = {
    "JAPY": ["Japy Freres"],
    "MEISSEN": ["Meissen"],
    "SEVRES": ["Sèvres"],
    "WEDGWOOD": ["Wedgwood"],
    "LLADRO": ["Lladró"],
}


def normalize_mark_text(value: str | None) -> str | None:
    if not value:
        return None
    text = value.upper().strip()
    text = re.sub(r"[^A-Z0-9\s&.\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def classify_mark_type(mark: dict[str, Any]) -> str | None:
    explicit = mark.get("mark_type")
    if explicit:
        return str(explicit).strip().lower()
    blob = " ".join(
        str(item)
        for item in [mark.get("text"), mark.get("evidence"), mark.get("location")]
        if item
    ).lower()
    for key, mark_type in MARK_TYPE_KEYWORDS.items():
        if key in blob:
            return mark_type
    return "text"


def lookup_manufacturer_candidates(
    normalized_text: str | None,
    reference: dict[str, list[str]] | None = None,
) -> list[str]:
    if not normalized_text:
        return []
    reference = reference or DEFAULT_MARK_REFERENCE
    matches: list[str] = []
    for key, candidates in reference.items():
        if key in normalized_text:
            matches.extend(candidates)
    deduped: list[str] = []
    for item in matches:
        if item not in deduped:
            deduped.append(item)
    return deduped


class MarkAnalysisService:
    """Analyze extracted marks and enrich with normalized evidence."""

    def __init__(self, reference: dict[str, list[str]] | None = None) -> None:
        self.reference = reference or DEFAULT_MARK_REFERENCE

    def analyze(self, identification: dict[str, Any]) -> list[dict[str, Any]]:
        raw_marks = identification.get("marks") or []
        if not raw_marks and identification.get("signature_text"):
            raw_marks = [
                {
                    "text": identification.get("signature_text"),
                    "mark_type": "signature",
                    "confidence": 0.5,
                }
            ]

        enriched = []
        for mark in raw_marks:
            text = mark.get("text")
            normalized_text = normalize_mark_text(text)
            mark_type = classify_mark_type(mark)
            candidates = lookup_manufacturer_candidates(normalized_text, self.reference)
            enriched.append(
                {
                    "text": text,
                    "normalized_text": normalized_text,
                    "location": mark.get("location"),
                    "mark_type": mark_type,
                    "confidence": float(mark.get("confidence", 0.0) or 0.0),
                    "evidence_image": mark.get("evidence_image"),
                    "evidence": mark.get("evidence"),
                    "manufacturer_candidates": candidates,
                }
            )
        return enriched
