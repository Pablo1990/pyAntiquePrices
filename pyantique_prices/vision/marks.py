"""Dedicated maker-mark analysis helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from .schemas import AntiqueImageSet, Candidate, MarkEvidence

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

MARK_REGION_PROMPT = """\
Review these antique photos and identify which images are most likely to contain maker marks,
labels, signatures, or impressed/stamped marks. Return valid JSON only as:
{"regions":[{"image_name":"front.jpg","location":"base","mark_type":"maker_mark","confidence":0.0,"evidence":"..."}]}

If nothing readable is visible, return {"regions":[]}.
"""

MARK_TEXT_PROMPT = """\
Review these antique photos and read any visible maker marks, signatures, labels, or stamps.
Return valid JSON only as:
{"marks":[{"image_name":"base.jpg","text":"...","mark_type":"maker_mark","confidence":0.0,"evidence":"..."}]}

If the mark is unreadable, leave text null and explain why in evidence.
"""


def normalize_mark_text(value: str | None) -> str | None:
    if not value:
        return None
    text = value.upper().strip()
    text = re.sub(r"[^A-Z0-9\s&.\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_mark(value: str | None) -> str | None:
    return normalize_mark_text(value)


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


def generate_mark_candidates(
    normalized_text: str | None,
    reference: dict[str, list[str]] | None = None,
) -> list[Candidate]:
    return [
        Candidate(name=name, confidence=0.65, evidence="Matched normalized maker-mark text.")
        for name in lookup_manufacturer_candidates(normalized_text, reference)
    ]


def _parse_json_blob(raw: str, key: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        return []
    items = payload.get(key)
    return items if isinstance(items, list) else []


def detect_mark_regions(
    image_set: AntiqueImageSet,
    *,
    client=None,
) -> list[dict[str, Any]]:
    if client is not None:
        try:
            raw = client.analyze_images(image_set.paths, MARK_REGION_PROMPT)
            regions = _parse_json_blob(raw, "regions")
            if regions:
                return regions
        except Exception:
            pass

    regions = []
    for image in image_set.images:
        if image.role.value in {"base", "maker_mark", "signature", "label", "back", "detail"}:
            regions.append(
                {
                    "image_name": image.name,
                    "location": image.role.value,
                    "mark_type": "maker_mark" if image.role.value == "maker_mark" else None,
                    "confidence": 0.35,
                    "evidence": f"Filename/role suggests potential mark view ({image.role.value}).",
                }
            )
    return regions


def extract_mark_text(
    image_set: AntiqueImageSet,
    *,
    client=None,
    regions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if client is not None:
        try:
            raw = client.analyze_images(image_set.paths, MARK_TEXT_PROMPT)
            marks = _parse_json_blob(raw, "marks")
            if marks:
                return marks
        except Exception:
            pass

    extracted = []
    for region in regions or []:
        extracted.append(
            {
                "image_name": region.get("image_name"),
                "text": None,
                "location": region.get("location"),
                "mark_type": region.get("mark_type"),
                "confidence": float(region.get("confidence", 0.0) or 0.0),
                "evidence": region.get("evidence") or "Likely mark view but text could not be read.",
            }
        )
    return extracted


class MarkAnalysisService:
    """Analyze extracted marks and enrich with normalized evidence."""

    def __init__(self, reference: dict[str, list[str]] | None = None) -> None:
        self.reference = reference or DEFAULT_MARK_REFERENCE

    def analyze(
        self,
        identification: dict[str, Any],
        *,
        image_set: AntiqueImageSet | None = None,
        client=None,
    ) -> list[dict[str, Any]]:
        raw_marks = identification.get("marks") or []
        if not raw_marks and image_set is not None:
            regions = detect_mark_regions(image_set, client=client)
            raw_marks = extract_mark_text(image_set, client=client, regions=regions)
        if not raw_marks and identification.get("signature_text"):
            raw_marks = [
                {
                    "text": identification.get("signature_text"),
                    "mark_type": "signature",
                    "confidence": 0.5,
                    "evidence": "Signature text surfaced during vision identification.",
                }
            ]

        enriched = []
        for mark in raw_marks:
            text = mark.get("text")
            normalized_text = normalize_mark_text(text)
            mark_type = classify_mark_type(mark)
            candidates = generate_mark_candidates(normalized_text, self.reference)
            enriched.append(
                MarkEvidence(
                    image_name=mark.get("image_name") or mark.get("evidence_image"),
                    text=text,
                    normalized_text=normalized_text,
                    location=mark.get("location"),
                    mark_type=mark_type,
                    confidence=float(mark.get("confidence", 0.0) or 0.0),
                    evidence_image=mark.get("evidence_image"),
                    evidence=mark.get("evidence"),
                    manufacturer_candidates=candidates,
                ).model_dump()
            )
        return enriched
