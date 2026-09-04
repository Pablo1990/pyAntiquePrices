"""Normalized text documents for semantic comparable retrieval."""

from __future__ import annotations

from typing import Any


def _as_text(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("value") or value.get("name") or value.get("text")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            text = _as_text(item)
            if text:
                result.append(text)
        return result
    text = _as_text(value)
    return [text] if text else []


def build_search_document(identification: dict | Any) -> str:
    data = identification if isinstance(identification, dict) else identification.model_dump()
    lines: list[str] = []

    for field in ("object_type", "subtype", "period"):
        text = _as_text(data.get(field))
        if text:
            lines.append(text)

    years = []
    if data.get("estimated_year_start") is not None:
        years.append(str(data["estimated_year_start"]))
    if data.get("estimated_year_end") is not None:
        years.append(str(data["estimated_year_end"]))
    if years:
        lines.append("circa " + "–".join(years))

    for label, field in (
        ("manufacturer", "manufacturer_candidates"),
        ("artist", "artist_candidates"),
        ("workshop", "workshop_candidates"),
    ):
        values = _as_list(data.get(field))
        if values:
            lines.append(f"possible {label} " + ", ".join(values))

    for field in ("country", "region", "condition", "rarity_assessment"):
        text = _as_text(data.get(field))
        if text:
            lines.append(text)

    for field in ("materials", "techniques", "styles", "provenance_clues"):
        values = _as_list(data.get(field))
        if values:
            lines.append(", ".join(values))

    marks = data.get("marks") or []
    for mark in marks:
        if not isinstance(mark, dict):
            continue
        bits = [
            _as_text(mark.get("text")),
            _as_text(mark.get("normalized_text")),
            _as_text(mark.get("mark_type")),
        ]
        mark_text = " ".join(bit for bit in bits if bit)
        if mark_text:
            lines.append(mark_text)

    return "\n".join(dict.fromkeys(line for line in lines if line))


def build_sale_search_document(sale: dict) -> str:
    document = {
        "object_type": sale.get("object_type"),
        "subtype": sale.get("subcategory"),
        "period": sale.get("period"),
        "manufacturer_candidates": [{"name": sale.get("manufacturer")}]
        if sale.get("manufacturer")
        else [],
        "artist_candidates": [{"name": sale.get("artist")}] if sale.get("artist") else [],
        "workshop_candidates": [{"name": sale.get("workshop")}] if sale.get("workshop") else [],
        "country": sale.get("country"),
        "region": sale.get("region"),
        "materials": sale.get("materials") or sale.get("material"),
        "techniques": sale.get("technique"),
        "condition": sale.get("condition"),
        "marks": sale.get("marks") or [],
        "provenance_clues": sale.get("provenance") or [],
        "rarity_assessment": sale.get("rarity_assessment"),
    }
    title = _as_text(sale.get("title"))
    description = _as_text(sale.get("description"))
    body = build_search_document(document)
    extra = "\n".join(part for part in [title, description, body] if part)
    return extra.strip()
