"""Ranking logic for comparable sales."""

from __future__ import annotations

import math
import re
from typing import Any

DEFAULT_SIGNAL_WEIGHTS = {
    "semantic": 0.50,
    "visual": 0.30,
    "structured": 0.20,
}

DEFAULT_STRUCTURED_WEIGHTS = {
    "object_type": 0.18,
    "manufacturer": 0.18,
    "artist": 0.12,
    "period": 0.12,
    "material": 0.10,
    "country": 0.08,
    "condition": 0.08,
    "dimensions": 0.07,
    "marks": 0.07,
}


def _normalized_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value") or value.get("name") or value.get("text")
    if value is None:
        return ""
    return str(value).strip().lower()


def _tokenize(value: Any) -> set[str]:
    text = _normalized_text(value)
    return {token for token in re.split(r"[^a-z0-9]+", text) if token}


def _match_fraction(source_values: list[str], target_values: list[str]) -> float:
    source = {item for item in source_values if item}
    target = {item for item in target_values if item}
    if not source or not target:
        return 0.0
    overlap = source.intersection(target)
    return len(overlap) / max(len(source), len(target))


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values = []
        for item in value:
            normalized = _normalized_text(item)
            if normalized:
                values.append(normalized)
        return values
    normalized = _normalized_text(value)
    return [normalized] if normalized else []


def _candidate_names(value: Any) -> list[str]:
    return _as_list(value)


def _extract_years(value: Any) -> tuple[int | None, int | None]:
    if isinstance(value, dict):
        start = value.get("estimated_year_start")
        end = value.get("estimated_year_end")
        if isinstance(start, int) or isinstance(end, int):
            return start, end
        value = value.get("period") or value.get("likely_period") or value.get("value")
    text = _normalized_text(value)
    years = [int(match) for match in re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", text)]
    if len(years) >= 2:
        return min(years), max(years)
    if len(years) == 1:
        return years[0], years[0]
    return None, None


def _range_overlap(
    left_start: int | None,
    left_end: int | None,
    right_start: int | None,
    right_end: int | None,
) -> float:
    if None in {left_start, left_end, right_start, right_end}:
        return 0.0
    start = max(left_start, right_start)
    end = min(left_end, right_end)
    if end < start:
        return 0.0
    left_span = max(1, left_end - left_start + 1)
    right_span = max(1, right_end - right_start + 1)
    return (end - start + 1) / max(left_span, right_span)


def object_type_similarity(query: dict, sale: dict) -> float:
    query_text = _normalized_text(query.get("object_type"))
    sale_text = _normalized_text(sale.get("object_type"))
    if not query_text or not sale_text:
        return 0.0
    if query_text == sale_text:
        return 1.0
    if query_text in sale_text or sale_text in query_text:
        return 0.85
    return _match_fraction(list(_tokenize(query_text)), list(_tokenize(sale_text)))


def manufacturer_similarity(query: dict, sale: dict) -> float:
    return _match_fraction(
        _candidate_names(query.get("manufacturer_candidates")),
        _as_list(sale.get("manufacturer")),
    )


def artist_similarity(query: dict, sale: dict) -> float:
    return _match_fraction(
        _candidate_names(query.get("artist_candidates")),
        _as_list(sale.get("artist")),
    )


def period_similarity(query: dict, sale: dict) -> float:
    query_years = (
        query.get("estimated_year_start"),
        query.get("estimated_year_end"),
    )
    sale_years = _extract_years(sale.get("period"))
    overlap = _range_overlap(*query_years, *sale_years)
    if overlap > 0:
        return overlap
    query_text = _normalized_text(query.get("period") or query.get("likely_period"))
    sale_text = _normalized_text(sale.get("period"))
    if not query_text or not sale_text:
        return 0.0
    if query_text == sale_text:
        return 1.0
    if query_text in sale_text or sale_text in query_text:
        return 0.75
    return _match_fraction(list(_tokenize(query_text)), list(_tokenize(sale_text)))


def material_similarity(query: dict, sale: dict) -> float:
    return _match_fraction(
        _as_list(query.get("materials")),
        _as_list(sale.get("materials") or sale.get("material")),
    )


def country_similarity(query: dict, sale: dict) -> float:
    query_text = _normalized_text(query.get("country"))
    sale_text = _normalized_text(sale.get("country"))
    if not query_text or not sale_text:
        return 0.0
    return 1.0 if query_text == sale_text else 0.0


def condition_similarity(query: dict, sale: dict) -> float:
    query_text = _normalized_text(query.get("condition"))
    sale_text = _normalized_text(sale.get("condition"))
    if not query_text or not sale_text:
        return 0.0
    if query_text == sale_text:
        return 1.0
    return 0.5 if query_text in sale_text or sale_text in query_text else 0.0


def dimensions_similarity(query: dict, sale: dict) -> float:
    values = []
    for field in ("height", "width", "depth", "diameter", "weight"):
        left = query.get(field)
        right = sale.get(field)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            continue
        if left <= 0 or right <= 0:
            continue
        ratio = abs(float(left) - float(right)) / max(float(left), float(right))
        values.append(max(0.0, 1.0 - min(1.0, ratio)))
    if not values:
        return 0.0
    return sum(values) / len(values)


def marks_similarity(query: dict, sale: dict) -> float:
    query_marks = [
        _normalized_text(mark.get("normalized_text") or mark.get("text"))
        for mark in query.get("marks", []) or []
        if isinstance(mark, dict)
    ]
    sale_marks = _as_list(sale.get("marks"))
    return _match_fraction(query_marks, sale_marks)


def explain_structured_similarity(query: dict, sale: dict) -> tuple[float, list[str]]:
    scores = {
        "same object type": object_type_similarity(query, sale),
        "same manufacturer": manufacturer_similarity(query, sale),
        "same artist": artist_similarity(query, sale),
        "same period": period_similarity(query, sale),
        "similar materials": material_similarity(query, sale),
        "same country": country_similarity(query, sale),
        "similar condition": condition_similarity(query, sale),
        "similar dimensions": dimensions_similarity(query, sale),
        "similar manufacturer mark": marks_similarity(query, sale),
    }
    weighted_total = 0.0
    total_weight = 0.0
    for label, score in scores.items():
        weight_key = {
            "same object type": "object_type",
            "same manufacturer": "manufacturer",
            "same artist": "artist",
            "same period": "period",
            "similar materials": "material",
            "same country": "country",
            "similar condition": "condition",
            "similar dimensions": "dimensions",
            "similar manufacturer mark": "marks",
        }[label]
        weight = DEFAULT_STRUCTURED_WEIGHTS.get(weight_key, 0.0)
        weighted_total += score * weight
        total_weight += weight
    reasons = [label for label, score in scores.items() if score >= 0.6]
    structured = weighted_total / total_weight if total_weight else 0.0
    return structured, reasons


def compute_overall_similarity(
    *,
    semantic_similarity: float,
    structured_similarity: float,
    visual_similarity: float | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    weights = weights or DEFAULT_SIGNAL_WEIGHTS
    signals = {
        "semantic": semantic_similarity,
        "structured": structured_similarity,
    }
    if visual_similarity is not None:
        signals["visual"] = visual_similarity

    total_weight = sum(weights.get(name, 0.0) for name in signals)
    if math.isclose(total_weight, 0.0):
        return 0.0
    score = sum(signals[name] * weights.get(name, 0.0) for name in signals) / total_weight
    return max(0.0, min(1.0, score))


def compute_structured_similarity(
    identification: dict,
    comparable: dict,
    semantic_similarity: float = 0.0,
    weights: dict | None = None,
) -> float:
    structured_similarity, _ = explain_structured_similarity(identification, comparable)
    signal_weights = dict(DEFAULT_SIGNAL_WEIGHTS)
    if weights:
        signal_weights.update(
            {key: value for key, value in weights.items() if key in signal_weights}
        )
    return compute_overall_similarity(
        semantic_similarity=semantic_similarity,
        structured_similarity=structured_similarity,
        weights=signal_weights,
    )
