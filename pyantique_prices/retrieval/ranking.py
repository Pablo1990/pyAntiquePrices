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


MIN_CANDIDATE_CONFIDENCE = 0.3


def _confident_candidate_names(value: Any) -> list[str]:
    """Candidate names, dropping low-confidence guesses (they cause false matches)."""
    if not isinstance(value, list):
        return _candidate_names(value)
    names = []
    for item in value:
        if isinstance(item, dict):
            conf = item.get("confidence")
            if isinstance(conf, (int, float)) and conf < MIN_CANDIDATE_CONFIDENCE:
                continue
        name = _normalized_text(item)
        if name:
            names.append(name)
    return names


def _name_similarity(query_names: list[str], sale_names: list[str]) -> float:
    """Best fuzzy match between name lists: exact, containment, token overlap.

    "meissen" vs "meissen porcelain manufactory" must match; exact-string
    comparison would score it 0.
    """
    best = 0.0
    for left in query_names:
        for right in sale_names:
            if not left or not right:
                continue
            if left == right:
                return 1.0
            if left in right or right in left:
                best = max(best, 0.85)
                continue
            lt, rt = _tokenize(left), _tokenize(right)
            if lt and rt:
                best = max(best, len(lt & rt) / max(len(lt), len(rt)))
    return best


def manufacturer_similarity(query: dict, sale: dict) -> float:
    return _name_similarity(
        _confident_candidate_names(query.get("manufacturer_candidates")),
        _as_list(sale.get("manufacturer")),
    )


def artist_similarity(query: dict, sale: dict) -> float:
    return _name_similarity(
        _confident_candidate_names(query.get("artist_candidates")),
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


_FIELD_LABELS = {
    "object_type": "same object type",
    "manufacturer": "same manufacturer",
    "artist": "same artist",
    "period": "same period",
    "material": "similar materials",
    "country": "same country",
    "condition": "similar condition",
    "dimensions": "similar dimensions",
    "marks": "similar manufacturer mark",
}


def _known_fields(query: dict, sale: dict) -> set[str]:
    """Fields for which BOTH sides carry information.

    A missing value is *unknown*, not a mismatch: scoring it 0 would punish
    sales (or photos) for sparse metadata and bury the best comparables.
    """
    known = set()
    if _normalized_text(query.get("object_type")) and _normalized_text(sale.get("object_type")):
        known.add("object_type")
    if _confident_candidate_names(query.get("manufacturer_candidates")) and _as_list(
        sale.get("manufacturer")
    ):
        known.add("manufacturer")
    if _confident_candidate_names(query.get("artist_candidates")) and _as_list(
        sale.get("artist")
    ):
        known.add("artist")
    q_period = (
        query.get("estimated_year_start") is not None
        or _normalized_text(query.get("period") or query.get("likely_period"))
    )
    if q_period and _normalized_text(sale.get("period")):
        known.add("period")
    if _as_list(query.get("materials")) and _as_list(sale.get("materials") or sale.get("material")):
        known.add("material")
    if _normalized_text(query.get("country")) and _normalized_text(sale.get("country")):
        known.add("country")
    if _normalized_text(query.get("condition")) and _normalized_text(sale.get("condition")):
        known.add("condition")
    if any(
        isinstance(query.get(f), (int, float)) and isinstance(sale.get(f), (int, float))
        and query[f] > 0 and sale[f] > 0
        for f in ("height", "width", "depth", "diameter", "weight")
    ):
        known.add("dimensions")
    if [m for m in (query.get("marks") or []) if isinstance(m, dict) and (m.get("normalized_text") or m.get("text"))] and _as_list(
        sale.get("marks")
    ):
        known.add("marks")
    return known


def explain_structured_similarity(
    query: dict,
    sale: dict,
    weights: dict[str, float] | None = None,
) -> tuple[float, list[str]]:
    """Weighted structured similarity over fields known on both sides.

    The result is the weighted mean over known fields, scaled by
    ``0.5 + 0.5 * coverage`` (share of total weight that was comparable) so a
    perfect match on one field does not outrank a strong match on many.
    """
    weights = {**DEFAULT_STRUCTURED_WEIGHTS, **(weights or {})}
    scorers = {
        "object_type": object_type_similarity,
        "manufacturer": manufacturer_similarity,
        "artist": artist_similarity,
        "period": period_similarity,
        "material": material_similarity,
        "country": country_similarity,
        "condition": condition_similarity,
        "dimensions": dimensions_similarity,
        "marks": marks_similarity,
    }
    known = _known_fields(query, sale)
    total_weight = sum(weights.get(k, 0.0) for k in scorers)
    known_weight = sum(weights.get(k, 0.0) for k in known)
    if not known or known_weight <= 0 or total_weight <= 0:
        return 0.0, []

    weighted_total = 0.0
    reasons = []
    for key in known:
        score = scorers[key](query, sale)
        weighted_total += score * weights.get(key, 0.0)
        if score >= 0.6:
            reasons.append(_FIELD_LABELS[key])
    coverage = known_weight / total_weight
    structured = (weighted_total / known_weight) * (0.5 + 0.5 * coverage)
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
