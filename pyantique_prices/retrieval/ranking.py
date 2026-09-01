"""Ranking logic for comparable sales."""

from __future__ import annotations

DEFAULT_RANKING_WEIGHTS = {
    "semantic": 0.30,
    "manufacturer": 0.20,
    "object_type": 0.15,
    "period": 0.10,
    "material": 0.10,
    "country": 0.05,
    "condition": 0.05,
    "dimensions": 0.05,
}


def _normalized_text(value) -> str:
    if isinstance(value, dict):
        value = value.get("value") or value.get("name") or value.get("text")
    if value is None:
        return ""
    return str(value).strip().lower()


def _match_fraction(source_values: list[str], target_values: list[str]) -> float:
    source = {item for item in source_values if item}
    target = {item for item in target_values if item}
    if not source or not target:
        return 0.0
    overlap = source.intersection(target)
    return len(overlap) / max(len(source), len(target))


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values = []
        for item in value:
            if isinstance(item, dict):
                normalized = _normalized_text(
                    item.get("name") or item.get("value") or item.get("text")
                )
            else:
                normalized = _normalized_text(item)
            if normalized:
                values.append(normalized)
        return values
    normalized = _normalized_text(value)
    return [normalized] if normalized else []


def compute_structured_similarity(
    identification: dict,
    comparable: dict,
    semantic_similarity: float = 0.0,
    weights: dict | None = None,
) -> float:
    weights = weights or DEFAULT_RANKING_WEIGHTS
    object_match = 1.0 if _normalized_text(identification.get("object_type")) in _normalized_text(comparable.get("object_type")) else 0.0
    country_match = 1.0 if _normalized_text(identification.get("country")) == _normalized_text(comparable.get("country")) and _normalized_text(identification.get("country")) else 0.0
    condition_match = 1.0 if _normalized_text(identification.get("condition")) == _normalized_text(comparable.get("condition")) and _normalized_text(identification.get("condition")) else 0.0
    manufacturer_match = _match_fraction(
        _as_list(identification.get("manufacturer_candidates")),
        _as_list([comparable.get("manufacturer")]),
    )
    period_match = 1.0 if _normalized_text(identification.get("likely_period")) and _normalized_text(identification.get("likely_period")) in _normalized_text(comparable.get("period")) else 0.0
    material_match = _match_fraction(
        _as_list(identification.get("materials")),
        _as_list(comparable.get("materials") or comparable.get("material")),
    )
    score = (
        semantic_similarity * weights.get("semantic", 0.0)
        + manufacturer_match * weights.get("manufacturer", 0.0)
        + object_match * weights.get("object_type", 0.0)
        + period_match * weights.get("period", 0.0)
        + material_match * weights.get("material", 0.0)
        + country_match * weights.get("country", 0.0)
        + condition_match * weights.get("condition", 0.0)
    )
    return max(0.0, min(1.0, score))
