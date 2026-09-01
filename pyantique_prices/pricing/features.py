"""Feature engineering for the pricing model."""

from __future__ import annotations

from typing import Any

import numpy as np

CONDITION_SCALE = {
    "exceptional": 1.00,
    "very good": 0.85,
    "good": 0.70,
    "fair": 0.50,
    "poor": 0.25,
    "extremely poor": 0.00,
}


def _extract_value(value):
    if isinstance(value, dict):
        return value.get("value")
    return value


def condition_to_float(condition: str | None) -> float:
    condition = _extract_value(condition)
    if condition is None:
        return 0.5
    return CONDITION_SCALE.get(condition.lower().strip(), 0.5)


def extract_features(identification: dict, comparables: list[dict]) -> dict[str, Any]:
    """Extract features for the pricing model from identification + comparables."""
    prices = [
        comparable.get("normalized_price")
        for comparable in comparables
        if comparable.get("normalized_price")
    ]
    object_type = _extract_value(identification.get("object_type")) or ""
    country = _extract_value(identification.get("country")) or ""
    condition = _extract_value(identification.get("condition"))

    return {
        "object_type": object_type,
        "country": country,
        "condition_score": condition_to_float(condition),
        "num_comparables": len(prices),
        "median_comparable_price": float(np.median(prices)) if prices else 0.0,
        "mean_comparable_price": float(np.mean(prices)) if prices else 0.0,
        "comparable_price_iqr": (
            float(np.percentile(prices, 75) - np.percentile(prices, 25))
            if len(prices) >= 2
            else 0.0
        ),
        "comparable_price_std": float(np.std(prices)) if len(prices) >= 2 else 0.0,
    }
