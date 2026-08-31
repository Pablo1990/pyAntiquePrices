from __future__ import annotations

from pyantique_prices.pricing.calibration import compute_metrics
from pyantique_prices.pricing.features import condition_to_float, extract_features
from pyantique_prices.pricing.model import PricePredictor
from pyantique_prices.pricing.quantiles import compute_quantiles


def test_condition_to_float_handles_known_and_unknown_values():
    assert condition_to_float("good") == 0.7
    assert condition_to_float("unknown") == 0.5
    assert condition_to_float(None) == 0.5


def test_extract_features_computes_price_statistics():
    identification = {"object_type": "clock", "country": "France", "condition": "good"}
    comparables = [
        {"normalized_price": 100.0},
        {"normalized_price": 200.0},
        {"normalized_price": 300.0},
    ]

    features = extract_features(identification, comparables)

    assert features["object_type"] == "clock"
    assert features["condition_score"] == 0.7
    assert features["median_comparable_price"] == 200.0
    assert features["num_comparables"] == 3


def test_price_predictor_returns_quantile_estimate():
    predictor = PricePredictor()
    result = predictor.predict(
        {},
        [
            {"normalized_price": 100.0},
            {"normalized_price": 200.0},
            {"normalized_price": 300.0},
            {"normalized_price": 400.0},
        ],
    )

    assert result is not None
    assert result["mid"] == 250.0
    assert result["valuation_available"] is True
    assert result["confidence_note"] == "Low confidence: 3-5 comparable sales."


def test_quantiles_and_metrics():
    quantiles = compute_quantiles([100.0, 200.0, 300.0, 400.0], [0.25, 0.5, 0.75])
    metrics = compute_metrics([100.0, 200.0], [110.0, 190.0])

    assert quantiles == {"p25": 175.0, "p50": 250.0, "p75": 325.0}
    assert metrics["mae"] == 10.0
    assert metrics["rmse"] == 10.0
