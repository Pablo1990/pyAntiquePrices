from __future__ import annotations

import pickle
from pathlib import Path

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


def test_extract_features_supports_structured_values():
    identification = {
        "object_type": {"value": "clock"},
        "country": {"value": "France"},
        "condition": {"value": "good"},
    }
    features = extract_features(identification, [{"normalized_price": 100.0}])
    assert features["object_type"] == "clock"
    assert features["country"] == "France"
    assert features["condition_score"] == 0.7


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
    assert result["method"] == "quantile_estimate"
    assert result["confidence_note"] == "Low confidence: 3-5 comparable sales."


def test_price_predictor_uses_reference_only_for_1_to_2_comparables():
    predictor = PricePredictor()
    result = predictor.predict({}, [{"normalized_price": 100.0}, {"normalized_price": 120.0}])

    assert result is not None
    assert result["valuation_available"] is False
    assert result["method"] == "reference_only"
    assert result["confidence_note"] == "Very low confidence: only 1-2 comparable sales."


def test_price_predictor_uses_artifact_model_when_available(tmp_path):
    models_dir = Path(tmp_path) / "models"
    quantiles_dir = models_dir / "quantile_models"
    quantiles_dir.mkdir(parents=True)

    model = {
        "model_type": "bucket_median_v1",
        "global_median": 900.0,
        "by_object": {"clock": 1000.0},
        "by_object_country": {"clock|france": 1200.0},
        "residual_quantiles": {
            "p10": -200.0,
            "p25": -100.0,
            "p50": 0.0,
            "p75": 150.0,
            "p90": 250.0,
        },
    }
    calibrator = {"scale": 1.0, "bias": 0.0}
    with (models_dir / "price_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    with (models_dir / "calibrator.pkl").open("wb") as handle:
        pickle.dump(calibrator, handle)

    predictor = PricePredictor(model_dir=str(models_dir))
    comparables = [{"normalized_price": float(1000 + i * 10)} for i in range(6)]
    result = predictor.predict(
        {"object_type": "clock", "country": "France", "condition_score": 0.7},
        comparables,
    )
    assert result is not None
    assert result["method"] == "model_quantile_estimate"
    assert result["mid"] == 1200.0


def test_quantiles_and_metrics():
    quantiles = compute_quantiles([100.0, 200.0, 300.0, 400.0], [0.25, 0.5, 0.75])
    metrics = compute_metrics([100.0, 200.0], [110.0, 190.0])

    assert quantiles == {"p25": 175.0, "p50": 250.0, "p75": 325.0}
    assert metrics["mae"] == 10.0
    assert metrics["rmse"] == 10.0
