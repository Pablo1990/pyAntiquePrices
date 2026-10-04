"""Tests for the similarity-weighted log-space estimator and v2 training."""

from __future__ import annotations

import math
from datetime import date

import pytest

from pyantique_prices.pricing.estimator import (
    effective_sample_size,
    estimate_from_comparables,
    weighted_quantile,
)
from pyantique_prices.pricing.model import PricePredictor
from pyantique_prices.pricing.training import (
    attach_residual_quantiles,
    predict_interval,
    prediction_level,
    train_bucket_model,
)


def _comp(price, sim=None, sale_date=None):
    comp = {"normalized_price": price}
    if sim is not None:
        comp["overall_similarity"] = sim
    if sale_date is not None:
        comp["sale_date"] = sale_date
    return comp


def test_weighted_quantile_matches_median_for_equal_weights():
    assert weighted_quantile([1, 2, 3], [1, 1, 1], 0.5) == 2.0


def test_effective_sample_size_penalises_dominant_weight():
    assert effective_sample_size([1, 1, 1, 1]) == pytest.approx(4.0)
    assert effective_sample_size([10, 0.1, 0.1, 0.1]) < 1.1


def test_close_matches_dominate_weak_ones():
    comps = [_comp(1000, 0.95), _comp(1100, 0.9), _comp(90, 0.1), _comp(80, 0.1), _comp(95, 0.1)]
    est = estimate_from_comparables(comps)
    assert est["quantiles"]["p50"] > 800  # unweighted median would be 95


def test_interval_is_never_degenerate_and_widens_with_few_samples():
    same = [_comp(500, 0.9)] * 3
    est = estimate_from_comparables(same)["quantiles"]
    assert est["p25"] < 500 < est["p75"]
    many = estimate_from_comparables([_comp(500, 0.9)] * 30)["quantiles"]
    assert (est["p75"] / est["p25"]) > (many["p75"] / many["p25"])


def test_quantiles_are_ordered():
    est = estimate_from_comparables([_comp(p, 0.5) for p in (50, 80, 120, 300, 900)])["quantiles"]
    assert est["p10"] < est["p25"] < est["p50"] < est["p75"] < est["p90"]


def test_old_sales_weigh_less():
    today = date(2026, 1, 1)
    comps = [_comp(1000, 0.5, date(2025, 6, 1)), _comp(100, 0.5, date(1960, 1, 1))]
    est = estimate_from_comparables(comps, today=today)
    assert est["quantiles"]["p50"] > 500


def test_prior_shrinks_towards_model_when_few_effective_comparables():
    comps = [_comp(100, 0.5), _comp(100, 0.5), _comp(100, 0.5)]
    no_prior = estimate_from_comparables(comps)["quantiles"]["p50"]
    with_prior = estimate_from_comparables(comps, prior_price=400)["quantiles"]["p50"]
    assert no_prior == pytest.approx(100)
    assert 100 < with_prior < 400


def test_no_priced_comparables_returns_none():
    assert estimate_from_comparables([{"normalized_price": None}]) is None


def test_bucket_model_ignores_tiny_buckets_and_uses_log_residuals():
    rows = [{"object_type": "vase", "country": "fr", "target_price": p} for p in (100, 200, 400)]
    rows.append({"object_type": "lamp", "country": "fr", "target_price": 5000})  # 1 sample
    model = train_bucket_model(rows)
    assert "lamp" not in model["by_object"]
    assert prediction_level(model, {"object_type": "lamp", "country": "fr"}) == "global"
    assert prediction_level(model, {"object_type": "vase", "country": "fr"}) == "object_country"

    attach_residual_quantiles(model, rows[:3])
    interval = predict_interval(model, {"object_type": "vase", "country": "fr"})
    assert interval["p25"] < interval["p50"] < interval["p75"]
    # Multiplicative: scaling every price scales the interval proportionally
    scaled = train_bucket_model([{**r, "target_price": r["target_price"] * 10} for r in rows[:3]])
    attach_residual_quantiles(scaled, [{**r, "target_price": r["target_price"] * 10} for r in rows[:3]])
    big = predict_interval(scaled, {"object_type": "vase", "country": "fr"})
    assert big["p75"] / interval["p75"] == pytest.approx(10, rel=1e-6)


def test_global_prior_is_not_used():
    predictor = PricePredictor(model_dir="/nonexistent")
    res = predictor.predict({}, [_comp(100, 0.5)] * 6)
    assert res["prior_level"] is None
    assert res["method"] == "similarity_weighted_estimate"
    assert math.isclose(res["mid"], 100.0, rel_tol=1e-6)
