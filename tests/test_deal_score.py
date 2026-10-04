from __future__ import annotations

import pytest

from pyantique_prices.deals import assess_deal, format_deal

VAL = {"p25": 200.0, "p50": 300.0, "p75": 420.0, "num_comparables": 12,
       "effective_comparables": 8.0, "valuation_available": True, "method": "similarity_weighted_estimate"}


def test_strong_buy_when_profitable_even_at_low_estimate():
    d = assess_deal(VAL, asking_price=100)
    assert d["verdict"] == "strong_buy"
    assert d["margin_low"] == pytest.approx(1.0)  # 200 / 100 - 1


def test_good_buy_when_only_typical_value_clears_margin():
    d = assess_deal(VAL, asking_price=200)  # low: 0 %, mid: +50 %
    assert d["verdict"] == "good_buy"
    assert d["profit_low"] == 0.0


def test_fair_and_overpriced():
    assert assess_deal(VAL, asking_price=280)["verdict"] == "fair"
    d = assess_deal(VAL, asking_price=400)
    assert d["verdict"] == "overpriced" and d["margin_mid"] < 0


def test_costs_and_resale_fees_change_the_verdict():
    cheap = assess_deal(VAL, asking_price=140)
    assert cheap["verdict"] == "strong_buy"
    costly = assess_deal(VAL, asking_price=140, shipping=40, buyer_premium_pct=25,
                         resale_fee_pct=15, resale_shipping=10, restoration=30)
    assert costly["all_in_cost"] == pytest.approx(140 * 1.25 + 70)
    assert costly["net_low"] == pytest.approx(200 * 0.85 - 10)
    assert costly["verdict"] != "strong_buy"


def test_keeping_ignores_selling_costs():
    d = assess_deal(VAL, asking_price=250, resale_fee_pct=50, for_resale=False)
    assert d["net_mid"] == 300.0 and d["verdict"] == "fair"


def test_weak_evidence_doubles_required_margin_and_warns():
    weak = dict(VAL, num_comparables=4, effective_comparables=2.0)
    d = assess_deal(weak, asking_price=150)  # mid +100 %, low +33 %
    assert d["evidence"] == "weak" and d["required_margin"] == pytest.approx(0.6)
    assert d["verdict"] == "good_buy"  # low margin 33 % < 60 %
    assert any("Weak evidence" in w for w in d["warnings"])


def test_low_identification_confidence_counts_as_weak():
    assert assess_deal(VAL, asking_price=100, identification_confidence=0.3)["evidence"] == "weak"


def test_suspiciously_cheap_warns():
    d = assess_deal(VAL, asking_price=20)
    assert any("Suspiciously cheap" in w for w in d["warnings"])


def test_no_valuation_or_bad_price_gives_no_verdict():
    assert assess_deal(None, asking_price=100)["verdict"] == "no_verdict"
    assert assess_deal(dict(VAL, valuation_available=False), asking_price=100)["verdict"] == "no_verdict"
    assert assess_deal(VAL, asking_price=0)["verdict"] == "no_verdict"


def test_calibration_factor_scales_values_within_bounds():
    d = assess_deal(VAL, asking_price=100, calibration_factor=0.8)
    assert d["value_mid"] == pytest.approx(240.0)
    assert assess_deal(VAL, asking_price=100, calibration_factor=9)["calibration_factor"] == 1.5


def test_max_asking_prices_hit_the_required_margin():
    d = assess_deal(VAL, asking_price=500, shipping=10, buyer_premium_pct=20)
    again = assess_deal(VAL, asking_price=d["max_asking_strong_buy"], shipping=10, buyer_premium_pct=20)
    assert again["margin_low"] == pytest.approx(0.30, abs=0.01)


def test_format_deal_mentions_verdict_and_warnings():
    text = "\n".join(format_deal(assess_deal(VAL, asking_price=100)))
    assert "STRONG BUY" in text and "Condition, repairs" in text
    assert format_deal(None) == []
