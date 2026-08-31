from __future__ import annotations

from pyantique_prices.data.normalizer import normalize_price


def test_normalize_price_to_eur():
    assert normalize_price(100.0, "USD", "EUR") == 92.0


def test_normalize_price_between_non_eur_currencies():
    result = normalize_price(100.0, "GBP", "USD")

    assert result is not None
    assert round(result, 2) == round(117.0 / 0.92, 2)


def test_normalize_price_returns_none_for_unsupported_currency():
    assert normalize_price(100.0, "SEK", "EUR") is None
