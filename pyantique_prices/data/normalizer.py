"""Currency normalization utilities."""

from __future__ import annotations

from typing import Optional

DEFAULT_RATES: dict[str, float] = {
    "EUR": 1.0,
    "GBP": 1.17,
    "USD": 0.92,
    "CHF": 1.04,
    "CAD": 0.68,
    "AUD": 0.60,
    "JPY": 0.0062,
}

SUPPORTED_CURRENCIES = set(DEFAULT_RATES.keys())


def normalize_price(
    price: float,
    from_currency: str,
    to_currency: str = "EUR",
    rates: Optional[dict[str, float]] = None,
) -> Optional[float]:
    """Convert price from one currency to another using configurable rates."""
    if rates is None:
        rates = DEFAULT_RATES
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    if from_currency not in rates or to_currency not in rates:
        return None
    price_in_eur = price * rates[from_currency] / rates.get("EUR", 1.0)
    if to_currency == "EUR":
        return price_in_eur
    return price_in_eur / rates[to_currency] * rates.get("EUR", 1.0)
