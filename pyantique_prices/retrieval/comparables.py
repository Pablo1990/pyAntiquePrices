"""Comparable retrieval from local historical sales database."""

from __future__ import annotations

DEFAULT_WEIGHTS = {
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
        value = value.get("value")
    if value is None:
        return ""
    return str(value).strip().lower()


def score_comparable(
    identification: dict,
    sale: dict,
    weights: dict | None = None,
) -> float:
    """Compute a similarity score between identification and a historical sale."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    score = 0.0

    ident_object = _normalized_text(identification.get("object_type"))
    sale_object = _normalized_text(sale.get("object_type"))
    ident_country = _normalized_text(identification.get("country"))
    sale_country = _normalized_text(sale.get("country"))
    ident_condition = _normalized_text(identification.get("condition"))
    sale_condition = _normalized_text(sale.get("condition"))

    if ident_object and sale_object:
        if ident_object in sale_object:
            score += weights.get("object_type", 0.15)

    if ident_country and sale_country:
        if ident_country == sale_country:
            score += weights.get("country", 0.05)

    if ident_condition and sale_condition:
        if ident_condition == sale_condition:
            score += weights.get("condition", 0.05)

    return score


def retrieve_comparables(session, identification: dict, top_k: int = 50) -> list[dict]:
    """Retrieve top-K comparable sales from the database."""
    from pyantique_prices.data.models import HistoricalSale

    sales = (
        session.query(HistoricalSale)
        .filter(
            HistoricalSale.normalized_price.is_not(None),
            HistoricalSale.usable_for_training.is_(True),
        )
        .all()
    )

    scored: list[tuple[float, dict]] = []
    for sale in sales:
        sale_dict = {
            "id": sale.id,
            "title": sale.title,
            "object_type": sale.object_type,
            "country": sale.country,
            "condition": sale.condition,
            "normalized_price": sale.normalized_price,
            "sale_date": str(sale.sale_date) if sale.sale_date else None,
            "auction_house": sale.auction_house,
        }
        scored.append((score_comparable(identification, sale_dict), sale_dict))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:top_k]]
