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


def score_comparable(
    identification: dict,
    sale: dict,
    weights: dict | None = None,
) -> float:
    """Compute a similarity score between identification and a historical sale."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    score = 0.0

    if identification.get("object_type") and sale.get("object_type"):
        if identification["object_type"].lower() in sale["object_type"].lower():
            score += weights.get("object_type", 0.15)

    if identification.get("country") and sale.get("country"):
        if identification["country"].lower() == sale["country"].lower():
            score += weights.get("country", 0.05)

    if identification.get("condition") and sale.get("condition"):
        if identification["condition"].lower() == sale["condition"].lower():
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
