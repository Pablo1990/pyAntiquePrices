"""Comparable retrieval from local historical sales database."""

from __future__ import annotations

import datetime

from .ranking import compute_structured_similarity

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


def _data_quality_score(sale: dict) -> float:
    checks = [
        bool(sale.get("title")),
        bool(sale.get("object_type")),
        bool(sale.get("country")),
        bool(sale.get("source_url")),
        sale.get("normalized_price") is not None,
    ]
    return sum(1 for item in checks if item) / len(checks)


def _is_recent_enough(sale: dict, max_sale_age_years: int) -> bool:
    sale_date = sale.get("sale_date")
    if not sale_date:
        return True
    try:
        dt = datetime.datetime.fromisoformat(str(sale_date))
    except ValueError:
        return True
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=max_sale_age_years * 365)
    return dt >= cutoff


def retrieve_comparables_details(
    session,
    identification: dict,
    *,
    top_k: int = 50,
    min_similarity: float = 0.05,
    max_sale_age_years: int = 80,
    min_data_quality_score: float = 0.4,
    weights: dict | None = None,
) -> dict:
    """Retrieve and filter comparable sales, returning counts and results."""
    from pyantique_prices.data.models import HistoricalSale

    sales = (
        session.query(HistoricalSale)
        .filter(
            HistoricalSale.normalized_price.is_not(None),
            HistoricalSale.usable_for_training.is_(True),
        )
        .all()
    )

    candidate_count = len(sales)
    scored: list[tuple[float, dict]] = []
    for sale in sales:
        materials = []
        if sale.material:
            materials = [item.strip().lower() for item in str(sale.material).split(",") if item.strip()]
        sale_dict = {
            "id": sale.id,
            "title": sale.title,
            "object_type": sale.object_type,
            "manufacturer": sale.manufacturer,
            "period": sale.period,
            "materials": materials,
            "country": sale.country,
            "condition": sale.condition,
            "normalized_price": sale.normalized_price,
            "sale_date": sale.sale_date.isoformat() if sale.sale_date else None,
            "auction_house": sale.auction_house,
            "source_url": sale.source_url,
        }
        semantic_score = score_comparable(identification, sale_dict, weights=weights)
        ranking_score = compute_structured_similarity(
            identification=identification,
            comparable=sale_dict,
            semantic_similarity=semantic_score,
            weights=weights,
        )
        sale_dict["retrieval_score"] = round(ranking_score, 6)
        sale_dict["semantic_score"] = round(semantic_score, 6)
        sale_dict["data_quality_score"] = round(_data_quality_score(sale_dict), 6)
        scored.append((ranking_score, sale_dict))

    scored.sort(key=lambda item: item[0], reverse=True)
    filtered = []
    for _, sale_dict in scored:
        if sale_dict["retrieval_score"] < min_similarity:
            continue
        if sale_dict.get("normalized_price") is None:
            continue
        if not _is_recent_enough(sale_dict, max_sale_age_years=max_sale_age_years):
            continue
        if sale_dict["data_quality_score"] < min_data_quality_score:
            continue
        filtered.append(sale_dict)
        if len(filtered) >= top_k:
            break

    return {
        "candidate_count": candidate_count,
        "usable_comparable_count": len(filtered),
        "comparables": filtered,
    }


def retrieve_comparables(session, identification: dict, top_k: int = 50) -> list[dict]:
    """Retrieve top-K comparable sales from the database."""
    details = retrieve_comparables_details(
        session,
        identification,
        top_k=top_k,
        min_similarity=0.0,
        min_data_quality_score=0.0,
    )
    return details["comparables"]
