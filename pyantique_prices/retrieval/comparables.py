"""Comparable retrieval from the local historical sales database."""

from __future__ import annotations

from datetime import date
from typing import Protocol

from pydantic import BaseModel, Field

from pyantique_prices.retrieval.documents import build_sale_search_document, build_search_document
from pyantique_prices.retrieval.ranking import (
    DEFAULT_SIGNAL_WEIGHTS,
    compute_overall_similarity,
    explain_structured_similarity,
)
from pyantique_prices.retrieval.vector_store import cosine_similarity


class ComparableResult(BaseModel):
    sale_id: str
    title: str
    auction_house: str | None = None
    sale_date: date | None = None
    price: float | None = None
    currency: str | None = None
    semantic_similarity: float
    visual_similarity: float | None = None
    structured_similarity: float
    overall_similarity: float
    match_reasons: list[str] = Field(default_factory=list)


class ComparableRetriever(Protocol):
    def search(
        self,
        query_text: str,
        query_image=None,
        top_k: int = 20,
    ) -> list[ComparableResult]:
        ...


def _safe_float_list(value) -> list[float]:
    if not isinstance(value, list):
        return []
    floats = []
    for item in value:
        if not isinstance(item, (int, float)):
            return []
        floats.append(float(item))
    return floats


def _lexical_similarity(left: str, right: str) -> float:
    left_tokens = {token for token in left.lower().split() if token}
    right_tokens = {token for token in right.lower().split() if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))


def _sale_to_dict(sale) -> dict:
    return {
        "id": sale.id,
        "title": sale.title,
        "description": sale.description,
        "category": sale.category,
        "subcategory": sale.subcategory,
        "object_type": sale.object_type,
        "manufacturer": sale.manufacturer,
        "artist": sale.artist,
        "workshop": sale.workshop,
        "period": sale.period,
        "materials": [
            item.strip().lower()
            for item in str(sale.material or "").split(",")
            if item.strip()
        ],
        "material": sale.material,
        "technique": sale.technique,
        "country": sale.country,
        "region": sale.region,
        "condition": sale.condition,
        "height": sale.height,
        "width": sale.width,
        "depth": sale.depth,
        "diameter": sale.diameter,
        "weight": sale.weight,
        "marks": [
            item.strip() for item in str(sale.marks or "").split(",") if item.strip()
        ],
        "provenance": sale.provenance,
        "normalized_price": sale.normalized_price,
        "price": sale.final_price or sale.hammer_price or sale.normalized_price,
        "currency": sale.currency or sale.normalized_currency,
        "sale_date": sale.sale_date,
        "auction_house": sale.auction_house,
        "source_url": sale.source_url,
        "image_urls": sale.image_urls,
        "text_embedding": _safe_float_list(sale.text_embedding),
        "image_embedding": _safe_float_list(sale.image_embedding),
    }


def score_comparable(
    identification: dict,
    sale: dict,
    weights: dict | None = None,
) -> float:
    query_text = build_search_document(identification)
    sale_text = build_sale_search_document(sale)
    semantic_similarity = _lexical_similarity(query_text, sale_text)
    structured_similarity, _ = explain_structured_similarity(identification, sale)
    return compute_overall_similarity(
        semantic_similarity=semantic_similarity,
        structured_similarity=structured_similarity,
        weights=weights or DEFAULT_SIGNAL_WEIGHTS,
    )


def _data_quality_score(sale: dict) -> float:
    checks = [
        bool(sale.get("title")),
        bool(sale.get("object_type")),
        bool(sale.get("price")),
        bool(sale.get("source_url")),
        bool(build_sale_search_document(sale)),
    ]
    return sum(1 for item in checks if item) / len(checks)


def _build_query_image_embedding(query_image, image_embedding_provider):
    if query_image is None or image_embedding_provider is None:
        return None
    return _safe_float_list(image_embedding_provider.embed(query_image))


class HybridComparableRetriever:
    """Hybrid comparable search over local HistoricalSale rows."""

    def __init__(
        self,
        session,
        *,
        identification: dict,
        text_embedding_provider=None,
        image_embedding_provider=None,
        signal_weights: dict[str, float] | None = None,
    ) -> None:
        self.session = session
        self.identification = identification
        self.text_embedding_provider = text_embedding_provider
        self.image_embedding_provider = image_embedding_provider
        self.signal_weights = dict(DEFAULT_SIGNAL_WEIGHTS)
        if signal_weights:
            self.signal_weights.update(signal_weights)

    def _semantic_similarity(self, query_text: str, sale: dict, query_embedding: list[float]) -> float:
        if query_embedding:
            sale_embedding = sale.get("text_embedding") or []
            if sale_embedding:
                return cosine_similarity(query_embedding, sale_embedding)
        return _lexical_similarity(query_text, build_sale_search_document(sale))

    def search(self, query_text: str, query_image=None, top_k: int = 20) -> list[ComparableResult]:
        from pyantique_prices.data.models import HistoricalSale

        query_embedding = []
        if self.text_embedding_provider is not None:
            query_embedding = _safe_float_list(self.text_embedding_provider.embed(query_text))
        query_image_embedding = _build_query_image_embedding(
            query_image,
            self.image_embedding_provider,
        )
        rows = (
            self.session.query(HistoricalSale)
            .filter(
                HistoricalSale.normalized_price.is_not(None),
                HistoricalSale.usable_for_training.is_(True),
            )
            .all()
        )
        results: list[ComparableResult] = []
        for row in rows:
            sale = _sale_to_dict(row)
            semantic_similarity = self._semantic_similarity(query_text, sale, query_embedding)
            structured_similarity, reasons = explain_structured_similarity(
                self.identification,
                sale,
            )
            visual_similarity = None
            if query_image_embedding is not None and sale.get("image_embedding"):
                visual_similarity = cosine_similarity(
                    query_image_embedding,
                    sale["image_embedding"],
                )
            overall_similarity = compute_overall_similarity(
                semantic_similarity=semantic_similarity,
                structured_similarity=structured_similarity,
                visual_similarity=visual_similarity,
                weights=self.signal_weights,
            )
            results.append(
                ComparableResult(
                    sale_id=str(sale["id"]),
                    title=sale.get("title") or "Untitled",
                    auction_house=sale.get("auction_house"),
                    sale_date=sale.get("sale_date").date() if sale.get("sale_date") else None,
                    price=sale.get("price"),
                    currency=sale.get("currency"),
                    semantic_similarity=round(semantic_similarity, 6),
                    visual_similarity=round(visual_similarity, 6)
                    if visual_similarity is not None
                    else None,
                    structured_similarity=round(structured_similarity, 6),
                    overall_similarity=round(overall_similarity, 6),
                    match_reasons=reasons,
                )
            )
        results.sort(key=lambda item: item.overall_similarity, reverse=True)
        return results[:top_k]


def retrieve_comparables_details(
    session,
    identification: dict,
    *,
    top_k: int = 20,
    min_similarity: float = 0.05,
    max_sale_age_years: int = 80,
    min_data_quality_score: float = 0.4,
    weights: dict | None = None,
    text_embedding_provider=None,
    image_embedding_provider=None,
    query_image=None,
) -> dict:
    """Retrieve and filter comparable sales, returning counts and results."""
    from datetime import datetime, timedelta
    from pyantique_prices.data.models import HistoricalSale

    query_text = build_search_document(identification)
    retriever = HybridComparableRetriever(
        session,
        identification=identification,
        text_embedding_provider=text_embedding_provider,
        image_embedding_provider=image_embedding_provider,
        signal_weights=weights,
    )
    results = retriever.search(query_text, query_image=query_image, top_k=max(top_k, 200))

    sales_by_id = {
        str(row.id): _sale_to_dict(row)
        for row in session.query(HistoricalSale)
        .filter(
            HistoricalSale.normalized_price.is_not(None),
            HistoricalSale.usable_for_training.is_(True),
        )
        .all()
    }
    candidate_count = len(sales_by_id)
    cutoff = datetime.utcnow().date() - timedelta(days=max_sale_age_years * 365)
    filtered = []
    for comparable in results:
        sale = sales_by_id.get(comparable.sale_id, {})
        payload = comparable.model_dump()
        payload["id"] = int(comparable.sale_id)
        payload["normalized_price"] = sale.get("normalized_price")
        payload["retrieval_score"] = comparable.overall_similarity
        payload["semantic_score"] = comparable.semantic_similarity
        payload["source_url"] = sale.get("source_url")
        payload["image_urls"] = sale.get("image_urls")
        payload["data_quality_score"] = _data_quality_score(sale)
        if comparable.sale_date and comparable.sale_date < cutoff:
            continue
        if comparable.overall_similarity < min_similarity:
            continue
        if payload["data_quality_score"] < min_data_quality_score:
            continue
        filtered.append(payload)
        if len(filtered) >= top_k:
            break

    return {
        "candidate_count": candidate_count,
        "usable_comparable_count": len(filtered),
        "comparables": filtered,
    }


def retrieve_comparables(session, identification: dict, top_k: int = 20) -> list[dict]:
    """Retrieve top-K comparable sales from the database."""
    details = retrieve_comparables_details(
        session,
        identification,
        top_k=top_k,
        min_similarity=0.0,
        min_data_quality_score=0.0,
    )
    return details["comparables"]
