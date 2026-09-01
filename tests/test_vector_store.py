from __future__ import annotations

from pyantique_prices.retrieval.ranking import compute_structured_similarity
from pyantique_prices.retrieval.vector_store import InMemoryVectorStore, cosine_similarity


def test_cosine_similarity_handles_empty_or_shape_mismatch():
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


def test_in_memory_vector_store_add_search_delete():
    store = InMemoryVectorStore()
    store.add(1, [1.0, 0.0], {"title": "clock"})
    store.add(2, [0.0, 1.0], {"title": "vase"})
    results = store.search([0.9, 0.1], top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == 1
    store.delete(1)
    assert store.search([0.9, 0.1], top_k=2)[0]["id"] == 2


def test_compute_structured_similarity_combines_signals():
    score = compute_structured_similarity(
        identification={
            "object_type": {"value": "clock"},
            "country": {"value": "France"},
            "condition": {"value": "good"},
            "materials": ["bronze"],
            "likely_period": {"value": "19th"},
            "manufacturer_candidates": [{"name": "Japy Freres"}],
        },
        comparable={
            "object_type": "Mantel clock",
            "country": "France",
            "condition": "good",
            "period": "19th century",
            "materials": ["bronze"],
            "manufacturer": "Japy Freres",
        },
        semantic_similarity=0.8,
    )
    assert score > 0.5
