from __future__ import annotations

from pyantique_prices.data.database import create_tables, get_engine, get_session_factory
from pyantique_prices.data.models import HistoricalSale
from pyantique_prices.retrieval.comparables import (
    HybridComparableRetriever,
    retrieve_comparables,
    retrieve_comparables_details,
    score_comparable,
)
from pyantique_prices.retrieval.documents import build_search_document
from pyantique_prices.retrieval.ranking import manufacturer_similarity, period_similarity


def _make_session():
    engine = get_engine("sqlite:///:memory:")
    create_tables(engine)
    session_factory = get_session_factory(engine)
    return session_factory()


def test_score_comparable_rewards_matching_fields():
    score = score_comparable(
        {
            "object_type": "clock",
            "country": "France",
            "condition": "good",
            "manufacturer_candidates": [{"name": "Japy Freres"}],
        },
        {"object_type": "Mantel Clock", "country": "France", "condition": "good"},
    )

    assert score > 0.3


def test_score_comparable_handles_structured_identification_values():
    score = score_comparable(
        {
            "object_type": {"value": "clock"},
            "country": {"value": "France"},
            "condition": {"value": "good"},
        },
        {"object_type": "Mantel Clock", "country": "France", "condition": "good"},
    )
    assert score > 0.2


def test_retrieve_comparables_orders_by_score():
    with _make_session() as session:
        session.add_all(
            [
                HistoricalSale(
                    title="French mantel clock",
                    object_type="Mantel clock",
                    country="France",
                    condition="good",
                    manufacturer="Japy Freres",
                    normalized_price=250.0,
                    usable_for_training=True,
                    text_embedding=[1.0, 0.0],
                ),
                HistoricalSale(
                    title="German vase",
                    object_type="Porcelain vase",
                    country="Germany",
                    condition="fair",
                    normalized_price=180.0,
                    usable_for_training=True,
                    text_embedding=[0.0, 1.0],
                ),
            ]
        )
        session.commit()

        results = retrieve_comparables(
            session,
            {
                "object_type": "clock",
                "country": "France",
                "condition": "good",
                "manufacturer_candidates": [{"name": "Japy Freres"}],
            },
            top_k=2,
        )

    assert len(results) == 2
    assert results[0]["title"] == "French mantel clock"


def test_retrieve_comparables_details_reports_counts():
    with _make_session() as session:
        session.add_all(
            [
                HistoricalSale(
                    title="French mantel clock",
                    object_type="Mantel clock",
                    country="France",
                    condition="good",
                    normalized_price=250.0,
                    usable_for_training=True,
                    source_url="https://example.com/clock-1",
                    text_embedding=[1.0, 0.0],
                ),
                HistoricalSale(
                    title="Unknown listing",
                    object_type=None,
                    country=None,
                    condition=None,
                    normalized_price=120.0,
                    usable_for_training=True,
                    source_url=None,
                    text_embedding=[0.0, 1.0],
                ),
            ]
        )
        session.commit()

        details = retrieve_comparables_details(
            session,
            {"object_type": "clock", "country": "France", "condition": "good"},
            top_k=5,
            min_similarity=0.01,
            min_data_quality_score=0.5,
        )
    assert details["candidate_count"] == 2
    assert details["usable_comparable_count"] == 1
    assert details["comparables"][0]["title"] == "French mantel clock"


def test_build_search_document_includes_normalized_fields():
    document = build_search_document(
        {
            "object_type": "French porcelain vase",
            "period": "late 19th century",
            "materials": ["porcelain", "gilt"],
            "marks": [{"text": "blue mark", "mark_type": "maker_mark"}],
        }
    )
    assert "French porcelain vase" in document
    assert "late 19th century" in document
    assert "blue mark" in document


def test_hybrid_retriever_uses_embedding_similarity():
    class _FakeTextEmbeddingProvider:
        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0] if "clock" in text.lower() else [0.0, 1.0]

    with _make_session() as session:
        session.add_all(
            [
                HistoricalSale(
                    title="French mantel clock",
                    object_type="Mantel clock",
                    manufacturer="Japy Freres",
                    country="France",
                    condition="good",
                    normalized_price=250.0,
                    usable_for_training=True,
                    text_embedding=[1.0, 0.0],
                ),
                HistoricalSale(
                    title="Studio pottery vase",
                    object_type="Pottery vase",
                    country="France",
                    condition="good",
                    normalized_price=200.0,
                    usable_for_training=True,
                    text_embedding=[0.0, 1.0],
                ),
            ]
        )
        session.commit()
        retriever = HybridComparableRetriever(
            session,
            identification={"object_type": "clock", "country": "France"},
            text_embedding_provider=_FakeTextEmbeddingProvider(),
        )

        results = retriever.search("French clock", top_k=2)

    assert results[0].title == "French mantel clock"
    assert results[0].semantic_similarity > results[1].semantic_similarity


def test_similarity_helpers_reward_manufacturer_and_period_matches():
    identification = {
        "manufacturer_candidates": [{"name": "Japy Freres"}],
        "period": "1880-1900",
        "estimated_year_start": 1880,
        "estimated_year_end": 1900,
    }
    sale = {"manufacturer": "Japy Freres", "period": "circa 1890"}
    assert manufacturer_similarity(identification, sale) == 1.0
    assert period_similarity(identification, sale) > 0.0
