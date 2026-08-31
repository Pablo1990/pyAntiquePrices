from __future__ import annotations

from pyantique_prices.data.database import create_tables, get_engine, get_session_factory
from pyantique_prices.data.models import HistoricalSale
from pyantique_prices.retrieval.comparables import retrieve_comparables, score_comparable


def _make_session():
    engine = get_engine("sqlite:///:memory:")
    create_tables(engine)
    session_factory = get_session_factory(engine)
    return session_factory()


def test_score_comparable_rewards_matching_fields():
    score = score_comparable(
        {"object_type": "clock", "country": "France", "condition": "good"},
        {"object_type": "Mantel Clock", "country": "France", "condition": "good"},
    )

    assert score == 0.25


def test_retrieve_comparables_orders_by_score():
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
                ),
                HistoricalSale(
                    title="German vase",
                    object_type="Porcelain vase",
                    country="Germany",
                    condition="fair",
                    normalized_price=180.0,
                    usable_for_training=True,
                ),
            ]
        )
        session.commit()

        results = retrieve_comparables(
            session,
            {"object_type": "clock", "country": "France", "condition": "good"},
            top_k=2,
        )

    assert len(results) == 2
    assert results[0]["title"] == "French mantel clock"
