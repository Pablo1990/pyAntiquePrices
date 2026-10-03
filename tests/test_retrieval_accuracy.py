"""Accuracy-oriented tests: unknown != mismatch, fuzzy makers, outliers."""

from __future__ import annotations

import pytest

from pyantique_prices.data.database import create_tables, get_engine, get_session_factory
from pyantique_prices.data.models import HistoricalSale
from pyantique_prices.data.outliers import flag_outliers
from pyantique_prices.retrieval.comparables import retrieve_comparables_details
from pyantique_prices.retrieval.ranking import (
    explain_structured_similarity,
    manufacturer_similarity,
)


def _session():
    engine = get_engine("sqlite:///:memory:")
    create_tables(engine)
    return get_session_factory(engine)()


def test_missing_sale_metadata_is_not_a_mismatch():
    query = {"object_type": "vase", "country": "France", "materials": ["glass"]}
    sparse = {"object_type": "vase"}  # country/material unknown
    wrong = {"object_type": "vase", "country": "Italy", "materials": ["wood"]}
    sparse_score, _ = explain_structured_similarity(query, sparse)
    wrong_score, _ = explain_structured_similarity(query, wrong)
    assert sparse_score > wrong_score  # a real mismatch must rank below unknown


def test_more_evidence_beats_single_field_match():
    query = {"object_type": "vase", "country": "France", "materials": ["glass"]}
    one = {"object_type": "vase"}
    many = {"object_type": "vase", "country": "France", "materials": ["glass"]}
    assert explain_structured_similarity(query, many)[0] > explain_structured_similarity(query, one)[0]


def test_manufacturer_match_is_fuzzy_and_confidence_aware():
    sale = {"manufacturer": "Meissen Porcelain Manufactory"}
    assert manufacturer_similarity({"manufacturer_candidates": [{"name": "Meissen", "confidence": 0.8}]}, sale) >= 0.85
    # A 10 %-confidence guess must not create a "same manufacturer" match
    assert manufacturer_similarity({"manufacturer_candidates": [{"name": "Meissen", "confidence": 0.1}]}, sale) == 0.0


def test_no_information_gives_zero_not_error():
    assert explain_structured_similarity({}, {}) == (0.0, [])


def test_flag_outliers_marks_extreme_price_and_retrieval_skips_it():
    with _session() as session:
        for price in (100, 110, 95, 105, 120, 98, 102, 115):
            session.add(HistoricalSale(title="Clock", object_type="clock", normalized_price=float(price),
                                       source_url=f"u{price}", usable_for_training=True))
        session.add(HistoricalSale(title="Clock typo", object_type="clock", normalized_price=1_000_000.0,
                                   source_url="typo", usable_for_training=True))
        session.commit()
        assert flag_outliers(session) == 1
        flagged = session.query(HistoricalSale).filter_by(outlier_flag=True).one()
        assert flagged.title == "Clock typo" and "z-score" in flagged.outlier_reason

        details = retrieve_comparables_details(
            session, {"object_type": "clock"}, top_k=20, min_similarity=0.0, min_data_quality_score=0.0
        )
        assert details["candidate_count"] == 8
        assert all(c["normalized_price"] < 1000 for c in details["comparables"])


def test_identical_prices_are_not_outliers():
    with _session() as session:
        for i in range(10):
            session.add(HistoricalSale(title="x", object_type="a", normalized_price=50.0, source_url=str(i)))
        session.commit()
        assert flag_outliers(session) == 0
