from __future__ import annotations

from pyantique_prices.data.database import create_tables, get_engine, get_session_factory
from pyantique_prices.data.models import HistoricalSale
from pyantique_prices.pricing.model import PricePredictor
from pyantique_prices.services.appraisal import AppraisalService


class _FakeAnalyzer:
    def analyze(self, images, context: str = ""):  # noqa: ARG002
        return {"object_type": "clock", "country": "France", "condition": "good"}


class _EmptyAnalyzer:
    def analyze(self, images, context: str = ""):  # noqa: ARG002
        return None


class _FakeFallbackEstimator:
    def estimate(self, images, context: str = ""):  # noqa: ARG002
        return {
            "low": 80.0,
            "mid": 100.0,
            "high": 150.0,
            "p25": 80.0,
            "p50": 100.0,
            "p75": 150.0,
            "valuation_available": False,
            "method": "legacy_web_fallback",
            "confidence_note": "Very low confidence",
        }


def _make_session_factory():
    engine = get_engine("sqlite:///:memory:")
    create_tables(engine)
    session_factory = get_session_factory(engine)
    with session_factory() as session:
        session.add(
            HistoricalSale(
                title="French mantel clock",
                object_type="Mantel clock",
                country="France",
                condition="good",
                normalized_price=250.0,
                usable_for_training=True,
            )
        )
        session.commit()
    return session_factory


def test_appraisal_service_uses_reference_estimate_for_1_to_2_comparables():
    service = AppraisalService(
        analyzer=_FakeAnalyzer(),
        retrieval_session_factory=_make_session_factory(),
        pricer=PricePredictor(),
    )

    result = service.appraise(images=["/tmp/a.jpg", "/tmp/b.jpg", "/tmp/c.jpg"])

    assert result["valuation"] is not None
    assert result["valuation"]["method"] == "reference_only"
    assert result["valuation_available"] is False
    assert result["candidate_count"] == 1
    assert result["usable_comparable_count"] == 1
    assert result["identification_confidence"] > 0.0
    assert result["valuation_confidence"] > 0.0
    assert "Reference estimate only." in " ".join(result["warnings"])


def test_appraisal_service_uses_fallback_estimate_when_no_local_comparables():
    service = AppraisalService(
        analyzer=_EmptyAnalyzer(),
        fallback_estimator=_FakeFallbackEstimator(),
    )

    result = service.appraise(images=["/tmp/a.jpg", "/tmp/b.jpg", "/tmp/c.jpg"])

    assert result["valuation"] is not None
    assert result["valuation"]["method"] == "legacy_web_fallback"
    assert result["valuation"]["mid"] == 100.0
    assert result["valuation_available"] is False
    assert result["valuation_confidence"] == 0.1
    assert any("rough fallback estimate" in warning for warning in result["warnings"])
