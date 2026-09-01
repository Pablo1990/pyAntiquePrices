from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from pyantique_prices.api.app import create_app
from pyantique_prices.config import Settings
from pyantique_prices.data.models import HistoricalSale


class _FakeAppraisalService:
    def appraise(self, images, context: str = "", currency: str | None = None):
        assert len(images) == 3
        assert all(Path(path).exists() for path in images)
        return {
            "request_id": "req-123",
            "identification": {
                "object_type": {"value": "Mantel clock", "confidence": 0.8},
                "condition": {"value": "good", "confidence": 0.7},
                "marks": [{"text": "Japy Frères", "confidence": 0.76}],
            },
            "comparables": [
                {"id": 99, "title": "Comparable clock", "normalized_price": 1200.0}
            ],
            "valuation": {"low": 1000.0, "mid": 1200.0, "high": 1500.0},
            "valuation_available": True,
            "warnings": [],
            "identification_confidence": 0.82,
            "valuation_confidence": 0.65,
            "currency": currency or "EUR",
            "evidence": [{"claim": "Clock type", "source": "vision"}],
            "candidate_count": 12,
            "usable_comparable_count": 8,
        }


def _make_client(tmp_path):
    db_path = tmp_path / "test.db"
    settings = Settings(database_url=f"sqlite:///{db_path}")
    app = create_app(settings=settings)
    app.state.appraisal_service = _FakeAppraisalService()
    return TestClient(app), app


def test_health_and_models(tmp_path):
    client, _ = _make_client(tmp_path)
    assert client.get("/health").json() == {"status": "ok"}
    body = client.get("/models").json()
    assert body["vision_model"]
    assert body["pricing_model"] == "price_predictor_v1"


def test_appraise_requires_between_3_and_5_images(tmp_path):
    client, _ = _make_client(tmp_path)
    files = [
        ("images", ("img1.jpg", b"a", "image/jpeg")),
        ("images", ("img2.jpg", b"b", "image/jpeg")),
    ]
    response = client.post("/appraise", files=files)
    assert response.status_code == 400


def test_appraise_persists_and_can_be_loaded(tmp_path):
    client, _ = _make_client(tmp_path)
    files = [
        ("images", ("img1.jpg", b"a", "image/jpeg")),
        ("images", ("img2.png", b"b", "image/png")),
        ("images", ("img3.webp", b"c", "image/webp")),
    ]
    response = client.post(
        "/appraise",
        files=files,
        data={"currency": "EUR", "location": "Madrid"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req-123"
    assert body["valuation"]["mid"] == 1200.0
    assert body["confidence"]["identification_confidence"] == 0.82
    assert body["candidate_count"] == 12
    assert body["usable_comparable_count"] == 8

    saved = client.get("/appraisals/1")
    assert saved.status_code == 200
    saved_body = saved.json()
    assert saved_body["request_id"] == "req-123"
    assert saved_body["comparable_ids"] == [99]


def test_get_sale_by_id(tmp_path):
    client, app = _make_client(tmp_path)
    with app.state.session_factory() as session:
        sale = HistoricalSale(
            title="French clock",
            object_type="Mantel clock",
            currency="EUR",
            final_price=900.0,
            normalized_currency="EUR",
            normalized_price=900.0,
            source_url="https://example.com/sale/1",
        )
        session.add(sale)
        session.commit()
        sale_id = sale.id

    response = client.get(f"/sales/{sale_id}")
    assert response.status_code == 200
    assert response.json()["title"] == "French clock"
