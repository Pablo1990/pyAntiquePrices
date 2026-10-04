"""Research links + deal verdict through AppraisalService and the REST API."""

from __future__ import annotations

import datetime

from fastapi.testclient import TestClient

from pyantique_prices.api.app import create_app
from pyantique_prices.config import Settings
from pyantique_prices.data.database import create_tables, get_engine, get_session_factory
from pyantique_prices.data.models import HistoricalSale
from pyantique_prices.ledger import add_purchase, record_sale
from pyantique_prices.pricing.model import PricePredictor
from pyantique_prices.services.appraisal import AppraisalService


class _Analyzer:
    def analyze(self, images, context: str = ""):  # noqa: ARG002
        return {"object_type": "mantel clock", "manufacturer_candidates": [{"name": "Japy", "confidence": 0.8}],
                "country": "France", "condition": "good"}


def _factory(n=8, price=300.0):
    engine = get_engine("sqlite:///:memory:")
    create_tables(engine)
    factory = get_session_factory(engine)
    with factory() as s:
        for i in range(n):
            s.add(HistoricalSale(title=f"Japy mantel clock {i}", object_type="mantel clock", manufacturer="Japy",
                                 country="France", condition="good", normalized_price=price + i, source_url=f"u{i}",
                                 sale_date=datetime.datetime(2025, 1, 1), usable_for_training=True))
        s.commit()
    return factory


def _service(factory, **kw):
    return AppraisalService(analyzer=_Analyzer(), retrieval_session_factory=factory,
                            pricer=PricePredictor(model_dir="/none"), **kw)


def test_result_always_has_research_links_and_no_deal_without_price():
    result = _service(_factory()).appraise(["a.jpg"])
    ids = {l["id"] for l in result["lookup_links"]["links"]}
    assert "ebay_sold" in ids and result["deal"] is None


def test_ebay_domain_is_configurable():
    result = _service(_factory(), ebay_domain="ebay.co.uk").appraise(["a.jpg"])
    link = next(l for l in result["lookup_links"]["links"] if l["id"] == "ebay_sold")
    assert "ebay.co.uk" in link["url"]


def test_asking_price_adds_deal_verdict_and_uses_costs():
    svc = _service(_factory())
    cheap = svc.appraise(["a.jpg"], asking_price=60)
    assert cheap["valuation_available"] and cheap["deal"]["verdict"] in {"strong_buy", "good_buy"}
    dear = svc.appraise(["a.jpg"], asking_price=900, deal_options={"shipping": 20})
    assert dear["deal"]["verdict"] == "overpriced" and dear["deal"]["all_in_cost"] == 920


def test_deal_uses_ledger_calibration_when_enough_sales():
    factory = _factory()
    with factory() as s:
        for i in range(5):  # historically sold at half the estimate
            it = add_purchase(s, title=str(i), price=10, valuation={"p50": 200.0})
            record_sale(s, it.id, price=100)
    deal = _service(factory).appraise(["a.jpg"], asking_price=100)["deal"]
    assert deal["calibration_factor"] == 0.5 and deal["value_mid"] < 200


def test_deal_without_valuation_is_no_verdict():
    result = _service(_factory(n=0)).appraise(["a.jpg"], asking_price=50)
    assert result["deal"]["verdict"] == "no_verdict"


class _Fake:
    def __init__(self):
        self.kwargs = None

    def appraise(self, images, context="", currency=None, **kwargs):
        self.kwargs = kwargs
        return {"request_id": "r", "identification": {}, "valuation": None, "warnings": [],
                "lookup_links": {"query": "q", "links": []}, "deal": {"verdict": "fair", "headline": "ok"}}


def test_api_passes_deal_form_fields_and_returns_deal(tmp_path):
    app = create_app(settings=Settings(database_url=f"sqlite:///{tmp_path/'t.db'}"))
    fake = _Fake()
    app.state.appraisal_service = fake
    files = [("images", (f"i{i}.jpg", b"x", "image/jpeg")) for i in range(3)]
    r = TestClient(app).post("/appraise", files=files, data={"asking_price": "120", "shipping": "15", "resale_fee_pct": "13"})
    assert r.status_code == 200 and r.json()["deal"]["verdict"] == "fair"
    assert fake.kwargs["asking_price"] == 120.0
    assert fake.kwargs["deal_options"]["shipping"] == 15.0 and fake.kwargs["deal_options"]["resale_fee_pct"] == 13.0


def test_api_without_asking_price_does_not_pass_deal_args(tmp_path):
    app = create_app(settings=Settings(database_url=f"sqlite:///{tmp_path/'t.db'}"))
    fake = _Fake()
    app.state.appraisal_service = fake
    files = [("images", (f"i{i}.jpg", b"x", "image/jpeg")) for i in range(3)]
    assert TestClient(app).post("/appraise", files=files).status_code == 200
    assert fake.kwargs == {}


def test_api_rejects_non_positive_asking_price(tmp_path):
    app = create_app(settings=Settings(database_url=f"sqlite:///{tmp_path/'t.db'}"))
    app.state.appraisal_service = _Fake()
    files = [("images", (f"i{i}.jpg", b"x", "image/jpeg")) for i in range(3)]
    assert TestClient(app).post("/appraise", files=files, data={"asking_price": "0"}).status_code == 422
