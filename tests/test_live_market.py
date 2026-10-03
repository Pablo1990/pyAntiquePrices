"""Tests for live, non-persisted eBay listings at appraisal time."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from pyantique_prices.services.appraisal import AppraisalService
from pyantique_prices.services.live_market import (
    EbayLiveListings,
    build_query,
    format_live_listings,
)

IDENT = {
    "object_type": {"value": "pocket watch", "confidence": 0.9},
    "subtype": {"value": "hunter case", "confidence": 0.6},
    "likely_period": {"value": "unknown", "confidence": 0.1},
    "manufacturer_candidates": [{"name": "Omega", "confidence": 0.7}],
    "materials": ["silver", "enamel"],
}

RECORDS = [
    {
        "title": "Omega silver hunter pocket watch",
        "final_price": 320.0,
        "currency": "EUR",
        "source_url": "https://www.ebay.es/itm/1",
        "image_url": "https://i.ebayimg.com/1.jpg",
        "condition": "Used",
        "price_basis": "asking",
    },
    {"title": "no url", "final_price": 1.0, "currency": "EUR", "source_url": None},
]


def _scraper(records=RECORDS):
    scraper = MagicMock()
    scraper.marketplace_id = "EBAY_ES"
    scraper.scrape.return_value = records
    return scraper


class TestBuildQuery:
    def test_uses_maker_type_subtype_material_and_skips_unknown(self):
        assert build_query(IDENT) == "Omega pocket watch hunter case silver"

    def test_caps_length_and_dedupes(self):
        ident = {"object_type": {"value": "a b c d e f g h i j"}, "subtype": {"value": "a b"}}
        assert build_query(ident) == "a b c d e f g h"

    def test_falls_back_to_context_keywords(self):
        assert build_query({}, "Extra keywords: reloj bolsillo") == "reloj bolsillo"

    def test_empty(self):
        assert build_query(None) == ""


class TestEbayLiveListings:
    def test_search_returns_in_memory_block(self):
        live = EbayLiveListings(scraper=_scraper(), max_results=5)
        block = live.search(IDENT)
        assert block["source"] == "eBay"
        assert block["marketplace"] == "EBAY_ES"
        assert block["price_basis"] == "asking"
        assert "not stored" in block["notice"]
        assert len(block["items"]) == 1  # item without URL dropped
        assert block["items"][0]["url"] == "https://www.ebay.es/itm/1"
        live.scraper.scrape.assert_called_once_with(
            "Omega pocket watch hunter case silver", max_results=5
        )

    def test_no_query_no_request(self):
        live = EbayLiveListings(scraper=_scraper())
        assert live.search({})["items"] == []
        live.scraper.scrape.assert_not_called()

    def test_from_settings_requires_keys_and_flag(self):
        base = dict(ebay_live_listings=True, ebay_client_id="id", ebay_client_secret="s",
                    ebay_live_max_results=3, ebay_live_mode="active")
        assert EbayLiveListings.from_settings(SimpleNamespace(**{**base, "ebay_live_listings": False})) is None
        assert EbayLiveListings.from_settings(SimpleNamespace(**{**base, "ebay_client_id": ""})) is None
        assert EbayLiveListings.from_settings(SimpleNamespace(**{**base, "ebay_live_mode": "bogus"})) is None

    def test_format(self):
        block = EbayLiveListings(scraper=_scraper()).search(IDENT)
        text = "\n".join(format_live_listings(block))
        assert "LIVE eBAY LISTINGS" in text and "320.00 EUR" in text
        assert "https://www.ebay.es/itm/1" in text
        assert format_live_listings(None) == []


class TestAppraisalServiceIntegration:
    def _service(self, live):
        analyzer = MagicMock()
        analyzer.analyze.return_value = IDENT
        return AppraisalService(analyzer=analyzer, live_market=live)

    def test_listings_shown_but_not_used_for_comparables_or_valuation(self):
        live = EbayLiveListings(scraper=_scraper())
        result = self._service(live).appraise(["img.jpg"])
        assert result["live_market_listings"]["items"][0]["price"] == 320.0
        assert result["comparables"] == []
        assert result["valuation"] is None

    def test_lookup_failure_is_a_warning(self):
        live = MagicMock()
        live.search.side_effect = RuntimeError("boom")
        result = self._service(live).appraise(["img.jpg"])
        assert result["live_market_listings"] is None
        assert any("Live eBay lookup failed" in w for w in result["warnings"])

    def test_not_persisted(self, tmp_path):
        from pyantique_prices.data.appraisals import save_appraisal
        from pyantique_prices.data.database import (
            create_tables, get_engine, get_session_factory,
        )
        engine = get_engine(f"sqlite:///{tmp_path / 'a.db'}")
        create_tables(engine)
        result = self._service(EbayLiveListings(scraper=_scraper())).appraise(["img.jpg"])
        with get_session_factory(engine)() as session:
            record = save_appraisal(session=session, result=result,
                                    input_metadata={}, model_versions={})
            stored = repr({c.name: getattr(record, c.name) for c in record.__table__.columns})
        assert "ebay.es/itm" not in stored
