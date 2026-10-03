"""Tests for pyantique_prices.scraping.sources."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyantique_prices.scraping.sources import (
    AICScraper,
    CatawikiScraper,
    EbayApiError,
    EbayApiScraper,
    EbayEsScraper,
    LibraryOfCongressScraper,
    _parse_date_loose,
    _parse_price,
)


# ---------------------------------------------------------------------------
# _parse_price
# ---------------------------------------------------------------------------

class TestParsePrice:
    def test_euro_symbol(self):
        price, currency = _parse_price("€ 1.234,50")
        assert currency == "EUR"
        assert abs(price - 1234.50) < 0.01

    def test_pound_symbol(self):
        price, currency = _parse_price("£450")
        assert currency == "GBP"
        assert price == 450.0

    def test_dollar_symbol(self):
        price, currency = _parse_price("$1,000.00")
        assert currency == "USD"
        assert price == 1000.0

    def test_eur_code(self):
        price, currency = _parse_price("EUR 300")
        assert currency == "EUR"
        assert price == 300.0

    def test_empty_string(self):
        assert _parse_price("") == (None, None)

    def test_no_number(self):
        price, currency = _parse_price("€ n/a")
        assert price is None
        assert currency == "EUR"


# ---------------------------------------------------------------------------
# _parse_date_loose
# ---------------------------------------------------------------------------

class TestParseDateLoose:
    def test_iso_date(self):
        dt = _parse_date_loose("2023-06-15")
        assert dt is not None
        assert dt.year == 2023 and dt.month == 6 and dt.day == 15

    def test_iso_datetime(self):
        dt = _parse_date_loose("2022-11-01T00:00:00")
        assert dt is not None
        assert dt.year == 2022

    def test_dmy_slash(self):
        dt = _parse_date_loose("25/12/2021")
        assert dt is not None
        assert dt.day == 25 and dt.month == 12

    def test_month_name(self):
        dt = _parse_date_loose("January 15, 2023")
        assert dt is not None
        assert dt.month == 1

    def test_empty_returns_none(self):
        assert _parse_date_loose("") is None

    def test_garbage_returns_none(self):
        assert _parse_date_loose("not-a-date") is None


# ---------------------------------------------------------------------------
# EbayEsScraper
# ---------------------------------------------------------------------------

class TestEbayEsScraper:
    def _make_scraper(self, allowed: bool = True) -> EbayEsScraper:
        scraper = EbayEsScraper(crawl_delay=0)
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = allowed
        mock_rp.crawl_delay.return_value = None
        scraper._robots = mock_rp
        return scraper

    def test_skips_when_disallowed(self):
        scraper = self._make_scraper(allowed=False)
        result = scraper.scrape("reloj")
        assert result == []

    def test_returns_empty_on_fetch_failure(self):
        scraper = self._make_scraper(allowed=True)
        scraper._fetch = MagicMock(return_value=None)
        result = scraper.scrape("reloj")
        assert result == []

    def test_parses_listings(self):
        html = """
        <html><body>
          <div class="s-item">
            <span class="s-item__title">Victorian pocket watch</span>
            <span class="s-item__price">€120</span>
            <span class="s-item__ended-date">2023-04-10</span>
            <a class="s-item__link" href="https://www.ebay.es/itm/12345">Link</a>
          </div>
          <div class="s-item">
            <span class="s-item__title">Shop on eBay</span>
          </div>
        </body></html>
        """
        items = EbayEsScraper._parse_listings(html)
        assert len(items) == 1
        assert "Victorian" in items[0]["title"]
        assert items[0]["currency"] == "EUR"
        assert items[0]["final_price"] == 120.0
        assert items[0]["auction_house"] == "eBay.es"

    def test_respects_max_results(self):
        cards = "".join(
            f'<div class="s-item"><span class="s-item__title">Item {i}</span></div>'
            for i in range(20)
        )
        html = f"<html><body>{cards}</body></html>"
        scraper = self._make_scraper(allowed=True)
        scraper._fetch = MagicMock(return_value=html)
        result = scraper.scrape("reloj", max_results=5)
        assert len(result) <= 5


# ---------------------------------------------------------------------------
# CatawikiScraper
# ---------------------------------------------------------------------------

class TestCatawikiScraper:
    def _make_scraper(self, allowed: bool = True) -> CatawikiScraper:
        scraper = CatawikiScraper(crawl_delay=0)
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = allowed
        mock_rp.crawl_delay.return_value = None
        scraper._robots = mock_rp
        return scraper

    def test_skips_when_disallowed(self):
        scraper = self._make_scraper(allowed=False)
        result = scraper.scrape("porcelana")
        assert result == []

    def test_returns_empty_on_fetch_failure(self):
        scraper = self._make_scraper(allowed=True)
        scraper._fetch = MagicMock(return_value=None)
        result = scraper.scrape("porcelana")
        assert result == []

    def test_parses_listings(self):
        html = """
        <html><body>
          <article class="lot-card">
            <h3 class="lot-card__title">Art Deco bronze lamp</h3>
            <span class="lot-card__price">€ 340,00</span>
            <time datetime="2022-09-05T18:00:00Z">5 Sep 2022</time>
            <a href="/en/l/123456-art-deco-bronze-lamp">View</a>
          </article>
        </body></html>
        """
        items = CatawikiScraper._parse_listings(html)
        assert len(items) == 1
        assert "Art Deco" in items[0]["title"]
        assert items[0]["currency"] == "EUR"
        assert items[0]["auction_house"] == "Catawiki"


# ---------------------------------------------------------------------------
# AICScraper
# ---------------------------------------------------------------------------

class TestAICScraper:
    def _make_scraper(self, allowed: bool = True) -> AICScraper:
        scraper = AICScraper(crawl_delay=0)
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = allowed
        mock_rp.crawl_delay.return_value = None
        scraper._robots = mock_rp
        return scraper

    def test_skips_when_disallowed(self):
        scraper = self._make_scraper(allowed=False)
        result = scraper.scrape("painting")
        assert result == []

    def test_returns_empty_on_fetch_failure(self):
        scraper = self._make_scraper(allowed=True)
        scraper._fetch = MagicMock(return_value=None)
        result = scraper.scrape("painting")
        assert result == []

    def test_parses_resources(self):
        html = """
        <html><body>
          <article>
            <h3><a href="/resources/conservation-of-metals">Conservation of Metals</a></h3>
          </article>
          <article>
            <h3><a href="/resources/textile-care">Textile Care Guide</a></h3>
          </article>
        </body></html>
        """
        items = AICScraper._parse_resources(html, "metals")
        assert len(items) == 2
        assert items[0]["auction_house"] == "AIC"
        assert items[0]["price_basis"] == "reference"
        assert items[0]["final_price"] is None


# ---------------------------------------------------------------------------
# LibraryOfCongressScraper
# ---------------------------------------------------------------------------

class TestLibraryOfCongressScraper:
    def _make_scraper(self, allowed: bool = True) -> LibraryOfCongressScraper:
        scraper = LibraryOfCongressScraper(crawl_delay=0)
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = allowed
        mock_rp.crawl_delay.return_value = None
        scraper._robots = mock_rp
        return scraper

    def test_skips_when_disallowed(self):
        scraper = self._make_scraper(allowed=False)
        result = scraper.scrape("preservation")
        assert result == []

    def test_returns_empty_on_fetch_failure(self):
        scraper = self._make_scraper(allowed=True)
        scraper._fetch = MagicMock(return_value=None)
        result = scraper.scrape("preservation")
        assert result == []

    def test_parses_resources(self):
        html = """
        <html><body>
          <li class="item-description">
            <h3><a href="/preservation/books-paper">Books &amp; Paper</a></h3>
            <time>2021-03-01</time>
          </li>
        </body></html>
        """
        items = LibraryOfCongressScraper._parse_resources(html, "paper")
        assert len(items) == 1
        assert items[0]["auction_house"] == "Library of Congress"
        assert items[0]["final_price"] is None
        assert items[0]["sale_date"] is not None


# ---------------------------------------------------------------------------
# EbayApiScraper
# ---------------------------------------------------------------------------

def _resp(status: int, payload: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.ok = 200 <= status < 300
    r.json.return_value = payload or {}
    r.text = str(payload)
    return r


class TestEbayApiScraper:
    def _make_scraper(self, mode: str = "sold", allowed: bool = True, **kw) -> EbayApiScraper:
        scraper = EbayApiScraper(
            crawl_delay=0,
            client_id=kw.pop("client_id", "id"),
            client_secret=kw.pop("client_secret", "secret"),
            marketplace_id="EBAY_ES",
            environment="production",
            mode=mode,
            **kw,
        )
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = allowed
        mock_rp.crawl_delay.return_value = None
        scraper._robots = mock_rp
        scraper._session = MagicMock()
        scraper._session.post.return_value = _resp(
            200, {"access_token": "tok", "expires_in": 7200}
        )
        return scraper

    def test_skips_when_robots_disallows(self):
        scraper = self._make_scraper(allowed=False)
        assert scraper.scrape("reloj") == []
        scraper._session.get.assert_not_called()

    def test_missing_credentials_raises(self):
        scraper = self._make_scraper(client_id="", client_secret="")
        scraper.client_id = scraper.client_secret = ""
        with pytest.raises(EbayApiError):
            scraper.scrape("reloj")

    def test_oauth_scope_error_raises_with_hint(self):
        scraper = self._make_scraper()
        scraper._session.post.return_value = _resp(
            400, {"error": "invalid_scope", "error_description": "The requested scope is invalid"}
        )
        with pytest.raises(EbayApiError, match="Marketplace Insights"):
            scraper.scrape("reloj")

    def test_sold_mode_parses_item_sales(self):
        scraper = self._make_scraper(mode="sold")
        scraper._session.get.return_value = _resp(
            200,
            {
                "total": 1,
                "itemSales": [
                    {
                        "itemId": "v1|1|0",
                        "title": "Reloj de bolsillo plata 1890",
                        "lastSoldPrice": {"value": "145.50", "currency": "EUR"},
                        "lastSoldDate": "2024-03-01T10:15:30.000Z",
                        "itemWebUrl": "https://www.ebay.es/itm/1",
                        "categories": [{"categoryId": "20081", "categoryName": "Antigüedades"}],
                    }
                ],
            },
        )
        items = scraper.scrape("reloj bolsillo")
        assert len(items) == 1
        item = items[0]
        assert item["final_price"] == 145.50
        assert item["currency"] == "EUR"
        assert item["sale_date"] == "2024-03-01T10:15:30"
        assert item["price_basis"] == "realized"
        assert item["auction_house"] == "eBay.es"
        assert item["category"] == "Antigüedades"

        _, kwargs = scraper._session.get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer tok"
        assert kwargs["headers"]["X-EBAY-C-MARKETPLACE-ID"] == "EBAY_ES"
        assert kwargs["params"]["category_ids"] == "20081"
        assert "marketplace_insights" in scraper._session.get.call_args[0][0]
        # Insights scope requested for sold mode
        assert "buy.marketplace.insights" in scraper._session.post.call_args.kwargs["data"]["scope"]

    def test_active_mode_uses_browse_and_marks_asking(self):
        scraper = self._make_scraper(mode="active")
        scraper._session.get.return_value = _resp(
            200,
            {
                "total": 1,
                "itemSummaries": [
                    {
                        "title": "Jarrón porcelana",
                        "price": {"value": "80.00", "currency": "EUR"},
                        "itemWebUrl": "https://www.ebay.es/itm/2",
                    }
                ],
            },
        )
        items = scraper.scrape("porcelana")
        assert items[0]["price_basis"] == "asking"
        assert items[0]["final_price"] == 80.0
        assert "/buy/browse/v1/" in scraper._session.get.call_args[0][0]

    def test_paginates_until_max_results(self):
        scraper = self._make_scraper(mode="active")
        page = lambda n: _resp(  # noqa: E731
            200,
            {
                "total": 1000,
                "next": "more",
                "itemSummaries": [
                    {"title": f"Item {i}", "itemWebUrl": f"https://e/{n}-{i}"}
                    for i in range(200)
                ],
            },
        )
        scraper._session.get.side_effect = [page(0), page(1)]
        items = scraper.scrape("x", max_results=250)
        assert len(items) == 250
        offsets = [c.kwargs["params"]["offset"] for c in scraper._session.get.call_args_list]
        limits = [c.kwargs["params"]["limit"] for c in scraper._session.get.call_args_list]
        assert offsets == [0, 200]
        assert limits == [200, 50]

    def test_forbidden_raises(self):
        scraper = self._make_scraper()
        scraper._session.get.return_value = _resp(403, {"errors": []})
        with pytest.raises(EbayApiError):
            scraper.scrape("reloj")

    def test_rate_limited_returns_partial(self):
        scraper = self._make_scraper(mode="active")
        scraper._session.get.return_value = _resp(429)
        assert scraper.scrape("reloj") == []

    def test_token_is_reused(self):
        scraper = self._make_scraper(mode="active")
        scraper._session.get.return_value = _resp(200, {"total": 0, "itemSummaries": []})
        scraper.scrape("a")
        scraper.scrape("b")
        assert scraper._session.post.call_count == 1

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            EbayApiScraper(mode="bogus", client_id="a", client_secret="b")


# ---------------------------------------------------------------------------
# robots.txt fetching (uses our session / User-Agent, not urllib's default)
# ---------------------------------------------------------------------------

class TestRobotsFetching:
    def _scraper(self, status: int = 200, text: str = "", exc: Exception | None = None):
        scraper = LibraryOfCongressScraper(crawl_delay=0)
        scraper._session = MagicMock()
        if exc is not None:
            scraper._session.get.side_effect = exc
        else:
            resp = MagicMock()
            resp.status_code = status
            resp.text = text
            scraper._session.get.return_value = resp
        return scraper

    def test_parses_rules_with_own_session(self):
        s = self._scraper(200, "User-agent: *\nDisallow: /private\n")
        assert s._is_allowed("/preservation") is True
        assert s._is_allowed("/private/x") is False
        url = s._session.get.call_args[0][0]
        assert url == "https://www.loc.gov/robots.txt"

    def test_404_means_allow_all(self):
        assert self._scraper(404)._is_allowed("/anything") is True

    def test_403_means_disallow_all(self):
        s = self._scraper(403)
        assert s._is_allowed("/anything") is False
        assert "403" in s._robots_status

    def test_server_error_disallows(self):
        assert self._scraper(503)._is_allowed("/anything") is False

    def test_network_error_disallows(self):
        import requests
        s = self._scraper(exc=requests.ConnectionError("boom"))
        assert s._is_allowed("/anything") is False

    def test_robots_fetched_once(self):
        s = self._scraper(200, "User-agent: *\nAllow: /\n")
        s._is_allowed("/a"); s._is_allowed("/b")
        assert s._session.get.call_count == 1


# ---------------------------------------------------------------------------
# scrape_sales.py – eBay licence guard
# ---------------------------------------------------------------------------

class TestScrapeSalesEbayGuard:
    def _run(self, tmp_path, *extra):
        import scripts.scrape_sales as ss
        calls = []

        class FakeEbay:
            def __init__(self, **kw):
                pass
            def scrape(self, keywords, max_results=50):
                calls.append(keywords)
                return [{"title": "t", "final_price": 1.0, "currency": "EUR",
                         "source_url": "https://www.ebay.es/itm/1", "price_basis": "asking"}]

        db = f"sqlite:///{tmp_path / 't.db'}"
        with patch.dict(ss._SCRAPERS, {"ebay_api": FakeEbay}), \
             patch.object(ss, "EbayApiScraper", FakeEbay):
            try:
                ss.main(["--keywords", "x", "--sources", "ebay_api", "--db-url", db, *extra])
            except SystemExit as exc:
                return calls, exc.code
        return calls, 0

    def test_refuses_to_store_without_permission(self, tmp_path, capsys):
        calls, code = self._run(tmp_path)
        assert code == 2
        assert calls == []
        assert "Refusing to store eBay data" in capsys.readouterr().err

    def test_dry_run_allowed_without_permission(self, tmp_path):
        calls, code = self._run(tmp_path, "--dry-run")
        assert code == 0 and calls == ["x"]

    def test_stores_with_permission(self, tmp_path):
        import sqlite3
        calls, code = self._run(tmp_path, "--ebay-data-permission")
        assert code == 0 and calls == ["x"]
        n = sqlite3.connect(tmp_path / "t.db").execute(
            "select count(*) from historical_sales").fetchone()[0]
        assert n == 1

    def test_ebay_api_not_in_defaults(self):
        import scripts.scrape_sales as ss
        assert "ebay_api" not in ss._DEFAULT_SOURCES
