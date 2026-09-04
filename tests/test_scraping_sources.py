"""Tests for pyantique_prices.scraping.sources."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyantique_prices.scraping.sources import (
    AICScraper,
    CatawikiScraper,
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
