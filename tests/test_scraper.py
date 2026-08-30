"""Tests for scrapers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pyantique_prices.scraper import (
    DuckDuckGoScraper,
    MultiSourceScraper,
    TodoColeccionScraper,
)

_TODOCOLECCION_BASE = "https://www.todocoleccion.net"


class TestTodoColeccionParseListings:
    def test_returns_empty_for_blank_html(self):
        result = TodoColeccionScraper._parse_listings("<html></html>", 5)
        assert result == []

    def test_parses_article_with_title_and_price(self):
        html = """
        <html><body>
          <article class="tc-ad">
            <h2>Reloj de bolsillo plata</h2>
            <span class="price">45,00 €</span>
          </article>
        </body></html>
        """
        result = TodoColeccionScraper._parse_listings(html, 5)
        assert len(result) == 1
        assert "Reloj" in result[0]["title"]
        assert "45" in result[0]["price"]

    def test_respects_max_results(self):
        items = "".join(
            f'<article class="tc-ad"><h2>Item {i}</h2></article>' for i in range(10)
        )
        html = f"<html><body>{items}</body></html>"
        result = TodoColeccionScraper._parse_listings(html, 3)
        assert len(result) <= 3


class TestTodoColeccionRobotsTxt:
    def test_skips_when_disallowed(self):
        scraper = TodoColeccionScraper()
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = False
        scraper._robots = mock_rp

        result = scraper.get_reference_prices("reloj")
        assert result == ""

    def test_proceeds_when_allowed(self):
        scraper = TodoColeccionScraper()
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = True
        scraper._robots = mock_rp
        scraper._fetch = MagicMock(return_value="<html></html>")

        result = scraper.get_reference_prices("reloj")
        scraper._fetch.assert_called_once()
        assert result == ""


class TestDuckDuckGoParseResults:
    def test_returns_empty_for_blank_html(self):
        result = DuckDuckGoScraper._parse_results("<html></html>", 5)
        assert result == []

    def test_parses_result_with_snippet(self):
        html = """
        <html><body>
          <div class="result">
            <h2 class="result__title">Victorian silver pocket watch auction</h2>
            <span class="result__snippet">Sold for €450 at Catawiki.</span>
          </div>
        </body></html>
        """
        result = DuckDuckGoScraper._parse_results(html, 5)
        assert len(result) == 1
        assert "Victorian" in result[0]["title"]
        assert "450" in result[0]["snippet"]

    def test_skips_when_disallowed(self):
        scraper = DuckDuckGoScraper()
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = False
        scraper._robots = mock_rp

        result = scraper.get_reference_prices("pocket watch")
        assert result == ""


class TestMultiSourceScraper:
    def test_combines_results_from_both_sources(self):
        scraper = MultiSourceScraper()
        scraper._todocoleccion.get_reference_prices = MagicMock(return_value="todocoleccion result")
        scraper._duckduckgo.get_reference_prices = MagicMock(return_value="duckduckgo result")

        result = scraper.get_reference_prices("reloj")
        assert "todocoleccion result" in result
        assert "duckduckgo result" in result

    def test_returns_partial_when_one_source_fails(self):
        scraper = MultiSourceScraper()
        scraper._todocoleccion.get_reference_prices = MagicMock(side_effect=RuntimeError("network"))
        scraper._duckduckgo.get_reference_prices = MagicMock(return_value="duckduckgo result")

        result = scraper.get_reference_prices("reloj")
        assert "duckduckgo result" in result

    def test_returns_empty_when_all_sources_fail(self):
        scraper = MultiSourceScraper()
        scraper._todocoleccion.get_reference_prices = MagicMock(return_value="")
        scraper._duckduckgo.get_reference_prices = MagicMock(return_value="")

        result = scraper.get_reference_prices("reloj")
        assert result == ""
