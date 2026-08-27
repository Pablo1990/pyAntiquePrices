"""Tests for TodoColeccionScraper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyantique_prices.scraper import TodoColeccionScraper, _BASE_URL


class TestBuildSearchUrl:
    def test_encodes_query(self):
        url = TodoColeccionScraper._build_search_url("reloj plata siglo XIX")
        assert "reloj+plata+siglo+XIX" in url or "reloj%20plata%20siglo%20XIX" in url
        assert url.startswith(_BASE_URL)


class TestParseListings:
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


class TestRobotsTxtCompliance:
    def test_skips_when_disallowed(self):
        scraper = TodoColeccionScraper()
        # Patch robots to disallow
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
        # Patch _fetch to return empty page
        scraper._fetch = MagicMock(return_value="<html></html>")

        result = scraper.get_reference_prices("reloj")
        # No listings found but scraper did proceed (no robots block)
        scraper._fetch.assert_called_once()
        assert result == ""
