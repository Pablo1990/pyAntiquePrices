"""Tests for scrapers."""

from __future__ import annotations

from unittest.mock import MagicMock
from urllib.parse import parse_qs, unquote_plus, urlparse

import pytest

from pyantique_prices.scraper import (
    DuckDuckGoScraper,
    MultiSourceScraper,
)


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

    def test_respects_max_results(self):
        items = "".join(
            f'<div class="result"><h2 class="result__title">Item {i}</h2></div>'
            for i in range(10)
        )
        html = f"<html><body>{items}</body></html>"
        result = DuckDuckGoScraper._parse_results(html, 3)
        assert len(result) <= 3

    def test_skips_when_disallowed(self):
        scraper = DuckDuckGoScraper()
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = False
        scraper._robots = mock_rp

        result = scraper.get_reference_prices("pocket watch")
        assert result == ""

    def test_proceeds_when_allowed(self):
        scraper = DuckDuckGoScraper()
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = True
        scraper._robots = mock_rp
        scraper._fetch = MagicMock(return_value="<html></html>")

        result = scraper.get_reference_prices("pocket watch")
        scraper._fetch.assert_called_once()
        assert result == ""

    def test_search_query_prioritises_spanish_marketplaces(self):
        scraper = DuckDuckGoScraper()
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = True
        scraper._robots = mock_rp
        scraper._fetch = MagicMock(return_value="<html></html>")

        scraper.get_reference_prices("pocket watch")

        fetch_url = scraper._fetch.call_args.args[0]
        encoded_query = parse_qs(urlparse(fetch_url).query)["q"][0]
        decoded_query = unquote_plus(encoded_query)
        assert "site:todocoleccion.net" in decoded_query
        assert "site:setdart.com" in decoded_query
        assert "spain" in decoded_query


class TestMultiSourceScraper:
    def test_returns_duckduckgo_result(self):
        scraper = MultiSourceScraper()
        scraper._duckduckgo.get_reference_prices = MagicMock(return_value="duckduckgo result")

        result = scraper.get_reference_prices("reloj")
        assert "duckduckgo result" in result

    def test_returns_empty_when_source_fails(self):
        scraper = MultiSourceScraper()
        scraper._duckduckgo.get_reference_prices = MagicMock(side_effect=RuntimeError("network"))

        result = scraper.get_reference_prices("reloj")
        assert result == ""

    def test_returns_empty_when_source_returns_empty(self):
        scraper = MultiSourceScraper()
        scraper._duckduckgo.get_reference_prices = MagicMock(return_value="")

        result = scraper.get_reference_prices("reloj")
        assert result == ""
