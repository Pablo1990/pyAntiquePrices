"""Bridge to existing pyantique_prices.scraper module."""

from __future__ import annotations

from pyantique_prices.scraper import search_auction_prices

from .base import BaseScraper


class ExistingScraper(BaseScraper):
    """Wrapper around the existing scraper functionality."""

    name = "existing"

    def search(self, keywords: str, max_results: int = 10) -> list[dict]:
        results = search_auction_prices(keywords)
        return results[:max_results] if results else []
