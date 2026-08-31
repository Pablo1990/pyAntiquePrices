"""Base scraper interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseScraper(ABC):
    """Base class for auction site scrapers."""

    name: str = "base"

    @abstractmethod
    def search(self, keywords: str, max_results: int = 10) -> list[dict]:
        """Search and return list of sale dicts."""
