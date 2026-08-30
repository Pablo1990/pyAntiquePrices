"""Web scrapers for antique reference prices – all robots.txt compliant."""

from __future__ import annotations

import logging
import time
import urllib.robotparser
from typing import Optional
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_USER_AGENT = "pyAntiquePrices/0.1 (+https://github.com/Pablo1990/pyAntiquePrices)"
_REQUEST_TIMEOUT = 15  # seconds
_DEFAULT_CRAWL_DELAY = 3.0


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _BaseScraper:
    """Shared HTTP helpers with robots.txt compliance and polite crawl delay."""

    base_url: str = ""

    def __init__(self, crawl_delay: float = _DEFAULT_CRAWL_DELAY) -> None:
        self.crawl_delay = crawl_delay
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": _USER_AGENT,
                "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )
        self._robots: Optional[urllib.robotparser.RobotFileParser] = None
        self._last_request_time: float = 0.0

    def _get_robots(self) -> urllib.robotparser.RobotFileParser:
        if self._robots is None:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = urljoin(self.base_url, "/robots.txt")
            rp.set_url(robots_url)
            try:
                self._polite_wait()
                rp.read()
                logger.debug("robots.txt fetched from %s", robots_url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not fetch robots.txt from %s: %s", robots_url, exc)
            self._robots = rp
        return self._robots

    def _is_allowed(self, path: str) -> bool:
        rp = self._get_robots()
        return rp.can_fetch(_USER_AGENT, urljoin(self.base_url, path))

    def _polite_wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        wait = max(0.0, self.crawl_delay - elapsed)
        if wait > 0:
            time.sleep(wait)

    def _fetch(self, url: str) -> Optional[str]:
        self._polite_wait()
        try:
            response = self._session.get(url, timeout=_REQUEST_TIMEOUT)
            self._last_request_time = time.monotonic()
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            logger.error("Request failed for %s: %s", url, exc)
            return None


# ---------------------------------------------------------------------------
# todocoleccion.net scraper
# ---------------------------------------------------------------------------

class TodoColeccionScraper(_BaseScraper):
    """Scrape todocoleccion.net for reference prices (robots.txt compliant).

    If the robots file disallows the search path the scrape is gracefully
    skipped and an empty string is returned.
    """

    base_url = "https://www.todocoleccion.net"
    _SEARCH_PATH = "/buscar/"

    def get_reference_prices(self, query: str, max_results: int = 5) -> str:
        """Return a formatted string of reference prices for *query*."""
        if not self._is_allowed(self._SEARCH_PATH):
            logger.warning("robots.txt disallows scraping %s – skipping.", self._SEARCH_PATH)
            return ""

        url = f"{self.base_url}{self._SEARCH_PATH}?q={quote_plus(query)}"
        logger.debug("Fetching: %s", url)
        html = self._fetch(url)
        if not html:
            return ""

        listings = self._parse_listings(html, max_results)
        if not listings:
            return ""

        lines = [f"Reference prices from todocoleccion.net for '{query}':"]
        for i, item in enumerate(listings, 1):
            price_str = f"  Price: {item['price']}" if item["price"] else ""
            lines.append(f"  {i}. {item['title']}{price_str}")
        return "\n".join(lines)

    @staticmethod
    def _parse_listings(html: str, max_results: int) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        results: list[dict] = []
        candidates = soup.select(
            "article.tc-ad, li.tc-ad, .search-results li, .results-list article, "
            "[class*='item-'], [class*='product-']"
        )
        for elem in candidates[:max_results]:
            title_tag = (
                elem.select_one("h2") or elem.select_one("h3")
                or elem.select_one(".title") or elem.select_one("[class*='title']")
            )
            price_tag = (
                elem.select_one(".price") or elem.select_one("[class*='price']")
                or elem.select_one("[itemprop='price']")
            )
            title = title_tag.get_text(strip=True) if title_tag else None
            price = price_tag.get_text(strip=True) if price_tag else None
            if title:
                results.append({"title": title, "price": price})
        return results


# ---------------------------------------------------------------------------
# DuckDuckGo HTML scraper (no API key required)
# ---------------------------------------------------------------------------

class DuckDuckGoScraper(_BaseScraper):
    """Search DuckDuckGo for similar antique listings (robots.txt compliant).

    DuckDuckGo's HTML endpoint (html.duckduckgo.com) is publicly accessible
    and does not require an API key.  The robots.txt is checked before every
    search path.
    """

    base_url = "https://html.duckduckgo.com"
    _SEARCH_PATH = "/html/"

    def get_reference_prices(self, query: str, max_results: int = 5) -> str:
        """Return formatted search snippets relevant to *query*."""
        if not self._is_allowed(self._SEARCH_PATH):
            logger.warning("DuckDuckGo robots.txt disallows scraping – skipping.")
            return ""

        # Append antique/price keywords to narrow results
        enriched = f"{query} antique price appraisal site:catawiki.com OR site:liveauctioneers.com OR site:invaluable.com"
        url = f"{self.base_url}{self._SEARCH_PATH}?q={quote_plus(enriched)}&kl=wt-wt"
        logger.debug("DuckDuckGo search: %s", url)
        html = self._fetch(url)
        if not html:
            return ""

        snippets = self._parse_results(html, max_results)
        if not snippets:
            return ""

        lines = [f"Web search results for similar antiques ('{query}'):"]
        for i, s in enumerate(snippets, 1):
            lines.append(f"  {i}. {s['title']}")
            if s["snippet"]:
                lines.append(f"     {s['snippet']}")
        return "\n".join(lines)

    @staticmethod
    def _parse_results(html: str, max_results: int) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        results: list[dict] = []
        for result in soup.select(".result")[:max_results]:
            title_tag = result.select_one(".result__title") or result.select_one("h2")
            snippet_tag = result.select_one(".result__snippet") or result.select_one(".snippet")
            title = title_tag.get_text(strip=True) if title_tag else None
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
            if title:
                results.append({"title": title, "snippet": snippet})
        return results


# ---------------------------------------------------------------------------
# Multi-source scraper (combines all sources)
# ---------------------------------------------------------------------------

class MultiSourceScraper:
    """Query multiple sources and combine results into a single context string.

    Sources attempted in order:
    1. todocoleccion.net  (Spanish specialist marketplace)
    2. DuckDuckGo HTML    (broad web search, auction sites)

    Each source is queried only if its robots.txt permits it.
    """

    def __init__(self, crawl_delay: float = _DEFAULT_CRAWL_DELAY) -> None:
        self._todocoleccion = TodoColeccionScraper(crawl_delay=crawl_delay)
        self._duckduckgo = DuckDuckGoScraper(crawl_delay=crawl_delay)

    def get_reference_prices(self, query: str, max_results: int = 5) -> str:
        """Return a combined reference string from all available sources."""
        parts: list[str] = []

        for scraper in (self._todocoleccion, self._duckduckgo):
            try:
                result = scraper.get_reference_prices(query, max_results=max_results)
                if result:
                    parts.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s failed: %s", type(scraper).__name__, exc)

        return "\n\n".join(parts)
