"""Web scraper for todocoleccion.net – robots.txt compliant."""

from __future__ import annotations

import logging
import time
import urllib.robotparser
from typing import Optional
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.todocoleccion.net"
_SEARCH_PATH = "/buscar/"
_USER_AGENT = "pyAntiquePrices/0.1 (+https://github.com/Pablo1990/pyAntiquePrices)"

# Polite crawl delay in seconds (also respects Crawl-delay in robots.txt)
_DEFAULT_CRAWL_DELAY = 3.0
_REQUEST_TIMEOUT = 15  # seconds


class TodoColeccionScraper:
    """Scrape todocoleccion.net for reference prices.

    The scraper honours the site's ``robots.txt`` before making any request.
    If the robots file disallows the search path it gracefully skips the
    scrape and returns an empty result.

    Parameters
    ----------
    crawl_delay:
        Minimum seconds to wait between requests.
    """

    def __init__(self, crawl_delay: float = _DEFAULT_CRAWL_DELAY) -> None:
        self.crawl_delay = crawl_delay
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": _USER_AGENT,
                "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            }
        )
        self._robots: Optional[urllib.robotparser.RobotFileParser] = None
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_reference_prices(self, query: str, max_results: int = 5) -> str:
        """Return a formatted string of reference prices for *query*.

        Parameters
        ----------
        query:
            Search keywords (e.g. ``"reloj de bolsillo plata siglo XIX"``).
        max_results:
            Maximum number of listings to include.

        Returns
        -------
        str
            A human-readable summary ready to be embedded in the LLM prompt,
            or an empty string if scraping is not permitted or fails.
        """
        if not self._is_allowed(_SEARCH_PATH):
            logger.warning(
                "robots.txt disallows scraping %s – skipping web search.", _SEARCH_PATH
            )
            return ""

        url = self._build_search_url(query)
        logger.debug("Fetching: %s", url)
        html = self._fetch(url)
        if not html:
            return ""

        listings = self._parse_listings(html, max_results)
        if not listings:
            return ""

        lines = [f"Reference prices from todocoleccion.net for query '{query}':"]
        for i, item in enumerate(listings, 1):
            price_str = f"  Price: {item['price']}" if item["price"] else ""
            lines.append(f"  {i}. {item['title']}{price_str}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_robots(self) -> urllib.robotparser.RobotFileParser:
        if self._robots is None:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = urljoin(_BASE_URL, "/robots.txt")
            rp.set_url(robots_url)
            try:
                self._polite_wait()
                rp.read()
                logger.debug("robots.txt fetched from %s", robots_url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not fetch robots.txt: %s", exc)
            self._robots = rp
        return self._robots

    def _is_allowed(self, path: str) -> bool:
        rp = self._get_robots()
        return rp.can_fetch(_USER_AGENT, urljoin(_BASE_URL, path))

    def _polite_wait(self) -> None:
        """Sleep to respect crawl delay."""
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

    @staticmethod
    def _build_search_url(query: str) -> str:
        return f"{_BASE_URL}{_SEARCH_PATH}?q={quote_plus(query)}"

    @staticmethod
    def _parse_listings(html: str, max_results: int) -> list[dict]:
        """Parse search results page and extract title + price."""
        soup = BeautifulSoup(html, "html.parser")
        results: list[dict] = []

        # todocoleccion uses article or li elements with class patterns
        # that can change over time – we use a resilient approach.
        candidates = soup.select(
            "article.tc-ad, "
            "li.tc-ad, "
            ".search-results li, "
            ".results-list article, "
            "[class*='item-'], "
            "[class*='product-']"
        )

        for elem in candidates[:max_results]:
            title_tag = (
                elem.select_one("h2")
                or elem.select_one("h3")
                or elem.select_one(".title")
                or elem.select_one("[class*='title']")
            )
            price_tag = (
                elem.select_one(".price")
                or elem.select_one("[class*='price']")
                or elem.select_one("[itemprop='price']")
            )
            title = title_tag.get_text(strip=True) if title_tag else None
            price = price_tag.get_text(strip=True) if price_tag else None
            if title:
                results.append({"title": title, "price": price})

        return results
