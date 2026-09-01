"""Robots.txt-compliant scrapers for antique auction data sources.

Supported sources
-----------------
* eBay.es  – completed/sold listings (2021-present)
* Catawiki – closed lots (2021-present)
* AIC      – American Institute for Conservation references
* LoC      – Library of Congress, Preservation resources

Each scraper:
  1. Fetches and parses ``robots.txt`` before every search path.
  2. Obeys the crawl-delay directive (defaults to 5 s if not specified).
  3. Returns a list of dicts that match the ``HistoricalSale`` column schema so
     results can be imported directly via ``import_csv`` or inserted manually.

Run from the CLI via ``scripts/scrape_sales.py``.
"""

from __future__ import annotations

import datetime
import logging
import re
import time
import urllib.robotparser
from typing import Optional
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_USER_AGENT = "pyAntiquePrices/0.1 (+https://github.com/Pablo1990/pyAntiquePrices)"
_REQUEST_TIMEOUT = 20
_DEFAULT_CRAWL_DELAY = 5.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _BaseAuctionScraper:
    """Shared polite-HTTP helpers with robots.txt compliance."""

    base_url: str = ""
    source_name: str = "unknown"

    def __init__(self, crawl_delay: float = _DEFAULT_CRAWL_DELAY) -> None:
        self.crawl_delay = crawl_delay
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": _USER_AGENT,
                "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                ),
            }
        )
        self._robots: Optional[urllib.robotparser.RobotFileParser] = None
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # robots.txt helpers
    # ------------------------------------------------------------------

    def _get_robots(self) -> urllib.robotparser.RobotFileParser:
        if self._robots is None:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = urljoin(self.base_url, "/robots.txt")
            rp.set_url(robots_url)
            try:
                self._polite_wait()
                rp.read()
                logger.debug("robots.txt fetched: %s", robots_url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not fetch robots.txt from %s: %s", robots_url, exc)
            self._robots = rp
        return self._robots

    def _is_allowed(self, path: str) -> bool:
        rp = self._get_robots()
        return rp.can_fetch(_USER_AGENT, urljoin(self.base_url, path))

    def _crawl_delay_from_robots(self) -> float:
        rp = self._get_robots()
        delay = rp.crawl_delay(_USER_AGENT)
        return float(delay) if delay is not None else self.crawl_delay

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scrape(self, keywords: str, max_results: int = 50) -> list[dict]:
        """Return list of sale dicts for *keywords*.  Always respects robots.txt."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# eBay.es – Completed / Sold listings
# ---------------------------------------------------------------------------

class EbayEsScraper(_BaseAuctionScraper):
    """Scrape eBay.es completed (sold) listings.

    Uses the public eBay.es search with ``LH_Sold=1&LH_Complete=1`` to filter
    for historical hammer prices.  Checks ``/robots.txt`` before every path.

    Notes
    -----
    eBay's robots.txt restricts many paths for automated crawlers.  If the
    search path is disallowed, this scraper logs a warning and returns an
    empty list – it will never violate the robots.txt directive.
    """

    base_url = "https://www.ebay.es"
    source_name = "ebay.es"

    # eBay completed-listings search path
    _SEARCH_PATH = "/sch/i.html"

    def scrape(self, keywords: str, max_results: int = 50) -> list[dict]:
        if not self._is_allowed(self._SEARCH_PATH):
            logger.warning(
                "eBay.es robots.txt disallows %s – skipping.", self._SEARCH_PATH
            )
            return []

        # Respect the crawl-delay from robots.txt if larger than our default.
        self.crawl_delay = max(self.crawl_delay, self._crawl_delay_from_robots())

        results: list[dict] = []
        page = 1
        per_page = 50

        while len(results) < max_results:
            url = (
                f"{self.base_url}{self._SEARCH_PATH}"
                f"?_nkw={quote_plus(keywords)}"
                f"&LH_Sold=1&LH_Complete=1"
                f"&_pgn={page}&_ipg={per_page}"
            )
            html = self._fetch(url)
            if not html:
                break

            page_results = self._parse_listings(html)
            if not page_results:
                break

            results.extend(page_results)
            if len(page_results) < per_page:
                break
            page += 1

        logger.info("eBay.es: scraped %d listings for '%s'", len(results), keywords)
        return results[:max_results]

    @staticmethod
    def _parse_listings(html: str) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        items: list[dict] = []

        for card in soup.select(".s-item"):
            title_tag = card.select_one(".s-item__title")
            price_tag = card.select_one(".s-item__price")
            date_tag = card.select_one(".s-item__ended-date")
            link_tag = card.select_one("a.s-item__link")

            title = title_tag.get_text(strip=True) if title_tag else None
            if not title or title.lower().startswith("shop on ebay"):
                continue

            price_text = price_tag.get_text(strip=True) if price_tag else ""
            price, currency = _parse_price(price_text)

            date_text = date_tag.get_text(strip=True) if date_tag else ""
            sale_date = _parse_date_loose(date_text)

            source_url = link_tag["href"].split("?")[0] if link_tag else None

            items.append(
                {
                    "title": title,
                    "final_price": price,
                    "currency": currency,
                    "sale_date": sale_date.isoformat() if sale_date else None,
                    "auction_house": "eBay.es",
                    "source_url": source_url,
                    "price_basis": "realized",
                }
            )

        return items


# ---------------------------------------------------------------------------
# Catawiki – Closed lots
# ---------------------------------------------------------------------------

class CatawikiScraper(_BaseAuctionScraper):
    """Scrape Catawiki closed auction lots.

    Catawiki's public search is used with ``status=closed`` to find historical
    sold lots.  robots.txt compliance is enforced before every path.
    """

    base_url = "https://www.catawiki.com"
    source_name = "catawiki"

    _SEARCH_PATH = "/en/s"

    def scrape(self, keywords: str, max_results: int = 50) -> list[dict]:
        if not self._is_allowed(self._SEARCH_PATH):
            logger.warning(
                "Catawiki robots.txt disallows %s – skipping.", self._SEARCH_PATH
            )
            return []

        self.crawl_delay = max(self.crawl_delay, self._crawl_delay_from_robots())

        url = (
            f"{self.base_url}{self._SEARCH_PATH}"
            f"?q={quote_plus(keywords)}&status=closed"
        )
        html = self._fetch(url)
        if not html:
            return []

        results = self._parse_listings(html)
        logger.info("Catawiki: scraped %d listings for '%s'", len(results), keywords)
        return results[:max_results]

    @staticmethod
    def _parse_listings(html: str) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        items: list[dict] = []

        # Catawiki renders lots inside <li data-lot-id="…"> elements or
        # article tags depending on the page version – we try both.
        cards = soup.select("article.lot-card, li[data-lot-id]")
        for card in cards:
            title_tag = card.select_one("[class*='lot-card__title'], h3, h2")
            price_tag = card.select_one(
                "[class*='lot-card__price'], [class*='hammer-price']"
            )
            date_tag = card.select_one("[class*='end-date'], time")
            link_tag = card.select_one("a")

            title = title_tag.get_text(strip=True) if title_tag else None
            if not title:
                continue

            price_text = price_tag.get_text(strip=True) if price_tag else ""
            price, currency = _parse_price(price_text)

            date_text = date_tag.get("datetime") or (
                date_tag.get_text(strip=True) if date_tag else ""
            )
            sale_date = _parse_date_loose(date_text)

            href = link_tag.get("href", "") if link_tag else ""
            source_url = urljoin("https://www.catawiki.com", href) if href else None

            items.append(
                {
                    "title": title,
                    "final_price": price,
                    "currency": currency,
                    "sale_date": sale_date.isoformat() if sale_date else None,
                    "auction_house": "Catawiki",
                    "source_url": source_url,
                    "price_basis": "realized",
                }
            )

        return items


# ---------------------------------------------------------------------------
# AIC – American Institute for Conservation
# ---------------------------------------------------------------------------

class AICScraper(_BaseAuctionScraper):
    """Scrape the AIC (culturalheritage.org) for conservation / valuation references.

    The AIC website publishes publicly accessible articles, guides, and
    resources about the conservation and valuation of cultural heritage
    objects.  No authentication or API key is required.
    """

    base_url = "https://www.culturalheritage.org"
    source_name = "aic"

    _SEARCH_PATH = "/find-a-conservator"

    def scrape(self, keywords: str, max_results: int = 50) -> list[dict]:  # noqa: ARG002
        """Return AIC reference entries related to *keywords*.

        Because AIC does not expose structured auction data the method returns
        reference records (title + URL) that can be stored as provenance /
        research links.  ``final_price`` will be ``None`` for all records.
        """
        # Check robots.txt for the resource directory
        resource_path = "/resources"
        if not self._is_allowed(resource_path):
            logger.warning(
                "AIC robots.txt disallows %s – skipping.", resource_path
            )
            return []

        self.crawl_delay = max(self.crawl_delay, self._crawl_delay_from_robots())

        search_url = (
            f"{self.base_url}/resources"
            f"?keywords={quote_plus(keywords)}"
        )
        html = self._fetch(search_url)
        if not html:
            return []

        results = self._parse_resources(html, keywords)
        logger.info("AIC: found %d resources for '%s'", len(results), keywords)
        return results[:max_results]

    @staticmethod
    def _parse_resources(html: str, keywords: str) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        items: list[dict] = []

        for article in soup.select("article, .resource-item, .views-row"):
            title_tag = article.select_one("h2, h3, .title, a")
            link_tag = article.select_one("a")

            title = title_tag.get_text(strip=True) if title_tag else None
            if not title:
                continue

            href = link_tag.get("href", "") if link_tag else ""
            source_url = urljoin("https://www.culturalheritage.org", href) if href else None

            items.append(
                {
                    "title": title,
                    "description": f"AIC conservation resource – keywords: {keywords}",
                    "auction_house": "AIC",
                    "source_url": source_url,
                    "final_price": None,
                    "currency": None,
                    "sale_date": None,
                    "price_basis": "reference",
                }
            )

        return items


# ---------------------------------------------------------------------------
# Library of Congress – Preservation
# ---------------------------------------------------------------------------

class LibraryOfCongressScraper(_BaseAuctionScraper):
    """Scrape Library of Congress preservation pages (loc.gov/preservation).

    The LoC preservation section publishes publicly accessible conservation
    and valuation references.  This scraper harvests title + URL tuples that
    can serve as provenance or research records in the database.
    """

    base_url = "https://www.loc.gov"
    source_name = "loc"

    _PRESERVATION_PATH = "/preservation"

    def scrape(self, keywords: str, max_results: int = 50) -> list[dict]:
        if not self._is_allowed(self._PRESERVATION_PATH):
            logger.warning(
                "LoC robots.txt disallows %s – skipping.", self._PRESERVATION_PATH
            )
            return []

        self.crawl_delay = max(self.crawl_delay, self._crawl_delay_from_robots())

        search_url = (
            f"{self.base_url}/search"
            f"?q={quote_plus(keywords)}&fa=subject_headings%3Apreservation"
        )
        html = self._fetch(search_url)
        if not html:
            return []

        results = self._parse_resources(html, keywords)
        logger.info("LoC: found %d resources for '%s'", len(results), keywords)
        return results[:max_results]

    @staticmethod
    def _parse_resources(html: str, keywords: str) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        items: list[dict] = []

        for li in soup.select(".result, .item, article, li.item-description"):
            title_tag = li.select_one("h3, h2, .item-description-title, a")
            link_tag = li.select_one("a")
            date_tag = li.select_one("time, .date")

            title = title_tag.get_text(strip=True) if title_tag else None
            if not title:
                continue

            href = link_tag.get("href", "") if link_tag else ""
            source_url = urljoin("https://www.loc.gov", href) if href else None

            date_text = date_tag.get_text(strip=True) if date_tag else ""
            sale_date = _parse_date_loose(date_text)

            items.append(
                {
                    "title": title,
                    "description": (
                        f"Library of Congress preservation reference – keywords: {keywords}"
                    ),
                    "auction_house": "Library of Congress",
                    "source_url": source_url,
                    "final_price": None,
                    "currency": None,
                    "sale_date": sale_date.isoformat() if sale_date else None,
                    "price_basis": "reference",
                }
            )

        return items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_price(text: str) -> tuple[Optional[float], Optional[str]]:
    """Extract a numeric price and ISO-4217 currency code from a free-text string.

    Supports common prefixes / suffixes: ``€``, ``EUR``, ``£``, ``GBP``,
    ``$``, ``USD``.  Returns ``(None, None)`` when no price can be parsed.
    """
    if not text:
        return None, None

    currency_map = {
        "€": "EUR",
        "£": "GBP",
        "$": "USD",
        "EUR": "EUR",
        "GBP": "GBP",
        "USD": "USD",
    }

    detected_currency = None
    for symbol, code in currency_map.items():
        if symbol in text:
            detected_currency = code
            break

    # Extract numeric part – strip thousands separator and normalise decimal.
    number_match = re.search(r"[\d.,]+", text.replace("\u00a0", ""))
    if not number_match:
        return None, detected_currency

    raw = number_match.group(0)
    # Handle European format (comma as decimal): 1.234,56 → 1234.56
    if "," in raw and "." in raw:
        if raw.index(",") > raw.index("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw and "." not in raw:
        raw = raw.replace(",", ".")

    try:
        return float(raw), detected_currency
    except ValueError:
        return None, detected_currency


def _parse_date_loose(text: str) -> Optional[datetime.datetime]:
    """Parse a date string in several common formats; return ``None`` on failure."""
    if not text:
        return None

    # ISO date embedded in a longer string
    iso_match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if iso_match:
        try:
            return datetime.datetime.strptime(iso_match.group(0), "%Y-%m-%d")
        except ValueError:
            pass

    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ):
        try:
            return datetime.datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue

    return None
