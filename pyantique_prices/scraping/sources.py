"""Robots.txt-compliant scrapers for antique auction data sources.

Supported sources
-----------------
* eBay API – official eBay REST APIs (Marketplace Insights / Browse);
  the recommended eBay source, see :class:`EbayApiScraper`
* eBay.es  – completed/sold listings via HTML (usually blocked by robots.txt)
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
        self._robots_status: str = "rules"
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # robots.txt helpers
    # ------------------------------------------------------------------

    def _get_robots(self) -> urllib.robotparser.RobotFileParser:
        """Fetch and parse ``robots.txt`` using our own session / User-Agent.

        ``RobotFileParser.read()`` uses urllib's default ``Python-urllib``
        agent, which many sites answer with 403 – and the parser then treats
        the whole site as disallowed.  Fetching it ourselves avoids that
        false negative while keeping the standard semantics:

        * 2xx      → parse the rules
        * 401/403  → everything disallowed (robots.txt is access-controlled)
        * other 4xx (e.g. 404) → everything allowed (no robots.txt)
        * 5xx / network error  → everything disallowed (be conservative)
        """
        if self._robots is None:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = urljoin(self.base_url, "/robots.txt")
            rp.set_url(robots_url)
            self._robots_status = "unknown"
            self._polite_wait()
            try:
                response = self._session.get(
                    robots_url,
                    timeout=_REQUEST_TIMEOUT,
                    headers={"Accept": "text/plain,*/*;q=0.8"},
                )
            except requests.RequestException as exc:
                logger.warning("Could not fetch %s: %s", robots_url, exc)
                rp.disallow_all = True
                self._robots_status = f"unreachable ({exc.__class__.__name__})"
            else:
                code = response.status_code
                if 200 <= code < 300:
                    rp.parse(response.text.splitlines())
                    self._robots_status = "rules"
                elif code in (401, 403):
                    rp.disallow_all = True
                    self._robots_status = f"HTTP {code} – access denied to robots.txt"
                elif 400 <= code < 500:
                    rp.allow_all = True
                    self._robots_status = f"HTTP {code} – no robots.txt"
                else:
                    rp.disallow_all = True
                    self._robots_status = f"HTTP {code} – server error"
                logger.debug("robots.txt %s: %s", robots_url, self._robots_status)
            finally:
                self._last_request_time = time.monotonic()
            rp.modified()  # mark as read so can_fetch() evaluates the rules
            self._robots = rp
        return self._robots

    def _is_allowed(self, path: str) -> bool:
        rp = self._get_robots()
        allowed = rp.can_fetch(_USER_AGENT, urljoin(self.base_url, path))
        if not allowed:
            status = getattr(self, "_robots_status", "rules")
            reason = (
                "a Disallow rule matches"
                if status == "rules"
                else f"robots.txt not usable: {status}"
            )
            logger.info(
                "%s: %s blocked by robots.txt (%s).", self.base_url, path, reason
            )
        return allowed

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

    # The path actually requested below – this is what must be checked
    # against robots.txt (loc.gov currently disallows /search for all agents).
    _SEARCH_PATH = "/search"

    def scrape(self, keywords: str, max_results: int = 50) -> list[dict]:
        if not self._is_allowed(self._SEARCH_PATH):
            logger.warning(
                "LoC robots.txt disallows %s – skipping.", self._SEARCH_PATH
            )
            return []

        self.crawl_delay = max(self.crawl_delay, self._crawl_delay_from_robots())

        search_url = (
            f"{self.base_url}{self._SEARCH_PATH}"
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
# eBay – official REST APIs (OAuth2 client-credentials)
# ---------------------------------------------------------------------------

class EbayApiError(RuntimeError):
    """Raised when the eBay API cannot be used (missing keys, auth failure)."""


class EbayApiScraper(_BaseAuctionScraper):
    """Fetch eBay listings through eBay's official REST APIs.

    This is the recommended way to get eBay data: eBay's ``robots.txt``
    disallows automated crawling of its search pages, so the HTML-based
    :class:`EbayEsScraper` usually returns nothing.  The API is the
    sanctioned channel and is governed by the eBay API License Agreement and
    per-application call limits instead.

    Two modes are supported:

    ``"sold"`` (default)
        Marketplace Insights API – completed sales with the realised price
        and sale date (last 90 days).  Access to this API is restricted: your
        eBay developer application must be approved for the
        ``buy.marketplace.insights`` scope.  Records get
        ``price_basis="realized"``.

    ``"active"``
        Browse API – currently *active* listings.  Available to every
        developer key, but the prices are asking prices, not hammer prices.
        Records get ``price_basis="asking"`` so they can be told apart from
        realised sales downstream.

    robots.txt is still fetched from the API host and honoured before every
    request, and the configured crawl delay is applied between calls.

    Credentials are read from the ``EBAY_CLIENT_ID`` / ``EBAY_CLIENT_SECRET``
    environment variables (see ``.env.example``) unless passed explicitly.
    """

    source_name = "ebay_api"

    _HOSTS = {
        "production": "https://api.ebay.com",
        "sandbox": "https://api.sandbox.ebay.com",
    }
    _TOKEN_PATH = "/identity/v1/oauth2/token"
    _BROWSE_PATH = "/buy/browse/v1/item_summary/search"
    _INSIGHTS_PATH = "/buy/marketplace_insights/v1_beta/item_sales/search"

    _SCOPE_BASE = "https://api.ebay.com/oauth/api_scope"
    _SCOPE_INSIGHTS = "https://api.ebay.com/oauth/api_scope/buy.marketplace.insights"

    # eBay caps ``limit`` at 200 for both endpoints; offset + limit <= 10 000.
    _PAGE_SIZE = 200
    _MAX_OFFSET = 10_000

    _MARKETPLACE_LABELS = {
        "EBAY_ES": "eBay.es",
        "EBAY_GB": "eBay.co.uk",
        "EBAY_US": "eBay.com",
        "EBAY_DE": "eBay.de",
        "EBAY_FR": "eBay.fr",
        "EBAY_IT": "eBay.it",
    }

    def __init__(
        self,
        crawl_delay: float = 1.0,
        *,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        marketplace_id: Optional[str] = None,
        environment: Optional[str] = None,
        mode: Optional[str] = None,
        category_ids: Optional[str] = None,
    ) -> None:
        super().__init__(crawl_delay=crawl_delay)
        # Imported lazily so the module has no hard dependency on config.
        from pyantique_prices.config import settings

        self.client_id = client_id or settings.ebay_client_id
        self.client_secret = client_secret or settings.ebay_client_secret
        self.marketplace_id = (marketplace_id or settings.ebay_marketplace_id).upper()
        env = (environment or settings.ebay_environment).lower()
        if env not in self._HOSTS:
            raise ValueError(f"EBAY_ENVIRONMENT must be one of {list(self._HOSTS)}")
        self.environment = env
        self.base_url = self._HOSTS[env]
        self.mode = (mode or "sold").lower()
        if self.mode not in ("sold", "active"):
            raise ValueError("mode must be 'sold' or 'active'")
        # eBay category 20081 = "Antiques"; restricts results to that tree.
        self.category_ids = (
            category_ids if category_ids is not None else settings.ebay_category_ids
        )

        self._session.headers.update({"Accept": "application/json"})
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0

    # ------------------------------------------------------------------
    # OAuth
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        if self._token and time.monotonic() < self._token_expiry - 60:
            return self._token

        if not self.client_id or not self.client_secret:
            raise EbayApiError(
                "eBay API credentials missing – set EBAY_CLIENT_ID and "
                "EBAY_CLIENT_SECRET (create a keyset at "
                "https://developer.ebay.com/my/keys)."
            )

        scope = self._SCOPE_INSIGHTS if self.mode == "sold" else self._SCOPE_BASE
        self._polite_wait()
        try:
            response = self._session.post(
                f"{self.base_url}{self._TOKEN_PATH}",
                auth=(self.client_id, self.client_secret),
                data={"grant_type": "client_credentials", "scope": scope},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise EbayApiError(f"eBay OAuth request failed: {exc}") from exc
        finally:
            self._last_request_time = time.monotonic()

        if response.status_code != 200:
            detail = _safe_json(response).get("error_description") or response.text[:200]
            hint = ""
            if self.mode == "sold" and "scope" in detail.lower():
                hint = (
                    " – your application is not approved for the Marketplace "
                    "Insights API. Request access from eBay, or use "
                    "--ebay-api-mode active (asking prices only)."
                )
            raise EbayApiError(
                f"eBay OAuth failed ({response.status_code}): {detail}{hint}"
            )

        payload = response.json()
        self._token = payload["access_token"]
        self._token_expiry = time.monotonic() + float(payload.get("expires_in", 7200))
        return self._token

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    def _api_get(self, path: str, params: dict) -> Optional[dict]:
        self._polite_wait()
        try:
            response = self._session.get(
                f"{self.base_url}{path}",
                params=params,
                headers={
                    "Authorization": f"Bearer {self._get_token()}",
                    "X-EBAY-C-MARKETPLACE-ID": self.marketplace_id,
                },
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            logger.error("eBay API request failed: %s", exc)
            return None
        finally:
            self._last_request_time = time.monotonic()

        if response.status_code == 429:
            logger.error("eBay API rate limit reached – stopping for now.")
            return None
        if response.status_code in (401, 403):
            raise EbayApiError(
                f"eBay API refused access to {path} ({response.status_code}): "
                f"{response.text[:200]}"
            )
        if not response.ok:
            logger.error(
                "eBay API error %s for %s: %s",
                response.status_code, path, response.text[:200],
            )
            return None
        return _safe_json(response)

    def scrape(self, keywords: str, max_results: int = 50) -> list[dict]:
        path = self._INSIGHTS_PATH if self.mode == "sold" else self._BROWSE_PATH
        if not self._is_allowed(path):
            logger.warning(
                "%s robots.txt disallows %s – skipping.", self.base_url, path
            )
            return []

        self.crawl_delay = max(self.crawl_delay, self._crawl_delay_from_robots())

        results: list[dict] = []
        offset = 0
        while len(results) < max_results and offset < self._MAX_OFFSET:
            limit = min(self._PAGE_SIZE, max_results - len(results))
            params = {"q": keywords, "limit": limit, "offset": offset}
            if self.category_ids:
                params["category_ids"] = self.category_ids

            data = self._api_get(path, params)
            if not data:
                break

            key = "itemSales" if self.mode == "sold" else "itemSummaries"
            page = [self._parse_item(item) for item in data.get(key) or []]
            page = [p for p in page if p]
            if not page:
                break
            results.extend(page)

            offset += limit
            if not data.get("next") or offset >= int(data.get("total", 0)):
                break

        logger.info(
            "eBay API (%s, %s): %d items for '%s'",
            self.mode, self.marketplace_id, len(results), keywords,
        )
        return results[:max_results]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_item(self, item: dict) -> Optional[dict]:
        title = item.get("title")
        if not title:
            return None

        if self.mode == "sold":
            price_obj = item.get("lastSoldPrice") or {}
            date_raw = item.get("lastSoldDate")
            basis = "realized"
        else:
            price_obj = item.get("price") or item.get("currentBidPrice") or {}
            date_raw = item.get("itemEndDate") or item.get("itemCreationDate")
            basis = "asking"

        try:
            price = float(price_obj["value"]) if price_obj.get("value") else None
        except (TypeError, ValueError):
            price = None

        sale_date = _parse_iso_utc(date_raw)
        categories = item.get("categories") or []
        category = categories[0].get("categoryName") if categories else None

        return {
            "title": title,
            "description": item.get("shortDescription"),
            "category": category,
            "final_price": price,
            "currency": price_obj.get("currency"),
            "sale_date": sale_date.isoformat() if sale_date else None,
            "auction_house": self._MARKETPLACE_LABELS.get(
                self.marketplace_id, self.marketplace_id
            ),
            "source_url": item.get("itemWebUrl") or item.get("itemHref"),
            "price_basis": basis,
        }


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


def _safe_json(response: requests.Response) -> dict:
    """Return the JSON body of *response* as a dict, or ``{}`` if it is not JSON."""
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_iso_utc(text: Optional[str]) -> Optional[datetime.datetime]:
    """Parse an eBay ISO-8601 timestamp (``2024-03-01T10:15:30.000Z``).

    Returns a naive UTC datetime truncated to whole seconds, so that its
    ``isoformat()`` matches the formats accepted by ``scrape_sales.py``.
    """
    if not text:
        return None
    try:
        dt = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return _parse_date_loose(text)
    if dt.tzinfo is not None:
        dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return dt.replace(microsecond=0)
