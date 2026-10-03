"""Live, per-appraisal eBay listings via eBay's official Browse API.

Why this design
---------------
The eBay API License Agreement (version of 3 Sep 2025) does not allow,
without eBay's express written permission, storing eBay Content (s. 9(g);
s. 3.1 only allows temporary intermediate copies), using it to suggest or
model prices (s. 9(e)), or using it to train algorithms (s. 9(j)).
Displayed listing data may be at most 6 hours old (s. 8.1(a)(3)).

So this module:

* queries eBay **at appraisal time** with keywords built from the
  identification, and keeps the results **in memory only** (nothing is
  written to the database, a cache or a file);
* returns them as a separate ``live_market_listings`` block with links back
  to each eBay item page, for the user to look at;
* never mixes them into ``comparables``, the pricing model or training data.

The appraisal persistence layer (``data/appraisals.py``) does not save this
block either.
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

NOTICE = (
    "Live eBay listings retrieved through the eBay API for reference only. "
    "Prices are sellers' asking prices, not sale prices. They are not stored "
    "and are not used to compute the valuation."
)

_MAX_QUERY_WORDS = 8


def _value(field: Any) -> Optional[str]:
    if isinstance(field, dict):
        field = field.get("value")
    if isinstance(field, str) and field.strip() and field.strip().lower() not in {
        "unknown", "n/a", "none", "null",
    }:
        return field.strip()
    return None


_MIN_MAKER_CONFIDENCE = 0.3
_MAKER_KEYS = ("manufacturer_candidates", "artist_candidates", "workshop_candidates")


def _best_maker(identification: dict) -> Optional[str]:
    """Highest-confidence maker/artist/workshop name, or a signature/mark.

    Candidates below ``_MIN_MAKER_CONFIDENCE`` are ignored: a wrong maker in
    the query makes eBay return unrelated items, which is worse than none.
    """
    pool: list[tuple[float, str]] = []
    marks = identification.get("marks") or []
    sources = [identification.get(key) or [] for key in _MAKER_KEYS]
    sources.extend(
        (mark.get("manufacturer_candidates") or [])
        for mark in marks
        if isinstance(mark, dict)
    )
    for candidates in sources:
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            name, conf = cand.get("name"), cand.get("confidence")
            if isinstance(name, str) and name.strip():
                pool.append((float(conf) if isinstance(conf, (int, float)) else 0.0, name.strip()))
    pool = [item for item in pool if item[0] >= _MIN_MAKER_CONFIDENCE]
    if pool:
        return max(pool, key=lambda item: item[0])[1]
    return _value(identification.get("signature_text"))


def build_query(identification: dict | None, context: str = "") -> str:
    """Build a short eBay search query from a structured identification.

    Uses the most discriminating fields first (maker/artist, object type,
    subtype, material, period) and keeps it to a handful of words, because
    eBay search matches all terms and long queries return nothing.
    """
    identification = identification or {}
    parts: list[str] = []

    maker = _best_maker(identification)
    if maker:
        parts.append(maker)

    for key in ("object_type", "subtype"):
        val = _value(identification.get(key))
        if val:
            parts.append(val)

    materials = identification.get("materials") or []
    if materials and isinstance(materials[0], str):
        parts.append(materials[0])

    period = _value(identification.get("likely_period")) or _value(
        identification.get("period")
    )
    if period:
        parts.append(period)

    if not parts and context:
        match = re.search(r"Extra keywords:\s*(.+)", context)
        parts.append(match.group(1) if match else context)

    words: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for word in re.findall(r"[\w'-]+", part):
            low = word.lower()
            if low not in seen:
                seen.add(low)
                words.append(word)
    return " ".join(words[:_MAX_QUERY_WORDS])


class EbayLiveListings:
    """Fetch eBay listings for one appraisal; results are never persisted."""

    source = "eBay"

    def __init__(self, scraper=None, max_results: int = 10, mode: str = "active") -> None:
        if scraper is None:
            from pyantique_prices.scraping.sources import EbayApiScraper

            scraper = EbayApiScraper(crawl_delay=1.0, mode=mode)
        self.scraper = scraper
        self.max_results = max_results
        self.mode = mode

    @classmethod
    def from_settings(cls, settings=None) -> Optional["EbayLiveListings"]:
        """Return an instance if live listings are enabled and keys are set."""
        if settings is None:
            from pyantique_prices.config import settings
        if not settings.ebay_live_listings:
            return None
        if not (settings.ebay_client_id and settings.ebay_client_secret):
            logger.info("eBay live listings disabled: EBAY_CLIENT_ID/SECRET not set.")
            return None
        try:
            return cls(
                max_results=settings.ebay_live_max_results,
                mode=settings.ebay_live_mode,
            )
        except ValueError as exc:
            logger.warning("eBay live listings disabled: %s", exc)
            return None

    def search(self, identification: dict | None, context: str = "") -> dict:
        query = build_query(identification, context)
        block: dict = {
            "source": self.source,
            "marketplace": getattr(self.scraper, "marketplace_id", None),
            "query": query,
            "retrieved_at": datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "price_basis": "realized" if self.mode == "sold" else "asking",
            "notice": NOTICE if self.mode != "sold" else NOTICE.replace(
                "Prices are sellers' asking prices, not sale prices. ",
                "Prices are recent eBay sale prices. ",
            ),
            "items": [],
        }
        if not query:
            return block

        records = self.scraper.scrape(query, max_results=self.max_results)
        block["items"] = [
            {
                "title": rec.get("title"),
                "price": rec.get("final_price"),
                "currency": rec.get("currency"),
                "url": rec.get("source_url"),
                "image_url": rec.get("image_url"),
                "condition": rec.get("condition"),
                "buying_options": rec.get("buying_options") or [],
                "date": rec.get("sale_date"),
            }
            for rec in records
            if rec.get("source_url")
        ]
        return block


def format_live_listings(block: dict | None, limit: int = 10) -> list[str]:
    """Plain-text lines for CLI / GUI output."""
    if not block:
        return []
    lines = [f"LIVE eBAY LISTINGS ({block.get('marketplace') or 'eBay'})"]
    lines.append(f"Search: {block.get('query') or '-'} | retrieved {block.get('retrieved_at')}")
    items = block.get("items") or []
    if not items:
        lines.append("No matching listings found.")
    for item in items[:limit]:
        price = item.get("price")
        price_txt = f"{price:.2f} {item.get('currency') or ''}".strip() if price is not None else "price n/a"
        lines.append(f"- {item.get('title') or 'Untitled'} | {price_txt}")
        lines.append(f"  {item.get('url')}")
    lines.append(f"Note: {block.get('notice')}")
    return lines
