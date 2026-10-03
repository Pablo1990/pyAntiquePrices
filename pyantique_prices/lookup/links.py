"""Build search links so *you* can check sold prices and listings by hand.

Why links instead of fetching
-----------------------------
Most price sources forbid automated collection (robots.txt, site terms, the
eBay API licence). A link that opens a normal search page in your own browser
fetches nothing, stores nothing and needs no keys, so it is compliant by
construction and works for every marketplace -- including ones with no API.

The URL templates below are best-effort: sites change them. Point
``LOOKUP_SITES_FILE`` at a JSON file to add, replace or remove sites without
touching the code::

    [{"id": "mysite", "name": "My site", "kind": "sold",
      "url": "https://example.com/search?q={q}", "note": "optional"}]

A user file entry with an existing ``id`` replaces the default; give
``"disabled": true`` to drop one.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from pyantique_prices.services.live_market import build_query

logger = logging.getLogger(__name__)

# kind: sold = realised prices | active = current asking prices
#       auction = auction archives / upcoming lots | images = visual comparison
DEFAULT_SITES: list[dict[str, Any]] = [
    {"id": "ebay_sold", "name": "eBay - sold & completed", "kind": "sold",
     "url": "https://{ebay_domain}/sch/i.html?_nkw={q}&LH_Sold=1&LH_Complete=1&_sacat=0",
     "note": "Sold listings are the closest free proxy for real prices.", "broad": True},
    {"id": "ebay_active", "name": "eBay - active listings", "kind": "active",
     "url": "https://{ebay_domain}/sch/i.html?_nkw={q}&_sacat=0",
     "note": "Asking prices, usually above what items sell for.", "broad": True},
    {"id": "catawiki", "name": "Catawiki", "kind": "auction",
     "url": "https://www.catawiki.com/en/s?q={q}"},
    {"id": "invaluable", "name": "Invaluable", "kind": "auction",
     "url": "https://www.invaluable.com/search?keyword={q}"},
    {"id": "liveauctioneers", "name": "LiveAuctioneers", "kind": "auction",
     "url": "https://www.liveauctioneers.com/search/?keyword={q}"},
    {"id": "saleroom", "name": "The Saleroom", "kind": "auction",
     "url": "https://www.the-saleroom.com/en-gb/search-filter?searchterm={q}"},
    {"id": "barnebys", "name": "Barnebys", "kind": "auction",
     "url": "https://www.barnebys.com/search?q={q}"},
    {"id": "todocoleccion", "name": "Todocoleccion", "kind": "active",
     "url": "https://www.todocoleccion.net/buscador?bu={q}"},
    {"id": "wallapop", "name": "Wallapop", "kind": "active",
     "url": "https://es.wallapop.com/app/search?keywords={q}"},
    {"id": "google_images", "name": "Google Images", "kind": "images",
     "url": "https://www.google.com/search?tbm=isch&q={q}",
     "note": "Compare form, decoration and marks visually."},
]

_REFERENCE_SITES: list[dict[str, Any]] = [
    {"id": "maker_reference", "name": "Maker / mark reference", "kind": "reference",
     "url": "https://www.google.com/search?q={q}+maker+mark+identification"},
]


def _load_user_sites(path: str | os.PathLike | None) -> list[dict[str, Any]]:
    if not path:
        return []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [item for item in data if isinstance(item, dict) and item.get("id")]
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring LOOKUP_SITES_FILE %s: %s", path, exc)
        return []


def _merge_sites(user_sites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sites = {site["id"]: dict(site) for site in DEFAULT_SITES}
    for site in user_sites:
        if site.get("disabled"):
            sites.pop(site["id"], None)
        elif site.get("url"):
            sites[site["id"]] = {"kind": "other", "name": site["id"], **site}
    return list(sites.values())


def _broad_query(identification: dict) -> str:
    """Type + subtype + main material only: finds the category price range."""
    broad = {
        key: identification.get(key)
        for key in ("object_type", "subtype", "materials")
        if identification.get(key)
    }
    return build_query(broad)


def _maker_name(identification: dict) -> str:
    for key in ("manufacturer_candidates", "artist_candidates", "workshop_candidates"):
        for cand in identification.get(key) or []:
            if isinstance(cand, dict) and cand.get("name") and (cand.get("confidence") or 0) >= 0.3:
                return str(cand["name"])
    return ""


def build_lookup_links(
    identification: dict | None,
    context: str = "",
    *,
    ebay_domain: str = "ebay.es",
    sites_file: str | os.PathLike | None = None,
) -> dict:
    """Return ``{"query", "broad_query", "links": [...]}`` for manual research.

    Each link is ``{"id", "name", "kind", "url", "query_kind", "note"}``.
    """
    identification = identification or {}
    specific = build_query(identification, context)
    broad = _broad_query(identification)
    if sites_file is None:
        sites_file = os.getenv("LOOKUP_SITES_FILE")
    sites = _merge_sites(_load_user_sites(sites_file))

    links: list[dict[str, Any]] = []

    def add(site: dict, query: str, query_kind: str) -> None:
        if not query:
            return
        url = site["url"].format(q=quote_plus(query), ebay_domain=ebay_domain)
        links.append({
            "id": site["id"] if query_kind == "specific" else f"{site['id']}_broad",
            "name": site["name"] if query_kind == "specific" else f"{site['name']} (broader)",
            "kind": site.get("kind", "other"),
            "url": url,
            "query_kind": query_kind,
            "note": site.get("note"),
        })

    for site in sites:
        add(site, specific, "specific")
        if site.get("broad") and broad and broad != specific:
            add(site, broad, "broad")

    maker = _maker_name(identification)
    if maker:
        for site in _REFERENCE_SITES:
            add(site, maker, "specific")

    return {"query": specific, "broad_query": broad, "links": links}


def format_lookup_links(block: dict | None, limit: int | None = None) -> list[str]:
    """Plain-text lines for CLI / GUI output."""
    if not block or not block.get("links"):
        return []
    lines = ["RESEARCH LINKS (open in your browser; nothing is fetched or stored)"]
    lines.append(f"Search: {block.get('query') or '-'}")
    for link in block["links"][:limit]:
        lines.append(f"- {link['name']}: {link['url']}")
    return lines
