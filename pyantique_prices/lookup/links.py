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

# kind: sold      realised prices (closest to "what it really sells for")
#       auction   auction archives / upcoming lots
#       active    marketplaces: asking prices, usually optimistic
#       retail    dealers' and designers' asking prices (upper bound)
#       reference makers, marks, museum collections, price guides
#       images    visual comparison
#
# "url" is a direct search URL. "site" builds a Google ``site:`` search
# instead, which keeps working when a site changes its own search URL; use it
# for any site whose URL pattern is uncertain. ``verified`` is True only where
# the direct URL was confirmed to return a results page.
DEFAULT_SITES: list[dict[str, Any]] = [
    # --- realised prices / marketplaces -------------------------------------
    {"id": "ebay_sold", "name": "eBay - sold & completed", "kind": "sold", "regions": ["global"],
     "url": "https://{ebay_domain}/sch/i.html?_nkw={q}&LH_Sold=1&LH_Complete=1&_sacat=0",
     "note": "Sold listings are the closest free proxy for real prices.", "broad": True},
    {"id": "ebay_active", "name": "eBay - active listings", "kind": "active", "regions": ["global"],
     "url": "https://{ebay_domain}/sch/i.html?_nkw={q}&_sacat=0",
     "note": "Asking prices, usually above what items sell for.", "broad": True},
    {"id": "catawiki", "name": "Catawiki", "kind": "auction", "regions": ["global", "es"],
     "url": "https://www.catawiki.com/en/s?q={q}", "verified": True,
     "note": "Open the lot to see closed-auction prices."},
    {"id": "todocoleccion", "name": "Todocoleccion", "kind": "active", "regions": ["es"],
     "url": "https://www.todocoleccion.net/buscador?bu={q}"},
    {"id": "wallapop", "name": "Wallapop", "kind": "active", "regions": ["es"],
     "url": "https://es.wallapop.com/app/search?keywords={q}"},
    {"id": "milanuncios", "name": "Milanuncios", "kind": "active", "regions": ["es"],
     "site": "milanuncios.com"},
    # --- auction archives ---------------------------------------------------
    {"id": "christies", "name": "Christie's past lots", "kind": "auction", "regions": ["global"],
     "site": "christies.com", "note": "Past lot pages show the realised price."},
    {"id": "sothebys", "name": "Sotheby's past lots", "kind": "auction", "regions": ["global"],
     "site": "sothebys.com"},
    {"id": "bonhams", "name": "Bonhams", "kind": "auction", "regions": ["global", "uk"],
     "site": "bonhams.com"},
    {"id": "liveauctioneers", "name": "LiveAuctioneers", "kind": "auction", "regions": ["global", "us"],
     "site": "liveauctioneers.com"},
    {"id": "invaluable", "name": "Invaluable", "kind": "auction", "regions": ["global", "us"],
     "site": "invaluable.com"},
    {"id": "saleroom", "name": "The Saleroom", "kind": "auction", "regions": ["uk"],
     "site": "the-saleroom.com"},
    {"id": "barnebys", "name": "Barnebys", "kind": "auction", "regions": ["global"],
     "site": "barnebys.com", "note": "Aggregates many auction houses."},
    {"id": "setdart", "name": "Setdart", "kind": "auction", "regions": ["es"], "site": "setdart.com"},
    {"id": "segre", "name": "Subastas Segre", "kind": "auction", "regions": ["es"], "site": "subastassegre.es"},
    {"id": "ansorena", "name": "Ansorena", "kind": "auction", "regions": ["es"], "site": "ansorena.com"},
    {"id": "alcala", "name": "Alcalá Subastas", "kind": "auction", "regions": ["es"], "site": "alcalasubastas.com"},
    {"id": "durantes", "name": "Durán Arte y Subastas", "kind": "auction", "regions": ["es"],
     "site": "duran-subastas.com"},
    {"id": "drouot", "name": "Drouot", "kind": "auction", "regions": ["fr"], "site": "drouot.com"},
    {"id": "interencheres", "name": "Interenchères", "kind": "auction", "regions": ["fr"],
     "site": "interencheres.com"},
    {"id": "lot_tissimo", "name": "Lot-tissimo", "kind": "auction", "regions": ["de"], "site": "lot-tissimo.com"},
    {"id": "sworders", "name": "Sworders", "kind": "auction", "regions": ["uk"], "site": "sworder.co.uk"},
    {"id": "dreweatts", "name": "Dreweatts", "kind": "auction", "regions": ["uk"], "site": "dreweatts.com"},
    {"id": "lyonturnbull", "name": "Lyon & Turnbull", "kind": "auction", "regions": ["uk"],
     "site": "lyonandturnbull.com"},
    {"id": "heritage", "name": "Heritage Auctions", "kind": "auction", "regions": ["us"], "site": "ha.com"},
    {"id": "doyle", "name": "Doyle", "kind": "auction", "regions": ["us"], "site": "doyle.com"},
    # --- art price databases (full data is paid; free search shows some results)
    {"id": "artnet", "name": "Artnet price database", "kind": "auction", "regions": ["global"],
     "categories": ["art"], "site": "artnet.com", "note": "Full results need a subscription."},
    {"id": "mutualart", "name": "MutualArt", "kind": "auction", "regions": ["global"],
     "categories": ["art"], "site": "mutualart.com"},
    # --- dealers (asking prices: an upper bound) ----------------------------
    {"id": "1stdibs", "name": "1stDibs", "kind": "retail", "regions": ["global"], "site": "1stdibs.com"},
    {"id": "rubylane", "name": "Ruby Lane", "kind": "retail", "regions": ["us"], "site": "rubylane.com"},
    {"id": "pamono", "name": "Pamono", "kind": "retail", "regions": ["global"],
     "categories": ["furniture", "design"], "site": "pamono.eu"},
    # --- price guides / references ------------------------------------------
    {"id": "worthpoint", "name": "WorthPoint (price guide)", "kind": "sold", "regions": ["global"],
     "site": "worthpoint.com", "note": "Full price history is a paid subscription."},
    {"id": "kovels", "name": "Kovels price guide", "kind": "reference", "regions": ["us"], "site": "kovels.com"},
    {"id": "antiquesnavigator", "name": "Antiques Navigator", "kind": "reference", "regions": ["us"],
     "site": "antiquesnavigator.com"},
    {"id": "met", "name": "Metropolitan Museum collection", "kind": "reference", "regions": ["global"],
     "site": "metmuseum.org", "note": "Compare form, period and attribution."},
    {"id": "vam", "name": "V&A collection", "kind": "reference", "regions": ["global", "uk"], "site": "collections.vam.ac.uk"},
    {"id": "europeana", "name": "Europeana", "kind": "reference", "regions": ["global", "es", "fr", "de"],
     "site": "europeana.eu"},
    {"id": "google_images", "name": "Google Images", "kind": "images", "regions": ["global"],
     "url": "https://www.google.com/search?tbm=isch&q={q}",
     "note": "Compare form, decoration and marks visually."},
    # --- category specialists -----------------------------------------------
    {"id": "chrono24", "name": "Chrono24", "kind": "active", "regions": ["global"], "categories": ["watches"],
     "url": "https://www.chrono24.com/search/index.htm?query={q}"},
    {"id": "numista", "name": "Numista", "kind": "reference", "regions": ["global"], "categories": ["coins"],
     "site": "numista.com"},
    {"id": "ngc", "name": "NGC price guide", "kind": "reference", "regions": ["global"], "categories": ["coins"],
     "site": "ngccoin.com"},
    {"id": "abebooks", "name": "AbeBooks", "kind": "active", "regions": ["global"], "categories": ["books"],
     "url": "https://www.abebooks.com/servlet/SearchResults?kn={q}"},
    {"id": "silver_marks", "name": "Silver marks", "kind": "reference", "regions": ["global"],
     "categories": ["silver_jewellery"], "site": "925-1000.com"},
    {"id": "ceramic_marks", "name": "Pottery & porcelain marks", "kind": "reference", "regions": ["global"],
     "categories": ["ceramics"], "site": "gotheborg.com"},
]

_REFERENCE_SITES: list[dict[str, Any]] = [
    {"id": "maker_reference", "name": "Maker / mark reference", "kind": "reference", "regions": ["global"],
     "url": "https://www.google.com/search?q={q}+maker+mark+identification"},
]

_GOOGLE_SITE = "https://www.google.com/search?q={q}+site%3A{site}"

# Words in the identification that select category-specific sites.
_CATEGORY_KEYWORDS = {
    "watches": ("watch", "clock", "chronometer", "reloj"),
    "coins": ("coin", "medal", "moneda", "banknote"),
    "books": ("book", "manuscript", "libro", "atlas", "map"),
    "art": ("painting", "print", "drawing", "etching", "engraving", "sculpture", "lithograph",
            "watercolour", "watercolor", "oil", "cuadro", "grabado", "escultura"),
    "silver_jewellery": ("silver", "gold", "jewel", "ring", "brooch", "necklace", "plata", "oro"),
    "ceramics": ("porcelain", "ceramic", "pottery", "faience", "earthenware", "vase", "cerámica", "porcelana"),
    "furniture": ("chair", "table", "cabinet", "desk", "commode", "sideboard", "mueble"),
    "design": ("lamp", "chair", "table", "sideboard", "mid-century", "art deco"),
}

DEFAULT_REGIONS = ("global", "es", "uk")


def infer_categories(identification: dict | None) -> set[str]:
    """Categories suggested by object type / subtype / materials."""
    ident = identification or {}
    parts = []
    for key in ("object_type", "subtype"):
        value = ident.get(key)
        parts.append(str(value.get("value") if isinstance(value, dict) else value or ""))
    parts.extend(str(m) for m in ident.get("materials") or [])
    text = " ".join(parts).lower()
    return {cat for cat, words in _CATEGORY_KEYWORDS.items() if any(w in text for w in words)}


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
        elif site.get("url") or site.get("site"):
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


def _wanted(site: dict, regions: set[str], categories: set[str]) -> bool:
    site_regions = set(site.get("regions") or ["global"])
    if "all" not in regions and not (site_regions & regions):
        return False
    site_cats = set(site.get("categories") or [])
    return not site_cats or bool(site_cats & categories)


def _site_url(site: dict, q: str, ebay_domain: str) -> str | None:
    if site.get("url"):
        return site["url"].format(q=quote_plus(q), ebay_domain=ebay_domain)
    if site.get("site"):
        return _GOOGLE_SITE.format(q=quote_plus(q), site=site["site"])
    return None


def build_lookup_links(
    identification: dict | None,
    context: str = "",
    *,
    ebay_domain: str = "ebay.es",
    sites_file: str | os.PathLike | None = None,
    regions: list[str] | tuple[str, ...] | None = None,
    categories: list[str] | None = None,
) -> dict:
    """Return ``{"query", "broad_query", "links": [...]}`` for manual research.

    ``regions`` (default ``LOOKUP_REGIONS`` env or global/es/uk; ``"all"`` for
    everything) and ``categories`` (default: inferred from the identification)
    select which sites appear. Each link is ``{"id", "name", "kind", "url",
    "query_kind", "note", "verified"}``.
    """
    identification = identification or {}
    specific = build_query(identification, context)
    broad = _broad_query(identification)
    if sites_file is None:
        sites_file = os.getenv("LOOKUP_SITES_FILE")
    sites = _merge_sites(_load_user_sites(sites_file))
    if regions is None:
        env = os.getenv("LOOKUP_REGIONS")
        regions = [r.strip().lower() for r in env.split(",") if r.strip()] if env else list(DEFAULT_REGIONS)
    region_set = {r.lower() for r in regions}
    category_set = set(categories) if categories is not None else infer_categories(identification)
    sites = [site for site in sites if _wanted(site, region_set, category_set)]

    links: list[dict[str, Any]] = []

    def add(site: dict, query: str, query_kind: str) -> None:
        if not query:
            return
        url = _site_url(site, query, ebay_domain)
        if not url:
            return
        links.append({
            "id": site["id"] if query_kind == "specific" else f"{site['id']}_broad",
            "name": site["name"] if query_kind == "specific" else f"{site['name']} (broader)",
            "kind": site.get("kind", "other"),
            "url": url,
            "query_kind": query_kind,
            "note": site.get("note"),
            "verified": bool(site.get("verified")),
        })

    for site in sites:
        add(site, specific, "specific")
        if site.get("broad") and broad and broad != specific:
            add(site, broad, "broad")

    maker = _maker_name(identification)
    if maker:
        for site in _REFERENCE_SITES:
            add(site, maker, "specific")

    order = {"sold": 0, "auction": 1, "active": 2, "retail": 3, "reference": 4, "images": 5}
    links.sort(key=lambda link: order.get(link["kind"], 9))  # stable: keeps site order

    return {
        "query": specific,
        "broad_query": broad,
        "regions": sorted(region_set),
        "categories": sorted(category_set),
        "links": links,
    }


_KIND_TITLES = {
    "sold": "Realised prices",
    "auction": "Auction houses (open a past lot for its result)",
    "active": "Marketplaces (asking prices)",
    "retail": "Dealers (asking prices, an upper bound)",
    "reference": "References (makers, marks, museums, price guides)",
    "images": "Images",
}


def format_lookup_links(block: dict | None, limit: int | None = None) -> list[str]:
    """Plain-text lines for CLI / GUI output, grouped by what each link gives you."""
    if not block or not block.get("links"):
        return []
    lines = ["RESEARCH LINKS (open in your browser; nothing is fetched or stored)"]
    lines.append(f"Search: {block.get('query') or '-'}")
    lines.append(
        "Triangulate: trust a price when sold results from 2-3 independent sources agree."
    )
    current = None
    for link in block["links"][:limit]:
        kind = link.get("kind", "other")
        if kind != current:
            current = kind
            lines.append(f"{_KIND_TITLES.get(current, current.title())}:")
        lines.append(f"- {link['name']}: {link['url']}")
    return lines
