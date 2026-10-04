"""One-click research links for an identified item (no scraping, nothing stored)."""

from .links import DEFAULT_SITES, build_lookup_links, format_lookup_links

__all__ = ["DEFAULT_SITES", "build_lookup_links", "format_lookup_links"]
