from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

from pyantique_prices.lookup import build_lookup_links, format_lookup_links

IDENT = {
    "object_type": "pocket watch",
    "subtype": "hunter case",
    "materials": ["silver"],
    "manufacturer_candidates": [{"name": "Omega", "confidence": 0.8}],
}


def _by_id(block):
    return {link["id"]: link for link in block["links"]}


def test_ebay_sold_link_is_a_normal_search_url_with_sold_filters():
    block = build_lookup_links(IDENT, ebay_domain="ebay.co.uk", sites_file="")
    link = _by_id(block)["ebay_sold"]
    parsed = urlparse(link["url"])
    qs = parse_qs(parsed.query)
    assert parsed.netloc == "ebay.co.uk"
    assert qs["_nkw"] == ["Omega pocket watch hunter case silver"]
    assert qs["LH_Sold"] == ["1"] and qs["LH_Complete"] == ["1"]


def test_broad_variant_for_sold_search_and_query_is_url_encoded():
    block = build_lookup_links({"object_type": "tête-à-tête", "subtype": "set", "manufacturer_candidates": [{"name": "A&B", "confidence": 0.9}]}, sites_file="")
    ids = _by_id(block)
    assert "ebay_sold_broad" in ids
    assert "&B" not in ids["ebay_sold"]["url"].split("_nkw=")[1].split("&LH")[0]  # & is encoded


def test_all_default_links_are_https_and_unique():
    block = build_lookup_links(IDENT, sites_file="")
    ids = [link["id"] for link in block["links"]]
    assert len(ids) == len(set(ids))
    assert all(link["url"].startswith("https://") for link in block["links"])
    assert "maker_reference" in ids  # a maker was identified


def test_no_identification_gives_no_links():
    assert build_lookup_links({}, sites_file="")["links"] == []
    assert format_lookup_links(build_lookup_links({}, sites_file="")) == []


def test_user_file_can_add_replace_and_disable_sites(tmp_path):
    f = tmp_path / "sites.json"
    f.write_text(json.dumps([
        {"id": "mine", "name": "Mine", "kind": "sold", "url": "https://example.org/s?q={q}"},
        {"id": "catawiki", "disabled": True},
        {"id": "barnebys", "name": "Barnebys ES", "url": "https://www.barnebys.es/search?q={q}"},
    ]))
    ids = _by_id(build_lookup_links(IDENT, sites_file=f))
    assert "mine" in ids and "catawiki" not in ids
    assert ids["barnebys"]["url"].startswith("https://www.barnebys.es/")


def test_bad_user_file_is_ignored(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{not json")
    assert build_lookup_links(IDENT, sites_file=f)["links"]


def test_format_lists_every_link():
    block = build_lookup_links(IDENT, sites_file="")
    text = "\n".join(format_lookup_links(block))
    assert "RESEARCH LINKS" in text and "eBay - sold" in text
