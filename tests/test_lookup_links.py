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


def test_google_site_search_is_used_for_site_entries_and_is_encoded():
    block = build_lookup_links(IDENT, sites_file="", regions=["all"])
    bonhams = _by_id(block)["bonhams"]
    qs = parse_qs(urlparse(bonhams["url"]).query)
    assert qs["q"] == ["Omega pocket watch hunter case silver site:bonhams.com"]
    assert bonhams["verified"] is False and _by_id(block)["catawiki"]["verified"] is True


def test_regions_filter_sites():
    es = {l["id"] for l in build_lookup_links(IDENT, sites_file="", regions=["es"])["links"]}
    assert "todocoleccion" in es and "saleroom" not in es and "drouot" not in es
    allr = {l["id"] for l in build_lookup_links(IDENT, sites_file="", regions=["all"])["links"]}
    assert {"drouot", "saleroom", "heritage", "lot_tissimo"} <= allr


def test_default_regions_and_env_override(monkeypatch):
    default = {l["id"] for l in build_lookup_links(IDENT, sites_file="")["links"]}
    assert "saleroom" in default and "drouot" not in default
    monkeypatch.setenv("LOOKUP_REGIONS", "fr")
    fr = {l["id"] for l in build_lookup_links(IDENT, sites_file="")["links"]}
    assert "drouot" in fr and "todocoleccion" not in fr


def test_category_sites_follow_the_object():
    watch = {l["id"] for l in build_lookup_links(IDENT, sites_file="")["links"]}
    assert "chrono24" in watch and "numista" not in watch and "abebooks" not in watch
    coin = {l["id"] for l in build_lookup_links({"object_type": "silver coin"}, sites_file="", regions=["all"])["links"]}
    assert "numista" in coin and "chrono24" not in coin
    forced = {l["id"] for l in build_lookup_links({"object_type": "thing"}, sites_file="", categories=["books"])["links"]}
    assert "abebooks" in forced


def test_links_are_grouped_realised_first_and_formatter_groups():
    block = build_lookup_links(IDENT, sites_file="", regions=["all"])
    kinds = [l["kind"] for l in block["links"]]
    assert kinds == sorted(kinds, key=["sold", "auction", "active", "retail", "reference", "images"].index)
    text = "\n".join(format_lookup_links(block))
    assert "Realised prices:" in text and "Triangulate" in text and "Images:" in text


def test_user_site_entry_can_use_site_search(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps([{"id": "mine", "name": "Mine", "kind": "auction", "site": "example.org"}]))
    assert "site%3Aexample.org" in _by_id(build_lookup_links(IDENT, sites_file=f))["mine"]["url"]
