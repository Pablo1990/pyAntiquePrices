from __future__ import annotations

import datetime
from types import SimpleNamespace

import pytest

from pyantique_prices.gui_logic import (
    VERDICT_STYLE, backtest_lines, deal_inputs_from_form, deal_summary_lines, import_lines,
    key_source_urls, ledger_row, link_groups, manual_valuation, parse_number, summary_lines,
)
from pyantique_prices.lookup import build_lookup_links


def test_parse_number_accepts_spanish_and_english_formats():
    assert parse_number("1,5", "x") == 1.5
    assert parse_number("1.234,50", "x") == 1234.5
    assert parse_number("1,234.50", "x") == 1234.5
    assert parse_number(" 25 € ", "x") == 25.0
    assert parse_number("13%", "x") == 13.0
    assert parse_number("", "x") is None and parse_number("  ", "x", default=0.0) == 0.0


def test_parse_number_errors_are_readable():
    with pytest.raises(ValueError, match="Shipping: 'abc' is not a number"):
        parse_number("abc", "Shipping")
    with pytest.raises(ValueError, match="cannot be below"):
        parse_number("-1", "Shipping")
    with pytest.raises(ValueError, match="cannot be above"):
        parse_number("150", "VAT", maximum=100)


def test_deal_inputs_empty_asking_means_no_deal_check():
    asking, options = deal_inputs_from_form({"asking": ""})
    assert asking is None and options["shipping"] == 0.0 and options["for_resale"] is True


def test_deal_inputs_parse_all_fields():
    asking, o = deal_inputs_from_form({"asking": "120", "shipping": "8,5", "premium": "25", "vat": "21",
                                       "restoration": "30", "resale_fee": "13", "keep": True})
    assert asking == 120.0 and o == {"shipping": 8.5, "buyer_premium_pct": 25.0, "vat_pct": 21.0,
                                     "restoration": 30.0, "for_resale": False, "resale_fee_pct": 13.0}
    with pytest.raises(ValueError):
        deal_inputs_from_form({"asking": "0"})


def test_manual_valuation_defaults_and_validation():
    v = manual_valuation({"p50": "200"})
    assert v["p25"] == 140.0 and v["p75"] == 280.0 and v["num_comparables"] == 5
    with pytest.raises(ValueError, match="typical sold price"):
        manual_valuation({})
    with pytest.raises(ValueError, match="Low ≤ typical ≤ high"):
        manual_valuation({"p50": "200", "p25": "300"})


def test_manual_valuation_feeds_the_deal_check():
    from pyantique_prices.deals import assess_deal

    deal = assess_deal(manual_valuation({"p50": "300", "p25": "200", "p75": "420", "n": "12"}), asking_price=100)
    assert deal["verdict"] == "strong_buy"


def test_every_verdict_has_a_style():
    for verdict in ("strong_buy", "good_buy", "fair", "overpriced", "no_verdict"):
        label, bg, fg = VERDICT_STYLE[verdict]
        assert label and bg.startswith("#") and fg.startswith("#")


def test_link_groups_and_key_sources():
    block = build_lookup_links({"object_type": "pocket watch"}, sites_file="")
    titles = [t for t, _ in link_groups(block)]
    assert titles[0] == "Realised prices" and "Images" in titles
    urls = key_source_urls(block, 4)
    assert len(urls) == 4 and "LH_Sold=1" in urls[0]
    assert link_groups(None) == [] and key_source_urls(None) == []


def test_deal_summary_lines_include_warnings():
    from pyantique_prices.deals import assess_deal

    deal = assess_deal({"p25": 100, "p50": 150, "p75": 200, "num_comparables": 10, "valuation_available": True},
                       asking_price=50)
    text = "\n".join(deal_summary_lines(deal))
    assert "All-in cost" in text and "⚠" in text and "strong buy" in text
    assert deal_summary_lines(None) == []


def test_ledger_row_and_summary_lines():
    item = SimpleNamespace(id=3, title="Vase", status="sold", acquired_date=datetime.datetime(2026, 1, 5),
                           acquired_price=50, acquired_costs=10, estimate_mid=120.0, sold_price=150.0, sold_fees=15.0)
    assert ledger_row(item) == ("3", "Vase", "sold", "2026-01-05", "60.00", "120.00", "150.00", "+75.00")
    held = SimpleNamespace(id=4, title="Lamp", status="held", acquired_date=None, acquired_price=20,
                           acquired_costs=0, estimate_mid=None, sold_price=None, sold_fees=0)
    assert ledger_row(held)[-3:] == ("", "", "")
    assert "No items yet" in summary_lines({"items": 0})[0]
    s = {"items": 2, "held": 1, "sold": 1, "kept": 0, "capital_in_held": 30.0, "realised_profit": 75.0, "roi": 1.25,
         "win_rate": 1.0, "median_days_held": 31, "estimates": {"rated_items": 1, "median_sold_to_estimate": 1.25,
                                                                "share_within_estimated_range": 1.0}}
    text = "\n".join(summary_lines(s, factor=0.9))
    assert "+75.00" in text and "ROI: 125%" in text and "×0.90" in text


def test_backtest_and_import_lines():
    assert backtest_lines({"error": "need more"}) == ["need more"]
    report = {"targets_scored": 50, "valuation_rate": 0.9,
              "model": {"mdape": 0.4, "bias_median_log_ratio": 0.1},
              "baseline_unweighted_median": {"mdape": 0.5}, "baseline_global_median": {"mdape": 0.9},
              "interval_p25_p75": {"coverage": 0.55}, "interval_p10_p90": {"coverage": 0.82}}
    text = "\n".join(backtest_lines(report))
    assert "40%" in text and "1.11×" in text and "55%" in text
    result = SimpleNamespace(rows_processed=5, rows_inserted=4, rows_skipped=1, duplicates=1, invalid_prices=0,
                             unsupported_currencies=0, asking_excluded=1, outliers_flagged=2,
                             mixed_price_basis=True, hammer_only=2, final_with_premium=2)
    text = "\n".join(import_lines(result))
    assert "added: 4" in text and "HAMMER_PREMIUM_RATE" in text and "flagged" in text


def test_identification_from_fields_and_estimate_line():
    from pyantique_prices.gui_logic import estimate_line, identification_from_fields

    ident = identification_from_fields({"object": "vase", "maker": "Galle", "material": "glass", "artist": "  "})
    assert ident["object_type"] == "vase" and ident["materials"] == ["glass"]
    assert ident["manufacturer_candidates"][0]["name"] == "Galle" and ident["artist_candidates"] == []
    assert identification_from_fields({})["object_type"] is None

    assert "no comparable sales" in estimate_line(None)
    line = estimate_line({"currency": "EUR", "valuation_available": True,
                          "valuation": {"low": 150, "mid": 220, "high": 300, "num_comparables": 12}})
    assert "150 – 300 EUR (typical 220)" in line and "12 comparable sales" in line
    ref = estimate_line({"valuation": {"low": 1, "mid": 2, "high": 3, "num_comparables": 1,
                                       "confidence_note": "Very low confidence"}})
    assert ref.startswith("Rough reference only") and "1 comparable sale ·" in ref
