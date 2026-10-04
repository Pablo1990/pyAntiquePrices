from __future__ import annotations

import csv
import datetime

import pytest

from pyantique_prices.data.database import create_tables, get_engine, get_session_factory
from pyantique_prices.data.models import HistoricalSale
from pyantique_prices.ledger import (
    add_purchase, calibration_factor, export_sales_csv, list_items, record_sale, summarize, sync_to_sales,
)

IDENT = {"object_type": {"value": "vase"}, "manufacturer_candidates": [{"name": "Galle", "confidence": 0.8}],
         "materials": ["glass"], "country": "France"}
VAL = {"p25": 80.0, "p50": 120.0, "p75": 180.0, "method": "similarity_weighted_estimate"}


def _s():
    engine = get_engine("sqlite:///:memory:")
    create_tables(engine)
    return get_session_factory(engine)()


def test_add_purchase_snapshots_identification_and_estimate():
    with _s() as s:
        item = add_purchase(s, title="Galle vase", price=50, costs=10, date="2026-01-05",
                            where="flea market", identification=IDENT, valuation=VAL,
                            deal={"verdict": "strong_buy"}, notes="chip on base")
        assert (item.object_type, item.manufacturer, item.material) == ("vase", "Galle", "glass")
        assert (item.estimate_low, item.estimate_mid, item.estimate_high) == (80, 120, 180)
        assert item.deal_verdict == "strong_buy" and item.notes == "chip on base"
        assert item.status == "held"


def test_sale_and_summary_profit_roi_and_estimate_accuracy():
    with _s() as s:
        a = add_purchase(s, title="a", price=50, costs=10, date="2026-01-01", valuation=VAL, deal={"verdict": "strong_buy"})
        b = add_purchase(s, title="b", price=100, date="2026-01-01", valuation=VAL, deal={"verdict": "fair"})
        add_purchase(s, title="c", price=30)  # still held
        record_sale(s, a.id, price=150, fees=15, date="2026-02-01")   # profit 150-15-60 = 75
        record_sale(s, b.id, price=90, fees=5, date="2026-03-02")     # profit 90-5-100 = -15
        out = summarize(s)
    assert (out["items"], out["sold"], out["held"]) == (3, 2, 1)
    assert out["realised_profit"] == 60.0
    assert out["roi"] == pytest.approx(60 / 160)
    assert out["win_rate"] == 0.5
    assert out["capital_in_held"] == 30.0
    assert out["estimates"]["rated_items"] == 2
    assert out["estimates"]["share_within_estimated_range"] == 1.0  # 150 and 90 both in [80, 180]
    assert out["estimates"]["median_sold_to_estimate"] == pytest.approx(0.968, abs=0.001)  # sqrt(150/120 * 90/120)
    assert out["by_verdict"]["strong_buy"]["profit"] == 75.0 and out["by_verdict"]["fair"]["wins"] == 0
    assert out["median_days_held"] == 45.5


def test_calibration_needs_enough_sales_then_uses_median_ratio():
    with _s() as s:
        for i in range(4):
            it = add_purchase(s, title=f"i{i}", price=10, valuation=VAL)
            record_sale(s, it.id, price=60)  # sold at half the 120 estimate
        assert calibration_factor(s) == 1.0  # only 4 sales
        it = add_purchase(s, title="i5", price=10, valuation=VAL)
        record_sale(s, it.id, price=60)
        assert calibration_factor(s) == pytest.approx(0.5)


def test_items_without_estimate_are_ignored_by_calibration():
    with _s() as s:
        for i in range(6):
            it = add_purchase(s, title=str(i), price=10)  # no valuation snapshot
            record_sale(s, it.id, price=99)
        assert calibration_factor(s) == 1.0


def test_validation_errors():
    with _s() as s:
        with pytest.raises(ValueError):
            add_purchase(s, title="x", price=-1)
        it = add_purchase(s, title="x", price=1)
        record_sale(s, it.id, price=5)
        with pytest.raises(ValueError):
            record_sale(s, it.id, price=6)
        with pytest.raises(KeyError):
            record_sale(s, 999, price=1)
        with pytest.raises(ValueError):
            list_items(s, "bogus")


def test_export_and_sync_use_only_sold_items_and_are_idempotent(tmp_path):
    with _s() as s:
        sold = add_purchase(s, title="sold vase", price=10, identification=IDENT, currency="EUR")
        add_purchase(s, title="held", price=10)
        record_sale(s, sold.id, price=75, date="2026-02-01", where="eBay")
        path = tmp_path / "mine.csv"
        assert export_sales_csv(s, path) == 1
        rows = list(csv.DictReader(path.open()))
        assert rows[0]["title"] == "sold vase" and rows[0]["final_price"] == "75.0"
        assert rows[0]["price_basis"] == "realized" and rows[0]["source_url"].startswith("ledger://")

        assert sync_to_sales(s) == 1
        assert sync_to_sales(s) == 0
        sale = s.query(HistoricalSale).one()
        assert sale.normalized_price == 75.0 and sale.object_type == "vase"
        assert sale.sale_date == datetime.datetime(2026, 2, 1)


def test_exported_csv_round_trips_through_the_importer(tmp_path):
    from pyantique_prices.data.importer import import_csv

    with _s() as s:
        it = add_purchase(s, title="v", price=10, identification=IDENT)
        record_sale(s, it.id, price=75, date="2026-02-01")
        path = tmp_path / "mine.csv"
        export_sales_csv(s, path)
    with _s() as other:
        result = import_csv(path, other)
        assert result.rows_inserted == 1 and other.query(HistoricalSale).one().normalized_price == 75.0
