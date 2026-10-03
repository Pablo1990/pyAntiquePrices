from __future__ import annotations

from pathlib import Path

from pyantique_prices.data.database import create_tables, get_engine, get_session_factory
from pyantique_prices.data.importer import import_csv
from pyantique_prices.data.models import HistoricalSale

FIXTURES = Path(__file__).parent / "fixtures"


def _make_session():
    engine = get_engine("sqlite:///:memory:")
    create_tables(engine)
    session_factory = get_session_factory(engine)
    return session_factory()


def test_import_csv_inserts_and_normalizes():
    with _make_session() as session:
        result = import_csv(FIXTURES / "sales_valid.csv", session, base_currency="EUR")
        sale = session.query(HistoricalSale).one()

    assert result.rows_processed == 1
    assert result.rows_inserted == 1
    assert sale.title == "Victorian Silver Pocket Watch"
    assert sale.normalized_price == 460.0


def test_import_csv_skips_duplicates():
    with _make_session() as session:
        first = import_csv(FIXTURES / "sales_valid.csv", session, base_currency="EUR")
        second = import_csv(FIXTURES / "sales_valid.csv", session, base_currency="EUR")

    assert first.rows_inserted == 1
    assert second.duplicates == 1
    assert second.rows_skipped == 1


def test_import_csv_counts_invalid_and_unsupported_rows():
    with _make_session() as session:
        result = import_csv(FIXTURES / "sales_invalid.csv", session, base_currency="EUR")
        count = session.query(HistoricalSale).count()

    assert result.rows_processed == 2
    assert result.invalid_prices == 1
    assert result.unsupported_currencies == 1
    assert result.rows_inserted == 0
    assert count == 0


def _write_csv(path, rows):
    import csv

    fields = ["title", "object_type", "currency", "final_price", "hammer_price", "price_basis", "source_url"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def test_asking_prices_are_stored_but_never_used_for_training(tmp_path):
    path = tmp_path / "s.csv"
    _write_csv(path, [{"title": "a", "object_type": "vase", "currency": "EUR",
                       "final_price": "100", "price_basis": "asking", "source_url": "u1"}])
    with _make_session() as session:
        result = import_csv(path, session)
        sale = session.query(HistoricalSale).one()
    assert result.asking_excluded == 1
    assert sale.usable_for_training is False


def test_hammer_only_rows_get_premium_uplift_and_mixed_basis_is_reported(tmp_path):
    path = tmp_path / "s.csv"
    _write_csv(path, [
        {"title": "h", "object_type": "vase", "currency": "EUR", "hammer_price": "100", "source_url": "u1"},
        {"title": "f", "object_type": "vase", "currency": "EUR", "final_price": "125", "source_url": "u2"},
    ])
    with _make_session() as session:
        result = import_csv(path, session, hammer_premium_rate=0.25)
        hammer = session.query(HistoricalSale).filter_by(title="h").one()
    assert hammer.normalized_price == 125.0
    assert hammer.price_basis == "hammer_plus_estimated_premium"
    assert result.mixed_price_basis is True

    with _make_session() as session:
        raw = import_csv(path, session)  # no uplift configured
        hammer = session.query(HistoricalSale).filter_by(title="h").one()
    assert hammer.normalized_price == 100.0 and hammer.price_basis == "hammer"
    assert raw.mixed_price_basis is True
