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
