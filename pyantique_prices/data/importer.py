"""CSV importer for historical sales data."""

from __future__ import annotations

import csv
import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    rows_processed: int = 0
    rows_inserted: int = 0
    rows_updated: int = 0
    rows_skipped: int = 0
    duplicates: int = 0
    invalid_prices: int = 0
    unsupported_currencies: int = 0


SUPPORTED_CURRENCIES = {"EUR", "GBP", "USD", "CHF", "CAD", "AUD", "JPY"}


def _parse_jsonish(value):
    if value in (None, ""):
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def import_csv(path: str | Path, session, base_currency: str = "EUR") -> ImportResult:
    """Import historical sales from a CSV file."""
    from .models import HistoricalSale
    from .normalizer import normalize_price

    result = ImportResult()
    csv_path = Path(path)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            result.rows_processed += 1

            try:
                price_str = row.get("final_price") or row.get("hammer_price") or ""
                price = float(price_str) if price_str else None
            except (TypeError, ValueError):
                result.invalid_prices += 1
                result.rows_skipped += 1
                continue

            currency = (row.get("currency") or "EUR").upper()
            if currency not in SUPPORTED_CURRENCIES:
                result.unsupported_currencies += 1
                result.rows_skipped += 1
                continue

            source_url = row.get("source_url")
            if source_url:
                existing = (
                    session.query(HistoricalSale)
                    .filter_by(source_url=source_url)
                    .first()
                )
                if existing:
                    result.duplicates += 1
                    result.rows_skipped += 1
                    continue

            sale_date = None
            date_str = row.get("sale_date")
            if date_str:
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        sale_date = datetime.datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue

            normalized = None
            if price is not None:
                normalized = normalize_price(price, currency, base_currency)

            sale = HistoricalSale(
                title=row.get("title"),
                description=row.get("description"),
                category=row.get("category"),
                subcategory=row.get("subcategory"),
                object_type=row.get("object_type"),
                period=row.get("period"),
                manufacturer=row.get("manufacturer"),
                artist=row.get("artist"),
                workshop=row.get("workshop"),
                material=row.get("material"),
                technique=row.get("technique"),
                condition=row.get("condition"),
                country=row.get("country"),
                region=row.get("region"),
                marks=row.get("marks"),
                provenance=row.get("provenance"),
                auction_house=row.get("auction_house"),
                sale_date=sale_date,
                currency=currency,
                final_price=price,
                image_urls=_parse_jsonish(row.get("image_urls")),
                source_url=source_url,
                original_currency=currency,
                original_price=price,
                normalized_currency=base_currency,
                normalized_price=normalized,
                price_basis=row.get("price_basis", "realized"),
                text_embedding=_parse_jsonish(row.get("text_embedding")),
                image_embedding=_parse_jsonish(row.get("image_embedding")),
            )
            session.add(sale)
            result.rows_inserted += 1

    session.commit()
    return result
