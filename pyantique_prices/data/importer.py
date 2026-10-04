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
    outliers_flagged: int = 0
    asking_excluded: int = 0
    hammer_only: int = 0
    final_with_premium: int = 0

    @property
    def mixed_price_basis(self) -> bool:
        """True when hammer-only and premium-inclusive prices are both present."""
        return self.hammer_only > 0 and self.final_with_premium > 0


SUPPORTED_CURRENCIES = {"EUR", "GBP", "USD", "CHF", "CAD", "AUD", "JPY"}


def _parse_jsonish(value):
    if value in (None, ""):
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _float_or_none(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def import_csv(
    path: str | Path,
    session,
    base_currency: str = "EUR",
    hammer_premium_rate: float = 0.0,
) -> ImportResult:
    """Import historical sales from a CSV file.

    Prices must be comparable, so each row's *price basis* is tracked:

    * ``final_price`` (hammer + buyer's premium) is preferred;
    * hammer-only rows are uplifted by ``hammer_premium_rate`` (e.g. 0.25) when
      given, otherwise kept as-is and counted in ``result.hammer_only`` (see
      ``result.mixed_price_basis``);
    * ``price_basis == "asking"`` rows are stored but never used as comparables
      or training data: asking prices are not sale prices.
    """
    from .models import HistoricalSale
    from .normalizer import normalize_price

    result = ImportResult()
    csv_path = Path(path)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            result.rows_processed += 1

            try:
                final_str = row.get("final_price") or ""
                hammer_str = row.get("hammer_price") or ""
                price = float(final_str or hammer_str) if (final_str or hammer_str) else None
            except (TypeError, ValueError):
                result.invalid_prices += 1
                result.rows_skipped += 1
                continue

            basis = (row.get("price_basis") or "realized").strip().lower()
            hammer = _float_or_none(hammer_str)
            if price is not None and not final_str and hammer is not None:
                # hammer-only row
                if hammer_premium_rate > 0:
                    price = hammer * (1.0 + hammer_premium_rate)
                    basis = "hammer_plus_estimated_premium"
                elif basis == "realized":
                    basis = "hammer"
                result.hammer_only += 1
            elif price is not None and final_str:
                result.final_with_premium += 1

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
                hammer_price=hammer,
                buyer_premium=_float_or_none(row.get("buyer_premium")),
                price_basis=basis,
                usable_for_training=basis != "asking",
                text_embedding=_parse_jsonish(row.get("text_embedding")),
                image_embedding=_parse_jsonish(row.get("image_embedding")),
            )
            if basis == "asking":
                result.asking_excluded += 1
            session.add(sale)
            result.rows_inserted += 1

    session.commit()
    if result.mixed_price_basis:
        logger.warning(
            "Mixed price bases: %d hammer-only vs %d premium-inclusive rows. Set "
            "HAMMER_PREMIUM_RATE (e.g. 0.25) so they are comparable.",
            result.hammer_only,
            result.final_with_premium,
        )
    try:
        from .outliers import flag_outliers

        result.outliers_flagged = flag_outliers(session)
    except Exception as exc:  # noqa: BLE001 - flagging must never lose an import
        logger.warning("Outlier flagging failed: %s", exc)
    return result
