#!/usr/bin/env python3
"""Scrape antique auction data from public sources and populate the database.

Supported sources
-----------------
* eBay.es         – completed / sold listings
* Catawiki        – closed lots
* AIC             – American Institute for Conservation references
* LoC             – Library of Congress preservation resources

All scrapers respect ``robots.txt`` and apply polite crawl delays.  If a
source's ``robots.txt`` disallows the target path the scraper skips it
gracefully with a logged warning.

Usage
-----
.. code-block:: bash

    # Scrape all sources for a keyword (dry-run, no DB write)
    python scripts/scrape_sales.py --keywords "reloj bolsillo antiguo" --dry-run

    # Scrape all sources and save to the default database
    python scripts/scrape_sales.py --keywords "reloj bolsillo antiguo"

    # Scrape only specific sources
    python scripts/scrape_sales.py --keywords "porcelana" --sources ebay catawiki

    # Limit results per source
    python scripts/scrape_sales.py --keywords "plata" --max-results 20

    # Use a custom database URL
    python scripts/scrape_sales.py --keywords "mueble" --db-url sqlite:///./data/test.db
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as ``python scripts/scrape_sales.py`` without installing the
# package first.
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyantique_prices.config import settings
from pyantique_prices.data.database import create_tables, get_engine, get_session_factory
from pyantique_prices.data.importer import SUPPORTED_CURRENCIES
from pyantique_prices.data.models import HistoricalSale
from pyantique_prices.data.normalizer import normalize_price
from pyantique_prices.scraping.sources import (
    AICScraper,
    CatawikiScraper,
    EbayEsScraper,
    LibraryOfCongressScraper,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("scrape_sales")

_SCRAPERS = {
    "ebay": EbayEsScraper,
    "catawiki": CatawikiScraper,
    "aic": AICScraper,
    "loc": LibraryOfCongressScraper,
}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _save_records(
    session,
    records: list[dict],
    base_currency: str,
    dry_run: bool,
) -> tuple[int, int]:
    """Persist *records* to the database.  Returns ``(inserted, skipped)``."""
    inserted = skipped = 0

    for rec in records:
        source_url = rec.get("source_url")

        # Skip duplicates by source URL
        if source_url:
            exists = (
                session.query(HistoricalSale)
                .filter_by(source_url=source_url)
                .first()
            )
            if exists:
                skipped += 1
                continue

        currency = (rec.get("currency") or "EUR").upper()
        if currency not in SUPPORTED_CURRENCIES and rec.get("final_price") is not None:
            logger.debug("Unsupported currency %s – skipping %s", currency, source_url)
            skipped += 1
            continue

        price = rec.get("final_price")
        normalized = None
        if price is not None:
            try:
                normalized = normalize_price(price, currency, base_currency)
            except Exception:  # noqa: BLE001
                normalized = None

        import datetime

        sale_date_raw = rec.get("sale_date")
        sale_date = None
        if sale_date_raw:
            if isinstance(sale_date_raw, datetime.datetime):
                sale_date = sale_date_raw
            else:
                for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                    try:
                        sale_date = datetime.datetime.strptime(sale_date_raw, fmt)
                        break
                    except ValueError:
                        continue

        sale = HistoricalSale(
            title=rec.get("title"),
            description=rec.get("description"),
            category=rec.get("category"),
            auction_house=rec.get("auction_house"),
            sale_date=sale_date,
            currency=currency,
            final_price=price,
            source_url=source_url,
            original_currency=currency,
            original_price=price,
            normalized_currency=base_currency,
            normalized_price=normalized,
            price_basis=rec.get("price_basis", "realized"),
        )

        if not dry_run:
            session.add(sale)
        inserted += 1

    if not dry_run:
        session.commit()

    return inserted, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape antique auction data into the pyAntiquePrices database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--keywords",
        required=True,
        help="Search keywords (e.g. 'reloj bolsillo antiguo').",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(_SCRAPERS),
        default=list(_SCRAPERS),
        help="Which sources to scrape. Defaults to all.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="Maximum results to fetch per source (default: 50).",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help=(
            "SQLAlchemy database URL. "
            "Defaults to DATABASE_URL env / config (%(default)s)."
        ),
    )
    parser.add_argument(
        "--base-currency",
        default=None,
        help=(
            "ISO-4217 currency for price normalisation. "
            "Defaults to BASE_CURRENCY env / config."
        ),
    )
    parser.add_argument(
        "--crawl-delay",
        type=float,
        default=5.0,
        help="Minimum seconds between HTTP requests per scraper (default: 5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print results without writing to the database.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    db_url = args.db_url or settings.database_url
    base_currency = (args.base_currency or settings.base_currency).upper()

    logger.info("Keywords   : %s", args.keywords)
    logger.info("Sources    : %s", ", ".join(args.sources))
    logger.info("Max results: %d per source", args.max_results)
    logger.info("DB URL     : %s", db_url)
    logger.info("Dry-run    : %s", args.dry_run)

    engine = get_engine(db_url)
    create_tables(engine)
    session_factory = get_session_factory(engine)

    total_inserted = total_skipped = 0

    for source_key in args.sources:
        scraper_cls = _SCRAPERS[source_key]
        scraper = scraper_cls(crawl_delay=args.crawl_delay)
        logger.info("--- Scraping: %s ---", source_key)

        try:
            records = scraper.scrape(args.keywords, max_results=args.max_results)
        except Exception as exc:  # noqa: BLE001
            logger.error("%s scraper failed: %s", source_key, exc)
            continue

        logger.info("%s: %d raw records retrieved", source_key, len(records))

        if args.dry_run:
            for rec in records:
                print(
                    f"  [{source_key}] {rec.get('title', '(no title)')!r}"
                    f"  price={rec.get('final_price')} {rec.get('currency')}"
                    f"  date={rec.get('sale_date')}"
                    f"  url={rec.get('source_url')}"
                )
            total_inserted += len(records)
            continue

        with session_factory() as session:
            inserted, skipped = _save_records(
                session, records, base_currency, dry_run=False
            )

        logger.info("%s: inserted=%d  skipped=%d", source_key, inserted, skipped)
        total_inserted += inserted
        total_skipped += skipped

    print()
    if args.dry_run:
        print(f"Dry-run complete – {total_inserted} records would be inserted.")
    else:
        print(f"Done – inserted: {total_inserted}  skipped: {total_skipped}")


if __name__ == "__main__":
    main()
