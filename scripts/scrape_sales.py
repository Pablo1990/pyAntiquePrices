#!/usr/bin/env python3
"""Scrape antique auction data from public sources and populate the database.

Supported sources
-----------------
* eBay API        – official eBay REST APIs (needs EBAY_CLIENT_ID/SECRET).
                    Writing eBay data to the DB requires written permission
                    from eBay – see ``--ebay-data-permission``.
* eBay.es         – completed / sold listings via HTML (usually blocked by
                    eBay's robots.txt – prefer ``ebay_api``)
* AIC             – American Institute for Conservation references
* LoC             – Library of Congress preservation resources

All scrapers respect ``robots.txt`` and apply polite crawl delays.  If a
source's ``robots.txt`` disallows the target path the scraper skips it
gracefully with a logged warning.

Usage
-----
.. code-block:: bash


    # Check the eBay API connection without storing anything
    python scripts/scrape_sales.py --keywords "reloj bolsillo" --sources ebay_api --ebay-api-mode active --dry-run

    # Store eBay results – ONLY with written permission from eBay
    python scripts/scrape_sales.py --keywords "reloj bolsillo" --sources ebay_api --ebay-data-permission

    # Reference records (no prices) from AIC, into a custom database
    python scripts/scrape_sales.py --keywords "mueble" --sources aic --db-url sqlite:///./data/test.db
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
    EbayApiError,
    EbayApiScraper,
    EbayEsScraper,
    LibraryOfCongressScraper,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("scrape_sales")

_SCRAPERS = {
    "ebay_api": EbayApiScraper,
    "ebay": EbayEsScraper,
    "aic": AICScraper,
    "loc": LibraryOfCongressScraper,
}

# No source runs by default: none of them can legally fill a price DB as-is.
#   ebay_api – eBay API licence forbids storing data without written permission
#   ebay     – eBay robots.txt disallows /sch/
#   aic/loc  – reference records without prices (LoC /search is disallowed)
# Catawiki was removed: its General Terms (15 Sep 2026, Art. 8) forbid scraping.
# Fill the DB from data you hold rights to with scripts/import_sales.py.

# Sources whose terms forbid storing their data without a separate licence.
_EBAY_LICENSE_NOTICE = """\
Refusing to store eBay data in the database.

The eBay API License Agreement (3 Sep 2025) does not allow, without eBay's
express written permission:
  * storing or copying eBay Content beyond temporary copies   (s. 9(g), 3.1)
  * using eBay Content to suggest or model prices             (s. 9(e))
  * using eBay Content to train algorithms / machine learning (s. 9(j))
A historical price DB for appraisal / model training needs all three.

Options:
  * Ask eBay for a data licence covering this use (e.g. via the Developer
    Program / Marketplace Insights application) and, once granted in writing,
    re-run with --ebay-data-permission.
  * Use --dry-run to check your API connection without storing anything.
  * Fill the DB from licensed sources instead: scripts/import_sales.py CSV.
Full text: https://developer.ebay.com/join/api-license-agreement
"""


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
        required=True,
        help=(
            "Which sources to scrape. 'ebay_api' "
            "only stores data with --ebay-data-permission. 'ebay' is the "
            "HTML scraper, blocked by eBay's robots.txt. 'aic' and 'loc' "
            "return reference records without prices."
        ),
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
    parser.add_argument(
        "--ebay-api-mode",
        choices=["sold", "active"],
        default="sold",
        help=(
            "For --sources ebay_api: 'sold' uses the Marketplace Insights API "
            "(realised prices, restricted access); 'active' uses the Browse "
            "API (asking prices of live listings). Default: sold."
        ),
    )
    parser.add_argument(
        "--ebay-data-permission",
        action="store_true",
        help=(
            "Confirm you hold express written permission from eBay to store "
            "eBay Content and use it for price modelling / ML. Required to "
            "write ebay_api results to the database."
        ),
    )
    parser.add_argument(
        "--ebay-marketplace",
        default=None,
        help="eBay marketplace ID, e.g. EBAY_ES, EBAY_GB (default: EBAY_MARKETPLACE_ID env).",
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

    if (
        "ebay_api" in args.sources
        and not args.dry_run
        and not args.ebay_data_permission
    ):
        print(_EBAY_LICENSE_NOTICE, file=sys.stderr)
        args.sources = [src for src in args.sources if src != "ebay_api"]
        if not args.sources:
            raise SystemExit(2)

    engine = get_engine(db_url)
    create_tables(engine)
    session_factory = get_session_factory(engine)

    total_inserted = total_skipped = 0

    for source_key in args.sources:
        scraper_cls = _SCRAPERS[source_key]
        logger.info("--- Scraping: %s ---", source_key)

        try:
            if scraper_cls is EbayApiScraper:
                scraper = scraper_cls(
                    crawl_delay=args.crawl_delay,
                    mode=args.ebay_api_mode,
                    marketplace_id=args.ebay_marketplace,
                )
            else:
                scraper = scraper_cls(crawl_delay=args.crawl_delay)
            records = scraper.scrape(args.keywords, max_results=args.max_results)
        except EbayApiError as exc:
            logger.error("%s: %s", source_key, exc)
            continue
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
