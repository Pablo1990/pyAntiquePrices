#!/usr/bin/env python3
"""Import historical auction sales from a CSV file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyantique_prices.config import settings
from pyantique_prices.data.database import create_tables, get_engine, get_session_factory
from pyantique_prices.data.importer import import_csv


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/import_sales.py <path/to/sales.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    engine = get_engine(settings.database_url)
    create_tables(engine)
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        result = import_csv(
            csv_path,
            session,
            base_currency=settings.base_currency,
            hammer_premium_rate=settings.hammer_premium_rate,
        )

    print(f"Rows processed:          {result.rows_processed}")
    print(f"Rows inserted:           {result.rows_inserted}")
    print(f"Rows updated:            {result.rows_updated}")
    print(f"Rows skipped:            {result.rows_skipped}")
    print(f"Duplicates:              {result.duplicates}")
    print(f"Invalid prices:          {result.invalid_prices}")
    print(f"Unsupported currencies:  {result.unsupported_currencies}")
    print(f"Outliers flagged:        {result.outliers_flagged}")
    print(f"Asking prices (unused):  {result.asking_excluded}")
    if result.mixed_price_basis:
        print(
            f"WARNING: {result.hammer_only} hammer-only and {result.final_with_premium} "
            "premium-inclusive prices are mixed. Set HAMMER_PREMIUM_RATE in .env."
        )


if __name__ == "__main__":
    main()
