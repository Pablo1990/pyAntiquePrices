#!/usr/bin/env python3
"""Evaluate pricing artifacts on a time-aware held-out split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyantique_prices.config import settings
from pyantique_prices.data.database import get_engine, get_session_factory
from pyantique_prices.data.models import HistoricalSale
from pyantique_prices.pricing.training import (
    evaluate_predictions,
    load_artifacts,
    prepare_training_records,
    time_aware_split,
)


def main() -> int:
    artifacts = load_artifacts("models")
    if artifacts is None:
        print("No model artifacts found. Run scripts/train_price_model.py first.")
        return 1

    engine = get_engine(settings.database_url)
    session_factory = get_session_factory(engine)
    with session_factory() as session:
        sales = (
            session.query(HistoricalSale)
            .filter(
                HistoricalSale.normalized_price.is_not(None),
                HistoricalSale.usable_for_training.is_(True),
            )
            .order_by(HistoricalSale.sale_date)
            .all()
        )
    records = prepare_training_records(sales)
    split = time_aware_split(records)
    report = {
        "validation": evaluate_predictions(
            artifacts["model"],
            split.validation,
            calibrator=artifacts["calibrator"],
        ),
        "test": evaluate_predictions(
            artifacts["model"],
            split.test,
            calibrator=artifacts["calibrator"],
        ),
        "split_strategy": "time_aware",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
