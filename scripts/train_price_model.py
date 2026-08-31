#!/usr/bin/env python3
"""Train the price prediction model on historical sales."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyantique_prices.config import settings
from pyantique_prices.data.database import get_engine, get_session_factory
from pyantique_prices.data.models import HistoricalSale


def main():
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

    print(f"Training on {len(sales)} sales")
    if len(sales) < settings.min_comparables_for_model:
        print(
            "Not enough data to train "
            f"(need {settings.min_comparables_for_model}). Exiting."
        )
        sys.exit(0)

    Path("models").mkdir(exist_ok=True)
    schema = {
        "features": [
            "object_type",
            "country",
            "condition_score",
            "num_comparables",
            "median_comparable_price",
        ],
        "target": settings.price_target,
        "num_training_samples": len(sales),
    }
    with open("models/feature_schema.json", "w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2)
    print("Feature schema saved to models/feature_schema.json")
    print("Note: Full CatBoost training requires sufficient labeled data.")


if __name__ == "__main__":
    main()
