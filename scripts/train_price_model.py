#!/usr/bin/env python3
"""Train the local pricing model and save model artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyantique_prices.config import settings
from pyantique_prices.data.database import get_engine, get_session_factory
from pyantique_prices.data.models import HistoricalSale
from pyantique_prices.pricing.training import (
    attach_residual_quantiles,
    evaluate_predictions,
    fit_calibrator,
    prepare_training_records,
    save_artifacts,
    time_aware_split,
    train_bucket_model,
)


def main() -> int:
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
    print(f"Training rows after filtering/deduplication: {len(records)}")
    if len(records) < settings.min_comparables_for_model:
        print(
            "Not enough data to train "
            f"(need {settings.min_comparables_for_model}). Exiting."
        )
        return 0

    split = time_aware_split(records)
    model = train_bucket_model(split.train)
    model = attach_residual_quantiles(model, split.validation or split.train)
    calibrator = fit_calibrator(model, split.validation)

    metrics = {
        "train": evaluate_predictions(model, split.train, calibrator=calibrator),
        "validation": evaluate_predictions(
            model,
            split.validation,
            calibrator=calibrator,
        ),
        "test": evaluate_predictions(model, split.test, calibrator=calibrator),
        "model_type": model["model_type"],
    }
    feature_schema = {
        "features": [
            "object_type",
            "country",
            "condition_score",
            "num_comparables",
            "median_comparable_price",
            "mean_comparable_price",
            "comparable_price_iqr",
            "comparable_price_std",
        ],
        "target": settings.price_target,
        "num_training_samples": len(split.train),
        "num_validation_samples": len(split.validation),
        "num_test_samples": len(split.test),
        "split_strategy": "time_aware",
    }

    save_artifacts(
        model_dir="models",
        model=model,
        calibrator=calibrator,
        metrics=metrics,
        feature_schema=feature_schema,
    )
    print("Saved artifacts:")
    print(" - models/price_model.pkl")
    print(" - models/quantile_models/p10.pkl ... p90.pkl")
    print(" - models/calibrator.pkl")
    print(" - models/feature_schema.json")
    print(" - models/metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
