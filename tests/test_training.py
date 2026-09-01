from __future__ import annotations

import datetime

from pyantique_prices.data.models import HistoricalSale
from pyantique_prices.pricing.training import (
    evaluate_predictions,
    fit_calibrator,
    prepare_training_records,
    time_aware_split,
    train_bucket_model,
)


def _sale(idx: int, price: float, source_url: str) -> HistoricalSale:
    return HistoricalSale(
        title=f"Sale {idx}",
        object_type="clock",
        country="France",
        condition="good",
        normalized_price=price,
        usable_for_training=True,
        source_url=source_url,
        sale_date=datetime.datetime(2020, 1, 1) + datetime.timedelta(days=idx),
    )


def test_prepare_training_records_deduplicates_source_url_and_date():
    sales = [
        _sale(1, 100.0, "https://example.com/1"),
        _sale(2, 120.0, "https://example.com/1"),
        _sale(3, 130.0, "https://example.com/3"),
    ]
    records = prepare_training_records(sales)
    assert len(records) == 2
    assert records[0]["target_price"] == 100.0
    assert records[1]["target_price"] == 130.0


def test_time_aware_split_preserves_order():
    records = [{"sale_date": f"2020-01-{i:02d}"} for i in range(1, 11)]
    split = time_aware_split(records)
    assert len(split.train) >= 1
    assert split.train[0]["sale_date"] < split.train[-1]["sale_date"]
    if split.validation and split.test:
        assert split.validation[-1]["sale_date"] < split.test[0]["sale_date"]


def test_evaluate_predictions_and_calibrator_return_expected_keys():
    rows = [
        {"object_type": "clock", "country": "france", "target_price": 100.0},
        {"object_type": "clock", "country": "france", "target_price": 130.0},
        {"object_type": "clock", "country": "france", "target_price": 160.0},
    ]
    model = train_bucket_model(rows)
    calibrator = fit_calibrator(model, rows)
    metrics = evaluate_predictions(model, rows, calibrator=calibrator)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "prediction_interval_coverage" in metrics
    assert metrics["count"] == 3
