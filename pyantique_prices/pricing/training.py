"""Training and evaluation helpers for local pricing artifacts."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import compute_metrics
from .features import condition_to_float


@dataclass
class DatasetSplit:
    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    test: list[dict[str, Any]]


def _text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("value") or "").strip().lower()
    if value is None:
        return ""
    return str(value).strip().lower()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _sale_key(record: dict[str, Any]) -> tuple:
    source_url = record.get("source_url") or ""
    if source_url:
        return ("source_url", source_url)
    return (
        "",
        record.get("auction_house") or "",
        record.get("lot_number") or "",
        record.get("sale_date") or "",
    )


def prepare_training_records(sales: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen = set()
    for sale in sales:
        price = getattr(sale, "normalized_price", None)
        if price is None or price <= 0:
            continue
        if getattr(sale, "usable_for_training", True) is False:
            continue
        key = _sale_key(
            {
                "source_url": getattr(sale, "source_url", None),
                "auction_house": getattr(sale, "auction_house", None),
                "lot_number": getattr(sale, "lot_number", None),
                "sale_date": getattr(sale, "sale_date", None),
            }
        )
        if any(key) and key in seen:
            continue
        if any(key):
            seen.add(key)

        sale_date = getattr(sale, "sale_date", None)
        records.append(
            {
                "object_type": _text(getattr(sale, "object_type", None)),
                "country": _text(getattr(sale, "country", None)),
                "condition_score": condition_to_float(getattr(sale, "condition", None)),
                "target_price": float(price),
                "target_log_price": float(np.log1p(price)),
                "sale_year": float(sale_date.year) if sale_date else 0.0,
                "sale_date": sale_date.isoformat() if sale_date else "",
                "source_url": getattr(sale, "source_url", None),
                "auction_house": getattr(sale, "auction_house", None),
                "lot_number": getattr(sale, "lot_number", None),
            }
        )
    records.sort(key=lambda item: item.get("sale_date") or "")
    return records


def time_aware_split(records: list[dict[str, Any]]) -> DatasetSplit:
    n = len(records)
    if n == 0:
        return DatasetSplit(train=[], validation=[], test=[])
    train_end = max(1, int(n * 0.7))
    val_end = max(train_end + 1, int(n * 0.85))
    val_end = min(val_end, n)
    return DatasetSplit(
        train=records[:train_end],
        validation=records[train_end:val_end],
        test=records[val_end:],
    )


def train_bucket_model(records: list[dict[str, Any]]) -> dict[str, Any]:
    prices = [row["target_price"] for row in records]
    global_median = float(np.median(prices)) if prices else 0.0
    by_object: dict[str, list[float]] = {}
    by_object_country: dict[str, list[float]] = {}
    for row in records:
        obj = row["object_type"]
        ctry = row["country"]
        by_object.setdefault(obj, []).append(row["target_price"])
        by_object_country.setdefault(f"{obj}|{ctry}", []).append(row["target_price"])

    model = {
        "model_type": "bucket_median_v1",
        "global_median": global_median,
        "by_object": {k: float(np.median(v)) for k, v in by_object.items()},
        "by_object_country": {
            k: float(np.median(v)) for k, v in by_object_country.items()
        },
        "residual_quantiles": {
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
        },
    }
    return model


def predict_price(model: dict[str, Any], features: dict[str, Any]) -> float:
    obj = _text(features.get("object_type"))
    ctry = _text(features.get("country"))
    key = f"{obj}|{ctry}"
    if key in model["by_object_country"]:
        return float(model["by_object_country"][key])
    if obj in model["by_object"]:
        return float(model["by_object"][obj])
    return float(model["global_median"])


def attach_residual_quantiles(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return model
    residuals = []
    for row in rows:
        pred = predict_price(model, row)
        residuals.append(row["target_price"] - pred)
    arr = np.array(residuals)
    model["residual_quantiles"] = {
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }
    return model


def predict_interval(
    model: dict[str, Any],
    features: dict[str, Any],
    calibrator: dict[str, float] | None = None,
) -> dict[str, float]:
    base = predict_price(model, features)
    cal = calibrator or {"scale": 1.0, "bias": 0.0}
    scale = _num(cal.get("scale"), 1.0)
    bias = _num(cal.get("bias"), 0.0)
    quantiles = model.get("residual_quantiles", {})
    p10 = max(0.0, scale * (base + _num(quantiles.get("p10"))) + bias)
    p25 = max(0.0, scale * (base + _num(quantiles.get("p25"))) + bias)
    p50 = max(0.0, scale * (base + _num(quantiles.get("p50"))) + bias)
    p75 = max(0.0, scale * (base + _num(quantiles.get("p75"))) + bias)
    p90 = max(0.0, scale * (base + _num(quantiles.get("p90"))) + bias)
    return {"p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90}


def evaluate_predictions(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    calibrator: dict[str, float] | None = None,
) -> dict[str, float]:
    if not rows:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "log_error": 0.0,
            "prediction_interval_coverage": 0.0,
            "prediction_interval_width": 0.0,
            "count": 0,
        }
    y_true = [row["target_price"] for row in rows]
    intervals = [predict_interval(model, row, calibrator=calibrator) for row in rows]
    y_pred = [item["p50"] for item in intervals]
    low = [item["p25"] for item in intervals]
    high = [item["p75"] for item in intervals]
    coverage = float(
        np.mean(
            [
                1.0 if low[idx] <= y_true[idx] <= high[idx] else 0.0
                for idx in range(len(y_true))
            ]
        )
    )
    width = float(np.mean([high[idx] - low[idx] for idx in range(len(y_true))]))
    metrics = compute_metrics(y_true, y_pred)
    metrics["prediction_interval_coverage"] = coverage
    metrics["prediction_interval_width"] = width
    metrics["count"] = len(rows)
    return metrics


def fit_calibrator(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    if not rows:
        return {"scale": 1.0, "bias": 0.0}
    preds = np.array([predict_price(model, row) for row in rows], dtype=float)
    true = np.array([row["target_price"] for row in rows], dtype=float)
    denom = float(np.dot(preds, preds))
    if math.isclose(denom, 0.0):
        return {"scale": 1.0, "bias": 0.0}
    scale = float(np.dot(preds, true) / denom)
    return {"scale": max(0.2, min(scale, 5.0)), "bias": 0.0}


def save_artifacts(
    model_dir: str | Path,
    model: dict[str, Any],
    calibrator: dict[str, float],
    metrics: dict[str, Any],
    feature_schema: dict[str, Any],
) -> None:
    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    quantile_dir = output / "quantile_models"
    quantile_dir.mkdir(parents=True, exist_ok=True)

    with (output / "price_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    with (output / "calibrator.pkl").open("wb") as handle:
        pickle.dump(calibrator, handle)

    for key, value in model.get("residual_quantiles", {}).items():
        with (quantile_dir / f"{key}.pkl").open("wb") as handle:
            pickle.dump({"quantile": key, "residual": float(value)}, handle)

    with (output / "feature_schema.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_schema, handle, indent=2)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def load_artifacts(model_dir: str | Path = "models") -> dict[str, Any] | None:
    root = Path(model_dir)
    model_path = root / "price_model.pkl"
    calibrator_path = root / "calibrator.pkl"
    if not model_path.exists() or not calibrator_path.exists():
        return None
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    with calibrator_path.open("rb") as handle:
        calibrator = pickle.load(handle)
    return {"model": model, "calibrator": calibrator}
