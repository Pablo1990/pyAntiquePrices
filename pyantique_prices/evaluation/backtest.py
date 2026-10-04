"""Point-in-time backtest of retrieval + pricing against realised sale prices.

For every held-out sale we pretend it is a new appraisal:

1. build the structured identification the vision model *would* have produced
   from the sale's own metadata (so this isolates retrieval + pricing from
   vision errors);
2. retrieve comparables using only sales **strictly before** the held-out
   date, excluding the sale itself (no look-ahead, no self-match);
3. price it with :class:`PricePredictor`;
4. compare with the realised ``normalized_price``.

Reported against two baselines so improvements are measurable:
``global_median`` (median of all earlier prices) and ``unweighted_median``
(plain median of the retrieved comparables -- the pre-fix estimator).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import median
from typing import Any

import numpy as np


@dataclass
class BacktestConfig:
    min_history: int = 30  # earlier sales required before a target is scored
    max_targets: int | None = 300
    top_k: int = 20
    min_similarity: float = 0.05
    min_data_quality_score: float = 0.4
    max_sale_age_years: int = 200  # backtest on whatever history exists
    seed: int = 7


def sale_to_identification(sale: Any) -> dict:
    """Structured identification equivalent to a perfect vision pass."""

    def cand(value):
        return [{"name": str(value), "confidence": 0.9}] if value else []

    materials = [m.strip() for m in str(sale.material or "").split(",") if m.strip()]
    marks = [{"text": m.strip(), "normalized_text": m.strip().lower(), "confidence": 0.9}
             for m in str(sale.marks or "").split(",") if m.strip()]
    return {
        "object_type": sale.object_type,
        "period": sale.period,
        "likely_period": sale.period,
        "manufacturer_candidates": cand(sale.manufacturer),
        "artist_candidates": cand(sale.artist),
        "workshop_candidates": cand(sale.workshop),
        "materials": materials,
        "country": sale.country,
        "region": sale.region,
        "condition": sale.condition,
        "height": sale.height,
        "width": sale.width,
        "depth": sale.depth,
        "diameter": sale.diameter,
        "weight": sale.weight,
        "marks": marks,
    }


def _metrics(rows: list[dict], key: str) -> dict:
    """Error metrics for the predictor named ``key`` over rows that have it."""
    pairs = [(r["actual"], r[key]) for r in rows if r.get(key) and r[key] > 0]
    if not pairs:
        return {"n": 0}
    actual = np.array([a for a, _ in pairs])
    pred = np.array([p for _, p in pairs])
    log_ratio = np.log(pred / actual)
    return {
        "n": len(pairs),
        "mdape": float(np.median(np.abs(pred - actual) / actual)),
        "mean_abs_log_error": float(np.mean(np.abs(log_ratio))),
        "bias_median_log_ratio": float(np.median(log_ratio)),  # >0 = overestimates
    }


def _coverage(rows: list[dict], lo: str, hi: str) -> dict:
    sel = [r for r in rows if r.get(lo) and r.get(hi)]
    if not sel:
        return {"n": 0}
    inside = [r[lo] <= r["actual"] <= r[hi] for r in sel]
    width = [math.log(r[hi] / r[lo]) for r in sel if r[lo] > 0]
    return {
        "n": len(sel),
        "coverage": float(np.mean(inside)),
        "median_log_width": float(np.median(width)) if width else None,
    }


def summarize(rows: list[dict]) -> dict:
    priced = [r for r in rows if r.get("estimate")]
    report = {
        "targets_scored": len(rows),
        "with_valuation": len(priced),
        "valuation_rate": len(priced) / len(rows) if rows else 0.0,
        "model": _metrics(priced, "estimate"),
        "baseline_unweighted_median": _metrics(priced, "unweighted_median"),
        "baseline_global_median": _metrics(rows, "global_median"),
        # Nominal coverage: P25-P75 should contain ~50 %, P10-P90 ~80 %.
        "interval_p25_p75": {**_coverage(priced, "p25", "p75"), "nominal": 0.50},
        "interval_p10_p90": {**_coverage(priced, "p10", "p90"), "nominal": 0.80},
    }
    by_type: dict[str, list[dict]] = {}
    for r in priced:
        by_type.setdefault(r["object_type"] or "unknown", []).append(r)
    report["by_object_type"] = {
        k: {**_metrics(v, "estimate"), "baseline_unweighted_median": _metrics(v, "unweighted_median").get("mdape")}
        for k, v in sorted(by_type.items(), key=lambda kv: -len(kv[1]))[:15]
        if len(v) >= 5
    }
    bins = {"3-5": (3, 5), "6-9": (6, 9), "10+": (10, 10**9)}
    report["by_comparable_count"] = {
        label: _metrics([r for r in priced if lo <= r["n_comparables"] <= hi], "estimate")
        for label, (lo, hi) in bins.items()
    }
    return report


def run_backtest(session, predictor, config: BacktestConfig | None = None) -> dict:
    """Run the point-in-time backtest and return a JSON-serialisable report."""
    from pyantique_prices.data.models import HistoricalSale
    from pyantique_prices.pricing.features import extract_features
    from pyantique_prices.retrieval.comparables import retrieve_comparables_details

    config = config or BacktestConfig()
    sales = (
        session.query(HistoricalSale)
        .filter(
            HistoricalSale.normalized_price.is_not(None),
            HistoricalSale.normalized_price > 0,
            HistoricalSale.usable_for_training.is_not(False),
            HistoricalSale.outlier_flag.is_not(True),
            HistoricalSale.sale_date.is_not(None),
        )
        .order_by(HistoricalSale.sale_date)
        .all()
    )
    if len(sales) <= config.min_history:
        return {"error": f"need more than {config.min_history} dated, priced sales; found {len(sales)}"}

    targets = sales[config.min_history:]
    if config.max_targets and len(targets) > config.max_targets:
        targets = random.Random(config.seed).sample(targets, config.max_targets)

    rows: list[dict] = []
    for sale in targets:
        earlier_prices = [s.normalized_price for s in sales if s.sale_date < sale.sale_date]
        if len(earlier_prices) < config.min_history:
            continue
        ident = sale_to_identification(sale)
        details = retrieve_comparables_details(
            session,
            ident,
            top_k=config.top_k,
            min_similarity=config.min_similarity,
            max_sale_age_years=config.max_sale_age_years,
            min_data_quality_score=config.min_data_quality_score,
            as_of=sale.sale_date,
            exclude_ids=[sale.id],
        )
        comps = details["comparables"]
        row: dict[str, Any] = {
            "actual": float(sale.normalized_price),
            "object_type": (sale.object_type or "").strip().lower(),
            "n_comparables": len(comps),
            "global_median": float(median(earlier_prices)),
        }
        prices = [c["normalized_price"] for c in comps if c.get("normalized_price")]
        if prices:
            row["unweighted_median"] = float(np.median(prices))
        valuation = predictor.predict(extract_features(ident, comps), comps) if comps else None
        if valuation and valuation.get("valuation_available"):
            row.update(
                estimate=valuation["p50"], p10=valuation["p10"], p25=valuation["p25"],
                p75=valuation["p75"], p90=valuation["p90"],
            )
        rows.append(row)

    report = summarize(rows)
    report["config"] = config.__dict__
    return report
