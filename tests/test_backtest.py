"""Backtest harness: no look-ahead, sane report, beats naive baselines on synthetic data."""

from __future__ import annotations

import datetime
import math
import random

from pyantique_prices.data.database import create_tables, get_engine, get_session_factory
from pyantique_prices.data.models import HistoricalSale
from pyantique_prices.evaluation import BacktestConfig, run_backtest
from pyantique_prices.pricing.model import PricePredictor
from pyantique_prices.retrieval.comparables import retrieve_comparables_details

BASE = {"vase": 200.0, "clock": 800.0, "painting": 3000.0, "lamp": 120.0}
MAKER = {"Meissen": 2.5, "Generic": 1.0, "Galle": 4.0}


def _session(n=240, seed=1):
    engine = get_engine("sqlite:///:memory:")
    create_tables(engine)
    session = get_session_factory(engine)()
    rng = random.Random(seed)
    start = datetime.datetime(2015, 1, 1)
    for i in range(n):
        obj = rng.choice(list(BASE))
        maker = rng.choice(list(MAKER))
        price = BASE[obj] * MAKER[maker] * math.exp(rng.gauss(0, 0.25))
        session.add(HistoricalSale(
            title=f"{maker} {obj}", object_type=obj, manufacturer=maker, country="France",
            material="glass", normalized_price=round(price, 2), source_url=f"u{i}",
            sale_date=start + datetime.timedelta(days=i * 10), usable_for_training=True,
            auction_house="X",
        ))
    session.commit()
    return session


def test_point_in_time_retrieval_never_sees_future_or_self():
    with _session() as session:
        target = session.query(HistoricalSale).order_by(HistoricalSale.sale_date).all()[100]
        details = retrieve_comparables_details(
            session, {"object_type": target.object_type}, top_k=200, min_similarity=0.0,
            min_data_quality_score=0.0, as_of=target.sale_date, exclude_ids=[target.id],
            max_sale_age_years=200,
        )
        assert details["candidate_count"] == 100
        ids = {c["id"] for c in details["comparables"]}
        assert target.id not in ids
        dates = {s.id: s.sale_date for s in session.query(HistoricalSale).all()}
        assert all(dates[i] < target.sale_date for i in ids)


def test_backtest_report_and_lift_over_baselines(tmp_path):
    with _session() as session:
        report = run_backtest(
            session,
            PricePredictor(model_dir=str(tmp_path / "no_models")),
            BacktestConfig(min_history=40, max_targets=60, max_sale_age_years=200),
        )
    assert report["targets_scored"] > 30
    assert report["valuation_rate"] > 0.8
    # Object type + maker explain most of the price: retrieval must beat the global median
    assert report["model"]["mdape"] < report["baseline_global_median"]["mdape"]
    # Noise is 25 % lognormal: intervals should cover roughly their nominal share
    assert 0.3 < report["interval_p25_p75"]["coverage"] <= 1.0
    assert report["interval_p10_p90"]["coverage"] >= report["interval_p25_p75"]["coverage"]
    assert "by_object_type" in report and "by_comparable_count" in report


def test_backtest_needs_enough_history():
    with _session(n=10) as session:
        assert "error" in run_backtest(session, PricePredictor(model_dir="/none"))
