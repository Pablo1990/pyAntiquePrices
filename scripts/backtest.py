#!/usr/bin/env python3
"""Measure real pricing accuracy with a point-in-time backtest.

Usage:  python scripts/backtest.py [--max-targets 300] [--min-history 30] [--out report.json]

Run it before and after any change to retrieval or pricing: the report shows
median absolute % error, bias, interval coverage versus nominal, and lift over
two baselines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyantique_prices.config import settings
from pyantique_prices.data.database import get_engine, get_session_factory
from pyantique_prices.evaluation import BacktestConfig, run_backtest
from pyantique_prices.pricing.model import PricePredictor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-targets", type=int, default=300)
    parser.add_argument("--min-history", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=settings.top_k_comparables)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, help="Write the JSON report here too")
    args = parser.parse_args()

    engine = get_engine(settings.database_url)
    session_factory = get_session_factory(engine)
    predictor = PricePredictor(
        min_comparables_for_model=settings.min_comparables_for_model,
        min_comparables_for_confidence=settings.min_comparables_for_confidence,
    )
    with session_factory() as session:
        report = run_backtest(
            session,
            predictor,
            BacktestConfig(
                min_history=args.min_history,
                max_targets=args.max_targets,
                top_k=args.top_k,
                seed=args.seed,
            ),
        )
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 1 if "error" in report else 0


if __name__ == "__main__":
    raise SystemExit(main())
