"""Price prediction model."""

from __future__ import annotations

from typing import Optional

import numpy as np


class PricePredictor:
    """Simple price predictor using comparable median as baseline."""

    def __init__(
        self,
        min_comparables_for_model: int = 6,
        min_comparables_for_confidence: int = 10,
    ) -> None:
        self.min_comparables_for_model = min_comparables_for_model
        self.min_comparables_for_confidence = min_comparables_for_confidence

    def predict(self, features: dict, comparables: list[dict]) -> Optional[dict]:
        """Return P25/P50/P75 estimates or None if insufficient data."""
        del features
        prices = [
            comparable.get("normalized_price")
            for comparable in comparables
            if comparable.get("normalized_price")
        ]
        n_prices = len(prices)
        if n_prices == 0:
            return None

        confidence_note = None
        if n_prices < 3:
            confidence_note = "Very low confidence: only 1-2 comparable sales."
        elif n_prices < 6:
            confidence_note = "Low confidence: 3-5 comparable sales."
        elif n_prices < self.min_comparables_for_confidence:
            confidence_note = "Moderate confidence: 6-9 comparable sales."

        p25 = float(np.percentile(prices, 25))
        p50 = float(np.percentile(prices, 50))
        p75 = float(np.percentile(prices, 75))
        valuation_available = n_prices >= 3

        return {
            "p25": round(p25, 2),
            "p50": round(p50, 2),
            "p75": round(p75, 2),
            "low": round(p25, 2),
            "mid": round(p50, 2),
            "high": round(p75, 2),
            "num_comparables": n_prices,
            "valuation_available": valuation_available,
            "method": "reference_only" if not valuation_available else "quantile_estimate",
            "confidence_note": confidence_note,
        }
