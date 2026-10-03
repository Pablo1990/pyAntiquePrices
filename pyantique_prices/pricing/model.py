"""Price prediction from comparable sales, optionally anchored by a trained prior."""

from __future__ import annotations

from typing import Optional

from .estimator import estimate_from_comparables
from .training import load_artifacts, predict_interval, prediction_level


class PricePredictor:
    """Similarity-weighted log-space estimate, shrunk towards a model prior.

    The trained bucket model (object type / country medians) is used only as a
    *prior* and only when its bucket is specific (not the global median). The
    actual comparables always drive the estimate, so a close match is never
    replaced by a category average.
    """

    def __init__(
        self,
        min_comparables_for_model: int = 6,
        min_comparables_for_confidence: int = 10,
        model_dir: str = "models",
        prior_strength: float = 4.0,
        similarity_power: float = 2.0,
        recency_half_life_years: float | None = 20.0,
    ) -> None:
        self.min_comparables_for_model = min_comparables_for_model
        self.min_comparables_for_confidence = min_comparables_for_confidence
        self.prior_strength = prior_strength
        self.similarity_power = similarity_power
        self.recency_half_life_years = recency_half_life_years
        self._artifacts = load_artifacts(model_dir)

    def _prior(self, features: dict) -> tuple[Optional[float], Optional[str]]:
        if not self._artifacts:
            return None, None
        model = self._artifacts["model"]
        level = prediction_level(model, features)
        if level == "global":
            return None, None
        interval = predict_interval(model, features, calibrator=self._artifacts["calibrator"])
        return float(interval["p50"]) or None, level

    def predict(self, features: dict, comparables: list[dict]) -> Optional[dict]:
        """Return P10..P90 estimates or None if there are no priced comparables."""
        prior, prior_level = (None, None)
        n_prices = sum(1 for c in comparables if c.get("normalized_price"))
        if n_prices >= self.min_comparables_for_model:
            prior, prior_level = self._prior(features)

        estimate = estimate_from_comparables(
            comparables,
            prior_price=prior,
            prior_strength=self.prior_strength,
            similarity_power=self.similarity_power,
            recency_half_life_years=self.recency_half_life_years,
        )
        if estimate is None:
            return None

        q = estimate["quantiles"]
        n = estimate["n"]
        n_eff = estimate["n_eff"]

        confidence_note = None
        if n < 3:
            confidence_note = "Very low confidence: only 1-2 comparable sales."
        elif n < 6:
            confidence_note = "Low confidence: 3-5 comparable sales."
        elif n < self.min_comparables_for_confidence:
            confidence_note = "Moderate confidence: 6-9 comparable sales."
        elif n_eff < 3:
            confidence_note = (
                "Moderate confidence: a few very close matches dominate the estimate."
            )

        method = "reference_only" if n < 3 else estimate["method"]
        return {
            "p10": round(q["p10"], 2),
            "p25": round(q["p25"], 2),
            "p50": round(q["p50"], 2),
            "p75": round(q["p75"], 2),
            "p90": round(q["p90"], 2),
            "low": round(q["p25"], 2),
            "mid": round(q["p50"], 2),
            "high": round(q["p75"], 2),
            "num_comparables": n,
            "effective_comparables": n_eff,
            "valuation_available": n >= 3,
            "method": method,
            "prior_level": prior_level,
            "confidence_note": confidence_note,
        }
