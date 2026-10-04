"""Similarity-weighted, log-space price estimation from comparable sales.

Why this exists
---------------
Auction prices are heavy-tailed and roughly log-normal, and comparables are
not equally informative. The previous estimator took plain percentiles of the
top-K comparables (so a weak match counted as much as a near-identical one) and
-- whenever a trained model existed -- discarded the comparables entirely in
favour of a coarse ``object_type|country`` median.

This module:

* weights each comparable by ``similarity ** similarity_power`` and by an
  exponential recency decay;
* computes weighted quantiles on ``log(price)``;
* reports an *effective* sample size (Kish) so a few dominant comparables do not
  masquerade as a large sample;
* shrinks towards an optional model prior with weight ``n_eff / (n_eff + k)``;
* widens the interval for small ``n_eff`` and never lets it collapse below a
  noise floor (auction results are noisy even for identical objects).
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Iterable

import numpy as np

# Log-space half-width floor for the inter-quartile range (~ +/-22 % around the
# median). Auction hammer prices of near-identical lots routinely differ by more.
DEFAULT_SIGMA_FLOOR = 0.20
# Ratio of the 10-90 % half-width to the 25-75 % half-width for a normal law.
_TAIL_RATIO = 1.2816 / 0.6745


def weighted_quantile(values: Iterable[float], weights: Iterable[float], q: float) -> float:
    """Weighted quantile with linear interpolation (``q`` in [0, 1])."""
    v = np.asarray(list(values), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    if v.size == 0:
        raise ValueError("weighted_quantile needs at least one value")
    order = np.argsort(v)
    v, w = v[order], w[order]
    total = w.sum()
    if total <= 0:
        w = np.ones_like(w)
        total = w.sum()
    positions = (np.cumsum(w) - 0.5 * w) / total
    return float(np.interp(q, positions, v))


def effective_sample_size(weights: Iterable[float]) -> float:
    """Kish effective sample size: (sum w)^2 / sum w^2."""
    w = np.asarray(list(weights), dtype=float)
    denom = float((w**2).sum())
    return float(w.sum() ** 2 / denom) if denom > 0 else 0.0


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def comparable_weight(
    comparable: dict,
    *,
    similarity_power: float = 2.0,
    recency_half_life_years: float | None = 20.0,
    today: date | None = None,
) -> float:
    """Weight of one comparable; 1.0 when it carries no similarity signal."""
    sim = comparable.get("overall_similarity", comparable.get("retrieval_score"))
    weight = max(float(sim), 1e-6) ** similarity_power if isinstance(sim, (int, float)) else 1.0
    if recency_half_life_years:
        sale_date = _as_date(comparable.get("sale_date"))
        if sale_date is not None:
            age_years = max(0.0, ((today or date.today()) - sale_date).days / 365.25)
            weight *= 0.5 ** (age_years / recency_half_life_years)
    return weight


def estimate_from_comparables(
    comparables: list[dict],
    *,
    prior_price: float | None = None,
    prior_strength: float = 4.0,
    similarity_power: float = 2.0,
    recency_half_life_years: float | None = 20.0,
    sigma_floor: float = DEFAULT_SIGMA_FLOOR,
    today: date | None = None,
) -> dict | None:
    """Return log-space weighted P10..P90 or ``None`` without usable prices."""
    prices, weights = [], []
    for comp in comparables:
        price = comp.get("normalized_price")
        if isinstance(price, (int, float)) and price > 0:
            prices.append(float(price))
            weights.append(
                comparable_weight(
                    comp,
                    similarity_power=similarity_power,
                    recency_half_life_years=recency_half_life_years,
                    today=today,
                )
            )
    if not prices:
        return None

    logs = np.log(prices)
    n_eff = effective_sample_size(weights)
    log_p25 = weighted_quantile(logs, weights, 0.25)
    log_p50 = weighted_quantile(logs, weights, 0.50)
    log_p75 = weighted_quantile(logs, weights, 0.75)

    method = "similarity_weighted_estimate"
    if prior_price and prior_price > 0 and prior_strength >= 0:
        w_comp = n_eff / (n_eff + prior_strength) if (n_eff + prior_strength) > 0 else 1.0
        log_p50 = w_comp * log_p50 + (1.0 - w_comp) * math.log(prior_price)
        method = "model_blended_estimate"

    # Spread: observed dispersion, inflated for small n_eff, floored for noise.
    inflate = 1.0 + 1.0 / math.sqrt(max(n_eff, 1.0))
    half_lo = max((log_p50 - log_p25) if log_p50 > log_p25 else 0.0, sigma_floor) * inflate
    half_hi = max((log_p75 - log_p50) if log_p75 > log_p50 else 0.0, sigma_floor) * inflate
    quantiles = {
        "p10": log_p50 - half_lo * _TAIL_RATIO,
        "p25": log_p50 - half_lo,
        "p50": log_p50,
        "p75": log_p50 + half_hi,
        "p90": log_p50 + half_hi * _TAIL_RATIO,
    }
    return {
        "quantiles": {k: float(math.exp(v)) for k, v in quantiles.items()},
        "n": len(prices),
        "n_eff": round(n_eff, 2),
        "method": method,
    }
