"""Quantile regression utilities."""

from __future__ import annotations

import numpy as np


def compute_quantiles(
    prices: list[float],
    quantiles: list[float] | None = None,
) -> dict[str, float]:
    """Compute quantile estimates from a list of prices."""
    if quantiles is None:
        quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    arr = np.array(prices)
    return {
        f"p{int(quantile * 100)}": float(np.percentile(arr, quantile * 100))
        for quantile in quantiles
    }
