"""Calibration utilities for the pricing model."""

from __future__ import annotations

import numpy as np


def compute_metrics(y_true: list[float], y_pred: list[float]) -> dict:
    """Compute MAE, RMSE, median absolute percentage error, log error."""
    yt = np.array(y_true)
    yp = np.array(y_pred)
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mape = float(np.median(np.abs((yt - yp) / (yt + 1e-9))))
    log_err = float(np.mean(np.abs(np.log1p(yt) - np.log1p(yp))))
    return {"mae": mae, "rmse": rmse, "mape": mape, "log_error": log_err}
