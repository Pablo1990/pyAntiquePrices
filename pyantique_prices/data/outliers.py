"""Robust outlier flagging for historical sale prices.

Mis-keyed prices, bundled lots or wrong currencies otherwise dominate
retrieval and training. We flag (never delete) with the modified z-score of
Iglewicz & Hoaglin on log prices, per ``object_type`` when the group is large
enough and over the whole table otherwise.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

MIN_GROUP_SIZE = 8
Z_THRESHOLD = 3.5


def _modified_z(logs: np.ndarray) -> np.ndarray:
    med = np.median(logs)
    mad = np.median(np.abs(logs - med))
    if mad < 1e-9:  # (near-)identical prices: nothing can be called an outlier
        return np.zeros_like(logs)
    return 0.6745 * (logs - med) / mad


def flag_outliers(session, z_threshold: float = Z_THRESHOLD) -> int:
    """Set ``outlier_flag`` on extreme prices; return the number flagged."""
    from .models import HistoricalSale

    rows = (
        session.query(HistoricalSale)
        .filter(HistoricalSale.normalized_price.is_not(None), HistoricalSale.normalized_price > 0)
        .all()
    )
    if not rows:
        return 0

    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        groups[(row.object_type or "").strip().lower()].append(row)

    global_logs = np.log([r.normalized_price for r in rows])
    global_z = dict(zip((id(r) for r in rows), _modified_z(global_logs)))

    flagged = 0
    for members in groups.values():
        if len(members) >= MIN_GROUP_SIZE:
            z = _modified_z(np.log([r.normalized_price for r in members]))
            zmap = dict(zip((id(r) for r in members), z))
            scope = "object type"
        else:
            zmap, scope = global_z, "all sales"
        for row in members:
            zi = float(zmap[id(row)])
            is_out = abs(zi) > z_threshold and math.isfinite(zi)
            if is_out and not row.outlier_flag:
                flagged += 1
            row.outlier_flag = is_out
            row.outlier_reason = (
                f"price modified z-score {zi:+.1f} within {scope}" if is_out else None
            )
    session.commit()
    return flagged
