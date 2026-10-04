"""Turn a valuation range plus your real costs into a buy / pass verdict.

The valuation (P25 / P50 / P75) is *uncertain*: valuing antiques from photos is
rarely better than +/-40 %. So the score never trusts the midpoint alone. It
asks, in order:

* **Would I still profit if the item sells at the low end (P25)?** -> strong buy
* **Would I profit at the typical value (P50)?**                   -> good buy
* **Would I at least break even at the typical value?**            -> fair
* otherwise                                                        -> overpriced

"Profit" means a minimum *margin* over the all-in cost (default 30 %), doubled
when the evidence is weak (few or dominated comparables, or an uncertain
identification). If you are buying to keep, set ``for_resale=False`` and
selling fees are ignored.

This is decision support, not advice: it cannot see condition problems,
repairs, fakes or provenance, and the verdict is only as good as the valuation.
"""

from __future__ import annotations

from typing import Any

from ..i18n import t

CALIBRATION_BOUNDS = (0.5, 1.5)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _invert_cost(max_cost: float, *, premium_pct: float, vat_pct: float, fixed: float) -> float:
    """Highest asking price whose all-in cost equals ``max_cost``."""
    base = (max_cost - fixed) / (1.0 + vat_pct / 100.0)
    return max(0.0, base / (1.0 + premium_pct / 100.0))


def assess_deal(
    valuation: dict | None,
    *,
    asking_price: float,
    shipping: float = 0.0,
    buyer_premium_pct: float = 0.0,
    vat_pct: float = 0.0,
    restoration: float = 0.0,
    other_costs: float = 0.0,
    for_resale: bool = True,
    resale_fee_pct: float = 0.0,
    resale_shipping: float = 0.0,
    min_margin: float = 0.30,
    calibration_factor: float = 1.0,
    identification_confidence: float | None = None,
) -> dict:
    """Return a verdict dict (see keys below); never raises on missing data."""
    warnings: list[str] = []
    result: dict[str, Any] = {
        "verdict": "no_verdict",
        "headline": t("No verdict: there is no usable valuation for this item."),
        "asking_price": asking_price,
        "warnings": warnings,
    }
    if asking_price is None or asking_price <= 0:
        result["headline"] = t("No verdict: enter the asking price (> 0).")
        return result
    if not valuation or not valuation.get("valuation_available", True):
        warnings.append(t("Add comparable sales (see the research links) before relying on a price."))
        return result
    p25, p50, p75 = (_num(valuation.get(k)) for k in ("p25", "p50", "p75"))
    if p50 <= 0:
        return result

    lo_b, hi_b = CALIBRATION_BOUNDS
    factor = min(hi_b, max(lo_b, _num(calibration_factor, 1.0) or 1.0))
    p25, p50, p75 = p25 * factor, p50 * factor, p75 * factor

    fixed = shipping + restoration + other_costs
    cost = asking_price * (1 + buyer_premium_pct / 100.0) * (1 + vat_pct / 100.0) + fixed

    def net(value: float) -> float:
        if not for_resale:
            return value
        return value * (1 - resale_fee_pct / 100.0) - resale_shipping

    net_low, net_mid, net_high = net(p25), net(p50), net(p75)
    margin_low = (net_low - cost) / cost
    margin_mid = (net_mid - cost) / cost

    n = int(_num(valuation.get("num_comparables")))
    n_eff = _num(valuation.get("effective_comparables"), float(n))
    weak = n < 6 or n_eff < 3 or (
        identification_confidence is not None and identification_confidence < 0.5
    )
    required = min_margin * (2.0 if weak else 1.0)
    if weak:
        warnings.append(
            t(
                "Weak evidence (few or dominated comparables, or an uncertain identification): "
                "the margin required for a good verdict is doubled to {required}.",
                required=f"{required:.0%}",
            )
        )
    if valuation.get("method") == "reference_only" or n < 3:
        warnings.append(t("Fewer than 3 comparables: treat this as a rough pointer only."))

    if margin_low >= required:
        verdict, headline = "strong_buy", t(
            "Strong buy: still {margin} over your all-in cost even at the low estimate.",
            margin=f"{margin_low:.0%}",
        )
    elif margin_mid >= required:
        verdict, headline = "good_buy", t(
            "Good buy at the typical value ({margin} margin), but {risk} if it sells at the low end.",
            margin=f"{margin_mid:.0%}",
            risk=t("a loss") if margin_low < 0 else t("a thin margin"),
        )
    elif margin_mid >= 0:
        verdict, headline = "fair", t("Fair price: about break-even at the typical value.")
    else:
        verdict, headline = "overpriced", t(
            "Overpriced: you would lose about {loss} at the typical value.", loss=f"{abs(margin_mid):.0%}"
        )

    if cost < 0.25 * net_low:
        warnings.append(
            t(
                "Suspiciously cheap (under a quarter of the low estimate). Check that the item is "
                "correctly identified, genuine and in the condition shown before getting excited."
            )
        )
    warnings.append(t("Condition, repairs, authenticity and provenance are not assessed."))
    if factor != 1.0:
        warnings.append(t("Estimates adjusted x{factor} using your own past sales.", factor=f"{factor:.2f}"))

    result.update(
        verdict=verdict,
        headline=headline,
        all_in_cost=round(cost, 2),
        value_low=round(p25, 2),
        value_mid=round(p50, 2),
        value_high=round(p75, 2),
        net_low=round(net_low, 2),
        net_mid=round(net_mid, 2),
        net_high=round(net_high, 2),
        profit_low=round(net_low - cost, 2),
        profit_mid=round(net_mid - cost, 2),
        margin_low=round(margin_low, 4),
        margin_mid=round(margin_mid, 4),
        required_margin=round(required, 4),
        evidence="weak" if weak else "ok",
        for_resale=for_resale,
        calibration_factor=round(factor, 3),
        max_asking_strong_buy=round(
            _invert_cost(net_low / (1 + required), premium_pct=buyer_premium_pct,
                         vat_pct=vat_pct, fixed=fixed), 2),
        max_asking_good_buy=round(
            _invert_cost(net_mid / (1 + required), premium_pct=buyer_premium_pct,
                         vat_pct=vat_pct, fixed=fixed), 2),
    )
    return result


_LABELS = {
    "strong_buy": "STRONG BUY",
    "good_buy": "GOOD BUY",
    "fair": "FAIR PRICE",
    "overpriced": "OVERPRICED",
    "no_verdict": "NO VERDICT",
}


def format_deal(deal: dict | None, currency: str = "EUR") -> list[str]:
    """Plain-text lines for CLI / GUI output."""
    if not deal:
        return []
    lines = [f"DEAL CHECK: {_LABELS.get(deal.get('verdict'), '?')}", deal.get("headline", "")]
    if deal.get("all_in_cost") is not None:
        lines.append(
            f"All-in cost {deal['all_in_cost']:.2f} {currency} | worth (low / typical / high): "
            f"{deal['value_low']:.2f} / {deal['value_mid']:.2f} / {deal['value_high']:.2f}"
        )
        if deal.get("for_resale"):
            lines.append(
                f"After selling costs (low / typical): {deal['net_low']:.2f} / {deal['net_mid']:.2f} "
                f"-> profit {deal['profit_low']:+.2f} / {deal['profit_mid']:+.2f}"
            )
        lines.append(
            f"Highest asking price for a strong buy: {deal['max_asking_strong_buy']:.2f} | "
            f"for a good buy: {deal['max_asking_good_buy']:.2f} {currency}"
        )
    lines.extend(f"! {w}" for w in deal.get("warnings", []))
    return lines
