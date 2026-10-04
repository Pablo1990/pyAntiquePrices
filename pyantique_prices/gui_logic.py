"""GUI-independent helpers: parsing form input, verdict styling, formatting.

Kept free of tkinter so it can be unit-tested anywhere.
"""

from __future__ import annotations

from typing import Any

from .i18n import t

# verdict -> (label, background, foreground)
VERDICT_STYLE: dict[str, tuple[str, str, str]] = {
    "strong_buy": ("STRONG BUY", "#1b7f3b", "#ffffff"),
    "good_buy": ("GOOD BUY", "#5aa02c", "#ffffff"),
    "fair": ("FAIR PRICE", "#d4a017", "#000000"),
    "overpriced": ("OVERPRICED", "#c0392b", "#ffffff"),
    "no_verdict": ("NO VERDICT", "#7f8c8d", "#ffffff"),
}

def verdict_style(verdict: str | None) -> tuple[str, str, str]:
    """``(label, background, foreground)`` with the label in the active language."""
    label, bg, fg = VERDICT_STYLE.get(verdict, VERDICT_STYLE["no_verdict"])
    return t(label), bg, fg


REGION_CHOICES = {
    "Global + Spain + UK": ["global", "es", "uk"],
    "Spain": ["global", "es"],
    "UK": ["global", "uk"],
    "France": ["global", "fr"],
    "Germany": ["global", "de"],
    "USA": ["global", "us"],
    "Everything": ["all"],
}


def parse_number(
    text: str | None,
    label: str,
    *,
    default: float | None = None,
    minimum: float | None = 0.0,
    maximum: float | None = None,
) -> float | None:
    """Parse a user-typed number; accepts ``1.234,50``-style and ``1,5``.

    Empty input returns ``default``. Raises ``ValueError`` with a message fit
    to show in a dialog.
    """
    raw = (text or "").strip().replace("€", "").replace("£", "").replace("$", "").replace("%", "")
    raw = raw.replace(" ", "")
    if not raw:
        return default
    if "," in raw and "." in raw:
        # the last separator is the decimal one
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw:
        raw = raw.replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(t("{label}: '{text}' is not a number.", label=label, text=text)) from None
    if minimum is not None and value < minimum:
        raise ValueError(t("{label} cannot be below {limit}.", label=label, limit=f"{minimum:g}"))
    if maximum is not None and value > maximum:
        raise ValueError(t("{label} cannot be above {limit}.", label=label, limit=f"{maximum:g}"))
    return value


def deal_inputs_from_form(form: dict[str, Any]) -> tuple[float | None, dict]:
    """Turn raw form strings into ``(asking_price, deal_options)``.

    ``asking_price`` is ``None`` when the field is empty (no deal check wanted).
    """
    asking = parse_number(form.get("asking"), t("Asking price"), minimum=0.0)
    if asking is not None and asking <= 0:
        raise ValueError(t("Asking price must be greater than 0."))
    options = {
        "shipping": parse_number(form.get("shipping"), t("Shipping"), default=0.0),
        "buyer_premium_pct": parse_number(form.get("premium"), t("Buyer's premium"), default=0.0, maximum=100.0),
        "vat_pct": parse_number(form.get("vat"), t("VAT"), default=0.0, maximum=100.0),
        "restoration": parse_number(form.get("restoration"), t("Restoration"), default=0.0),
        "for_resale": not bool(form.get("keep")),
        "resale_fee_pct": parse_number(form.get("resale_fee"), t("Selling fee"), default=0.0, maximum=100.0),
    }
    return asking, options


def manual_valuation(form: dict[str, Any]) -> dict:
    """Valuation dict from numbers the user found by hand (sold prices)."""
    p50 = parse_number(form.get("p50"), t("Typical sold price"), minimum=0.0)
    if not p50:
        raise ValueError(t("Enter the typical sold price you found (the middle of the sold results)."))
    p25 = parse_number(form.get("p25"), t("Low sold price"), default=p50 * 0.7)
    p75 = parse_number(form.get("p75"), t("High sold price"), default=p50 * 1.4)
    if not (p25 <= p50 <= p75):
        raise ValueError(t("Low ≤ typical ≤ high is required for the sold prices."))
    n = int(parse_number(form.get("n"), t("Number of sold results"), default=5.0, minimum=1.0))
    return {
        "p25": p25, "p50": p50, "p75": p75,
        "num_comparables": n, "effective_comparables": float(n),
        "valuation_available": True, "method": "manual",
    }


def link_groups(block: dict | None) -> list[tuple[str, list[dict]]]:
    """Research links grouped under human-readable headings, in display order."""
    from pyantique_prices.lookup.links import _KIND_TITLES

    if not block:
        return []
    groups: dict[str, list[dict]] = {}
    for link in block.get("links", []):
        groups.setdefault(link.get("kind", "other"), []).append(link)
    return [(t(_KIND_TITLES[kind]) if kind in _KIND_TITLES else kind.title(), links) for kind, links in groups.items()]


def key_source_urls(block: dict | None, limit: int = 4) -> list[str]:
    """The few links worth opening first: realised-price sources, then auctions."""
    links = (block or {}).get("links", [])
    ordered = [l for l in links if l.get("kind") == "sold"] + [l for l in links if l.get("kind") == "auction"]
    return [l["url"] for l in ordered[:limit]]


def deal_summary_lines(deal: dict | None, currency: str = "EUR") -> list[str]:
    """Detail lines shown under the verdict banner."""
    if not deal:
        return []
    lines = []
    if deal.get("all_in_cost") is not None:
        lines.append(t("All-in cost: {amount} {currency}", amount=f"{deal['all_in_cost']:.2f}", currency=currency))
        lines.append(
            t(
                "Worth (low / typical / high): {low} / {mid} / {high} {currency}",
                low=f"{deal['value_low']:.2f}",
                mid=f"{deal['value_mid']:.2f}",
                high=f"{deal['value_high']:.2f}",
                currency=currency,
            )
        )
        if deal.get("for_resale"):
            lines.append(
                t(
                    "After selling costs: {net_low} / {net_mid}  →  profit {profit_low} / {profit_mid} {currency}",
                    net_low=f"{deal['net_low']:.2f}",
                    net_mid=f"{deal['net_mid']:.2f}",
                    profit_low=f"{deal['profit_low']:+.2f}",
                    profit_mid=f"{deal['profit_mid']:+.2f}",
                    currency=currency,
                )
            )
        lines.append(
            t(
                "Highest asking price for a strong buy: {strong} · for a good buy: {good} {currency}",
                strong=f"{deal['max_asking_strong_buy']:.2f}",
                good=f"{deal['max_asking_good_buy']:.2f}",
                currency=currency,
            )
        )
    lines.extend(f"⚠ {w}" for w in deal.get("warnings", []))
    return lines


def ledger_row(item: Any) -> tuple[str, ...]:
    """Display values for one ledger row."""
    from pyantique_prices.ledger.service import cost_basis, profit

    def money(value):
        return f"{value:,.2f}" if value is not None else ""

    sold = item.sold_price if item.status == "sold" else None
    gain = profit(item)
    return (
        str(item.id),
        item.title or "",
        t(item.status),
        item.acquired_date.date().isoformat() if item.acquired_date else "",
        money(cost_basis(item)),
        money(item.estimate_mid),
        money(sold),
        f"{gain:+,.2f}" if gain is not None else "",
    )


def summary_lines(summary: dict, currency: str = "EUR", factor: float = 1.0) -> list[str]:
    """Plain-language ledger summary."""
    if not summary or not summary.get("items"):
        return [t("No items yet. Add what you buy and mark it sold to learn how well the estimates work.")]
    pct = lambda v: t("n/a") if v is None else f"{v * 100:.0f}%"  # noqa: E731
    lines = [
        t(
            "{items} items: {held} held, {sold} sold, {kept} kept",
            items=summary["items"], held=summary["held"], sold=summary["sold"], kept=summary["kept"],
        ),
        t(
            "Money tied up in held items: {amount} {currency}",
            amount=f"{summary['capital_in_held']:,.2f}", currency=currency,
        ),
        t(
            "Realised profit: {profit} {currency}   ROI: {roi}   Win rate: {win}",
            profit=f"{summary['realised_profit']:+,.2f}", currency=currency,
            roi=pct(summary["roi"]), win=pct(summary["win_rate"]),
        ),
    ]
    if summary.get("median_days_held") is not None:
        lines.append(t("Median days to sell: {days}", days=f"{summary['median_days_held']:g}"))
    est = summary.get("estimates") or {}
    if est.get("rated_items"):
        lines.append(
            t(
                "Estimate accuracy ({n} sold items): typically sold for {ratio}× the estimate; "
                "{share} landed inside the estimated range.",
                n=est["rated_items"],
                ratio=f"{est['median_sold_to_estimate']:.2f}",
                share=pct(est["share_within_estimated_range"]),
            )
        )
        lines.append(
            t("Correction applied to new deal checks: ×{factor}", factor=f"{factor:.2f}")
            + ("" if factor != 1.0 else t(" (needs at least 5 sold items with an estimate)"))
        )
    return lines


def backtest_lines(report: dict) -> list[str]:
    """Readable backtest result."""
    if "error" in report:
        return [report["error"]]
    pct = lambda v: t("n/a") if v is None else f"{v * 100:.0f}%"  # noqa: E731
    m, base, glob = report["model"], report["baseline_unweighted_median"], report["baseline_global_median"]
    lines = [
        t(
            "Items re-appraised: {n} (a price estimate was possible for {rate})",
            n=report["targets_scored"], rate=pct(report["valuation_rate"]),
        ),
        t(
            "Typical error (median % off): {err}   [plain median of comparables: {base}; "
            "one price for everything: {glob}]",
            err=pct(m.get("mdape")), base=pct(base.get("mdape")), glob=pct(glob.get("mdape")),
        ),
    ]
    if m.get("bias_median_log_ratio") is not None:
        import math

        lines.append(t("Bias: estimates are typically {ratio}× the real price", ratio=f"{math.exp(m['bias_median_log_ratio']):.2f}"))
    i50, i80 = report["interval_p25_p75"], report["interval_p10_p90"]
    lines.append(
        t(
            "Range reliability: the 'typical range' held the real price {c50} of the time "
            "(ideal 50%); the wide range {c80} (ideal 80%).",
            c50=pct(i50.get("coverage")), c80=pct(i80.get("coverage")),
        )
    )
    return lines


def import_lines(result: Any) -> list[str]:
    """Readable summary of a CSV import."""
    lines = [
        t(
            "Rows read: {read} · added: {added} · skipped: {skipped}",
            read=result.rows_processed, added=result.rows_inserted, skipped=result.rows_skipped,
        ),
    ]
    if result.duplicates:
        lines.append(t("Duplicates ignored: {n}", n=result.duplicates))
    if result.invalid_prices:
        lines.append(t("Rows with an invalid price: {n}", n=result.invalid_prices))
    if result.unsupported_currencies:
        lines.append(t("Rows with an unsupported currency: {n}", n=result.unsupported_currencies))
    if result.asking_excluded:
        lines.append(t("Asking prices stored but not used as comparables: {n}", n=result.asking_excluded))
    if result.outliers_flagged:
        lines.append(t("Extreme prices flagged and excluded: {n}", n=result.outliers_flagged))
    if result.mixed_price_basis:
        lines.append(
            t(
                "⚠ {hammer} hammer-only and {final} premium-inclusive prices are mixed. "
                "Set HAMMER_PREMIUM_RATE in .env so they are comparable.",
                hammer=result.hammer_only, final=result.final_with_premium,
            )
        )
    return lines


def identification_from_fields(fields: dict[str, str]) -> dict:
    """Minimal identification from what the user typed (for research links)."""

    def clean(key: str) -> str:
        return (fields.get(key) or "").strip()

    def cand(name: str) -> list[dict]:
        return [{"name": name, "confidence": 0.9}] if name else []

    return {
        "object_type": clean("object") or None,
        "subtype": clean("subtype") or None,
        "period": clean("period") or None,
        "likely_period": clean("period") or None,
        "materials": [clean("material")] if clean("material") else [],
        "manufacturer_candidates": cand(clean("maker")),
        "artist_candidates": cand(clean("artist")),
    }


def estimate_line(result: dict | None) -> str:
    """One-line summary of the price estimate for the results header."""
    result = result or {}
    valuation = result.get("valuation")
    currency = result.get("currency", "EUR")
    if not valuation:
        return t(
            "No price estimate yet: there are no comparable sales in your database. Use the research "
            "links below, then add what you learn on the Data tab."
        )
    low, mid, high = valuation.get("low"), valuation.get("mid"), valuation.get("high")
    n = valuation.get("num_comparables")
    prefix = t("Estimated value") if result.get("valuation_available") else t("Rough reference only")
    parts = [t("{prefix}: {low} – {high} {currency} (typical {mid})", prefix=prefix, low=f"{low:,.0f}", high=f"{high:,.0f}", currency=currency, mid=f"{mid:,.0f}")]
    if n is not None:
        parts.append(t("{n} comparable sales", n=n) if n != 1 else t("1 comparable sale"))
    note = valuation.get("confidence_note")
    if note:
        parts.append(note)
    return " · ".join(parts)
