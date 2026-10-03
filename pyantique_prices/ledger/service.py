"""Ledger operations. All data stays in your local database.

Why keep a ledger
-----------------
* It is the only price data that is *yours* to use however you like: your own
  purchases and sales are facts about your own transactions.
* It tells you how good the estimates really are (:func:`summarize`) and lets
  the deal score correct a consistent bias (:func:`calibration_factor`).
* :func:`sync_to_sales` can feed your realised sales back in as comparables,
  and :func:`export_sales_csv` shares them in the standard import format.
"""

from __future__ import annotations

import csv
import datetime
import math
import statistics
from pathlib import Path
from typing import Any

from pyantique_prices.data.models import HistoricalSale, LedgerItem

MIN_ITEMS_FOR_CALIBRATION = 5
STATUSES = {"held", "sold", "kept"}


def _as_datetime(value) -> datetime.datetime | None:
    if value is None or isinstance(value, datetime.datetime):
        return value
    if isinstance(value, datetime.date):
        return datetime.datetime.combine(value, datetime.time())
    return datetime.datetime.fromisoformat(str(value)[:10])


def add_purchase(
    session,
    *,
    title: str,
    price: float,
    costs: float = 0.0,
    date=None,
    where: str | None = None,
    currency: str = "EUR",
    identification: dict | None = None,
    valuation: dict | None = None,
    deal: dict | None = None,
    **fields: Any,
) -> LedgerItem:
    """Record a purchase. ``identification``/``valuation``/``deal`` are optional
    snapshots from an appraisal; explicit ``fields`` (object_type, notes, ...)
    take precedence."""
    if price is None or price < 0:
        raise ValueError("price must be >= 0")
    ident = identification or {}

    def first_name(key):
        for cand in ident.get(key) or []:
            if isinstance(cand, dict) and cand.get("name"):
                return cand["name"]
        return None

    def text(key):
        value = ident.get(key)
        return value.get("value") if isinstance(value, dict) else value

    val = valuation or {}
    item = LedgerItem(
        title=title,
        object_type=text("object_type"),
        manufacturer=first_name("manufacturer_candidates"),
        artist=first_name("artist_candidates"),
        period=text("period") or text("likely_period"),
        material=", ".join(ident.get("materials") or []) or None,
        condition=text("condition"),
        country=text("country"),
        status="held",
        acquired_date=_as_datetime(date) or datetime.datetime.now(),
        acquired_price=float(price),
        acquired_costs=float(costs or 0.0),
        acquired_where=where,
        currency=currency,
        estimate_low=val.get("p25") or val.get("low"),
        estimate_mid=val.get("p50") or val.get("mid"),
        estimate_high=val.get("p75") or val.get("high"),
        estimate_method=val.get("method"),
        deal_verdict=(deal or {}).get("verdict"),
    )
    for key, value in fields.items():
        if value is not None and hasattr(item, key):
            setattr(item, key, value)
    session.add(item)
    session.commit()
    return item


def _get(session, item_id: int) -> LedgerItem:
    item = session.get(LedgerItem, item_id)
    if item is None:
        raise KeyError(f"No ledger item with id {item_id}")
    return item


def record_sale(session, item_id: int, *, price: float, fees: float = 0.0, date=None,
                where: str | None = None) -> LedgerItem:
    """Mark an item as sold."""
    if price is None or price < 0:
        raise ValueError("price must be >= 0")
    item = _get(session, item_id)
    if item.status == "sold":
        raise ValueError(f"Item {item_id} is already recorded as sold")
    item.status = "sold"
    item.sold_price = float(price)
    item.sold_fees = float(fees or 0.0)
    item.sold_date = _as_datetime(date) or datetime.datetime.now()
    item.sold_where = where
    session.commit()
    return item


def list_items(session, status: str | None = None) -> list[LedgerItem]:
    if status and status not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}")
    query = session.query(LedgerItem)
    if status:
        query = query.filter(LedgerItem.status == status)
    return query.order_by(LedgerItem.acquired_date, LedgerItem.id).all()


def cost_basis(item: LedgerItem) -> float:
    return (item.acquired_price or 0.0) + (item.acquired_costs or 0.0)


def profit(item: LedgerItem) -> float | None:
    if item.status != "sold" or item.sold_price is None:
        return None
    return item.sold_price - (item.sold_fees or 0.0) - cost_basis(item)


def summarize(session) -> dict:
    """Profit/loss, hit rates and how accurate the tool's estimates were."""
    items = list_items(session)
    sold = [i for i in items if i.status == "sold" and i.sold_price is not None]
    held = [i for i in items if i.status == "held"]
    profits = [profit(i) for i in sold]
    invested_sold = sum(cost_basis(i) for i in sold)
    days = [
        (i.sold_date - i.acquired_date).days
        for i in sold if i.sold_date and i.acquired_date
    ]

    rated = [i for i in sold if i.estimate_mid and i.estimate_mid > 0 and i.sold_price > 0]
    log_ratios = [math.log(i.sold_price / i.estimate_mid) for i in rated]
    in_range = [
        i for i in rated
        if i.estimate_low and i.estimate_high and i.estimate_low <= i.sold_price <= i.estimate_high
    ]
    by_verdict: dict[str, dict] = {}
    for i in sold:
        if i.deal_verdict:
            bucket = by_verdict.setdefault(i.deal_verdict, {"n": 0, "profit": 0.0, "wins": 0})
            bucket["n"] += 1
            bucket["profit"] += profit(i)
            bucket["wins"] += 1 if profit(i) > 0 else 0

    return {
        "items": len(items),
        "held": len(held),
        "sold": len(sold),
        "kept": sum(1 for i in items if i.status == "kept"),
        "capital_in_held": round(sum(cost_basis(i) for i in held), 2),
        "realised_revenue": round(sum(i.sold_price for i in sold), 2),
        "realised_profit": round(sum(profits), 2),
        "roi": round(sum(profits) / invested_sold, 4) if invested_sold else None,
        "win_rate": round(sum(1 for p in profits if p > 0) / len(profits), 4) if profits else None,
        "median_days_held": statistics.median(days) if days else None,
        "estimates": {
            "rated_items": len(rated),
            # >1 means items sold for more than estimated (tool is too pessimistic)
            "median_sold_to_estimate": round(math.exp(statistics.median(log_ratios)), 3) if log_ratios else None,
            "share_within_estimated_range": round(len(in_range) / len(rated), 4) if rated else None,
        },
        "by_verdict": {k: {**v, "profit": round(v["profit"], 2)} for k, v in by_verdict.items()},
    }


def calibration_factor(session, min_items: int = MIN_ITEMS_FOR_CALIBRATION) -> float:
    """Median (sold price / estimated midpoint) over your sales; 1.0 if too few.

    Feed it to ``assess_deal(calibration_factor=...)`` so the estimates adapt to
    your market and your tool's habitual bias.
    """
    ratios = [
        i.sold_price / i.estimate_mid
        for i in list_items(session, "sold")
        if i.estimate_mid and i.estimate_mid > 0 and i.sold_price and i.sold_price > 0
    ]
    if len(ratios) < min_items:
        return 1.0
    return float(statistics.median(ratios))


SALES_CSV_FIELDS = [
    "title", "description", "object_type", "manufacturer", "artist", "period", "material",
    "condition", "country", "auction_house", "sale_date", "currency", "final_price",
    "price_basis", "source_url",
]


def _sold_rows(session) -> list[dict]:
    rows = []
    for i in list_items(session, "sold"):
        if not i.sold_price:
            continue
        rows.append({
            "title": i.title, "description": i.description, "object_type": i.object_type,
            "manufacturer": i.manufacturer, "artist": i.artist, "period": i.period,
            "material": i.material, "condition": i.condition, "country": i.country,
            "auction_house": i.sold_where or "private sale",
            "sale_date": i.sold_date.date().isoformat() if i.sold_date else "",
            "currency": i.currency or "EUR", "final_price": i.sold_price,
            "price_basis": "realized", "source_url": f"ledger://{i.id}",
        })
    return rows


def export_sales_csv(session, path: str | Path) -> int:
    """Write your sold items in the ``import_sales.py`` CSV format; returns rows."""
    rows = _sold_rows(session)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SALES_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def sync_to_sales(session, base_currency: str = "EUR") -> int:
    """Add your realised sales to the comparables database (idempotent)."""
    from pyantique_prices.data.normalizer import normalize_price

    added = 0
    for row in _sold_rows(session):
        if session.query(HistoricalSale).filter_by(source_url=row["source_url"]).first():
            continue
        normalized = normalize_price(row["final_price"], row["currency"], base_currency)
        if normalized is None:
            continue
        session.add(HistoricalSale(
            title=row["title"], description=row["description"], object_type=row["object_type"],
            manufacturer=row["manufacturer"], artist=row["artist"], period=row["period"],
            material=row["material"], condition=row["condition"], country=row["country"],
            auction_house=row["auction_house"],
            sale_date=_as_datetime(row["sale_date"]) if row["sale_date"] else None,
            currency=row["currency"], final_price=row["final_price"],
            original_currency=row["currency"], original_price=row["final_price"],
            normalized_currency=base_currency, normalized_price=normalized,
            price_basis="realized", source_url=row["source_url"], usable_for_training=True,
        ))
        added += 1
    session.commit()
    return added
