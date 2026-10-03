"""SQLAlchemy ORM models for historical auction sales."""

from __future__ import annotations

import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase


def _utcnow() -> datetime.datetime:
    """Naive UTC timestamp (what SQLite stores); replaces deprecated utcnow()."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    """Base class for ORM models."""


class HistoricalSale(Base):
    __tablename__ = "historical_sales"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500))
    description = Column(Text)
    category = Column(String(200))
    subcategory = Column(String(200))
    object_type = Column(String(200))
    period = Column(String(200))
    manufacturer = Column(String(200))
    artist = Column(String(200))
    workshop = Column(String(200))
    material = Column(String(500))
    technique = Column(String(500))
    condition = Column(String(100))
    country = Column(String(100))
    region = Column(String(200))
    dimensions = Column(JSON)
    height = Column(Float)
    width = Column(Float)
    depth = Column(Float)
    diameter = Column(Float)
    weight = Column(Float)
    marks = Column(Text)
    provenance = Column(Text)
    auction_house = Column(String(200))
    auction_location = Column(String(200))
    sale_date = Column(DateTime)
    lot_number = Column(String(50))
    currency = Column(String(10))
    hammer_price = Column(Float)
    buyer_premium = Column(Float)
    final_price = Column(Float)
    estimate_low = Column(Float)
    estimate_high = Column(Float)
    image_urls = Column(JSON)
    source_url = Column(String(1000))
    original_currency = Column(String(10))
    original_price = Column(Float)
    normalized_currency = Column(String(10))
    normalized_price = Column(Float)
    price_basis = Column(String(50))
    outlier_flag = Column(Boolean, default=False)
    outlier_reason = Column(String(500))
    usable_for_training = Column(Boolean, default=True)
    text_embedding = Column(JSON)
    image_embedding = Column(JSON)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(
        DateTime,
        default=_utcnow,
        onupdate=_utcnow,
    )


class AppraisalRecord(Base):
    __tablename__ = "appraisals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), unique=True, nullable=False, index=True)
    model_versions = Column(JSON)
    input_metadata = Column(JSON)
    identification = Column(JSON)
    comparable_ids = Column(JSON)
    valuation = Column(JSON)
    calibration = Column(JSON)
    confidence = Column(JSON)
    warnings = Column(JSON)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(
        DateTime,
        default=_utcnow,
        onupdate=_utcnow,
    )


class LedgerItem(Base):
    """An item you bought (and maybe sold): your own, private transaction record."""

    __tablename__ = "ledger_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    object_type = Column(String(200))
    manufacturer = Column(String(200))
    artist = Column(String(200))
    period = Column(String(200))
    material = Column(String(500))
    condition = Column(String(100))
    country = Column(String(100))
    notes = Column(Text)
    photos = Column(JSON)  # list of local file paths

    # Purchase
    status = Column(String(20), default="held", nullable=False)  # held | sold | kept
    acquired_date = Column(DateTime)
    acquired_price = Column(Float, nullable=False)
    acquired_costs = Column(Float, default=0.0)  # shipping, premium, VAT, restoration
    acquired_where = Column(String(200))
    source_url = Column(String(1000))
    currency = Column(String(10), default="EUR")

    # What the tool thought when you bought it (for calibration)
    estimate_low = Column(Float)
    estimate_mid = Column(Float)
    estimate_high = Column(Float)
    estimate_method = Column(String(100))
    deal_verdict = Column(String(30))

    # Sale
    sold_date = Column(DateTime)
    sold_price = Column(Float)
    sold_fees = Column(Float, default=0.0)  # platform fees + postage you paid
    sold_where = Column(String(200))

    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
