"""SQLAlchemy ORM models for historical auction sales."""

from __future__ import annotations

import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase


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
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
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
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )
