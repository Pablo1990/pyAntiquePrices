"""API schemas for AntiqueGPT endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AppraiseResponse(BaseModel):
    request_id: str
    identification: dict[str, Any] | None = None
    marks: list[dict[str, Any]] = Field(default_factory=list)
    condition: dict[str, Any] | None = None
    comparables: list[dict[str, Any]] = Field(default_factory=list)
    valuation: dict[str, Any] | None = None
    confidence: dict[str, float] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    model_version: dict[str, str] = Field(default_factory=dict)
    valuation_available: bool = False
    currency: str = "EUR"
    candidate_count: int = 0
    usable_comparable_count: int = 0
    # Live eBay listings fetched for this request only (not stored, not
    # used for the valuation). See pyantique_prices/services/live_market.py.
    live_market_listings: dict[str, Any] | None = None
    # Search links for manual research (nothing fetched or stored) and, when an
    # asking price was supplied, a buy/pass verdict.
    lookup_links: dict[str, Any] | None = None
    deal: dict[str, Any] | None = None
