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
