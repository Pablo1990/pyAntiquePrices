"""Pydantic schemas for structured antique identification."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MarkEvidence(BaseModel):
    text: Optional[str] = None
    normalized_text: Optional[str] = None
    location: Optional[str] = None
    mark_type: Optional[str] = None
    confidence: float = 0.0
    evidence_image: Optional[str] = None
    evidence: Optional[str] = None


class EvidenceValue(BaseModel):
    value: Optional[str] = None
    confidence: float = 0.0
    evidence: Optional[str] = None
    evidence_images: list[str] = Field(default_factory=list)


class AntiqueIdentification(BaseModel):
    object_type: EvidenceValue = Field(default_factory=EvidenceValue)
    subtype: EvidenceValue = Field(default_factory=EvidenceValue)
    likely_period: EvidenceValue = Field(default_factory=EvidenceValue)

    estimated_year_start: Optional[int] = None
    estimated_year_end: Optional[int] = None

    manufacturer_candidates: list[dict] = Field(default_factory=list)
    artist_candidates: list[dict] = Field(default_factory=list)
    workshop_candidates: list[dict] = Field(default_factory=list)

    country: EvidenceValue = Field(default_factory=EvidenceValue)
    region: EvidenceValue = Field(default_factory=EvidenceValue)

    materials: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    styles: list[str] = Field(default_factory=list)

    condition: EvidenceValue = Field(default_factory=EvidenceValue)
    dimensions: Optional[dict] = None

    marks: list[MarkEvidence] = Field(default_factory=list)

    signature_text: Optional[str] = None

    provenance_clues: list[str] = Field(default_factory=list)

    rarity: EvidenceValue = Field(default_factory=EvidenceValue)

    image_quality: EvidenceValue = Field(default_factory=EvidenceValue)

    contradictions: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
