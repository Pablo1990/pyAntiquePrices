"""Pydantic schemas for structured antique identification."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def _strip_or_none(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("value") or value.get("name") or value.get("text")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class Candidate(BaseModel):
    name: str
    confidence: float = 0.0
    evidence: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_candidate(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value, "confidence": 0.5}
        return value


class MarkEvidence(BaseModel):
    image_name: Optional[str] = None
    text: Optional[str] = None
    normalized_text: Optional[str] = None
    location: Optional[str] = None
    mark_type: Optional[str] = None
    confidence: float = 0.0
    evidence_image: Optional[str] = None
    evidence: Optional[str] = None
    manufacturer_candidates: list[Candidate] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_mark(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"text": value, "confidence": 0.5}
        return value


class EvidenceValue(BaseModel):
    value: Optional[str] = None
    confidence: float = 0.0
    evidence: Optional[str] = None
    evidence_images: list[str] = Field(default_factory=list)


class ImageRole(str, Enum):
    front = "front"
    back = "back"
    side = "side"
    base = "base"
    maker_mark = "maker_mark"
    signature = "signature"
    detail = "detail"
    label = "label"
    unknown = "unknown"


class AntiqueImage(BaseModel):
    path: str
    name: str
    role: ImageRole = ImageRole.unknown


class AntiqueImageSet(BaseModel):
    images: list[AntiqueImage] = Field(default_factory=list)

    @classmethod
    def from_paths(
        cls,
        paths: list[Path | str],
        roles: list[ImageRole] | None = None,
    ) -> "AntiqueImageSet":
        role_values = roles or []
        images = []
        for index, raw_path in enumerate(paths):
            path = Path(raw_path)
            role = role_values[index] if index < len(role_values) else ImageRole.unknown
            images.append(AntiqueImage(path=str(path), name=path.name, role=role))
        return cls(images=images)

    @property
    def roles(self) -> list[ImageRole]:
        return [image.role for image in self.images]

    @property
    def paths(self) -> list[str]:
        return [image.path for image in self.images]


class Identification(BaseModel):
    image_roles: dict[str, ImageRole] = Field(default_factory=dict)

    object_type: Optional[str] = None
    subtype: Optional[str] = None

    period: Optional[str] = None
    likely_period: Optional[str] = None
    estimated_year_start: Optional[int] = None
    estimated_year_end: Optional[int] = None

    manufacturer_candidates: list[Candidate] = Field(default_factory=list)
    artist_candidates: list[Candidate] = Field(default_factory=list)
    workshop_candidates: list[Candidate] = Field(default_factory=list)

    country: Optional[str] = None
    region: Optional[str] = None

    materials: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    styles: list[str] = Field(default_factory=list)

    condition: Optional[str] = None

    height: float | None = None
    width: float | None = None
    depth: float | None = None
    diameter: float | None = None
    weight: float | None = None
    dimensions: Optional[dict[str, Any]] = None

    marks: list[MarkEvidence] = Field(default_factory=list)

    signature_text: Optional[str] = None
    provenance_clues: list[str] = Field(default_factory=list)

    rarity_assessment: Optional[str] = None
    rarity: Optional[str] = None
    image_quality: Optional[str] = None

    contradictions: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)

    normalized_description: Optional[str] = None
    search_document: Optional[str] = None
    source_images: list[str] = Field(default_factory=list)
    raw_model_output: Optional[str] = None

    @field_validator(
        "object_type",
        "subtype",
        "period",
        "likely_period",
        "country",
        "region",
        "condition",
        "signature_text",
        "rarity_assessment",
        "rarity",
        "image_quality",
        mode="before",
    )
    @classmethod
    def _coerce_text_fields(cls, value: Any) -> Any:
        return _strip_or_none(value)

    @field_validator("materials", "techniques", "styles", "provenance_clues", mode="before")
    @classmethod
    def _coerce_list_fields(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            parts = [part.strip() for part in value.replace(";", ",").split(",")]
            return [part for part in parts if part]
        if isinstance(value, list):
            result = []
            for item in value:
                text = _strip_or_none(item)
                if text:
                    result.append(text)
            return result
        text = _strip_or_none(value)
        return [text] if text else []

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if not payload.get("period") and payload.get("likely_period"):
            payload["period"] = payload.get("likely_period")
        if not payload.get("likely_period") and payload.get("period"):
            payload["likely_period"] = payload.get("period")
        if not payload.get("rarity_assessment") and payload.get("rarity"):
            payload["rarity_assessment"] = payload.get("rarity")
        return payload


AntiqueIdentification = Identification
