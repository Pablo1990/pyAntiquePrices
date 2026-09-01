"""Multi-image antique analysis using OllamaClient."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from .marks import MarkAnalysisService
from .ollama import OllamaClient, is_context_overflow_error
from .schemas import AntiqueIdentification

MIN_IMAGES = 3
MAX_IMAGES = 5
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_CONTEXT_CHARS = 600


def validate_images(images: Sequence[Path | str]) -> list[Path]:
    """Validate image paths and return list of Paths."""
    paths = [Path(image) for image in images]
    if len(paths) < MIN_IMAGES:
        raise ValueError(f"At least {MIN_IMAGES} images required, got {len(paths)}")
    if len(paths) > MAX_IMAGES:
        raise ValueError(f"At most {MAX_IMAGES} images allowed, got {len(paths)}")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {path.suffix}")
    return paths


class MultiImageAnalyzer:
    """Analyze an antique from multiple images."""

    def __init__(self, client: OllamaClient, mark_service: MarkAnalysisService | None = None) -> None:
        self.client = client
        self.mark_service = mark_service or MarkAnalysisService()

    @staticmethod
    def _to_evidence(value):
        if isinstance(value, dict):
            return value
        if value is None:
            return {}
        return {
            "value": str(value),
            "confidence": 0.5,
            "evidence": "Auto-converted from unstructured model output",
            "evidence_images": [],
        }

    @staticmethod
    def _extract_text_field(raw: str, keys: list[str]) -> str | None:
        for key in keys:
            pattern = rf"(?im)^\s*{re.escape(key)}\s*[:\-]\s*(.+?)\s*$"
            match = re.search(pattern, raw)
            if match:
                value = match.group(1).strip()
                if value:
                    return value
        return None

    def _fallback_from_text(self, raw: str) -> dict:
        object_type = self._extract_text_field(raw, ["object", "object type", "item"])
        period = self._extract_text_field(raw, ["period", "date", "era"])
        country = self._extract_text_field(raw, ["country", "origin", "region"])
        condition = self._extract_text_field(raw, ["condition", "state"])
        materials_text = self._extract_text_field(raw, ["materials", "material", "medium"])
        materials = []
        if materials_text:
            materials = [item.strip() for item in re.split(r"[,;/]", materials_text) if item.strip()]
        payload = {}
        if object_type:
            payload["object_type"] = object_type
        if period:
            payload["likely_period"] = period
        if country:
            payload["country"] = country
        if condition:
            payload["condition"] = condition
        if materials:
            payload["materials"] = materials
        return payload

    def _normalize_payload(self, payload: dict) -> dict:
        evidence_fields = [
            "object_type",
            "subtype",
            "likely_period",
            "country",
            "region",
            "condition",
            "rarity",
            "image_quality",
        ]
        normalized = dict(payload)
        for field in evidence_fields:
            normalized[field] = self._to_evidence(normalized.get(field))

        marks = normalized.get("marks")
        if isinstance(marks, list):
            normalized_marks = []
            for mark in marks:
                if isinstance(mark, str):
                    normalized_marks.append({"text": mark, "confidence": 0.5})
                elif isinstance(mark, dict):
                    normalized_marks.append(mark)
            normalized["marks"] = normalized_marks
        return normalized

    @staticmethod
    def _extract_json(raw: str) -> dict:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start : end + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _truncate_context(context: str) -> str:
        context = context.strip()
        if len(context) <= MAX_CONTEXT_CHARS:
            return context
        return f"{context[:MAX_CONTEXT_CHARS].rstrip()}…"

    def analyze(self, images: Sequence[Path | str], context: str = "") -> dict:
        """Run multi-image analysis and return structured identification."""
        paths = validate_images(images)
        from .prompts import (
            COMPACT_MULTI_IMAGE_PROMPT,
            MULTI_IMAGE_PROMPT,
            SYSTEM_PROMPT,
        )

        truncated_context = self._truncate_context(context)
        prompt = MULTI_IMAGE_PROMPT.format(context=truncated_context or "None provided")
        try:
            raw = self.client.analyze_images(paths, prompt, system=SYSTEM_PROMPT)
        except Exception as exc:  # noqa: BLE001
            if not is_context_overflow_error(exc):
                raise
            compact_prompt = COMPACT_MULTI_IMAGE_PROMPT.format(
                context=truncated_context or "None provided"
            )
            try:
                raw = self.client.analyze_images(paths, compact_prompt, system=None)
            except Exception as retry_exc:  # noqa: BLE001
                if is_context_overflow_error(retry_exc):
                    raise ValueError(
                        "Vision analysis request exceeded Ollama context size. "
                        "Increase OLLAMA_NUM_CTX or reduce the amount of context text."
                    ) from retry_exc
                raise
        payload = self._extract_json(raw)
        if not payload:
            payload = self._fallback_from_text(raw)
        normalized = self._normalize_payload(payload)
        identification = AntiqueIdentification.model_validate(normalized).model_dump()
        enriched_marks = self.mark_service.analyze(identification)
        identification["marks"] = enriched_marks
        if not identification.get("manufacturer_candidates"):
            from_marks = []
            for mark in enriched_marks:
                for candidate in mark.get("manufacturer_candidates", []):
                    from_marks.append({"name": candidate, "confidence": mark.get("confidence", 0.0)})
            identification["manufacturer_candidates"] = from_marks
        identification["source_images"] = [str(path) for path in paths]
        identification["raw_model_output"] = raw
        return identification
