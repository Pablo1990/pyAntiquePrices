"""Multi-image antique analysis using OllamaClient."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .ollama import OllamaClient

MIN_IMAGES = 3
MAX_IMAGES = 5
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


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

    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    def analyze(self, images: Sequence[Path | str], context: str = "") -> dict:
        """Run multi-image analysis and return raw text + image metadata."""
        paths = validate_images(images)
        from .prompts import MULTI_IMAGE_PROMPT, SYSTEM_PROMPT

        prompt = MULTI_IMAGE_PROMPT.format(context=context or "None provided")
        raw = self.client.analyze_images(paths, prompt, system=SYSTEM_PROMPT)
        return {"raw": raw, "images": [str(path) for path in paths]}
