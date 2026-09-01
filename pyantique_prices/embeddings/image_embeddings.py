"""Image embedding interfaces for optional visual similarity."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ImageEmbedder(Protocol):
    def embed(self, image: str | Path) -> list[float]:
        """Return an embedding for one image."""


class NullImageEmbedder:
    """No-op image embedder used when visual embeddings are disabled."""

    def embed(self, image: str | Path) -> list[float]:
        del image
        return []
