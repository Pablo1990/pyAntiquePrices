"""Image embedding interfaces for optional visual similarity."""

from __future__ import annotations

from typing import Protocol

from PIL import Image


class ImageEmbeddingProvider(Protocol):
    def embed(self, image: Image.Image) -> list[float]:
        """Return an embedding for one image."""


class CLIPCompatibleImageEmbeddingProvider:
    """Adapter for CLIP-compatible backends supplied by the caller."""

    def __init__(self, backend) -> None:
        self.backend = backend

    def embed(self, image: Image.Image) -> list[float]:
        if hasattr(self.backend, "embed_image"):
            vector = self.backend.embed_image(image)
        else:
            vector = self.backend(image)
        return [float(value) for value in vector]


class NullImageEmbeddingProvider:
    """No-op image embedder used when visual embeddings are disabled."""

    def embed(self, image: Image.Image) -> list[float]:
        del image
        return []


ImageEmbedder = ImageEmbeddingProvider
NullImageEmbedder = NullImageEmbeddingProvider
