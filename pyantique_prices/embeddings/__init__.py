"""Embedding interfaces and implementations."""

from .image_embeddings import (
    CLIPCompatibleImageEmbeddingProvider,
    ImageEmbedder,
    ImageEmbeddingProvider,
    NullImageEmbedder,
    NullImageEmbeddingProvider,
)
from .ollama_embeddings import (
    EmbeddingProviderError,
    MalformedEmbeddingResponseError,
    MissingEmbeddingModelError,
    OllamaTextEmbedder,
    OllamaTextEmbeddingProvider,
    OllamaUnavailableError,
    TextEmbeddingProvider,
)

__all__ = [
    "CLIPCompatibleImageEmbeddingProvider",
    "EmbeddingProviderError",
    "ImageEmbedder",
    "ImageEmbeddingProvider",
    "MalformedEmbeddingResponseError",
    "MissingEmbeddingModelError",
    "NullImageEmbedder",
    "NullImageEmbeddingProvider",
    "OllamaTextEmbedder",
    "OllamaTextEmbeddingProvider",
    "OllamaUnavailableError",
    "TextEmbeddingProvider",
]
