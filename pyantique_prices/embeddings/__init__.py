"""Embedding interfaces and implementations."""

from .image_embeddings import ImageEmbedder, NullImageEmbedder
from .ollama_embeddings import OllamaTextEmbedder

__all__ = ["ImageEmbedder", "NullImageEmbedder", "OllamaTextEmbedder"]
