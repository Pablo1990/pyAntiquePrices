"""Text embedding helpers backed by Ollama."""

from __future__ import annotations

from typing import Protocol

from pyantique_prices.vision.ollama import OllamaClient


class EmbeddingProviderError(RuntimeError):
    """Base exception for embedding provider failures."""


class OllamaUnavailableError(EmbeddingProviderError):
    """Raised when Ollama cannot be reached."""


class MissingEmbeddingModelError(EmbeddingProviderError):
    """Raised when the configured embedding model is not available."""


class MalformedEmbeddingResponseError(EmbeddingProviderError):
    """Raised when Ollama returns an invalid embedding payload."""


class TextEmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]:
        """Return an embedding for one text query."""


class OllamaTextEmbeddingProvider:
    """Generate text embeddings with a configured Ollama model."""

    def __init__(
        self,
        host: str,
        model: str,
        num_ctx: int = 8192,
        *,
        client: OllamaClient | None = None,
        require_model: bool = True,
    ) -> None:
        self.client = client or OllamaClient(host=host, model=model, num_ctx=num_ctx)
        self.model = model
        self.require_model = require_model

    def embed(self, text: str) -> list[float]:
        if not self.client.health_check():
            raise OllamaUnavailableError(
                f"Ollama is unavailable at {self.client.host}."
            )
        if self.require_model and not self.client.model_available(self.model):
            raise MissingEmbeddingModelError(
                f"Embedding model '{self.model}' is not available in Ollama."
            )
        try:
            embedding = self.client.embed_text(text)
        except EmbeddingProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise OllamaUnavailableError(f"Failed to generate embedding: {exc}") from exc
        if not isinstance(embedding, list) or not embedding or not all(
            isinstance(value, (int, float)) for value in embedding
        ):
            raise MalformedEmbeddingResponseError(
                "Ollama returned a malformed embedding response."
            )
        return [float(value) for value in embedding]


OllamaTextEmbedder = OllamaTextEmbeddingProvider
