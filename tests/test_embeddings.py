from __future__ import annotations

from pyantique_prices.embeddings import (
    MalformedEmbeddingResponseError,
    MissingEmbeddingModelError,
    OllamaTextEmbeddingProvider,
    OllamaUnavailableError,
)


class _HealthyClient:
    host = "http://example.test:11434"

    def __init__(self, embedding=None, available=True, healthy=True):
        self._embedding = embedding if embedding is not None else [0.1, 0.2]
        self._available = available
        self._healthy = healthy

    def health_check(self) -> bool:
        return self._healthy

    def model_available(self, model: str | None = None) -> bool:  # noqa: ARG002
        return self._available

    def embed_text(self, text: str) -> list[float] | object:  # noqa: ARG002
        return self._embedding


def test_ollama_embedding_provider_returns_embedding():
    provider = OllamaTextEmbeddingProvider(
        host="http://example.test:11434",
        model="embeddinggemma",
        client=_HealthyClient(),
    )
    assert provider.embed("clock") == [0.1, 0.2]


def test_ollama_embedding_provider_handles_unavailable_ollama():
    provider = OllamaTextEmbeddingProvider(
        host="http://example.test:11434",
        model="embeddinggemma",
        client=_HealthyClient(healthy=False),
    )
    try:
        provider.embed("clock")
    except OllamaUnavailableError as exc:
        assert "unavailable" in str(exc).lower()
    else:
        raise AssertionError("Expected OllamaUnavailableError")


def test_ollama_embedding_provider_handles_missing_model():
    provider = OllamaTextEmbeddingProvider(
        host="http://example.test:11434",
        model="missing-model",
        client=_HealthyClient(available=False),
    )
    try:
        provider.embed("clock")
    except MissingEmbeddingModelError as exc:
        assert "missing-model" in str(exc)
    else:
        raise AssertionError("Expected MissingEmbeddingModelError")


def test_ollama_embedding_provider_handles_malformed_response():
    provider = OllamaTextEmbeddingProvider(
        host="http://example.test:11434",
        model="embeddinggemma",
        client=_HealthyClient(embedding={"embedding": "bad"}),
    )
    try:
        provider.embed("clock")
    except MalformedEmbeddingResponseError:
        pass
    else:
        raise AssertionError("Expected MalformedEmbeddingResponseError")
