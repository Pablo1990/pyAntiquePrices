"""Text embedding helper backed by Ollama."""

from __future__ import annotations

from pyantique_prices.vision.ollama import OllamaClient


class OllamaTextEmbedder:
    """Generate text embeddings with a configured Ollama model."""

    def __init__(self, host: str, model: str) -> None:
        self.client = OllamaClient(host=host, model=model)

    def embed(self, text: str) -> list[float]:
        return self.client.embed_text(text)
