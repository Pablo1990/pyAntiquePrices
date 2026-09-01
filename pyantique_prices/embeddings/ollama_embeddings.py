"""Text embedding helper backed by Ollama."""

from __future__ import annotations

from pyantique_prices.vision.ollama import OllamaClient


class OllamaTextEmbedder:
    """Generate text embeddings with a configured Ollama model."""

    def __init__(self, host: str, model: str, num_ctx: int = 8192) -> None:
        self.client = OllamaClient(host=host, model=model, num_ctx=num_ctx)

    def embed(self, text: str) -> list[float]:
        return self.client.embed_text(text)
