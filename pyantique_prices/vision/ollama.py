"""Ollama client abstraction."""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OllamaClient:
    """Abstraction over Ollama API calls."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen3-vl:8b",
        num_ctx: int = 8192,
    ) -> None:
        self.host = host
        self.model = model
        self.num_ctx = num_ctx
        self._client = None

    def _get_client(self):
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.host)
        return self._client

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            client.list()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama health check failed: %s", exc)
            return False

    def model_available(self, model: str | None = None) -> bool:
        model = model or self.model
        try:
            client = self._get_client()
            models = client.list()
            model_items = getattr(models, "models", None)
            if model_items is None and isinstance(models, dict):
                model_items = models.get("models", [])
            names = [
                getattr(item, "model", None)
                or getattr(item, "name", None)
                or item.get("name", "")
                for item in model_items or []
            ]
            return any(model in name for name in names)
        except Exception:  # noqa: BLE001
            return False

    def analyze_images(
        self,
        images: list[Path | str],
        prompt: str,
        system: str | None = None,
    ) -> str:
        """Send images + prompt to vision model, return text response."""
        client = self._get_client()
        encoded = []
        for image in images:
            path = Path(image)
            encoded.append(base64.b64encode(path.read_bytes()).decode())

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt, "images": encoded})

        response = client.chat(
            model=self.model,
            messages=messages,
            options={"num_ctx": self.num_ctx},
        )
        message = getattr(response, "message", None)
        if message is not None:
            return getattr(message, "content", "")
        return response["message"]["content"]


def is_context_overflow_error(exc: Exception) -> bool:
    message = str(exc)
    patterns = [
        r"exceeds the available context size",
        r"exceed_context_size_error",
        r"\bn_ctx\b",
        r"\bcontext size\b",
    ]
    return any(re.search(pattern, message, re.IGNORECASE) for pattern in patterns)

    def embed_text(self, text: str) -> list[float]:
        """Generate text embeddings."""
        client = self._get_client()
        response = client.embeddings(model=self.model, prompt=text)
        embedding = getattr(response, "embedding", None)
        if embedding is not None:
            return embedding
        return response["embedding"]
