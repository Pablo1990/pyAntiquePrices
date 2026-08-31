"""Central configuration loaded from environment variables / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    ollama_vision_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_VISION_MODEL", "minicpm-v")
    )
    ollama_embed_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    )
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "sqlite:///./data/antiquegpt.db"
        )
    )
    base_currency: str = field(default_factory=lambda: os.getenv("BASE_CURRENCY", "EUR"))
    top_k_comparables: int = field(
        default_factory=lambda: int(os.getenv("TOP_K_COMPARABLES", "50"))
    )
    min_comparables_for_model: int = field(
        default_factory=lambda: int(os.getenv("MIN_COMPARABLES_FOR_MODEL", "6"))
    )
    min_comparables_for_confidence: int = field(
        default_factory=lambda: int(os.getenv("MIN_COMPARABLES_FOR_CONFIDENCE", "10"))
    )
    enable_image_embeddings: bool = field(
        default_factory=lambda: os.getenv("ENABLE_IMAGE_EMBEDDINGS", "false").lower()
        == "true"
    )
    price_target: str = field(
        default_factory=lambda: os.getenv(
            "PRICE_TARGET", "normalized_realized_price"
        )
    )


settings = Settings()
