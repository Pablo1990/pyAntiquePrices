"""Central configuration loaded from environment variables / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(*paths: Path) -> None:
    """Load ``KEY=VALUE`` lines from ``.env`` files into ``os.environ``.

    Looks in the current directory and the project root by default.  Values
    already set in the real environment are never overwritten, so an
    ``export`` in the shell always wins over the file.
    """
    candidates = paths or (Path.cwd() / ".env", _PROJECT_ROOT / ".env")
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv()


@dataclass
class Settings:
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    ollama_vision_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:8b")
    )
    ollama_embed_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
    )
    ollama_num_ctx: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    )
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "sqlite:///./data/antiquegpt.db"
        )
    )
    base_currency: str = field(default_factory=lambda: os.getenv("BASE_CURRENCY", "EUR"))
    top_k_comparables: int = field(
        default_factory=lambda: int(os.getenv("TOP_K_COMPARABLES", "20"))
    )
    min_similarity: float = field(
        default_factory=lambda: float(os.getenv("MIN_SIMILARITY", "0.05"))
    )
    max_sale_age_years: int = field(
        default_factory=lambda: int(os.getenv("MAX_SALE_AGE_YEARS", "80"))
    )
    min_data_quality_score: float = field(
        default_factory=lambda: float(os.getenv("MIN_DATA_QUALITY_SCORE", "0.4"))
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
    semantic_weight: float = field(
        default_factory=lambda: float(os.getenv("SEMANTIC_WEIGHT", "0.50"))
    )
    visual_weight: float = field(
        default_factory=lambda: float(os.getenv("VISUAL_WEIGHT", "0.30"))
    )
    structured_weight: float = field(
        default_factory=lambda: float(os.getenv("STRUCTURED_WEIGHT", "0.20"))
    )
    price_target: str = field(
        default_factory=lambda: os.getenv(
            "PRICE_TARGET", "normalized_realized_price"
        )
    )
    # eBay developer API (https://developer.ebay.com/my/keys)
    ebay_client_id: str = field(default_factory=lambda: os.getenv("EBAY_CLIENT_ID", ""))
    ebay_client_secret: str = field(
        default_factory=lambda: os.getenv("EBAY_CLIENT_SECRET", "")
    )
    ebay_marketplace_id: str = field(
        default_factory=lambda: os.getenv("EBAY_MARKETPLACE_ID", "EBAY_ES")
    )
    ebay_environment: str = field(
        default_factory=lambda: os.getenv("EBAY_ENVIRONMENT", "production")
    )
    # Live, per-appraisal eBay listings (shown, never stored). Needs the keys above.
    ebay_live_listings: bool = field(
        default_factory=lambda: os.getenv("EBAY_LIVE_LISTINGS", "true").lower() == "true"
    )
    ebay_live_max_results: int = field(
        default_factory=lambda: int(os.getenv("EBAY_LIVE_MAX_RESULTS", "10"))
    )
    # "active" = Browse API (any keyset); "sold" = Marketplace Insights (approval needed)
    ebay_live_mode: str = field(
        default_factory=lambda: os.getenv("EBAY_LIVE_MODE", "active").lower()
    )
    # 20081 = eBay "Antiques" top-level category; empty string = no filter.
    ebay_category_ids: str = field(
        default_factory=lambda: os.getenv("EBAY_CATEGORY_IDS", "20081")
    )


settings = Settings()
