from __future__ import annotations

from pyantique_prices.config import Settings


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("TOP_K_COMPARABLES", raising=False)
    monkeypatch.delenv("MIN_SIMILARITY", raising=False)

    settings = Settings()

    assert settings.ollama_host == "http://localhost:11434"
    assert settings.top_k_comparables == 50
    assert settings.min_similarity == 0.05
    assert settings.enable_image_embeddings is False


def test_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://example.test:11434")
    monkeypatch.setenv("BASE_CURRENCY", "USD")
    monkeypatch.setenv("ENABLE_IMAGE_EMBEDDINGS", "true")

    settings = Settings()

    assert settings.ollama_host == "http://example.test:11434"
    assert settings.base_currency == "USD"
    assert settings.enable_image_embeddings is True
