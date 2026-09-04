"""FastAPI application factory for AntiqueGPT endpoints."""

from __future__ import annotations

from fastapi import FastAPI

from pyantique_prices.config import Settings
from pyantique_prices.data.database import create_tables, get_engine, get_session_factory
from pyantique_prices.embeddings import (
    NullImageEmbeddingProvider,
    OllamaTextEmbeddingProvider,
)
from pyantique_prices.pricing.model import PricePredictor
from pyantique_prices.services.appraisal import AppraisalService, LegacyWebFallbackEstimator
from pyantique_prices.vision.analyzer import MultiImageAnalyzer
from pyantique_prices.vision.ollama import OllamaClient

from .appraisals import router as appraisals_router
from .health import router as health_router
from .sales import router as sales_router


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    engine = get_engine(settings.database_url)
    create_tables(engine)
    session_factory = get_session_factory(engine)

    client = OllamaClient(
        host=settings.ollama_host,
        model=settings.ollama_vision_model,
        num_ctx=settings.ollama_num_ctx,
    )
    analyzer = MultiImageAnalyzer(client=client)
    text_embedding_provider = OllamaTextEmbeddingProvider(
        host=settings.ollama_host,
        model=settings.ollama_embed_model,
        num_ctx=settings.ollama_num_ctx,
        require_model=False,
    )
    image_embedding_provider = NullImageEmbeddingProvider()
    pricer = PricePredictor(
        min_comparables_for_model=settings.min_comparables_for_model,
        min_comparables_for_confidence=settings.min_comparables_for_confidence,
    )
    service = AppraisalService(
        analyzer=analyzer,
        retrieval_session_factory=session_factory,
        text_embedding_provider=text_embedding_provider,
        image_embedding_provider=image_embedding_provider,
        pricer=pricer,
        fallback_estimator=LegacyWebFallbackEstimator(model=settings.ollama_vision_model),
        base_currency=settings.base_currency,
        min_comparables_for_model=settings.min_comparables_for_model,
        min_comparables_for_confidence=settings.min_comparables_for_confidence,
        top_k_comparables=settings.top_k_comparables,
        min_similarity=settings.min_similarity,
        max_sale_age_years=settings.max_sale_age_years,
        min_data_quality_score=settings.min_data_quality_score,
        similarity_weights={
            "semantic": settings.semantic_weight,
            "visual": settings.visual_weight,
            "structured": settings.structured_weight,
        },
    )

    app = FastAPI(title="AntiqueGPT API")
    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.appraisal_service = service
    app.state.model_version = {
        "vision_model": settings.ollama_vision_model,
        "pricing_model": "price_predictor_v1",
    }
    app.include_router(health_router)
    app.include_router(appraisals_router)
    app.include_router(sales_router)
    return app


app = create_app()
