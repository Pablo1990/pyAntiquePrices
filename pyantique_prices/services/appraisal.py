"""High-level appraisal service orchestrating the full pipeline."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


def _extract_confidence(value) -> float | None:
    if isinstance(value, dict):
        confidence = value.get("confidence")
        if isinstance(confidence, (int, float)):
            return float(confidence)
    return None


def _compute_identification_confidence(identification: dict | None) -> float:
    if not identification:
        return 0.0
    confidence_values = []
    for key in [
        "object_type",
        "subtype",
        "likely_period",
        "country",
        "region",
        "condition",
        "rarity",
        "image_quality",
    ]:
        confidence = _extract_confidence(identification.get(key))
        if confidence is not None:
            confidence_values.append(max(0.0, min(1.0, confidence)))
    for mark in identification.get("marks", []) or []:
        mark_confidence = mark.get("confidence")
        if isinstance(mark_confidence, (int, float)):
            confidence_values.append(max(0.0, min(1.0, float(mark_confidence))))
    if confidence_values:
        return float(sum(confidence_values) / len(confidence_values))
    if identification.get("object_type"):
        return 0.35
    return 0.0


def _compute_valuation_confidence(
    n_comparables: int,
    valuation_available: bool,
) -> float:
    if n_comparables <= 0:
        return 0.0
    if n_comparables < 3:
        return 0.2
    if n_comparables < 6:
        return 0.35
    if n_comparables < 10:
        return 0.55 if valuation_available else 0.4
    return 0.75 if valuation_available else 0.5


def _fallback_valuation_confidence(n_comparables: int) -> float:
    if n_comparables > 0:
        return _compute_valuation_confidence(n_comparables, False)
    return 0.1


class LegacyWebFallbackEstimator:
    """Use the preserved legacy analyzer as a last-resort rough estimate."""

    def __init__(self, model: str = "qwen3-vl:8b") -> None:
        self.model = model

    def estimate(self, images: Sequence[Path | str], context: str = "") -> dict | None:
        from pyantique_prices.analyzer import AntiqueAnalyzer
        from pyantique_prices.scraper import MultiSourceScraper

        first_image = next(iter(images), None)
        if first_image is None:
            return None

        analyzer = AntiqueAnalyzer(
            model=self.model,
            reasoning_model=self.model,
            deep_thinking=False,
        )
        appraisal_text = analyzer.analyse(
            first_image,
            context=context,
            scraper=MultiSourceScraper(),
        )
        price_range = analyzer.parse_price_range(appraisal_text)
        if not price_range:
            return None
        low, high = price_range
        mid = round((low + high) / 2, 2)
        return {
            "p25": round(low, 2),
            "p50": mid,
            "p75": round(high, 2),
            "low": round(low, 2),
            "mid": mid,
            "high": round(high, 2),
            "num_comparables": 0,
            "valuation_available": False,
            "method": "legacy_web_fallback",
            "confidence_note": "Very low confidence: fallback estimate from legacy web references.",
            "appraisal_text": appraisal_text,
        }


class AppraisalService:
    """Orchestrates multi-image vision analysis, retrieval, and pricing."""

    def __init__(
        self,
        analyzer=None,
        retrieval_session=None,
        retrieval_session_factory=None,
        text_embedding_provider=None,
        image_embedding_provider=None,
        pricer=None,
        fallback_estimator=None,
        base_currency: str = "EUR",
        min_comparables_for_model: int = 6,
        min_comparables_for_confidence: int = 10,
        top_k_comparables: int = 50,
        min_similarity: float = 0.05,
        max_sale_age_years: int = 80,
        min_data_quality_score: float = 0.4,
        similarity_weights: dict[str, float] | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.retrieval_session = retrieval_session
        self.retrieval_session_factory = retrieval_session_factory
        self.text_embedding_provider = text_embedding_provider
        self.image_embedding_provider = image_embedding_provider
        self.pricer = pricer
        self.fallback_estimator = fallback_estimator
        self.base_currency = base_currency
        self.min_comparables_for_model = min_comparables_for_model
        self.min_comparables_for_confidence = min_comparables_for_confidence
        self.top_k_comparables = top_k_comparables
        self.min_similarity = min_similarity
        self.max_sale_age_years = max_sale_age_years
        self.min_data_quality_score = min_data_quality_score
        self.similarity_weights = similarity_weights or {}

    def appraise(
        self,
        images: Sequence[Path | str],
        context: str = "",
        currency: str | None = None,
    ) -> dict:
        """Run the full appraisal pipeline."""
        request_id = str(uuid.uuid4())
        currency = currency or self.base_currency

        result: dict = {
            "request_id": request_id,
            "identification": None,
            "comparables": [],
            "valuation": None,
            "valuation_available": False,
            "warnings": [],
            "identification_confidence": 0.0,
            "valuation_confidence": 0.0,
            "currency": currency,
            "candidate_count": 0,
            "usable_comparable_count": 0,
        }

        if self.analyzer:
            try:
                analysis = self.analyzer.analyze(images, context=context)
                result["identification"] = analysis
                result["identification_confidence"] = _compute_identification_confidence(
                    analysis
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Vision analysis failed: %s", exc)
                result["warnings"].append(f"Vision analysis failed: {exc}")
        else:
            result["warnings"].append("No vision analyzer configured.")

        if (self.retrieval_session or self.retrieval_session_factory) and result.get(
            "identification"
        ):
            try:
                from pyantique_prices.retrieval.comparables import (
                    retrieve_comparables_details,
                )

                if self.retrieval_session_factory:
                    with self.retrieval_session_factory() as session:
                        comparable_details = retrieve_comparables_details(
                            session,
                            result["identification"],
                            top_k=self.top_k_comparables,
                            min_similarity=self.min_similarity,
                            max_sale_age_years=self.max_sale_age_years,
                            min_data_quality_score=self.min_data_quality_score,
                            weights=self.similarity_weights,
                            text_embedding_provider=self.text_embedding_provider,
                            image_embedding_provider=self.image_embedding_provider,
                        )
                else:
                    comparable_details = retrieve_comparables_details(
                        self.retrieval_session,
                        result["identification"],
                        top_k=self.top_k_comparables,
                        min_similarity=self.min_similarity,
                        max_sale_age_years=self.max_sale_age_years,
                        min_data_quality_score=self.min_data_quality_score,
                        weights=self.similarity_weights,
                        text_embedding_provider=self.text_embedding_provider,
                        image_embedding_provider=self.image_embedding_provider,
                    )
                result["comparables"] = comparable_details["comparables"]
                result["candidate_count"] = comparable_details["candidate_count"]
                result["usable_comparable_count"] = comparable_details[
                    "usable_comparable_count"
                ]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Comparable retrieval failed: %s", exc)
                result["warnings"].append(f"Comparable retrieval failed: {exc}")

        n_comparables = len(result["comparables"])
        if n_comparables == 0:
            fallback_used = False
            if self.fallback_estimator:
                try:
                    fallback = self.fallback_estimator.estimate(images, context=context)
                    if fallback:
                        result["valuation"] = fallback
                        result["valuation_available"] = False
                        result["valuation_confidence"] = _fallback_valuation_confidence(0)
                        result.setdefault("evidence", []).append(
                            {
                                "claim": f"Estimated value {currency} {fallback['low']} – {fallback['high']}",
                                "source": "legacy_web_fallback",
                                "confidence": result["valuation_confidence"],
                                "evidence": {
                                    "note": "Derived from the preserved legacy single-image workflow using scraped web references.",
                                },
                            }
                        )
                        result["warnings"].append(
                            "No comparable sales were found in the local database. Returning a very rough fallback estimate from the preserved legacy web-reference workflow."
                        )
                        fallback_used = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Fallback estimation failed: %s", exc)
                    result["warnings"].append(f"Fallback estimation failed: {exc}")

            if not fallback_used:
                result["warnings"].append(
                    "No comparable sales found in the local database. Cannot estimate price."
                )
            result["warnings"].append(
                "Import historical sales first: "
                "`python scripts/import_sales.py data/sales.csv` "
                "then `python scripts/index_sales.py`."
            )
        else:
            if self.pricer and result["identification"]:
                try:
                    from pyantique_prices.pricing.features import extract_features

                    features = extract_features(
                        result["identification"],
                        result["comparables"],
                    )
                    valuation = self.pricer.predict(features, result["comparables"])
                    if valuation:
                        result["valuation"] = valuation
                        result["valuation_available"] = valuation.get(
                            "valuation_available",
                            False,
                        )
                        result["valuation_confidence"] = _compute_valuation_confidence(
                            n_comparables,
                            result["valuation_available"],
                        )
                        if n_comparables < 3:
                            result["warnings"].append(
                                "Very few comparables (1-2). Reference estimate only."
                            )
                        elif n_comparables < self.min_comparables_for_model:
                            result["warnings"].append(
                                "Limited comparables (3-5). Wide estimate."
                            )
                        elif n_comparables < self.min_comparables_for_confidence:
                            result["warnings"].append(
                                "Usable but uncertain comparables (6-9)."
                            )
                except Exception as exc:  # noqa: BLE001
                    logger.error("Pricing failed: %s", exc)
                    result["warnings"].append(f"Pricing failed: {exc}")
            else:
                result["valuation_confidence"] = _compute_valuation_confidence(
                    n_comparables,
                    False,
                )

        result["warnings"].append(
            "This is an AI-assisted market estimate, not a formal appraisal."
        )
        return result
