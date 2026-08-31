"""High-level appraisal service orchestrating the full pipeline."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


class AppraisalService:
    """Orchestrates multi-image vision analysis, retrieval, and pricing."""

    def __init__(
        self,
        analyzer=None,
        retrieval_session=None,
        pricer=None,
        base_currency: str = "EUR",
    ) -> None:
        self.analyzer = analyzer
        self.retrieval_session = retrieval_session
        self.pricer = pricer
        self.base_currency = base_currency

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
        }

        if self.analyzer:
            try:
                analysis = self.analyzer.analyze(images, context=context)
                result["identification"] = analysis
            except Exception as exc:  # noqa: BLE001
                logger.error("Vision analysis failed: %s", exc)
                result["warnings"].append(f"Vision analysis failed: {exc}")
        else:
            result["warnings"].append("No vision analyzer configured.")

        if self.retrieval_session and result.get("identification"):
            try:
                from pyantique_prices.retrieval.comparables import retrieve_comparables

                comparables = retrieve_comparables(
                    self.retrieval_session,
                    result["identification"],
                )
                result["comparables"] = comparables
            except Exception as exc:  # noqa: BLE001
                logger.warning("Comparable retrieval failed: %s", exc)
                result["warnings"].append(f"Comparable retrieval failed: {exc}")

        n_comparables = len(result["comparables"])
        if n_comparables == 0:
            result["warnings"].append(
                "No comparable sales found. Cannot estimate price."
            )
        elif n_comparables < 3:
            result["warnings"].append("Very few comparables (1-2). Estimate unreliable.")
            result["valuation_available"] = False
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
                except Exception as exc:  # noqa: BLE001
                    logger.error("Pricing failed: %s", exc)
                    result["warnings"].append(f"Pricing failed: {exc}")

        result["warnings"].append(
            "This is an AI-assisted market estimate, not a formal appraisal."
        )
        return result
