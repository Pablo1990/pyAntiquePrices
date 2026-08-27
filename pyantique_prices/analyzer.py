"""Ollama-powered antique analyser."""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llava"

_SYSTEM_PROMPT = (
    "You are an expert antique appraiser with decades of experience in fine art, "
    "furniture, ceramics, silver, jewellery and collectibles. "
    "When presented with an image you carefully examine craftsmanship, style, "
    "materials, patina, maker's marks and provenance clues. "
    "You always give a structured, confident assessment."
)

_USER_TEMPLATE = """\
Please analyse the following antique item image and provide a detailed appraisal.

Additional context provided by the user:
{context}

Reference prices found online for similar items:
{reference_prices}

Your appraisal must include:
1. **Description**: What the item appears to be (type, style, origin, materials).
2. **Estimated Age**: The probable period or decade of manufacture, with reasoning.
3. **Condition Assessment**: Based solely on what is visible in the image.
4. **Estimated Price Range**: A realistic market range in EUR (and USD if relevant), \
distinguishing auction estimate from retail/dealer price.
5. **Key factors**: The main factors that increase or decrease value.
6. **Confidence**: Your confidence level (low / medium / high) with an explanation.

Be specific and justify each point with observable evidence from the image.
"""


class AntiqueAnalyzer:
    """Analyse antique images with a local Ollama vision model.

    Parameters
    ----------
    model:
        The Ollama model to use (default: ``llava``).  Any multimodal model
        available in your local Ollama installation can be used.
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self.model = model
        self.on_pull_progress = None  # optional callable(status: str)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        image_path: str | Path,
        context: str = "",
        reference_prices: str = "",
    ) -> str:
        """Return a full appraisal string for the supplied image.

        Parameters
        ----------
        image_path:
            Path to the image file (JPEG, PNG, WEBP, …).
        context:
            Optional free-text description the user provides about the item.
        reference_prices:
            Formatted string of reference prices scraped from the web
            (already fetched by :class:`TodoColeccionScraper`).

        Returns
        -------
        str
            The model's appraisal text.
        """
        import ollama  # imported lazily so the package loads without ollama running

        self._ensure_model(ollama)

        image_data = self._encode_image(image_path)
        prompt = _USER_TEMPLATE.format(
            context=context.strip() or "No additional context provided.",
            reference_prices=reference_prices.strip() or "No reference prices available.",
        )
        logger.debug("Sending request to Ollama model '%s'", self.model)
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_data],
                },
            ],
        )
        return response["message"]["content"]

    def _ensure_model(self, ollama) -> None:
        """Pull *self.model* if it is not already available locally."""
        try:
            local_models = [m["name"] for m in ollama.list().get("models", [])]
            # Normalise names: strip the ":latest" suffix for comparison
            def _base(name: str) -> str:
                return name.split(":")[0]

            if not any(_base(m) == _base(self.model) for m in local_models):
                logger.info("Model '%s' not found locally – pulling from Ollama…", self.model)
                print(f"Downloading model '{self.model}' – this may take a few minutes…")
                for progress in ollama.pull(self.model, stream=True):
                    status = progress.get("status", "")
                    if status:
                        print(f"  {status}", end="\r", flush=True)
                        if callable(self.on_pull_progress):
                            self.on_pull_progress(status)
                print()  # newline after progress
                logger.info("Model '%s' pulled successfully.", self.model)
        except Exception as exc:  # noqa: BLE001
            # If we cannot verify local models (e.g. Ollama not running) let
            # the subsequent chat() call surface the real error.
            logger.debug("Could not verify/pull model: %s", exc)

    def list_available_models(self) -> list[str]:
        """Return a list of locally available Ollama model names."""
        import ollama

        models = ollama.list()
        return [m["name"] for m in models.get("models", [])]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}

    @classmethod
    def collect_images(cls, path: str | Path) -> list[Path]:
        """Return a sorted list of image files under *path*.

        If *path* is a file it is returned as a single-element list.
        If *path* is a directory all image files found directly inside it
        (non-recursive) are returned, sorted by filename.
        """
        p = Path(path)
        if p.is_file():
            return [p]
        if p.is_dir():
            return sorted(
                f for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in cls._IMAGE_EXTENSIONS
            )
        raise FileNotFoundError(f"Path not found: {p}")

    @staticmethod
    def _encode_image(image_path: str | Path) -> str:
        """Return the base64-encoded content of an image file."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        with path.open("rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")

    @staticmethod
    def parse_price_range(appraisal_text: str) -> Optional[tuple[float, float]]:
        """Extract the first EUR price range found in *appraisal_text*.

        Returns a ``(low, high)`` tuple or ``None`` if no range is found.
        """
        # Matches patterns like "500 – 800 EUR", "€500-€800", "1,200 to 1,500 EUR"
        pattern = r"(?:€|EUR\s*)?([\d,\.]+)\s*(?:–|-|to)\s*(?:€|EUR\s*)?([\d,\.]+)\s*(?:EUR|€)?"
        match = re.search(pattern, appraisal_text, re.IGNORECASE)
        if match:
            low = float(match.group(1).replace(",", ""))
            high = float(match.group(2).replace(",", ""))
            return (low, high)
        return None
