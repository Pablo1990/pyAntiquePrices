"""Ollama-powered antique analyser."""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Best locally-runnable vision models in roughly descending quality order.
# Users can pick any Ollama-compatible model.
RECOMMENDED_MODELS = [
    "llama3.2-vision",       # Meta – excellent vision + reasoning
    "deepseek-r1",           # DeepSeek – strong chain-of-thought reasoning (use with llava for images)
    "llava:34b",             # LLaVA large – strong multimodal
    "llava",                 # LLaVA base – fallback
    "gemma3",                # Google Gemma 3 – good general reasoning
    "mistral-small3.1",      # Mistral – capable vision
]

_DEFAULT_MODEL = "llama3.2-vision"

# -------------------------------------------------------------------------
# Prompts
# -------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a world-class antique appraiser with decades of hands-on experience in \
fine art, furniture, ceramics, silver, jewellery, clocks, toys, books and all \
categories of collectibles. You have appraised items for leading auction houses \
(Christie's, Sotheby's, Catawiki) and private clients across Europe and the Americas.

When you examine an image you follow a rigorous methodology:
  • Identify the object type, cultural origin and decorative style.
  • Assess manufacturing techniques (hand-made vs. industrial, casting, gilding, etc.).
  • Note patina, wear patterns, repairs, losses and any visible marks or signatures.
  • Compare with known reference pieces and market sales data you have memorised.
  • Apply current market conditions: collector demand, rarity, provenance weight.

You ALWAYS reason step-by-step before giving a final answer, explicitly showing \
your chain of thought so that the reasoning can be verified. \
Your final appraisal is structured, specific and justified by visible evidence."""

_USER_TEMPLATE_STANDARD = """\
Please examine the antique item in this image and provide a detailed appraisal.

Additional context provided by the owner:
{context}

Reference data found online for similar items:
{reference_prices}

Structure your response as follows:
1. **Description**: Object type, style/period, probable origin, materials and \
construction technique.
2. **Estimated Age**: Most likely decade or period of manufacture; explain the \
visual clues that led you to this conclusion.
3. **Condition Assessment**: Visible condition issues (chips, cracks, fading, \
restorations, missing parts); overall grade (Excellent / Good / Fair / Poor).
4. **Estimated Price Range**: Realistic EUR market range, separately for:
   - Auction estimate (hammer price at a major house)
   - Retail / dealer price
   Justify the range with reference to the condition and comparable sales.
5. **Key Value Factors**: Top 3-5 factors that raise or lower the value of \
this specific piece.
6. **Confidence Level**: Low / Medium / High – explain what additional \
information would increase your confidence.

Be specific and cite observable evidence from the image for every claim."""

_USER_TEMPLATE_DEEP = """\
Please examine the antique item in this image. \
Before giving your final structured appraisal, work through the following \
reasoning steps explicitly (this "thinking" section will be shown to the user):

<thinking>
Step 1 – Object identification:
  What is the most likely object type? List alternatives and rule them out.

Step 2 – Style and period analysis:
  What stylistic features narrow the period? Consider form, decoration, \
proportion and any maker's marks.

Step 3 – Material and technique assessment:
  What materials are present? How was it made? Does the construction method \
constrain the date?

Step 4 – Condition and authenticity:
  What wear is visible? Are there signs of restoration? Does the ageing look \
consistent and genuine?

Step 5 – Market comparables:
  What comparable pieces or auction results come to mind? What price brackets \
did they achieve?

Step 6 – Synthesis:
  Combine all the above into a probability-weighted estimate of age and value.
</thinking>

After the thinking section, provide your final appraisal:

Additional context provided by the owner:
{context}

Reference data found online for similar items:
{reference_prices}

**Final Appraisal**

1. **Description**: Object type, style/period, probable origin, materials.
2. **Estimated Age**: Most likely decade or period, justified by visual evidence.
3. **Condition Assessment**: Visible issues; overall grade (Excellent/Good/Fair/Poor).
4. **Estimated Price Range**:
   - Auction estimate: …
   - Retail / dealer price: …
5. **Key Value Factors**: Top 3-5 factors raising or lowering value.
6. **Confidence Level**: Low / Medium / High – what would raise it?"""


class AntiqueAnalyzer:
    """Analyse antique images with a local Ollama vision model.

    Parameters
    ----------
    model:
        The Ollama model to use (default: ``llama3.2-vision``).
    deep_thinking:
        When ``True`` the prompt explicitly asks the model to show its
        chain-of-thought before the final answer.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        deep_thinking: bool = True,
    ) -> None:
        self.model = model
        self.deep_thinking = deep_thinking
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
            Formatted string of reference prices scraped from the web.

        Returns
        -------
        str
            The model's appraisal text.
        """
        import ollama  # imported lazily so the package loads without ollama running

        self._ensure_model(ollama)

        image_data = self._encode_image(image_path)
        template = _USER_TEMPLATE_DEEP if self.deep_thinking else _USER_TEMPLATE_STANDARD
        prompt = template.format(
            context=context.strip() or "No additional context provided.",
            reference_prices=reference_prices.strip() or "No reference prices available.",
        )
        logger.debug("Sending request to Ollama model '%s' (deep_thinking=%s)",
                     self.model, self.deep_thinking)
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
        pattern = r"(?:€|EUR\s*)?([\d,\.]+)\s*(?:–|-|to)\s*(?:€|EUR\s*)?([\d,\.]+)\s*(?:EUR|€)?"
        match = re.search(pattern, appraisal_text, re.IGNORECASE)
        if match:
            low = float(match.group(1).replace(",", ""))
            high = float(match.group(2).replace(",", ""))
            return (low, high)
        return None
