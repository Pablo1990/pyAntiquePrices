"""Ollama-powered antique analyser."""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Vision models confirmed to work with Ollama ≥ 0.30 (new llama.cpp backend).
# Note: llama3.2-vision uses the 'mllama' architecture which is NOT supported
# by Ollama 0.30+.  The models below all use supported architectures.
RECOMMENDED_MODELS = [
    "minicpm-v",             # Best overall: compact, accurate vision+reasoning, all Ollama versions
    "llava:13b",             # LLaVA 13B – strong multimodal, widely supported
    "llava",                 # LLaVA 7B – reliable fallback, small footprint
    "moondream",             # Tiny but capable vision model
    "gemma3",                # Google Gemma 3 (vision variant) – good reasoning
    "mistral-small3.1",      # Mistral vision – capable, medium size
]

_DEFAULT_MODEL = "minicpm-v"

# -------------------------------------------------------------------------
# Prompts
# -------------------------------------------------------------------------

_KEYWORDS_PROMPT = """\
Look at the image and identify the antique or collectible object shown.
Return ONLY a comma-separated list of 5-7 concise search keywords \
(no bullet points, no sentences, no explanation) that an auction specialist \
would type into a search engine to find comparable sold items for this specific piece.
Include: object type, style/period, probable origin, main material, and any \
distinctive decorative feature visible.
Example output format: French ormolu mantel clock, Empire period, gilt bronze, \
porcelain dial, 19th century"""

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
        The Ollama model to use (default: ``minicpm-v``).  Any multimodal
        model available in your local Ollama installation can be used.
        See ``RECOMMENDED_MODELS`` for a list of models known to work with
        Ollama ≥ 0.30.
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

    def generate_search_keywords(self, image_path: str | Path) -> str:
        """Return a comma-separated keyword string derived from the image.

        Makes a quick vision-model call to identify the object and produce
        5-7 search terms suitable for querying auction-site databases.
        Returns an empty string on failure (non-fatal – appraisal continues).
        """
        import ollama  # noqa: PLC0415

        try:
            image_data = self._encode_image(image_path)
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": _KEYWORDS_PROMPT,
                        "images": [image_data],
                    }
                ],
            )
            keywords = response["message"]["content"].strip()
            # Strip any accidental leading labels like "Keywords:" or bullets
            keywords = re.sub(r"^[*\-•]?\s*keywords?:\s*", "", keywords, flags=re.IGNORECASE)
            logger.debug("Auto-generated keywords: %s", keywords)
            return keywords
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not auto-generate keywords: %s", exc)
            return ""

    def analyse(
        self,
        image_path: str | Path,
        context: str = "",
        reference_prices: str = "",
        scraper=None,
        extra_keywords: str = "",
    ) -> str:
        """Return a full appraisal string for the supplied image.

        Scraping is performed automatically: the model first identifies the
        object and generates search keywords from the image, then those
        keywords (plus any *extra_keywords* supplied by the caller) are used
        to fetch reference prices from auction sites.  Passing a pre-fetched
        *reference_prices* string bypasses the automatic scraping step.

        Parameters
        ----------
        image_path:
            Path to the image file (JPEG, PNG, WEBP, …).
        context:
            Optional free-text description the user provides about the item.
        reference_prices:
            Pre-fetched reference prices string.  When supplied, automatic
            scraping is skipped.
        scraper:
            A ``MultiSourceScraper`` instance.  When ``None`` and
            *reference_prices* is empty a new ``MultiSourceScraper`` is
            created automatically.
        extra_keywords:
            Additional keywords (e.g. from the user) appended to the
            auto-generated keywords before searching.

        Returns
        -------
        str
            The model's appraisal text.
        """
        import ollama  # imported lazily so the package loads without ollama running

        self._ensure_model(ollama)

        # ── Step 1: auto-generate keywords and scrape reference prices ──────
        if not reference_prices.strip():
            if callable(self.on_pull_progress):
                self.on_pull_progress("Identifying object for price search…")
            auto_keywords = self.generate_search_keywords(image_path)

            # Merge auto-generated + user-supplied keywords
            all_keywords_parts = [p.strip() for p in [auto_keywords, extra_keywords] if p.strip()]
            search_query = ", ".join(all_keywords_parts)

            if search_query:
                if callable(self.on_pull_progress):
                    self.on_pull_progress(f"Searching comparable prices: {search_query[:60]}…")
                logger.info("Auto-scraping with keywords: %s", search_query)
                if scraper is None:
                    from .scraper import MultiSourceScraper  # noqa: PLC0415
                    scraper = MultiSourceScraper()
                try:
                    reference_prices = scraper.get_reference_prices(search_query)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Auto-scrape failed: %s", exc)

        # ── Step 2: full appraisal ───────────────────────────────────────────
        image_data = self._encode_image(image_path)
        template = _USER_TEMPLATE_DEEP if self.deep_thinking else _USER_TEMPLATE_STANDARD
        prompt = template.format(
            context=context.strip() or "No additional context provided.",
            reference_prices=reference_prices.strip() or "No reference prices available.",
        )
        logger.debug("Sending request to Ollama model '%s' (deep_thinking=%s)",
                     self.model, self.deep_thinking)
        try:
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
        except Exception as exc:
            self._raise_if_not_vision(exc)
            raise
        return response["message"]["content"]

    @staticmethod
    def _raise_if_not_vision(exc: Exception) -> None:
        """Re-raise *exc* as a descriptive ValueError when it signals that the
        chosen model does not support multimodal / vision requests."""
        msg = str(exc).lower()
        if "multimodal" in msg or "does not support multimodal" in msg:
            vision_list = ", ".join(RECOMMENDED_MODELS)
            raise ValueError(
                f"The selected model does not support image (vision) requests.\n\n"
                f"Please choose a vision-capable model, for example:\n  {vision_list}\n\n"
                f"You can change the model in the GUI dropdown or via --model on the CLI.\n\n"
                f"Original error: {exc}"
            ) from exc

    def _ensure_model(self, ollama) -> None:
        """Pull *self.model* if it is not already available locally.

        Raises
        ------
        RuntimeError
            If the pull fails so the error is visible to the caller rather
            than being silently swallowed.
        """
        try:
            # ollama.list() returns a ListResponse object in newer SDK versions
            # and a plain dict in older ones – handle both.
            list_response = ollama.list()
            if hasattr(list_response, "models"):
                # New SDK: ListResponse with .models attribute (list of Model objects)
                local_names = [
                    getattr(m, "model", None) or getattr(m, "name", None) or str(m)
                    for m in list_response.models
                ]
            else:
                # Old SDK: plain dict {"models": [{"name": "..."}]}
                local_names = [m.get("name", "") for m in list_response.get("models", [])]
        except Exception as exc:  # noqa: BLE001
            # Ollama is not running or unreachable – let chat() surface the error
            logger.debug("Could not list local models: %s", exc)
            return

        def _base(name: str) -> str:
            return (name or "").split(":")[0].lower()

        if any(_base(n) == _base(self.model) for n in local_names):
            return  # model already present

        logger.info("Model '%s' not found locally – pulling from Ollama…", self.model)
        msg = f"Downloading model '{self.model}' – this may take a few minutes…"
        print(msg)
        if callable(self.on_pull_progress):
            self.on_pull_progress(msg)
        try:
            for progress in ollama.pull(self.model, stream=True):
                # progress may be a ProgressResponse object or a plain dict
                if hasattr(progress, "status"):
                    status = progress.status or ""
                else:
                    status = progress.get("status", "") if isinstance(progress, dict) else ""
                if status:
                    print(f"  {status}", end="\r", flush=True)
                    if callable(self.on_pull_progress):
                        self.on_pull_progress(status)
            print()  # newline after progress
            logger.info("Model '%s' pulled successfully.", self.model)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download model '{self.model}': {exc}\n"
                "Run `ollama pull " + self.model + "` manually to install it."
            ) from exc

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
