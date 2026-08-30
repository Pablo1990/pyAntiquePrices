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

_PASS1_PROMPT = """\
You are an expert antique appraiser. Examine the image carefully, together with \
any context the owner has provided.

Your task has two parts — answer BOTH, separated by "---":

PART A – Identification (3-5 sentences):
Identify the object: type, probable origin, approximate period/style, \
visible materials and any distinctive features or marks. \
Take into account the owner's context when refining your identification. \
Be specific — "18th-century Chinese blue-and-white export porcelain bowl" \
not "a bowl".

PART B – Search keywords (one line, comma-separated, no explanation):
List 6-8 auction-specialist search keywords that would find the most \
comparable sold items on platforms such as Catawiki, LiveAuctioneers or \
Invaluable. Derive these from your identification above PLUS the owner's \
context. Include: object type, cultural origin, period/style, main material, \
distinctive feature, and (if relevant) any maker or school.
Example: Chinese blue and white porcelain bowl, Kangxi period, export ware, \
floral medallion, 18th century, Qing dynasty"""

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
  Review the reference data found online (shown below). How do those \
comparable pieces compare to this item in terms of quality, rarity and \
condition? Are the prices consistent with your initial assessment? \
Revise your estimate if the data suggests a different range.

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

    def _pass1_identify(self, image_path: str | Path, context: str = "") -> tuple[str, str]:
        """Pass 1 of the two-pass pipeline.

        Sends the image and owner context to the model with a lightweight
        prompt that asks for:
          - A concise object identification (3-5 sentences)
          - A comma-separated list of auction-specialist search keywords

        Returns
        -------
        (identification, keywords) tuple.
        Both are empty strings on failure (non-fatal – pipeline continues).
        """
        import ollama  # noqa: PLC0415

        try:
            image_data = self._encode_image(image_path)
            user_content = _PASS1_PROMPT
            if context.strip():
                user_content = (
                    f"Owner's context: {context.strip()}\n\n" + _PASS1_PROMPT
                )
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": user_content,
                        "images": [image_data],
                    },
                ],
            )
            raw = response["message"]["content"].strip()
            logger.debug("Pass-1 raw output: %s", raw[:200])

            # Split on the separator "---"
            parts = re.split(r"\n\s*---\s*\n", raw, maxsplit=1)
            identification = parts[0].strip()
            keywords_raw = parts[1].strip() if len(parts) > 1 else raw

            # Extract just the keywords line (last non-empty line of part B,
            # or the whole part if it is a single line)
            kw_lines = [ln.strip() for ln in keywords_raw.splitlines() if ln.strip()]
            # Drop any header like "PART B" or "Keywords:"
            kw_lines = [
                re.sub(r"^[*\-•]?\s*(part\s+b|keywords?)\s*[-–:]\s*", "", ln, flags=re.IGNORECASE)
                for ln in kw_lines
            ]
            kw_lines = [ln for ln in kw_lines if ln]
            keywords = kw_lines[-1] if kw_lines else ""

            logger.debug("Pass-1 identification: %s", identification[:100])
            logger.debug("Pass-1 keywords: %s", keywords)
            return identification, keywords
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pass-1 identification failed: %s", exc)
            return "", ""

    def generate_search_keywords(self, image_path: str | Path, context: str = "") -> str:
        """Return a comma-separated keyword string derived from the image.

        Convenience wrapper around :meth:`_pass1_identify` that returns only
        the keyword portion.  Retained for backward compatibility.
        """
        _, keywords = self._pass1_identify(image_path, context=context)
        return keywords

    def analyse(
        self,
        image_path: str | Path,
        context: str = "",
        reference_prices: str = "",
        scraper=None,
        extra_keywords: str = "",
    ) -> str:
        """Return a full appraisal string for the supplied image.

        Two-pass pipeline
        -----------------
        **Pass 1** (fast): the model sees the image and the owner's context and
        produces a concise identification plus targeted auction search keywords.

        **Scraping**: those keywords (merged with any user-supplied
        *extra_keywords*) are used to fetch comparable prices from auction
        sites via DuckDuckGo.

        **Pass 2** (deep): the model re-examines the image with the scraped
        prices injected into the prompt.  It explicitly compares its initial
        identification against the market data and revises if needed, then
        produces the final structured appraisal.

        Passing a pre-fetched *reference_prices* string bypasses Passes 1 and
        the scraping step, going straight to Pass 2.

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
            A ``MultiSourceScraper`` instance.  When ``None`` a new one is
            created automatically.
        extra_keywords:
            Additional keywords (e.g. from the user) appended to the
            Pass-1-generated keywords before searching.

        Returns
        -------
        str
            The model's appraisal text, prefixed with the Pass-1
            identification block.
        """
        import ollama  # imported lazily so the package loads without ollama running

        self._ensure_model(ollama)

        # ── Pass 1: identify + generate search keywords ──────────────────────
        identification = ""
        if not reference_prices.strip():
            if callable(self.on_pull_progress):
                self.on_pull_progress("Pass 1 – identifying object and generating search terms…")
            identification, auto_keywords = self._pass1_identify(image_path, context=context)

            # Merge Pass-1 keywords + user-supplied extra keywords
            all_keywords_parts = [p.strip() for p in [auto_keywords, extra_keywords] if p.strip()]
            search_query = ", ".join(all_keywords_parts)

            if search_query:
                if callable(self.on_pull_progress):
                    self.on_pull_progress(f"Scraping comparable prices: {search_query[:60]}…")
                logger.info("Scraping with keywords: %s", search_query)
                if scraper is None:
                    from .scraper import MultiSourceScraper  # noqa: PLC0415
                    scraper = MultiSourceScraper()
                try:
                    reference_prices = scraper.get_reference_prices(search_query)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Scrape failed: %s", exc)

        # ── Pass 2: full deep-thinking appraisal with scraped prices ─────────
        if callable(self.on_pull_progress):
            self.on_pull_progress("Pass 2 – deep-thinking appraisal with market data…")

        # Enrich context with the Pass-1 identification so Pass 2 can build on it
        enriched_context = context.strip()
        if identification:
            sep = "\n\n" if enriched_context else ""
            enriched_context = (
                f"[Initial identification from visual analysis]\n{identification}"
                f"{sep}{enriched_context}"
            )

        image_data = self._encode_image(image_path)
        template = _USER_TEMPLATE_DEEP if self.deep_thinking else _USER_TEMPLATE_STANDARD
        prompt = template.format(
            context=enriched_context or "No additional context provided.",
            reference_prices=reference_prices.strip() or "No reference prices available.",
        )
        logger.debug("Pass-2 request to model '%s' (deep_thinking=%s)",
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
