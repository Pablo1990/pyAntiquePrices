"""Entry point: ``python -m pyantique_prices`` or ``pyantique-prices`` CLI."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    from .config import Settings

    settings = Settings()
    parser = argparse.ArgumentParser(
        prog="pyantique-prices",
        description="Estimate antique identification and value from 3-5 photos.",
    )
    parser.add_argument(
        "--cli",
        metavar="IMAGE_OR_FOLDER",
        help=(
            "Run in headless CLI mode with a single image, or a folder containing "
            "3-5 photos of one object for the AntiqueGPT workflow."
        ),
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Optional 3-5 image paths for the AntiqueGPT object workflow.",
    )
    parser.add_argument(
        "--context",
        default="",
        help="Free-text context about the item (CLI mode only).",
    )
    parser.add_argument(
        "--keywords",
        default="",
        help="Extra search keywords to supplement the auto-generated ones (CLI mode only).",
    )
    parser.add_argument(
        "--model",
        default=settings.ollama_vision_model,
        help=(
            "Ollama vision model for image analysis "
            f"(default: {settings.ollama_vision_model})."
        ),
    )
    parser.add_argument(
        "--reasoning-model",
        default=None,
        help=(
            "Pass-2 reasoning model. For Ollama, this is the Ollama model name. "
            "For Hugging Face, this is the base model repo id."
        ),
    )
    parser.add_argument(
        "--reasoning-backend",
        choices=("ollama", "huggingface"),
        default="ollama",
        help="Pass-2 backend for reasoning and price estimation (default: ollama).",
    )
    parser.add_argument(
        "--reasoning-adapter",
        default=None,
        help=(
            "Optional Hugging Face PEFT adapter repo id for Pass 2, "
            "for example jordanmatsumoto/pricing-specialist."
        ),
    )
    parser.add_argument(
        "--deep-thinking",
        action="store_true",
        default=True,
        help="Enable chain-of-thought reasoning prompt (default: on).",
    )
    parser.add_argument(
        "--no-deep-thinking",
        dest="deep_thinking",
        action="store_false",
        help="Disable chain-of-thought reasoning (faster, less accurate).",
    )
    args = parser.parse_args(argv)

    if args.cli or args.images:
        return _run_cli(args)
    else:
        return _run_gui()


def _run_cli(args) -> int:
    from pathlib import Path

    from .analyzer import AntiqueAnalyzer

    if (args.reasoning_backend == "huggingface" or args.reasoning_adapter) and not args.reasoning_model:
        print(
            "Error: --reasoning-model is required when using --reasoning-backend huggingface "
            "or --reasoning-adapter.",
            file=sys.stderr,
        )
        return 1

    if args.images:
        images = [Path(image) for image in args.images]
        target = images[0]
        for image in images:
            if not image.exists():
                print(f"Error: Image not found: {image}", file=sys.stderr)
                return 1
    else:
        target = Path(args.cli)
        try:
            images = AntiqueAnalyzer.collect_images(target)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    if not images:
        print("No image files found.", file=sys.stderr)
        return 1

    if _should_use_object_workflow(target, images):
        return _run_object_cli(args, images)

    print(
        "Using legacy single-image workflow. For the new AntiqueGPT object workflow, "
        "pass a folder with 3-5 photos of the same object.",
        file=sys.stderr,
    )
    from .scraper import MultiSourceScraper

    scraper = MultiSourceScraper()
    analyzer = AntiqueAnalyzer(
        model=args.model,
        reasoning_model=args.reasoning_model,
        reasoning_backend=args.reasoning_backend,
        reasoning_adapter=args.reasoning_adapter,
        deep_thinking=args.deep_thinking,
    )

    def _on_progress(status: str) -> None:
        print(f"  {status}", end="\r", flush=True)

    analyzer.on_pull_progress = _on_progress

    mode = "deep thinking" if args.deep_thinking else "standard"
    total = len(images)

    for idx, img_path in enumerate(images, 1):
        print(f"\n{'='*60}")
        print(f"Image {idx}/{total}: {img_path}")
        print(f"{'='*60}")
        reasoning_backend = args.reasoning_backend
        if args.reasoning_adapter and reasoning_backend == "ollama":
            reasoning_backend = "huggingface"
        reasoning_model = args.reasoning_model or args.model
        print(
            f"Analysing with vision model '{args.model}' and {reasoning_backend} "
            f"reasoning model '{reasoning_model}' [{mode}]…"
        )
        print(f"  Pass 1 – identifying object…", end="\r", flush=True)
        try:
            result = analyzer.analyse(
                img_path,
                context=args.context,
                extra_keywords=args.keywords,
                scraper=scraper,
            )
            print()  # clear progress line
            print("\n--- APPRAISAL ---\n")
            print(result)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {exc}", file=sys.stderr)

    return 0


def _should_use_object_workflow(target, images) -> bool:
    return bool((target.is_dir() or len(images) > 1) and 3 <= len(images) <= 5)


def _run_object_cli(args, images) -> int:
    from .config import Settings
    from .embeddings import NullImageEmbeddingProvider, OllamaTextEmbeddingProvider
    from .services.appraisal import AppraisalService, LegacyWebFallbackEstimator
    from .vision.analyzer import MultiImageAnalyzer
    from .vision.marks import MarkAnalysisService
    from .vision.ollama import OllamaClient

    settings = Settings()
    session_factory = None
    db_warning = None
    pricing_warning = None

    try:
        from .data.database import create_tables, get_engine, get_session_factory

        engine = get_engine(settings.database_url)
        create_tables(engine)
        session_factory = get_session_factory(engine)
    except ModuleNotFoundError as exc:
        if exc.name == "sqlalchemy":
            db_warning = (
                "SQLAlchemy is not installed; running without local sales retrieval "
                "or appraisal persistence."
            )
        else:
            raise

    pricer = None
    try:
        from .pricing.model import PricePredictor

        pricer = PricePredictor(
            min_comparables_for_model=settings.min_comparables_for_model,
            min_comparables_for_confidence=settings.min_comparables_for_confidence,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            pricing_warning = (
                "NumPy is not installed; running without the numerical pricing model."
            )
        else:
            raise

    client = OllamaClient(
        host=settings.ollama_host,
        model=args.model,
        num_ctx=settings.ollama_num_ctx,
    )
    analyzer = MultiImageAnalyzer(client=client, mark_service=MarkAnalysisService())
    text_embedding_provider = OllamaTextEmbeddingProvider(
        host=settings.ollama_host,
        model=settings.ollama_embed_model,
        num_ctx=settings.ollama_num_ctx,
        require_model=False,
    )
    service = AppraisalService(
        analyzer=analyzer,
        retrieval_session_factory=session_factory,
        text_embedding_provider=text_embedding_provider,
        image_embedding_provider=NullImageEmbeddingProvider(),
        pricer=pricer,
        fallback_estimator=LegacyWebFallbackEstimator(model=args.model),
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

    context = args.context.strip()
    if args.keywords.strip():
        keyword_context = f"Extra keywords: {args.keywords.strip()}"
        context = f"{context}\n{keyword_context}".strip() if context else keyword_context

    if args.reasoning_model or args.reasoning_adapter or args.reasoning_backend != "ollama":
        print(
            "Note: the AntiqueGPT object workflow uses structured vision analysis plus "
            "a numerical pricing pipeline; reasoning-model CLI options are ignored.",
            file=sys.stderr,
        )

    print(
        f"\n{'='*60}\n"
        f"AntiqueGPT object workflow ({len(images)} photos)\n"
        f"{'='*60}\n"
        f"Analysing with vision model '{args.model}'...",
    )
    result = service.appraise(images, context=context, currency=settings.base_currency)
    if db_warning:
        result.setdefault("warnings", []).append(db_warning)
    if pricing_warning:
        result.setdefault("warnings", []).append(pricing_warning)
    print(_format_object_result(result))
    return 0


def _extract_value(field):
    if isinstance(field, dict):
        return field.get("value")
    return field


def _format_candidates(candidates) -> str:
    if not candidates:
        return "N/A"
    lines = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate.get("name"):
            continue
        confidence = float(candidate.get("confidence", 0.0) or 0.0)
        evidence = candidate.get("evidence")
        chunk = f"{candidate['name']} ({confidence * 100:.0f}%)"
        if evidence:
            chunk += f" – {evidence}"
        lines.append(chunk)
    return "; ".join(lines) or "N/A"


def _format_object_result(result: dict) -> str:
    identification = result.get("identification") or {}
    valuation = result.get("valuation") or {}
    marks = identification.get("marks") or []

    lines = ["", "--- APPRAISAL ---", ""]
    lines.append("OBJECT IDENTIFICATION")
    lines.append(f"Object: {_extract_value(identification.get('object_type')) or 'N/A'}")
    lines.append(
        f"Period: {_extract_value(identification.get('period') or identification.get('likely_period')) or 'N/A'}"
    )
    lines.append(
        f"Manufacturer candidates: {_format_candidates(identification.get('manufacturer_candidates'))}"
    )
    lines.append(
        f"Artist candidates: {_format_candidates(identification.get('artist_candidates'))}"
    )
    lines.append(
        f"Materials: {', '.join(identification.get('materials', [])) or 'N/A'}"
    )
    lines.append(
        f"Condition: {_extract_value(identification.get('condition')) or 'N/A'}"
    )
    lines.append("")
    lines.append("MAKER MARKS")
    if marks:
        for mark in marks:
            candidates = _format_candidates(mark.get("manufacturer_candidates"))
            lines.append(
                f"- Mark: {mark.get('text') or 'Unreadable'} | "
                f"Confidence: {mark.get('confidence', 0.0):.2f} | "
                f"Evidence: {mark.get('evidence') or 'N/A'} | "
                f"Candidates: {candidates}"
            )
    else:
        lines.append("No marks detected.")
    lines.append("")
    lines.append("TOP COMPARABLES")
    lines.append(
        f"Candidates: {result.get('candidate_count', 0)} | "
        f"Usable: {result.get('usable_comparable_count', 0)}"
    )
    for comparable in result.get("comparables", [])[:10]:
        lines.append(
            f"- Auction: {comparable.get('auction_house') or 'Unknown'} | "
            f"Date: {comparable.get('sale_date') or 'N/A'} | "
            f"Object: {comparable.get('title') or 'Untitled'} | "
            f"Similarity: {comparable.get('overall_similarity', comparable.get('retrieval_score', 0.0)):.3f} | "
            f"Price: {comparable.get('normalized_price')} {result.get('currency', 'EUR')}"
        )
    lines.append("")
    lines.append("LEGACY PRICE ESTIMATE")
    if valuation:
        label = (
            "Estimated market value"
            if result.get("valuation_available")
            else "Reference-only estimate"
        )
        lines.append(
            f"{label}: {result.get('currency', 'EUR')} "
            f"{valuation.get('low')} – {valuation.get('high')}"
        )
        if valuation.get("mid") is not None:
            lines.append(f"Midpoint (P50): {valuation.get('mid')}")
    else:
        lines.append("No valuation available.")
    lines.append("")
    lines.append("CONFIDENCE")
    lines.append(
        f"Identification confidence: {result.get('identification_confidence', 0.0) * 100:.0f}%"
    )
    lines.append(
        f"Valuation confidence: {result.get('valuation_confidence', 0.0) * 100:.0f}%"
    )
    lines.append("")
    lines.append("WARNINGS")
    warnings = result.get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("None.")
    return "\n".join(lines)


def _run_gui() -> int:
    try:
        from .gui import run_gui

        run_gui()
        return 0
    except ImportError as exc:
        print(
            f"GUI unavailable ({exc}). Use --cli <image> for headless mode.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
