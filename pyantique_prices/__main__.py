"""Entry point: ``python -m pyantique_prices`` or ``pyantique-prices`` CLI."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pyantique-prices",
        description="Estimate the price and age of antiques from an image.",
    )
    parser.add_argument(
        "--cli",
        metavar="IMAGE_OR_FOLDER",
        help="Run in headless CLI mode with the given image path or folder.",
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
        default="minicpm-v",
        help="Ollama vision model for Pass 1 image analysis (default: minicpm-v).",
    )
    parser.add_argument(
        "--reasoning-model",
        default=None,
        help="Ollama model for Pass 2 reasoning and price estimation (default: same as --model).",
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

    if args.cli:
        return _run_cli(args)
    else:
        return _run_gui()


def _run_cli(args) -> int:
    from pathlib import Path

    from .analyzer import AntiqueAnalyzer
    from .scraper import MultiSourceScraper

    target = Path(args.cli)
    try:
        images = AntiqueAnalyzer.collect_images(target)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not images:
        print("No image files found.", file=sys.stderr)
        return 1

    scraper = MultiSourceScraper()
    analyzer = AntiqueAnalyzer(
        model=args.model,
        reasoning_model=args.reasoning_model,
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
        reasoning_model = args.reasoning_model or args.model
        print(
            f"Analysing with vision model '{args.model}' and reasoning model "
            f"'{reasoning_model}' [{mode}]…"
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
