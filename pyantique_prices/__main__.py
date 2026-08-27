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
        help="Search keywords for reference prices (CLI mode only).",
    )
    parser.add_argument(
        "--model",
        default="llava",
        help="Ollama model to use (default: llava).",
    )
    args = parser.parse_args(argv)

    if args.cli:
        return _run_cli(args)
    else:
        return _run_gui()


def _run_cli(args) -> int:
    from pathlib import Path

    from .analyzer import AntiqueAnalyzer
    from .scraper import TodoColeccionScraper

    target = Path(args.cli)
    try:
        images = AntiqueAnalyzer.collect_images(target)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not images:
        print("No image files found.", file=sys.stderr)
        return 1

    scraper = TodoColeccionScraper()
    reference_prices = ""
    if args.keywords:
        print(f"Fetching reference prices for: {args.keywords}")
        try:
            reference_prices = scraper.get_reference_prices(args.keywords)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not fetch reference prices: {exc}", file=sys.stderr)

    analyzer = AntiqueAnalyzer(model=args.model)
    total = len(images)

    for idx, img_path in enumerate(images, 1):
        print(f"\n{'='*60}")
        print(f"Image {idx}/{total}: {img_path}")
        print(f"{'='*60}")
        print(f"Analysing with model '{args.model}'…")
        try:
            result = analyzer.analyse(
                img_path, context=args.context, reference_prices=reference_prices
            )
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
