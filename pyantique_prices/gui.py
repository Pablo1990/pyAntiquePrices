"""Tkinter GUI for the multi-photo AntiqueGPT workflow."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .config import Settings
from .pricing.model import PricePredictor
from .services.appraisal import AppraisalService
from .vision.analyzer import MAX_IMAGES, MIN_IMAGES, SUPPORTED_EXTENSIONS, MultiImageAnalyzer
from .vision.marks import MarkAnalysisService
from .vision.ollama import OllamaClient

_WINDOW_TITLE = "AntiqueGPT"
_WINDOW_MIN_W = 920
_WINDOW_MIN_H = 720
_PAD = 8


class App(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title(_WINDOW_TITLE)
        self.minsize(_WINDOW_MIN_W, _WINDOW_MIN_H)
        self.resizable(True, True)
        self._settings = Settings()
        self._image_paths: list[Path] = []
        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.LabelFrame(self, text="Antique object input", padding=_PAD)
        top.pack(fill=tk.X, padx=_PAD, pady=_PAD)

        image_row = ttk.Frame(top)
        image_row.pack(fill=tk.X, pady=(0, _PAD))
        ttk.Label(image_row, text=f"Photos ({MIN_IMAGES}-{MAX_IMAGES}):").pack(side=tk.LEFT)
        self._img_var = tk.StringVar(value="No photos selected")
        ttk.Entry(image_row, textvariable=self._img_var, state="readonly", width=72).pack(
            side=tk.LEFT,
            padx=_PAD,
        )
        ttk.Button(image_row, text="Select photos…", command=self._browse_images).pack(
            side=tk.LEFT
        )
        ttk.Button(image_row, text="Clear", command=self._clear_images).pack(
            side=tk.LEFT,
            padx=(4, 0),
        )

        model_row = ttk.Frame(top)
        model_row.pack(fill=tk.X, pady=(0, _PAD))
        ttk.Label(model_row, text="Vision model:").pack(side=tk.LEFT)
        self._model_var = tk.StringVar(value=self._settings.ollama_vision_model)
        ttk.Entry(model_row, textvariable=self._model_var, width=28).pack(
            side=tk.LEFT,
            padx=_PAD,
        )
        ttk.Label(model_row, text="Currency:").pack(side=tk.LEFT)
        self._currency_var = tk.StringVar(value=self._settings.base_currency)
        ttk.Entry(model_row, textvariable=self._currency_var, width=8).pack(
            side=tk.LEFT,
            padx=(4, _PAD),
        )
        ttk.Label(model_row, text="Location:").pack(side=tk.LEFT)
        self._location_var = tk.StringVar()
        ttk.Entry(model_row, textvariable=self._location_var, width=20).pack(
            side=tk.LEFT,
            padx=(4, 0),
        )

        dim_row = ttk.Frame(top)
        dim_row.pack(fill=tk.X, pady=(0, _PAD))
        ttk.Label(dim_row, text="Known dimensions:").pack(side=tk.LEFT)
        self._dimensions_var = tk.StringVar()
        ttk.Entry(dim_row, textvariable=self._dimensions_var, width=40).pack(
            side=tk.LEFT,
            padx=_PAD,
        )
        ttk.Label(dim_row, text="Provenance:").pack(side=tk.LEFT)
        self._provenance_var = tk.StringVar()
        ttk.Entry(dim_row, textvariable=self._provenance_var, width=30).pack(
            side=tk.LEFT,
            padx=(4, 0),
        )

        ttk.Label(top, text="Description / context:").pack(anchor=tk.W)
        self._context_text = scrolledtext.ScrolledText(top, height=4, wrap=tk.WORD)
        self._context_text.pack(fill=tk.X, pady=(2, 0))

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=_PAD)
        self._analyse_btn = ttk.Button(
            btn_frame,
            text="Analyze object",
            command=self._start_analysis,
        )
        self._analyse_btn.pack(side=tk.LEFT)
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(btn_frame, textvariable=self._status_var, foreground="grey").pack(
            side=tk.LEFT,
            padx=_PAD,
        )

        self._progress = ttk.Progressbar(self, mode="indeterminate")
        self._progress.pack(fill=tk.X, padx=_PAD, pady=(2, 0))

        result_frame = ttk.LabelFrame(self, text="Appraisal", padding=_PAD)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=_PAD, pady=_PAD)
        self._result_text = scrolledtext.ScrolledText(
            result_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self._result_text.pack(fill=tk.BOTH, expand=True)

    def _browse_images(self) -> None:
        paths = filedialog.askopenfilenames(
            title=f"Select {MIN_IMAGES}-{MAX_IMAGES} antique photos",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp"), ("All files", "*.*")],
        )
        if not paths:
            return
        selected = [Path(path) for path in paths]
        self._image_paths = selected
        label = ", ".join(path.name for path in selected[:3])
        if len(selected) > 3:
            label = f"{label}, … ({len(selected)} selected)"
        self._img_var.set(label)

    def _clear_images(self) -> None:
        self._image_paths = []
        self._img_var.set("No photos selected")

    def _start_analysis(self) -> None:
        if len(self._image_paths) < MIN_IMAGES or len(self._image_paths) > MAX_IMAGES:
            messagebox.showwarning(
                "Invalid photo count",
                f"Please select between {MIN_IMAGES} and {MAX_IMAGES} photos.",
            )
            return

        for path in self._image_paths:
            if not path.exists():
                messagebox.showerror("Not found", f"Cannot find:\n{path}")
                return
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                messagebox.showwarning(
                    "Unsupported format",
                    "Supported formats: JPEG, PNG, WebP.",
                )
                return

        model = self._model_var.get().strip() or self._settings.ollama_vision_model
        currency = (self._currency_var.get().strip() or self._settings.base_currency).upper()
        context = self._context_text.get("1.0", tk.END).strip()
        location = self._location_var.get().strip()
        dimensions = self._dimensions_var.get().strip()
        provenance = self._provenance_var.get().strip()

        self._analyse_btn.config(state=tk.DISABLED)
        self._progress.start(8)
        self._set_status("Analyzing antique object…")
        self._set_result("")

        thread = threading.Thread(
            target=self._run_analysis,
            args=(list(self._image_paths), model, currency, context, location, dimensions, provenance),
            daemon=True,
        )
        thread.start()

    def _run_analysis(
        self,
        image_paths: list[Path],
        model: str,
        currency: str,
        context: str,
        location: str,
        dimensions: str,
        provenance: str,
    ) -> None:
        try:
            settings = Settings()
            session_factory = None
            save_appraisal_fn = None
            db_warning = None
            try:
                from .data.appraisals import save_appraisal as _save_appraisal
                from .data.database import create_tables, get_engine, get_session_factory

                engine = get_engine(settings.database_url)
                create_tables(engine)
                session_factory = get_session_factory(engine)
                save_appraisal_fn = _save_appraisal
            except ModuleNotFoundError as exc:
                if exc.name == "sqlalchemy":
                    db_warning = (
                        "SQLAlchemy is not installed; running without local sales "
                        "database, comparable retrieval, and appraisal persistence."
                    )
                else:
                    raise
            client = OllamaClient(host=settings.ollama_host, model=model)
            analyzer = MultiImageAnalyzer(client=client, mark_service=MarkAnalysisService())
            pricer = PricePredictor(
                min_comparables_for_model=settings.min_comparables_for_model,
                min_comparables_for_confidence=settings.min_comparables_for_confidence,
            )
            service = AppraisalService(
                analyzer=analyzer,
                retrieval_session_factory=session_factory,
                pricer=pricer,
                base_currency=settings.base_currency,
                min_comparables_for_model=settings.min_comparables_for_model,
                min_comparables_for_confidence=settings.min_comparables_for_confidence,
                top_k_comparables=settings.top_k_comparables,
                min_similarity=settings.min_similarity,
                max_sale_age_years=settings.max_sale_age_years,
                min_data_quality_score=settings.min_data_quality_score,
            )
            full_context = _build_context(
                context=context,
                location=location,
                known_dimensions=dimensions,
                provenance=provenance,
            )
            result = service.appraise(image_paths, context=full_context, currency=currency)
            if db_warning:
                result.setdefault("warnings", []).append(db_warning)

            if session_factory and save_appraisal_fn:
                with session_factory() as session:
                    save_appraisal_fn(
                        session=session,
                        result=result,
                        input_metadata={
                            "currency": currency,
                            "location": location or None,
                            "known_dimensions": dimensions or None,
                            "provenance": provenance or None,
                            "user_description": context or None,
                            "num_images": len(image_paths),
                            "source_images": [str(path) for path in image_paths],
                        },
                        model_versions={
                            "vision_model": model,
                            "pricing_model": result.get("valuation", {}).get("method", "unknown"),
                        },
                    )

            formatted = _format_appraisal(result)
            self._after_safe(self._on_analysis_done, formatted)
        except Exception as exc:  # noqa: BLE001
            self._after_safe(self._on_analysis_error, str(exc))

    def _on_analysis_done(self, result: str) -> None:
        self._progress.stop()
        self._set_result(result)
        self._set_status("Analysis complete.")
        self._analyse_btn.config(state=tk.NORMAL)

    def _on_analysis_error(self, message: str) -> None:
        self._progress.stop()
        self._set_result(f"Error:\n{message}")
        self._set_status("Analysis failed.")
        self._analyse_btn.config(state=tk.NORMAL)
        messagebox.showerror("Analysis error", message)

    def _set_status(self, text: str) -> None:
        self._status_var.set(text)

    def _set_result(self, text: str) -> None:
        self._result_text.config(state=tk.NORMAL)
        self._result_text.delete("1.0", tk.END)
        self._result_text.insert(tk.END, text)
        self._result_text.config(state=tk.DISABLED)

    def _after_safe(self, func, *args) -> None:
        self.after(0, func, *args)


def _build_context(
    *,
    context: str,
    location: str,
    known_dimensions: str,
    provenance: str,
) -> str:
    parts = []
    if context:
        parts.append(f"Description: {context}")
    if location:
        parts.append(f"Location: {location}")
    if known_dimensions:
        parts.append(f"Known dimensions: {known_dimensions}")
    if provenance:
        parts.append(f"Provenance: {provenance}")
    return "\n".join(parts)


def _extract_value(field):
    if isinstance(field, dict):
        return field.get("value")
    return field


def _format_appraisal(result: dict) -> str:
    identification = result.get("identification") or {}
    valuation = result.get("valuation") or {}
    marks = identification.get("marks") or []
    manufacturer_candidates = identification.get("manufacturer_candidates") or []
    manufacturers = ", ".join(
        candidate.get("name", "")
        for candidate in manufacturer_candidates
        if isinstance(candidate, dict) and candidate.get("name")
    ) or "N/A"

    lines = []
    lines.append("IDENTIFICATION")
    lines.append("-" * 43)
    lines.append(f"Object: {_extract_value(identification.get('object_type')) or 'N/A'}")
    lines.append(f"Period: {_extract_value(identification.get('likely_period')) or 'N/A'}")
    lines.append(f"Manufacturer candidates: {manufacturers}")
    lines.append(
        f"Materials: {', '.join(identification.get('materials', [])) or 'N/A'}"
    )
    lines.append(f"Condition: {_extract_value(identification.get('condition')) or 'N/A'}")
    lines.append("")

    lines.append("MARKS")
    lines.append("-" * 43)
    if marks:
        for mark in marks:
            lines.append(
                f"- {mark.get('text') or 'N/A'} | type={mark.get('mark_type') or 'N/A'} | "
                f"confidence={mark.get('confidence', 0.0):.2f} | "
                f"candidates={', '.join(mark.get('manufacturer_candidates', [])) or 'N/A'}"
            )
    else:
        lines.append("No marks detected.")
    lines.append("")

    lines.append("COMPARABLE SALES")
    lines.append("-" * 43)
    lines.append(
        f"Candidates: {result.get('candidate_count', 0)} | "
        f"Usable: {result.get('usable_comparable_count', 0)}"
    )
    comparables = result.get("comparables", [])
    for comparable in comparables[:10]:
        lines.append(
            f"- {comparable.get('title') or 'Untitled'} | "
            f"{comparable.get('normalized_price')} {result.get('currency', 'EUR')} | "
            f"score={comparable.get('retrieval_score', 0.0):.3f}"
        )
    lines.append("")

    lines.append("VALUATION")
    lines.append("-" * 43)
    if valuation and result.get("valuation_available"):
        lines.append(
            f"Estimated market value: {result.get('currency', 'EUR')} "
            f"{valuation.get('low')} – {valuation.get('high')}"
        )
        lines.append(f"Midpoint (P50): {valuation.get('mid')}")
    elif valuation:
        lines.append(
            f"Reference-only estimate: {result.get('currency', 'EUR')} "
            f"{valuation.get('low')} – {valuation.get('high')}"
        )
    else:
        lines.append("No valuation available.")
    lines.append("")

    lines.append("CONFIDENCE")
    lines.append("-" * 43)
    lines.append(
        f"Identification confidence: {result.get('identification_confidence', 0.0) * 100:.0f}%"
    )
    lines.append(
        f"Valuation confidence: {result.get('valuation_confidence', 0.0) * 100:.0f}%"
    )
    lines.append("")

    lines.append("WARNINGS")
    lines.append("-" * 43)
    warnings = result.get("warnings", [])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("None.")
    return "\n".join(lines)


def run_gui() -> None:
    app = App()
    app.mainloop()
