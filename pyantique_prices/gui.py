"""Minimal Tkinter GUI for pyAntiquePrices."""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

from .analyzer import AntiqueAnalyzer, RECOMMENDED_MODELS
from .scraper import MultiSourceScraper

logger = logging.getLogger(__name__)

_WINDOW_TITLE = "pyAntiquePrices – Antique Appraiser"
_WINDOW_MIN_W = 820
_WINDOW_MIN_H = 680
_PAD = 8


class App(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title(_WINDOW_TITLE)
        self.minsize(_WINDOW_MIN_W, _WINDOW_MIN_H)
        self.resizable(True, True)

        self._image_path: Optional[Path] = None
        self._analyzer = AntiqueAnalyzer()
        self._scraper = MultiSourceScraper()

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ---- Top frame: image selector + options ----
        top = ttk.LabelFrame(self, text="Item details", padding=_PAD)
        top.pack(fill=tk.X, padx=_PAD, pady=_PAD)

        # Image / folder row
        img_row = ttk.Frame(top)
        img_row.pack(fill=tk.X, pady=(0, _PAD))
        ttk.Label(img_row, text="Image / folder:").pack(side=tk.LEFT)
        self._img_var = tk.StringVar(value="No file or folder selected")
        ttk.Entry(img_row, textvariable=self._img_var, state="readonly", width=48).pack(
            side=tk.LEFT, padx=_PAD
        )
        ttk.Button(img_row, text="Image…", command=self._browse_image).pack(side=tk.LEFT)
        ttk.Button(img_row, text="Folder…", command=self._browse_folder).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        # Model row – combobox with recommended models + free-text
        model_row = ttk.Frame(top)
        model_row.pack(fill=tk.X, pady=(0, _PAD))
        ttk.Label(model_row, text="Ollama model:").pack(side=tk.LEFT)
        self._model_var = tk.StringVar(value=RECOMMENDED_MODELS[0])
        model_cb = ttk.Combobox(
            model_row,
            textvariable=self._model_var,
            values=RECOMMENDED_MODELS,
            width=28,
        )
        model_cb.pack(side=tk.LEFT, padx=_PAD)
        ttk.Label(
            model_row,
            text="(or type any Ollama model name)",
            foreground="grey",
        ).pack(side=tk.LEFT)

        # Options row: deep thinking checkbox
        opts_row = ttk.Frame(top)
        opts_row.pack(fill=tk.X, pady=(0, _PAD))
        self._deep_thinking_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opts_row,
            text="Deep thinking  (chain-of-thought reasoning – slower but more accurate)",
            variable=self._deep_thinking_var,
        ).pack(side=tk.LEFT)

        # Search keywords row
        kw_row = ttk.Frame(top)
        kw_row.pack(fill=tk.X, pady=(0, _PAD))
        ttk.Label(kw_row, text="Extra search keywords (optional):").pack(side=tk.LEFT)
        self._keywords_var = tk.StringVar()
        ttk.Entry(kw_row, textvariable=self._keywords_var, width=48).pack(
            side=tk.LEFT, padx=_PAD
        )
        ttk.Label(
            kw_row,
            text="Keywords are auto-generated from the image; add extras here if needed",
            foreground="grey",
        ).pack(side=tk.LEFT)

        # Context area
        ttk.Label(top, text="Additional context / description:").pack(anchor=tk.W)
        self._context_text = scrolledtext.ScrolledText(top, height=4, wrap=tk.WORD)
        self._context_text.pack(fill=tk.X, pady=(2, 0))

        # ---- Action button ----
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=_PAD)
        self._analyse_btn = ttk.Button(
            btn_frame,
            text="Analyse antique",
            command=self._start_analysis,
        )
        self._analyse_btn.pack(side=tk.LEFT)
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(btn_frame, textvariable=self._status_var, foreground="grey").pack(
            side=tk.LEFT, padx=_PAD
        )

        # ---- Progress bar ----
        self._progress = ttk.Progressbar(self, mode="determinate")
        self._progress.pack(fill=tk.X, padx=_PAD, pady=(2, 0))

        # ---- Result area ----
        result_frame = ttk.LabelFrame(self, text="Appraisal", padding=_PAD)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=_PAD, pady=_PAD)
        self._result_text = scrolledtext.ScrolledText(
            result_frame, wrap=tk.WORD, state=tk.DISABLED
        )
        self._result_text.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _browse_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select antique image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.gif"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._image_path = Path(path)
            self._img_var.set(str(self._image_path))

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(title="Select folder with antique images")
        if path:
            self._image_path = Path(path)
            self._img_var.set(str(self._image_path))

    def _start_analysis(self) -> None:
        if self._image_path is None:
            messagebox.showwarning("No selection", "Please select an image or folder first.")
            return

        if not self._image_path.exists():
            messagebox.showerror("Not found", f"Cannot find:\n{self._image_path}")
            return

        model = self._model_var.get().strip()
        if not model:
            messagebox.showwarning("No model", "Please enter an Ollama model name.")
            return

        # Warn if the selected model is not in the recommended vision list
        if model not in RECOMMENDED_MODELS:
            proceed = messagebox.askyesno(
                "Vision support warning",
                f"'{model}' is not in the list of known vision-capable models.\n\n"
                f"This tool requires a model that can process images. "
                f"If '{model}' does not support vision, the analysis will fail.\n\n"
                f"Recommended models: {', '.join(RECOMMENDED_MODELS)}\n\n"
                "Continue anyway?",
            )
            if not proceed:
                return

        try:
            images = AntiqueAnalyzer.collect_images(self._image_path)
        except FileNotFoundError as exc:
            messagebox.showerror("Error", str(exc))
            return

        if not images:
            messagebox.showwarning("No images", "No image files found in the selected folder.")
            return

        # Capture all Tkinter values on the main thread before handing to worker
        context = self._context_text.get("1.0", tk.END).strip()
        keywords = self._keywords_var.get().strip()
        deep_thinking = self._deep_thinking_var.get()

        self._analyse_btn.config(state=tk.DISABLED)
        self._progress["value"] = 0
        self._progress["maximum"] = len(images)
        self._set_status(f"Starting analysis of {len(images)} image(s)…")
        self._set_result("")

        thread = threading.Thread(
            target=self._run_analysis,
            args=(images, model, context, keywords, deep_thinking),
            daemon=True,
        )
        thread.start()

    def _run_analysis(
        self,
        images: list[Path],
        model: str,
        context: str,
        keywords: str,
        deep_thinking: bool,
    ) -> None:
        """Background worker – must not touch Tk widgets directly."""
        self._analyzer.model = model
        self._analyzer.deep_thinking = deep_thinking
        self._analyzer.on_pull_progress = self._on_pull_progress

        total = len(images)
        all_results: list[str] = []

        for idx, img_path in enumerate(images, 1):
            mode = "deep thinking" if deep_thinking else "standard"
            self._after_safe(
                self._set_status,
                f"[{mode}] Analysing image {idx}/{total}: {img_path.name}…",
            )
            try:
                result = self._analyzer.analyse(
                    img_path,
                    context=context,
                    extra_keywords=keywords,
                    scraper=self._scraper,
                )
                all_results.append(
                    f"{'='*60}\n"
                    f"Image {idx}/{total}: {img_path.name}\n"
                    f"{'='*60}\n"
                    f"{result}\n"
                )
            except Exception as exc:  # noqa: BLE001
                all_results.append(
                    f"{'='*60}\n"
                    f"Image {idx}/{total}: {img_path.name}  [ERROR]\n"
                    f"{'='*60}\n"
                    f"{exc}\n"
                )
            self._after_safe(self._set_progress, idx)

        self._after_safe(self._on_analysis_done, "\n".join(all_results))

    def _on_analysis_done(self, result: str) -> None:
        self._set_result(result)
        self._set_status("Analysis complete.")
        self._analyse_btn.config(state=tk.NORMAL)

    def _on_analysis_error(self, message: str) -> None:
        self._set_result(f"Error:\n{message}")
        self._set_status("Analysis failed.")
        self._progress["value"] = 0
        self._analyse_btn.config(state=tk.NORMAL)
        messagebox.showerror("Analysis error", message)

    def _on_pull_progress(self, status: str) -> None:
        """Called from the background thread to relay download progress to the GUI."""
        self._after_safe(self._set_status, f"Downloading model: {status}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        self._status_var.set(text)

    def _set_progress(self, value: int) -> None:
        self._progress["value"] = value

    def _set_result(self, text: str) -> None:
        self._result_text.config(state=tk.NORMAL)
        self._result_text.delete("1.0", tk.END)
        self._result_text.insert(tk.END, text)
        self._result_text.config(state=tk.DISABLED)

    def _after_safe(self, func, *args) -> None:
        """Schedule *func* on the Tk main thread."""
        self.after(0, func, *args)


def run_gui() -> None:
    """Launch the GUI application."""
    app = App()
    app.mainloop()
