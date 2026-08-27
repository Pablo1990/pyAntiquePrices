"""Minimal Tkinter GUI for pyAntiquePrices."""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

from .analyzer import AntiqueAnalyzer
from .scraper import TodoColeccionScraper

logger = logging.getLogger(__name__)

_WINDOW_TITLE = "pyAntiquePrices – Antique Appraiser"
_WINDOW_MIN_W = 780
_WINDOW_MIN_H = 640
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
        self._scraper = TodoColeccionScraper()

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ---- Top frame: image selector + context ----
        top = ttk.LabelFrame(self, text="Item details", padding=_PAD)
        top.pack(fill=tk.X, padx=_PAD, pady=_PAD)

        # Image row
        img_row = ttk.Frame(top)
        img_row.pack(fill=tk.X, pady=(0, _PAD))

        ttk.Label(img_row, text="Image:").pack(side=tk.LEFT)
        self._img_var = tk.StringVar(value="No file selected")
        ttk.Entry(img_row, textvariable=self._img_var, state="readonly", width=55).pack(
            side=tk.LEFT, padx=_PAD
        )
        ttk.Button(img_row, text="Browse…", command=self._browse_image).pack(side=tk.LEFT)

        # Model row
        model_row = ttk.Frame(top)
        model_row.pack(fill=tk.X, pady=(0, _PAD))
        ttk.Label(model_row, text="Ollama model:").pack(side=tk.LEFT)
        self._model_var = tk.StringVar(value="llava")
        ttk.Entry(model_row, textvariable=self._model_var, width=30).pack(
            side=tk.LEFT, padx=_PAD
        )

        # Search keywords row
        kw_row = ttk.Frame(top)
        kw_row.pack(fill=tk.X, pady=(0, _PAD))
        ttk.Label(kw_row, text="Search keywords (optional):").pack(side=tk.LEFT)
        self._keywords_var = tk.StringVar()
        ttk.Entry(kw_row, textvariable=self._keywords_var, width=50).pack(
            side=tk.LEFT, padx=_PAD
        )
        ttk.Label(
            kw_row,
            text="Used to look up reference prices on todocoleccion.net",
            foreground="grey",
        ).pack(side=tk.LEFT)

        # Context area
        ttk.Label(top, text="Additional context / description:").pack(
            anchor=tk.W
        )
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
        self._progress = ttk.Progressbar(self, mode="indeterminate")
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

    def _start_analysis(self) -> None:
        if self._image_path is None:
            messagebox.showwarning("No image", "Please select an image first.")
            return

        if not self._image_path.exists():
            messagebox.showerror("File not found", f"Cannot find:\n{self._image_path}")
            return

        model = self._model_var.get().strip()
        if not model:
            messagebox.showwarning("No model", "Please enter an Ollama model name.")
            return

        # Capture all widget values here on the main thread before handing off
        # to the background thread – Tkinter widgets must NOT be accessed from
        # any thread other than the main one.
        context = self._context_text.get("1.0", tk.END).strip()
        keywords = self._keywords_var.get().strip()
        image_path = self._image_path

        self._analyse_btn.config(state=tk.DISABLED)
        self._progress.start(10)
        self._set_status("Fetching reference prices…")
        self._set_result("")

        thread = threading.Thread(
            target=self._run_analysis,
            args=(image_path, model, context, keywords),
            daemon=True,
        )
        thread.start()

    def _run_analysis(
        self,
        image_path: Path,
        model: str,
        context: str,
        keywords: str,
    ) -> None:
        """Background worker – must not touch Tk widgets directly."""
        try:
            reference_prices = ""
            if keywords:
                try:
                    reference_prices = self._scraper.get_reference_prices(keywords)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Scraper error: %s", exc)

            self._after_safe(self._set_status, "Analysing image with Ollama…")

            self._analyzer.model = model
            # Provide a progress callback so model downloads are visible in the GUI
            self._analyzer.on_pull_progress = self._on_pull_progress
            result = self._analyzer.analyse(
                image_path,
                context=context,
                reference_prices=reference_prices,
            )
            self._after_safe(self._on_analysis_done, result)
        except Exception as exc:  # noqa: BLE001
            self._after_safe(self._on_analysis_error, str(exc))

    def _on_analysis_done(self, result: str) -> None:
        self._set_result(result)
        self._set_status("Analysis complete.")
        self._progress.stop()
        self._analyse_btn.config(state=tk.NORMAL)

    def _on_analysis_error(self, message: str) -> None:
        self._set_result(f"Error:\n{message}")
        self._set_status("Analysis failed.")
        self._progress.stop()
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
