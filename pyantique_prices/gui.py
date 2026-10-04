"""Tkinter GUI: appraise photos, check a price, keep a ledger, manage your data."""

from __future__ import annotations

import datetime
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .config import Settings
from .embeddings import NullImageEmbeddingProvider, OllamaTextEmbeddingProvider
from .gui_logic import (
    REGION_CHOICES,
    backtest_lines,
    estimate_line,
    identification_from_fields,
    import_lines,
    ledger_row,
    manual_valuation,
    parse_number,
    summary_lines,
)
from .gui_panels import CostsForm, FormDialog, LinksPanel, VerdictPanel
from .services.appraisal import AppraisalService, LegacyWebFallbackEstimator
from .vision.analyzer import MAX_IMAGES, MIN_IMAGES, SUPPORTED_EXTENSIONS, MultiImageAnalyzer
from .vision.marks import MarkAnalysisService
from .vision.ollama import OllamaClient

_WINDOW_TITLE = "AntiqueGPT"
_WINDOW_MIN_W = 1000
_WINDOW_MIN_H = 880
_PAD = 8


def _today() -> str:
    return datetime.date.today().isoformat()


def _build_service(settings: Settings, model: str, session_factory) -> AppraisalService:
    """Wire the appraisal service from settings (shared by every GUI action)."""
    from pyantique_prices.services.live_market import EbayLiveListings

    client = OllamaClient(host=settings.ollama_host, model=model, num_ctx=settings.ollama_num_ctx)
    analyzer = MultiImageAnalyzer(client=client, mark_service=MarkAnalysisService())
    text_embedding_provider = OllamaTextEmbeddingProvider(
        host=settings.ollama_host,
        model=settings.ollama_embed_model,
        num_ctx=settings.ollama_num_ctx,
        require_model=False,
    )
    pricer = None
    try:
        from .pricing.model import PricePredictor

        pricer = PricePredictor(
            min_comparables_for_model=settings.min_comparables_for_model,
            min_comparables_for_confidence=settings.min_comparables_for_confidence,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise
    return AppraisalService(
        analyzer=analyzer,
        retrieval_session_factory=session_factory,
        text_embedding_provider=text_embedding_provider,
        image_embedding_provider=NullImageEmbeddingProvider(),
        pricer=pricer,
        fallback_estimator=LegacyWebFallbackEstimator(model=model),
        live_market=EbayLiveListings.from_settings(),
        ebay_domain=settings.ebay_domain,
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


class App(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title(_WINDOW_TITLE)
        self.minsize(_WINDOW_MIN_W, _WINDOW_MIN_H)
        self.resizable(True, True)
        self._settings = Settings()
        self._image_paths: list[Path] = []
        self._last_result: dict | None = None
        self._session_factory = None
        self._db_error: str | None = None
        self._init_db()

        self._tabs = ttk.Notebook(self)
        self._tabs.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._appraise_tab = ttk.Frame(self._tabs)
        self._check_tab = ttk.Frame(self._tabs)
        self._ledger_tab = ttk.Frame(self._tabs)
        self._data_tab = ttk.Frame(self._tabs)
        self._tabs.add(self._appraise_tab, text="  Appraise photos  ")
        self._tabs.add(self._check_tab, text="  Check a price  ")
        self._tabs.add(self._ledger_tab, text="  My ledger  ")
        self._tabs.add(self._data_tab, text="  Data & accuracy  ")
        self._build_appraise_tab()
        self._build_check_tab()
        self._build_ledger_tab()
        self._build_data_tab()
        self.refresh_ledger()
        self.refresh_data_stats()

    # ------------------------------------------------------------------ DB
    def _init_db(self) -> None:
        try:
            from .data.database import create_tables, get_engine, get_session_factory

            engine = get_engine(self._settings.database_url)
            create_tables(engine)
            self._session_factory = get_session_factory(engine)
        except ModuleNotFoundError as exc:
            self._db_error = f"{exc.name} is not installed, so the ledger and sales data are unavailable."
        except Exception as exc:  # noqa: BLE001
            self._db_error = f"Could not open the database: {exc}"

    def _session(self):
        if self._session_factory is None:
            raise RuntimeError(self._db_error or "Database unavailable.")
        return self._session_factory()

    def _calibration(self) -> float:
        try:
            from .ledger import calibration_factor

            with self._session() as session:
                return calibration_factor(session)
        except Exception:  # noqa: BLE001
            return 1.0

    def _error(self, title: str, exc: Exception | str) -> None:
        messagebox.showerror(title, str(exc), parent=self)

    # ------------------------------------------------------ tab 1: appraise
    def _build_appraise_tab(self) -> None:
        tab = self._appraise_tab
        top = ttk.LabelFrame(tab, text="The object", padding=_PAD)
        top.pack(fill=tk.X, padx=_PAD, pady=(_PAD, 4))

        image_row = ttk.Frame(top)
        image_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(image_row, text=f"Photos ({MIN_IMAGES}-{MAX_IMAGES}):").pack(side=tk.LEFT)
        self._img_var = tk.StringVar(value="No photos selected")
        ttk.Entry(image_row, textvariable=self._img_var, state="readonly", width=64).pack(side=tk.LEFT, padx=_PAD)
        ttk.Button(image_row, text="Select photos…", command=self._browse_images).pack(side=tk.LEFT)
        ttk.Button(image_row, text="Clear", command=self._clear_images).pack(side=tk.LEFT, padx=(4, 0))

        model_row = ttk.Frame(top)
        model_row.pack(fill=tk.X, pady=2)
        ttk.Label(model_row, text="Vision model:").pack(side=tk.LEFT)
        self._model_var = tk.StringVar(value=self._settings.ollama_vision_model)
        ttk.Entry(model_row, textvariable=self._model_var, width=24).pack(side=tk.LEFT, padx=_PAD)
        ttk.Label(model_row, text="Currency:").pack(side=tk.LEFT)
        self._currency_var = tk.StringVar(value=self._settings.base_currency)
        ttk.Entry(model_row, textvariable=self._currency_var, width=6).pack(side=tk.LEFT, padx=(4, _PAD))
        ttk.Label(model_row, text="Location:").pack(side=tk.LEFT)
        self._location_var = tk.StringVar()
        ttk.Entry(model_row, textvariable=self._location_var, width=16).pack(side=tk.LEFT, padx=(4, _PAD))
        ttk.Label(model_row, text="Dimensions:").pack(side=tk.LEFT)
        self._dimensions_var = tk.StringVar()
        ttk.Entry(model_row, textvariable=self._dimensions_var, width=16).pack(side=tk.LEFT, padx=(4, _PAD))
        self._provenance_var = tk.StringVar()
        prov_row = ttk.Frame(top)
        prov_row.pack(fill=tk.X, pady=2)
        ttk.Label(prov_row, text="Provenance:").pack(side=tk.LEFT)
        ttk.Entry(prov_row, textvariable=self._provenance_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        ttk.Label(top, text="Description / context (anything the seller says):").pack(anchor=tk.W, pady=(4, 0))
        self._context_text = scrolledtext.ScrolledText(top, height=2, wrap=tk.WORD)
        self._context_text.pack(fill=tk.X, pady=(2, 0))

        self._costs = CostsForm(tab)
        self._costs.pack(fill=tk.X, padx=_PAD, pady=4)

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=_PAD)
        self._analyse_btn = ttk.Button(btn_frame, text="Analyze object", command=self._start_analysis)
        self._analyse_btn.pack(side=tk.LEFT)
        self._recheck_btn = ttk.Button(btn_frame, text="Recalculate deal", command=self._recheck_deal, state=tk.DISABLED)
        self._recheck_btn.pack(side=tk.LEFT, padx=(6, 0))
        self._save_btn = ttk.Button(btn_frame, text="I bought it → save to ledger…", command=self._save_to_ledger,
                                    state=tk.DISABLED)
        self._save_btn.pack(side=tk.LEFT, padx=(6, 0))
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(btn_frame, textvariable=self._status_var, foreground="grey").pack(side=tk.LEFT, padx=_PAD)

        self._progress = ttk.Progressbar(tab, mode="indeterminate")
        self._progress.pack(fill=tk.X, padx=_PAD, pady=(2, 0))

        self._results = ttk.Notebook(tab)
        self._results.pack(fill=tk.BOTH, expand=True, padx=_PAD, pady=_PAD)
        summary = ttk.Frame(self._results, padding=_PAD)
        self._results.add(summary, text="Verdict & research links")
        self._verdict = VerdictPanel(summary)
        self._verdict.pack(fill=tk.X)
        self._estimate_var = tk.StringVar(value="")
        ttk.Label(summary, textvariable=self._estimate_var, wraplength=900, justify=tk.LEFT,
                  font=("TkDefaultFont", 10, "bold")).pack(fill=tk.X, pady=(8, 4))
        self._links = LinksPanel(summary)
        self._links.pack(fill=tk.BOTH, expand=True)
        report = ttk.Frame(self._results)
        self._results.add(report, text="Full report")
        self._result_text = scrolledtext.ScrolledText(report, wrap=tk.WORD, state=tk.DISABLED)
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
            messagebox.showwarning("Invalid photo count",
                                   f"Please select between {MIN_IMAGES} and {MAX_IMAGES} photos.", parent=self)
            return
        for path in self._image_paths:
            if not path.exists():
                messagebox.showerror("Not found", f"Cannot find:\n{path}", parent=self)
                return
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                messagebox.showwarning("Unsupported format", "Supported formats: JPEG, PNG, WebP.", parent=self)
                return
        try:
            asking, deal_options = self._costs.inputs()
        except ValueError as exc:
            messagebox.showwarning("Check the price fields", str(exc), parent=self)
            return

        model = self._model_var.get().strip() or self._settings.ollama_vision_model
        currency = (self._currency_var.get().strip() or self._settings.base_currency).upper()
        context = self._context_text.get("1.0", tk.END).strip()
        location = self._location_var.get().strip()
        dimensions = self._dimensions_var.get().strip()
        provenance = self._provenance_var.get().strip()

        self._analyse_btn.config(state=tk.DISABLED)
        self._progress.start(8)
        self._set_status("Analyzing the object (the vision model can take a minute)…")
        self._set_result("")
        self._verdict.clear("Working…")
        self._links.clear("Working…")
        self._estimate_var.set("")

        threading.Thread(
            target=self._run_analysis,
            args=(list(self._image_paths), model, currency, context, location, dimensions, provenance,
                  asking, deal_options),
            daemon=True,
        ).start()

    def _run_analysis(
        self,
        image_paths: list[Path],
        model: str,
        currency: str,
        context: str,
        location: str,
        dimensions: str,
        provenance: str,
        asking: float | None = None,
        deal_options: dict | None = None,
    ) -> None:
        try:
            settings = Settings()
            service = _build_service(settings, model, self._session_factory)
            full_context = _build_context(
                context=context, location=location, known_dimensions=dimensions, provenance=provenance
            )
            extra = {}
            if asking is not None:
                extra = {"asking_price": asking, "deal_options": deal_options}
            result = service.appraise(image_paths, context=full_context, currency=currency, **extra)
            if self._db_error:
                result.setdefault("warnings", []).append(self._db_error)

            if self._session_factory is not None:
                from .data.appraisals import persist_appraisal

                _, persistence_warning = persist_appraisal(
                    session_factory=self._session_factory,
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
                    model_versions={"vision_model": model, "pricing_model": _pricing_model_name(result)},
                )
                if persistence_warning:
                    result.setdefault("warnings", []).append(persistence_warning)
            self._after_safe(self._on_analysis_done, result)
        except Exception as exc:  # noqa: BLE001
            self._after_safe(self._on_analysis_error, str(exc))

    def _on_analysis_done(self, result: dict) -> None:
        self._stop_progress()
        self._last_result = result
        currency = result.get("currency", "EUR")
        self._verdict.show(result.get("deal"), currency)
        if not result.get("deal"):
            self._verdict.clear("Enter an asking price above and press 'Recalculate deal' for a buy / pass verdict.")
        self._estimate_var.set(estimate_line(result))
        self._links.show(result.get("lookup_links"))
        self._set_result(_format_appraisal(result))
        self._results.select(0)
        self._set_status("Analysis complete.")
        self._analyse_btn.config(state=tk.NORMAL)
        self._recheck_btn.config(state=tk.NORMAL)
        self._save_btn.config(state=tk.NORMAL)

    def _on_analysis_error(self, message: str) -> None:
        self._stop_progress()
        self._set_result(f"Error:\n{message}")
        self._verdict.clear("The analysis failed. See the Full report tab.")
        self._links.clear()
        self._set_status("Analysis failed.")
        self._analyse_btn.config(state=tk.NORMAL)
        messagebox.showerror("Analysis error", message, parent=self)

    def _recheck_deal(self) -> None:
        """Recompute the verdict from the last estimate (no new vision run)."""
        if not self._last_result:
            return
        try:
            asking, options = self._costs.inputs()
        except ValueError as exc:
            messagebox.showwarning("Check the price fields", str(exc), parent=self)
            return
        if asking is None:
            messagebox.showinfo("Asking price needed", "Type the asking price first.", parent=self)
            return
        from .deals import assess_deal

        deal = assess_deal(
            self._last_result.get("valuation"),
            asking_price=asking,
            identification_confidence=self._last_result.get("identification_confidence"),
            calibration_factor=self._calibration(),
            **options,
        )
        self._last_result["deal"] = deal
        self._verdict.show(deal, self._last_result.get("currency", "EUR"))
        self._set_result(_format_appraisal(self._last_result))

    def _save_to_ledger(self) -> None:
        result = self._last_result
        if not result:
            return
        ident = result.get("identification") or {}
        maker = next((c["name"] for c in (ident.get("manufacturer_candidates") or [])
                      if isinstance(c, dict) and c.get("name")), "")
        obj = _extract_value(ident.get("object_type")) or ""
        deal = result.get("deal") or {}
        try:
            asking, options = self._costs.inputs()
        except ValueError:
            asking, options = None, {}
        extra_costs = 0.0
        if asking is not None:
            extra_costs = (deal.get("all_in_cost") or asking) - asking
        fields = [
            ("title", "Item", f"{maker} {obj}".strip()),
            ("price", "Price paid", f"{asking:g}" if asking else ""),
            ("costs", "Extra costs (shipping, premium…)", f"{max(extra_costs, 0):.2f}"),
            ("where", "Where did you buy it?", ""),
            ("date", "Date (YYYY-MM-DD)", _today()),
            ("notes", "Notes", ""),
        ]

        def save(values: dict[str, str]) -> None:
            from . import ledger

            if not values["title"]:
                raise ValueError("Give the item a name.")
            price = parse_number(values["price"], "Price paid", minimum=0.0)
            if price is None:
                raise ValueError("Enter the price you paid.")
            with self._session() as session:
                ledger.add_purchase(
                    session, title=values["title"], price=price,
                    costs=parse_number(values["costs"], "Extra costs", default=0.0),
                    date=_parse_date(values["date"]), where=values["where"] or None,
                    currency=result.get("currency", "EUR"), identification=ident,
                    valuation=result.get("valuation"), deal=result.get("deal"), notes=values["notes"] or None,
                )

        dialog = FormDialog(self, "Save to ledger", fields, save,
                            intro="Records this purchase together with the estimate, so you can see later "
                                  "how accurate it was.")
        dialog.show_modal()
        self.refresh_ledger()
        self._set_status("Saved to your ledger (see the 'My ledger' tab).")

    # --------------------------------------------------- tab 2: check a price
    def _build_check_tab(self) -> None:
        tab = self._check_tab
        tab.columnconfigure(0, weight=1, uniform="cols")
        tab.columnconfigure(1, weight=1, uniform="cols")
        tab.rowconfigure(0, weight=1)

        left = ttk.Frame(tab, padding=_PAD)
        left.grid(row=0, column=0, sticky="nsew")
        box = ttk.LabelFrame(left, text="1. What is it?", padding=_PAD)
        box.pack(fill=tk.X)
        self._check_vars: dict[str, tk.StringVar] = {}
        for i, (key, label) in enumerate([
            ("object", "Object (e.g. pocket watch)"), ("subtype", "Type / style"), ("maker", "Maker / brand"),
            ("artist", "Artist"), ("material", "Material"), ("period", "Period / date"),
        ]):
            ttk.Label(box, text=label + ":").grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar()
            self._check_vars[key] = var
            ttk.Entry(box, textvariable=var, width=26).grid(row=i, column=1, sticky=tk.EW, padx=(6, 0), pady=2)
        ttk.Label(box, text="Where to look:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self._region_var = tk.StringVar(value=next(iter(REGION_CHOICES)))
        ttk.Combobox(box, textvariable=self._region_var, values=list(REGION_CHOICES), state="readonly",
                     width=24).grid(row=6, column=1, sticky=tk.EW, padx=(6, 0), pady=2)
        box.columnconfigure(1, weight=1)
        ttk.Button(box, text="Show research links", command=self._check_links).grid(
            row=7, column=0, columnspan=2, sticky=tk.EW, pady=(8, 0))
        self._check_links_panel = LinksPanel(left)
        self._check_links_panel.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        right = ttk.Frame(tab, padding=_PAD)
        right.grid(row=0, column=1, sticky="nsew")
        found = ttk.LabelFrame(right, text="2. What did the SOLD results show?", padding=_PAD)
        found.pack(fill=tk.X)
        self._found_vars: dict[str, tk.StringVar] = {}
        for i, (key, label) in enumerate([
            ("p50", "Typical sold price"), ("p25", "Lowest normal sold price"),
            ("p75", "Highest normal sold price"), ("n", "How many sold results"),
        ]):
            ttk.Label(found, text=label + ":").grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar()
            self._found_vars[key] = var
            ttk.Entry(found, textvariable=var, width=12).grid(row=i, column=1, sticky=tk.W, padx=(6, 0), pady=2)
        ttk.Label(found, foreground="#777777", wraplength=380, justify=tk.LEFT,
                  text="Only the typical price is required. Fewer than 6 sold results counts as weak evidence "
                       "and raises the margin the verdict asks for.").grid(row=4, column=0, columnspan=2,
                                                                          sticky=tk.W, pady=(4, 0))
        self._check_costs = CostsForm(right, title="3. The deal", columns=2)
        self._check_costs.pack(fill=tk.X, pady=8)
        ttk.Button(right, text="Check the deal", command=self._check_deal).pack(anchor=tk.W)
        self._check_verdict = VerdictPanel(right)
        self._check_verdict.pack(fill=tk.X, pady=(10, 0))

    def _check_links(self) -> None:
        from .lookup import build_lookup_links

        fields = {k: v.get() for k, v in self._check_vars.items()}
        ident = identification_from_fields(fields)
        if not any([ident["object_type"], ident["subtype"], ident["manufacturer_candidates"],
                    ident["artist_candidates"]]):
            messagebox.showinfo("Describe the item", "Enter at least an object, maker or artist.", parent=self)
            return
        block = build_lookup_links(ident, ebay_domain=self._settings.ebay_domain,
                                   regions=REGION_CHOICES[self._region_var.get()])
        self._check_links_panel.show(block)

    def _check_deal(self) -> None:
        from .deals import assess_deal

        try:
            valuation = manual_valuation({k: v.get() for k, v in self._found_vars.items()})
            asking, options = self._check_costs.inputs()
            if asking is None:
                raise ValueError("Enter the asking price in section 3.")
        except ValueError as exc:
            messagebox.showwarning("Check the fields", str(exc), parent=self)
            return
        deal = assess_deal(valuation, asking_price=asking, calibration_factor=self._calibration(), **options)
        self._check_verdict.show(deal, self._settings.base_currency)

    # ------------------------------------------------------- tab 3: ledger
    def _build_ledger_tab(self) -> None:
        tab = self._ledger_tab
        bar = ttk.Frame(tab, padding=_PAD)
        bar.pack(fill=tk.X)
        for text, command in [
            ("Add a purchase…", self._ledger_add),
            ("Mark as sold…", self._ledger_sell),
            ("Export sold items (CSV)…", self._ledger_export),
            ("Add sold items to my price data", self._ledger_sync),
        ]:
            ttk.Button(bar, text=text, command=command).pack(side=tk.LEFT, padx=(0, 6))
        columns = ("id", "item", "status", "bought", "paid", "estimate", "sold", "profit")
        headings = ("#", "Item", "Status", "Bought", "Paid (all-in)", "Estimate", "Sold for", "Profit")
        widths = (40, 300, 70, 90, 100, 90, 90, 90)
        frame = ttk.Frame(tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=_PAD)
        self._ledger_tree = ttk.Treeview(frame, columns=columns, show="headings", height=12, selectmode="browse")
        for col, head, width in zip(columns, headings, widths):
            self._ledger_tree.heading(col, text=head)
            self._ledger_tree.column(col, width=width, anchor=tk.W if col in ("item", "status") else tk.E)
        self._ledger_tree.tag_configure("gain", foreground="#1b7f3b")
        self._ledger_tree.tag_configure("loss", foreground="#c0392b")
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self._ledger_tree.yview)
        self._ledger_tree.configure(yscrollcommand=scroll.set)
        self._ledger_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._ledger_tree.bind("<Double-1>", lambda _e: self._ledger_sell())
        box = ttk.LabelFrame(tab, text="How you are doing", padding=_PAD)
        box.pack(fill=tk.X, padx=_PAD, pady=_PAD)
        self._ledger_summary = tk.StringVar()
        ttk.Label(box, textvariable=self._ledger_summary, justify=tk.LEFT, wraplength=900).pack(anchor=tk.W)

    def refresh_ledger(self) -> None:
        tree = self._ledger_tree
        tree.delete(*tree.get_children())
        if self._session_factory is None:
            self._ledger_summary.set(self._db_error or "Database unavailable.")
            return
        from . import ledger
        from .ledger.service import profit

        with self._session() as session:
            items = ledger.list_items(session)
            for item in items:
                gain = profit(item)
                tag = () if gain is None else (("gain",) if gain >= 0 else ("loss",))
                tree.insert("", tk.END, iid=str(item.id), values=ledger_row(item), tags=tag)
            summary = ledger.summarize(session)
            factor = ledger.calibration_factor(session)
        self._ledger_summary.set("\n".join(summary_lines(summary, self._settings.base_currency, factor)))

    def _ledger_add(self) -> None:
        fields = [("title", "Item", ""), ("price", "Price paid", ""), ("costs", "Extra costs", "0"),
                  ("where", "Where", ""), ("date", "Date (YYYY-MM-DD)", _today()),
                  ("est", "Your estimate of its value (optional)", "")]

        def save(values: dict[str, str]) -> None:
            from . import ledger

            if not values["title"]:
                raise ValueError("Give the item a name.")
            price = parse_number(values["price"], "Price paid")
            if price is None:
                raise ValueError("Enter the price you paid.")
            est = parse_number(values["est"], "Estimate", minimum=0.0)
            with self._session() as session:
                ledger.add_purchase(
                    session, title=values["title"], price=price,
                    costs=parse_number(values["costs"], "Extra costs", default=0.0),
                    date=_parse_date(values["date"]), where=values["where"] or None,
                    currency=self._settings.base_currency,
                    valuation={"p50": est, "method": "manual"} if est else None,
                )

        FormDialog(self, "Add a purchase", fields, save).show_modal()
        self.refresh_ledger()

    def _selected_item_id(self) -> int | None:
        selection = self._ledger_tree.selection()
        return int(selection[0]) if selection else None

    def _ledger_sell(self) -> None:
        item_id = self._selected_item_id()
        if item_id is None:
            messagebox.showinfo("Select an item", "Click the item you sold first.", parent=self)
            return
        status = self._ledger_tree.set(str(item_id), "status")
        if status == "sold":
            messagebox.showinfo("Already sold", "This item is already marked as sold.", parent=self)
            return
        fields = [("price", "Sold for", ""), ("fees", "Selling fees + postage", "0"),
                  ("date", "Date (YYYY-MM-DD)", _today()), ("where", "Where", "")]

        def save(values: dict[str, str]) -> None:
            from . import ledger

            price = parse_number(values["price"], "Sale price")
            if price is None:
                raise ValueError("Enter the price it sold for.")
            with self._session() as session:
                ledger.record_sale(session, item_id, price=price,
                                   fees=parse_number(values["fees"], "Fees", default=0.0),
                                   date=_parse_date(values["date"]), where=values["where"] or None)

        FormDialog(self, "Mark as sold", fields, save).show_modal()
        self.refresh_ledger()

    def _ledger_export(self) -> None:
        path = filedialog.asksaveasfilename(parent=self, title="Export sold items", defaultextension=".csv",
                                            initialfile="my_sales.csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            from . import ledger

            with self._session() as session:
                count = ledger.export_sales_csv(session, path)
            messagebox.showinfo("Exported", f"Wrote {count} sold item(s) to:\n{path}", parent=self)
        except Exception as exc:  # noqa: BLE001
            self._error("Export failed", exc)

    def _ledger_sync(self) -> None:
        try:
            from . import ledger

            with self._session() as session:
                added = ledger.sync_to_sales(session, self._settings.base_currency)
            messagebox.showinfo(
                "Price data updated",
                f"Added {added} sold item(s) to your comparable sales." if added else
                "Nothing new: all your sold items are already in your price data.", parent=self)
            self.refresh_data_stats()
        except Exception as exc:  # noqa: BLE001
            self._error("Could not update", exc)

    # ---------------------------------------------- tab 4: data & accuracy
    def _build_data_tab(self) -> None:
        tab = self._data_tab
        intro = ttk.Label(
            tab, wraplength=900, justify=tk.LEFT, padding=_PAD,
            text="The price estimate is only as good as the sales you give it. Use data you have the right to "
                 "use: your own deals (the ledger), data you licensed, or a CSV an auction house allowed you "
                 "to use. Do not bulk-copy results from sites whose terms forbid it.")
        intro.pack(fill=tk.X)
        self._stats_var = tk.StringVar()
        ttk.Label(tab, textvariable=self._stats_var, padding=(_PAD, 0), font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)
        bar = ttk.Frame(tab, padding=_PAD)
        bar.pack(fill=tk.X)
        ttk.Button(bar, text="Import sales (CSV)…", command=self._import_csv).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(bar, text="Save a blank CSV template…", command=self._save_template).pack(side=tk.LEFT, padx=(0, 6))
        self._backtest_btn = ttk.Button(bar, text="Test the accuracy on my data", command=self._run_backtest)
        self._backtest_btn.pack(side=tk.LEFT)
        self._data_log = scrolledtext.ScrolledText(tab, height=14, wrap=tk.WORD, state=tk.DISABLED)
        self._data_log.pack(fill=tk.BOTH, expand=True, padx=_PAD, pady=(0, _PAD))
        self._log_data("Import sales to give the estimator real prices to compare against. "
                       "'Test the accuracy' re-appraises each of your past sales using only earlier ones and "
                       "shows how far off the estimates are.")

    def _log_data(self, text: str) -> None:
        self._data_log.config(state=tk.NORMAL)
        self._data_log.delete("1.0", tk.END)
        self._data_log.insert(tk.END, text)
        self._data_log.config(state=tk.DISABLED)

    def refresh_data_stats(self) -> None:
        if self._session_factory is None:
            self._stats_var.set(self._db_error or "Database unavailable.")
            return
        from sqlalchemy import func

        from .data.models import HistoricalSale

        with self._session() as session:
            total = session.query(func.count(HistoricalSale.id)).scalar() or 0
            usable = session.query(func.count(HistoricalSale.id)).filter(
                HistoricalSale.normalized_price.is_not(None), HistoricalSale.usable_for_training.is_not(False),
                HistoricalSale.outlier_flag.is_not(True)).scalar() or 0
            first, last = session.query(func.min(HistoricalSale.sale_date), func.max(HistoricalSale.sale_date)).one()
        span = f", {first.year}–{last.year}" if first and last else ""
        self._stats_var.set(f"Your price data: {total} sales, {usable} usable as comparables{span}")

    def _import_csv(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Choose a sales CSV",
                                          filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            from .data.importer import import_csv

            with self._session() as session:
                result = import_csv(path, session, base_currency=self._settings.base_currency,
                                    hammer_premium_rate=self._settings.hammer_premium_rate)
            self._log_data("Import finished.\n\n" + "\n".join(import_lines(result)) +
                           "\n\nNext: press 'Test the accuracy on my data'.")
            self.refresh_data_stats()
        except Exception as exc:  # noqa: BLE001
            self._error("Import failed", exc)

    def _save_template(self) -> None:
        from .toolkit_cli import SALES_TEMPLATE_HEADER, SALES_TEMPLATE_ROW

        path = filedialog.asksaveasfilename(parent=self, title="Save the CSV template", defaultextension=".csv",
                                            initialfile="my_sales_template.csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        Path(path).write_text(SALES_TEMPLATE_HEADER + "\n" + SALES_TEMPLATE_ROW + "\n", encoding="utf-8")
        self._log_data(f"Template saved to {path}.\nReplace the example row with your own sales, then import it.")

    def _run_backtest(self) -> None:
        if self._session_factory is None:
            self._error("Unavailable", self._db_error or "Database unavailable.")
            return
        self._backtest_btn.config(state=tk.DISABLED)
        self._log_data("Testing… this can take a minute for large data sets.")
        threading.Thread(target=self._backtest_worker, daemon=True).start()

    def _backtest_worker(self) -> None:
        try:
            from .evaluation import BacktestConfig, run_backtest
            from .pricing.model import PricePredictor

            predictor = PricePredictor(
                min_comparables_for_model=self._settings.min_comparables_for_model,
                min_comparables_for_confidence=self._settings.min_comparables_for_confidence,
            )
            with self._session() as session:
                report = run_backtest(session, predictor, BacktestConfig(max_targets=200))
            text = "\n".join(backtest_lines(report))
        except Exception as exc:  # noqa: BLE001
            text = f"The test failed: {exc}"
        self._after_safe(self._on_backtest_done, text)

    def _on_backtest_done(self, text: str) -> None:
        self._backtest_btn.config(state=tk.NORMAL)
        self._log_data(text)

    # ---------------------------------------------------------------- misc
    def _stop_progress(self) -> None:
        self._progress.stop()
        self._progress.config(value=0)

    def _set_status(self, text: str) -> None:
        self._status_var.set(text)

    def _set_result(self, text: str) -> None:
        self._result_text.config(state=tk.NORMAL)
        self._result_text.delete("1.0", tk.END)
        self._result_text.insert(tk.END, text)
        self._result_text.config(state=tk.DISABLED)

    def _after_safe(self, func, *args) -> None:
        self.after(0, func, *args)


def _parse_date(text: str):
    """``YYYY-MM-DD`` (or empty = today) -> ``datetime.date``."""
    text = (text or "").strip()
    if not text:
        return datetime.date.today()
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"Date '{text}' must look like 2026-03-14.") from None


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


def _format_candidates(candidates) -> str:
    if not candidates:
        return "N/A"
    parts = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate.get("name"):
            continue
        confidence = float(candidate.get("confidence", 0.0) or 0.0)
        evidence = candidate.get("evidence")
        text = f"{candidate['name']} ({confidence * 100:.0f}%)"
        if evidence:
            text += f" – {evidence}"
        parts.append(text)
    return "; ".join(parts) or "N/A"


def _pricing_model_name(result: dict) -> str:
    valuation = result.get("valuation") or {}
    if isinstance(valuation, dict):
        return valuation.get("method", "unknown")
    return "unknown"


def _format_appraisal(result: dict) -> str:
    identification = result.get("identification") or {}
    valuation = result.get("valuation") or {}
    marks = identification.get("marks") or []

    lines = []
    lines.append("OBJECT IDENTIFICATION")
    lines.append("-" * 43)
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
    lines.append(f"Condition: {_extract_value(identification.get('condition')) or 'N/A'}")
    lines.append("")

    lines.append("MAKER MARKS")
    lines.append("-" * 43)
    if marks:
        for mark in marks:
            lines.append(
                f"- Mark: {mark.get('text') or 'Unreadable'} | "
                f"Confidence: {mark.get('confidence', 0.0):.2f} | "
                f"Evidence: {mark.get('evidence') or 'N/A'} | "
                f"Candidates: {_format_candidates(mark.get('manufacturer_candidates'))}"
            )
    else:
        lines.append("No marks detected.")
    lines.append("")

    lines.append("TOP COMPARABLES")
    lines.append("-" * 43)
    lines.append(
        f"Candidates: {result.get('candidate_count', 0)} | "
        f"Usable: {result.get('usable_comparable_count', 0)}"
    )
    comparables = result.get("comparables", [])
    for comparable in comparables[:10]:
        lines.append(
            f"- Auction: {comparable.get('auction_house') or 'Unknown'} | "
            f"Date: {comparable.get('sale_date') or 'N/A'} | "
            f"Object: {comparable.get('title') or 'Untitled'} | "
            f"Similarity: {comparable.get('overall_similarity', comparable.get('retrieval_score', 0.0)):.3f} | "
            f"Price: {comparable.get('normalized_price')} {result.get('currency', 'EUR')}"
        )
    lines.append("")

    from pyantique_prices.services.live_market import format_live_listings

    live_lines = format_live_listings(result.get("live_market_listings"))
    if live_lines:
        lines.append(live_lines[0])
        lines.append("-" * 43)
        lines.extend(live_lines[1:])
        lines.append("")

    lines.append("PRICE ESTIMATE")
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

    from pyantique_prices.deals import format_deal
    from pyantique_prices.lookup import format_lookup_links

    deal_lines = format_deal(result.get("deal"), result.get("currency", "EUR"))
    if deal_lines:
        lines.append(deal_lines[0])
        lines.append("-" * 43)
        lines.extend(deal_lines[1:])
        lines.append("")
    link_lines = format_lookup_links(result.get("lookup_links"))
    if link_lines:
        lines.append(link_lines[0].split(" (")[0])
        lines.append("-" * 43)
        lines.extend(link_lines[1:])
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
