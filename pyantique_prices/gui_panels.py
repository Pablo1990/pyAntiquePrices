"""Reusable tkinter widgets for the GUI: links, verdict banner, costs form, dialogs."""

from __future__ import annotations

import sys
import tkinter as tk
import webbrowser
from tkinter import ttk
from typing import Any, Callable

from .gui_logic import (
    VERDICT_STYLE,
    deal_inputs_from_form,
    deal_summary_lines,
    key_source_urls,
    link_groups,
)

_PAD = 6
_LINK_COLOR = "#1a5fb4"
_HAND = "pointinghand" if sys.platform == "darwin" else "hand2"


class LinksPanel(ttk.Frame):
    """Clickable research links grouped by what they tell you."""

    def __init__(self, parent, opener: Callable[[str], Any] | None = None) -> None:
        super().__init__(parent)
        self._open = opener or webbrowser.open_new_tab
        self._block: dict | None = None
        self.link_urls: dict[str, str] = {}  # tag -> url (also used by tests)

        bar = ttk.Frame(self)
        bar.pack(fill=tk.X)
        ttk.Label(bar, text="Research links - click to open in your browser", font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
        self._open_key = ttk.Button(bar, text="Open the key sources", command=self.open_key_sources, state=tk.DISABLED)
        self._open_key.pack(side=tk.RIGHT)

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self._text = tk.Text(body, height=10, wrap=tk.WORD, relief=tk.FLAT, borderwidth=1, state=tk.DISABLED,
                             padx=8, pady=6, cursor="arrow")
        scroll = ttk.Scrollbar(body, orient=tk.VERTICAL, command=self._text.yview)
        self._text.configure(yscrollcommand=scroll.set)
        self._text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._text.tag_configure("heading", font=("TkDefaultFont", 10, "bold"), spacing1=8, spacing3=2)
        self._text.tag_configure("note", foreground="#777777")
        self._text.tag_configure("hint", foreground="#555555", spacing3=4)

        self._menu = tk.Menu(self, tearoff=0)
        self._menu_url: str | None = None
        self._menu.add_command(label="Copy link", command=self._copy_link)
        self.clear()

    # -- public ----------------------------------------------------------
    def clear(self, message: str = "Research links appear here after an appraisal or a search.") -> None:
        self._block = None
        self.link_urls = {}
        self._open_key.config(state=tk.DISABLED)
        self._write(lambda t: t.insert(tk.END, message, "note"))

    def show(self, block: dict | None) -> None:
        if not block or not block.get("links"):
            self.clear("No links: identify the object or enter at least an object type or maker.")
            return
        self._block = block
        self.link_urls = {}

        def fill(text: tk.Text) -> None:
            text.insert(tk.END, "Trust a price when SOLD results from 2-3 independent sources agree. "
                                "Asking prices are an upper bound.\n", "hint")
            index = 0
            for title, links in link_groups(block):
                text.insert(tk.END, f"{title}\n", "heading")
                for link in links:
                    tag = f"link{index}"
                    index += 1
                    self.link_urls[tag] = link["url"]
                    text.insert(tk.END, "  ↗ " + link["name"], ("link", tag))
                    if link.get("note"):
                        text.insert(tk.END, f" - {link['note']}", "note")
                    text.insert(tk.END, "\n")
                    text.tag_configure(tag, foreground=_LINK_COLOR, underline=True)
                    text.tag_bind(tag, "<Button-1>", lambda _e, t=tag: self.open_link(t))
                    for seq in ("<Button-3>", "<Button-2>", "<Control-Button-1>"):
                        text.tag_bind(tag, seq, lambda e, t=tag: self._popup(e, t))
                    text.tag_bind(tag, "<Enter>", lambda _e: text.config(cursor=_HAND))
                    text.tag_bind(tag, "<Leave>", lambda _e: text.config(cursor="arrow"))

        self._write(fill)
        self._open_key.config(state=tk.NORMAL if key_source_urls(block) else tk.DISABLED)

    def open_link(self, tag: str) -> None:
        url = self.link_urls.get(tag)
        if url:
            self._open(url)

    def open_key_sources(self, limit: int = 4) -> list[str]:
        urls = key_source_urls(self._block, limit)
        for url in urls:
            self._open(url)
        return urls

    # -- internals -------------------------------------------------------
    def _write(self, fill: Callable[[tk.Text], None]) -> None:
        self._text.config(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        fill(self._text)
        self._text.config(state=tk.DISABLED)

    def _popup(self, event, tag: str) -> str:
        self._menu_url = self.link_urls.get(tag)
        self._menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _copy_link(self) -> None:
        if self._menu_url:
            self.clipboard_clear()
            self.clipboard_append(self._menu_url)


class VerdictPanel(ttk.Frame):
    """Coloured buy/pass banner with the numbers behind it."""

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self._banner = tk.Label(self, text="", font=("TkDefaultFont", 18, "bold"), pady=8)
        self._banner.pack(fill=tk.X)
        self._headline = ttk.Label(self, text="", wraplength=820, justify=tk.LEFT, font=("TkDefaultFont", 11))
        self._headline.pack(fill=tk.X, pady=(6, 2))
        self._detail = tk.Text(self, height=5, wrap=tk.WORD, relief=tk.FLAT, state=tk.DISABLED, padx=2, pady=2,
                               background=ttk.Style().lookup("TFrame", "background") or "#ececec")
        self._detail.pack(fill=tk.X)
        self.clear()

    def clear(self, message: str = "Enter an asking price to get a buy / pass verdict.") -> None:
        self._banner.config(text="NO DEAL CHECK YET", bg="#bdc3c7", fg="#2c3e50")
        self._headline.config(text=message)
        self._set_detail([])

    def show(self, deal: dict | None, currency: str = "EUR") -> None:
        if not deal:
            self.clear()
            return
        label, bg, fg = VERDICT_STYLE.get(deal.get("verdict"), VERDICT_STYLE["no_verdict"])
        self._banner.config(text=label, bg=bg, fg=fg)
        self._headline.config(text=deal.get("headline", ""))
        self._set_detail(deal_summary_lines(deal, currency))

    @property
    def banner_text(self) -> str:
        return self._banner.cget("text")

    def _set_detail(self, lines: list[str]) -> None:
        self._detail.config(state=tk.NORMAL)
        self._detail.delete("1.0", tk.END)
        self._detail.insert(tk.END, "\n".join(lines))
        self._detail.config(state=tk.DISABLED)


class CostsForm(ttk.LabelFrame):
    """Asking price and the real costs of buying (and reselling) an item."""

    FIELDS = (
        ("asking", "Asking price"),
        ("shipping", "Shipping"),
        ("premium", "Buyer's premium %"),
        ("vat", "VAT / import tax %"),
        ("restoration", "Restoration"),
        ("resale_fee", "Selling fee %"),
    )

    def __init__(self, parent, title: str = "Is the price worth it? (optional)", columns: int = 3) -> None:
        super().__init__(parent, text=title, padding=_PAD)
        self.vars: dict[str, tk.StringVar] = {}
        for i, (key, label) in enumerate(self.FIELDS):
            row, col = divmod(i, columns)
            cell = ttk.Frame(self)
            cell.grid(row=row, column=col, sticky=tk.W, padx=(0, 14), pady=2)
            ttk.Label(cell, text=label + ":").pack(side=tk.LEFT)
            var = tk.StringVar()
            self.vars[key] = var
            ttk.Entry(cell, textvariable=var, width=9).pack(side=tk.LEFT, padx=(4, 0))
        rows = -(-len(self.FIELDS) // columns)
        self.keep_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="I'm buying to keep it (ignore selling costs)", variable=self.keep_var).grid(
            row=rows, column=0, columnspan=columns, sticky=tk.W, pady=(4, 0))
        ttk.Label(
            self, foreground="#777777",
            wraplength=420 if columns < 3 else 900, justify=tk.LEFT,
            text="Leave 'Asking price' empty to skip the deal check. Amounts are in your currency; decimals can use , or .",
        ).grid(row=rows + 1, column=0, columnspan=columns, sticky=tk.W)

    def raw(self) -> dict[str, Any]:
        values: dict[str, Any] = {key: var.get() for key, var in self.vars.items()}
        values["keep"] = self.keep_var.get()
        return values

    def inputs(self) -> tuple[float | None, dict]:
        """``(asking_price, deal_options)``; raises ``ValueError`` with a readable message."""
        return deal_inputs_from_form(self.raw())


class FormDialog(tk.Toplevel):
    """Small modal form. ``on_ok(values)`` may raise ``ValueError`` to keep it open."""

    def __init__(self, parent, title: str, fields: list[tuple[str, str, str]],
                 on_ok: Callable[[dict[str, str]], None], intro: str = "") -> None:
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.resizable(False, False)
        self._on_ok = on_ok
        self.vars: dict[str, tk.StringVar] = {}
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        row = 0
        if intro:
            ttk.Label(frame, text=intro, wraplength=380, justify=tk.LEFT, foreground="#555555").grid(
                row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))
            row += 1
        first = None
        for key, label, default in fields:
            ttk.Label(frame, text=label + ":").grid(row=row, column=0, sticky=tk.W, pady=3, padx=(0, 8))
            var = tk.StringVar(value=default)
            self.vars[key] = var
            entry = ttk.Entry(frame, textvariable=var, width=34)
            entry.grid(row=row, column=1, sticky=tk.EW, pady=3)
            first = first or entry
            row += 1
        self._error = ttk.Label(frame, text="", foreground="#c0392b", wraplength=380, justify=tk.LEFT)
        self._error.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        buttons = ttk.Frame(frame)
        buttons.grid(row=row + 1, column=0, columnspan=2, sticky=tk.E, pady=(10, 0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="OK", command=self.submit).pack(side=tk.RIGHT, padx=(0, 6))
        self.bind("<Return>", lambda _e: self.submit())
        self.bind("<Escape>", lambda _e: self.destroy())
        if first is not None:
            first.focus_set()

    def values(self) -> dict[str, str]:
        return {key: var.get().strip() for key, var in self.vars.items()}

    def submit(self) -> bool:
        try:
            self._on_ok(self.values())
        except (ValueError, KeyError) as exc:
            self._error.config(text=str(exc).strip("'\""))
            return False
        self.destroy()
        return True

    def show_modal(self) -> None:
        self.grab_set()
        self.wait_window(self)
