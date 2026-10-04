"""Drive the real tkinter window (skipped when no display is available)."""

import gc

import pytest

tk = pytest.importorskip("tkinter")

from pyantique_prices import gui, i18n  # noqa: E402
from pyantique_prices.deals import assess_deal  # noqa: E402
from pyantique_prices.gui_panels import FormDialog  # noqa: E402
from pyantique_prices.lookup import build_lookup_links  # noqa: E402

IDENT = {"object_type": "pocket watch", "manufacturer_candidates": [{"name": "Omega", "confidence": 0.9}],
         "artist_candidates": []}
VAL = {"low": 120, "mid": 200, "high": 320, "p25": 150, "p50": 200, "p75": 260, "num_comparables": 8,
       "effective_comparables": 6.0, "valuation_available": True, "method": "similarity_weighted_estimate"}


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/gui.db")
    monkeypatch.setenv("APP_LANGUAGE", "en")
    monkeypatch.setattr(i18n, "_CONFIG_PATH", tmp_path / "cfg.json")
    try:
        window = gui.App()
    except tk.TclError:
        pytest.skip("no display available")
    window.withdraw()
    # dialogs: submit immediately instead of blocking in wait_window
    monkeypatch.setattr(FormDialog, "show_modal", lambda self: self.submit())
    monkeypatch.setattr(gui.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(gui.messagebox, "showwarning", lambda *a, **k: None)
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: None)
    yield window
    window.destroy()
    gc.collect()  # drop Tk variables on this thread, not a worker thread
    i18n.set_language("en")


def _result(asking=90):
    return {"identification": IDENT, "valuation": VAL, "valuation_available": True, "currency": "EUR",
            "deal": assess_deal(VAL, asking_price=asking), "identification_confidence": 0.8,
            "lookup_links": build_lookup_links(IDENT, regions=["global"])}


def test_analysis_done_shows_verdict_links_and_report(app):
    app._on_analysis_done(_result())
    assert app._verdict.banner_text == "STRONG BUY"
    assert app._links.link_urls
    assert "Estimated value" in app._estimate_var.get()
    assert str(app._recheck_btn.cget("state")) == "normal"


def test_recheck_deal_uses_costs_form(app):
    app._on_analysis_done(_result(asking=90))
    app._costs.vars["asking"].set("400")
    app._recheck_deal()
    assert app._verdict.banner_text == "OVERPRICED"


def test_check_a_price_tab(app):
    app._found_vars["p50"].set("200")
    app._check_costs.vars["asking"].set("1.000,50")
    app._check_deal()
    assert app._check_verdict.banner_text == "OVERPRICED"
    app._check_vars["object"].set("pocket watch")
    app._check_links()
    assert app._check_links_panel.link_urls


def test_ledger_flow_from_gui(app):
    app._on_analysis_done(_result(asking=90))
    app._costs.vars["asking"].set("90")
    app._save_to_ledger()  # default values from the deal
    assert len(app._ledger_tree.get_children()) == 1
    item = app._ledger_tree.get_children()[0]
    app._ledger_tree.selection_set(item)
    app._ledger_sell()  # price empty -> dialog stays open, nothing saved
    assert app._ledger_tree.set(item, "status") == "held"


def test_links_open_with_injected_opener(app):
    opened = []
    app._links._open = opened.append
    app._on_analysis_done(_result())
    assert app._links.open_key_sources()
    assert opened


def test_switch_language_keeps_inputs_and_translates_everything(app):
    app._on_analysis_done(_result(asking=90))
    app._costs.vars["asking"].set("90")
    app._model_var.set("my-model")
    app._lang_var.set("Español")
    app._on_language_chosen()
    assert i18n.get_language() == "es"
    assert app._tabs.tab(0, "text").strip() == "Tasar con fotos"
    assert app._model_var.get() == "my-model" and app._costs.vars["asking"].get() == "90"
    assert app._verdict.banner_text == "COMPRA MUY BUENA"
    assert "Valor estimado" in app._estimate_var.get()
    report = app._result_text.get("1.0", tk.END)
    assert "ESTIMACIÓN DE PRECIO" in report and "PRICE ESTIMATE" not in report
    assert any("Precios realizados" in line for line in [report])
    # and back again
    app._lang_var.set("English")
    app._on_language_chosen()
    assert app._verdict.banner_text == "STRONG BUY"
    assert i18n.detect_language() == "en"


def test_spanish_ledger_and_check_flow(app):
    app._lang_var.set("Español")
    app._on_language_chosen()
    app._found_vars["p50"].set("200")
    app._check_costs.vars["asking"].set("400")
    app._check_deal()
    assert app._check_verdict.banner_text == "DEMASIADO CARO"
    app._check_vars["object"].set("reloj")
    app._region_var.set("España")
    app._check_links()
    assert app._check_links_panel.link_urls
    app._ledger_add()  # empty -> dialog stays open
    app._on_analysis_done(_result())
    app._costs.vars["asking"].set("90")
    app._save_to_ledger()
    item = app._ledger_tree.get_children()[0]
    assert app._ledger_tree.set(item, "status") == "en stock"
