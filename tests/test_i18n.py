import ast
import pathlib
import string

import pytest

pytest.importorskip("tkinter")

from pyantique_prices import i18n
from pyantique_prices.gui_logic import REGION_CHOICES, VERDICT_STYLE
from pyantique_prices.gui_panels import CostsForm
from pyantique_prices.i18n_es import ES
from pyantique_prices.lookup import links as links_mod
from pyantique_prices.services.live_market import NOTICE

PKG = pathlib.Path(i18n.__file__).parent
FILES = ["gui.py", "gui_panels.py", "gui_logic.py", "deals/score.py", "lookup/links.py", "services/live_market.py"]


@pytest.fixture(autouse=True)
def _english_after():
    yield
    i18n.set_language("en")


def _literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal(node.left), _literal(node.right)
        return left + right if left is not None and right is not None else None
    return None


def _source_keys() -> set[str]:
    keys = set()
    for name in FILES:
        for node in ast.walk(ast.parse((PKG / name).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "t" and node.args:
                value = _literal(node.args[0])
                if value is not None:
                    keys.add(value)
    return keys


def _dynamic_keys() -> set[str]:
    keys = set(links_mod._KIND_TITLES.values()) | set(REGION_CHOICES)
    keys |= {label for label, _bg, _fg in VERDICT_STYLE.values()}
    keys |= {label for _key, label in CostsForm.FIELDS}
    keys |= {NOTICE, NOTICE.replace("Prices are sellers' asking prices, not sale prices. ", "Prices are recent eBay sale prices. ")}
    keys |= {"held", "sold", "kept"}
    keys |= {site["note"] for site in [*links_mod.DEFAULT_SITES, *links_mod._REFERENCE_SITES] if site.get("note")}
    return keys


def _fields(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_every_message_has_a_spanish_translation():
    missing = sorted((_source_keys() | _dynamic_keys()) - set(ES))
    assert not missing, f"missing Spanish translations: {missing}"


def test_placeholders_match_between_languages():
    for english, spanish in ES.items():
        assert _fields(english) == _fields(spanish), english


def test_language_switch_and_fallback():
    i18n.set_language("es")
    assert i18n.t("Cancel") == "Cancelar"
    assert i18n.t("All-in cost: {amount} {currency}", amount="1.00", currency="EUR") == "Coste total: 1.00 EUR"
    assert i18n.t("not translated {x}", x=1) == "not translated 1"  # falls back to English
    i18n.set_language("fr")  # unsupported -> English
    assert i18n.get_language() == "en"


def test_deal_verdict_is_translated():
    from pyantique_prices.deals import assess_deal

    val = {"p25": 150, "p50": 200, "p75": 260, "num_comparables": 8, "effective_comparables": 6.0,
           "valuation_available": True}
    i18n.set_language("es")
    assert "Compra muy buena" in assess_deal(val, asking_price=90)["headline"]
    i18n.set_language("en")
    assert assess_deal(val, asking_price=90)["headline"].startswith("Strong buy")


def test_saved_language_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(i18n, "_CONFIG_PATH", tmp_path / "cfg.json")
    monkeypatch.setenv("APP_LANGUAGE", "en")
    assert i18n.detect_language() == "en"
    i18n.save_language("es")
    assert i18n.detect_language() == "es"
