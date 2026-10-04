"""Tiny translation layer (English / Spanish) for the GUI.

``t("English text {name}", name=value)`` returns the text in the active
language. Unknown strings fall back to English, so a missing translation never
breaks the app. Spanish strings live in :mod:`pyantique_prices.i18n_es`.
"""

from __future__ import annotations

import json
import locale
import os
from pathlib import Path

LANGUAGES = {"en": "English", "es": "Español"}
_CONFIG_PATH = Path(os.environ.get("PYANTIQUE_PRICES_CONFIG", Path.home() / ".pyantique_prices.json"))
_current = "en"


def get_language() -> str:
    return _current


def set_language(code: str) -> str:
    """Select ``en`` or ``es`` (anything else falls back to English)."""
    global _current
    _current = code if code in LANGUAGES else "en"
    return _current


def t(text: str, /, **fields) -> str:
    """Translate ``text`` (a ``str.format`` template) and fill in ``fields``."""
    if _current != "en":
        from .i18n_es import ES

        text = ES.get(text, text)
    return text.format(**fields) if fields else text


def detect_language() -> str:
    """Saved choice > ``APP_LANGUAGE`` env var > system locale > English."""
    try:
        saved = json.loads(_CONFIG_PATH.read_text(encoding="utf-8")).get("language")
        if saved in LANGUAGES:
            return saved
    except (OSError, ValueError, AttributeError):
        pass
    env = (os.environ.get("APP_LANGUAGE") or "").strip().lower()[:2]
    if env in LANGUAGES:
        return env
    try:
        system = (locale.getlocale()[0] or os.environ.get("LANG", "")).lower()
    except Exception:  # noqa: BLE001
        system = ""
    return "es" if system.startswith("es") else "en"


def save_language(code: str) -> None:
    try:
        data = {}
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        data["language"] = code
        _CONFIG_PATH.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass  # a read-only home folder must never break the app
