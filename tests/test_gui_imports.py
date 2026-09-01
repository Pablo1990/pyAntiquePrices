from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def test_gui_module_imports_when_numpy_is_missing(monkeypatch):
    pytest.importorskip("tkinter")
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy" or name.startswith("numpy."):
            raise ModuleNotFoundError("No module named 'numpy'")
        return real_import(name, globals, locals, fromlist, level)

    sys.modules.pop("pyantique_prices.gui", None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.import_module("pyantique_prices.gui")
    assert hasattr(module, "run_gui")
