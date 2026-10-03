import os
from pathlib import Path
from pyantique_prices.config import load_dotenv

def test_load_dotenv(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# c\nEBAY_T1=abc\nexport EBAY_T2="q x"\nEBAY_T3=keep\n\n')
    monkeypatch.setenv("EBAY_T3", "shell")
    monkeypatch.delenv("EBAY_T1", raising=False); monkeypatch.delenv("EBAY_T2", raising=False)
    load_dotenv(f)
    assert os.environ["EBAY_T1"] == "abc"
    assert os.environ["EBAY_T2"] == "q x"
    assert os.environ["EBAY_T3"] == "shell"
