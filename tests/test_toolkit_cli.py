from __future__ import annotations

import csv
import io
import json

import pytest

from pyantique_prices.__main__ import _format_object_result, main
from pyantique_prices.toolkit_cli import SALES_TEMPLATE_HEADER, SALES_TEMPLATE_ROW


@pytest.fixture(autouse=True)
def _db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'cli.db'}")


def test_links_command_prints_urls(capsys):
    assert main(["links", "--object", "pocket watch", "--maker", "Omega"]) == 0
    out = capsys.readouterr().out
    assert "RESEARCH LINKS" in out and "LH_Sold=1" in out and "Omega" in out


def test_links_json_and_missing_input(capsys):
    assert main(["links", "--object", "vase", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["links"]
    assert main(["links"]) == 1


def test_deal_command_verdicts_and_exit_codes(capsys):
    assert main(["deal", "--asking", "100", "--p25", "200", "--p50", "300", "--p75", "420"]) == 0
    assert "STRONG BUY" in capsys.readouterr().out
    assert main(["deal", "--asking", "100"]) == 1  # no valuation -> error


def test_deal_reads_appraisal_json(tmp_path, capsys):
    f = tmp_path / "a.json"
    f.write_text(json.dumps({"valuation": {"p25": 200, "p50": 300, "p75": 420, "num_comparables": 10,
                                           "effective_comparables": 7, "valuation_available": True}}))
    assert main(["deal", "--asking", "400", "--appraisal", str(f), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "overpriced"


def test_ledger_round_trip_via_cli(tmp_path, capsys):
    assert main(["ledger", "add", "Omega watch", "--price", "100", "--costs", "10", "--est-mid", "200"]) == 0
    assert main(["ledger", "sell", "1", "--price", "250", "--fees", "20"]) == 0
    assert "profit +120.00" in capsys.readouterr().out
    assert main(["ledger", "list"]) == 0
    assert "Omega watch" in capsys.readouterr().out
    assert main(["ledger", "summary"]) == 0
    assert json.loads(capsys.readouterr().out)["realised_profit"] == 120.0
    out = tmp_path / "mine.csv"
    assert main(["ledger", "export", str(out)]) == 0
    assert list(csv.DictReader(out.open()))[0]["final_price"] == "250.0"
    assert main(["ledger", "sync"]) == 0
    assert main(["ledger", "sell", "1", "--price", "1"]) == 1  # already sold
    assert main(["ledger", "sell", "99", "--price", "1"]) == 1  # unknown id


def test_template_is_importable(tmp_path, capsys):
    from pyantique_prices.data.database import create_tables, get_engine, get_session_factory
    from pyantique_prices.data.importer import import_csv

    assert main(["template"]) == 0
    text = capsys.readouterr().out
    assert text.splitlines()[0] == SALES_TEMPLATE_HEADER
    path = tmp_path / "t.csv"
    path.write_text(text)
    assert len(next(csv.reader(io.StringIO(SALES_TEMPLATE_ROW)))) == len(SALES_TEMPLATE_HEADER.split(","))
    engine = get_engine("sqlite:///:memory:")
    create_tables(engine)
    with get_session_factory(engine)() as s:
        assert import_csv(path, s).rows_inserted == 1


def test_object_report_includes_deal_and_links():
    result = {"identification": {"object_type": "vase"}, "currency": "EUR",
              "deal": {"verdict": "fair", "headline": "Fair price: ok."},
              "lookup_links": {"query": "vase", "links": [{"name": "eBay - sold", "url": "https://x/y"}]}}
    text = _format_object_result(result)
    assert "DEAL CHECK: FAIR PRICE" in text and "RESEARCH LINKS" in text and "https://x/y" in text
