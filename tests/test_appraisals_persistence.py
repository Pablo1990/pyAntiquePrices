from __future__ import annotations

from sqlalchemy.exc import OperationalError

from pyantique_prices.data.appraisals import _build_persistence_warning, persist_appraisal
from pyantique_prices.data.database import create_tables, get_engine, get_session_factory


def test_persist_appraisal_saves_record_to_writable_database():
    engine = get_engine("sqlite:///:memory:")
    create_tables(engine)
    session_factory = get_session_factory(engine)

    record, warning = persist_appraisal(
        session_factory=session_factory,
        result={"request_id": "req-1", "comparables": [], "warnings": []},
        input_metadata={"currency": "EUR"},
        model_versions={"vision_model": "qwen3-vl:8b"},
    )

    assert warning is None
    assert record is not None
    assert record.request_id == "req-1"


def test_persist_appraisal_returns_warning_for_readonly_database(monkeypatch):
    engine = get_engine("sqlite:///:memory:")
    create_tables(engine)
    session_factory = get_session_factory(engine)

    def _broken_save(*args, **kwargs):  # noqa: ARG001
        raise OperationalError(
            "INSERT INTO appraisals ...",
            {},
            Exception("attempt to write a readonly database"),
        )

    monkeypatch.setattr("pyantique_prices.data.appraisals.save_appraisal", _broken_save)

    record, warning = persist_appraisal(
        session_factory=session_factory,
        result={"request_id": "req-2", "comparables": [], "warnings": []},
        input_metadata={"currency": "EUR"},
        model_versions={"vision_model": "qwen3-vl:8b"},
    )

    assert record is None
    assert warning is not None
    assert "database is read-only" in warning


def test_build_persistence_warning_handles_generic_database_error():
    warning = _build_persistence_warning(RuntimeError("disk full"))
    assert warning == "Appraisal result could not be saved to the local database: disk full"
