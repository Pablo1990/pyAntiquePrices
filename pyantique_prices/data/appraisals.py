"""Helpers for storing and loading appraisals."""

from __future__ import annotations

from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from .models import AppraisalRecord


def save_appraisal(
    session,
    result: dict[str, Any],
    input_metadata: dict[str, Any],
    model_versions: dict[str, Any],
) -> AppraisalRecord:
    record = AppraisalRecord(
        request_id=result["request_id"],
        model_versions=model_versions,
        input_metadata=input_metadata,
        identification=result.get("identification"),
        comparable_ids=[item.get("id") for item in result.get("comparables", [])],
        valuation=result.get("valuation"),
        calibration=result.get("calibration"),
        confidence={
            "identification_confidence": result.get("identification_confidence"),
            "valuation_confidence": result.get("valuation_confidence"),
        },
        warnings=result.get("warnings", []),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def persist_appraisal(
    session_factory,
    result: dict[str, Any],
    input_metadata: dict[str, Any],
    model_versions: dict[str, Any],
) -> tuple[AppraisalRecord | None, str | None]:
    session = session_factory()
    try:
        record = save_appraisal(
            session=session,
            result=result,
            input_metadata=input_metadata,
            model_versions=model_versions,
        )
        return record, None
    except SQLAlchemyError as exc:
        session.rollback()
        return None, _build_persistence_warning(exc)
    finally:
        session.close()


def _build_persistence_warning(exc: Exception) -> str:
    message = str(exc).strip() or type(exc).__name__
    lowered = message.lower()
    if "readonly" in lowered:
        return (
            "Appraisal result could not be saved because the configured SQLite "
            "database is read-only. Check DATABASE_URL permissions or choose a "
            "writable location."
        )
    return f"Appraisal result could not be saved to the local database: {message}"


def get_appraisal_by_id(session, appraisal_id: int) -> AppraisalRecord | None:
    return session.query(AppraisalRecord).filter_by(id=appraisal_id).first()
