"""Helpers for storing and loading appraisals."""

from __future__ import annotations

from typing import Any

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


def get_appraisal_by_id(session, appraisal_id: int) -> AppraisalRecord | None:
    return session.query(AppraisalRecord).filter_by(id=appraisal_id).first()
