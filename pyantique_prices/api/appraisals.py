"""Appraisal endpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from pyantique_prices.data.appraisals import get_appraisal_by_id, save_appraisal

from .schemas import AppraiseResponse

router = APIRouter()
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _build_context(
    user_description: str | None,
    provenance: str | None,
    location: str | None,
    known_dimensions: str | None,
) -> str:
    parts = []
    if user_description:
        parts.append(f"User description: {user_description}")
    if provenance:
        parts.append(f"Provenance: {provenance}")
    if location:
        parts.append(f"Location: {location}")
    if known_dimensions:
        parts.append(f"Known dimensions: {known_dimensions}")
    return "\n".join(parts)


def _to_response(result: dict[str, Any], model_version: dict[str, str]) -> AppraiseResponse:
    identification = result.get("identification") or {}
    marks = identification.get("marks", []) if isinstance(identification, dict) else []
    condition = identification.get("condition") if isinstance(identification, dict) else None
    return AppraiseResponse(
        request_id=result["request_id"],
        identification=identification,
        marks=marks,
        condition=condition,
        comparables=result.get("comparables", []),
        valuation=result.get("valuation"),
        confidence={
            "identification_confidence": float(result.get("identification_confidence", 0.0)),
            "valuation_confidence": float(result.get("valuation_confidence", 0.0)),
        },
        evidence=result.get("evidence", []),
        warnings=result.get("warnings", []),
        model_version=model_version,
        valuation_available=bool(result.get("valuation_available", False)),
        currency=result.get("currency", "EUR"),
        candidate_count=int(result.get("candidate_count", 0)),
        usable_comparable_count=int(result.get("usable_comparable_count", 0)),
    )


@router.post("/appraise", response_model=AppraiseResponse)
async def appraise(
    request: Request,
    images: list[UploadFile] = File(...),
    currency: str | None = Form(default=None),
    location: str | None = Form(default=None),
    known_dimensions: str | None = Form(default=None),
    user_description: str | None = Form(default=None),
    provenance: str | None = Form(default=None),
) -> AppraiseResponse:
    if len(images) < 3 or len(images) > 5:
        raise HTTPException(status_code=400, detail="Please upload between 3 and 5 images.")

    temp_paths: list[Path] = []
    try:
        for upload in images:
            extension = Path(upload.filename or "").suffix.lower()
            if upload.content_type not in ALLOWED_IMAGE_TYPES and extension not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported image type. Allowed: JPEG, PNG, WebP.",
                )
            with tempfile.NamedTemporaryFile(suffix=extension or ".jpg", delete=False) as tmp:
                tmp.write(await upload.read())
                temp_paths.append(Path(tmp.name))

        appraisal_service = request.app.state.appraisal_service
        model_version = request.app.state.model_version
        context = _build_context(user_description, provenance, location, known_dimensions)
        result = appraisal_service.appraise(temp_paths, context=context, currency=currency)

        session_factory = request.app.state.session_factory
        with session_factory() as session:
            save_appraisal(
                session=session,
                result=result,
                input_metadata={
                    "currency": currency,
                    "location": location,
                    "known_dimensions": known_dimensions,
                    "user_description": user_description,
                    "provenance": provenance,
                    "num_images": len(images),
                },
                model_versions=model_version,
            )

        return _to_response(result, model_version=model_version)
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)


@router.get("/appraisals/{appraisal_id}")
def get_appraisal(appraisal_id: int, request: Request) -> dict:
    session_factory = request.app.state.session_factory
    with session_factory() as session:
        record = get_appraisal_by_id(session, appraisal_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Appraisal not found")
        return {
            "id": record.id,
            "request_id": record.request_id,
            "model_versions": record.model_versions or {},
            "input_metadata": record.input_metadata or {},
            "identification": record.identification,
            "comparable_ids": record.comparable_ids or [],
            "valuation": record.valuation,
            "calibration": record.calibration,
            "confidence": record.confidence or {},
            "warnings": record.warnings or [],
            "created_at": record.created_at.isoformat() if record.created_at else None,
        }
