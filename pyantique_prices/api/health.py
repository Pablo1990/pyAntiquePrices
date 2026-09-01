"""Health and model metadata endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/models")
def models(request: Request) -> dict[str, str]:
    model_version = getattr(request.app.state, "model_version", {})
    return {
        "vision_model": model_version.get("vision_model", ""),
        "pricing_model": model_version.get("pricing_model", ""),
    }
