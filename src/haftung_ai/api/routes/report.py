"""GET /report/{id} — retrieve analysis report."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()

# Shared with analyze module
from haftung_ai.api.routes.analyze import _results


@router.get("/{analysis_id}")
def get_report(analysis_id: str):
    """Retrieve analysis results by ID."""
    if analysis_id not in _results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    result = _results[analysis_id]
    return {
        "analysis_id": analysis_id,
        "report": result.get("report", {}),
        "causation": result.get("causation_output", {}),
        "confidence": result.get("confidence_score"),
        "validation": result.get("validation_details"),
    }
