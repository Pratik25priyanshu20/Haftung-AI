"""POST /analyze — run accident analysis."""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from haftung_ai.agents.orchestrator import run_analysis

router = APIRouter()

# In-memory result store (production would use a database)
_results: dict[str, dict] = {}


@router.post("")
async def analyze_accident(
    video: UploadFile = File(...),
    can_log: UploadFile = File(...),
    variant: str = Form("S2"),
):
    """Run full accident analysis pipeline."""
    analysis_id = str(uuid.uuid4())
    tmp_dir = Path(tempfile.mkdtemp())

    video_path = tmp_dir / video.filename
    can_path = tmp_dir / can_log.filename
    with video_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    with can_path.open("wb") as f:
        shutil.copyfileobj(can_log.file, f)

    result = run_analysis(str(video_path), str(can_path), variant=variant)

    _results[analysis_id] = result

    return {
        "analysis_id": analysis_id,
        "variant": variant,
        "primary_cause": result.get("primary_cause", "Unknown"),
        "accident_type": result.get("accident_type", "unknown"),
        "confidence": result.get("confidence_score", 0.0),
        "errors": result.get("errors", []),
    }
