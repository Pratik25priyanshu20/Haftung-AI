"""JSON mode + Pydantic structured output helpers."""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel


def extract_json_object(text: str) -> dict[str, Any]:
    """Robustly extract the first JSON object from LLM text."""
    if not text:
        return {}
    text = re.sub(r"```(json)?", "", text, flags=re.IGNORECASE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    json_str = text[start : end + 1]
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {}


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """Robustly extract the first JSON array from LLM text."""
    if not text:
        return []
    text = re.sub(r"```(json)?", "", text, flags=re.IGNORECASE).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    json_str = text[start : end + 1]
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []


def parse_structured_output(text: str, model: type[BaseModel]) -> BaseModel | None:
    """Parse LLM text into a Pydantic model, returning None on failure."""
    data = extract_json_object(text)
    if not data:
        return None
    try:
        return model.model_validate(data)
    except Exception:
        return None
