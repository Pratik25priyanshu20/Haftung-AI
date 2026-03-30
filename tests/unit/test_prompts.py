"""Tests for prompt templates."""
from haftung_ai.llm.prompts import (
    CAUSATION_S1_PROMPT,
    CAUSATION_S2_PROMPT,
    CAUSATION_S3_PROMPT,
    CAUSATION_SYSTEM_PROMPT,
    CONTRADICTION_PROMPT,
    EVIDENCE_EXTRACTION_PROMPT,
    REPORT_GENERATION_PROMPT,
    REPORT_SYSTEM_PROMPT,
    VALIDATION_PROMPT,
)


def test_system_prompt_exists():
    assert len(CAUSATION_SYSTEM_PROMPT) > 50
    assert "accident analyst" in CAUSATION_SYSTEM_PROMPT


def test_s1_prompt_no_legal():
    assert "No access to legal references" in CAUSATION_S1_PROMPT
    assert "{scene_description}" in CAUSATION_S1_PROMPT


def test_s2_prompt_contains_legal():
    assert "{legal_context}" in CAUSATION_S2_PROMPT
    assert "StVO" in CAUSATION_S2_PROMPT


def test_s3_prompt_contains_evidence():
    assert "{evidence_summary}" in CAUSATION_S3_PROMPT
    assert "claim" in CAUSATION_S3_PROMPT


def test_evidence_prompt():
    assert len(EVIDENCE_EXTRACTION_PROMPT) > 50
    assert "evidence extraction" in EVIDENCE_EXTRACTION_PROMPT


def test_contradiction_prompt():
    assert "{stmt_a}" in CONTRADICTION_PROMPT
    assert "{stmt_b}" in CONTRADICTION_PROMPT


def test_report_prompt():
    assert "accident report" in REPORT_GENERATION_PROMPT
    assert "Accident Sequence" in REPORT_GENERATION_PROMPT


def test_report_system_prompt():
    assert "technical expert" in REPORT_SYSTEM_PROMPT


def test_validation_prompt():
    assert "{context}" in VALIDATION_PROMPT
    assert "{analysis}" in VALIDATION_PROMPT


def test_all_prompts_are_strings():
    for prompt in [
        CAUSATION_SYSTEM_PROMPT,
        CAUSATION_S1_PROMPT,
        CAUSATION_S2_PROMPT,
        CAUSATION_S3_PROMPT,
        EVIDENCE_EXTRACTION_PROMPT,
        CONTRADICTION_PROMPT,
        REPORT_GENERATION_PROMPT,
        REPORT_SYSTEM_PROMPT,
        VALIDATION_PROMPT,
    ]:
        assert isinstance(prompt, str)
        assert len(prompt) > 20
