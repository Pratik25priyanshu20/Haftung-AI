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
    assert "Unfallanalytiker" in CAUSATION_SYSTEM_PROMPT


def test_s1_prompt_no_legal():
    assert "Kein Zugang" in CAUSATION_S1_PROMPT
    assert "{scene_description}" in CAUSATION_S1_PROMPT


def test_s2_prompt_contains_legal():
    assert "{legal_context}" in CAUSATION_S2_PROMPT
    assert "StVO" in CAUSATION_S2_PROMPT


def test_s3_prompt_contains_evidence():
    assert "{evidence_summary}" in CAUSATION_S3_PROMPT
    assert "Behauptung" in CAUSATION_S3_PROMPT


def test_evidence_prompt():
    assert len(EVIDENCE_EXTRACTION_PROMPT) > 50
    assert "Beweisextraktion" in EVIDENCE_EXTRACTION_PROMPT


def test_contradiction_prompt():
    assert "{stmt_a}" in CONTRADICTION_PROMPT
    assert "{stmt_b}" in CONTRADICTION_PROMPT


def test_report_prompt():
    assert "Unfallbericht" in REPORT_GENERATION_PROMPT
    assert "Unfallhergang" in REPORT_GENERATION_PROMPT


def test_report_system_prompt():
    assert "Sachverständiger" in REPORT_SYSTEM_PROMPT


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
