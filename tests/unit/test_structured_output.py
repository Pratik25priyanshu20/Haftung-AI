"""Tests for JSON extraction and structured output parsing."""
from pydantic import BaseModel

from haftung_ai.llm.structured_output import (
    extract_json_array,
    extract_json_object,
    parse_structured_output,
)


class SimpleModel(BaseModel):
    name: str
    value: int = 0


def test_extract_json_object_simple():
    text = '{"key": "value", "num": 42}'
    result = extract_json_object(text)
    assert result == {"key": "value", "num": 42}


def test_extract_json_object_with_code_block():
    text = '```json\n{"key": "value"}\n```'
    result = extract_json_object(text)
    assert result == {"key": "value"}


def test_extract_json_object_with_surrounding_text():
    text = 'Here is the result: {"answer": "yes"} and that is the end.'
    result = extract_json_object(text)
    assert result == {"answer": "yes"}


def test_extract_json_object_empty():
    assert extract_json_object("") == {}
    assert extract_json_object("no json here") == {}


def test_extract_json_array_simple():
    text = '[{"a": 1}, {"a": 2}]'
    result = extract_json_array(text)
    assert len(result) == 2
    assert result[0]["a"] == 1


def test_extract_json_array_with_code_block():
    text = '```json\n[{"item": "one"}]\n```'
    result = extract_json_array(text)
    assert len(result) == 1


def test_extract_json_array_empty():
    assert extract_json_array("") == []
    assert extract_json_array("no array") == []


def test_parse_structured_output_success():
    text = '{"name": "test", "value": 42}'
    result = parse_structured_output(text, SimpleModel)
    assert result is not None
    assert result.name == "test"
    assert result.value == 42


def test_parse_structured_output_missing_required():
    text = '{"value": 42}'
    result = parse_structured_output(text, SimpleModel)
    assert result is None


def test_parse_structured_output_invalid():
    text = "not json at all"
    result = parse_structured_output(text, SimpleModel)
    assert result is None


def test_extract_json_object_nested():
    text = '{"outer": {"inner": [1, 2, 3]}}'
    result = extract_json_object(text)
    assert result["outer"]["inner"] == [1, 2, 3]
