"""Tests for PDF generator."""

from haftung_ai.report.pdf_generator import PDFGenerator


def test_generator_init():
    gen = PDFGenerator()
    assert gen.templates_dir is not None


def test_custom_templates_dir(tmp_path):
    gen = PDFGenerator(templates_dir=tmp_path)
    assert gen.templates_dir == tmp_path


def test_generate_produces_output(tmp_path):
    """Test that generate creates either a PDF or HTML fallback."""
    gen = PDFGenerator()
    report_data = {
        "metadata": {
            "title": "Test Report",
            "date": "2024-01-01",
            "location": "Berlin",
            "case_number": "HAF-001",
        },
        "sections": [
            {"title": "Zusammenfassung", "content": "Test content", "section_type": "summary"},
        ],
        "primary_cause": "Test cause",
        "accident_type": "rear_end",
        "responsibility": [
            {"party": "ego", "percentage": 30.0, "rationale": "Test"},
        ],
        "contributing_factors": [],
        "confidence_score": 0.85,
        "variant": "S2",
    }
    output = tmp_path / "test_report.pdf"
    result = gen.generate(report_data, output)
    # Either PDF or HTML fallback should exist
    assert result.exists()
