"""PDF report generator using WeasyPrint + Jinja2."""
from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"


class PDFGenerator:
    """Generate German accident report PDF from structured data."""

    def __init__(self, templates_dir: Path | None = None):
        self.templates_dir = templates_dir or TEMPLATES_DIR
        self.env = Environment(loader=FileSystemLoader(str(self.templates_dir)))

    def generate(self, report_data: dict, output_path: str | Path, scene_diagram_path: str | None = None) -> Path:
        """Generate PDF from report data.

        Args:
            report_data: Structured report dictionary from ReportAgent.
            output_path: Where to save the PDF.
            scene_diagram_path: Optional path to BEV scene diagram image.

        Returns:
            Path to generated PDF.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        template = self.env.get_template("unfallbericht.html")
        html = template.render(
            report=report_data,
            scene_diagram=scene_diagram_path,
        )

        try:
            from weasyprint import HTML

            HTML(string=html).write_pdf(str(output_path))
            logger.info("PDF generated: %s", output_path)
        except (ImportError, OSError):
            # Fallback: save HTML if weasyprint or system libs not available
            html_path = output_path.with_suffix(".html")
            html_path.write_text(html, encoding="utf-8")
            logger.warning("WeasyPrint not available, saved HTML: %s", html_path)
            return html_path

        return output_path
