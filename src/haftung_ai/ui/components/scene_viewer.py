"""Scene diagram viewer component."""
from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_scene_viewer(diagram_path: str | None):
    """Render BEV scene diagram."""
    if diagram_path and Path(diagram_path).exists():
        st.subheader("Unfallszene (Draufsicht)")
        st.image(str(diagram_path), use_container_width=True)
    else:
        st.info("Keine Unfallskizze verfügbar.")
