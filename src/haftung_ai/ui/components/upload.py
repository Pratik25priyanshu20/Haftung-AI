"""Upload component for Streamlit UI."""
from __future__ import annotations

import streamlit as st


def render_upload():
    """Render file upload widgets."""
    col1, col2 = st.columns(2)
    with col1:
        video = st.file_uploader("Dashcam-Video hochladen", type=["mp4", "avi", "mov"], key="video_upload")
    with col2:
        can_log = st.file_uploader("CAN-Bus-Log hochladen", type=["csv", "asc", "blf"], key="can_upload")
    return video, can_log
