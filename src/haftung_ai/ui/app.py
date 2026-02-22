"""Streamlit frontend for Haftung_AI."""
from __future__ import annotations

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Haftung_AI", page_icon="🚗", layout="wide")
st.title("Haftung_AI — Unfallursachenanalyse")
st.markdown("Automatisierte Analyse von Dashcam-Videos und CAN-Bus-Daten")

# Sidebar
with st.sidebar:
    st.header("Konfiguration")
    variant = st.selectbox("Systemvariante", ["S1", "S2", "S3"], index=1)
    st.markdown("""
    - **S1**: Ohne RAG (Baseline)
    - **S2**: Mit RAG (StVO/Rechtsprechung)
    - **S3**: RAG + Constraints (vollständige Evidenzprüfung)
    """)

# Upload
col1, col2 = st.columns(2)
with col1:
    video_file = st.file_uploader("Dashcam-Video", type=["mp4", "avi", "mov"])
with col2:
    can_file = st.file_uploader("CAN-Bus-Log", type=["csv", "asc", "blf"])

if st.button("Analyse starten", type="primary", disabled=not (video_file and can_file)):
    with st.spinner("Analyse läuft..."):
        try:
            files = {
                "video": (video_file.name, video_file, "video/mp4"),
                "can_log": (can_file.name, can_file, "text/csv"),
            }
            data = {"variant": variant}
            response = requests.post(f"{API_URL}/analyze", files=files, data=data, timeout=300)
            response.raise_for_status()
            result = response.json()

            st.success(f"Analyse abgeschlossen (ID: {result['analysis_id']})")

            # Results
            col1, col2, col3 = st.columns(3)
            col1.metric("Unfalltyp", result.get("accident_type", "N/A"))
            col2.metric("Konfidenz", f"{result.get('confidence', 0):.1%}")
            col3.metric("Variante", variant)

            st.subheader("Primäre Unfallursache")
            st.write(result.get("primary_cause", "Nicht bestimmt"))

            if result.get("errors"):
                st.error("Fehler: " + ", ".join(result["errors"]))

            # Fetch full report
            report_response = requests.get(f"{API_URL}/report/{result['analysis_id']}")
            if report_response.ok:
                report_data = report_response.json()
                with st.expander("Vollständiger Bericht"):
                    st.json(report_data.get("report", {}))
                with st.expander("Causation Details"):
                    st.json(report_data.get("causation", {}))

        except requests.exceptions.ConnectionError:
            st.error("API nicht erreichbar. Bitte starten Sie den Server mit: uvicorn haftung_ai.api.main:app")
        except Exception as e:
            st.error(f"Fehler: {e}")
