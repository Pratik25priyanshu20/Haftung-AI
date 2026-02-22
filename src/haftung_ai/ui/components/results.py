"""Results display component for Streamlit UI."""
from __future__ import annotations

import streamlit as st


def render_results(result: dict):
    """Render analysis results."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Unfalltyp", result.get("accident_type", "N/A"))
    col2.metric("Konfidenz", f"{result.get('confidence', 0):.1%}")
    col3.metric("Variante", result.get("variant", "N/A"))

    st.subheader("Primäre Unfallursache")
    st.write(result.get("primary_cause", "Nicht bestimmt"))

    if result.get("contributing_factors"):
        st.subheader("Beitragende Faktoren")
        for f in result["contributing_factors"]:
            st.markdown(f"- **{f.get('factor', 'N/A')}** ({f.get('severity', '')})")
