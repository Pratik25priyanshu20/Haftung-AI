"""Haftung_AI Demo Dashboard — S1 vs S2 vs S3 portfolio showcase."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
SCENARIOS_DIR = Path(__file__).resolve().parents[3] / "evaluation" / "dataset" / "scenarios"

VARIANT_COLORS = {"S1": "#FF6B6B", "S2": "#4ECDC4", "S3": "#45B7D1"}
VARIANT_LABELS = {
    "S1": "No RAG (Baseline)",
    "S2": "RAG (StVO/Case Law)",
    "S3": "RAG + Constraints",
}
CATEGORY_LABELS = {
    "rear_end": "Rear-End Collision",
    "side_collision": "Side Collision",
    "head_on": "Head-On Collision",
    "intersection": "Intersection",
    "pedestrian": "Pedestrian",
    "single_vehicle": "Single Vehicle",
}


# ── Data Loading ───────────────────────────────────────────────────────────
@st.cache_data
def load_scenarios() -> dict[str, list[dict]]:
    """Load all scenario JSON files, grouped by category."""
    grouped: dict[str, list[dict]] = {}
    if not SCENARIOS_DIR.exists():
        return grouped
    for path in sorted(SCENARIOS_DIR.glob("*.json")):
        with open(path) as f:
            scenario = json.load(f)
        cat = scenario.get("category", "unknown")
        grouped.setdefault(cat, []).append(scenario)
    return grouped


def get_scenario_label(scenario: dict) -> str:
    """Human-readable label for scenario dropdown."""
    sid = scenario.get("scenario_id", "unknown")
    text = scenario.get("scenario_text", "")
    preview = text[:60] + "..." if len(text) > 60 else text
    return f"{sid} — {preview}"


# ── Execution ──────────────────────────────────────────────────────────────
def run_analysis_safe(text: str, variant: str) -> tuple[dict | None, str | None]:
    """Run pipeline with error handling. Returns (result, error_message)."""
    try:
        from haftung_ai.agents.orchestrator import run_text_analysis

        result = run_text_analysis(text, variant=variant)
        if not result or not result.get("causation_output"):
            return result, "Analysis returned empty causation results."
        return result, None
    except Exception as e:
        logger.exception("Analysis failed for %s", variant)
        return None, f"{type(e).__name__}: {e}"


def check_service_health() -> dict[str, bool]:
    """Check connectivity for Groq API and Qdrant."""
    health: dict[str, bool] = {"groq": False, "qdrant": False}

    # Groq: check if API key is set
    try:
        from haftung_ai.config.settings import get_settings

        settings = get_settings()
        health["groq"] = bool(settings.GROQ_API_KEY)
    except Exception:
        pass

    # Qdrant: try connecting
    try:
        from qdrant_client import QdrantClient

        from haftung_ai.config.settings import get_settings

        settings = get_settings()
        client = QdrantClient(url=settings.QDRANT_URL, timeout=3)
        client.get_collections()
        health["qdrant"] = True
    except Exception:
        pass

    return health


# ── Visualizations ─────────────────────────────────────────────────────────
def render_confidence_gauge(score: float | None, label: str = "Confidence") -> None:
    """Render a confidence score as colored progress bar."""
    if score is None:
        st.caption(f"{label}: N/A")
        return
    score = max(0.0, min(1.0, float(score)))
    color = "normal" if score >= 0.7 else ("off" if score >= 0.4 else "off")
    st.caption(label)
    st.progress(score, text=f"{score:.0%}")


def render_responsibility_pie(responsibility: list[dict], title: str = "") -> None:
    """Render a matplotlib pie chart for responsibility distribution."""
    if not responsibility:
        st.info("No responsibility data.")
        return

    labels = [r.get("party", "?") for r in responsibility]
    sizes = [r.get("percentage", 0) for r in responsibility]

    if sum(sizes) == 0:
        st.info("No responsibility percentages.")
        return

    fig, ax = plt.subplots(figsize=(3, 3))
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.0f%%", colors=colors, startangle=90,
        textprops={"fontsize": 8},
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_responsibility_comparison(results: dict[str, dict]) -> None:
    """Grouped bar chart comparing responsibility across variants."""
    all_parties: list[str] = []
    for v, state in results.items():
        causation = state.get("causation_output", {})
        for r in causation.get("responsibility", []):
            party = r.get("party", "?")
            if party not in all_parties:
                all_parties.append(party)

    if not all_parties:
        st.info("No responsibility data to compare.")
        return

    variants = list(results.keys())
    x = np.arange(len(all_parties))
    width = 0.8 / len(variants)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, v in enumerate(variants):
        causation = results[v].get("causation_output", {})
        resp_map = {r.get("party", "?"): r.get("percentage", 0) for r in causation.get("responsibility", [])}
        values = [resp_map.get(p, 0) for p in all_parties]
        offset = (i - len(variants) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=f"{v}: {VARIANT_LABELS.get(v, v)}", color=VARIANT_COLORS.get(v, "#999"))
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.0f}%",
                        ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Responsibility %")
    ax.set_xticks(x)
    ax.set_xticklabels(all_parties, fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.set_title("Responsibility Distribution by Variant", fontsize=11, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_variant_badge(variant: str) -> str:
    """Return colored markdown badge for a variant."""
    color = VARIANT_COLORS.get(variant, "#999")
    label = VARIANT_LABELS.get(variant, variant)
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-weight:bold;font-size:0.85em">{variant}: {label}</span>'


# ── Tab Renderers ──────────────────────────────────────────────────────────
def render_analysis_tab(results: dict[str, dict]) -> None:
    """Tab 1: Side-by-side analysis results per variant."""
    cols = st.columns(len(results))
    for col, (variant, state) in zip(cols, results.items()):
        with col:
            st.markdown(render_variant_badge(variant), unsafe_allow_html=True)
            st.markdown("---")

            causation = state.get("causation_output", {})
            if not causation:
                st.warning("No causation output.")
                continue

            # Accident type
            st.metric("Accident Type", causation.get("accident_type", "N/A"))

            # Confidence
            confidence = state.get("confidence_score") or causation.get("confidence")
            render_confidence_gauge(confidence)

            # Primary cause
            st.markdown("**Primary Cause**")
            st.info(causation.get("primary_cause", "N/A"))

            # Responsibility pie
            render_responsibility_pie(causation.get("responsibility", []))

            # Contributing factors
            factors = causation.get("contributing_factors", [])
            if factors:
                st.markdown("**Contributing Factors**")
                for f in factors:
                    if isinstance(f, dict):
                        severity = f.get("severity", "")
                        icon = {"primary": "**[P]**", "secondary": "[S]", "minor": "[m]"}.get(severity, "[-]")
                        ref = f" _{f['legal_reference']}_" if f.get("legal_reference") else ""
                        st.markdown(f"- {icon} {f.get('factor', 'N/A')}{ref}")

            # Errors/warnings
            for err in state.get("errors", []):
                st.error(err)
            for warn in state.get("warnings", []):
                st.warning(warn)


def render_comparison_tab(results: dict[str, dict], ground_truth: dict | None) -> None:
    """Tab 2: Summary comparison table + charts + deltas."""
    # Summary table
    rows = []
    for variant, state in results.items():
        causation = state.get("causation_output", {})
        confidence = state.get("confidence_score") or causation.get("confidence")
        claims = causation.get("claims", [])
        chunks = state.get("retrieved_chunks", [])
        rows.append({
            "Variant": variant,
            "Primary Cause": causation.get("primary_cause", "N/A"),
            "Accident Type": causation.get("accident_type", "N/A"),
            "Confidence": f"{confidence:.0%}" if confidence else "N/A",
            "Claims": len(claims),
            "RAG Chunks": len(chunks),
        })

    if ground_truth:
        rows.append({
            "Variant": "Ground Truth",
            "Primary Cause": ground_truth.get("primary_cause", "N/A"),
            "Accident Type": ground_truth.get("accident_type", "N/A"),
            "Confidence": "-",
            "Claims": len(ground_truth.get("expected_claims", [])),
            "RAG Chunks": "-",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Responsibility comparison chart
    st.markdown("#### Responsibility Comparison")
    render_responsibility_comparison(results)

    # Delta indicators
    variants = list(results.keys())
    if len(variants) >= 2:
        st.markdown("#### Variant Deltas")
        pairs = []
        if "S1" in results and "S2" in results:
            pairs.append(("S2", "S1"))
        if "S2" in results and "S3" in results:
            pairs.append(("S3", "S2"))

        for newer, older in pairs:
            c_new = results[newer].get("confidence_score") or results[newer].get("causation_output", {}).get("confidence")
            c_old = results[older].get("confidence_score") or results[older].get("causation_output", {}).get("confidence")

            chunks_new = len(results[newer].get("retrieved_chunks", []))
            chunks_old = len(results[older].get("retrieved_chunks", []))

            cols = st.columns(3)
            with cols[0]:
                st.markdown(f"**{newer} vs {older}**")
            with cols[1]:
                if c_new is not None and c_old is not None:
                    delta = c_new - c_old
                    st.metric("Confidence", f"{c_new:.0%}", delta=f"{delta:+.0%}")
                else:
                    st.metric("Confidence", "N/A")
            with cols[2]:
                st.metric("RAG Chunks", chunks_new, delta=chunks_new - chunks_old)


def render_evidence_tab(results: dict[str, dict]) -> None:
    """Tab 3: RAG evidence, legal refs, claims, contradictions."""
    for variant, state in results.items():
        st.markdown(render_variant_badge(variant), unsafe_allow_html=True)

        if variant == "S1":
            st.info("S1 (No RAG) does not retrieve external evidence.")
            st.markdown("---")
            continue

        # Retrieved chunks
        chunks = state.get("retrieved_chunks", [])
        if chunks:
            st.markdown(f"**Retrieved Chunks** ({len(chunks)})")
            for i, chunk in enumerate(chunks):
                with st.expander(f"Chunk {i+1} — score: {chunk.get('score', 'N/A'):.3f}" if isinstance(chunk.get("score"), (int, float)) else f"Chunk {i+1}"):
                    st.code(chunk.get("content", "N/A"), language=None)
                    meta = chunk.get("metadata", {})
                    if meta:
                        st.caption(f"Source: {meta.get('source', 'N/A')} | Section: {meta.get('section', 'N/A')}")
        else:
            st.warning("No chunks retrieved.")

        # Legal references from causation
        causation = state.get("causation_output", {})
        legal_refs = causation.get("legal_references", [])
        if legal_refs:
            st.markdown("**Legal References**")
            for ref in legal_refs:
                st.markdown(f"- {ref}")

        # Claims table
        claims = causation.get("claims", [])
        if claims:
            st.markdown(f"**Claims** ({len(claims)})")
            claims_data = []
            for c in claims:
                if isinstance(c, dict):
                    claims_data.append({
                        "Statement": c.get("statement", "N/A"),
                        "Source": c.get("source_type", "N/A"),
                        "Confidence": f"{c['confidence']:.0%}" if isinstance(c.get("confidence"), (int, float)) else "N/A",
                    })
            if claims_data:
                st.dataframe(pd.DataFrame(claims_data), use_container_width=True, hide_index=True)

        # Contradictions
        contradictions = state.get("contradictions", [])
        if state.get("has_contradictions") and contradictions:
            st.markdown("**Contradictions Detected**")
            for contradiction in contradictions:
                if isinstance(contradiction, dict):
                    st.warning(
                        f"**{contradiction.get('type', 'Contradiction')}**: "
                        f"{contradiction.get('description', contradiction.get('statement_a', 'N/A'))} "
                        f"vs {contradiction.get('statement_b', '')}"
                    )

        st.markdown("---")


def render_ground_truth_tab(results: dict[str, dict], ground_truth: dict | None) -> None:
    """Tab 4: Match/mismatch against ground truth."""
    if not ground_truth:
        st.info("Ground truth is only available for pre-loaded scenarios.")
        return

    gt_cause = ground_truth.get("primary_cause", "")
    gt_type = ground_truth.get("accident_type", "")
    gt_responsibility = {r["party"]: r["percentage"] for r in ground_truth.get("responsibility", [])}
    gt_stvo = set(ground_truth.get("relevant_stvo", []))
    gt_claims = ground_truth.get("expected_claims", [])

    st.markdown("#### Ground Truth Reference")
    st.markdown(f"**Primary Cause:** {gt_cause}")
    st.markdown(f"**Accident Type:** {gt_type}")
    st.markdown(f"**Relevant StVO:** {', '.join(gt_stvo) if gt_stvo else 'N/A'}")

    st.markdown("---")

    for variant, state in results.items():
        st.markdown(render_variant_badge(variant), unsafe_allow_html=True)
        causation = state.get("causation_output", {})

        # Primary cause match
        pred_cause = causation.get("primary_cause", "")
        cause_match = gt_cause.lower() in pred_cause.lower() or pred_cause.lower() in gt_cause.lower()
        if cause_match:
            st.success(f"Primary cause MATCH: {pred_cause}")
        else:
            st.error(f"Primary cause MISMATCH: predicted '{pred_cause}' vs expected '{gt_cause}'")

        # Accident type match
        pred_type = causation.get("accident_type", "")
        if pred_type == gt_type:
            st.success(f"Accident type MATCH: {pred_type}")
        else:
            st.error(f"Accident type MISMATCH: predicted '{pred_type}' vs expected '{gt_type}'")

        # Responsibility comparison
        pred_resp = {r.get("party", "?"): r.get("percentage", 0) for r in causation.get("responsibility", [])}
        if pred_resp:
            resp_cols = st.columns(len(pred_resp) + 1)
            resp_cols[0].markdown("**Party**")
            for i, (party, pct) in enumerate(pred_resp.items()):
                gt_pct = gt_responsibility.get(party)
                if gt_pct is not None:
                    delta = pct - gt_pct
                    resp_cols[i + 1].metric(party, f"{pct:.0f}%", delta=f"{delta:+.0f}pp" if delta != 0 else "exact")
                else:
                    resp_cols[i + 1].metric(party, f"{pct:.0f}%", delta="no GT")

        # StVO coverage
        pred_refs = set(causation.get("legal_references", []))
        chunks = state.get("retrieved_chunks", [])
        chunk_text = " ".join(c.get("content", "") for c in chunks if isinstance(c, dict))
        retrieved_stvo = {s for s in gt_stvo if s in chunk_text or s in " ".join(pred_refs)}

        if gt_stvo:
            covered = len(retrieved_stvo)
            total = len(gt_stvo)
            st.markdown(f"**StVO Coverage:** {covered}/{total} relevant laws referenced")
            for s in gt_stvo:
                if s in retrieved_stvo:
                    st.markdown(f"- {s}")
                else:
                    st.markdown(f"- ~~{s}~~ (missing)")

        # Expected claims coverage
        if gt_claims:
            pred_text = " ".join([
                causation.get("primary_cause", ""),
                causation.get("reasoning", ""),
                " ".join(c.get("statement", "") for c in causation.get("claims", []) if isinstance(c, dict)),
            ]).lower()
            st.markdown(f"**Expected Claims Coverage**")
            for claim in gt_claims:
                # Simple keyword overlap check
                keywords = [w for w in claim.lower().split() if len(w) > 4]
                matched = sum(1 for kw in keywords if kw in pred_text)
                ratio = matched / len(keywords) if keywords else 0
                if ratio >= 0.5:
                    st.markdown(f"- {claim}")
                else:
                    st.markdown(f"- ~~{claim}~~ (not covered)")

        st.markdown("---")


def render_report_tab(results: dict[str, dict]) -> None:
    """Tab 5: German Unfallbericht from each variant."""
    for variant, state in results.items():
        report = state.get("report", {})
        if not report:
            st.info(f"{variant}: No report generated.")
            continue

        with st.expander(f"{variant} — Unfallbericht", expanded=len(results) == 1):
            if isinstance(report, dict):
                for section, content in report.items():
                    st.markdown(f"**{section.replace('_', ' ').title()}**")
                    if isinstance(content, str):
                        st.write(content)
                    elif isinstance(content, list):
                        for item in content:
                            st.write(f"- {item}")
                    elif isinstance(content, dict):
                        for k, v in content.items():
                            st.write(f"**{k}:** {v}")
                    st.markdown("")
            else:
                st.write(report)


# ── Sidebar ────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple[str, list[str], dict | None, bool]:
    """Render sidebar controls. Returns (scenario_text, variants, ground_truth, is_preloaded)."""
    with st.sidebar:
        st.title("Haftung_AI Demo")

        # ── Scenario Selection ──
        st.markdown("### Scenario Selection")
        scenarios = load_scenarios()

        use_custom = st.checkbox("Custom text input")

        scenario_text = ""
        ground_truth = None
        is_preloaded = False

        if use_custom:
            scenario_text = st.text_area(
                "Accident description (German)",
                height=200,
                placeholder="Am 15. März 2024 gegen 08:15 Uhr ereignete sich auf der B27...",
            )
        else:
            if not scenarios:
                st.warning(f"No scenarios found in {SCENARIOS_DIR}")
            else:
                categories = list(scenarios.keys())
                category = st.selectbox(
                    "Category",
                    categories,
                    format_func=lambda c: CATEGORY_LABELS.get(c, c),
                )
                scenario_list = scenarios.get(category, [])
                if scenario_list:
                    selected = st.selectbox(
                        "Scenario",
                        scenario_list,
                        format_func=get_scenario_label,
                    )
                    if selected:
                        scenario_text = selected.get("scenario_text", "")
                        ground_truth = selected.get("ground_truth")
                        is_preloaded = True

        # ── Variants ──
        st.markdown("### Variants")
        col1, col2, col3 = st.columns(3)
        s1 = col1.checkbox("S1", value=True)
        s2 = col2.checkbox("S2", value=True)
        s3 = col3.checkbox("S3", value=True)
        variants = [v for v, on in [("S1", s1), ("S2", s2), ("S3", s3)] if on]

        st.caption(
            "**S1** No RAG | **S2** RAG | **S3** RAG + Constraints"
        )

        # ── Run Button ──
        run_clicked = st.button(
            "Run Analysis",
            type="primary",
            disabled=not scenario_text or not variants,
            use_container_width=True,
        )

        if run_clicked:
            st.session_state.run_requested = True

        # ── Service Health ──
        st.markdown("### Status")
        health = check_service_health()
        st.markdown(
            f"Groq API: {'🟢' if health['groq'] else '🔴'} &nbsp;&nbsp; "
            f"Qdrant: {'🟢' if health['qdrant'] else '🔴'}",
            unsafe_allow_html=True,
        )
        if not health["groq"]:
            st.caption("Set GROQ_API_KEY in .env")
        if not health["qdrant"]:
            st.caption("S2/S3 need Qdrant — run `make docker-up`")

        # ── About ──
        with st.expander("About"):
            st.markdown(
                "**Haftung_AI** — LLM-powered accident causation analysis.\n\n"
                "Three system variants compare how RAG retrieval and "
                "constraint enforcement affect causation accuracy.\n\n"
                "- **S1**: Pure LLM inference (baseline)\n"
                "- **S2**: RAG-augmented with StVO & case law\n"
                "- **S3**: S2 + claim-level validation loop\n\n"
                "30 German accident scenarios across 6 categories. "
                "Built with LangGraph, Groq (LLaMA 3.3 70B), Qdrant, "
                "and sentence-transformers."
            )

    return scenario_text, variants, ground_truth, is_preloaded


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Haftung_AI Demo",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Haftung_AI — Accident Causation Analysis")
    st.caption("LLM-powered S1 vs S2 vs S3 comparison for German traffic accidents")

    scenario_text, variants, ground_truth, is_preloaded = render_sidebar()

    # ── Run Analysis ──
    if st.session_state.get("run_requested"):
        st.session_state.run_requested = False
        results: dict[str, dict] = {}

        progress = st.progress(0, text="Starting analysis...")
        for i, variant in enumerate(variants):
            progress.progress(
                (i) / len(variants),
                text=f"Running {variant} ({VARIANT_LABELS.get(variant, '')})...",
            )
            result, error = run_analysis_safe(scenario_text, variant)
            if error:
                st.error(f"**{variant}** failed: {error}")
            if result:
                results[variant] = result

        progress.progress(1.0, text="Done!")

        if results:
            st.session_state.results = results
            st.session_state.ground_truth = ground_truth
            st.session_state.is_preloaded = is_preloaded
            st.session_state.last_text = scenario_text

    # ── Display Results ──
    results = st.session_state.get("results")
    if not results:
        st.markdown("---")
        st.markdown(
            "Select a scenario and click **Run Analysis** to see results.\n\n"
            "**Tip:** Start with S1 only — it works without Qdrant. "
            "Add S2/S3 once services are running."
        )
        return

    ground_truth = st.session_state.get("ground_truth")
    is_preloaded = st.session_state.get("is_preloaded", False)

    # Show scenario text
    with st.expander("Scenario Text", expanded=False):
        st.write(st.session_state.get("last_text", ""))

    # Tabs
    tab_names = ["Analysis", "Comparison", "Evidence & RAG", "Ground Truth", "Report"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_analysis_tab(results)

    with tabs[1]:
        render_comparison_tab(results, ground_truth)

    with tabs[2]:
        render_evidence_tab(results)

    with tabs[3]:
        render_ground_truth_tab(results, ground_truth if is_preloaded else None)

    with tabs[4]:
        render_report_tab(results)


if __name__ == "__main__":
    main()
