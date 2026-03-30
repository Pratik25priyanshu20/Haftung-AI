"""Haftung_AI — Automotive Accident Causation Intelligence Platform."""
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

VARIANT_COLORS = {"S1": "#E74C3C", "S2": "#00BCD4", "S3": "#2ECC71"}
VARIANT_LABELS = {
    "S1": "Baseline",
    "S2": "RAG-Augmented",
    "S3": "RAG + Validation",
}
VARIANT_DESCRIPTIONS = {
    "S1": "Pure LLM inference from scene data",
    "S2": "Augmented with StVO & case law retrieval",
    "S3": "Full pipeline with claim validation loop",
}
CATEGORY_LABELS = {
    "rear_end": "Rear-End Collision",
    "side_collision": "Side Collision",
    "head_on": "Head-On Collision",
    "intersection": "Intersection Incident",
    "pedestrian": "Pedestrian Involved",
    "single_vehicle": "Single Vehicle",
}

# German → English translations for display
DE_EN_MAP = {
    "fahrzeug a": "Vehicle A",
    "fahrzeug b": "Vehicle B",
    "fahrzeug c": "Vehicle C",
    "fahrzeug_a": "Vehicle A",
    "fahrzeug_b": "Vehicle B",
    "fahrzeug_c": "Vehicle C",
    "ego": "Ego Vehicle",
    "ego_vehicle": "Ego Vehicle",
    "other_1": "Other Vehicle 1",
    "other_2": "Other Vehicle 2",
    "other": "Other Vehicle",
    "fussgänger": "Pedestrian",
    "fußgänger": "Pedestrian",
    "fussgaenger": "Pedestrian",
    "pedestrian": "Pedestrian",
    "radfahrer": "Cyclist",
    "motorrad": "Motorcycle",
    "lkw": "Truck",
    "seitenkollision": "Side Collision",
    "auffahrunfall": "Rear-End Collision",
    "frontalzusammenstoss": "Head-On Collision",
    "kreuzungsunfall": "Intersection Collision",
}


def translate_de(text: str) -> str:
    """Translate known German terms to English for display."""
    if not text:
        return text
    low = text.lower().strip()
    if low in DE_EN_MAP:
        return DE_EN_MAP[low]
    # Try partial matching for compound terms
    result = text
    for de, en in DE_EN_MAP.items():
        if de in result.lower():
            # Case-insensitive replace
            idx = result.lower().find(de)
            result = result[:idx] + en + result[idx + len(de):]
    return result


# ── Theme & Styling ───────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-primary: #050508;
        --bg-card: #0c0c10;
        --bg-card-hover: #101016;
        --border-subtle: rgba(255,255,255,0.04);
        --border-hover: rgba(0,188,212,0.2);
        --text-primary: #f0f0f5;
        --text-secondary: rgba(255,255,255,0.55);
        --text-muted: rgba(255,255,255,0.3);
        --accent: #00BCD4;
        --accent-glow: rgba(0,188,212,0.15);
        --success: #00E676;
        --warning: #FFB300;
        --danger: #FF5252;
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }

    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1440px;
    }

    /* ── Hero Header ── */
    .hero-header {
        background: linear-gradient(135deg, #08080c 0%, #0d1117 40%, #0f1923 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.75rem 2.5rem;
        margin-bottom: 1.25rem;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(0,188,212,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-header::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,188,212,0.15), transparent);
    }
    .hero-title {
        font-family: var(--font-sans);
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.5px;
        margin: 0 0 0.2rem 0;
    }
    .hero-subtitle {
        font-family: var(--font-sans);
        font-size: 0.82rem;
        font-weight: 400;
        color: var(--text-secondary);
        margin: 0;
    }
    .hero-badges {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.75rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(0,188,212,0.08);
        color: var(--accent);
        font-family: var(--font-mono);
        font-size: 0.65rem;
        font-weight: 500;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid rgba(0,188,212,0.12);
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .hero-badge-alt {
        background: rgba(255,255,255,0.03);
        color: var(--text-muted);
        border-color: var(--border-subtle);
    }

    /* ── Cards ── */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.6rem;
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        border-color: var(--border-hover);
        background: var(--bg-card-hover);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: transparent;
        transition: background 0.2s ease;
    }
    .metric-card:hover::before {
        background: var(--accent);
    }
    .metric-label {
        font-family: var(--font-mono);
        font-size: 0.65rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-family: var(--font-sans);
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.3;
    }
    .metric-value-sm {
        font-family: var(--font-sans);
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.3;
    }
    .metric-delta-pos { color: var(--success); font-size: 0.78rem; font-weight: 600; }
    .metric-delta-neg { color: var(--danger); font-size: 0.78rem; font-weight: 600; }

    /* ── Variant Header ── */
    .variant-header {
        background: linear-gradient(135deg, var(--bg-card), #101018);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 0.9rem 1.15rem;
        margin-bottom: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .variant-indicator {
        width: 36px; height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: var(--font-mono);
        font-size: 0.75rem;
        font-weight: 700;
        color: white;
        flex-shrink: 0;
    }
    .variant-name {
        font-family: var(--font-sans);
        font-size: 0.9rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .variant-desc {
        font-family: var(--font-sans);
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 1px;
    }

    /* ── Confidence ── */
    .conf-bar-outer {
        background: rgba(255,255,255,0.03);
        border-radius: 6px;
        height: 6px;
        overflow: hidden;
        margin-top: 0.4rem;
    }
    .conf-bar-inner {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .conf-high { background: linear-gradient(90deg, #00C853, #00E676); box-shadow: 0 0 12px rgba(0,200,83,0.2); }
    .conf-med { background: linear-gradient(90deg, #FF8F00, #FFB300); box-shadow: 0 0 12px rgba(255,179,0,0.2); }
    .conf-low { background: linear-gradient(90deg, #D50000, #FF5252); box-shadow: 0 0 12px rgba(255,82,82,0.2); }

    /* ── Info Block ── */
    .info-block {
        background: rgba(0,188,212,0.04);
        border-left: 2px solid rgba(0,188,212,0.3);
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        font-family: var(--font-sans);
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.55;
    }

    /* ── Factor Tags ── */
    .factor-tag {
        display: inline-block;
        font-family: var(--font-sans);
        font-size: 0.72rem;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 3px 4px 3px 0;
        font-weight: 500;
        line-height: 1.4;
    }
    .factor-primary {
        background: rgba(255,82,82,0.1);
        color: var(--danger);
        border: 1px solid rgba(255,82,82,0.18);
    }
    .factor-secondary {
        background: rgba(255,179,0,0.1);
        color: var(--warning);
        border: 1px solid rgba(255,179,0,0.18);
    }
    .factor-minor {
        background: rgba(255,255,255,0.04);
        color: var(--text-muted);
        border: 1px solid var(--border-subtle);
    }

    /* ── Legal Chips ── */
    .legal-chip {
        display: inline-block;
        background: rgba(0,230,118,0.07);
        color: var(--success);
        border: 1px solid rgba(0,230,118,0.15);
        font-family: var(--font-mono);
        font-size: 0.72rem;
        font-weight: 500;
        padding: 3px 10px;
        border-radius: 6px;
        margin: 2px 4px 2px 0;
    }

    /* ── Status ── */
    .status-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0;
        font-family: var(--font-sans);
        font-size: 0.78rem;
    }
    .status-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .status-dot.ok {
        background: var(--success);
        box-shadow: 0 0 8px rgba(0,230,118,0.4);
    }
    .status-dot.err {
        background: var(--danger);
        box-shadow: 0 0 8px rgba(255,82,82,0.4);
    }
    .status-label { color: var(--text-secondary); }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-primary) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(255,255,255,0.02);
        border-radius: 10px;
        padding: 3px;
        border: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-sans);
        font-weight: 500;
        font-size: 0.8rem;
        border-radius: 8px;
        padding: 0.45rem 1.1rem;
        color: var(--text-muted);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0,188,212,0.1) !important;
        color: var(--accent) !important;
        border-bottom: none !important;
    }

    /* ── Table ── */
    .cmp-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-family: var(--font-sans);
        font-size: 0.8rem;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-subtle);
    }
    .cmp-table th {
        background: rgba(255,255,255,0.02);
        color: var(--text-muted);
        text-transform: uppercase;
        font-family: var(--font-mono);
        font-size: 0.65rem;
        font-weight: 500;
        letter-spacing: 1px;
        padding: 0.65rem 1rem;
        text-align: left;
        border-bottom: 1px solid var(--border-subtle);
    }
    .cmp-table td {
        padding: 0.7rem 1rem;
        color: var(--text-secondary);
        border-bottom: 1px solid rgba(255,255,255,0.02);
    }
    .cmp-table tr:hover td {
        background: rgba(0,188,212,0.02);
    }
    .cmp-table .gt-row td {
        background: rgba(0,230,118,0.03);
        color: rgba(0,230,118,0.6);
        font-style: italic;
    }

    /* ── Divider ── */
    .section-divider {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
        margin: 1.25rem 0;
    }

    /* ── Empty State ── */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        font-family: var(--font-sans);
    }
    .empty-state-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }
    .empty-state-desc {
        font-size: 0.82rem;
        color: var(--text-muted);
        line-height: 1.6;
        max-width: 480px;
        margin: 0 auto;
    }

    /* ── Chunk Card ── */
    .chunk-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s ease;
    }
    .chunk-card:hover { border-color: var(--border-hover); }
    .chunk-score {
        font-family: var(--font-mono);
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--accent);
        letter-spacing: 0.5px;
    }
    .chunk-content {
        font-size: 0.8rem;
        color: var(--text-secondary);
        line-height: 1.55;
        margin-top: 0.35rem;
    }
    .chunk-source {
        font-family: var(--font-mono);
        font-size: 0.65rem;
        color: var(--text-muted);
        margin-top: 0.35rem;
    }

    /* ── Match Indicators ── */
    .match-pass {
        background: rgba(0,230,118,0.05);
        border: 1px solid rgba(0,230,118,0.12);
        border-left: 3px solid var(--success);
        border-radius: 0 8px 8px 0;
        padding: 0.55rem 0.9rem;
        margin: 0.35rem 0;
        font-family: var(--font-sans);
        font-size: 0.8rem;
        color: var(--success);
    }
    .match-fail {
        background: rgba(255,82,82,0.05);
        border: 1px solid rgba(255,82,82,0.12);
        border-left: 3px solid var(--danger);
        border-radius: 0 8px 8px 0;
        padding: 0.55rem 0.9rem;
        margin: 0.35rem 0;
        font-family: var(--font-sans);
        font-size: 0.8rem;
        color: var(--danger);
    }

    /* ── Liability Bar (replaces donut) ── */
    .liability-bar-container {
        display: flex;
        border-radius: 8px;
        overflow: hidden;
        height: 32px;
        margin: 0.5rem 0;
        border: 1px solid var(--border-subtle);
    }
    .liability-segment {
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: var(--font-mono);
        font-size: 0.72rem;
        font-weight: 600;
        color: white;
        transition: width 0.5s ease;
        position: relative;
    }
    .liability-legend {
        display: flex;
        gap: 1rem;
        margin-top: 0.4rem;
        flex-wrap: wrap;
    }
    .liability-legend-item {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        font-family: var(--font-sans);
        font-size: 0.72rem;
        color: var(--text-secondary);
    }
    .liability-legend-dot {
        width: 8px; height: 8px;
        border-radius: 3px;
        flex-shrink: 0;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.12); }

    /* ── Hide Streamlit UI ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
"""


# ── Data Loading ───────────────────────────────────────────────────────────
@st.cache_data
def load_scenarios() -> dict[str, list[dict]]:
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
    sid = scenario.get("scenario_id", "unknown")
    text = scenario.get("scenario_text", "")
    preview = text[:55] + "..." if len(text) > 55 else text
    return f"{sid} — {preview}"


# ── Execution ──────────────────────────────────────────────────────────────
def run_analysis_safe(text: str, variant: str) -> tuple[dict | None, str | None]:
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
    health: dict[str, bool] = {"groq": False, "qdrant": False}
    try:
        from haftung_ai.config.settings import get_settings
        settings = get_settings()
        health["groq"] = bool(settings.GROQ_API_KEY)
    except Exception:
        pass
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


# ── Component Helpers ─────────────────────────────────────────────────────
def render_variant_header(variant: str) -> str:
    color = VARIANT_COLORS.get(variant, "#999")
    label = VARIANT_LABELS.get(variant, variant)
    desc = VARIANT_DESCRIPTIONS.get(variant, "")
    return (
        f'<div class="variant-header">'
        f'  <div class="variant-indicator" style="background:{color}">{variant}</div>'
        f'  <div><div class="variant-name">{label}</div>'
        f'  <div class="variant-desc">{desc}</div></div>'
        f'</div>'
    )


def render_confidence_bar(score: float | None, label: str = "Confidence Score") -> str:
    if score is None:
        return (
            f'<div class="metric-card">'
            f'  <div class="metric-label">{label}</div>'
            f'  <div class="metric-value" style="color:var(--text-muted)">N/A</div>'
            f'</div>'
        )
    score = max(0.0, min(1.0, float(score)))
    pct = score * 100
    level = "high" if score >= 0.7 else ("med" if score >= 0.4 else "low")
    color_map = {"high": "#00E676", "med": "#FFB300", "low": "#FF5252"}
    return (
        f'<div class="metric-card">'
        f'  <div class="metric-label">{label}</div>'
        f'  <div class="metric-value" style="color:{color_map[level]}">{score:.1%}</div>'
        f'  <div class="conf-bar-outer">'
        f'    <div class="conf-bar-inner conf-{level}" style="width:{pct}%"></div>'
        f'  </div>'
        f'</div>'
    )


def render_metric_card(label: str, value: str, small: bool = False) -> str:
    cls = "metric-value-sm" if small else "metric-value"
    return (
        f'<div class="metric-card">'
        f'  <div class="metric-label">{label}</div>'
        f'  <div class="{cls}">{value}</div>'
        f'</div>'
    )


def format_accident_type(raw: str) -> str:
    label = CATEGORY_LABELS.get(raw)
    if label:
        return label
    return translate_de(raw.replace("_", " ").title())


def render_liability_bar(responsibility: list[dict]) -> str:
    """Render a horizontal stacked bar for liability distribution."""
    if not responsibility:
        return '<div style="color:var(--text-muted);font-size:0.78rem;padding:0.5rem 0">No liability data available</div>'

    palette = ["#00BCD4", "#FF5252", "#FFB300", "#7C4DFF", "#00E676", "#FF6E40"]
    total = sum(r.get("percentage", 0) for r in responsibility)
    if total == 0:
        return '<div style="color:var(--text-muted);font-size:0.78rem;padding:0.5rem 0">No liability percentages</div>'

    segments = ""
    legend_items = ""
    for i, r in enumerate(responsibility):
        pct = r.get("percentage", 0)
        party = translate_de(r.get("party", "?"))
        color = palette[i % len(palette)]
        segments += f'<div class="liability-segment" style="width:{pct}%;background:{color}">{pct:.0f}%</div>'
        legend_items += (
            f'<div class="liability-legend-item">'
            f'  <div class="liability-legend-dot" style="background:{color}"></div>'
            f'  {party} ({pct:.0f}%)'
            f'</div>'
        )

    return (
        f'<div class="metric-card">'
        f'  <div class="metric-label">Liability Distribution</div>'
        f'  <div class="liability-bar-container">{segments}</div>'
        f'  <div class="liability-legend">{legend_items}</div>'
        f'</div>'
    )


def render_factors_html(factors: list[dict]) -> str:
    if not factors:
        return ""
    tags = []
    for f in factors:
        if not isinstance(f, dict):
            continue
        severity = f.get("severity", "minor")
        cls = f"factor-{severity}" if severity in ("primary", "secondary", "minor") else "factor-minor"
        text = translate_de(f.get("factor", "N/A"))
        ref = f.get("legal_reference", "")
        ref_str = f' <span style="opacity:0.4">({ref})</span>' if ref else ""
        tags.append(f'<span class="factor-tag {cls}">{text}{ref_str}</span>')
    return " ".join(tags)


def render_legal_chips(refs: list[str]) -> str:
    if not refs:
        return '<span style="color:var(--text-muted);font-size:0.78rem">None referenced</span>'
    return " ".join(f'<span class="legal-chip">{r}</span>' for r in refs)


# ── Chart Rendering ───────────────────────────────────────────────────────
def render_comparison_bar_chart(results: dict[str, dict]) -> None:
    all_parties: list[str] = []
    for v, state in results.items():
        causation = state.get("causation_output", {})
        for r in causation.get("responsibility", []):
            party = r.get("party", "?")
            if party not in all_parties:
                all_parties.append(party)

    if not all_parties:
        return

    variants = list(results.keys())
    x = np.arange(len(all_parties))
    width = 0.65 / len(variants)

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#050508")
    ax.set_facecolor("#050508")

    for i, v in enumerate(variants):
        causation = results[v].get("causation_output", {})
        resp_map = {r.get("party", "?"): r.get("percentage", 0) for r in causation.get("responsibility", [])}
        values = [resp_map.get(p, 0) for p in all_parties]
        offset = (i - len(variants) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width,
            label=f"{v}: {VARIANT_LABELS.get(v, v)}",
            color=VARIANT_COLORS.get(v, "#999"),
            edgecolor="#050508", linewidth=0.5,
            zorder=3,
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=8,
                    color="#f0f0f5", fontweight="bold",
                )

    ax.set_ylabel("Liability %", fontsize=9, color="#f0f0f5", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([translate_de(p) for p in all_parties], fontsize=9, color="#f0f0f5")
    ax.set_ylim(0, 110)
    ax.tick_params(colors="#f0f0f5", which="both")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.grid(True, alpha=0.04, color="white")
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, labelcolor="#f0f0f5", loc="upper right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Tab Renderers ─────────────────────────────────────────────────────────
def render_analysis_tab(results: dict[str, dict]) -> None:
    cols = st.columns(len(results))
    for col, (variant, state) in zip(cols, results.items()):
        with col:
            st.markdown(render_variant_header(variant), unsafe_allow_html=True)

            causation = state.get("causation_output", {})
            if not causation:
                st.markdown('<div class="info-block">Analysis did not return causation data.</div>', unsafe_allow_html=True)
                continue

            # Accident type
            acc_type = format_accident_type(causation.get("accident_type", "N/A"))
            st.markdown(render_metric_card("Accident Classification", acc_type), unsafe_allow_html=True)

            # Confidence
            confidence = state.get("confidence_score") or causation.get("confidence")
            st.markdown(render_confidence_bar(confidence), unsafe_allow_html=True)

            # Primary cause
            cause = translate_de(causation.get("primary_cause", "N/A"))
            st.markdown(
                f'<div class="metric-card">'
                f'  <div class="metric-label">Root Cause</div>'
                f'  <div class="info-block">{cause}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Liability bar
            st.markdown(render_liability_bar(causation.get("responsibility", [])), unsafe_allow_html=True)

            # Contributing factors
            factors = causation.get("contributing_factors", [])
            if factors:
                st.markdown(
                    f'<div class="metric-card">'
                    f'  <div class="metric-label">Contributing Factors</div>'
                    f'  <div style="margin-top:0.4rem">{render_factors_html(factors)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Legal references
            legal_refs = causation.get("legal_references", [])
            st.markdown(
                f'<div class="metric-card">'
                f'  <div class="metric-label">Legal References</div>'
                f'  <div style="margin-top:0.4rem">{render_legal_chips(legal_refs)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            for err in state.get("errors", []):
                st.error(err)


def render_comparison_tab(results: dict[str, dict], ground_truth: dict | None) -> None:
    rows_html = ""
    for variant, state in results.items():
        causation = state.get("causation_output", {})
        confidence = state.get("confidence_score") or causation.get("confidence")
        claims = causation.get("claims", [])
        chunks = state.get("retrieved_chunks", [])
        conf_str = f"{confidence:.0%}" if confidence else "—"
        color = VARIANT_COLORS.get(variant, "#999")
        rows_html += (
            f'<tr>'
            f'<td><span style="color:{color};font-weight:700">{variant}</span> {VARIANT_LABELS.get(variant, "")}</td>'
            f'<td>{translate_de(causation.get("primary_cause", "N/A"))}</td>'
            f'<td>{format_accident_type(causation.get("accident_type", "N/A"))}</td>'
            f'<td style="font-weight:700">{conf_str}</td>'
            f'<td>{len(claims)}</td>'
            f'<td>{len(chunks)}</td>'
            f'</tr>'
        )

    if ground_truth:
        rows_html += (
            f'<tr class="gt-row">'
            f'<td>Ground Truth</td>'
            f'<td>{translate_de(ground_truth.get("primary_cause", "N/A"))}</td>'
            f'<td>{format_accident_type(ground_truth.get("accident_type", "N/A"))}</td>'
            f'<td>—</td>'
            f'<td>{len(ground_truth.get("expected_claims", []))}</td>'
            f'<td>—</td>'
            f'</tr>'
        )

    st.markdown(
        f'<table class="cmp-table">'
        f'<thead><tr>'
        f'<th>System</th><th>Root Cause</th><th>Classification</th>'
        f'<th>Confidence</th><th>Claims</th><th>RAG Chunks</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label" style="padding:0.4rem 0">LIABILITY COMPARISON ACROSS SYSTEMS</div>', unsafe_allow_html=True)
    render_comparison_bar_chart(results)

    # Delta indicators
    variants = list(results.keys())
    if len(variants) >= 2:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label" style="padding:0.4rem 0">SYSTEM IMPROVEMENT DELTAS</div>', unsafe_allow_html=True)

        pairs = []
        if "S1" in results and "S2" in results:
            pairs.append(("S2", "S1"))
        if "S2" in results and "S3" in results:
            pairs.append(("S3", "S2"))

        for newer, older in pairs:
            c_new = results[newer].get("confidence_score") or results[newer].get("causation_output", {}).get("confidence")
            c_old = results[older].get("confidence_score") or results[older].get("causation_output", {}).get("confidence")

            delta_cols = st.columns(3)
            with delta_cols[0]:
                cn = VARIANT_COLORS.get(newer, "#fff")
                co = VARIANT_COLORS.get(older, "#fff")
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Comparison</div>'
                    f'<div class="metric-value-sm"><span style="color:{cn}">{newer}</span>'
                    f' <span style="color:var(--text-muted)">vs</span> '
                    f'<span style="color:{co}">{older}</span></div></div>',
                    unsafe_allow_html=True,
                )
            with delta_cols[1]:
                if c_new is not None and c_old is not None:
                    delta = c_new - c_old
                    dcls = "metric-delta-pos" if delta >= 0 else "metric-delta-neg"
                    dsign = "+" if delta >= 0 else ""
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-label">Confidence Delta</div>'
                        f'<div class="metric-value-sm">{c_new:.0%} <span class="{dcls}">{dsign}{delta:.0%}</span></div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(render_metric_card("Confidence Delta", "N/A"), unsafe_allow_html=True)
            with delta_cols[2]:
                chunks_new = len(results[newer].get("retrieved_chunks", []))
                chunks_old = len(results[older].get("retrieved_chunks", []))
                cd = chunks_new - chunks_old
                dcls = "metric-delta-pos" if cd >= 0 else "metric-delta-neg"
                dsign = "+" if cd >= 0 else ""
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">RAG Chunks</div>'
                    f'<div class="metric-value-sm">{chunks_new} <span class="{dcls}">{dsign}{cd}</span></div></div>',
                    unsafe_allow_html=True,
                )


def render_evidence_tab(results: dict[str, dict]) -> None:
    for variant, state in results.items():
        st.markdown(render_variant_header(variant), unsafe_allow_html=True)

        if variant == "S1":
            st.markdown(
                '<div class="info-block">Baseline system (S1) operates without retrieval-augmented generation. '
                'No external legal evidence is retrieved.</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            continue

        chunks = state.get("retrieved_chunks", [])
        if chunks:
            st.markdown(f'<div class="metric-label" style="padding:0.4rem 0">RETRIEVED KNOWLEDGE CHUNKS ({len(chunks)})</div>', unsafe_allow_html=True)
            for chunk in chunks:
                score = chunk.get("score", 0)
                score_str = f"RELEVANCE {score:.3f}" if isinstance(score, (int, float)) else "RELEVANCE N/A"
                content = chunk.get("content", "N/A")
                meta = chunk.get("metadata", {})
                source = meta.get("source_name", meta.get("source", "unknown"))
                st.markdown(
                    f'<div class="chunk-card">'
                    f'  <div class="chunk-score">{score_str}</div>'
                    f'  <div class="chunk-content">{content[:500]}{"..." if len(content) > 500 else ""}</div>'
                    f'  <div class="chunk-source">Source: {source}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="info-block">No knowledge chunks were retrieved.</div>', unsafe_allow_html=True)

        causation = state.get("causation_output", {})
        legal_refs = causation.get("legal_references", [])
        if legal_refs:
            st.markdown('<div class="metric-label" style="padding:0.6rem 0 0.25rem 0">LEGAL REFERENCES</div>', unsafe_allow_html=True)
            st.markdown(render_legal_chips(legal_refs), unsafe_allow_html=True)

        claims = causation.get("claims", [])
        if claims:
            st.markdown(f'<div class="metric-label" style="padding:0.6rem 0 0.25rem 0">EXTRACTED CLAIMS ({len(claims)})</div>', unsafe_allow_html=True)
            claims_data = []
            for c in claims:
                if isinstance(c, dict):
                    conf = c.get("confidence")
                    conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) else "—"
                    claims_data.append({
                        "Statement": translate_de(c.get("statement", "N/A")),
                        "Source": c.get("source_type", "N/A").upper(),
                        "Conf.": conf_str,
                        "Valid": "Yes" if c.get("supported") else "No",
                    })
            if claims_data:
                st.dataframe(pd.DataFrame(claims_data), use_container_width=True, hide_index=True)

        contradictions = state.get("contradictions", [])
        if state.get("has_contradictions") and contradictions:
            st.markdown('<div class="metric-label" style="padding:0.6rem 0 0.25rem 0;color:var(--danger)">CONTRADICTIONS DETECTED</div>', unsafe_allow_html=True)
            for contradiction in contradictions:
                if isinstance(contradiction, dict):
                    st.markdown(
                        f'<div class="match-fail">'
                        f'<b>{contradiction.get("type", "Conflict")}:</b> '
                        f'{translate_de(contradiction.get("description", contradiction.get("statement_a", "N/A")))} '
                        f'vs {translate_de(contradiction.get("statement_b", ""))}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


def render_ground_truth_tab(results: dict[str, dict], ground_truth: dict | None) -> None:
    if not ground_truth:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-title">No Ground Truth Available</div>'
            '<div class="empty-state-desc">Ground truth validation is only available for pre-loaded evaluation scenarios. '
            'Select a scenario from the sidebar to enable this view.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    gt_cause = ground_truth.get("primary_cause", "")
    gt_type = ground_truth.get("accident_type", "")
    gt_responsibility = {r["party"]: r["percentage"] for r in ground_truth.get("responsibility", [])}
    gt_stvo = set(ground_truth.get("relevant_stvo", []))
    gt_claims = ground_truth.get("expected_claims", [])

    ref_cols = st.columns(3)
    with ref_cols[0]:
        st.markdown(render_metric_card("Expected Cause", translate_de(gt_cause), small=True), unsafe_allow_html=True)
    with ref_cols[1]:
        st.markdown(render_metric_card("Expected Type", format_accident_type(gt_type), small=True), unsafe_allow_html=True)
    with ref_cols[2]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Relevant Laws</div>'
            f'<div style="margin-top:0.35rem">{render_legal_chips(sorted(gt_stvo))}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    for variant, state in results.items():
        st.markdown(render_variant_header(variant), unsafe_allow_html=True)
        causation = state.get("causation_output", {})

        check_cols = st.columns(2)
        with check_cols[0]:
            pred_cause = causation.get("primary_cause", "")
            cause_match = gt_cause.lower() in pred_cause.lower() or pred_cause.lower() in gt_cause.lower()
            if cause_match:
                st.markdown(f'<div class="match-pass">Root Cause MATCH: {translate_de(pred_cause)}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="match-fail">Root Cause MISMATCH<br>Predicted: {translate_de(pred_cause)}<br>Expected: {translate_de(gt_cause)}</div>', unsafe_allow_html=True)

        with check_cols[1]:
            pred_type = causation.get("accident_type", "")
            if pred_type == gt_type:
                st.markdown(f'<div class="match-pass">Classification MATCH: {format_accident_type(pred_type)}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="match-fail">Classification MISMATCH<br>Predicted: {format_accident_type(pred_type)}<br>Expected: {format_accident_type(gt_type)}</div>', unsafe_allow_html=True)

        pred_resp = {r.get("party", "?"): r.get("percentage", 0) for r in causation.get("responsibility", [])}
        if pred_resp:
            resp_cols = st.columns(len(pred_resp))
            for i, (party, pct) in enumerate(pred_resp.items()):
                gt_pct = gt_responsibility.get(party)
                with resp_cols[i]:
                    if gt_pct is not None:
                        delta = pct - gt_pct
                        dcls = "metric-delta-pos" if abs(delta) < 5 else "metric-delta-neg"
                        dstr = f'{delta:+.0f}pp' if delta != 0 else "exact"
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-label">{translate_de(party)}</div>'
                            f'<div class="metric-value-sm">{pct:.0f}% <span class="{dcls}">{dstr}</span></div></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(render_metric_card(translate_de(party), f"{pct:.0f}%", small=True), unsafe_allow_html=True)

        if gt_stvo:
            pred_refs = set(causation.get("legal_references", []))
            chunks = state.get("retrieved_chunks", [])
            chunk_text = " ".join(c.get("content", "") for c in chunks if isinstance(c, dict))
            retrieved_stvo = {s for s in gt_stvo if s in chunk_text or s in " ".join(pred_refs)}
            covered = len(retrieved_stvo)
            total = len(gt_stvo)
            pct_cov = covered / total if total > 0 else 0

            stvo_html = ""
            for s in sorted(gt_stvo):
                if s in retrieved_stvo:
                    stvo_html += f'<span class="legal-chip">{s}</span> '
                else:
                    stvo_html += f'<span class="factor-tag factor-minor" style="text-decoration:line-through">{s}</span> '

            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Legal Coverage: {covered}/{total} ({pct_cov:.0%})</div>'
                f'<div style="margin-top:0.4rem">{stvo_html}</div></div>',
                unsafe_allow_html=True,
            )

        if gt_claims:
            pred_text = " ".join([
                causation.get("primary_cause", ""),
                causation.get("reasoning", ""),
                " ".join(c.get("statement", "") for c in causation.get("claims", []) if isinstance(c, dict)),
            ]).lower()

            claims_html = ""
            matched_count = 0
            for claim in gt_claims:
                keywords = [w for w in claim.lower().split() if len(w) > 4]
                matched = sum(1 for kw in keywords if kw in pred_text)
                ratio = matched / len(keywords) if keywords else 0
                if ratio >= 0.5:
                    claims_html += f'<div class="match-pass" style="padding:0.4rem 0.8rem;margin:2px 0;font-size:0.78rem">{translate_de(claim)}</div>'
                    matched_count += 1
                else:
                    claims_html += f'<div class="match-fail" style="padding:0.4rem 0.8rem;margin:2px 0;font-size:0.78rem">{translate_de(claim)}</div>'

            st.markdown(
                f'<div class="metric-label" style="padding:0.4rem 0">EXPECTED CLAIMS COVERAGE ({matched_count}/{len(gt_claims)})</div>'
                f'{claims_html}',
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


def render_report_tab(results: dict[str, dict]) -> None:
    for variant, state in results.items():
        report = state.get("report", {})
        if not report:
            st.markdown(render_variant_header(variant), unsafe_allow_html=True)
            st.markdown('<div class="info-block">No accident report was generated for this variant.</div>', unsafe_allow_html=True)
            continue

        with st.expander(f"{variant} — {VARIANT_LABELS.get(variant, '')} — Accident Report", expanded=len(results) == 1):
            if isinstance(report, dict):
                for section, content in report.items():
                    section_title = section.replace("_", " ").title()
                    st.markdown(f'<div class="metric-label" style="padding:0.4rem 0">{section_title.upper()}</div>', unsafe_allow_html=True)
                    if isinstance(content, str):
                        st.markdown(f'<div class="info-block">{content}</div>', unsafe_allow_html=True)
                    elif isinstance(content, list):
                        for item in content:
                            st.markdown(f"- {item}")
                    elif isinstance(content, dict):
                        for k, v in content.items():
                            st.markdown(f"**{k}:** {v}")
            else:
                st.write(report)


# ── Sidebar ───────────────────────────────────────────────────────────────
def render_sidebar() -> tuple[str, list[str], dict | None, bool]:
    with st.sidebar:
        st.markdown(
            '<div style="padding:0.25rem 0 0.75rem 0">'
            '<div style="font-family:Inter,sans-serif;font-size:1.2rem;font-weight:800;color:white;letter-spacing:-0.5px">'
            'HAFTUNG<span style="color:#00BCD4">_AI</span></div>'
            '<div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;color:rgba(255,255,255,0.25);letter-spacing:1.5px;text-transform:uppercase;margin-top:2px">'
            'Causation Intelligence Platform</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label" style="padding:0 0 0.4rem 0">SCENARIO INPUT</div>', unsafe_allow_html=True)

        scenarios = load_scenarios()
        use_custom = st.checkbox("Custom text input", key="custom_toggle")

        scenario_text = ""
        ground_truth = None
        is_preloaded = False

        if use_custom:
            scenario_text = st.text_area("Accident description", height=160, placeholder="Describe the accident scenario...")
        else:
            if not scenarios:
                st.warning(f"No scenarios found in {SCENARIOS_DIR}")
            else:
                categories = list(scenarios.keys())
                category = st.selectbox("Category", categories, format_func=lambda c: CATEGORY_LABELS.get(c, c))
                scenario_list = scenarios.get(category, [])
                if scenario_list:
                    selected = st.selectbox("Scenario", scenario_list, format_func=get_scenario_label)
                    if selected:
                        scenario_text = selected.get("scenario_text", "")
                        ground_truth = selected.get("ground_truth")
                        is_preloaded = True

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label" style="padding:0 0 0.4rem 0">ANALYSIS SYSTEMS</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        s1 = col1.checkbox("S1", value=True, help="Baseline")
        s2 = col2.checkbox("S2", value=True, help="RAG-augmented")
        s3 = col3.checkbox("S3", value=True, help="RAG + validation")
        variants = [v for v, on in [("S1", s1), ("S2", s2), ("S3", s3)] if on]

        st.markdown(
            '<div style="font-family:Inter,sans-serif;font-size:0.68rem;color:rgba(255,255,255,0.25);padding:0.15rem 0 0.5rem 0;line-height:1.5">'
            '<b style="color:rgba(255,255,255,0.4)">S1</b> Baseline '
            '<b style="color:rgba(255,255,255,0.4)">S2</b> +RAG '
            '<b style="color:rgba(255,255,255,0.4)">S3</b> +Validation</div>',
            unsafe_allow_html=True,
        )

        run_clicked = st.button("Run Analysis", type="primary", disabled=not scenario_text or not variants, use_container_width=True)
        if run_clicked:
            st.session_state.run_requested = True

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label" style="padding:0 0 0.4rem 0">SYSTEM STATUS</div>', unsafe_allow_html=True)

        health = check_service_health()
        for name, ok, hint in [
            ("Groq LLM API", health["groq"], "Set GROQ_API_KEY in .env"),
            ("Qdrant Vector DB", health["qdrant"], "Run: make docker-up"),
        ]:
            dot_cls = "ok" if ok else "err"
            status_text = "Connected" if ok else "Offline"
            st.markdown(
                f'<div class="status-row">'
                f'  <div class="status-dot {dot_cls}"></div>'
                f'  <div class="status-label">{name}: {status_text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if not ok:
                st.caption(hint)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        with st.expander("About"):
            st.markdown(
                "**Haftung_AI** is an LLM-powered accident causation analysis platform "
                "for the automotive industry.\n\n"
                "It combines computer vision, CAN bus telemetry, and legal RAG retrieval "
                "to determine fault and liability under German traffic law (StVO).\n\n"
                "Three system variants demonstrate the measurable impact of retrieval augmentation "
                "and constraint validation on analysis accuracy.\n\n"
                "Built with LangGraph, Groq (LLaMA 3.3 70B), Qdrant, and SentenceTransformers."
            )

    return scenario_text, variants, ground_truth, is_preloaded


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Haftung_AI — Causation Intelligence",
        page_icon="H",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="hero-header">'
        '  <p class="hero-title">Accident Causation Intelligence</p>'
        '  <p class="hero-subtitle">Multi-agent analysis pipeline comparing baseline, RAG-augmented, and constraint-validated systems</p>'
        '  <div class="hero-badges">'
        '    <span class="hero-badge">LLM-Powered</span>'
        '    <span class="hero-badge">German Traffic Law (StVO)</span>'
        '    <span class="hero-badge hero-badge-alt">LLaMA 3.3 70B</span>'
        '    <span class="hero-badge hero-badge-alt">Qdrant RAG</span>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )

    scenario_text, variants, ground_truth, is_preloaded = render_sidebar()

    if st.session_state.get("run_requested"):
        st.session_state.run_requested = False
        results: dict[str, dict] = {}

        progress = st.progress(0, text="Initializing analysis pipeline...")
        for i, variant in enumerate(variants):
            progress.progress(i / len(variants), text=f"Running {variant} — {VARIANT_LABELS.get(variant, '')}...")
            result, error = run_analysis_safe(scenario_text, variant)
            if error:
                st.error(f"**{variant}** failed: {error}")
            if result:
                results[variant] = result

        progress.progress(1.0, text="Analysis complete.")

        if results:
            st.session_state.results = results
            st.session_state.ground_truth = ground_truth
            st.session_state.is_preloaded = is_preloaded
            st.session_state.last_text = scenario_text

    results = st.session_state.get("results")
    if not results:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-title">Select a Scenario to Begin</div>'
            '<div class="empty-state-desc">'
            'Choose an accident scenario from the sidebar and click <b>Run Analysis</b> '
            'to compare causation outputs across system variants.<br><br>'
            'Start with <b>S1</b> only for fast results, or enable all three for a full comparison.'
            '</div></div>',
            unsafe_allow_html=True,
        )
        return

    ground_truth = st.session_state.get("ground_truth")
    is_preloaded = st.session_state.get("is_preloaded", False)

    with st.expander("View Scenario Input"):
        st.markdown(f'<div class="info-block">{st.session_state.get("last_text", "")}</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Analysis", "Comparison", "Evidence & RAG", "Ground Truth", "Report"])

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
