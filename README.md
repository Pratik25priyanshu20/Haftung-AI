# Haftung_AI

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0-FF6B00?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiPjwvc3ZnPg==)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-8B5CF6)](https://groq.com)
[![Tests](https://img.shields.io/badge/tests-244_passing-22c55e)]()

> **Multi-agent accident causation analysis platform powered by LLM inference, retrieval-augmented generation, and constraint validation under German traffic law (StVO).**

Haftung_AI orchestrates a pipeline of specialized agents -- vision, telemetry, RAG retrieval, evidence extraction, contradiction detection, causation reasoning, validation, and report generation -- to determine fault and liability in traffic accidents. Three structurally distinct system variants (S1, S2, S3) are evaluated side-by-side across 30 ground-truth scenarios to quantify the measurable impact of legal knowledge retrieval and claim-level constraint enforcement on analysis accuracy.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [System Variants](#system-variants)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Demo Dashboard](#demo-dashboard)
- [Project Structure](#project-structure)
- [Evaluation Framework](#evaluation-framework)
- [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [Development](#development)
- [Limitations](#limitations)
- [License](#license)

---

## Key Features

- **Three-variant comparison** -- Baseline (S1), RAG-augmented (S2), and RAG + constraint validation (S3) pipelines compiled as separate LangGraph state machines
- **Legal RAG retrieval** -- Adaptive dense/hybrid retrieval from a Qdrant vector database containing StVO statutes and German case law, with BM25 reranking
- **Multi-agent pipeline** -- 9 specialized agents (Vision, Telemetry, RAG, Evidence, Contradiction, Causation, Validation, Report, TextInput) orchestrated via LangGraph StateGraph
- **Constraint enforcement loop** -- S3 validation agent triggers re-analysis when confidence falls below threshold, improving accuracy through iterative refinement
- **Multi-run validation** -- LLM judge runs N=3 times per scenario with rubric-based scoring across 4 criteria, tracking mean and standard deviation to mitigate single-judge bias
- **Configurable confidence scoring** -- Weighted combination of LLM judge score, evidence coverage, and base confidence with tunable weights via environment variables or grid search ablation
- **Computer vision pipeline** -- YOLOv8 object detection, DeepSORT multi-object tracking, Kalman smoothing, scene graph construction, and TTC-based collision detection
- **CAN bus telemetry** -- Parser supporting CSV/ASC/BLF formats, speed profiling, braking event detection, anomaly classification, and ego-vehicle trajectory reconstruction
- **ISO 26262 safety analysis** -- ASIL classification, plausibility checks, sensor health monitoring, and diagnostic trouble code logging
- **Professional report generation** -- Jinja2-templated accident reports rendered to PDF via WeasyPrint with HTML fallback
- **Interactive dashboard** -- Streamlit UI with scenario browser, variant comparison, evidence inspector, ground truth validation, and report viewer
- **Comprehensive evaluation** -- 30 scenarios, 10+ metrics, statistical significance tests, judge variance analysis, and confidence weight ablation studies

---

## Architecture

```
                        ┌─────────────────────────────────────────────────────────┐
                        │              Haftung_AI Analysis Pipeline                │
                        └─────────────────────────────────────────────────────────┘

  S1  Baseline          TextInput ──▶ Causation ──▶ Report

  S2  RAG-Augmented     TextInput ──▶ RAG ──▶ Causation ──▶ Evidence ──▶ Contradiction
                                                            ──▶ Validation ──▶ Report

  S3  RAG + Validation  TextInput ──▶ RAG ──▶ Causation ──▶ Evidence ──▶ Contradiction
                                                            ──▶ Validation ──┐
                                              ▲                              │
                                              └──── correction loop ◀────────┘
                                                                         ──▶ Report
```

Each variant is compiled as a separate LangGraph `StateGraph` at build time. All agents share a common `HaftungState` (TypedDict with ~40 fields) that flows through the pipeline, progressively enriched at each node.

For video + CAN input mode, `TextInput` is replaced by `Vision` (YOLOv8 + DeepSORT + Kalman) and `Telemetry` (CAN parsing + speed profiling) agents.

---

## System Variants

| Variant | Pipeline | Description |
|---------|----------|-------------|
| **S1** | TextInput &rarr; Causation &rarr; Report | Pure LLM inference from scenario data. No external knowledge, no validation. Serves as the performance baseline. |
| **S2** | TextInput &rarr; RAG &rarr; Causation &rarr; Evidence &rarr; Contradiction &rarr; Validation &rarr; Report | Augmented with retrieval from StVO statutes and case law. Evidence extraction and contradiction detection provide structured claim support. Multi-run validation produces calibrated confidence scores. |
| **S3** | S2 + conditional loop from Validation back to Causation | Full constraint enforcement. If validation confidence falls below threshold (default 0.7), the system loops back to re-run causation with updated contradiction penalties. Terminates when confidence is sufficient or max iterations reached. |

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM Inference** | Groq API (LLaMA 3.3 70B Versatile) | Fast inference for causation reasoning, evidence extraction, validation judging, and report generation |
| **Orchestration** | LangGraph (StateGraph) | Compile-time variant graphs with conditional edges and correction loops |
| **Vector Database** | Qdrant | Dense vector storage and retrieval for StVO statutes and case law |
| **Embeddings** | BAAI/bge-large-en-v1.5 (1024-dim) | Semantic embedding for retrieval queries and knowledge base chunks |
| **Reranking** | BM25 (rank-bm25) | Lexical reranking combined with dense scores via configurable hybrid alpha |
| **Object Detection** | YOLOv8 (Ultralytics) | Real-time vehicle, pedestrian, and cyclist detection from dashcam video |
| **Tracking** | DeepSORT | Appearance-based multi-object tracking with track identity maintenance |
| **State Estimation** | Kalman Filter + RTS Smoother | Position and velocity estimation with backward-pass smoothing |
| **Telemetry** | Custom CAN parser | CSV/ASC/BLF parsing, speed profiling, braking event detection |
| **Safety** | ISO 26262 ASIL | Safety integrity classification, TTC computation, plausibility validation |
| **Reports** | Jinja2 + WeasyPrint | Templated accident report generation with PDF and HTML output |
| **API** | FastAPI + Uvicorn | REST endpoints for analysis, report retrieval, and health checks |
| **Dashboard** | Streamlit | Interactive scenario browser with 5-tab analysis comparison UI |
| **Testing** | pytest (244 tests) | Unit, integration, and evaluation test suites with coverage reporting |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for Qdrant vector database)
- [Groq API key](https://console.groq.com) (free tier available)

### Installation

```bash
git clone https://github.com/pratik25priyanshu20/haftung-ai.git
cd haftung-ai
pip install -e ".[dev]"
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` and set the required values:

```env
GROQ_API_KEY=gsk_your_api_key_here    # Required
GROQ_MODEL=llama-3.3-70b-versatile    # Default model
QDRANT_URL=http://localhost:6333       # Default Qdrant endpoint
```

### Start Services

```bash
# Start Qdrant (required for S2/S3 RAG retrieval)
make docker-up

# Ingest German traffic law into the vector database
make ingest
```

### Launch Dashboard

```bash
make demo
```

Open [http://localhost:8502](http://localhost:8502) in your browser.

### Run All Three Variants

1. Select a scenario category and scenario from the sidebar
2. Enable S1, S2, and S3 checkboxes
3. Click **Run Analysis**
4. Compare results across the 5 tabs: Analysis, Comparison, Evidence & RAG, Ground Truth, Report

---

## Demo Dashboard

The interactive Streamlit dashboard (`src/haftung_ai/ui/demo.py`) provides a professional analysis interface:

| Tab | Content |
|-----|---------|
| **Analysis** | Side-by-side causation results per variant -- accident classification, confidence score with gradient bar, root cause, liability distribution (horizontal stacked bar), contributing factors (severity-tagged), and legal references |
| **Comparison** | Summary table across all variants, grouped liability bar chart, and system improvement delta metrics (confidence and RAG chunk deltas) |
| **Evidence & RAG** | Retrieved knowledge chunks with relevance scores, legal reference chips, extracted claims table with source and validation status, contradiction alerts |
| **Ground Truth** | Match/mismatch indicators for root cause and classification, liability deviation per party, legal coverage ratio, expected claims coverage |
| **Report** | Structured accident report output per variant |

The sidebar provides a scenario picker (6 categories, 30 scenarios with ground truth), custom text input mode, variant toggles, and live service health indicators.

---

## Project Structure

```
haftung-ai/
├── src/haftung_ai/
│   ├── agents/                 # LangGraph pipeline nodes
│   │   ├── orchestrator.py     # Graph builder (S1/S2/S3 variants)
│   │   ├── vision_agent.py     # YOLOv8 + DeepSORT + Kalman + scene graph
│   │   ├── telemetry_agent.py  # CAN parsing + speed profiling
│   │   ├── rag_node.py         # Adaptive retrieval orchestration
│   │   ├── evidence_agent.py   # LLM-based evidence extraction
│   │   ├── contradiction_agent.py  # Pairwise contradiction detection
│   │   ├── causation_agent.py  # 3-variant causation reasoning
│   │   ├── validation_agent.py # Multi-run LLM judge + confidence scoring
│   │   └── report_agent.py     # Accident report generation
│   ├── api/                    # FastAPI REST backend
│   │   ├── main.py             # App configuration and CORS
│   │   └── routes/             # analyze, report, health, stream endpoints
│   ├── config/
│   │   └── settings.py         # Pydantic BaseSettings (env-driven)
│   ├── llm/
│   │   ├── client.py           # Groq client with rate limiting and retries
│   │   ├── prompts.py          # All agent prompt templates (English)
│   │   └── structured_output.py # JSON extraction and Pydantic validation
│   ├── perception/
│   │   ├── detector.py         # YOLOv8 wrapper (auto device selection)
│   │   ├── tracker.py          # DeepSORT multi-object tracker
│   │   ├── kalman.py           # Kalman filter + RTS backward smoother
│   │   ├── scene_graph.py      # Spatial relation graph builder
│   │   ├── impact_detector.py  # TTC-based collision detection
│   │   └── video_input.py      # Frame reader (MP4/AVI)
│   ├── rag/
│   │   ├── embeddings.py       # SentenceTransformer service
│   │   ├── vectorstore.py      # Qdrant client wrapper
│   │   ├── knowledge_base.py   # Document ingestion pipeline
│   │   ├── retrieval.py        # Adaptive dense/hybrid retriever
│   │   └── reranker.py         # BM25 reranking
│   ├── report/
│   │   ├── pdf_generator.py    # WeasyPrint + Jinja2 PDF generation
│   │   ├── scene_diagram.py    # Bird's-eye-view diagram renderer
│   │   └── templates/          # HTML report template
│   ├── safety/
│   │   ├── asil.py             # ISO 26262 ASIL classification
│   │   ├── ttc.py              # Time-to-collision computation
│   │   ├── plausibility.py     # Range and physics validation
│   │   ├── sensor_health.py    # Confidence degradation tracking
│   │   └── safety_manager.py   # Aggregated safety evaluation
│   ├── telemetry/
│   │   ├── can_parser.py       # CSV/ASC/BLF format parser
│   │   ├── speed_profile.py    # Braking event detection
│   │   ├── anomaly_detector.py # Severity classification
│   │   └── ego_reconstructor.py # Trajectory integration
│   ├── types/                  # Pydantic models and TypedDicts
│   └── ui/
│       ├── demo.py             # Streamlit analysis dashboard
│       ├── app.py              # Main Streamlit application
│       └── components/         # Upload, results, scene viewer widgets
├── evaluation/
│   ├── dataset/
│   │   ├── scenarios/          # 30 JSON scenarios with ground truth
│   │   └── accidents/          # Video + CAN log test data
│   ├── runners/
│   │   ├── run_experiment.py       # Single-variant evaluation
│   │   ├── run_all_systems.py      # Comparative S1/S2/S3 evaluation
│   │   ├── run_stability.py        # N=5 rerun variance analysis
│   │   ├── run_judge_variance.py   # LLM judge inter-run variance
│   │   └── run_weight_ablation.py  # Confidence weight grid search
│   ├── metrics/
│   │   ├── aggregate.py            # Master metric aggregator
│   │   ├── cause_taxonomy.py       # 20-class keyword taxonomy accuracy
│   │   ├── causation_accuracy.py   # Fuzzy string matching
│   │   ├── responsibility_mae.py   # Mean absolute error
│   │   ├── factors_f1.py           # Precision/recall/F1
│   │   ├── calibration.py          # ECE + Brier score
│   │   ├── hallucination.py        # Unsupported claim rate
│   │   └── retrieval_quality.py    # P@5, MRR, nDCG@5
│   └── analysis/
│       ├── compare_systems.py      # Cross-variant comparison tables
│       ├── statistical_tests.py    # Mann-Whitney U, t-tests
│       ├── results_table.py        # CSV/JSON export
│       └── plot_results.py         # Matplotlib visualizations
├── scripts/
│   ├── ingest_knowledge_base.py    # Qdrant ingestion CLI
│   ├── create_text_scenarios.py    # Scenario generation
│   ├── create_eval_dataset.py      # Dataset assembly
│   └── generate_synthetic_can.py   # Parametric CAN data generation
├── tests/
│   ├── unit/                   # 206 unit tests
│   ├── integration/            # Pipeline and API integration tests
│   └── evaluation/             # Metric computation tests
├── data/
│   └── knowledge_base/
│       ├── stvo/               # German Road Traffic Regulations
│       └── case_law/           # Court decisions and precedents
├── pyproject.toml
├── Makefile                    # 35+ development and evaluation targets
├── docker-compose.yml          # Qdrant + API + Streamlit services
└── .env.example                # Environment variable template
```

---

## Evaluation Framework

### Dataset

30 hand-authored German accident scenarios across 6 categories, each with structured ground truth:

| Category | Count | Example Scenarios |
|----------|------:|-------------------|
| Rear-End Collision | 5 | Highway chain reaction, stop-and-go traffic |
| Side Collision | 5 | Lane change without signaling, blind spot |
| Head-On Collision | 5 | Overtaking maneuver, wrong-way driver |
| Intersection | 5 | Right-of-way violation, red-light running |
| Pedestrian | 5 | Crosswalk incident, school zone |
| Single Vehicle | 5 | Aquaplaning, tire blowout, road debris |

**Ground truth fields:** `primary_cause`, `primary_cause_taxonomy_id`, `accident_type`, `contributing_factors`, `responsibility` (per-party percentages), `relevant_stvo`, `expected_claims`, `legal_references`

### Metrics

| Metric | Module | Description |
|--------|--------|-------------|
| Causation Accuracy (Taxonomy) | `cause_taxonomy.py` | 20-class keyword taxonomy with exact match scoring |
| Causation Accuracy (Fuzzy) | `causation_accuracy.py` | Substring-based fuzzy matching |
| Responsibility MAE | `responsibility_mae.py` | Mean absolute error of per-party liability percentages |
| Contributing Factors F1 | `factors_f1.py` | Precision, recall, and F1 for contributing factor identification |
| ECE | `calibration.py` | Expected Calibration Error across 10 confidence bins |
| Brier Score | `calibration.py` | Mean squared error of confidence estimates |
| Hallucination Rate | `hallucination.py` | Fraction of claims not supported by evidence |
| Retrieval Quality | `retrieval_quality.py` | Precision@5, Mean Reciprocal Rank, nDCG@5 |

### Running Evaluations

```bash
# Core evaluation
make eval-text-all              # Run S1/S2/S3 on all 30 scenarios
make eval-text-stability        # 5 reruns per variant for variance analysis
make eval-tables                # Generate comparison tables and statistical tests

# Advanced analysis
make eval-judge-variance        # Quantify LLM judge inter-run variance (N=5)
make eval-weight-ablation       # Grid search over confidence weight combinations
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Run accident analysis (multipart: video, CAN log, variant) |
| `GET` | `/report/{analysis_id}` | Retrieve analysis report by ID |
| `GET` | `/health` | Service health check (Groq API, Qdrant connectivity) |
| `GET` | `/stream/{analysis_id}` | Server-sent events for real-time analysis progress |

### Example

```bash
# Text-based analysis via the Streamlit dashboard
make demo

# Video + CAN analysis via API
make serve    # Start FastAPI on :8000

curl -X POST http://localhost:8000/analyze \
  -F "video=@dashcam.mp4" \
  -F "can_log=@vehicle.csv" \
  -F "variant=S2"
```

---

## Docker Deployment

Three services defined in `docker-compose.yml`:

| Service | Port | Description |
|---------|------|-------------|
| `qdrant` | 6333 | Qdrant vector database with persistent storage |
| `api` | 8000 | FastAPI analysis backend (depends on Qdrant) |
| `streamlit` | 8501 | Streamlit UI (depends on API) |

```bash
make docker-up          # Start all services
make docker-down        # Stop all services
make docker-build       # Rebuild images
```

For development, you can run Qdrant standalone and use the local dev server:

```bash
docker compose up -d qdrant     # Only start Qdrant
make ingest                     # Ingest knowledge base
make serve                      # Start FastAPI dev server on :8000
make demo                       # Start Streamlit dashboard on :8502
```

---

## Development

### Commands

```bash
make install            # Install package in editable mode
make dev                # Install with dev dependencies
make lint               # Run ruff linter
make format             # Run ruff formatter
make typecheck          # Run mypy type checking
make test               # Unit tests with coverage (206 tests)
make test-all           # Full test suite (unit + integration + evaluation, 244 tests)
make clean              # Remove caches and build artifacts
```

### Configuration

All settings are managed via environment variables with sensible defaults. See `src/haftung_ai/config/settings.py` for the complete list:

| Category | Key Variables | Defaults |
|----------|--------------|----------|
| LLM | `GROQ_MODEL`, `GROQ_TEMPERATURE`, `GROQ_MAX_TOKENS` | `llama-3.3-70b-versatile`, `0.1`, `4096` |
| RAG | `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K_RETRIEVAL`, `HYBRID_ALPHA` | `1000`, `200`, `5`, `0.6` |
| Validation | `VALIDATION_THRESHOLD`, `CONFIDENCE_W_LLM/COVERAGE/BASE` | `0.7`, `0.4/0.3/0.3` |
| Perception | `DETECTOR_MODEL`, `CONFIDENCE_THRESHOLD` | `yolov8n`, `0.25` |
| Safety | `TTC_WARNING_THRESHOLD`, `TTC_CRITICAL_THRESHOLD` | `3.0s`, `1.5s` |

---

## Limitations

1. **Synthetic data.** All CAN bus telemetry is parametrically generated (`scripts/generate_synthetic_can.py`) using internal CAN ID conventions (e.g., `0x100` = speed at 0.1 km/h resolution) that do not correspond to production vehicle DBC files. The 30 evaluation scenarios are hand-authored templates. No real accident data, dashcam footage, or police reports are used.

2. **Evaluation scope.** Metrics are computed over 30 scenarios (6 categories x 5 variations). This sample size demonstrates relative variant differences (S1 vs S2 vs S3) but is insufficient for population-level accuracy claims.

3. **LLM-as-judge bias.** The validation agent's judge shares the same model family as the analysis pipeline. Multi-run aggregation (N=3, mean +/- std) and dedicated variance analysis (`make eval-judge-variance`) partially mitigate this, but systematic bias remains possible.

4. **Confidence weight sensitivity.** Default validation weights (0.4/0.3/0.3) are heuristic. Sensitivity can be verified via `make eval-weight-ablation`, which performs a grid search over all weight combinations summing to 1.0.

5. **Rate limits.** The Groq free tier imposes daily token limits. Running all three variants on complex scenarios can exhaust the quota. Consider upgrading to the Dev tier or reducing the number of judge runs for intensive evaluation sessions.

---

## License

MIT -- see [LICENSE](LICENSE).
