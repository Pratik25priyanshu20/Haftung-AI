# Haftung_AI

[![CI](https://github.com/pratik25priyanshu20/haftung-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/pratik25priyanshu20/haftung-ai/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0-orange)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-purple)

**LLM-powered multi-agent system for automated accident causation analysis under German traffic law (StVO).**

Haftung_AI runs three structurally different analysis variants — S1 (baseline), S2 (RAG-augmented), S3 (RAG + constraint enforcement) — over the same accident scenario and compares them side-by-side. A 30-scenario evaluation framework with ground-truth labels measures how retrieval-augmented generation and claim-level validation improve causation accuracy.

---

## Architecture

```
                         ┌──────────────────────────────────────────────────────┐
                         │                   Haftung_AI Pipeline                │
                         └──────────────────────────────────────────────────────┘

  S1 (Baseline)          Vision ─▶ Telemetry ─▶ Causation ─▶ Report

  S2 (RAG)               Vision ─▶ Telemetry ─▶ RAG ─▶ Causation ─▶ Evidence
                                                          ─▶ Contradiction ─▶ Validation ─▶ Report

  S3 (RAG+Constraints)   Vision ─▶ Telemetry ─▶ RAG ─▶ Causation ─▶ Evidence
                                                          ─▶ Contradiction ─▶ Validation ─┐
                                                               ▲                           │
                                                               └── (correction loop) ◀─────┘
                                                                                      ─▶ Report
```

### Tech Stack

| Layer | Technology |
|---|---|
| LLM inference | Groq API (LLaMA 3.3 70B Versatile) |
| Agent orchestration | LangGraph (StateGraph, compile-time variants) |
| Perception | YOLOv8 (Ultralytics) + DeepSORT tracking + Kalman filtering |
| RAG retrieval | Qdrant vector DB + BGE-large-en-v1.5 embeddings + BM25 reranking |
| Document processing | PyPDF + LangChain text splitters |
| Safety analysis | ASIL classification, TTC computation, plausibility checks |
| Report generation | Jinja2 + WeasyPrint (German Unfallbericht PDF) |
| API | FastAPI + Uvicorn |
| UI | Streamlit (main app + demo dashboard) |
| Telemetry | CAN bus parsing, speed profiling, anomaly detection |
| CI | GitHub Actions (ruff, mypy, pytest, codecov) |

---

## System Variants

| Variant | Pipeline Nodes | Description |
|---|---|---|
| **S1** | Vision &rarr; Telemetry &rarr; Causation &rarr; Report | Pure LLM inference without external knowledge |
| **S2** | Vision &rarr; Telemetry &rarr; RAG &rarr; Causation &rarr; Evidence &rarr; Contradiction &rarr; Validation &rarr; Report | RAG-augmented with StVO statutes and case law |
| **S3** | Same as S2 + conditional loop from Validation back to Causation | S2 + claim-level constraint enforcement; validation can reject and re-trigger causation |

Text-only mode replaces Vision/Telemetry with a TextInput node for scenario-based evaluation.

---

## Project Structure

```
src/haftung_ai/
├── agents/              # LangGraph nodes: orchestrator, causation, evidence, contradiction, report, validation, vision, telemetry, RAG
├── api/                 # FastAPI app and route handlers (analyze, report, health, stream)
│   └── routes/
├── config/              # Pydantic settings (env-based configuration)
├── llm/                 # Groq client, prompt templates, structured output parsing
├── perception/          # YOLOv8 detector, DeepSORT tracker, Kalman filter, scene graph, impact detection
├── rag/                 # Embedding generation, Qdrant vectorstore, hybrid retrieval, BM25 reranker, knowledge base ingestion
├── report/              # PDF generation (WeasyPrint + Jinja2 Unfallbericht template)
│   └── templates/
├── safety/              # ASIL classification, TTC, plausibility checks, DTC logging, sensor health
├── telemetry/           # CAN bus parser, speed profiling, anomaly detection, ego-vehicle reconstruction
├── types/               # Pydantic models: state, detection, track, ego, safety, world_model, causation, report, telemetry
└── ui/                  # Streamlit app + demo dashboard
    └── components/      # Upload, results, scene viewer widgets
```

```
evaluation/
├── dataset/
│   ├── accidents/       # Video + CAN log test data
│   └── scenarios/       # 30 text-only scenarios (6 categories x 5 each)
├── runners/             # run_experiment, run_all_systems, run_stability
└── analysis/            # compare_systems, statistical_tests, results_table, plot_results
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for Qdrant)
- [Groq API key](https://console.groq.com)

### Install

```bash
git clone https://github.com/pratik25priyanshu20/haftung-ai.git
cd haftung-ai
pip install -e ".[dev]"
```

### Environment

```bash
cp .env.example .env
# Edit .env — set GROQ_API_KEY at minimum
```

### Start services

```bash
make docker-up          # Starts Qdrant (required for S2/S3)
make ingest             # Ingest StVO knowledge base into Qdrant
```

### Run a demo

```bash
make demo               # Opens Streamlit demo dashboard on :8502
```

---

## Demo Dashboard

`make demo` launches an interactive Streamlit dashboard (`src/haftung_ai/ui/demo.py`) with:

| Tab | Content |
|---|---|
| **Analysis** | Side-by-side causation results per variant (accident type, primary cause, confidence, responsibility pie charts) |
| **Comparison** | Summary table + grouped bar charts + variant delta metrics (confidence, RAG chunks) |
| **Evidence & RAG** | Retrieved chunks, legal references, claim tables, contradiction alerts |
| **Ground Truth** | Match/mismatch against scenario labels (cause, type, StVO coverage, expected claims) |
| **Report** | German Unfallbericht output per variant |

The sidebar provides a **scenario picker** (6 categories, 30 scenarios with ground truth) or custom text input, variant toggles (S1/S2/S3), and live service health indicators for Groq and Qdrant.

---

## Evaluation

### Dataset

30 German accident scenarios across 6 categories:

| Category | Count | Examples |
|---|---|---|
| Rear-End Collision | 5 | Highway chain reaction, stop-and-go |
| Side Collision | 5 | Lane change, blind spot |
| Head-On Collision | 5 | Overtaking, wrong-way driver |
| Intersection | 5 | Right-of-way, red-light violation |
| Pedestrian | 5 | Crosswalk, jaywalking, school zone |
| Single Vehicle | 5 | Aquaplaning, tire blowout |

Each scenario includes `scenario_text`, `ground_truth` (primary cause, accident type, responsibility distribution, relevant StVO sections, expected claims).

### Metrics

- **Primary cause accuracy** — exact / fuzzy match against ground truth
- **Accident type accuracy** — categorical match
- **Responsibility deviation** — absolute percentage-point delta per party
- **StVO coverage** — fraction of relevant statutes referenced
- **Claim coverage** — keyword overlap with expected claims
- **Confidence calibration** — model confidence vs actual correctness

### Run evaluation

```bash
make eval-text-all        # Run all 3 variants across 30 scenarios
make eval-text-stability  # 5 reruns per variant for stability analysis
make eval-tables          # Generate comparison tables, statistical tests, plots
```

---

## Docker

Three services defined in `docker-compose.yml`:

| Service | Port | Description |
|---|---|---|
| `api` | 8000 | FastAPI analysis backend |
| `qdrant` | 6333 | Qdrant vector database |
| `streamlit` | 8501 | Streamlit UI |

```bash
make docker-up            # Start all services
make docker-down          # Stop all services
make docker-build         # Rebuild images
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze` | Run accident analysis (multipart: video, CAN log, variant) |
| `GET` | `/report/{analysis_id}` | Retrieve analysis report by ID |
| `GET` | `/health` | Service health check |

### Example

```bash
curl -X POST http://localhost:8000/analyze \
  -F "video=@dashcam.mp4" \
  -F "can_log=@vehicle.csv" \
  -F "variant=S2"
```

---

## Development

```bash
make lint                 # Ruff linter
make format               # Ruff formatter
make typecheck            # Mypy type checking
make test                 # Unit tests with coverage
make test-all             # All tests (unit + integration + evaluation)
make clean                # Remove caches and build artifacts
```

---

## License

MIT -- see [LICENSE](LICENSE).
