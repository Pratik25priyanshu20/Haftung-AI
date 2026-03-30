.PHONY: install dev lint format typecheck test test-unit test-integration test-eval test-all \
       serve ui demo docker-up docker-down ingest eval eval-stability clean \
       eval-text eval-text-all eval-text-stability eval-tables create-text-scenarios \
       eval-judge-variance eval-weight-ablation

# ─── Setup ──────────────────────────────────────────────────────────────
install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# ─── Code Quality ───────────────────────────────────────────────────────
lint:
	ruff check src/ tests/ evaluation/

format:
	ruff format src/ tests/ evaluation/

typecheck:
	mypy src/haftung_ai/ --ignore-missing-imports

# ─── Tests ──────────────────────────────────────────────────────────────
test: test-unit

test-unit:
	pytest tests/unit/ -v --cov=src/haftung_ai --cov-report=term-missing

test-integration:
	pytest tests/integration/ -v

test-eval:
	pytest tests/evaluation/ -v

test-all:
	pytest tests/ -v --cov=src/haftung_ai --cov-report=term-missing --cov-report=html

# ─── Run Services ───────────────────────────────────────────────────────
serve:
	uvicorn haftung_ai.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run src/haftung_ai/ui/app.py --server.port 8501

demo:
	streamlit run src/haftung_ai/ui/demo.py --server.port 8502

# ─── Docker ─────────────────────────────────────────────────────────────
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

# ─── Knowledge Base ─────────────────────────────────────────────────────
ingest:
	python scripts/ingest_knowledge_base.py

ingest-recreate:
	python scripts/ingest_knowledge_base.py --recreate

# ─── Evaluation ─────────────────────────────────────────────────────────
DATASET ?= evaluation/dataset/accidents
RESULTS ?= evaluation/results

eval:
	python evaluation/runners/run_all_systems.py --dataset $(DATASET) --output $(RESULTS)

eval-s1:
	python evaluation/runners/run_experiment.py --dataset $(DATASET) --variant S1 --output $(RESULTS)

eval-s2:
	python evaluation/runners/run_experiment.py --dataset $(DATASET) --variant S2 --output $(RESULTS)

eval-s3:
	python evaluation/runners/run_experiment.py --dataset $(DATASET) --variant S3 --output $(RESULTS)

eval-stability:
	python evaluation/runners/run_stability.py --dataset $(DATASET) --variant S2 --reruns 5 --output $(RESULTS)

eval-compare:
	python evaluation/analysis/compare_systems.py --results $(RESULTS)
	python evaluation/analysis/plot_results.py --results $(RESULTS)
	python evaluation/analysis/statistical_tests.py --results $(RESULTS)

# ─── Text-Only Evaluation ───────────────────────────────────────────────
SCENARIOS ?= evaluation/dataset/scenarios

eval-text:
	python evaluation/runners/run_experiment.py --dataset $(SCENARIOS) --variant S2 --output $(RESULTS) --text

eval-text-all:
	python evaluation/runners/run_all_systems.py --dataset $(SCENARIOS) --output $(RESULTS) --text

eval-text-stability:
	@for variant in S1 S2 S3; do \
		echo "=== Stability test for $$variant ==="; \
		python evaluation/runners/run_stability.py --dataset $(SCENARIOS) --variant $$variant --reruns 5 --output $(RESULTS) --text; \
	done

eval-tables:
	python evaluation/analysis/compare_systems.py --results $(RESULTS) --text
	python evaluation/analysis/statistical_tests.py --results $(RESULTS) --text
	python evaluation/analysis/results_table.py --results $(RESULTS) --text
	python evaluation/analysis/plot_results.py --results $(RESULTS) --text

# ─── Judge Variance & Weight Ablation ──────────────────────────────────
eval-judge-variance:
	python evaluation/runners/run_judge_variance.py --scenarios $(SCENARIOS) --variant S2 --n-runs 5 --output $(RESULTS)

eval-weight-ablation:
	python evaluation/runners/run_weight_ablation.py --scenarios $(SCENARIOS) --output $(RESULTS)

# ─── Dataset ────────────────────────────────────────────────────────────
create-dataset:
	python scripts/create_eval_dataset.py

create-text-scenarios:
	python scripts/create_text_scenarios.py

generate-can:
	python scripts/generate_synthetic_can.py

# ─── Cleanup ────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
