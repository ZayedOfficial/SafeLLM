.PHONY: dataset train-verifier train-scout train-analyst \
        eval-full eval-external repro-paper deploy clean

# =========================================================
# DATASET
# =========================================================
dataset:
	@echo "📦 Building SafeLang-1M dataset..."
	python datasets/safelang_1m.py --stream --max-samples 1000

dataset-full:
	@echo "📦 Building FULL SafeLang-1M dataset (slow)..."
	python datasets/safelang_1m.py

dataset-publish:
	@echo "🚀 Publishing dataset to HuggingFace Hub..."
	python datasets/safelang_1m.py --push-to-hub

# =========================================================
# TRAINING  (verifier runs locally; scout/analyst via Colab)
# =========================================================
train-verifier:
	@echo "🔧 Training DeBERTa-v3-large verifier..."
	python train.py verifier \
		--model microsoft/deberta-v3-large \
		--epochs 5 \
		--batch-size 8 \
		--seed 42 \
		--certify

train-verifier-fast:
	@echo "🔧 Training verifier (quick test, 1 epoch)..."
	python train.py verifier --epochs 1 --batch-size 4 --seed 42

train-scout:
	@echo "🤗 Scout training: see notebooks/train_scout.ipynb (Colab A100)"
	python train.py scout --lora-r 16 --seed 42

train-analyst:
	@echo "🤗 Analyst training: see notebooks/train_analyst.ipynb (Colab A100)"
	python train.py analyst --lora-r 32 --seed 42

train-all: train-verifier train-scout train-analyst

# =========================================================
# EVALUATION
# =========================================================
eval-full:
	@echo "📊 Running full held-out evaluation..."
	python eval/external_benchmarks.py --benchmarks all --max-samples 500

eval-mock:
	@echo "📊 Running evaluation in mock mode (no API)..."
	python eval/external_benchmarks.py --benchmarks jailbreakbench harmbench --mock --max-samples 50

# =========================================================
# PAPER
# =========================================================
tables:
	@echo "📋 Generating LaTeX tables..."
	python eval/generate_tables.py --mock

paper:
	@echo "📄 Generating IEEE paper..."
	python paper/generate_ieee_paper.py

repro-paper: tables paper
	@echo "✅ Paper artefacts ready in paper/"
	@echo "   Compile: cd paper && pdflatex safelang1m_ieeesp2027.tex"

# =========================================================
# DEPLOYMENT
# =========================================================
serve:
	@echo "🚀 Starting SafeLang API on port 8000..."
	cd deploy && python -m uvicorn app:app --host 0.0.0.0 --port 8000

onnx-export:
	@echo "📦 Exporting verifier to ONNX..."
	python deploy/onnx_export.py --model-id checkpoints/verifier/best

docker-build:
	docker compose -f deploy/docker-compose.yml build

docker-up:
	docker compose -f deploy/docker-compose.yml up -d

docker-down:
	docker compose -f deploy/docker-compose.yml down

# =========================================================
# TESTS
# =========================================================
test:
	@echo "🧪 Running test suite..."
	python -m pytest tests/ -v --tb=short

test-quick:
	@echo "🧪 Running quick unit tests (no API)..."
	python -m pytest tests/test_arbiter.py tests/test_metrics.py -v

# =========================================================
# SMOKE TEST (no GPU, no API needed)
# =========================================================
smoke:
	@echo "💨 Running smoke tests..."
	python -c "from models.certified_verifier import CertifiedVerifier, LipschitzCertifier; print('✅ models.certified_verifier OK')"
	python -c "from arbiter.z3_safety import SafetyArbiter; print('✅ arbiter.z3_safety OK')"
	python -c "from eval.metrics import compute_metrics; print('✅ eval.metrics OK')"
	python -c "from pipeline import SafeLangPipeline, SafetyResult; print('✅ pipeline OK')"
	python -m pytest tests/test_arbiter.py tests/test_metrics.py -v --tb=short
	python eval/external_benchmarks.py --benchmarks jailbreakbench --mock --max-samples 20
	python eval/generate_tables.py --mock
	python paper/generate_ieee_paper.py
	@echo ""
	@echo "✅ All smoke tests passed!"

# =========================================================
# FULL REPRO
# =========================================================
repro-full: dataset-full train-all eval-full repro-paper
	@echo ""
	@echo "🎉 Full reproduction complete!"

# =========================================================
# CLEANUP
# =========================================================
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf checkpoints/ wandb/ runs/ .pytest_cache/

clean-paper:
	cd paper && rm -f *.aux *.bbl *.blg *.log *.out *.pdf *.synctex.gz *.toc 2>/dev/null; true
