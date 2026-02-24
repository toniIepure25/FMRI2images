PY ?= python3
SUBJECT ?= subj01
CACHE_ROOT ?= cache
DEVICE ?= cuda

# Optional .env loading (best-effort)
-include .env
export

# ============================================================================
# Phony targets
# ============================================================================

.PHONY: help setup preflight doctor smoke
.PHONY: prepare data models index preprocess preextract clip-cache build-clip-cache
.PHONY: train ablation
.PHONY: eval-recon eval-shared1000 summarize-shared1000 compare-evals
.PHONY: test test-quick
.PHONY: clean clean-logs download-sd check-headers
.PHONY: ridge ablate fit-preproc
.PHONY: paper manifest-check repro-check

# ============================================================================
# Help
# ============================================================================

help:
	@echo "fmri2img -- fMRI-to-Image Neural Decoding (Phase 2: ViT-L/14, 768-D)"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          Install package in editable mode"
	@echo "  make preflight      System readiness checks (Python, CUDA, disk)"
	@echo "  make doctor         Comprehensive readiness check"
	@echo "  make prepare        Full data pipeline (data + models + index + preprocess + clip-cache)"
	@echo ""
	@echo "Data Preparation:"
	@echo "  make index          Build canonical NSD index"
	@echo "  make preprocess     Fit preprocessing pipeline (scaler + reliability + PCA)"
	@echo "  make preextract     Pre-extract ROI-masked fMRI features (fast training)"
	@echo "  make clip-cache     Build CLIP embeddings cache"
	@echo ""
	@echo "Training:"
	@echo "  make train CONFIG=configs/experiments/B0_deterministic.yaml"
	@echo "                      Train a single experiment"
	@echo "  make ablation       Run full B0-N4 ablation ladder"
	@echo "  make ridge          Train Ridge baseline (fMRI -> CLIP)"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval-recon     Evaluate reconstruction quality"
	@echo "  make eval-shared1000  Paper-grade Shared1000 evaluation"
	@echo "  make compare-evals  Aggregate evaluations with bootstrap CIs"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run full test suite"
	@echo "  make test-quick     Run fast unit tests only"
	@echo ""
	@echo "Utilities:"
	@echo "  make download-sd    Download Stable Diffusion model"
	@echo "  make check-headers  Validate NSD index header bounds"
	@echo "  make clean          Remove build artifacts and caches"
	@echo ""
	@echo "Environment Variables:"
	@echo "  SUBJECT=subj01      NSD subject ID (default: subj01)"
	@echo "  DEVICE=cuda         Compute device (default: cuda)"
	@echo "  LIMIT=N             Limit number of samples to process"
	@echo "  GPU=0               GPU device ID for training"
	@echo ""

# ============================================================================
# Setup
# ============================================================================

setup:
	@echo "=== Installing fmri2img (editable) ==="
	pip install -e ".[train,diffusion]"
	@echo "Done. Next: make preflight"

preflight:
	$(PY) scripts/utils/preflight.py

doctor:
	$(PY) scripts/utils/doctor.py

smoke:
	$(PY) scripts/utils/smoke.py

# ============================================================================
# Data Preparation
# ============================================================================

prepare: data models index preprocess preextract clip-cache
	@echo "=== Prepare complete ==="

data:
	$(PY) scripts/utils/verify_dataset.py --subject $(SUBJECT)

models:
	@mkdir -p $(CACHE_ROOT)/.markers
	@if [ -f "$(CACHE_ROOT)/.markers/models.ok" ]; then \
		echo "models: already prepared ($(CACHE_ROOT)/.markers/models.ok)"; \
	else \
		$(PY) scripts/utils/fetch_models.py && \
		date -Iseconds > "$(CACHE_ROOT)/.markers/models.ok"; \
	fi

index:
	@mkdir -p data/indices/nsd_index
	@if [ -f "data/indices/nsd_index/subject=$(SUBJECT)/index.parquet" ]; then \
		echo "index: already present (data/indices/nsd_index/subject=$(SUBJECT)/index.parquet)"; \
	else \
		$(PY) scripts/build/build_full_index.py --subject $(SUBJECT) \
			--output data/indices/nsd_index/subject=$(SUBJECT)/index.parquet; \
	fi

preprocess:
	@mkdir -p $(CACHE_ROOT)/.markers
	@if [ -f "$(CACHE_ROOT)/.markers/preprocess_$(SUBJECT).ok" ]; then \
		echo "preprocess: already prepared ($(CACHE_ROOT)/.markers/preprocess_$(SUBJECT).ok)"; \
	else \
		mkdir -p $(CACHE_ROOT)/preproc && \
		$(PY) scripts/build/fit_preprocessing.py \
			--subject $(SUBJECT) \
			--index-file data/indices/nsd_index/subject=$(SUBJECT)/index.parquet \
			--output-dir $(CACHE_ROOT)/preproc/subject=$(SUBJECT) && \
		date -Iseconds > "$(CACHE_ROOT)/.markers/preprocess_$(SUBJECT).ok"; \
	fi

fit-preproc: preprocess

preextract:
	@mkdir -p $(CACHE_ROOT)/.markers
	@if [ -f "$(CACHE_ROOT)/.markers/preextract_$(SUBJECT).ok" ]; then \
		echo "preextract: already done ($(CACHE_ROOT)/.markers/preextract_$(SUBJECT).ok)"; \
	else \
		mkdir -p $(CACHE_ROOT)/preextracted && \
		$(PY) scripts/build/preextract_fmri.py \
			--subject $(SUBJECT) \
			--index-file data/indices/nsd_index/subject=$(SUBJECT)/index.parquet \
			--output-dir $(CACHE_ROOT)/preextracted/subject=$(SUBJECT) && \
		date -Iseconds > "$(CACHE_ROOT)/.markers/preextract_$(SUBJECT).ok"; \
	fi

clip-cache: build-clip-cache
	@mkdir -p $(CACHE_ROOT)/.markers
	@date -Iseconds > "$(CACHE_ROOT)/.markers/clip_cache_$(SUBJECT).ok"

build-clip-cache:
	@mkdir -p outputs/clip_cache
	@if [ -f "$${CACHE:-outputs/clip_cache/clip.parquet}" ]; then \
		echo "clip-cache: already present ($${CACHE:-outputs/clip_cache/clip.parquet})"; \
	else \
		$(PY) scripts/build/build_clip_cache.py \
			$${INDEX_FILE:+--index-file $$INDEX_FILE} \
			$${INDEX_ROOT:+--index-root $$INDEX_ROOT} \
			$${SUBJECT:+--subject $$SUBJECT} \
			--cache $${CACHE:-outputs/clip_cache/clip.parquet} \
			--batch $${BATCH:-128} \
			--device $(DEVICE) \
			$${LIMIT:+--limit $$LIMIT}; \
	fi

# ============================================================================
# Training
# ============================================================================

# Single experiment: make train CONFIG=configs/experiments/B0_deterministic.yaml
train:
	@test -n "$${CONFIG}" || (echo "ERROR: set CONFIG=configs/experiments/<file>.yaml" && exit 2)
	$(PY) scripts/training/train_unified.py \
		--config $${CONFIG} \
		--gpu $${GPU:-0} \
		$${SUBJECT:+--subject $$SUBJECT}

# Full B0-N4 ablation ladder (training only)
ablation:
	bash scripts/training/run_ablation_ladder.sh \
		--subjects "$${SUBJECTS:-subj01 subj02 subj05 subj07}" \
		--gpu $${GPU:-0}

# Full pipeline: data prep + training + reconstruction + evaluation + aggregation
full-pipeline:
	bash scripts/orchestration/run_full_ablation.sh \
		--subjects "$${SUBJECTS:-subj01 subj02 subj05 subj07}" \
		--gpu $${GPU:-0}

# Aggregate ablation results into paper tables
aggregate:
	$(PY) scripts/evaluation/aggregate_ablation.py \
		--results-dir experimental_results \
		--subjects $${SUBJECTS:-subj01 subj02 subj05 subj07}

# Ridge baseline
ridge:
	@echo "=== Training Ridge Baseline ==="
	$(PY) scripts/training/train_ridge.py \
		--index-root data/indices/nsd_index \
		--subject $(SUBJECT) \
		--use-preproc \
		--clip-cache outputs/clip_cache/clip.parquet \
		--alpha-grid "0.1,1,3,10,30,100" \
		--limit $${LIMIT:-2048}
	@echo "=== Ridge training complete ==="

# Ridge ablation study
ablate:
	$(PY) scripts/analysis/ablate_preproc_and_ridge.py \
		--index-root data/indices/nsd_index \
		--subject $(SUBJECT) \
		--clip-cache outputs/clip_cache/clip.parquet \
		--rel-grid "0.05,0.1,0.2" \
		--k-grid "512,1024,4096" \
		--limit $${LIMIT:-4096}

# ============================================================================
# Evaluation
# ============================================================================

eval-recon:
	@echo "=== Evaluating Reconstructions ==="
	$(PY) scripts/evaluation/eval_reconstruction.py \
		--index-root data/indices/nsd_index \
		--subject $(SUBJECT) \
		--recon-dir $${RECON_DIR:-outputs/recon/$(SUBJECT)/production_final} \
		--clip-cache outputs/clip_cache/clip.parquet \
		--out-csv outputs/reports/$(SUBJECT)/recon_eval.csv \
		--out-json outputs/reports/$(SUBJECT)/recon_eval.json \
		--out-fig outputs/reports/$(SUBJECT)/recon_grid.png

SHARED1000_OUT := outputs/eval_shared1000
STRATEGIES := single best_of_8 boi_lite
REP_MODE := avg
SEEDS := 0 1 2

eval-shared1000:
	@echo "=== Paper-Grade Shared1000 Evaluation ==="
	@mkdir -p $(SHARED1000_OUT)/$(SUBJECT)
	$(PY) scripts/evaluation/eval_shared1000_full.py \
		--subject $(SUBJECT) \
		--encoder-checkpoint $${ENCODER_CKPT} \
		--encoder-type $${ENCODER_TYPE:-unified} \
		--output-dir $(SHARED1000_OUT)/$(SUBJECT) \
		--rep-mode $(REP_MODE) \
		--strategies $(STRATEGIES) \
		--seeds $(SEEDS) \
		--clip-cache $${CLIP_CACHE:-outputs/clip_cache/clip.parquet} \
		$${USE_CEILING:+--use-noise-ceiling} \
		--device $(DEVICE)

summarize-shared1000:
	$(PY) scripts/evaluation/summarize_shared1000.py \
		--eval-dir $(SHARED1000_OUT) \
		--output-dir $(SHARED1000_OUT) \
		--subjects $${SUBJECTS:-subj01 subj02 subj05 subj07} \
		--strategies $(STRATEGIES) \
		--rep-mode $(REP_MODE)

compare-evals:
	$(PY) scripts/analysis/compare_evals.py \
		--report-dir outputs/reports/$(SUBJECT) \
		--out-csv outputs/reports/$(SUBJECT)/recon_compare.csv \
		--out-tex outputs/reports/$(SUBJECT)/recon_compare.tex \
		--out-md  outputs/reports/$(SUBJECT)/recon_compare.md \
		--out-fig outputs/reports/$(SUBJECT)/recon_compare.png \
		$${PATTERN:+--pattern $$PATTERN} \
		$${BOOTS:+--boots $$BOOTS}

# ============================================================================
# Testing
# ============================================================================

test:
	$(PY) -m pytest tests/ -v

test-quick:
	$(PY) -m pytest tests/ -v -x --ignore=tests/unit -k "not slow"

# ============================================================================
# Utilities
# ============================================================================

check-headers:
	@echo "=== Validating Index Headers ==="
	$(PY) scripts/utils/check_index_headers.py \
		data/indices/nsd_index/subject=$(SUBJECT)/index.parquet \
		--max-files 10

download-sd:
	$(PY) scripts/utils/download_sd_model.py \
		--model-id $${MODEL:-sd2-community/stable-diffusion-2-1}

paper:
	$(PY) scripts/analysis/build_paper_artifacts.py

manifest-check:
	@test -n "$${OUTPUT_DIR}" || (echo "ERROR: set OUTPUT_DIR=..." && exit 2)
	$(PY) scripts/utils/write_run_manifest.py --output-dir $${OUTPUT_DIR}

repro-check:
	$(PY) -c "from fmri2img.utils.manifest import gather_env_info; print('git_commit' in gather_env_info())"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache/ build/ dist/ *.egg-info/

clean-logs:
	rm -rf outputs/logs/*.log
