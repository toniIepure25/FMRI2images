PY=python
PREPROC_FLAG := $(if $(USE_PREPROC),--use-preproc,)
PREPROC_DIR_FLAG := $(if $(PREPROC_DIR),--preproc-dir $(PREPROC_DIR),)

.PHONY: setup index test demo sanity read-index check-index clean build-clip-cache check-headers clip-cache-small smoke-tests clean-logs help ridge repair-adapter

help:
	@echo "Bachelor V2 - fMRI to Image Pipeline"
	@echo ""
	@echo "Smoke Tests & Quick Sanity Checks:"
	@echo "  make check-headers      - Validate beta_index bounds in index files"
	@echo "  make clip-cache-small   - Build small CLIP cache (256 samples)"
	@echo "  make smoke-tests        - Run all smoke tests (headers + small cache)"
	@echo ""
	@echo "Main Targets:"
	@echo "  make setup              - Install package in development mode"
	@echo "  make index              - Build canonical NSD index"
	@echo "  make build-clip-cache   - Build CLIP embeddings cache"
	@echo "  make fit-preproc        - Fit preprocessing pipeline (scaler + reliability + PCA)"
	@echo "  make ridge              - Train Ridge baseline (fMRI → CLIP)"
	@echo "  make mlp                - Train MLP encoder (fMRI → CLIP)"
	@echo "  make clip-adapter       - Train CLIP adapter (512D → 768/1024D)"
	@echo "  make test               - Run comprehensive tests"
	@echo "  make test-reliability   - Run reliability module tests"
	@echo "  make demo               - Run IO layer demo"
	@echo "  make train-smoke        - Run training smoke test"
	@echo ""
	@echo "Evaluation & Reporting:"
	@echo "  make eval-recon         - Evaluate reconstructions (512-D space)"
	@echo "  make eval-recon-adapter - Evaluate reconstructions (768/1024-D target space)"
	@echo "  make recon-eval         - Generate + evaluate (512-D, one-click)"
	@echo "  make recon-eval-adapter - Generate + evaluate (768/1024-D, one-click)"
	@echo "  make compare-evals      - Aggregate multiple evaluations with bootstrap CIs"
	@echo ""
	@echo "Paper-Grade Shared1000 Evaluation:"
	@echo "  make eval-shared1000    - Run comprehensive Shared1000 evaluation"
	@echo "  make summarize-shared1000 - Aggregate results across subjects/strategies"
	@echo "  make test-pipeline      - Run evaluation pipeline smoke test"
	@echo ""
	@echo "Environment Variables:"
	@echo "  USE_PREPROC=1                     - Enable preprocessing (auto-detected from checkpoint if not set)"
	@echo "  PREPROC_DIR=<path>                - Override preprocessing directory (auto-discovered if not set)"
	@echo "  MODEL=<model-id>                  - Override diffusion model (e.g., stabilityai/stable-diffusion-2-1)"
	@echo "  LIMIT=<n>                         - Limit number of samples to process"
	@echo ""
	@echo "Diffusion Image Generation:"
	@echo "  make download-sd        - Download Stable Diffusion model (one-time, ~5GB)"
	@echo "  make check-sd           - Check if SD model is cached"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean              - Clean cache and build artifacts"
	@echo "  make clean-logs         - Remove log files"
	@echo "  make repair-adapter     - Backfill missing metadata in adapter checkpoint"
	@echo ""

setup:
	pip install -e .

# Build canonical index with unified API
index:
	$(PY) -m fmri2img.data.nsd_index_builder --subjects $${SUBJECTS:-subj01} $${MAX_TRIALS:+--max-trials $$MAX_TRIALS} --output-format parquet

# Build CLIP embeddings cache with resume support
build-clip-cache:
	$(PY) scripts/build_clip_cache.py \
		$${INDEX_FILE:+--index-file $$INDEX_FILE} \
		$${INDEX_ROOT:+--index-root $$INDEX_ROOT} \
		$${SUBJECT:+--subject $$SUBJECT} \
		--cache $${CACHE:-outputs/clip_cache/clip.parquet} \
		--batch $${BATCH:-128} \
		--device $${DEVICE:-cuda} \
		$${LIMIT:+--limit $$LIMIT}

# Run comprehensive tests
test:
	$(PY) -m pytest src/fmri2img/scripts/test_*.py -v

# Run IO layer demo with unified API
demo:
	$(PY) src/fmri2img/scripts/io_layer_demo.py

# Alias for demo (for backward compatibility)
sanity: demo

# Read canonical index with filtering
read-index:
	$(PY) src/fmri2img/scripts/nsd_index_reader.py --index $${INDEX:-data/indices/nsd_index/subject=subj01/index.parquet} --subject $${SUBJECT:-subj01} --n 10

# Check index header bounds (verbose version with all beta files)
check-index:
	$(PY) scripts/check_index_headers.py data/indices/nsd_index/subject=subj01/index.parquet

# Quick header validation (checks only 10 files for smoke test)
check-headers:
	@echo "=== Validating Index Headers (10 files sample) ==="
	@$(PY) scripts/check_index_headers.py \
		data/indices/nsd_index/subject=subj01/index.parquet \
		--max-files 10
	@echo "✅ Header validation passed"

# Build small CLIP cache for smoke testing (256 samples, uses configs/clip.yaml)
clip-cache-small:
	@echo "=== Building Small CLIP Cache (256 samples) ==="
	@mkdir -p outputs/clip_cache
	@$(PY) scripts/build_clip_cache.py \
		--index-file data/indices/nsd_index/subject=subj01/index.parquet \
		--cache outputs/clip_cache/clip_smoke.parquet \
		--batch 64 \
		--device cuda \
		--limit 256
	@echo "✅ Small CLIP cache built successfully"

# Run all smoke tests
smoke-tests: check-headers clip-cache-small
	@echo ""
	@echo "✅ All smoke tests passed!"

# Train Ridge baseline (fMRI → CLIP)
ridge:
	@echo "=== Training Ridge Baseline ==="
	@$(PY) scripts/train_ridge.py \
		--index-root data/indices/nsd_index \
		--subject subj01 \
		--use-preproc \
		--clip-cache outputs/clip_cache/clip.parquet \
		--alpha-grid "0.1,1,3,10,30,100" \
		--limit 2048
	@echo "✅ Ridge training complete"

# Ablation study: reliability threshold × PCA dimensionality
ablate:
	@echo "=== Ridge Ablation Study ==="
	@$(PY) scripts/ablate_preproc_and_ridge.py \
		--index-root data/indices/nsd_index \
		--subject subj01 \
		--clip-cache outputs/clip_cache/clip.parquet \
		--rel-grid "0.05,0.1,0.2" \
		--k-grid "512,1024,4096" \
		--limit $${LIMIT:-4096}
	@echo "✅ Ablation study complete: outputs/reports/subj01/ablation_ridge.csv"

# Ablation study running MLP for quick comparison
ablate-mlp:
	@echo "=== MLP Ablation Study ==="
	@$(PY) scripts/ablate_preproc_and_ridge.py \
		--index-root data/indices/nsd_index \
		--subject subj01 \
		--clip-cache outputs/clip_cache/clip.parquet \
		--model mlp \
		--rel-grid "0.1,0.2" \
		--k-grid "512,1024" \
		--hidden 1024 --dropout 0.1 --lr 1e-3 --wd 1e-4 --epochs 50 --patience 7 \
		--batch-size 256 --limit $${LIMIT:-2048}
	@echo "✅ MLP ablation study complete: outputs/reports/subj01/ablation_ridge.csv"

# Train MLP encoder (fMRI → CLIP)
mlp:
	@echo "=== Training MLP Encoder ==="
	@$(PY) scripts/train_mlp.py \
		--index-root data/indices/nsd_index \
		--subject subj01 \
		--use-preproc \
		--clip-cache outputs/clip_cache/clip.parquet \
		--hidden 1024 --dropout 0.1 \
		--lr 1e-3 --wd 1e-4 --epochs 50 --patience 7 \
		--batch-size 256 --limit $${LIMIT:-2048}
	@echo "✅ MLP training complete"

# Train CLIP adapter (512D → 768/1024D for diffusion models)
clip-adapter:
	@echo "=== Training CLIP Adapter ==="
	@$(PY) scripts/train_clip_adapter.py \
		--index-root data/indices/nsd_index \
		--subject subj01 \
		--clip-cache outputs/clip_cache/clip.parquet \
		--model-id stabilityai/stable-diffusion-2-1 \
		--epochs 30 --batch-size 256 --limit $${LIMIT:-4096} \
		--out checkpoints/clip_adapter/subj01/adapter.pt
	@echo "✅ CLIP adapter training complete"

# Evaluate reconstructed images (512-D ViT-B/32 space)
eval-recon:
	@echo "=== Evaluating Reconstruction (512-D) ==="
	@$(PY) scripts/eval_reconstruction.py \
		--index-root data/indices/nsd_index \
		--subject $${SUBJECT:-subj01} \
		--recon-dir $${RECON_DIR:-outputs/recon/subj01/run_001} \
		--clip-cache outputs/clip_cache/clip.parquet \
		--out-csv outputs/reports/$${SUBJECT:-subj01}/recon_eval.csv \
		--out-fig outputs/reports/$${SUBJECT:-subj01}/recon_grid.png
	@echo "✅ Reconstruction evaluation complete"

# Evaluate reconstructed images with adapter (768/1024-D target space)
eval-recon-adapter:
	@echo "=== Evaluating Reconstruction (1024-D target) ==="
	@$(PY) scripts/eval_reconstruction.py \
		--index-root data/indices/nsd_index \
		--subject $${SUBJECT:-subj01} \
		--recon-dir $${RECON_DIR:-outputs/recon/subj01/run_001} \
		--clip-cache outputs/clip_cache/clip.parquet \
		--use-adapter --model-id stabilityai/stable-diffusion-2-1 \
		--out-csv outputs/reports/$${SUBJECT:-subj01}/recon_eval_1024.csv \
		--out-fig outputs/reports/$${SUBJECT:-subj01}/recon_grid_1024.png
	@echo "✅ Reconstruction evaluation complete"

# One-click: Generate + Evaluate reconstructions (512-D, no adapter)
recon-eval:
	@echo "=== Reconstruct & Evaluate (512-D, no adapter) ==="
	@$(PY) scripts/run_reconstruct_and_eval.py \
		--subject $${SUBJECT:-subj01} \
		--encoder $${ENCODER:-mlp} \
		--ckpt $${CKPT:-checkpoints/mlp/subj01/mlp.pt} \
		--clip-cache outputs/clip_cache/clip.parquet \
		$${MODEL:+--model-id $$MODEL} \
		$(PREPROC_FLAG) \
		$(PREPROC_DIR_FLAG) \
		--output-dir outputs/recon/$${SUBJECT:-subj01}/auto_no_adapter \
		--report-dir outputs/reports/$${SUBJECT:-subj01} \
		--limit $${LIMIT:-64} \
		$${INDEX_ROOT:+--index-root $$INDEX_ROOT} \
		$${INDEX_FILE:+--index-file $$INDEX_FILE}
	@echo "✅ Reconstruct & evaluate complete"

# One-click: Generate + Evaluate reconstructions (768/1024-D, with adapter)
recon-eval-adapter:
	@echo "=== Reconstruct & Evaluate (1024-D, with adapter) ==="
	@$(PY) scripts/run_reconstruct_and_eval.py \
		--subject $${SUBJECT:-subj01} \
		--encoder $${ENCODER:-mlp} \
		--ckpt $${CKPT:-checkpoints/mlp/subj01/mlp.pt} \
		--clip-cache outputs/clip_cache/clip.parquet \
		--use-adapter \
		--adapter $${ADAPTER:-checkpoints/clip_adapter/subj01/adapter.pt} \
		--model-id $${MODEL:-stabilityai/stable-diffusion-2-1} \
		$(PREPROC_FLAG) \
		$(PREPROC_DIR_FLAG) \
		--output-dir outputs/recon/$${SUBJECT:-subj01}/auto_with_adapter \
		--report-dir outputs/reports/$${SUBJECT:-subj01} \
		--limit $${LIMIT:-64} \
		$${INDEX_ROOT:+--index-root $$INDEX_ROOT} \
		$${INDEX_FILE:+--index-file $$INDEX_FILE}
	@echo "✅ Reconstruct & evaluate complete"

# Aggregate multiple evaluations with bootstrap confidence intervals
compare-evals:
	@echo "=== Comparing Evaluations ==="
	@$(PY) scripts/compare_evals.py \
		--report-dir outputs/reports/$${SUBJECT:-subj01} \
		--out-csv outputs/reports/$${SUBJECT:-subj01}/recon_compare.csv \
		--out-tex outputs/reports/$${SUBJECT:-subj01}/recon_compare.tex \
		--out-md  outputs/reports/$${SUBJECT:-subj01}/recon_compare.md \
		--out-fig outputs/reports/$${SUBJECT:-subj01}/recon_compare.png \
		$${PATTERN:+--pattern $$PATTERN} \
		$${BOOTS:+--boots $$BOOTS}
	@echo "✅ Evaluation comparison complete"

train-smoke:
	$(PY) scripts/train_smoke.py --index-root $${INDEX:-data/indices/nsd_index} $(ARGS)

# Fit preprocessing pipeline with split-half reliability
fit-preproc:
	@echo "=== Fitting Preprocessing Pipeline ==="
	@mkdir -p outputs/preproc/$${SUBJECT:-subj01}
	$(PY) scripts/nsd_fit_preproc.py \
		--subject $${SUBJECT:-subj01} \
		--k $${K:-4096} \
		--reliability-thr $${THR:-0.1} \
		--min-variance $${MINVAR:-1e-6} \
		--min-repeat-ids $${MINREP:-20} \
		--seed $${SEED:-42} \
		$${NOPCA:+--no-pca} \
		$${ROI:+--roi-mode $$ROI}
	@echo "✅ Preprocessing fitted successfully"

test-preproc:
	$(PY) -m pytest src/fmri2img/scripts/test_preprocess.py -v

test-reliability:
	$(PY) -m pytest src/fmri2img/scripts/test_reliability.py -v

# Clean up cache and build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf cache/
	rm -f test_unified_index.parquet

# Clean up log files
clean-logs:
	@echo "Removing log files from outputs/logs/..."
	@rm -rf outputs/logs/*.log
	@echo "✅ Logs cleaned"

# Download Stable Diffusion model to cache (one-time)
download-sd:
	@echo "Downloading Stable Diffusion model to cache..."
	$(PY) scripts/download_sd_model.py --model-id $${MODEL:-stabilityai/stable-diffusion-2-1}
	@echo "✅ Model downloaded and cached"

# Check if Stable Diffusion model is cached
check-sd:
	@echo "Checking HuggingFace cache status..."
	$(PY) scripts/check_hf_cache.py --model-id $${MODEL:-stabilityai/stable-diffusion-2-1}

# Repair adapter checkpoint metadata (backfill missing fields)
repair-adapter:
	@echo "=== Repairing Adapter Metadata ==="
	@$(PY) scripts/repair_adapter_metadata.py \
		--adapter $${ADAPTER:-checkpoints/clip_adapter/subj01/adapter.pt} \
		--subject $${SUBJECT:-subj01} \
		--model-id $${MODEL:-stabilityai/stable-diffusion-2-1}
	@echo "✅ Adapter metadata repaired"

# ════════════════════════════════════════════════════════════════════════════
# PUBLICATION-READY AUTOMATION
# ════════════════════════════════════════════════════════════════════════════

SUBJECTS := subj01 subj02 subj03
MODEL_ID := stabilityai/stable-diffusion-2-1
CACHE_DIR := outputs/clip_cache
RECON_DIR := outputs/recon
REPORTS_DIR := outputs/reports
FIGURES_DIR := $(REPORTS_DIR)/figures
TARGET_CACHE := $(CACHE_DIR)/target_clip_$(shell echo $(MODEL_ID) | sed 's/\//_/g').parquet

.PHONY: pipeline build_target_cache reconstruct_all eval_all_subjects summarize_reports generate_figures

# Complete publication pipeline
pipeline: build_target_cache reconstruct_all eval_all_subjects summarize_reports generate_figures
	@echo "════════════════════════════════════════════════════════════════"
	@echo "✅ PUBLICATION PIPELINE COMPLETE!"
	@echo "════════════════════════════════════════════════════════════════"
	@echo "📊 Reports:      $(REPORTS_DIR)/summary_*.csv"
	@echo "📈 Figures:      $(FIGURES_DIR)/"
	@echo "🖼️  Reconstructions: $(RECON_DIR)/"
	@echo "════════════════════════════════════════════════════════════════"

# Build 1024-D target CLIP cache for SD 2.1
build_target_cache:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "Building 1024-D CLIP cache for $(MODEL_ID)..."
	@echo "════════════════════════════════════════════════════════════════"
	@mkdir -p $(CACHE_DIR)
	$(PY) scripts/nsd_build_clip_cache.py \
		--model-id $(MODEL_ID) \
		--output-dir $(CACHE_DIR) \
		--device cuda \
		--batch-size 32

# Reconstruct test sets for all subjects
reconstruct_all: $(foreach subj,$(SUBJECTS),reconstruct_$(subj))

reconstruct_%:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "Reconstructing test set for $*..."
	@echo "════════════════════════════════════════════════════════════════"
	@mkdir -p $(RECON_DIR)/$*/ridge_diffusion/images
	$(PY) scripts/decode_diffusion.py \
		--subject $* \
		--model-id $(MODEL_ID) \
		--output-dir $(RECON_DIR)/$*/ridge_diffusion \
		--split test \
		--num-inference-steps 50 \
		--guidance-scale 7.5 \
		--device cuda \
		--batch-size 4

# Evaluate all subjects with all gallery types
eval_all_subjects: $(foreach subj,$(SUBJECTS),eval_full_$(subj))

eval_full_%:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "Evaluating reconstructions for $* (3 gallery types)..."
	@echo "════════════════════════════════════════════════════════════════"
	@mkdir -p $(REPORTS_DIR)/$*
	@# Gallery: matched
	$(PY) scripts/eval_reconstruction.py \
		--subject $* \
		--recon-dir $(RECON_DIR)/$*/ridge_diffusion/images \
		--clip-cache $(TARGET_CACHE) \
		--use-adapter \
		--model-id $(MODEL_ID) \
		--gallery matched \
		--image-source hdf5 \
		--out-csv $(REPORTS_DIR)/$*/eval_matched.csv \
		--out-json $(REPORTS_DIR)/$*/eval_matched.json \
		--out-fig $(REPORTS_DIR)/$*/eval_matched_grid.png
	@# Gallery: test
	$(PY) scripts/eval_reconstruction.py \
		--subject $* \
		--recon-dir $(RECON_DIR)/$*/ridge_diffusion/images \
		--clip-cache $(TARGET_CACHE) \
		--use-adapter \
		--model-id $(MODEL_ID) \
		--gallery test \
		--image-source hdf5 \
		--out-csv $(REPORTS_DIR)/$*/eval_test.csv \
		--out-json $(REPORTS_DIR)/$*/eval_test.json \
		--out-fig $(REPORTS_DIR)/$*/eval_test_grid.png
	@# Gallery: all
	$(PY) scripts/eval_reconstruction.py \
		--subject $* \
		--recon-dir $(RECON_DIR)/$*/ridge_diffusion/images \
		--clip-cache $(TARGET_CACHE) \
		--use-adapter \
		--model-id $(MODEL_ID) \
		--gallery all \
		--image-source hdf5 \
		--out-csv $(REPORTS_DIR)/$*/eval_all.csv \
		--out-json $(REPORTS_DIR)/$*/eval_all.json \
		--out-fig $(REPORTS_DIR)/$*/eval_all_grid.png

# Summarize all evaluation reports
summarize_reports:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "Summarizing evaluation reports..."
	@echo "════════════════════════════════════════════════════════════════"
	$(PY) scripts/summarize_reports.py \
		--reports-dir $(REPORTS_DIR) \
		--output-csv $(REPORTS_DIR)/summary_by_subject.csv \
		--output-md $(REPORTS_DIR)/SUMMARY.md

# Generate publication figures
generate_figures:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "Generating publication figures..."
	@echo "════════════════════════════════════════════════════════════════"
	@mkdir -p $(FIGURES_DIR)
	$(PY) scripts/plot_metrics.py \
		--reports-dir $(REPORTS_DIR) \
		--output-dir $(FIGURES_DIR) \
		--subjects $(SUBJECTS)

# ════════════════════════════════════════════════════════════════════════════
# PAPER-GRADE SHARED1000 EVALUATION
# ════════════════════════════════════════════════════════════════════════════

SHARED1000_OUT := outputs/eval_shared1000
STRATEGIES := single best_of_8 boi_lite
REP_MODE := avg
SEEDS := 0 1 2

.PHONY: eval-shared1000 summarize-shared1000 test-pipeline

# Run comprehensive Shared1000 evaluation for one subject
eval-shared1000:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "Running Paper-Grade Shared1000 Evaluation"
	@echo "Subject:    $${SUBJECT:-subj01}"
	@echo "Strategies: $(STRATEGIES)"
	@echo "Rep Mode:   $(REP_MODE)"
	@echo "Seeds:      $(SEEDS)"
	@echo "════════════════════════════════════════════════════════════════"
	@mkdir -p $(SHARED1000_OUT)/$${SUBJECT:-subj01}
	$(PY) scripts/eval_shared1000_full.py \
		--subject $${SUBJECT:-subj01} \
		--encoder-checkpoint $${ENCODER_CKPT:-checkpoints/mlp/$${SUBJECT:-subj01}/mlp.pt} \
		--encoder-type $${ENCODER_TYPE:-mlp} \
		--output-dir $(SHARED1000_OUT)/$${SUBJECT:-subj01} \
		--rep-mode $(REP_MODE) \
		--strategies $(STRATEGIES) \
		--seeds $(SEEDS) \
		--clip-cache $${CLIP_CACHE:-outputs/clip_cache/clip.parquet} \
		$${USE_CEILING:+--use-noise-ceiling} \
		$${ENCODING_CKPT:+--encoding-model-checkpoint $$ENCODING_CKPT} \
		--device $${DEVICE:-cuda}

# Aggregate results across subjects and strategies
summarize-shared1000:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "Summarizing Shared1000 Results"
	@echo "Subjects:   $${SUBJECTS:-subj01 subj02 subj03}"
	@echo "Strategies: $(STRATEGIES)"
	@echo "════════════════════════════════════════════════════════════════"
	$(PY) scripts/summarize_shared1000.py \
		--eval-dir $(SHARED1000_OUT) \
		--output-dir $(SHARED1000_OUT) \
		--subjects $${SUBJECTS:-subj01 subj02 subj03} \
		--strategies $(STRATEGIES) \
		--rep-mode $(REP_MODE)
	@echo "✅ Summary complete: $(SHARED1000_OUT)/SUMMARY.*"

# Quick smoke test of evaluation pipeline
test-pipeline:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "Running Evaluation Pipeline Smoke Test"
	@echo "════════════════════════════════════════════════════════════════"
	@mkdir -p outputs/eval_test
	$(PY) scripts/eval_shared1000_full.py \
		--subject subj01 \
		--encoder-checkpoint checkpoints/mlp/subj01/mlp.pt \
		--encoder-type mlp \
		--output-dir outputs/eval_test \
		--smoke
	@echo "✅ Smoke test passed"
