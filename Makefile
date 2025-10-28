PY=python

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
	$(PY) -m fmri2img.data.nsd_index_builder --subjects $${SUBJECTS:-subj01} --max-trials $${MAX_TRIALS:-} --output-format parquet --use-s3

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
