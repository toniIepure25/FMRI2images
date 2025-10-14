PY=python

.PHONY: setup index test demo sanity read-index check-index clean build-clip-cache

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

# Check index header bounds
check-index:
	$(PY) scripts/check_index_headers.py

train-smoke:
	$(PY) scripts/train_smoke.py --index-root $${INDEX:-data/indices/nsd_index} $(ARGS)

test-preproc:
	$(PY) -m pytest src/fmri2img/scripts/test_preprocess.py -v

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
