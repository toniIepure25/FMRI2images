PY=uv run
export AWS_REGION=us-east-2

.PHONY: setup nsd-manifest sanity

setup:
\tuv sync

nsd-manifest:
\t$(PY) python scripts/nsd_build_manifest.py

sanity:
\t$(PY) python scripts/nsd_sanity_check.py --limit 3
