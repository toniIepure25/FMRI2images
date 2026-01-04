# Paper track (draft)

This folder is the **paper-style, research-facing documentation** for this repository.
It’s written to support:

- a thesis write-up or paper submission,
- artifact/reproducibility expectations (manifests, exact configs, seeds),
- clearly testable claims.

If you’re looking for *how to run the code*, start at `docs/README.md` (engineering docs).

## Structure

- `method.md` — modeling choices and training objective
- `evaluation_protocol.md` — evaluation setup and metrics definitions
- `experiments.md` — experiment catalog (links to per-run cards)
- `results.md` — tables/plots produced by `scripts/build_paper_artifacts.py`
- `limitations.md` — known limitations and failure modes
- `reproducibility.md` — how to reproduce runs and verify artifacts
- `claims.md` — explicit claims + the evidence path in the repo

## Conventions

- Citations: we keep a lightweight bib in `REFERENCES.bib` and a human-readable map in `docs/refs.md`.
- Repository citation metadata: `CITATION.cff`.
- Runs: every experiment should emit a `manifest.json` using `fmri2img.utils.manifest.write_manifest()`.

## What counts as “reproducible” here

A run is considered reproducible if:

1. It records git commit + dirty state.
2. It records the effective config (post-merge).
3. It records key input hashes (index, checkpoint, cache).
4. It stores enough identifiers to map reconstructions ⟷ stimuli deterministically (see evaluator).
