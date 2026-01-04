# Evaluation protocol (draft)

This repo’s reconstruction evaluation is designed to be **research-grade**:

- manifest-first mapping (when present)
- fail-fast on unmatched reconstructions
- strict embedding-dimension checks
- diagnostic outputs on mapping failures

## Inputs

- Reconstruction directory containing generated images.
- NSD index parquet for the subject.
- CLIP cache parquet (+ metadata sidecar if present).

## Stimulus identity rules

Reconstruction filenames are parsed into a stimulus key (e.g., NSD id / trial id). The evaluator resolves this key against the cache/index.

**Hard requirement:** every evaluated reconstruction must map to exactly one GT stimulus.

## Metrics

- CLIPScore: cosine similarity between recon image embedding and GT image embedding.
- Retrieval@K: rank of GT within a gallery of embeddings.
- Rank statistics: mean/median rank, MRR.

Interpretation note:

If the gallery is large (e.g., 10k images), it’s normal for Retrieval@10 to be 0.0 unless the model is extremely strong.
