# Method (draft)

This project reconstructs images from fMRI by predicting an embedding (CLIP or diffusion-conditioning) from brain activity and decoding it into an image.

## Pipeline overview

1. **Data**: Natural Scenes Dataset (NSD) subject-specific trials.
2. **Preprocessing** (optional): scaling, reliability filtering, PCA.
3. **Encoding / mapping**: fMRI → CLIP embedding (ridge/MLP/two-stage).
4. **Decoding**: embedding → image via diffusion.
5. **Evaluation**: CLIP-space similarity + retrieval/ranking metrics with strict stimulus ID matching.

## Models

- Ridge baseline: linear fMRI → CLIP.
- MLP: nonlinear fMRI → CLIP.
- Two-stage: supports probabilistic outputs (KL, NLL variants) depending on training mode.
- CLIP adapter: maps 768-D CLIP space into diffusion target space (e.g., 768/1024-D).

## Objectives

- Deterministic models: regression loss in embedding space.
- Probabilistic models: may combine reconstruction loss with KL (and optional NLL variants).

See code for exact losses and toggles in training scripts and configs.

## References

See `docs/paper/refs.md` and `REFERENCES.bib`.
