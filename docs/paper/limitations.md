# Limitations (draft)

## Dataset / scope

- Results are NSD-subject-specific and may not generalize.
- The retrieval gallery size heavily affects Retrieval@K.

## Evaluation limitations

- CLIPScore correlates imperfectly with human judgments.
- Some failure modes (mode collapse, blurry recon) are not captured by embedding metrics.

## Engineering limitations

- Runtime and VRAM requirements can be high for diffusion decoding.
- Full reproducibility depends on stable external model weights availability.
