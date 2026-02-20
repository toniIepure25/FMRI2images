# System Configurations

Infrastructure and external service settings.

## Files

| Config | Purpose |
|--------|---------|
| `clip.yaml` | CLIP model selection and caching (Phase 2: ViT-L/14, 768-D) |
| `data.yaml` | NSD dataset paths, S3 access, preprocessing defaults |
| `logging.yaml` | Log format and verbosity |

These are referenced by `base.yaml` and generally do not need modification.
Override specific fields in experiment configs when needed.
