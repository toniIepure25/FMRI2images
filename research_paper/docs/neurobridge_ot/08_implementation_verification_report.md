# NeuroBridge-OT: 8-Subject Implementation Verification Report

## Summary

The NeuroBridge-OT architecture is designed for **all 8 NSD subjects**. This document
audits data availability, readiness, and blockers for the primary 8-subject protocol.

**Critical note**: Pre-extracted fMRI features (`.npy`) live on the H100 pod at
`/home/jovyan/work/FMRI2images_cache/preextracted/` or equivalent. They are
gitignored and must be regenerated after pod restarts via `make preextract`.

## Per-Subject Audit Table

| Field | subj01 | subj02 | subj03 | subj04 | subj05 | subj06 | subj07 | subj08 |
|---|---|---|---|---|---|---|---|---|
| **fMRI features path** | `cache/preextracted/subject=subj01/fmri_features.npy` | (same pattern) | ... | ... | ... | ... | ... | ... |
| **Pre-extracted exists (pod)** | YES | YES | PENDING | PENDING | YES | PENDING | YES | PENDING |
| **Feature shape (expected)** | (30000, ~15724) | (30000, ~15xxx) | (30000, ~15xxx) | (30000, ~15xxx) | (30000, ~15xxx) | (30000, ~15xxx) | (30000, ~15xxx) | (30000, ~15xxx) |
| **nsdgeneral voxels** | ~15,724 | ~14,500–16,500 | varies | varies | varies | varies | varies | varies |
| **ROI index availability** | YES (NSD atlas) | YES | YES | YES | YES | YES | YES | YES |
| **n_rois (NeuroBridge-OT)** | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 |
| **index.parquet** | EXISTS (local) | GENERATE | GENERATE | GENERATE | GENERATE | GENERATE | GENERATE | GENERATE |
| **Trials (expected)** | 30,000 | 30,000 | 30,000 | 30,000 | 30,000 | 30,000 | 30,000 | 30,000 |
| **Sessions** | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 |
| **SHARED1000 availability** | YES | YES | YES | YES | YES | YES | YES | YES |
| **Expected SHARED1000 count** | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| **SHARED1000 reps per image** | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| **SHARED1000 trials** | 3000 | 3000 | 3000 | 3000 | 3000 | 3000 | 3000 | 3000 |
| **Missing nsdIds** | AUDIT | AUDIT | AUDIT | AUDIT | AUDIT | AUDIT | AUDIT | AUDIT |
| **CLIP target coverage** | 9,841+ (from clip.parquet) | Same gallery | Same | Same | Same | Same | Same | Same |
| **Token target coverage** | If token cache built | Same | Same | Same | Same | Same | Same | Same |
| **DINO/SigLIP/OpenCLIP** | NOT AVAILABLE | — | — | — | — | — | — | — |
| **Teacher: V61a** | subj01 ONLY | NO | NO | NO | NO | NO | NO | NO |
| **Teacher: V62a** | subj01 ONLY | NO | NO | NO | NO | NO | NO | NO |
| **Teacher: V66a** | subj01/02/05/07 | subj01/02/05/07 | NO | NO | subj01/02/05/07 | NO | subj01/02/05/07 | NO |
| **Single-subject training** | YES | YES | YES | YES | YES | YES | YES | YES |
| **Multi-subject training** | YES | YES | YES | YES | YES | YES | YES | YES |
| **LOSO target** | YES | YES | YES | YES | YES | YES | YES | YES |
| **Blockers** | None (data ready on pod) | Pre-extract needed | Pre-extract + index needed | Same | Pre-extract needed | Pre-extract + index needed | Pre-extract needed | Pre-extract + index needed |

## Data Preparation Commands (for all 8 subjects)

```bash
# On the H100 pod:
set -a && source .env && set +a

# Build index for all subjects
for subj in subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08; do
    make index SUBJECT=$subj
done

# Pre-extract fMRI features (ROI-masked)
for subj in subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08; do
    make preextract SUBJECT=$subj
done

# Build CLIP cache (shared across subjects)
make clip-cache
```

## SHARED1000 Gallery Audit

### Expected behavior

All 8 NSD subjects view the same 1,000 SHARED images. Each image is shown 3 times
per subject, producing 3,000 SHARED1000 trials per subject (out of 30,000 total).

For SHARED1000 evaluation:
- The 3 repetitions are averaged to produce 1 embedding per image per subject.
- Gallery size = 1,000 unique CLIP embeddings (same for all subjects).
- Gallery is identical across subjects (same 1,000 nsdIds).

### Why previous results report ~982 instead of 1000

In previous implementations, some subjects may show fewer than 1,000 valid
SHARED1000 results due to:
1. Missing beta data for certain sessions (incomplete scanning).
2. Flagged trials removed during quality control.
3. Missing CLIP embeddings for some nsdIds.
4. Index building filtering out problematic trials.

The actual count must be verified per-subject after `make index` and reported
in the run manifest. **Metrics are only directly comparable if gallery nsdIds match.**

### Verification command

```bash
python3 scripts/neurobridge_ot/audit_neurobridge_data.py \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --check-shared1000
```

## Teacher Artifact Coverage

| Teacher Model | Available Subjects | Missing Subjects | Notes |
|---|---|---|---|
| V61a (token-target MLP) | subj01 | subj02-08 | subj01-specific architecture |
| V62a (CLS MLP) | subj01 | subj02-08 | subj01-specific architecture |
| V66a (Multi-subj ROI Trans) | subj01, subj02, subj05, subj07 | subj03, subj04, subj06, subj08 | Trained on 4 subjects |
| Triple-fusion scores | subj01 | subj02-08 | subj01-only system |

**Implications for 8-subject experiments:**
- Teacher distillation is NOT uniformly available across all 8 subjects.
- Experiments comparing with/without teacher should either:
  (a) run teacher loss only for subjects with coverage; or
  (b) run teacher experiments separately as teacher-coverage-limited.
- Teacher availability MUST be logged in every run manifest.
- NEVER silently bias results by having teacher loss active for only some subjects.

## Readiness Summary

| Status | Count | Subjects |
|---|---|---|
| Fully ready on pod (4 legacy) | 4 | subj01, subj02, subj05, subj07 |
| Needs index + preextract | 4 | subj03, subj04, subj06, subj08 |
| All NSD raw data available | 8 | All (NSD provides all 8 subjects) |

**Critical action**: Before running any 8-subject experiment, execute:
```bash
python3 scripts/neurobridge_ot/audit_neurobridge_data.py \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --check-rois --check-teachers --check-shared1000
```

If any subject fails audit, the experiment MUST NOT proceed with that subject
silently excluded. Either fix the data or explicitly document the exclusion.
