# NeuroBridge-OT: Overview

## Motivation

Current fMRI-to-CLIP visual decoding achieves strong subject-specific retrieval (77.2% R@1 on SHARED1000 for subj01 via triple-fusion). However, this performance relies on subject-specific models (V61a, V62a) trained on ~30,000 trials per subject and does not transfer to unseen subjects.

The generalization workstream showed:
- True zero-shot cross-subject transfer is weak (~5-15% R@1).
- Multi-subject joint training on seen subjects (V66a) reaches ~45-55% R@1.
- Few-shot adaptation with 100-500 samples shows promise.

**NeuroBridge-OT bridges this gap** by learning a shared cortical-semantic representation via optimal-transport alignment while supporting subject-specific adaptation through hyper-networks.

## Core Idea

Instead of forcing all subjects through a single fixed architecture, NeuroBridge-OT:

1. **Tokenizes** each subject's variable-anatomy fMRI into a fixed set of ROI tokens.
2. **Aligns** these tokens to a shared canonical cortical space via differentiable optimal transport.
3. **Processes** aligned tokens through a shared semantic transformer.
4. **Decodes** into CLIP embeddings with uncertainty estimation.
5. **Adapts** to new subjects via hyper-generated lightweight adapters from subject fingerprints.

## Relation to Existing Models

| Model | Architecture | Subject Support | Limitation |
|---|---|---|---|
| V61a | Token-target MLP | subj01 only | No transfer capability |
| V62a | CLS MLP | subj01 only | No transfer capability |
| V66a | Multi-subject ROI Transformer | 4 subjects (subj01/02/05/07) | Weak on unseen subjects |
| Triple Fusion | Score-level fusion | subj01 only | Requires all 3 models |
| **NeuroBridge-OT** | OT-aligned ROI Transformer | All 8 NSD subjects | Novel; experiments pending |

## What This Architecture Does NOT Claim

- It does not claim to "solve" subject-independent brain decoding.
- It does not claim zero-shot performance matches subject-specific models.
- All claims must be backed by experimental evidence from the defined protocols.
- The architecture supports and explicitly evaluates 7 distinct generalization regimes.
