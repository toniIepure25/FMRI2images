# NeuroBridge-OT: Training Objectives

## Total Loss

$$\mathcal{L}_{total} = \mathcal{L}_{clip} + \lambda_{reg} \mathcal{L}_{reg} + \lambda_{token} \mathcal{L}_{token} + \lambda_{teacher} \mathcal{L}_{teacher} + \lambda_{ot} \mathcal{L}_{ot} + \lambda_{shared} \mathcal{L}_{shared} + \lambda_{adv} \mathcal{L}_{adv} + \lambda_{vmf} \mathcal{L}_{vmf} + \lambda_{cal} \mathcal{L}_{cal}$$

All components are optional and controlled by YAML config weights.

## Component Details

### 1. CLIP Contrastive (InfoNCE)

$$\mathcal{L}_{clip} = -\frac{1}{2B} \sum_{i=1}^{B} \left[ \log \frac{e^{s \cdot sim(q_i, k_i)}}{\sum_j e^{s \cdot sim(q_i, k_j)}} + \log \frac{e^{s \cdot sim(k_i, q_i)}}{\sum_j e^{s \cdot sim(k_i, q_j)}} \right]$$

- `s = exp(logit_scale)` is a learnable temperature parameter.
- Symmetric loss computes both query→key and key→query.
- Config: `contrastive.temperature`, `contrastive.symmetric`.

### 2. Embedding Regression

$$\mathcal{L}_{reg} = 1 - \frac{1}{B} \sum_{i=1}^{B} cos(\hat{y}_i, y_i)$$

Simple cosine distance between predicted and target CLIP embeddings.

### 3. Token Regression (optional)

$$\mathcal{L}_{token} = \frac{1}{B \cdot T \cdot D} \sum_{i,t,d} (\hat{z}_{i,t,d} - z_{i,t,d})^2$$

MSE loss on 257×768 token-level predictions. Only active when token cache exists.

### 4. Teacher Distillation

$$\mathcal{L}_{teacher} = 1 - \frac{1}{B} \sum_{i} cos(\hat{y}_i, y_i^{teacher})$$

Cosine distance to teacher predictions (V61a/V62a/V66a). Only active when teacher artifacts are available and nsdId-aligned.

### 5. Optimal Transport Alignment

$$\mathcal{L}_{ot} = \langle T^*, C \rangle = \sum_{m,k} T^*_{m,k} \cdot C_{m,k}$$

The Sinkhorn-optimal transport cost. Encourages meaningful alignment between subject ROI tokens and canonical tokens.

### 6. Shared-Image Consistency

$$\mathcal{L}_{shared} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \max(0, \delta - cos(e_i, e_j))$$

Hinge loss encouraging embeddings of the same image across different subjects to be similar. $\mathcal{P}$ = pairs with same nsdId, different subject.

### 7. Subject Adversarial

$$\mathcal{L}_{adv} = -\sum_{i} \log p(s_i | GRL(e_i))$$

Cross-entropy loss for subject classification on gradient-reversed features. The gradient reversal layer (Ganin et al., 2016) ensures the main encoder learns subject-invariant representations.

### 8. vMF Uncertainty (NLL)

$$\mathcal{L}_{vmf} = -\frac{1}{B} \sum_{i} \kappa_i \cdot cos(\mu_i, y_i) + \lambda_{reg} \cdot (\kappa_i - \kappa_{target})^2$$

Negative log-likelihood under von Mises-Fisher distribution (Bessel constant cancels). Includes kappa regularization to prevent collapse/explosion.

### 9. Calibration

$$\mathcal{L}_{cal} = BCE(confidence_i, \mathbb{1}[cos(\hat{y}_i, y_i) > 0.5])$$

Binary cross-entropy training the confidence head to predict retrieval quality.

## Default Weights

| Component | Weight | Rationale |
|---|---|---|
| Contrastive | 1.0 | Primary objective |
| Regression | 0.5 | Direct embedding alignment |
| Token | 0.0 | Only when cache available |
| Teacher | 0.3 | Knowledge transfer |
| OT | 0.1 | Alignment regularization |
| Shared | 0.1 | Cross-subject consistency |
| Adversarial | 0.1 | Subject invariance |
| vMF | 0.3 | Uncertainty calibration |
| Calibration | 0.0 | Optional post-training |

## Enabling/Disabling Components

Set any weight to 0.0 in the YAML config to disable that loss component. The loss module skips computation when weight is zero.
