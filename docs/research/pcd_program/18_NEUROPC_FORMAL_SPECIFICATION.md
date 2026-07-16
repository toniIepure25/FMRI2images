# 18 — Formal Specification: NCD (Neural-Constrained Decoder)

**Date:** 2026-07-16 · Supersedes the mission's "NeuroPC" naming (see `17` §1).

**Terminology (per reviewer O-2).** The model is a **Neural-Constrained Decoder**. We claim a
**neural-prediction constraint**, never "identifiability" — masked-ROI prediction makes
representations *predictive of held-out activity*; it does **not** make them identifiable in
the statistical sense (unique up to a known transformation). The stronger word is not earned
and is not used.

---

## 1. Notation

| Symbol | Meaning | Shape |
|---|---|---|
| `s ∈ {1..S}` | subject index | S = 8 |
| `R` | ROI nodes (per-ROI, **never pooled**) | R = 17 |
| `x^{(s)} ∈ ℝ^{V_s}` | flat fMRI, subject `s` | (B, V_s) |
| `y_r^{(s)}` | voxels of ROI `r` | (B, V_{s,r}) |
| `h_r ∈ ℝ^d` | ROI node embedding | (B, R, d) |
| `M ⊂ {1..R}` | masked ROI subset | \|M\| = ⌈ρR⌉ |
| `ŷ_r` | predicted activity of masked ROI `r` | (B, V_{s,r}) |
| `z ∈ ℝ^d` | fused cortical state | (B, d) |
| `μ ∈ S^{D−1}` | CLIP direction | (B, 768) |
| `c ∈ {perception, imagery}` | acquisition state | scalar |

## 2. Observation model with low-rank subject adaptation

Per-ROI projection, shared weights plus a **low-rank subject residual** — the capacity fix
from F-001, deliberately *not* the hypothesis:

```
h_r^{(s)} = LN( ( W_r + U_r^{(s)} (V_r^{(s)})ᵀ ) y_r^{(s)} + b_r )
W_r ∈ ℝ^{d×V_r},  U_r^{(s)} ∈ ℝ^{d×k},  V_r^{(s)} ∈ ℝ^{V_r×k},  k ≪ d   (k = 8)
```

`W_r` is shared across subjects. Only `U_r^{(s)}, V_r^{(s)}` are subject-specific.
Per-subject parameters drop from ~96M (PCD, ~57% of the model) to
`S · R · k · (d + V_r)` ≈ **~7M**, a ~14× reduction.

**Voxel-count mismatch across subjects** is handled by per-subject `V_r`, exactly as PCD's
existing per-subject buffers already do (`_roi_idx_{subj}_{i}`) — reuse that machinery.

## 3. Cortical state — no bypass

```
H   = [h_1 … h_R] + PosEmb + SubjEmb^{(s)}      (B, R, d)
H'  = TransformerEncoder(H)                     (B, R, d)   # lateral messages only
z   = Attn(q, H')                               (B, d)
```

**All R = 17 ROIs are nodes, including every category-selective ROI separately**
(FFA1, FFA2, PPA, EBA, OFA, OPA, RSC). No level pooling — this is what makes per-ROI
contrasts possible at all, and it is the concrete departure from PCD's 4-level grouping
(which Hi-DREAM owns anyway, `14` §1).

**No `nsdgeneral_other` bypass.** It enters as a **constrained context node** under a strict
budget: fixed random projection `P ∈ ℝ^{d×V_other}` (frozen, not learned) followed by a
single learned `d×d` map. Parameter cost `d²` instead of `d·10000`, a ~13× reduction, and it
cannot dominate by capacity. Alternatives (exclude entirely / functional parcellation) are
ablation arms in §7, not the default.

## 4. The neural-prediction constraint — the one independent variable

Sample a mask `M` per batch (ratio ρ = 0.3). Zero the masked ROI nodes **before** the
encoder, then predict their true activity from the remaining context:

```
H_masked = H with h_r ← [MASK] for r ∈ M
H'_masked = TransformerEncoder(H_masked)
ŷ_r = g_r( H'_masked[r] )        for r ∈ M           # g_r: d → V_{s,r}
L_neural = (1/|M|) Σ_{r∈M} || ŷ_r − y_r^{(s)} ||² / V_{s,r}
```

**Leakage-safety is structural:** `y_r` for `r ∈ M` never enters the forward pass that
predicts it. The target is held out **by construction, per batch**, not by a split.

**Objective:**
```
L = L_retrieval(μ, CLIP(image))  +  λ_neural · L_neural  +  λ_reg · L_reg
```

`λ_neural` is **the only manipulated variable.** Everything else is held fixed across arms.

**Gradient contract (from T8/F-002, non-negotiable):** every interpreted quantity must have
an identifying objective, enforced by a test. `ŷ_r` is interpreted → it is supervised by
`L_neural`. **No quantity is interpreted unless a gradient test proves the loss reaches it.**
Any head added later without an objective is a bug, not a feature.

## 5. Controls demanded by the panel (all load-bearing)

| Control | Purpose | Objection |
|---|---|---|
| **Matched-regularization arm** — `λ_neural = 0`, with dropout/noise **tuned to match the training loss and effective capacity** of the `λ_neural > 0` arm | separates "constraint" from "any regularizer" | **O-1** |
| **Stimulus-shuffled neural arm** — `L_neural` computed against ROI activity from a *different stimulus* (same subject/session) | if this works equally well, the objective is learning **functional connectivity**, not visual representation | **O-3** |
| **Strong external baseline** — report transfer for ≥1 strong published model, or scope the claim to our capacity regime explicitly | the effect may be specific to a weak (16.4%) base model | **O-5** |
| **Per-ROI mean baseline** for `L_neural` | proves the constraint predicts anything out-of-sample at all | kill criterion, `17` §6 |

## 6. Estimand (per O-6 — **not** a ratio)

Let `A_c^{(s)}` be image-level R@1 for subject `s` in condition `c ∈ {perception, imagery}`,
on a **fixed, protocol-standardised gallery**.

**Primary estimand — a paired difference on a variance-stabilised scale:**
```
Δ^{(s)} = [ φ(A_imagery^{(s)} | λ>0) − φ(A_imagery^{(s)} | λ=0,matched-reg) ]
          −  γ · [ φ(A_perception^{(s)} | λ>0) − φ(A_perception^{(s)} | λ=0,matched-reg) ]
where φ = arcsin(√·)
```
Estimate `E_s[Δ]` by hierarchical bootstrap over subjects **and** stimuli. The `γ` term
enforces *"at matched perception performance"* — an imagery gain purchased by a perception
loss is a capacity trade, not a mechanism (`17` §6). `γ` is fixed **before** unblinding.

Equivalent and preferred formulation: a hierarchical model with a
`condition × λ_neural` **interaction term**, subject as a random effect. **No ratios.**

## 7. Ablation mapping (each arm = one limiting case)

| Arm | Change | Tests |
|---|---|---|
| `lambda_neural: 0.0` | no constraint | main effect |
| `lambda_neural: 0.0 + matched reg` | O-1 control | constraint vs regularizer |
| `neural_target: shuffled` | O-3 control | representation vs connectivity |
| `mask_ratio: {0.15, 0.3, 0.5}` | dose | monotone response? |
| `subject_adapter: {full, lowrank_k8, shared_only}` | capacity location | F-001 hypothesis |
| `other_node: {context, excluded, unconstrained}` | T7 | does the bypass matter? |
| `roi_nodes: {per_roi, pcd_4level}` | grouping | our departure vs Hi-DREAM's |
| `PCD_v4 checkpoint` | as-is baseline | the incumbent |
| flat MLP, param-matched | floor | |
| linear ridge | **NSD-Imagery's own winner** | the baseline that beats complex models |

## 8. Budget (d = 512, R = 17, k = 8, S = 8)

| Component | Params |
|---|---|
| Shared ROI proj `W_r` (17 × 512 × ~340 avg) | ~3.0M |
| Low-rank subject adapters (8 × 17 × 8 × (512+340)) | ~0.9M |
| `other` context node (frozen P + d²) | ~0.26M |
| Transformer encoder (4 layers, d=512) | ~12.6M |
| Neural-prediction heads `g_r` (17 × 512 × ~340) | ~3.0M |
| Retrieval head (512 → 768) | ~0.8M |
| **Total** | **~20.5M** |

**vs PCD's 167M — an 8× reduction.** This is the point: NSD-Imagery found complex models
overfit; our own model overfits by 82 pp; the successor must be small. FLOPs: to be measured
by `scripts/utils/count_flops.py` (**not yet written** — required before any matched
comparison; `NOT_MEASURED` is not an acceptable registry entry twice).

## 9. Pseudocode

```python
def forward(x, subject_id, mask=None, return_neural=False):
    h = [roi_proj(r, gather(x, roi_idx[subject_id][r]), subject_id)   # low-rank adapted
         for r in range(R)]
    H = stack(h, dim=1) + pos_embed + subject_embed(subject_id)       # (B, R, d)
    if mask is not None:
        H = where(mask, mask_token, H)                                 # (B, R, d)
    Hp = encoder(H)                                                    # (B, R, d)
    z  = attn_pool(Hp)                                                 # (B, d)
    mu = normalize(retrieval_head(z))                                  # (B, 768)
    y_hat = {r: g[r](Hp[:, r]) for r in masked_rois} if return_neural else None
    return mu, y_hat
```

## 10. Deliberately excluded from v1

Recurrence (mission-permitted, but NSD cannot resolve it and it adds capacity to a model
whose disease is capacity); precision weighting (no objective → would recreate F-002);
per-level or any uncertainty head (kappa is degenerate, T9 — and re-adding an unsupervised
head would repeat the exact bug this program exists to have learned from); reconstruction;
diffusion. Each may return **only** after v1's single knob has been read.
