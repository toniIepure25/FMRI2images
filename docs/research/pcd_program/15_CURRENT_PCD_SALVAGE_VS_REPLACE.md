# 15 — Current PCD: Salvage vs Replace

**Date:** 2026-07-16 · Each Gate 0 finding treated as a **design constraint**, not a bug ticket.

---

## 1. Finding-by-finding diagnosis

### T6 — Prediction flows low→high, not high→low
- **Root cause:** conceptual. Someone wrote the residual in the feed-forward direction and
  labelled it Rao–Ballard. Never tested.
- **Local or structural:** **structural for the claim, trivial for the code.** Reversing the
  edge is ~10 lines. But reversing it buys nothing: Hi-DREAM owns the hierarchy and ESANN
  2025 owns "PC dynamics improve predictivity" (`14` §1).
- **Reuse:** the *mechanism* (encode the component of one ROI group not predictable from
  another) is fine and reusable. The *label* is not.
- **Invalidates a direction?** Yes — Candidate A as a mechanistic PC claim.
- **Successor:** keep cross-level residuals; drop the direction claim; make direction an
  arm only if it is ever cheap.

### T8/T13 — Per-level kappa heads never trained
- **Root cause:** `PCDModel.forward` returns only `(mu, kappa)`; `level_kappas` goes to a
  dict no loss reads. **The deeper cause is architectural: the model exposes an interpretable
  quantity through a channel the objective cannot see.**
- **Local or structural:** the *bug* is local (one loss term). The *pathology is structural
  and it is the most valuable thing this project has found* — an interpreted quantity with no
  identifying objective is not merely untrained, it is **unfalsifiable**. Because it projects
  trained features, it yields stable, plausible, category-varying figures.
- **Reuse:** heads reusable; the pattern must be banned.
- **Successor constraint (load-bearing):** **every interpreted quantity must have an
  identifying objective, enforced by a gradient test.** This generalises beyond PCD and is
  the seed of the selected thesis (`14` §3).

### F-001 — 82 pp overfit gap, four regularization rounds failed
- **Root cause:** hypothesised — capacity is in the wrong place (per-subject ROI projections
  ≈57% of params, UNVERIFIED). More likely, per `14` §3: **nothing constrains intermediate
  representations to be neurally meaningful**, so the model is free to memorise
  perception-specific voxel patterns.
- **Local or structural:** **structural.** Regularization is the wrong instrument; four
  rounds proved it. This is the same pathology NSD-Imagery documented independently.
- **Successor:** replace regularization-by-noise with **constraint-by-objective**
  (masked-ROI neural prediction). Shrink capacity via low-rank adapters as hygiene, not as
  the hypothesis.

### T7 — `nsdgeneral_other` ≈64% of voxels, bypasses the hierarchy
- **Root cause:** design shortcut — a residual dump given an unmediated path to the aggregator.
- **Local or structural:** structural. It makes every hierarchy statement unfalsifiable:
  the hierarchy can be decorative while the model still scores.
- **Successor:** **no unrestricted bypass** (per mission Workstream 5). Options in
  `18_NEUROPC_FORMAL_SPECIFICATION.md` §4.

### T11/T14 — Metric fork (16.4% vs 3.3%) and selection on `val_loss`
- **Root cause:** evaluation and selection evolved separately from reporting.
- **Local or structural:** local, but **poisons every comparison** until fixed.
- **Successor:** protocol standardisation is a **precondition**, not a task
  (`19_CONFIRMATORY_EXPERIMENT_PLAN.md` §1).

### T10 — Splits clean, SHARED1000 sealed
- **Not a failure — the project's most valuable asset.** Verified against NSD's own stimulus
  table. **Reuse wholesale.** Do not rebuild the split pipeline.

---

## 2. Component-level verdict

| Component | Verdict | Rationale |
|---|---|---|
| Split pipeline + `split.json` | **KEEP wholesale** | verified clean (T10); the asset |
| Per-ROI tokenisation (17 ROIs incl. FFA1/FFA2/PPA/EBA/OFA/OPA/RSC separately) | **KEEP** | **the data layer already preserves per-ROI granularity** — only the *level grouping* pools them. Mission Workstream 5's "preserve individual category-selective ROIs" costs a grouping change, not a rewrite |
| `ROIProjection` (no bottleneck) | **KEEP, constrain** | replace with shared + low-rank residual. **Never re-enable `bottleneck_dim`** (hung v3 at ep2, F-001) |
| `LevelTransformerEncoder` | **KEEP** | generic small transformer, unobjectionable |
| `PredictionHead` / cross-level residual | **KEEP mechanism, DROP label** | reusable; call it a cross-level residual (D-002) |
| `HierarchicalAggregator` | **REPLACE** | enables the level-3 bypass (T7); `n_levels`/`dropout` args are dead code |
| `PerLevelKappaHeads` | **DELETE** | never trained (T13); no objective; reintroduce only with a proper scoring rule |
| Global vMF head (`mu`, `kappa`) | **KEEP `mu`, NEUTER `kappa`** | kappa degenerate, CV≈5% (T9). Retrieval head kept for comparability; drop `kappa_reg` |
| 4-level fixed grouping | **REPLACE** | Hi-DREAM owns it (`14` §1); pooling FFA/PPA/EBA/OFA/OPA/RSC blocks per-ROI analysis |
| `ablation_mode` {full,no_errors,reversed,random} | **KEEP as diagnostic, DEMOTE from contribution** | Hi-DREAM already published this control. Also: `random.Random(42)` is a single fixed shuffle — needs a seed distribution for any use at all |
| `scripts/analysis/pcd_neuroscience_analysis.py` | **DELETE/REWRITE** | reads untrained kappas; treats magnitude as information |
| `train_unified.py`, queue, EMA, losses | **KEEP** | infrastructure works; the run reached epoch 113 |
| Checkpoints (ep98/ep112) | **KEEP as baseline arm** | the "current PCD" baseline in `19` §3; hashed in `05` |

## 3. Verdict: salvage the substrate, replace the thesis

**This is not a rewrite.** Roughly 70% of the code survives — splits, tokenisation, encoders,
training loop, retrieval head, evaluation. What is replaced is small and specific: the
aggregator (kills the bypass), the subject projections (low-rank), the level grouping
(per-ROI nodes), and the kappa heads (deleted). What is *added* is one thing: a **masked-ROI
neural-prediction objective**.

**What is actually being replaced is the research question**, and that was never in the code.
