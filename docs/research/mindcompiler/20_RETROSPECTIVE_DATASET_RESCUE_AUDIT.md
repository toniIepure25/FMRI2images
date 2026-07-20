# 20 — Retrospective Dataset Rescue Audit (Gate M0.2)

**Date:** 2026-07-17 · **No downloads performed.** Metadata/publication inspection only.

---

## 0. THE CORRECTION — my previous absolute statement was WRONG

**I wrote:** *"No public dataset has ≥3 mental states on the same content at trial level."*

**That was false as an existence claim.** **Oedekoven et al. (2017)** measured **21 participants
watching, immediately retrieving, and retrieving after one week, the same 24 short videos.**
That is **three states on identical content**. It exists, it is published, and it is verified.

The true statement is narrower and entirely different in its consequences:

> **Three-state data exist and are verified. They are not *openly downloadable* at trial level.
> The published NeuroVault release (collection 2814) contains group-level t-maps only.**

**"No dataset exists" and "the data exist but require an author request" are not the same
finding, and I conflated them.** The first terminates a retrospective program; the second
converts it into an access task. This materially changes the Gate M0.1 verdict.

## 1. Precise access classification (replaces all absolute statements)

| Class | Meaning | Datasets |
|---|---|---|
| **1 — Fully public, directly downloadable** | usable today | **Li, Yang & Bao 2026** (Dryad, 7.37 GB); **ds001132** (OpenNeuro); NSD-Imagery; NSD-Synthetic |
| **2 — Public processed only** | insufficient granularity | **Oedekoven NeuroVault 2814 — group-level t-maps ONLY** |
| **3 — Available on reasonable request** | access task, not a data gap | **Oedekoven 2017 trial/event-level** (paper states data available from corresponding author) |
| **4 — Exists, access unconfirmed** | audit incomplete | **Li et al. 2023** (imagery/perception/illusion); **Roy Dataset 2** (512 conditions, 3 subjects) |
| **5 — Prospective acquisition required** | does not exist | **MindStates-7T** |

## 2. Dataset A — Oedekoven et al. 2017 · **THREE STATES** · class 2/3

*Reinstatement of memory representations for lifelike events over the course of a week.*
Sci Rep 7, doi:10.1038/s41598-017-13938-4 · PMC5662713 · NeuroVault collection 2814.

| Property | Verified |
|---|---|
| Participants | **21** |
| Content | **24 short videos** — unique event identities |
| **States** | **encoding (watch) → immediate retrieval → delayed retrieval (1 week)** |
| Same content across states | **YES — all three phases on the same 24 videos** |
| Behavioural | free recall outside scanner after delayed retrieval; memory performance scores |
| Public artifact | **NeuroVault 2814 — group-level t-images only. NOT trial-level.** |
| Trial-level access | **Class 3 — on reasonable request** (draft in `21`) |

**Reported findings relevant to us:** widespread within- and between-subject reinstatement in a
posterior-midline core retrieval network across all phases; in precuneus, bilateral MTG and
left hippocampus, **reinstatement between retrieval phases correlated with memory
performance**.

> **This is the single highest-value target in the program.** It is the only verified dataset
> supporting **E-M3** (encoding → immediate → delayed), i.e. multi-stage predictive
> factorization on shared content. 21 subjects is a serious n by this field's standards.
> **Caveat: 24 videos is a small content set for held-out-content evaluation** — the same
> structural limit as NSD-Imagery's 12, though with far more subjects.

## 3. Dataset B — Li, Yang & Bao 2026 · **PUBLIC NOW** · class 1

*Spatial reorganization of object representations in high-level visual cortex distinguishes
working memory from perception.* **Science Advances**, 8 May 2026,
doi:10.1126/sciadv.aea7764 · preprint bioRxiv 2025.06.29.662186 · Dryad **7.37 GB**
(`wm.zip`, `wm_avged.zip`, `control.zip`).

**States:** perception + visual working memory (1-item and 2-item). **Not** imagery, **not**
episodic recall. Matched bilateral presentation; fMRI decoding; LOC focus.

### The finding that forces a theory revision

> *"Perception kept object information largely **contralateral**, whereas VWM produced robust
> **ipsilateral** representation even when memorizing bilateral items. […] VWM engages
> **70–90% of ipsilateral LOC territories**, far exceeding those recruited during unilateral
> perception."*

**This is spatial EXPANSION and REDISTRIBUTION — not contraction.** Roy et al. report
early-visual *contraction* for perception→imagery. Li et al. report *ipsilateral expansion*
for perception→VWM.

> **Different internally-generated states reorganize information in opposite geometric
> directions.** A pure "information contraction" thesis is **refuted by data already
> published**. The revised theory (`§5`) is not a softening — it is what the evidence says.

**Practical:** do **not** download 7.37 GB blindly. `wm_avged.zip` (averaged) is likely the
minimum proof-of-concept subset; whether **trial-level** responses exist in `wm.zip` is
**UNVERIFIED** and is the first thing to check via the Dryad file manifest.

## 4. Dataset C — Li et al. 2023 · class 4

*Neural Representations in Visual and Parietal Cortex Differentiate between Imagined,
Perceived, and Illusory Experiences.* **Perception + imagery + illusion** on matched
orientation content — a **third condition** of a genuinely different kind.

**Not audited in full this session.** Participant counts, trial counts, pairing, eye-tracking
and data availability all **UNVERIFIED**.

> **Illusion is not working memory and not episodic recall.** It probes the objective/subjective
> internality distinction. It must not be substituted for a memory state to manufacture a
> "three-state" claim.

## 5. Dataset D — ds001132 (Chen et al., naturalistic encoding–recall) · class 1

Movie viewing + **spoken** recall; public on OpenNeuro; transcripts; event segmentation.

**Principal confound, and it is disqualifying for the primary endpoint:** recall is **produced
by speech**. Motor, articulatory and language-production activity is entangled with the
retrieval representation. Usable as **external naturalistic validation** of event-level
retention orderings; **not** usable for clean state-transition geometry.

## 6. Usable state pairs and sequences — the bottom line

**Two-state, downloadable today:**
- perception → **working memory** (Li/Yang/Bao, class 1) ← **E-M2 runs on this**
- perception → **imagery** (NSD-Imagery, class 1) ← **E-M1 runs on this**

**Three-state sequence, verified, access required:**
- **encoding → immediate retrieval → delayed retrieval** (Oedekoven, class 3) ← **E-M3**

**Three-state, existence verified, access unconfirmed:**
- perception → imagery → illusion (Li et al. 2023, class 4)

**Not available anywhere:** perception → WM → imagery → recall factorially on shared content
with controlled delay and vividness. **That remains Track P / MindStates-7T.**
