# Defense Presentation Script

**Duration:** ~9 min 30 sec (target: 10 min) — restructured: V61a architecture added, redundancies removed
**Presenter:** Iepure Antoniu
**Committee:** UBB FMI AI
**Date:** June 2026

---

## Slide 1 — Title [15 sec]

> Good afternoon. My name is Iepure Antoniu, and my thesis presents a system that identifies which image a person was viewing — using only their brain activity — with 86.3% accuracy on a held-out benchmark of one thousand images.
>
> Let me walk you through how.

**Delivery:** Confident, measured. Make eye contact. Let the 86.3% land.

---

## Slide 2 — The Question [30 sec]

> Imagine a person lying in an fMRI scanner, viewing natural photographs. The scanner records blood-oxygen-level-dependent signals from cortical voxels — thousands of small volumetric measurements of neural activity.

*[click — reveal gold text]*

> The question this thesis asks: from ONLY that brain signal — can we identify the EXACT image they were viewing?

*[click — reveal clarification]*

> Not the category. Not "an animal" or "a building." The specific photograph — out of a gallery of one thousand possibilities.

**Delivery:** Slow, cinematic. Pause after "which image they saw?" for 1–2 seconds.

---

## Slide 2b — Visual Intuition [20 sec]

> Look at this fruit bowl image. Right now, as you look at it, your visual cortex is processing it — V1 detects edges and orientations, V2 processes textures, V4 handles color and shape, and the inferotemporal cortex recognizes the objects.

> The fMRI scanner captures all this activity simultaneously from these regions — V1 through IT, plus category-selective areas like the Fusiform Face Area and Parahippocampal Place Area. That's what we decode from: 15,724 voxels spanning these visual areas.

**Delivery:** Point to the brain diagram regions as you name them. Engage the committee.

---

## Slide 3 — Motivation & Scope [45 sec]

> Why does this problem matter? Functional MRI captures rich visual representations — but at the level of blood-oxygen signals in individual cortical voxels. Bridging the gap between these signals and image identity has direct applications in assistive neurotechnology and cognitive neuroscience.

*[click]*

> Many prior approaches in this field emphasize category-level decoding — classifying whether the person saw "a face" or "a scene" — or they focus on reconstruction quality. This thesis takes a stricter approach: exact image retrieval from a held-out gallery.

*[click — reveal right panel]*

> What does the thesis achieve? A retrieval system that, given only brain activity, identifies the exact photograph from a gallery of one thousand — with 86.3% accuracy. The method: frozen score-level fusion of three complementary decoders. All reported experiments use a single subject.

*[click — bottom text]*

**Transition:** "Before describing the model, let me define the data and evaluation protocol precisely."

---

## Slide 4 — Dataset & Evaluation Protocol [50 sec]

> How was the dataset built? Allen and colleagues put subjects in a 7-Tesla fMRI scanner and showed them natural photographs – 10,000 unique images from COCO, each shown 3 times. That gives 30,000 total trials. On every single trial, the scanner records blood-oxygen activity from roughly 15,700 cortical voxels covering visual cortex.

*[click — example image appears]*

> *[point to the fruit bowl image]* Look at this image right now. Just as you, the committee members, are looking at this photograph – that's exactly what the subjects did. And in this moment, specific patterns are activating in your brain corresponding to YOUR internal representation of this scene – based on your visual experience. The fMRI scanner captures exactly those activation patterns. That's the raw signal we work with: 15,724 numbers per trial, one photo per trial, 30,000 pairs.

*[click — details]*

> Importantly, we don't use raw fMRI time series. NSD provides pre-computed single-trial beta coefficients — the output of a denoised general linear model. These capture the trial-specific neural response with noise already removed by the NSD preprocessing pipeline. We then z-score per session and mask to the nsdgeneral ROI.

> The protocol: we split by image identity so no photo leaks across train and test. 1000 images are held out completely as the SHARED1000 benchmark. Task: given a new brain pattern, identify which exact photo from 1000. Chance: 0.1%. Our system: 86.3%.

**Transition:** "With the benchmark defined, the key modeling choice is the target space."

---

## Slide 5 — Predict Meaning, Not Pixels [50 sec]

> The core insight of this work: do not try to reconstruct pixels. Instead, predict meaning.

> OpenAI's CLIP model maps images into a 768-dimensional unit hypersphere where cosine similarity directly encodes semantic similarity. CLIP provides a shared language between images, text descriptions, and — crucially — our predicted brain embeddings.

> Our decoder learns a mapping from approximately 15,700 brain voxels into this CLIP space. The function maps from the nsdgeneral ROI voxels to an L2-normalized 768-dimensional embedding.

*[click — teal text]*

> The retrieval logic follows directly: if the predicted embedding lands near the true CLIP embedding of the viewed image, then nearest-neighbour search identifies the correct image from the gallery.

> The diagram on the right illustrates the geometry: semantic clusters form naturally in CLIP space, and our decoded fMRI embedding lands near the ground truth.

**Transition:** "Let me show you the strongest expert in our system."

---

## Slide 6 — V61a: Dominant Expert Architecture [35 sec]

> Let me start with the strongest expert — V61a — which carries 70% of the fusion weight.

> Follow the diagram left to right: 15,724 nsdgeneral voxels enter a wide MLP encoder — 4 layers from 8192 down to 2048 — with residual skip connections and dropout. The vMF decoder head outputs a direction mu on the unit hypersphere plus a concentration kappa via softplus. The target is the full 197K-dimensional token space — all 257 spatial tokens from ViT-L/14, not just the CLS summary. At inference, MC-TTA-16 performs 16 stochastic passes and averages on the sphere: 79.1% becomes 83.1%.

*[click — Wide MLP card]*

> Why so wide? Brain signals are 15,000-dimensional but extremely noisy. Broad layers with skip connections capture distributed patterns.

*[click — Token target card]*

> Why tokens? 257 spatial patches preserve both semantic AND spatial information — a single CLS vector throws spatial layout away.

*[click — MC-TTA card]*

> MC-TTA leverages the kappa confidence: we run 16 passes with active dropout, average directions on the sphere, and gain 4 full percentage points. 83.1% — strongest single expert, weight 0.7 in fusion.

**Transition:** "Now let me show you the full system that combines three such experts."

---

## Slide 6b — System Architecture [40 sec]

> The architecture, reading left to right:

> Input: 7-Tesla fMRI from NSD Subject 01, masked to 15,724 nsdgeneral voxels, per-session z-scored.

> Three independently trained decoder experts — all using vMF-NCE — each mapping brain activity into CLIP space but with different target spaces:

> V61a — the Token Expert — 197K-dimensional token space, weight 0.7. V62a — 768-D CLS space, weight 0.1. V66a — 768-D ROI-pretrained, weight 0.2.

> Their CSLS-corrected scores are z-normalized and combined with fixed weights, frozen before test evaluation.

**Transition:** "But how do we TRAIN these experts on a normalized space? That requires a special loss."

---

## Slide 6c — Training Loss: vMF-NCE [25 sec]

> How do we train these experts? Since CLIP embeddings live on the unit hypersphere, we need a loss function that respects this geometry.

> *[gesture to sphere diagram]* The von Mises-Fisher distribution is the natural probability distribution for directions on a sphere. The model predicts mu — a direction — and kappa — how confident it is. High kappa means a tight cluster around the prediction. Low kappa means the model is uncertain — the probability mass spreads out.

*[click — three reasons]*

> Why vMF-NCE? It's cosine-native — matches CLIP's metric. It's contrastive — ranks against hard negatives. And kappa gives per-trial uncertainty — which enables MC-TTA.

*[click — impact]*

> The concrete payoff: kappa enables 16-draw Monte Carlo test-time augmentation on V61a, giving +4 percentage points. All three experts use vMF-NCE as their training loss.

**Transition:** "Now — the contribution: how we combine these experts."

---

## Slide 7 — Core Contribution: Frozen Score Fusion [65 sec]

> This is the main intellectual contribution.

> The formula: the final score for query i against gallery item j is 0.7 times V61a's z-normalized CSLS score, plus 0.1 times V62a's, plus 0.2 times V66a's. CSLS at k=3 corrects for hubness — penalizing vectors that appear as nearest neighbours to too many queries. Z-normalization ensures scale differences don't dominate.

> Three critical protocol points: weights selected on validation only, frozen before SHARED1000, no test-time tuning.

*[click — table and explanation]*

> The evidence: V61a alone reaches 83.1%. Triple fusion reaches 86.3%. That's +3.2 percentage points — at least 32 additional images correctly identified from the gallery.

> Why does this work? The experts target different embedding spaces and have different failure modes. The R@5 rise from 96.9% to 98.0% confirms complementary correction, not redundant ensembling. If experts were redundant, they'd reinforce the same mistakes.

> I say "empirically stronger" deliberately — I have not run a McNemar test, so I report the observed improvement without claiming formal statistical significance.

**Transition:** "What does this achieve on the benchmark?"

---

## Slide 8 — SHARED1000 Retrieval: 86.3% [50 sec]

> 86.3% Recall at 1.

*[Pause 2 seconds — let the number breathe]*

> For 863 out of 1,000 held-out test images, the system identifies the exact photograph from the full gallery — using only brain activity. Chance is 0.1%. We are 863 times above chance.

> The 95% confidence interval is [84.0, 88.4], computed using the Clopper-Pearson exact binomial method on 863 successes out of 1,000 trials.

*[click — R@5 and MRR]*

> Recall at 5 is 98% — the correct image is almost always in the top five. Mean Reciprocal Rank is 0.914 — the median predicted rank is 1.

*[click — pills]*

> That's plus 3.2 percentage points over V61a alone — the best single expert at 83.1%. And plus 9.1 points over our prior two-expert baseline at 77.2%.

**Transition:** "Before showing qualitative examples, let me put our result in context."

---

## Slide 8b — Context in the Literature [40 sec]

> Let me put our result in context. There are two metric families shown here.

> On the left: RETRIEVAL — identifying the exact image. MindEye achieves 93.2% in a gallery of 982 images. MindEye2 reaches 97.4% but uses 300-image batches and 8 subjects with shared pretraining. We reach 86.3% in a gallery of 1000 with a single subject. Brain Diffuser does NOT do retrieval at all.

> On the right: RECONSTRUCTION — generating pixels that look like the original. Brain Diffuser is strongest on SSIM because that is exactly what they optimize for.

*[click — approach comparison fragment]*

> The fundamental difference: Brain Diffuser regresses fMRI to a VAE latent and runs latent diffusion to GENERATE new pixels from scratch. They never search a gallery. We predict CLIP embeddings and search for the EXACT image in a gallery of 1000. Our 86.3% means we correctly identify the exact stimulus 86.3% of the time. A fundamentally different question.

*[click — disclaimer]*

> Different galleries, subjects, and protocols — the metrics are not directly comparable.

**Transition:** "Let me show specific examples of what the experts retrieve."

---

## Slide 9 — Qualitative Retrieval Evidence [50 sec]

> Two real SHARED1000 examples from the thesis. Each column shows one expert's top-1 retrieved image.

*[click — row 1]*

> Surfer, NSD 8262: V61a retrieves correctly at rank 1. V62a returns a wrong scene at rank 5. V66a misses at rank 8. Fusion: correct.

*[click — row 2]*

> Skier: V61a correct. V62a returns a related sport at rank 10. V66a misses at rank 11. Fusion: correct.

*[click — bottom text]*

> The scientific point here: these two examples illustrate that fusion preserves V61a's correct retrievals — it does not degrade them. The actual 3.2-point gain comes from the disjoint subset of queries where V61a fails but the weaker experts provide corrective scores.

**Transition:** "Let me show exactly what the wrong retrievals look like."

---

## Slide 9b — What Each Expert Retrieves [40 sec]

> Now let me show you what the WRONG retrievals actually look like — these are the exact figures from the thesis.

*[click — surfer row]*

> For the surfer query (NSD 8262): V61a gets it perfectly — same image at rank 1. But V62a retrieves a person with arms raised — semantically related (human + outdoor activity) but completely wrong identity, rank 5. V66a returns a person jumping at rank 8. Fusion: correct, because V61a's dominant weight pushes the correct answer to the top.

*[click — seagulls row]*

> For the seagulls: V61a correct again. V62a retrieves a flock of sheep on a hillside — the model captured "group of animals" but wrong species, rank 4. V66a gets a crowd of people with livestock at rank 12. Fusion: correct.

*[click — key insight]*

> The key insight: the errors are semantically plausible — the wrong experts retrieve images that share abstract semantics (groups, animals, outdoor activity). But the errors are geometrically different from V61a's error patterns. That geometric independence is exactly what makes score fusion gain 3.2 percentage points.

**Transition:** "Beyond retrieval, the thesis demonstrates a downstream application."

---

## Slide 10 — Downstream: Image Reconstruction [40 sec]

> I want to be explicit: the thesis contribution is retrieval, not reconstruction. Reconstruction is a controlled add-on demonstrating downstream utility.

> You see the visual flow here: ground truth, retrieved anchor, SDXL output. For perfect retrievals, the anchor IS the ground truth — diffusion preserves it. For non-perfect cases, it refines the anchor toward the brain signal.

> The pipeline: triple-fusion top-1 as IP-Adapter visual anchor for SDXL 1.0, with the fMRI-predicted CLIP embedding injected into the text pathway. Best-of-16 selection.

*[click — metrics table]*

> Results on 141 SHARED1000 examples: modest but non-degrading. SSIM from 0.202 to 0.214, CLIP sim from 0.744 to 0.750. AlexNet layer-5 two-way at 86.2% on the 137 non-perfect examples. Anchor-dependent, not protocol-comparable to MindEye2.

**Transition:** "To conclude — the contributions and their scope."

---

## Slide 11 — Contributions & Scope [60 sec]

*[click through each contribution]*

> Four contributions:

> First: a complete, leakage-free pipeline from fMRI to CLIP retrieval — single subject, image-level splits, no information leakage.

> Second: three complementary decoder experts — token, CLS, and ROI — targeting different embedding spaces with different training objectives.

> Third: frozen CSLS score fusion reaching 86.3% Recall at 1 on SHARED1000, with exact confidence interval. No test-time tuning, no leakage.

> Fourth: a controlled SDXL plus IP-Adapter reconstruction add-on.

*[click — what didn't work]*

> But equally important: what didn't work. Purely probabilistic vMF decoders were hard to stabilize without careful kappa regularization. Richer token targets sometimes degraded retrieval geometry due to hubness. And learned fusion gates overfit — frozen score fusion proved more robust than any trainable alternative. These negative results narrow the future search space and are as scientifically valuable as the positive ones.

*[click — limitations]*

> Limitations: single subject, fixed gallery of 1000, anchor-dependent reconstruction, not protocol-comparable to multi-subject works.

> Future: multi-subject generalisation, learned end-to-end fusion, per-query rescue analysis.

---

## Slide 12 — Thank You [10 sec]

> Thank you for your attention. I welcome your questions.

**Delivery:** Smile. Make eye contact with each committee member. Stand still.

---

---

## BACKUP SLIDES — Quick Reference for Q&A

### If asked about evaluation protocol:
> "We split at the image level. Each unique image is assigned to exactly one partition. Since NSD shows each image three times, all three repetitions stay together. SHARED1000 is entirely excluded from both training and validation. The weights 0.7, 0.1, 0.2 were selected on validation and frozen — they were never adjusted on test data."

### If asked "Is this mind reading?":
> "No. This is controlled visual perception decoding under a fixed-gallery retrieval protocol. The subject is viewing a photograph — we identify which one. It does not decode arbitrary thoughts, memories, or imagination."

### If asked "Does it reconstruct arbitrary images?":
> "No. The retrieval system selects from a fixed gallery of known images. The reconstruction add-on generates a variant conditioned on the retrieved anchor — it does not hallucinate novel images from brain activity alone."

### If asked "Why CLIP and not a different embedding?":
> "CLIP provides a 768-dimensional semantic space where visual content, text, and our predicted brain embeddings can all be compared via cosine similarity. Its multimodal alignment makes it a natural choice for bridging brain signals to image semantics."

### If asked "Why von Mises-Fisher instead of a standard Gaussian or MSE loss?":
> "CLIP embeddings are L2-normalized — they live on the unit hypersphere, not in unconstrained Euclidean space. The von Mises-Fisher distribution is the maximum-entropy distribution for directional data on the sphere — it's analogous to the Gaussian for Euclidean space. Using vMF means our training loss respects the intrinsic geometry of the target space: cosine similarity is the natural score function, and the concentration parameter kappa gives a principled uncertainty estimate per trial. A Gaussian in the ambient space would assign probability mass outside the hypersphere, which is geometrically incorrect. In practice, all three experts use vMF-NCE — a contrastive loss weighted by kappa — which combines the benefits of directional probabilistic modelling with hard negative ranking."

### If asked "What does kappa give you practically?":
> "Kappa is the concentration parameter of the vMF distribution — it measures how confident the decoder is about the predicted direction. High kappa means the model is confident; low kappa means uncertain. We exploit this in two ways: first, as a kappa regularizer during training to prevent collapse; second, at inference time for V61a, where we perform 16 stochastic forward passes with dropout active, average the resulting directions on the hypersphere, and re-normalize — this is Monte Carlo test-time augmentation. It reduces per-trial prediction variance and gives a +4 percentage-point gain over a single forward pass."

### If asked "Why not just use V61a alone?":
> "V61a alone achieves 83.1%. It fails on approximately 169 out of 1,000 queries. The weaker experts, despite lower individual accuracy, make different errors — they fail on different queries. The z-normalized weighted combination rescues a net positive number of those 169 failures, yielding the 3.2 percentage-point improvement."

### If asked "Why these specific weights?":
> "We performed a grid search over weight combinations on the validation set, optimizing CSLS Recall at 1. The optimal weights — 0.7, 0.1, 0.2 — were frozen and never modified during SHARED1000 evaluation. This prevents overfitting to the test set."

### If asked "How does CSLS work?":
> "CSLS — Cross-domain Similarity Local Scaling — adjusts cosine similarity by subtracting the mean similarity of each vector to its k nearest neighbours. This penalizes 'hub' vectors that are spuriously close to many queries, reducing retrieval errors caused by hubness in high-dimensional spaces."

### If asked "Can this generalise to other subjects?":
> "The pipeline architecture is subject-agnostic — each expert takes a voxel vector as input. For a new subject, one would retrain the experts on that subject's data. Multi-subject generalisation, potentially with shared encoder layers, is listed as future work."

### If asked "Why is reconstruction not the main contribution?":
> "Reconstruction quality depends heavily on the retrieval anchor — it conditions on the top-1 retrieved image. A system with perfect retrieval would produce near-perfect reconstructions trivially. The scientifically interesting question is the retrieval itself: can we identify the exact image from brain activity alone? That is what the thesis answers."

### If asked about the confidence interval:
> "The 95% CI [84.0, 88.4] is computed using the Clopper-Pearson exact method — a conservative binomial confidence interval based on 863 successes out of 1,000 independent trials. It makes no distributional assumptions."

### If asked "How does this compare to MindEye2?":
> "A direct numerical comparison is not scientifically valid. MindEye2 uses multi-subject pretraining across 7 NSD subjects, a different target space, different gallery sizes, and a different evaluation split protocol. We answer a narrower but well-defined question: what is achievable with three single-subject experts and frozen score fusion? Our result — 86.3% on SHARED1000 — demonstrates strong single-subject performance under strict no-leakage conditions."

---

## Presentation Checklist (pre-defense)

- [ ] Laptop connected to projector, resolution 1280×720 tested
- [ ] Browser opened to `index.html` in full-screen (F11 or F key in Reveal.js)
- [ ] Speaker notes visible on presenter screen (S key opens speaker view)
- [ ] Backup slides accessible via slide number (type number + Enter)
- [ ] Timer visible to presenter (phone or watch)
- [ ] Water bottle nearby
- [ ] Practice run completed at least 2 times under 10 minutes
- [ ] Laser pointer / clicker tested

---

## Key Numbers to Memorize

| Metric | Value | Source |
|--------|-------|--------|
| Main result | 86.3% CSLS R@1 | triple_fusion_shared1000_86.json |
| CI | [84.0, 88.4] | Clopper-Pearson exact, n=1000 |
| V61a alone | 83.1% | Same JSON |
| Prior baseline | 77.2% | V35+N1v28a fusion |
| Gain | +3.2 pp | 86.3 − 83.1 |
| R@5 | 98.0% | Same JSON |
| MRR | 0.914 | Same JSON |
| Chance | 0.1% | 1/1000 |
| Voxels | 15,724 | nsdgeneral mask |
| Trials | ~30,000 | 3 reps × 10,000 images |
| CLIP dim | 768 | ViT-L/14 |
| V61a dim | 197,376 | 257 × 768 tokens |
| Recon AlexNet(5) | 86.2% | metrics_summary.json |
| CSLS k | 3 | Fixed hyperparameter |
| Fusion weights | 0.7 / 0.1 / 0.2 | Selected on validation |
