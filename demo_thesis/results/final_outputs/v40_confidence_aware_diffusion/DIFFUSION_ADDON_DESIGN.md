# DIFFUSION ADDON DESIGN

## Audit of the Previous Addon

The previous true-diffusion addon had three main failure modes:

1. It applied one global diffusion recipe to every case, so even perfect retrievals were denoised and damaged.
2. It built the prior by directly averaging top-k latents, which washed out structure and encouraged blur / semantic averaging.
3. It used a soft semantic selector that was too permissive, so decorative or abstract samples could beat more faithful anchor-preserving outputs.

This made the old addon visually destructive and only weakly aligned with the frozen retrieval system.

## What V40 Reuses

- the frozen tri-expert retrieval system and its exact approved fusion recipe
- the frozen shortlist / top-k retrieval outputs
- the NSD stimulus image loading path
- the existing reconstruction metrics (MSE, PSNR, SSIM, pixel correlation, LPIPS, CLIP similarity)

## What V40 Replaces

- replaces naive top-k latent averaging with an anchor-first residual prior
- adds a confidence-aware controller with four routes:
  - identity_pass
  - low_strength_refine
  - guided_refine
  - exploratory_refine
- adds stronger candidate selection using:
  - anchor similarity
  - shortlist consistency
  - prior similarity
  - layout preservation
  - edge preservation
  - color-drift penalty
- always keeps the anchor and prior themselves as selectable candidates, so diffusion cannot force a worse output

## Why This Is More Scientifically Defensible

V40 is architecture-aligned because it treats the frozen tri-fusion shortlist as the semantic backbone and diffusion only as a refinement stage. It respects the retrieval evidence, preserves strong retrievals, and only broadens exploration when the shortlist itself is uncertain.

Ground truth is used for evaluation and reporting only. It is never used to build the prior, choose the route, or select the final candidate.

## Honest Limitations

- If retrieval is already exact, diffusion is usually unnecessary and often worse; V40 therefore explicitly allows pass-through behavior.
- The controller is still heuristic because it is inference-only and deliberately avoids new training.
- Hard cases remain limited by the semantic content of the frozen shortlist itself.
