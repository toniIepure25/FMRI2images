# Why MINDIR matters (`SCIENTIFIC_VALUE.md`)

MINDIR is more than another brain-to-image decoder. Most imagery decoding asks *how well* we can reconstruct an
imagined image. MINDIR asks a structural, cost-oriented question: **which components of imagery does perception
fail to specify, how do they vary across people, and how little data are needed to calibrate to them.**

- **Support vs private geometry:** a principled split of the imagined state into a perception-supported part and
  a subject-specific residual — a lens most decoders do not apply.
- **Participant-specific calibration:** the residual does not transfer across people (under the tested analysis),
  so per-person calibration is necessary — and its *data cost* becomes the interesting quantity.
- **Sample efficiency:** a small, quantified calibration frontier (estimator-specific), not an accuracy score.
- **Acquisition reliability:** which measurements you acquire matters; repeat-geometry predicts calibration
  success — directly useful for designing efficient imagery experiments.
- **Zero-shot potential:** if the missing geometry can be predicted from non-imagery data, imagery decoding gets
  much cheaper (preregistered, unproven).
- **Cross-state potential:** if the same shared-scaffold + private-correction structure recurs in other internal
  states (e.g. delayed recall), it points to a general organization of internally generated states (preregistered).

## What we know vs what we hope to discover
- **KNOW (development cohort, N=8):** perception is incomplete; a low-dim subject-specific residual exists; a
  small, fragile calibration frontier; repeat-geometry predicts success.
- **HOPE TO DISCOVER (requires new data):** that these replicate; that geometry is a stable cross-session
  fingerprint; that it is predictable zero/near-zero-shot; that it generalizes across internal states.

No biological generality is claimed today. The value proposition is a rigorous, falsifiable, reproducible program
with a frozen replication ready to run.
