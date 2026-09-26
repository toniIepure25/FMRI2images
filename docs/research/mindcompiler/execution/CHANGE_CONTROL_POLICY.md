# Change-control policy for proposed modifications

Classify every requested change:

- **ADMINISTRATIVE** (wording, contacts, formatting) → apply; version-bump docs.
- **ENGINEERING** (software, timing implementation, BIDS mapping) → apply with tests; no scientific effect.
- **ACQUISITION** (sequence params, site hardware, run/session partition) → apply if it preserves the frozen
  design targets; record as site config; MR physicist signs off.
- **SCIENTIFIC NON-PRIMARY** (secondary/exploratory endpoints, added behavioral measures) → PI review; may add
  as clearly-labelled secondary; must not alter primary.
- **SCIENTIFIC PRIMARY** (the frozen primary calibration estimator, its training-only quality monitors, the
  operational success criterion, the primary endpoints, the participant-N requirement, and the cross-validation
  folds) → requires an explicit determination: does the original preregistration remain valid, or is a **new
  protocol version / new confirmatory design** required? **Never silently modify the frozen protocol.** No change
  is driven by observed outcomes.
