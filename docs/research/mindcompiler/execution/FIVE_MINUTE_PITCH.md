# 5-minute scientific narrative (repository-independent)

Mental imagery and perception recruit overlapping visual cortex, but similarity of activation does not mean the
representations are identical. I asked a specific, quantitative question: how much of the imagined ("target")
neural state is already specified by perception, how much is unique to the individual, and how little direct
imagery data are needed to calibrate a model to the part perception cannot supply.

Using the public Natural Scenes Dataset (eight participants, which I treat strictly as a development cohort), I
decomposed each person's imagined state into a component lying inside the perception-supported subspace and a
residual outside it. Perception carried substantial, reproducible information — but a real, low-dimensional
residual remained. That residual's orientation did not transfer across people, and was not predicted by the
perception phenotype, anatomy, or behaviour I tested. It is, in that specific sense, subject-specific.

Crucially, that subject-specific geometry is not noise: a small amount of direct imagery calibration recovered
much of it. Under a fixed estimator, roughly eight imagery observations recovered at least half of the gap to a
within-person oracle in two visual streams. But this frontier was fragile — which measurements were acquired
mattered as much as how many, and the repeat-to-repeat geometry of the outside-of-perception component predicted
whether a calibration generalized.

I then asked what that failure-predictive structure is. Exploratory analyses ruled out a discrete latent state,
a scalar reliability variable, and a single dominant drift mode — the structure is distributed.

All of this is development-cohort evidence. To test whether it generalizes, I have prospectively frozen and
preregistered an independent replication in a new cohort, with a fixed analysis and an outcome firewall, plus
three preregistered secondary geometric endpoints. The natural next steps — knowing online when enough imagery
data has been collected, predicting a person's imagery geometry with little or no imagery data, and testing
whether the same structure appears in other internal states like delayed recall — are all preregistered and
require new data. What I need now is a supervisor, an MRI site, and scientific feedback.
