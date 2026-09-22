# 7T vs 3T — technical note (engineering/operational tradeoffs only)

This note weighs practical tradeoffs for the O2.16 replication. It reaches **no** conclusion that one field
strength is automatically scientifically superior, and it does **not** change the frozen scientific design.

## 7T (e.g. MedUni Vienna HFMRC, Magnetom 7T)
- Potentially higher spatial resolution / SNR for fine functional geometry.
- More B0/B1 inhomogeneity and susceptibility artifacts; sequence optimization is more involved.
- Stricter safety and hardware constraints (SAR, shimming); potentially greater setup complexity.
- The historical discovery data are 7T high-resolution, so field strength and ~1.8 mm resolution are more
  directly comparable.

## 3T (e.g. University of Vienna MR Center, MAGNETOM Skyra)
- Broader standardization and easier, well-established cognitive-fMRI operations.
- A mature, integrated stimulus/response stack is often already in place (display, response pad, eye-tracking,
  physio, audio), reducing engineering/timing-validation burden.
- Potentially simpler replication logistics and scheduling.

## Guidance
The replication should prioritize **protocol fidelity + stable acquisition + a genuinely independent cohort**
over field strength alone. A well-controlled 3T acquisition that faithfully implements the frozen design and
delivers stable timing is preferable to a higher-field acquisition with greater operational risk — and vice
versa if a 7T site offers both fidelity and stability. The choice is an engineering-feasibility decision for the
PI + MRI physicist, informed by the compatibility matrix and the site's answers to the questionnaire; the frozen
O2.16 scientific design is not adjusted to favour either option.
