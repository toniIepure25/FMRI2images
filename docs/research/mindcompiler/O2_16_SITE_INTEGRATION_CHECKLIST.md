# O2.16 site-integration checklist

To be completed with a real MRI facility and PI **before** any confirmatory acquisition. Every unresolved item
below is a genuine blocker; none may be fabricated. This software provides SIMULATION/PILOT tooling only — it
does not grant scanner access, ethics approval, or consent.

## Scanner / sequence
- [ ] Scanner vendor / model — `TBD_BY_MRI_FACILITY`
- [ ] Field strength (target 7T where feasible) — `TBD`
- [ ] Functional sequence + voxel size (~1.8 mm isotropic target) — `TBD`
- [ ] TR (~1.6 s target) / TE / flip angle / multiband / phase encoding — `TBD`
- [ ] Distortion correction plan — `TBD`
- [ ] Structural: T1w ≤ 1 mm (+ T2w recommended) — `TBD`
- [ ] Dummy (non-steady-state) volume count — `TBD`
- [ ] Inter-run pause / run duration confirmed — `TBD`

## Synchronisation
- [ ] Trigger interface + exact trigger code (do **not** assume "5") — `TBD`
- [ ] Clock test: monotonic timing verified on the presentation machine — `TBD`
- [ ] Expected-vs-observed TR check passes on real triggers — `TBD`

## Presentation / response hardware
- [ ] Projector / display, resolution, refresh rate — `TBD`
- [ ] Viewing distance, screen dimensions, stimulus visual angle — `TBD`
- [ ] PsychoPy (pinned) installed; visual flip timing validated on the actual display — `TBD`
- [ ] Button box model + response mapping — `TBD`
- [ ] Audio system (if any) — `TBD`

## Data / BIDS
- [ ] Real-site BIDS destination path — `TBD`
- [ ] Official `bids-validator` run on a real dry-run dataset (PASS) — `TBD` (internal schema check only so far)
- [ ] Pseudonymous ID scheme approved; identity mapping stored outside the research dataset — `TBD`

## Stimuli
- [ ] 512 perception anchors bound to real files with verified SHAs and permissions — `TBD` (placeholders now)
- [ ] 12 imagery identities: final naturalistic stimuli + cue text, permissions cleared, `placeholder=false` — `TBD`

## Human authorization
- [ ] Ethics protocol ID + approving institution + valid dates — `TBD`
- [ ] Consent version + data-processing basis — `TBD`
- [ ] MRI safety screening process — `TBD`
- [ ] Site/institutional authorization — `TBD`
- [ ] Stimulus-use permissions — `TBD`

## Operations
- [ ] Scanner operator + emergency-stop procedure — `TBD`
- [ ] Phantom / hardware test if the facility requires — `TBD`
- [ ] **Actual-site engineering dry run** completed and reviewed — `TBD`

## Go / no-go for confirmatory
- [ ] Scanner sync success; event reconstruction 100%; acceptable missing-trigger rate; no critical BIDS error;
      stimuli display correctly; participant understands task; button box works.
- [ ] **Neural effect size is NOT a go/no-go criterion.**
- [ ] Authorization manifest complete → CONFIRMATORY guard passes (it fails closed otherwise; no override flag).
