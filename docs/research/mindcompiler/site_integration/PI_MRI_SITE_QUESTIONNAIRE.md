# PI / MRI physicist site questionnaire (O2.16 independent replication)

Each question maps to an unresolved config field (`TBD_SITE_OPERATOR`). Please answer only what your facility
supports; "not possible" answers are as useful as "yes". No confirmatory scanning is proposed until ethics and
authorization are complete.

## Scanner
- Which scanner would we use (vendor / model / **field strength**)?  → `field_strength`, `model`
- Is it research-dedicated, and what is the typical booking granularity?  → booking
- Head coil (channel count)?  → `head_coil`

## Sequence
- Recommended whole-brain functional sequence + **voxel size** (we target ~1.8 mm isotropic)?  → `voxel_size`
- **TR** (we target ~1.6 s), **TE**, flip angle, matrix/FOV, slices?  → `TR`, `TE`, `flip_angle`
- Multiband factor, partial Fourier, phase-encoding direction, bandwidth/echo spacing?  → `multiband`, `phase_encoding`
- Distortion correction (fieldmap / opposite-PE)?  → `distortion_correction`
- **Dummy / non-steady-state volumes** and reference scans?  → `dummy_volumes`
- Shim method?

## Trigger
- What **trigger** signal marks each volume, and exactly what key/code does it deliver to the task computer?
  (We do not assume "5".)  → `trigger_event`, `trigger_device`
- Trigger interface (keyboard emulation / serial / parallel)?

## Display
- **Display** hardware for stimulus presentation, its native resolution and **refresh rate**?  → `display`, `display_refresh_hz`
- Viewing distance / screen dimensions for visual-angle calibration?

## Button box
- **Button** box model and how presses appear to the task computer (which keys)?  → `response_device`, button mapping

## Task computer
- What presentation machine/OS is available; can we install pinned **PsychoPy**?

## Network
- Network access / isolation on the presentation machine?

## DICOM export
- **DICOM** export path and pseudonymization point?

## BIDS
- Is `dcm2niix` / a **BIDS** conversion pipeline available, and can we run `bids-validator`?

## Run length limits
- Maximum comfortable single-run and single-session duration?

## Participant safety
- MRI safety screening process; any **7T-specific** constraints?

## 7T-specific requirements
- If 7T: B0/B1 shimming, SAR limits, sequence optimization, additional safety steps?

## Pilot access
- Can we run an engineering pilot (≤ 2 participants, non-confirmatory) for hardware validation?

## Ethics
- Local **ethics** pathway and typical timeline; is PI sponsorship required?

## Data protection
- Data-protection basis and where research pseudonymization begins?

## Cost / booking
- Cost model and booking process for pilot vs confirmatory sessions?

## Staff support
- Is an MR operator/physicist available during pilot and confirmatory sessions?
