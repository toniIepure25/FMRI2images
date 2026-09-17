# O2.16-DATA scanner / site requirements

Status-neutral engineering spec. **Candidate-site values are `TBD_BY_MRI_FACILITY`** until a real facility confirms a sequence. Nothing below asserts an implemented configuration.

| Parameter | Required / preferred | Historical target | Candidate-site value | Compatible? | Action required |
|---|---|---|---|---|---|
| Field strength | 7T preferred | 7T | TBD_BY_MRI_FACILITY | TBD | confirm 7T availability |
| Head coil | multi-channel | (NSD 7T) | TBD | TBD | confirm coil |
| Voxel size | ~1.8 mm iso | 1.8 mm | TBD | TBD | confirm resolution feasible |
| TR | ~1.6 s | ~1.6 s | TBD | TBD | confirm TR |
| TE | site-optimal | (NSD) | TBD | TBD | confirm TE |
| Flip angle | site-optimal | (NSD) | TBD | TBD | confirm FA |
| Multiband | as needed for coverage | (NSD MB) | TBD | TBD | confirm MB factor |
| Phase encoding | AP/PA (fieldmap) | (NSD) | TBD | TBD | confirm PE + fieldmaps |
| Field maps / distortion | required | yes | TBD | TBD | confirm fieldmap protocol |
| Number of slices | whole-brain | whole-brain | TBD | TBD | confirm coverage |
| Whole-brain coverage | required (ventral+lateral streams) | yes | TBD | TBD | confirm coverage |
| Structural | T1w <=1 mm (+T2w) | T1w<=1mm | TBD | TBD | confirm structural |
| Response device | MRI-safe button box (>=2 buttons) | yes | TBD | TBD | confirm device |
| Stimulus display | MRI-safe projector/goggles | yes | TBD | TBD | confirm display |
| Synchronization trigger | scanner->stimulus TTL | yes | TBD | TBD | confirm trigger |
| Physiological recording | optional (QC only) | optional | TBD | TBD | confirm availability |

If a candidate site cannot meet the representation-critical rows (field strength class, voxel size, whole-brain coverage, native-space) -> `O2_16_DATA_SITE_COMPATIBILITY_BLOCKED` and consider a separately frozen conceptual-replication gate.
