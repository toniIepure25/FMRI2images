# Site-config freeze procedure

How a candidate site config becomes a frozen `SITE_CONFIG_v1`. No scientific parameter is ever outcome-tuned;
only operational hardware/sequence values are filled by the site.

1. **Populate candidate config** — start from `*_CANDIDATE.yaml` (verified public values + `TBD_SITE_OPERATOR`).
2. **Human MRI physicist review** — physicist fills sequence + hardware fields; PI reviews.
3. **Hardware dry-run** — run the trigger / frame-timing / response harness at the site (no participant).
4. **Fix operational defects only** — timing/mapping/hardware issues; never scientific parameters.
5. **Freeze `SITE_CONFIG_v1`** — set `status: SITE_CONFIG_v1`; every `TBD_SITE_OPERATOR` resolved.
6. **Hash** the frozen config.
7. **Commit.**
8. **Push.**
9. **Server-verify** (remote HEAD == local HEAD).
10. **Only then** run the engineering pilot.

## Engineering-pilot freeze (before first pilot)
Freeze together: PsychoPy version, site config, trigger mapping, response mapping, display parameters, run
partition, timing, instructions, stimuli, and BIDS mapping. The pilot may expose engineering defects; fixes are
documented and versioned (new protocol version + hash).

## Confirmatory freeze (before first confirmatory participant)
Require ALL, none marked true until real: site engineering certification; pilot complete; final protocol
version; ethics active; consent active; stimulus permissions active; real-site BIDS validation; participant-
independence system; Stage-A/Stage-B firewall; scanner config; PI authorization. The confirmatory guard fails
closed until every one is genuinely satisfied — there is no override flag.
