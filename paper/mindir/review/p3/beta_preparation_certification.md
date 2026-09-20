# Beta-preparation certification (P3 Methods closure)

**Verified against sealed artifact** `artifacts/mindcompiler/roy_s2_5r/beta_version_evidence.json` and
`artifacts/mindcompiler/roy_s2_7r/final_method_concordance.csv` (row `M-BETA`). No new analysis.

## Sealed evidence (verbatim fields)
- `project_branches.B0 = "nsdimagerybetas_fithrf (= b2)"`
- `project_branches.B1 = "nsdimagerybetas_fithrf_GLMdenoise_RR (= b3)"`
- `nsd_manual_mapping = {b1: betas_assumehrf, b2: betas_fithrf, b3: betas_fithrf_GLMdenoise_RR}`
- `conclusion = "PUBLIC_EVIDENCE_STRONGLY_IDENTIFIES_B0_AS_B2_COMPATIBLE"`
- `caveat = "'similar to b2' != bitwise-identical released file"`
- concordance `M-BETA`: paper "'similar to b2'" vs ours "B0 = nsdimagerybetas_fithrf (b2-compatible)" →
  `CLOSE_DEFENSIBLE_RECONSTRUCTION`.

## Certified Methods wording (approved for the manuscript)
> The primary analyses used the NSD-Imagery **B0** beta preparation, `nsdimagerybetas_fithrf`, corresponding to
> the public NSD **`betas_fithrf`** (b2-compatible) preparation. The public method description is "similar to
> b2"; we therefore do **not** claim that the released public file is bitwise identical to the beta file used in
> the original Roy et al. analysis. A GLMdenoise/RR (b3-compatible) preparation, `nsdimagerybetas_fithrf_
> GLMdenoise_RR` (branch **B1**), was examined only as a measurement-preparation **sensitivity** branch and is
> **not** the primary preparation.

## Guards preserved
- `ORIGINAL_CODE_REPRODUCTION_UNAVAILABLE` and `BITWISE_REPLICATION_UNAVAILABLE_WITHOUT_ORIGINAL_CODE_AND_SEEDS`
  retained.
- The manuscript must **not** state the primary imagery analysis used GLMdenoise_RR (that is the B1 sensitivity
  branch).

**Status: RESOLVED — beta-preparation placeholder closed against sealed evidence.**
