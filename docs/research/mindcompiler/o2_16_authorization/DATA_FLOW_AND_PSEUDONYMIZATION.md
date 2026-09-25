<!-- DOC_ID: O216-AUTH-10 | VERSION: 0.1.0-DRAFT | DATE: 2026-09-25 | SCIENTIFIC_PROTOCOL_SHA: 2da2cc791054406b42eb1896b9b5f123b5b6bacb8386e6fb0f1e19649cf18907 | SITE_CONFIG_SHA: TBD_SITE_OPERATOR | APPROVAL_STATUS: DRAFT_FOR_HUMAN_REVIEW -->

# Data flow and pseudonymization

> **DRAFT_FOR_HUMAN_REVIEW.** No approval (PI, ethics, site, data-protection, consent, stimulus licence) has been granted. This is a draft to be transferred into the institution's official forms.


```
[Identity data] --(secure, offline key)--> [pseudonym sub-XXXX]
Scanner -> DICOM (PHI in headers) -> pseudonymize + deface anatomicals -> BIDS (pseudonymous) ->
frozen preprocessing -> Stage-A predictors (no outcomes) --(human release token)--> Stage-B protected outcomes
```
PHI exists at the scanner/DICOM stage; **research pseudonymization begins before data enter the research zone**.
No cloud upload of identifiable information. Stage-A cannot read Stage-B; Stage-B stays locked until a
human-created release token with full certification exists.
