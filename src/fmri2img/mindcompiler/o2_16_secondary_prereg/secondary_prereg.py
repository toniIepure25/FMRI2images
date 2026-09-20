"""MINDIR O2.16-SEC -- Independent-cohort SECONDARY endpoint PREREGISTRATION (freeze-only, ZERO inference).

This module freezes the DECISION LOGIC (support rules, six-test multiple-testing plan, program-status
classification, participant-unit + historical-exclusion + primary/secondary-separation guards) for three
prospectively-declared secondary replication endpoints, to be evaluated ONLY on a future genuinely-independent
O2.16 cohort. It computes NO scientific outcome and reads NO historical N=8 data. Historical N=8 contributes
definitions/frozen formulas/descriptive references ONLY -- never replication inference.

Secondary family (EXACTLY 3 findings x 2 ROIs = 6 tests), Holm @ alpha=0.05 across the six:
  S1  X1 principal-angle invariant  B_angles          (donor-mean LOSO E_SHARED_angle + pipeline-triviality guard)
  S2  O2.15 OUTSIDE-support agreement Q_OUT -> margin  (participant Spearman(Q_OUT, ROBUSTNESS_MARGIN), Fisher-z)
  S3  X4 geometric anisotropy       ANISO_CENTRED      (CENTRED top-fraction vs isotropic null + cross-half repro)

Advanced ONLY because each survived its own prospectively-defined discovery guard. Failed hypotheses
(X1 spectral/support/compression; X2 K2-K1; X3 angular-scaffold; X3 scale->margin; X4-H2 MODE_GAP->margin)
are PERMANENTLY EXCLUDED from the positive family. No X5. No historical feature search."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F

ROIS = ("ventral", "lateral")
ENDPOINTS = ("S1_Bangles", "S2_QOUT", "S3_ANISO")          # the ONLY positive replication endpoints
SIX_TESTS = tuple((roi, e) for e in ENDPOINTS for roi in ROIS)   # exactly 6
ALPHA = 0.05
POS_FRACTION = 0.75                                         # >=75% independent participants effect>0
REPRO_FRACTION = 0.75                                       # S3 cross-half reproducibility participant fraction
S3_STATISTIC = "ANISO_CENTRED"                             # CENTRED top-fraction, null expectation 0 -- NEVER the ratio
HISTORICAL_N8 = tuple("subj0%d" % i for i in range(1, 9))  # excluded from ALL replication inference
FAILED_HYPOTHESES = (
    "X1_spectral_block", "X1_support_block", "X1_compression_block",   # pipeline-trivial
    "X2_K2_minus_K1",                                                   # 0/8 positive; Student-t >= K2
    "X3_repeat_stable_angular_scaffold",                               # failed
    "X3_student_t_scale_to_margin",                                    # failed
    "X4_H2_mode_gap_to_margin",                                        # failed + opposite sign
)


def frac_pos(effects):
    e = [v for v in effects if v is not None and np.isfinite(v)]
    return (sum(v > 0 for v in e) / len(e)) if e else 0.0


def _median(effects):
    e = [v for v in effects if v is not None and np.isfinite(v)]
    return float(np.median(e)) if e else float("nan")


def replication_inference_set(participants):
    """Independent-cohort ONLY: drop any historical N=8 id. Returns (kept, dropped)."""
    kept = [p for p in participants if p not in HISTORICAL_N8]
    dropped = [p for p in participants if p in HISTORICAL_N8]
    return kept, dropped


def participant_signflip_p(effects):
    """One-sided sign-flip participant test (exact 2^N when feasible; else the deterministic MC frozen by O2.16).
    Historical participants must already be excluded upstream."""
    return F.signflip_p_onesided(list(effects))


def six_test_holm(pvals_by_test):
    """Holm across EXACTLY the six preregistered tests. Keys must be the six (roi, endpoint) pairs."""
    assert set(pvals_by_test.keys()) == set(SIX_TESTS), "Holm family must be exactly the six preregistered tests"
    keyed = {"%s|%s" % (roi, e): pvals_by_test[(roi, e)] for (roi, e) in pvals_by_test}
    rej = F.holm(keyed, alpha=ALPHA)
    return {(roi, e): bool(rej["%s|%s" % (roi, e)]) for (roi, e) in pvals_by_test}


# ---- per-ROI support rules (each requires its finding's own frozen guard) ----
def support_S1(effects, holm_reject, triviality_pass):
    """PRINCIPAL_ANGLE_INVARIANT_REPLICATED iff median E_SHARED_angle>0 AND >=75% >0 AND Holm-reject AND the
    independent cohort independently reproduces the X1 non-triviality guard."""
    return bool(_median(effects) > 0 and frac_pos(effects) >= POS_FRACTION and holm_reject and triviality_pass)


def support_S2(rhos, holm_reject):
    """Q_OUT_FAILURE_MODE_REPLICATED iff median participant rho>0 AND >=75% rho>0 AND Holm-reject. No min effect."""
    return bool(_median(rhos) > 0 and frac_pos(rhos) >= POS_FRACTION and holm_reject)


def support_S3(aniso_centred, holm_reject, repro_participant_frac, triviality_pass):
    """GEOMETRIC_ANISOTROPY_REPLICATED iff median ANISO_CENTRED>0 AND >=75% >0 AND Holm-reject AND >=75%
    participants pass cross-half reproducibility AND norm/rank-matched isotropic triviality control passes.
    Uses the CENTRED statistic ONLY (null expectation 0); the unstable ratio is never an inference input."""
    return bool(_median(aniso_centred) > 0 and frac_pos(aniso_centred) >= POS_FRACTION and holm_reject
                and repro_participant_frac >= REPRO_FRACTION and triviality_pass)


def classify_program(support):
    """support[roi][endpoint] -> bool. Returns exactly one frozen secondary program status.
    A finding 'replicates' (consistently) iff it replicates in BOTH ROIs; a finding replicating in exactly one
    ROI is a substantial ROI difference -> MULTIREGIME. INCONCLUSIVE is set upstream on technical failure only."""
    both = {e: bool(support["ventral"][e] and support["lateral"][e]) for e in ENDPOINTS}
    xor = {e: bool(support["ventral"][e] != support["lateral"][e]) for e in ENDPOINTS}
    any_roi = {e: bool(support["ventral"][e] or support["lateral"][e]) for e in ENDPOINTS}
    n_both = sum(both.values())
    if n_both == 3:
        return "THREE_GEOMETRIC_DISCOVERY_FINDINGS_INDEPENDENTLY_REPLICATED"
    if not any(any_roi.values()):
        return "GEOMETRIC_DISCOVERY_FINDINGS_NOT_REPLICATED"
    if any(xor.values()):
        return "GEOMETRIC_DISCOVERY_REPLICATION_MULTIREGIME"
    return "GEOMETRIC_DISCOVERY_REPLICATION_PARTIAL"


def assert_no_failed_hypothesis(endpoint_family):
    """Guard: the positive replication family may contain none of the frozen failed hypotheses."""
    bad = set(endpoint_family) & set(FAILED_HYPOTHESES)
    if bad:
        raise ValueError("failed hypotheses may never enter the positive family: %s" % sorted(bad))
    return True


def seal_secondary(primary_status, secondary_status):
    """Primary/secondary separation: the primary O2.16 status is returned VERBATIM; the secondary status can
    never modify it. Reporting is primary-first by construction (primary key precedes secondary)."""
    return {"o2_16_primary_status": primary_status, "secondary_status": secondary_status,
            "separation": "secondary NEVER modifies primary; no secondary positive rescues a failed primary; "
                          "no secondary negative erases a successful primary"}
