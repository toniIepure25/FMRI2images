"""O2 shared + subject-specific perception->imagery operator driver (3 phases).

--freeze     : config + manifests + contracts + prior-art (before any outcome)
--phase-a    : VISION-ONLY common-space K-selection + validation (go/no-go BEFORE imagery)
--phase-b [--rois ...] : PRIMARY imagery-zero-shot LOSO (ventral/lateral/parietal); inference; status
--phase-c    : secondary V1/V3 + Delta_s decomposition + gauge-invariant geometry

B0 only. Target-subject imagery NEVER used for SRM/lambda/T. Global identity holdout. No raw-W
cross-subject comparison. Raw HDF5 / full native transforms never committed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from fmri2img.mindcompiler.operator_o1 import folds as ofo                     # noqa: E402
from fmri2img.mindcompiler.operator_o2 import pipeline as pl                   # noqa: E402
from fmri2img.mindcompiler.operator_o2 import shared_ops as so                 # noqa: E402
from fmri2img.mindcompiler.operator_o2 import srm as SRM                       # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_roi as roi        # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp  # noqa: E402

DATA = _REPO / "data/nsd"
OUT = _REPO / "artifacts/mindcompiler/operator_o2"
B0 = ("nsdimagerybetas_fithrf", "31485ff0e4cb9e90b6f83f2f550714b1a688993604f5795148a2746df7b42b64")
ALL = ["subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"]
PRIMARY = ["ventral", "lateral", "parietal"]; SECONDARY = ["V1", "V3"]; EXCLUDED = ["V2", "hV4"]
K_CANDIDATES = [2, 3, 4, 5, 6, 7]


def _git(a, d=""):
    for pre in (["git", "-c", "safe.directory=*", "-C", str(_REPO)], ["git", "-C", str(_REPO)]):
        try:
            return subprocess.check_output(pre + a, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            continue
    return d


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _identities():
    tt = sp.build_trial_table(str(DATA / "nsddata/bdata/nsdimagery"), subject="subj01")
    ids = sorted(tt["identity"].unique())
    fam = {i: tt.loc[tt.identity == i, "family"].iloc[0] for i in ids}
    return ids, fam


def load_roi(rname):
    """All-subject B0 centroids for one ROI: cvis/cimg {(subj,identity): raw vector}, min voxel count."""
    cvis, cimg, nvox = {}, {}, {}
    for s in ALL:
        tt = sp.build_trial_table(str(DATA / "nsddata/bdata/nsdimagery"), subject=s)
        ids = sorted(tt["identity"].unique())
        R = str(DATA / f"nsddata/ppdata/{s}/func1pt8mm/roi"); P = str(DATA / f"nsddata/ppdata/{s}/func1pt8mm")
        N = str(DATA / f"nsddata_betas/ppdata/{s}/func1pt8mm/betas_fithrf/ncsnr.nii.gz")
        sel = roi.select_roi_voxels(rname, R, P, N); xyz = sel["xyz"]
        p = DATA / f"nsddata_betas/ppdata/{s}/func1pt8mm/{B0[0]}/betas_nsdimagery.hdf5"
        M = sp.extract_v1_matrix(str(p), tt["beta_index0"].values, xyz)
        for i in ids:
            cvis[(s, i)] = M[tt.index[(tt.state == "vision") & (tt.identity == i)]].mean(0)
            cimg[(s, i)] = M[tt.index[(tt.state == "imagery") & (tt.identity == i)]].mean(0)
        nvox[s] = int(xyz.shape[1])
    return cvis, cimg, nvox


def _kcap(nvox, ids):
    return [K for K in K_CANDIDATES if K <= min(nvox.values()) - 1 and K <= len(ids) - 1]


# -------------------------------------------------------------------------- FREEZE
def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    ids, fam = _identities()
    outer_id = ofo.outer_folds(ids, fam)
    rows = []
    for ti, target in enumerate(ALL):
        for k, f in enumerate(outer_id):
            for i in f.test:
                rows.append(dict(target_subject=target, identity_fold=k, role_identity="test", identity=i, family=fam[i]))
            for i in f.train:
                rows.append(dict(target_subject=target, identity_fold=k, role_identity="train", identity=i, family=fam[i]))
    pd.DataFrame(rows).to_csv(OUT / "outer_subject_identity_manifest.csv", index=False)
    cfg = {"gate": "O2", "analysis_class": "FROZEN_CROSS_SUBJECT_AND_CROSS_STIMULUS_OPERATOR_GENERALIZATION",
           "primary_claim_class": "IMAGERY_ZERO_SHOT_SUBJECT_TRANSFER_WITH_VISION_ONLY_CALIBRATION",
           "primary_beta": "B0 only", "B0_measurement_dependence_is_a_limitation": True,
           "participants": ALL, "primary_rois": PRIMARY, "secondary_rois": SECONDARY, "excluded_rois": EXCLUDED,
           "excluded_justification": "V2/hV4 lacked cohort Holm evidence in O1 and O1.1 classified their failures MULTIFACTORIAL",
           "common_space_method": "DETERMINISTIC_SHARED_RESPONSE_MODEL", "common_space_from": "VISION_ONLY",
           "same_W_applied_to_vision_and_imagery": True, "K_candidates": K_CANDIDATES,
           "K_selection": "nested vision-only: inner LOO subject (7) x inner identity fold (5) = 35 cells; max mean VISION_IDENTITY_MARGIN; tie smaller K",
           "outer_design": "8 LOSO subject folds x 6 identity folds x 3 primary ROIs = 144 primary cells; each: 7 train subjects x 10 train identities, 1 target x 2 held-out identities",
           "global_identity_holdout": True, "target_subject_vision_only_calibration": True,
           "shared_operator": "FULL_SHARED_SCALAR_RIDGE (gauge-equivariant; NO per-target lambda, NO diagonal)",
           "ridge_grid": {"lo": so.RIDGE_LO, "hi": so.RIDGE_HI, "n": so.RIDGE_N},
           "lambda_selection": "training-subjects-only nested CV (LOO train subject x identity holdout); mean held-out identity pattern r; tie larger lambda",
           "baselines": {"S0": "GROUP_IMAGERY_MEAN (train subjects x train identities, shared space)",
                         "S1": "GLOBAL_SHARED_GAIN (gauge-invariant); NO diagonal common-space baseline"},
           "primary_metric": "NATIVE_HELD_OUT_PATTERN_R (Pearson across target-native voxels)",
           "primary_effect": "G_shared = r(T_shared) - r(S0)", "secondary_effect": "G_beyond_gain = r(T_shared) - r(S1)",
           "inference": {"unit": "participant (N=8)", "test": "exact 2^8 sign-flip", "correction": "Holm across 3 primary ROIs", "alpha": 0.05},
           "delta_s": "subject-specific residual operator for TRAINING subjects only (never target); scalar ridge; identity-held-out CV",
           "gauge_invariants_only": ["singular values", "spectral/frobenius norm", "effective rank", "NONTRIVIAL_TRANSFORMATION_INDEX", "polar stretch"],
           "no_offdiagonal_fraction_reported": True,
           "prohibited": ["target-subject imagery in SRM/lambda/T", "test-identity leakage", "B1 primary", "raw-W cross-subject comparison",
                          "diagonal/per-target lambda in common coords", "V2/hV4 primary", "zero-calibration wording", "causal claim"],
           "immutable": {"O1_STATUS": "STIMULUS_INVARIANT_OPERATOR_PARTIAL", "O1_1_STATUS": "OPERATOR_HETEROGENEITY_REGION_STRUCTURE_DOMINANT",
                         "TRACK_R": "INDEPENDENT_METHOD_REPRODUCTION_PARTIAL"}}
    body = json.dumps(cfg, indent=2, sort_keys=True)
    (OUT / "o2_frozen_config.json").write_text(json.dumps({"config_sha256": hashlib.sha256(body.encode()).hexdigest(), **cfg}, indent=2))
    (OUT / "common_space_contract.json").write_text(json.dumps({
        "method": "DETERMINISTIC_SHARED_RESPONSE_MODEL", "fit_on": "VISION ONLY (never imagery)",
        "new_subject": "vision-only calibration on TRAIN identities; orthogonal Procrustes to trained S",
        "same_W_both_states": True, "hyperalignment": "COMMON_SPACE_METHOD_SENSITIVITY_FUTURE (not an O2 selection branch)",
        "K_selection_metric": "VISION_IDENTITY_MARGIN (no imagery metric permitted)"}, indent=2))
    (OUT / "operator_contract.json").write_text(json.dumps({
        "estimator": "FULL_SHARED_SCALAR_RIDGE", "gauge_equivariant": "T' = Q^T T Q for X'=XQ,Y'=YQ",
        "forbidden": ["per-target lambda", "diagonal penalty", "coordinate-specific sparsity"],
        "S0": "GROUP_IMAGERY_MEAN", "S1": "GLOBAL_SHARED_GAIN", "no_diagonal_baseline": True}, indent=2))
    (OUT / "scientific_status_contract.json").write_text(json.dumps({
        "SHARED_OPERATOR_SUPPORTED_requires_ALL": {
            "A": "COMMON_SPACE_VALIDATED", "B": "G_shared Holm-rejects in >=2/3 primary ROIs",
            "C": "all 3 primary ROIs median G_shared > 0", "D": "for each ROI under B, >=6/8 participants G_shared>0",
            "E": "G_beyond_gain positive median in >=2/3 ROIs AND Holm-rejects in >=1/3",
            "F": "leave-one-participant sensitivity does not reverse median G_shared sign in any ROI under B"},
        "SHARED_OPERATOR_PARTIAL": "common space validates AND (only one ROI passes OR beats S0 not gain coherently OR substantial heterogeneity)",
        "SHARED_OPERATOR_NOT_SUPPORTED": "common space validates AND zero ROI passes G_shared inference AND median G_shared<=0 in >=2/3 ROIs",
        "SHARED_OPERATOR_INCONCLUSIVE": "common space fails / too few evaluable / leakage / numerical failure"}, indent=2))
    (OUT / "prior_art_registry.json").write_text(json.dumps(PRIOR_ART, indent=2))
    print("frozen config sha:", json.loads((OUT / "o2_frozen_config.json").read_text())["config_sha256"])
    return 0


PRIOR_ART = {"novelty_label_overall": "NOVELTY_CANDIDATE",
    "question": "State-to-state perception->imagery OPERATOR tested on an imagery-unseen participant AND held-out stimulus identities, with vision-only SRM calibration and a shared-vs-subject-specific decomposition.",
    "close_prior_art": [
        {"work": "Shared Response Model (Chen et al. NeurIPS 2015)", "relation": "the common-space method used here; aligns responses, not a state-to-state operator", "label": "METHOD_USED"},
        {"work": "Hyperalignment (Haxby/Guntupalli)", "relation": "cross-subject common space incl. held-out subjects; not perception->imagery operator", "label": "CLOSE_PRIOR_ART_EXISTS"},
        {"work": "MindEye2 / MindBridge / MindAligner / MindAdapter / Duala (2024-2026)", "relation": "cross-subject VISUAL DECODING (brain->image), some new-subject calibration; not a neural-state->state operator transferred zero-shot to imagery", "label": "CLOSE_PRIOR_ART_EXISTS"},
        {"work": "Roy et al. 2025", "relation": "within-subject vision->imagery transformation, same-identity", "label": "CLOSE_PRIOR_ART_EXISTS"}],
    "novelty_statement": "Cross-subject alignment (SRM/hyperalignment) and cross-subject visual decoding are established; a SHARED perception->imagery neural-state OPERATOR transferred zero-shot (imagery-wise) to a new subject on held-out identities, with shared/subject-specific split, is a NOVELTY_CANDIDATE. No 'first ever'.",
    "sources": ["https://dl.acm.org/doi/10.5555/2969239.2969291", "https://arxiv.org/pdf/2403.11207",
                "https://arxiv.org/pdf/2502.05034", "https://pubmed.ncbi.nlm.nih.gov/40950062/"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--phase-a", action="store_true")
    ap.add_argument("--phase-b", action="store_true")
    ap.add_argument("--phase-c", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        return freeze()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from operator_o2_phases import phase_a, phase_b, phase_c   # local import to keep driver lean
    if a.phase_a:
        return phase_a(load_roi, _identities, _kcap, OUT, ALL, PRIMARY, K_CANDIDATES, _git, _sha)
    if a.phase_b:
        return phase_b(load_roi, _identities, OUT, ALL, PRIMARY, _git, _sha)
    if a.phase_c:
        return phase_c(load_roi, _identities, OUT, ALL, SECONDARY, _git, _sha)
    print("specify --freeze | --phase-a | --phase-b | --phase-c", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
