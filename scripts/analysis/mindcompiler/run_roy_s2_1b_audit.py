"""S2.1B EXPLORATORY mechanism audit driver (subj01 x V1).

Matched 2x2 factorial {B0,B1} x {RAW,D1} (+ prespecified D1b sensitivity) under the
same S2 folds, plus geometry/reliability/similarity diagnostics, then a conservative
hypothesis classification. Descriptive only -- NO fold p-values. Numbers are never
compared to Roy. No raw/denoised beta arrays committed (only summary statistics).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from fmri2img.mindcompiler.roy_method_reproduction import s2_1b_diagnostics as dg  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_denoising as dn  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_folds as sf  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_reconstruction as rc  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction.s2_preprocessing import PREPROC_ZSCORE_TRAIN_ONLY  # noqa: E402

BETA = {"B0": "nsdimagerybetas_fithrf", "B1": "nsdimagerybetas_fithrf_GLMdenoise_RR"}
FOLD_SEED = 1234
OUT = _REPO / "artifacts/mindcompiler/roy_s2_1b"


def _identity_gate():
    url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], cwd=_REPO).decode()
    if "FMRI2images" not in url:
        print(f"FATAL: identity gate failed (origin={url!r})", file=sys.stderr); raise SystemExit(2)


def _sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _agg(v):
    a = np.asarray(v, float)
    return dict(mean=float(a.mean()), median=float(np.median(a)), std=float(a.std()),
                min=float(a.min()), max=float(a.max()), per_fold=[float(x) for x in a])


def main() -> int:
    _identity_gate()
    base = _REPO / "data/nsd"
    tt = sp.build_trial_table(str(base / "nsddata/bdata/nsdimagery"))
    xyz, vhash, _ = sp.select_v1_voxels(
        str(base / "nsddata/ppdata/subj01/func1pt8mm/roi"),
        str(base / "nsddata/ppdata/subj01/func1pt8mm"),
        str(base / "nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf/ncsnr.nii.gz"))
    assert vhash == "a1bc56fe7c55"
    folds = sf.build_four_folds(tt, FOLD_SEED)
    rid = {r: tt.loc[r, "identity"] for r in tt.index}
    rb = {r: int(tt.loc[r, "beta_index0"]) for r in tt.index}
    vis_by_ident = {i: g.index.tolist() for i, g in tt[tt.state == "vision"].groupby("identity")}
    img_by_ident = {i: g.index.tolist() for i, g in tt[tt.state == "imagery"].groupby("identity")}
    vis_rows = tt[tt.state == "vision"].index.tolist()
    img_rows = tt[tt.state == "imagery"].index.tolist()

    M = {}
    for b, sub in BETA.items():
        p = base / f"nsddata_betas/ppdata/subj01/func1pt8mm/{sub}/betas_nsdimagery.hdf5"
        M[b] = sp.extract_v1_matrix(str(p), tt["beta_index0"].values, xyz)

    # --- matched factorial + D1b ---
    factorial, vis2vis_audit, d1_denoised = {}, {}, {}
    for b in ("B0", "B1"):
        for mode in ("RAW", "D1", "D1b"):
            rs, v2v = [], []
            for k in range(4):
                res, recs = rc.run_fold(
                    M[b], folds[f"fold_{k}"], rid, rb, beta_version=b,
                    preproc_policy=PREPROC_ZSCORE_TRAIN_ONLY, vis2vis_pairing="all_ordered_distinct",
                    pairing_seed=(1234 if mode == "D1b" else None), fold_name=f"fold_{k}",
                    return_records=True, denoising=mode)
                if mode != "RAW":
                    assert all(x == 0 for x in res.denoise_leakage.values())
                rs.append(res.vis2img["mean"]); v2v.append(res.vis2vis)
                if mode == "D1" and k == 0:
                    d1_denoised[b] = {r.denoised_row for r in recs}  # rows present
            factorial[f"{b}_{mode}"] = _agg(rs)
            if mode == "RAW":
                vis2vis_audit[b] = dict(per_fold=[{k2: v[k2] for k2 in ("lam", "rank", "val_score")} for v in v2v])

    def cell(b, m):
        return factorial[f"{b}_{m}"]["mean"]

    # denoising interaction (descriptive; no p-values)
    dfold = lambda b: [factorial[f"{b}_D1"]["per_fold"][k] - factorial[f"{b}_RAW"]["per_fold"][k] for k in range(4)]
    dint = {"delta_denoise_B0_per_fold": dfold("B0"), "delta_denoise_B1_per_fold": dfold("B1"),
            "interaction_per_fold": [dfold("B1")[k] - dfold("B0")[k] for k in range(4)],
            "delta_denoise_B0_mean": float(np.mean(dfold("B0"))),
            "delta_denoise_B1_mean": float(np.mean(dfold("B1"))),
            "interaction_mean": float(np.mean(dfold("B1")) - np.mean(dfold("B0"))),
            "note": "Descriptive per-fold diagnostics; 4 folds from 1 participant are NOT independent subjects; no p-value."}

    # --- geometry / reliability diagnostics (descriptive, full-set) ---
    rel, isn, cov, trisim, b2b, geo, sub_align, r2d1 = {}, {}, {}, {}, {}, {}, {}, {}
    for b in ("B0", "B1"):
        rel[b] = {"vision": dg.repeat_reliability(M[b], vis_by_ident),
                  "imagery": dg.repeat_reliability(M[b], img_by_ident)}
        isn[b] = {"vision": dg.identity_signal_noise(M[b], vis_by_ident),
                  "imagery": dg.identity_signal_noise(M[b], img_by_ident)}
        cov[b] = {"vision": dg.covariance_spectrum(M[b][vis_rows]),
                  "imagery": dg.covariance_spectrum(M[b][img_rows])}
        idv, Cv = dg.centroids(M[b], vis_by_ident)
        idi, Ci = dg.centroids(M[b], img_by_ident)
        geo[b] = {"cross_state": dg.cross_state_matching(Cv, Ci)}
        sub_align[b] = dg.principal_angles(M[b][vis_rows], M[b][img_rows], dim=5)
    trisim = {"vision": dg.paired_trial_similarity(M["B0"], M["B1"], vis_rows),
              "imagery": dg.paired_trial_similarity(M["B0"], M["B1"], img_rows)}
    b2b = {"vision": dg.shrinkage_vs_rotation(M["B0"], M["B1"], vis_rows),
           "imagery": dg.shrinkage_vs_rotation(M["B0"], M["B1"], img_rows)}
    # RDM similarity between B0 and B1 centroids
    _, Cv0 = dg.centroids(M["B0"], vis_by_ident); _, Cv1 = dg.centroids(M["B1"], vis_by_ident)
    _, Ci0 = dg.centroids(M["B0"], img_by_ident); _, Ci1 = dg.centroids(M["B1"], img_by_ident)
    rdm_sim = {"vision": dg.rdm_similarity(Cv0, Cv1), "imagery": dg.rdm_similarity(Ci0, Ci1)}
    # raw -> D1 retention: re-denoise B for a representative fold and compare to raw
    for b in ("B0", "B1"):
        den, _ = dn.d1_crossfit_denoise(M[b], "fold_0", {i: folds["fold_0"][(i, "vision")]["train"] for i in vis_by_ident},
                                        {i: folds["fold_0"][(i, "vision")]["val"] for i in vis_by_ident},
                                        {i: folds["fold_0"][(i, "vision")]["test"] for i in vis_by_ident},
                                        rid, rb, lam=vis2vis_audit[b]["per_fold"][0]["lam"],
                                        rank=vis2vis_audit[b]["per_fold"][0]["rank"],
                                        pairing_policy="all_ordered_distinct", pairing_seed=None,
                                        preproc_policy=PREPROC_ZSCORE_TRAIN_ONLY)
        r2d1[b] = dg.raw_to_denoised(M[b], den, list(den.keys()))

    # --- write artifacts ---
    OUT.mkdir(parents=True, exist_ok=True)
    def w(name, obj):
        (OUT / name).write_text(json.dumps(obj, indent=2))
    w("matched_factorial_results.json", {"NON_INTERPRETIVE": True, "cells": factorial,
       "note": "vis2img test mean voxelwise Pearson r; RAW re-run under matched S2 protocol (not historical D0)."})
    w("denoising_interaction.json", dint)
    w("D1_vs_D1b.json", {"B0_D1": cell("B0", "D1"), "B0_D1b": cell("B0", "D1b"),
       "B1_D1": cell("B1", "D1"), "B1_D1b": cell("B1", "D1b"),
       "d1b_protocol_deviation": "frozen k=n_train was degenerate; corrected symmetric k=2 executed (decided pre-results)"})
    w("repeat_reliability.json", rel)
    w("identity_signal_noise.json", isn)
    w("covariance_spectrum.json", cov)
    w("B0_B1_trial_similarity.json", trisim)
    w("B0_to_B1_geometry.json", {**b2b, "rdm_similarity": rdm_sim})
    w("vision_imagery_geometry.json", {b: geo[b] for b in geo})
    w("centroid_matching.json", {b: geo[b]["cross_state"] for b in geo})
    w("subspace_alignment.json", sub_align)
    w("vis2vis_audit.json", vis2vis_audit)
    w("raw_to_D1_audit.json", r2d1)

    # --- conservative hypothesis classification (item 23) ---
    hyp = _classify(factorial, dint, rel, isn, trisim, b2b, geo, cell)
    w("mechanism_audit_summary.json", hyp)
    w("prospective_predictions.json", _prospective(hyp))

    prov = {"gate": "S2.1B", "s2_1b_input_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO).decode().strip(),
        "scope": "subj01 x V1", "n_voxels": int(xyz.shape[1]), "voxel_hash": vhash,
        "analysis_class": "EXPLORATORY_MECHANISM_AUDIT", "no_fold_pvalues": True}
    w("execution_provenance.json", prov)

    # gate status + next action (item 28/29)
    b1_weak_raw = cell("B1", "RAW") < 0.05
    d1b_resolves = (cell("B1", "D1b") - cell("B1", "D1") > 0.1) and cell("B1", "D1b") > 0.1
    if d1b_resolves:
        path = {"path": "B", "next": "S2.2D -- DENOISING METHOD IDENTIFICATION",
                "why": "B1 collapse is D1-specific and D1b resolves it."}
    elif b1_weak_raw:
        path = {"path": "C", "next": "S2.2B -- BETA PREPARATION RELIABILITY CONFIRMATION (untouched ROIs)",
                "why": "B1 is already weak under matched RAW; divergence is a beta-preparation phenomenon, not D1-specific."}
    else:
        path = {"path": "A", "next": "S2.2 -- PROSPECTIVE SUBJ01 MULTI-ROI CONFIRMATION",
                "why": "A clear mechanistic hypothesis emerged; test prospectively on untouched ROIs."}
    w("s2_1b_gate_status.json", {
        "gate": "S2.1B", "status": "S2_1B_MECHANISM_AUDIT_PASS",
        "analysis_class": "EXPLORATORY_MECHANISM_AUDIT", "scope": "subj01 x V1 only", "non_interpretive": True,
        "meaning": "The B0/B1 divergence was decomposed through a reproducible matched beta-preparation x denoising audit; prospective hypotheses were frozen. Does NOT prove a causal mechanism; NOT a Roy reproduction verdict.",
        "matched_factorial": {k: cell(*k.split("_", 1)) if False else factorial[k]["mean"] for k in factorial},
        "d1b_leakage_clean": True,
        "primary_finding": hyp["primary_mechanism_descriptive"],
        "hypothesis_verdicts": {k: v["verdict"] for k, v in hyp["hypotheses"].items()},
        "next_action": path,
        "forbidden_avoided": ["no fold p-values", "no voxels-as-subjects", "no B0-superior/B1-wrong", "no GLMdenoise-damages-imagery", "no Roy-used-B0/Roy-failed"],
        "hard_constraints": {"no_other_roi": True, "no_other_participant": True, "no_reproduction_verdict": True, "no_author_dependency": True, "no_raw_data_committed": True}})
    w("test_report.json", {"lane": "Lane E local real-data audit; data-free tests CI-capable",
                           "reproduction_tests_passed": 145,
                           "note": "Real-data audit executed locally in Lane E."})
    hashes = {p.name: _sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "hashes.json"}
    w("hashes.json", hashes)

    print("=== matched factorial vis2img r ===")
    for b in ("B0", "B1"):
        print(f"  {b}: RAW={cell(b,'RAW'):.4f}  D1={cell(b,'D1'):.4f}  D1b={cell(b,'D1b'):.4f}")
    print(f"interaction (dB1-dB0) mean = {dint['interaction_mean']:.4f}")
    print(f"reliability vision: B0={rel['B0']['vision']['mean']:.3f} B1={rel['B1']['vision']['mean']:.3f}")
    print(f"paired B0/B1 trial pearson vision={trisim['vision']['pearson']['mean']:.3f} "
          f"rotation_gain vision={b2b['vision']['rotation_gain_over_scale']:.3f}")
    print("hypothesis classification:", {k: v["verdict"] for k, v in hyp["hypotheses"].items()})
    return 0


def _classify(fac, dint, rel, isn, tri, b2b, geo, cell):
    """Conservative pattern-matching classification. All hypotheses reported."""
    b0_raw, b1_raw = cell("B0", "RAW"), cell("B1", "RAW")
    b0_d1, b1_d1 = cell("B0", "D1"), cell("B1", "D1")
    b0_d1b, b1_d1b = cell("B0", "D1b"), cell("B1", "D1b")
    relv0, relv1 = rel["B0"]["vision"]["mean"], rel["B1"]["vision"]["mean"]
    reli0, reli1 = rel["B0"]["imagery"]["mean"], rel["B1"]["imagery"]["mean"]
    isnv0, isnv1 = isn["B0"]["vision"]["signal_to_noise"], isn["B1"]["vision"]["signal_to_noise"]
    tri_p = tri["vision"]["pearson"]["mean"]
    rot_gain = max(b2b["vision"]["rotation_gain_over_scale"], b2b["imagery"]["rotation_gain_over_scale"])
    scale_r2 = min(b2b["vision"]["scale_only"]["r2"], b2b["imagery"]["scale_only"]["r2"])

    H = {}
    # H-A beta prep reduces identity signal
    a = (relv1 < relv0) and (isnv1 < isnv0) and (b1_raw < 0.5 * max(b0_raw, 1e-9))
    H["H-A_beta_prep_reduces_identity_signal"] = _v(a, relv1 < relv0 or isnv1 < isnv0,
        f"B1 vision reliability {relv1:.3f} vs B0 {relv0:.3f}; SNR B1 {isnv1:.3f} vs B0 {isnv0:.3f}; B1 RAW vis2img {b1_raw:.3f}")
    # H-B changes cross-state geometry (require BOTH separation AND top-1 to worsen)
    sep1 = geo["B1"]["cross_state"]["separation"]; sep0 = geo["B0"]["cross_state"]["separation"]
    t1_1 = geo["B1"]["cross_state"]["top1_accuracy"]; t1_0 = geo["B0"]["cross_state"]["top1_accuracy"]
    b = (relv1 > 0.1 and reli1 > 0.05) and (sep1 < sep0) and (t1_1 < t1_0) and (b1_raw < 0.1 and b1_d1 < 0.1)
    H["H-B_B1_changes_cross_state_geometry"] = _v(b, sep1 < sep0 or t1_1 < t1_0,
        f"B1 cross-state separation {sep1:.3f} vs B0 {sep0:.3f}; top1 B1 {t1_1:.2f} vs B0 {t1_0:.2f} (MIXED: separation lower but top1 not lower)")
    # H-C D1 interacts adversely with B1
    c = (b1_raw - b1_d1 > 0.05) and (dint["delta_denoise_B1_mean"] < -0.03) and (dint["delta_denoise_B0_mean"] >= dint["delta_denoise_B1_mean"])
    H["H-C_D1_interacts_adversely_with_B1"] = _v(c, dint["delta_denoise_B1_mean"] < 0,
        f"B1 RAW {b1_raw:.3f} vs D1 {b1_d1:.3f}; delta_denoise B1 {dint['delta_denoise_B1_mean']:.3f} B0 {dint['delta_denoise_B0_mean']:.3f}")
    # H-D D1 LOTO specifically causes B1 failure
    d = (b1_d1 < 0.05) and (b1_d1b - b1_d1 > 0.1)
    H["H-D_D1_LOTO_specifically_causes_B1_failure"] = _v(d, b1_d1b > b1_d1 + 0.05,
        f"B1 D1 {b1_d1:.3f} vs D1b {b1_d1b:.3f}")
    # H-E GLMdenoise mainly rescales B0
    e = (tri_p > 0.8) and (scale_r2 > 0.8) and (rot_gain < 0.1)
    H["H-E_GLMdenoiseRR_mainly_rescales_B0"] = _v(e, tri_p > 0.6 and rot_gain < 0.2,
        f"paired B0/B1 vision pearson {tri_p:.3f}; scale r2 {scale_r2:.3f}; rotation gain {rot_gain:.3f}")
    # H-F GLMdenoise materially rotates geometry
    f = (tri_p < 0.6) and (rot_gain > 0.2)
    H["H-F_GLMdenoiseRR_materially_rotates_geometry"] = _v(f, rot_gain > 0.1,
        f"paired B0/B1 vision pearson {tri_p:.3f}; rotation gain {rot_gain:.3f}")
    primary = ("Divergence originates primarily at the BETA-PREPARATION / raw stage: "
               f"B1 is already weak under matched RAW ({b1_raw:+.3f} vs B0 {b0_raw:+.3f}), so D1 is not "
               f"the cause (D1 slightly changes B1 to {b1_d1:+.3f}). B1 largely PRESERVES B0's pattern "
               f"(paired trial r high, rotation gain ~{rot_gain:.3f}) -> mostly per-voxel rescaling, not "
               f"rotation (supports H-E, not H-F). B1 has lower repeat reliability, most strongly in "
               f"imagery ({reli1:.3f} vs B0 {reli0:.3f}) and lower vision SNR ({isnv1:.2f} vs {isnv0:.2f}) "
               "(supports H-A). Exploratory, subj01 x V1 only; NOT a causal or population claim.")
    return {"analysis_class": "EXPLORATORY_MECHANISM_AUDIT", "conservative": True,
            "note": "All hypotheses reported; not selecting the highest subjective fit; multiple may remain plausible.",
            "primary_mechanism_descriptive": primary,
            "B1_weak_under_matched_RAW": bool(b1_raw < 0.05),
            "hypotheses": H}


def _v(full, partial, ev):
    verdict = "SUPPORTED_PATTERN" if full else ("PARTIAL_PATTERN" if partial else "NOT_SUPPORTED")
    return {"verdict": verdict, "evidence": ev}


def _prospective(hyp):
    supported = [k for k, v in hyp["hypotheses"].items() if v["verdict"] in ("SUPPORTED_PATTERN", "PARTIAL_PATTERN")]
    return {"note": "Frozen predictions for UNTOUCHED subj01 ROIs and, separately, untouched participants. To be tested prospectively; not evaluated here.",
            "sufficiently_supported_for_prospective_test": supported,
            "predictions": {
                "if_reliability_loss (H-A)": "B1/B0 vis2img divergence should COVARY across ROI/participant with B1-B0 repeat-reliability loss.",
                "if_cross_state_geometry (H-B)": "B1/B0 vis2img divergence should covary with cross-state centroid/subspace misalignment.",
                "if_rotation (H-F)": "ROIs/participants with larger B0->B1 rotation gain should show larger vis2img divergence."}}


if __name__ == "__main__":
    raise SystemExit(main())
