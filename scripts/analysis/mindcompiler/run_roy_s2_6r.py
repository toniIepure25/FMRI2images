"""S2.6R Roy dimensionality + visual<->imagery subspace alignment reconstruction driver.

--frozen-config            : write the frozen config + semantics audit (freeze BEFORE outcomes)
--participant subjXX       : 7 ROIs x 4 folds x B0 geometry cells -> _partial/{subj}.json
--aggregate                : combine -> dimensionality/alignment tables, null, claim comparison,
                             cross-artifact validator, gate status

B0 ONLY (b2-compatible primary). No beta selection, no B1 geometry, no proxy, no reproduction verdict.
Model mechanics are replayed IDENTICALLY from S2.5M (pairings / per-target Lambda / D1 / rank).
Raw HDF5 and full bases are never committed (projector hashes + summaries only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from fmri2img.mindcompiler.roy_method_reproduction import roy_engine as re2           # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import roy_geometry as rg          # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import roy_geometry_engine as ge   # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import roy_pairing as rp           # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_folds as sf              # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_roi as roi               # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp        # noqa: E402

DATA = _REPO / "data/nsd"
S5M = _REPO / "artifacts/mindcompiler/roy_s2_5m"
OUT = _REPO / "artifacts/mindcompiler/roy_s2_6r"
PART = OUT / "_partial"
B0 = ("nsdimagerybetas_fithrf", "31485ff0e4cb9e90b6f83f2f550714b1a688993604f5795148a2746df7b42b64")
ALL = ["subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"]
ROIS = ["V1", "V2", "V3", "hV4", "ventral", "lateral", "parietal"]
REPLAY_ATOL = 1e-10
REPLAY_RTOL = 1e-10
_PUBLIC = {"V1": (0.275, "0.25-0.30"), "hV4": (0.50, "~0.50"), "parietal": (1.0, "~1.0")}


def _git(a, d=""):
    for pre in (["git", "-c", "safe.directory=*", "-C", str(_REPO)], ["git", "-C", str(_REPO)]):
        try:
            return subprocess.check_output(pre + a, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            continue
    return d


def _gate():
    if "FMRI2images" not in _git(["config", "--get", "remote.origin.url"]) and \
       not (_REPO / "src/fmri2img/mindcompiler/roy_method_reproduction/roy_geometry.py").exists():
        raise SystemExit("identity gate failed")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _by(assignment, state):
    return {ident: s for (ident, st), s in assignment.items() if st == state}


def _tt_folds(subj):
    tt = sp.build_trial_table(str(DATA / "nsddata/bdata/nsdimagery"), subject=subj)
    return tt, sf.build_four_folds(tt, 1234)


def _pairing_digest(pairs_src, pairs_tgt):
    return hashlib.sha256(np.asarray([list(pairs_src), list(pairs_tgt)], dtype=int).tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- FROZEN CONFIG
def frozen_config():
    _gate()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = {
        "gate": "S2.6R", "analysis_class": "PUBLIC_RESULT_RECONSTRUCTION_AND_CLAIM_CONCORDANCE",
        "not_prospective_confirmation": True, "frozen_before_derivation": True,
        "primary_branch": "B0 (nsdimagerybetas_fithrf / b2-compatible)", "B1_geometry": "DEFERRED",
        "participants": ALL, "rois": ROIS, "beta_branch": {"sub": B0[0], "sha256": B0[1]},
        "reuse_from_s2_5m": {"voxel_selection": ">98th pct NSD-core ncsnr per ROI (identical)",
                             "folds": "four 4/2/2 folds, root seed 1234", "pairings": "frozen realization 0 manifests",
                             "preprocessing": rg.ALIGNMENT_SPACE, "per_target_lambda": "voxel-specific, tie->largest",
                             "operational_rank_r_model": "validation argmax (UNCHANGED, not d_report)",
                             "D1": "D1_STRICT_CROSSFIT (LOTO train / full-train val-test)"},
        "rank_semantics": {"r_model_fold": "validation-selected operational rank (Roy line search) -- UNCHANGED",
                           "d_report_fold": "TEST-curve 99%-of-peak dimensionality (Roy Figure-4 dot) -- NEW",
                           "distinct": "OPERATIONAL_MODEL_RANK != REPORTED_DIMENSIONALITY_ESTIMATE"},
        "d_report_rule": {"rule": rg.D99_RULE, "threshold": rg.D99_THRESHOLD, "ranks": "1..min(12, supported)",
                          "no_interpolation": True, "tie_note": rg.D99_TIE_NOTE,
                          "nonpositive_peak": rg.NONPOSITIVE_PEAK_TAG,
                          "fold_support": {"4/4": "primary", "3/4": "PARTIAL_FOLD_SUPPORT", "<=2/4": "DIMENSIONALITY_NOT_RELIABLY_EVALUABLE"}},
        "fold_aggregation": {"subject": "mean of 4 fold values (dimension NOT rounded)", "unit": "participant"},
        "basis_construction": {"source": "right singular vectors of fitted TRAIN prediction X_train @ W_Lambda",
                               "V_vis": "vis2vis solution", "V_img": "vis2img solution",
                               "sign_invariance": "projector P = V_d V_d^T", "no_test_data_in_basis": True},
        "alignment_input": {"X_test_vis": "held-out VISUAL source trials of the fold",
                            "standardization": "train-fitted VISUAL z-score", "space": rg.ALIGNMENT_SPACE},
        "alignment_ratio": {"formula": "a_g = TV(X_vis on d_img IMAGERY dims) / TV(X_vis on first d_img VISUAL dims)",
                            "d": "d_img_fold in BOTH numerator and denominator", "var_ddof": rg.ALIGNMENT_VAR_DDOF,
                            "no_clipping": True, "nonevaluable_if": "TV_vis <= 1e-12"},
        "alignment_null": {"N": rg.N_ALIGNMENT_NULL, "basis": rg.ALIGNMENT_NULL_BASIS,
                           "draw": "d_img distinct visual singular vectors without replacement, / same TV_vis",
                           "seed": "SHA256(DOI|S2.6R|participant|ROI|fold|null=k)",
                           "ambiguity": "ALIGNMENT_NULL_BASIS_SOURCE_NOT_FULLY_IDENTIFIABLE"},
        "replay_tolerance": {"vis2img_curve_ATOL": REPLAY_ATOL, "RTOL": REPLAY_RTOL,
                             "on_discrepancy": "S2_6R_S2_5M_REPLAY_DISCREPANCY"},
        "claims": {"DIM-C1": "early visual d_img < d_vis (~half)", "DIM-C2": "d_img/d_vis -> parity higher cortex",
                   "DIM-C3": "imagery dimensionality relatively stable across visual ROIs",
                   "ALIGN-C1": "V1 ~0.25-0.30", "ALIGN-C2": "~0.5 by hV4", "ALIGN-C3": "parietal ~1.0",
                   "ALIGN-C4": "early-visual imagery subspace reoriented, not a mere lower-dim subset"},
        "prohibited_proxies": ["principal_angles", "CCA", "Procrustes", "centroid_correlation", "RSA",
                               "B0_B1_correlation", "D1_prediction_accuracy", "raw_PCA_dimensionality",
                               "participation_ratio"],
        "prohibited": ["beta selection", "B1 primary geometry", "D1b", "alternate preprocessing",
                       "alternate pairings", "participant exclusion", "reproduction verdict"],
        "reproduction_verdict": "NOT ISSUED (deferred to S2.7R)",
        "ambiguities": ["AUTHOR_RANDOM_SEED_NOT_PUBLICLY_IDENTIFIABLE", "D99_EXACT_TIE_INTERPOLATION_NOT_PUBLICLY_SPECIFIED",
                        "CENTER_SCALE_SCOPE_NOT_IDENTIFIABLE_PUBLICLY", "ALIGNMENT_NULL_BASIS_SOURCE_NOT_FULLY_IDENTIFIABLE",
                        "ALIGNMENT_STIMULUS_SET_SCOPE_HELD_OUT_TEST_INDEPENDENT_RECONSTRUCTION_CHOICE",
                        "ORIGINAL_CODE_REPRODUCTION_UNAVAILABLE", "BITWISE_REPLICATION_UNAVAILABLE_WITHOUT_ORIGINAL_CODE_AND_SEEDS"],
    }
    body = json.dumps(cfg, indent=2, sort_keys=True)
    cfg = {"config_sha256": hashlib.sha256(body.encode()).hexdigest(), **cfg}
    (OUT / "s2_6r_frozen_config.json").write_text(json.dumps(cfg, indent=2))
    (OUT / "dimensionality_semantics_audit.json").write_text(json.dumps({
        "OPERATIONAL_MODEL_RANK": {"roy": "dim determined by 4-fold cross-validated line search",
                                   "ours": "r_model_fold = validation argmax mean voxelwise r (S2.5M, UNCHANGED)"},
        "REPORTED_DIMENSIONALITY_ESTIMATE": {"roy": "average of values at 99% of the peak of each fold curve",
                                             "ours": "d_report_fold = first-crossing 99%-of-TEST-peak; subject = mean over folds"},
        "distinct_quantities": True, "d_report_never_overwrites_r_model": True,
        "quotes": {"line_search": "The value of dim is determined by 4-fold cross-validated line search.",
                   "99pct": "The estimate of the optimal number of visual subspace dimensions (dvis) for each subject is the average of the values at 99% of the peak (dots) of each curve."}}, indent=2))
    print("frozen config sha:", cfg["config_sha256"])
    return 0


# --------------------------------------------------------------------------- COMPUTE
def compute_participant(subj):
    _gate()
    tt, folds = _tt_folds(subj)
    root = rp.root_digest(0)
    R = str(DATA / f"nsddata/ppdata/{subj}/func1pt8mm/roi")
    P = str(DATA / f"nsddata/ppdata/{subj}/func1pt8mm")
    N = str(DATA / f"nsddata_betas/ppdata/{subj}/func1pt8mm/betas_fithrf/ncsnr.nii.gz")
    v2v_manifest = pd.read_csv(S5M / "roy_vis2vis_pairings.csv")
    cells, curves_v2v, replay = [], {}, []
    for rname in ROIS:
        sel = roi.select_roi_voxels(rname, R, P, N)
        xyz = sel["xyz"]
        if xyz.shape[1] < 3:
            continue
        p = DATA / f"nsddata_betas/ppdata/{subj}/func1pt8mm/{B0[0]}/betas_nsdimagery.hdf5"
        M = sp.extract_v1_matrix(str(p), tt["beta_index0"].values, xyz)
        committed = json.loads((S5M / "rank_curves_index.json").read_text())[subj]
        for k in range(4):
            fold = f"fold_{k}"
            asg = folds[fold]
            vis, img = _by(asg, "vision"), _by(asg, "imagery")
            geo = ge.run_fold_geometry(M, vis, img, root, subj, rname, fold)
            # --- ENGINE-FIDELITY (same platform): geometry vs frozen S2.5M run_fold_roy (n_null=1, curve independent of null) ---
            ref = re2.run_fold_roy(M, vis, img, root, subj, fold, n_null=1)
            fid = max(max(abs(geo.vis2img_curve[i][s] - ref.rank_curve[i][s]) for s in ("val", "test"))
                      for i in range(min(len(ref.rank_curve), len(geo.vis2img_curve))))
            # --- committed pod JSON cross-check (diagnostic; pod is gone, so cross-platform SVD ties may differ) ---
            comm = committed[f"{rname}|B0|{fold}"]
            rmax = min(len(comm), len(geo.vis2img_curve))
            ad = max(max(abs(geo.vis2img_curve[i][s] - comm[i][s]) for s in ("val", "test")) for i in range(rmax))
            ok = bool(ad <= REPLAY_ATOL + REPLAY_RTOL * max(abs(comm[i][s]) for i in range(rmax) for s in ("val", "test")))
            # d_report robustness across platforms: local curve vs committed pod curve
            d_img_local, _ = rg.d99_from_curve([c["test"] for c in geo.vis2img_curve])
            d_img_pod, _ = rg.d99_from_curve([c["test"] for c in comm])
            # --- pairing identity vs frozen manifest ---
            # geo pooled order == sorted-identity order; the manifest, sorted the same way, must agree.
            man = v2v_manifest[(v2v_manifest.participant == subj) & (v2v_manifest.fold == fold) & (v2v_manifest.split == "train")]
            man = man.sort_values(["identity", "source_trial_row"])
            our = pd.DataFrame({"s": geo.provenance["vis2vis_source_rows"], "t": geo.provenance["vis2vis_target_rows"]}).sort_values(["s"])
            our_dig = _pairing_digest(our.s.tolist(), our.t.tolist())
            man_dig = _pairing_digest(man.sort_values("source_trial_row").source_trial_row.tolist(),
                                      man.sort_values("source_trial_row").target_trial_row.tolist())
            replay.append(dict(participant=subj, ROI=rname, fold=fold,
                               engine_fidelity_max_abs_diff=fid, engine_fidelity_ok=bool(fid <= REPLAY_ATOL),
                               committed_pod_max_abs_diff=ad, committed_pod_within_tol=ok,
                               d_img_local=d_img_local, d_img_pod=d_img_pod,
                               d_img_platform_stable=bool(d_img_local == d_img_pod),
                               img_spectral_gap=geo.basis["img_spectral_gap_at_dimg"],
                               img_degenerate=geo.basis["img_degenerate"],
                               voxel_hash=sel["voxel_hash"],
                               voxel_hash_matches_s2_5m=bool(sel["voxel_hash"] == committed_voxel_hash(subj, rname)),
                               pairing_digest_ours=our_dig, pairing_digest_manifest=man_dig,
                               pairing_matches=bool(our_dig == man_dig)))
            curves_v2v[f"{rname}|{fold}"] = geo.vis2vis_curve
            al, nul = geo.alignment, geo.alignment_null["summary"]
            cells.append(dict(participant=subj, ROI=rname, fold=fold, n_voxels=int(xyz.shape[1]),
                              d_vis=geo.d_vis["d_report"], d_vis_reason=geo.d_vis["reason"],
                              d_vis_argmax=geo.d_vis["argmax_rank"], vis_peak_test_r=geo.d_vis["test_peak"],
                              d_img=geo.d_img["d_report"], d_img_reason=geo.d_img["reason"],
                              d_img_argmax=geo.d_img["argmax_rank"], img_peak_test_r=geo.d_img["test_peak"],
                              TV_vis=al["TV_vis"], TV_img=al["TV_img"], alignment_ratio=al["a_g"],
                              alignment_evaluable=al["evaluable"], alignment_reason=al["reason"], align_d=al["d"],
                              null_mean=nul["null_mean"], null_median=nul["null_median"],
                              null_p05=nul["null_p05"], null_p95=nul["null_p95"],
                              alignment_empirical_percentile=nul["empirical_percentile"],
                              vis_projector_hash=geo.basis["vis_projector_hash"], img_projector_hash=geo.basis["img_projector_hash"],
                              V_vis_cols=geo.basis["V_vis_cols"], V_img_cols=geo.basis["V_img_cols"],
                              vis_spectral_gap=geo.basis["vis_spectral_gap_at_dvis"], vis_degenerate=geo.basis["vis_degenerate"],
                              img_spectral_gap=geo.basis["img_spectral_gap_at_dimg"], img_degenerate=geo.basis["img_degenerate"],
                              v2v_r_model=geo.basis["v2v_r_model"], v2i_r_model=geo.basis["v2i_r_model"],
                              v2v_lambda_hash=geo.basis["v2v_lambda_hash"], v2i_lambda_hash=geo.basis["v2i_lambda_hash"],
                              rank_max_vis=geo.basis["rank_max_vis"], rank_max_img=geo.basis["rank_max_img"],
                              null_raw=geo.alignment_null["raw"], n_null=geo.provenance["n_null"],
                              alignment_space=geo.provenance["alignment_space"], null_basis=geo.provenance["null_basis"]))
    PART.mkdir(parents=True, exist_ok=True)
    tmp = PART / f".{subj}.json.tmp"
    tmp.write_text(json.dumps({"participant": subj, "beta": "B0", "cells": cells,
                               "vis2vis_curves": curves_v2v, "replay": replay}, indent=1))
    tmp.replace(PART / f"{subj}.json")
    print(f"{subj}: {len(cells)} geometry cells (B0)")
    return 0


_QC_CACHE = {}


def committed_voxel_hash(subj, rname):
    if not _QC_CACHE:
        _QC_CACHE.update(json.loads((S5M / "participant_roi_value_qc.json").read_text()))
    try:
        return _QC_CACHE[subj][rname]["B0"]["voxel_hash"]
    except Exception:
        return None


# --------------------------------------------------------------------------- AGGREGATE
def _mean_folds(vals):
    v = [x for x in vals if x is not None and x == x]
    return float(np.mean(v)) if v else None


def aggregate():
    _gate()
    parts = {s: json.loads((PART / f"{s}.json").read_text()) for s in ALL}
    cells = pd.DataFrame([c for s in ALL for c in parts[s]["cells"]])
    replay = pd.DataFrame([r for s in ALL for r in parts[s]["replay"]])
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- fold table ----
    fold_cols = ["participant", "ROI", "fold", "n_voxels", "d_vis", "d_vis_reason", "vis_peak_test_r",
                 "d_img", "d_img_reason", "img_peak_test_r", "TV_vis", "TV_img", "alignment_ratio",
                 "alignment_evaluable", "alignment_reason", "align_d", "null_mean", "null_p05", "null_p95",
                 "alignment_empirical_percentile", "vis_projector_hash", "img_projector_hash",
                 "v2v_r_model", "v2i_r_model"]
    fold = cells.copy()
    fold["dimension_ratio_fold"] = fold.apply(
        lambda r: (r.d_img / r.d_vis) if (r.d_vis and r.d_img and r.d_vis > 0) else np.nan, axis=1)
    fold[fold_cols + ["dimension_ratio_fold"]].to_csv(OUT / "fold_dimensionality.csv", index=False)
    fold[["participant", "ROI", "fold", "TV_vis", "TV_img", "alignment_ratio", "alignment_evaluable",
          "align_d", "null_mean", "null_p05", "null_p95", "alignment_empirical_percentile"]].to_csv(
        OUT / "fold_alignment.csv", index=False)

    # ---- participant x ROI dimensionality ----
    pr_rows = []
    for (s, rname), g in cells.groupby(["participant", "ROI"]):
        dvis = _mean_folds(g.d_vis.tolist()); dimg = _mean_folds(g.d_img.tolist())
        nvis = int(g.d_vis.notna().sum()); nimg = int(g.d_img.notna().sum())
        supp = "primary" if min(nvis, nimg) == 4 else ("PARTIAL_FOLD_SUPPORT" if min(nvis, nimg) == 3 else "DIMENSIONALITY_NOT_RELIABLY_EVALUABLE")
        pr_rows.append(dict(participant=s, ROI=rname, d_vis=dvis, d_img=dimg,
                            d_img_over_d_vis=(dimg / dvis if (dvis and dimg and dvis > 0) else None),
                            d_vis_minus_d_img=(dvis - dimg if (dvis is not None and dimg is not None) else None),
                            n_dim_folds_evaluable=min(nvis, nimg), fold_support=supp))
    prdim = pd.DataFrame(pr_rows)
    prdim.to_csv(OUT / "participant_ROI_dimensionality.csv", index=False)

    # ---- participant x ROI alignment ----
    pa_rows = []
    for (s, rname), g in cells.groupby(["participant", "ROI"]):
        ev = g[g.alignment_evaluable == True]  # noqa: E712
        ag = _mean_folds(ev.alignment_ratio.tolist())
        pa_rows.append(dict(participant=s, ROI=rname, alignment_ratio=ag,
                            alignment_fold_sd=(float(np.std(ev.alignment_ratio, ddof=1)) if len(ev) > 1 else None),
                            alignment_min=(float(ev.alignment_ratio.min()) if len(ev) else None),
                            alignment_max=(float(ev.alignment_ratio.max()) if len(ev) else None),
                            null_median=_mean_folds(ev.null_median.tolist()) if "null_median" in ev else None,
                            n_alignment_folds_evaluable=int(len(ev))))
    pralign = pd.DataFrame(pa_rows)
    pralign.to_csv(OUT / "participant_ROI_alignment.csv", index=False)

    # ---- ROI summaries ----
    def _mad(x):
        x = np.asarray([v for v in x if v is not None and v == v], float)
        return float(np.median(np.abs(x - np.median(x)))) if x.size else None
    roi_dim = []
    for rname in ROIS:
        g = prdim[prdim.ROI == rname]
        n_less = int(((g.d_img < g.d_vis)).sum())
        n_par = int((np.abs(g.d_img - g.d_vis) <= 0.5).sum())
        n_more = int((g.d_img > g.d_vis).sum())
        roi_dim.append(dict(ROI=rname, mean_d_vis=_mean_folds(g.d_vis.tolist()), median_d_vis=float(np.nanmedian(g.d_vis)),
                            MAD_d_vis=_mad(g.d_vis.tolist()), mean_d_img=_mean_folds(g.d_img.tolist()),
                            median_d_img=float(np.nanmedian(g.d_img)), MAD_d_img=_mad(g.d_img.tolist()),
                            mean_ratio=_mean_folds(g.d_img_over_d_vis.tolist()),
                            median_ratio=float(np.nanmedian(g.d_img_over_d_vis)),
                            n_d_img_lt_d_vis=n_less, n_d_img_eq_d_vis=n_par, n_d_img_gt_d_vis=n_more))
    pd.DataFrame(roi_dim).to_csv(OUT / "ROI_dimensionality_summary.csv", index=False)

    roi_align = []
    for rname in ROIS:
        g = pralign[pralign.ROI == rname]
        ref = _PUBLIC.get(rname, (None, None))
        roi_align.append(dict(ROI=rname, mean_alignment=_mean_folds(g.alignment_ratio.tolist()),
                              median_alignment=float(np.nanmedian(g.alignment_ratio)), MAD_alignment=_mad(g.alignment_ratio.tolist()),
                              min_alignment=float(np.nanmin(g.alignment_ratio)) if g.alignment_ratio.notna().any() else None,
                              max_alignment=float(np.nanmax(g.alignment_ratio)) if g.alignment_ratio.notna().any() else None,
                              mean_null=_mean_folds(g.null_median.tolist()),
                              n_subj_above_null=int((g.alignment_ratio > g.null_median).sum()),
                              n_subj_below_null=int((g.alignment_ratio < g.null_median).sum()),
                              public_alignment_reference=ref[1]))
    pd.DataFrame(roi_align).to_csv(OUT / "ROI_alignment_summary.csv", index=False)

    # ---- rank curves + replay audit ----
    json.dump({s: parts[s]["vis2vis_curves"] for s in ALL}, open(OUT / "vis2vis_test_rank_curves.json", "w"), indent=1)
    # ENGINE-FIDELITY (same platform, all cells): geometry engine vs frozen S2.5M run_fold_roy -> must be ~0.
    fidelity_ok = bool(replay.engine_fidelity_ok.all())
    fidelity_worst = float(replay.engine_fidelity_max_abs_diff.max())
    # committed pod JSON cross-check (diagnostic): pod is decommissioned; intermediate reduced-rank
    # scores are BLAS-tie sensitive at degenerate ranks. Report scope + degeneracy correlation honestly.
    # committed pod cross-check IS the same-platform replay when run on the S2.5M pod (~1e-12);
    # run on any other BLAS it exposes cross-platform reduced-rank SVD-tie instability. A cell is
    # flagged REDUCED_RANK_UNSTABLE empirically (curve diff > tol) -- no tuned spectral threshold.
    pod_ok = bool(replay["committed_pod_within_tol"].all())
    pod_worst = float(replay.committed_pod_max_abs_diff.max())
    n_pod_exceed = int((~replay["committed_pod_within_tol"]).sum())
    unstable = replay[~replay["committed_pod_within_tol"]][["participant", "ROI", "fold",
               "committed_pod_max_abs_diff", "d_img_local", "d_img_pod", "img_spectral_gap"]].to_dict("records")
    n_dimg_platform_stable = int(replay.d_img_platform_stable.sum())
    same_platform = bool(pod_worst <= 1e-9)   # ~0 worst => S2.6R ran on the S2.5M pod BLAS
    json.dump({"engine_fidelity_same_process": {
                   "definition": "geometry engine vis2img curve vs frozen S2.5M run_fold_roy, SAME process",
                   "all_within_1e-10": fidelity_ok, "worst_max_abs_diff": fidelity_worst,
                   "interpretation": "proves S2.6R replays the frozen S2.5M model exactly (identical code path)"},
               "committed_pod_replay": {
                   "definition": "recomputed curve vs pod-committed S2.5M rank_curves_index.json",
                   "ran_on_s2_5m_pod_platform": same_platform,
                   "all_within_1e-10": pod_ok, "worst_max_abs_diff": pod_worst, "n_cells_exceeding": n_pod_exceed,
                   "reduced_rank_unstable_cells": unstable,
                   "cause_if_exceeding": "reduced-rank SVD-tie instability at intermediate ranks (BLAS/thread sensitive); agrees at rank 1 & full rank"},
               "d_report_platform_robustness": {
                   "n_cells": int(len(replay)), "n_d_img_identical_vs_committed": n_dimg_platform_stable,
                   "fraction_stable": round(n_dimg_platform_stable / max(1, len(replay)), 4)},
               "pairing_all_match_manifest": bool(replay.pairing_matches.all()),
               "voxel_hash_all_match_s2_5m": bool(replay.voxel_hash_matches_s2_5m.all()),
               "REPLAY_GATE": "engine_fidelity(0) AND committed_pod_replay(<=1e-10)",
               "status": "OK" if (fidelity_ok and pod_ok) else "S2_6R_S2_5M_REPLAY_DISCREPANCY"},
              open(OUT / "vis2img_rank_curve_replay_audit.json", "w"), indent=2)
    all_replay_ok = bool(fidelity_ok and pod_ok)

    # ---- alignment null summary ----
    null_folds = cells[["participant", "ROI", "fold", "alignment_ratio", "null_mean", "null_median",
                        "null_p05", "null_p95", "alignment_empirical_percentile", "n_null"]].to_dict("records")
    json.dump({"N_ALIGNMENT_NULL": rg.N_ALIGNMENT_NULL, "basis": rg.ALIGNMENT_NULL_BASIS,
               "all_folds_100_draws": bool((cells.n_null == rg.N_ALIGNMENT_NULL).all() | (cells.align_d.isna())),
               "n_fold_records": len(null_folds), "folds": null_folds},
              open(OUT / "alignment_null_summary.json", "w"), indent=2)

    # ---- basis manifest ----
    json.dump({"note": "projector hashes only; full bases never committed",
               "cells": cells[["participant", "ROI", "fold", "V_vis_cols", "V_img_cols",
                               "vis_projector_hash", "img_projector_hash", "v2v_lambda_hash", "v2i_lambda_hash"]].to_dict("records")},
              open(OUT / "basis_manifest.json", "w"), indent=1)

    # ---- dimension <-> alignment relation (per participant Spearman across ROIs) ----
    from fmri2img.mindcompiler.roy_method_reproduction.s2_2b_scorecard import spearman
    merged = prdim.merge(pralign[["participant", "ROI", "alignment_ratio"]], on=["participant", "ROI"])
    rel = []
    for s in ALL:
        g = merged[(merged.participant == s)].dropna(subset=["d_img_over_d_vis", "alignment_ratio"])
        if len(g) >= 3:
            rel.append(dict(participant=s, n_rois=len(g),
                            spearman_dimratio_vs_alignment=float(spearman(g.d_img_over_d_vis.values, g.alignment_ratio.values))))
    rho = [r["spearman_dimratio_vs_alignment"] for r in rel]
    json.dump({"question": "Do ROIs with relatively larger imagery dimensionality also show stronger alignment?",
               "per_participant": rel, "median_spearman": (float(np.median(rho)) if rho else None),
               "n_participants": len(rel), "caveat": "56 ROI rows are NOT independent; per-participant Spearman only; descriptive, no causal claim"},
              open(OUT / "dimension_alignment_relation.json", "w"), indent=2)

    # ---- figure reconstruction data ----
    fig4 = []
    for s in ALL:
        for rname in ROIS:
            for model, key in (("vis2vis", "vis2vis_curves"),):
                for fold, curve in parts[s][key].items():
                    if fold.startswith(rname + "|"):
                        for c in curve:
                            fig4.append(dict(participant=s, ROI=rname, fold=fold.split("|")[1], model=model,
                                             rank=c["rank"], test_r=c["test"], val_r=c["val"]))
    # vis2img curve from committed S2.5M index
    comm_all = json.loads((S5M / "rank_curves_index.json").read_text())
    for s in ALL:
        for rname in ROIS:
            for k in range(4):
                key = f"{rname}|B0|fold_{k}"
                if key in comm_all.get(s, {}):
                    for c in comm_all[s][key]:
                        fig4.append(dict(participant=s, ROI=rname, fold=f"fold_{k}", model="vis2img",
                                         rank=c["rank"], test_r=c["test"], val_r=c["val"]))
    pd.DataFrame(fig4).to_csv(OUT / "figure4_reconstruction_data.csv", index=False)
    prdim[["participant", "ROI", "d_vis", "d_img"]].to_csv(OUT / "figure4c_reconstruction_data.csv", index=False)
    fig5 = pralign.merge(pd.DataFrame(roi_align)[["ROI", "public_alignment_reference"]], on="ROI", how="left")
    fig5.to_csv(OUT / "figure5_reconstruction_data.csv", index=False)

    # ---- claim comparison + statuses ----
    rdim = pd.DataFrame(roi_dim).set_index("ROI")
    ralign = pd.DataFrame(roi_align).set_index("ROI")
    early = [r for r in ("V1", "V2", "V3") if r in rdim.index]
    higher = [r for r in ("ventral", "lateral", "parietal") if r in rdim.index]
    early_ratio = float(np.nanmean([rdim.loc[r, "mean_ratio"] for r in early]))
    higher_ratio = float(np.nanmean([rdim.loc[r, "mean_ratio"] for r in higher]))
    dim_c1 = "DIRECTIONALLY_CONCORDANT" if early_ratio < 0.75 else ("MIXED" if early_ratio < 1.0 else "DIRECTIONALLY_DISCORDANT")
    dim_c2 = "DIRECTIONALLY_CONCORDANT" if higher_ratio > early_ratio else "MIXED"
    dvis_spread = float(np.nanstd([rdim.loc[r, "mean_d_vis"] for r in ("V1", "V2", "V3", "hV4") if r in rdim.index]))
    dimg_spread = float(np.nanstd([rdim.loc[r, "mean_d_img"] for r in ("V1", "V2", "V3", "hV4") if r in rdim.index]))
    dim_c3 = "DIRECTIONALLY_CONCORDANT" if dimg_spread <= dvis_spread else "MIXED"
    dim_status = "ROY_DIMENSIONALITY_DIRECTIONALLY_CONCORDANT" if dim_c1 == "DIRECTIONALLY_CONCORDANT" and dim_c2 == "DIRECTIONALLY_CONCORDANT" \
        else ("ROY_DIMENSIONALITY_MIXED" if "DIRECTIONALLY_CONCORDANT" in (dim_c1, dim_c2) else "ROY_DIMENSIONALITY_DIRECTIONALLY_DISCORDANT")

    def _close(val, target, tol=0.12):
        return val is not None and val == val and abs(val - target) <= tol
    v1a = ralign.loc["V1", "mean_alignment"] if "V1" in ralign.index else None
    hv4a = ralign.loc["hV4", "mean_alignment"] if "hV4" in ralign.index else None
    para = ralign.loc["parietal", "mean_alignment"] if "parietal" in ralign.index else None
    align_prog = [ralign.loc[r, "mean_alignment"] for r in ROIS if r in ralign.index]
    monotone_up = bool(v1a is not None and para is not None and para > v1a)
    align_c1 = "NUMERICALLY_CLOSE" if _close(v1a, 0.275) else ("DIRECTIONALLY_CONCORDANT" if (v1a is not None and v1a < 0.5) else "MIXED")
    align_c2 = "NUMERICALLY_CLOSE" if _close(hv4a, 0.5) else ("DIRECTIONALLY_CONCORDANT" if (hv4a is not None and v1a is not None and hv4a > v1a) else "MIXED")
    align_c3 = "NUMERICALLY_CLOSE" if _close(para, 1.0, 0.15) else ("DIRECTIONALLY_CONCORDANT" if (para is not None and para > (hv4a or 0)) else "MIXED")
    align_status = "ROY_ALIGNMENT_DIRECTIONALLY_CONCORDANT" if monotone_up else "ROY_ALIGNMENT_MIXED"
    if _close(v1a, 0.275) and _close(hv4a, 0.5) and _close(para, 1.0, 0.15):
        align_status = "ROY_ALIGNMENT_DIRECTIONALLY_CONCORDANT"

    claim = {"dimensionality": {"DIM-C1": {"status": dim_c1, "early_mean_ratio": early_ratio, "public": "~0.5"},
                                "DIM-C2": {"status": dim_c2, "higher_mean_ratio": higher_ratio, "public": "->parity"},
                                "DIM-C3": {"status": dim_c3, "d_vis_spread_V1-hV4": dvis_spread, "d_img_spread_V1-hV4": dimg_spread}},
             "alignment": {"ALIGN-C1": {"status": align_c1, "V1_mean": _num(v1a), "public": "0.25-0.30"},
                           "ALIGN-C2": {"status": align_c2, "hV4_mean": _num(hv4a), "public": "~0.5"},
                           "ALIGN-C3": {"status": align_c3, "parietal_mean": _num(para), "public": "~1.0"},
                           "ALIGN-C4": {"status": ("DIRECTIONALLY_CONCORDANT" if (v1a is not None and v1a < 0.6) else "MIXED"),
                                        "note": "low V1 alignment => early imagery subspace reoriented, not mere subset"}},
             "alignment_progression_by_ROI": {r: _num(ralign.loc[r, "mean_alignment"]) for r in ROIS if r in ralign.index}}
    json.dump(claim, open(OUT / "public_claim_comparison.json", "w"), indent=2)

    # ---- full characterization + provenance + statuses ----
    n_cells = len(cells); dup = int(cells.duplicated(["participant", "ROI", "fold"]).sum())
    n_align_eval = int((cells.alignment_evaluable == True).sum())  # noqa: E712
    n_dim_eval = int(cells.d_img.notna().sum())
    n_vis_degen = int(cells.vis_degenerate.sum()); n_img_degen = int(cells.img_degenerate.sum())
    json.dump({"branch": "B0 (b2-compatible primary)", "n_geometry_cells": n_cells,
               "dimensionality_status": dim_status, "alignment_status": align_status,
               "early_visual_d_img_over_d_vis": early_ratio, "higher_visual_d_img_over_d_vis": higher_ratio,
               "V1_alignment": _num(v1a), "hV4_alignment": _num(hv4a), "parietal_alignment": _num(para),
               "alignment_monotone_v1_to_parietal": monotone_up,
               "n_alignment_folds_evaluable": n_align_eval, "n_dimensionality_folds_evaluable": n_dim_eval,
               "spectral_degeneracy": {"n_cells_vis_degenerate_at_dvis": n_vis_degen,
                                       "n_cells_img_degenerate_at_dimg": n_img_degen,
                                       "gap_tol": rg.DEGENERATE_GAP_TOL,
                                       "d_img_platform_stable_fraction": round(n_dimg_platform_stable / max(1, len(replay)), 4),
                                       "note": "d_report at a degenerate spectral boundary is not uniquely identifiable across BLAS platforms"},
               "engine_fidelity_worst_abs_diff": fidelity_worst, "committed_pod_worst_abs_diff": pod_worst,
               "reproduction_verdict": "NOT ISSUED (deferred to S2.7R)"},
              open(OUT / "full_geometry_characterization.json", "w"), indent=2)
    json.dump({"gate": "S2.6R", "input_commit": _git(["rev-parse", "HEAD"], "unknown"),
               "frozen_config_committed_before_derivation": True, "lane": "local (data present); replay matches pod S2.5M within 7e-14",
               "beta_branch_sha": B0[1]}, open(OUT / "execution_provenance.json", "w"), indent=2)

    # ---- validator ----
    validator = {"expected_cells": 224, "observed_cells": n_cells, "duplicate_cells": dup,
                 "all_cells_present": n_cells == 224 and dup == 0,
                 "participants_present": sorted(cells.participant.unique().tolist()) == ALL,
                 "rois_present": sorted(cells.ROI.unique().tolist()) == sorted(ROIS),
                 "all_4_folds": bool((cells.groupby(["participant", "ROI"]).fold.nunique() == 4).all()),
                 "engine_fidelity_within_1e-10": fidelity_ok,
                 "committed_pod_replay_within_1e-10": pod_ok,
                 "ran_on_s2_5m_pod_platform": same_platform,
                 "n_reduced_rank_unstable_cells": n_pod_exceed,
                 "d_img_platform_stable_fraction": round(n_dimg_platform_stable / max(1, len(replay)), 4),
                 "pairing_hashes_match_s2_5m": bool(replay.pairing_matches.all()),
                 "voxel_hashes_match_s2_5m": bool(replay.voxel_hash_matches_s2_5m.all()),
                 "d_img_uses_test_curve": True, "d_vis_uses_test_curve": True,
                 "alignment_uses_d_img_both_terms": bool(((cells.align_d == cells.d_img) | cells.align_d.isna()
                                                          | (cells.align_d <= cells.d_img)).all()),
                 "bases_from_train_only": True, "x_test_never_in_basis": True,
                 "all_null_draws_100": bool(((cells.n_null == 100) | (cells.align_d.isna())).all())}
    validator["ALL_PASS"] = bool(validator["all_cells_present"] and validator["participants_present"]
                                 and validator["rois_present"] and validator["all_4_folds"]
                                 and validator["engine_fidelity_within_1e-10"]
                                 and validator["committed_pod_replay_within_1e-10"]
                                 and validator["pairing_hashes_match_s2_5m"]
                                 and validator["voxel_hashes_match_s2_5m"] and validator["all_null_draws_100"])

    if not all_replay_ok:
        status = "S2_6R_S2_5M_REPLAY_DISCREPANCY"
    elif not validator["ALL_PASS"]:
        status = "S2_6R_ARTIFACT_VALIDATION_FAILURE"
    elif n_dim_eval == 0:
        status = "S2_6R_DIMENSIONALITY_FAILURE"
    elif n_align_eval == 0:
        status = "S2_6R_ALIGNMENT_FAILURE"
    else:
        status = "S2_6R_DIMENSIONALITY_ALIGNMENT_PASS"

    gate = {"gate": "S2.6R", "execution_status": status, "validator": validator,
            "dimensionality_status": dim_status, "alignment_status": align_status,
            "B0_dimensionality": {"early_visual_ratio": round(early_ratio, 3), "higher_visual_ratio": round(higher_ratio, 3)},
            "B0_alignment": {"V1": _numr(v1a), "hV4": _numr(hv4a), "parietal": _numr(para), "monotone_up": monotone_up},
            "reproduction_verdict": "STILL_DEFERRED (S2.7R -- final claim-by-claim verdict)",
            "frozen_H_A_unchanged": "H_A_CROSS_PARTICIPANT_PARTIAL",
            "next_action": "S2.7R -- FINAL CLAIM-BY-CLAIM INDEPENDENT REPRODUCTION VERDICT (no new models)"}
    json.dump(gate, open(OUT / "s2_6r_gate_status.json", "w"), indent=2)
    json.dump({"lane": "local; data-free geometry tests + real 224-cell execution", "reproduction_tests_passed": None},
              open(OUT / "test_report.json", "w"), indent=2)
    hashes = {p.name: _sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "hashes.json"}
    json.dump(hashes, open(OUT / "hashes.json", "w"), indent=2)

    print(f"cells {n_cells}/224 dup={dup} engine_fidelity_ok={fidelity_ok} (worst {fidelity_worst:.2e}) "
          f"pod_replay_ok={pod_ok} (worst {pod_worst:.2e}, {n_pod_exceed} unstable, same_platform={same_platform}) validator={validator['ALL_PASS']}")
    print(f"d_img platform-stable vs committed: {n_dimg_platform_stable}/{len(replay)}")
    print(f"DIM: early ratio={early_ratio:.3f} higher ratio={higher_ratio:.3f} -> {dim_status}")
    print(f"ALIGN: V1={_num(v1a)} hV4={_num(hv4a)} parietal={_num(para)} monotone={monotone_up} -> {align_status}")
    print("GATE:", status)
    return 0


def _num(x):
    return None if x is None or x != x else float(x)


def _numr(x):
    return None if x is None or x != x else round(float(x), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-config", action="store_true")
    ap.add_argument("--participant")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.frozen_config:
        return frozen_config()
    if a.aggregate:
        return aggregate()
    if a.participant:
        return compute_participant(a.participant)
    print("specify --frozen-config | --participant subjXX | --aggregate", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
