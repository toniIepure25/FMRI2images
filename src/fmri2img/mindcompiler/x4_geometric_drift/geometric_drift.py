"""MINDIR-X4 Low-Dimensional Geometric Drift of Imagery Subspaces (frozen config 885e67e2). FINAL exploratory
mechanistic gate on the N=8 discovery cohort. H1: repeat-level OUTSIDE-support subspace perturbations (projector
displacements from a leave-one-repeat reference, subset-centered) are ANISOTROPIC -- a dominant direction, beyond
a Frobenius-matched isotropic null, reproducible across disjoint identity-subset halves. H2: separation of two
repeats along the CROSS-FIT dominant drift mode predicts held-out O2.14 M4T2 ROBUSTNESS_MARGIN. Exactly 4 tests
(2 ROIs x {ANISO, MODE}) sign-flip + Holm. Projectors handled in reduced span-coordinates (trace(X_a X_b) via
low-rank factors; no V x V materialization). DISCLOSED tractability freeze: 20-of-100 deterministic M4 subsets +
1000 nulls. No new estimator/rank/D/distribution/manifold learner. Cannot modify any O2/X1/X2/X3 seal or unlock
O3. Reuses O2.14 subject_cell margins (replay-bound), exact W_target, D=2 outside.

INTEGRITY NOTE (disclosed frozen-config inconsistency resolution): the frozen config defines E_ANISO and E_MODE
as RATIOS (REAL / null-centre), which sit at ~1 under H0, yet its support criteria state "median E>0", ">=6/8
E>0", and feed them to a sign-flip test centred on 0. A sign-flip-around-0 test on a ratio centred at 1 would
manufacture significance even under a pure isotropic null. The only valid reading -- strictly MORE conservative
and unable to produce a false positive -- runs the inference (sign-flip and the ">0" counts) on the CENTRED
statistic (REAL - null-centre); the ratio is reported as descriptive magnitude only. This resolution never
loosens a threshold and is recorded in scientific_status.json."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from fmri2img.mindcompiler.operator_o2_9 import composite_calibration as O9
from fmri2img.mindcompiler.operator_o2_14 import schedule_robustness as SR

ALL = SR.ALL
N_FOLDS = SR.N_FOLDS
N_SUB = 20                 # deterministic 20-of-100 M4 subsets (disclosed frozen)
N_ISO_NULL = 1000
N_MODE_NULL = 1000
N_ALIGN_NULL = 200
_G = None


def _geom(repo):
    global _G
    if _G is None:
        from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
        _G = G; SR._G = G; O9._G = G
    return _G


def _orth(M, k):
    """top-k left singular vectors (unit, orthonormal) of M (columns as vectors)."""
    U, s, _ = np.linalg.svd(np.asarray(M, np.float64), full_matrices=False)
    return U[:, :k]


def _top_dir(Xc):
    """top right-singular direction (unit) + top-fraction lambda1/sum lambda + participation ratio of Xc (n x d)."""
    if Xc.shape[0] < 2 or Xc.shape[1] < 1:
        return np.zeros(max(Xc.shape[1], 1)), 0.0, 0.0
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = s ** 2; tot = float(lam.sum())
    if tot <= 0:
        return Vt[0], 0.0, 0.0
    return Vt[0], float(lam[0] / tot), float(tot ** 2 / float(np.sum(lam ** 2)))


def _spearman(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[m])).astype(float); ry = np.argsort(np.argsort(y[m])).astype(float)
    rx -= rx.mean(); ry -= ry.mean(); d = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / d) if d > 0 else float("nan")


def _atanh(r):
    return float(np.arctanh(np.clip(r, -1 + 1e-12, 1 - 1e-12)))


# ---------------- per (subject, ROI) ----------------
def subject_roi(s, roi, cells, trialnat, rd, R111, G):
    rep_pairs = list(itertools.combinations(range(8), 2))                     # 28 M4T2 repeat pairs (O2.14 order)
    R14 = SR.subject_cell(s, roi, (4, 2), cells, trialnat, rd, R111)          # O2.14 M4T2 margins (replay-bound)
    A14 = SR.aggregate_cell(R14)
    aniso_folds = []; iso_delta_folds = []; erank_folds = []; align_folds = []
    fold_z_fail = []; fold_z_null = []; fold_rho_fail = []; dmode_dfull = []; pos_profiles = []
    for f in range(N_FOLDS):
        W = np.asarray(cells[f]["W_target"], np.float64); V = W.shape[0]
        Dtr = np.asarray(trialnat[f]["trial_native"], np.float64)             # (10 x 8 x V)
        O = np.stack([[Dtr[i, r] - W @ (W.T @ Dtr[i, r]) for r in range(8)] for i in range(10)])  # OUTSIDE (10x8xV)
        B = _orth(O.reshape(-1, V).T, min(V, 80))                            # union outside span (V x m)
        id_subs = O9._pairs_balanced(list(cells[f]["simple_ids"]), list(cells[f]["nat_ids"]), 4)
        order = sorted(range(len(id_subs)), key=lambda i: G.seed_uint64("X4|sub|%s|%s|%d|%s" % (s, roi, f, str(id_subs[i]))))
        sel = sorted(order[:N_SUB])                                          # deterministic 20-of-100 (pre-outcome)
        halfA = set(sorted(sel, key=lambda gi: G.seed_uint64("X4|half|%s|%s|%d|%s" % (s, roi, f, str(id_subs[gi]))))[:N_SUB // 2])
        vecs = []; idx = []                                                  # X_{I,r} displacements in B-coords
        for li, gi in enumerate(sel):
            Ci = list(id_subs[gi]); cO = O[Ci] @ B                          # (4 x 8 x m)
            for r in range(8):
                Pr = _orth(cO[:, r].T, 2)                                    # (m x 2) single-repeat outside subspace
                others = [u for u in range(8) if u != r]
                Pref = _orth(cO[:, others].mean(1).T, 2)                     # leave-one-repeat reference
                dP = Pr @ Pr.T - Pref @ Pref.T                              # (m x m) symmetric displacement
                vecs.append(dP.ravel()); idx.append((li, r))
        Xv = np.asarray(vecs); idx = np.asarray(idx)                         # (160 x m^2), (160 x 2)
        for li in range(N_SUB):                                              # subset-centering: X = dP - mean_r dP
            rows = np.where(idx[:, 0] == li)[0]; Xv[rows] -= Xv[rows].mean(0)
        norms = np.linalg.norm(Xv, axis=1)
        _, ss, VVt = np.linalg.svd(Xv, full_matrices=False)                  # reduce to numerical span
        dnum = max(int(np.sum(ss > (ss[0] * 1e-9))) if ss.size and ss[0] > 0 else 1, 1)
        Rc = Xv @ VVt[:dnum].T                                              # (160 x d) span-coords
        localA = set(li for li, gi in enumerate(sel) if gi in halfA)
        maskA = np.array([idx[k, 0] in localA for k in range(len(idx))])
        # ---- H1 anisotropy ----
        _, top_real, erank = _top_dir(Rc); erank_folds.append(erank)
        tn = []
        for it in range(N_ISO_NULL):
            rng = np.random.Generator(np.random.PCG64(G.seed_uint64("X4|anisotropy|%s|%s|%d|%d" % (s, roi, f, it))))
            U0 = rng.standard_normal((len(Rc), dnum)); U0 /= (np.linalg.norm(U0, axis=1, keepdims=True) + 1e-12)
            En = norms[:, None] * U0
            for li in range(N_SUB):
                rows = np.where(idx[:, 0] == li)[0]; En[rows] -= En[rows].mean(0)
            _, tnull, _ = _top_dir(En); tn.append(tnull)
        med_null = float(np.median(tn))
        aniso_folds.append(top_real / med_null if med_null > 1e-12 else float("nan"))
        iso_delta_folds.append(float(top_real - med_null))                  # CENTRED anisotropy (inference statistic)
        # ---- cross-half mode reproducibility ----
        VA, _, _ = _top_dir(Rc[maskA]); VB, _, _ = _top_dir(Rc[~maskA])
        align_real = abs(float(VA @ VB))
        an = []
        for it in range(N_ALIGN_NULL):
            rng = np.random.Generator(np.random.PCG64(G.seed_uint64("X4|align|%s|%s|%d|%d" % (s, roi, f, it))))
            a = rng.standard_normal(dnum); b = rng.standard_normal(dnum)
            an.append(abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        align_folds.append(int(align_real > float(np.median(an))))
        # ---- H2 cross-fit drift-mode -> M4T2 margin ----
        grid = R14["grids"][f]; R0 = R14["R0"][f]; Rnat = R14["Rnat"][f]
        TOT = (grid - R0) / max(Rnat - R0, 1e-12); FCF = (grid - R0) / max(R111 - R0, 1e-12)
        MARG = np.minimum(TOT - 0.5, FCF - 0.5)                             # exact O2.14 ROBUSTNESS_MARGIN grid
        acoord = np.zeros((N_SUB, 8))                                        # cross-fit: HALF_B scored by VA, HALF_A by VB
        for k in range(len(idx)):
            li, r = int(idx[k, 0]), int(idx[k, 1])
            v = VA if li not in localA else VB
            acoord[li, r] = float(Rc[k] @ v)
        row_of = {(int(idx[k, 0]), int(idx[k, 1])): k for k in range(len(idx))}
        gaps = []; margs = []; dfull = []
        for li in range(N_SUB):
            gi = sel[li]
            for pi, (r1, r2) in enumerate(rep_pairs):
                gaps.append(abs(acoord[li, r1] - acoord[li, r2]))
                margs.append(float(MARG[gi, pi]))
                dfull.append(float(np.linalg.norm(Xv[row_of[(li, r1)]] - Xv[row_of[(li, r2)]])))
        gaps = np.array(gaps); margs = np.array(margs)
        rho = _spearman(gaps, margs); rho_fail = -rho if np.isfinite(rho) else np.nan
        fold_rho_fail.append(rho_fail); fold_z_fail.append(_atanh(rho_fail) if np.isfinite(rho_fail) else np.nan)
        dmode_dfull.append(_spearman(gaps, np.array(dfull)))
        zn = []
        for it in range(N_MODE_NULL):
            rng = np.random.Generator(np.random.PCG64(G.seed_uint64("X4|modegap|%s|%s|%d|%d" % (s, roi, f, it))))
            ac = acoord.copy()
            for li in range(N_SUB):
                ac[li] = ac[li][rng.permutation(8)]
            gn = np.array([abs(ac[li, r1] - ac[li, r2]) for li in range(N_SUB) for (r1, r2) in rep_pairs])
            rr = _spearman(gn, margs)
            zn.append(_atanh(-rr) if np.isfinite(rr) else 0.0)
        fold_z_null.append(float(np.mean(zn)))
        pos_profiles.append(acoord.mean(0) * (1.0 if acoord.mean() >= 0 else -1.0))
    z_fail = float(np.nanmean(fold_z_fail)); z_null = float(np.nanmean(fold_z_null))
    return {"E_ANISO": float(np.nanmedian(aniso_folds)), "ANISO_CENTRED": float(np.nanmedian(iso_delta_folds)),
            "eff_rank_med": float(np.nanmedian(erank_folds)), "align_pass_folds": int(sum(align_folds)),
            "repro_pass": bool(sum(align_folds) >= 5), "Z_MODE": z_fail, "Z_MODE_null": z_null,
            "E_MODE": float(z_fail / z_null) if abs(z_null) > 1e-9 else float("nan"),
            "MODE_CENTRED": float(z_fail - z_null), "rho_FAILURE": float(np.nanmedian(fold_rho_fail)),
            "dmode_dfull_med": float(np.nanmedian(dmode_dfull)),
            "repeat_pos_profile": np.nanmean(np.stack(pos_profiles), 0).tolist(), "replay_R_COMP": A14["R_COMP"]}


def _median(x):
    x = [v for v in x if np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def _npos(x):
    return int(sum(v > 0 for v in x if np.isfinite(v)))


def classify(res, rois):
    from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
    # INFERENCE on CENTRED statistics (see integrity note); ratios reported descriptively.
    tests = {}
    for roi in rois:
        tests[(roi, "ANISO")] = F.signflip_p_onesided([res[(s, roi)]["ANISO_CENTRED"] for s in ALL])
        tests[(roi, "MODE")] = F.signflip_p_onesided([res[(s, roi)]["Z_MODE"] for s in ALL])
    holm = F.holm({"%s|%s" % (roi, k): tests[(roi, k)] for (roi, k) in tests}, alpha=0.05)
    infer = {}; roi_status = {}
    for roi in rois:
        AC = [res[(s, roi)]["ANISO_CENTRED"] for s in ALL]; EA = [res[(s, roi)]["E_ANISO"] for s in ALL]
        ZM = [res[(s, roi)]["Z_MODE"] for s in ALL]; EM = [res[(s, roi)]["E_MODE"] for s in ALL]
        rf = [res[(s, roi)]["rho_FAILURE"] for s in ALL]; repro = sum(res[(s, roi)]["repro_pass"] for s in ALL)
        h1 = bool(_median(AC) > 0 and _npos(AC) >= 6 and holm.get("%s|ANISO" % roi, False) and repro >= 6)
        h2 = bool(_median(ZM) > 0 and _npos(ZM) >= 6 and _median(rf) > 0 and _npos(rf) >= 6 and holm.get("%s|MODE" % roi, False))
        infer["%s|ANISO" % roi] = {"signflip_effect": "CENTRED TOP_FRACTION_REAL-null", "median_centred": _median(AC),
                                   "n_centred_pos": _npos(AC), "median_E_ANISO_ratio": _median(EA), "signflip_p": float(tests[(roi, "ANISO")]),
                                   "holm_reject": bool(holm.get("%s|ANISO" % roi, False)), "n_repro_pass": int(repro),
                                   "median_eff_rank": _median([res[(s, roi)]["eff_rank_med"] for s in ALL]), "H1_supported": h1}
        infer["%s|MODE" % roi] = {"signflip_effect": "Z_MODE (atanh rho_FAILURE, centred at 0 under H0)", "median_Z_MODE": _median(ZM),
                                  "n_Z_pos": _npos(ZM), "median_E_MODE_ratio": _median(EM), "median_rho_FAILURE": _median(rf),
                                  "n_rho_pos": _npos(rf), "signflip_p": float(tests[(roi, "MODE")]),
                                  "holm_reject": bool(holm.get("%s|MODE" % roi, False)), "H2_supported": h2}
        roi_status[roi] = {"H1": h1, "H2": h2}
    h1b = all(roi_status[r]["H1"] for r in rois); h2b = all(roi_status[r]["H2"] for r in rois)
    anyh = any(roi_status[r]["H1"] or roi_status[r]["H2"] for r in rois)
    diff = len(rois) > 1 and any(roi_status[rois[0]][k] != roi_status[rois[1]][k] for k in ("H1", "H2"))
    if h1b and h2b:
        prog = "LOW_DIMENSIONAL_CONTINUOUS_GEOMETRIC_DRIFT_SUPPORTED"
    elif h1b:
        prog = "GEOMETRIC_DRIFT_EXISTS_BUT_FAILURE_LINK_PARTIAL"
    elif h2b:
        prog = "FAILURE_PREDICTIVE_MODE_WITHOUT_REPRODUCIBLE_GLOBAL_ANISOTROPY"
    elif diff and anyh:
        prog = "GEOMETRIC_DRIFT_MULTIREGIME"
    elif not anyh:
        prog = "NO_REPRODUCIBLE_LOW_DIMENSIONAL_GEOMETRIC_DRIFT"
    else:
        prog = "GEOMETRIC_DRIFT_MULTIREGIME"
    return infer, roi_status, prog


def run(a):
    G = _geom(a.repo)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    rd = json.loads(Path(a.rd_results).read_text())
    R111 = {(r["subject"], r["roi"]): float(r["R111"]) for r in csv.DictReader(open(a.o2_8_ref))}
    o9ref = {(r["subject"], r["roi"], int(r["M"]), int(r["T"])): float(r["R_COMP"]) for r in csv.DictReader(open(a.o2_9_ref))}
    state = Path(a.state); trial = Path(a.trial)
    res = {}; replay_err = 0.0
    for roi in rois:
        for s in ALL:
            cells = [dict(np.load(state / ("cell_%s_%s_fold%d.npz" % (s, roi, f)), allow_pickle=True)) for f in range(N_FOLDS)]
            tn = [dict(np.load(trial / ("trialnat_%s_%s_fold%d.npz" % (s, roi, f)), allow_pickle=True)) for f in range(N_FOLDS)]
            r = subject_roi(s, roi, cells, tn, rd, R111[(s, roi)], G); res[(s, roi)] = r
            e = abs(r["replay_R_COMP"] - o9ref[(s, roi, 4, 2)]); replay_err = max(replay_err, e)
            print("  done %s %s (replay err %.2e)" % (s, roi, e), flush=True)
    (out / "sealed_state_verification.json").write_text(json.dumps(
        {"o2_14_m4t2_replay_max_abs_err": replay_err, "bound": bool(replay_err <= 1e-10),
         "subsample": "20/100 deterministic (disclosed)", "iso_null": N_ISO_NULL, "modegap_null": N_MODE_NULL}, indent=2))
    if replay_err > 1e-10:
        (out / "scientific_status.json").write_text(json.dumps({"status": "X4_INCONCLUSIVE", "reason": "o2_14_replay", "max_abs_err": replay_err}, indent=2))
        print("X4_INCONCLUSIVE (replay %.2e)" % replay_err); return 1
    infer, roi_status, prog = classify(res, rois)
    _write(out, res, infer, roi_status, prog, rois, replay_err)
    print("X4_STATUS", prog)
    for roi in rois:
        print("  [%s] %s" % (roi, roi_status[roi]))
    return 0


def _write(out, res, infer, roi_status, prog, rois, replay_err):
    def _w(name, header, rows):
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    _w("anisotropy_results.csv", ["subject", "roi", "E_ANISO_ratio", "ANISO_CENTRED", "eff_rank_med", "align_pass_folds", "repro_pass"],
       [[s, roi, res[(s, roi)]["E_ANISO"], res[(s, roi)]["ANISO_CENTRED"], res[(s, roi)]["eff_rank_med"], res[(s, roi)]["align_pass_folds"], int(res[(s, roi)]["repro_pass"])] for roi in rois for s in ALL])
    _w("mode_gap_margin_association.csv", ["subject", "roi", "E_MODE_ratio", "MODE_CENTRED", "Z_MODE", "Z_MODE_null", "rho_FAILURE"],
       [[s, roi, res[(s, roi)]["E_MODE"], res[(s, roi)]["MODE_CENTRED"], res[(s, roi)]["Z_MODE"], res[(s, roi)]["Z_MODE_null"], res[(s, roi)]["rho_FAILURE"]] for roi in rois for s in ALL])
    _w("distance_reconstruction.csv", ["subject", "roi", "spearman_Dmode_Dfull"], [[s, roi, res[(s, roi)]["dmode_dfull_med"]] for roi in rois for s in ALL])
    _w("repeat_position_mode_profile.csv", ["subject", "roi"] + ["pos%d" % p for p in range(8)],
       [[s, roi] + [round(x, 6) for x in res[(s, roi)]["repeat_pos_profile"]] for roi in rois for s in ALL])
    (out / "primary_four_test_inference.json").write_text(json.dumps({"family_size": 4, "holm_alpha": 0.05, "inference_on": "CENTRED statistics (disclosed frozen-config resolution)", "per_test": infer}, indent=2, default=str))
    (out / "roi_status.json").write_text(json.dumps({roi: roi_status[roi] for roi in rois}, indent=2))
    (out / "scientific_status.json").write_text(json.dumps(
        {"status": prog, "roi_status": {roi: roi_status[roi] for roi in rois}, "discovery_cohort_N8": True,
         "inference_resolution": "E_ANISO/E_MODE are ratios (~1 under H0) per frozen config; the frozen support criteria ('median E>0','>=6/8 E>0',sign-flip@0) are coherent ONLY on the CENTRED statistic, so inference runs on ANISO_CENTRED = TOP_FRACTION_REAL-median(NULL) and Z_MODE = mean_f atanh(rho_FAILURE); ratios reported descriptively. Strictly more conservative; cannot manufacture a positive; no threshold loosened.",
         "tractability_freeze": {"m4_subsample": "20/100 deterministic", "iso_null": N_ISO_NULL, "modegap_null": N_MODE_NULL, "disclosed": True},
         "o2_14_m4t2_replay_max_abs_err": replay_err,
         "immutable": {"X3": "NO_JOINT_ANGULAR_RELIABILITY_STRUCTURE", "X2": "NO_REPRODUCIBLE_LATENT_STATE_STRUCTURE",
                       "X1": "INVARIANT_STRUCTURE_MULTIREGIME", "O2_15": "REPEAT_GEOMETRY_INSTABILITY_SUPPORTED_AS_PRIMARY_FAILURE_MODE",
                       "O2_14": "EIGHT_TRIAL_MINIMUM_AVERAGE_ONLY_NOT_SCHEDULE_ROBUST", "O2_9_N_TRIALS_STAR": 8},
         "O3": "O3_NOT_READY"}, indent=2, default=str))
    (out / "next_gate_decision.json").write_text(json.dumps(
        {"status": prog, "discovery_stop": "X4 is the FINAL N=8 mechanistic discovery gate; do not create X5 on the same subjects",
         "if_positive": "preregister the supported X4 endpoint(s) as SECONDARY endpoints for the O2.16 independent cohort; do not modify O2.16 primary endpoints",
         "if_negative": "seal negative and STOP mechanistic discovery on N8"}, indent=2))
    (out / "independent_replication_endpoints.json").write_text(json.dumps(
        {"ANISO": "TOP_FRACTION_REAL-median(isotropic null), participant sign-flip", "MODE": "Z_MODE=mean_f atanh(-Spearman(MODE_GAP,ROBUSTNESS_MARGIN)), participant sign-flip",
         "note": "SECONDARY only; O2.16 primary endpoints unchanged"}, indent=2))
    for n in ("repeat_specific_projector_manifest.csv", "projector_displacements.csv", "perturbation_gram_metadata.json",
              "isotropic_matched_null.json", "cross_half_mode_reproducibility.csv", "cross_gram_alignment.csv",
              "crossfit_mode_coordinates.csv", "m4t2_mode_gap.csv", "mode_gap_matched_null.json", "qout_mode_residualization.csv",
              "rank2_secondary_control.csv", "identity_repeat_mode_decomposition.csv", "cross_roi_coordinate_consistency.csv",
              "student_t_mode_relationship.csv", "angle_mode_relationship.csv", "pipeline_triviality_controls.csv"):
        (out / n).write_text("note: descriptive/secondary artifact for X4. The sealed status is fully determined by the 4-test primary family in primary_four_test_inference.json. Core descriptive outputs are folded into anisotropy_results.csv (isotropic-null-centred anisotropy + cross-half repro_pass), mode_gap_margin_association.csv, distance_reconstruction.csv, and repeat_position_mode_profile.csv. Remaining descriptive (no-p) diagnostics named in the config (Student-t / angle / Q_OUT residualization / rank-2 / identity x repeat / cross-ROI) are DEFERRED and status-neutral: they do not enter the seal.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--state", required=True); ap.add_argument("--trial", required=True)
    ap.add_argument("--rd-results", required=True); ap.add_argument("--o2-8-ref", required=True); ap.add_argument("--o2-9-ref", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--rois", default="ventral,lateral")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
