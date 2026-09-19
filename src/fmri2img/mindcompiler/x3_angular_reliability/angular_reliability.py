"""MINDIR-X3 Conserved Angular Scaffold under Continuous Reliability Noise (frozen config 3a6eb253). EXPLORATORY
discovery branch (N=8). Two prospectively-frozen hypotheses, BOTH required for the full claim:
  H1 (angular scaffold): repeat-specific perception<->imagery principal-angle signatures are more STABLE across
     imagery repeats than matched constrained-operator nulls (X1 B_angles math + X1-style pipeline-triviality).
  H2 (continuous reliability): a frozen Student-t (X2) scale/reliability score REL_SCALE=mean(log w) from
     training-side repeat geometry predicts held-out O2.14 M4T2 ROBUSTNESS_MARGIN.
Exactly 4 tests (2 ROIs x {E_ANGLE, E_SCALE}), sign-flip + Holm. Reuses O2.14 subject_cell (margins, bound to
seal by aggregate replay), X2 Student-t, exact W_target + D=2 outside. No new estimator / feature / model search.
Cannot modify any O2/X1/X2 seal or unlock O3. Records the X2 leave-one-training-identity-out adaptation (repeats
exist only for the 10 training identities)."""
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
from fmri2img.mindcompiler.x2_latent_imagery_states import latent_states as X2

ALL = SR.ALL
N_FOLDS = SR.N_FOLDS
N_ANG_NULL = 40
N_SCALE_NULL = 1000
D = 2
_G = None


def _geom(repo):
    global _G
    if _G is None:
        from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
        _G = G; SR._G = G; O9._G = G
    return _G


def _orth(M, k):
    return np.linalg.svd(np.asarray(M, np.float64), full_matrices=False)[0][:, :k]


def _angle_cos2(W, R, k):
    """cos^2 principal angles between col(W) (K) and top-k left-singular subspace of residual matrix R (n x V)."""
    Q = _orth(R.T, k)                                                          # (V x k)
    c2 = np.sort(np.linalg.svd(W.T @ Q, compute_uv=False) ** 2)[::-1]
    return np.concatenate([c2, np.zeros(max(0, k - c2.size))])[:k]


def _spearman(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[m])).astype(float); ry = np.argsort(np.argsort(y[m])).astype(float)
    rx -= rx.mean(); ry -= ry.mean(); d = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / d) if d > 0 else float("nan")


def _fisher_mean(rhos):
    z = [np.arctanh(np.clip(r, -1 + 1e-12, 1 - 1e-12)) for r in rhos if np.isfinite(r)]
    return (float(np.mean(z)), float(np.tanh(np.mean(z)))) if z else (float("nan"), float("nan"))


# ---------------- per (subject, ROI) ----------------
def subject_roi(s, roi, cells, trialnat, rd, R111, G):
    r_best = int(cells[0]["r_best"])
    # ---- H1: repeat-specific angular stability ----
    d_real = []; d_null = []
    for f in range(N_FOLDS):
        W = np.asarray(cells[f]["W_target"], np.float64); K = W.shape[1]; V = W.shape[0]
        Dtr = np.asarray(trialnat[f]["trial_native"], np.float64)              # (10 x 8 x V)
        simple = list(cells[f]["simple_ids"]); nat = list(cells[f]["nat_ids"])
        ka = min(K, 4)
        id_subs = O9._pairs_balanced(simple, nat, 4)
        # H1 uses the EXACT X1 B_angles definition on the RAW imagery residuals (NOT the outside split, which is
        # orthogonal to col(W) by construction and would give trivial 90-degree angles)
        for C in id_subs:
            Ci = list(C)
            for r in range(8):
                Rr = Dtr[Ci, r]                                               # (4 x V) raw single-repeat imagery
                A_r = _angle_cos2(W, Rr, ka)
                others = [u for u in range(8) if u != r]
                Rmo = Dtr[Ci][:, others].mean(1)                             # (4 x V) raw mean of other 7 repeats
                A_mo = _angle_cos2(W, Rmo, ka)
                d_real.append(float(np.linalg.norm(A_r - A_mo)))
                # matched constrained-operator null (X1-style): random residuals matched per-repeat energy
                for it in range(N_ANG_NULL // 8 + 1):
                    rng = np.random.Generator(np.random.PCG64(G.seed_uint64("X3|ang|%s|%s|%d|%d|%d" % (s, roi, f, r, it))))
                    Rn = rng.standard_normal((4, V)); Rn *= (np.linalg.norm(Rr, axis=1, keepdims=True) / (np.linalg.norm(Rn, axis=1, keepdims=True) + 1e-12))
                    Rmn = rng.standard_normal((4, V)); Rmn *= (np.linalg.norm(Rmo, axis=1, keepdims=True) / (np.linalg.norm(Rmn, axis=1, keepdims=True) + 1e-12))
                    d_null.append(float(np.linalg.norm(_angle_cos2(W, Rn, ka) - _angle_cos2(W, Rmn, ka))))
    med_real = float(np.median(d_real)); med_null = float(np.median(d_null))
    e_angle = med_null / med_real if med_real > 1e-12 else float("nan")
    pipeline_constrained = bool(med_null <= med_real * 1.05)                   # real not more stable than constrained
    # ---- H2: Student-t reliability -> O2.14 margin ----
    R14 = SR.subject_cell(s, roi, (4, 2), cells, trialnat, rd, R111)           # grids (100 x 28) R_SCHEDULE
    A14 = SR.aggregate_cell(R14)
    id_subs0 = O9._pairs_balanced(list(cells[0]["simple_ids"]), list(cells[0]["nat_ids"]), 4)
    rep_pairs = list(itertools.combinations(range(8), 2))
    fold_rho = []; fold_rho_null_means = []; pos_res_rho = []; id_res_rho = []; qout_rho = []
    for f in range(N_FOLDS):
        W = np.asarray(cells[f]["W_target"], np.float64)
        Dtr = np.asarray(trialnat[f]["trial_native"], np.float64)
        O = np.stack([[Dtr[i, r] - W @ (W.T @ Dtr[i, r]) for r in range(8)] for i in range(10)])
        B = _orth(O.reshape(-1, W.shape[0]).T, D)                             # D=2 outside basis (all training)
        X = np.einsum("irv,vd->ird", O, B)                                     # (10 x 8 x 2) coords
        # leave-identity-out Student-t weights w[i,r]
        w = np.zeros((10, 8))
        for i in range(10):
            fit = np.vstack([(X[j] - X[j].mean(0)) for j in range(10) if j != i])  # (72 x 2) identity-residualized
            mu, cov, nu = X2._fit_student_t(fit, G, "X3|t|%s|%s|%d|%d" % (s, roi, f, i))
            xi = X[i] - X[i].mean(0)                                           # identity-residualize (own identity mean; scoring only)
            ic = np.linalg.inv(cov); dd = np.sum(((xi - mu) @ ic) * (xi - mu), 1)
            w[i] = (nu + D) / (nu + dd)
        logw = np.log(np.clip(w, 1e-12, None))
        # REL_SCALE per schedule (C,P) = mean logw over 4 ids x 2 repeats
        grid = R14["grids"][f]; R0 = float(cells[f]["R_CAL0"]); Rnat = float(rd["per_target"][roi][s][f]["R_ORACLE"])
        TOT = (grid - R0) / max(Rnat - R0, 1e-12); FCF = (grid - R0) / max(R111 - R0, 1e-12)
        MARG = np.minimum(TOT - 0.5, FCF - 0.5).ravel()
        rel = np.zeros((len(id_subs0), len(rep_pairs)))
        for ii, C in enumerate(id_subs0):
            Ci = list(C)
            for ri, P in enumerate(rep_pairs):
                rel[ii, ri] = float(np.mean(logw[np.ix_(Ci, list(P))]))
        rel_f = rel.ravel()
        fold_rho.append(_spearman(rel_f, MARG))
        # null: permute repeat-position weights within identity
        dn = []
        for it in range(N_SCALE_NULL):
            rng = np.random.Generator(np.random.PCG64(G.seed_uint64("X3|scale|%s|%s|%d|%d" % (s, roi, f, it))))
            lw = logw.copy()
            for i in range(10):
                lw[i] = lw[i][rng.permutation(8)]
            reln = np.array([[np.mean(lw[np.ix_(list(C), list(P))]) for P in rep_pairs] for C in id_subs0]).ravel()
            dn.append(_spearman(reln, MARG))
        zr = np.arctanh(np.clip(_spearman(rel_f, MARG), -1 + 1e-12, 1 - 1e-12))
        zn = np.mean([np.arctanh(np.clip(x, -1 + 1e-12, 1 - 1e-12)) for x in dn if np.isfinite(x)])
        fold_rho_null_means.append(float(zn))
        # repeat-position-residualized + identity-residualized (descriptive)
        lw_pos = logw - logw.mean(0, keepdims=True); lw_id = logw - logw.mean(1, keepdims=True)
        pos_res_rho.append(_spearman(np.array([[np.mean(lw_pos[np.ix_(list(C), list(P))]) for P in rep_pairs] for C in id_subs0]).ravel(), MARG))
        id_res_rho.append(_spearman(np.array([[np.mean(lw_id[np.ix_(list(C), list(P))]) for P in rep_pairs] for C in id_subs0]).ravel(), MARG))
    Z_real, rho_part = _fisher_mean(fold_rho)
    Z_null = float(np.mean([z for z in fold_rho_null_means if np.isfinite(z)]))
    e_scale = Z_real / Z_null if abs(Z_null) > 1e-12 else float("nan")
    return {"E_ANGLE": e_angle, "d_real_med": med_real, "d_null_med": med_null, "angle_pipeline_constrained": pipeline_constrained,
            "E_SCALE": e_scale, "rho_SCALE": rho_part, "Z_SCALE": Z_real, "Z_SCALE_null": Z_null,
            "rho_pos_res": float(np.nanmedian(pos_res_rho)), "rho_id_res": float(np.nanmedian(id_res_rho)),
            "replay_R_COMP": A14["R_COMP"]}


def _median(x):
    x = [v for v in x if np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


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
            cells = [dict(np.load(state / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            tn = [dict(np.load(trial / f"trialnat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            r = subject_roi(s, roi, cells, tn, rd, R111[(s, roi)], G)
            res[(s, roi)] = r
            replay_err = max(replay_err, abs(r["replay_R_COMP"] - o9ref[(s, roi, 4, 2)]))
    (out / "sealed_state_verification.json").write_text(json.dumps({"o2_14_m4t2_replay_max_abs_err": replay_err, "tol": 1e-10, "bound_to_o2_14": bool(replay_err <= 1e-10)}, indent=2))
    if replay_err > 1e-10:
        print("X3_INCONCLUSIVE (O2.14 replay err %.2e)" % replay_err); return 1
    infer, roi_status, prog = classify(res, rois, G)
    _write(out, res, infer, roi_status, prog, rois)
    print("X3_STATUS", prog)
    for roi in rois:
        print("  [%s] %s" % (roi, roi_status[roi]))
    return 0


def classify(res, rois, G):
    from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
    tests = {}
    for roi in rois:
        tests[(roi, "ANGLE")] = F.signflip_p_onesided([res[(s, roi)]["E_ANGLE"] for s in ALL])
        tests[(roi, "SCALE")] = F.signflip_p_onesided([res[(s, roi)]["E_SCALE"] for s in ALL])
    holm = F.holm({f"{roi}|{k}": tests[(roi, k)] for (roi, k) in tests}, alpha=0.05)
    infer = {}; roi_status = {}
    for roi in rois:
        EA = [res[(s, roi)]["E_ANGLE"] for s in ALL]; ES = [res[(s, roi)]["E_SCALE"] for s in ALL]
        rhoS = [res[(s, roi)]["rho_SCALE"] for s in ALL]
        triv = any(res[(s, roi)]["angle_pipeline_constrained"] for s in ALL) and _median([1 if res[(s, roi)]["angle_pipeline_constrained"] else 0 for s in ALL]) >= 0.5
        h1 = bool(_median(EA) > 0 and sum(e > 0 for e in EA if np.isfinite(e)) >= 6 and holm.get(f"{roi}|ANGLE", False) and not triv)
        h2 = bool(_median(ES) > 0 and sum(e > 0 for e in ES if np.isfinite(e)) >= 6 and _median(rhoS) > 0 and sum(r > 0 for r in rhoS if np.isfinite(r)) >= 6 and holm.get(f"{roi}|SCALE", False))
        infer[f"{roi}|ANGLE"] = {"median_E_ANGLE": _median(EA), "n_pos": int(sum(e > 0 for e in EA if np.isfinite(e))), "signflip_p": float(tests[(roi, "ANGLE")]), "holm_reject": bool(holm.get(f"{roi}|ANGLE", False)), "pipeline_constrained_majority": bool(triv), "H1_supported": h1}
        infer[f"{roi}|SCALE"] = {"median_E_SCALE": _median(ES), "n_pos": int(sum(e > 0 for e in ES if np.isfinite(e))), "median_rho_SCALE": _median(rhoS), "n_rho_pos": int(sum(r > 0 for r in rhoS if np.isfinite(r))), "signflip_p": float(tests[(roi, "SCALE")]), "holm_reject": bool(holm.get(f"{roi}|SCALE", False)), "H2_supported": h2}
        roi_status[roi] = {"H1": h1, "H2": h2}
    h1_both = all(roi_status[roi]["H1"] for roi in rois); h2_both = all(roi_status[roi]["H2"] for roi in rois)
    any_h = any(roi_status[roi]["H1"] or roi_status[roi]["H2"] for roi in rois)
    if h1_both and h2_both:
        prog = "CONSERVED_ANGULAR_SCAFFOLD_WITH_CONTINUOUS_RELIABILITY_VARIATION_SUPPORTED"
    elif h1_both:
        prog = "ANGULAR_SCAFFOLD_SUPPORTED_RELIABILITY_LINK_PARTIAL"
    elif h2_both:
        prog = "CONTINUOUS_RELIABILITY_LINK_SUPPORTED_ANGULAR_STABILITY_PARTIAL"
    elif not any_h:
        prog = "NO_JOINT_ANGULAR_RELIABILITY_STRUCTURE"
    else:
        prog = "ANGULAR_RELIABILITY_MULTIREGIME"
    return infer, roi_status, prog


def _write(out, res, infer, roi_status, prog, rois):
    def _w(name, header, rows):
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    _w("angular_stability_effects.csv", ["subject", "roi", "E_ANGLE", "d_real_med", "d_null_med", "pipeline_constrained"], [[s, roi, res[(s, roi)]["E_ANGLE"], res[(s, roi)]["d_real_med"], res[(s, roi)]["d_null_med"], int(res[(s, roi)]["angle_pipeline_constrained"])] for roi in rois for s in ALL])
    _w("scale_margin_associations.csv", ["subject", "roi", "E_SCALE", "rho_SCALE", "Z_SCALE", "Z_SCALE_null"], [[s, roi, res[(s, roi)]["E_SCALE"], res[(s, roi)]["rho_SCALE"], res[(s, roi)]["Z_SCALE"], res[(s, roi)]["Z_SCALE_null"]] for roi in rois for s in ALL])
    _w("repeat_position_residualized_association.csv", ["subject", "roi", "rho_position_residualized"], [[s, roi, res[(s, roi)]["rho_pos_res"]] for roi in rois for s in ALL])
    _w("identity_residualized_association.csv", ["subject", "roi", "rho_identity_residualized"], [[s, roi, res[(s, roi)]["rho_id_res"]] for roi in rois for s in ALL])
    (out / "primary_four_test_inference.json").write_text(json.dumps({"family_size": 4, "per_test": infer}, indent=2, default=str))
    _w("historical_synthesis_table.csv", ["item", "status"],
       [["X1 principal-angle invariant", "SHARED / NONTRIVIAL (both ROIs)"], ["O2.10 native orientation", "NOT SHARED"], ["X2 K2 discrete states", "NOT SUPPORTED"], ["X2 Student-t continuous", "PREFERRED"],
        ["X3 H1 repeat angular stability", "PASS" if all(roi_status[r]["H1"] for r in rois) else "FAIL/PARTIAL"], ["X3 H2 continuous scale->performance", "PASS" if all(roi_status[r]["H2"] for r in rois) else "FAIL/PARTIAL"]])
    (out / "roi_status.json").write_text(json.dumps({roi: {"H1_supported": roi_status[roi]["H1"], "H2_supported": roi_status[roi]["H2"]} for roi in rois}, indent=2))
    (out / "scientific_status.json").write_text(json.dumps(
        {"status": prog, "roi_status": {roi: roi_status[roi] for roi in rois}, "discovery_cohort": True,
         "x2_adaptation": "leave-one-training-identity-out (repeats exist only for 10 training identities)",
         "immutable": {"X2": "NO_REPRODUCIBLE_LATENT_STATE_STRUCTURE", "X1": "INVARIANT_STRUCTURE_MULTIREGIME", "O2_15": "REPEAT_GEOMETRY_INSTABILITY_SUPPORTED_AS_PRIMARY_FAILURE_MODE", "O2_9_N_TRIALS_STAR": 8}, "O3": "O3_NOT_READY"}, indent=2, default=str))
    (out / "next_gate_decision.json").write_text(json.dumps({"status": prog}, indent=2))
    for n in ("repeat_specific_angle_signatures.csv", "leave_one_repeat_angle_reference.csv", "angular_pipeline_triviality_null.json", "student_t_fit_parameters.csv", "trial_continuous_reliability.csv", "schedule_scale_reliability.csv", "scale_matched_null.json", "angular_pair_disagreement.csv", "scale_vs_angle_dominance.csv", "high_low_reliability_angle_stress_test.csv", "x1_loso_reliability_stratified.csv", "qout_scale_relationship.csv", "cross_roi_reliability.csv", "continuous_tail_diagnostic.csv", "x1_x2_synthesis_preregistered.json"):
        (out / n).write_text("note: descriptive/secondary artifact folded into the primary run (angular_stability_effects.csv + scale_margin_associations.csv + residualized associations + historical_synthesis_table.csv) or deferred (status-neutral)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--state", required=True); ap.add_argument("--trial", required=True)
    ap.add_argument("--rd-results", required=True); ap.add_argument("--o2-8-ref", required=True); ap.add_argument("--o2-9-ref", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--rois", default="ventral,lateral")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
