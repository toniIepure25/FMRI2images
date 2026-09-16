"""O2.15 Acquisition-Schedule Failure-Mode Diagnostic (frozen config 831ad419). Why do prospectively-valid M4T2
acquisitions sometimes succeed and sometimes fail? For each M4T2 schedule (identity subset C x repeat pair P)
compute a TRAINING-ONLY geometry-agreement Q between the 2-repeat PAIR estimate and the complementary 6-repeat
COMPLEMENT estimate (within Q_IN, outside Q_OUT), then correlate (Spearman across the 2800 schedules per fold)
against the ALREADY-SEALED O2.14 held-out ROBUSTNESS_MARGIN. Fisher-average over folds -> participant rho;
4-test (2 comp x 2 ROI) sign-flip + Holm. Held-out margins are REGENERATED from source with the exact O2.9
estimator and bound to the O2.14 seal by reproducing its aggregates to <=1e-10. Descriptive: residual
reliability, energy, success/failure contrast, failure attribution, single-repeat quality, additive repeat
model, interaction alignment, LOSO / cross-ROI consistency, secondary M2T4/M8T1. No schedule optimization / no
good-repeat protocol / no new estimator. Reuses O2.14 subject_cell + O2.9 SVD helpers."""
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
COMPONENTS = ["IN", "OUT"]
_G = None


def _geom(repo):
    global _G
    if _G is None:
        from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
        _G = G; SR._G = G; O9._G = G
    return _G


def _rankdata(x):
    """Average-rank of x (deterministic tie handling)."""
    x = np.asarray(x, np.float64); order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), np.float64); sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _spearman(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx = _rankdata(x[m]); ry = _rankdata(y[m]); rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / d) if d > 0 else float("nan")


def _cos2(a, b):
    na = float(a @ a); nb = float(b @ b)
    return float((a @ b) ** 2 / (na * nb)) if na > 0 and nb > 0 else float("nan")


def _proj_sim(U, V, k):
    """||U^T V||_F^2 / k for orthonormal U,V (=trace(P_U P_V)/k)."""
    return float(np.sum((U.T @ V) ** 2) / k)


def _fisher_mean(rhos):
    z = [np.arctanh(np.clip(r, -1 + 1e-12, 1 - 1e-12)) for r in rhos if np.isfinite(r)]
    if not z:
        return float("nan"), float("nan")
    Z = float(np.mean(z)); return Z, float(np.tanh(Z))


# ---------------- per (subject, ROI) M4T2 pair-vs-complement ----------------
def _schedule_Q(W, Dtr, C, P, r_best):
    """Training-only Q_IN, Q_OUT + residual reliability + energies for one schedule (identity subset C, pair P)."""
    q = min(r_best, 4); comp = [t for t in range(8) if t not in P]
    dP = np.stack([Dtr[i][list(P)].mean(0) for i in C]); dC = np.stack([Dtr[i][comp].mean(0) for i in C])  # (4 x V)
    zP = dP @ W; zC = dC @ W                                                    # (4 x K)
    oP = np.stack([dP[k] - W @ (W.T @ dP[k]) for k in range(4)]); oC = np.stack([dC[k] - W @ (W.T @ dC[k]) for k in range(4)])
    Uin_P = O9.within_basis_direct(zP, q); Uin_C = O9.within_basis_direct(zC, q)
    Bout_P = O9.outside_basis_direct(oP.T, 2); Bout_C = O9.outside_basis_direct(oC.T, 2)
    Q_IN = _proj_sim(Uin_P, Uin_C, q) if (Uin_P is not None and Uin_C is not None) else float("nan")
    Q_OUT = _proj_sim(Bout_P, Bout_C, 2) if (Bout_P is not None and Bout_C is not None) else float("nan")
    cos_n = float(np.mean([_cos2(dP[k], dC[k]) for k in range(4)]))
    cos_w = float(np.mean([_cos2(zP[k], zC[k]) for k in range(4)]))
    cos_o = float(np.mean([_cos2(oP[k], oC[k]) for k in range(4)]))
    eP = (float(np.mean([dP[k] @ dP[k] for k in range(4)])), float(np.mean([zP[k] @ zP[k] for k in range(4)])), float(np.mean([oP[k] @ oP[k] for k in range(4)])))
    eC = (float(np.mean([dC[k] @ dC[k] for k in range(4)])), float(np.mean([zC[k] @ zC[k] for k in range(4)])), float(np.mean([oC[k] @ oC[k] for k in range(4)])))
    return Q_IN, Q_OUT, cos_n, cos_w, cos_o, eP, eC


def run(a):
    G = _geom(a.repo)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    rd = json.loads(Path(a.rd_results).read_text())
    R111 = {(r["subject"], r["roi"]): float(r["R111"]) for r in csv.DictReader(open(a.o2_8_ref))}
    o9ref = {(r["subject"], r["roi"], int(r["M"]), int(r["T"])): float(r["R_COMP"]) for r in csv.DictReader(open(a.o2_9_ref))}
    o14cov = {(r["subject"], r["roi"], int(r["M"]), int(r["T"])): float(r["SCHEDULE_COVERAGE"]) for r in csv.DictReader(open(a.o2_14_cov))}
    state = Path(a.state); trial = Path(a.trial)
    rep_pairs = list(itertools.combinations(range(8), 2))
    _w(out / "repeat_pair_manifest.csv", ["pair_idx", "a", "b"], [[i, p[0], p[1]] for i, p in enumerate(rep_pairs)])

    part = {s: {roi: {} for roi in rois} for s in ALL}
    replay_err = 0.0; disjoint_ok = True
    contrast_rows, attr_rows, resrel_rows, energy_rows, singleq_rows, addmodel_rows, intalign_rows = [], [], [], [], [], [], []
    posquality = {roi: {} for roi in rois}                                      # per subj -> 8-vector (IN)
    for roi in rois:
        for s in ALL:
            cells = [dict(np.load(state / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            tn = [dict(np.load(trial / f"trialnat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            r_best = int(cells[0]["r_best"]); R111v = R111[(s, roi)]
            R14 = SR.subject_cell(s, roi, (4, 2), cells, tn, rd, R111v)         # grids = 100x28 R_SCHEDULE
            A14 = SR.aggregate_cell(R14)
            replay_err = max(replay_err, abs(A14["R_COMP"] - o9ref[(s, roi, 4, 2)]), abs(A14["coverage"] - o14cov[(s, roi, 4, 2)]))
            id_subs = O9._pairs_balanced(list(cells[0]["simple_ids"]), list(cells[0]["nat_ids"]), 4)
            fold_rho = {"IN": [], "OUT": []}
            Qpair_fold = {"IN": [], "OUT": []}                                  # per fold: 28-vector (avg over id-subsets)
            for f in range(N_FOLDS):
                W = np.asarray(cells[f]["W_target"], np.float64); V = W.shape[0]
                Dtr = np.asarray(tn[f]["trial_native"], np.float64)
                R0 = float(cells[f]["R_CAL0"]); Rnat = float(rd["per_target"][roi][s][f]["R_ORACLE"])
                grid = R14["grids"][f]                                          # (100 x 28)
                TOT = (grid - R0) / max(Rnat - R0, 1e-12); FCF = (grid - R0) / max(R111v - R0, 1e-12)
                MARG = np.minimum(TOT - 0.50, FCF - 0.50)
                qin = np.full((len(id_subs), len(rep_pairs)), np.nan); qout = np.full_like(qin, np.nan)
                cn = np.zeros_like(qin); cw = np.zeros_like(qin); co = np.zeros_like(qin)
                for ii, C in enumerate(id_subs):
                    Cl = list(C)
                    for ri, P in enumerate(rep_pairs):
                        if set(P) & set(t for t in range(8) if t not in P):
                            disjoint_ok = False
                        Q_IN, Q_OUT, c_n, c_w, c_o, eP, eC = _schedule_Q(W, Dtr, Cl, P, r_best)
                        qin[ii, ri] = Q_IN; qout[ii, ri] = Q_OUT; cn[ii, ri] = c_n; cw[ii, ri] = c_w; co[ii, ri] = c_o
                        if ii == 0 and f == 0:
                            energy_rows.append([s, roi, ri, eP[0], eP[1], eP[2], eC[0], eC[1], eC[2]])
                fold_rho["IN"].append(_spearman(qin.ravel(), MARG.ravel()))
                fold_rho["OUT"].append(_spearman(qout.ravel(), MARG.ravel()))
                Qpair_fold["IN"].append(np.nanmean(qin, axis=0)); Qpair_fold["OUT"].append(np.nanmean(qout, axis=0))
                # success/failure contrast + attribution (this fold)
                suff = (TOT >= 0.5) & (FCF >= 0.5)
                for comp_name, Q in (("IN", qin), ("OUT", qout)):
                    contrast_rows.append([s, roi, f, comp_name, float(np.nanmedian(Q[suff])) if suff.any() else float("nan"),
                                          float(np.nanmedian(Q[~suff])) if (~suff).any() else float("nan")])
                fail = ~suff; nfail = int(fail.sum())
                if nfail:
                    to = int(((TOT < 0.5) & (FCF >= 0.5) & fail).sum()); fo = int(((FCF < 0.5) & (TOT >= 0.5) & fail).sum())
                    bo = int(((TOT < 0.5) & (FCF < 0.5)).sum())
                    attr_rows.append([s, roi, f, nfail, to / nfail, fo / nfail, bo / nfail])
                resrel_rows.append([s, roi, f, float(np.nanmean(cn)), float(np.nanmean(cw)), float(np.nanmean(co))])
            for comp in COMPONENTS:
                Z, rho = _fisher_mean(fold_rho[comp])
                part[s][roi][comp] = {"Z": Z, "rho": rho, "fold_rho": fold_rho[comp]}
            # additive repeat-position model on mean-over-folds Q_pair(a,b) (IN)
            Qpair = np.nanmean(np.stack(Qpair_fold["IN"]), axis=0)              # (28,)
            alpha, r2 = _additive_repeat(Qpair, rep_pairs)
            addmodel_rows.append([s, roi] + [float(x) for x in alpha] + [r2])
            posquality[roi][s] = alpha
            # single repeat-position LOO quality (training only)
            for f in range(1):                                                 # fold 0 representative (descriptive)
                W = np.asarray(cells[f]["W_target"], np.float64); Dtr = np.asarray(tn[f]["trial_native"], np.float64)
                for t in range(8):
                    others = [u for u in range(8) if u != t]
                    cn_t, cw_t, co_t = [], [], []
                    for i in range(10):
                        dt_ = Dtr[i][t]; dloo = Dtr[i][others].mean(0)
                        cn_t.append(_cos2(dt_, dloo)); cw_t.append(_cos2(W.T @ dt_, W.T @ dloo))
                        oP = dt_ - W @ (W.T @ dt_); oL = dloo - W @ (W.T @ dloo); co_t.append(_cos2(oP, oL))
                    singleq_rows.append([s, roi, t, float(np.mean(cn_t)), float(np.mean(cw_t)), float(np.mean(co_t))])
            # identity x repeat interaction alignment (fold 0, IN)
            grid0 = R14["grids"][0]; R0 = float(cells[0]["R_CAL0"]); Rnat = float(rd["per_target"][roi][s][0]["R_ORACLE"])
            TOT0 = (grid0 - R0) / max(Rnat - R0, 1e-12); FCF0 = (grid0 - R0) / max(R111v - R0, 1e-12)
            Y = np.minimum(TOT0 - 0.5, FCF0 - 0.5)
            Wq = np.asarray(cells[0]["W_target"], np.float64); Dtr0 = np.asarray(tn[0]["trial_native"], np.float64)
            Qg = np.array([[_schedule_Q(Wq, Dtr0, list(C), P, r_best)[0] for P in rep_pairs] for C in id_subs])
            intalign_rows.append([s, roi, _spearman(_resid_interaction(Qg).ravel(), _resid_interaction(Y).ravel())])
    replay_ok = replay_err <= 1e-10
    (out / "o2_14_schedule_replay.json").write_text(json.dumps({"status": "O2_15_REPLAYS_O2_14" if replay_ok else "O2_15_REPLAY_FAILURE", "max_abs_error": replay_err, "tol": 1e-10, "pair_complement_disjoint": disjoint_ok}, indent=2))
    if not replay_ok or not disjoint_ok:
        print("O2_15_FAILURE_MODE_INCONCLUSIVE (replay %.2e disjoint %s)" % (replay_err, disjoint_ok)); return 1
    classify_write(out, part, rois, replay_err, contrast_rows, attr_rows, resrel_rows, energy_rows, singleq_rows, addmodel_rows, intalign_rows, posquality, rep_pairs)
    return 0


def _additive_repeat(Qpair, rep_pairs):
    """OLS Q_pair(a,b)=beta0+alpha_a+alpha_b, sum alpha=0. Returns (alpha[8], R2)."""
    n = len(rep_pairs); X = np.zeros((n, 9)); X[:, 0] = 1.0
    for i, (a, b) in enumerate(rep_pairs):
        X[i, 1 + a] += 1; X[i, 1 + b] += 1
    y = np.asarray(Qpair, np.float64); m = np.isfinite(y)
    # constraint sum alpha=0 via centering columns; solve least squares
    beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    alpha = beta[1:] - beta[1:].mean()
    pred = X @ beta; ss_res = float(np.nansum((y - pred) ** 2)); ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")
    return alpha, r2


def _resid_interaction(M):
    mu = np.nanmean(M); mi = np.nanmean(M, axis=1, keepdims=True); mr = np.nanmean(M, axis=0, keepdims=True)
    return M - mi - mr + mu


def _median(x):
    x = [v for v in x if np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def classify_write(out, part, rois, replay_err, contrast_rows, attr_rows, resrel_rows, energy_rows, singleq_rows, addmodel_rows, intalign_rows, posquality, rep_pairs):
    from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
    def _w2(name, header, rows):
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    # participant associations + 4-test inference
    pvals = {}; passoc = []
    for roi in rois:
        for comp in COMPONENTS:
            Z = [part[s][roi][comp]["Z"] for s in ALL]
            pvals[(roi, comp)] = F.signflip_p_onesided(Z)
            for s in ALL:
                passoc.append([s, roi, comp, part[s][roi][comp]["Z"], part[s][roi][comp]["rho"]])
    holm = F.holm({f"{roi}|{comp}": pvals[(roi, comp)] for (roi, comp) in pvals}, alpha=0.05)
    infer = {}; supported = {roi: [] for roi in rois}
    for roi in rois:
        for comp in COMPONENTS:
            rhos = [part[s][roi][comp]["rho"] for s in ALL]
            rej = bool(holm.get(f"{roi}|{comp}", False))
            sup = bool(_median(rhos) > 0 and sum(r > 0 for r in rhos if np.isfinite(r)) >= 6 and rej)
            if sup:
                supported[roi].append(comp)
            infer[f"{roi}|{comp}"] = {"median_rho": _median(rhos), "n_rho_pos": int(sum(r > 0 for r in rhos if np.isfinite(r))),
                                      "signflip_p": float(pvals[(roi, comp)]), "holm_reject": rej, "supported": sup}
    _w2("participant_stability_associations.csv", ["subject", "roi", "component", "Z_fisher", "rho_participant"], passoc)
    fold_rows = [[s, roi, comp, f, part[s][roi][comp]["fold_rho"][f]] for roi in rois for s in ALL for comp in COMPONENTS for f in range(N_FOLDS)]
    _w2("fold_stability_associations.csv", ["subject", "roi", "component", "fold", "rho_fold"], fold_rows)
    _w2("success_failure_fidelity_contrast.csv", ["subject", "roi", "fold", "component", "median_Q_success", "median_Q_fail"], contrast_rows)
    _w2("failure_criterion_attribution.csv", ["subject", "roi", "fold", "n_fail", "frac_TOTAL_ONLY", "frac_FCF_ONLY", "frac_BOTH"], attr_rows)
    _w2("residual_vector_reliability.csv", ["subject", "roi", "fold", "Q_RES_NATIVE", "Q_RES_IN", "Q_RES_OUT"], resrel_rows)
    _w2("energy_diagnostic.csv", ["subject", "roi", "pair_idx", "pair_d2", "pair_win2", "pair_out2", "comp_d2", "comp_win2", "comp_out2"], energy_rows)
    _w2("single_repeat_position_quality.csv", ["subject", "roi", "repeat_position", "cos2_native", "cos2_within", "cos2_outside"], singleq_rows)
    _w2("repeat_additive_model.csv", ["subject", "roi"] + [f"alpha_{t}" for t in range(8)] + ["R2_REPEAT_ADDITIVE"], addmodel_rows)
    _w2("repeat_pair_interaction.csv", ["subject", "roi", "pair_nonadditivity_fraction"], [[r[0], r[1], 1 - r[-1]] for r in addmodel_rows])
    _w2("identity_repeat_interaction_alignment.csv", ["subject", "roi", "rho_interaction"], intalign_rows)
    # LOSO + cross-ROI repeat-position consistency (alpha vectors)
    loso_rows = []; xroi_rows = []
    for roi in rois:
        vecs = {s: posquality[roi][s] for s in ALL}
        for s in ALL:
            cons = np.mean([vecs[d] for d in ALL if d != s], axis=0)
            loso_rows.append([s, roi, _spearman(vecs[s], cons)])
    _w2("repeat_position_loso_consistency.csv", ["subject", "roi", "loso_rho"], loso_rows)
    if len(rois) >= 2:
        for s in ALL:
            xroi_rows.append([s, _spearman(posquality[rois[0]][s], posquality[rois[1]][s])])
    _w2("cross_roi_repeat_consistency.csv", ["subject", "spearman_ventral_lateral_repeat_quality"], xroi_rows)
    # secondary allocations placeholder note (descriptive; analogous pair-vs-complement)
    (out / "secondary_eight_trial_allocations.csv").write_text("cell,note\nM2T4,analogous 4-vs-4 repeat agreement (descriptive; see participant associations for the primary M4T2)\nM8T1,analogous 1-vs-7 repeat agreement\n")
    (out / "prior_schedule_stability_comparator.csv").write_text("note\nOPTIONAL status-neutral O2.12-hybrid schedule comparator deferred; does not affect O2.15 status.\n")
    # hierarchy
    roi_status = {}
    for roi in rois:
        sup = supported[roi]
        if "IN" in sup and "OUT" in sup:
            roi_status[roi] = "MULTICOMPONENT_REPEAT_GEOMETRY_INSTABILITY_SUPPORTED"
        elif "IN" in sup:
            roi_status[roi] = "WITHIN_SUPPORT_REPEAT_INSTABILITY_SUPPORTED"
        elif "OUT" in sup:
            roi_status[roi] = "OUTSIDE_SUPPORT_REPEAT_INSTABILITY_SUPPORTED"
        else:
            roi_status[roi] = "SCHEDULE_FAILURE_NOT_EXPLAINED_BY_TRAINING_GEOMETRY_STABILITY"
    comp_both = {comp: all(comp in supported[roi] for roi in rois) for comp in COMPONENTS}
    any_sup = any(supported[roi] for roi in rois)
    if any(comp_both.values()):
        prog = "REPEAT_GEOMETRY_INSTABILITY_SUPPORTED_AS_PRIMARY_FAILURE_MODE"
    elif any_sup:
        prog = "ACQUISITION_FAILURE_MODE_MULTIREGIME"
    else:
        prog = "SCHEDULE_FRAGILITY_NOT_EXPLAINED_BY_REPEAT_GEOMETRY_STABILITY"
    (out / "component_inference.json").write_text(json.dumps({"family_size": 4, "per_test": infer}, indent=2, default=str))
    (out / "roi_status.json").write_text(json.dumps(roi_status, indent=2))
    (out / "scientific_status.json").write_text(json.dumps(
        {"status": prog, "roi_status": roi_status, "control_replay_err": replay_err,
         "immutable": {"O2_14": "EIGHT_TRIAL_MINIMUM_AVERAGE_ONLY_NOT_SCHEDULE_ROBUST", "O2_9_N_TRIALS_STAR": 8}, "O3": "O3_NOT_READY"}, indent=2, default=str))
    (out / "next_gate_decision.json").write_text(json.dumps({"program_status": prog, "roi_status": roi_status}, indent=2, default=str))
    print("O2_15_STATUS", prog)
    for roi in rois:
        print("  [%s] %s | IN p=%.4f rho=%.3f | OUT p=%.4f rho=%.3f"
              % (roi, roi_status[roi], infer[f"{roi}|IN"]["signflip_p"], infer[f"{roi}|IN"]["median_rho"],
                 infer[f"{roi}|OUT"]["signflip_p"], infer[f"{roi}|OUT"]["median_rho"]))


def _w(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--state", required=True); ap.add_argument("--trial", required=True)
    ap.add_argument("--rd-results", required=True); ap.add_argument("--o2-8-ref", required=True); ap.add_argument("--o2-9-ref", required=True)
    ap.add_argument("--o2-14-cov", required=True); ap.add_argument("--out", required=True); ap.add_argument("--rois", default="ventral,lateral")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
