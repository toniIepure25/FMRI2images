"""O2.14 Target-Imagery Calibration Necessity / Robustness (frozen config 942fb9e). Acquisition-schedule
robustness audit of the sealed O2.9 8-observation minimum: for every prospectively enumerated acquisition
schedule (one balanced identity subset x one repeat-position subset), fit the EXACT O2.9 no-prior estimator
from that schedule alone and evaluate on the fold's two held-out identities. A schedule is SUFFICIENT iff
TOTAL>=0.50 and FCF>=0.50; a cell is SCHEDULE_ROBUST_SUFFICIENT iff the O2.9 average criterion holds AND median
participant schedule-coverage>=0.75 AND >=6/8 participants>=0.75 (0.75 frozen before outcomes). Also: identity/
repeat/interaction variance decomposition, M4T2 LOFO stability + repeat/identity diagnostics + jackknife, and
the direct-imagery necessity checklist. No new estimator, no prior, no p-value; participant is the unit.
Reuses O2.9 topk_energy_gram / _pairs_balanced verbatim (control replay == O2.9 to <=1e-10)."""
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

ALL = O9.ALL
N_FOLDS = O9.N_FOLDS
PRIMARY = [(2, 1), (2, 2), (4, 1), (6, 1), (2, 4), (4, 2), (8, 1)]
REFERENCE = [(10, 8)]
ALL_CELLS = PRIMARY + REFERENCE
M_SUB = O9.M_SUB; T_SUB = O9.T_SUB; D_OUT = 2
ROBUST_Q = 0.75
_G = None


def _geom(repo):
    global _G
    if _G is None:
        from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
        _G = G
    return _G


def _quantiles(x):
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return {q: float("nan") for q in ("Q10", "Q25", "median", "Q75", "Q90")}
    return {"Q10": float(np.quantile(x, 0.10)), "Q25": float(np.quantile(x, 0.25)), "median": float(np.median(x)),
            "Q75": float(np.quantile(x, 0.75)), "Q90": float(np.quantile(x, 0.90))}


def schedules_for(cell, simple, nat):
    M, T = cell
    ids = O9._pairs_balanced(simple, nat, M)
    reps = list(itertools.combinations(range(8), T))
    return ids, reps


# ---------------- per (subject, ROI, cell) schedule sweep ----------------
def subject_cell(s, roi, cell, cells, trialnat, rd, R111):
    """Returns per-fold arrays of schedule TOTAL (grid id x rep) + fold R0/Rnat + schedule sufficiency.
    Reuses the exact O2.9 estimator so grand-mean R_SCHEDULE == O2.9 R_COMP."""
    G = _G; M, T = cell
    r_best = int(cells[0]["r_best"])
    fold_grids = []; fold_R0 = []; fold_Rnat = []
    for f in range(N_FOLDS):
        cell0 = cells[f]; W = np.asarray(cell0["W_target"], np.float64); K = int(cell0["K"]); V = W.shape[0]
        dt = [np.asarray(x, np.float64) for x in list(cell0["deltas_test"])]
        dt_n2 = [float(d @ d) for d in dt]; g = [W.T @ d for d in dt]
        simple = list(cell0["simple_ids"]); nat = list(cell0["nat_ids"])
        Dtr = np.asarray(trialnat[f]["trial_native"], np.float64)
        R0 = float(cell0["R_CAL0"]); Rnat = float(rd["per_target"][roi][s][f]["R_ORACLE"])
        fold_R0.append(R0); fold_Rnat.append(Rnat)
        id_subs, rep_subs = schedules_for(cell, simple, nat)
        q = min(r_best, M)
        grid = np.full((len(id_subs), len(rep_subs)), np.nan)
        for ri, S in enumerate(rep_subs):
            Sl = list(S)
            di = np.stack([Dtr[i][Sl].mean(0) for i in range(10)])
            zc = np.stack([W.T @ di[i] for i in range(10)])
            dout = np.stack([di[i] - W @ zc[i] for i in range(10)])
            Zg = zc @ zc.T; Og = dout @ dout.T
            ZCG = np.stack([zc @ g[ti] for ti in range(2)]); OH = np.stack([dout @ dt[ti] for ti in range(2)])
            for ii, C in enumerate(id_subs):
                Ci = list(C); rc = []
                good = True
                for ti in range(2):
                    ew, _ = O9.topk_energy_gram(Zg[np.ix_(Ci, Ci)], ZCG[ti][Ci], q, max(K, M))
                    eo, _ = O9.topk_energy_gram(Og[np.ix_(Ci, Ci)], OH[ti][Ci], D_OUT, V)
                    if ew is None or eo is None:
                        good = False; break
                    rc.append((ew + eo) / dt_n2[ti])
                if good:
                    grid[ii, ri] = float(np.mean(rc))
        fold_grids.append(grid)
    return {"grids": fold_grids, "R0": fold_R0, "Rnat": fold_Rnat, "R111": R111, "r_best": r_best,
            "n_id": fold_grids[0].shape[0], "n_rep": fold_grids[0].shape[1]}


def _total_fcf(R, R0, Rnat, R111):
    return (R - R0) / max(Rnat - R0, 1e-12), (R - R0) / max(R111 - R0, 1e-12)


def aggregate_cell(res):
    """From per-fold TOTAL grids -> schedule coverage, margins, variance fractions, R_COMP (O2.9 replay),
    participant-average TOTAL/FCF."""
    grids = res["grids"]; R0 = res["R0"]; Rnat = res["Rnat"]; R111 = res["R111"]
    fold_cov = []; margins = []; f_id = []; f_rep = []; f_int = []
    fold_Rcomp = []; complete = True; exp = res["n_id"] * res["n_rep"]
    for f in range(N_FOLDS):
        Y = grids[f]
        if np.isnan(Y).any() or Y.size != exp:
            complete = False
        Rg = Y.copy()  # R_SCHEDULE grid
        TOT = (Rg - R0[f]) / max(Rnat[f] - R0[f], 1e-12)
        FCF = (Rg - R0[f]) / max(R111 - R0[f], 1e-12)
        suff = (TOT >= 0.50) & (FCF >= 0.50)
        fold_cov.append(float(np.mean(suff)))
        marg = np.minimum(TOT - 0.50, FCF - 0.50)
        margins.append(marg.ravel())
        fold_Rcomp.append(float(np.mean(Rg)))
        # variance decomposition on TOTAL grid
        mu = float(TOT.mean()); mu_i = TOT.mean(1); mu_r = TOT.mean(0)
        ss_id = res["n_rep"] * float(np.sum((mu_i - mu) ** 2))
        ss_rep = res["n_id"] * float(np.sum((mu_r - mu) ** 2))
        ss_int = float(np.sum((TOT - mu_i[:, None] - mu_r[None, :] + mu) ** 2))
        ss_tot = float(np.sum((TOT - mu) ** 2))
        if ss_tot > 1e-15:
            f_id.append(ss_id / ss_tot); f_rep.append(ss_rep / ss_tot); f_int.append(ss_int / ss_tot)
    coverage = float(np.mean(fold_cov))
    R_comp = float(np.mean(fold_Rcomp))
    R0m = float(np.mean(R0)); Rnatm = float(np.mean(Rnat))
    tot_avg, fcf_avg = _total_fcf(R_comp, R0m, Rnatm, R111)
    allmarg = np.concatenate(margins) if margins else np.array([])
    return {"coverage": coverage, "fold_cov": fold_cov, "cov_min": float(np.min(fold_cov)), "cov_max": float(np.max(fold_cov)),
            "cov_sd": float(np.std(fold_cov)), "margin_q": _quantiles(allmarg), "R_COMP": R_comp,
            "TOTAL_avg": float(np.clip(tot_avg, 0, 1)), "FCF_avg": float(np.clip(fcf_avg, 0, 1)),
            "TOTAL_avg_u": tot_avg, "FCF_avg_u": fcf_avg, "fold_Rcomp": fold_Rcomp, "R0": R0, "Rnat": Rnat, "R111": R111,
            "F_ID": float(np.mean(f_id)) if f_id else float("nan"), "F_REPEAT": float(np.mean(f_rep)) if f_rep else float("nan"),
            "F_INTERACTION": float(np.mean(f_int)) if f_int else float("nan"), "complete": complete}


def _median(x):
    return float(np.median(x)) if len(x) else float("nan")


def o2_9_sufficient(tots, fcfs):
    return (_median(tots) >= 0.50 and sum(t >= 0.5 for t in tots) >= 6 and _median(fcfs) >= 0.50 and sum(x >= 0.5 for x in fcfs) >= 6)


# ---------------- driver ----------------
def run(a):
    G = _geom(a.repo)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    rd = json.loads(Path(a.rd_results).read_text())
    R111 = {(r["subject"], r["roi"]): float(r["R111"]) for r in csv.DictReader(open(a.o2_8_ref))}
    o9ref = {(r["subject"], r["roi"], int(r["M"]), int(r["T"])): float(r["R_COMP"]) for r in csv.DictReader(open(a.o2_9_ref))}
    state = Path(a.state); trial = Path(a.trial)
    agg = {s: {roi: {} for roi in rois} for s in ALL}
    replay_err = 0.0
    for roi in rois:
        for s in ALL:
            cells = [dict(np.load(state / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            tn = [dict(np.load(trial / f"trialnat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            for cell in ALL_CELLS:
                res = subject_cell(s, roi, cell, cells, tn, rd, R111[(s, roi)])
                A = aggregate_cell(res)
                A["res"] = res
                agg[s][roi][cell] = A
                replay_err = max(replay_err, abs(A["R_COMP"] - o9ref[(s, roi, cell[0], cell[1])]))
    replay_ok = replay_err <= 1e-10
    (out / "o2_9_replay.json").write_text(json.dumps(
        {"status": "O2_14_REPLAYS_O2_9_RESOURCE_FRONTIER" if replay_ok else "O2_14_REPLAY_FAILURE", "max_abs_error": replay_err, "tol": 1e-10}, indent=2))
    if not replay_ok:
        print("O2_14 replay FAILED err=%.2e" % replay_err); return 1
    classify_and_write(out, agg, rois, replay_err, a)
    return 0


def classify_and_write(out, agg, rois, replay_err, a):
    def _w(name, header, rows):
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    # participant / fold coverage + robustness rows
    pcov, fcov, marg_rows, var_rows, rob_rows = [], [], [], [], []
    for roi in rois:
        for s in ALL:
            for cell in ALL_CELLS:
                A = agg[s][roi][cell]
                pcov.append([s, roi, cell[0], cell[1], cell[0] * cell[1], A["coverage"], A["cov_min"], A["cov_max"], A["cov_sd"], int(A["complete"])])
                for f in range(N_FOLDS):
                    fcov.append([s, roi, cell[0], cell[1], f, A["fold_cov"][f]])
                q = A["margin_q"]
                marg_rows.append([s, roi, cell[0], cell[1], q["Q10"], q["Q25"], q["median"], q["Q75"], q["Q90"]])
                var_rows.append([s, roi, cell[0], cell[1], A["F_ID"], A["F_REPEAT"], A["F_INTERACTION"]])
    _w("participant_schedule_coverage.csv", ["subject", "roi", "M", "T", "N", "SCHEDULE_COVERAGE", "cov_min_fold", "cov_max_fold", "cov_sd_fold", "complete"], pcov)
    _w("fold_schedule_coverage.csv", ["subject", "roi", "M", "T", "fold", "SCHEDULE_COVERAGE_FOLD"], fcov)
    _w("robustness_margin.csv", ["subject", "roi", "M", "T", "Q10", "Q25", "median", "Q75", "Q90"], marg_rows)
    _w("identity_repeat_variance_decomposition.csv", ["subject", "roi", "M", "T", "F_ID", "F_REPEAT", "F_INTERACTION"], var_rows)
    _w("schedule_level_results.csv", ["subject", "roi", "M", "T", "N", "R_COMP", "TOTAL_avg", "FCF_avg", "SCHEDULE_COVERAGE"],
       [[s, roi, c[0], c[1], c[0] * c[1], agg[s][roi][c]["R_COMP"], agg[s][roi][c]["TOTAL_avg"], agg[s][roi][c]["FCF_avg"], agg[s][roi][c]["coverage"]] for roi in rois for s in ALL for c in ALL_CELLS])
    # resource robustness table + classification
    robust = {}
    for roi in rois:
        for cell in PRIMARY:
            tots = [agg[s][roi][cell]["TOTAL_avg"] for s in ALL]; fcfs = [agg[s][roi][cell]["FCF_avg"] for s in ALL]
            covs = [agg[s][roi][cell]["coverage"] for s in ALL]
            o9ok = o2_9_sufficient(tots, fcfs)
            cov_ok = (_median(covs) >= ROBUST_Q and sum(c >= ROBUST_Q for c in covs) >= 6)
            complete = all(agg[s][roi][cell]["complete"] for s in ALL)
            rob = bool(o9ok and cov_ok and complete)
            robust[(roi, cell)] = rob
            rob_rows.append([roi, cell[0], cell[1], cell[0] * cell[1], o9ok, _median(covs), int(sum(c >= ROBUST_Q for c in covs)), cov_ok, int(complete), rob])
    _w("resource_robustness_table.csv", ["roi", "M", "T", "N", "o2_9_avg_sufficient", "median_coverage", "n_cov_ge075", "coverage_ok", "complete", "SCHEDULE_ROBUST_SUFFICIENT"], rob_rows)
    common_robust = [cell for cell in PRIMARY if all(robust[(roi, cell)] for roi in rois)]
    n_star = min((M * T for (M, T) in common_robust), default=None)
    minimal = [c for c in common_robust if c[0] * c[1] == n_star] if n_star is not None else []
    # 8-trial allocation comparison
    alloc = []
    for cell in [(2, 4), (4, 2), (8, 1)]:
        for roi in rois:
            tots = [agg[s][roi][cell]["TOTAL_avg"] for s in ALL]; covs = [agg[s][roi][cell]["coverage"] for s in ALL]
            q = _quantiles(np.concatenate([np.array(list(agg[s][roi][cell]["margin_q"].values())) for s in ALL]))
            alloc.append([cell[0], cell[1], roi, _median(tots), _median(covs), _median([agg[s][roi][cell]["margin_q"]["median"] for s in ALL]), _median([agg[s][roi][cell]["cov_sd"] for s in ALL])])
    _w("eight_trial_allocation_comparison.csv", ["M", "T", "roi", "median_TOTAL_avg", "median_coverage", "median_margin_median", "median_cov_sd_fold"], alloc)
    # M4T2 LOFO
    lofo_rows = []; lofo_stable = True
    for omit in range(N_FOLDS):
        keep = [f for f in range(N_FOLDS) if f != omit]
        okrois = {}
        for roi in rois:
            tots, fcfs = [], []
            for s in ALL:
                A = agg[s][roi][(4, 2)]
                Rc = float(np.mean([A["fold_Rcomp"][f] for f in keep]))
                R0 = float(np.mean([A["R0"][f] for f in keep])); Rn = float(np.mean([A["Rnat"][f] for f in keep]))
                t, fc = _total_fcf(Rc, R0, Rn, A["R111"])
                tots.append(float(np.clip(t, 0, 1))); fcfs.append(float(np.clip(fc, 0, 1)))
            okrois[roi] = o2_9_sufficient(tots, fcfs)
            lofo_rows.append([omit, roi, _median(tots), _median(fcfs), okrois[roi]])
        if not all(okrois.values()):
            lofo_stable = False
    _w("outer_fold_leave_one_out.csv", ["omitted_fold", "roi", "median_TOTAL", "median_FCF", "o2_9_sufficient"], lofo_rows)
    # M4T2 repeat-position + identity-subset diagnostics (participant-first over folds, mean over the other axis)
    rep_rows, id_rows = [], []
    for roi in rois:
        A0 = agg[ALL[0]][roi][(4, 2)]["res"]; nrep = A0["n_rep"]; nid = A0["n_id"]
        for ri in range(nrep):
            covs = []; tots = []
            for s in ALL:
                A = agg[s][roi][(4, 2)]
                per_fold = []
                for f in range(N_FOLDS):
                    col = A["res"]["grids"][f][:, ri]
                    TOT = (col - A["R0"][f]) / max(A["Rnat"][f] - A["R0"][f], 1e-12)
                    FCF = (col - A["R0"][f]) / max(A["R111"] - A["R0"][f], 1e-12)
                    per_fold.append((float(np.mean(TOT)), float(np.mean((TOT >= 0.5) & (FCF >= 0.5)))))
                tots.append(float(np.mean([x[0] for x in per_fold]))); covs.append(float(np.mean([x[1] for x in per_fold])))
            rep_rows.append([roi, ri, _median(tots), _median(covs)])
        for ii in range(nid):
            covs = []; tots = []
            for s in ALL:
                A = agg[s][roi][(4, 2)]
                per_fold = []
                for f in range(N_FOLDS):
                    row = A["res"]["grids"][f][ii, :]
                    TOT = (row - A["R0"][f]) / max(A["Rnat"][f] - A["R0"][f], 1e-12)
                    FCF = (row - A["R0"][f]) / max(A["R111"] - A["R0"][f], 1e-12)
                    per_fold.append((float(np.mean(TOT)), float(np.mean((TOT >= 0.5) & (FCF >= 0.5)))))
                tots.append(float(np.mean([x[0] for x in per_fold]))); covs.append(float(np.mean([x[1] for x in per_fold])))
            id_rows.append([roi, ii, _median(tots), _median(covs)])
    _w("repeat_position_diagnostic.csv", ["roi", "repeat_pair_idx", "median_TOTAL", "median_coverage"], rep_rows)
    _w("identity_subset_diagnostic.csv", ["roi", "identity_subset_idx", "median_TOTAL", "median_coverage"], id_rows)
    # participant jackknife (M4T2)
    jk_rows = []
    for omit in ALL:
        for roi in rois:
            keep = [s for s in ALL if s != omit]
            tots = [agg[s][roi][(4, 2)]["TOTAL_avg"] for s in keep]; covs = [agg[s][roi][(4, 2)]["coverage"] for s in keep]
            jk_rows.append([omit, roi, _median(tots), _median(covs)])
    _w("participant_jackknife.csv", ["omitted_subject", "roi", "median_TOTAL", "median_coverage"], jk_rows)
    # non-imagery evidence ladder (sealed metrics, descriptive references)
    _w("nonimagery_evidence_ladder.csv", ["stage", "N_obs", "method", "sealed_status"],
       [["O2.10", 0, "donor transfer", "COMPOSITE_TARGET_STATE_GEOMETRY_SUBJECT_SPECIFIC_DOMINANT"],
        ["O2.11", 0, "perception-only prediction", "SUBJECT_SPECIFIC_GEOMETRY_NOT_PREDICTABLE_FROM_TESTED_PERCEPTION_PHENOTYPE"],
        ["O2.13", 0, "covariate-corrected prediction", "NO_REPRODUCIBLE_TESTED_COVARIATE_STRUCTURE"],
        ["O2.9", 2, "imagery-only sub-eight", "MINIMAL_COMPOSITE_TARGET_STATE_CALIBRATION_ESTABLISHED (no common sub-8)"],
        ["O2.12", "2-6", "fixed perception-prior hybrid", "PERCEPTION_PRIOR_BENEFICIAL_BUT_EIGHT_TRIAL_MINIMUM_NOT_REDUCED"],
        ["O2.9", 8, "M4T2", "MINIMAL_COMPOSITE_TARGET_STATE_CALIBRATION_ESTABLISHED (N_TRIALS_STAR=8)"],
        ["O2.9", 80, "M10T8 full-resource", "reference"]])
    # classification
    m4t2_common = all(robust[(roi, (4, 2))] for roi in rois)
    sub8_common = [c for c in common_robust if c[0] * c[1] < 8]
    roi_status = {}
    for roi in rois:
        r_cells = [c for c in PRIMARY if robust[(roi, c)]]
        if robust[(roi, (4, 2))] and not any(c[0] * c[1] < 8 for c in r_cells):
            roi_status[roi] = "EIGHT_TRIAL_TARGET_IMAGERY_CALIBRATION_ROBUST"
        elif any(c[0] * c[1] < 8 for c in r_cells):
            roi_status[roi] = "SUBEIGHT_SCHEDULE_ROBUST_CALIBRATION_EMERGES"
        elif o2_9_sufficient([agg[s][roi][(4, 2)]["TOTAL_avg"] for s in ALL], [agg[s][roi][(4, 2)]["FCF_avg"] for s in ALL]):
            roi_status[roi] = "EIGHT_TRIAL_MINIMUM_AVERAGE_ONLY_NOT_SCHEDULE_ROBUST"
        else:
            roi_status[roi] = "TARGET_IMAGERY_CALIBRATION_ROBUSTNESS_INCONCLUSIVE"
    if sub8_common:
        primary = "SUBEIGHT_SCHEDULE_ROBUST_CALIBRATION_EMERGES"
    elif m4t2_common and lofo_stable:
        primary = "EIGHT_TRIAL_TARGET_IMAGERY_CALIBRATION_ROBUST"
    elif roi_status["ventral"] != roi_status["lateral"]:
        primary = "TARGET_IMAGERY_CALIBRATION_ROBUSTNESS_MULTIREGIME"
    elif all(o2_9_sufficient([agg[s][roi][(4, 2)]["TOTAL_avg"] for s in ALL], [agg[s][roi][(4, 2)]["FCF_avg"] for s in ALL]) for roi in rois):
        primary = "EIGHT_TRIAL_MINIMUM_AVERAGE_ONLY_NOT_SCHEDULE_ROBUST"
    else:
        primary = "O2_14_CALIBRATION_ROBUSTNESS_INCONCLUSIVE"
    # direct-imagery necessity checklist
    m4t2_replays = all(o2_9_sufficient([agg[s][roi][(4, 2)]["TOTAL_avg"] for s in ALL], [agg[s][roi][(4, 2)]["FCF_avg"] for s in ALL]) for roi in rois)
    checklist = {"1_no_N0_sufficient": True, "2_no_common_subeight_o2_9": True, "3_no_o2_12_prior_enabled_subeight": True,
                 "4_o2_13_no_reproducible_covariate": True, "5_m4t2_common_replay_sufficient": bool(m4t2_replays),
                 "6_m4t2_common_schedule_robust": bool(m4t2_common), "7_m4t2_lofo_stable": bool(lofo_stable)}
    necessity = all(checklist.values())
    (out / "direct_imagery_necessity_checklist.json").write_text(json.dumps({"checklist": checklist, "necessity_supported": necessity}, indent=2))
    (out / "robust_minimal_resource_set.json").write_text(json.dumps(
        {"N_TRIALS_ROBUST_STAR": n_star, "ROBUST_MINIMAL_RESOURCE_SET": [{"M": M, "T": T} for (M, T) in minimal],
         "common_robust_cells": [{"M": M, "T": T, "N": M * T} for (M, T) in common_robust], "historical_N_TRIALS_STAR": 8, "M4T2_LOFO_STABLE": bool(lofo_stable)}, indent=2))
    (out / "roi_status.json").write_text(json.dumps(roi_status, indent=2))
    if primary == "EIGHT_TRIAL_TARGET_IMAGERY_CALIBRATION_ROBUST" and necessity:
        prog = "DIRECT_TARGET_IMAGERY_NECESSITY_ROBUST_AT_EIGHT_TRIALS_WITHIN_TESTED_REPRESENTATIONS"
    elif primary == "EIGHT_TRIAL_TARGET_IMAGERY_CALIBRATION_ROBUST":
        prog = "EIGHT_TRIAL_CALIBRATION_ROBUST_WITHOUT_NECESSITY_CONCLUSION"
    else:
        prog = primary
    (out / "scientific_status.json").write_text(json.dumps(
        {"primary_status": primary, "program_status": prog, "roi_status": roi_status, "N_TRIALS_ROBUST_STAR": n_star,
         "M4T2_LOFO_STABLE": bool(lofo_stable), "necessity_supported": necessity, "control_replay_err": replay_err,
         "immutable": {"O2_13": "NO_REPRODUCIBLE_TESTED_COVARIATE_STRUCTURE", "O2_9_N_TRIALS_STAR": 8}, "O3": "O3_NOT_READY"}, indent=2, default=str))
    (out / "next_gate_decision.json").write_text(json.dumps({"primary_status": primary, "program_status": prog, "N_TRIALS_ROBUST_STAR": n_star}, indent=2, default=str))
    print("O2_14_STATUS", primary, "| PROGRAM", prog, "| N_TRIALS_ROBUST_STAR", n_star, "| LOFO_STABLE", lofo_stable, "| necessity", necessity)
    for roi in rois:
        print("  [%s] %s" % (roi, roi_status[roi]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--state", required=True); ap.add_argument("--trial", required=True)
    ap.add_argument("--rd-results", required=True); ap.add_argument("--o2-8-ref", required=True); ap.add_argument("--o2-9-ref", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--rois", default="ventral,lateral")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
