"""O2.9 Minimal Composite Target-State Calibration (frozen config 037686de). OPERATIONAL resource frontier
over the M x T grid: direct-SVD within-support target subspace (q=min(r_best,M); NO Procrustes) + fixed D=2
outside-support basis, composite P_COMP = P_IN + P_OUT (orthogonal ranges). Consumes ONLY sealed state +
certified O2.7 trial residuals. No nulls/p-values (O2.8 already established the components). Gram-optimized
evaluation, mathematically equivalent to the direct SVD/projector (data-free equivalence tested)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ALL = [f"subj0{i}" for i in range(1, 9)]
N_FOLDS = 6
M_GRID = [2, 4, 6, 8, 10]
T_GRID = [1, 2, 4, 8]
M_SUB = {2: 25, 4: 100, 6: 100, 8: 25, 10: 1}
T_SUB = {1: 8, 2: 28, 4: 70, 8: 1}
D_OUT = 2
EPS = np.finfo(np.float64).eps
_G = None


def _geom(repo):
    global _G
    if _G is None:
        sys.path.insert(0, str(Path(repo) / "src"))
        from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
        _G = G
    return _G


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _canon(U):
    U = np.asarray(U, np.float64).copy()
    for k in range(U.shape[1]):
        j = int(np.argmax(np.abs(U[:, k])))
        if U[j, k] < 0:
            U[:, k] *= -1.0
    return U


def _num_rank(sv, maxshape):
    if sv.size == 0:
        return 0
    tol = EPS * maxshape * float(sv[0])
    return int(np.sum(sv > tol))


def topk_energy_gram(Gsub, hsub, k, maxshape):
    """Gram fast path: energy = sum over top-k singular directions of (v_j . h)^2 / lambda_j, where (lambda,v)
    = eig(Gram). Returns (energy, numerical_rank) or (None, rank) if rank<k."""
    lam, V = np.linalg.eigh(np.asarray(Gsub, np.float64))
    order = np.argsort(lam)[::-1]
    lam = lam[order]; V = V[:, order]
    sv = np.sqrt(np.clip(lam, 0.0, None))
    rank = _num_rank(sv, maxshape)
    if rank < k:
        return None, rank
    e = 0.0
    for j in range(k):
        if lam[j] > 0:
            e += float((V[:, j] @ hsub) ** 2 / lam[j])
    return e, rank


# --- direct-path helpers (used only in the data-free equivalence test) -------
def within_basis_direct(Zc, q):
    U, s, _ = np.linalg.svd(np.asarray(Zc, np.float64).T, full_matrices=False)   # (K x M).T -> left sv in K
    if _num_rank(s, max(Zc.shape)) < q:
        return None
    return _canon(U[:, :q])


def outside_basis_direct(Dout, D):
    U, s, _ = np.linalg.svd(np.asarray(Dout, np.float64), full_matrices=False)
    if _num_rank(s, max(Dout.shape)) < D:
        return None
    return _canon(U[:, :D])


def _pairs_balanced(simple, nat, M):
    h = M // 2
    return [tuple(cs) + tuple(cn) for cs in itertools.combinations(sorted(simple), h)
            for cn in itertools.combinations(sorted(nat), h)]


# --- per (subject, ROI) resource frontier ------------------------------------
def subject_roi(s, roi, cells, trialnat, extnat, rd, leak, cert, want_sealed_replay=False):
    Rnat_folds = [float(rd["per_target"][roi][s][f]["R_ORACLE"]) for f in range(N_FOLDS)]
    R0_folds = [float(cells[f]["R_CAL0"]) for f in range(N_FOLDS)]
    r_best = int(cells[0]["r_best"])
    # accumulate per (M,T): list over (fold, repeat-subset) of mean-over-identity-subset R_COMP; evaluable flag
    acc = {(M, T): {"vals": [], "evaluable": True, "rank_fail": 0} for M in M_GRID for T in T_GRID}
    sealed_R = {}                                                            # M10,D2 via delta_native (replay)
    spectra = []
    for f in range(N_FOLDS):
        cell = cells[f]
        W = np.asarray(cell["W_target"], np.float64); K = int(cell["K"])
        dt = [np.asarray(x, np.float64) for x in list(cell["deltas_test"])]
        dt_n2 = [float(d @ d) for d in dt]; g = [W.T @ d for d in dt]         # K-space test coords
        simple = list(cell["simple_ids"]); nat = list(cell["nat_ids"])
        Dtr = np.asarray(trialnat[f]["trial_native"], np.float64)            # (10 x 8 x V)
        maxshape_in = lambda M: max(K, M)
        V = W.shape[0]
        # sealed-centroid replay (M10, D2, q=r_best) from delta_native
        if want_sealed_replay:
            Dn = np.asarray(extnat[f]["delta_native"], np.float64)
            zc = np.stack([W.T @ Dn[i] for i in range(10)])                  # (10 x K)
            dout = np.stack([Dn[i] - W @ zc[i] for i in range(10)])          # (10 x V)
            Zg = zc @ zc.T; Og = dout @ dout.T
            for ti in range(2):
                zcg = zc @ g[ti]; oh = dout @ dt[ti]
                ew, _ = topk_energy_gram(Zg, zcg, min(r_best, 10), maxshape_in(10))
                eo, _ = topk_energy_gram(Og, oh, D_OUT, V)
                sealed_R.setdefault(f, []).append((ew + eo) / dt_n2[ti])
        for T in T_GRID:
            for S in itertools.combinations(range(8), T):
                Sl = list(S)
                di = np.stack([Dtr[i][Sl].mean(0) for i in range(10)])       # (10 x V) limited residuals
                zc = np.stack([W.T @ di[i] for i in range(10)])              # (10 x K)
                dout = np.stack([di[i] - W @ zc[i] for i in range(10)])      # (10 x V)
                leak["max"] = max(leak["max"], float(np.max([abs(W.T @ dout[i]).max() for i in range(10)])))
                Zg = zc @ zc.T; Og = dout @ dout.T
                # cross vectors to the 2 held-out tests
                ZCG = np.stack([zc @ g[ti] for ti in range(2)])              # (2 x 10)
                OH = np.stack([dout @ dt[ti] for ti in range(2)])            # (2 x 10)
                if f == 0 and T == 8 and want_sealed_replay is False:
                    svz = np.sqrt(np.clip(np.linalg.eigvalsh(Zg)[::-1], 0, None))
                    svo = np.sqrt(np.clip(np.linalg.eigvalsh(Og)[::-1], 0, None))
                    q10 = min(r_best, 10)
                    spectra.append([s, roi, f, r_best, q10,
                                    float(svz[q10] / svz[q10 - 1]) if len(svz) > q10 and svz[q10 - 1] > 0 else "",
                                    float(svo[0]), float(svo[1] if len(svo) > 1 else 0.0),
                                    float(svo[2] if len(svo) > 2 else 0.0),
                                    float(svo[2] / svo[1]) if len(svo) > 2 and svo[1] > 0 else ""])
                for M in M_GRID:
                    q = min(r_best, M)
                    subs = _pairs_balanced(simple, nat, M)
                    if not subs:
                        acc[(M, T)]["evaluable"] = False; continue
                    vals = []
                    for C in subs:
                        Ci = list(C)
                        rc = []
                        ok = True
                        for ti in range(2):
                            ew, rw = topk_energy_gram(Zg[np.ix_(Ci, Ci)], ZCG[ti][Ci], q, maxshape_in(M))
                            eo, ro = topk_energy_gram(Og[np.ix_(Ci, Ci)], OH[ti][Ci], D_OUT, V)
                            if ew is None or eo is None:
                                ok = False; break
                            rc.append((ew + eo) / dt_n2[ti])
                        if not ok:
                            acc[(M, T)]["evaluable"] = False; acc[(M, T)]["rank_fail"] += 1; vals = None; break
                        vals.append(float(np.mean(rc)))
                    if vals is not None:
                        acc[(M, T)]["vals"].append(float(np.mean(vals)))
    R0 = float(np.mean(R0_folds)); Rnat = float(np.mean(Rnat_folds))
    out = {"R0": R0, "R_native": Rnat, "r_best": r_best, "spectra": spectra, "cells": {}}
    if want_sealed_replay:
        out["sealed_M10D2"] = float(np.mean([np.mean(v) for v in sealed_R.values()]))
    for (M, T), a in acc.items():
        exp_sched = N_FOLDS * M_SUB[M] * T_SUB[T]
        evaluable = a["evaluable"] and len(a["vals"]) == N_FOLDS * T_SUB[T]   # all fold x repeat-subset present
        if not evaluable:
            out["cells"][(M, T)] = {"evaluable": False, "rank_fail": a["rank_fail"], "n_present": len(a["vals"]), "expected_foldrep": N_FOLDS * T_SUB[T]}
            continue
        R_COMP = float(np.mean(a["vals"]))
        tot = (R_COMP - R0) / max(Rnat - R0, 1e-12)
        fcf = (R_COMP - R0) / max(out.get("R111_ref", {}).get((M, T), np.nan) if False else 1.0, 1e-12)  # filled later
        out["cells"][(M, T)] = {"evaluable": True, "R_COMP": R_COMP, "N_TRIALS": M * T, "q": min(M, r_best),
                                "TOTAL_RECOVERY_unclipped": tot, "TOTAL_RECOVERY": float(np.clip(tot, 0, 1))}
    return out


# --- aggregation / status ----------------------------------------------------
def _median(x):
    return float(np.median(x)) if len(x) else float("nan")


def finalize(per_subj, R111, rois):
    """Attach FULL_CAPACITY_FRACTION (needs R111 per participant) and compute group cell success."""
    group = {roi: {} for roi in rois}
    for roi in rois:
        for M in M_GRID:
            for T in T_GRID:
                cellvals = [per_subj[s][roi]["cells"][(M, T)] for s in ALL]
                if not all(c.get("evaluable") for c in cellvals):
                    group[roi][(M, T)] = {"evaluable": False, "n_evaluable": sum(c.get("evaluable", False) for c in cellvals)}
                    continue
                tot, fcf = [], []
                for s in ALL:
                    c = per_subj[s][roi]["cells"][(M, T)]; R0 = per_subj[s][roi]["R0"]
                    r111 = R111[(s, roi)]
                    ff = (c["R_COMP"] - R0) / max(r111 - R0, 1e-12)
                    c["FULL_CAPACITY_FRACTION_unclipped"] = ff; c["FULL_CAPACITY_FRACTION"] = float(np.clip(ff, 0, 1))
                    tot.append(c["TOTAL_RECOVERY"]); fcf.append(c["FULL_CAPACITY_FRACTION"])
                suff = (_median(tot) >= 0.50 and sum(t >= 0.5 for t in tot) >= 6
                        and _median(fcf) >= 0.50 and sum(x >= 0.5 for x in fcf) >= 6)
                group[roi][(M, T)] = {"evaluable": True, "n_evaluable": 8, "N_TRIALS": M * T,
                                      "median_TOTAL": _median(tot), "n_TOTAL_ge0.5": int(sum(t >= 0.5 for t in tot)),
                                      "median_FCF": _median(fcf), "n_FCF_ge0.5": int(sum(x >= 0.5 for x in fcf)),
                                      "sufficient": bool(suff)}
    # ROI status
    roi_status = {}
    for roi in rois:
        cells = group[roi]
        any_suff = any(cells[(M, T)].get("sufficient") for M in M_GRID for T in T_GRID)
        full = cells[(10, 8)]
        if any_suff:
            roi_status[roi] = "COMPOSITE_OPERATIONAL_CALIBRATION_ESTABLISHED"
        elif full.get("evaluable") and not full.get("sufficient"):
            roi_status[roi] = "COMPOSITE_OPERATIONAL_RECOVERY_NOT_ESTABLISHED_AT_FULL_RESOURCE"
        else:
            roi_status[roi] = "COMPOSITE_RESOURCE_FRONTIER_INCONCLUSIVE"
    # common success
    common = [(M, T) for M in M_GRID for T in T_GRID
              if all(group[roi][(M, T)].get("sufficient") for roi in rois)]
    n_star = min((M * T for (M, T) in common), default=None)
    minimal_set = [(M, T) for (M, T) in common if M * T == n_star] if n_star is not None else []
    # common pareto (nondominated)
    pareto = []
    for (M, T) in common:
        dominated = any((M2 <= M and T2 <= T and (M2 < M or T2 < T)) for (M2, T2) in common)
        if not dominated:
            pareto.append((M, T))
    # program status
    if all(roi_status[roi] == "COMPOSITE_RESOURCE_FRONTIER_INCONCLUSIVE" for roi in rois):
        prog = "O2_9_COMPOSITE_CALIBRATION_INCONCLUSIVE"
    elif common:
        prog = "MINIMAL_COMPOSITE_TARGET_STATE_CALIBRATION_ESTABLISHED"
    elif all(any(group[roi][(M, T)].get("sufficient") for M in M_GRID for T in T_GRID) for roi in rois):
        prog = "COMPOSITE_TARGET_STATE_CALIBRATION_MULTIREGIME"
    elif any(roi_status[roi] == "COMPOSITE_OPERATIONAL_RECOVERY_NOT_ESTABLISHED_AT_FULL_RESOURCE" for roi in rois):
        prog = "COMPOSITE_OPERATIONAL_RECOVERY_NOT_ESTABLISHED"
    else:
        prog = "O2_9_COMPOSITE_CALIBRATION_INCONCLUSIVE"
    return group, roi_status, common, n_star, minimal_set, pareto, prog


def run(a):
    _geom(a.repo)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    state_dir, trial_dir, ext_dir = Path(a.state), Path(a.trial), Path(a.ext)
    rd = json.loads(Path(a.rd_results).read_text()); rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    # provenance
    stateman = {f["name"]: f for f in json.loads(Path(a.manifest).read_text())["files"]}
    tman = json.loads(Path(a.trial_manifest).read_text()); exman = json.loads(Path(a.ext_manifest).read_text())
    tby = {f["name"]: f for f in tman["files"]}; eby = {f["name"]: f for f in exman["files"]}
    prov = {"rd_match": (a.rd_results_sha256 is None or _sha(a.rd_results) == a.rd_results_sha256),
            "state_ok": all(stateman.get(f"cell_{s}_{r}_fold{f}.npz", {}).get("sha256") == _sha(state_dir / f"cell_{s}_{r}_fold{f}.npz") for r in rois for s in ALL for f in range(N_FOLDS)),
            "trial_ok": all(tby.get(f"trialnat_{s}_{r}_fold{f}.npz", {}).get("sha256") == _sha(trial_dir / f"trialnat_{s}_{r}_fold{f}.npz") for r in rois for s in ALL for f in range(N_FOLDS)),
            "ext_ok": all(eby.get(f"deltanat_{s}_{r}_fold{f}.npz", {}).get("sha256") == _sha(ext_dir / f"deltanat_{s}_{r}_fold{f}.npz") for r in rois for s in ALL for f in range(N_FOLDS))}
    (out / "sealed_state_verification.json").write_text(json.dumps(prov, indent=2))
    if not all(prov.values()):
        print("O2_9_COMPOSITE_CALIBRATION_INCONCLUSIVE (provenance)"); return 1
    R111 = {(r["subject"], r["roi"]): float(r["R111"]) for r in csv.DictReader(open(a.o2_8_ref))}

    leak = {"max": 0.0}; cert = {}
    per = {s: {} for s in ALL}; replay_err = 0.0
    for roi in rois:
        for s in ALL:
            cells = [dict(np.load(state_dir / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            tn = [dict(np.load(trial_dir / f"trialnat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            en = [dict(np.load(ext_dir / f"deltanat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            r = subject_roi(s, roi, cells, tn, en, rd, leak, cert, want_sealed_replay=True)
            per[s][roi] = r
            replay_err = max(replay_err, abs(r["sealed_M10D2"] - R111[(s, roi)]))
    replay_ok = replay_err <= 1e-10
    (out / "o2_8_r111_replay.json").write_text(json.dumps(
        {"status": "O2_9_FULL_COMPOSITE_REPLAYS_O2_8_R111" if replay_ok else "O2_9_R111_REPLAY_FAILURE", "max_abs_error": replay_err, "tol": 1e-10}, indent=2))
    if not replay_ok:
        print("O2_9 R111 replay failed err=%.2e" % replay_err); return 1

    group, roi_status, common, n_star, minimal_set, pareto, prog = finalize(per, R111, rois)

    def _w(name, header, rows):
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    rc_rows, part_rows, tr_rows, fcf_rows, roi_fr = [], [], [], [], []
    for roi in rois:
        for M in M_GRID:
            for T in T_GRID:
                g = group[roi][(M, T)]
                roi_fr.append([roi, M, T, M * T, g.get("evaluable"), g.get("median_TOTAL", ""), g.get("n_TOTAL_ge0.5", ""),
                               g.get("median_FCF", ""), g.get("n_FCF_ge0.5", ""), g.get("sufficient", "")])
        for s in ALL:
            for M in M_GRID:
                for T in T_GRID:
                    c = per[s][roi]["cells"][(M, T)]
                    if c.get("evaluable"):
                        part_rows.append([s, roi, M, T, M * T, c["q"], c["R_COMP"], c["TOTAL_RECOVERY"], c.get("FULL_CAPACITY_FRACTION", "")])
    _w("roi_resource_frontier.csv", ["roi", "M", "T", "N_TRIALS", "evaluable", "median_TOTAL", "n_TOTAL_ge0.5", "median_FCF", "n_FCF_ge0.5", "sufficient"], roi_fr)
    _w("participant_resource_frontier.csv", ["subject", "roi", "M", "T", "N_TRIALS", "q", "R_COMP", "TOTAL_RECOVERY", "FULL_CAPACITY_FRACTION"], part_rows)
    _w("resource_cell_results.csv", ["roi", "M", "T", "N_TRIALS", "median_TOTAL", "median_FCF", "sufficient"],
       [[roi, M, T, M * T, group[roi][(M, T)].get("median_TOTAL", ""), group[roi][(M, T)].get("median_FCF", ""), group[roi][(M, T)].get("sufficient", "")] for roi in rois for M in M_GRID for T in T_GRID])
    # spectra diagnostics
    spec = [row for roi in rois for s in ALL for row in per[s][roi]["spectra"]]
    _w("within_support_rank_diagnostic.csv", ["subject", "roi", "fold", "r_best", "q", "boundary_ratio_sq1_sq"], [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in spec])
    _w("outside_d2_rank_diagnostic.csv", ["subject", "roi", "fold", "s1", "s2", "s3", "s3_over_s2"], [[r[0], r[1], r[2], r[6], r[7], r[8], r[9]] for r in spec])
    _w("calibration_schedule_manifest.csv", ["M", "T", "N_TRIALS", "n_id_subsets", "n_repeat_subsets", "schedules_per_participant"],
       [[M, T, M * T, M_SUB[M], T_SUB[T], N_FOLDS * M_SUB[M] * T_SUB[T]] for M in M_GRID for T in T_GRID])
    _w("common_resource_frontier.csv", ["M", "T", "N_TRIALS"], [[M, T, M * T] for (M, T) in common])
    # monotonicity
    mono = {}
    for roi in rois:
        mono[roi] = {"T_fixed_M_increasing": {}, "M_fixed_T_increasing": {}}
        for T in T_GRID:
            seq = [group[roi][(M, T)].get("median_TOTAL") for M in M_GRID if group[roi][(M, T)].get("evaluable")]
            mono[roi]["T_fixed_M_increasing"][T] = bool(all(a2 <= b2 + 1e-9 for a2, b2 in zip(seq, seq[1:]))) if len(seq) > 1 else None
        for M in M_GRID:
            seq = [group[roi][(M, T)].get("median_TOTAL") for T in T_GRID if group[roi][(M, T)].get("evaluable")]
            mono[roi]["M_fixed_T_increasing"][M] = bool(all(a2 <= b2 + 1e-9 for a2, b2 in zip(seq, seq[1:]))) if len(seq) > 1 else None
    (out / "monotonicity_diagnostic.json").write_text(json.dumps(mono, indent=2, default=str))
    (out / "minimal_resource_set.json").write_text(json.dumps(
        {"N_TRIALS_STAR": n_star, "minimal_resource_set": [{"M": M, "T": T, "N_TRIALS": M * T} for (M, T) in minimal_set]}, indent=2))
    (out / "pareto_frontier.json").write_text(json.dumps([{"M": M, "T": T, "N_TRIALS": M * T} for (M, T) in pareto], indent=2))
    (out / "full_capacity_fraction.csv").write_text("see participant_resource_frontier.csv (FULL_CAPACITY_FRACTION column)\n")
    (out / "total_recovery_comp.csv").write_text("see participant_resource_frontier.csv (TOTAL_RECOVERY column)\n")
    (out / "roi_status.json").write_text(json.dumps({roi: roi_status[roi] for roi in rois}, indent=2))
    m10t8 = {roi: per[ALL[0]][roi]["cells"][(10, 8)] for roi in rois}
    (out / "scientific_status.json").write_text(json.dumps(
        {"status": prog, "roi_status": roi_status, "common_success": [{"M": M, "T": T} for (M, T) in common],
         "N_TRIALS_STAR": n_star, "minimal_resource_set": [{"M": M, "T": T} for (M, T) in minimal_set],
         "r111_replay": replay_ok, "r111_replay_err": replay_err, "outside_leak_max": leak["max"],
         "immutable": {"O2_8": "RESIDUAL_CAPACITY_MULTICOMPONENT"}, "O3": "O3_NOT_READY"}, indent=2, default=str))
    print("O2_9_STATUS", prog, "N_TRIALS_STAR", n_star)
    for roi in rois:
        g = group[roi]
        print("  [%s] %s | M10T8 median_TOTAL=%.3f FCF=%.3f suff=%s | #sufficient cells=%d"
              % (roi, roi_status[roi], g[(10, 8)].get("median_TOTAL", float("nan")), g[(10, 8)].get("median_FCF", float("nan")),
                 g[(10, 8)].get("sufficient"), sum(g[(M, T)].get("sufficient", False) for M in M_GRID for T in T_GRID)))
    print("  common-success:", common, " minimal:", minimal_set, " pareto:", pareto)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--state", required=True); ap.add_argument("--trial", required=True)
    ap.add_argument("--ext", required=True); ap.add_argument("--rd-results", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--trial-manifest", required=True); ap.add_argument("--ext-manifest", required=True); ap.add_argument("--o2-8-ref", required=True)
    ap.add_argument("--rd-results-sha256", default=None); ap.add_argument("--out", required=True); ap.add_argument("--rois", default="ventral,lateral")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
