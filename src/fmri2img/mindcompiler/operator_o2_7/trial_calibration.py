"""O2.7 Minimal-Trial Target-State Calibration (frozen config d61cd6ff). DIAGNOSTIC; conditional on the
sealed O2.6 M=2,D=1 estimator with reduced calibration repeats. No model search (no M/D change, no
ridge/CCA/RRR/NN/new W_target/residual). Consumes ONLY sealed O2.3A-RD/O2.4R/O2.6 state + the replay-certified
trial-residual extension. Evaluates ALL repeat-position subsets for T in {1,2,4} (+T=8 reference) and reports
E_TRIAL, axis fidelity, TOTAL_RECOVERY, T_STAR, per-ROI + program status."""
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
ROIS_PRIMARY = ["ventral", "lateral"]
N_FOLDS = 6
T_PRIMARY = [1, 2, 4]
T_ALL = [1, 2, 4, 8]
EXPECTED_SUBSETS = {1: 8, 2: 28, 4: 70, 8: 1}
N_NULL = 100
EPS = 1e-12
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


def _canon1(u):
    u = np.asarray(u, np.float64)
    j = int(np.argmax(np.abs(u)))
    return u * (-1.0 if u[j] < 0 else 1.0)


def _outside_axis(d1, d2, W):
    """First left singular vector of [delta_out_1, delta_out_2] (D=1), canonical sign; orthogonal to col(W)."""
    Do = np.stack([d1 - W @ (W.T @ d1), d2 - W @ (W.T @ d2)], axis=1)      # (V x 2)
    U, _, _ = np.linalg.svd(Do, full_matrices=False)
    return _canon1(U[:, 0])


def _null_axis(seed, W, V, G):
    rng = np.random.Generator(np.random.PCG64(G.seed_uint64(seed)))
    g = rng.standard_normal(V)
    g = g - W @ (W.T @ g)
    n = np.linalg.norm(g)
    return _canon1(g / n) if n > 0 else g


def _frac(b, delta):
    d2 = float(delta @ delta)
    return 0.0 if d2 <= 0 else float((b @ delta) ** 2 / d2)


def balanced_pairs(simple_ids, nat_ids):
    return [(a, b) for a in sorted(simple_ids) for b in sorted(nat_ids)]     # 25 pairs (M=2)


# --- per (subject, ROI) trial-calibration frontier ---------------------------
def subject_roi(s, roi, cells, trialnat, rd, G, leak):
    per_T = {T: {"R_AUG": [], "R_BASE": [], "R_NULL": [], "AXFID": [], "TRAIN": []} for T in T_ALL}
    R0_folds = [float(c["R_CAL0"]) for c in cells]
    Rnat_folds = [float(rd["per_target"][roi][s][f]["R_ORACLE"]) for f in range(N_FOLDS)]
    for f in range(N_FOLDS):
        cell = cells[f]; tn = trialnat[f]
        W = np.asarray(cell["W_target"], np.float64); U_res = np.asarray(cell["U_res"], np.float64)
        X = np.asarray(cell["X"], np.float64); K = int(cell["K"]); V = W.shape[0]
        deltas_test = [np.asarray(x, np.float64) for x in list(cell["deltas_test"])]
        simple_ids = list(cell["simple_ids"]); nat_ids = list(cell["nat_ids"])
        Dtrial = np.asarray(tn["trial_native"], np.float64)                  # (10 x 8 x V)
        pairs = balanced_pairs(simple_ids, nat_ids)
        for pi, (i1, i2) in enumerate(pairs):
            # full-repeat T=8 reference axis for this pair
            d1_8 = Dtrial[i1].mean(0); d2_8 = Dtrial[i2].mean(0)
            b8 = _outside_axis(d1_8, d2_8, W)
            for T in T_ALL:
                for si, S in enumerate(itertools.combinations(range(8), T)):
                    Sl = list(S)
                    d1 = Dtrial[i1][Sl].mean(0); d2 = Dtrial[i2][Sl].mean(0)
                    Z = np.stack([W.T @ d1, W.T @ d2])
                    Q = G.orthogonal_procrustes(X[[i1, i2]], Z)
                    P_IN = G.native_projector(Q, U_res, W)
                    R_base = float(np.mean([G.retention(P_IN, dt) for dt in deltas_test]))
                    bT = _outside_axis(d1, d2, W)
                    leak["max"] = max(leak["max"], float(abs(W.T @ bT).max()))
                    r_aug = R_base + float(np.mean([_frac(bT, dt) for dt in deltas_test]))
                    nvals = []
                    for it in range(N_NULL):
                        seed = "O2.7|%s|%s|%d|%d|%d|%d|%d" % (s, roi, f, pi, T, si, it)
                        bn = _null_axis(seed, W, V, G)
                        nvals.append(R_base + float(np.mean([_frac(bn, dt) for dt in deltas_test])))
                    tr = float(np.mean([G.retention(P_IN, Dtrial[i][Sl].mean(0)) + _frac(bT, Dtrial[i][Sl].mean(0))
                                        for i in (i1, i2)]))
                    per_T[T]["R_AUG"].append(r_aug); per_T[T]["R_BASE"].append(R_base)
                    per_T[T]["R_NULL"].append(float(np.mean(nvals)))
                    per_T[T]["AXFID"].append(float((bT @ b8) ** 2)); per_T[T]["TRAIN"].append(tr)
    R0 = float(np.mean(R0_folds)); Rnat = float(np.mean(Rnat_folds))
    out = {"R0": R0, "R_native": Rnat, "T": {}}
    R8 = float(np.mean(per_T[8]["R_AUG"]))
    for T in T_ALL:
        p = per_T[T]
        R_T = float(np.mean(p["R_AUG"]))
        out["T"][T] = {"R_AUG": R_T, "R_BASE": float(np.mean(p["R_BASE"])), "R_NULL_mean": float(np.mean(p["R_NULL"])),
                       "E_TRIAL": R_T - float(np.mean(p["R_NULL"])), "AXIS_FIDELITY": float(np.mean(p["AXFID"])),
                       "TRAIN_retention": float(np.mean(p["TRAIN"])),
                       "TOTAL_RECOVERY": float(np.clip((R_T - R0) / max(Rnat - R0, EPS), 0, 1)),
                       "TOTAL_RECOVERY_unclipped": (R_T - R0) / max(Rnat - R0, EPS),
                       "FULL_RESOURCE_FRACTION": float(np.clip((R_T - R0) / max(R8 - R0, EPS), 0, 1)),
                       "FULL_RESOURCE_FRACTION_unclipped": (R_T - R0) / max(R8 - R0, EPS),
                       "n_cells": len(p["R_AUG"])}
    out["R8"] = R8
    return out


# --- inference + status ------------------------------------------------------
def _median(x):
    return float(np.median(x)) if len(x) else float("nan")


def aggregate(per_subj, G):
    from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
    rois = list(next(iter(per_subj.values())).keys())
    # participant E_TRIAL / TOTAL per (roi,T)
    pvals = {}
    grp = {roi: {} for roi in rois}
    for roi in rois:
        for T in T_ALL:
            E = [per_subj[s][roi]["T"][T]["E_TRIAL"] for s in ALL]
            tot = [per_subj[s][roi]["T"][T]["TOTAL_RECOVERY"] for s in ALL]
            grp[roi][T] = {"median_E_TRIAL": _median(E), "n_E_pos": sum(e > 0 for e in E),
                           "median_TOTAL": _median(tot), "n_TOTAL_ge0.5": sum(t >= 0.5 for t in tot),
                           "median_AXIS_FIDELITY": _median([per_subj[s][roi]["T"][T]["AXIS_FIDELITY"] for s in ALL]),
                           "median_FULL_RESOURCE": _median([per_subj[s][roi]["T"][T]["FULL_RESOURCE_FRACTION"] for s in ALL]),
                           "E": E}
            if T in T_PRIMARY:
                pvals["%s|T%d" % (roi, T)] = F.signflip_p_onesided(E)
    holm = F.holm(pvals, alpha=0.05)                                         # 6-test family
    roi_status = {}
    for roi in rois:
        tstar = "BELOW_FULL_RESOURCE_NOT_ESTABLISHED"
        for T in T_PRIMARY:
            g = grp[roi][T]; key = "%s|T%d" % (roi, T)
            g["holm_reject"] = bool(holm.get(key, False)); g["signflip_p"] = float(pvals[key])
            ok = (g["median_E_TRIAL"] > 0 and g["n_E_pos"] >= 6 and g["holm_reject"]
                  and g["median_TOTAL"] >= 0.50 and g["n_TOTAL_ge0.5"] >= 6)
            g["meets_all"] = ok
            if ok and tstar == "BELOW_FULL_RESOURCE_NOT_ESTABLISHED":
                tstar = T
        st = {1: "ONE_REPEAT_TARGET_STATE_CALIBRATION_SUFFICIENT", 2: "TWO_REPEAT_TARGET_STATE_CALIBRATION_SUFFICIENT",
              4: "FOUR_REPEAT_TARGET_STATE_CALIBRATION_SUFFICIENT"}.get(tstar, "REDUCED_REPEAT_TARGET_STATE_CALIBRATION_NOT_ESTABLISHED")
        roi_status[roi] = {"T_STAR": tstar, "status": st, "per_T": grp[roi]}
    tv, tl = roi_status["ventral"]["T_STAR"], roi_status["lateral"]["T_STAR"]
    def _int(x): return x if isinstance(x, int) else None
    iv, il = _int(tv), _int(tl)
    if iv is None or il is None:
        prog = "REDUCED_TRIAL_TARGET_STATE_CALIBRATION_NOT_ESTABLISHED"
    elif iv == 1 and il == 1:
        prog = "MINIMAL_TARGET_STATE_CALIBRATION_TWO_TOTAL_TRIALS"
    elif iv <= 2 and il <= 2:
        prog = "MINIMAL_TARGET_STATE_CALIBRATION_FOUR_TOTAL_TRIALS"
    elif iv <= 4 and il <= 4:
        prog = "MINIMAL_TARGET_STATE_CALIBRATION_EIGHT_TOTAL_TRIALS"
    else:
        prog = "TARGET_STATE_TRIAL_CALIBRATION_MULTIREGIME"
    return roi_status, prog


# --- T=8 exact O2.6 replay certification -------------------------------------
def certify_t8_replay(per_subj, o2_6_aug_csv, rois):
    """At T=8 the M2,D1 R_AUG per participant must equal the sealed O2.6 M2,d1 R_AUG within 1e-10."""
    ref = {}
    for r in csv.DictReader(open(o2_6_aug_csv)):
        if int(r["M"]) == 2 and int(r["d"]) == 1:
            ref[(r["subject"], r["roi"])] = float(r["R_AUG"])
    max_err = 0.0
    for roi in rois:
        for s in ALL:
            got = per_subj[s][roi]["T"][8]["R_AUG"]
            if (s, roi) in ref:
                max_err = max(max_err, abs(got - ref[(s, roi)]))
    return {"ok": max_err <= 1e-10, "max_abs_error": max_err,
            "status": "O2_7_T8_REPLAYS_O2_6_M2_D1" if max_err <= 1e-10 else "O2_7_FULL_REPEAT_REPLAY_FAILURE"}


def run(repo, state_dir, trial_dir, rd_results_path, manifest_path, trial_manifest, rd_sha, o2_6_aug_csv, out, rois):
    G = _geom(repo)
    out.mkdir(parents=True, exist_ok=True)
    prov = {"path": str(rd_results_path), "sha256": _sha(rd_results_path), "expected_sha256": rd_sha,
            "match": (rd_sha is None or _sha(rd_results_path) == rd_sha)}
    (out / "rd_results_provenance.json").write_text(json.dumps(prov, indent=2))
    if not prov["match"]:
        print("O2_7_TRIAL_CALIBRATION_INCONCLUSIVE (rd_results provenance mismatch)"); return 1
    rd = json.loads(Path(rd_results_path).read_text())
    # sealed-state hash verification
    man = {f["name"]: f for f in json.loads(Path(manifest_path).read_text())["files"]}
    hv = {"checked": 0, "ok": True, "mismatches": []}
    for roi in rois:
        for s in ALL:
            for f in range(N_FOLDS):
                n = f"cell_{s}_{roi}_fold{f}.npz"; hv["checked"] += 1
                if man.get(n, {}).get("sha256") != _sha(state_dir / n):
                    hv["ok"] = False; hv["mismatches"].append(n)
    (out / "sealed_state_verification.json").write_text(json.dumps(hv, indent=2))
    if not hv["ok"]:
        print("O2_7_TRIAL_CALIBRATION_INCONCLUSIVE (sealed-state hash mismatch)"); return 1
    # trial-state extension hash verification
    tman = {f["name"]: f for f in json.loads(Path(trial_manifest).read_text())["files"]}
    tv = {"checked": 0, "matched": 0, "ok": True}
    for roi in rois:
        for s in ALL:
            for f in range(N_FOLDS):
                n = f"trialnat_{s}_{roi}_fold{f}.npz"; tv["checked"] += 1
                p = trial_dir / n
                if p.exists() and tman.get(n, {}).get("sha256") == _sha(p) and tman[n]["bytes"] == p.stat().st_size:
                    tv["matched"] += 1
                else:
                    tv["ok"] = False
    (out / "trial_state_replay_certification.json").write_text(json.dumps(tv, indent=2))
    if not (tv["ok"] and tv["checked"] == tv["matched"] == 96):
        print("O2_7_TRIAL_STATE_EXTENSION_FAILURE (hash)"); return 1

    leak = {"max": 0.0}
    per_subj = {s: {} for s in ALL}
    for roi in rois:
        for s in ALL:
            cells = [dict(np.load(state_dir / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            tn = [dict(np.load(trial_dir / f"trialnat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            per_subj[s][roi] = subject_roi(s, roi, cells, tn, rd, G, leak)
    # T=8 exact O2.6 replay
    t8 = certify_t8_replay(per_subj, o2_6_aug_csv, rois)
    (out / "t8_o2_6_replay_certification.json").write_text(json.dumps(t8, indent=2))
    if not t8["ok"]:
        print(t8["status"]); return 1
    roi_status, prog = aggregate(per_subj, G)

    def _w(name, header, rows):
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    fr, ax, frf, tr, gc = [], [], [], [], []
    for roi in rois:
        for T in T_ALL:
            g = roi_status[roi]["per_T"][T]
            fr.append([roi, T, g["median_E_TRIAL"], g["n_E_pos"], g.get("signflip_p", ""), g.get("holm_reject", ""),
                       g["median_TOTAL"], g["n_TOTAL_ge0.5"], g["median_AXIS_FIDELITY"], g["median_FULL_RESOURCE"]])
        for s in ALL:
            for T in T_ALL:
                v = per_subj[s][roi]["T"][T]
                ax.append([s, roi, T, v["AXIS_FIDELITY"]]); frf.append([s, roi, T, v["FULL_RESOURCE_FRACTION"], v["FULL_RESOURCE_FRACTION_unclipped"]])
                tr.append([s, roi, T, v["TOTAL_RECOVERY"], v["TOTAL_RECOVERY_unclipped"], v["E_TRIAL"], v["R_AUG"], v["R_BASE"]])
                gc.append([s, roi, T, v["TRAIN_retention"], v["R_AUG"]])
    _w("trial_frontier_results.csv", ["roi", "T", "median_E_TRIAL", "n_E_pos", "signflip_p", "holm_reject", "median_TOTAL", "n_TOTAL_ge0.5", "median_AXIS_FIDELITY", "median_FULL_RESOURCE"], fr)
    _w("participant_roi_trial_frontier.csv", ["subject", "roi", "T", "TOTAL_RECOVERY", "TOTAL_RECOVERY_unclipped", "E_TRIAL", "R_AUG", "R_BASE"], tr)
    _w("axis_fidelity_results.csv", ["subject", "roi", "T", "AXIS_FIDELITY"], ax)
    _w("full_resource_fraction.csv", ["subject", "roi", "T", "FULL_RESOURCE_FRACTION", "unclipped"], frf)
    _w("total_recovery_results.csv", ["subject", "roi", "T", "TOTAL_RECOVERY", "unclipped", "E_TRIAL", "R_AUG", "R_BASE"], tr)
    _w("generalization_control.csv", ["subject", "roi", "T", "train_calibration_retention", "heldout_R_AUG"], gc)
    (out / "repeat_schedule_manifest.csv").write_text("T,n_repeat_subsets,total_trials\n" + "\n".join("%d,%d,%d" % (T, EXPECTED_SUBSETS[T], 2 * T) for T in T_ALL) + "\n")
    prim = {"family": [("%s|T%d" % (roi, T)) for roi in rois for T in T_PRIMARY], "family_size": 6,
            "per_test": {roi: {T: {"signflip_p": roi_status[roi]["per_T"][T].get("signflip_p"),
                                   "holm_reject": roi_status[roi]["per_T"][T].get("holm_reject"),
                                   "median_E_TRIAL": roi_status[roi]["per_T"][T]["median_E_TRIAL"],
                                   "n_E_pos": roi_status[roi]["per_T"][T]["n_E_pos"]}
                               for T in T_PRIMARY} for roi in rois}}
    (out / "primary_inference.json").write_text(json.dumps(prim, indent=2, default=str))
    (out / "minimum_repeat_budget.json").write_text(json.dumps({roi: {"T_STAR": roi_status[roi]["T_STAR"], "total_trials": (2 * roi_status[roi]["T_STAR"] if isinstance(roi_status[roi]["T_STAR"], int) else None)} for roi in rois}, indent=2, default=str))
    (out / "primary_roi_status.json").write_text(json.dumps({roi: {"status": roi_status[roi]["status"], "T_STAR": roi_status[roi]["T_STAR"]} for roi in rois}, indent=2, default=str))
    def _overfit(roi):
        tr1 = _median([per_subj[s][roi]["T"][1]["TRAIN_retention"] for s in ALL])
        ho1 = _median([per_subj[s][roi]["T"][1]["R_AUG"] for s in ALL])
        return "LOW_REPEAT_TARGET_STATE_OVERFIT_SIGNATURE" if (tr1 > 0.9 and ho1 < 0.3) else "none"
    overfit = {roi: _overfit(roi) for roi in rois}
    (out / "scientific_status.json").write_text(json.dumps(
        {"status": prog, "roi_status": {roi: roi_status[roi]["status"] for roi in rois},
         "T_STAR": {roi: roi_status[roi]["T_STAR"] for roi in rois}, "t8_replay": t8["status"],
         "outside_leak_max": leak["max"], "overfit_signature": overfit,
         "immutable": {"O2_6": "TARGET_STATE_MISSING_BASIS_LOW_COMPLEXITY"}, "O3": "O3_NOT_READY"}, indent=2, default=str))
    (out / "matched_null_summary.json").write_text(json.dumps({"n_null_per_cell": N_NULL, "seed": "O2.7|subject|ROI|fold|identity_pair|T|repeat_subset_id|null_iteration", "outside_leak_max": leak["max"]}, indent=2))
    print("O2_7_STATUS", prog)
    for roi in rois:
        print("  [%s] %s T_STAR=%s" % (roi, roi_status[roi]["status"], roi_status[roi]["T_STAR"]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--state", required=True)
    ap.add_argument("--trial", required=True); ap.add_argument("--rd-results", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--trial-manifest", required=True)
    ap.add_argument("--rd-results-sha256", default=None); ap.add_argument("--o2-6-aug", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--rois", default="ventral,lateral")
    a = ap.parse_args()
    return run(a.repo, Path(a.state), Path(a.trial), Path(a.rd_results), Path(a.manifest), Path(a.trial_manifest),
               a.rd_results_sha256, Path(a.o2_6_aug), Path(a.out), [r.strip() for r in a.rois.split(",") if r.strip()])


if __name__ == "__main__":
    raise SystemExit(main())
