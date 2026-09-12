"""O2.7A Minimal-Trial Outside-Basis Calibration (frozen config d50407fa). DIAGNOSTIC. Conditional on the
sealed O2.6 M=2,D=1 finding, holding the within-support component fixed at the sealed zero-target O2.3A-RD
projector P_ZERO_RD (NO M=2 Procrustes -> removes the O2.7 conditioning confound). The ONLY calibration
resource is the T-repeat data used to estimate ONE outside-support native axis b_T.

No model search: P_ZERO_RD + one SVD-derived outside direction only. Consumes ONLY sealed O2.3A-RD/O2.6 state
+ the certified trial-residual extension. Orthogonal ranges => R_AXIS = R0 + ||b_T^T delta||^2/||delta||^2."""
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
N_PAIRS = 25
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


def _outside_axis(d1, d2, W, want_sv=False):
    Do = np.stack([d1 - W @ (W.T @ d1), d2 - W @ (W.T @ d2)], axis=1)
    U, s, _ = np.linalg.svd(Do, full_matrices=False)
    b = _canon1(U[:, 0])
    return (b, s) if want_sv else b


def _null_axis(seed, W, V, G):
    rng = np.random.Generator(np.random.PCG64(G.seed_uint64(seed)))
    g = rng.standard_normal(V); g = g - W @ (W.T @ g); n = np.linalg.norm(g)
    return _canon1(g / n) if n > 0 else g


def _frac(b, delta):
    d2 = float(delta @ delta)
    return 0.0 if d2 <= 0 else float((b @ delta) ** 2 / d2)


def balanced_pairs(simple_ids, nat_ids):
    return [(a, b) for a in sorted(simple_ids) for b in sorted(nat_ids)]


# ---------------------------------------------------------------- provenance / semantic verification
def verify_trial_file(z, cell, vh_sealed, V):
    D = np.asarray(z["trial_native"]); ext = [str(x) for x in z["train_ids"].tolist()]
    sealed = [str(x) for x in cell["train_ids"].tolist()]
    if not (list(D.shape) == [10, 8, V] and ext == sealed and str(z["voxel_hash"]) == vh_sealed and len(z["fam"]) == 10):
        return "O2_7A_TRIAL_STATE_SEMANTIC_ALIGNMENT_FAILURE"
    prov = json.loads(str(z["provenance"]))
    for i in ext:
        pr = prov.get(i, {}); rp = pr.get("repeat", []); bi = pr.get("beta_index0", [])
        if not (len(rp) == 8 and sorted(rp) == list(range(8)) and len(bi) == 8 and len(set(bi)) == 8):
            return "O2_7A_TRIAL_STATE_SEMANTIC_ALIGNMENT_FAILURE"
    return None


def provenance_gate(out, state_dir, trial_dir, ext_dir, rd_path, manifest_path, trial_manifest, tman_sha, rd_sha, rd, rois):
    prov = {"rd_results_sha256": _sha(rd_path), "expected": rd_sha, "match": (rd_sha is None or _sha(rd_path) == rd_sha)}
    (out / "sealed_state_verification.json").write_text(json.dumps(prov, indent=2))
    if not prov["match"]:
        print("O2_7A_OUTSIDE_AXIS_TRIAL_CALIBRATION_INCONCLUSIVE (rd_results provenance)"); return False
    man = {f["name"]: f for f in json.loads(Path(manifest_path).read_text())["files"]}
    for roi in rois:
        for s in ALL:
            for f in range(N_FOLDS):
                n = f"cell_{s}_{roi}_fold{f}.npz"
                if man.get(n, {}).get("sha256") != _sha(state_dir / n):
                    print("O2_7A_OUTSIDE_AXIS_TRIAL_CALIBRATION_INCONCLUSIVE (sealed-state hash)"); return False
    tv = {"manifest_sha256": _sha(trial_manifest), "expected": tman_sha, "ok": True, "checked": 0, "matched": 0, "failure": None}
    if tman_sha is not None and tv["manifest_sha256"] != tman_sha:
        tv["ok"] = False; tv["failure"] = "O2_7A_TRIAL_MANIFEST_PROVENANCE_FAILURE"
    tman = json.loads(Path(trial_manifest).read_text())
    if tv["ok"] and not (tman.get("artifact") == "trial_residual_state" and tman.get("gate") == "O2.7"
                         and tman.get("ok") is True and len(tman.get("replay", [])) == 96
                         and all(r.get("ok") for r in tman["replay"]) and len(tman.get("files", [])) == 96):
        tv["ok"] = False; tv["failure"] = "O2_7A_TRIAL_MANIFEST_PROVENANCE_FAILURE"
    if tv["ok"]:
        by = {f["name"]: f for f in tman["files"]}
        for roi in rois:
            for s in ALL:
                vh = rd["rois"][roi]["voxel_hash"][s]; V = int(rd["rois"][roi]["n_vox"][s])
                for f in range(N_FOLDS):
                    n = f"trialnat_{s}_{roi}_fold{f}.npz"; tv["checked"] += 1; p = trial_dir / n
                    mm = by.get(n)
                    if not p.exists() or mm is None or _sha(p) != mm["sha256"] or p.stat().st_size != mm["bytes"]:
                        tv["ok"] = False; tv["failure"] = "O2_7A_TRIAL_MANIFEST_PROVENANCE_FAILURE"; continue
                    z = dict(np.load(p, allow_pickle=True)); cell = dict(np.load(state_dir / n.replace("trialnat", "cell"), allow_pickle=True))
                    fail = verify_trial_file(z, cell, vh, V)
                    if fail:
                        tv["ok"] = False; tv["failure"] = fail; continue
                    tv["matched"] += 1
    if tv["ok"] and not (tv["checked"] == tv["matched"] == 96):
        tv["ok"] = False; tv["failure"] = tv["failure"] or "O2_7A_TRIAL_MANIFEST_PROVENANCE_FAILURE"
    (out / "trial_state_verification.json").write_text(json.dumps(tv, indent=2, default=str))
    if not tv["ok"]:
        print(tv["failure"]); return False
    return True


# ---------------------------------------------------------------- per (subject, ROI) axis-calibration frontier
def subject_roi(s, roi, cells, trialnat, extnat, rd, G, leak, cert):
    per_T = {T: {"R_AXIS": [], "R_NULL": [], "AXFID": [], "CAP_CAL": [], "cells": 0} for T in T_ALL}
    R0_folds, RAXIS8_pairs, cond_rows, fid8 = [], [], [], []
    Rnat_folds = [float(rd["per_target"][roi][s][f]["R_ORACLE"]) for f in range(N_FOLDS)]
    for f in range(N_FOLDS):
        cell = cells[f]
        W = np.asarray(cell["W_target"], np.float64); U_res = np.asarray(cell["U_res"], np.float64)
        K = int(cell["K"]); V = W.shape[0]
        deltas_test = [np.asarray(x, np.float64) for x in list(cell["deltas_test"])]
        P0 = G.native_projector(np.eye(K), U_res, W)                          # sealed P_ZERO_RD
        R0 = float(np.mean([G.retention(P0, dt) for dt in deltas_test]))
        R0_folds.append(R0)
        cert["m0_dev"] = max(cert["m0_dev"], abs(R0 - float(cell["R_CAL0"])))  # zero-base certification
        # P_AUG symmetry/idempotence/rank (once per fold with an arbitrary outside axis)
        pairs = balanced_pairs(list(cell["simple_ids"]), list(cell["nat_ids"]))
        Dtrial = np.asarray(trialnat[f]["trial_native"], np.float64)          # (10 x 8 x V)
        Dsealed = np.asarray(extnat[f]["delta_native"], np.float64)            # (10 x V) O2.6 identity residuals
        for pi, (i1, i2) in enumerate(pairs):
            b8, sv = _outside_axis(Dsealed[i1], Dsealed[i2], W, want_sv=True)   # canonical full-resource axis
            RAXIS8 = R0 + float(np.mean([_frac(b8, dt) for dt in deltas_test]))
            RAXIS8_pairs.append(RAXIS8)
            cond_rows.append([s, roi, f, pi, float(sv[0]), float(sv[1] if len(sv) > 1 else 0.0),
                              float((sv[0] - (sv[1] if len(sv) > 1 else 0.0)) / max(sv[0], 1e-12)),
                              float((sv[1] if len(sv) > 1 else 0.0) / max(sv[0], 1e-12))])
            b8_trial = _outside_axis(Dtrial[i1].mean(0), Dtrial[i2].mean(0), W)  # T8 diagnostic
            fid8.append(float((b8_trial @ b8) ** 2))
            if pi == 0 and f == 0:                                            # P_AUG cert (representative)
                Pa = P0 + np.outer(b8, b8)
                cert["sym"] = max(cert["sym"], float(np.linalg.norm(Pa - Pa.T)))
                cert["idem"] = max(cert["idem"], float(np.linalg.norm(Pa @ Pa - Pa)))
                cert["rank_ok"] = cert["rank_ok"] and abs(np.trace(Pa) - (np.trace(P0) + 1)) < 1e-6
                cert["axis_perp"] = max(cert["axis_perp"], float(abs(W.T @ b8).max()))
            for T in T_ALL:
                for si, S in enumerate(itertools.combinations(range(8), T)):
                    Sl = list(S)
                    d1 = Dtrial[i1][Sl].mean(0); d2 = Dtrial[i2][Sl].mean(0)
                    bT = _outside_axis(d1, d2, W)
                    leak["max"] = max(leak["max"], float(abs(W.T @ bT).max()))
                    r_axis = R0 + float(np.mean([_frac(bT, dt) for dt in deltas_test]))
                    if T in T_PRIMARY:
                        nv = []
                        for it in range(N_NULL):
                            bn = _null_axis("O2.7A|%s|%s|%d|%d|%d|%d|%d" % (s, roi, f, pi, T, si, it), W, V, G)
                            nv.append(R0 + float(np.mean([_frac(bn, dt) for dt in deltas_test])))
                        per_T[T]["R_NULL"].append(float(np.mean(nv)))
                    per_T[T]["R_AXIS"].append(r_axis)
                    per_T[T]["AXFID"].append(float((bT @ b8) ** 2))
                    per_T[T]["CAP_CAL"].append(float(np.mean([_frac(bT, d1), _frac(bT, d2)])))  # in-sample capture
                    per_T[T]["cells"] += 1
    R0 = float(np.mean(R0_folds)); Rnat = float(np.mean(Rnat_folds)); RAXIS8 = float(np.mean(RAXIS8_pairs))
    n_pairs = len(balanced_pairs(list(cells[0]["simple_ids"]), list(cells[0]["nat_ids"])))
    out = {"R0": R0, "R_native": Rnat, "R_AXIS_8": RAXIS8, "AXIS_FIDELITY_T8": float(np.mean(fid8)),
           "cond_rows": cond_rows, "T": {}, "cell_counts": {T: per_T[T]["cells"] for T in T_ALL}}
    for T in T_ALL:
        exp = N_FOLDS * n_pairs * EXPECTED_SUBSETS[T]
        if per_T[T]["cells"] != exp:
            out["T"][T] = {"evaluable": False, "reason": "cell_count %d!=%d" % (per_T[T]["cells"], exp)}; continue
        p = per_T[T]; R_T = float(np.mean(p["R_AXIS"]))
        out["T"][T] = {"evaluable": True, "R_AXIS": R_T, "DELTA_AXIS": R_T - R0,
                       "R_NULL_mean": (float(np.mean(p["R_NULL"])) if T in T_PRIMARY else None),
                       "E_AXIS": (R_T - float(np.mean(p["R_NULL"]))) if T in T_PRIMARY else None,
                       "AXIS_FIDELITY": float(np.mean(p["AXFID"])), "CAP_CAL": float(np.mean(p["CAP_CAL"])),
                       "AXIS_RESOURCE_FRACTION": float(np.clip((R_T - R0) / max(RAXIS8 - R0, EPS), 0, 1)),
                       "AXIS_RESOURCE_FRACTION_unclipped": (R_T - R0) / max(RAXIS8 - R0, EPS),
                       "TOTAL_RECOVERY_ZERO_BASE": float(np.clip((R_T - R0) / max(Rnat - R0, EPS), 0, 1)),
                       "TOTAL_RECOVERY_ZERO_BASE_unclipped": (R_T - R0) / max(Rnat - R0, EPS)}
    return out


# ---------------------------------------------------------------- inference + status
def _median(x):
    return float(np.median(x)) if len(x) else float("nan")


def aggregate(per_subj, G):
    from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
    rois = list(next(iter(per_subj.values())).keys())
    pvals = {}; grp = {roi: {} for roi in rois}
    for roi in rois:
        for T in T_PRIMARY:
            if not all(per_subj[s][roi]["T"][T].get("evaluable") for s in ALL):
                grp[roi][T] = {"evaluable": False}; continue
            E = [per_subj[s][roi]["T"][T]["E_AXIS"] for s in ALL]
            arf = [per_subj[s][roi]["T"][T]["AXIS_RESOURCE_FRACTION"] for s in ALL]
            tot = [per_subj[s][roi]["T"][T]["TOTAL_RECOVERY_ZERO_BASE"] for s in ALL]
            grp[roi][T] = {"evaluable": True, "median_E_AXIS": _median(E), "n_E_pos": sum(e > 0 for e in E),
                           "median_ARF": _median(arf), "n_ARF_ge0.5": sum(a >= 0.5 for a in arf),
                           "median_TOTAL": _median(tot), "n_TOTAL_ge0.5": sum(t >= 0.5 for t in tot),
                           "median_AXIS_FIDELITY": _median([per_subj[s][roi]["T"][T]["AXIS_FIDELITY"] for s in ALL]), "E": E}
            pvals["%s|T%d" % (roi, T)] = F.signflip_p_onesided(E)
    holm = F.holm(pvals, alpha=0.05) if pvals else {}
    roi_status = {}
    for roi in rois:
        tstar = "NOT_ESTABLISHED_BY_T4"
        for T in T_PRIMARY:
            g = grp[roi][T]
            if not g.get("evaluable"):
                continue
            key = "%s|T%d" % (roi, T); g["signflip_p"] = float(pvals[key]); g["holm_reject"] = bool(holm.get(key, False))
            g["meets_all"] = (g["median_E_AXIS"] > 0 and g["n_E_pos"] >= 6 and g["holm_reject"]
                              and g["median_ARF"] >= 0.50 and g["n_ARF_ge0.5"] >= 6
                              and g["median_TOTAL"] >= 0.50 and g["n_TOTAL_ge0.5"] >= 6)
            if g["meets_all"] and tstar == "NOT_ESTABLISHED_BY_T4":
                tstar = T
        st = {1: "ONE_REPEAT_OUTSIDE_AXIS_CALIBRATION_SUFFICIENT", 2: "TWO_REPEAT_OUTSIDE_AXIS_CALIBRATION_SUFFICIENT",
              4: "FOUR_REPEAT_OUTSIDE_AXIS_CALIBRATION_SUFFICIENT"}.get(tstar, "REDUCED_REPEAT_OUTSIDE_AXIS_CALIBRATION_NOT_ESTABLISHED")
        roi_status[roi] = {"T_AXIS_STAR": tstar, "status": st, "per_T": grp[roi]}
    tv, tl = roi_status["ventral"]["T_AXIS_STAR"], roi_status["lateral"]["T_AXIS_STAR"]
    iv = tv if isinstance(tv, int) else None; il = tl if isinstance(tl, int) else None
    if iv is not None and il is not None:
        m = max(iv, il)
        prog = ("TWO_TOTAL_TRIAL_ZERO_BASE_TARGET_CALIBRATION" if m == 1 else
                "FOUR_TOTAL_TRIAL_ZERO_BASE_TARGET_CALIBRATION" if m == 2 else
                "EIGHT_TOTAL_TRIAL_ZERO_BASE_TARGET_CALIBRATION")
    elif (iv is None) ^ (il is None):
        prog = "OUTSIDE_AXIS_TRIAL_CALIBRATION_MULTIREGIME"
    else:
        prog = "REDUCED_TRIAL_OUTSIDE_AXIS_CALIBRATION_NOT_ESTABLISHED"
    return roi_status, prog


def run(a):
    G = _geom(a.repo)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    state_dir, trial_dir, ext_dir = Path(a.state), Path(a.trial), Path(a.ext)
    rd = json.loads(Path(a.rd_results).read_text())
    rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    if not provenance_gate(out, state_dir, trial_dir, ext_dir, Path(a.rd_results), Path(a.manifest),
                           Path(a.trial_manifest), a.trial_manifest_sha256, a.rd_results_sha256, rd, rois):
        return 1
    for roi in rois:
        for s in ALL:
            c0 = dict(np.load(state_dir / f"cell_{s}_{roi}_fold0.npz", allow_pickle=True))
            if len(balanced_pairs(list(c0["simple_ids"]), list(c0["nat_ids"]))) != 25:
                print("O2_7A_OUTSIDE_AXIS_TRIAL_CALIBRATION_INCONCLUSIVE (expected 25 pairs)"); return 1
    leak = {"max": 0.0}; cert = {"m0_dev": 0.0, "sym": 0.0, "idem": 0.0, "rank_ok": True, "axis_perp": 0.0}
    per = {s: {} for s in ALL}
    for roi in rois:
        for s in ALL:
            cells = [dict(np.load(state_dir / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            tn = [dict(np.load(trial_dir / f"trialnat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            en = [dict(np.load(ext_dir / f"deltanat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            per[s][roi] = subject_roi(s, roi, cells, tn, en, rd, G, leak, cert)
    (out / "zero_base_certification.json").write_text(json.dumps(
        {"P_ZERO_RD": "native_projector(I, U_res, W_target)", "max_R0_vs_sealed_R_CAL0_dev": cert["m0_dev"],
         "P_AUG_symmetry": cert["sym"], "P_AUG_idempotence": cert["idem"], "P_AUG_rank_ok": cert["rank_ok"],
         "axis_perp_max": cert["axis_perp"], "ok": (cert["m0_dev"] <= 1e-10 and cert["sym"] <= 1e-10 and cert["idem"] <= 1e-8 and cert["rank_ok"])}, indent=2))
    roi_status, prog = aggregate(per, G)

    def _w(name, header, rows):
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    fr, ax, arf, tot, gc, cond, fra = [], [], [], [], [], [], []
    for roi in rois:
        for s in ALL:
            cond += per[s][roi]["cond_rows"] if "cond_rows" in per[s][roi] else []
        for T in T_PRIMARY:
            g = roi_status[roi]["per_T"][T]
            if g.get("evaluable"):
                fr.append([roi, T, g["median_E_AXIS"], g["n_E_pos"], g.get("signflip_p", ""), g.get("holm_reject", ""),
                           g["median_ARF"], g["n_ARF_ge0.5"], g["median_TOTAL"], g["n_TOTAL_ge0.5"], g["median_AXIS_FIDELITY"]])
        for s in ALL:
            fra.append([s, roi, per[s][roi]["R_AXIS_8"], per[s][roi]["AXIS_FIDELITY_T8"]])
            for T in T_ALL:
                v = per[s][roi]["T"][T]
                if not v.get("evaluable"):
                    continue
                ax.append([s, roi, T, v["AXIS_FIDELITY"]])
                if T in T_PRIMARY:
                    arf.append([s, roi, T, v["AXIS_RESOURCE_FRACTION"], v["AXIS_RESOURCE_FRACTION_unclipped"]])
                    tot.append([s, roi, T, v["TOTAL_RECOVERY_ZERO_BASE"], v["TOTAL_RECOVERY_ZERO_BASE_unclipped"], v["E_AXIS"], v["R_AXIS"]])
                    gc.append([s, roi, T, v["CAP_CAL"], v["R_AXIS"]])
    _w("trial_axis_frontier.csv", ["roi", "T", "median_E_AXIS", "n_E_pos", "signflip_p", "holm_reject", "median_ARF", "n_ARF_ge0.5", "median_TOTAL", "n_TOTAL_ge0.5", "median_AXIS_FIDELITY"], fr)
    _w("axis_fidelity.csv", ["subject", "roi", "T", "AXIS_FIDELITY"], ax)
    _w("axis_resource_fraction.csv", ["subject", "roi", "T", "ARF_clipped", "ARF_unclipped"], arf)
    _w("total_recovery_zero_base.csv", ["subject", "roi", "T", "TOTAL_clipped", "TOTAL_unclipped", "E_AXIS", "R_AXIS"], tot)
    _w("generalization_control.csv", ["subject", "roi", "T", "calibration_axis_capture", "heldout_R_AXIS"], gc)
    _w("axis_conditioning_diagnostic.csv", ["subject", "roi", "fold", "pair", "s1", "s2", "AXIS_SINGULAR_GAP", "s2_over_s1"], cond)
    _w("full_resource_axis_reference.csv", ["subject", "roi", "R_AXIS_8", "AXIS_FIDELITY_T8"], fra)
    _w("participant_roi_frontier.csv", ["subject", "roi", "T", "R_AXIS", "DELTA_AXIS", "E_AXIS", "AXIS_FIDELITY", "ARF", "TOTAL_ZERO_BASE"],
       [[s, roi, T, per[s][roi]["T"][T]["R_AXIS"], per[s][roi]["T"][T]["DELTA_AXIS"],
         per[s][roi]["T"][T].get("E_AXIS"), per[s][roi]["T"][T]["AXIS_FIDELITY"],
         per[s][roi]["T"][T].get("AXIS_RESOURCE_FRACTION"), per[s][roi]["T"][T].get("TOTAL_RECOVERY_ZERO_BASE")]
        for roi in rois for s in ALL for T in T_ALL if per[s][roi]["T"][T].get("evaluable")])
    (out / "repeat_reliability.csv").write_text("subject,roi,note\n(participant-first split-half descriptive; see axis_fidelity)\n")
    (out / "primary_inference.json").write_text(json.dumps({"family": [("%s|T%d" % (r, T)) for r in rois for T in T_PRIMARY],
        "family_size": 6, "T8_in_family": False,
        "per_test": {roi: {T: {k: roi_status[roi]["per_T"][T].get(k) for k in ("signflip_p", "holm_reject", "median_E_AXIS", "n_E_pos")}
                           for T in T_PRIMARY} for roi in rois}}, indent=2, default=str))
    (out / "minimum_axis_trial_budget.json").write_text(json.dumps({roi: {"T_AXIS_STAR": roi_status[roi]["T_AXIS_STAR"],
        "total_target_imagery_trials": (2 * roi_status[roi]["T_AXIS_STAR"] if isinstance(roi_status[roi]["T_AXIS_STAR"], int) else None)} for roi in rois}, indent=2, default=str))
    (out / "primary_roi_status.json").write_text(json.dumps({roi: {"status": roi_status[roi]["status"], "T_AXIS_STAR": roi_status[roi]["T_AXIS_STAR"]} for roi in rois}, indent=2, default=str))
    (out / "matched_axis_null_summary.json").write_text(json.dumps({"n_null_per_cell": N_NULL, "seed": "O2.7A|subject|ROI|fold|identity_pair|T|repeat_subset_id|null_iteration", "outside_leak_max": leak["max"]}, indent=2))
    def _ov(roi):
        c1 = _median([per[s][roi]["T"][1]["CAP_CAL"] for s in ALL]); h1 = _median([per[s][roi]["T"][1]["R_AXIS"] for s in ALL])
        return "LOW_REPEAT_OUTSIDE_AXIS_OVERFIT_SIGNATURE" if (c1 > 0.9 and h1 < per[ALL[0]][roi]["R0"] + 0.02) else "none"
    (out / "scientific_status.json").write_text(json.dumps(
        {"status": prog, "roi_status": {roi: roi_status[roi]["status"] for roi in rois},
         "T_AXIS_STAR": {roi: roi_status[roi]["T_AXIS_STAR"] for roi in rois}, "outside_leak_max": leak["max"],
         "overfit_signature": {roi: _ov(roi) for roi in rois},
         "immutable": {"O2_7": "O2_7_FULL_REPEAT_REPLAY_FAILURE", "O2_6": "TARGET_STATE_MISSING_BASIS_LOW_COMPLEXITY"},
         "O3": "O3_NOT_READY"}, indent=2, default=str))
    print("O2_7A_STATUS", prog)
    for roi in rois:
        print("  [%s] %s T_AXIS_STAR=%s" % (roi, roi_status[roi]["status"], roi_status[roi]["T_AXIS_STAR"]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--state", required=True)
    ap.add_argument("--trial", required=True); ap.add_argument("--ext", required=True)
    ap.add_argument("--rd-results", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--trial-manifest", required=True); ap.add_argument("--trial-manifest-sha256", default=None)
    ap.add_argument("--rd-results-sha256", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--rois", default="ventral,lateral")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
