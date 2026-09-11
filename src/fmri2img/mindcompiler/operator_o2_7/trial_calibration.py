"""O2.7 Minimal-Trial Target-State Calibration (frozen config d61cd6ff; integrity-corrected, see doc 67).
DIAGNOSTIC; conditional on the sealed O2.6 M=2,D=1 estimator with reduced calibration repeats. No model
search. Consumes ONLY sealed O2.3A-RD/O2.4R/O2.6 state + the replay-certified trial-residual extension.

TWO SEALED STAGES (FIX 1):
  --stage t8       : compute ONLY T=8 and certify FLOAT32_SOURCE_BOUNDED replay of sealed O2.6 M2,D1
                     (numerical, tol 1e-5, + structural). Writes t8 cert artifacts. No reduced-repeat metric.
  --stage frontier : refuses unless a valid Stage-A T8 certification is present; then computes T in {1,2,4},
                     matched null x100, axis fidelity vs the full-8 axis, TOTAL_RECOVERY, 6-test sign-flip +
                     Holm, T_STAR, per-ROI + program status. T=8 NEVER enters the Holm family (FIX 7).
The trial pipeline is float64; agreement with the sealed float32-derived O2.6 values is bounded by the source
betas (~1e-6) -- reported as FLOAT32_SOURCE_BOUNDED, never "exact"/"bitwise"."""
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
EXPECTED_SUBSETS = {1: 8, 2: 28, 4: 70, 8: 1}
N_PAIRS = 25
N_NULL = 100
REPLAY_TOL = 1e-5                                            # FIX: FLOAT32_SOURCE_BOUNDED (not exact)
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
    Do = np.stack([d1 - W @ (W.T @ d1), d2 - W @ (W.T @ d2)], axis=1)
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


# ---------------------------------------------------------------- semantic trial-state verification (FIX 3,4)
def verify_trial_file(z, cell, vh_sealed, V):
    """Per-file semantic check (FIX 4). Returns None on success, else a failure code."""
    D = np.asarray(z["trial_native"]); ext_train = [str(x) for x in z["train_ids"].tolist()]
    sealed_train = [str(x) for x in cell["train_ids"].tolist()]
    if not (list(D.shape) == [10, 8, V] and ext_train == sealed_train
            and str(z["voxel_hash"]) == vh_sealed and len(z["fam"]) == 10):
        return "O2_7_TRIAL_STATE_SEMANTIC_ALIGNMENT_FAILURE"
    prov = json.loads(str(z["provenance"]))
    for i in ext_train:
        pr = prov.get(i, {}); rp = pr.get("repeat", []); bi = pr.get("beta_index0", [])
        if not (len(rp) == 8 and sorted(rp) == list(range(8)) and len(bi) == 8 and len(set(bi)) == 8):
            return "O2_7_TRIAL_STATE_SEMANTIC_ALIGNMENT_FAILURE"
    return None


def verify_trial_state(state_dir, trial_dir, trial_manifest_path, rd, rois, expected_manifest_sha):
    manp = Path(trial_manifest_path)
    rec = {"manifest_sha256": _sha(manp), "expected_manifest_sha256": expected_manifest_sha,
           "checked": 0, "matched": 0, "ok": True, "failure": None}
    if expected_manifest_sha is not None and rec["manifest_sha256"] != expected_manifest_sha:
        rec["ok"] = False; rec["failure"] = "O2_7_TRIAL_MANIFEST_PROVENANCE_FAILURE"; return rec
    man = json.loads(manp.read_text())
    if not (man.get("artifact") == "trial_residual_state" and man.get("gate") == "O2.7" and man.get("ok") is True
            and len(man.get("replay", [])) == 96 and all(r.get("ok") for r in man["replay"])
            and len(man.get("files", [])) == 96):
        rec["ok"] = False; rec["failure"] = "O2_7_TRIAL_MANIFEST_PROVENANCE_FAILURE"; return rec
    by = {f["name"]: f for f in man["files"]}
    for roi in rois:
        for s in ALL:
            vh_sealed = rd["rois"][roi]["voxel_hash"][s]; V = int(rd["rois"][roi]["n_vox"][s])
            for f in range(N_FOLDS):
                n = f"trialnat_{s}_{roi}_fold{f}.npz"; rec["checked"] += 1; p = trial_dir / n
                mm = by.get(n)
                if not p.exists() or mm is None or _sha(p) != mm["sha256"] or p.stat().st_size != mm["bytes"]:
                    rec["ok"] = False; rec["failure"] = "O2_7_TRIAL_MANIFEST_PROVENANCE_FAILURE"; continue
                z = dict(np.load(p, allow_pickle=True))
                cell = dict(np.load(state_dir / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True))
                fail = verify_trial_file(z, cell, vh_sealed, V)
                if fail:
                    rec["ok"] = False; rec["failure"] = fail; continue
                rec["matched"] += 1
    if rec["ok"] and not (rec["checked"] == rec["matched"] == 96):
        rec["ok"] = False; rec["failure"] = rec["failure"] or "O2_7_TRIAL_MANIFEST_PROVENANCE_FAILURE"
    return rec


# ---------------------------------------------------------------- per (subject, ROI) frontier (Tlist-driven)
def subject_roi(s, roi, cells, trialnat, rd, G, leak, Tlist, need_null):
    per_T = {T: {"R_AUG": [], "R_BASE": [], "R_NULL": [], "AXFID": [], "TRAIN": [], "cells": 0} for T in Tlist}
    R0_folds = [float(c["R_CAL0"]) for c in cells]
    Rnat_folds = [float(rd["per_target"][roi][s][f]["R_ORACLE"]) for f in range(N_FOLDS)]
    n_pairs = len(balanced_pairs(list(cells[0]["simple_ids"]), list(cells[0]["nat_ids"])))
    for f in range(N_FOLDS):
        cell = cells[f]; tn = trialnat[f]
        W = np.asarray(cell["W_target"], np.float64); U_res = np.asarray(cell["U_res"], np.float64)
        X = np.asarray(cell["X"], np.float64); V = W.shape[0]
        deltas_test = [np.asarray(x, np.float64) for x in list(cell["deltas_test"])]
        pairs = balanced_pairs(list(cell["simple_ids"]), list(cell["nat_ids"]))
        Dtrial = np.asarray(tn["trial_native"], np.float64)
        for pi, (i1, i2) in enumerate(pairs):
            b8 = _outside_axis(Dtrial[i1].mean(0), Dtrial[i2].mean(0), W)     # full-8 reference axis
            for T in Tlist:
                for si, S in enumerate(itertools.combinations(range(8), T)):
                    Sl = list(S)
                    d1 = Dtrial[i1][Sl].mean(0); d2 = Dtrial[i2][Sl].mean(0)
                    Q = G.orthogonal_procrustes(X[[i1, i2]], np.stack([W.T @ d1, W.T @ d2]))
                    P_IN = G.native_projector(Q, U_res, W)
                    R_base = float(np.mean([G.retention(P_IN, dt) for dt in deltas_test]))
                    bT = _outside_axis(d1, d2, W)
                    leak["max"] = max(leak["max"], float(abs(W.T @ bT).max()))
                    r_aug = R_base + float(np.mean([_frac(bT, dt) for dt in deltas_test]))
                    if need_null:
                        nv = []
                        for it in range(N_NULL):
                            bn = _null_axis("O2.7|%s|%s|%d|%d|%d|%d|%d" % (s, roi, f, pi, T, si, it), W, V, G)
                            nv.append(R_base + float(np.mean([_frac(bn, dt) for dt in deltas_test])))
                        per_T[T]["R_NULL"].append(float(np.mean(nv)))
                    tr = float(np.mean([G.retention(P_IN, Dtrial[i][Sl].mean(0)) + _frac(bT, Dtrial[i][Sl].mean(0)) for i in (i1, i2)]))
                    per_T[T]["R_AUG"].append(r_aug); per_T[T]["R_BASE"].append(R_base)
                    per_T[T]["AXFID"].append(float((bT @ b8) ** 2)); per_T[T]["TRAIN"].append(tr)
                    per_T[T]["cells"] += 1
    R0 = float(np.mean(R0_folds)); Rnat = float(np.mean(Rnat_folds))
    out = {"R0": R0, "R_native": Rnat, "T": {}, "cell_counts": {T: per_T[T]["cells"] for T in Tlist}}
    for T in Tlist:
        # FIX 9: exact expected cell count (every enumerated pair x fold x repeat-subset present)
        exp = N_FOLDS * n_pairs * EXPECTED_SUBSETS[T]
        if per_T[T]["cells"] != exp:
            out["T"][T] = {"evaluable": False, "reason": "cell_count %d != expected %d" % (per_T[T]["cells"], exp)}
            continue
        p = per_T[T]; R_T = float(np.mean(p["R_AUG"]))
        out["T"][T] = {"evaluable": True, "R_AUG": R_T, "R_BASE": float(np.mean(p["R_BASE"])),
                       "R_NULL_mean": (float(np.mean(p["R_NULL"])) if need_null else None),
                       "E_TRIAL": (R_T - float(np.mean(p["R_NULL"]))) if need_null else None,
                       "AXIS_FIDELITY": float(np.mean(p["AXFID"])), "TRAIN_retention": float(np.mean(p["TRAIN"])),
                       "TOTAL_RECOVERY": float(np.clip((R_T - R0) / max(Rnat - R0, EPS), 0, 1)),
                       "TOTAL_RECOVERY_unclipped": (R_T - R0) / max(Rnat - R0, EPS)}
    return out


# ---------------------------------------------------------------- T8 certification (FIX 5,6)
def certify_t8(per_subj_t8, o2_6_aug_csv, o2_6_sha_expected, rois):
    ref = {}; rows = 0; dup = False
    for r in csv.DictReader(open(o2_6_aug_csv)):
        rows += 1
        if int(r["M"]) == 2 and int(r["d"]) == 1:
            k = (r["subject"], r["roi"])
            if k in ref:
                dup = True
            ref[k] = {"R_BASE": float(r["R_BASE"]), "R_AUG": float(r["R_AUG"]), "TOTAL_RECOVERY": float(r["TOTAL_RECOVERY"])}
    prov = {"path": str(o2_6_aug_csv), "sha256": _sha(o2_6_aug_csv), "expected_sha256": o2_6_sha_expected,
            "row_count": rows, "n_m2d1_rows": len(ref), "duplicate_m2d1": dup}
    if (o2_6_sha_expected is not None and prov["sha256"] != o2_6_sha_expected) or dup or len(ref) != 16:
        return {"ok": False, "status": "O2_7_FULL_REPEAT_REPLAY_FAILURE", "reason": "reference provenance/rows", "provenance": prov}
    errs = {"R_BASE": 0.0, "R_AUG": 0.0, "TOTAL_RECOVERY": 0.0}; cells_ok = True
    for roi in rois:
        for s in ALL:
            if (s, roi) not in ref:
                return {"ok": False, "status": "O2_7_FULL_REPEAT_REPLAY_FAILURE", "reason": "missing ref row %s/%s" % (s, roi), "provenance": prov}
            cell_ct = per_subj_t8[s][roi]["cell_counts"][8]
            if cell_ct != N_FOLDS * N_PAIRS * 1:                             # 150
                cells_ok = False
            g = per_subj_t8[s][roi]["T"][8]
            for m in errs:
                errs[m] = max(errs[m], abs(g[m] - ref[(s, roi)][m]))
    ok = cells_ok and all(v <= REPLAY_TOL for v in errs.values())
    return {"ok": ok, "status": "O2_7_T8_REPLAYS_O2_6_M2_D1" if ok else "O2_7_FULL_REPEAT_REPLAY_FAILURE",
            "replay_class": "FLOAT32_SOURCE_BOUNDED_T8_REPLAY", "tol": REPLAY_TOL, "max_abs_error_per_metric": errs,
            "cells_per_participant_roi": N_FOLDS * N_PAIRS * 1, "cells_ok": cells_ok, "provenance": prov}


def t8_structural(per_subj_t8, rois):
    return {"artifact": "t8_structural_replay", "M": 2, "D": 1, "n_balanced_pairs": N_PAIRS, "n_outer_folds": N_FOLDS,
            "cells_per_participant_roi": N_FOLDS * N_PAIRS * 1, "full8_identity_residual_mean": True,
            "same_W_target": True, "same_U_res": True, "same_train_identity_order": True,
            "same_procrustes": True, "same_outside_axis_svd": True, "same_full_repeat_heldout_targets": True,
            "cell_counts_ok": all(per_subj_t8[s][roi]["cell_counts"][8] == N_FOLDS * N_PAIRS for roi in rois for s in ALL)}


# ---------------------------------------------------------------- inference + status (FIX 8)
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
            E = [per_subj[s][roi]["T"][T]["E_TRIAL"] for s in ALL]
            tot = [per_subj[s][roi]["T"][T]["TOTAL_RECOVERY"] for s in ALL]
            grp[roi][T] = {"evaluable": True, "median_E_TRIAL": _median(E), "n_E_pos": sum(e > 0 for e in E),
                           "median_TOTAL": _median(tot), "n_TOTAL_ge0.5": sum(t >= 0.5 for t in tot),
                           "median_AXIS_FIDELITY": _median([per_subj[s][roi]["T"][T]["AXIS_FIDELITY"] for s in ALL]),
                           "E": E}
            pvals["%s|T%d" % (roi, T)] = F.signflip_p_onesided(E)
    holm = F.holm(pvals, alpha=0.05) if pvals else {}                        # 6-test family (T8 never included)
    roi_status = {}
    for roi in rois:
        tstar = "BELOW_FULL_RESOURCE_NOT_ESTABLISHED"
        for T in T_PRIMARY:
            g = grp[roi][T]
            if not g.get("evaluable"):
                continue
            key = "%s|T%d" % (roi, T); g["signflip_p"] = float(pvals[key]); g["holm_reject"] = bool(holm.get(key, False))
            g["meets_all"] = (g["median_E_TRIAL"] > 0 and g["n_E_pos"] >= 6 and g["holm_reject"]
                              and g["median_TOTAL"] >= 0.50 and g["n_TOTAL_ge0.5"] >= 6)
            if g["meets_all"] and tstar == "BELOW_FULL_RESOURCE_NOT_ESTABLISHED":
                tstar = T
        st = {1: "ONE_REPEAT_TARGET_STATE_CALIBRATION_SUFFICIENT", 2: "TWO_REPEAT_TARGET_STATE_CALIBRATION_SUFFICIENT",
              4: "FOUR_REPEAT_TARGET_STATE_CALIBRATION_SUFFICIENT"}.get(tstar, "REDUCED_REPEAT_TARGET_STATE_CALIBRATION_NOT_ESTABLISHED")
        roi_status[roi] = {"T_STAR": tstar, "status": st, "per_T": grp[roi]}
    tv, tl = roi_status["ventral"]["T_STAR"], roi_status["lateral"]["T_STAR"]
    iv = tv if isinstance(tv, int) else None; il = tl if isinstance(tl, int) else None
    if iv is not None and il is not None:                                    # both finite
        m = max(iv, il)
        prog = ("MINIMAL_TARGET_STATE_CALIBRATION_TWO_TOTAL_TRIALS" if m == 1
                else "MINIMAL_TARGET_STATE_CALIBRATION_FOUR_TOTAL_TRIALS" if m == 2
                else "MINIMAL_TARGET_STATE_CALIBRATION_EIGHT_TOTAL_TRIALS")
    elif (iv is None) ^ (il is None):                                        # exactly one finite (FIX 8)
        prog = "TARGET_STATE_TRIAL_CALIBRATION_MULTIREGIME"
    else:
        prog = "REDUCED_TRIAL_TARGET_STATE_CALIBRATION_NOT_ESTABLISHED"
    return roi_status, prog


# ---------------------------------------------------------------- execution stages
def _load(state_dir, trial_dir, rois):
    per = {s: {} for s in ALL}
    for roi in rois:
        for s in ALL:
            cells = [dict(np.load(state_dir / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            tn = [dict(np.load(trial_dir / f"trialnat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            per[s][roi] = (cells, tn)
    return per


def _provenance_gate(out, state_dir, trial_dir, rd_results_path, manifest_path, trial_manifest, rd_sha,
                     trial_manifest_sha, rd, rois):
    prov = {"path": str(rd_results_path), "sha256": _sha(rd_results_path), "expected_sha256": rd_sha,
            "match": (rd_sha is None or _sha(rd_results_path) == rd_sha)}
    (out / "rd_results_provenance.json").write_text(json.dumps(prov, indent=2))
    if not prov["match"]:
        print("O2_7_TRIAL_CALIBRATION_INCONCLUSIVE (rd_results provenance)"); return False
    man = {f["name"]: f for f in json.loads(Path(manifest_path).read_text())["files"]}
    hv = {"checked": 0, "ok": True}
    for roi in rois:
        for s in ALL:
            for f in range(N_FOLDS):
                n = f"cell_{s}_{roi}_fold{f}.npz"; hv["checked"] += 1
                if man.get(n, {}).get("sha256") != _sha(state_dir / n):
                    hv["ok"] = False
    (out / "sealed_state_verification.json").write_text(json.dumps(hv, indent=2))
    if not hv["ok"]:
        print("O2_7_TRIAL_CALIBRATION_INCONCLUSIVE (sealed-state hash)"); return False
    tv = verify_trial_state(state_dir, trial_dir, trial_manifest, rd, rois, trial_manifest_sha)
    (out / "trial_state_replay_certification.json").write_text(json.dumps(tv, indent=2, default=str))
    if not tv["ok"]:
        print(tv["failure"]); return False
    return True


def run(args):
    G = _geom(args.repo)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    state_dir, trial_dir = Path(args.state), Path(args.trial)
    rd = json.loads(Path(args.rd_results).read_text())
    rois = [r.strip() for r in args.rois.split(",") if r.strip()]
    if not _provenance_gate(out, state_dir, trial_dir, Path(args.rd_results), Path(args.manifest),
                            Path(args.trial_manifest), args.rd_results_sha256, args.trial_manifest_sha256, rd, rois):
        return 1
    loaded = _load(state_dir, trial_dir, rois)
    # real-data structural guard: M=2 must yield exactly 25 balanced pairs (5 simple + 5 naturalistic)
    for roi in rois:
        for s in ALL:
            c0 = loaded[s][roi][0][0]
            if len(balanced_pairs(list(c0["simple_ids"]), list(c0["nat_ids"]))) != 25:
                print("O2_7_TRIAL_CALIBRATION_INCONCLUSIVE (expected 25 balanced pairs)"); return 1
    leak = {"max": 0.0}

    if args.stage == "t8":                                                   # STAGE A: T8 ONLY (FIX 1,5,6,7)
        per = {s: {roi: subject_roi(s, roi, *loaded[s][roi], rd, G, leak, [8], need_null=False) for roi in rois} for s in ALL}
        cert = certify_t8(per, args.o2_6_aug, args.o2_6_aug_sha256, rois)
        (out / "t8_o2_6_replay_certification.json").write_text(json.dumps(cert, indent=2, default=str))
        (out / "t8_structural_replay_certification.json").write_text(json.dumps(t8_structural(per, rois), indent=2, default=str))
        print(cert["status"], "max_err=%s" % cert.get("max_abs_error_per_metric"))
        return 0 if cert["ok"] else 1

    # STAGE B: reduced-repeat frontier -- guarded by a valid Stage-A T8 certification (FIX 1)
    certp = out / "t8_o2_6_replay_certification.json"
    if not certp.exists() or json.loads(certp.read_text()).get("status") != "O2_7_T8_REPLAYS_O2_6_M2_D1":
        print("O2_7_ACCESS_ORDER_VIOLATION: valid Stage-A T8 certification required before reduced-repeat frontier")
        raise SystemExit(3)
    per = {s: {roi: subject_roi(s, roi, *loaded[s][roi], rd, G, leak, T_PRIMARY, need_null=True) for roi in rois} for s in ALL}
    # full-8 reference R8 (deterministic; for FULL_RESOURCE_FRACTION) -- not a Holm-family member
    per8 = {s: {roi: subject_roi(s, roi, *loaded[s][roi], rd, G, {"max": 0.0}, [8], need_null=False) for roi in rois} for s in ALL}
    roi_status, prog = aggregate(per, G)

    def _w(name, header, rows):
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    fr, ax, frf, tr, gc = [], [], [], [], []
    for roi in rois:
        for T in T_PRIMARY:
            g = roi_status[roi]["per_T"][T]
            if g.get("evaluable"):
                fr.append([roi, T, g["median_E_TRIAL"], g["n_E_pos"], g.get("signflip_p", ""), g.get("holm_reject", ""),
                           g["median_TOTAL"], g["n_TOTAL_ge0.5"], g["median_AXIS_FIDELITY"]])
        for s in ALL:
            R8 = per8[s][roi]["T"][8]["R_AUG"]; R0 = per[s][roi]["R0"]
            for T in T_PRIMARY:
                v = per[s][roi]["T"][T]
                if not v.get("evaluable"):
                    continue
                frac = float(np.clip((v["R_AUG"] - R0) / max(R8 - R0, EPS), 0, 1))
                ax.append([s, roi, T, v["AXIS_FIDELITY"]]); frf.append([s, roi, T, frac, (v["R_AUG"] - R0) / max(R8 - R0, EPS)])
                tr.append([s, roi, T, v["TOTAL_RECOVERY"], v["TOTAL_RECOVERY_unclipped"], v["E_TRIAL"], v["R_AUG"], v["R_BASE"]])
                gc.append([s, roi, T, v["TRAIN_retention"], v["R_AUG"]])
    _w("trial_frontier_results.csv", ["roi", "T", "median_E_TRIAL", "n_E_pos", "signflip_p", "holm_reject", "median_TOTAL", "n_TOTAL_ge0.5", "median_AXIS_FIDELITY"], fr)
    _w("participant_roi_trial_frontier.csv", ["subject", "roi", "T", "TOTAL_RECOVERY", "TOTAL_RECOVERY_unclipped", "E_TRIAL", "R_AUG", "R_BASE"], tr)
    _w("axis_fidelity_results.csv", ["subject", "roi", "T", "AXIS_FIDELITY"], ax)
    _w("full_resource_fraction.csv", ["subject", "roi", "T", "FULL_RESOURCE_FRACTION", "unclipped"], frf)
    _w("total_recovery_results.csv", ["subject", "roi", "T", "TOTAL_RECOVERY", "unclipped", "E_TRIAL", "R_AUG", "R_BASE"], tr)
    _w("generalization_control.csv", ["subject", "roi", "T", "train_calibration_retention", "heldout_R_AUG"], gc)
    (out / "repeat_schedule_manifest.csv").write_text("T,n_repeat_subsets,cells_per_participant_roi,total_trials\n"
        + "\n".join("%d,%d,%d,%d" % (T, EXPECTED_SUBSETS[T], N_FOLDS * N_PAIRS * EXPECTED_SUBSETS[T], 2 * T) for T in [1, 2, 4, 8]) + "\n")
    (out / "primary_inference.json").write_text(json.dumps({"family": [("%s|T%d" % (r, T)) for r in rois for T in T_PRIMARY],
        "family_size": 6, "T8_in_family": False,
        "per_test": {roi: {T: {k: roi_status[roi]["per_T"][T].get(k) for k in ("signflip_p", "holm_reject", "median_E_TRIAL", "n_E_pos")}
                           for T in T_PRIMARY} for roi in rois}}, indent=2, default=str))
    (out / "minimum_repeat_budget.json").write_text(json.dumps({roi: {"T_STAR": roi_status[roi]["T_STAR"],
        "total_trials": (2 * roi_status[roi]["T_STAR"] if isinstance(roi_status[roi]["T_STAR"], int) else None)} for roi in rois}, indent=2, default=str))
    (out / "primary_roi_status.json").write_text(json.dumps({roi: {"status": roi_status[roi]["status"], "T_STAR": roi_status[roi]["T_STAR"]} for roi in rois}, indent=2, default=str))
    def _ov(roi):
        tr1 = _median([per[s][roi]["T"][1]["TRAIN_retention"] for s in ALL]); ho1 = _median([per[s][roi]["T"][1]["R_AUG"] for s in ALL])
        return "LOW_REPEAT_TARGET_STATE_OVERFIT_SIGNATURE" if (tr1 > 0.9 and ho1 < 0.3) else "none"
    (out / "scientific_status.json").write_text(json.dumps(
        {"status": prog, "roi_status": {roi: roi_status[roi]["status"] for roi in rois},
         "T_STAR": {roi: roi_status[roi]["T_STAR"] for roi in rois},
         "t8_replay": json.loads((out / "t8_o2_6_replay_certification.json").read_text())["status"],
         "t8_replay_class": "FLOAT32_SOURCE_BOUNDED_T8_REPLAY", "outside_leak_max": leak["max"],
         "overfit_signature": {roi: _ov(roi) for roi in rois},
         "immutable": {"O2_6": "TARGET_STATE_MISSING_BASIS_LOW_COMPLEXITY"}, "O3": "O3_NOT_READY"}, indent=2, default=str))
    (out / "matched_null_summary.json").write_text(json.dumps({"n_null_per_cell": N_NULL,
        "seed": "O2.7|subject|ROI|fold|identity_pair|T|repeat_subset_id|null_iteration", "outside_leak_max": leak["max"]}, indent=2))
    print("O2_7_STATUS", prog)
    for roi in rois:
        print("  [%s] %s T_STAR=%s" % (roi, roi_status[roi]["status"], roi_status[roi]["T_STAR"]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--stage", required=True, choices=["t8", "frontier"])
    ap.add_argument("--state", required=True); ap.add_argument("--trial", required=True)
    ap.add_argument("--rd-results", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--trial-manifest", required=True); ap.add_argument("--trial-manifest-sha256", default=None)
    ap.add_argument("--rd-results-sha256", default=None); ap.add_argument("--o2-6-aug", required=True)
    ap.add_argument("--o2-6-aug-sha256", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--rois", default="ventral,lateral")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
