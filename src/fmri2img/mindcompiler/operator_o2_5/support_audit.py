"""O2.5 Perception-Support Ceiling Audit (frozen config b241a5f0). DIAGNOSTIC ONLY -- fits NO new transfer
model. Consumes ONLY the sealed O2.3A-RD/O2.4R derived state (per-cell npz + rd_results.json), verified by
committed SHA256. No raw betas, no SRM/K/rank refit, no new calibration fit.

Decomposes the O2.4R failure into: full perception-support ceiling (any subspace in col(W_target)),
rank-matched support oracle (best rank-r subspace inside col(W_target), target-informed, train-only), and the
sealed native oracle. Reports gap fractions F_FULL / F_RANK / F_EST, principal angles, and a per-ROI
mechanism classification. Reuses the committed certified geometry (retention, Procrustes, native_projector)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ALL = [f"subj0{i}" for i in range(1, 9)]
ROIS_PRIMARY = ["ventral", "lateral"]
N_FOLDS = 6
EPS = 1e-12
_G = None


def _geom(repo: Path):
    global _G
    if _G is None:
        sys.path.insert(0, str(repo / "src"))
        from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
        _G = G
    return _G


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _canon_cols(U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=np.float64).copy()
    for k in range(U.shape[1]):
        j = int(np.argmax(np.abs(U[:, k])))
        if U[j, k] < 0:
            U[:, k] *= -1.0
    return U


# ------------------------------------------------------------------ hash verification (Part: sealed state)
def verify_state_hashes(state_dir: Path, manifest_path: Path, needed):
    man = json.loads(manifest_path.read_text())
    by_name = {f["name"]: f for f in man["files"]}
    rec = {"manifest": str(manifest_path), "n_manifest": man["n_files"], "checked": 0, "ok": True, "mismatches": []}
    for name in needed:
        if name not in by_name:
            rec["ok"] = False; rec["mismatches"].append({"name": name, "why": "not in manifest"}); continue
        got = _sha256_file(state_dir / name)
        rec["checked"] += 1
        if got != by_name[name]["sha256"]:
            rec["ok"] = False
            rec["mismatches"].append({"name": name, "expected": by_name[name]["sha256"], "got": got})
    return rec


# ------------------------------------------------------------------ rank-matched support-oracle basis
def support_oracle_basis(Z: np.ndarray, r: int) -> np.ndarray:
    """Top-r left singular vectors of Z.T (K x r), canonical signs. Z = outer-TRAINING target residual coords
    only (Z_i = W_target.T delta_target_i); NO test residual enters this fit. Rank is the frozen r_best."""
    Usvd, _, _ = np.linalg.svd(np.asarray(Z, np.float64).T, full_matrices=False)
    return _canon_cols(Usvd[:, :int(r)])


# ------------------------------------------------------------------ per-cell computation
def cell_metrics(cell: dict, r_oracle_sealed: float, G):
    W = np.asarray(cell["W_target"], np.float64)              # (p x K)
    U_res = np.asarray(cell["U_res"], np.float64)             # (K x r)
    X = np.asarray(cell["X"], np.float64)                     # (10 x K) RD template coords
    Z = np.asarray(cell["Z"], np.float64)                     # (10 x K) target-imagery coords
    deltas = [np.asarray(d, np.float64) for d in cell["deltas_test"]]   # 2 x (p,)
    r = int(cell["r_best"]); K = int(cell["K"])
    # Part 1: W_target orthonormality + P_SUPPORT
    orth_err = float(np.linalg.norm(W.T @ W - np.eye(K), "fro"))
    P_SUP = W @ W.T
    sym_err = float(np.linalg.norm(P_SUP - P_SUP.T, "fro"))
    idem_err = float(np.linalg.norm(P_SUP @ P_SUP - P_SUP, "fro"))
    # Part 2: full support ceiling (mean over 2 held-out identities)
    R_full = float(np.mean([G.retention(P_SUP, d) for d in deltas]))
    # baseline M0 (sealed P_ZERO_RD retention)
    R0 = float(cell["R_CAL0"])
    # Part 3 bound inputs: M0 and reconstructed M=10 calibrated projector (single all-10 subset)
    P0 = G.native_projector(np.eye(K), U_res, W)
    R_m0_recon = float(np.mean([G.retention(P0, d) for d in deltas]))
    Q10 = G.orthogonal_procrustes(X, Z)
    P10 = G.native_projector(Q10, U_res, W)
    R_m10 = float(np.mean([G.retention(P10, d) for d in deltas]))
    # Part 4: rank-matched support oracle (train-only Z; top-r of Z.T inside col(W_target))
    U_sup_or = support_oracle_basis(Z, r)                    # (K x r)
    B = W @ U_sup_or                                         # (p x r)
    P_sup_or = B @ B.T
    R_sup_or = float(np.mean([G.retention(P_sup_or, d) for d in deltas]))
    # Part 10: principal angles span(U_res) vs span(U_sup_or)
    cs = np.clip(np.linalg.svd(U_res.T @ U_sup_or, compute_uv=False), -1.0, 1.0)
    angles = np.degrees(np.arccos(cs))
    return {"orth_err": orth_err, "sym_err": sym_err, "idem_err": idem_err,
            "R0": R0, "R_m0_recon": R_m0_recon, "R_m10": R_m10, "R_full": R_full,
            "R_sup_or": R_sup_or, "R_native_oracle": float(r_oracle_sealed), "r_best": r, "K": K,
            "pa_cos": cs.tolist(), "pa_deg": angles.tolist(),
            "pa_mean_sq_cos": float(np.mean(cs ** 2)), "pa_min_cos": float(np.min(cs))}


# ------------------------------------------------------------------ aggregation + classification
def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def classify_roi(part):
    """part[subject] = participant dict with F_FULL/F_RANK/F_EST (or NOT_EVALUABLE flags)."""
    evaluable = [s for s in ALL if part[s].get("evaluable_native")]
    n_eval = len(evaluable)
    if n_eval < 7:
        return {"status": "SUPPORT_MAPPING_MULTIREGIME_OR_INCONCLUSIVE", "reason": "n_evaluable<7",
                "n_evaluable": n_eval}
    ffull = [part[s]["F_FULL"] for s in evaluable]
    frank = [part[s]["F_RANK"] for s in evaluable]
    est = [part[s]["F_EST"] for s in evaluable if part[s].get("evaluable_est")]
    med_ff = float(np.median(ffull)); med_fr = float(np.median(frank))
    med_fe = float(np.median(est)) if est else float("nan")
    n_ff_lo = sum(x < 0.50 for x in ffull); n_fr_lo = sum(x < 0.50 for x in frank)
    n_fe_lo = sum(x < 0.50 for x in est)
    out = {"n_evaluable": n_eval, "median_F_FULL": med_ff, "median_F_RANK": med_fr,
           "median_F_EST": med_fe, "n_F_FULL_lt_0.5": n_ff_lo, "n_F_RANK_lt_0.5": n_fr_lo,
           "n_F_EST_lt_0.5": n_fe_lo, "n_est_evaluable": len(est)}
    if med_ff < 0.50 and n_ff_lo >= 6:
        out["status"] = "PERCEPTION_SUPPORT_CEILING_DOMINANT"
    elif med_ff >= 0.50 and med_fr < 0.50 and n_fr_lo >= 6:
        out["status"] = "RANK_MATCHED_SUPPORT_LIMIT_DOMINANT"
    elif med_fr >= 0.50 and est and med_fe < 0.50 and n_fe_lo >= 6:
        out["status"] = "WITHIN_SUPPORT_MAPPING_ESTIMATOR_DOMINANT"
    elif med_fr >= 0.50 and est and med_fe >= 0.50:
        out["status"] = "SUPPORT_AND_MAPPING_ADEQUATE"
    else:
        out["status"] = "SUPPORT_MAPPING_MULTIREGIME_OR_INCONCLUSIVE"
    return out


def program_status(roi_status):
    v, l = roi_status["ventral"]["status"], roi_status["lateral"]["status"]
    if any(s == "SUPPORT_MAPPING_MULTIREGIME_OR_INCONCLUSIVE" for s in (v, l)):
        # inconclusive only if a ROI is genuinely inconclusive/evaluability failure
        pass
    if v == l == "PERCEPTION_SUPPORT_CEILING_DOMINANT":
        return "TARGET_IMAGERY_RESIDUAL_OUTSIDE_PERCEPTION_SUPPORT_DOMINANT"
    if v == l == "RANK_MATCHED_SUPPORT_LIMIT_DOMINANT":
        return "TARGET_IMAGERY_RESIDUAL_RANK_LIMIT"
    if v == l == "WITHIN_SUPPORT_MAPPING_ESTIMATOR_DOMINANT":
        return "TARGET_IMAGERY_RESIDUAL_WITHIN_SUPPORT_MAPPING_LIMIT"
    if "SUPPORT_MAPPING_MULTIREGIME_OR_INCONCLUSIVE" in (v, l) and not (v == l):
        # one ROI inconclusive -> provenance/evaluability path unless the other is a clean mechanism;
        # treat mixed-with-inconclusive as diagnostic inconclusive
        return "O2_5_SUPPORT_DIAGNOSTIC_INCONCLUSIVE"
    return "TARGET_IMAGERY_RESIDUAL_SUPPORT_MULTIREGIME"


def run_audit(repo: Path, state_dir: Path, rd_results_path: Path, manifest_path: Path, out: Path, rois):
    G = _geom(repo)
    out.mkdir(parents=True, exist_ok=True)
    rd = json.loads(rd_results_path.read_text())
    # sealed-state hash verification (all consumed cell files)
    needed = [f"cell_{s}_{roi}_fold{f}.npz" for roi in rois for s in ALL for f in range(N_FOLDS)]
    hv = verify_state_hashes(state_dir, manifest_path, needed)
    hv["rd_results_sha256"] = _sha256_file(rd_results_path)
    (out / "sealed_state_verification.json").write_text(json.dumps(hv, indent=2))
    if not hv["ok"]:
        (out / "scientific_status.json").write_text(json.dumps({"status": "O2_5_SUPPORT_DIAGNOSTIC_INCONCLUSIVE",
                                                                "reason": "sealed-state hash mismatch"}, indent=2))
        print("O2_5_STATUS O2_5_SUPPORT_DIAGNOSTIC_INCONCLUSIVE (hash mismatch)"); return 1

    wcert, bcert, full_rows, ror_rows, gap_rows, pa_rows = [], [], [], [], [], []
    roi_part = {}
    bound_ok = True; orth_ok = True
    for roi in rois:
        part = {}
        for s in ALL:
            folds = []
            for f in range(N_FOLDS):
                cell = dict(np.load(state_dir / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True))
                cell["deltas_test"] = list(cell["deltas_test"])
                r_or = rd["per_target"][roi][s][f]["R_ORACLE"]
                m = cell_metrics(cell, r_or, G)
                # provenance: r_best in cell matches rd_results
                assert int(cell["r_best"]) == int(rd["per_target"][roi][s][f]["r_best"]), "r_best provenance mismatch"
                folds.append(m)
                if m["orth_err"] > 1e-8:
                    orth_ok = False
                if not (m["R_m0_recon"] <= m["R_full"] + 1e-10 and m["R_m10"] <= m["R_full"] + 1e-10):
                    bound_ok = False
                wcert.append([s, roi, f, m["orth_err"], m["sym_err"], m["idem_err"]])
                bcert.append([s, roi, f, m["R_m0_recon"], m["R_m10"], m["R_full"],
                              int(m["R_m0_recon"] <= m["R_full"] + 1e-10), int(m["R_m10"] <= m["R_full"] + 1e-10)])
                full_rows.append([s, roi, f, m["R0"], m["R_full"]])
                ror_rows.append([s, roi, f, m["R_sup_or"], m["R_native_oracle"], m["r_best"]])
                pa_rows.append([s, roi, f, m["pa_mean_sq_cos"], m["pa_min_cos"],
                                ";".join("%.6f" % c for c in m["pa_cos"]), ";".join("%.3f" % a for a in m["pa_deg"])])
            # participant-first aggregation (mean over 6 folds)
            R0 = float(np.mean([m["R0"] for m in folds]))
            R_m10 = float(np.mean([m["R_m10"] for m in folds]))
            R_full = float(np.mean([m["R_full"] for m in folds]))
            R_sup_or = float(np.mean([m["R_sup_or"] for m in folds]))
            R_nat = float(np.mean([m["R_native_oracle"] for m in folds]))
            den_nat = R_nat - R0
            rec = {"R0": R0, "R_m10": R_m10, "R_full": R_full, "R_sup_or": R_sup_or, "R_native": R_nat,
                   "den_native": den_nat}
            if den_nat <= EPS:
                rec.update({"evaluable_native": False, "note": "NOT_EVALUABLE_FOR_NATIVE_GAP"})
            else:
                rec["evaluable_native"] = True
                rec["F_FULL_unclipped"] = (R_full - R0) / den_nat
                rec["F_FULL"] = _clip01(rec["F_FULL_unclipped"])
                rec["F_RANK_unclipped"] = (R_sup_or - R0) / den_nat
                rec["F_RANK"] = _clip01(rec["F_RANK_unclipped"])
                den_sup = R_sup_or - R0
                if den_sup <= EPS:
                    rec.update({"evaluable_est": False, "note_est": "NOT_EVALUABLE_FOR_ESTIMATOR_EFFICIENCY"})
                else:
                    rec["evaluable_est"] = True
                    rec["F_EST_unclipped"] = (R_m10 - R0) / den_sup
                    rec["F_EST"] = _clip01(rec["F_EST_unclipped"])
            part[s] = rec
            gap_rows.append([s, roi, R0, R_m10, R_full, R_sup_or, R_nat,
                             rec.get("F_FULL", ""), rec.get("F_RANK", ""), rec.get("F_EST", ""),
                             rec.get("evaluable_native", False), rec.get("evaluable_est", "")])
        roi_part[roi] = part

    roi_status = {roi: classify_roi(roi_part[roi]) for roi in rois}
    prim = [r for r in rois if r in ROIS_PRIMARY]
    prog = program_status(roi_status) if set(prim) == set(ROIS_PRIMARY) else "O2_5_SUPPORT_DIAGNOSTIC_INCONCLUSIVE"
    if not (bound_ok and orth_ok):
        prog = "O2_5_W_TARGET_ORTHONORMALITY_FAILURE" if not orth_ok else "O2_5_SUPPORT_BOUND_IMPLEMENTATION_FAILURE"

    # write CSVs
    def _w(name, header, rows):
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    _w("w_target_certification.csv", ["subject", "roi", "fold", "orth_err_fro", "sym_err", "idem_err"], wcert)
    _w("support_bound_certification.csv", ["subject", "roi", "fold", "R_m0_recon", "R_m10", "R_SUPPORT_FULL", "m0_le_full", "m10_le_full"], bcert)
    _w("full_support_results.csv", ["subject", "roi", "fold", "R0", "R_SUPPORT_FULL"], full_rows)
    _w("rank_matched_support_oracle.csv", ["subject", "roi", "fold", "R_SUPPORT_ORACLE_R", "R_NATIVE_ORACLE", "r_best"], ror_rows)
    _w("gap_fraction_results.csv", ["subject", "roi", "R0", "R_M10", "R_SUPPORT_FULL", "R_SUPPORT_ORACLE_R", "R_NATIVE_ORACLE", "F_FULL", "F_RANK", "F_EST", "evaluable_native", "evaluable_est"], gap_rows)
    _w("principal_angle_results.csv", ["subject", "roi", "fold", "mean_sq_cos", "min_cos", "cosines", "angles_deg"], pa_rows)
    # participant-roi summary
    prs = []
    for roi in rois:
        for s in ALL:
            r = roi_part[roi][s]
            prs.append([s, roi, r["R0"], r["R_m10"], r["R_full"], r["R_sup_or"], r["R_native"],
                        r.get("F_FULL", ""), r.get("F_RANK", ""), r.get("F_EST", ""),
                        r.get("evaluable_native", False), r.get("evaluable_est", "")])
    _w("participant_roi_summary.csv", ["subject", "roi", "R0", "R_M10", "R_SUPPORT_FULL", "R_SUPPORT_ORACLE_R", "R_NATIVE_ORACLE", "F_FULL", "F_RANK", "F_EST", "evaluable_native", "evaluable_est"], prs)
    (out / "primary_roi_classification.json").write_text(json.dumps(roi_status, indent=2, default=str))
    (out / "scientific_status.json").write_text(json.dumps(
        {"status": prog, "roi_status": {r: roi_status[r]["status"] for r in rois},
         "bound_ok": bound_ok, "orthonormality_ok": orth_ok,
         "immutable": {"O2_3A_RD": "CORE_ANCHOR_RD_INCONCLUSIVE",
                       "O2_4R": "TARGET_STATE_ORIENTATION_NOT_RECOVERED_BY_ORTHOGONAL_CALIBRATION"},
         "O3": "O3_NOT_READY"}, indent=2))
    print("O2_5_STATUS", prog)
    for roi in rois:
        st = roi_status[roi]
        print("  [%s] %s | median F_FULL=%.3f F_RANK=%.3f F_EST=%s (n_eval=%d)" %
              (roi, st["status"], st.get("median_F_FULL", float("nan")), st.get("median_F_RANK", float("nan")),
               ("%.3f" % st["median_F_EST"]) if st.get("median_F_EST") == st.get("median_F_EST") else "nan",
               st.get("n_evaluable", 0)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--state", required=True)
    ap.add_argument("--rd-results", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--rois", default="ventral,lateral")
    a = ap.parse_args()
    return run_audit(Path(a.repo), Path(a.state), Path(a.rd_results), Path(a.manifest), Path(a.out),
                     [r.strip() for r in a.rois.split(",") if r.strip()])


if __name__ == "__main__":
    raise SystemExit(main())
