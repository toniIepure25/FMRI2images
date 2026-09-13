"""O2.8 Residual-Capacity Decomposition (frozen config f2af6206). DIAGNOSTIC full-resource identity-centroid
2x2x2 factorial (A outside-identity-diversity M2/M10 x B outside-dimension D1/D2 x C within-support orientation
P_ZERO_RD/P_SUPPORT_ORACLE_R) with an exact 3-factor Shapley decomposition of the remaining native-oracle gap.
Consumes ONLY sealed O2.3A-RD/O2.5/O2.6/O2.7A state; no raw betas, no trial data, no model search.
Orthogonal ranges => R_ABC = retention(P_IN_C, delta) + ||B_OUT_D^T delta||^2/||delta||^2."""
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
N_PAIRS = 25
CELLS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
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


def support_oracle_basis(Z, r):
    U, _, _ = np.linalg.svd(np.asarray(Z, np.float64).T, full_matrices=False)
    return _canon(U[:, :int(r)])                                             # (K x r), O2.5 convention


def outside_basis(Dout, D):
    """Top-D left singular vectors of the (V x m) outside-support residual matrix; canonical signs. None if
    numerical rank < D."""
    U, s, _ = np.linalg.svd(np.asarray(Dout, np.float64), full_matrices=False)
    rank = int(np.sum(s > (s.max() * 1e-9))) if s.size else 0
    return None if rank < D else _canon(U[:, :D])


def _frac(B, delta):
    d2 = float(delta @ delta)
    return 0.0 if d2 <= 0 else float((B.T @ delta) @ (B.T @ delta) / d2)


def _pairs(simple, nat):
    return [(a, b) for a in sorted(simple) for b in sorted(nat)]


def shapley(R):
    """Exact 3-factor Shapley of v(S)=R_S - R000 over the 8 cells R[(a,b,c)]. Returns {A,B,C} phi."""
    R000 = R[(0, 0, 0)]
    def v(cell): return R[cell] - R000
    W = {0: 1.0 / 3, 1: 1.0 / 6, 2: 1.0 / 3}
    phi = {}
    for idx, fac in enumerate("ABC"):
        others = [j for j in range(3) if j != idx]
        tot = 0.0
        for rsz in range(3):
            for combo in itertools.combinations(others, rsz):
                base = [0, 0, 0]
                for j in combo:
                    base[j] = 1
                wi = base[:]; wi[idx] = 1
                tot += W[len(combo)] * (v(tuple(wi)) - v(tuple(base)))
        phi[fac] = tot
    return phi


# ---------------------------------------------------------------- per (subject, ROI)
def subject_roi(s, roi, cells, extnat, rd, G, cert, leak):
    R_folds = {c: [] for c in CELLS}
    Rnat_folds, c1_replay = [], []
    for f in range(N_FOLDS):
        cell = cells[f]
        W = np.asarray(cell["W_target"], np.float64); U_res = np.asarray(cell["U_res"], np.float64)
        Z = np.asarray(cell["Z"], np.float64); K = int(cell["K"]); r_best = int(cell["r_best"])
        dt = [np.asarray(x, np.float64) for x in list(cell["deltas_test"])]
        P_ZERO = G.native_projector(np.eye(K), U_res, W)
        R_IN = {0: float(np.mean([G.retention(P_ZERO, d) for d in dt]))}
        # C1 within-support oracle (O2.5)
        Usup = support_oracle_basis(Z, r_best); Bsup = W @ Usup; P_SUPP = Bsup @ Bsup.T
        R_IN[1] = float(np.mean([G.retention(P_SUPP, d) for d in dt]))
        c1_replay.append({"subject": s, "roi": roi, "fold": f, "R_SUPPORT_ORACLE_R": R_IN[1]})
        cert["m0_dev"] = max(cert["m0_dev"], abs(R_IN[0] - float(cell["R_CAL0"])))
        # outside components of the 10 outer-training identities
        Dn = np.asarray(extnat[f]["delta_native"], np.float64)               # (10 x V)
        dout = np.stack([Dn[k] - W @ (W.T @ Dn[k]) for k in range(10)])       # (10 x V)
        leak["max"] = max(leak["max"], float(np.max([abs(W.T @ dout[k]).max() for k in range(10)])))
        # A1 (M=10) bases
        B_A1 = {D: outside_basis(dout.T, D) for D in (1, 2)}
        # A0 (M=2) averaged outside fractions
        simple = list(cell["simple_ids"]); nat = list(cell["nat_ids"]); pairs = _pairs(simple, nat)
        fracA0 = {1: [], 2: []}
        for (i1, i2) in pairs:
            Do = np.stack([dout[i1], dout[i2]], axis=1)                       # (V x 2)
            for D in (1, 2):
                B = outside_basis(Do, D)
                fracA0[D].append(np.mean([_frac(B, d) for d in dt]) if B is not None else np.nan)
        fr = {(0, 1): float(np.mean(fracA0[1])), (0, 2): float(np.mean(fracA0[2])),
              (1, 1): float(np.mean([_frac(B_A1[1], d) for d in dt])) if B_A1[1] is not None else np.nan,
              (1, 2): float(np.mean([_frac(B_A1[2], d) for d in dt])) if B_A1[2] is not None else np.nan}
        for (a, b, c) in CELLS:
            R_folds[(a, b, c)].append(R_IN[c] + fr[(a, b + 1)])               # b in {0,1} -> D in {1,2}
        Rnat_folds.append(float(rd["per_target"][roi][s][f]["R_ORACLE"]))
        if f == 0:                                                            # representative P_ABC certification
            B = B_A1[2]
            if B is not None:
                Pabc = P_SUPP + B @ B.T
                cert["sym"] = max(cert["sym"], float(np.linalg.norm(Pabc - Pabc.T)))
                cert["idem"] = max(cert["idem"], float(np.linalg.norm(Pabc @ Pabc - Pabc)))
                cert["cross"] = max(cert["cross"], float(np.linalg.norm(P_SUPP @ (B @ B.T))))
                cert["rank_ok"] = cert["rank_ok"] and abs(np.trace(Pabc) - (np.trace(P_SUPP) + 2)) < 1e-6
    R = {c: float(np.mean(R_folds[c])) for c in CELLS}                        # participant cell values
    if any(np.isnan(v) for v in R.values()):
        return {"evaluable": False, "reason": "RANK_INSUFFICIENT", "c1_replay": c1_replay}
    Rnat = float(np.mean(Rnat_folds))
    phi = shapley(R)
    G_REMAIN = Rnat - R[(0, 0, 0)]
    return {"evaluable": True, "R": R, "R_native": Rnat, "R000": R[(0, 0, 0)], "R111": R[(1, 1, 1)],
            "G_REMAIN": G_REMAIN, "phi": phi, "shapley_sum": sum(phi.values()),
            "FULL_CLOSURE": (R[(1, 1, 1)] - R[(0, 0, 0)]) / max(G_REMAIN, 1e-12),
            "UNEXPLAINED": (Rnat - R[(1, 1, 1)]) / max(G_REMAIN, 1e-12), "c1_replay": c1_replay}


# ---------------------------------------------------------------- aggregate / inference / status
def _median(x):
    return float(np.median(x)) if len(x) else float("nan")


def aggregate(per_subj, G):
    from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
    rois = list(next(iter(per_subj.values())).keys())
    pvals = {}; roi_out = {}
    for roi in rois:
        ev = [s for s in ALL if per_subj[s][roi].get("evaluable")]
        if len(ev) < 7:
            roi_out[roi] = {"status": "RESIDUAL_CAPACITY_DECOMPOSITION_INCONCLUSIVE", "n_evaluable": len(ev)}; continue
        phis = {fac: [per_subj[s][roi]["phi"][fac] for s in ev] for fac in "ABC"}
        fc = [per_subj[s][roi]["FULL_CLOSURE"] for s in ev]
        supp = {}
        for fac in "ABC":
            pvals["%s|%s" % (roi, fac)] = F.signflip_p_onesided(phis[fac])
        roi_out[roi] = {"n_evaluable": len(ev), "phi_median": {fac: _median(phis[fac]) for fac in "ABC"},
                        "n_phi_pos": {fac: int(sum(x > 0 for x in phis[fac])) for fac in "ABC"},
                        "median_FULL_CLOSURE": _median([min(max(x, 0.0), 1.0) for x in fc]),
                        "n_FC_ge0.5": int(sum(min(max(x, 0.0), 1.0) >= 0.5 for x in fc)),
                        "phis": phis}
    holm = F.holm(pvals, alpha=0.05) if pvals else {}
    prog_inputs = {}
    for roi in rois:
        o = roi_out[roi]
        if "phi_median" not in o:
            prog_inputs[roi] = o["status"]; continue
        supported = []
        for fac in "ABC":
            key = "%s|%s" % (roi, fac); rej = bool(holm.get(key, False))
            o.setdefault("supported", {})[fac] = bool(o["phi_median"][fac] > 0 and o["n_phi_pos"][fac] >= 6 and rej)
            o.setdefault("signflip_p", {})[fac] = float(pvals[key]); o.setdefault("holm_reject", {})[fac] = rej
            if o["supported"][fac]:
                supported.append(fac)
        fc_sufficient = (o["median_FULL_CLOSURE"] >= 0.50 and o["n_FC_ge0.5"] >= 6)
        o["full_closure_sufficient"] = bool(fc_sufficient)
        if not fc_sufficient:
            st = "RESIDUAL_CAPACITY_BEYOND_TESTED_FACTORS"
        elif supported == ["A"]:
            st = "OUTSIDE_IDENTITY_DIVERSITY_LIMIT"
        elif supported == ["B"]:
            st = "SECOND_OUTSIDE_DIMENSION_LIMIT"
        elif supported == ["C"]:
            st = "WITHIN_SUPPORT_ORIENTATION_LIMIT"
        elif len(supported) >= 2:
            st = "MULTICOMPONENT_RESIDUAL_CAPACITY_LIMIT"
        else:
            st = "CAPACITY_INTERACTION_DOMINANT"
        o["status"] = st; prog_inputs[roi] = st
    sv, sl = prog_inputs.get("ventral"), prog_inputs.get("lateral")
    limit_map = {"OUTSIDE_IDENTITY_DIVERSITY_LIMIT": "RESIDUAL_CAPACITY_OUTSIDE_IDENTITY_DOMINANT",
                 "SECOND_OUTSIDE_DIMENSION_LIMIT": "RESIDUAL_CAPACITY_SECOND_OUTSIDE_DIMENSION_DOMINANT",
                 "WITHIN_SUPPORT_ORIENTATION_LIMIT": "RESIDUAL_CAPACITY_WITHIN_SUPPORT_ORIENTATION_DOMINANT"}
    if "RESIDUAL_CAPACITY_DECOMPOSITION_INCONCLUSIVE" in (sv, sl):
        prog = "O2_8_RESIDUAL_CAPACITY_INCONCLUSIVE"
    elif sv == "RESIDUAL_CAPACITY_BEYOND_TESTED_FACTORS" or sl == "RESIDUAL_CAPACITY_BEYOND_TESTED_FACTORS":
        prog = "RESIDUAL_CAPACITY_BEYOND_TESTED_FACTORS"
    elif sv == sl and sv in limit_map:
        prog = limit_map[sv]
    else:
        prog = "RESIDUAL_CAPACITY_MULTICOMPONENT"
    return roi_out, prog


def run(a):
    G = _geom(a.repo)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    state_dir, ext_dir = Path(a.state), Path(a.ext)
    rd = json.loads(Path(a.rd_results).read_text()); rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    # provenance
    prov = {"rd_results_sha256": _sha(a.rd_results), "expected": a.rd_results_sha256,
            "rd_match": (a.rd_results_sha256 is None or _sha(a.rd_results) == a.rd_results_sha256)}
    stateman = {f["name"]: f for f in json.loads(Path(a.manifest).read_text())["files"]}
    extman = json.loads(Path(a.ext_manifest).read_text())
    prov["state_hashes_ok"] = all(stateman.get(f"cell_{s}_{roi}_fold{f}.npz", {}).get("sha256") == _sha(state_dir / f"cell_{s}_{roi}_fold{f}.npz")
                                  for roi in rois for s in ALL for f in range(N_FOLDS))
    prov["ext_manifest_ok"] = bool(extman.get("gate") == "O2.6" and extman.get("ok") is True and len(extman.get("files", [])) == 96)
    exby = {f["name"]: f for f in extman["files"]}
    prov["ext_hashes_ok"] = all(exby.get(f"deltanat_{s}_{roi}_fold{f}.npz", {}).get("sha256") == _sha(ext_dir / f"deltanat_{s}_{roi}_fold{f}.npz")
                                for roi in rois for s in ALL for f in range(N_FOLDS))
    (out / "sealed_state_verification.json").write_text(json.dumps(prov, indent=2))
    if not (prov["rd_match"] and prov["state_hashes_ok"] and prov["ext_manifest_ok"] and prov["ext_hashes_ok"]):
        print("O2_8_RESIDUAL_CAPACITY_INCONCLUSIVE (provenance)"); return 1

    cert = {"m0_dev": 0.0, "sym": 0.0, "idem": 0.0, "cross": 0.0, "rank_ok": True}; leak = {"max": 0.0}
    per = {s: {} for s in ALL}; c1_all = []
    for roi in rois:
        for s in ALL:
            cells = [dict(np.load(state_dir / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            en = [dict(np.load(ext_dir / f"deltanat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            r = subject_roi(s, roi, cells, en, rd, G, cert, leak); per[s][roi] = r; c1_all += r["c1_replay"]
    # replay gate 1: baseline R000 vs sealed O2.7A R_AXIS_8
    axis_ref = {(x["subject"], x["roi"]): float(x["R_AXIS_8"]) for x in csv.DictReader(open(a.o2_7a_ref))}
    base_err = 0.0
    for roi in rois:
        for s in ALL:
            if per[s][roi].get("evaluable"):
                base_err = max(base_err, abs(per[s][roi]["R000"] - axis_ref[(s, roi)]))
    base_ok = base_err <= 1e-10
    (out / "o2_7a_baseline_replay.json").write_text(json.dumps(
        {"status": "O2_8_BASELINE_REPLAYS_O2_7A_T8" if base_ok else "O2_8_BASELINE_REPLAY_FAILURE", "max_abs_error": base_err, "tol": 1e-10}, indent=2))
    # replay gate 2: support oracle vs sealed O2.5
    o25 = {(x["subject"], x["roi"], int(x["fold"])): float(x["R_SUPPORT_ORACLE_R"]) for x in csv.DictReader(open(a.o2_5_ref))}
    orc_err = max((abs(x["R_SUPPORT_ORACLE_R"] - o25[(x["subject"], x["roi"], x["fold"])])
                   for x in c1_all if (x["subject"], x["roi"], x["fold"]) in o25), default=1.0)
    orc_ok = orc_err <= 1e-10
    (out / "o2_5_support_oracle_replay.json").write_text(json.dumps(
        {"status": "O2_8_SUPPORT_ORACLE_REPLAYS_O2_5" if orc_ok else "O2_8_SUPPORT_ORACLE_REPLAY_FAILURE", "max_abs_error": orc_err, "tol": 1e-10}, indent=2))
    if not (base_ok and orc_ok):
        print("O2_8 replay gate failed base_err=%.2e orc_err=%.2e" % (base_err, orc_err)); return 1

    # Shapley efficiency certification
    eff = max((abs(per[s][roi]["shapley_sum"] - (per[s][roi]["R111"] - per[s][roi]["R000"]))
               for roi in rois for s in ALL if per[s][roi].get("evaluable")), default=0.0)
    if eff > 1e-12:
        print("O2_8 SHAPLEY_EFFICIENCY_FAILURE %.2e" % eff); return 1
    roi_out, prog = aggregate(per, G)

    def _w(name, header, rows):
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    fc_rows, sh_rows, nf_rows, cell_rows, part_cells, inter_rows = [], [], [], [], [], []
    for roi in rois:
        for s in ALL:
            r = per[s][roi]
            if not r.get("evaluable"):
                continue
            R = r["R"]
            for (a2, b2, c2) in CELLS:
                part_cells.append([s, roi, a2, b2, c2, R[(a2, b2, c2)]])
            sh_rows.append([s, roi, r["phi"]["A"], r["phi"]["B"], r["phi"]["C"], r["shapley_sum"], r["R111"] - r["R000"]])
            gr = max(r["G_REMAIN"], 1e-12)
            nf_rows.append([s, roi, r["phi"]["A"] / gr, r["phi"]["B"] / gr, r["phi"]["C"] / gr,
                            1 - (r["phi"]["A"] + r["phi"]["B"] + r["phi"]["C"]) / gr])
            fc_rows.append([s, roi, r["R000"], r["R111"], r["R_native"], r["G_REMAIN"], r["FULL_CLOSURE"], r["UNEXPLAINED"]])
            # interaction contrasts (descriptive)
            AB = (R[(1, 1, 0)] - R[(1, 0, 0)]) - (R[(0, 1, 0)] - R[(0, 0, 0)])
            AC = (R[(1, 0, 1)] - R[(1, 0, 0)]) - (R[(0, 0, 1)] - R[(0, 0, 0)])
            BC = (R[(0, 1, 1)] - R[(0, 1, 0)]) - (R[(0, 0, 1)] - R[(0, 0, 0)])
            ABC = R[(1, 1, 1)] - R[(1, 1, 0)] - R[(1, 0, 1)] - R[(0, 1, 1)] + R[(1, 0, 0)] + R[(0, 1, 0)] + R[(0, 0, 1)] - R[(0, 0, 0)]
            inter_rows.append([s, roi, AB, AC, BC, ABC])
    for roi in rois:
        o = roi_out[roi]
        if "phi_median" in o:
            cell_rows.append([roi, o["phi_median"]["A"], o["phi_median"]["B"], o["phi_median"]["C"],
                              o["median_FULL_CLOSURE"], o.get("status")])
    _w("participant_factorial_cells.csv", ["subject", "roi", "A", "B", "C", "R"], part_cells)
    _w("factorial_cell_results.csv", ["roi", "median_phi_A", "median_phi_B", "median_phi_C", "median_FULL_CLOSURE", "status"], cell_rows)
    _w("shapley_contributions.csv", ["subject", "roi", "phi_A", "phi_B", "phi_C", "shapley_sum", "R111_minus_R000"], sh_rows)
    _w("normalized_capacity_fractions.csv", ["subject", "roi", "F_A", "F_B", "F_C", "F_UNEXPLAINED"], nf_rows)
    _w("full_closure_results.csv", ["subject", "roi", "R000", "R111", "R_native", "G_REMAIN", "FULL_CLOSURE", "UNEXPLAINED"], fc_rows)
    _w("interaction_diagnostics.csv", ["subject", "roi", "AxB", "AxC", "BxC", "AxBxC"], inter_rows)
    (out / "component_inference.json").write_text(json.dumps(
        {"family": [("%s|%s" % (r, f)) for r in rois for f in "ABC"], "family_size": 6,
         "per_roi": {roi: {"signflip_p": roi_out[roi].get("signflip_p"), "holm_reject": roi_out[roi].get("holm_reject"),
                           "supported": roi_out[roi].get("supported"), "phi_median": roi_out[roi].get("phi_median"),
                           "n_phi_pos": roi_out[roi].get("n_phi_pos")} for roi in rois}}, indent=2, default=str))
    (out / "roi_mechanism_status.json").write_text(json.dumps(
        {roi: {"status": roi_out[roi].get("status"), "supported": roi_out[roi].get("supported"),
               "full_closure_sufficient": roi_out[roi].get("full_closure_sufficient"),
               "median_FULL_CLOSURE": roi_out[roi].get("median_FULL_CLOSURE"), "n_evaluable": roi_out[roi].get("n_evaluable")} for roi in rois}, indent=2, default=str))
    (out / "scientific_status.json").write_text(json.dumps(
        {"status": prog, "roi_status": {roi: roi_out[roi].get("status") for roi in rois},
         "shapley_efficiency_max_err": eff, "P_AUG_cert": cert, "outside_leak_max": leak["max"],
         "baseline_replay": base_ok, "support_oracle_replay": orc_ok,
         "immutable": {"O2_7A": "REDUCED_TRIAL_OUTSIDE_AXIS_CALIBRATION_NOT_ESTABLISHED", "O2_6": "TARGET_STATE_MISSING_BASIS_LOW_COMPLEXITY"},
         "O3": "O3_NOT_READY"}, indent=2, default=str))
    print("O2_8_STATUS", prog)
    for roi in rois:
        o = roi_out[roi]
        print("  [%s] %s | phi_median A=%.4f B=%.4f C=%.4f | FC=%.3f" %
              (roi, o.get("status"), o.get("phi_median", {}).get("A", float("nan")),
               o.get("phi_median", {}).get("B", float("nan")), o.get("phi_median", {}).get("C", float("nan")),
               o.get("median_FULL_CLOSURE", float("nan"))))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--state", required=True); ap.add_argument("--ext", required=True)
    ap.add_argument("--rd-results", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--ext-manifest", required=True)
    ap.add_argument("--o2-7a-ref", required=True); ap.add_argument("--o2-5-ref", required=True)
    ap.add_argument("--rd-results-sha256", default=None); ap.add_argument("--out", required=True); ap.add_argument("--rois", default="ventral,lateral")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
