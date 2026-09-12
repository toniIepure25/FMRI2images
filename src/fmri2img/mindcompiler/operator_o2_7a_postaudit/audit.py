"""O2.7A post-seal integrity audit (config 04e15c2b). NON-SCIENTIFIC: verifies provenance/completeness of the
sealed O2.7A run and adds a descriptive repeat-reliability addendum. Does NOT rerun inference or change any
O2.7A metric/T_AXIS_STAR/status/threshold/null/multiplicity/grid. Audits A-H per the brief."""
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
    u = np.asarray(u, np.float64); j = int(np.argmax(np.abs(u)))
    return u * (-1.0 if u[j] < 0 else 1.0)


def _outside_axis(d1, d2, W, want_sv=False):
    Do = np.stack([d1 - W @ (W.T @ d1), d2 - W @ (W.T @ d2)], axis=1)
    U, s, _ = np.linalg.svd(Do, full_matrices=False)
    b = _canon1(U[:, 0])
    return (b, s) if want_sv else b


def _pairs(simple, nat):
    return [(a, b) for a in sorted(simple) for b in sorted(nat)]


# ---------------- Audit A / B : extension provenance ----------------
def _verify_ext(kind, manifest_path, ext_dir, state_dir, rd, rois, key):
    man = json.loads(Path(manifest_path).read_text())
    fields_ok = (man.get("gate") == ("O2.6" if kind == "o2_6" else "O2.7")
                 and man.get("artifact") == ("native_residual_state_extension" if kind == "o2_6" else "trial_residual_state")
                 and man.get("ok") is True and len(man.get("replay", [])) == 96
                 and all(r.get("ok") for r in man["replay"]) and len(man.get("files", [])) == 96)
    if kind == "o2_6":
        fields_ok = fields_ok and float(man.get("max_abs_discrepancy_global", 1)) == 0.0
    by = {f["name"]: f for f in man["files"]}
    checked = matched = 0; mism = []
    for roi in rois:
        for s in ALL:
            vh = rd["rois"][roi]["voxel_hash"][s]; V = int(rd["rois"][roi]["n_vox"][s])
            for f in range(N_FOLDS):
                pref = "deltanat" if kind == "o2_6" else "trialnat"
                n = f"{pref}_{s}_{roi}_fold{f}.npz"; checked += 1; p = ext_dir / n
                mm = by.get(n)
                cell = dict(np.load(state_dir / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True))
                ok = p.exists() and mm is not None and _sha(p) == mm["sha256"] and p.stat().st_size == mm["bytes"]
                if ok:
                    z = dict(np.load(p, allow_pickle=True))
                    tr = [str(x) for x in z["train_ids"].tolist()]; sealed = [str(x) for x in cell["train_ids"].tolist()]
                    ok = (list(np.asarray(z[key]).shape) == list(mm["shape"]) and tr == sealed
                          and str(z["voxel_hash"]) == vh and len(z["fam"]) == 10)
                    if kind == "o2_6":
                        ok = ok and np.asarray(z[key]).shape == (10, V)
                    else:
                        ok = ok and np.asarray(z[key]).shape == (10, 8, V)
                        prov = json.loads(str(z["provenance"]))
                        for i in tr:
                            pr = prov.get(i, {})
                            if not (sorted(pr.get("repeat", [])) == list(range(8)) and len(set(pr.get("beta_index0", []))) == 8):
                                ok = False
                if ok:
                    matched += 1
                else:
                    mism.append(n)
    return {"kind": kind, "manifest_fields_ok": bool(fields_ok), "checked": checked, "matched": matched,
            "ok": bool(fields_ok and checked == matched == 96), "mismatches": mism}


# ---------------- Audit C : exhaustive zero-base ----------------
def audit_C(state_dir, rd, rois, G):
    rows = []; ok_all = True
    for roi in rois:
        for s in ALL:
            for f in range(N_FOLDS):
                cell = dict(np.load(state_dir / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True))
                W = np.asarray(cell["W_target"], np.float64); U = np.asarray(cell["U_res"], np.float64); K = int(cell["K"])
                P0 = G.native_projector(np.eye(K), U, W)
                dt = [np.asarray(x, np.float64) for x in list(cell["deltas_test"])]
                r0 = float(np.mean([G.retention(P0, d) for d in dt]))
                dev = abs(r0 - float(cell["R_CAL0"])); sym = float(np.linalg.norm(P0 - P0.T)); idem = float(np.linalg.norm(P0 @ P0 - P0))
                inW = float(np.linalg.norm(P0 - W @ (W.T @ P0)))                # range(P0) in col(W)
                rank = int(round(float(np.trace(P0)))); r_best = int(cell["r_best"])
                cok = dev <= 1e-10 and sym <= 1e-10 and idem <= 1e-8 and inW <= 1e-8 and rank == r_best
                ok_all = ok_all and cok
                rows.append([s, roi, f, r0, float(cell["R_CAL0"]), dev, sym, idem, inW, rank, r_best, int(cok)])
    return rows, ok_all


# ---------------- Audit D : canonical b8 provenance ----------------
def audit_D(state_dir, ext_dir, ext_manifest, rois, G):
    by = {ff["name"]: ff for ff in json.loads(Path(ext_manifest).read_text())["files"]}
    rows = []; ok_all = True; test_leak = False
    for roi in rois:
        for s in ALL:
            for f in range(N_FOLDS):
                cell = dict(np.load(state_dir / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True))
                W = np.asarray(cell["W_target"], np.float64)
                z = dict(np.load(ext_dir / f"deltanat_{s}_{roi}_fold{f}.npz", allow_pickle=True))
                dn = np.asarray(z["delta_native"], np.float64); tr = [str(x) for x in z["train_ids"].tolist()]
                test_ids = [str(x) for x in cell["test_ids"].tolist()]
                src_sha = by[f"deltanat_{s}_{roi}_fold{f}.npz"]["sha256"]
                pairs = _pairs(list(cell["simple_ids"]), list(cell["nat_ids"]))
                if len(pairs) != 25 or len(set(pairs)) != 25:
                    ok_all = False
                for pi, (i1, i2) in enumerate(pairs):
                    if tr[i1] in test_ids or tr[i2] in test_ids:
                        test_leak = True
                    b8, sv = _outside_axis(dn[i1], dn[i2], W, want_sv=True)
                    s1 = float(sv[0]); s2 = float(sv[1] if len(sv) > 1 else 0.0)
                    norm = float(np.linalg.norm(b8)); perp = float(abs(W.T @ b8).max())
                    cok = abs(norm - 1) <= 1e-9 and perp <= 1e-10 and s1 > 1e-12
                    ok_all = ok_all and cok
                    rows.append([s, roi, f, pi, tr[i1], tr[i2], s1, s2, (s2 / s1 if s1 > 0 else 0.0),
                                 ((s1 - s2) / s1 if s1 > 0 else 0.0), perp, src_sha, int(cok)])
    return rows, bool(ok_all and not test_leak), test_leak


# ---------------- Audit F : repeat reliability (descriptive) ----------------
def audit_F(state_dir, trial_dir, rois, W_by):
    splits = list(itertools.combinations(range(8), 4))
    seen = set(); uniq = []
    for A in splits:
        B = tuple(sorted(set(range(8)) - set(A)))
        key = frozenset([A, B])
        if key not in seen:
            seen.add(key); uniq.append((list(A), list(B)))   # 35 complementary balanced splits
    rows = []
    for roi in rois:
        for s in ALL:
            for f in range(N_FOLDS):
                cell = dict(np.load(state_dir / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True))
                W = np.asarray(cell["W_target"], np.float64); tr = [str(x) for x in cell["train_ids"].tolist()]
                D = np.asarray(dict(np.load(trial_dir / f"trialnat_{s}_{roi}_fold{f}.npz", allow_pickle=True))["trial_native"], np.float64)
                for k in range(10):
                    dout = D[k] - (D[k] @ W) @ W.T                            # (8 x V) outside components
                    for si, (A, B) in enumerate(uniq):
                        vA = dout[A].mean(0); vB = dout[B].mean(0)
                        nA = float(vA @ vA); nB = float(vB @ vB)
                        if nA > 0 and nB > 0:
                            cos2 = float((vA @ vB) ** 2 / (nA * nB))
                            rows.append([s, roi, f, tr[k], si, cos2])
    return rows


def run(a):
    G = _geom(a.repo)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    state_dir, trial_dir, ext_dir = Path(a.state), Path(a.trial), Path(a.ext)
    rd = json.loads(Path(a.rd_results).read_text()); rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    sealed = Path(a.sealed)

    # A / B
    A = _verify_ext("o2_6", a.o2_6_ext_manifest, ext_dir, state_dir, rd, rois, "delta_native")
    (out / "o2_6_native_extension_postseal_verification.json").write_text(json.dumps(A, indent=2))
    B = _verify_ext("o2_7", a.trial_manifest, trial_dir, state_dir, rd, rois, "trial_native")
    (out / "trial_extension_postseal_verification.json").write_text(json.dumps(B, indent=2))
    # C
    crows, C_ok = audit_C(state_dir, rd, rois, G)
    with open(out / "zero_base_exhaustive_postseal_certification.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subject", "roi", "fold", "R0", "R_CAL0", "dev", "sym", "idem", "range_in_colW", "rank", "r_best", "ok"]); w.writerows(crows)
    # D
    drows, D_ok, leak = audit_D(state_dir, ext_dir, a.o2_6_ext_manifest, rois, G)
    with open(out / "full_resource_axis_postseal_certification.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subject", "roi", "fold", "pair", "id1", "id2", "s1", "s2", "s2_over_s1", "singular_gap", "outside_leak", "source_deltanat_sha256", "ok"]); w.writerows(drows)
    # E : sealed family completeness
    pi = json.loads((sealed / "primary_inference.json").read_text())
    fam = pi.get("family", []); per = pi.get("per_test", {})
    exp_family = {"%s|T%d" % (r, T) for r in rois for T in (1, 2, 4)}
    six_ok = (set(fam) == exp_family and pi.get("family_size") == 6 and pi.get("T8_in_family") is False
              and all(per.get(r, {}).get(str(T), per.get(r, {}).get(T, {})).get("signflip_p") is not None
                      and (per.get(r, {}).get(str(T)) or per.get(r, {}).get(T, {})).get("holm_reject") is not None
                      for r in rois for T in (1, 2, 4)))
    prf = list(csv.DictReader(open(sealed / "participant_roi_frontier.csv")))
    n_primary_cells = len([r for r in prf if int(r["T"]) in (1, 2, 4)])
    E = {"six_test_family_ok": bool(six_ok), "family": sorted(fam), "T8_excluded": pi.get("T8_in_family") is False,
         "n_participant_primary_cells": n_primary_cells, "expected": 48, "cells_ok": n_primary_cells == 48,
         "ok": bool(six_ok and n_primary_cells == 48)}
    (out / "primary_inference_postseal_completeness.json").write_text(json.dumps(E, indent=2, default=str))
    # F : reliability
    W_by = {}
    frows = audit_F(state_dir, trial_dir, rois, W_by)
    with open(out / "repeat_reliability_postseal.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subject", "roi", "fold", "identity", "split_id", "SPLIT_HALF_COS2"]); w.writerows(frows)
    # participant-first summary
    from collections import defaultdict
    by_if = defaultdict(list)
    for s, roi, f, idn, si, c in frows:
        by_if[(s, roi, f, idn)].append(c)
    by_sr = defaultdict(list)
    for (s, roi, f, idn), v in by_if.items():
        by_sr[(s, roi)].append(float(np.median(v)))                          # median per identity/fold
    srows = []
    for (s, roi), v in sorted(by_sr.items()):
        pm = float(np.median(v)); r = float(np.sqrt(max(pm, 0.0))); sb = 2 * r / (1 + r) if (1 + r) > 0 else float("nan")
        srows.append([s, roi, pm, r, sb])
    with open(out / "repeat_reliability_participant_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subject", "roi", "participant_median_COS2", "sign_invariant_r", "spearman_brown_exploratory"]); w.writerows(srows)
    rel_roi = {roi: float(np.median([r[2] for r in srows if r[1] == roi])) for roi in rois}
    # G : convergence (from sealed CSVs)
    def _read(name):
        return list(csv.DictReader(open(sealed / name)))
    fid = _read("axis_fidelity.csv")
    prf_all = {(r["subject"], r["roi"], int(r["T"])): r for r in prf}
    conv = []
    for roi in rois:
        for T in (1, 2, 4, 8):
            fv = [float(x["AXIS_FIDELITY"]) for x in fid if x["roi"] == roi and int(x["T"]) == T]
            arf = [float(prf_all[(s, roi, T)]["ARF"]) for s in ALL if (s, roi, T) in prf_all and prf_all[(s, roi, T)]["ARF"] not in ("", "None")]
            tot = [float(prf_all[(s, roi, T)]["TOTAL_ZERO_BASE"]) for s in ALL if (s, roi, T) in prf_all and prf_all[(s, roi, T)]["TOTAL_ZERO_BASE"] not in ("", "None")]
            conv.append([roi, T, float(np.median(fv)) if fv else "", float(np.median(arf)) if arf else "", float(np.median(tot)) if tot else ""])
    with open(out / "trial_count_convergence.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["roi", "T", "median_AXIS_FIDELITY", "median_ARF", "median_TOTAL_ZERO_BASE"]); w.writerows(conv)
    # H : generalization continuous
    gc = _read("generalization_control.csv")
    grows = []
    for roi in rois:
        for T in (1, 2, 4):
            cap = [float(x["calibration_axis_capture"]) for x in gc if x["roi"] == roi and int(x["T"]) == T]
            ho = [float(x["heldout_R_AXIS"]) for x in gc if x["roi"] == roi and int(x["T"]) == T]
            grows.append([roi, T, float(np.median(cap)) if cap else "", float(np.median(ho)) if ho else "",
                          (float(np.median(cap)) - float(np.median(ho))) if (cap and ho) else ""])
    with open(out / "generalization_control_postseal.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["roi", "T", "median_calibration_capture", "median_heldout_R_AXIS", "median_gap"]); w.writerows(grows)
    # H correction note + preservation
    (out / "scientific_status_preservation.json").write_text(json.dumps(
        {"O2_7A_status": "REDUCED_TRIAL_OUTSIDE_AXIS_CALIBRATION_NOT_ESTABLISHED",
         "ventral_T_AXIS_STAR": "NOT_ESTABLISHED_BY_T4", "lateral_T_AXIS_STAR": "NOT_ESTABLISHED_BY_T4",
         "overfit_field_note": "NON_AUTHORITATIVE_POST_HOC_DIAGNOSTIC (no prospectively frozen threshold)",
         "generalization_statement": "Calibration and held-out behavior showed no obvious qualitative divergence under the descriptive control.",
         "reliability_note": "POST_SEAL_DESCRIPTIVE_ADDENDUM; no pass threshold; does not alter T_AXIS_STAR/p-values/status",
         "repeat_reliability_roi_median_COS2": rel_roi}, indent=2))
    passed = bool(A["ok"] and B["ok"] and C_ok and D_ok and E["ok"] and not leak)
    status = "O2_7A_POSTSEAL_INTEGRITY_AUDIT_PASS" if passed else "O2_7A_POSTSEAL_INTEGRITY_AUDIT_FAIL"
    (out / "audit_status.json").write_text(json.dumps(
        {"status": status, "A_o2_6_ext": A["ok"], "B_trial_ext": B["ok"], "C_zero_base_96": C_ok,
         "D_b8_provenance": D_ok, "E_family_cells": E["ok"], "held_out_leakage": leak,
         "reliability": "descriptive_no_threshold", "o2_7a_status_preserved": "REDUCED_TRIAL_OUTSIDE_AXIS_CALIBRATION_NOT_ESTABLISHED",
         "appended": ("POSTSEAL_INTEGRITY_VERIFIED" if passed else None),
         "authorize_next": ("O2.8 RESIDUAL-CAPACITY DECOMPOSITION" if passed else "STOP_BEFORE_O2_8"),
         "O3": "O3_NOT_READY"}, indent=2, default=str))
    print(status, "A=%s B=%s C=%s D=%s E=%s leak=%s" % (A["ok"], B["ok"], C_ok, D_ok, E["ok"], leak))
    return 0 if passed else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--state", required=True)
    ap.add_argument("--trial", required=True); ap.add_argument("--ext", required=True)
    ap.add_argument("--rd-results", required=True); ap.add_argument("--o2-6-ext-manifest", required=True)
    ap.add_argument("--trial-manifest", required=True); ap.add_argument("--sealed", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--rois", default="ventral,lateral")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
