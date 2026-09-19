"""MINDIR-X1 Gauge-Invariant Shared Transformation Law (frozen config 85c0f628). EXPLORATORY discovery branch
(N=8 development cohort). Tests whether subjects share coordinate-INVARIANT properties of the perception->imagery
transformation despite subject-specific orientations. Reuses ONLY sealed O2 geometry (W_target + delta_native);
all invariants live in native voxel space and are invariant to orthogonal rotation of W_target columns and to
singular/eigenvector sign flips. Four frozen invariant blocks (A spectral shape, B principal angles, C support
partition, D compression) -> per-subject signature -> LOSO donor-mean prediction vs per-component permutation
null -> participant E_SHARED -> exact 8-test (4 blocks x 2 ROIs) sign-flip + Holm. Cannot modify any O2 seal or
unlock O3. No new estimator / no hyperparameter search."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

ALL = [f"subj0{i}" for i in range(1, 9)]
N_FOLDS = 6
BLOCKS = ["A_spectral", "B_angles", "C_support", "D_compression"]
N_NULL = 1000
O2_10_ORIENTATION = {"ventral": "COMPOSITE_GEOMETRY_NOT_SHARED", "lateral": "COMPOSITE_GEOMETRY_NOT_SHARED"}  # immutable historical (O2.10)
_G = None


def _geom(repo):
    global _G
    if _G is None:
        from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
        _G = G
    return _G


def _svals(M):
    s = np.linalg.svd(np.asarray(M, np.float64), compute_uv=False)
    return s[s > s[0] * 1e-12] if s.size and s[0] > 0 else s


def _entropy(p):
    p = p[p > 0]
    return float(-np.sum(p * np.log(p))) if p.size else 0.0


def _partic(s):
    s2 = s ** 2
    return float((s2.sum()) ** 2 / np.sum(s2 ** 2)) if s2.sum() > 0 else 0.0


def _orth(M, k):
    U, s, _ = np.linalg.svd(np.asarray(M, np.float64), full_matrices=False)
    return U[:, :k]


# ---------------- invariant blocks (native voxel space; rotation/sign invariant) ----------------
def invariants(W, Dn, r, r_common):
    """W: (V x K) orthonormal perception support; Dn: (10 x V) imagery residuals; r=r_best; r_common frozen."""
    W = np.asarray(W, np.float64); Dn = np.asarray(Dn, np.float64); V, K = W.shape
    Z = Dn @ W                                                                 # (10 x K) within-support coords
    O = Dn - Z @ W.T                                                           # (10 x V) outside residuals
    rc = int(r_common)
    # A spectral shape (within-support coordinate spectrum)
    sA = _svals(Z)
    sAn = sA / sA.sum() if sA.sum() > 0 else sA
    a_vec = list(sAn[:rc]) + [0.0] * max(0, rc - len(sAn))
    a_ent = _entropy((sA ** 2) / np.sum(sA ** 2)) if sA.sum() > 0 else 0.0
    a_erank = _partic(sA); a_cond = float(sA[0] / sA[min(r, len(sA)) - 1]) if len(sA) >= 1 else 1.0
    A = np.array(a_vec + [a_ent, a_erank, a_cond], np.float64)
    # B principal angles: col(W) (K) vs raw imagery support top-(r+2) of Dn
    Qimg = _orth(Dn.T, min(r + 2, Dn.shape[0], V))                             # (V x (<=r+2))
    cos2 = np.sort(_svals(W.T @ Qimg) ** 2)[::-1]                              # cos^2 principal angles
    b_vec = list(cos2[:rc]) + [0.0] * max(0, rc - len(cos2))
    b_mean = float(np.mean(cos2)) if cos2.size else 0.0; b_med = float(np.median(cos2)) if cos2.size else 0.0
    b_maxcos = float(cos2[0]) if cos2.size else 0.0; b_mincos = float(cos2[-1]) if cos2.size else 0.0
    b_ent = _entropy(cos2 / cos2.sum()) if cos2.sum() > 0 else 0.0
    B = np.array(b_vec + [b_mean, b_med, b_maxcos, b_mincos, b_ent], np.float64)
    # C support partition
    e_in = float(np.sum([w @ w for w in Z @ W.T])); e_tot = float(np.sum(Dn ** 2))
    frac_in = e_in / e_tot if e_tot > 0 else 0.0; frac_out = 1.0 - frac_in
    ratio = frac_out / max(frac_in, 1e-12)
    sO = _svals(O); out_erank = _partic(sO)
    oe = (sO ** 2) / np.sum(sO ** 2) if sO.sum() > 0 else sO
    C = np.array([frac_in, frac_out, ratio, out_erank, float(oe[0]) if len(oe) else 0.0, float(oe[1]) if len(oe) > 1 else 0.0], np.float64)
    # D compression (full imagery operator spectrum)
    sD = _svals(Dn); e = (sD ** 2) / np.sum(sD ** 2) if sD.sum() > 0 else sD
    top4 = list(e[:4]) + [0.0] * max(0, 4 - len(e))
    eff = _partic(sD); cum = np.cumsum(e); cumk = [float(cum[k - 1]) if len(cum) >= k else 1.0 for k in (1, 2, 3, 4)]
    D = np.array(top4 + [eff] + cumk, np.float64)
    return {"A_spectral": A, "B_angles": B, "C_support": C, "D_compression": D}


# ---------------- LOSO donor-mean vs permutation null ----------------
def _wdist(x, m, sd):
    return float(np.sqrt(np.sum(((x - m) / sd) ** 2)))


def e_shared(target, donors, block, seed_prefix, G):
    D = np.stack(donors)                                                       # (7 x d)
    mu = D.mean(0); sd = D.std(0, ddof=0); sd = np.where(sd < 1e-12, 1.0, sd)
    w = target / sd; m = mu / sd
    dist_pred = float(np.sqrt(np.sum((w - m) ** 2)))
    dn = []
    for it in range(N_NULL):
        rng = np.random.Generator(np.random.PCG64(G.seed_uint64("%s|%d" % (seed_prefix, it))))
        dn.append(float(np.sqrt(np.sum((rng.permutation(w) - m) ** 2))))
    return float(np.mean(dn) - dist_pred), dist_pred, float(np.mean(dn))


# ---------------- driver ----------------
def run(a):
    G = _geom(a.repo)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    state = Path(a.state); ext = Path(a.ext)
    cells = {s: {roi: [dict(np.load(state / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    extn = {s: {roi: [dict(np.load(ext / f"deltanat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    prov = {"state_cells": sum(len(cells[s][r]) for s in ALL for r in rois), "ext_cells": sum(len(extn[s][r]) for s in ALL for r in rois), "expected": 8 * len(rois) * N_FOLDS}
    (out / "sealed_state_verification.json").write_text(json.dumps(prov, indent=2))
    (out / "operator_equivalence_definition.json").write_text(json.dumps(
        {"equivalence_class": "invariants unchanged under W_target->W_target R (R orthogonal KxK), singular/eigenvector sign flips, identity ordering; principal angles invariant to V-space orthogonal transforms",
         "gauge_invariant_means": "tested basis-invariant scalars only; NOT physical gauge symmetry"}, indent=2))
    # r_common per ROI
    r_common = {roi: min(int(cells[s][roi][0]["r_best"]) for s in ALL) for roi in rois}
    # signatures (fold-mean per subject x roi x block)
    sig = {s: {roi: {} for roi in rois} for s in ALL}
    sig_rows = []
    for roi in rois:
        for s in ALL:
            blocks_fold = {b: [] for b in BLOCKS}
            for f in range(N_FOLDS):
                W = np.asarray(cells[s][roi][f]["W_target"], np.float64); r = int(cells[s][roi][f]["r_best"])
                Dn = np.asarray(extn[s][roi][f]["delta_native"], np.float64)
                inv = invariants(W, Dn, r, r_common[roi])
                for b in BLOCKS:
                    blocks_fold[b].append(inv[b])
            for b in BLOCKS:
                sig[s][roi][b] = np.mean(np.stack(blocks_fold[b]), 0)
            sig_rows.append([s, roi, r_common[roi]] + [round(float(x), 6) for b in BLOCKS for x in sig[s][roi][b]])
    _w(out / "subject_invariant_signature.csv", ["subject", "roi", "r_common", "signature..."], sig_rows)
    # per-block CSVs
    for b, fn in (("A_spectral", "spectral_shape.csv"), ("B_angles", "principal_angle_spectrum.csv"), ("C_support", "support_partition.csv"), ("D_compression", "compression_signature.csv")):
        _w(out / fn, ["subject", "roi", "components..."], [[s, roi] + [round(float(x), 6) for x in sig[s][roi][b]] for roi in rois for s in ALL])
    # LOSO E_SHARED + fold robustness + pipeline triviality + inference
    part = {roi: {b: {} for b in BLOCKS} for roi in rois}
    loso_rows = []
    for roi in rois:
        for b in BLOCKS:
            for s in ALL:
                donors = [sig[d][roi][b] for d in ALL if d != s]
                es, dp, dn = e_shared(sig[s][roi][b], donors, b, "X1|%s|%s|%s" % (s, roi, b), G)
                part[roi][b][s] = es
                loso_rows.append([s, roi, b, es, dp, dn])
    _w(out / "loso_invariant_prediction.csv", ["subject", "roi", "block", "E_SHARED", "dist_pred", "dist_null_mean"], loso_rows)
    part_status, infer, roi_status, prog, triviality, synth = classify(part, sig, rois, r_common, cells, extn, G)
    _write(out, part, infer, roi_status, prog, triviality, synth, sig, rois, G, cells, extn, r_common)
    print("X1_STATUS", prog)
    for roi in rois:
        print("  [%s] %s" % (roi, roi_status[roi]))
    return 0


def _median(x):
    return float(np.median(x)) if len(x) else float("nan")


def classify(part, sig, rois, r_common, cells, extn, G):
    from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
    keys = [(roi, b) for roi in rois for b in BLOCKS]
    pvals = {(roi, b): F.signflip_p_onesided([part[roi][b][s] for s in ALL]) for (roi, b) in keys}
    holm = F.holm({f"{roi}|{b}": pvals[(roi, b)] for (roi, b) in keys}, alpha=0.05)
    # pipeline triviality: synthetic constrained-null cohort (random ops preserving rank+energy+support sizes)
    triviality = {}
    for roi in rois:
        for b in BLOCKS:
            real = np.stack([sig[s][roi][b] for s in ALL])
            D_obs = _pairwise_med(real)
            # constrained null: random Dn per subject preserving V,10,r via random gaussian (rank>=r), same K
            nulls = []
            for it in range(50):
                rng = np.random.Generator(np.random.PCG64(G.seed_uint64("X1|triv|%s|%s|%d" % (roi, b, it))))
                cohort = []
                for s in ALL:
                    W = np.asarray(cells[s][roi][0]["W_target"], np.float64); r = int(cells[s][roi][0]["r_best"]); V = W.shape[0]
                    Dn = rng.standard_normal((10, V))
                    cohort.append(invariants(W, Dn, r, r_common[roi])[b])
                nulls.append(_pairwise_med(np.stack(cohort)))
            D_null = float(np.mean(nulls))
            # flag if random-constrained cohort is NOT meaningfully more dispersed than real (invariant forced similar by pipeline)
            constrained = bool(D_null <= D_obs * 1.10)
            triviality[(roi, b)] = {"D_obs": D_obs, "D_null_constrained": D_null, "PIPELINE_CONSTRAINED": constrained}
    infer = {}; supported = {roi: [] for roi in rois}
    for (roi, b) in keys:
        E = [part[roi][b][s] for s in ALL]; rej = bool(holm.get(f"{roi}|{b}", False))
        triv = triviality[(roi, b)]["PIPELINE_CONSTRAINED"]
        sup = bool(_median(E) > 0 and sum(e > 0 for e in E) >= 6 and rej and not triv)
        if sup:
            supported[roi].append(b)
        infer[f"{roi}|{b}"] = {"median_E": _median(E), "n_E_pos": int(sum(e > 0 for e in E)), "signflip_p": float(pvals[(roi, b)]),
                               "holm_reject": rej, "pipeline_constrained": triv, "supported": sup}
    roi_status = {}
    for roi in rois:
        n = len(supported[roi])
        roi_status[roi] = ("INVARIANTS_SUPPORTED(%d)" % n) if n else "NO_INVARIANTS_SUPPORTED"
    both = {b: all(b in supported[roi] for roi in rois) for b in BLOCKS}
    n_both = sum(both.values()); any_sup = any(supported[roi] for roi in rois)
    orientation_neg = all(O2_10_ORIENTATION[roi] == "COMPOSITE_GEOMETRY_NOT_SHARED" for roi in rois)
    if n_both >= 2 and orientation_neg:
        prog = "SHARED_INTRINSIC_STRUCTURE_PRIVATE_ORIENTATION"
    elif n_both >= 2:
        prog = "SHARED_INTRINSIC_TRANSFORMATION_STRUCTURE_SUPPORTED"
    elif any_sup and (supported[rois[0]] != supported[rois[1]]):
        prog = "INVARIANT_STRUCTURE_MULTIREGIME"
    elif not any_sup:
        prog = "NO_REPRODUCIBLE_SHARED_INTRINSIC_STRUCTURE"
    else:
        prog = "INVARIANT_STRUCTURE_MULTIREGIME"
    synth = [[roi, O2_10_ORIENTATION[roi], roi_status[roi], "|".join(supported[roi])] for roi in rois]
    return part, infer, roi_status, prog, triviality, synth


def _pairwise_med(X):
    n = len(X); Xn = X / (np.linalg.norm(X, axis=0) + 1e-12)                    # component-normalized
    d = [float(np.linalg.norm(Xn[i] - Xn[j])) for i in range(n) for j in range(i + 1, n)]
    return float(np.median(d)) if d else 0.0


def _basis_invariance(cells, extn, rois, r_common, G):
    """Apply random orthogonal rotation to W columns + sign flips; require invariants unchanged."""
    rng = np.random.default_rng(0); worst = 0.0
    for roi in rois:
        s = ALL[0]; W = np.asarray(cells[s][roi][0]["W_target"], np.float64); r = int(cells[s][roi][0]["r_best"])
        Dn = np.asarray(extn[s][roi][0]["delta_native"], np.float64); K = W.shape[1]
        R = np.linalg.qr(rng.standard_normal((K, K)))[0]
        i0 = invariants(W, Dn, r, r_common[roi]); i1 = invariants(W @ R, Dn, r, r_common[roi])
        for b in BLOCKS:
            worst = max(worst, float(np.max(np.abs(i0[b] - i1[b]))))
    return worst


def _write(out, part, infer, roi_status, prog, triviality, synth, sig, rois, G, cells, extn, r_common):
    (out / "component_inference.json").write_text(json.dumps({"family_size": 8, "per_test": infer}, indent=2, default=str))
    binv = _basis_invariance(cells, extn, rois, r_common, G)
    (out / "basis_invariance_tests.json").write_text(json.dumps({"max_abs_invariant_change_under_W_rotation": binv, "tol": 1e-10, "pass": bool(binv <= 1e-9)}, indent=2))
    _w(out / "pipeline_triviality_controls.csv", ["roi", "block", "D_obs", "D_null_constrained", "PIPELINE_CONSTRAINED"],
       [[roi, b, triviality[(roi, b)]["D_obs"], triviality[(roi, b)]["D_null_constrained"], int(triviality[(roi, b)]["PIPELINE_CONSTRAINED"])] for roi in rois for b in BLOCKS])
    _w(out / "orientation_vs_invariant_synthesis.csv", ["roi", "o2_10_orientation_transfer", "x1_invariant_status", "supported_blocks"], synth)
    # matched null summary + random-op negative control (already embedded via triviality)
    (out / "matched_null_results.json").write_text(json.dumps(
        {"loso_permutation_null": {"type": "per_component_permutation", "n": N_NULL, "seed": "X1|subject|roi|block|iter"},
         "constrained_random_operator_negative_control": "see pipeline_triviality_controls.csv (random gaussian ops preserving V/10/K/rank)"}, indent=2, default=str))
    # fold robustness (LOFO sign of E_SHARED per supported block)
    fr_rows = []
    for roi in rois:
        for b in [bb for bb in BLOCKS if infer[f"{roi}|{bb}"]["supported"]]:
            stable = True
            for omit in range(N_FOLDS):
                pos = 0
                for s in ALL:
                    sigs = {d: _fold_sig(d, roi, b, cells, extn, r_common, omit) for d in ALL}
                    es, _, _ = e_shared(sigs[s], [sigs[d] for d in ALL if d != s], b, "X1|LOFO|%s|%s|%s|%d" % (s, roi, b, omit), G)
                    pos += (es > 0)
                if pos < 6:
                    stable = False
            fr_rows.append([roi, b, int(stable)])
    _w(out / "fold_robustness.csv", ["roi", "block", "LOFO_STABLE"], fr_rows)
    # cross-ROI consistency (participant signature correlation ventral vs lateral)
    xroi = []
    if len(rois) >= 2:
        for s in ALL:
            v = np.concatenate([sig[s][rois[0]][b] for b in BLOCKS]); l = np.concatenate([sig[s][rois[1]][b] for b in BLOCKS])
            vn = (v - v.mean()) / (v.std() + 1e-12); ln = (l - l.mean()) / (l.std() + 1e-12)
            xroi.append([s, float(np.mean(vn * ln))])
    _w(out / "cross_roi_consistency.csv", ["subject", "signature_correlation_ventral_lateral"], xroi)
    # subject identifiability (descriptive: LOFO orientation self-match via within-support subspace)
    ident = []
    for roi in rois:
        correct = 0
        for s in ALL:
            tgt = _within_sub(s, roi, 0, cells, extn)
            best = max(ALL, key=lambda d: _subspace_sim(tgt, _within_sub(d, roi, 1, cells, extn)))
            correct += (best == s)
        ident.append([roi, correct, len(ALL)])
    _w(out / "subject_identifiability.csv", ["roi", "correct_self_match", "n"], ident)
    (out / "roi_status.json").write_text(json.dumps(roi_status, indent=2))
    (out / "scientific_status.json").write_text(json.dumps(
        {"status": prog, "roi_status": roi_status, "basis_invariance_max_change": binv, "discovery_cohort": True,
         "immutable_o2": "unchanged", "O3": "O3_NOT_READY"}, indent=2, default=str))
    (out / "next_gate_decision.json").write_text(json.dumps({"status": prog,
        "next": "X1.1 MINIMAL CANONICAL FORM" if prog == "SHARED_INTRINSIC_STRUCTURE_PRIVATE_ORIENTATION" else ("X1.1 ROI-SPECIFIC CLASSES" if prog == "INVARIANT_STRUCTURE_MULTIREGIME" else "seal negative")}, indent=2))


def _fold_sig(s, roi, b, cells, extn, r_common, omit):
    vs = []
    for f in range(N_FOLDS):
        if f == omit:
            continue
        W = np.asarray(cells[s][roi][f]["W_target"], np.float64); r = int(cells[s][roi][f]["r_best"])
        Dn = np.asarray(extn[s][roi][f]["delta_native"], np.float64)
        vs.append(invariants(W, Dn, r, r_common[roi])[b])
    return np.mean(np.stack(vs), 0)


def _within_sub(s, roi, f, cells, extn):
    W = np.asarray(cells[s][roi][f]["W_target"], np.float64); r = int(cells[s][roi][f]["r_best"])
    Dn = np.asarray(extn[s][roi][f]["delta_native"], np.float64)
    Z = Dn @ W; U = _orth(Z.T, min(r, W.shape[1]))                             # within-support orientation (K-space)
    return W @ U                                                              # (V x r) native orientation


def _subspace_sim(A, B):
    k = min(A.shape[1], B.shape[1])
    return float(np.sum((_orth(A, k).T @ _orth(B, k)) ** 2) / k)


def _w(path, header, rows):
    if str(path).endswith(".json_rows"):
        return
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        if header:
            w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--state", required=True); ap.add_argument("--ext", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--rois", default="ventral,lateral")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
