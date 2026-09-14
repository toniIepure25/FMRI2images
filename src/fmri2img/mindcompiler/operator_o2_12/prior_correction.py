"""O2.12 Minimal Imagery Calibration with Perception-Only Prior Control (frozen config f63c6adc). Tests whether
the FAILED O2.11 perception-only geometry prediction can still REDUCE the sealed O2.9 8-trial target-imagery
calibration burden when used ONLY as a fixed ONE-PSEUDO-OBSERVATION covariance prior.

CONTROL = the exact O2.9 estimator (reuses topk_energy_gram verbatim -> guarantees O2.9 replay).
HYBRID  = same limited target residuals + frozen O2.11 prior via a one-pseudo-observation covariance prior:
          within  C_HYB_IN  = C_DATA_IN  + (e_IN/r_best) U_PRIOR U_PRIOR^T   (top r_best eigvectors)
          outside C_HYB_OUT = C_DATA_OUT + (e_OUT/2)     B_PRIOR B_PRIOR^T   (top 2 eigvectors)
Both Gram-optimized (no K x K or V x V materialization). Geometry prior->hybrid->oracle similarities are dt-free
(oracle uses the 10 training-identity residuals, not held-out) and are computed in Stage A.

Two sealed stages:
  --stage predict  : verify inputs + O2.11 prior hashes + O2.9 control replay (sealed) + dt-free hybrid geometry
                     diagnostics; freeze/hash; NO held-out HYBRID retention.
  --stage evaluate : open held-out imagery -> R_CONTROL / R_HYBRID -> participant DELTA_PRIOR -> exact 8-test
                     sign-flip + Holm -> operational sufficiency -> common prior-enabled frontier -> seal.
No lambda/prior-weight/model search; no rotation-fit / correlation / regression / network estimators."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from fmri2img.mindcompiler.operator_o2_9 import composite_calibration as O9

ALL = [f"subj0{i}" for i in range(1, 9)]
N_FOLDS = 6
PRIMARY_CELLS = [(2, 1), (2, 2), (4, 1), (6, 1)]                                # all O2.9 cells with M*T<8
REFERENCE_CELLS = [(4, 2), (10, 8)]                                            # descriptive only
ALL_CELLS = PRIMARY_CELLS + REFERENCE_CELLS
M_SUB = O9.M_SUB; T_SUB = O9.T_SUB; D_OUT = 2
EPS = np.finfo(np.float64).eps
_G = None


def _geom(repo):
    global _G
    if _G is None:
        from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
        _G = G
    return _G


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------- hybrid Gram energy + subspace-similarity primitives ----------------
def _hyb_topk(Gmat, lam_diag, k):
    """Top-k eigen-decomposition of Lsqrt Gmat Lsqrt (== nonzero spectrum of C_HYB in its span).
    Returns (nu_topk, W_topk, ls) where the fitted subspace vectors are u_j = Zin ls*W[:,j]/sqrt(nu_j)."""
    ls = np.sqrt(np.asarray(lam_diag, np.float64))
    A = (ls[:, None] * np.asarray(Gmat, np.float64)) * ls[None, :]
    A = 0.5 * (A + A.T)
    nu, W = np.linalg.eigh(A)
    order = np.argsort(nu)[::-1]
    nu = nu[order]; W = W[:, order]
    if nu.size < k or nu[k - 1] <= EPS * max(A.shape) * max(nu[0], 0.0):
        return None, None, ls
    return nu[:k], W[:, :k], ls


def _hyb_energy(nu, W, ls, rhs, k):
    """Projection energy of test vector onto the top-k fitted subspace: sum_j (u_j . target)^2 where
    u_j . target = W[:,j] . (ls*rhs) / sqrt(nu_j) (rhs = Zin^T (test coord))."""
    b = W.T @ (ls * np.asarray(rhs, np.float64))
    return float(np.sum(b[:k] ** 2 / nu[:k]))


def _hyb_sim(nu, W, ls, MrhsMat, k, rank_out):
    """||U_HYB^T Ref||_F^2 where Ref is an orthonormal basis and MrhsMat = Zin^T Ref ((n) x rank_out).
    U_HYB[:,j]^T Ref = W[:,j] . (ls[:,None]*MrhsMat) / sqrt(nu_j)."""
    B = (W.T @ (ls[:, None] * np.asarray(MrhsMat, np.float64))) / np.sqrt(nu)[:, None]
    return float(np.sum(B[:k] ** 2))


# ---------------- core per (subject, ROI) ----------------
def core(s, roi, cells, trialnat, extnat, priors, rd, leak, hybrid_heldout, want_refskip=None):
    """Compute per-cell R_CONTROL (always; == O2.9), R_HYBRID (only if hybrid_heldout), and dt-free geometry
    similarities (always). Participant-first grand-mean aggregation. Returns per-cell dict."""
    G = _G
    r_best = int(cells[0]["r_best"])
    R0 = float(np.mean([float(cells[f]["R_CAL0"]) for f in range(N_FOLDS)]))
    Rnat = float(np.mean([float(rd["per_target"][roi][s][f]["R_ORACLE"]) for f in range(N_FOLDS)]))
    acc = {c: {"ctrl": [], "hyb": [], "evaluable": True, "rank_fail": 0} for c in ALL_CELLS}
    # dt-free similarity accumulators: prior->oracle, hybrid->oracle, hybrid->prior (within & outside)
    sim = {c: {"po_in": [], "ho_in": [], "hp_in": [], "po_out": [], "ho_out": [], "hp_out": []} for c in ALL_CELLS}
    Ts_needed = sorted({T for (_, T) in ALL_CELLS})
    for f in range(N_FOLDS):
        cell = cells[f]
        W = np.asarray(cell["W_target"], np.float64); K = int(cell["K"]); V = W.shape[0]
        dt = [np.asarray(x, np.float64) for x in list(cell["deltas_test"])]
        dt_n2 = [float(d @ d) for d in dt]; g = [W.T @ d for d in dt]
        simple = list(cell["simple_ids"]); nat = list(cell["nat_ids"])
        Dtr = np.asarray(trialnat[f]["trial_native"], np.float64)             # (10 x 8 x V)
        pr = priors[f]
        B_IN_PRED = np.asarray(pr["B_IN_PRED"], np.float64)                   # (V x r_best)
        B_OUT_PRED = np.asarray(pr["B_OUT_PRED"], np.float64)                 # (V x 2)
        U_PRIOR = W.T @ B_IN_PRED                                             # (K x r_best), orthonormal
        BP = B_OUT_PRED                                                       # (V x 2)
        # target OWN full-resource oracle geometry (10 training-identity residuals; dt-FREE)
        Dn = np.asarray(extnat[f]["delta_native"], np.float64)               # (10 x V)
        zc_or = np.stack([W.T @ Dn[i] for i in range(10)])                    # (10 x K)
        Uor = O9.within_basis_direct(zc_or, r_best)                          # (K x r_best)
        dout_or = np.stack([Dn[i] - W @ zc_or[i] for i in range(10)])
        Bor = O9.outside_basis_direct(dout_or.T, D_OUT)                      # (V x 2)
        # dt-free prior->oracle sims (constant across schedules within this fold)
        po_in = float(np.sum((U_PRIOR.T @ Uor) ** 2)) / r_best if Uor is not None else float("nan")
        po_out = float(np.sum((BP.T @ Bor) ** 2)) / D_OUT if Bor is not None else float("nan")
        # cross terms for hybrid sims (dt-free)
        UPUor = None if Uor is None else zc_or  # placeholder, computed per Ci below
        BPBor = None if Bor is None else Dn      # placeholder
        for T in Ts_needed:
            for S in itertools.combinations(range(8), T):
                Sl = list(S)
                di = np.stack([Dtr[i][Sl].mean(0) for i in range(10)])        # (10 x V)
                zc = np.stack([W.T @ di[i] for i in range(10)])               # (10 x K)
                dout = np.stack([di[i] - W @ zc[i] for i in range(10)])       # (10 x V)
                leak["max"] = max(leak["max"], float(np.max([np.abs(W.T @ dout[i]).max() for i in range(10)])))
                Zg = zc @ zc.T; Og = dout @ dout.T                            # (10 x 10) grams
                ZCG = np.stack([zc @ g[ti] for ti in range(2)])              # (2 x 10)
                OH = np.stack([dout @ dt[ti] for ti in range(2)])           # (2 x 10)
                ZU = zc @ U_PRIOR                                             # (10 x r_best)
                DP = dout @ BP                                                # (10 x 2)
                UG = np.stack([U_PRIOR.T @ g[ti] for ti in range(2)])        # (2 x r_best)
                BPd = np.stack([BP.T @ dt[ti] for ti in range(2)])          # (2 x 2)
                diagZ = np.diag(Zg).copy(); diagO = np.diag(Og).copy()
                # dt-free cross for hybrid->oracle sims
                ZUor = zc @ Uor if Uor is not None else None                 # (10 x r_best)  <z_i, Uor_c>
                DOor = dout @ Bor if Bor is not None else None               # (10 x 2)
                UPUor = U_PRIOR.T @ Uor if Uor is not None else None         # (r x r)
                BPBor = BP.T @ Bor if Bor is not None else None              # (2 x 2)
                for (M, T2) in ALL_CELLS:
                    if T2 != T:
                        continue
                    q = min(r_best, M)
                    subs = O9._pairs_balanced(simple, nat, M)
                    if not subs:
                        acc[(M, T)]["evaluable"] = False; continue
                    cvals, hvals = [], []
                    po_ins, ho_ins, hp_ins, po_outs, ho_outs, hp_outs = [], [], [], [], [], []
                    ok = True
                    for C in subs:
                        Ci = list(C)
                        # ----- CONTROL (exact O2.9) -----
                        rc = []
                        good = True
                        for ti in range(2):
                            ew, rw = O9.topk_energy_gram(Zg[np.ix_(Ci, Ci)], ZCG[ti][Ci], q, max(K, M))
                            eo, ro = O9.topk_energy_gram(Og[np.ix_(Ci, Ci)], OH[ti][Ci], D_OUT, V)
                            if ew is None or eo is None:
                                good = False; break
                            rc.append((ew + eo) / dt_n2[ti])
                        if not good:
                            ok = False; break
                        cvals.append(float(np.mean(rc)))
                        # ----- HYBRID (one-pseudo-observation prior) -----
                        e_IN = float(np.mean(diagZ[Ci])); e_OUT = float(np.mean(diagO[Ci]))
                        if e_IN <= EPS or e_OUT <= EPS:
                            ok = False; break
                        n_in = len(Ci) + r_best
                        Gin = np.zeros((n_in, n_in)); Gin[:M, :M] = Zg[np.ix_(Ci, Ci)]
                        Gin[:M, M:] = ZU[Ci]; Gin[M:, :M] = ZU[Ci].T; Gin[M:, M:] = np.eye(r_best)
                        lam_in = np.concatenate([np.ones(M), np.full(r_best, e_IN / r_best)])
                        nu_i, W_i, ls_i = _hyb_topk(Gin, lam_in, r_best)
                        n_out = len(Ci) + D_OUT
                        Gout = np.zeros((n_out, n_out)); Gout[:M, :M] = Og[np.ix_(Ci, Ci)]
                        Gout[:M, M:] = DP[Ci]; Gout[M:, :M] = DP[Ci].T; Gout[M:, M:] = np.eye(D_OUT)
                        lam_out = np.concatenate([np.ones(M), np.full(D_OUT, e_OUT / D_OUT)])
                        nu_o, W_o, ls_o = _hyb_topk(Gout, lam_out, D_OUT)
                        if nu_i is None or nu_o is None:
                            ok = False; break
                        rh = []
                        for ti in range(2):
                            ewh = _hyb_energy(nu_i, W_i, ls_i, np.concatenate([ZCG[ti][Ci], UG[ti]]), r_best)
                            eoh = _hyb_energy(nu_o, W_o, ls_o, np.concatenate([OH[ti][Ci], BPd[ti]]), D_OUT)
                            rh.append((ewh + eoh) / dt_n2[ti])
                        hvals.append(float(np.mean(rh)))
                        # ----- dt-free geometry sims -----
                        if Uor is not None:
                            Muor_in = np.vstack([ZUor[Ci], UPUor])            # (M+r) x r
                            ho_ins.append(_hyb_sim(nu_i, W_i, ls_i, Muor_in, r_best, r_best) / r_best)
                            Mprior_in = np.vstack([ZU[Ci], np.eye(r_best)])   # Zin^T U_PRIOR
                            hp_ins.append(_hyb_sim(nu_i, W_i, ls_i, Mprior_in, r_best, r_best) / r_best)
                            po_ins.append(po_in)
                        if Bor is not None:
                            Muor_out = np.vstack([DOor[Ci], BPBor])           # (M+2) x 2
                            ho_outs.append(_hyb_sim(nu_o, W_o, ls_o, Muor_out, D_OUT, D_OUT) / D_OUT)
                            Mprior_out = np.vstack([DP[Ci], np.eye(D_OUT)])   # Zout^T B_PRIOR
                            hp_outs.append(_hyb_sim(nu_o, W_o, ls_o, Mprior_out, D_OUT, D_OUT) / D_OUT)
                            po_outs.append(po_out)
                    if not ok:
                        acc[(M, T)]["evaluable"] = False; acc[(M, T)]["rank_fail"] += 1; continue
                    acc[(M, T)]["ctrl"].append(float(np.mean(cvals)))
                    if hybrid_heldout:
                        acc[(M, T)]["hyb"].append(float(np.mean(hvals)))
                    if ho_ins:
                        sim[(M, T)]["po_in"].append(float(np.mean(po_ins))); sim[(M, T)]["ho_in"].append(float(np.mean(ho_ins))); sim[(M, T)]["hp_in"].append(float(np.mean(hp_ins)))
                    if ho_outs:
                        sim[(M, T)]["po_out"].append(float(np.mean(po_outs))); sim[(M, T)]["ho_out"].append(float(np.mean(ho_outs))); sim[(M, T)]["hp_out"].append(float(np.mean(hp_outs)))
    out = {"R0": R0, "R_native": Rnat, "r_best": r_best, "cells": {}}
    for (M, T) in ALL_CELLS:
        a = acc[(M, T)]
        exp = N_FOLDS * T_SUB[T]
        evaluable = a["evaluable"] and len(a["ctrl"]) == exp
        d = {"evaluable": bool(evaluable), "N_TRIALS": M * T, "q": min(M, r_best), "rank_fail": a["rank_fail"]}
        if evaluable:
            d["R_CONTROL"] = float(np.mean(a["ctrl"]))
            if hybrid_heldout and len(a["hyb"]) == exp:
                d["R_HYBRID"] = float(np.mean(a["hyb"]))
            sm = sim[(M, T)]
            for kk in sm:
                d["SIM_" + kk] = float(np.mean(sm[kk])) if sm[kk] else float("nan")
        out["cells"][(M, T)] = d
    return out


# ---------------- inference ----------------
def _median(x):
    return float(np.median(x)) if len(x) else float("nan")


def classify(per, R111, rois):
    from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
    # participant DELTA_PRIOR, recovery, fcf per ROI x primary cell
    part = {roi: {c: {} for c in ALL_CELLS} for roi in rois}
    for roi in rois:
        for c in ALL_CELLS:
            for s in ALL:
                cc = per[s][roi]["cells"][c]; R0 = per[s][roi]["R0"]; Rnat = per[s][roi]["R_native"]
                if not cc.get("evaluable") or "R_HYBRID" not in cc:
                    part[roi][c][s] = None; continue
                dp = cc["R_HYBRID"] - cc["R_CONTROL"]
                tot = (cc["R_HYBRID"] - R0) / max(Rnat - R0, 1e-12)
                fcf = (cc["R_HYBRID"] - R0) / max(R111[(s, roi)] - R0, 1e-12)
                part[roi][c][s] = {"DELTA": dp, "TOTAL": float(np.clip(tot, 0, 1)), "TOTAL_u": tot,
                                   "FCF": float(np.clip(fcf, 0, 1)), "FCF_u": fcf, "R_CONTROL": cc["R_CONTROL"], "R_HYBRID": cc["R_HYBRID"]}
    # primary family = 4 sub-eight cells x rois (exactly 8); Holm across all
    pvals = {}
    for roi in rois:
        for c in PRIMARY_CELLS:
            vals = [part[roi][c][s] for s in ALL]
            if any(v is None for v in vals):
                pvals[(roi, c)] = None
            else:
                pvals[(roi, c)] = F.signflip_p_onesided([v["DELTA"] for v in vals])
    technical = any(v is None for v in pvals.values())
    holm = {} if technical else F.holm({f"{r}|{M}x{T}": pvals[(r, (M, T))] for (r, (M, T)) in pvals}, alpha=0.05)
    infer = {}; roi_status = {}
    for roi in rois:
        prior_enabled = []; gain_supported = []; hyb_sufficient = []
        for c in PRIMARY_CELLS:
            vals = [part[roi][c][s] for s in ALL]
            if any(v is None for v in vals):
                infer[f"{roi}|{c[0]}x{c[1]}"] = {"evaluable": False}; continue
            D = [v["DELTA"] for v in vals]; TOT = [v["TOTAL"] for v in vals]; FCF = [v["FCF"] for v in vals]
            rej = bool(holm.get(f"{roi}|{c[0]}x{c[1]}", False))
            gs = bool(_median(D) > 0 and sum(d > 0 for d in D) >= 6 and rej)
            hs = bool(_median(TOT) >= 0.5 and sum(t >= 0.5 for t in TOT) >= 6 and _median(FCF) >= 0.5 and sum(x >= 0.5 for x in FCF) >= 6)
            pe = bool(gs and hs)
            if gs:
                gain_supported.append(c)
            if hs:
                hyb_sufficient.append(c)
            if pe:
                prior_enabled.append(c)
            infer[f"{roi}|{c[0]}x{c[1]}"] = {"evaluable": True, "median_DELTA": _median(D), "n_DELTA_pos": int(sum(d > 0 for d in D)),
                                             "signflip_p": float(pvals[(roi, c)]), "holm_reject": rej, "gain_supported": gs,
                                             "median_TOTAL": _median(TOT), "n_TOTAL_ge": int(sum(t >= 0.5 for t in TOT)),
                                             "median_FCF": _median(FCF), "n_FCF_ge": int(sum(x >= 0.5 for x in FCF)),
                                             "hybrid_sufficient": hs, "prior_enabled": pe}
        if technical:
            roi_status[roi] = "PERCEPTION_PRIOR_CALIBRATION_INCONCLUSIVE"
        elif prior_enabled:
            roi_status[roi] = "PERCEPTION_PRIOR_REDUCES_IMAGERY_CALIBRATION"
        elif gain_supported:
            roi_status[roi] = "PERCEPTION_PRIOR_IMPROVES_BUT_NOT_TO_OPERATIONAL_THRESHOLD"
        elif hyb_sufficient:
            roi_status[roi] = "PERCEPTION_PRIOR_OPERATIONALLY_SUFFICIENT_WITHOUT_GAIN_EVIDENCE"
        else:
            roi_status[roi] = "PERCEPTION_PRIOR_NO_SUPPORTED_BENEFIT"
        infer[f"{roi}|__cells__"] = {"prior_enabled": [list(c) for c in prior_enabled],
                                     "gain_supported": [list(c) for c in gain_supported], "hybrid_sufficient": [list(c) for c in hyb_sufficient]}
    # common prior-enabled
    def _pe(roi, c):
        return infer.get(f"{roi}|{c[0]}x{c[1]}", {}).get("prior_enabled", False)
    def _hs(roi, c):
        return infer.get(f"{roi}|{c[0]}x{c[1]}", {}).get("hybrid_sufficient", False)
    common_pe = [c for c in PRIMARY_CELLS if all(_pe(roi, c) for roi in rois)]
    common_hs = [c for c in PRIMARY_CELLS if all(_hs(roi, c) for roi in rois)]
    n_star = min((M * T for (M, T) in common_pe), default=None)
    minimal_set = [c for c in common_pe if c[0] * c[1] == n_star] if n_star is not None else []
    if technical:
        prog = "O2_12_PRIOR_CORRECTION_INCONCLUSIVE"
    elif common_pe:
        prog = "PERCEPTION_PRIOR_REDUCES_COMPOSITE_CALIBRATION_BELOW_EIGHT_TRIALS"
    elif common_hs:
        prog = "SUBEIGHT_HYBRID_RECOVERY_WITHOUT_PRIOR_UTILITY_ESTABLISHED"
    elif any(infer.get(f"{roi}|{c[0]}x{c[1]}", {}).get("gain_supported", False) for roi in rois for c in PRIMARY_CELLS):
        prog = "PERCEPTION_PRIOR_BENEFICIAL_BUT_EIGHT_TRIAL_MINIMUM_NOT_REDUCED"
    else:
        prog = "PERCEPTION_PRIOR_DOES_NOT_REDUCE_TARGET_IMAGERY_BURDEN"
    return part, infer, roi_status, common_pe, common_hs, n_star, minimal_set, prog, technical


# ---------------- loaders ----------------
def _load(a, rois):
    state = {s: {roi: [dict(np.load(Path(a.state) / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    trial = {s: {roi: [dict(np.load(Path(a.trial) / f"trialnat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    ext = {s: {roi: [dict(np.load(Path(a.ext) / f"deltanat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    prior = {s: {roi: [dict(np.load(Path(a.prior_dir) / f"pred_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    return state, trial, ext, prior


def _verify_prior_hashes(a, rois):
    man = json.loads(Path(a.prior_manifest).read_text()); by = {f["name"]: f for f in man["files"]}
    ok = True; checked = 0
    for roi in rois:
        for s in ALL:
            for f in range(N_FOLDS):
                nm = f"pred_{s}_{roi}_fold{f}.npz"; checked += 1
                p = Path(a.prior_dir) / nm
                if not p.exists() or by.get(nm, {}).get("sha256") != _sha(p):
                    ok = False
    return {"checked": checked, "ok": ok, "manifest": Path(a.prior_manifest).name}


# ---------------- Stage A ----------------
def stage_predict(a):
    G = _geom(a.repo)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    rd = json.loads(Path(a.rd_results).read_text())
    o9ref = {}
    for r in csv.DictReader(open(a.o2_9_ref)):
        o9ref[(r["subject"], r["roi"], int(r["M"]), int(r["T"]))] = float(r["R_COMP"])
    pv = _verify_prior_hashes(a, rois)
    (out / "o2_11_prior_verification.json").write_text(json.dumps(pv, indent=2))
    if not pv["ok"]:
        print("O2_12_PRIOR_CORRECTION_INCONCLUSIVE (prior manifest hash mismatch)"); return 1
    state, trial, ext, prior = _load(a, rois)
    leak = {"max": 0.0}; per = {s: {} for s in ALL}
    replay_err = 0.0
    for roi in rois:
        for s in ALL:
            r = core(s, roi, state[s][roi], trial[s][roi], ext[s][roi], prior[s][roi], rd, leak, hybrid_heldout=False)
            per[s][roi] = r
            for (M, T) in ALL_CELLS:
                cc = r["cells"][(M, T)]
                if cc.get("evaluable"):
                    replay_err = max(replay_err, abs(cc["R_CONTROL"] - o9ref[(s, roi, M, T)]))
    replay_ok = replay_err <= 1e-10
    (out / "o2_9_control_replay.json").write_text(json.dumps(
        {"status": "O2_12_CONTROL_REPLAYS_O2_9" if replay_ok else "O2_12_CONTROL_REPLAY_FAILURE",
         "max_abs_error": replay_err, "tol": 1e-10, "cells": [f"{M}x{T}" for (M, T) in ALL_CELLS]}, indent=2))
    if not replay_ok:
        print("O2_12 control replay FAILED err=%.2e" % replay_err); return 1
    # dt-free geometry diagnostics
    rows = []
    for roi in rois:
        for s in ALL:
            for (M, T) in ALL_CELLS:
                cc = per[s][roi]["cells"][(M, T)]
                if cc.get("evaluable"):
                    rows.append([s, roi, M, T, cc.get("SIM_po_in"), cc.get("SIM_ho_in"), cc.get("SIM_hp_in"),
                                 cc.get("SIM_po_out"), cc.get("SIM_ho_out"), cc.get("SIM_hp_out")])
    with open(out / "prior_geometry_correction.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subject", "roi", "M", "T", "SIM_prior_oracle_IN", "SIM_hybrid_oracle_IN", "PRIOR_ALIGNMENT_IN",
                                        "SIM_prior_oracle_OUT", "SIM_hybrid_oracle_OUT", "PRIOR_ALIGNMENT_OUT"]); w.writerows(rows)
    man = {"gate": "O2.12", "stage": "predict", "prior_verification": pv, "control_replay_err": replay_err,
           "outside_leak_max": leak["max"], "files": []}
    for nm in ["o2_11_prior_verification.json", "o2_9_control_replay.json", "prior_geometry_correction.csv"]:
        man["files"].append({"name": nm, "sha256": _sha(out / nm)})
    (out / "stage_a_hybrid_fit_manifest.json").write_text(json.dumps(man, indent=2, default=str))
    print("O2_12_STAGE_A_SEALED control_replay_err=%.2e leak=%.2e" % (replay_err, leak["max"]))
    return 0


# ---------------- Stage B ----------------
def stage_evaluate(a):
    G = _geom(a.repo)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    rd = json.loads(Path(a.rd_results).read_text())
    R111 = {(r["subject"], r["roi"]): float(r["R111"]) for r in csv.DictReader(open(a.o2_8_ref))}
    o9ref = {}
    for r in csv.DictReader(open(a.o2_9_ref)):
        o9ref[(r["subject"], r["roi"], int(r["M"]), int(r["T"]))] = float(r["R_COMP"])
    # provenance: Stage-A manifest + prior hashes
    man = json.loads(Path(a.stage_a_manifest).read_text())
    pv = _verify_prior_hashes(a, rois)
    if not pv["ok"]:
        print("O2_12_PRIOR_CORRECTION_INCONCLUSIVE (prior hash mismatch in Stage B)"); return 1
    state, trial, ext, prior = _load(a, rois)
    leak = {"max": 0.0}; per = {s: {} for s in ALL}; replay_err = 0.0
    for roi in rois:
        for s in ALL:
            r = core(s, roi, state[s][roi], trial[s][roi], ext[s][roi], prior[s][roi], rd, leak, hybrid_heldout=True)
            per[s][roi] = r
            for (M, T) in ALL_CELLS:
                cc = r["cells"][(M, T)]
                if cc.get("evaluable"):
                    replay_err = max(replay_err, abs(cc["R_CONTROL"] - o9ref[(s, roi, M, T)]))
    if replay_err > 1e-10:
        (out / "o2_9_control_replay.json").write_text(json.dumps({"status": "O2_12_CONTROL_REPLAY_FAILURE", "max_abs_error": replay_err}, indent=2))
        print("O2_12 control replay FAILED in Stage B err=%.2e" % replay_err); return 1
    part, infer, roi_status, common_pe, common_hs, n_star, minimal_set, prog, technical = classify(per, R111, rois)
    _write(out, per, part, infer, roi_status, common_pe, common_hs, n_star, minimal_set, prog, technical, replay_err, leak, rois, o9ref, R111)
    print("O2_12_STATUS", prog, "N_TRIALS_PRIOR_STAR", n_star)
    for roi in rois:
        print("  [%s] %s" % (roi, roi_status[roi]))
    return 0


def _write(out, per, part, infer, roi_status, common_pe, common_hs, n_star, minimal_set, prog, technical, replay_err, leak, rois, o9ref, R111):
    def _w(name, header, rows):
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    ctrl_rows, hyb_rows, gain_rows, tot_rows, fcf_rows, harm_rows = [], [], [], [], [], []
    for roi in rois:
        for c in ALL_CELLS:
            for s in ALL:
                cc = per[s][roi]["cells"][c]
                if cc.get("evaluable"):
                    ctrl_rows.append([s, roi, c[0], c[1], c[0] * c[1], cc["q"], cc["R_CONTROL"], cc.get("R_HYBRID", "")])
                    if "R_HYBRID" in cc:
                        hyb_rows.append([s, roi, c[0], c[1], c[0] * c[1], cc["R_HYBRID"], cc["R_CONTROL"], cc["R_HYBRID"] - cc["R_CONTROL"]])
        for c in PRIMARY_CELLS:
            for s in ALL:
                v = part[roi][c][s]
                if v is not None:
                    gain_rows.append([s, roi, c[0], c[1], c[0] * c[1], v["R_CONTROL"], v["R_HYBRID"], v["DELTA"]])
                    tot_rows.append([s, roi, c[0], c[1], v["TOTAL"], v["TOTAL_u"]])
                    fcf_rows.append([s, roi, c[0], c[1], v["FCF"], v["FCF_u"]])
            vals = [part[roi][c][s] for s in ALL]
            if all(v is not None for v in vals):
                D = [v["DELTA"] for v in vals]; neg = [d for d in D if d < 0]
                harm_rows.append([roi, c[0], c[1], int(sum(d < 0 for d in D)), (float(np.median(neg)) if neg else "")])
    _w("control_resource_results.csv", ["subject", "roi", "M", "T", "N_TRIALS", "q", "R_CONTROL", "R_HYBRID"], ctrl_rows)
    _w("hybrid_resource_results.csv", ["subject", "roi", "M", "T", "N_TRIALS", "R_HYBRID", "R_CONTROL", "DELTA_PRIOR"], hyb_rows)
    _w("participant_prior_gain.csv", ["subject", "roi", "M", "T", "N_TRIALS", "R_CONTROL", "R_HYBRID", "DELTA_PRIOR"], gain_rows)
    _w("hybrid_total_recovery.csv", ["subject", "roi", "M", "T", "TOTAL_RECOVERY_HYB", "TOTAL_RECOVERY_HYB_unclipped"], tot_rows)
    _w("hybrid_full_capacity_fraction.csv", ["subject", "roi", "M", "T", "FCF_HYB", "FCF_HYB_unclipped"], fcf_rows)
    _w("prior_harm_diagnostic.csv", ["roi", "M", "T", "n_DELTA_neg", "median_negative_loss"], harm_rows)
    # roi frontier + common frontier
    rf = []
    for roi in rois:
        for c in PRIMARY_CELLS:
            ii = infer.get(f"{roi}|{c[0]}x{c[1]}", {})
            rf.append([roi, c[0], c[1], c[0] * c[1], ii.get("evaluable"), ii.get("median_DELTA", ""), ii.get("n_DELTA_pos", ""),
                       ii.get("signflip_p", ""), ii.get("holm_reject", ""), ii.get("gain_supported", ""),
                       ii.get("median_TOTAL", ""), ii.get("median_FCF", ""), ii.get("hybrid_sufficient", ""), ii.get("prior_enabled", "")])
    _w("roi_prior_frontier.csv", ["roi", "M", "T", "N_TRIALS", "evaluable", "median_DELTA", "n_DELTA_pos", "signflip_p", "holm_reject",
                                  "gain_supported", "median_TOTAL", "median_FCF", "hybrid_sufficient", "prior_enabled"], rf)
    _w("common_prior_frontier.csv", ["M", "T", "N_TRIALS", "kind"],
       [[c[0], c[1], c[0] * c[1], "prior_enabled"] for c in common_pe] + [[c[0], c[1], c[0] * c[1], "hybrid_sufficient"] for c in common_hs])
    # reference comparisons
    for (Mr, Tr), fn in ((tuple(REFERENCE_CELLS[0]), "reference_m4t2_comparison.csv"), (tuple(REFERENCE_CELLS[1]), "reference_m10t8_comparison.csv")):
        rr = []
        for roi in rois:
            for s in ALL:
                cc = per[s][roi]["cells"][(Mr, Tr)]
                if cc.get("evaluable") and "R_HYBRID" in cc:
                    rr.append([s, roi, Mr, Tr, cc["R_CONTROL"], cc["R_HYBRID"], cc["R_HYBRID"] - cc["R_CONTROL"],
                               o9ref[(s, roi, Mr, Tr)], cc.get("SIM_ho_in"), cc.get("SIM_ho_out")])
        _w(fn, ["subject", "roi", "M", "T", "R_CONTROL", "R_HYBRID", "DELTA_PRIOR", "O2_9_R_COMP", "SIM_hybrid_oracle_IN", "SIM_hybrid_oracle_OUT"], rr)
    (out / "primary_prior_gain_inference.json").write_text(json.dumps({"family_size": 8, "per_test": {k: v for k, v in infer.items() if "__cells__" not in k and "|" in k}, "technical": technical}, indent=2, default=str))
    (out / "minimal_prior_resource_set.json").write_text(json.dumps(
        {"N_TRIALS_PRIOR_STAR": n_star, "MINIMAL_PRIOR_RESOURCE_SET": [{"M": M, "T": T, "N_TRIALS": M * T} for (M, T) in minimal_set],
         "TRIAL_REDUCTION": (8 - n_star) if n_star is not None else None, "RELATIVE_REDUCTION": ((8 - n_star) / 8) if n_star is not None else None,
         "common_prior_enabled": [{"M": M, "T": T} for (M, T) in common_pe]}, indent=2, default=str))
    (out / "roi_status.json").write_text(json.dumps(roi_status, indent=2))
    (out / "scientific_status.json").write_text(json.dumps(
        {"status": prog, "roi_status": roi_status, "N_TRIALS_PRIOR_STAR": n_star,
         "MINIMAL_PRIOR_RESOURCE_SET": [{"M": M, "T": T} for (M, T) in minimal_set],
         "control_replay_err": replay_err, "outside_leak_max": leak["max"],
         "immutable": {"O2_11": "SUBJECT_SPECIFIC_GEOMETRY_NOT_PREDICTABLE_FROM_TESTED_PERCEPTION_PHENOTYPE",
                       "O2_10": "COMPOSITE_TARGET_STATE_GEOMETRY_SUBJECT_SPECIFIC_DOMINANT",
                       "O2_9": "MINIMAL_COMPOSITE_TARGET_STATE_CALIBRATION_ESTABLISHED", "O2_9_N_TRIALS_STAR": 8}, "O3": "O3_NOT_READY"}, indent=2, default=str))
    (out / "next_gate_decision.json").write_text(json.dumps({"program_status": prog, "N_TRIALS_PRIOR_STAR": n_star}, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--stage", required=True, choices=["predict", "evaluate"])
    ap.add_argument("--state", required=True); ap.add_argument("--trial", required=True); ap.add_argument("--ext", required=True)
    ap.add_argument("--rd-results", required=True); ap.add_argument("--prior-dir", required=True); ap.add_argument("--prior-manifest", required=True)
    ap.add_argument("--o2-9-ref", required=True); ap.add_argument("--o2-8-ref", default=""); ap.add_argument("--stage-a-manifest", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--rois", default="ventral,lateral")
    a = ap.parse_args()
    return stage_predict(a) if a.stage == "predict" else stage_evaluate(a)


if __name__ == "__main__":
    raise SystemExit(main())
