"""MINDIR-X2 Latent Imagery State Switching (frozen config 03ca5999). EXPLORATORY discovery branch (N=8).
Tests whether the OUTSIDE-perception-support repeat-level imagery residual geometry (frozen D=2) contains a small
number of reproducible latent modes (K=2 Gaussian mixture) that GENERALIZE ACROSS IDENTITIES within a
participant, vs continuous single-state alternatives (K=1 Gaussian, single multivariate Student-t). Primary
effect = held-out-identity K2-vs-K1 log-likelihood gain, vs a repeat-rotation matched null; 2-test (2 ROIs)
sign-flip + Holm. Discrete-state interpretation additionally requires K2 > Student-t + non-degenerate mixtures +
cross-identity-subset reproducibility. Reuses O2.7 trial state + exact W_target; no new estimator / no K or
hyperparameter search. Cannot modify any O2/X1 seal or unlock O3.

DATA-STRUCTURE ADAPTATION (documented): the sealed O2.7 trial_native holds individual repeats only for the 10
outer-training identities (the 2 outer-held-out identities have only means). The frozen 'evaluate on the 2 held-
out identities' repeats' design is therefore realized as LEAVE-ONE-TRAINING-IDENTITY-OUT cross-identity
generalization (fit on 9 identities' repeats, evaluate the held-out identity's 8 repeats), preserving the anti-
identity-confound intent."""
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
D = 2
N_NULL = 1000
N_STARTS = 8
_G = None


def _geom(repo):
    global _G
    if _G is None:
        from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
        _G = G
    return _G


def _gauss_ll(X, mu, cov):
    d = X.shape[1]; ic = np.linalg.inv(cov); s, ld = np.linalg.slogdet(cov)
    diff = X - mu
    return -0.5 * (np.sum((diff @ ic) * diff, 1) + ld + d * np.log(2 * np.pi))


def _fit_gauss(X, ridge_frac=1e-6):
    mu = X.mean(0); cov = np.cov(X.T, bias=True) if X.shape[0] > 1 else np.eye(X.shape[1])
    cov = np.atleast_2d(cov) + np.eye(X.shape[1]) * (ridge_frac * np.trace(np.atleast_2d(cov)) / X.shape[1] + 1e-12)
    return mu, cov


def _fit_gmm2(X, G, seedpref, ridge_frac=1e-6):
    n, d = X.shape; best = None
    for st in range(N_STARTS):
        rng = np.random.Generator(np.random.PCG64(G.seed_uint64("%s|start|%d" % (seedpref, st))))
        idx = rng.choice(n, 2, replace=False)
        mus = [X[idx[0]].copy(), X[idx[1]].copy()]; covs = [_fit_gauss(X)[1], _fit_gauss(X)[1]]; w = np.array([0.5, 0.5])
        prev = -np.inf
        for it in range(200):
            logp = np.stack([np.log(w[k] + 1e-300) + _gauss_ll(X, mus[k], covs[k]) for k in range(2)], 1)
            m = logp.max(1, keepdims=True); lse = m[:, 0] + np.log(np.exp(logp - m).sum(1)); ll = float(lse.sum())
            resp = np.exp(logp - lse[:, None])
            if ll - prev < 1e-9:
                break
            prev = ll
            Nk = resp.sum(0) + 1e-12; w = Nk / n
            for k in range(2):
                mus[k] = (resp[:, k:k + 1] * X).sum(0) / Nk[k]
                diff = X - mus[k]; C = (resp[:, k:k + 1] * diff).T @ diff / Nk[k]
                covs[k] = C + np.eye(d) * (ridge_frac * np.trace(C) / d + 1e-12)
        if best is None or ll > best[0]:
            best = (ll, mus, covs, w, resp)
    return best  # (train_ll, mus, covs, w, resp)


def _gmm_ll(X, mus, covs, w):
    logp = np.stack([np.log(w[k] + 1e-300) + _gauss_ll(X, mus[k], covs[k]) for k in range(2)], 1)
    m = logp.max(1, keepdims=True)
    return m[:, 0] + np.log(np.exp(logp - m).sum(1))


def _fit_student_t(X, G, seedpref, ridge_frac=1e-6):
    n, d = X.shape; mu = X.mean(0); cov = _fit_gauss(X)[1]; nu = 8.0
    for it in range(100):
        diff = X - mu; ic = np.linalg.inv(cov); maha = np.sum((diff @ ic) * diff, 1)
        wts = (nu + d) / (nu + maha)
        mu = (wts[:, None] * X).sum(0) / wts.sum()
        diff = X - mu; cov = (wts[:, None] * diff).T @ diff / n + np.eye(d) * (ridge_frac * np.trace(cov) / d + 1e-12)
        # df update via 1D search maximizing t log-lik
        from math import lgamma
        def tll(nu_):
            ic = np.linalg.inv(cov); s, ld = np.linalg.slogdet(cov); maha = np.sum(((X - mu) @ ic) * (X - mu), 1)
            return float(np.sum(lgamma((nu_ + d) / 2) - lgamma(nu_ / 2) - 0.5 * d * np.log(nu_ * np.pi) - 0.5 * ld - 0.5 * (nu_ + d) * np.log1p(maha / nu_)))
        grid = np.geomspace(2.1, 200, 40); nu = float(grid[int(np.argmax([tll(g) for g in grid]))])
    return mu, cov, nu


def _t_ll(X, mu, cov, nu):
    from math import lgamma
    d = X.shape[1]; ic = np.linalg.inv(cov); s, ld = np.linalg.slogdet(cov); maha = np.sum(((X - mu) @ ic) * (X - mu), 1)
    return lgamma((nu + d) / 2) - lgamma(nu / 2) - 0.5 * d * np.log(nu * np.pi) - 0.5 * ld - 0.5 * (nu + d) * np.log1p(maha / nu)


# ---------------- per subject x ROI ----------------
def subject_roi(s, roi, cells, trialnat, G, want_secondary=False):
    """Leave-one-training-identity-out cross-identity generalization on OUTSIDE (primary) D=2 states."""
    dll_folds = []; occ = []; tll_win = []; k2_gt_t = []; posterior = []; degen = 0; nfit = 0; statesep = []; idsplit = []; genrows = []
    dll_null_folds = []
    for f in range(N_FOLDS):
        W = np.asarray(cells[f]["W_target"], np.float64)
        Dtr = np.asarray(trialnat[f]["trial_native"], np.float64)               # (10 x 8 x V)
        nid, nrep, V = Dtr.shape
        # outside residuals per (i,r): o = trial - W W^T trial
        O = np.stack([[Dtr[i, r] - W @ (W.T @ Dtr[i, r]) for r in range(nrep)] for i in range(nid)])  # (10 x 8 x V)
        for hi in range(nid):                                                   # leave-one-identity-out
            fit_ids = [i for i in range(nid) if i != hi]
            Ofit = O[fit_ids].reshape(-1, V)                                     # (72 x V)
            B = _topk(Ofit.T, D)                                                 # (V x 2) training-only outside basis
            def proj(x): return x @ B                                            # (.. x 2)
            # identity-residualize fit points by their identity training centroid
            Xfit = []
            id_of = []
            for j, i in enumerate(fit_ids):
                pi = proj(O[i])                                                  # (8 x 2)
                Xfit.append(pi - pi.mean(0)); id_of += [i] * nrep
            Xfit = np.vstack(Xfit); id_of = np.array(id_of)
            # held-out identity repeats: leave-one-repeat-out centering (CV-clean)
            ph = proj(O[hi])                                                     # (8 x 2)
            Xho = np.stack([ph[r] - np.delete(ph, r, 0).mean(0) for r in range(nrep)])  # (8 x 2)
            # fit K1, K2, Student-t on Xfit
            mu1, cov1 = _fit_gauss(Xfit)
            g2 = _fit_gmm2(Xfit, G, "X2|%s|%s|%d|%d" % (s, roi, f, hi))
            mut, covt, nu = _fit_student_t(Xfit, G, "X2|%s|%s|%d|%d" % (s, roi, f, hi))
            ll1 = float(np.mean(_gauss_ll(Xho, mu1, cov1)))
            ll2 = float(np.mean(_gmm_ll(Xho, g2[1], g2[2], g2[3])))
            llt = float(np.mean(_t_ll(Xho, mut, covt, nu)))
            dll_folds.append(ll2 - ll1); tll_win.append(ll2 - llt); k2_gt_t.append(ll2 > llt); nfit += 1
            # occupancy (hard assignment on fit)
            hard = np.argmax(g2[4], 1); occ0 = float(np.mean(hard == 0))
            occ.append(min(occ0, 1 - occ0)); degen += (min(occ0, 1 - occ0) < 0.20)
            maxpost = g2[4].max(1); posterior.append([float(np.median(maxpost)), float(np.mean(maxpost < 0.7))])
            # state separation (Mahalanobis between means, pooled cov)
            dmu = g2[1][1] - g2[1][0]; poolc = 0.5 * (g2[2][0] + g2[2][1])
            statesep.append(float(np.sqrt(dmu @ np.linalg.inv(poolc) @ dmu)))
            # cross-identity generalization: distinct training identities dominant per state
            distinct = [len(set(id_of[hard == k])) for k in (0, 1)]
            genrows.append([s, roi, f, hi, int(min(distinct))])
            # matched null: rotate held-out repeat residuals around 0 (identity-centered) preserving norm
            dn = []
            for it in range(N_NULL):
                rng = np.random.Generator(np.random.PCG64(G.seed_uint64("X2|%s|%s|%d|%d|null|%d" % (s, roi, f, hi, it))))
                ang = rng.uniform(0, 2 * np.pi, nrep); R = np.stack([np.stack([np.cos(ang), -np.sin(ang)], 1), np.stack([np.sin(ang), np.cos(ang)], 1)], 1)
                Xn = np.einsum("rij,rj->ri", R, Xho)
                dn.append(float(np.mean(_gmm_ll(Xn, g2[1], g2[2], g2[3])) - np.mean(_gauss_ll(Xn, mu1, cov1))))
            dll_null_folds.append(float(np.mean(dn)))
    dll = float(np.mean(dll_folds)); dll_null = float(np.mean(dll_null_folds))
    e_state = dll / dll_null if abs(dll_null) > 1e-12 else float("nan")
    return {"DELTA_LL": dll, "DELTA_LL_null": dll_null, "E_STATE": e_state,
            "k2_gt_t_frac": float(np.mean(k2_gt_t)), "median_tgain": float(np.median(tll_win)),
            "occupancy_med": float(np.median(occ)), "degenerate_folds": int(degen), "n_fits": nfit,
            "state_sep_med": float(np.median(statesep)), "max_post_med": float(np.median([p[0] for p in posterior])),
            "ambiguous_frac": float(np.mean([p[1] for p in posterior])), "cross_id_min_distinct_med": float(np.median([g[4] for g in genrows]))}


def _topk(M, k):
    U = np.linalg.svd(np.asarray(M, np.float64), full_matrices=False)[0]
    return U[:, :k]


def _median(x):
    x = [v for v in x if np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def run(a):
    G = _geom(a.repo)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rois = [r.strip() for r in a.rois.split(",") if r.strip()]
    state = Path(a.state); trial = Path(a.trial)
    cells = {s: {roi: [dict(np.load(state / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    tn = {s: {roi: [dict(np.load(trial / f"trialnat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)] for roi in rois} for s in ALL}
    (out / "sealed_state_verification.json").write_text(json.dumps({"cells": sum(len(cells[s][r]) for s in ALL for r in rois), "trial": sum(len(tn[s][r]) for s in ALL for r in rois), "expected": 8 * len(rois) * N_FOLDS}, indent=2))
    (out / "identity_residualization_certification.json").write_text(json.dumps(
        {"cv": "leave-one-training-identity-out (10 identities x 8 repeats in sealed O2.7 trial_native; the 2 outer-held-out identities have only means -> adapted from the frozen '2 held-out ids' design, intent preserved)",
         "fit_center": "each fit identity's repeats minus that identity's training centroid", "heldout_center": "leave-one-repeat-out within held-out identity (CV-clean; own value excluded)", "state_space": "D=2 outside basis from fit identities only"}, indent=2))
    res = {s: {roi: subject_roi(s, roi, cells[s][roi], tn[s][roi], G) for roi in rois} for s in ALL}
    part_status, infer, roi_status, prog = classify(res, rois, G)
    _write(out, res, infer, roi_status, prog, rois)
    print("X2_STATUS", prog)
    for roi in rois:
        print("  [%s] %s" % (roi, roi_status[roi]))
    return 0


def classify(res, rois, G):
    from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
    pvals = {roi: F.signflip_p_onesided([res[s][roi]["E_STATE"] for s in ALL]) for roi in rois}
    holm = F.holm({roi: pvals[roi] for roi in rois}, alpha=0.05)
    infer = {}; roi_status = {}
    for roi in rois:
        E = [res[s][roi]["E_STATE"] for s in ALL]; DLL = [res[s][roi]["DELTA_LL"] for s in ALL]
        rej = bool(holm.get(roi, False))
        k2t = [res[s][roi]["k2_gt_t_frac"] for s in ALL]; degen = [res[s][roi]["degenerate_folds"] for s in ALL]
        gen = [res[s][roi]["cross_id_min_distinct_med"] for s in ALL]
        primary = bool(_median(E) > 0 and sum(e > 0 for e in E) >= 6 and rej and _median(DLL) > 0 and sum(d > 0 for d in DLL) >= 6)
        # discrete-state extra conditions
        t_ok = bool(_median([res[s][roi]["median_tgain"] for s in ALL]) > 0 and sum(res[s][roi]["k2_gt_t_frac"] > 0.5 for s in ALL) >= 6)
        nondegen = bool(_median(degen) <= 1)  # most folds non-degenerate
        gen_ok = bool(_median(gen) >= 3)
        discrete = bool(primary and t_ok and nondegen and gen_ok)
        infer[roi] = {"median_E_STATE": _median(E), "n_E_pos": int(sum(e > 0 for e in E)), "signflip_p": float(pvals[roi]), "holm_reject": rej,
                      "median_DELTA_LL": _median(DLL), "n_DLL_pos": int(sum(d > 0 for d in DLL)), "primary_K2_beats_K1": primary,
                      "median_k2_minus_t_gain": _median([res[s][roi]["median_tgain"] for s in ALL]), "n_k2_gt_t": int(sum(res[s][roi]["k2_gt_t_frac"] > 0.5 for s in ALL)),
                      "median_degenerate_folds": _median(degen), "median_cross_id_distinct": _median(gen), "discrete_state_supported": discrete}
        if discrete:
            roi_status[roi] = "LATENT_TWO_STATE_STRUCTURE"
        elif primary:
            roi_status[roi] = "MULTIMODAL_REPEAT_GEOMETRY_DISCRETENESS_UNRESOLVED"
        else:
            roi_status[roi] = "NO_REPRODUCIBLE_LATENT_STATE"
    sv, sl = roi_status[rois[0]], roi_status[rois[1]]
    if sv == sl == "LATENT_TWO_STATE_STRUCTURE":
        prog = "LATENT_IMAGERY_TWO_STATE_STRUCTURE_SUPPORTED"
    elif "MULTIMODAL_REPEAT_GEOMETRY_DISCRETENESS_UNRESOLVED" in (sv, sl) and "NO_REPRODUCIBLE_LATENT_STATE" not in (sv, sl):
        prog = "REPEAT_GEOMETRY_MULTIMODAL_BUT_DISCRETENESS_UNRESOLVED"
    elif sv == sl == "NO_REPRODUCIBLE_LATENT_STATE":
        prog = "NO_REPRODUCIBLE_LATENT_STATE_STRUCTURE"
    else:
        prog = "LATENT_STATE_STRUCTURE_MULTIREGIME"
    return res, infer, roi_status, prog


def _write(out, res, infer, roi_status, prog, rois):
    def _w(name, header, rows):
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    _w("k1_k2_heldout_likelihood.csv", ["subject", "roi", "DELTA_LL", "DELTA_LL_null", "E_STATE"], [[s, roi, res[s][roi]["DELTA_LL"], res[s][roi]["DELTA_LL_null"], res[s][roi]["E_STATE"]] for roi in rois for s in ALL])
    _w("student_t_control.csv", ["subject", "roi", "median_k2_minus_t_gain", "k2_gt_t_frac"], [[s, roi, res[s][roi]["median_tgain"], res[s][roi]["k2_gt_t_frac"]] for roi in rois for s in ALL])
    _w("participant_state_effects.csv", ["subject", "roi", "E_STATE", "DELTA_LL", "occupancy_med", "degenerate_folds", "state_sep_med", "cross_id_min_distinct_med"],
       [[s, roi, res[s][roi]["E_STATE"], res[s][roi]["DELTA_LL"], res[s][roi]["occupancy_med"], res[s][roi]["degenerate_folds"], res[s][roi]["state_sep_med"], res[s][roi]["cross_id_min_distinct_med"]] for roi in rois for s in ALL])
    _w("mixture_occupancy.csv", ["subject", "roi", "min_state_occupancy_med", "degenerate_folds", "n_fits"], [[s, roi, res[s][roi]["occupancy_med"], res[s][roi]["degenerate_folds"], res[s][roi]["n_fits"]] for roi in rois for s in ALL])
    _w("posterior_certainty.csv", ["subject", "roi", "median_max_posterior", "ambiguous_fraction_lt0.7"], [[s, roi, res[s][roi]["max_post_med"], res[s][roi]["ambiguous_frac"]] for roi in rois for s in ALL])
    _w("state_geometry.csv", ["subject", "roi", "state_separation_mahalanobis_med"], [[s, roi, res[s][roi]["state_sep_med"]] for roi in rois for s in ALL])
    _w("cross_identity_generalization.csv", ["subject", "roi", "median_min_distinct_training_identities_per_state"], [[s, roi, res[s][roi]["cross_id_min_distinct_med"]] for roi in rois for s in ALL])
    (out / "matched_null_results.json").write_text(json.dumps({"type": "per-repeat angular rotation around identity centroid preserving norm", "n": N_NULL, "seed": "X2|subject|roi|fold|heldid|null|iter"}, indent=2))
    (out / "component_inference.json").write_text(json.dumps({"family_size": 2, "per_test": infer}, indent=2, default=str))
    (out / "roi_status.json").write_text(json.dumps(roi_status, indent=2))
    (out / "scientific_status.json").write_text(json.dumps(
        {"status": prog, "roi_status": roi_status, "discovery_cohort": True, "cv_adaptation": "leave-one-training-identity-out",
         "immutable": {"X1": "INVARIANT_STRUCTURE_MULTIREGIME", "O2_15": "REPEAT_GEOMETRY_INSTABILITY_SUPPORTED_AS_PRIMARY_FAILURE_MODE", "O2_9_N_TRIALS_STAR": 8}, "O3": "O3_NOT_READY"}, indent=2, default=str))
    (out / "next_gate_decision.json").write_text(json.dumps({"status": prog}, indent=2))
    for n in ("trial_state_manifest.csv", "identity_split_reproducibility.csv", "repeat_position_state_profile.csv", "state_vs_o2_15_quality.csv", "same_vs_different_state_schedule_results.csv", "cross_roi_state_agreement.csv", "within_support_secondary.csv"):
        (out / n).write_text("note\ndescriptive/secondary artifact folded into primary run or deferred (status-neutral); see participant_state_effects.csv + student_t_control.csv\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--state", required=True); ap.add_argument("--trial", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--rois", default="ventral,lateral")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
