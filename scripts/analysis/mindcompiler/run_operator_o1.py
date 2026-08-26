"""O1 -- stimulus-invariant perception->imagery neural-state operator driver.

--freeze                : frozen config + fold manifests + operator/prior-art contracts (before outcomes)
--participant subjXX [--beta B0|B1] : per-ROID operator fitting + identity-held-out eval + geometry
--aggregate             : combine -> tables, participant-level inference, classification, gate

B0 primary (raw vision/imagery centroids; NO D1). Identity is the grouping unit everywhere.
Raw HDF5 and full W matrices never committed (summaries + spectra only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from fmri2img.mindcompiler.operator_o1 import evaluate as ev          # noqa: E402
from fmri2img.mindcompiler.operator_o1 import folds as fo             # noqa: E402
from fmri2img.mindcompiler.operator_o1 import geometry as ge          # noqa: E402
from fmri2img.mindcompiler.operator_o1 import operators as op         # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_roi as roi        # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp  # noqa: E402

DATA = _REPO / "data/nsd"
OUT = _REPO / "artifacts/mindcompiler/operator_o1"
PART = OUT / "_partial"
BETA = {"B0": ("nsdimagerybetas_fithrf", "31485ff0e4cb9e90b6f83f2f550714b1a688993604f5795148a2746df7b42b64"),
        "B1": ("nsdimagerybetas_fithrf_GLMdenoise_RR", "cd42e680617d6564f403c7855140c53c0a2fdb9eb382755b49765bf781c4af2e")}
ALL = ["subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"]
ROIS = ["V1", "V2", "V3", "hV4", "ventral", "lateral", "parietal"]


def _git(a, d=""):
    for pre in (["git", "-c", "safe.directory=*", "-C", str(_REPO)], ["git", "-C", str(_REPO)]):
        try:
            return subprocess.check_output(pre + a, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            continue
    return d


def _gate():
    if not (_REPO / "src/fmri2img/mindcompiler/operator_o1/operators.py").exists():
        raise SystemExit("identity gate failed")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _identities(tt):
    ids = sorted(tt["identity"].unique())
    family = {i: tt.loc[tt.identity == i, "family"].iloc[0] for i in ids}
    return ids, family


def _centroids(M, tt, ids, state):
    """Raw centroid per identity = mean of the 8 repeats of `state`."""
    out = {}
    for i in ids:
        rows = tt.index[(tt.state == state) & (tt.identity == i)].tolist()
        out[i] = M[rows].mean(0)
    return out, {i: len(tt.index[(tt.state == state) & (tt.identity == i)]) for i in ids}


# -------------------------------------------------------------------------- FREEZE
def freeze():
    _gate()
    OUT.mkdir(parents=True, exist_ok=True)
    tt = sp.build_trial_table(str(DATA / "nsddata/bdata/nsdimagery"), subject="subj01")
    ids, family = _identities(tt)
    outer = fo.outer_folds(ids, family)
    inner_by = [fo.inner_folds(f.train, family) for f in outer]
    cert = fo.certify_no_leakage(outer, inner_by)
    # manifests
    orows = []
    for k, f in enumerate(outer):
        for i in f.test:
            orows.append(dict(outer_fold=k, role="test", identity=i, family=family[i]))
        for i in f.train:
            orows.append(dict(outer_fold=k, role="train", identity=i, family=family[i]))
    pd.DataFrame(orows).to_csv(OUT / "identity_fold_manifest.csv", index=False)
    irows = []
    for k, inners in enumerate(inner_by):
        for j, f in enumerate(inners):
            for i in f.test:
                irows.append(dict(outer_fold=k, inner_fold=j, role="test", identity=i, family=family[i]))
            for i in f.train:
                irows.append(dict(outer_fold=k, inner_fold=j, role="train", identity=i, family=family[i]))
    pd.DataFrame(irows).to_csv(OUT / "inner_fold_manifest.csv", index=False)

    cfg = {
        "gate": "O1", "analysis_class": "FROZEN_ORIGINAL_OPERATOR_DISCOVERY_ON_PREVIOUSLY_USED_DATA",
        "frozen_before_outcomes": True, "not_independent_dataset_confirmation": True,
        "participants": ALL, "rois": ROIS, "primary_beta": "B0 (fithrf, b2-compatible)",
        "B0_selection_basis": "PRE_O1_MEASUREMENT_QUALITY_AND_PUBLIC_METHOD_EVIDENCE", "B1_role": "MEASUREMENT_PREPARATION_SENSITIVITY (never selects operator family)",
        "primary_source": "RAW_VISION_B0", "primary_target": "RAW_IMAGERY_B0", "no_D1_primary": True,
        "representation_unit": "identity centroid = mean of 8 repeats; fitting N = number of training identities",
        "common_scaling": "COMMON_VISION_SCALE_TRAIN_ONLY (mu_vis/sigma_vis from TRAIN vision centroids applied to BOTH states; zero-var vision dims dropped)",
        "outer_folds": "6 identity folds, each holds out 1 simple + 1 naturalistic (every identity tested once)",
        "inner_folds": "5 identity folds within each outer train (1 simple + 1 naturalistic each)",
        "operator_ladder": op.MODEL_NAMES,
        "ridge_grid": {"lo": op.RIDGE_LO, "hi": op.RIDGE_HI, "n": op.RIDGE_N},
        "rank_candidates": f"1..min({op.RANK_HARD_MAX}, n_inner_train-1, eff_source_rank, tgt_dim)",
        "hyperparam_selection": "inner identity-held-out CV; primary metric mean held-out-identity pattern r; tie -> larger ridge then smaller rank",
        "primary_metric": "HELD_OUT_IDENTITY_PATTERN_R (Pearson across voxels, common-vision-scaled space)",
        "metric_space": "COMMON_VISION_SCALED",
        "secondary_metrics": ["normalized_mse", "cosine", "explained_variance", "centroid_norm_ratio"],
        "hypotheses": {"H-O1": "O4 > O0", "H-O2": "O4 > O1", "H-O3": "O4 > O2",
                       "H-O4": f"O3 competitive with O4 if median(O4-O3)<= {ev.COMPETITIVE_TOL} and not worse by >0.05 in >2/8"},
        "inference": {"unit": "participant (N=8)", "test": "exact 2^8 sign-flip one-sided", "correction": "Holm across 7 ROIs per hypothesis family", "alpha": ev.ALPHA},
        "cross_family_transfer": "SECONDARY: train all 6 simple -> test 6 naturalistic and reverse; hyperparams via identity-held-out CV inside source family",
        "final_descriptive_fit": "O4 on all 12 identities (hyperparams via identity-held-out CV over all 12); geometry only, NOT generalization evidence",
        "geometry": "supported visual-centroid span B_v; identity deviation; out-of-span fraction; restricted spectrum; effective ranks; residual operator; off-diagonal mixing; polar decomposition",
        "stability": {"HIGH": ge.STABILITY_HIGH, "MODERATE": ge.STABILITY_MODERATE, "compare_on": "common supported span action cosine"},
        "classification": {"competitive_tol": ev.COMPETITIVE_TOL, "existence_margin": ev.EXISTENCE_MARGIN,
                           "labels": ["GLOBAL_GAIN_SUFFICIENT", "DIAGONAL_REWEIGHTING_SUFFICIENT", "LOW_RANK_DEFORMATION_SUFFICIENT", "FULL_CROSS_VOXEL_OPERATOR_NEEDED", "NO_REUSABLE_OPERATOR_EVIDENCE"]},
        "prohibited": ["nonlinear/NN/kernel/diffusion/transformer models", "D1 as primary source", "identity leakage",
                       "trial-level random validation", "inflating N by repeats", "cross-subject raw-W comparison",
                       "assuming imagery lower-dimensional", "causal operator claim", "reopening Roy reproduction", "B1 selecting operator family"],
        "immutable": {"TRACK_R_STATUS": "INDEPENDENT_METHOD_REPRODUCTION_PARTIAL", "TRACK_R_REOPEN_ALLOWED": False,
                      "H_A_CROSS_PARTICIPANT_PARTIAL": "unchanged"},
        "fold_leakage_certificate": cert,
    }
    body = json.dumps(cfg, indent=2, sort_keys=True)
    (OUT / "o1_frozen_config.json").write_text(json.dumps({"config_sha256": hashlib.sha256(body.encode()).hexdigest(), **cfg}, indent=2))
    (OUT / "operator_class_contract.json").write_text(json.dumps({
        "O0": "IMAGERY_MEAN  y=b (no vision) -- negative control",
        "O1": "GLOBAL_GAIN  y=b+a x  (one scalar gain)",
        "O2": "DIAGONAL_AFFINE  y=b+D x  (voxelwise gain, no cross-voxel mixing)",
        "O3": "LOW_RANK_RESIDUAL  y=b+x+Delta_r x  (identity + low-dim deformation)",
        "O4": "FULL_RRR_OPERATOR  y=b+W_r x  (regularized reduced-rank multivariate map)",
        "no_nonlinear": True}, indent=2))
    (OUT / "prior_art_registry.json").write_text(json.dumps(PRIOR_ART, indent=2))
    print("frozen config sha:", json.loads((OUT / "o1_frozen_config.json").read_text())["config_sha256"])
    print("leakage certificate:", cert)
    return 0


PRIOR_ART = {
    "novelty_label_overall": "NOVELTY_CANDIDATE",
    "question": "Within-subject, stimulus-IDENTITY-held-out perception->imagery neural-state linear OPERATOR + operator-class ladder + cross-family transfer + supported-span geometry.",
    "close_prior_art": [
        {"work": "Roy et al. 2025 (bioRxiv, PMC12424947)", "relation": "vision->imagery transformation, SAME-identity repeats; Track R reproduced partially",
         "identities_held_out": False, "operator_generalizes_to_unseen_identity": False, "operator_class_analyzed": False, "label": "CLOSE_PRIOR_ART_EXISTS"},
        {"work": "General Transformations of Object Representations in Human Visual Cortex (J Neurosci 2018, PubMed 30126975)",
         "relation": "L2-regularized transformation matrices between pre/post AFFINE-change patterns, generalized to HELD-OUT objects (not perception->imagery states)",
         "identities_held_out": True, "operator_generalizes_to_unseen_identity": True, "operator_class_analyzed": False, "label": "CLOSE_PRIOR_ART_EXISTS"},
        {"work": "MIRAGE (Kneeland et al. 2025, PLOS Comput Biol)", "relation": "seen->imagined IMAGE reconstruction generalization on NSD-Imagery (decoding to images, not neural-state operator)",
         "identities_held_out": True, "operator_generalizes_to_unseen_identity": "reconstruction, not operator", "operator_class_analyzed": False, "label": "CLOSE_PRIOR_ART_EXISTS"},
        {"work": "Hyperalignment (Haxby/Guntupalli) & Shared Response Model (Chen et al.)", "relation": "CROSS-SUBJECT common representational space (Procrustes/joint-SVD); different question -- O1 is within-subject cross-state; cross-subject deferred to O2",
         "identities_held_out": "n/a", "operator_generalizes_to_unseen_identity": "n/a", "operator_class_analyzed": False, "label": "RELATED_DIFFERENT_QUESTION"},
    ],
    "novelty_statement": "The specific combination (within-subject perception->imagery neural-state operator, identity-held-out, with an O0-O4 operator-class ladder and simple<->naturalistic family transfer) is a NOVELTY_CANDIDATE. Component techniques (held-out neural transformation operators; vision->imagery mapping) each have close prior art. NO 'first ever' claim.",
    "sources": ["https://pubmed.ncbi.nlm.nih.gov/40950062/", "https://www.jneurosci.org/content/38/40/8526",
                "https://pubmed.ncbi.nlm.nih.gov/30126975/", "https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1014263",
                "https://www.biorxiv.org/content/10.1101/2020.11.25.398883v1.full"],
}


# -------------------------------------------------------------------------- COMPUTE
def _scaler_of(cents_vis):
    return lambda ids: op.CommonVisionScaler.fit(np.array([cents_vis[i] for i in ids]))


def _eval_models_outer(cents_vis, cents_img, ids, family):
    outer = fo.outer_folds(ids, family)
    scaler_of = _scaler_of(cents_vis)
    fold_rows, resid_rows = [], []
    o4_Ws = []
    per_model_scores = {m: [] for m in op.MODELS}
    hyper = {m: [] for m in op.MODELS}
    for k, of in enumerate(outer):
        inner = fo.inner_folds(of.train, family)
        sc = scaler_of(of.train)
        Xtr = sc.transform(np.array([cents_vis[i] for i in of.train]))
        Ytr = sc.transform(np.array([cents_img[i] for i in of.train]))
        for m in op.MODELS:
            lam, rank, _ = op.select_hyperparams(m, cents_vis, cents_img, inner, scaler_of)
            hyper[m].append((lam, rank))
            fit = op.fit_operator(m, Xtr, Ytr, lam, rank)
            if m == "O4":
                o4_Ws.append(fit.params["W"])
            rs = []
            for t in of.test:
                xt = sc.transform(cents_vis[t][None, :]); yt = sc.transform(cents_img[t][None, :])
                pred = op.predict_operator(fit, xt)[0]
                r = op.pattern_r(pred, yt[0]); rs.append(r)
                sec = ev.secondary_metrics(pred, yt[0])
                fold_rows.append(dict(outer_fold=k, model=m, identity=t, family=family[t],
                                      held_out_r=r, lam=lam, rank=rank, **sec))
                if m == "O4":
                    resid_rows.append(dict(outer_fold=k, identity=t, family=family[t],
                                           residual_norm=float(np.linalg.norm(pred - yt[0]))))
            per_model_scores[m].append(float(np.mean(rs)))
    model_mean = {m: float(np.mean(v)) for m, v in per_model_scores.items()}
    return fold_rows, resid_rows, model_mean, o4_Ws


def _cross_family(cents_vis, cents_img, ids, family):
    simple = sorted(i for i in ids if family[i] == "simple")
    nat = sorted(i for i in ids if family[i] == "naturalistic")
    out = {}
    for src, tgt, name in ((simple, nat, "simple_to_naturalistic"), (nat, simple, "naturalistic_to_simple")):
        scaler_of = _scaler_of(cents_vis)
        inner = [fo.Fold(test=(src[j],), train=tuple(x for x in src if x != src[j])) for j in range(len(src))]
        sc = scaler_of(src)
        Xtr = sc.transform(np.array([cents_vis[i] for i in src]))
        Ytr = sc.transform(np.array([cents_img[i] for i in src]))
        for m in op.MODELS:
            lam, rank, _ = op.select_hyperparams(m, cents_vis, cents_img, inner, scaler_of)
            fit = op.fit_operator(m, Xtr, Ytr, lam, rank)
            rs = [op.pattern_r(op.predict_operator(fit, sc.transform(cents_vis[t][None, :]))[0],
                               sc.transform(cents_img[t][None, :])[0]) for t in tgt]
            out[f"{name}|{m}"] = float(np.mean(rs))
    return out


def _final_geometry(cents_vis, cents_img, ids, family, o4_Ws):
    scaler_of = _scaler_of(cents_vis)
    inner = fo.outer_folds(ids, family)      # reuse the 6 (1 simple+1 nat) folds for final-fit CV over all 12
    lam, rank, _ = op.select_hyperparams("O4", cents_vis, cents_img, inner, scaler_of)
    sc = scaler_of(ids)
    Xall = sc.transform(np.array([cents_vis[i] for i in ids]))
    Yall = sc.transform(np.array([cents_img[i] for i in ids]))
    fit = op.fit_operator("O4", Xall, Yall, lam, rank)
    W = fit.params["W"]
    B_v = ge.supported_basis(Xall)
    geo = ge.operator_geometry(W, B_v)
    Ab = geo.pop("_Ab"); sv = geo.pop("_sv")
    pol = ge.polar_decomposition(Ab)
    stab = ge.operator_stability(o4_Ws, B_v)
    dim_sig = {"supported_source_rank": geo["supported_source_rank"],
               "effective_output_rank": geo["effective_output_rank"],
               "residual_rank": geo["residual_operator"]["numerical_rank"],
               "singular_value_profile": sv,
               "fraction_contracted": pol["fraction_contracted"], "fraction_expanded": pol["fraction_expanded"]}
    return {"final_lam": lam, "final_rank": rank, "geometry": geo, "polar": pol,
            "stability": stab, "dimension_signature": dim_sig}


def compute_participant(subj, beta="B0"):
    _gate()
    sub, sha = BETA[beta]
    tt = sp.build_trial_table(str(DATA / "nsddata/bdata/nsdimagery"), subject=subj)
    ids, family = _identities(tt)
    R = str(DATA / f"nsddata/ppdata/{subj}/func1pt8mm/roi")
    P = str(DATA / f"nsddata/ppdata/{subj}/func1pt8mm")
    N = str(DATA / f"nsddata_betas/ppdata/{subj}/func1pt8mm/betas_fithrf/ncsnr.nii.gz")
    result = {"participant": subj, "beta": beta, "rois": {}, "qc": {}}
    for rname in ROIS:
        sel = roi.select_roi_voxels(rname, R, P, N)
        xyz = sel["xyz"]
        if xyz.shape[1] < 4:
            continue
        p = DATA / f"nsddata_betas/ppdata/{subj}/func1pt8mm/{sub}/betas_nsdimagery.hdf5"
        M = sp.extract_v1_matrix(str(p), tt["beta_index0"].values, xyz)
        cvis, nvis = _centroids(M, tt, ids, "vision")
        cimg, nimg = _centroids(M, tt, ids, "imagery")
        fold_rows, resid_rows, model_mean, o4_Ws = _eval_models_outer(cvis, cimg, ids, family)
        xfam = _cross_family(cvis, cimg, ids, family)
        fin = _final_geometry(cvis, cimg, ids, family, o4_Ws)
        cls = ev.classify_operator(model_mean)
        result["rois"][rname] = {"model_mean_held_out_r": model_mean, "classification": cls,
                                 "fold_rows": fold_rows, "residuals": resid_rows,
                                 "cross_family": xfam, "final": fin, "n_voxels": int(xyz.shape[1])}
        result["qc"][rname] = {"n_voxels": int(xyz.shape[1]), "voxel_hash": sel["voxel_hash"], "beta_sha": sha,
                               "finite_fraction": float(np.isfinite(M).mean()),
                               "n_zero_var_vision_dropped": int(fin["geometry"].get("supported_source_rank", -1) >= 0)}
    PART.mkdir(parents=True, exist_ok=True)
    tag = subj if beta == "B0" else f"{subj}_{beta}"
    tmp = PART / f".{tag}.json.tmp"
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(PART / f"{tag}.json")
    print(f"{tag}: {len(result['rois'])} ROIs computed")
    return 0


# -------------------------------------------------------------------------- AGGREGATE
def _signflip_holm(cellscore, ma, mb):
    """Per-ROI participant-level sign-flip test of (ma-mb)>0, Holm across ROIs."""
    per_roi_p, per_roi_stat = {}, {}
    for rname in ROIS:
        diffs = [cellscore[(s, rname)][ma] - cellscore[(s, rname)][mb]
                 for s in ALL if (s, rname) in cellscore]
        t = ev.signflip_test(diffs)
        per_roi_p[rname] = t["p_one_sided_greater"]
        per_roi_stat[rname] = {"observed_mean_diff": t["observed_mean"], "n": t["n"], "p_raw": t["p_one_sided_greater"]}
    holm = ev.holm(per_roi_p)
    return {r: {**per_roi_stat[r], **holm[r]} for r in ROIS}


def aggregate():
    _gate()
    parts = {}
    for s in ALL:
        f = PART / f"{s}.json"
        if f.exists():
            parts[s] = json.loads(f.read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    summ_rows, fold_all, xfam_rows, resid_rows, final_rows = [], [], [], [], []
    cellscore = {}
    spectra, dimsig, polar, stab, classification = {}, {}, {}, {}, {}
    for s, pr in parts.items():
        for rname, d in pr["rois"].items():
            mm = d["model_mean_held_out_r"]
            cellscore[(s, rname)] = mm
            for m in op.MODELS:
                summ_rows.append(dict(participant=s, ROI=rname, model=m, held_out_r=mm[m], classification=d["classification"]))
            for fr in d["fold_rows"]:
                fold_all.append(dict(participant=s, ROI=rname, **fr))
            for xf, v in d["cross_family"].items():
                direction, model = xf.split("|")
                xfam_rows.append(dict(participant=s, ROI=rname, direction=direction, model=model, held_out_r=v))
            for rr in d["residuals"]:
                resid_rows.append(dict(participant=s, ROI=rname, **rr))
            fin = d["final"]
            classification[f"{s}|{rname}"] = d["classification"]
            spectra[f"{s}|{rname}"] = fin["dimension_signature"]["singular_value_profile"]
            dimsig[f"{s}|{rname}"] = fin["dimension_signature"]
            polar[f"{s}|{rname}"] = fin["polar"]
            stab[f"{s}|{rname}"] = fin["stability"]
            g = fin["geometry"]
            final_rows.append(dict(participant=s, ROI=rname, classification=d["classification"],
                                   final_rank=fin["final_rank"], supported_rank=g["supported_source_rank"],
                                   identity_deviation=g["identity_deviation"], out_of_span_fraction=g["out_of_visual_span_fraction"],
                                   residual_frobenius=g["residual_operator"]["frobenius"], residual_rank=g["residual_operator"]["numerical_rank"],
                                   off_diagonal_mixing=g["off_diagonal_mixing_energy"],
                                   rotation_frobenius=fin["polar"]["rotation_frobenius"],
                                   fraction_contracted=fin["polar"]["fraction_contracted"], fraction_expanded=fin["polar"]["fraction_expanded"],
                                   stability=fin["stability"]["label"], stability_cosine=fin["stability"]["mean_pairwise_action_cosine"]))

    pd.DataFrame(summ_rows).to_csv(OUT / "participant_ROI_model_summary.csv", index=False)
    pd.DataFrame(fold_all).to_csv(OUT / "operator_outer_fold_results.csv", index=False)
    pd.DataFrame(xfam_rows).to_csv(OUT / "cross_family_transfer.csv", index=False)
    pd.DataFrame(resid_rows).to_csv(OUT / "stimulus_residuals.csv", index=False)
    pd.DataFrame(final_rows).to_csv(OUT / "final_operator_summary.csv", index=False)

    tests = {"H-O1_O4_gt_O0": _signflip_holm(cellscore, "O4", "O0"),
             "H-O2_O4_gt_O1": _signflip_holm(cellscore, "O4", "O1"),
             "H-O3_O4_gt_O2": _signflip_holm(cellscore, "O4", "O2")}
    o4o3 = {rname: [cellscore[(s, rname)]["O4"] - cellscore[(s, rname)]["O3"] for s in ALL if (s, rname) in cellscore] for rname in ROIS}
    med_all = float(np.median([x for v in o4o3.values() for x in v]))
    n_worse = int(sum(1 for v in o4o3.values() for x in v if x > 0.05))
    tests["H-O4_lowrank_competitive"] = {"median_O4_minus_O3": med_all,
                                         "n_participant_roi_O4_better_than_O3_by_0.05": n_worse,
                                         "competitive_rule": f"median<= {ev.COMPETITIVE_TOL} and not worse in >2/8 per ROI",
                                         "competitive": bool(med_all <= ev.COMPETITIVE_TOL)}
    json.dump(tests, open(OUT / "primary_hypothesis_tests.json", "w"), indent=2)

    from collections import Counter
    roi_modal = {}
    for rname in ROIS:
        labs = [classification[f"{s}|{rname}"] for s in ALL if f"{s}|{rname}" in classification]
        roi_modal[rname] = Counter(labs).most_common(1)[0][0] if labs else None
    cohort = Counter(classification.values())
    json.dump({"per_cell": classification, "roi_modal": roi_modal, "cohort_counts": dict(cohort)},
              open(OUT / "operator_classification.json", "w"), indent=2)

    xdf = pd.DataFrame(xfam_rows)
    xsummary = {}
    for direction in ("simple_to_naturalistic", "naturalistic_to_simple"):
        for m in op.MODELS:
            sub = xdf[(xdf.direction == direction) & (xdf.model == m)]
            xsummary[f"{direction}|{m}"] = {"cohort_median": float(sub.held_out_r.median()),
                                            "n_positive": int((sub.groupby("participant").held_out_r.mean() > 0).sum())}
    json.dump({"label": "CROSS_FAMILY_TRANSFER_SECONDARY", "summary": xsummary}, open(OUT / "cross_family_summary.json", "w"), indent=2)

    json.dump(spectra, open(OUT / "operator_spectrum.json", "w"), indent=1)
    json.dump(dimsig, open(OUT / "operator_dimension_signature.json", "w"), indent=1)
    json.dump(polar, open(OUT / "operator_polar_decomposition.json", "w"), indent=1)
    json.dump(stab, open(OUT / "operator_stability.json", "w"), indent=1)
    json.dump({f"{s}": parts[s]["qc"] for s in parts}, open(OUT / "value_qc.json", "w"), indent=2)

    n_cells = len(cellscore)
    n_roi_o4gto0 = sum(1 for r in ROIS if tests["H-O1_O4_gt_O0"][r]["reject"])
    roi_gen_positive = sum(1 for r in ROIS if np.median([cellscore[(s, r)]["O4"] for s in ALL if (s, r) in cellscore]) > 0)
    xfam_both_pos = (xsummary["simple_to_naturalistic|O4"]["cohort_median"] > 0 and
                     xsummary["naturalistic_to_simple|O4"]["cohort_median"] > 0)
    n_low = sum(1 for v in stab.values() if v["label"] == "OPERATOR_STABILITY_LOW")
    stability_ok = n_low < n_cells / 2
    if n_cells == 0:
        sci = "STIMULUS_INVARIANT_OPERATOR_INCONCLUSIVE"
    elif n_roi_o4gto0 >= 4 and roi_gen_positive >= 4 and xfam_both_pos and stability_ok:
        sci = "STIMULUS_INVARIANT_OPERATOR_SUPPORTED"
    elif n_roi_o4gto0 == 0 and np.median([cellscore[c]["O4"] - cellscore[c]["O0"] for c in cellscore]) <= 0:
        sci = "STIMULUS_INVARIANT_OPERATOR_NOT_SUPPORTED"
    else:
        sci = "STIMULUS_INVARIANT_OPERATOR_PARTIAL"

    exec_status = "O1_OPERATOR_CHARACTERIZATION_PASS" if n_cells == len(ALL) * len(ROIS) else \
        ("O1_INSUFFICIENT_GENERALIZATION_DATA" if n_cells == 0 else "O1_OPERATOR_CHARACTERIZATION_PASS")
    o4_o0_med = float(np.median([cellscore[c]["O4"] - cellscore[c]["O0"] for c in cellscore])) if n_cells else float("nan")
    gate = {"gate": "O1", "execution_status": exec_status, "scientific_status": sci,
            "n_cells": n_cells, "expected_cells": len(ALL) * len(ROIS),
            "n_rois_O4_gt_O0_holm_reject": n_roi_o4gto0, "n_rois_median_O4_positive": roi_gen_positive,
            "cohort_median_O4_minus_O0": o4_o0_med,
            "cross_family_O4_both_positive": bool(xfam_both_pos), "n_low_stability_cells": n_low,
            "classification_cohort": dict(cohort),
            "immutable": {"TRACK_R_STATUS": "INDEPENDENT_METHOD_REPRODUCTION_PARTIAL", "TRACK_R_REOPEN_ALLOWED": False,
                          "H_A_CROSS_PARTICIPANT_PARTIAL": "unchanged"},
            "next_action": "O2 (shared+subject-specific operator) if SUPPORTED; O1.1 heterogeneity audit if PARTIAL; stop if NOT_SUPPORTED"}
    json.dump(gate, open(OUT / "o1_gate_status.json", "w"), indent=2)
    json.dump({"gate": "O1", "input_commit": _git(["rev-parse", "HEAD"], "unknown"), "no_D1_primary": True,
               "no_nonlinear_models": True, "identity_grouping_unit": True,
               "TRACK_R_REOPEN_ALLOWED": False}, open(OUT / "execution_provenance.json", "w"), indent=2)
    json.dump({"lane": "pod compute; data-free operator tests CI-capable", "reproduction_tests_passed": None},
              open(OUT / "test_report.json", "w"), indent=2)
    hashes = {p.name: _sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "hashes.json"}
    json.dump(hashes, open(OUT / "hashes.json", "w"), indent=2)

    print(f"cells {n_cells}/{len(ALL)*len(ROIS)} | O4-O0 Holm-reject ROIs {n_roi_o4gto0}/7 | median O4-O0 {o4_o0_med:.4f}")
    print(f"cross-family O4 both positive: {xfam_both_pos} | low-stability cells {n_low}/{n_cells}")
    print(f"classification cohort: {dict(cohort)}")
    print(f"EXEC: {exec_status} | SCIENCE: {sci}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--participant"); ap.add_argument("--beta", default="B0")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        return freeze()
    if a.aggregate:
        return aggregate()
    if a.participant:
        return compute_participant(a.participant, a.beta)
    print("specify --freeze | --participant subjXX [--beta B0|B1] | --aggregate", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
