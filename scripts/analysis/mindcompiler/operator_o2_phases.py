"""O2 phase implementations (kept out of the lean driver): A vision-only common-space validation,
B primary imagery-zero-shot LOSO, C secondary V1/V3 + Delta_s + gauge-invariant geometry."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fmri2img.mindcompiler.operator_o1 import folds as ofo
from fmri2img.mindcompiler.operator_o1.operators import CommonVisionScaler
from fmri2img.mindcompiler.operator_o2 import pipeline as pl
from fmri2img.mindcompiler.operator_o2 import shared_ops as so
from fmri2img.mindcompiler.operator_o2 import srm as SRM


def _signflip(diffs):
    d = np.asarray([x for x in diffs if x == x], float); n = d.size
    if n == 0:
        return {"n": 0, "observed_mean": float("nan"), "p_one_sided_greater": float("nan")}
    obs = float(d.mean()); ge = sum(1 for s in itertools.product((-1., 1.), repeat=n) if float((np.array(s) * d).mean()) >= obs - 1e-12)
    return {"n": n, "observed_mean": obs, "n_assignments": 2 ** n, "p_one_sided_greater": ge / 2 ** n}


def _holm(pvals, alpha=0.05):
    items = sorted(((k, v) for k, v in pvals.items() if v == v), key=lambda kv: kv[1]); m = len(items); out = {}; run = 0.0
    for rank, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - rank) * p)); out[k] = {"raw_p": p, "holm_p": run, "reject": run <= alpha}
    return out


def _median(x):
    x = [v for v in x if v == v]
    return float(np.median(x)) if x else float("nan")


# ============================================================ PHASE A
def phase_a(load_roi, identities, kcap, OUT, ALL, PRIMARY, K_CANDIDATES, _git, _sha):
    ids, fam = identities(); outer = ofo.outer_folds(ids, fam)
    val_rows, dim_rows = [], []
    for rname in PRIMARY:
        cvis, _, nvox = load_roi(rname); Ks = kcap(nvox, ids)
        for target in ALL:
            train_subjects = [s for s in ALL if s != target]
            for kf, f in enumerate(outer):
                train_ids = list(f.train); inner = ofo.inner_folds(train_ids, fam)
                scoreK = {}
                for K in Ks:
                    margins = []
                    for it in train_subjects:
                        fitsub = [s for s in train_subjects if s != it]
                        for inf in inner:
                            margins.append(pl.vision_identity_margin(cvis, fitsub, it, list(inf.train), list(inf.test), K))
                    scoreK[K] = float(np.mean(margins))
                    val_rows.append(dict(ROI=rname, target_subject=target, identity_fold=kf, K=K, mean_vision_margin=scoreK[K]))
                Kstar = max(Ks, key=lambda k: (scoreK[k], -k))
                dim_rows.append(dict(ROI=rname, target_subject=target, identity_fold=kf, K_selected=Kstar, selected_margin=scoreK[Kstar]))
        print(f"phase-A {rname} done", flush=True)
    val = pd.DataFrame(val_rows); dim = pd.DataFrame(dim_rows)
    val.to_csv(OUT / "common_space_validation.csv", index=False)
    dim.to_csv(OUT / "common_space_dimension_selection.csv", index=False)
    roi_med = {r: _median(dim[dim.ROI == r].selected_margin) for r in PRIMARY}
    frac_pos = float((dim.selected_margin > 0).mean())
    subj_fail = [s for s in ALL if all(_median(dim[(dim.ROI == r) & (dim.target_subject == s)].selected_margin) <= 0 for r in PRIMARY)]
    validated = all(roi_med[r] > 0 for r in PRIMARY) and frac_pos >= 0.75 and len(subj_fail) == 0
    status = {"COMMON_SPACE_VALIDATED": bool(validated), "roi_median_selected_margin": roi_med,
              "fraction_outer_cells_positive_margin": frac_pos, "subjects_failing_all_primary_rois": subj_fail,
              "K_selected_distribution": {r: dict(dim[dim.ROI == r].K_selected.value_counts().sort_index()) for r in PRIMARY},
              "status": "COMMON_SPACE_VALIDATED" if validated else "O2_COMMON_SPACE_VALIDATION_FAILURE"}
    status["K_selected_distribution"] = {r: {int(k): int(v) for k, v in d.items()} for r, d in status["K_selected_distribution"].items()}
    json.dump(status, open(OUT / "common_space_validation_status.json", "w"), indent=2)
    print("PHASE-A:", status["status"], "| roi medians", {r: round(v, 3) for r, v in roi_med.items()}, "| frac_pos", round(frac_pos, 2))
    return 0


# ============================================================ lambda selection (training subjects only)
def _select_lambda(cvis, cimg, train_subjects, inner, fam, K):
    grid = so.ridge_grid(); acc = {float(l): [] for l in grid}
    for val_subj in train_subjects:
        fitsub = [s for s in train_subjects if s != val_subj]
        for inf in inner:
            calib = list(inf.train); test = list(inf.test)
            sc = {s: CommonVisionScaler.fit(np.array([cvis[(s, i)] for i in calib])) for s in fitsub + [val_subj]}
            Xtr = [sc[s].transform(np.array([cvis[(s, i)] for i in calib])).T for s in fitsub]
            Ws, S = SRM.det_srm(Xtr, K); Wmap = dict(zip(fitsub, Ws))
            Wval = SRM.new_subject_W(sc[val_subj].transform(np.array([cvis[(val_subj, i)] for i in calib])).T, S)
            Xsh = np.vstack([SRM.project(Wmap[s], sc[s].transform(np.array([cvis[(s, i)] for i in calib])).T).T for s in fitsub])
            Ysh = np.vstack([SRM.project(Wmap[s], sc[s].transform(np.array([cimg[(s, i)] for i in calib])).T).T for s in fitsub])
            for lam in grid:
                T, b = so.fit_shared_ridge(Xsh, Ysh, lam)
                for t in test:
                    x = SRM.project(Wval, sc[val_subj].transform(cvis[(val_subj, t)][None, :]).T).T[0]
                    yhat = SRM.reconstruct(Wval, (b + x @ T)[:, None])[:, 0]
                    ytrue = sc[val_subj].transform(cimg[(val_subj, t)][None, :])[0]
                    acc[float(lam)].append(so.pattern_r(yhat, ytrue))
    best = max(grid, key=lambda l: (float(np.nanmean(acc[float(l)])), l))
    return float(best)


# ============================================================ PHASE B / C core
def _run_primary(load_roi, identities, OUT, ALL, ROIS, dim_csv, _git, _sha, tag, holm_family_size):
    ids, fam = identities(); outer = ofo.outer_folds(ids, fam)
    dim = pd.read_csv(OUT / dim_csv)
    pred_rows, leak_rows, cell = [], [], {}
    spectra, deltas = {}, {}
    for rname in ROIS:
        cvis, cimg, nvox = load_roi(rname)
        for target in ALL:
            train_subjects = [s for s in ALL if s != target]
            gs, gg = [], []
            for kf, f in enumerate(outer):
                train_ids = list(f.train); test_ids = list(f.test); inner = ofo.inner_folds(train_ids, fam)
                sel = dim[(dim.ROI == rname) & (dim.target_subject == target) & (dim.identity_fold == kf)]
                K = int(sel.K_selected.iloc[0]) if len(sel) else min(4, min(nvox.values()) - 1)
                lam = _select_lambda(cvis, cimg, train_subjects, inner, fam, K)
                out = pl.evaluate_outer_cell(cvis, cimg, train_subjects, target, train_ids, test_ids, K, lam)
                leak_rows.append(dict(ROI=rname, target_subject=target, identity_fold=kf, **out["leakage"],
                                      target_vision_only_calibration=out["target_vision_only_calibration"]))
                rT = _median([r["r_Tshared"] for r in out["rows"]]); rS0 = _median([r["r_S0"] for r in out["rows"]]); rS1 = _median([r["r_S1"] for r in out["rows"]])
                gs.append(rT - rS0); gg.append(rT - rS1)
                for r in out["rows"]:
                    pred_rows.append(dict(ROI=rname, target_subject=target, identity_fold=kf, K=K, lam=lam, **r))
                if kf == 0:
                    spectra[f"{target}|{rname}"] = so.shared_operator_spectrum(out["T"])
            cell[(target, rname)] = {"G_shared": float(np.mean(gs)), "G_beyond_gain": float(np.mean(gg))}
        print(f"{tag} {rname} done", flush=True)
    return ids, fam, pred_rows, leak_rows, cell, spectra


def phase_b(load_roi, identities, OUT, ALL, PRIMARY, _git, _sha):
    st = json.load(open(OUT / "common_space_validation_status.json"))
    if not st["COMMON_SPACE_VALIDATED"]:
        json.dump({"gate": "O2", "execution_status": "O2_COMMON_SPACE_VALIDATION_FAILURE"}, open(OUT / "scientific_status.json", "w"), indent=2)
        print("STOP: common space not validated"); return 0
    ids, fam, pred_rows, leak_rows, cell, spectra = _run_primary(
        load_roi, identities, OUT, ALL, PRIMARY, "common_space_dimension_selection.csv", _git, _sha, "phase-B", 3)
    pd.DataFrame(pred_rows).to_csv(OUT / "outer_fold_predictions.csv", index=False)
    leak = pd.DataFrame(leak_rows); leak.to_csv(OUT / "leakage_certification.csv", index=False)
    leak_ok = bool(leak[[c for c in leak.columns if c.endswith("_SRM") or c.endswith("_lambda") or c.endswith("_Tshared")]].sum().sum() == 0)
    summ = pd.DataFrame([dict(target_subject=s, ROI=r, **cell[(s, r)]) for s in ALL for r in PRIMARY])
    summ.to_csv(OUT / "participant_ROI_primary_summary.csv", index=False)
    # O1 (within-subject) vs O2 (cross-subject) transfer fraction (diagnostic)
    o1 = pd.read_csv(OUT.parent / "operator_o1/participant_ROI_model_summary.csv")
    trans = []
    for s in ALL:
        for r in PRIMARY:
            o1c = o1[(o1.participant == s) & (o1.ROI == r)]
            g_native = (float(o1c[o1c.model == "O4"].held_out_r.iloc[0]) - float(o1c[o1c.model == "O0"].held_out_r.iloc[0])) if len(o1c) else float("nan")
            g_o2 = cell[(s, r)]["G_shared"]
            frac = (g_o2 / g_native) if (g_native == g_native and g_native > 0.02) else None
            trans.append(dict(subject=s, ROI=r, G_native_O1=g_native, G_shared_O2=g_o2, shared_transfer_fraction=frac))
    tdf = pd.DataFrame(trans)
    json.dump({"per_cell": trans, "median_transfer_fraction_where_native_gt_0.02": _median([t["shared_transfer_fraction"] for t in trans if t["shared_transfer_fraction"] is not None]),
               "note": "diagnostic only; not in primary classification"}, open(OUT / "O1_vs_O2_transfer.json", "w"), indent=2)
    # inference
    gsh = {r: [cell[(s, r)]["G_shared"] for s in ALL] for r in PRIMARY}
    gbg = {r: [cell[(s, r)]["G_beyond_gain"] for s in ALL] for r in PRIMARY}
    t_gs = {r: _signflip(gsh[r]) for r in PRIMARY}; h_gs = _holm({r: t_gs[r]["p_one_sided_greater"] for r in PRIMARY})
    t_gg = {r: _signflip(gbg[r]) for r in PRIMARY}; h_gg = _holm({r: t_gg[r]["p_one_sided_greater"] for r in PRIMARY})
    json.dump({"G_shared": {r: {**t_gs[r], **h_gs[r], "n_positive": int(sum(1 for v in gsh[r] if v > 0))} for r in PRIMARY}},
              open(OUT / "primary_hypothesis_tests.json", "w"), indent=2)
    json.dump({"G_beyond_gain": {r: {**t_gg[r], **h_gg[r], "median": _median(gbg[r]), "n_positive": int(sum(1 for v in gbg[r] if v > 0))} for r in PRIMARY}},
              open(OUT / "global_gain_comparison.json", "w"), indent=2)
    # leave-one-participant sensitivity (median G_shared sign stability)
    sens = {}
    for r in PRIMARY:
        base = _median(gsh[r]); flips = []
        for i in range(8):
            sub = [gsh[r][j] for j in range(8) if j != i]
            if (np.sign(_median(sub)) != np.sign(base)) and base != 0:
                flips.append(ALL[i])
        sens[r] = {"median_G_shared": base, "sign_flips_on_leave_one_out": flips}
    json.dump(sens, open(OUT / "leave_one_participant_sensitivity.json", "w"), indent=2)
    json.dump(spectra, open(OUT / "shared_operator_spectrum.json", "w"), indent=1)
    # gauge invariance runtime audit
    rng = np.random.default_rng(0); K = 5; X = rng.standard_normal((30, K)); Y = rng.standard_normal((30, K)); Q, _ = np.linalg.qr(rng.standard_normal((K, K)))
    T, b = so.fit_shared_ridge(X, Y, 1.0); Tp, bp = so.fit_shared_ridge(X @ Q, Y @ Q, 1.0)
    gi = {"T_prime_equals_Qt_T_Q_max_abs_err": float(np.abs(Tp - Q.T @ T @ Q).max()),
          "prediction_equivariant_max_abs_err": float(np.abs(so.predict_shared(Tp, bp, X @ Q) - so.predict_shared(T, b, X) @ Q).max()),
          "nti_rotation_invariant": bool(abs(so.nontrivial_transformation_index(T) - so.nontrivial_transformation_index(Q.T @ T @ Q)) < 1e-9),
          "PASS": True}
    gi["PASS"] = bool(gi["T_prime_equals_Qt_T_Q_max_abs_err"] < 1e-7 and gi["prediction_equivariant_max_abs_err"] < 1e-7 and gi["nti_rotation_invariant"])
    json.dump(gi, open(OUT / "gauge_invariance_audit.json", "w"), indent=2)
    # scientific status
    B = sum(1 for r in PRIMARY if h_gs[r]["reject"])
    C = all(_median(gsh[r]) > 0 for r in PRIMARY)
    D = all((sum(1 for v in gsh[r] if v > 0) >= 6) for r in PRIMARY if h_gs[r]["reject"])
    E = (sum(1 for r in PRIMARY if _median(gbg[r]) > 0) >= 2) and (sum(1 for r in PRIMARY if h_gg[r]["reject"]) >= 1)
    F = all(len(sens[r]["sign_flips_on_leave_one_out"]) == 0 for r in PRIMARY if h_gs[r]["reject"])
    A = True
    if not leak_ok:
        status = "SHARED_OPERATOR_INCONCLUSIVE"; exec_status = "O2_SUBJECT_OR_IDENTITY_LEAKAGE_FAILURE"
    else:
        exec_status = "O2_SHARED_OPERATOR_PASS"
        if A and B >= 2 and C and D and E and F:
            status = "SHARED_OPERATOR_SUPPORTED"
        elif B == 0 and sum(1 for r in PRIMARY if _median(gsh[r]) <= 0) >= 2:
            status = "SHARED_OPERATOR_NOT_SUPPORTED"
        else:
            status = "SHARED_OPERATOR_PARTIAL"
    gate = {"gate": "O2", "execution_status": exec_status, "scientific_status": status,
            "common_space_validated": True, "leakage_clean": leak_ok,
            "n_rois_G_shared_holm_reject": B, "all_rois_median_G_shared_positive": bool(C),
            "G_shared_median_by_ROI": {r: _median(gsh[r]) for r in PRIMARY},
            "G_beyond_gain_median_by_ROI": {r: _median(gbg[r]) for r in PRIMARY},
            "criteria": {"A_common_space": A, "B_holm_reject_ge2": B >= 2, "C_all_median_pos": bool(C),
                         "D_6of8_per_reject_roi": bool(D), "E_beyond_gain": bool(E), "F_no_sign_flip": bool(F)},
            "immutable": {"O1_STATUS": "STIMULUS_INVARIANT_OPERATOR_PARTIAL", "TRACK_R": "INDEPENDENT_METHOD_REPRODUCTION_PARTIAL"},
            "B0_measurement_dependence_caveat": True}
    json.dump(gate, open(OUT / "scientific_status.json", "w"), indent=2)
    json.dump({"gate": "O2", "phase": "B", "input_commit": _git(["rev-parse", "HEAD"], "?"), "no_B1": True,
               "no_raw_W_cross_subject": True, "target_imagery_never_used": leak_ok}, open(OUT / "execution_provenance.json", "w"), indent=2)
    json.dump({"lane": "pod compute", "o2_tests": None}, open(OUT / "test_report.json", "w"), indent=2)
    print(f"PHASE-B: {status} | Holm-reject {B}/3 | G_shared med {[round(_median(gsh[r]),4) for r in PRIMARY]} | leak_ok {leak_ok}")
    return 0


def _kselect_secondary(load_roi, identities, OUT, ALL, SECONDARY):
    """Vision-only K-selection for the secondary ROIs (identical procedure to Phase A)."""
    K_CANDIDATES = [2, 3, 4, 5, 6, 7]
    ids, fam = identities(); outer = ofo.outer_folds(ids, fam); dim_rows = []
    for rname in SECONDARY:
        cvis, _, nvox = load_roi(rname)
        Ks = [K for K in K_CANDIDATES if K <= min(nvox.values()) - 1 and K <= len(ids) - 1]
        for target in ALL:
            train_subjects = [s for s in ALL if s != target]
            for kf, f in enumerate(outer):
                train_ids = list(f.train); inner = ofo.inner_folds(train_ids, fam); scoreK = {}
                for K in Ks:
                    margins = [pl.vision_identity_margin(cvis, [s for s in train_subjects if s != it], it, list(inf.train), list(inf.test), K)
                               for it in train_subjects for inf in inner]
                    scoreK[K] = float(np.mean(margins))
                Kstar = max(Ks, key=lambda k: (scoreK[k], -k))
                dim_rows.append(dict(ROI=rname, target_subject=target, identity_fold=kf, K_selected=Kstar, selected_margin=scoreK[Kstar]))
        print(f"phase-C kselect {rname} done", flush=True)
    pd.DataFrame(dim_rows).to_csv(OUT / "common_space_dimension_selection_secondary.csv", index=False)


def phase_c(load_roi, identities, OUT, ALL, SECONDARY, _git, _sha):
    _kselect_secondary(load_roi, identities, OUT, ALL, SECONDARY)
    ids, fam, pred_rows, leak_rows, cell, spectra = _run_primary(
        load_roi, identities, OUT, ALL, SECONDARY, "common_space_dimension_selection_secondary.csv", _git, _sha, "phase-C", 2)
    gsh = {r: [cell[(s, r)]["G_shared"] for s in ALL] for r in SECONDARY}
    gbg = {r: [cell[(s, r)]["G_beyond_gain"] for s in ALL] for r in SECONDARY}
    t = {r: _signflip(gsh[r]) for r in SECONDARY}; h = _holm({r: t[r]["p_one_sided_greater"] for r in SECONDARY})
    json.dump({"label": "PREDECLARED_SECONDARY_ROIS", "cannot_change_primary_status": True,
               "G_shared": {r: {**t[r], **h[r], "median": _median(gsh[r]), "n_positive": int(sum(1 for v in gsh[r] if v > 0))} for r in SECONDARY},
               "G_beyond_gain_median": {r: _median(gbg[r]) for r in SECONDARY},
               "spectra": spectra}, open(OUT / "secondary_V1_V3.json", "w"), indent=2)
    # Delta_s decomposition + shared-action fraction (training subjects only), on primary ROIs
    _delta_and_action(load_roi, identities, OUT, ALL, ["ventral", "lateral", "parietal"])
    print(f"PHASE-C: secondary V1/V3 G_shared med {[round(_median(gsh[r]),4) for r in SECONDARY]}")
    return 0


def _delta_and_action(load_roi, identities, OUT, ALL, ROIS):
    ids, fam = identities(); outer = ofo.outer_folds(ids, fam)
    dim = pd.read_csv(OUT / "common_space_dimension_selection.csv")
    frac_rows, delta_rows = [], []
    for rname in ROIS:
        cvis, cimg, nvox = load_roi(rname)
        # use fold_0 training set (all 8 subjects are training here since Delta_s is for TRAINING subjects;
        # fit shared on all 8 x 10 train identities of fold_0, then per-subject Delta_s)
        f = outer[0]; train_ids = list(f.train)
        sel = dim[(dim.ROI == rname) & (dim.identity_fold == 0)]
        K = int(sel.K_selected.mode().iloc[0]) if len(sel) else min(4, min(nvox.values()) - 1)
        sc = {s: CommonVisionScaler.fit(np.array([cvis[(s, i)] for i in train_ids])) for s in ALL}
        Xtr = [sc[s].transform(np.array([cvis[(s, i)] for i in train_ids])).T for s in ALL]
        Ws, S = SRM.det_srm(Xtr, K); Wmap = dict(zip(ALL, Ws))
        Xsh = {s: SRM.project(Wmap[s], sc[s].transform(np.array([cvis[(s, i)] for i in train_ids])).T).T for s in ALL}
        Ysh = {s: SRM.project(Wmap[s], sc[s].transform(np.array([cimg[(s, i)] for i in train_ids])).T).T for s in ALL}
        Xall = np.vstack([Xsh[s] for s in ALL]); Yall = np.vstack([Ysh[s] for s in ALL])
        # lambda_shared via simple grid on pooled (reuse ridge grid, pick smallest CV error omitted -> median lam)
        T, b = so.fit_shared_ridge(Xall, Yall, 1.0)
        inner = ofo.inner_folds(train_ids, fam)
        pos = {i: k for k, i in enumerate(train_ids)}
        for s in ALL:
            # lambda_delta via identity-held-out CV within this training subject
            def cv(lam):
                rs = []
                for inf in inner:
                    tr = [pos[i] for i in inf.train]; te = [pos[i] for i in inf.test]
                    D = pl.fit_delta_s(Xsh[s][tr], Ysh[s][tr], T, lam)
                    bd = Ysh[s][tr].mean(0) - Xsh[s][tr].mean(0) @ (T + D)
                    for j in te:
                        rs.append(so.pattern_r(bd + Xsh[s][j] @ (T + D), Ysh[s][j]))
                return float(np.nanmean(rs))
            best = max(so.ridge_grid(), key=lambda l: (cv(l), l))
            D = pl.fit_delta_s(Xsh[s], Ysh[s], T, float(best))
            frac = so.shared_action_fraction(Xsh[s] - Xsh[s].mean(0), T, D)
            frac_rows.append(dict(ROI=rname, subject=s, shared_action_fraction=frac, lambda_delta=float(best)))
            delta_rows.append(dict(ROI=rname, subject=s, delta_frobenius=float(np.linalg.norm(D)),
                                   delta_nti=so.nontrivial_transformation_index(D + np.eye(D.shape[0]))))
    fr = pd.DataFrame(frac_rows)
    med = float(fr.shared_action_fraction.median())
    lab = ("SHARED_COMPONENT_DOMINANT" if med >= 0.67 else "MIXED_SHARED_AND_SUBJECT_SPECIFIC" if med >= 0.33 else "SUBJECT_SPECIFIC_COMPONENT_DOMINANT")
    json.dump({"per_subject_roi": frac_rows, "median_shared_action_fraction": med, "decomposition_status": lab,
               "note": "training subjects only; Delta_target never estimated"}, open(OUT / "shared_action_fraction.json", "w"), indent=2)
    json.dump({"per_subject_roi": delta_rows, "note": "T_s = T_shared + Delta_s (training subjects only)"},
              open(OUT / "subject_specific_delta_summary.json", "w"), indent=2)
