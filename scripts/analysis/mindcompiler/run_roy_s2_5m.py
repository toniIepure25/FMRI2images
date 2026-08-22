"""S2.5M public-method-aligned reconstruction driver.

--pairings                 : generate roy_vis2vis/vis2img pairing manifests (freeze; needs TSVs)
--participant subjXX [--realization N]  : compute 7 ROIs x 2 betas x 4 folds for one participant
--aggregate                : combine partials -> summaries, null, comparison, validator, gate

NO beta selection; no H-A rescue; no reproduction verdict. Raw HDF5 never committed.
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

from fmri2img.mindcompiler.roy_method_reproduction import roy_engine as re  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import roy_pairing as rp  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_folds as sf  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_roi as roi  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction.s2_2b_scorecard import spearman  # noqa: E402

DATA = _REPO / "data/nsd"
OUT = _REPO / "artifacts/mindcompiler/roy_s2_5m"
PART = OUT / "_partial"
BETA = {"B0": ("nsdimagerybetas_fithrf", "31485ff0e4cb9e90b6f83f2f550714b1a688993604f5795148a2746df7b42b64"),
        "B1": ("nsdimagerybetas_fithrf_GLMdenoise_RR", "cd42e680617d6564f403c7855140c53c0a2fdb9eb382755b49765bf781c4af2e")}
ALL = ["subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"]
NEW = ALL[1:]
ROIS = ["V1", "V2", "V3", "hV4", "ventral", "lateral", "parietal"]


def _git(a, d=""):
    for pre in (["git", "-c", "safe.directory=*", "-C", str(_REPO)], ["git", "-C", str(_REPO)]):
        try:
            return subprocess.check_output(pre + a, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            continue
    return d


def _gate():
    if "FMRI2images" not in _git(["config", "--get", "remote.origin.url"]) and \
       not (_REPO / "src/fmri2img/mindcompiler/roy_method_reproduction/roy_engine.py").exists():
        raise SystemExit("identity gate failed")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _tt_folds(subj):
    tt = sp.build_trial_table(str(DATA / "nsddata/bdata/nsdimagery"), subject=subj)
    folds = sf.build_four_folds(tt, 1234)
    rb = {r: int(tt.loc[r, "beta_index0"]) for r in tt.index}
    return tt, folds, rb


def _by(assignment, state):
    return {ident: sp_ for (ident, st), sp_ in assignment.items() if st == state}


def gen_pairings(realization=0):
    _gate()
    root = rp.root_digest(realization)
    v2v_rows, v2i_rows = [], []
    for subj in ALL:
        tt, folds, rb = _tt_folds(subj)
        for k in range(4):
            asg = folds[f"fold_{k}"]
            vis, img = _by(asg, "vision"), _by(asg, "imagery")
            for split in ("train", "val", "test"):
                for ident in sorted(vis):
                    cs_v = rp.child_seed(root, subj, f"fold_{k}", split, ident, "vis2vis")
                    s, t = rp.vis2vis_pairs(vis[ident][split], cs_v)
                    for a, b in zip(s, t):
                        v2v_rows.append(dict(participant=subj, fold=f"fold_{k}", split=split, identity=ident,
                                             model="vis2vis", source_trial_row=a, source_beta_index=rb[a],
                                             target_trial_row=b, target_beta_index=rb[b],
                                             pairing_policy=rp.V2V_POLICY, root_seed_digest=root[:16],
                                             child_seed_digest=hashlib.sha256(str(cs_v).encode()).hexdigest()[:16],
                                             derivation_version=rp.POLICY_VERSION))
                    cs_i = rp.child_seed(root, subj, f"fold_{k}", split, ident, "vis2img")
                    v, im = rp.vis2img_pairs(vis[ident][split], img[ident][split], cs_i)
                    for a, b in zip(v, im):
                        v2i_rows.append(dict(participant=subj, fold=f"fold_{k}", split=split, identity=ident,
                                             model="vis2img", source_trial_row=a, source_beta_index=rb[a],
                                             target_trial_row=b, target_beta_index=rb[b],
                                             pairing_policy=rp.V2I_POLICY, root_seed_digest=root[:16],
                                             child_seed_digest=hashlib.sha256(str(cs_i).encode()).hexdigest()[:16],
                                             derivation_version=rp.POLICY_VERSION))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(v2v_rows).to_csv(OUT / "roy_vis2vis_pairings.csv", index=False)
    pd.DataFrame(v2i_rows).to_csv(OUT / "roy_vis2img_pairings.csv", index=False)
    # validate invariants
    dv = pd.DataFrame(v2v_rows); di = pd.DataFrame(v2i_rows)
    assert (dv.source_trial_row != dv.target_trial_row).all(), "vis2vis fixed point"
    json.dump({"realization": realization, "root_digest": root,
               "vis2vis_rows": len(dv), "vis2img_rows": len(di),
               "vis2vis_no_fixed_point": bool((dv.source_trial_row != dv.target_trial_row).all()),
               "AUTHOR_RANDOM_SEED_NOT_PUBLICLY_IDENTIFIABLE": True},
              open(OUT / "pairing_seed_registry.json", "w"), indent=2)
    print(f"pairings: vis2vis {len(dv)} rows, vis2img {len(di)} rows (realization {realization})")
    return 0


def compute_participant(subj, realization=0):
    _gate()
    tt, folds, _ = _tt_folds(subj)
    root = rp.root_digest(realization)
    R = str(DATA / f"nsddata/ppdata/{subj}/func1pt8mm/roi")
    P = str(DATA / f"nsddata/ppdata/{subj}/func1pt8mm")
    N = str(DATA / f"nsddata_betas/ppdata/{subj}/func1pt8mm/betas_fithrf/ncsnr.nii.gz")
    cells, leak_rows, qc, curves = [], [], {}, {}
    for rname in ROIS:
        sel = roi.select_roi_voxels(rname, R, P, N)
        xyz = sel["xyz"]
        if xyz.shape[1] < 3:
            continue
        for b, (sub, sha) in BETA.items():
            p = DATA / f"nsddata_betas/ppdata/{subj}/func1pt8mm/{sub}/betas_nsdimagery.hdf5"
            M = sp.extract_v1_matrix(str(p), tt["beta_index0"].values, xyz)
            qc.setdefault(rname, {})[b] = dict(n_voxels=int(xyz.shape[1]), voxel_hash=sel["voxel_hash"], beta_sha=sha,
                                               finite_fraction=float(np.isfinite(M).mean()),
                                               constant_voxels=int((M.var(0) < 1e-12).sum()))
            for k in range(4):
                asg = folds[f"fold_{k}"]
                cell = re.run_fold_roy(M, _by(asg, "vision"), _by(asg, "imagery"), root, subj, f"fold_{k}",
                                       null_seed=int(hashlib.sha256(f"{subj}|{rname}|{b}|{k}|null".encode()).hexdigest()[:8], 16))
                assert all(v == 0 for v in cell.denoise_leakage.values())
                leak_rows.append(dict(participant=subj, ROI=rname, beta=b, fold=k, **cell.denoise_leakage,
                                      n_denoised_records=cell.n_denoise_records))
                cells.append(dict(participant=subj, ROI=rname, beta=b, fold=k, n_voxels=int(xyz.shape[1]),
                                  v2v_lambda_median=cell.vis2vis["lambda_median"], v2v_r_model=cell.vis2vis["r_model"],
                                  v2i_lambda_min=cell.vis2img["lambda_min"], v2i_lambda_median=cell.vis2img["lambda_median"],
                                  v2i_lambda_max=cell.vis2img["lambda_max"], v2i_n_unique_lambdas=cell.vis2img["n_unique_lambdas"],
                                  v2i_r_model=cell.vis2img["r_model"], val_score=cell.vis2img["val_score"],
                                  test_mean_r=cell.vis2img["test_mean_r"], test_median_r=cell.vis2img["test_median_r"],
                                  finite_frac=cell.vis2img["finite_frac"], target_constant=cell.vis2img["target_constant"],
                                  prediction_constant=cell.vis2img["prediction_constant"],
                                  null_mean=cell.null["null_mean"], null_p95_mean=cell.null["null_p95_mean"],
                                  fraction_voxels_above_null_p95=cell.null["fraction_voxels_above_null_p95"]))
                curves[f"{rname}|{b}|fold_{k}"] = cell.rank_curve
    PART.mkdir(parents=True, exist_ok=True)
    tag = subj if realization == 0 else f"{subj}_r{realization}"
    tmp = PART / f".{tag}.json.tmp"
    tmp.write_text(json.dumps({"participant": subj, "realization": realization, "cells": cells,
                               "leakage": leak_rows, "qc": qc, "rank_curves": curves}, indent=1))
    tmp.replace(PART / f"{tag}.json")
    print(f"{tag}: {len(cells)} cells, leakage-clean")
    return 0


def _med(x):
    x = [v for v in x if v == v]
    return float(np.median(x)) if x else float("nan")


def aggregate():
    _gate()
    parts = {s: json.loads((PART / f"{s}.json").read_text()) for s in ALL}
    cells = pd.DataFrame([c for s in ALL for c in parts[s]["cells"]])
    leak = pd.DataFrame([r for s in ALL for r in parts[s]["leakage"]])
    d = (cells.groupby(["participant", "ROI", "beta"])
         .agg(test_r=("test_mean_r", "mean"), test_median=("test_mean_r", "median"),
              r_model=("v2i_r_model", "median"), frac_above_null=("fraction_voxels_above_null_p95", "mean"),
              n_unique_lambda=("v2i_n_unique_lambdas", "median"), n_voxels=("n_voxels", "first")).reset_index())
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "participant_ROI_summary.csv", index=False)
    cells.to_csv(OUT / "aligned_vis2img_fold_results.csv", index=False)
    cells[["participant", "ROI", "beta", "fold", "v2v_lambda_median", "v2v_r_model"]].to_csv(OUT / "aligned_vis2vis_fold_results.csv", index=False)
    leak.to_csv(OUT / "D1_leakage_certification.csv", index=False)
    cells[["participant", "ROI", "beta", "fold", "v2i_lambda_min", "v2i_lambda_median", "v2i_lambda_max", "v2i_n_unique_lambdas"]].to_csv(OUT / "lambda_summary_by_cell.csv", index=False)
    cells[["participant", "ROI", "beta", "fold", "null_mean", "null_p95_mean", "fraction_voxels_above_null_p95"]].to_csv(OUT / "prediction_null_summary.csv", index=False)

    def branch(beta):
        vals = {s: _med(d[(d.participant == s) & (d.beta == beta)]["test_r"]) for s in NEW}
        fa = {s: _med(d[(d.participant == s) & (d.beta == beta)]["frac_above_null"]) for s in NEW}
        return dict(participant_test_r=vals, median=_med(list(vals.values())),
                    n_positive=int(sum(1 for v in vals.values() if v > 0)),
                    median_fraction_voxels_above_null=_med(list(fa.values())))
    b0, b1 = branch("B0"), branch("B1")
    part_summary = pd.DataFrame([dict(participant=s, B0_test_r=_med(d[(d.participant == s) & (d.beta == "B0")]["test_r"]),
                                      B1_test_r=_med(d[(d.participant == s) & (d.beta == "B1")]["test_r"])) for s in ALL])
    part_summary.to_csv(OUT / "participant_summary.csv", index=False)

    # S2.4R vs S2.5M (B0 new-cohort median)
    old = json.loads((_REPO / "artifacts/mindcompiler/roy_s2_4r/s2_4r_gate_status.json").read_text())
    comp = {"B0_new_cohort_median": {"S2_4R_scalar_lambda_indexaligned": round(old["B0_D1_new_cohort_median"], 4),
                                     "S2_5M_pertarget_royaligned": round(b0["median"], 4)},
            "B1_new_cohort_median": {"S2_4R": round(old["B1_D1_new_cohort_median"], 4), "S2_5M": round(b1["median"], 4)},
            "note": "Descriptive post-hoc method-remediation comparison; no pipeline selected by score."}
    json.dump(comp, open(OUT / "S2_4R_vs_S2_5M_comparison.json", "w"), indent=2)

    n_cells = len(cells); dup = int(cells.duplicated(["participant", "ROI", "beta", "fold"]).sum())
    leak_ok = bool(leak[["self_target", "self_source", "val_test_in_training", "cross_identity_target", "training_row_not_train_split"]].sum().sum() == 0)
    scalar_fallback = bool((cells["v2i_n_unique_lambdas"] <= 1).all())  # would indicate scalar collapse
    validator = {"expected_cells": 448, "observed_cells": n_cells, "duplicate_cells": dup,
                 "all_cells_present": n_cells == 448, "leakage_clean": leak_ok,
                 "no_scalar_lambda_fallback": (not scalar_fallback),
                 "n_unique_lambda_median": float(cells["v2i_n_unique_lambdas"].median())}
    for name, obj in [("B0_primary_characterization.json", {"branch": "B0 (b2-compatible primary)", **b0}),
                      ("B1_sensitivity_characterization.json", {"branch": "B1 (b3 sensitivity; NOT a failure)", **b1}),
                      ("method_gap_resolution.json", {"vis2img_pairing": "MATCH (random within-identity)",
                          "vis2vis_pairing": "MATCH (within-split derangement)", "per_target_lambda": "MATCH (voxel-specific)",
                          "operational_rank": "CLOSE_DEFENSIBLE (validation argmax)", "prediction_null": "CLOSE_DEFENSIBLE (test-fold shuffle; N=1000)",
                          "note": "resolves the S2.5R prediction-class method gaps; dimensionality/alignment still DEFERRED_TO_S2_6R"}),
                      ("prediction_concordance_update.json", {"B0_status": "PREDICTION_DIRECTIONALLY_CONCORDANT_ONLY" if b0["median"] > 0 else "PREDICTION_NOT_CONCORDANT",
                          "B0_median_r": round(b0["median"], 4), "B0_n_positive": b0["n_positive"],
                          "B0_median_fraction_voxels_above_null_p95": round(b0["median_fraction_voxels_above_null"], 4),
                          "note": "prediction now tested against a paper-style shuffle null; still not a full Roy verdict"}),
                      ("remaining_ambiguities.json", {"items": ["AUTHOR_RANDOM_SEED_UNKNOWN", "EXACT_DENOISING_TRAIN_DEPENDENCY_UNKNOWN",
                          "CENTER_SCALE_SCOPE_UNKNOWN", "ORIGINAL_CODE_UNAVAILABLE", "BITWISE_REPLICATION_UNAVAILABLE",
                          "ROY_DIMENSIONALITY_RESULT_NOT_YET_RECONSTRUCTED", "ROY_ALIGNMENT_RATIO_NOT_YET_RECONSTRUCTED"],
                          "FULL_INDEPENDENT_REPRODUCTION_VERDICT": "STILL_DEFERRED (S2.6R)"})]:
        (OUT / name).write_text(json.dumps(obj, indent=2))
    json.dump({s: parts[s]["qc"] for s in ALL}, open(OUT / "participant_roi_value_qc.json", "w"), indent=2)
    json.dump({k: parts[s]["rank_curves"] for s in ALL for k in [s]}, open(OUT / "rank_curves_index.json", "w"), indent=2)
    json.dump({"gate": "S2.5M", "input_commit": _git(["rev-parse", "HEAD"], "unknown"),
               "frozen_before_outcomes": True, "SAVE_FULL_RANK_CURVES": True, "DIMENSIONALITY_VERDICT": "DEFERRED_TO_S2_6R"},
              open(OUT / "execution_provenance.json", "w"), indent=2)
    json.dump({"lane": "pod compute; data-free tests CI-capable", "reproduction_tests_passed": None}, open(OUT / "test_report.json", "w"), indent=2)

    status = "S2_5M_PUBLIC_METHOD_ALIGNMENT_PASS" if (validator["all_cells_present"] and leak_ok and validator["no_scalar_lambda_fallback"] and dup == 0) else \
        ("S2_5M_D1_LEAKAGE_FAILURE" if not leak_ok else ("S2_5M_PER_TARGET_RIDGE_FAILURE" if scalar_fallback else "S2_5M_INCOMPLETE_EXECUTION"))
    gate = {"gate": "S2.5M", "status": status, "validator": validator,
            "B0_PUBLIC_METHOD_ALIGNED_CHARACTERIZATION": {"median_test_r": round(b0["median"], 4), "n_positive": b0["n_positive"],
                "median_fraction_voxels_above_null_p95": round(b0["median_fraction_voxels_above_null"], 4)},
            "B1_METHOD_SENSITIVITY_CHARACTERIZATION": {"median_test_r": round(b1["median"], 4), "n_positive": b1["n_positive"]},
            "reproduction_verdict": "STILL_DEFERRED (dimensionality+alignment -> S2.6R)",
            "frozen_H_A_unchanged": "H_A_CROSS_PARTICIPANT_PARTIAL",
            "next_action": {"next": "S2.6R -- ROY-DEFINED DIMENSIONALITY AND ALIGNMENT RECONSTRUCTION (method-aligned B0; no proxies)",
                            "not_done_here": "No dimensionality/alignment verdict; no beta selection."}}
    json.dump(gate, open(OUT / "s2_5m_gate_status.json", "w"), indent=2)
    hashes = {p.name: _sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "hashes.json"}
    json.dump(hashes, open(OUT / "hashes.json", "w"), indent=2)
    print(f"validator cells {n_cells}/448 leak_clean={leak_ok} no_scalar={validator['no_scalar_lambda_fallback']}")
    print(f"B0 median test r={b0['median']:.4f} ({b0['n_positive']}/7) frac>null={b0['median_fraction_voxels_above_null']:.3f}")
    print(f"B1 median test r={b1['median']:.4f} ({b1['n_positive']}/7)")
    print(f"S2.4R B0={old['B0_D1_new_cohort_median']:.4f} -> S2.5M B0={b0['median']:.4f}")
    print("GATE:", status)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairings", action="store_true")
    ap.add_argument("--participant"); ap.add_argument("--realization", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.pairings:
        return gen_pairings(a.realization)
    if a.aggregate:
        return aggregate()
    if a.participant:
        return compute_participant(a.participant, a.realization)
    print("specify --pairings | --participant subjXX | --aggregate", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
