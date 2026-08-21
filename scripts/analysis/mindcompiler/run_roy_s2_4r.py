"""S2.4R full-cohort D1 characterization (per-participant compute + aggregate).

Executes the frozen D1 pipeline (vis2vis -> D1_STRICT_CROSSFIT -> vis2img) for one
participant across 7 ROIs x 2 betas x 4 folds, or aggregates all partials into the
full-cohort characterization. D1-vs-RAW reuses the frozen S2.3 RAW. Descriptive;
participant is the unit; no beta selection; no Roy verdict; no raw arrays committed.

    python run_roy_s2_4r.py --participant subj02      # one participant (parallel-safe)
    python run_roy_s2_4r.py --aggregate               # after all 8 partials exist
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

from fmri2img.mindcompiler.roy_method_reproduction import s2_folds as sf  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_reconstruction as rc  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_roi as roi  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction.s2_2b_scorecard import spearman  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction.s2_preprocessing import PREPROC_ZSCORE_TRAIN_ONLY  # noqa: E402

DATA = _REPO / "data/nsd"
OUT = _REPO / "artifacts/mindcompiler/roy_s2_4r"
PART = OUT / "_partial"
BETA = {"B0": ("nsdimagerybetas_fithrf", "31485ff0e4cb9e90b6f83f2f550714b1a688993604f5795148a2746df7b42b64"),
        "B1": ("nsdimagerybetas_fithrf_GLMdenoise_RR", "cd42e680617d6564f403c7855140c53c0a2fdb9eb382755b49765bf781c4af2e")}
ALL = ["subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"]
NEW = ["subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"]
ROIS = ["V1", "V2", "V3", "hV4", "ventral", "lateral", "parietal"]
NEAR_NEUTRAL = 0.02


def d1_category(g: float) -> str:
    """Frozen descriptive D1-vs-RAW gain category (threshold 0.02)."""
    return "D1_GAIN" if g > NEAR_NEUTRAL else ("D1_LOSS" if g < -NEAR_NEUTRAL else "D1_NEAR_NEUTRAL")


def cohort_beta_status(s_d1_beta_loss, n_required: int = 5) -> str:
    """Frozen cohort beta-sensitivity status over the NEW-cohort participant losses."""
    import numpy as _np
    vals = [v for v in s_d1_beta_loss if v == v]  # drop NaN
    med = float(_np.median(vals)) if vals else float("nan")
    n_pos = sum(1 for v in vals if v > 0)
    n_neg = sum(1 for v in vals if v < 0)
    if med > 0 and n_pos >= n_required:
        return "D1_BETA_SENSITIVITY_PERSISTS"
    if med < 0 and n_neg >= n_required:
        return "D1_BETA_SENSITIVITY_REVERSES"
    return "D1_BETA_SENSITIVITY_MIXED"


def _git(args, d=""):
    for pre in (["git", "-c", "safe.directory=*", "-C", str(_REPO)], ["git", "-C", str(_REPO)]):
        try:
            return subprocess.check_output(pre + args, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            continue
    return d


def _gate():
    ok = "FMRI2images" in _git(["config", "--get", "remote.origin.url"]) or \
         (_REPO / "src/fmri2img/mindcompiler/roy_method_reproduction/s2_roi.py").exists()
    if not ok:
        raise SystemExit("identity gate failed")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compute_participant(subj: str) -> int:
    _gate()
    tt = sp.build_trial_table(str(DATA / "nsddata/bdata/nsdimagery"), subject=subj)
    folds = sf.build_four_folds(tt, 1234)
    rid = {r: tt.loc[r, "identity"] for r in tt.index}
    rb = {r: int(tt.loc[r, "beta_index0"]) for r in tt.index}
    R = str(DATA / f"nsddata/ppdata/{subj}/func1pt8mm/roi")
    P = str(DATA / f"nsddata/ppdata/{subj}/func1pt8mm")
    N = str(DATA / f"nsddata_betas/ppdata/{subj}/func1pt8mm/betas_fithrf/ncsnr.nii.gz")
    cells, qc, leak_rows = [], {}, []
    for rname in ROIS:
        sel = roi.select_roi_voxels(rname, R, P, N)
        xyz = sel["xyz"]
        if xyz.shape[1] < 3:
            continue
        M = {}
        for b, (sub, sha) in BETA.items():
            p = DATA / f"nsddata_betas/ppdata/{subj}/func1pt8mm/{sub}/betas_nsdimagery.hdf5"
            M[b] = sp.extract_v1_matrix(str(p), tt["beta_index0"].values, xyz)
        qc[rname] = {b: dict(n_voxels=int(xyz.shape[1]), voxel_hash=sel["voxel_hash"], beta_sha=BETA[b][1],
                             finite_fraction=float(np.isfinite(M[b]).mean()),
                             constant_voxels=int((M[b].var(axis=0) < 1e-12).sum()),
                             int16_min=float((M[b] * 300).min()), int16_max=float((M[b] * 300).max()),
                             saturation=int((np.abs(M[b] * 300) >= 32767).sum())) for b in BETA}
        for b in BETA:
            for k in range(4):
                res, recs = rc.run_fold(M[b], folds[f"fold_{k}"], rid, rb, beta_version=b,
                                        preproc_policy=PREPROC_ZSCORE_TRAIN_ONLY,
                                        vis2vis_pairing="all_ordered_distinct", pairing_seed=None,
                                        fold_name=f"fold_{k}", return_records=True, denoising="D1")
                assert all(v == 0 for v in res.denoise_leakage.values()), (subj, rname, b, k, res.denoise_leakage)
                dep_hash = hashlib.sha256(
                    ("|".join(f"{r.denoised_row}:{r.model_id}" for r in recs)).encode()).hexdigest()[:16]
                leak_rows.append(dict(participant=subj, ROI=rname, beta=b, fold=f"fold_{k}",
                                      n_denoised_records=len(recs), **res.denoise_leakage, dependency_hash=dep_hash))
                cells.append(dict(participant=subj, ROI=rname, beta=b, fold=k, n_voxels=int(xyz.shape[1]),
                                  v2v_lam=res.vis2vis["lam"], v2v_rank=res.vis2vis["rank"], v2v_val=res.vis2vis["val_score"],
                                  v2i_lam=res.vis2img["lam"], v2i_rank=res.vis2img["rank"],
                                  v2i_val=res.vis2img["val_score"], test_r=res.vis2img["mean"],
                                  median=res.vis2img["median"], std=res.vis2img["std"],
                                  finite_frac=res.vis2img["finite_frac"],
                                  target_constant=res.vis2img["target_constant"],
                                  prediction_constant=res.vis2img["prediction_constant"]))
    PART.mkdir(parents=True, exist_ok=True)
    tmp = PART / f".{subj}.json.tmp"
    tmp.write_text(json.dumps({"participant": subj, "cells": cells, "qc": qc, "leakage": leak_rows}, indent=1))
    tmp.replace(PART / f"{subj}.json")  # atomic
    print(f"{subj}: {len(cells)} fold-cells, {len(leak_rows)} leakage rows, all leakage-clean")
    return 0


def _med(x):
    return float(np.median(x)) if len(x) else float("nan")


def aggregate() -> int:
    _gate()
    parts = {s: json.loads((PART / f"{s}.json").read_text()) for s in ALL}
    cells = pd.DataFrame([c for s in ALL for c in parts[s]["cells"]])
    leak = pd.DataFrame([r for s in ALL for r in parts[s]["leakage"]])
    # per participant x ROI x beta four-fold aggregate D1 test r
    d1 = (cells.groupby(["participant", "ROI", "beta"])
          .agg(D1_r=("test_r", "mean"), D1_median=("test_r", "median"), D1_std=("test_r", "std"),
               D1_min=("test_r", "min"), D1_max=("test_r", "max"),
               median_rank=("v2i_rank", "median"), rank_min=("v2i_rank", "min"), rank_max=("v2i_rank", "max"),
               v2v_val=("v2v_val", "mean"), n_voxels=("n_voxels", "first"))
          .reset_index())
    d1w = d1.pivot_table(index=["participant", "ROI"], columns="beta", values="D1_r").reset_index()
    d1w.columns = ["participant", "ROI", "D1_B0", "D1_B1"]
    # reuse frozen S2.3 RAW
    raw = pd.read_csv(_REPO / "artifacts/mindcompiler/roy_s2_3/ROI_level_primary_table.csv")[
        ["participant", "ROI", "B0_RAW_r", "B1_RAW_r", "B0_img_rel", "B1_img_rel", "B0_vis_rel", "B1_vis_rel"]]
    t = d1w.merge(raw, on=["participant", "ROI"], how="left")
    t["G_B0"] = t["D1_B0"] - t["B0_RAW_r"]
    t["G_B1"] = t["D1_B1"] - t["B1_RAW_r"]
    t["RAW_loss"] = t["B0_RAW_r"] - t["B1_RAW_r"]
    t["D1_loss"] = t["D1_B0"] - t["D1_B1"]
    t["Delta_beta_sensitivity"] = t["D1_loss"] - t["RAW_loss"]
    t["L_rel_img"] = t["B0_img_rel"] - t["B1_img_rel"]
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "participant_ROI_D1_table.csv", index=False)
    cells.to_csv(OUT / "vis2img_D1_fold_results.csv", index=False)
    cells[["participant", "ROI", "beta", "fold", "v2v_lam", "v2v_rank", "v2v_val"]].to_csv(OUT / "vis2vis_fold_results.csv", index=False)
    leak.to_csv(OUT / "D1_leakage_certification.csv", index=False)
    (OUT / "D1_vs_RAW_by_participant_roi.csv").write_text(
        t[["participant", "ROI", "D1_B0", "D1_B1", "B0_RAW_r", "B1_RAW_r", "G_B0", "G_B1",
           "RAW_loss", "D1_loss", "Delta_beta_sensitivity"]].to_csv(index=False))

    def part_summ(df):
        out = {}
        for s in df["participant"].unique():
            g = df[df.participant == s]
            out[s] = dict(D1_B0=_med(g["D1_B0"]), D1_B1=_med(g["D1_B1"]),
                          S_D1_beta_loss=_med(g["D1_loss"]), G_B0_subject=_med(g["G_B0"]),
                          G_B1_subject=_med(g["G_B1"]), Delta_subject=_med(g["Delta_beta_sensitivity"]),
                          S_rel_img=_med(g["L_rel_img"]))
        return out
    ps = part_summ(t)
    pd.DataFrame([dict(participant=s, **ps[s]) for s in ALL]).to_csv(OUT / "participant_D1_summary.csv", index=False)

    d1_categories = {s: {"B0": d1_category(ps[s]["G_B0_subject"]), "B1": d1_category(ps[s]["G_B1_subject"])} for s in ALL}

    # cohort beta-sensitivity (NEW cohort only)
    sdl = [ps[s]["S_D1_beta_loss"] for s in NEW]
    n_pos = sum(1 for v in sdl if v > 0); n_neg = sum(1 for v in sdl if v < 0)
    med_sdl = _med(sdl)
    beta_status = cohort_beta_status(sdl)

    # ROI profile (NEW cohort)
    roi_prof = []
    for r in ROIS:
        for b in ("B0", "B1"):
            col = "D1_B0" if b == "B0" else "D1_B1"
            gg = t[(t.ROI == r) & (t.participant.isin(NEW))]
            roi_prof.append(dict(ROI=r, beta=b, median_D1_r=_med(gg[col]), mean_D1_r=float(gg[col].mean()),
                                 mad=float(np.median(np.abs(gg[col] - np.median(gg[col])))),
                                 n_positive=int((gg[col] > 0).sum())))
    pd.DataFrame(roi_prof).to_csv(OUT / "ROI_D1_summary.csv", index=False)

    # branch cohort characterization (NEW)
    def branch(col):
        vals = [_med(t[(t.participant == s)][col]) for s in NEW]
        return dict(participant_values={NEW[i]: vals[i] for i in range(len(NEW))},
                    median=_med(vals), mean=float(np.mean(vals)),
                    mad=float(np.median(np.abs(np.array(vals) - np.median(vals)))),
                    min=float(np.min(vals)), max=float(np.max(vals)),
                    n_positive=int(sum(1 for v in vals if v > 0)), n_negative=int(sum(1 for v in vals if v < 0)))
    b0_char = branch("D1_B0"); b1_char = branch("D1_B1")

    # reliability context (no H-A reopening)
    rho_img = spearman([ps[s]["S_rel_img"] for s in NEW], [ps[s]["S_D1_beta_loss"] for s in NEW])

    # validator: 448 cells, leakage clean
    n_cells = len(cells)
    dup = cells.duplicated(subset=["participant", "ROI", "beta", "fold"]).sum()
    leak_ok = bool((leak[["self_target", "self_source", "val_test_in_training",
                          "cross_identity_target", "training_row_not_train_split"]].sum().sum()) == 0)
    validator = {"expected_cells": 448, "observed_cells": int(n_cells), "duplicate_cells": int(dup),
                 "all_cells_present": bool(n_cells == 448), "leakage_clean": leak_ok,
                 "participants": sorted(cells.participant.unique().tolist()),
                 "rois": sorted(cells.ROI.unique().tolist()), "betas": sorted(cells.beta.unique().tolist())}

    def w(n, o):
        (OUT / n).write_text(json.dumps(o, indent=2))
    w("D1_vs_RAW_summary.json", {"per_participant": {s: dict(G_B0=ps[s]["G_B0_subject"], G_B1=ps[s]["G_B1_subject"]) for s in ALL},
                                 "D1_categories": d1_categories, "near_neutral_threshold": NEAR_NEUTRAL})
    w("beta_sensitivity_D1.json", {"S_D1_beta_loss": {s: ps[s]["S_D1_beta_loss"] for s in ALL},
                                   "new_cohort_median": med_sdl, "n_positive_of_7": n_pos, "n_negative_of_7": n_neg,
                                   "cohort_status": beta_status,
                                   "Delta_beta_sensitivity_subject": {s: ps[s]["Delta_subject"] for s in ALL},
                                   "note": "Descriptive; NOT causal; not used to select a beta."})
    w("reliability_context.json", {"rho_imagery_reliability_loss_vs_S_D1_beta_loss": rho_img,
                                   "note": "Context only; H_A_CROSS_PARTICIPANT_PARTIAL NOT reopened; no p-value rescue."})
    w("subj07_D1_context.json", {"D1_B0": ps["subj07"]["D1_B0"], "D1_B1": ps["subj07"]["D1_B1"],
                                 "G_B0": ps["subj07"]["G_B0_subject"], "G_B1": ps["subj07"]["G_B1_subject"],
                                 "S_D1_beta_loss": ps["subj07"]["S_D1_beta_loss"],
                                 "roi_D1": t[t.participant == "subj07"][["ROI", "D1_B0", "D1_B1", "B0_RAW_r", "B1_RAW_r"]].to_dict("records"),
                                 "note": "subj07 fully included (VALID_SCIENTIFIC_HETEROGENEITY); not excluded."})
    w("full_cohort_characterization.json", {
        "analysis_class": "FROZEN_FULL_COHORT_INDEPENDENT_METHOD_RECONSTRUCTION_CHARACTERIZATION",
        "reference_D1": "subj01", "new_D1_cohort": NEW,
        "B0_D1_COHORT_CHARACTERIZATION": b0_char, "B1_D1_COHORT_CHARACTERIZATION": b1_char,
        "beta_sensitivity_status": beta_status, "reliability_context_rho": rho_img,
        "validator": validator})
    w("paper_comparability_inventory.json", {
        "DIRECTLY_COMPARABLE_PUBLIC_CLAIMS": ["two-stage vis2vis->vis2img structure", "reduced-rank ridge",
            "4/2/2 split", "ridge grid 1e-3..1e5", "12 identities / 6+6 / 8 repeats", "vision denoised before imagery",
            "positive perception->imagery mapping in early visual ROIs (qualitative direction)"],
        "NOT_IDENTIFIABLE_FROM_PUBLIC_METHODS": ["exact beta version", "exact author denoising implementation",
            "exact centering/scaling scope", "exact fold dependency graph", "exact pairing/seed choices", "tie-breaking"],
        "note": "For S2.5R only; concordance rubric MUST be frozen independently before comparison."})
    w("input_integrity_report.json", {"note": "Reuses S2.3 participant_acquisition_registry; 16 HDF5 verified.",
                                      "registry": "artifacts/mindcompiler/roy_s2_3/participant_acquisition_registry.json"})
    w("participant_roi_value_qc.json", {s: parts[s]["qc"] for s in ALL})
    w("execution_provenance.json", {"gate": "S2.4R", "input_commit": _git(["rev-parse", "HEAD"], "unknown"),
                                    "frozen_config_before_new_D1": True, "cells": int(n_cells)})
    w("test_report.json", {"lane": "pod compute (Lane E env); data-free tests CI-capable", "reproduction_tests_passed": None})

    status = "S2_4R_FULL_COHORT_D1_PASS" if (validator["all_cells_present"] and leak_ok and dup == 0) else \
        ("S2_4R_LEAKAGE_FAILURE" if not leak_ok else "S2_4R_INCOMPLETE_EXECUTION")
    gate = {"gate": "S2.4R", "status": status, "validator": validator,
            "B0_D1_new_cohort_median": b0_char["median"], "B0_n_positive": b0_char["n_positive"],
            "B1_D1_new_cohort_median": b1_char["median"], "B1_n_positive": b1_char["n_positive"],
            "beta_sensitivity_status": beta_status, "reliability_context_rho": rho_img,
            "frozen_H_A_unchanged": "H_A_CROSS_PARTICIPANT_PARTIAL",
            "next_action": {"next": "S2.5R -- PAPER-CONCORDANCE AND INDEPENDENT-REPRODUCTION VERDICT AUDIT (freeze rubric independently first)",
                            "not_done_here": "No Roy verdict; no Track-O operator work."},
            "note": "Execution/characterization gate; NOT a Roy reproduction verdict; no beta selected."}
    w("s2_4r_gate_status.json", gate)
    hashes = {p.name: _sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "hashes.json"}
    w("hashes.json", hashes)

    print(f"validator: cells {n_cells}/448 present={validator['all_cells_present']} dup={dup} leakage_clean={leak_ok}")
    print(f"B0 D1 new-cohort median r={b0_char['median']:.4f} ({b0_char['n_positive']}/7 positive)")
    print(f"B1 D1 new-cohort median r={b1_char['median']:.4f} ({b1_char['n_positive']}/7 positive)")
    print(f"beta-sensitivity: {beta_status} (median S_D1_beta_loss={med_sdl:.4f}, {n_pos}/7 positive)")
    print(f"reliability-context rho(imagery loss, S_D1_beta_loss)={rho_img:.3f}")
    print("D1 categories:", {s: d1_categories[s] for s in NEW})
    print("GATE:", status)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        return aggregate()
    if a.participant:
        return compute_participant(a.participant)
    print("specify --participant subjXX or --aggregate", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
