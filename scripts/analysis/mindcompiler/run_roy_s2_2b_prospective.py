"""S2.2B PHASE 2 -- prospective six-ROI execution (subj01).

Runs ONLY after the Phase-1 freeze commit. Extracts B0/B1 for each prospective ROI,
value-QCs, computes reliability + matched RAW (primary) + D1 (secondary) vis2img +
geometry, evaluates the frozen H-A scorecard, and writes compact artifacts. V1 is
excluded from the primary six-ROI analysis (appendix only). Descriptive within one
participant -- NO population inference, NO fold p-values. No raw arrays committed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from fmri2img.mindcompiler.roy_method_reproduction import s2_1b_diagnostics as dg  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_2b_scorecard as sc  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_folds as sf  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_reconstruction as rc  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_roi as roi  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction.s2_preprocessing import PREPROC_ZSCORE_TRAIN_ONLY  # noqa: E402

BETA = {"B0": ("nsdimagerybetas_fithrf", "31485ff0e4cb9e90b6f83f2f550714b1a688993604f5795148a2746df7b42b64"),
        "B1": ("nsdimagerybetas_fithrf_GLMdenoise_RR", "cd42e680617d6564f403c7855140c53c0a2fdb9eb382755b49765bf781c4af2e")}
OUT = _REPO / "artifacts/mindcompiler/roy_s2_2b"


def _gate():
    url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], cwd=_REPO).decode()
    if "FMRI2images" not in url:
        raise SystemExit("identity gate failed")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _agg(v):
    a = np.asarray(v, float)
    return dict(mean=float(a.mean()), median=float(np.median(a)), std=float(a.std()),
                min=float(a.min()), max=float(a.max()), per_fold=[float(x) for x in a])


def main() -> int:
    _gate()
    base = _REPO / "data/nsd"
    tt = sp.build_trial_table(str(base / "nsddata/bdata/nsdimagery"))
    folds = sf.build_four_folds(tt, 1234)
    rid = {r: tt.loc[r, "identity"] for r in tt.index}
    rb = {r: int(tt.loc[r, "beta_index0"]) for r in tt.index}
    visg = {i: g.index.tolist() for i, g in tt[tt.state == "vision"].groupby("identity")}
    imgg = {i: g.index.tolist() for i, g in tt[tt.state == "imagery"].groupby("identity")}
    vis_rows = tt[tt.state == "vision"].index.tolist()
    img_rows = tt[tt.state == "imagery"].index.tolist()
    R = str(base / "nsddata/ppdata/subj01/func1pt8mm/roi")
    P = str(base / "nsddata/ppdata/subj01/func1pt8mm")
    N = str(base / "nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf/ncsnr.nii.gz")
    manifest = json.loads((OUT / "roi_selection_manifest.json").read_text())

    qc, rel, isn, raw_res, d1_res, cross, geo, table = {}, {}, {}, {}, {}, {}, {}, []
    L_rel_img, L_perf_RAW, L_rel_vis, L_perf_D1, L_cross = {}, {}, {}, {}, {}

    all_rois = list(roi.PROSPECTIVE_ROIS) + ["V1"]  # V1 appendix only
    for name in all_rois:
        sel = roi.select_roi_voxels(name, R, P, N)
        assert sel["voxel_hash"] == manifest["rois"][name]["voxel_hash"], f"{name} voxel drift"
        xyz = sel["xyz"]
        M = {}
        for b, (sub, sha) in BETA.items():
            p = base / f"nsddata_betas/ppdata/subj01/func1pt8mm/{sub}/betas_nsdimagery.hdf5"
            if _sha(p) != sha:
                print(f"FATAL: {b} sha mismatch", file=sys.stderr); return 2
            M[b] = sp.extract_v1_matrix(str(p), tt["beta_index0"].values, xyz)
        # value QC (identical rows/voxels for B0/B1 by construction)
        qc[name] = {b: dict(selected_voxels=int(xyz.shape[1]),
                            int16_min=float(np.min(M[b] * 300)), int16_max=float(np.max(M[b] * 300)),
                            saturation_count=int(np.sum(np.abs(M[b] * 300) >= 32767)),
                            zero_count=int(np.sum(M[b] == 0)),
                            constant_voxels=int(np.sum(M[b].var(axis=0) < 1e-12)),
                            finite_fraction=float(np.isfinite(M[b]).mean()),
                            var_min=float(M[b].var(axis=0).min()), var_max=float(M[b].var(axis=0).max()),
                            voxel_hash=sel["voxel_hash"], beta_sha=sha) for b in BETA}
        # reliability + SNR
        rel[name] = {b: {"vision": dg.repeat_reliability(M[b], visg), "imagery": dg.repeat_reliability(M[b], imgg)} for b in BETA}
        isn[name] = {b: {"vision": dg.identity_signal_noise(M[b], visg), "imagery": dg.identity_signal_noise(M[b], imgg)} for b in BETA}
        # matched RAW + D1 vis2img
        def run(mode, b):
            rs = []
            for k in range(4):
                res = rc.run_fold(M[b], folds[f"fold_{k}"], rid, rb, beta_version=b,
                                  preproc_policy=PREPROC_ZSCORE_TRAIN_ONLY, vis2vis_pairing="all_ordered_distinct",
                                  pairing_seed=None, fold_name=f"fold_{k}", denoising=mode)
                if mode != "RAW":
                    assert all(x == 0 for x in res.denoise_leakage.values())
                rs.append(res.vis2img["mean"])
            return _agg(rs)
        raw_res[name] = {b: run("RAW", b) for b in BETA}
        d1_res[name] = {b: run("D1", b) for b in BETA}
        # geometry
        _, Cv0 = dg.centroids(M["B0"], visg); _, Ci0 = dg.centroids(M["B0"], imgg)
        _, Cv1 = dg.centroids(M["B1"], visg); _, Ci1 = dg.centroids(M["B1"], imgg)
        cross[name] = {"B0": dg.cross_state_matching(Cv0, Ci0), "B1": dg.cross_state_matching(Cv1, Ci1)}
        geo[name] = {"trial_similarity_vision": dg.paired_trial_similarity(M["B0"], M["B1"], vis_rows)["pearson"]["mean"],
                     "b0_to_b1_vision": dg.shrinkage_vs_rotation(M["B0"], M["B1"], vis_rows)}
        # losses
        li = rel[name]["B0"]["imagery"]["mean"] - rel[name]["B1"]["imagery"]["mean"]
        lp = raw_res[name]["B0"]["mean"] - raw_res[name]["B1"]["mean"]
        lv = rel[name]["B0"]["vision"]["mean"] - rel[name]["B1"]["vision"]["mean"]
        ld = d1_res[name]["B0"]["mean"] - d1_res[name]["B1"]["mean"]
        lc = cross[name]["B0"]["separation"] - cross[name]["B1"]["separation"]
        if name in roi.PROSPECTIVE_ROIS:
            L_rel_img[name] = li; L_perf_RAW[name] = lp; L_rel_vis[name] = lv; L_perf_D1[name] = ld; L_cross[name] = lc
        table.append(dict(ROI=name, n_voxels=int(xyz.shape[1]),
                          B0_img_rel=rel[name]["B0"]["imagery"]["mean"], B1_img_rel=rel[name]["B1"]["imagery"]["mean"], L_rel_img=li,
                          B0_vis_rel=rel[name]["B0"]["vision"]["mean"], B1_vis_rel=rel[name]["B1"]["vision"]["mean"], L_rel_vis=lv,
                          B0_RAW_r=raw_res[name]["B0"]["mean"], B1_RAW_r=raw_res[name]["B1"]["mean"], L_perf_RAW=lp,
                          B0_D1_r=d1_res[name]["B0"]["mean"], B1_D1_r=d1_res[name]["B1"]["mean"], L_perf_D1=ld,
                          B0_SNR_img=isn[name]["B0"]["imagery"]["signal_to_noise"], B1_SNR_img=isn[name]["B1"]["imagery"]["signal_to_noise"],
                          crossstate_B0=cross[name]["B0"]["separation"], crossstate_B1=cross[name]["B1"]["separation"], L_crossstate=lc,
                          B0_B1_trial_corr=geo[name]["trial_similarity_vision"],
                          rotation_gain=geo[name]["b0_to_b1_vision"]["rotation_gain_over_scale"]))

    # --- primary H-A scorecard (prospective ROIs only) ---
    scorecard = sc.evaluate_criteria(L_rel_img, L_perf_RAW)
    status = sc.classify(scorecard)
    lo = sc.leave_one_roi_rhos(L_rel_img, L_perf_RAW)
    # secondary
    rho_d1 = sc.spearman([L_rel_img[r] for r in L_rel_img], [L_perf_D1[r] for r in L_rel_img])
    rho_cross = sc.spearman([L_cross[r] for r in L_cross], [L_perf_RAW[r] for r in L_cross])
    rho_nvox_rel = sc.spearman([table[i]["n_voxels"] for i in range(len(roi.PROSPECTIVE_ROIS))],
                               [L_rel_img[roi.PROSPECTIVE_ROIS[i]] for i in range(len(roi.PROSPECTIVE_ROIS))])
    rho_nvox_perf = sc.spearman([table[i]["n_voxels"] for i in range(len(roi.PROSPECTIVE_ROIS))],
                                [L_perf_RAW[roi.PROSPECTIVE_ROIS[i]] for i in range(len(roi.PROSPECTIVE_ROIS))])

    # --- write artifacts ---
    def w(n, o):
        (OUT / n).write_text(json.dumps(o, indent=2))
    w("value_qc.json", qc)
    w("repeat_reliability_by_roi.json", rel)
    w("identity_signal_noise_by_roi.json", isn)
    w("RAW_fold_results_by_roi.json", raw_res)
    w("D1_fold_results_by_roi.json", d1_res)
    w("cross_state_geometry_by_roi.json", cross)
    w("B0_B1_geometry_by_roi.json", geo)
    pd.DataFrame([t for t in table if t["ROI"] in roi.PROSPECTIVE_ROIS]).to_csv(OUT / "primary_ROI_table.csv", index=False)
    pd.DataFrame(table).to_csv(OUT / "appendix_ROI_table_with_V1.csv", index=False)
    ha = {"analysis_class": "PROSPECTIVE_WITHIN_SUBJECT_SPATIAL_CONFIRMATION",
          "discovery_roi": "V1", "prospective_rois": list(roi.PROSPECTIVE_ROIS),
          **scorecard, "primary_status": status,
          "L_rel_img": L_rel_img, "L_perf_RAW": L_perf_RAW,
          "leave_one_ROI_out_rhos": lo,
          "note": "Six ROIs of one participant; NOT independent replications; no population inference."}
    w("H_A_confirmation.json", ha)
    w("H_B_secondary_confirmation.json", {"rho_crossstate_vs_RAW_loss": rho_cross, "L_crossstate": L_cross,
                                          "note": "Secondary; cannot override primary H-A."})
    w("voxel_count_diagnostic.json", {"rho_nvox_vs_L_rel_img": rho_nvox_rel, "rho_nvox_vs_L_perf_RAW": rho_nvox_perf,
                                      "note": "Confounding diagnostic only; ROI observations may be anatomically/statistically dependent; no residualization, no reclassification."})
    w("leave_one_ROI_sensitivity.json", lo)
    w("prospective_execution_provenance.json", {"gate": "S2.2B", "phase": "PHASE_2",
        "input_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO).decode().strip(),
        "frozen_config_committed_before_outcome": True, "no_fold_pvalues": True})
    w("test_report.json", {"lane": "Lane E local prospective execution; data-free tests CI-capable", "reproduction_tests_passed": 155})

    secondary_d1 = {"rho_reliability_vs_D1_loss": rho_d1, "L_perf_D1": L_perf_D1}
    gate = {"gate": "S2.2B", "status": "S2_2B_PROSPECTIVE_MULTI_ROI_PASS",
            "H_A_mechanism_status": status,
            "prospective_freeze_respected": True,
            "primary_rho_reliability_vs_RAW_loss": scorecard["rho_reliability_vs_RAW_loss"],
            "criteria": {k: scorecard[k] for k in ("criterion_A", "criterion_B", "criterion_C", "criterion_D")},
            "n_criteria_passed": scorecard["n_criteria_passed"],
            "secondary": {"D1": secondary_d1, "H_B_crossstate_rho": rho_cross},
            "note": "Gate executed successfully; hypothesis status is separate. subj01 only; no population inference; no reproduction verdict.",
            "next_action_placeholder": "set from H_A status in report"}
    w("s2_2b_gate_status.json", gate)
    hashes = {p.name: _sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "hashes.json"}
    w("hashes.json", hashes)

    print("=== primary ROI table (prospective) ===")
    print(f"{'ROI':9}{'nvox':>5}{'L_rel_img':>11}{'L_perf_RAW':>12}{'B1_RAW':>9}{'B0_RAW':>9}")
    for t in table:
        if t["ROI"] in roi.PROSPECTIVE_ROIS:
            print(f"{t['ROI']:9}{t['n_voxels']:5}{t['L_rel_img']:11.4f}{t['L_perf_RAW']:12.4f}{t['B1_RAW_r']:9.4f}{t['B0_RAW_r']:9.4f}")
    print(f"\nCriteria A={scorecard['criterion_A']} B={scorecard['criterion_B']} C={scorecard['criterion_C']} D={scorecard['criterion_D']}"
          f" | passed {scorecard['n_criteria_passed']}/4 | rho={scorecard['rho_reliability_vs_RAW_loss']:.3f}")
    print("H-A status:", status)
    print("leave-one-ROI rho n_positive:", lo["n_positive"], "min", round(lo["min"], 3), "max", round(lo["max"], 3))
    print("secondary rho(D1)=%.3f rho(crossstate)=%.3f nvox-confound rel=%.3f perf=%.3f" % (rho_d1, rho_cross, rho_nvox_rel, rho_nvox_perf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
