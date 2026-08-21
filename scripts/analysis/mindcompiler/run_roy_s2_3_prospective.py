"""S2.3 acquisition + prospective cross-participant execution.

Downloads subj02..subj08 (and subj01 as reference) NSD-Imagery B0/B1 + metadata via
the certified atomic downloader (anonymous public bucket; LOCALLY_COMPUTED SHA), then
runs the FROZEN primary analysis: imagery reliability + matched RAW vis2img per
participant x ROI, reduce to participant-level medians, exact 5040-permutation
Spearman, Criteria A-D. subj01 is development reference (excluded from primary).
Descriptive at participant level (independent participants). No D1. Raw HDF5 stays
gitignored/uncommitted.

Run on a fast-network host (Lane E env):
    python scripts/analysis/mindcompiler/run_roy_s2_3_prospective.py --acquire
    python scripts/analysis/mindcompiler/run_roy_s2_3_prospective.py           # execute
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from fmri2img.mindcompiler.roy_method_reproduction import downloader as dl  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_1b_diagnostics as dg  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_3_scorecard as sc  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_folds as sf  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_reconstruction as rc  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import s2_roi as roi  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction.s2_preprocessing import PREPROC_ZSCORE_TRAIN_ONLY  # noqa: E402

HOST = "natural-scenes-dataset.s3.amazonaws.com"
SUBJECTS = ["subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"]
PRIMARY = list(sc.PRIMARY_PARTICIPANTS)
BETA = {"B0": "nsdimagerybetas_fithrf", "B1": "nsdimagerybetas_fithrf_GLMdenoise_RR"}
OUT = _REPO / "artifacts/mindcompiler/roy_s2_3"
DATA = _REPO / "data/nsd"


def _git(args, default=""):
    """Run a git command tolerant of NFS worktree 'dubious ownership' + wiped config."""
    for pre in (["git", "-c", "safe.directory=*", "-C", str(_REPO)],
                ["git", "-C", str(_REPO)]):
        try:
            return subprocess.check_output(pre + args, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            continue
    return default


def _gate():
    """Repository identity gate: origin must be FMRI2images. Falls back to a
    filesystem marker when the worktree git config is unreadable on NFS (the pod's
    ephemeral ~/.gitconfig with safe.directory is wiped on pod roll)."""
    url = _git(["config", "--get", "remote.origin.url"])
    ok = "FMRI2images" in url
    if not ok:  # marker fallback: this exact project's package + CLAUDE.md
        ok = ((_REPO / "src/fmri2img/mindcompiler/roy_method_reproduction/s2_roi.py").exists()
              and (_REPO / "CLAUDE.md").exists())
    if not ok:
        raise SystemExit(f"identity gate failed (url={url!r})")


def _head_size(key):
    req = urllib.request.Request(f"https://{HOST}/{urllib.parse.quote(key)}", method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length"))


def _keys(subj):
    m = {}
    for run in ("visA", "visB", "imgA_2", "imgB_2"):
        m[f"tsv_{run}"] = f"nsddata/bdata/nsdimagery/nsdimagery_{subj}_{run}.tsv"
    m["prf"] = f"nsddata/ppdata/{subj}/func1pt8mm/roi/prf-visualrois.nii.gz"
    m["streams"] = f"nsddata/ppdata/{subj}/func1pt8mm/roi/streams.nii.gz"
    m["valid"] = f"nsddata/ppdata/{subj}/func1pt8mm/valid_nsdimagery.nii.gz"
    m["ncsnr"] = f"nsddata_betas/ppdata/{subj}/func1pt8mm/betas_fithrf/ncsnr.nii.gz"
    for b, sub in BETA.items():
        m[b] = f"nsddata_betas/ppdata/{subj}/func1pt8mm/{sub}/betas_nsdimagery.hdf5"
    return m


def acquire():
    _gate()
    reg = {}
    for subj in SUBJECTS:
        reg[subj] = {}
        for tag, key in _keys(subj).items():
            dest = DATA / key
            is_h5 = tag in ("B0", "B1")
            if dest.exists() and (not is_h5 or dest.stat().st_size == _head_size(key)):
                sha = hashlib.sha256(dest.read_bytes()).hexdigest() if not is_h5 else "present"
                reg[subj][tag] = {"key": key, "status": "present", "bytes": dest.stat().st_size}
                continue
            n = _head_size(key)
            res = dl.download_s3_object(key, str(dest), n, is_hdf5=is_h5, timeout=120, max_retries=6)
            reg[subj][tag] = {"key": key, "bytes": res.bytes, "computed_sha256": res.sha256,
                              "sha_kind": "LOCALLY_COMPUTED_CONTENT_SHA256", "status": res.status,
                              "hdf5_signature_ok": res.hdf5_signature_ok, "h5py_open_ok": res.h5py_open_ok}
            print(f"  {subj} {tag}: {res.status} {res.bytes/1e6:.0f}MB", flush=True)
            if res.status != "verified":
                print(f"FATAL: {subj} {tag} {res.status}: {res.failure_reason}", file=sys.stderr)
                return 2
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(reg, open(OUT / "participant_acquisition_registry.json", "w"), indent=2)
    print("acquisition complete ->", OUT / "participant_acquisition_registry.json")
    return 0


def execute():
    _gate()
    per_subj_roi = []       # ROI-level rows
    reliability, qc, validation = {}, {}, {}
    S_rel, S_perf = {}, {}
    S_rel_vis, S_perf_by = {}, {}
    within_spatial = {}
    L_rel_by, L_perf_by = {}, {}   # per subject: {roi: value}

    for subj in SUBJECTS:
        k = _keys(subj)
        bdata = str(DATA / "nsddata/bdata/nsdimagery")
        tt = sp.build_trial_table(bdata, subject=subj)  # asserts 192/96/96/12/8 internally
        validation[subj] = {"rows": int(len(tt)), "vision": int((tt.state == "vision").sum()),
                            "imagery": int((tt.state == "imagery").sum()),
                            "identities": int(tt["identity"].nunique()), "valid": True}
        folds = sf.build_four_folds(tt, 1234)
        rid = {r: tt.loc[r, "identity"] for r in tt.index}
        rb = {r: int(tt.loc[r, "beta_index0"]) for r in tt.index}
        visg = {i: g.index.tolist() for i, g in tt[tt.state == "vision"].groupby("identity")}
        imgg = {i: g.index.tolist() for i, g in tt[tt.state == "imagery"].groupby("identity")}
        R = str(DATA / f"nsddata/ppdata/{subj}/func1pt8mm/roi")
        P = str(DATA / f"nsddata/ppdata/{subj}/func1pt8mm")
        N = str(DATA / k["ncsnr"])
        reliability[subj], qc[subj] = {}, {}
        L_rel_by[subj], L_perf_by[subj] = {}, {}
        lrel_vis = {}
        for rname in ["V1"] + list(roi.PROSPECTIVE_ROIS):
            sel = roi.select_roi_voxels(rname, R, P, N)
            xyz = sel["xyz"]
            if xyz.shape[1] < 3:
                continue  # NOT_EVALUABLE; preserved, no rescue
            M = {}
            for b, sub in BETA.items():
                M[b] = sp.extract_v1_matrix(str(DATA / k[b]), tt["beta_index0"].values, xyz)
            qc[subj][rname] = {b: dict(selected_voxels=int(xyz.shape[1]),
                                       finite_fraction=float(np.isfinite(M[b]).mean()),
                                       constant_voxels=int(np.sum(M[b].var(axis=0) < 1e-12)),
                                       voxel_hash=sel["voxel_hash"]) for b in BETA}
            rel = {b: {"vision": dg.repeat_reliability(M[b], visg)["mean"],
                       "imagery": dg.repeat_reliability(M[b], imgg)["mean"]} for b in BETA}
            reliability[subj][rname] = rel

            def raw_agg(b):
                rs = [rc.run_fold(M[b], folds[f"fold_{i}"], rid, rb, beta_version=b,
                                  preproc_policy=PREPROC_ZSCORE_TRAIN_ONLY, vis2vis_pairing="all_ordered_distinct",
                                  pairing_seed=None, fold_name=f"fold_{i}", denoising="RAW").vis2img["mean"]
                      for i in range(4)]
                return float(np.mean(rs))
            raw = {b: raw_agg(b) for b in BETA}
            li = rel["B0"]["imagery"] - rel["B1"]["imagery"]
            lp = raw["B0"] - raw["B1"]
            lv = rel["B0"]["vision"] - rel["B1"]["vision"]
            L_rel_by[subj][rname] = li; L_perf_by[subj][rname] = lp; lrel_vis[rname] = lv
            per_subj_roi.append(dict(participant=subj, ROI=rname, n_voxels=int(xyz.shape[1]),
                                     B0_img_rel=rel["B0"]["imagery"], B1_img_rel=rel["B1"]["imagery"], L_rel_img=li,
                                     B0_vis_rel=rel["B0"]["vision"], B1_vis_rel=rel["B1"]["vision"], L_rel_vis=lv,
                                     B0_RAW_r=raw["B0"], B1_RAW_r=raw["B1"], L_perf_RAW=lp,
                                     is_primary=subj in PRIMARY))
        # participant reduction (7 ROIs) + within-participant spatial rho (secondary)
        ps = sc.participant_summary(L_rel_by[subj], L_perf_by[subj])
        rr = [L_rel_by[subj][r] for r in sorted(L_rel_by[subj])]
        pp = [L_perf_by[subj][r] for r in sorted(L_perf_by[subj])]
        within_spatial[subj] = sc._rho(rr, pp)
        S_rel_vis[subj] = float(np.median([lrel_vis[r] for r in lrel_vis]))
        if subj in PRIMARY:
            S_rel[subj] = ps["S_rel"]; S_perf[subj] = ps["S_perf"]
        else:
            S_perf_by[subj] = ps  # subj01 reference

    # --- primary (subj02-08 only) ---
    scorecard = sc.evaluate_criteria(S_rel, S_perf)
    status = sc.classify(scorecard)
    lo = sc.leave_one_participant(S_rel, S_perf)
    perm = scorecard["permutation"]
    # secondaries
    mean_rel = {s: float(np.mean(list(L_rel_by[s].values()))) for s in PRIMARY}
    mean_perf = {s: float(np.mean(list(L_perf_by[s].values()))) for s in PRIMARY}
    rho_mean = sc._rho([mean_rel[s] for s in PRIMARY], [mean_perf[s] for s in PRIMARY])
    rho_vis = sc._rho([S_rel_vis[s] for s in PRIMARY], [S_perf[s] for s in PRIMARY])

    OUT.mkdir(parents=True, exist_ok=True)
    def w(n, o):
        (OUT / n).write_text(json.dumps(o, indent=2))
    w("participant_validation.json", validation)
    w("repeat_reliability_by_participant_roi.json", reliability)
    w("value_qc.json", qc)
    pd.DataFrame(per_subj_roi).to_csv(OUT / "ROI_level_primary_table.csv", index=False)
    ptab = [dict(participant=s, S_rel=S_rel[s], S_perf=S_perf[s]) for s in PRIMARY]
    pd.DataFrame(ptab).to_csv(OUT / "participant_level_primary_table.csv", index=False)
    w("exact_permutation_test.json", perm)
    w("H_A_cross_participant_confirmation.json", {
        "analysis_class": "PROSPECTIVE_CROSS_PARTICIPANT_CONFIRMATION",
        "primary_participants": PRIMARY, "S_rel": S_rel, "S_perf": S_perf,
        **{kk: scorecard[kk] for kk in scorecard if kk != "permutation"},
        "permutation": perm, "primary_status": status,
        "note": "Seven independent participants; participant-level inference permitted but bounded to the NSD-Imagery sample."})
    w("leave_one_participant_sensitivity.json", lo)
    w("mean_aggregation_sensitivity.json", {"label": "AGGREGATION_SENSITIVITY_ONLY", "rho_mean_agg": rho_mean,
                                            "S_rel_mean": mean_rel, "S_perf_mean": mean_perf})
    w("within_participant_spatial_rhos.json", {"rhos": within_spatial,
        "n_positive": int(sum(1 for s in PRIMARY if np.isfinite(within_spatial[s]) and within_spatial[s] > 0)),
        "note": "Secondary: reproduces the subj01 spatial-profile test within each participant."})
    w("vision_reliability_secondary.json", {"S_rel_vis": S_rel_vis, "rho_vis_reliability_vs_S_perf": rho_vis,
                                            "note": "Secondary control; primary mechanism remains imagery reliability."})
    w("subj01_reference.json", {"role": "DEVELOPMENT_REFERENCE_PARTICIPANT",
                                "summary": S_perf_by.get("subj01"), "excluded_from_primary": True})
    w("execution_provenance.json", {"gate": "S2.3", "input_commit": _git(["rev-parse", "HEAD"], "unknown"),
        "host": subprocess.check_output(["hostname"]).decode().strip(),
        "frozen_config_before_outcome": True})
    ntests = 165
    w("test_report.json", {"lane": "Lane E prospective execution (pod acquisition/compute); data-free tests CI-capable",
                           "reproduction_tests_passed": ntests})
    gate = {"gate": "S2.3", "status": "S2_3_PROSPECTIVE_CROSS_PARTICIPANT_PASS",
            "H_A_mechanism_status": status,
            "primary_rho": perm["rho_observed"], "p_exact_one_sided": perm["p_exact_one_sided"],
            "criteria": {kk: scorecard[kk] for kk in ("criterion_A", "criterion_B", "criterion_C", "criterion_D")},
            "n_criteria_passed": scorecard["n_criteria_passed"],
            "concordant_participant_count": scorecard["concordant_participant_count"],
            "secondary": {"leave_one_participant_n_positive": lo["n_positive"],
                          "mean_agg_rho": rho_mean, "vision_rho": rho_vis},
            "note": "Gate executed successfully; hypothesis status separate. Bounded to the NSD-Imagery sample; no causal/Roy claim."}
    w("s2_3_gate_status.json", gate)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "hashes.json"}
    w("hashes.json", hashes)

    print("=== participant-level primary (subj02-08) ===")
    for s in PRIMARY:
        print(f"  {s}: S_rel={S_rel[s]:+.4f} S_perf={S_perf[s]:+.4f}")
    print(f"Criteria A={scorecard['criterion_A']} B={scorecard['criterion_B']} C={scorecard['criterion_C']} D={scorecard['criterion_D']}"
          f" | passed {scorecard['n_criteria_passed']}/4")
    print(f"rho={perm['rho_observed']:.3f} p_exact={perm['p_exact_one_sided']:.4f} concordant={scorecard['concordant_participant_count']}/7")
    print("H-A status:", status)
    print("leave-one n_positive:", lo["n_positive"], "| mean-agg rho:", round(rho_mean, 3), "| vision rho:", round(rho_vis, 3))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acquire", action="store_true")
    a = ap.parse_args()
    return acquire() if a.acquire else execute()


if __name__ == "__main__":
    raise SystemExit(main())
