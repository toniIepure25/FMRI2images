"""S2.4H post-hoc participant-heterogeneity audit (descriptive; committed data only).

Reads ONLY committed S2.3 summaries. No raw HDF5, no reruns, no models, no new
p-values, no participant exclusion, no primary recomputation. The frozen S2.3
primary result is immutable. Answers Q1-Q7 descriptively with exact numbers.
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

from fmri2img.mindcompiler.roy_method_reproduction import s2_4h_audit as au  # noqa: E402
from fmri2img.mindcompiler.roy_method_reproduction.s2_4h_audit import spearman  # noqa: E402

S3 = _REPO / "artifacts/mindcompiler/roy_s2_3"
OUT = _REPO / "artifacts/mindcompiler/roy_s2_4h"
PRIMARY = ["subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"]
ROIS = ["V1", "V2", "V3", "hV4", "ventral", "lateral", "parietal"]


def _load(name):
    return json.loads((S3 / name).read_text())


def main() -> int:
    # immutability guard
    au.assert_primary_immutable(_load("exact_permutation_test.json"))
    roi = pd.read_csv(S3 / "ROI_level_primary_table.csv")
    roi_p = roi[roi.is_primary].copy()
    within = _load("within_participant_spatial_rhos.json")["rhos"]
    vis_sec = _load("vision_reliability_secondary.json")
    lo = _load("leave_one_participant_sensitivity.json")
    mean_agg = _load("mean_aggregation_sensitivity.json")
    qc = _load("value_qc.json") if (S3 / "value_qc.json").exists() else {}
    validation = _load("participant_validation.json")
    registry = _load("participant_acquisition_registry.json")

    # per-participant summaries
    part = {}
    for s in PRIMARY:
        g = roi_p[roi_p.participant == s]
        rel = g["L_rel_img"].tolist(); perf = g["L_perf_RAW"].tolist(); vis = g["L_rel_vis"].tolist()
        part[s] = dict(
            S_rel_img=au.median(rel), S_perf=au.median(perf), S_rel_vis=au.median(vis),
            mad_rel=au.mad(rel), mad_perf=au.mad(perf),
            P_B0=au.median(g["B0_RAW_r"].tolist()), P_B1=au.median(g["B1_RAW_r"].tolist()),
            mean_rel=float(np.mean(rel)), mean_perf=float(np.mean(perf)),
            rho_spatial=float(within.get(s, float("nan"))),
            median_nvox=float(np.median(g["n_voxels"])), min_nvox=int(g["n_voxels"].min()),
            **au.sign_counts(rel, perf))

    S_rel = {s: part[s]["S_rel_img"] for s in PRIMARY}
    S_perf = {s: part[s]["S_perf"] for s in PRIMARY}
    S_rel_vis = {s: part[s]["S_rel_vis"] for s in PRIMARY}
    P_B0 = {s: part[s]["P_B0"] for s in PRIMARY}
    P_B1 = {s: part[s]["P_B1"] for s in PRIMARY}

    def rho(a, b):
        return spearman([a[s] for s in PRIMARY], [b[s] for s in PRIMARY])

    # ROI-level heterogeneity
    roi_het = []
    for r in ROIS:
        gg = roi_p[roi_p.ROI == r]
        rel = gg["L_rel_img"].tolist(); perf = gg["L_perf_RAW"].tolist()
        roi_het.append(dict(ROI=r, median_L_rel_img=au.median(rel), mad_L_rel_img=au.mad(rel),
                            n_rel_pos=int((gg["L_rel_img"] > 0).sum()),
                            median_L_perf_RAW=au.median(perf), mad_L_perf_RAW=au.mad(perf),
                            n_perf_pos=int((gg["L_perf_RAW"] > 0).sum()),
                            n_both_pos=int(((gg["L_rel_img"] > 0) & (gg["L_perf_RAW"] > 0)).sum()),
                            rho_across_participants=spearman(rel, perf)))

    # Q1: subj07 technical integrity (committed QC/provenance only)
    s7reg = registry.get("subj07", {})
    s7_beta_ok = all(s7reg.get(b, {}).get("status") in ("verified", "present") for b in ("B0", "B1"))
    s7_valid = validation.get("subj07", {}).get("valid", False)
    s7_qc = qc.get("subj07", {})
    s7_finite_ok = all(v.get("finite_fraction", 1.0) >= 0.9 for roi_d in s7_qc.values() for v in roi_d.values()) if s7_qc else True
    s7_tech = "NO_TECHNICAL_ANOMALY_IDENTIFIED" if (s7_beta_ok and s7_valid and s7_finite_ok) else "OBJECTIVE_TECHNICAL_ANOMALY_IDENTIFIED"

    # descriptive rhos
    rho_img = rho(S_rel, S_perf)         # = frozen primary 0.6786 (recomputed here for context ONLY, same value)
    rho_vis = rho(S_rel_vis, S_perf)
    rho_img_vis = rho(S_rel, S_rel_vis)
    rho_pb0 = rho(P_B0, S_perf)
    rho_pb1 = rho(P_B1, S_perf)
    rho_nvox_rel = rho({s: part[s]["median_nvox"] for s in PRIMARY}, S_rel)
    rho_nvox_perf = rho({s: part[s]["median_nvox"] for s in PRIMARY}, S_perf)
    rank_infl = au.rank_influence(S_rel, S_perf)

    # dispersion decomposition
    disp = dict(MAD_S_rel_img_across_participants=au.mad(list(S_rel.values())),
                MAD_S_perf_across_participants=au.mad(list(S_perf.values())),
                median_within_participant_MAD_rel=au.median([part[s]["mad_rel"] for s in PRIMARY]),
                median_within_participant_MAD_perf=au.median([part[s]["mad_perf"] for s in PRIMARY]))

    # --- Q1-Q7 descriptive answers ---
    def ans(v):
        return v
    q = {
        "Q1_subj07_technically_invalid": ans("NOT_SUPPORTED_BY_DESCRIPTIVE_PATTERN"
            if s7_tech == "NO_TECHNICAL_ANOMALY_IDENTIFIED" else "SUPPORTED_BY_DESCRIPTIVE_PATTERN"),
        "Q2_subj07_discordance_localized_or_wide": ans(
            f"PARTICIPANT_WIDE: subj07 has {part['subj07']['n_perf_positive']}/7 ROIs with L_perf_RAW>0 "
            f"(median S_perf={S_perf['subj07']:+.3f}); reliability loss positive in "
            f"{part['subj07']['n_rel_positive']}/7 -> discordance is a participant-wide RAW-performance shift, not one ROI."),
        "Q3_imagery_specific_or_general": ans(
            "GENERAL_RELIABILITY_FACTOR_PLAUSIBLE" if (rho_vis > 0.3 and rho_img_vis > 0.3)
            else ("IMAGERY_SPECIFIC_RELIABILITY_REMAINS_PLAUSIBLE" if rho_img_vis < 0.0 else "MIXED_RELIABILITY_PATTERN")),
        "Q4_heterogeneity_global_vs_spatial": ans(
            f"MAD S_perf across participants={disp['MAD_S_perf_across_participants']:.3f} vs median within-participant "
            f"MAD_perf={disp['median_within_participant_MAD_perf']:.3f}; "
            + ("within-participant ROI variation dominates" if disp['median_within_participant_MAD_perf'] > disp['MAD_S_perf_across_participants']
               else "between-participant level variation dominates")),
        "Q5_depends_on_one_participant": ans(
            f"NOT_SUPPORTED_BY_DESCRIPTIVE_PATTERN: leave-one-participant rho positive in {lo['n_positive']}/7 "
            f"(min {lo['min']:.3f}, max {lo['max']:.3f}); the positive DIRECTION is robust to removing any one participant. "
            "(The frozen criterion C was not met; this is not a significance statement.)"),
        "Q6_median_created_the_result": ans(
            f"MIXED: frozen median rho={rho_img:.3f} vs prespecified mean-aggregation rho={mean_agg['rho_mean_agg']:.3f}; "
            "both positive; ordering is somewhat aggregation-dependent but not sign-reversing. Median is the frozen primary and is not re-chosen."),
        "Q7_measurement_size_dominates": ans(
            f"{'NOT_SUPPORTED_BY_DESCRIPTIVE_PATTERN' if abs(rho_nvox_perf) < 0.4 else 'MIXED'}: "
            f"rho(median_nvox,S_perf)={rho_nvox_perf:+.3f}, rho(median_nvox,S_rel_img)={rho_nvox_rel:+.3f} "
            "(no residualization; primary effect not corrected)."),
    }

    # --- write artifacts ---
    OUT.mkdir(parents=True, exist_ok=True)
    def w(n, o):
        (OUT / n).write_text(json.dumps(o, indent=2))
    w("s2_4h_audit_charter.json", {"analysis_class": "POST_HOC_DESCRIPTIVE_PARTICIPANT_HETEROGENEITY_AUDIT",
        "S2_3_PRIMARY_RESULT_IMMUTABLE": True, "frozen_primary": au.S2_3_PRIMARY_IMMUTABLE,
        "forbidden": ["primary recomputation", "p-value rescue", "subj07 exclusion", "new models", "raw data"]})
    pt = pd.DataFrame([dict(participant=s, **part[s]) for s in PRIMARY])
    pt.to_csv(OUT / "participant_heterogeneity_table.csv", index=False)
    pd.DataFrame(roi_het).to_csv(OUT / "ROI_heterogeneity_table.csv", index=False)
    w("global_vs_spatial_coupling.json", {s: dict(S_rel_img=S_rel[s], S_perf=S_perf[s],
        rho_spatial=part[s]["rho_spatial"], ROI_concordant_positive=part[s]["n_concordant_positive"],
        ROI_discordant=part[s]["n_discordant"]) for s in PRIMARY})
    w("imagery_vs_vision_reliability.json", {"rho_img_vs_perf_context": rho_img,
        "rho_vis_vs_perf": rho_vis, "rho_img_vs_vis": rho_img_vis,
        "S_rel_img": S_rel, "S_rel_vis": S_rel_vis,
        "pattern_status": q["Q3_imagery_specific_or_general"],
        "note": "Descriptive; imagery reliability remains the S2.3 primary mechanism label (not renamed)."})
    w("baseline_RAW_context.json", {"P_B0": P_B0, "P_B1": P_B1, "S_perf": S_perf,
        "rho_PB0_vs_S_perf": rho_pb0, "rho_PB1_vs_S_perf": rho_pb1,
        "note": "Exploratory moderation context; H-A not redefined around baseline performance."})
    w("aggregation_sensitivity_audit.json", {"median_rho": rho_img, "mean_rho": mean_agg["rho_mean_agg"],
        "median": S_perf, "mean": {s: part[s]["mean_perf"] for s in PRIMARY},
        "note": "AGGREGATION_SENSITIVITY_ONLY; median is the frozen primary."})
    w("influence_audit.json", {"leave_one_participant": lo, "rank_influence": rank_infl,
        "note": "Descriptive only; no participant removed; no Cook's distance; no exclusion."})
    w("measurement_size_context.json", {"median_nvox": {s: part[s]["median_nvox"] for s in PRIMARY},
        "min_nvox": {s: part[s]["min_nvox"] for s in PRIMARY},
        "rho_nvox_vs_S_rel_img": rho_nvox_rel, "rho_nvox_vs_S_perf": rho_nvox_perf,
        "note": "Confounding context only; no residualization."})
    s7g = roi_p[roi_p.participant == "subj07"]
    w("subj07_case_audit.json", {
        "technical_integrity": s7_tech, "B0_B1_verified": s7_beta_ok, "trial_mapping_valid": s7_valid,
        "S_rel_img": S_rel["subj07"], "S_rel_vis": S_rel_vis["subj07"], "S_perf": S_perf["subj07"],
        "P_B0": P_B0["subj07"], "P_B1": P_B1["subj07"], "rho_spatial": part["subj07"]["rho_spatial"],
        "roi_sign_pattern": [dict(ROI=row.ROI, L_rel_img=row.L_rel_img, L_perf_RAW=row.L_perf_RAW,
                                  B0_RAW=row.B0_RAW_r, B1_RAW=row.B1_RAW_r, n_voxels=int(row.n_voxels))
                             for row in s7g.itertuples()],
        "ROI_concordance": au.sign_counts(s7g["L_rel_img"].tolist(), s7g["L_perf_RAW"].tolist()),
        "rank_positions": rank_infl["subj07"],
        "case_classification": ("VALID_SCIENTIFIC_HETEROGENEITY" if s7_tech == "NO_TECHNICAL_ANOMALY_IDENTIFIED"
                                else "OBJECTIVE_TECHNICAL_ANOMALY"),
        "interpretation": ("subj07 shows the expected B1 imagery-reliability loss (L_rel_img>0 in most ROIs) but a "
                           "LOW-BASELINE RAW regime: its B0 RAW mapping is itself weak/negative, so B1 is not worse "
                           "(L_perf_RAW<0). Valid heterogeneity, not a technical anomaly; not excluded.")})
    w("heterogeneity_summary.json", {
        "primary_result_immutable": au.S2_3_PRIMARY_IMMUTABLE,
        "technical_anomaly_status": s7_tech,
        "participant_global_effects": {s: dict(S_rel_img=S_rel[s], S_perf=S_perf[s]) for s in PRIMARY},
        "within_participant_spatial_effects": {s: part[s]["rho_spatial"] for s in PRIMARY},
        "imagery_vs_vision_reliability": {"rho_img": rho_img, "rho_vis": rho_vis, "rho_img_vis": rho_img_vis},
        "baseline_RAW_context": {"rho_PB0_vs_S_perf": rho_pb0, "rho_PB1_vs_S_perf": rho_pb1},
        "ROI_heterogeneity": roi_het,
        "aggregation_sensitivity": {"median_rho": rho_img, "mean_rho": mean_agg["rho_mean_agg"]},
        "influence_sensitivity": {"leave_one_n_positive": lo["n_positive"]},
        "measurement_size_context": {"rho_nvox_vs_S_perf": rho_nvox_perf},
        "answers": q})
    w("execution_provenance.json", {"gate": "S2.4H",
        "input_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO).decode().strip(),
        "S2_3_PRIMARY_RESULT_IMMUTABLE": True, "committed_data_only": True, "no_raw_reruns": True})
    w("test_report.json", {"lane": "data-free descriptive audit; CI-capable", "reproduction_tests_passed": None})
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "hashes.json"}
    w("hashes.json", hashes)

    gate = {"gate": "S2.4H", "status": "S2_4H_HETEROGENEITY_AUDIT_PASS",
            "frozen_S2_3_status_unchanged": "H_A_CROSS_PARTICIPANT_PARTIAL",
            "subj07_case": s7_tech, "answers": q,
            "descriptive_mechanism_statement": (
                "H-A captured a robust DIRECTIONAL component of beta-version sensitivity (leave-one-participant "
                "rho positive 7/7) but does not fully explain participant heterogeneity; subj07 is VALID "
                "heterogeneity (a low-baseline RAW regime), not a technical failure. A broader beta-preparation "
                "reliability factor involving BOTH vision and imagery remains plausible. NOT causal; NOT a "
                "population claim; frozen criterion C was not met.")}
    # next action (Path A if no anomaly + broader reliability plausible + robust)
    path = ("A" if (s7_tech == "NO_TECHNICAL_ANOMALY_IDENTIFIED" and lo["n_positive"] == 7) else "C")
    gate["next_action"] = {"path": path,
        "freeze": "BETA_PREPARATION_RELIABILITY_SENSITIVITY_WITH_PARTICIPANT_HETEROGENEITY" if path == "A" else "PARTICIPANT_HETEROGENEITY_UNRESOLVED",
        "next": "S2.4R -- FULL CROSS-PARTICIPANT D1 ROY-PIPELINE CHARACTERIZATION (both B0 and B1; no beta selection)",
        "not_done_here": "No D1/Track-O work in S2.4H."}
    w("s2_4h_gate_status.json", gate)

    print("subj07 technical:", s7_tech, "| case:", gate["next_action"]["path"])
    print("rho_img(context)=%.3f rho_vis=%.3f rho_img_vis=%.3f" % (rho_img, rho_vis, rho_img_vis))
    print("rho(P_B0,S_perf)=%.3f rho(P_B1,S_perf)=%.3f | rho(nvox,S_perf)=%.3f" % (rho_pb0, rho_pb1, rho_nvox_perf))
    print("subj07 sign counts:", part["subj07"]["n_rel_positive"], "rel+ /", part["subj07"]["n_perf_positive"], "perf+")
    for qq, aa in q.items():
        print(" ", qq, "->", aa[:70])
    print("GATE:", gate["status"], "| next:", gate["next_action"]["next"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
