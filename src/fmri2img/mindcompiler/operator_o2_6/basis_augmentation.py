"""O2.6 Target-State Basis Augmentation Frontier (frozen config f4e94517). DIAGNOSTIC ONLY: SVD augmentation
of the OUTSIDE-perception-support target residual; NO ridge/affine/CCA/RRR/NN/new SRM/new W_target/higher K.

Consumes ONLY sealed O2.3A-RD/O2.4R state + the replay-certified native-residual extension (delta_native for
the 10 outer-training identities per subject x ROI x fold). Reuses the committed geometry (orthogonal
Procrustes = the sealed O2.4R within-support estimator; native_projector; frozen null RNG).

Because range(P_IN) is in col(W_target) and B_OUT is orthogonal to col(W_target), the augmented retention
decomposes exactly: R_AUG = R_BASE + ||B_OUT^T delta||^2/||delta||^2 (verified in tests via explicit P_AUG).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ALL = [f"subj0{i}" for i in range(1, 9)]
ROIS_PRIMARY = ["ventral", "lateral"]
M_GRID = [2, 4, 6, 8, 10]
D_GRID = [1, 2, 4, 6, 8, 10]
EXPECTED_SUBSETS = {2: 25, 4: 100, 6: 100, 8: 25, 10: 1}   # frozen balanced-subset counts per M
N_FOLDS = 6
N_NULL = 100
EPS = 1e-12
_G = None


def _geom(repo: Path):
    global _G
    if _G is None:
        sys.path.insert(0, str(repo / "src"))
        from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
        _G = G
    return _G


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _canon(U):
    U = np.asarray(U, np.float64).copy()
    for k in range(U.shape[1]):
        j = int(np.argmax(np.abs(U[:, k])))
        if U[j, k] < 0:
            U[:, k] *= -1.0
    return U


# --- outside-support helpers -------------------------------------------------
def outside_component(delta, W):
    d = np.asarray(delta, np.float64)
    return d - W @ (W.T @ d)


def outside_basis(D_out, d):
    """Top-d left singular vectors of the (V x M) outside-support residual matrix; canonical signs.
    Returns (V x d) or None if numerical rank < d (RANK_INSUFFICIENT)."""
    U, s, _ = np.linalg.svd(np.asarray(D_out, np.float64), full_matrices=False)
    rank = int(np.sum(s > (s.max() * 1e-9))) if s.size else 0
    if rank < d:
        return None
    return _canon(U[:, :d])


def null_basis(seed_text, W, V, d, G):
    rng = np.random.Generator(np.random.PCG64(G.seed_uint64(seed_text)))
    Gm = rng.standard_normal((V, d))
    Gm = Gm - W @ (W.T @ Gm)                                   # project out perception support
    Q, _ = np.linalg.qr(Gm)
    return _canon(Q[:, :d])


def _proj_frac(B, delta):
    """||B^T delta||^2 / ||delta||^2."""
    d2 = float(delta @ delta)
    if d2 <= 0:
        return 0.0
    return float((B.T @ delta) @ (B.T @ delta) / d2)


def balanced_subsets(simple_ids, nat_ids, M):
    if M == 0:
        return [tuple()]
    h = M // 2
    out = []
    for cs in itertools.combinations(sorted(simple_ids), h):
        for cn in itertools.combinations(sorted(nat_ids), h):
            out.append(tuple(cs) + tuple(cn))
    return out


# --- per (subject, ROI) frontier ---------------------------------------------
def frontier_subject_roi(s, roi, cells, dnat, rd, G, ext_bound):
    """Returns per (M,d) fold-averaged participant metrics + terminal per-fold E_AUG list."""
    per_md = {(M, d): {"folds": [], "fold_insuf": 0, "subset_insuf": 0}
              for M in M_GRID for d in D_GRID if d <= M}
    term_E_by_fold = []                                        # E_AUG at (M=10,d=10) per fold
    R0_folds, Rnat_folds = [], []
    for f in range(N_FOLDS):
        cell = cells[f]
        W = np.asarray(cell["W_target"], np.float64); U_res = np.asarray(cell["U_res"], np.float64)
        X = np.asarray(cell["X"], np.float64); Z = np.asarray(cell["Z"], np.float64)
        K = int(cell["K"]); V = W.shape[0]
        deltas_test = [np.asarray(x, np.float64) for x in list(cell["deltas_test"])]
        simple_ids = list(cell["simple_ids"]); nat_ids = list(cell["nat_ids"])
        D_train = np.asarray(dnat[f]["delta_native"], np.float64)           # (10 x V) outer-training residuals
        R0_folds.append(float(cell["R_CAL0"])); Rnat_folds.append(float(rd["per_target"][roi][s][f]["R_ORACLE"]))
        # outside components (10 train) + certification of orthogonality to W
        Dout = np.stack([outside_component(D_train[k], W) for k in range(10)])   # (10 x V)
        for k in range(10):
            leak = float(np.linalg.norm(W.T @ Dout[k])) / max(1.0, float(np.linalg.norm(D_train[k])))
            ext_bound["max_leak"] = max(ext_bound["max_leak"], leak)
            ext_bound["rows"].append([s, roi, f, k, leak])
        # dimension spectrum of the 10 outside-support training residuals (Part O)
        sv = np.linalg.svd(Dout.T, compute_uv=False)
        e = sv ** 2; cum = np.cumsum(e) / max(e.sum(), 1e-30)
        stable = float(e.sum() / max(e[0], 1e-30)); effr = float((e.sum() ** 2) / max((e ** 2).sum(), 1e-30))
        ext_bound["spectrum"].append([s, roi, f, ";".join("%.5f" % x for x in sv),
                                      ";".join("%.4f" % x for x in cum), effr, stable])
        # all-training outside oracle per d
        oracle_B = {d: outside_basis(Dout.T, d) for d in D_GRID}
        for M in M_GRID:
            subs = balanced_subsets(simple_ids, nat_ids, M)
            n_expected = EXPECTED_SUBSETS[M]
            assert len(subs) == n_expected, "subset count %d != expected %d for M=%d" % (len(subs), n_expected, M)
            # per-d accumulators for THIS fold (only complete-subset cells are used)
            acc = {d: {"aug": [], "base": [], "cond": [], "null": [], "train": [], "insuf": 0}
                   for d in D_GRID if d <= M}
            for si, C in enumerate(subs):
                Ccols = list(C)
                Q = G.orthogonal_procrustes(X[Ccols], Z[Ccols])
                P_IN = G.native_projector(Q, U_res, W)
                R_base = float(np.mean([G.retention(P_IN, dt) for dt in deltas_test]))
                Dout_C = Dout[Ccols].T                                       # (V x M)
                for d in [x for x in D_GRID if x <= M]:
                    B = outside_basis(Dout_C, d)
                    if B is None:
                        acc[d]["insuf"] += 1                                # RANK_INSUFFICIENT subset -> do NOT drop
                        continue
                    r_aug = R_base + float(np.mean([_proj_frac(B, dt) for dt in deltas_test]))
                    Bor = oracle_B[d]                                        # all-training conditional oracle (same P_IN)
                    r_cond = R_base + (float(np.mean([_proj_frac(Bor, dt) for dt in deltas_test])) if Bor is not None else 0.0)
                    nvals = []                                              # matched random-outside null (100)
                    for it in range(N_NULL):
                        seed = "O2.6|%s|%s|%d|%d|%d|%d|%d" % (s, roi, f, M, si, d, it)
                        Bn = null_basis(seed, W, V, d, G)
                        nvals.append(R_base + float(np.mean([_proj_frac(Bn, dt) for dt in deltas_test])))
                    tr = [float(G.retention(P_IN, D_train[k]) + _proj_frac(B, D_train[k]))
                          for k in Ccols if float(D_train[k] @ D_train[k]) > 0]
                    acc[d]["aug"].append(r_aug); acc[d]["base"].append(R_base); acc[d]["cond"].append(r_cond)
                    acc[d]["null"].append(float(np.mean(nvals))); acc[d]["train"].append(float(np.mean(tr)) if tr else np.nan)
            # FIX 4: a fold (M,d) is evaluable ONLY if EVERY enumerated balanced subset produced a valid basis
            for d in acc:
                a = acc[d]; complete = (a["insuf"] == 0 and len(a["aug"]) == n_expected)
                per_md[(M, d)]["subset_insuf"] += a["insuf"]
                if complete:
                    per_md[(M, d)]["folds"].append({"R_AUG": float(np.mean(a["aug"])), "R_BASE": float(np.mean(a["base"])),
                        "R_COND": float(np.mean(a["cond"])), "R_NULL": float(np.mean(a["null"])),
                        "TRAIN": float(np.nanmean(a["train"]))})
                    if M == 10 and d == 10:
                        term_E_by_fold.append(float(np.mean(a["aug"]) - np.mean(a["null"])))
                else:
                    per_md[(M, d)]["fold_insuf"] += 1
    R0 = float(np.mean(R0_folds)); Rnat = float(np.mean(Rnat_folds))
    out = {"R0": R0, "R_native": Rnat, "term_E_folds": term_E_by_fold, "term_evaluable": len(term_E_by_fold) == N_FOLDS, "md": {}}
    for (M, d), v in per_md.items():
        folds = v["folds"]
        # FIX 5: participant (M,d) evaluable ONLY if ALL 6 outer folds are evaluable
        if len(folds) != N_FOLDS:
            out["md"]["%d_%d" % (M, d)] = {"evaluable": False, "reason": "PARTICIPANT_MD_RANK_INSUFFICIENT",
                                           "n_folds_evaluable": len(folds), "fold_insuf": v["fold_insuf"],
                                           "subset_insuf": v["subset_insuf"]}
            continue
        R_AUG = float(np.mean([x["R_AUG"] for x in folds])); R_BASE = float(np.mean([x["R_BASE"] for x in folds]))
        R_COND = float(np.mean([x["R_COND"] for x in folds])); R_NULL = float(np.mean([x["R_NULL"] for x in folds]))
        TRAIN = float(np.nanmean([x["TRAIN"] for x in folds]))
        E_AUG = R_AUG - R_NULL
        den_cond = R_COND - R_BASE
        obr = (R_AUG - R_BASE) / max(den_cond, EPS)
        tot = (R_AUG - R0) / max(Rnat - R0, EPS)
        out["md"]["%d_%d" % (M, d)] = {
            "evaluable": True, "R_BASE": R_BASE, "R_AUG": R_AUG, "DELTA_OUT": R_AUG - R_BASE, "R_NULL_mean": R_NULL,
            "E_AUG": E_AUG, "R_COND": R_COND, "R_COND_MINUS_R_BASE": den_cond,
            "OUT_BASIS_RECOVERY": float(np.clip(obr, 0, 1)), "OUT_BASIS_RECOVERY_unclipped": obr,
            "OUT_BASIS_RECOVERY_flag": ("DENOMINATOR_DEGENERATE" if den_cond <= 1e-12 else "ok"),   # FIX 8
            "TOTAL_RECOVERY": float(np.clip(tot, 0, 1)), "TOTAL_RECOVERY_unclipped": tot,
            "TRAIN_retention": TRAIN, "subset_insuf": v["subset_insuf"]}
    return out


# --- aggregation / inference / status ----------------------------------------
def _median(x):
    return float(np.median(x)) if len(x) else float("nan")


def aggregate(per_subj, G):
    from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
    rois = list(next(iter(per_subj.values())).keys())
    roi_status = {}; frontier_rows = []; term = {}
    for roi in rois:
        # group medians per (M,d) -- a group cell is COMPLETE only if all 8 participants are evaluable (FIX 5)
        md_group = {}
        for M in M_GRID:
            for d in D_GRID:
                if d > M:
                    continue
                key = "%d_%d" % (M, d)
                ev = [per_subj[s][roi]["md"][key] for s in ALL if per_subj[s][roi]["md"].get(key, {}).get("evaluable")]
                n_ev = len(ev)
                complete = (n_ev == 8)
                tots = [x["TOTAL_RECOVERY"] for x in ev]; eaug = [x["E_AUG"] for x in ev]; obr = [x["OUT_BASIS_RECOVERY"] for x in ev]
                g = {"M": M, "d": d, "n_evaluable_participants": n_ev, "complete": complete,
                     "median_TOTAL": _median(tots), "n_TOTAL_ge0.5": sum(t >= 0.5 for t in tots),
                     "median_E_AUG": _median(eaug), "n_E_pos": sum(e > 0 for e in eaug),
                     "median_OUT_BASIS_RECOVERY": _median(obr)}
                md_group[key] = g
                frontier_rows.append([roi, M, d, n_ev, g["median_TOTAL"], g["n_TOTAL_ge0.5"], g["median_E_AUG"],
                                      g["n_E_pos"], g["median_OUT_BASIS_RECOVERY"]])
        # terminal (M=10,d=10): valid ONLY if all 8 participants evaluable (6/6 folds each) (FIX 6)
        term_ok = all(per_subj[s][roi].get("term_evaluable") and per_subj[s][roi]["md"].get("10_10", {}).get("evaluable")
                      for s in ALL)
        if term_ok:
            eterm = [float(np.mean(per_subj[s][roi]["term_E_folds"])) for s in ALL]
            tot_term = [per_subj[s][roi]["md"]["10_10"]["TOTAL_RECOVERY"] for s in ALL]
            term[roi] = {"terminal_complete": True, "E_AUG": eterm, "median_E_AUG": _median(eterm),
                         "n_E_pos": sum(e > 0 for e in eterm), "signflip_p": float(F.signflip_p_onesided(eterm)),
                         "median_TOTAL": _median(tot_term), "n_TOTAL_ge0.5": sum(t >= 0.5 for t in tot_term)}
        else:
            term[roi] = {"terminal_complete": False, "signflip_p": 1.0, "median_E_AUG": float("nan"),
                         "n_E_pos": 0, "median_TOTAL": float("nan"), "n_TOTAL_ge0.5": 0}
        roi_status[roi] = {"md_group": md_group}
    # Holm across exactly the 2 terminal tests
    holm = F.holm({roi: term[roi]["signflip_p"] for roi in rois}, alpha=0.05)
    for roi in rois:
        t = term[roi]
        feasible = (t["terminal_complete"] and t["median_E_AUG"] > 0 and t["n_E_pos"] >= 6 and bool(holm[roi])
                    and t["median_TOTAL"] >= 0.50 and t["n_TOTAL_ge0.5"] >= 6)
        t["holm_reject"] = bool(holm[roi]); t["terminal_feasible"] = feasible
        # D50 at M=10 -- only from COMPLETE cells (all 8 participants evaluable) (FIX 7)
        def _cell_ok(key):
            g = roi_status[roi]["md_group"].get(key)
            return bool(g and g["complete"] and g["median_TOTAL"] >= 0.50 and g["n_TOTAL_ge0.5"] >= 6
                        and g["median_E_AUG"] > 0 and g["n_E_pos"] >= 6)
        d50 = "NOT_REACHED"
        if feasible:
            for d in D_GRID:
                if _cell_ok("%d_%d" % (10, d)):
                    d50 = d; break
        m50 = "NOT_REACHED"
        if isinstance(d50, int):
            for M in M_GRID:
                if _cell_ok("%d_%d" % (M, d50)):
                    m50 = M; break
        # ROI status
        if not t["terminal_complete"]:
            st = "TARGET_BASIS_AUGMENTATION_INCONCLUSIVE"        # FIX 6: terminal rank-insufficient -> inconclusive
        elif not feasible:
            st = "TARGET_BASIS_AUGMENTATION_NOT_RECOVERED_AT_MAX_BUDGET"
        elif d50 == "NOT_REACHED":
            st = "TARGET_BASIS_AUGMENTATION_INCONCLUSIVE"
        elif d50 <= 2:
            st = "LOW_DIMENSION_TARGET_BASIS_AUGMENTATION"
        elif d50 in (4, 6):
            st = "MODERATE_DIMENSION_TARGET_BASIS_AUGMENTATION"
        else:
            st = "HIGH_DIMENSION_TARGET_BASIS_AUGMENTATION"
        roi_status[roi].update({"terminal": t, "D50": d50, "M50": m50, "status": st})
    # program status
    sv, sl = roi_status["ventral"]["status"], roi_status["lateral"]["status"]
    d50v, d50l = roi_status["ventral"]["D50"], roi_status["lateral"]["D50"]
    m50v, m50l = roi_status["ventral"]["M50"], roi_status["lateral"]["M50"]
    notrec = "TARGET_BASIS_AUGMENTATION_NOT_RECOVERED_AT_MAX_BUDGET"
    if sv == sl == notrec:
        prog = "TARGET_STATE_BASIS_NOT_RECOVERED_WITH_10_IDENTITIES_10_DIMS"
    elif "TARGET_BASIS_AUGMENTATION_INCONCLUSIVE" in (sv, sl):
        prog = "O2_6_TARGET_BASIS_INCONCLUSIVE"
    elif isinstance(d50v, int) and isinstance(d50l, int) and d50v <= 2 and d50l <= 2 and isinstance(m50v, int) and isinstance(m50l, int):
        prog = "TARGET_STATE_MISSING_BASIS_LOW_COMPLEXITY"
    elif isinstance(d50v, int) and isinstance(d50l, int):
        prog = "TARGET_STATE_MISSING_BASIS_RECOVERABLE"
    else:
        prog = "TARGET_STATE_MISSING_BASIS_MULTIREGIME"
    return roi_status, term, frontier_rows, prog


def verify_ext_file(p, mm, sealed_cell, vdim_sealed, vh_sealed):
    """Per-file check. Returns None on success, else a failure code (FIX 1 hash / FIX 2 identity alignment)."""
    if not p.exists() or mm is None:
        return "O2_6_NATIVE_RESIDUAL_EXTENSION_HASH_FAILURE"
    if _sha256_file(p) != mm["sha256"] or p.stat().st_size != mm["bytes"]:
        return "O2_6_NATIVE_RESIDUAL_EXTENSION_HASH_FAILURE"
    z = dict(np.load(p, allow_pickle=True))
    D = np.asarray(z["delta_native"])
    if list(D.shape) != list(mm["shape"]) or D.shape[0] != 10 or D.shape[1] != vdim_sealed:
        return "O2_6_NATIVE_RESIDUAL_EXTENSION_HASH_FAILURE"
    ext_train = [str(x) for x in z["train_ids"].tolist()]
    sealed_train = [str(x) for x in sealed_cell["train_ids"].tolist()]
    if ext_train != sealed_train or str(z["voxel_hash"]) != vh_sealed or len(z["fam"]) != 10:
        return "O2_6_NATIVE_RESIDUAL_IDENTITY_ALIGNMENT_FAILURE"
    return None


def verify_extension(state_dir, ext_dir, ext_manifest_path, rd, rois):
    """FIX 1: verify every deltanat file against the COMMITTED extension manifest (exists/name/sha256/bytes/
    shape/exactly-10-residuals/voxel-dim). FIX 2: verify identity ORDER (train_ids element-for-element vs the
    sealed RD cell), voxel_hash vs the sealed ROI hash, and family alignment. Any failure STOPS the gate."""
    man = json.loads(Path(ext_manifest_path).read_text())
    by = {f["name"]: f for f in man["files"]}
    rec = {"checked": 0, "matched": 0, "mismatches": [], "ok": True, "failure": None}
    for roi in rois:
        for s in ALL:
            vdim_sealed = int(rd["rois"][roi]["n_vox"][s]); vh_sealed = rd["rois"][roi]["voxel_hash"][s]
            for f in range(N_FOLDS):
                name = f"deltanat_{s}_{roi}_fold{f}.npz"; p = ext_dir / name
                rec["checked"] += 1
                cell_p = state_dir / f"cell_{s}_{roi}_fold{f}.npz"
                cell = dict(np.load(cell_p, allow_pickle=True)) if cell_p.exists() else {"train_ids": np.array([], dtype=object)}
                fail = verify_ext_file(p, by.get(name), cell, vdim_sealed, vh_sealed)
                if fail:
                    rec["ok"] = False; rec["failure"] = fail
                    rec["mismatches"].append({"name": name, "failure": fail})
                else:
                    rec["matched"] += 1
    if rec["ok"] and not (rec["checked"] == rec["matched"] == 96):
        rec["ok"] = False; rec["failure"] = "O2_6_NATIVE_RESIDUAL_EXTENSION_HASH_FAILURE"
    return rec


def run_frontier(repo, state_dir, ext_dir, rd_results_path, manifest_path, ext_manifest, rd_results_sha256, out, rois):
    G = _geom(repo)
    out.mkdir(parents=True, exist_ok=True)
    # FIX 3: rd_results provenance (verify SHA against the committed/sealed value)
    rd_sha = _sha256_file(rd_results_path)
    prov = {"path": str(rd_results_path), "sha256": rd_sha, "expected_sha256": rd_results_sha256,
            "match": (rd_results_sha256 is None or rd_sha == rd_results_sha256)}
    (out / "rd_results_provenance.json").write_text(json.dumps(prov, indent=2))
    if not prov["match"]:
        print("O2_6_TARGET_BASIS_INCONCLUSIVE (rd_results provenance mismatch)"); return 1
    rd = json.loads(rd_results_path.read_text())
    man = json.loads(manifest_path.read_text())
    by = {f["name"]: f for f in man["files"]}
    hv = {"checked": 0, "ok": True, "mismatches": []}
    needed = [f"cell_{s}_{roi}_fold{f}.npz" for roi in rois for s in ALL for f in range(N_FOLDS)]
    for n in needed:
        if by.get(n, {}).get("sha256") != _sha256_file(state_dir / n):
            hv["ok"] = False; hv["mismatches"].append(n)
        hv["checked"] += 1
    (out / "sealed_state_verification.json").write_text(json.dumps(hv, indent=2))
    if not hv["ok"]:
        print("O2_6_TARGET_BASIS_INCONCLUSIVE (sealed-state hash mismatch)"); return 1
    # FIX 1+2: verify the certified native-residual extension (hash/size/shape) AND identity alignment
    ev = verify_extension(state_dir, ext_dir, ext_manifest, rd, rois)
    (out / "native_residual_extension_verification.json").write_text(json.dumps(ev, indent=2, default=str))
    if not ev["ok"]:
        print(ev["failure"]); return 1
    ext_bound = {"max_leak": 0.0, "rows": [], "spectrum": []}
    per_subj = {s: {} for s in ALL}
    for roi in rois:
        for s in ALL:
            cells = [dict(np.load(state_dir / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            dnat = [dict(np.load(ext_dir / f"deltanat_{s}_{roi}_fold{f}.npz", allow_pickle=True)) for f in range(N_FOLDS)]
            per_subj[s][roi] = frontier_subject_roi(s, roi, cells, dnat, rd, G, ext_bound)
    roi_status, term, frontier_rows, prog = aggregate(per_subj, G)

    def _w(name, header, rows):
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    _w("total_recovery_frontier.csv", ["roi", "M", "d", "n_evaluable_participants", "median_TOTAL", "n_TOTAL_ge0.5", "median_E_AUG", "n_E_pos", "median_OUT_BASIS_RECOVERY"], frontier_rows)
    # augmentation results (participant x M x d)
    aug_rows = []
    for roi in rois:
        for s in ALL:
            for key, v in per_subj[s][roi]["md"].items():
                if not v.get("evaluable"):
                    continue
                M, d = key.split("_")
                aug_rows.append([s, roi, M, d, v["R_BASE"], v["R_AUG"], v["DELTA_OUT"], v["R_NULL_mean"], v["E_AUG"],
                                 v["R_COND"], v["OUT_BASIS_RECOVERY"], v["TOTAL_RECOVERY"], v["TRAIN_retention"]])
    _w("augmentation_results.csv", ["subject", "roi", "M", "d", "R_BASE", "R_AUG", "DELTA_OUT", "R_NULL_mean", "E_AUG", "R_COND", "OUT_BASIS_RECOVERY", "TOTAL_RECOVERY", "TRAIN_retention"], aug_rows)
    # generalization control
    gc = []
    for roi in rois:
        for s in ALL:
            for key, v in per_subj[s][roi]["md"].items():
                if "TRAIN_retention" in v:
                    M, d = key.split("_")
                    gc.append([s, roi, M, d, v["TRAIN_retention"], v["R_AUG"]])
    _w("generalization_control.csv", ["subject", "roi", "M", "d", "train_calibration_retention", "heldout_R_AUG"], gc)
    _w("outside_support_certification.csv", ["subject", "roi", "fold", "identity_idx", "leak_ratio"], ext_bound["rows"])
    _w("dimension_spectrum.csv", ["subject", "roi", "fold", "singular_values", "cum_energy_frac", "effective_rank", "stable_rank"], ext_bound["spectrum"])
    _w("conditional_outside_oracle.csv", ["subject", "roi", "M", "d", "R_BASE", "R_COND", "R_AUG"],
       [[r[0], r[1], r[2], r[3], r[4], r[9], r[5]] for r in aug_rows])
    _w("outside_basis_recovery.csv", ["subject", "roi", "M", "d", "OUT_BASIS_RECOVERY", "R_AUG_minus_BASE", "R_COND_minus_BASE"],
       [[r[0], r[1], r[2], r[3], r[10], r[6], r[9] - r[4]] for r in aug_rows])
    _w("calibration_subset_manifest.csv", ["M", "n_balanced_subsets"], [[m, n] for m, n in zip(M_GRID, [25, 100, 100, 25, 1])])
    null_e = [r[8] for r in aug_rows]
    (out / "random_basis_null_summary.json").write_text(json.dumps(
        {"n_null_per_cell": N_NULL, "seed_scheme": "O2.6|subject|ROI|fold|M|subset_id|D|null_iteration -> SHA256 -> uint64 -> PCG64",
         "E_AUG_mean_over_cells": float(np.mean(null_e)) if null_e else None,
         "E_AUG_median_over_cells": float(np.median(null_e)) if null_e else None}, indent=2))
    term_out = {roi: {k: term[roi][k] for k in ("median_E_AUG", "n_E_pos", "signflip_p", "median_TOTAL", "n_TOTAL_ge0.5", "holm_reject", "terminal_feasible")} for roi in rois}
    (out / "terminal_inference.json").write_text(json.dumps(term_out, indent=2, default=str))
    (out / "resource_frontier.json").write_text(json.dumps({roi: {"D50": roi_status[roi]["D50"], "M50": roi_status[roi]["M50"]} for roi in rois}, indent=2, default=str))
    (out / "primary_roi_status.json").write_text(json.dumps({roi: {"status": roi_status[roi]["status"], "D50": roi_status[roi]["D50"], "M50": roi_status[roi]["M50"], "terminal": term_out[roi]} for roi in rois}, indent=2, default=str))
    overfit = {}
    for roi in rois:
        g = per_subj[ALL[0]][roi]["md"].get("10_10", {})
        overfit[roi] = "TARGET_STATE_BASIS_IDENTITY_SPECIFIC_OVERFIT_SIGNATURE" if (g.get("TRAIN_retention", 0) > 0.9 and g.get("R_AUG", 1) < 0.3) else "none"
    (out / "scientific_status.json").write_text(json.dumps(
        {"status": prog, "roi_status": {roi: roi_status[roi]["status"] for roi in rois},
         "D50": {roi: roi_status[roi]["D50"] for roi in rois}, "M50": {roi: roi_status[roi]["M50"] for roi in rois},
         "outside_support_leak_max": ext_bound["max_leak"], "overfit_signature": overfit,
         "immutable": {"O2_5": "TARGET_IMAGERY_RESIDUAL_OUTSIDE_PERCEPTION_SUPPORT_DOMINANT",
                       "O2_4R": "TARGET_STATE_ORIENTATION_NOT_RECOVERED_BY_ORTHOGONAL_CALIBRATION"},
         "O3": "O3_NOT_READY"}, indent=2, default=str))
    print("O2_6_STATUS", prog)
    for roi in rois:
        print("  [%s] %s D50=%s M50=%s | terminal median_E=%.4f p=%.4g Holm=%s median_TOTAL=%.3f" %
              (roi, roi_status[roi]["status"], roi_status[roi]["D50"], roi_status[roi]["M50"],
               term[roi]["median_E_AUG"], term[roi]["signflip_p"], term[roi]["holm_reject"], term[roi]["median_TOTAL"]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--state", required=True)
    ap.add_argument("--ext", required=True); ap.add_argument("--rd-results", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ext-manifest", required=True)          # FIX 1: committed extension manifest
    ap.add_argument("--rd-results-sha256", default=None)      # FIX 3: expected committed rd_results sha
    ap.add_argument("--rois", default="ventral,lateral")
    a = ap.parse_args()
    return run_frontier(Path(a.repo), Path(a.state), Path(a.ext), Path(a.rd_results), Path(a.manifest),
                        Path(a.ext_manifest), a.rd_results_sha256, Path(a.out),
                        [r.strip() for r in a.rois.split(",") if r.strip()])


if __name__ == "__main__":
    raise SystemExit(main())
