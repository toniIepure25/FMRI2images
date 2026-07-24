"""Reproducible subj01 x V1 x B0 x D0 engineering smoke for the Roy reproduction.

Regenerates the entire smoke from source metadata (behavioral TSVs, ROI/SNR
NIfTIs, the B0 HDF5) so the committed numerical result is reproducible from
repository code rather than an uncommitted inline script.

This module exactly mirrors the logic of the original S1.6 inline run: the same
run-beta boundaries, TSV trial-order mapping, RNG(seed) call sequence
(vision-split then imagery-split, identities in groupby/alphabetical order),
all-ordered-distinct vis2vis pairing (P0), and index-aligned vis2img pairing.

Nothing here interprets the numbers; the smoke validates the pipeline only.

Layout note: NSD betas are stored ``(trial, Z, Y, X)`` -- the reverse of the
NIfTI ``(X, Y, Z)`` -- so a NIfTI voxel ``(x, y, z)`` is ``betas[:, z, y, x]``.
Betas are int16 scaled x300; convert to percent signal change (``/300``) before
any linear algebra.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd

# Official 12-run acquisition order and beta-per-trial multiplicity.
RUN_ORDER = ["visA", "attA", "imgA_1", "visB", "attB", "imgB_1",
             "visC", "attC", "imgC_1", "imgA_2", "imgB_2", "imgC_2"]
TRIALS_PER_RUN = 48
BETAS_PER_TRIAL = {"vis": 1, "img": 1, "att": 2}
SELECTED_RUNS = ["visA", "visB", "imgA_2", "imgB_2"]
FAMILY = {"visA": "simple", "imgA_2": "simple", "visB": "naturalistic", "imgB_2": "naturalistic"}
STATE = {"visA": "vision", "visB": "vision", "imgA_2": "imagery", "imgB_2": "imagery"}
STIM_SET = {"visA": "A", "imgA_2": "A", "visB": "B", "imgB_2": "B"}


def run_beta_boundaries() -> Dict[str, Tuple[int, int]]:
    """Zero-based half-open [start, end) beta interval per run; sums to 720."""
    bounds, cur = {}, 0
    for r in RUN_ORDER:
        kind = "att" if r.startswith("att") else ("vis" if r.startswith("vis") else "img")
        n = TRIALS_PER_RUN * BETAS_PER_TRIAL[kind]
        bounds[r] = (cur, cur + n)
        cur += n
    assert cur == 720, f"beta boundaries sum to {cur}, expected 720"
    return bounds


def build_trial_table(bdata_dir: str) -> pd.DataFrame:
    """Build the 192-row Roy subset from the four selected behavioral TSVs.

    Beta trial order within each run follows the TSV ``TRIAL`` sequence (the NSD
    convention; ``designmatrixGLMsingle`` is condition-collapsed and is NOT the
    single-trial order).
    """
    bounds = run_beta_boundaries()
    rows: List[dict] = []
    for run in SELECTED_RUNS:
        b0, _ = bounds[run]
        df = pd.read_csv(f"{bdata_dir}/nsdimagery_subj01_{run}.tsv", sep="\t")
        df = df.sort_values("TRIAL").reset_index(drop=True)
        for k, (_, r) in enumerate(df.iterrows()):
            rows.append(dict(
                beta_index0=b0 + k, beta_id1=b0 + k + 1, run_name=run,
                tsv_trial=int(r["TRIAL"]), state=STATE[run], stim_set=STIM_SET[run],
                family=FAMILY[run], condition=str(r["CONDITION"]),
                identity=f"{STIM_SET[run]}:{r['CONDITION']}", cue=str(r["CUE"]),
                framefile=str(r["FRAMEFILE"]), trialonset=float(r["TRIALONSET"]),
                roy_inclusion=True, source_tsv=f"nsdimagery_subj01_{run}.tsv"))
    tt = pd.DataFrame(rows)
    tt["repeat"] = tt.groupby(["identity", "run_name"]).cumcount()
    _validate_trial_table(tt)
    return tt


def _validate_trial_table(tt: pd.DataFrame) -> None:
    assert len(tt) == 192, f"{len(tt)} rows, expected 192"
    assert (tt.state == "vision").sum() == 96 and (tt.state == "imagery").sum() == 96
    assert tt["identity"].nunique() == 12
    assert tt.groupby("family")["identity"].nunique().to_dict() == {"simple": 6, "naturalistic": 6}
    for st in ("vision", "imagery"):
        c = tt[tt.state == st].groupby("identity").size()
        assert c.min() == 8 and c.max() == 8, f"{st} repeats/identity not 8: {c.to_dict()}"
    assert tt["beta_index0"].is_unique
    assert not tt["run_name"].isin(["attA", "attB", "attC", "visC", "imgC_1", "imgC_2",
                                     "imgA_1", "imgB_1"]).any()


def select_v1_voxels(roi_dir: str, ppdata_dir: str, ncsnr_path: str) -> Tuple[np.ndarray, str, float]:
    """Return (xyz, hash, threshold) for the 98th-pct NSD-core-SNR V1 voxels.

    V1 = prf-visualrois labels {1: V1v, 2: V1d}. Selection is strict ``>`` the
    98th percentile within V1-valid-finite voxels.
    """
    import nibabel as nib
    pv = nib.load(f"{roi_dir}/prf-visualrois.nii.gz").get_fdata()
    valid = nib.load(f"{ppdata_dir}/valid_nsdimagery.nii.gz").get_fdata()
    ncsnr = nib.load(ncsnr_path).get_fdata()
    mask = np.isin(np.round(pv).astype(int), [1, 2]) & (valid > 0) & np.isfinite(ncsnr)
    xs, ys, zs = np.where(mask)
    snr = ncsnr[mask]
    thr = float(np.percentile(snr, 98))
    keep = snr > thr
    xyz = np.stack([xs[keep], ys[keep], zs[keep]])
    vhash = hashlib.sha256(xyz.tobytes()).hexdigest()[:12]
    return xyz, vhash, thr


def extract_v1_matrix(betas_path: str, beta_indices: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Extract selected (trial, voxel) betas as float32 PSC (``/300``), read-only."""
    import h5py
    xs, ys, zs = xyz
    with h5py.File(betas_path, "r") as f:
        b = f["betas"]  # (720, Z, Y, X)
        M = np.empty((len(beta_indices), xs.shape[0]), dtype=np.float32)
        for j in range(xs.shape[0]):
            M[:, j] = b[beta_indices, zs[j], ys[j], xs[j]].astype(np.float32)
    return M / 300.0


def make_splits(tt: pd.DataFrame, seed: int) -> Dict:
    """Deterministic per-identity 4/2/2 split for each state.

    Mirrors the original: one RNG, vision-state splits first then imagery-state,
    identities consumed in pandas groupby (alphabetical) order.
    """
    tt = tt.reset_index(drop=True).assign(_row=lambda d: range(len(d)))
    rng = np.random.default_rng(seed)

    def split_state(state: str):
        sub = tt[tt.state == state]
        out = {}
        for ident, g in sub.groupby("identity"):
            r = g.sort_values("repeat")["_row"].values
            perm = rng.permutation(r)
            out[ident] = dict(train=perm[:4], val=perm[4:6], test=perm[6:8])
        return out

    return {"vision": split_state("vision"), "imagery": split_state("imagery"), "_tt": tt}


Vis2VisPolicy = Literal["all-ordered-distinct", "derangement"]


def vis2vis_pairs(vsplit: Dict, part: str, policy: Vis2VisPolicy, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """vis2vis pairs for one partition.

    P0 ``all-ordered-distinct``: every ordered (source, target), source != target,
    per identity -> 4x3=12 train pairs/identity (144 total). P1 ``derangement``:
    one no-self permutation -> 4 train pairs/identity (48 total).
    """
    X: List[int] = []
    Y: List[int] = []
    rng = np.random.default_rng(seed)
    for ident in vsplit:
        if ident == "_tt":
            continue
        r = vsplit[ident][part]
        if policy == "all-ordered-distinct":
            for a in range(len(r)):
                for c in range(len(r)):
                    if a != c:
                        X.append(r[a]); Y.append(r[c])
        elif policy == "derangement":
            perm = rng.permutation(len(r))
            # ensure no fixed point
            while np.any(perm == np.arange(len(r))):
                perm = rng.permutation(len(r))
            for a in range(len(r)):
                X.append(r[a]); Y.append(r[perm[a]])
        else:
            raise ValueError(policy)
    return np.array(X), np.array(Y)


def vis2img_pairs(vsplit: Dict, isplit: Dict, part: str) -> Tuple[np.ndarray, np.ndarray]:
    """Index-aligned within-identity, within-split vision->imagery pairs."""
    X: List[int] = []
    Y: List[int] = []
    for ident in vsplit:
        if ident == "_tt":
            continue
        vr = vsplit[ident][part]
        ir = isplit[ident][part]
        n = min(len(vr), len(ir))
        for k in range(n):
            X.append(vr[k]); Y.append(ir[k])
    return np.array(X), np.array(Y)
