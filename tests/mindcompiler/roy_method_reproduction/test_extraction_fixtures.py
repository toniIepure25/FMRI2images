"""Data-free extraction fixtures for the two host-dependent steps.

Real NSD data is never touched: we synthesise a tiny HDF5 beta volume and tiny
ROI/SNR NIfTIs with KNOWN ground truth, then assert that

* ``extract_v1_matrix`` honours the reversed ``(trial, Z, Y, X)`` HDF5 layout,
  reads ``betas[:, z, y, x]`` for a NIfTI ``(x, y, z)`` voxel, and applies the
  ``/300`` int16 percent-signal-change scaling exactly; and
* ``select_v1_voxels`` restricts to V1 labels ``{1, 2}``, requires ``valid > 0``
  and finite SNR, and keeps voxels STRICTLY above the 98th SNR percentile.

These are the two places a silent index transpose or an off-by-one percentile
would corrupt every downstream number, so they are pinned against ground truth.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from fmri2img.mindcompiler.roy_method_reproduction import smoke_pipeline as sp

h5py = pytest.importorskip("h5py")
nib = pytest.importorskip("nibabel")


# --- item 7: synthetic HDF5 extraction --------------------------------------

def _make_betas(path, n_trials, Z, Y, X):
    """int16 betas laid out (trial, Z, Y, X); value encodes its own coordinates."""
    b = np.empty((n_trials, Z, Y, X), dtype=np.int16)
    for t in range(n_trials):
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    # a small deterministic function, distinct per (t,z,y,x)
                    b[t, z, y, x] = (t * 1000 + z * 100 + y * 10 + x) % 30000
    with h5py.File(path, "w") as f:
        f.create_dataset("betas", data=b)
    return b


def test_extract_v1_matrix_layout_and_scaling(tmp_path):
    n_trials, Z, Y, X = 12, 3, 4, 5
    ground = _make_betas(tmp_path / "betas.hdf5", n_trials, Z, Y, X)
    # three NIfTI-space voxels (x, y, z)
    xyz = np.array([[1, 3, 2], [0, 1, 2], [3, 0, 1]]).T  # (x,y,z) within bounds; shape (3, n_vox)
    beta_indices = np.array([0, 2, 5, 11])
    M = sp.extract_v1_matrix(str(tmp_path / "betas.hdf5"), beta_indices, xyz)

    assert M.shape == (len(beta_indices), xyz.shape[1])
    assert M.dtype == np.float32
    for j in range(xyz.shape[1]):
        x, y, z = xyz[:, j]
        for i, t in enumerate(beta_indices):
            # the reversed layout: NIfTI (x,y,z) -> betas[t, z, y, x], then /300
            expected = ground[t, z, y, x].astype(np.float32) / 300.0
            assert M[i, j] == pytest.approx(expected), f"voxel {j} trial {t}"


def test_extract_v1_matrix_is_percent_signal_change(tmp_path):
    _make_betas(tmp_path / "b.hdf5", 4, 2, 2, 2)
    xyz = np.array([[0], [0], [0]])
    with h5py.File(tmp_path / "b.hdf5", "r") as f:
        raw = f["betas"][0, 0, 0, 0]
    M = sp.extract_v1_matrix(str(tmp_path / "b.hdf5"), np.array([0]), xyz)
    assert M[0, 0] == pytest.approx(raw / 300.0)  # scaling applied, not raw int16


# --- item 8: synthetic ROI / SNR selection ----------------------------------

def _nii(path, arr):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4)), str(path))


def _make_roi_snr(dirpath, prf, valid, ncsnr):
    roi = dirpath / "roi"; pp = dirpath; roi.mkdir(parents=True, exist_ok=True)
    _nii(roi / "prf-visualrois.nii.gz", prf)
    _nii(pp / "valid_nsdimagery.nii.gz", valid)
    ncsnr_path = dirpath / "ncsnr.nii.gz"
    _nii(ncsnr_path, ncsnr)
    return str(roi), str(pp), str(ncsnr_path)


def test_select_v1_voxels_percentile_and_masks(tmp_path):
    # 10x1x1 volume. V1 (labels 1/2) at x=0..7; non-V1 elsewhere.
    shape = (10, 1, 1)
    prf = np.zeros(shape); prf[0:4, 0, 0] = 1; prf[4:8, 0, 0] = 2; prf[8:10, 0, 0] = 3
    valid = np.ones(shape); valid[7, 0, 0] = 0  # drop one V1 voxel via validity
    snr = np.zeros(shape)
    # V1-valid voxels are x in {0,1,2,3,4,5,6} -> assign ascending SNR
    snr_vals = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5, 5: 0.6, 6: 0.9}
    for x, v in snr_vals.items():
        snr[x, 0, 0] = v
    snr[8, 0, 0] = 5.0  # huge SNR but NOT V1 -> must be excluded
    roi, pp, ncsnr = _make_roi_snr(tmp_path, prf, valid, snr)

    xyz, vhash, thr = sp.select_v1_voxels(roi, pp, ncsnr)
    # 98th percentile of the 7 V1-valid SNRs; strict > keeps only the top voxel.
    vals = np.array(sorted(snr_vals.values()))
    assert thr == pytest.approx(float(np.percentile(vals, 98)))
    kept_x = sorted(xyz[0].tolist())
    assert kept_x == [6], f"expected only x=6 above 98th pct, got {kept_x}"
    # excluded: the non-V1 high-SNR voxel and the invalidated V1 voxel
    assert 8 not in kept_x and 7 not in kept_x


def test_select_v1_voxels_excludes_nonfinite_snr(tmp_path):
    shape = (5, 1, 1)
    prf = np.ones(shape)  # all V1v
    valid = np.ones(shape)
    snr = np.array([0.1, 0.2, np.nan, 0.4, 0.9]).reshape(shape)
    roi, pp, ncsnr = _make_roi_snr(tmp_path, prf, valid, snr)
    xyz, vhash, thr = sp.select_v1_voxels(roi, pp, ncsnr)
    # NaN voxel never enters the pool (finite mask) nor the percentile base.
    finite_vals = np.array([0.1, 0.2, 0.4, 0.9])
    assert thr == pytest.approx(float(np.percentile(finite_vals, 98)))
    assert 2 not in xyz[0].tolist()


def test_select_v1_voxels_hash_is_stable(tmp_path):
    shape = (6, 1, 1)
    prf = np.ones(shape); valid = np.ones(shape)
    snr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.9]).reshape(shape)
    roi, pp, ncsnr = _make_roi_snr(tmp_path, prf, valid, snr)
    xyz, vhash, _ = sp.select_v1_voxels(roi, pp, ncsnr)
    assert vhash == hashlib.sha256(xyz.tobytes()).hexdigest()[:12]
