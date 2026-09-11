"""O2.6 native-residual STATE EXTENSION (frozen config f4e94517). Derives, ONCE, the native target imagery
residuals delta_native for the OUTER-TRAINING identities (subject x ROI x fold x identity) using the EXACT
sealed O2.3A-RD residual code (O2.2 residual: imagery centroid orthogonal to the outer-training vision span).
Does NOT redefine the residual, refit SRM/K/rank, or touch W_target.

Before trusting the extension it REPLAYS the two sealed outer-test deltas_test per fold and requires
max |discrepancy| <= 1e-10 and identical ROI voxel hashes, else O2_6_NATIVE_RESIDUAL_STATE_EXTENSION_FAILURE.
Runs on the cluster (reads the preserved B0 NSD-Imagery data on the PVC via the committed cohort loaders)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ALL = [f"subj0{i}" for i in range(1, 9)]
ROIS_PRIMARY = ["ventral", "lateral"]
N_FOLDS = 6
REPLAY_TOL = 1e-10


def _coh(repo: Path):
    sys.path.insert(0, str(repo / "src"))
    from fmri2img.mindcompiler.operator_o2_3a_rd import cohort as CH
    return CH


def _sha_arr(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()).hexdigest()


def run_extension(repo: Path, data: Path, state_dir: Path, rd_results_path: Path, out: Path, rois):
    CH = _coh(repo)
    out.mkdir(parents=True, exist_ok=True)
    rd = json.loads(rd_results_path.read_text())
    manifest = {"gate": "O2.6", "artifact": "native_residual_state_extension", "tol": REPLAY_TOL,
                "replay": [], "files": [], "ok": True}
    max_disc_global = 0.0
    for roi in rois:
        for s in ALL:
            xyz, vh, nv = CH.roi_xyz(data, repo, s, roi)
            ids, V, I, fam = CH.imagery_centroids(data, repo, s, roi, xyz)
            idx = {i: k for k, i in enumerate(ids)}
            sealed_vh = rd["rois"][roi]["voxel_hash"][s]
            vh_ok = (vh == sealed_vh)
            for f in range(N_FOLDS):
                cell = dict(np.load(state_dir / f"cell_{s}_{roi}_fold{f}.npz", allow_pickle=True))
                train_ids = [str(x) for x in cell["train_ids"].tolist()]
                test_ids = [str(x) for x in cell["test_ids"].tolist()]
                Vtr = np.stack([V[idx[i]] for i in train_ids])          # (10 x p) outer-training vision
                B_vis, mu_vis = CH.vis_span_basis(Vtr)
                delta_native = {i: CH.delta_perp(I[idx[i]], B_vis, mu_vis) for i in ids}
                # replay: the two sealed outer-test residuals must match exactly
                sealed_test = [np.asarray(d, np.float64) for d in list(cell["deltas_test"])]
                disc = max(float(np.max(np.abs(delta_native[test_ids[k]] - sealed_test[k]))) for k in range(2))
                max_disc_global = max(max_disc_global, disc)
                ok = (disc <= REPLAY_TOL) and vh_ok
                if not ok:
                    manifest["ok"] = False
                # persist native residuals for the 10 outer-training identities (ordered by train_ids)
                D_train = np.stack([delta_native[i] for i in train_ids])   # (10 x p)
                fpath = out / f"deltanat_{s}_{roi}_fold{f}.npz"
                np.savez(fpath, delta_native=D_train, train_ids=np.array(train_ids, dtype=object),
                         fam=np.array([fam[i] for i in train_ids], dtype=object), voxel_hash=vh)
                manifest["files"].append({"name": fpath.name, "sha256": hashlib.sha256(fpath.read_bytes()).hexdigest(),
                                          "bytes": fpath.stat().st_size, "shape": list(D_train.shape)})
                manifest["replay"].append({"subject": s, "roi": roi, "fold": f, "max_abs_discrepancy": disc,
                                           "voxel_hash_match": vh_ok, "ok": ok})
    manifest["max_abs_discrepancy_global"] = max_disc_global
    (out / "native_residual_state_extension_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if manifest["ok"]:
        print("O2_6_STATE_EXTENSION_OK max_disc=%.2e n=%d" % (max_disc_global, len(manifest["replay"])))
        return 0
    print("O2_6_NATIVE_RESIDUAL_STATE_EXTENSION_FAILURE max_disc=%.2e" % max_disc_global)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--state", required=True); ap.add_argument("--rd-results", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--rois", default="ventral,lateral")
    a = ap.parse_args()
    return run_extension(Path(a.repo), Path(a.data), Path(a.state), Path(a.rd_results), Path(a.out),
                         [r.strip() for r in a.rois.split(",") if r.strip()])


if __name__ == "__main__":
    raise SystemExit(main())
