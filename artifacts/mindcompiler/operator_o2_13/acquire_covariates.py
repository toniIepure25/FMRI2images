"""O2.13 covariate acquisition + extraction (one-time authorized NSD S3 read-only fetch).
Family B (behavior): main-NSD continuous-recognition d' + criterion c from responses.tsv (ISOLD truth,
ISCORRECT -> response; loglinear corrected). Family A (anatomy): streams-parcel (ventral=5, lateral=6)
surface area = sum vertex areas; mean cortical thickness = mean vertex thickness; over both hemispheres.
Writes raw files (hashed) + covariates.json + covariate_manifest.csv to the PVC and prints them."""
import hashlib
import json
import math
import os
import urllib.request
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.stats import norm

BASE = "https://natural-scenes-dataset.s3.amazonaws.com/"
SUBS = [f"subj0{i}" for i in range(1, 9)]
ROOT = Path("/rd/data/data/nsd_covariates")
RAW = ROOT / "raw"
STREAMS = {"ventral": 5, "lateral": 6}
HEMIS = ["lh", "rh"]


def fetch(key):
    dst = RAW / key
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size == 0:
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(BASE + key, dst)
                break
            except Exception as e:
                if attempt == 3:
                    raise
    return dst


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def behavior(sub):
    key = f"nsddata/ppdata/{sub}/behav/responses.tsv"
    p = fetch(key)
    rows = p.read_text().splitlines()
    hdr = rows[0].split("\t")
    idx = {c: i for i, c in enumerate(hdr)}
    iso = idx["ISOLD"]; isc = idx["ISCORRECT"]; miss = idx.get("MISSINGDATA")
    n_old = n_new = hits = fa = used = skipped = 0
    for line in rows[1:]:
        f = line.split("\t")
        if not f or len(f) <= max(iso, isc):
            continue
        if miss is not None and f[miss] not in ("0", "0.0", ""):
            try:
                if float(f[miss]) != 0:
                    skipped += 1; continue
            except ValueError:
                skipped += 1; continue
        try:
            old = int(round(float(f[iso]))); corr = int(round(float(f[isc])))
        except ValueError:
            skipped += 1; continue
        if old not in (0, 1) or corr not in (0, 1):
            skipped += 1; continue
        resp_old = old if corr == 1 else (1 - old)                # response = truth iff correct
        used += 1
        if old == 1:
            n_old += 1; hits += (resp_old == 1)
        else:
            n_new += 1; fa += (resp_old == 1)
    H = (hits + 0.5) / (n_old + 1); F = (fa + 0.5) / (n_new + 1)
    zH = float(norm.ppf(H)); zF = float(norm.ppf(F))
    return {"source": key, "sha256": sha(p), "n_old": n_old, "n_new": n_new, "hits": hits, "false_alarms": fa,
            "used_trials": used, "skipped_trials": skipped, "H": H, "F": F,
            "d_prime": zH - zF, "criterion_c": -0.5 * (zH + zF)}


def anatomy(sub):
    files = {}
    area = {}; thick = {}; strm = {}
    for h in HEMIS:
        ka = f"nsddata/freesurfer/{sub}/surf/{h}.area"; kt = f"nsddata/freesurfer/{sub}/surf/{h}.thickness"
        ks = f"nsddata/freesurfer/{sub}/label/{h}.streams.mgz"
        pa, pt, ps = fetch(ka), fetch(kt), fetch(ks)
        files[ka] = sha(pa); files[kt] = sha(pt); files[ks] = sha(ps)
        area[h] = np.asarray(nib.freesurfer.io.read_morph_data(str(pa)), np.float64)
        thick[h] = np.asarray(nib.freesurfer.io.read_morph_data(str(pt)), np.float64)
        sv = np.asarray(nib.load(str(ps)).get_fdata(), np.float64).squeeze()
        strm[h] = np.round(sv).astype(int)
        assert area[h].shape == thick[h].shape == strm[h].shape, (sub, h, area[h].shape, thick[h].shape, strm[h].shape)
    out = {"sources": files, "n_vertices": {h: int(area[h].size) for h in HEMIS}}
    for roi, lab in STREAMS.items():
        a_sum = 0.0; t_vals = []
        nv = 0
        for h in HEMIS:
            m = strm[h] == lab
            a_sum += float(area[h][m].sum())
            t_vals.append(thick[h][m]); nv += int(m.sum())
        tv = np.concatenate(t_vals) if t_vals else np.array([])
        out[roi] = {"streams_label": lab, "n_vertices_roi": nv,
                    "surface_area_mm2": a_sum, "mean_thickness_mm": float(tv.mean()) if tv.size else float("nan")}
    return out


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    cov = {"gate": "O2.13", "base_url": BASE, "streams_labels": STREAMS, "subjects": {}}
    for sub in SUBS:
        b = behavior(sub); a = anatomy(sub)
        cov["subjects"][sub] = {"behavior": b, "anatomy": a}
        print("%s  d'=%.4f  c=%.4f  (old=%d new=%d hit=%d fa=%d)  | vent area=%.1f thick=%.3f  lat area=%.1f thick=%.3f"
              % (sub, b["d_prime"], b["criterion_c"], b["n_old"], b["n_new"], b["hits"], b["false_alarms"],
                 a["ventral"]["surface_area_mm2"], a["ventral"]["mean_thickness_mm"],
                 a["lateral"]["surface_area_mm2"], a["lateral"]["mean_thickness_mm"]))
    (ROOT / "covariates.json").write_text(json.dumps(cov, indent=2))
    # flat manifest CSV
    lines = ["subject,d_prime,criterion_c,n_old,n_new,hits,false_alarms,used_trials,behav_sha256,"
             "ventral_surface_area_mm2,ventral_mean_thickness_mm,lateral_surface_area_mm2,lateral_mean_thickness_mm"]
    for sub in SUBS:
        b = cov["subjects"][sub]["behavior"]; a = cov["subjects"][sub]["anatomy"]
        lines.append("%s,%.10f,%.10f,%d,%d,%d,%d,%d,%s,%.6f,%.6f,%.6f,%.6f" % (
            sub, b["d_prime"], b["criterion_c"], b["n_old"], b["n_new"], b["hits"], b["false_alarms"], b["used_trials"],
            b["sha256"], a["ventral"]["surface_area_mm2"], a["ventral"]["mean_thickness_mm"],
            a["lateral"]["surface_area_mm2"], a["lateral"]["mean_thickness_mm"]))
    (ROOT / "covariate_manifest.csv").write_text("\n".join(lines) + "\n")
    print("=== COVARIATE_MANIFEST ==="); print("\n".join(lines))
    print("=== COVARIATES_JSON_SHA ===", sha(ROOT / "covariates.json"))
    print("O2_13_COVARIATES_ACQUIRED")


if __name__ == "__main__":
    main()
