"""Site configuration loading + cross-check against frozen recommended targets. Differences produce
REVIEW_REQUIRED (a PI/MRI physicist decides), never a silent pass or a hard scientific FAIL."""
from __future__ import annotations

from . import config as C

RECOMMENDED = {"field_strength_T": 7.0, "voxel_mm": 1.8, "TR_s": 1.6, "T1_mm_max": 1.0, "T2_recommended": True}

REQUIRED_SITE_FIELDS = ["institution", "site_id", "scanner_vendor", "model", "field_strength", "head_coil",
                        "TR", "TE", "flip_angle", "voxel_size", "multiband", "phase_encoding",
                        "distortion_correction", "display", "response_device", "trigger_device", "trigger_event",
                        "dummy_volumes", "operator_contact", "bids_root"]


def load_site_yaml(path):
    try:
        import yaml
    except Exception:
        import json
        return json.loads(open(path).read()) if str(path).endswith(".json") else _naive_yaml(open(path).read())
    with open(path) as f:
        return yaml.safe_load(f)


def _naive_yaml(text):
    d = {}
    for line in text.splitlines():
        line = line.split("#")[0].rstrip()
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1); d[k.strip()] = v.strip().strip('"') or None
    return d


def validate_site(site: dict):
    tbd = [k for k in REQUIRED_SITE_FIELDS if k not in site or C.is_tbd(site.get(k))]
    reviews = []

    def num(x):
        try:
            return float(str(x).replace("T", "").replace("mm", "").replace("s", "").strip())
        except Exception:
            return None
    fs = num(site.get("field_strength"))
    if fs is not None and abs(fs - RECOMMENDED["field_strength_T"]) > 1e-6:
        reviews.append("field_strength %s != recommended 7T -> REVIEW_REQUIRED" % site.get("field_strength"))
    vs = num(site.get("voxel_size"))
    if vs is not None and abs(vs - RECOMMENDED["voxel_mm"]) > 0.2:
        reviews.append("voxel_size %s != ~1.8mm -> REVIEW_REQUIRED" % site.get("voxel_size"))
    tr = num(site.get("TR"))
    if tr is not None and abs(tr - RECOMMENDED["TR_s"]) > 0.2:
        reviews.append("TR %s != ~1.6s -> REVIEW_REQUIRED" % site.get("TR"))
    status = "SITE_TBD_INCOMPLETE" if tbd else ("SITE_REVIEW_REQUIRED" if reviews else "SITE_MATCHES_TARGETS")
    return {"status": status, "tbd_fields": tbd, "reviews": reviews, "recommended": RECOMMENDED,
            "decision": "a real PI / MRI physicist decides acceptability; software does not auto-approve"}
