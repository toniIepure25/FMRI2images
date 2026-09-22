"""BIDS-compatible events writer + validator. In SIMULATION no BOLD is produced (marker SIMULATED_NO_BOLD).
Internal structural validation is always available; the official bids-validator is run only if installed, and
its PASS is never equated with the internal schema PASS."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

from . import event_schema as ES


def _val(v):
    return ES.NA if v is None else v


def write_events(records, out_dir, sub, ses, task, run):
    d = Path(out_dir) / ("sub-%s" % sub) / ("ses-%s" % ses) / "func"
    d.mkdir(parents=True, exist_ok=True)
    base = "sub-%s_ses-%s_task-%s_run-%02d" % (sub, ses, task, run)
    tsv = d / (base + "_events.tsv")
    cols = ES.EVENT_COLUMNS + [c for c in ES.EXTRA_COLUMNS if any(c in r for r in records)]
    with open(tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(cols)
        for r in records:
            w.writerow([_val(r.get(c)) for c in cols])
    (d / (base + "_events.json")).write_text(json.dumps({k: ES.EVENTS_JSON[k] for k in cols if k in ES.EVENTS_JSON}, indent=2))
    (d / (base + "_events.tsv.SIMULATED_NO_BOLD")).write_text("No BOLD acquired (SIMULATION/engineering). Events only.\n")
    return str(tsv)


def write_scansync(rows, out_dir, sub, ses, task, run):
    d = Path(out_dir) / ("sub-%s" % sub) / ("ses-%s" % ses) / "func"
    d.mkdir(parents=True, exist_ok=True)
    base = "sub-%s_ses-%s_task-%s_run-%02d_scansync.tsv" % (sub, ses, task, run)
    p = d / base
    cols = ["trigger_count", "monotonic_s", "wall_utc", "est_volume_index", "inter_trigger_interval_s", "anomaly"]
    with open(p, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    return str(p)


def internal_validate(bids_root):
    """Structural check: dataset_description present, at least one events.tsv with required columns."""
    root = Path(bids_root)
    issues = []
    if not (root / "dataset_description.json").exists():
        issues.append("missing_dataset_description")
    tsvs = list(root.rglob("*_events.tsv"))
    if not tsvs:
        issues.append("no_events_tsv")
    for t in tsvs:
        header = t.read_text().splitlines()[0].split("\t") if t.exists() else []
        missing = [c for c in ES.EVENT_COLUMNS if c not in header]
        if missing:
            issues.append("missing_columns:%s:%s" % (t.name, ",".join(missing)))
    ok = len(issues) == 0
    return {"status": "INTERNAL_BIDS_SCHEMA_PASS" if ok else "INTERNAL_BIDS_SCHEMA_FAIL",
            "official_validator_equivalent": False, "n_events_tsv": len(tsvs), "issues": issues}


def official_validate(bids_root):
    exe = shutil.which("bids-validator")
    if not exe:
        return {"status": "BIDS_VALIDATOR_UNAVAILABLE", "ran": False}
    try:
        r = subprocess.run([exe, str(bids_root), "--json"], capture_output=True, text=True, timeout=120)
        return {"status": "BIDS_VALIDATOR_PASS" if r.returncode == 0 else "BIDS_VALIDATOR_FAIL",
                "ran": True, "returncode": r.returncode}
    except Exception as e:
        return {"status": "BIDS_VALIDATOR_ERROR", "ran": True, "error": str(e)[:200]}


def ensure_dataset_description(bids_root, name="O2.16 independent replication (SIMULATED)"):
    root = Path(bids_root); root.mkdir(parents=True, exist_ok=True)
    dd = root / "dataset_description.json"
    if not dd.exists():
        dd.write_text(json.dumps({"Name": name, "BIDSVersion": "1.8.0", "DatasetType": "raw",
                                  "GeneratedBy": [{"Name": "mindir-o216 experiment stack", "Version": "0.1.0"}]}, indent=2))
    return str(dd)
