#!/usr/bin/env python3
"""Patch the trials check in infer-by-nsd to work without trial index."""

MAIN_PY = "/home/jovyan/work/FMRI2images/demo_thesis/backend/main.py"

with open(MAIN_PY, "r") as f:
    content = f.read()

# Patch the trials check to allow precomputed-only mode
old = """    trials = _trials_for_nsd_id(nsd_id)
    if not trials:
        raise HTTPException(404, f"NSD id {nsd_id} not found in the trial index")"""

new = """    trials = _trials_for_nsd_id(nsd_id) if _TRIAL_INDEX is not None else []
    if not trials and not _has_precomputed:
        raise HTTPException(404, f"NSD id {nsd_id} not found in the trial index")
    if not trials:
        trials = [0]  # placeholder trial index for precomputed-only mode"""

if old in content:
    content = content.replace(old, new)
    print("Patched: trials check allows precomputed-only mode")
else:
    print("WARNING: Could not find trials check block")

with open(MAIN_PY, "w") as f:
    f.write(content)
print("Done.")
