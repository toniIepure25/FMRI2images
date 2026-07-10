#!/usr/bin/env python3
"""
Patch the infer-by-nsd endpoint in main.py to allow reconstruction
using precomputed V62a μ when live models aren't available.
"""
import re

MAIN_PY = "/home/jovyan/work/FMRI2images/demo_thesis/backend/main.py"

with open(MAIN_PY, "r") as f:
    content = f.read()

# The old code block that requires features + all 3 models:
old_block = """    if _FMRI_FEATURES is None:
        return JSONResponse(status_code=503, content={"ok": False, "reason": "features_not_loaded"})
    if _V62_MODEL is None or _V61_MODEL is None or _V66_MODEL is None:
        return JSONResponse(status_code=503, content={"ok": False, "reason": "triple_fusion_models_not_loaded"})"""

# New code: allow endpoint to proceed when precomputed predictions are available
new_block = """    _has_live_models = (_FMRI_FEATURES is not None and _V62_MODEL is not None
                       and _V61_MODEL is not None and _V66_MODEL is not None)
    _has_precomputed = bool(_SHARED1000_PREDS.get("v62") is not None)
    if not _has_live_models and not _has_precomputed:
        return JSONResponse(status_code=503, content={"ok": False, "reason": "features_and_precomputed_both_unavailable"})"""

if old_block in content:
    content = content.replace(old_block, new_block)
    print("Patched: model/feature requirement checks → allow precomputed fallback")
else:
    print("WARNING: Could not find the exact old_block to patch (model checks)")

# Now patch the live V62a forward pass section to use precomputed when models unavailable
old_fwd = """    # Live V62a μ for SDXL+IPA prompt-embed injection.  Run on the rep-averaged
    # per-session z-scored beta to match what the model was trained on.
    import torch as _torch
    t_fwd = _time.time()
    enc62 = _get_encoder_type(_V62_CONFIG)
    sess_col = "session" if _TRIAL_INDEX is not None and "session" in _TRIAL_INDEX.columns else None
    rep_mus, rep_kappas = [], []
    for t in trials:
        beta = _FMRI_FEATURES[int(t)].astype(np.float32)
        sess = int(_TRIAL_INDEX.iloc[int(t)][sess_col]) if sess_col else None
        beta = _zscore_per_session(beta, sess)
        x = _torch.from_numpy(beta).unsqueeze(0).float().to(_DEVICE)
        r = _model_forward(_V62_MODEL, x, _V62_CONFIG, enc62, enable_dropout=False)
        rep_mus.append(r["mu"])
        if r["kappa"] is not None:
            rep_kappas.append(r["kappa"])
    stack = np.stack(rep_mus, axis=0)
    if rep_kappas and len(rep_kappas) == stack.shape[0]:
        w = np.asarray(rep_kappas, dtype=np.float32)
        w = w / (w.sum() + 1e-8)
        v62_mu = (stack * w[:, None]).sum(axis=0)
        kappa_avg = float(np.mean(rep_kappas))
    else:
        v62_mu = stack.mean(axis=0)
        kappa_avg = 0.0
    v62_mu = v62_mu / (np.linalg.norm(v62_mu) + 1e-8)
    fwd_ms = (_time.time() - t_fwd) * 1000.0"""

new_fwd = """    # V62a μ for SDXL+IPA prompt-embed injection.
    # Prefer live forward pass when models are loaded; fall back to precomputed.
    t_fwd = _time.time()
    if _has_live_models:
        import torch as _torch
        enc62 = _get_encoder_type(_V62_CONFIG)
        sess_col = "session" if _TRIAL_INDEX is not None and "session" in _TRIAL_INDEX.columns else None
        rep_mus, rep_kappas = [], []
        for t in trials:
            beta = _FMRI_FEATURES[int(t)].astype(np.float32)
            sess = int(_TRIAL_INDEX.iloc[int(t)][sess_col]) if sess_col else None
            beta = _zscore_per_session(beta, sess)
            x = _torch.from_numpy(beta).unsqueeze(0).float().to(_DEVICE)
            r = _model_forward(_V62_MODEL, x, _V62_CONFIG, enc62, enable_dropout=False)
            rep_mus.append(r["mu"])
            if r["kappa"] is not None:
                rep_kappas.append(r["kappa"])
        stack = np.stack(rep_mus, axis=0)
        if rep_kappas and len(rep_kappas) == stack.shape[0]:
            w = np.asarray(rep_kappas, dtype=np.float32)
            w = w / (w.sum() + 1e-8)
            v62_mu = (stack * w[:, None]).sum(axis=0)
            kappa_avg = float(np.mean(rep_kappas))
        else:
            v62_mu = stack.mean(axis=0)
            kappa_avg = 0.0
        v62_mu = v62_mu / (np.linalg.norm(v62_mu) + 1e-8)
        _mu_source = "LIVE_LOCAL"
    else:
        v62_mu = _SHARED1000_PREDS["v62"][row].copy()
        v62_mu = v62_mu / (np.linalg.norm(v62_mu) + 1e-8)
        kappa_avg = 0.0
        _mu_source = "CACHED_LOCAL"
        logger.info("Using precomputed V62a mu for nsd_id=%d (models not loaded)", nsd_id)
    fwd_ms = (_time.time() - t_fwd) * 1000.0"""

if old_fwd in content:
    content = content.replace(old_fwd, new_fwd)
    print("Patched: V62a forward pass → precomputed fallback")
else:
    print("WARNING: Could not find the exact old_fwd block to patch")

# Patch provenance and n_voxels to handle missing features
old_nvox = '        "n_voxels": int(_FMRI_FEATURES.shape[1]),'
new_nvox = '        "n_voxels": int(_FMRI_FEATURES.shape[1]) if _FMRI_FEATURES is not None else 0,'
if old_nvox in content:
    content = content.replace(old_nvox, new_nvox)
    print("Patched: n_voxels guard for None features")

# Patch the "live_v62_mu_for_recon" field to reflect the source
old_live_mu = '            "live_v62_mu_for_recon": True,'
new_live_mu = '            "live_v62_mu_for_recon": _has_live_models,'
if old_live_mu in content:
    content = content.replace(old_live_mu, new_live_mu)
    print("Patched: live_v62_mu_for_recon reflects actual source")

# Patch model_forward provenance to reflect source
old_prov_fwd = '            "model_forward": "LIVE_LOCAL",'
new_prov_fwd = '            "model_forward": _mu_source,'
# Only replace the one inside the infer-by-nsd result dict (there might be others)
# We'll replace the first occurrence after the "infer-by-nsd" function
idx = content.find("def infer_by_nsd")
if idx >= 0:
    # Find the provenance dict within the function
    prov_idx = content.find(old_prov_fwd, idx)
    if prov_idx >= 0:
        content = content[:prov_idx] + new_prov_fwd + content[prov_idx + len(old_prov_fwd):]
        print("Patched: model_forward provenance → dynamic")
    else:
        print("WARNING: Could not find model_forward provenance in infer_by_nsd")

with open(MAIN_PY, "w") as f:
    f.write(content)

print("\nAll patches applied. Restart uvicorn to take effect.")
