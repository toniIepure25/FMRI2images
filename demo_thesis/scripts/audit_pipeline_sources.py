#!/usr/bin/env python3
"""Audit pipeline_cases.json and live backend for source consistency."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"

def load_cases():
    p = PUBLIC_DIR / "data" / "pipeline_cases.json"
    if not p.exists():
        print(f"MISSING: {p}")
        sys.exit(1)
    return json.loads(p.read_text())

def check_image(rel_path: str) -> dict:
    full = PUBLIC_DIR / rel_path.lstrip("/")
    result = {"path": rel_path, "exists": full.exists()}
    if full.exists():
        result["size_kb"] = round(full.stat().st_size / 1024, 1)
        with open(full, "rb") as f:
            result["md5"] = hashlib.md5(f.read()).hexdigest()
    return result

def audit_case(c: dict) -> list[str]:
    warnings = []
    cid = c.get("id", "?")
    m = c.get("metrics", {})

    if isinstance(m.get("cosine"), (int, float)) and m["cosine"] > 1.0:
        warnings.append(f"  [{cid}] metrics.cosine={m['cosine']:.3f} > 1.0 — likely CSLS, not cosine")

    if m.get("pixcorr") == 0.0:
        warnings.append(f"  [{cid}] pixcorr=0.0 — probably unmeasured default, should be null")
    if m.get("ssim") == 0.0:
        warnings.append(f"  [{cid}] ssim=0.0 — probably unmeasured default, should be null")

    for ri in c.get("retrievedImages", []):
        img = check_image(ri.get("image", ""))
        if not img["exists"]:
            warnings.append(f"  [{cid}] retrieved_{ri['rank']} image MISSING: {ri['image']}")
        elif img["size_kb"] < 2:
            warnings.append(f"  [{cid}] retrieved_{ri['rank']} image suspiciously small ({img['size_kb']} KB): {ri['image']}")

    target = check_image(c.get("targetImage", ""))
    if not target["exists"]:
        warnings.append(f"  [{cid}] target image MISSING: {c.get('targetImage', '?')}")

    recon_path = c.get("diffusionFinal") or c.get("reconstructionImage")
    recon = check_image(recon_path) if recon_path else {"path": "", "exists": False}
    if recon_path and not recon["exists"]:
        warnings.append(f"  [{cid}] reconstruction image MISSING: {recon_path}")

    if recon.get("exists") and target.get("exists"):
        if recon.get("md5") == target.get("md5"):
            warnings.append(f"  [{cid}] WARNING: reconstruction == target image (same hash)")

    ret1 = [ri for ri in c.get("retrievedImages", []) if ri.get("rank") == 1]
    if ret1:
        ret1_img = check_image(ret1[0].get("image", ""))
        if recon.get("exists") and ret1_img.get("exists"):
            if recon.get("md5") == ret1_img.get("md5"):
                warnings.append(f"  [{cid}] WARNING: reconstruction == retrieved#1 image (same hash)")

    return warnings

def check_backend(base_url: str, trial_idx: int | None):
    try:
        import urllib.request
        resp = urllib.request.urlopen(f"{base_url}/api/health", timeout=3)
        health = json.loads(resp.read())
        print(f"\nBackend: {health.get('status', '?')}")
        mm = health.get("model_meta")
        if mm:
            enc = mm.get("encoder_type", "?")
            mtype = mm.get("model_type", "?")
            print(f"Model: {enc} encoder + {mtype} head")
        print(f"Inference: {'available' if health.get('inference_available') else 'unavailable'}")
        print(f"Reconstruction: {'available' if health.get('reconstruction_available') else 'not available'}")

        if trial_idx is not None and health.get("inference_available"):
            resp2 = urllib.request.urlopen(f"{base_url}/api/infer/{trial_idx}", timeout=10)
            infer = json.loads(resp2.read())
            top_k = infer.get("top_k", [])
            kappa = infer.get("kappa")
            delta = infer.get("delta")
            print(f"\nLive inference trial={trial_idx}:")
            print(f"  kappa={kappa}, delta={delta}")
            for tk in top_k[:5]:
                print(f"  rank={tk['rank']} nsd_id={tk['nsd_id']} cosine={tk['cosine']:.4f} csls={tk.get('csls', '?')}")
    except Exception as e:
        print(f"\nBackend check failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Audit pipeline sources")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--trial-idx", type=int, default=None)
    args = parser.parse_args()

    cases = load_cases()
    print(f"Cases loaded: {len(cases)}")

    all_warnings = []
    n_clean = 0
    for c in cases:
        w = audit_case(c)
        all_warnings.extend(w)
        if not w:
            n_clean += 1

    print(f"Clean cases: {n_clean}/{len(cases)}")
    if all_warnings:
        print(f"\nWarnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(w)
    else:
        print("No warnings.")

    check_backend(args.base_url, args.trial_idx)

if __name__ == "__main__":
    main()
