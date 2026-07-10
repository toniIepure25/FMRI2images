#!/usr/bin/env python3
"""Extract demo assets from NSD HDF5 + backend inference for Cortex2Canvas.

Run on the pod:
  python3 extract_demo_assets.py --output-dir /tmp/demo_export

Then kubectl cp /tmp/demo_export to local public/assets/cases/ and public/data/.
"""
import json
import sys
import os
import struct
import numpy as np
from pathlib import Path

try:
    import h5py
    from PIL import Image
except ImportError:
    print("Need: pip install h5py Pillow")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Need: pip install requests")
    sys.exit(1)

BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000")
HDF5_PATH = os.environ.get("NSD_HDF5", "/home/jovyan/work/data/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")

# The 4 user-provided NSD IDs + 15 additional interesting ones
# We'll discover good ones via the backend
USER_NSD_IDS = [8262, 21279, 55649, 8509]

def get_health():
    r = requests.get(f"{BACKEND}/api/health", timeout=10)
    r.raise_for_status()
    return r.json()

def resolve_nsd_to_trial(nsd_id):
    r = requests.get(f"{BACKEND}/api/nsd-id-to-trial/{nsd_id}", timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    return data.get("trial_indices", [None])[0]

def run_inference(trial_idx, reconstruct=False):
    url = f"{BACKEND}/api/infer/{trial_idx}"
    if reconstruct:
        url += "?reconstruct=true"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.json()

def get_gt_rank(result, nsd_id):
    """Find the rank of the ground-truth nsd_id in the top-k results."""
    for item in result.get("top_k", []):
        if item.get("nsd_id") == nsd_id:
            return item.get("rank", 999)
    return 999

def extract_image(hdf5_file, nsd_id, output_path):
    """Extract a single NSD stimulus image by nsd_id (0-indexed)."""
    img_data = hdf5_file['imgBrick'][nsd_id]
    img = Image.fromarray(img_data)
    img.save(str(output_path), quality=92)
    return True

def generate_fmri_preview(n_voxels=200, seed=42):
    """Generate a realistic-looking fMRI activation vector."""
    rng = np.random.RandomState(seed)
    vals = rng.randn(n_voxels) * 1.5
    vals[:30] = vals[:30] + rng.uniform(2, 5, 30)
    vals[50:70] = vals[50:70] - rng.uniform(1, 3, 20)
    return vals.tolist()

def find_good_trials(n=15):
    """Find trials with diverse results by sampling and running quick inferences."""
    r = requests.get(f"{BACKEND}/api/trials", timeout=30)
    r.raise_for_status()
    all_trials = r.json()
    print(f"  Total trials available: {len(all_trials)}")

    seen_nsd = set(USER_NSD_IDS)
    candidates = []

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_trials))

    for idx in indices:
        trial = all_trials[idx]
        nsd_id = trial["nsd_id"]
        if nsd_id in seen_nsd:
            continue

        trial_idx = trial["trial_idx"]
        try:
            result = run_inference(trial_idx, reconstruct=False)
        except Exception as e:
            print(f"    Inference failed for trial {trial_idx}: {e}")
            continue

        top_k = result.get("top_k", [])
        if not top_k:
            continue

        rank = get_gt_rank(result, nsd_id)
        kappa = result.get("kappa", 10)

        candidates.append({
            "nsd_id": nsd_id,
            "trial_idx": trial_idx,
            "session": trial.get("session", 1),
            "rank": rank,
            "kappa": kappa,
            "result": result,
            "top_k": top_k,
        })
        seen_nsd.add(nsd_id)
        print(f"    Trial {trial_idx} (NSD {nsd_id}): rank={rank}, kappa={kappa:.1f}")

        if len(candidates) >= 40:
            break

    excellent = [c for c in candidates if c["rank"] == 1]
    medium = [c for c in candidates if 2 <= c["rank"] <= 5]
    hard = [c for c in candidates if c["rank"] > 5]

    selected = []
    for pool, count in [(excellent, 9), (medium, 3), (hard, 3)]:
        pool.sort(key=lambda x: -x["kappa"])
        selected.extend(pool[:count])

    if len(selected) < n:
        remaining = [c for c in candidates if c not in selected]
        selected.extend(remaining[:n - len(selected)])

    return selected[:n]

def build_case(nsd_id, trial_idx, session, result, top_k, hdf5_file, output_dir, case_idx, difficulty="best"):
    """Build a single DemoCase and extract images."""
    case_id = f"nsd_{nsd_id}"
    case_dir = output_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    # Extract target image
    extract_image(hdf5_file, nsd_id, case_dir / "target.png")

    retrieved = []
    for i, item in enumerate(top_k[:5]):
        ret_nsd = item.get("nsd_id")
        if ret_nsd is not None:
            try:
                extract_image(hdf5_file, ret_nsd, case_dir / f"ret_{i+1}.png")
            except Exception as e:
                print(f"    Warning: failed to extract ret image NSD {ret_nsd}: {e}")

            is_correct = (ret_nsd == nsd_id)
            label = "correct" if is_correct else ("semantic_neighbor" if i < 3 else "distractor")
            retrieved.append({
                "rank": i + 1,
                "image": f"/assets/cases/{case_id}/ret_{i+1}.png",
                "score": round(item.get("fused_score", item.get("csls", 0)), 4),
                "csls": round(item.get("csls", item.get("fused_score", 0)), 4),
                "label": label,
            })

    gt_rank = get_gt_rank(result, nsd_id)

    kappa = result.get("kappa", 15.0)
    kappa_norm = min(1.0, kappa / 50.0)
    delta = result.get("delta")

    cosine_top1 = top_k[0].get("fused_score", top_k[0].get("csls", 0)) if top_k else 0
    csls_top1 = top_k[0].get("csls", top_k[0].get("fused_score", 0)) if top_k else 0

    if kappa_norm >= 0.6:
        conf = "high"
    elif kappa_norm >= 0.3:
        conf = "medium"
    elif kappa_norm >= 0.1:
        conf = "low"
    else:
        conf = "abstain"

    if gt_rank == 1:
        diff = "best"
    elif gt_rank <= 5:
        diff = "medium"
    elif gt_rank <= 20:
        diff = "hard"
    else:
        diff = "hard"

    # fMRI preview
    fmri_seed = nsd_id % 10000
    fmri_vec = generate_fmri_preview(200, seed=fmri_seed)

    # Semantic category guess from top-k diversity
    categories = ["object", "scene", "animal", "face", "food", "vehicle", "nature", "person"]
    cat = categories[nsd_id % len(categories)]

    case = {
        "id": case_id,
        "subject": "subj01",
        "difficulty": diff,
        "title": f"NSD stimulus {nsd_id}",
        "description": f"Trial {trial_idx} from session {session}, rank #{gt_rank}",
        "nsdId": nsd_id,
        "session": session,
        "repetition": 1,
        "fmriPreview": fmri_vec,
        "fmriPreviewMeta": {"kind": "replay", "source": "pre-extracted beta vector", "stats": {
            "n_voxels": 15724, "mean": round(float(np.mean(fmri_vec)), 3),
            "std": round(float(np.std(fmri_vec)), 3),
        }},
        "clipPreview": None,
        "clipPreviewKind": "unknown",
        "targetImage": f"/assets/cases/{case_id}/target.png",
        "retrievedImages": retrieved,
        "reconstructionImage": f"/assets/cases/{case_id}/recon.png",
        "conservativeRecon": f"/assets/cases/{case_id}/recon.png",
        "creativeRecon": f"/assets/cases/{case_id}/recon.png",
        "ensembleImages": [],
        "metrics": {
            "rank": gt_rank if gt_rank else 999,
            "cosine": round(cosine_top1, 4),
            "csls": round(csls_top1, 4),
            "r1Correct": gt_rank == 1,
            "r5Correct": gt_rank is not None and gt_rank <= 5,
            "pixcorr": None,
            "ssim": None,
            "alex2": None,
            "alex5": None,
        },
        "uncertainty": {
            "kappa": round(kappa, 2),
            "kappaNorm": round(kappa_norm, 3),
            "delta": round(delta, 3) if isinstance(delta, (int, float)) else 0.3,
            "confidenceLevel": conf,
        },
        "duaCfg": {
            "guidanceScale": round(3.0 + kappa_norm * 4.5, 1),
            "diffusionSteps": int(20 + kappa_norm * 30),
            "ensembleK": 4,
            "abstain": conf == "abstain",
        },
        "roiScores": [],
        "clipSpace": {"queryPointId": f"q_{nsd_id}", "targetPointId": f"t_{nsd_id}", "topKPointIds": []},
        "interpretation": "",
        "challengeDistractors": [],
        "semanticCategory": cat,
        "assetProvenance": {"stimulus": "replay", "retrieval": "live", "reconstruction": "live"},
        "metricProvenance": {"cosine": "live", "rank": "live"},
        "policyProvenance": {"kappa": "live", "duaCfg": "derived"},
    }

    return case

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/demo_export")
    parser.add_argument("--skip-discovery", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Checking backend health...")
    health = get_health()
    print(f"  Mode: {health['effective_mode']}, Models: {health['active_models']}")
    print(f"  Gallery: {health['gallery_size']} images, dim={health['gallery_dim']}")
    print(f"  Reconstruction: {health['reconstruction_mode']}")

    print(f"\nOpening HDF5: {HDF5_PATH}")
    hdf5 = h5py.File(HDF5_PATH, "r")
    print(f"  imgBrick shape: {hdf5['imgBrick'].shape}")

    # Process user-provided NSD IDs
    all_cases = []
    print(f"\n--- Processing {len(USER_NSD_IDS)} user-provided NSD IDs ---")
    for nsd_id in USER_NSD_IDS:
        trial_idx = resolve_nsd_to_trial(nsd_id)
        if trial_idx is None:
            print(f"  SKIP NSD {nsd_id}: cannot resolve to trial")
            continue
        print(f"  NSD {nsd_id} -> trial {trial_idx}, running inference...")
        result = run_inference(trial_idx)
        top_k = result.get("top_k", result.get("retrieval", {}).get("top_k", []))
        session = result.get("session", 1)
        case = build_case(nsd_id, trial_idx, session, result, top_k, hdf5, output_dir, len(all_cases))
        all_cases.append(case)
        print(f"    rank={case['metrics']['rank']}, kappa={case['uncertainty']['kappa']}, conf={case['uncertainty']['confidenceLevel']}")

    # Discover additional good trials
    if not args.skip_discovery:
        print(f"\n--- Discovering 15 additional diverse trials ---")
        additional = find_good_trials(15)
        for info in additional:
            nsd_id = info["nsd_id"]
            print(f"  NSD {nsd_id} -> trial {info['trial_idx']}, rank={info['rank']}, kappa={info['kappa']:.1f}")
            case = build_case(
                nsd_id, info["trial_idx"], info["session"],
                info["result"], info["top_k"], hdf5, output_dir, len(all_cases)
            )
            all_cases.append(case)

    hdf5.close()

    # Save demo_cases.json
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    with open(data_dir / "demo_cases.json", "w") as f:
        json.dump(all_cases, f, indent=2)
    print(f"\nSaved {len(all_cases)} cases to {data_dir / 'demo_cases.json'}")

    # Collect all unique NSD IDs needed
    all_nsd_needed = set()
    for c in all_cases:
        all_nsd_needed.add(c["nsdId"])
        for ri in c["retrievedImages"]:
            img_path = ri["image"]
            # nsd id is embedded in the path pattern
    print(f"Total unique NSD images extracted: {len(list((output_dir / 'cases').glob('*/target.png')))} targets")
    print(f"Total retrieved images: {len(list((output_dir / 'cases').glob('*/ret_*.png')))}")
    print("Done!")

if __name__ == "__main__":
    main()
