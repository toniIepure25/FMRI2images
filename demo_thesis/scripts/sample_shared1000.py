#!/usr/bin/env python3
"""Sample SHARED1000 NSD IDs to find the best demo cases."""
import numpy as np, json, requests

BACKEND = "http://localhost:8000"
ids = np.load("/home/jovyan/work/FMRI2images/experimental_results/V62a_cls_retrieval_768d/subj01/metrics/shared1000_nsd_ids.npy")
user_ids = {8262, 21279, 55649, 8509}

rng = np.random.RandomState(42)
available = [int(x) for x in ids if int(x) not in user_ids]
rng.shuffle(available)

results = []
for nsd_id in available[:50]:
    try:
        r = requests.get(f"{BACKEND}/api/infer-by-nsd/{nsd_id}", timeout=30).json()
        gt_rank = 999
        for item in r.get("top_k", []):
            if item.get("nsd_id") == nsd_id:
                gt_rank = item.get("rank", 999)
                break
        kappa = r.get("kappa", 0)
        results.append({"nsd_id": nsd_id, "rank": gt_rank, "kappa": round(kappa, 2)})
        status = "rank-1" if gt_rank == 1 else f"rank-{gt_rank}"
        print(f"NSD {nsd_id}: {status}, kappa={kappa:.1f}")
    except Exception as e:
        print(f"NSD {nsd_id}: FAILED {e}")

r1 = sum(1 for r in results if r["rank"] == 1)
r5 = sum(1 for r in results if r["rank"] <= 5)
print(f"\nSample R@1: {r1}/{len(results)} ({100*r1/len(results):.0f}%)")
print(f"Sample R@5: {r5}/{len(results)} ({100*r5/len(results):.0f}%)")

excellent = sorted([r for r in results if r["rank"] == 1], key=lambda x: -x["kappa"])
medium = sorted([r for r in results if 2 <= r["rank"] <= 5], key=lambda x: x["rank"])
hard = sorted([r for r in results if r["rank"] > 5], key=lambda x: x["rank"])

selected = excellent[:10] + medium[:3] + hard[:2]
print(f"\nSelected {len(selected)} for demo:")
for s in selected:
    print(f"  NSD {s['nsd_id']}: rank={s['rank']}, kappa={s['kappa']}")
print(f"SELECTED_IDS: {json.dumps([s['nsd_id'] for s in selected])}")
