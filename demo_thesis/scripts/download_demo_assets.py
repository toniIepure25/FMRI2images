"""
Download real COCO images for all demo cases so the frontend works fully offline.

For each case:
  - target.png = actual NSD stimulus (COCO image)
  - ret_1.png  = same as target (rank-1 correct retrieval)
  - ret_2..5   = other COCO images as semantic neighbors/distractors
"""
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parent.parent
CASES_JSON = DEMO_ROOT / "public" / "data" / "demo_cases.json"
OUTPUT_DIR = DEMO_ROOT / "public" / "assets" / "cases"

NSD_STIM_INFO_URL = (
    "https://natural-scenes-dataset.s3.amazonaws.com/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
)

COCO_TRAIN_URL = "http://images.cocodataset.org/train2017/{:012d}.jpg"
COCO_VAL_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"


def download_file(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download a file, return True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.read())
        return True
    except Exception as e:
        print(f"  WARN: Failed to download {url}: {e}")
        return False


def get_nsd_to_coco_mapping(nsd_ids: set[int]) -> dict[int, tuple[int, str]]:
    """Download NSD stim info and extract nsdId -> (cocoId, cocoSplit) mapping."""
    print("Downloading NSD stimulus info CSV...")
    req = urllib.request.Request(NSD_STIM_INFO_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")

    mapping: dict[int, tuple[int, str]] = {}
    reader = csv.DictReader(io.StringIO(raw))
    for row in reader:
        nsd_id = int(row["nsdId"])
        if nsd_id in nsd_ids:
            coco_id = int(row["cocoId"])
            coco_split = row.get("cocoSplit", "train2017")
            mapping[nsd_id] = (coco_id, coco_split)
    return mapping


def get_coco_url(coco_id: int, split: str) -> str:
    if "val" in split:
        return COCO_VAL_URL.format(coco_id)
    return COCO_TRAIN_URL.format(coco_id)


def pick_distractor_nsd_ids(all_nsd_ids: list[int], target_nsd_id: int, count: int = 4) -> list[int]:
    """Pick `count` other NSD IDs to use as distractors for a given case."""
    others = [n for n in all_nsd_ids if n != target_nsd_id]
    # Rotate based on target to get different distractors for each case
    idx = all_nsd_ids.index(target_nsd_id) if target_nsd_id in all_nsd_ids else 0
    result = []
    for i in range(count):
        pick_idx = (idx + i + 1) % len(others)
        result.append(others[pick_idx])
    return result


def main():
    print("=" * 60)
    print("Cortex2Canvas - Download Real Demo Assets")
    print("=" * 60)

    # Load cases
    cases = json.loads(CASES_JSON.read_text(encoding="utf-8-sig"))
    print(f"Loaded {len(cases)} cases from demo_cases.json")

    # Collect all NSD IDs needed
    case_nsd_ids = [c["nsdId"] for c in cases]
    all_nsd_ids_needed = set(case_nsd_ids)

    # We also need distractor images - pick extras from the NSD catalog
    # Use a fixed set of known good NSD IDs as distractors pool
    distractor_pool_extra = [
        1000, 2000, 3000, 4000, 5000, 6000, 7000, 9000, 10000,
        11000, 12000, 15000, 18000, 20000, 25000, 30000, 35000,
        40000, 45000, 50000, 52000, 54000, 56000, 58000, 62000,
        64000, 66000, 68000, 71000, 72000
    ]
    all_nsd_ids_needed.update(distractor_pool_extra)

    # Download NSD -> COCO mapping
    mapping = get_nsd_to_coco_mapping(all_nsd_ids_needed)
    print(f"Resolved {len(mapping)} NSD->COCO mappings")

    # Check which target IDs we got
    missing_targets = [nid for nid in case_nsd_ids if nid not in mapping]
    if missing_targets:
        print(f"WARNING: No COCO mapping for NSD IDs: {missing_targets}")

    # Build distractor pool (mapped NSD IDs that aren't targets)
    distractor_pool = [nid for nid in distractor_pool_extra if nid in mapping]
    print(f"Distractor pool: {len(distractor_pool)} images available")

    # Download all needed images
    success_count = 0
    fail_count = 0

    for ci, case in enumerate(cases):
        nsd_id = case["nsdId"]
        case_folder_name = f"nsd_{nsd_id}"

        # Check if image paths in JSON point to a different folder
        target_path = case.get("targetImage", "")
        if "/assets/cases/" in target_path:
            case_folder_name = target_path.split("/assets/cases/")[1].split("/")[0]

        case_dir = OUTPUT_DIR / case_folder_name
        case_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{ci+1}/{len(cases)}] {case_folder_name} (NSD {nsd_id})")

        # Download target image
        if nsd_id in mapping:
            coco_id, split = mapping[nsd_id]
            url = get_coco_url(coco_id, split)
            target_dest = case_dir / "target.png"
            # Download as jpg first, then we'll keep as-is (browsers handle jpg fine even with .png extension)
            # Actually let's save as .jpg and also make .png version
            jpg_dest = case_dir / "target.jpg"
            if download_file(url, jpg_dest):
                # Copy jpg as png (browsers will render jpg data in .png file fine)
                if not target_dest.exists():
                    import shutil
                    shutil.copy2(jpg_dest, target_dest)
                print(f"  target.png OK (COCO {coco_id})")
                success_count += 1
            else:
                fail_count += 1

            # ret_1 = same as target for rank-1 cases
            rank = case.get("metrics", {}).get("rank", 1)
            ret1_dest = case_dir / "ret_1.png"
            if rank == 1 and target_dest.exists() and not ret1_dest.exists():
                import shutil
                shutil.copy2(target_dest, ret1_dest)
                print(f"  ret_1.png = target (rank 1 correct)")
                success_count += 1
            elif not ret1_dest.exists():
                # For non-rank-1 cases, ret_1 is a different image
                if distractor_pool:
                    d_nsd = distractor_pool[ci % len(distractor_pool)]
                    d_coco_id, d_split = mapping[d_nsd]
                    d_url = get_coco_url(d_coco_id, d_split)
                    d_jpg = case_dir / "ret_1.jpg"
                    if download_file(d_url, d_jpg):
                        import shutil
                        shutil.copy2(d_jpg, ret1_dest)
                        success_count += 1
        else:
            print(f"  SKIP - no COCO mapping for NSD {nsd_id}")
            fail_count += 1

        # Download ret_2 through ret_5 (distractors/neighbors)
        for ret_idx in range(2, 6):
            ret_dest = case_dir / f"ret_{ret_idx}.png"
            if ret_dest.exists() and ret_dest.stat().st_size > 0:
                success_count += 1
                continue

            # Pick a distractor
            pool_idx = (ci * 4 + ret_idx - 2) % len(distractor_pool) if distractor_pool else 0
            if pool_idx < len(distractor_pool):
                d_nsd = distractor_pool[pool_idx]
                if d_nsd in mapping:
                    d_coco_id, d_split = mapping[d_nsd]
                    d_url = get_coco_url(d_coco_id, d_split)
                    d_jpg = case_dir / f"ret_{ret_idx}.jpg"
                    if download_file(d_url, d_jpg):
                        import shutil
                        shutil.copy2(d_jpg, ret_dest)
                        print(f"  ret_{ret_idx}.png OK")
                        success_count += 1
                    else:
                        fail_count += 1
                else:
                    fail_count += 1
            else:
                fail_count += 1

    print(f"\n{'=' * 60}")
    print(f"Done! Success: {success_count}, Failed: {fail_count}")
    print(f"Assets in: {OUTPUT_DIR}")
    print("=" * 60)

    # Clean up .jpg duplicates
    for jpg in OUTPUT_DIR.rglob("*.jpg"):
        png_sibling = jpg.with_suffix(".png")
        if png_sibling.exists():
            jpg.unlink()


if __name__ == "__main__":
    main()
