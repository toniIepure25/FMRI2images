"""
Build NSD -> COCO category mapping cache.

Creates a parquet file mapping each nsdId to its dominant COCO category and supercategory.
For multi-object images, selects the category with the largest total bounding box area.

Output: cache/nsd_category_labels.parquet
"""
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NSD_DATA_ROOT = Path(os.environ.get("NSD_DATA_ROOT", "/home/jovyan/work/data/nsd"))
COCO_ROOT = Path(os.environ.get("COCO_ROOT", "/home/jovyan/work/data/coco"))
CACHE_ROOT = Path(os.environ.get("CACHE_ROOT", "/home/jovyan/work/FMRI2images/cache"))

OUTPUT_PATH = CACHE_ROOT / "nsd_category_labels.parquet"


def load_coco_annotations():
    """Load COCO instances annotations and build image_id -> category info."""
    annotations_dir = COCO_ROOT / "annotations"
    
    # COCO category definitions (same across train/val)
    categories = {}
    supercategories = {}
    
    # Image-level category accumulator: image_id -> {category_id: total_area}
    image_categories = defaultdict(lambda: defaultdict(float))
    
    for split in ["train2017", "val2017"]:
        ann_file = annotations_dir / f"instances_{split}.json"
        if not ann_file.exists():
            logger.warning("Missing: %s", ann_file)
            continue
        
        logger.info("Loading %s...", ann_file)
        with open(ann_file) as f:
            data = json.load(f)
        
        # Build category lookup
        for cat in data["categories"]:
            categories[cat["id"]] = cat["name"]
            supercategories[cat["id"]] = cat["supercategory"]
        
        # Accumulate annotations by image
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            area = ann.get("area", 0)
            image_categories[img_id][cat_id] += area
        
        logger.info("  Loaded %d annotations for %d images", 
                   len(data["annotations"]), len(data["images"]))
    
    logger.info("Total images with annotations: %d", len(image_categories))
    logger.info("Categories: %d, Supercategories: %d unique", 
               len(categories), len(set(supercategories.values())))
    
    return image_categories, categories, supercategories


def build_nsd_to_category_mapping(image_categories, categories, supercategories):
    """Map nsdId -> dominant COCO category via cocoId."""
    # Load NSD stim info
    stim_info_path = NSD_DATA_ROOT / "nsddata" / "experiments" / "nsd" / "nsd_stim_info_merged.csv"
    logger.info("Loading stim info: %s", stim_info_path)
    
    stim_df = pd.read_csv(stim_info_path)
    logger.info("Stim info: %d entries, columns: %s", len(stim_df), stim_df.columns.tolist()[:10])
    
    # nsdId is 0-indexed in the CSV (row index or explicit column)
    if "nsdId" not in stim_df.columns:
        stim_df["nsdId"] = stim_df.index
    
    records = []
    n_matched = 0
    n_no_annotation = 0
    
    for _, row in stim_df.iterrows():
        nsd_id = int(row["nsdId"])
        coco_id = int(row["cocoId"]) if pd.notna(row.get("cocoId")) else None
        coco_split = row.get("cocoSplit", "")
        
        if coco_id is None or coco_id not in image_categories:
            n_no_annotation += 1
            records.append({
                "nsdId": nsd_id,
                "cocoId": coco_id,
                "category_id": -1,
                "category_name": "unknown",
                "supercategory": "unknown",
                "dominant_area_fraction": 0.0,
                "n_objects": 0,
                "n_categories": 0,
            })
            continue
        
        # Find dominant category (largest total area)
        cat_areas = image_categories[coco_id]
        total_area = sum(cat_areas.values())
        dominant_cat_id = max(cat_areas, key=cat_areas.get)
        dominant_area = cat_areas[dominant_cat_id]
        
        records.append({
            "nsdId": nsd_id,
            "cocoId": coco_id,
            "category_id": dominant_cat_id,
            "category_name": categories.get(dominant_cat_id, "unknown"),
            "supercategory": supercategories.get(dominant_cat_id, "unknown"),
            "dominant_area_fraction": dominant_area / total_area if total_area > 0 else 0,
            "n_objects": sum(1 for _ in cat_areas.values()),
            "n_categories": len(cat_areas),
        })
        n_matched += 1
    
    logger.info("Matched: %d, No annotation: %d", n_matched, n_no_annotation)
    
    df = pd.DataFrame(records)
    return df


def main():
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Load COCO annotations
    image_categories, categories, supercategories = load_coco_annotations()
    
    # Build mapping
    df = build_nsd_to_category_mapping(image_categories, categories, supercategories)
    
    # Save
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Saved: %s (%d rows)", OUTPUT_PATH, len(df))
    
    # Summary statistics
    valid = df[df["category_name"] != "unknown"]
    logger.info("\n=== Category Distribution (supercategories) ===")
    supercat_counts = valid["supercategory"].value_counts()
    for cat, count in supercat_counts.items():
        logger.info("  %-15s: %5d (%.1f%%)", cat, count, 100 * count / len(valid))
    
    logger.info("\n=== Top 20 Fine Categories ===")
    finecat_counts = valid["category_name"].value_counts().head(20)
    for cat, count in finecat_counts.items():
        logger.info("  %-20s: %5d", cat, count)
    
    logger.info("\n=== Summary ===")
    logger.info("Total NSD images: %d", len(df))
    logger.info("With COCO annotations: %d (%.1f%%)", len(valid), 100 * len(valid) / len(df))
    logger.info("Unique supercategories: %d", valid["supercategory"].nunique())
    logger.info("Unique fine categories: %d", valid["category_name"].nunique())
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
