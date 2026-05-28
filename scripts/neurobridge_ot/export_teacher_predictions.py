#!/usr/bin/env python3
"""
Export Teacher Predictions for NeuroBridge-OT Distillation
============================================================

Exports embeddings/scores from existing subject-specific experts
(V61a, V62a, V66a) for use in teacher-student distillation.

Usage:
    python scripts/neurobridge_ot/export_teacher_predictions.py \
        --teacher-checkpoint experimental_results/V61a/subj01/checkpoints/best.pt \
        --teacher-name V61a \
        --subject subj01 \
        --output-dir experimental_results/neurobridge_ot/teachers
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export Teacher Predictions")
    parser.add_argument("--teacher-checkpoint", type=str, required=True)
    parser.add_argument("--teacher-name", type=str, required=True,
                        help="Teacher model name (e.g., V61a, V62a, V66a)")
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "all"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / args.subject / args.teacher_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting teacher predictions: %s for %s", args.teacher_name, args.subject)

    ckpt_path = Path(args.teacher_checkpoint)
    if not ckpt_path.exists():
        logger.error("Teacher checkpoint not found: %s", ckpt_path)
        logger.info("To use distillation, ensure teacher checkpoints exist.")
        logger.info("Expected teacher artifacts:")
        logger.info("  V61a: experimental_results/V61a_finetune_difflr/{subj}/checkpoints/best.pt")
        logger.info("  V62a: experimental_results/V62a_cls_retrieval_768d/{subj}/checkpoints/best.pt")
        logger.info("  V66a: experimental_results/V66a_roi_pretrain/{subj}/checkpoints/best.pt")

        # Create placeholder manifest documenting missing artifacts
        manifest = {
            "teacher_name": args.teacher_name,
            "subject": args.subject,
            "checkpoint_path": str(ckpt_path),
            "status": "checkpoint_not_found",
            "message": "Teacher artifacts are required for distillation but were not found. "
                       "Train the teacher model first or provide a valid checkpoint path.",
        }
        with open(output_dir / "export_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return

    if args.dry_run:
        logger.info("DRY RUN — would export from %s", ckpt_path)
        return

    # Load teacher model
    from fmri2img.models.unified_model import create_model, load_model

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        teacher_config = ckpt.get("config", {}).get("model", ckpt.get("config", {}))
        model = create_model(teacher_config).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        logger.info("Loaded teacher model: %s", args.teacher_name)
    except Exception as e:
        logger.error("Failed to load teacher: %s", e)
        manifest = {
            "teacher_name": args.teacher_name,
            "subject": args.subject,
            "status": "load_failed",
            "error": str(e),
        }
        with open(output_dir / "export_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return

    # Load data
    import pandas as pd

    cache_root = Path(os.environ.get("CACHE_ROOT", "cache")) / "preextracted"
    fmri_path = cache_root / f"subject={args.subject}" / "fmri_features.npy"
    index_path = Path("data/indices/nsd_index") / f"subject={args.subject}" / "index.parquet"

    if not fmri_path.exists() or not index_path.exists():
        logger.error("Data not found for %s", args.subject)
        return

    fmri_data = np.load(fmri_path)
    index_df = pd.read_parquet(index_path)
    nsd_ids = index_df["nsdId"].values

    logger.info("Data: %d trials, %d voxels", fmri_data.shape[0], fmri_data.shape[1])

    # Run predictions in batches
    all_preds = []
    n_total = len(fmri_data)
    batch_size = args.batch_size

    with torch.no_grad():
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch = torch.from_numpy(fmri_data[start:end].astype(np.float32)).to(device)
            output = model(batch)

            if isinstance(output, tuple):
                pred = output[0]  # mu for vMF models
            elif isinstance(output, dict):
                pred = output.get("clip_embedding", output.get("mu", list(output.values())[0]))
            else:
                pred = output

            all_preds.append(pred.cpu().numpy())

            if args.limit_batches and (start // batch_size) >= args.limit_batches - 1:
                break

    predictions = np.concatenate(all_preds, axis=0)
    actual_nsd_ids = nsd_ids[:len(predictions)]

    # Save
    np.save(output_dir / "predictions.npy", predictions)
    np.save(output_dir / "nsd_ids.npy", actual_nsd_ids)

    manifest = {
        "teacher_name": args.teacher_name,
        "subject": args.subject,
        "checkpoint_path": str(ckpt_path),
        "status": "success",
        "n_predictions": len(predictions),
        "embedding_dim": predictions.shape[1],
        "prediction_path": str(output_dir / "predictions.npy"),
        "nsd_ids_path": str(output_dir / "nsd_ids.npy"),
    }
    with open(output_dir / "export_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Exported %d predictions (dim=%d) to %s",
                len(predictions), predictions.shape[1], output_dir)


if __name__ == "__main__":
    main()
