"""
Unified Training Script for Research Experiments
================================================

Publication-grade training script with:
- Per-subject data loading (index, ROI mask, preprocessing)
- Reproducible seeding
- Structured output directory per experiment x subject
- Per-epoch CSV metrics + summary JSON
- Run manifest with system info and git commit
- Best + last checkpointing with optional periodic saves
- vMF-NCE-SPCL and MultiTask losses for N3/N4 experiments
- Gradient accumulation and mixed precision

Usage:
    python scripts/training/train_unified.py --config configs/experiments/B0_deterministic.yaml --gpu 0
    python scripts/training/train_unified.py --config configs/experiments/N1_vmf_nce.yaml --subject subj02 --gpu 0
"""

import argparse
import csv
import json
import logging
import math
import hashlib
import os
import platform
import random
import shutil
import subprocess
import sys
import gc
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from fmri2img.models.unified_model import create_model
from fmri2img.embedding_preproc import EmbeddingPreprocessor
from fmri2img.contrastive.queue import create_memory_queue
from fmri2img.losses.infonce_queue import InfoNCEQueueLoss
from fmri2img.losses.gaussian_nll import GaussianNLLLoss
from fmri2img.losses.gaussian_nce import GaussianNCELoss
from fmri2img.losses.vmf_nce import (
    VonMisesFisherNCELoss,
    MixtureVonMisesFisherNCELoss,
    MixtureComponentDiversityLoss,
    VonMisesFisherNLLLoss,
    KappaSPCLVMFNCELoss,
    DeltaSPCLVMFNCELoss,
    MultiTaskVMFNCELoss,
    kappa_regularizer,
    vmf_rdrop_loss,
)
from fmri2img.training.kl_schedule import KLScheduler
from fmri2img.losses.mixco import mixco_augment, mixco_nce_loss
from fmri2img.losses.softclip import SoftCLIPLoss, VMFSoftCLIPLoss
from fmri2img.losses.legacy_compact_distill import LegacyCompactDistillLoss
from fmri2img.losses.component_legacy_compact_distill import ComponentLegacyCompactDistillLoss
from fmri2img.losses.legacy_teacher_distill import LegacyTeacherDistillLoss
from fmri2img.losses.shortlist_teacher_distill import ShortlistTeacherDistillLoss
from fmri2img.losses.tri_teacher_distill import TriTeacherDistillLoss
from fmri2img.losses.hierarchical_clip_loss import HierarchicalCLIPLoss
from fmri2img.losses.cka_loss import CKALoss
from fmri2img.losses.direct_alignment import DirectAlignmentLoss
from fmri2img.losses.uniformity import UniformityLoss
from fmri2img.eval.embedding_eval import (
    compute_retrieval_metrics as _compute_retrieval,
    compute_retrieval_metrics_csls as _compute_retrieval_csls,
    compute_mixture_vmf_retrieval_metrics as _compute_mixture_vmf_retrieval,
    compute_mixture_vmf_retrieval_metrics_csls as _compute_mixture_vmf_retrieval_csls,
    compute_mixture_component_diagnostics as _compute_mixture_component_diagnostics,
    score_mixture_vmf_gallery as _score_mixture_vmf_gallery,
    csls_from_score_matrix as _csls_from_score_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exponential Moving Average (EMA) for model weights
# ---------------------------------------------------------------------------

class ModelEMA:
    """Maintains exponential moving average of model parameters for smoother
    validation metrics and better generalization."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """Swap model weights with EMA shadow weights for evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original model weights after evaluation."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def build_manifest(config: Dict[str, Any], args: argparse.Namespace,
                   model: nn.Module, device: str) -> Dict[str, Any]:
    """Build a run manifest with system info, config hash, and model stats."""
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    gpu_name = "N/A"
    if device.startswith("cuda"):
        try:
            gpu_name = torch.cuda.get_device_name(int(device.split(":")[-1]))
        except Exception:
            pass

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "config_path": str(args.config),
        "subject": args.subject or config.get("data", {}).get("subject", "subj01"),
        "experiment": config.get("experiment", {}).get("name", "unknown"),
        "seed": config.get("data", {}).get("seed", 42),
        "model_params_total": n_params,
        "model_params_trainable": n_trainable,
        "system": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda or "N/A",
            "gpu": gpu_name,
            "git_commit": _git_commit(),
        },
    }


# ---------------------------------------------------------------------------
# Metrics logger
# ---------------------------------------------------------------------------

class MetricsLogger:
    """Writes per-epoch metrics to CSV and accumulates for summary JSON."""

    def __init__(self, output_dir: Path):
        self.metrics_dir = output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.metrics_dir / "training_log.csv"
        self._all_fields: List[str] = []
        self._history: List[Dict[str, Any]] = []

    def log_epoch(self, epoch: int, lr: float,
                  train_metrics: Dict[str, float],
                  val_metrics: Dict[str, float]) -> None:
        row = {"epoch": epoch, "lr": lr}
        for k, v in train_metrics.items():
            row[f"train_{k}"] = v
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v
        self._history.append(row)

        new_keys = [k for k in row if k not in self._all_fields]
        if new_keys:
            self._all_fields.extend(new_keys)
            self._rewrite_csv()
        else:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, self._all_fields, extrasaction="ignore")
                writer.writerow(row)

    def _rewrite_csv(self) -> None:
        """Rewrite the full CSV when new columns appear."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, self._all_fields, extrasaction="ignore")
            writer.writeheader()
            for row in self._history:
                writer.writerow(row)

    def write_summary(self, best_epoch: int, best_val_loss: float,
                      wall_time_s: float, manifest: Dict[str, Any],
                      best_r1: float = 0.0, best_metric: float = 0.0,
                      checkpoint_metric: str = "r@1") -> None:
        summary = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_r@1": best_r1,
            "checkpoint_metric": checkpoint_metric,
            "best_checkpoint_metric_value": best_metric,
            "total_epochs": len(self._history),
            "wall_time_seconds": round(wall_time_s, 1),
            "manifest": manifest,
        }
        if self._history:
            summary["final_train_loss"] = self._history[-1].get("train_loss")
            summary["final_val_loss"] = self._history[-1].get("val_loss")
            summary["final_r@1"] = self._history[-1].get("val_r@1")
        with open(self.metrics_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else None


def _merge_summary_metrics(metrics_dir: Path) -> None:
    """Backfill high-signal post-training metrics into summary.json."""
    summary_path = metrics_dir / "summary.json"
    summary = _load_json_if_exists(summary_path)
    if summary is None:
        return

    updates: Dict[str, Any] = {}

    val_fused = _load_json_if_exists(metrics_dir / "val_fused_metrics.json")
    if val_fused is not None:
        updates["best_fused_r@1"] = float(val_fused.get("fused", {}).get("fused_r@1", 0.0))
        tri_best = val_fused.get("tri_fused_best") or val_fused.get("tri_fused_frozen")
        if isinstance(tri_best, dict):
            updates["best_tri_fused_r@1"] = float(tri_best.get("R@1", 0.0))

    shared_compact = (
        _load_json_if_exists(metrics_dir / "shared1000_metrics_compact.json")
        or _load_json_if_exists(metrics_dir / "shared1000_metrics.json")
    )
    if shared_compact is not None:
        updates["final_compact_csls_r@1"] = float(shared_compact.get("csls_r@1", 0.0))
        if "mixture_r@1" in shared_compact:
            updates["final_mixture_r@1"] = float(shared_compact.get("mixture_r@1", 0.0))
        if "mixture_csls_r@1" in shared_compact:
            updates["final_mixture_csls_r@1"] = float(shared_compact.get("mixture_csls_r@1", 0.0))

    shared_two_stage = _load_json_if_exists(metrics_dir / "shared1000_two_stage_rerank.json")
    if shared_two_stage is not None:
        updates["final_rerank_only_r@1"] = float(
            shared_two_stage.get("rerank_only", {}).get("rerank_r@1", 0.0)
        )

    shared_fused = _load_json_if_exists(metrics_dir / "shared1000_fused_metrics.json")
    if shared_fused is not None:
        updates["final_shared1000_fused_r@1"] = float(
            shared_fused.get("fused", {}).get("fused_r@1", 0.0)
        )
        tri_best = shared_fused.get("tri_fused_best") or shared_fused.get("tri_fused_frozen")
        if isinstance(tri_best, dict):
            updates["final_shared1000_tri_fused_r@1"] = float(tri_best.get("R@1", 0.0))

    if not updates:
        return

    summary.update(updates)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)


def _metric_lower_is_better(metric_name: str) -> bool:
    metric_name = str(metric_name)
    return (
        metric_name == "median_rank"
        or metric_name.endswith("median_rank")
        or metric_name == "loss"
        or metric_name.endswith("loss")
    )


def _sanitize_metric_name(metric_name: str) -> str:
    safe = []
    for ch in str(metric_name):
        if ch.isalnum():
            safe.append(ch)
        else:
            safe.append("_")
    sanitized = "".join(safe).strip("_")
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized or "metric"


def _aggregate_rows_by_nsd_id(
    values: np.ndarray,
    nsd_ids: np.ndarray,
    *,
    reduce: str = "mean",
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate row-aligned arrays from trial level to unique-image level."""
    values = np.asarray(values)
    nsd_ids = np.asarray(nsd_ids, dtype=np.int32)
    unique_ids = np.unique(nsd_ids)
    out_shape = (len(unique_ids),) + values.shape[1:]
    aggregated = np.zeros(out_shape, dtype=np.float32)

    for i, uid in enumerate(unique_ids):
        mask = nsd_ids == uid
        if reduce == "first":
            aggregated[i] = values[mask][0]
        elif reduce == "mean":
            aggregated[i] = values[mask].mean(axis=0)
        else:
            raise ValueError(f"Unknown reduce mode: {reduce}")

    if normalize and aggregated.ndim >= 2:
        aggregated = aggregated / np.maximum(
            np.linalg.norm(aggregated, axis=-1, keepdims=True),
            1e-8,
        )
    return aggregated.astype(np.float32), unique_ids.astype(np.int32)


def _aggregate_component_outputs_by_nsd_id(
    component_mu: np.ndarray,
    component_kappa: np.ndarray,
    component_logits: Optional[np.ndarray],
    nsd_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    unique_ids = np.unique(nsd_ids)
    n_img = len(unique_ids)
    n_comp = component_mu.shape[1]
    dim = component_mu.shape[2]
    agg_mu = np.zeros((n_img, n_comp, dim), dtype=np.float32)
    agg_kappa = np.zeros((n_img, n_comp), dtype=np.float32)
    agg_logits = np.zeros((n_img, n_comp), dtype=np.float32) if component_logits is not None else None
    for i, uid in enumerate(unique_ids):
        mask = nsd_ids == uid
        mu_i = component_mu[mask].mean(axis=0)
        agg_mu[i] = mu_i / np.maximum(np.linalg.norm(mu_i, axis=-1, keepdims=True), 1e-8)
        agg_kappa[i] = component_kappa[mask].mean(axis=0)
        if agg_logits is not None and component_logits is not None:
            agg_logits[i] = component_logits[mask].mean(axis=0)
    return agg_mu, agg_kappa, agg_logits, unique_ids.astype(np.int32)


def _save_compact_component_arrays(
    metrics_dir: Path,
    prefix: str,
    component_mu: Optional[np.ndarray],
    component_kappa: Optional[np.ndarray],
    component_logits: Optional[np.ndarray],
) -> None:
    if component_mu is not None:
        np.save(metrics_dir / f"{prefix}_predictions_compact_component_mu.npy", component_mu)
    if component_kappa is not None:
        np.save(metrics_dir / f"{prefix}_predictions_compact_component_kappa.npy", component_kappa)
    if component_logits is not None:
        np.save(metrics_dir / f"{prefix}_predictions_compact_component_logits.npy", component_logits)


def _save_aux_vmf_arrays(
    metrics_dir: Path,
    prefix: str,
    vmf_preds: Optional[np.ndarray],
    vmf_kappas: Optional[np.ndarray],
) -> None:
    if vmf_preds is not None:
        np.save(metrics_dir / f"{prefix}_predictions_vmf.npy", vmf_preds)
    if vmf_kappas is not None:
        np.save(metrics_dir / f"{prefix}_kappas_vmf.npy", vmf_kappas)


def _extract_vmf_outputs_for_losses(
    model_type: str,
    pred: torch.Tensor,
    aux: Any,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return the vMF prediction/kappa pair used by vMF-style losses.

    For classic vMF models, the primary retrieval prediction is itself the vMF
    mean direction. For dense-vMF hybrid models, the dense prediction remains
    primary while the auxiliary vMF branch is used only for confidence losses
    and diagnostics.
    """
    if model_type == "dense_vmf_hybrid":
        if isinstance(aux, dict):
            vmf_mu = aux.get("vmf_mu")
            vmf_kappa = aux.get("kappa", aux.get("concentration"))
            if torch.is_tensor(vmf_mu) and torch.is_tensor(vmf_kappa):
                return vmf_mu, vmf_kappa
        return None, None

    if model_type in ("vmf", "vmf_dcf", "vmf_triple"):
        if isinstance(aux, dict):
            vmf_kappa = aux.get("kappa", aux.get("concentration"))
            if torch.is_tensor(vmf_kappa):
                return pred, vmf_kappa
            return None, None
        if torch.is_tensor(aux):
            return pred, aux
    return None, None


def _deterministic_component_jitter_like(
    tensor: torch.Tensor,
    key: str,
    scale: float,
) -> torch.Tensor:
    if scale <= 0:
        return torch.zeros_like(tensor)
    seed_bytes = hashlib.sha256(key.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, byteorder="little", signed=False)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
    noise = noise.to(device=tensor.device, dtype=tensor.dtype)
    return noise * scale


def _adapt_pretrained_tensor_for_model(
    key: str,
    source_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Adapt selected pretrained tensors across decoder shape upgrades.

    For multi-hypothesis compact heads we tile the legacy single-head weights,
    but add a tiny deterministic jitter so the optimizer does not start in a
    perfectly symmetric 4x-clone solution.
    """
    if not torch.is_tensor(source_tensor):
        return None

    def _repeat_with_jitter(source: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
        if target.shape[0] % source.shape[0] != 0:
            return None
        repeat = target.shape[0] // source.shape[0]
        if source.ndim == 2:
            repeated = source.repeat(repeat, 1)
        elif source.ndim == 1:
            repeated = source.repeat(repeat)
        else:
            return None
        base_scale = float(source.detach().float().std().item())
        if not math.isfinite(base_scale):
            base_scale = 1e-3
        jitter_scale = max(base_scale, 1e-3) * 1e-3
        return repeated + _deterministic_component_jitter_like(repeated, key, jitter_scale)

    def _copy_if_same_tail(source: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
        if source.shape == target.shape:
            return source
        if source.ndim == target.ndim == 2 and source.shape[1] == target.shape[1]:
            if source.shape[0] == target.shape[0]:
                return source
        if source.ndim == target.ndim == 1 and source.shape[0] == target.shape[0]:
            return source
        return None

    if key == "decoder.retrieval_mu_head.weight":
        if source_tensor.ndim == 2 and target_tensor.ndim == 2 and source_tensor.shape[1] == target_tensor.shape[1]:
            return _repeat_with_jitter(source_tensor, target_tensor)
    elif key == "decoder.retrieval_mu_head.bias":
        if source_tensor.ndim == 1 and target_tensor.ndim == 1:
            return _repeat_with_jitter(source_tensor, target_tensor)
    elif key == "decoder.retrieval_kappa_head.weight":
        if source_tensor.ndim == 2 and target_tensor.ndim == 2 and source_tensor.shape[1] == target_tensor.shape[1]:
            return _repeat_with_jitter(source_tensor, target_tensor)
    elif key == "decoder.retrieval_kappa_head.bias":
        if source_tensor.ndim == 1 and target_tensor.ndim == 1:
            return _repeat_with_jitter(source_tensor, target_tensor)

    return None


def _prepare_compatible_pretrained_state_dict(
    model: nn.Module,
    source_state_dict: Dict[str, torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], list[str], list[str], list[str], list[str]]:
    """Filter/adapt a source state dict so load_state_dict(strict=False) is safe.

    Returns
    -------
    compatible_state_dict
        Keys ready to pass to load_state_dict.
    matched_keys
        Keys copied without adaptation.
    adapted_keys
        Keys copied after an explicit shape adaptation.
    initialized_keys
        Keys that do not exist in the source checkpoint but are deliberately
        initialized here. For multi-hypothesis logits we now keep the model's
        default random init to break symmetry, so this is typically empty.
    skipped_shape_keys
        Keys present in both states but skipped because shapes were incompatible
        and no principled adaptation rule exists.
    """
    model_state = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    matched: list[str] = []
    adapted: list[str] = []
    initialized: list[str] = []
    skipped_shape: list[str] = []

    for key, source_tensor in source_state_dict.items():
        target_tensor = model_state.get(key)
        if target_tensor is None:
            continue
        if source_tensor.shape == target_tensor.shape:
            compatible[key] = source_tensor
            matched.append(key)
            continue
        adapted_tensor = _adapt_pretrained_tensor_for_model(key, source_tensor, target_tensor)
        if adapted_tensor is not None and adapted_tensor.shape == target_tensor.shape:
            compatible[key] = adapted_tensor.to(dtype=target_tensor.dtype)
            adapted.append(key)
        else:
            skipped_shape.append(key)

    alias_map = {
        "decoder.dense_head.weight": "decoder.retrieval_mu_head.weight",
        "decoder.dense_head.bias": "decoder.retrieval_mu_head.bias",
        "decoder.vmf_mu_head.weight": "decoder.retrieval_mu_head.weight",
        "decoder.vmf_mu_head.bias": "decoder.retrieval_mu_head.bias",
        "decoder.vmf_kappa_head.weight": "decoder.retrieval_kappa_head.weight",
        "decoder.vmf_kappa_head.bias": "decoder.retrieval_kappa_head.bias",
        "decoder.rerank_head.weight": "decoder.rerank_head.weight",
        "decoder.rerank_head.bias": "decoder.rerank_head.bias",
        "decoder.regression_head.weight": "decoder.regression_head.weight",
        "decoder.regression_head.bias": "decoder.regression_head.bias",
    }
    for target_key, source_key in alias_map.items():
        if target_key in compatible or target_key not in model_state:
            continue
        source_tensor = source_state_dict.get(source_key)
        if source_tensor is None:
            continue
        target_tensor = model_state[target_key]
        if source_tensor.shape == target_tensor.shape:
            compatible[target_key] = source_tensor.to(dtype=target_tensor.dtype)
            adapted.append(target_key)

    # Keep newly introduced mixture-logit parameters at the decoder's random
    # init rather than zeroing them out; a non-symmetric start is important for
    # multi-hypothesis specialization.

    return compatible, matched, adapted, initialized, skipped_shape


# ---------------------------------------------------------------------------
# Embedding column resolution
# ---------------------------------------------------------------------------

_EMBEDDING_COL_PRIORITY = [
    "fused", "final", "embedding", "clip_embedding", "clip512",
]


def resolve_embedding_column(
    df: pd.DataFrame,
    override: Optional[str] = None,
) -> str:
    """Pick the best embedding column from a DataFrame.

    Args:
        df:       Embeddings DataFrame.
        override: Explicit column name from config (``data.embedding_column``).
                  Takes precedence when set and present in *df*.

    Returns:
        The selected column name.

    Raises:
        ValueError: If no suitable column can be found.
    """
    if override and override in df.columns:
        return override
    for col in _EMBEDDING_COL_PRIORITY:
        if col in df.columns:
            return col
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        emb_cols = [c for c in df.columns if c.startswith("embedding_")]
    if emb_cols:
        return emb_cols[0]
    raise ValueError(
        f"No embedding column found. Available: {list(df.columns)}"
    )


# Global that will be set by main() after config is loaded
_EMBEDDING_COLUMN_OVERRIDE: Optional[str] = None

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NSDDataset(Dataset):
    """Dataset for NSD fMRI and CLIP embeddings (fallback when pre-extracted features unavailable)."""

    MAX_CACHED_SESSIONS = 5

    def __init__(
        self,
        index_df: pd.DataFrame,
        embeddings_df: pd.DataFrame,
        roi_mask_path: Optional[Path] = None,
    ):
        from collections import OrderedDict

        self.index_df = index_df.reset_index(drop=True)
        self.embeddings_df = embeddings_df
        self.roi_mask = None
        self.beta_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        self._s3_fs = None

        if "nsdId" in embeddings_df.columns:
            self.embedding_lookup = {
                row["nsdId"]: i for i, (_, row) in enumerate(embeddings_df.iterrows())
            }
        else:
            self.embedding_lookup = {i: i for i in range(len(embeddings_df))}

        if roi_mask_path and roi_mask_path.exists():
            import nibabel as nib
            mask_img = nib.load(roi_mask_path)
            mask_data = mask_img.get_fdata()
            self.roi_mask = mask_data > 0.5
            logger.info("Loaded ROI mask: %d voxels", int(self.roi_mask.sum()))

        logger.info("NSDDataset created: %d trials (LRU cache: %d sessions)", len(self.index_df), self.MAX_CACHED_SESSIONS)

    @property
    def s3_fs(self):
        if self._s3_fs is None:
            import s3fs
            self._s3_fs = s3fs.S3FileSystem(
                key=os.environ.get("S3_ACCESS_KEY", ""),
                secret=os.environ.get("S3_SECRET_KEY", ""),
                client_kwargs={
                    "endpoint_url": os.environ.get(
                        "S3_ENDPOINT", "http://s3hub-intern.cs.ubbcluj.ro:9000"
                    )
                },
            )
        return self._s3_fs

    def __len__(self) -> int:
        return len(self.index_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.index_df.iloc[idx]
        beta_path = row["beta_path"]
        beta_idx = row.get("beta_index", row.get("volume_index", 0))

        if beta_path not in self.beta_cache:
            import nibabel as nib

            if beta_path.startswith("s3://"):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=True) as tmp:
                    with self.s3_fs.open(beta_path, "rb") as f_in:
                        tmp.write(f_in.read())
                        tmp.flush()
                    img = nib.load(tmp.name)
                    self.beta_cache[beta_path] = img.get_fdata(dtype=np.float32)
            else:
                img = nib.load(beta_path)
                self.beta_cache[beta_path] = img.get_fdata(dtype=np.float32)

            while len(self.beta_cache) > self.MAX_CACHED_SESSIONS:
                self.beta_cache.popitem(last=False)
        else:
            self.beta_cache.move_to_end(beta_path)

        beta_vol = self.beta_cache[beta_path][..., beta_idx]

        if self.roi_mask is not None:
            fmri = beta_vol[self.roi_mask].flatten()
        else:
            fmri = beta_vol.flatten()

        nsdId = row["nsdId"]
        emb_idx = self.embedding_lookup.get(nsdId)
        if emb_idx is None:
            raise KeyError(
                f"nsdId={nsdId} not found in CLIP cache "
                f"({len(self.embedding_lookup)} entries)"
            )

        col = resolve_embedding_column(self.embeddings_df, _EMBEDDING_COLUMN_OVERRIDE)
        embedding = self.embeddings_df.iloc[emb_idx][col]

        fmri_tensor = torch.tensor(np.asarray(fmri, dtype=np.float32))
        emb_tensor = torch.tensor(np.asarray(embedding, dtype=np.float32))
        return fmri_tensor, emb_tensor


class PreextractedNSDDataset(Dataset):
    """Fast dataset backed by pre-extracted float32 numpy arrays.

    Loads the entire feature matrix into RAM (~1.8 GB for 30k x 15724)
    so that __getitem__ is a simple array index -- no NIfTI I/O.

    When *token_cache* is provided the CLIP targets come from a
    :class:`~fmri2img.data.token_clip_cache.TokenCLIPCache` HDF5 file
    (shape ``(T, D)`` per image) and are returned **flat** ``(T*D,)``.
    The Parquet ``embeddings_df`` is still needed for nsdId lookup but
    the embedding column is ignored.

    When *dual_target* is ``True`` **and** a *token_cache* is available,
    each sample is returned as a **dict** with explicit keys::

        {"fmri": Tensor, "retrieval_target": Tensor, "rich_target": Tensor,
         "subject_id": Tensor(long), "nsd_id": Tensor(long)}

    This replaces the fragile positional-tuple convention.  When
    ``dual_target=False`` the legacy tuple interface is preserved.
    """

    def __init__(
        self,
        features_path: Path,
        index_df: pd.DataFrame,
        embeddings_df: pd.DataFrame,
        average_repetitions: bool = False,
        token_cache=None,
        rerank_cache=None,
        dual_target: bool = False,
        retrieval_projector=None,
    ):
        self.features = np.load(features_path, mmap_mode=None)  # (N, V) float32
        self.index_df = index_df.reset_index(drop=True)
        self.embeddings_df = embeddings_df
        self.token_cache = token_cache  # Optional[TokenCLIPCache]
        self.rerank_cache = rerank_cache  # Optional[CompressedTargetCache]
        self.dual_target = dual_target
        self.retrieval_projector = retrieval_projector
        self._retrieval_dim = (
            int(self.retrieval_projector.output_dim)
            if self.retrieval_projector is not None else None
        )

        if dual_target and token_cache is None:
            raise ValueError(
                "dual_target=True requires a token_cache (HDF5 token targets). "
                "Set data.token_cache_path in your config."
            )

        if len(self.features) != len(self.index_df):
            raise ValueError(
                f"Feature rows ({len(self.features)}) != index rows ({len(self.index_df)}). "
                "Re-run: make preextract SUBJECT=<subject>"
            )

        if average_repetitions and "nsdId" in self.index_df.columns:
            n_raw = len(self.index_df)
            unique_ids = self.index_df["nsdId"].unique()
            avg_features = np.zeros((len(unique_ids), self.features.shape[1]), dtype=np.float32)
            new_rows = []
            nsd_vals = self.index_df["nsdId"].values
            for i, nsd_id in enumerate(unique_ids):
                mask = nsd_vals == nsd_id
                avg_features[i] = self.features[mask].mean(axis=0)
                new_rows.append(self.index_df[mask].iloc[0].to_dict())
            self.features = avg_features
            self.index_df = pd.DataFrame(new_rows).reset_index(drop=True)
            logger.info(
                "Repetition averaging: %d trials -> %d unique images (SNR ~%.2fx)",
                n_raw, len(unique_ids), np.sqrt(n_raw / len(unique_ids)),
            )

        if "nsdId" in embeddings_df.columns:
            self.embedding_lookup = {
                row["nsdId"]: i for i, (_, row) in enumerate(embeddings_df.iterrows())
            }
        else:
            self.embedding_lookup = {i: i for i in range(len(embeddings_df))}

        logger.info(
            "PreextractedNSDDataset: %d trials, %d voxels (%.2f GB in RAM), dual_target=%s",
            *self.features.shape, self.features.nbytes / 1e9, dual_target,
        )

    def __len__(self) -> int:
        return len(self.features)

    def _get_cls_embedding(self, nsd_id: int) -> np.ndarray:
        """Return the CLS/pooled embedding (768-D) from embeddings_df."""
        emb_idx = self.embedding_lookup.get(nsd_id)
        if emb_idx is None:
            raise KeyError(
                f"nsdId={nsd_id} not found in CLIP cache "
                f"({len(self.embedding_lookup)} entries)"
            )
        col = resolve_embedding_column(self.embeddings_df, _EMBEDDING_COLUMN_OVERRIDE)
        cls_emb = np.asarray(self.embeddings_df.iloc[emb_idx][col], dtype=np.float32)
        if self.retrieval_projector is not None:
            cls_emb = self.retrieval_projector.transform(cls_emb)
        return cls_emb

    def __getitem__(self, idx: int):
        fmri = self.features[idx]
        nsd_id = int(self.index_df.iloc[idx]["nsdId"])

        if self.dual_target:
            cls_emb = self._get_cls_embedding(nsd_id)
            token_emb = self.token_cache.get_flat(nsd_id)
            if self._retrieval_dim is not None and cls_emb.shape[0] != self._retrieval_dim:
                raise ValueError(
                    f"retrieval_target dim mismatch for nsdId={nsd_id}: "
                    f"got {cls_emb.shape[0]}, expected {self._retrieval_dim}"
                )
            out = {
                "fmri": torch.from_numpy(np.asarray(fmri, dtype=np.float32)),
                "retrieval_target": torch.from_numpy(cls_emb),
                "rich_target": torch.from_numpy(np.asarray(token_emb, dtype=np.float32)),
                "subject_id": torch.tensor(0, dtype=torch.long),
                "nsd_id": torch.tensor(nsd_id, dtype=torch.long),
            }
            if self.rerank_cache is not None:
                out["rerank_target"] = torch.from_numpy(self.rerank_cache[nsd_id])
            return out

        if self.token_cache is not None:
            embedding = self.token_cache.get_flat(nsd_id)
        else:
            emb_idx = self.embedding_lookup.get(nsd_id)
            if emb_idx is None:
                raise KeyError(
                    f"nsdId={nsd_id} not found in CLIP cache "
                    f"({len(self.embedding_lookup)} entries)"
                )

            col = resolve_embedding_column(self.embeddings_df, _EMBEDDING_COLUMN_OVERRIDE)
            embedding = self.embeddings_df.iloc[emb_idx][col]

        fmri_tensor = torch.from_numpy(np.asarray(fmri, dtype=np.float32))
        emb_tensor = torch.from_numpy(np.asarray(embedding, dtype=np.float32))
        return fmri_tensor, emb_tensor


# ---------------------------------------------------------------------------
# Config & path helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return config


def resolve_subject(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    """Resolve subject from CLI override or config."""
    return args.subject or config.get("data", {}).get("subject", "subj01")


def build_retrieval_target_projector(
    config: Dict[str, Any],
    embeddings_df: pd.DataFrame,
) -> Optional[Any]:
    """Create a deterministic retrieval-target projector when dims differ."""
    decoder_cfg = config.get("model", {}).get("decoder", {})
    target_dim = int(decoder_cfg.get("retrieval_dim", 768))
    emb_col = resolve_embedding_column(embeddings_df, _EMBEDDING_COLUMN_OVERRIDE)
    sample = np.asarray(embeddings_df.iloc[0][emb_col], dtype=np.float32)
    input_dim = int(sample.shape[0])

    if target_dim == input_dim:
        return None

    proj_cfg = config.get("data", {}).get("retrieval_target_projection", {})
    if proj_cfg.get("enabled", True) is False:
        raise ValueError(
            f"decoder.retrieval_dim={target_dim} but compact target dim is {input_dim} "
            "and retrieval_target_projection.enabled=false"
        )

    from fmri2img.data.fixed_target_projector import FixedTargetProjector

    projector = FixedTargetProjector(
        input_dim=input_dim,
        output_dim=target_dim,
        seed=int(proj_cfg.get("seed", config.get("data", {}).get("seed", 42))),
        method=str(proj_cfg.get("method", "orthogonal_lift")),
        l2_normalize=True,
    )
    logger.info(
        "Compact retrieval targets will be projected: %d -> %d (method=%s, seed=%d)",
        input_dim,
        target_dim,
        projector.method,
        projector.seed,
    )
    return projector


def resolve_index_path(subject: str) -> Path:
    """Resolve the NSD index parquet path for a subject."""
    return Path(f"data/indices/nsd_index/subject={subject}/index.parquet")


def resolve_roi_mask_path(subject: str) -> Path:
    nsd_root = os.environ.get("NSD_DATA_ROOT", "data/nsd")
    candidates = [
        Path(nsd_root) / "nsddata" / "ppdata" / subject / "func1pt8mm" / "roi" / "nsdgeneral.nii.gz",
        Path(nsd_root) / "nsddata" / "ppdata" / subject / "func1pt8mm" / "nsdgeneral.nii.gz",
        Path(nsd_root) / "ppdata" / subject / "func1pt8mm" / "roi" / "nsdgeneral.nii.gz",
        Path(nsd_root) / "ppdata" / subject / "func1pt8mm" / "nsdgeneral.nii.gz",
        Path("data") / "nsd" / "ppdata" / subject / "func1pt8mm" / "roi" / "nsdgeneral.nii.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def resolve_preproc_artifact(subject: str, config: Dict[str, Any]) -> str:
    """Build preprocessing artifact path, substituting the subject."""
    preproc_cfg = config.get("preprocessing", {})
    artifact = preproc_cfg.get("artifact_path", "")
    if artifact:
        return artifact.replace("subj01", subject)
    return f"cache/embedding_preproc/{subject}_center_pcr_k8.pkl"


def find_embeddings_path() -> Optional[Path]:
    candidates = [
        Path("outputs/clip_cache/clip_multilayer.parquet"),
        Path("cache/clip_embeddings/nsd_clipcache_multilayer.parquet"),
        Path("cache/clip_embeddings/nsd_clipvitl14.parquet"),
        Path("cache/clip_embeddings/embeddings_ViT-B-32.parquet"),
        Path("cache/clip_embeddings/text_clip.parquet"),
        Path("outputs/clip_cache/clip.parquet"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def setup_preprocessing(config: Dict[str, Any], subject: str,
                        device: str) -> Tuple[Optional[EmbeddingPreprocessor], bool]:
    """Returns (preprocessor, needs_fit). If needs_fit=True, caller must fit on training data."""
    preproc_cfg = config.get("preprocessing", {})
    if not preproc_cfg.get("enabled", False):
        logger.info("Preprocessing disabled")
        return None, False

    artifact_path = Path(resolve_preproc_artifact(subject, config))
    if artifact_path.exists():
        logger.info("Loading preprocessor from %s", artifact_path)
        return EmbeddingPreprocessor.load(artifact_path), False

    mode = preproc_cfg.get("mode", "center_pcr")
    k_components = preproc_cfg.get("k_components", 8)
    whiten_eps = preproc_cfg.get("whiten_eps", 1e-5)
    logger.warning("Preproc artifact not found at %s — will auto-fit on training embeddings", artifact_path)
    return EmbeddingPreprocessor(
        mode=mode, k_components=k_components, whiten_eps=whiten_eps
    ), True


# ---------------------------------------------------------------------------
# Loss setup (including N3/N4 losses)
# ---------------------------------------------------------------------------

def setup_losses(config: Dict[str, Any], device: str,
                 queue: Optional[nn.Module] = None) -> Dict[str, nn.Module]:
    """Setup all loss functions from config, including SPCL and MultiTask."""
    loss_cfg = config.get("loss", {})
    losses: Dict[str, nn.Module] = {}

    if loss_cfg.get("mse", {}).get("enabled", False):
        mse_reduction = loss_cfg["mse"].get("reduction", "mean")
        losses["mse"] = nn.MSELoss(reduction=mse_reduction)
        logger.info("MSE loss enabled (reduction=%s)", mse_reduction)

    if loss_cfg.get("infonce", {}).get("enabled", False):
        c = loss_cfg["infonce"]
        use_q = c.get("use_queue", False) and queue is not None
        losses["infonce"] = InfoNCEQueueLoss(
            temperature=c.get("temperature", 0.07),
            learnable_temperature=c.get("learnable_temperature", True),
            use_queue=use_q,
            symmetric=c.get("symmetric", True),
        )
        logger.info("InfoNCE loss enabled (queue=%s)", use_q)

    if loss_cfg.get("gaussian_nll", {}).get("enabled", False):
        c = loss_cfg["gaussian_nll"]
        losses["gaussian_nll"] = GaussianNLLLoss(
            logvar_min=c.get("logvar_min", -10.0),
            logvar_max=c.get("logvar_max", 5.0),
            learnable_global_scale=c.get("learnable_global_scale", False),
            reduction=c.get("reduction", "mean"),
        )
        logger.info("Gaussian NLL loss enabled")

    if loss_cfg.get("gaussian_nce", {}).get("enabled", False):
        c = loss_cfg["gaussian_nce"]
        use_q = c.get("use_queue", False) and queue is not None
        losses["gaussian_nce"] = GaussianNCELoss(
            use_temperature=c.get("use_temperature", True),
            temperature=c.get("temperature", 1.0),
            use_queue=use_q,
            symmetric=c.get("symmetric", False),
            clamp_logvar=c.get("clamp_logvar", True),
            logvar_min=c.get("logvar_min", -10.0),
            logvar_max=c.get("logvar_max", 5.0),
        )
        logger.info("Gaussian-NCE loss enabled (queue=%s)", use_q)

    model_cfg = config.get("model", {})
    decoder_cfg = model_cfg.get("decoder", {})
    vmf_kappa_is_log = "log_kappa_min" in decoder_cfg or "log_kappa_max" in decoder_cfg
    if model_cfg.get("posterior") == "vmf" or "kappa_min" in decoder_cfg:
        vmf_kappa_is_log = False

    if loss_cfg.get("vmf_nll", {}).get("enabled", False):
        c = loss_cfg["vmf_nll"]
        # dim auto-detected from embedding_dim; fallback 768 for legacy configs
        _nll_dim = c.get("dim", config.get("model", {}).get("decoder", {}).get("token_dim", 768))
        losses["vmf_nll"] = VonMisesFisherNLLLoss(
            dim=_nll_dim, kappa_is_log=vmf_kappa_is_log,
        )
        logger.info("vMF-NLL loss enabled (kappa_is_log=%s)", vmf_kappa_is_log)

    if loss_cfg.get("vmf_nce", {}).get("enabled", False):
        c = loss_cfg["vmf_nce"]
        use_q = c.get("use_queue", False) and queue is not None
        _learnable_tau = c.get("learnable_temperature", False)
        losses["vmf_nce"] = VonMisesFisherNCELoss(
            tau=c.get("tau", 0.07), use_queue=use_q, kappa_is_log=vmf_kappa_is_log,
            learnable_temperature=_learnable_tau,
            use_arctanh=c.get("use_arctanh", False),
            margin_base=c.get("margin_base", 0.0),
            margin_kappa_ref=c.get("margin_kappa_ref", 50.0),
            label_smoothing=c.get("label_smoothing", 0.0),
            hard_negative_weight=c.get("hard_negative_weight", 0.0),
            hard_neg_k=c.get("hard_neg_k", 16),
            use_csls_training=c.get("use_csls_training", False),
            csls_k=c.get("csls_k", 10),
            isf_weight=c.get("isf_weight", 0.0),
        )
        logger.info("vMF-NCE loss enabled (queue=%s, tau=%s, learnable_tau=%s, arctanh=%s, "
                     "margin=%.2f, hard_neg=%.2f, csls_train=%s, isf=%.2f)",
                     use_q, c.get("tau", 0.07), _learnable_tau,
                     c.get("use_arctanh", False), c.get("margin_base", 0.0),
                     c.get("hard_negative_weight", 0.0),
                     c.get("use_csls_training", False), c.get("isf_weight", 0.0))


    if loss_cfg.get("vmf_nce_mixture", {}).get("enabled", False):
        c = loss_cfg["vmf_nce_mixture"]
        use_q = c.get("use_queue", False) and queue is not None
        _learnable_tau = c.get("learnable_temperature", False)
        losses["vmf_nce_mixture"] = MixtureVonMisesFisherNCELoss(
            tau=c.get("tau", 0.07), use_queue=use_q, kappa_is_log=vmf_kappa_is_log,
            learnable_temperature=_learnable_tau,
            use_arctanh=c.get("use_arctanh", False),
            margin_base=c.get("margin_base", 0.0),
            margin_kappa_ref=c.get("margin_kappa_ref", 50.0),
            label_smoothing=c.get("label_smoothing", 0.0),
            hard_negative_weight=c.get("hard_negative_weight", 0.0),
            hard_neg_k=c.get("hard_neg_k", 16),
            use_csls_training=c.get("use_csls_training", False),
            csls_k=c.get("csls_k", 10),
            isf_weight=c.get("isf_weight", 0.0),
        )
        logger.info("Mixture vMF-NCE loss enabled (queue=%s, tau=%s, learnable_tau=%s, arctanh=%s)",
                     use_q, c.get("tau", 0.07), _learnable_tau, c.get("use_arctanh", False))

    if loss_cfg.get("mixture_diversity", {}).get("enabled", False):
        c = loss_cfg["mixture_diversity"]
        losses["mixture_diversity"] = MixtureComponentDiversityLoss(
            cosine_margin=c.get("cosine_margin", 0.85),
            entropy_target=c.get("entropy_target", 0.75),
            kappa_std_target=c.get("kappa_std_target", 0.25),
            direction_weight=c.get("direction_weight", 1.0),
            entropy_weight=c.get("entropy_weight", 0.10),
            kappa_weight=c.get("kappa_weight", 0.05),
        )
        logger.info(
            "Mixture diversity loss enabled (weight=%.3f, cos_margin=%.2f, entropy_target=%.2f, kappa_std_target=%.2f)",
            c.get("weight", 0.05),
            c.get("cosine_margin", 0.85),
            c.get("entropy_target", 0.75),
            c.get("kappa_std_target", 0.25),
        )

    # --- N4: kappa-SPCL (or Delta-SPCL) ---
    if loss_cfg.get("vmf_nce_spcl", {}).get("enabled", False):
        c = loss_cfg["vmf_nce_spcl"]
        use_q = c.get("use_queue", False) and queue is not None
        if c.get("use_delta", False):
            losses["vmf_nce_spcl"] = DeltaSPCLVMFNCELoss(
                tau=c.get("tau", 0.07),
                use_queue=use_q,
                kappa_is_log=vmf_kappa_is_log,
                initial_curriculum_t=c.get("initial_curriculum_t", 100.0),
                delta_weight=c.get("delta_weight", 10.0),
            )
            logger.info("Delta-SPCL loss enabled (curriculum_t=%.1f, delta_weight=%.1f)",
                         c.get("initial_curriculum_t", 100.0), c.get("delta_weight", 10.0))
        else:
            losses["vmf_nce_spcl"] = KappaSPCLVMFNCELoss(
                tau=c.get("tau", 0.07),
                use_queue=use_q,
                kappa_is_log=vmf_kappa_is_log,
                initial_curriculum_t=c.get("initial_curriculum_t", 100.0),
                hard_negative_weight=c.get("hard_negative_weight", 0.0),
                hard_neg_k=c.get("hard_neg_k", 16),
                use_csls_training=c.get("use_csls_training", False),
                csls_k=c.get("csls_k", 10),
                isf_weight=c.get("isf_weight", 0.0),
            )
            logger.info("vMF-NCE-SPCL loss enabled (curriculum_t=%.1f, hard_neg=%.2f, csls_train=%s, isf=%.2f)",
                         c.get("initial_curriculum_t", 100.0),
                         c.get("hard_negative_weight", 0.0),
                         c.get("use_csls_training", False),
                         c.get("isf_weight", 0.0))

    # --- SoftCLIP knowledge distillation ---
    if loss_cfg.get("softclip", {}).get("enabled", False):
        c = loss_cfg["softclip"]
        use_q = c.get("use_queue", False) and queue is not None
        if c.get("vmf_mode", False):
            losses["softclip"] = VMFSoftCLIPLoss(
                teacher_tau=c.get("teacher_tau", c.get("tau", 0.05)),
                use_queue=use_q,
                symmetric=c.get("symmetric", True),
            )
            logger.info("vMF-SoftCLIP loss enabled (teacher_tau=%.3f, queue=%s, symmetric=%s)",
                         c.get("teacher_tau", c.get("tau", 0.05)), use_q, c.get("symmetric", True))
        else:
            losses["softclip"] = SoftCLIPLoss(
                tau=c.get("tau", 0.07),
                use_queue=use_q,
                symmetric=c.get("symmetric", True),
            )
            logger.info("SoftCLIP loss enabled (tau=%.3f, queue=%s, symmetric=%s)",
                         c.get("tau", 0.07), use_q, c.get("symmetric", True))

    # --- V30d: Rerank SoftCLIP (on PCA-compressed targets) ---
    if loss_cfg.get("rerank_softclip", {}).get("enabled", False):
        c = loss_cfg["rerank_softclip"]
        losses["rerank_softclip"] = SoftCLIPLoss(
            tau=c.get("tau", 0.07),
            use_queue=False,  # rerank head has its own space, no shared queue
            symmetric=c.get("symmetric", True),
        )
        logger.info("Rerank SoftCLIP loss enabled (tau=%.3f, symmetric=%s)",
                     c.get("tau", 0.07), c.get("symmetric", True))

    # --- V33: shortlist-local teacher distillation (rerank -> compact) ---
    if loss_cfg.get("shortlist_teacher_distill", {}).get("enabled", False):
        c = loss_cfg["shortlist_teacher_distill"]
        losses["shortlist_teacher_distill"] = ShortlistTeacherDistillLoss(
            compact_k=c.get("compact_k", 16),
            teacher_k=c.get("teacher_k", 16),
            teacher_temperature=c.get("teacher_temperature", 0.07),
            student_temperature=c.get("student_temperature", 0.07),
            teacher_rank_gate=c.get("teacher_rank_gate", 20),
        )
        logger.info(
            "Shortlist teacher distill enabled "
            "(weight=%.3f, compact_k=%d, teacher_k=%d, teacher_tau=%.3f, student_tau=%.3f, start_epoch=%d, gate<=%d)",
            c.get("weight", 0.15),
            c.get("compact_k", 16),
            c.get("teacher_k", 16),
            c.get("teacher_temperature", 0.07),
            c.get("student_temperature", 0.07),
            c.get("start_epoch", 10),
            c.get("teacher_rank_gate", 20),
        )

    # --- V35: legacy-teacher distillation (frozen N1v28a -> compact) ---
    if loss_cfg.get("legacy_teacher_distill", {}).get("enabled", False):
        c = loss_cfg["legacy_teacher_distill"]
        losses["legacy_teacher_distill"] = LegacyTeacherDistillLoss(
            compact_k=c.get("compact_k", 16),
            teacher_k=c.get("teacher_k", 16),
            teacher_temperature=c.get("teacher_temperature", 0.07),
            student_temperature=c.get("student_temperature", 0.07),
            teacher_rank_gate=c.get("teacher_rank_gate", 20),
        )
        logger.info(
            "Legacy teacher distill enabled "
            "(weight=%.3f, compact_k=%d, teacher_k=%d, teacher_tau=%.3f, student_tau=%.3f, start_epoch=%d, gate<=%d)",
            c.get("weight", 0.15),
            c.get("compact_k", 16),
            c.get("teacher_k", 16),
            c.get("teacher_temperature", 0.07),
            c.get("student_temperature", 0.07),
            c.get("start_epoch", 10),
            c.get("teacher_rank_gate", 20),
        )

    # --- V38: top-k legacy -> compact distillation over full in-batch logits ---
    if loss_cfg.get("legacy_compact_distill", {}).get("enabled", False):
        c = loss_cfg["legacy_compact_distill"]
        losses["legacy_compact_distill"] = LegacyCompactDistillLoss(
            teacher_temperature=c.get("teacher_tau", c.get("teacher_temperature", 0.07)),
            student_temperature=c.get("student_tau", c.get("student_temperature", 0.07)),
            topk=c.get("topk", 12),
        )
        logger.info(
            "Legacy compact distill enabled "
            "(weight=%.3f, teacher_tau=%.3f, student_tau=%.3f, topk=%d, start_epoch=%d)",
            c.get("weight", 0.15),
            c.get("teacher_tau", c.get("teacher_temperature", 0.07)),
            c.get("student_tau", c.get("student_temperature", 0.07)),
            c.get("topk", 12),
            c.get("start_epoch", 10),
        )

    # --- V44: component-responsibility legacy -> compact distillation ---
    if loss_cfg.get("component_legacy_compact_distill", {}).get("enabled", False):
        c = loss_cfg["component_legacy_compact_distill"]
        losses["component_legacy_compact_distill"] = ComponentLegacyCompactDistillLoss(
            teacher_temperature=c.get("teacher_tau", c.get("teacher_temperature", 0.07)),
            student_temperature=c.get("student_tau", c.get("student_temperature", 0.07)),
            topk=c.get("topk", 12),
            use_component_logits_prior=c.get("use_component_logits_prior", True),
            prior_weight=c.get("prior_weight", 0.10),
        )
        logger.info(
            "Component legacy compact distill enabled "
            "(weight=%.3f, teacher_tau=%.3f, student_tau=%.3f, topk=%d, start_epoch=%d, logits_prior=%s, prior_weight=%.3f)",
            c.get("weight", 0.10),
            c.get("teacher_tau", c.get("teacher_temperature", 0.07)),
            c.get("student_tau", c.get("student_temperature", 0.07)),
            c.get("topk", 12),
            c.get("start_epoch", 10),
            c.get("use_component_logits_prior", True),
            c.get("prior_weight", 0.10),
        )

    # --- V36: combined rerank + legacy tri-teacher distillation ---
    if loss_cfg.get("tri_teacher_distill", {}).get("enabled", False):
        c = loss_cfg["tri_teacher_distill"]
        losses["tri_teacher_distill"] = TriTeacherDistillLoss(
            compact_topk=c.get("compact_topk", c.get("compact_k", 16)),
            teacher_topk=c.get("teacher_topk", c.get("teacher_k", 16)),
            teacher_tau=c.get("teacher_tau", c.get("teacher_temperature", 0.07)),
            student_tau=c.get("student_tau", c.get("student_temperature", 0.07)),
            gate_max_rank=c.get("gate_max_rank", c.get("teacher_rank_gate", 20)),
            mode=c.get("mode", "weighted_logits"),
            rerank_teacher_weight=c.get("rerank_teacher_weight", 0.35),
            legacy_teacher_weight=c.get("legacy_teacher_weight", 0.65),
            symmetric=c.get("symmetric", False),
        )
        logger.info(
            "Tri-teacher distill enabled "
            "(weight=%.3f, mode=%s, rerank_w=%.3f, legacy_w=%.3f, compact_topk=%d, "
            "teacher_topk=%d, teacher_tau=%.3f, student_tau=%.3f, start_epoch=%d, gate<=%d, symmetric=%s)",
            c.get("weight", 0.20),
            c.get("mode", "weighted_logits"),
            c.get("rerank_teacher_weight", 0.35),
            c.get("legacy_teacher_weight", 0.65),
            c.get("compact_topk", c.get("compact_k", 16)),
            c.get("teacher_topk", c.get("teacher_k", 16)),
            c.get("teacher_tau", c.get("teacher_temperature", 0.07)),
            c.get("student_tau", c.get("student_temperature", 0.07)),
            c.get("start_epoch", 10),
            c.get("gate_max_rank", c.get("teacher_rank_gate", 20)),
            c.get("symmetric", False),
        )

    # --- N3/N4: MultiTask vMF-NCE ---
    if loss_cfg.get("vmf_nce_multitask", {}).get("enabled", False):
        c = loss_cfg["vmf_nce_multitask"]
        vmf_active_cfg = (
            loss_cfg.get("vmf_nce_spcl", {})
            if loss_cfg.get("vmf_nce_spcl", {}).get("enabled")
            else loss_cfg.get("vmf_nce", {})
        )
        mt_use_q = vmf_active_cfg.get("use_queue", False) and queue is not None
        mt_tau = vmf_active_cfg.get("tau", 0.07)
        losses["vmf_nce_multitask"] = MultiTaskVMFNCELoss(
            tau=mt_tau,
            use_queue=mt_use_q,
            lambda_aux=c.get("lambda_aux", 0.5),
            kappa_is_log=vmf_kappa_is_log,
        )
        logger.info("MultiTask vMF-NCE loss enabled (lambda_aux=%.2f, tau=%.3f, queue=%s)",
                     c.get("lambda_aux", 0.5), mt_tau, mt_use_q)

    # --- Hierarchical CLIP alignment (v8) ---
    if loss_cfg.get("hierarchical_clip", {}).get("enabled", False):
        c = loss_cfg["hierarchical_clip"]
        losses["hierarchical_clip"] = HierarchicalCLIPLoss(
            tier_indices=c.get("tier_indices"),
            tier_clip_columns=c.get("tier_clip_columns"),
        )
        logger.info("Hierarchical CLIP loss enabled (weight=%.3f)",
                     c.get("weight", 0.5))

    # --- CKA representational alignment (v8) ---
    if loss_cfg.get("cka", {}).get("enabled", False):
        c = loss_cfg["cka"]
        losses["cka"] = CKALoss(eps=c.get("eps", 1e-8))
        _per_subj = c.get("per_subject", False)
        logger.info("CKA loss enabled (weight=%.3f, per_subject=%s)",
                     c.get("weight", 0.5), _per_subj)

    # --- Direct cosine alignment (V11) ---
    if loss_cfg.get("direct_alignment", {}).get("enabled", False):
        losses["direct_alignment"] = DirectAlignmentLoss()
        logger.info("Direct alignment loss enabled (weight=%.3f)",
                     loss_cfg["direct_alignment"].get("weight", 0.5))

    # --- Spherical uniformity regularisation (V11) ---
    if loss_cfg.get("uniformity", {}).get("enabled", False):
        c = loss_cfg["uniformity"]
        losses["uniformity"] = UniformityLoss(t=c.get("t", 2.0))
        logger.info("Uniformity loss enabled (weight=%.3f, t=%.1f)",
                     c.get("weight", 0.1), c.get("t", 2.0))

    # Regression MSE is not a standard nn.Module loss object — it uses
    # F.mse_loss inline on the decoder's un-normalised regression head output.
    # Just log that it's configured.
    if loss_cfg.get("regression_mse", {}).get("enabled", False):
        _rm_w = loss_cfg["regression_mse"].get("weight", 1.0)
        logger.info("Dual-head regression MSE enabled (weight=%.3f) — requires "
                     "decoder.regression_head=true", _rm_w)

    return losses


def setup_kl_scheduler(config: Dict[str, Any]) -> Optional[KLScheduler]:
    """Setup KL annealing scheduler if enabled."""
    kl_cfg = config.get("loss", {}).get("kl", {})
    if not kl_cfg.get("enabled", False):
        return None
    return KLScheduler(
        start_weight=kl_cfg.get("weight_start", 0.0),
        end_weight=kl_cfg.get("weight_end", 0.001),
        n_steps=kl_cfg.get("anneal_steps", 10000),
        anneal_type=kl_cfg.get("anneal_type", "linear"),
        free_bits=kl_cfg.get("free_bits", 0.5),
        free_bits_aggregate=kl_cfg.get("free_bits_type", "dimension"),
    )


# ---------------------------------------------------------------------------
# KL divergence helpers
# ---------------------------------------------------------------------------

def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def _ive(v: float, z: torch.Tensor) -> torch.Tensor:
    return torch.special.i1e(z) if abs(v - 1.0) < 0.01 else torch.special.i0e(z)


def compute_vmf_kl(mu: torch.Tensor, log_kappa: torch.Tensor, dim: int) -> torch.Tensor:
    kappa = log_kappa.exp().squeeze(-1)
    half_d = dim / 2.0
    log_sphere = (
        half_d * math.log(2 * math.pi)
        + torch.lgamma(torch.tensor(half_d, device=mu.device))
        - torch.tensor(half_d, device=mu.device) * math.log(1.0)
    )
    log_c_kappa = (
        (half_d - 1) * torch.log(kappa + 1e-8)
        - half_d * math.log(2 * math.pi)
        - torch.log(_ive(half_d - 1, kappa) + 1e-10)
        - kappa
    )
    kl = (
        kappa * _ive(half_d, kappa) / (_ive(half_d - 1, kappa) + 1e-10)
        - log_c_kappa
        + log_sphere
    )
    return kl.mean()


def _get_legacy_teacher_mask_and_voxels(
    model: nn.Module,
    subject_ids: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], int]:
    if subject_ids is None:
        return None, 0
    canonical_subject = getattr(model, "_canonical_subject", None)
    subj_map = getattr(model, "_subj_int_to_id", None)
    if not canonical_subject or not subj_map:
        return None, 0
    canonical_int = next((int(i) for i, sid in subj_map.items() if sid == canonical_subject), None)
    canonical_voxels = int(getattr(model, "_canonical_voxels", 0) or 0)
    if canonical_int is None:
        return None, canonical_voxels
    return subject_ids == canonical_int, canonical_voxels


def _maybe_mask_legacy_tensor(tensor: Optional[torch.Tensor], mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None or mask is None:
        return tensor
    return tensor[mask]


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    losses: Dict[str, nn.Module],
    loss_weights: Dict[str, float],
    device: str,
    kl_scheduler: Optional[KLScheduler],
    queue: Optional[nn.Module],
    preprocessor: Optional[EmbeddingPreprocessor],
    global_step: int,
    grad_accum_steps: int = 1,
    scaler: Optional[torch.amp.GradScaler] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    config_ref: Optional[Dict[str, Any]] = None,
    vmf_is_log: bool = True,
    mixco_cfg: Optional[Dict[str, Any]] = None,
    ema: Optional[ModelEMA] = None,
    amp_dtype: Optional[torch.dtype] = None,
    current_epoch: int = 0,
    log_sigmas: Optional[nn.ParameterDict] = None,
    legacy_teacher_model: Optional[nn.Module] = None,
) -> Tuple[Dict[str, float], int]:
    """Train for one epoch with gradient accumulation and optional AMP."""
    model.train()
    epoch_metrics: Dict[str, list] = {}
    _amp_dtype = amp_dtype or torch.float16
    use_amp = (scaler is not None and scaler.is_enabled()) or (_amp_dtype == torch.bfloat16)
    model_type = getattr(model, "model_type", "deterministic")

    optimizer.zero_grad()
    _is_vmf_triple = model_type == "vmf_triple"
    pbar = tqdm(dataloader, desc="Training")
    for step_in_epoch, batch in enumerate(pbar):
        hier_targets = None
        _rich_target = None
        _rerank_target = None
        _batch_nsd_ids = None

        # --- Dict-batch (dual_target mode for vmf_triple) ---
        if isinstance(batch, dict):
            fmri = batch["fmri"].to(device, dtype=torch.float32)
            gt_embedding = batch["retrieval_target"].to(device, dtype=torch.float32)
            _rich_target = batch["rich_target"].to(device, dtype=torch.float32)
            subject_ids = batch["subject_id"].to(device)
            _batch_nsd_ids = batch["nsd_id"]
            if "rerank_target" in batch:
                _rerank_target = batch["rerank_target"].to(device, dtype=torch.float32)

            if step_in_epoch == 0 and _is_vmf_triple:
                _rd = gt_embedding.shape[-1]
                _td = _rich_target.shape[-1]
                _rrd = _rerank_target.shape[-1] if _rerank_target is not None else None
                _dec = getattr(model, "decoder", None)
                if _dec is not None:
                    assert _rd == _dec.retrieval_dim, (
                        f"retrieval_target dim ({_rd}) != decoder.retrieval_dim ({_dec.retrieval_dim})"
                    )
                    assert _td == _dec.token_dim, (
                        f"rich_target dim ({_td}) != decoder.token_dim ({_dec.token_dim})"
                    )
                    if _rrd is not None and getattr(_dec, "has_rerank", False):
                        assert _rrd == _dec.rerank_dim, (
                            f"rerank_target dim ({_rrd}) != decoder.rerank_dim ({_dec.rerank_dim})"
                        )
        # --- Legacy tuple batch ---
        elif len(batch) == 4:
            fmri, gt_embedding, subject_ids, hier_targets = batch
            subject_ids = subject_ids.to(device)
            fmri = fmri.to(device, dtype=torch.float32)
            gt_embedding = gt_embedding.to(device, dtype=torch.float32)
            if hier_targets is not None:
                hier_targets = {k: v.to(device, dtype=torch.float32)
                                for k, v in hier_targets.items()}
        elif len(batch) == 3:
            fmri, gt_embedding, subject_ids = batch
            subject_ids = subject_ids.to(device)
            fmri = fmri.to(device, dtype=torch.float32)
            gt_embedding = gt_embedding.to(device, dtype=torch.float32)
        else:
            fmri, gt_embedding = batch
            subject_ids = None
            fmri = fmri.to(device, dtype=torch.float32)
            gt_embedding = gt_embedding.to(device, dtype=torch.float32)

        fmri_teacher = fmri

        # fMRI noise augmentation: Gaussian noise to reduce overfitting
        _noise_std = (config_ref or {}).get("training", {}).get("fmri_noise_std", 0)
        if _noise_std > 0:
            fmri = fmri + torch.randn_like(fmri) * _noise_std

        # Voxel dropout: randomly zero voxels, scaled to preserve magnitude
        _voxel_drop = (config_ref or {}).get("training", {}).get("voxel_dropout", 0)
        if _voxel_drop > 0:
            mask = torch.bernoulli(torch.full_like(fmri, 1.0 - _voxel_drop))
            fmri = fmri * mask / (1.0 - _voxel_drop)

        if preprocessor is not None:
            gt_embedding_np = gt_embedding.cpu().numpy()
            gt_embedding_proc = preprocessor.transform(gt_embedding_np)
            gt_embedding = torch.from_numpy(gt_embedding_proc).float().to(device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=_amp_dtype):
            output = model(fmri, subject_ids=subject_ids) if subject_ids is not None else model(fmri)
            if isinstance(output, tuple):
                pred, aux = output
            else:
                pred, aux = output, None
            _vmf_pred_head, _vmf_aux_head = _extract_vmf_outputs_for_losses(model_type, pred, aux)

            total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            batch_metrics: Dict[str, float] = {}

            is_gaussian = model_type == "gaussian" and aux is not None
            is_vmf = _vmf_pred_head is not None and _vmf_aux_head is not None

            # Projection head for KD losses (SoftCLIP, MixCo) only.
            # vMF-NCE uses raw pred so kappa gets proper gradient flow
            # through cos_sim(mu, target) instead of a random projection.
            _proj_head = getattr(model, "projection_head", None)
            pred_for_contrast = _proj_head(pred) if _proj_head is not None else pred
            _legacy_teacher_mask, _legacy_teacher_voxels = _get_legacy_teacher_mask_and_voxels(model, subject_ids)
            _legacy_teacher_pred = None
            if legacy_teacher_model is not None and _rich_target is not None:
                with torch.no_grad():
                    if _legacy_teacher_mask is not None:
                        if bool(_legacy_teacher_mask.any().item()):
                            _teacher_fmri = fmri_teacher[_legacy_teacher_mask]
                            if _legacy_teacher_voxels > 0:
                                _teacher_fmri = _teacher_fmri[:, :_legacy_teacher_voxels]
                            _legacy_out = legacy_teacher_model(_teacher_fmri)
                            _legacy_teacher_pred = _legacy_out[0] if isinstance(_legacy_out, tuple) else _legacy_out
                    else:
                        _legacy_out = (
                            legacy_teacher_model(fmri_teacher, subject_ids=subject_ids)
                            if subject_ids is not None
                            else legacy_teacher_model(fmri_teacher)
                        )
                        _legacy_teacher_pred = _legacy_out[0] if isinstance(_legacy_out, tuple) else _legacy_out
            _compact_component_mu = getattr(model, "_last_compact_component_mu", None)
            _compact_component_kappa = getattr(model, "_last_compact_component_kappa", None)
            _compact_component_logits = getattr(model, "_last_compact_component_logits", None)
            _pred_for_legacy = _maybe_mask_legacy_tensor(pred, _legacy_teacher_mask)
            _gt_for_legacy = _maybe_mask_legacy_tensor(gt_embedding, _legacy_teacher_mask)
            _rich_target_for_legacy = _maybe_mask_legacy_tensor(_rich_target, _legacy_teacher_mask)
            _compact_component_mu_for_legacy = _maybe_mask_legacy_tensor(_compact_component_mu, _legacy_teacher_mask)
            _compact_component_kappa_for_legacy = _maybe_mask_legacy_tensor(_compact_component_kappa, _legacy_teacher_mask)
            _compact_component_logits_for_legacy = _maybe_mask_legacy_tensor(_compact_component_logits, _legacy_teacher_mask)

            # --- Deterministic / regression losses ---
            if "mse" in losses and not is_gaussian:
                l = losses["mse"](pred, gt_embedding)
                total_loss = total_loss + loss_weights.get("mse", 1.0) * l
                batch_metrics["mse"] = l.item()

            if "infonce" in losses and not is_gaussian and not is_vmf:
                l = losses["infonce"](pred_for_contrast, gt_embedding, queue=queue)
                total_loss = total_loss + loss_weights.get("infonce", 1.0) * l
                batch_metrics["infonce"] = l.item()

            # --- SoftCLIP knowledge distillation (works for all model types) ---
            if "softclip" in losses:
                if isinstance(losses["softclip"], VMFSoftCLIPLoss) and is_vmf:
                    l = losses["softclip"](_vmf_pred_head, _vmf_aux_head, gt_embedding, queue=queue)
                else:
                    l = losses["softclip"](pred_for_contrast, gt_embedding, queue=queue)
                total_loss = total_loss + loss_weights.get("softclip", 1.0) * l
                batch_metrics["softclip"] = l.item()

            # --- Gaussian losses ---
            if "gaussian_nll" in losses and is_gaussian:
                l = losses["gaussian_nll"](pred, aux, gt_embedding)
                total_loss = total_loss + loss_weights.get("gaussian_nll", 1.0) * l
                batch_metrics["nll"] = l.item()

            if "gaussian_nce" in losses and is_gaussian:
                l = losses["gaussian_nce"](pred, aux, gt_embedding, queue=queue)
                total_loss = total_loss + loss_weights.get("gaussian_nce", 1.0) * l
                batch_metrics["gnce"] = l.item()

            # --- vMF losses ---
            if "vmf_nll" in losses and is_vmf:
                l = losses["vmf_nll"](_vmf_pred_head, _vmf_aux_head, gt_embedding)
                total_loss = total_loss + loss_weights.get("vmf_nll", 1.0) * l
                batch_metrics["vmf_nll"] = l.item()

            if "vmf_nce" in losses and is_vmf:
                l = losses["vmf_nce"](_vmf_pred_head, _vmf_aux_head, gt_embedding, queue=queue)
                total_loss = total_loss + loss_weights.get("vmf_nce", 1.0) * l
                batch_metrics["vmf_nce"] = l.item()

            _mix_cfg = (config_ref or {}).get("loss", {}).get("vmf_nce_mixture", {})
            _mix_start_epoch = int(_mix_cfg.get("start_epoch", 0))
            if (
                "vmf_nce_mixture" in losses
                and is_vmf
                and _compact_component_mu is not None
                and _compact_component_kappa is not None
            ):
                if current_epoch > _mix_start_epoch:
                    l = losses["vmf_nce_mixture"](
                        _compact_component_mu,
                        _compact_component_kappa,
                        gt_embedding,
                        component_logits=_compact_component_logits,
                        queue=queue,
                    )
                    if torch.isfinite(l):
                        total_loss = total_loss + loss_weights.get("vmf_nce_mixture", 1.0) * l
                        batch_metrics["vmf_nce_mixture"] = l.item()
                    else:
                        logger.warning("Non-finite vmf_nce_mixture at step %d epoch %d -- skipping mixture loss", global_step, current_epoch)
                        batch_metrics["vmf_nce_mixture"] = 0.0
                else:
                    batch_metrics["vmf_nce_mixture"] = 0.0

            _mix_div_cfg = (config_ref or {}).get("loss", {}).get("mixture_diversity", {})
            _mix_div_start_epoch = int(_mix_div_cfg.get("start_epoch", 0))
            if (
                "mixture_diversity" in losses
                and is_vmf
                and _compact_component_mu is not None
                and _compact_component_kappa is not None
                and _compact_component_mu.ndim == 3
                and _compact_component_mu.shape[1] > 1
            ):
                if current_epoch > _mix_div_start_epoch:
                    _mix_div_l, _mix_div_stats = losses["mixture_diversity"](
                        _compact_component_mu,
                        _compact_component_kappa,
                        component_logits=_compact_component_logits,
                    )
                    if torch.isfinite(_mix_div_l):
                        total_loss = total_loss + loss_weights.get("mixture_diversity", 1.0) * _mix_div_l
                        batch_metrics["mixture_diversity"] = _mix_div_l.item()
                        batch_metrics["mixture_pairwise_cos"] = _mix_div_stats["pairwise_cos_mean"]
                        batch_metrics["mixture_weight_entropy_norm"] = _mix_div_stats["weight_entropy_norm"]
                        batch_metrics["mixture_top_weight_mean"] = _mix_div_stats["top_weight_mean"]
                        batch_metrics["mixture_kappa_std_mean"] = _mix_div_stats["kappa_std_mean"]
                    else:
                        logger.warning("Non-finite mixture_diversity at step %d epoch %d -- skipping diversity loss", global_step, current_epoch)
                        batch_metrics["mixture_diversity"] = 0.0
                else:
                    batch_metrics["mixture_diversity"] = 0.0

            # --- vMF-NCE-SPCL (N4) or Delta-SPCL ---
            if "vmf_nce_spcl" in losses and is_vmf:
                spcl_kwargs = dict(queue=queue)
                if isinstance(losses["vmf_nce_spcl"], DeltaSPCLVMFNCELoss):
                    dcf_ex = getattr(model, "_last_dcf_extras", {})
                    spcl_kwargs["delta"] = dcf_ex.get("delta")
                l = losses["vmf_nce_spcl"](_vmf_pred_head, _vmf_aux_head, gt_embedding, **spcl_kwargs)
                total_loss = total_loss + loss_weights.get("vmf_nce_spcl", 1.0) * l
                batch_metrics["vmf_nce_spcl"] = l.item()

            # --- MultiTask vMF-NCE (N3/N4) ---
            if "vmf_nce_multitask" in losses and is_vmf:
                dcf_extras = getattr(model, "_last_dcf_extras", {})
                mt_total, mt_fused, mt_aux = losses["vmf_nce_multitask"](
                    _vmf_pred_head, _vmf_aux_head, gt_embedding,
                    per_roi_mus=dcf_extras.get("per_roi_mus"),
                    per_roi_kappas=dcf_extras.get("per_roi_kappas"),
                    queue=queue,
                )
                total_loss = total_loss + loss_weights.get("vmf_nce_multitask", 1.0) * mt_total
                batch_metrics["mt_fused"] = mt_fused.item()
                batch_metrics["mt_aux"] = mt_aux.item()

            # --- Hierarchical CLIP alignment (v8) ---
            if "hierarchical_clip" in losses and is_vmf and hier_targets is not None:
                dcf_extras = getattr(model, "_last_dcf_extras", {})
                pr_mus = dcf_extras.get("per_roi_mus")
                alphas = dcf_extras.get("cls_to_roi_alpha")
                if pr_mus is not None and alphas is not None:
                    # Map column names to tier names
                    from fmri2img.losses.hierarchical_clip_loss import DEFAULT_TIER_CLIP_COLUMNS
                    col_to_tier = {v: k for k, v in DEFAULT_TIER_CLIP_COLUMNS.items()}
                    tier_tgts = {}
                    for col, tgt in hier_targets.items():
                        tier_name = col_to_tier.get(col)
                        if tier_name is not None:
                            tier_tgts[tier_name] = F.normalize(tgt.float(), p=2, dim=-1)
                    h_loss, h_details = losses["hierarchical_clip"](pr_mus, alphas, tier_tgts)
                    total_loss = total_loss + loss_weights.get("hierarchical_clip", 0.5) * h_loss
                    batch_metrics["hier_clip"] = h_loss.item()
                    for k, v in h_details.items():
                        batch_metrics[k] = v

            # --- CKA representational alignment (v8) ---
            if "cka" in losses:
                _cka_per_subj = (config_ref or {}).get("loss", {}).get(
                    "cka", {}).get("per_subject", False)
                if _cka_per_subj and subject_ids is not None:
                    # Per-subject CKA: align each subject's manifold
                    # structure to CLIP independently
                    _cka_parts = []
                    for sid in subject_ids.unique():
                        mask = subject_ids == sid
                        if mask.sum() >= 4:  # need >=4 for meaningful CKA
                            _cka_parts.append(
                                losses["cka"](pred[mask], gt_embedding[mask])
                            )
                    if _cka_parts:
                        cka_loss = torch.stack(_cka_parts).mean()
                    else:
                        cka_loss = losses["cka"](pred, gt_embedding)
                else:
                    cka_loss = losses["cka"](pred, gt_embedding)
                total_loss = total_loss + loss_weights.get("cka", 0.5) * cka_loss
                batch_metrics["cka"] = cka_loss.item()

            # --- Single queue enqueue (after all contrastive losses read the queue) ---
            if queue is not None:
                queue.enqueue(gt_embedding.detach())

            # --- Dual-head / triple-head regression MSE ---
            _reg_pred = getattr(model, "_last_reg_pred", None)
            _reg_mse_cfg = (config_ref or {}).get("loss", {}).get("regression_mse", {})
            if _reg_mse_cfg.get("enabled", False) and _reg_pred is not None:
                _reg_mse_start_epoch = int(_reg_mse_cfg.get("start_epoch", 0))
                if current_epoch > _reg_mse_start_epoch:
                    _reg_mse_w = loss_weights.get("regression_mse", _reg_mse_cfg.get("weight", 1.0))
                    _reg_target = _rich_target if _rich_target is not None else gt_embedding
                    _reg_loss = F.mse_loss(_reg_pred, _reg_target, reduction="mean")
                    total_loss = total_loss + _reg_mse_w * _reg_loss
                    batch_metrics["reg_mse"] = _reg_loss.item()
                else:
                    batch_metrics["reg_mse"] = 0.0

            # --- V30d: Rerank head SoftCLIP ---
            _rerank_pred = getattr(model, "_last_rerank_pred", None)
            _rerank_cfg = (config_ref or {}).get("loss", {}).get("rerank_softclip", {})
            if "rerank_softclip" in losses and _rerank_pred is not None and _rerank_target is not None:
                _rerank_start_epoch = int(_rerank_cfg.get("start_epoch", 0))
                if current_epoch > _rerank_start_epoch:
                    _rerank_w = loss_weights.get("rerank_softclip", _rerank_cfg.get("weight", 1.0))
                    _rerank_loss = losses["rerank_softclip"](_rerank_pred, _rerank_target, queue=None)
                    total_loss = total_loss + _rerank_w * _rerank_loss
                    batch_metrics["rerank_softclip"] = _rerank_loss.item()
                else:
                    batch_metrics["rerank_softclip"] = 0.0

            # --- V33: shortlist-local teacher distillation ---
            _std_cfg = (config_ref or {}).get("loss", {}).get("shortlist_teacher_distill", {})
            if (
                "shortlist_teacher_distill" in losses
                and _rerank_pred is not None
                and _rerank_target is not None
            ):
                _std_loss, _std_stats = losses["shortlist_teacher_distill"](
                    compact_pred=pred,
                    retrieval_target=gt_embedding,
                    rerank_pred=_rerank_pred,
                    rerank_target=_rerank_target,
                    return_stats=True,
                )
                _std_start_epoch = int(_std_cfg.get("start_epoch", 10))
                if current_epoch > _std_start_epoch:
                    _std_w = loss_weights.get(
                        "shortlist_teacher_distill",
                        _std_cfg.get("weight", 0.15),
                    )
                    total_loss = total_loss + _std_w * _std_loss
                    batch_metrics["shortlist_teacher_distill"] = _std_loss.item()
                else:
                    batch_metrics["shortlist_teacher_distill"] = 0.0
                batch_metrics["shortlist_teacher_gate_frac"] = _std_stats["gate_frac"]
                batch_metrics["shortlist_teacher_pos_rank_mean"] = _std_stats["teacher_pos_rank_mean"]
                batch_metrics["shortlist_teacher_pos_rank_median"] = _std_stats["teacher_pos_rank_median"]
                batch_metrics["shortlist_student_pos_rank_mean"] = _std_stats["student_pos_rank_mean"]
                batch_metrics["shortlist_student_pos_rank_median"] = _std_stats["student_pos_rank_median"]

            # --- V35: legacy-teacher distillation (frozen N1v28a -> compact) ---
            _ltd_cfg = (config_ref or {}).get("loss", {}).get("legacy_teacher_distill", {})
            if (
                "legacy_teacher_distill" in losses
                and _legacy_teacher_pred is not None
                and _rich_target is not None
            ):
                _ltd_loss, _ltd_stats = losses["legacy_teacher_distill"](
                    compact_pred=_pred_for_legacy,
                    retrieval_target=_gt_for_legacy,
                    teacher_pred=_legacy_teacher_pred,
                    teacher_target=_rich_target_for_legacy,
                    return_stats=True,
                )
                _ltd_start_epoch = int(_ltd_cfg.get("start_epoch", 10))
                if current_epoch > _ltd_start_epoch:
                    _ltd_w = loss_weights.get(
                        "legacy_teacher_distill",
                        _ltd_cfg.get("weight", 0.15),
                    )
                    total_loss = total_loss + _ltd_w * _ltd_loss
                    batch_metrics["legacy_teacher_distill"] = _ltd_loss.item()
                else:
                    batch_metrics["legacy_teacher_distill"] = 0.0
                batch_metrics["legacy_teacher_gate_frac"] = _ltd_stats["gate_frac"]
                batch_metrics["legacy_teacher_pos_rank_mean"] = _ltd_stats["teacher_pos_rank_mean"]
                batch_metrics["legacy_teacher_pos_rank_median"] = _ltd_stats["teacher_pos_rank_median"]
                batch_metrics["legacy_student_pos_rank_mean"] = _ltd_stats["student_pos_rank_mean"]
                batch_metrics["legacy_student_pos_rank_median"] = _ltd_stats["student_pos_rank_median"]

            # --- V38: full-batch legacy -> compact distillation ---
            _lcd_cfg = (config_ref or {}).get("loss", {}).get("legacy_compact_distill", {})
            if (
                "legacy_compact_distill" in losses
                and _legacy_teacher_pred is not None
                and _rich_target is not None
            ):
                _lcd_loss, _lcd_stats = losses["legacy_compact_distill"](
                    compact_pred=_pred_for_legacy,
                    retrieval_target=_gt_for_legacy,
                    teacher_pred=_legacy_teacher_pred,
                    teacher_target=_rich_target_for_legacy,
                    return_stats=True,
                )
                _lcd_start_epoch = int(_lcd_cfg.get("start_epoch", 10))
                if current_epoch > _lcd_start_epoch:
                    _lcd_w = loss_weights.get(
                        "legacy_compact_distill",
                        _lcd_cfg.get("weight", 0.15),
                    )
                    total_loss = total_loss + _lcd_w * _lcd_loss
                    batch_metrics["legacy_compact_distill"] = _lcd_loss.item()
                else:
                    batch_metrics["legacy_compact_distill"] = 0.0
                batch_metrics["legacy_compact_teacher_topk_hit_frac"] = _lcd_stats["teacher_topk_hit_frac"]
                batch_metrics["legacy_compact_teacher_pos_rank_mean"] = _lcd_stats["teacher_pos_rank_mean"]
                batch_metrics["legacy_compact_teacher_pos_rank_median"] = _lcd_stats["teacher_pos_rank_median"]
                batch_metrics["legacy_compact_student_pos_rank_mean"] = _lcd_stats["student_pos_rank_mean"]
                batch_metrics["legacy_compact_student_pos_rank_median"] = _lcd_stats["student_pos_rank_median"]
                batch_metrics["legacy_compact_active_candidate_size_mean"] = _lcd_stats["active_candidate_size_mean"]

            # --- V44: component-responsibility legacy -> compact distillation ---
            _clcd_cfg = (config_ref or {}).get("loss", {}).get("component_legacy_compact_distill", {})
            if (
                "component_legacy_compact_distill" in losses
                and _legacy_teacher_pred is not None
                and _rich_target is not None
                and _compact_component_mu is not None
                and _compact_component_kappa is not None
            ):
                _clcd_loss, _clcd_stats = losses["component_legacy_compact_distill"](
                    compact_component_mu=_compact_component_mu_for_legacy,
                    compact_component_kappa=_compact_component_kappa_for_legacy,
                    retrieval_target=_gt_for_legacy,
                    teacher_pred=_legacy_teacher_pred,
                    teacher_target=_rich_target_for_legacy,
                    component_logits=_compact_component_logits_for_legacy,
                    return_stats=True,
                )
                _clcd_start_epoch = int(_clcd_cfg.get("start_epoch", 10))
                if current_epoch > _clcd_start_epoch:
                    _clcd_w = loss_weights.get(
                        "component_legacy_compact_distill",
                        _clcd_cfg.get("weight", 0.10),
                    )
                    total_loss = total_loss + _clcd_w * _clcd_loss
                    batch_metrics["component_legacy_compact_distill"] = _clcd_loss.item()
                else:
                    batch_metrics["component_legacy_compact_distill"] = 0.0
                batch_metrics["component_legacy_teacher_topk_hit_frac"] = _clcd_stats["teacher_topk_hit_frac"]
                batch_metrics["component_legacy_teacher_pos_rank_mean"] = _clcd_stats["teacher_pos_rank_mean"]
                batch_metrics["component_legacy_teacher_pos_rank_median"] = _clcd_stats["teacher_pos_rank_median"]
                batch_metrics["component_legacy_student_pos_rank_mean"] = _clcd_stats["student_pos_rank_mean"]
                batch_metrics["component_legacy_student_pos_rank_median"] = _clcd_stats["student_pos_rank_median"]
                batch_metrics["component_legacy_active_candidate_size_mean"] = _clcd_stats["active_candidate_size_mean"]
                batch_metrics["component_legacy_responsible_component_entropy"] = _clcd_stats["responsible_component_entropy"]
                batch_metrics["component_legacy_responsible_component_top_rate"] = _clcd_stats["responsible_component_top_rate"]
                batch_metrics["component_legacy_responsible_component_mean"] = _clcd_stats["responsible_component_mean"]

            # --- V36: combined rerank + legacy teacher distillation ---
            _tri_cfg = (config_ref or {}).get("loss", {}).get("tri_teacher_distill", {})
            if (
                "tri_teacher_distill" in losses
                and _rerank_pred is not None
                and _rerank_target is not None
                and _legacy_teacher_pred is not None
                and _rich_target is not None
            ):
                _tri_loss, _tri_stats = losses["tri_teacher_distill"](
                    compact_pred=_pred_for_legacy,
                    retrieval_target=_gt_for_legacy,
                    rerank_pred=_maybe_mask_legacy_tensor(_rerank_pred, _legacy_teacher_mask),
                    rerank_target=_maybe_mask_legacy_tensor(_rerank_target, _legacy_teacher_mask),
                    legacy_pred=_legacy_teacher_pred,
                    legacy_target=_rich_target_for_legacy,
                    return_stats=True,
                )
                _tri_start_epoch = int(_tri_cfg.get("start_epoch", 10))
                if current_epoch > _tri_start_epoch:
                    _tri_w = loss_weights.get(
                        "tri_teacher_distill",
                        _tri_cfg.get("weight", 0.20),
                    )
                    total_loss = total_loss + _tri_w * _tri_loss
                    batch_metrics["tri_teacher_distill"] = _tri_loss.item()
                else:
                    batch_metrics["tri_teacher_distill"] = 0.0
                batch_metrics["tri_teacher_gate_frac"] = _tri_stats["gate_frac"]
                batch_metrics["tri_teacher_candidate_size_mean"] = _tri_stats["candidate_size_mean"]
                batch_metrics["tri_rerank_teacher_pos_rank_mean"] = _tri_stats["rerank_teacher_pos_rank_mean"]
                batch_metrics["tri_rerank_teacher_pos_rank_median"] = _tri_stats["rerank_teacher_pos_rank_median"]
                batch_metrics["tri_legacy_teacher_pos_rank_mean"] = _tri_stats["legacy_teacher_pos_rank_mean"]
                batch_metrics["tri_legacy_teacher_pos_rank_median"] = _tri_stats["legacy_teacher_pos_rank_median"]
                batch_metrics["tri_combined_teacher_pos_rank_mean"] = _tri_stats["combined_teacher_pos_rank_mean"]
                batch_metrics["tri_combined_teacher_pos_rank_median"] = _tri_stats["combined_teacher_pos_rank_median"]
                batch_metrics["tri_student_pos_rank_mean"] = _tri_stats["student_pos_rank_mean"]
                batch_metrics["tri_student_pos_rank_median"] = _tri_stats["student_pos_rank_median"]

            # --- Kappa regularizer ---
            kappa_reg_cfg = config_ref.get("loss", {}).get("kappa_reg", {}) if config_ref else {}
            if kappa_reg_cfg.get("enabled", False) and is_vmf and _vmf_aux_head is not None:
                kappa_vals = _vmf_aux_head.squeeze(-1) if not vmf_is_log else _vmf_aux_head.exp().squeeze(-1)
                kr = kappa_regularizer(kappa_vals, kappa_reg_cfg.get("lambda_kappa", 0.01))
                total_loss = total_loss + kr
                batch_metrics["kappa_reg"] = kr.item()

            # --- Direct cosine alignment (V11, fixed V19: use raw pred) ---
            if "direct_alignment" in losses and is_vmf:
                da_loss = losses["direct_alignment"](_vmf_pred_head, gt_embedding)
                total_loss = total_loss + loss_weights.get("direct_alignment", 0.5) * da_loss
                batch_metrics["direct_align"] = da_loss.item()

            # --- Spherical uniformity regularisation (V11) ---
            if "uniformity" in losses:
                uni_loss = losses["uniformity"](pred_for_contrast)
                total_loss = total_loss + loss_weights.get("uniformity", 0.1) * uni_loss
                batch_metrics["uniformity"] = uni_loss.item()

            # --- R-Drop: consistency between two forward passes (V9) ---
            _rdrop_cfg = (config_ref or {}).get("loss", {}).get("r_drop", {})
            _rdrop_start = _rdrop_cfg.get("start_epoch", 0)
            if _rdrop_cfg.get("enabled", False) and is_vmf and current_epoch >= _rdrop_start:
                output2 = model(fmri, subject_ids=subject_ids) if subject_ids is not None else model(fmri)
                pred2, aux2 = (output2 if isinstance(output2, tuple) else (output2, None))
                _vmf_pred_head2, _vmf_aux_head2 = _extract_vmf_outputs_for_losses(model_type, pred2, aux2)
                if _vmf_aux_head2 is not None and _vmf_pred_head2 is not None:
                    k1 = _vmf_aux_head.squeeze(-1) if not vmf_is_log else _vmf_aux_head.exp().squeeze(-1)
                    k2 = _vmf_aux_head2.squeeze(-1) if not vmf_is_log else _vmf_aux_head2.exp().squeeze(-1)
                    rd_loss = vmf_rdrop_loss(_vmf_pred_head, k1, _vmf_pred_head2, k2)
                    _rdrop_w = _rdrop_cfg.get("weight", 0.5)
                    total_loss = total_loss + _rdrop_w * rd_loss
                    batch_metrics["r_drop"] = rd_loss.item()

            # --- Kappa statistics ---
            if is_vmf and _vmf_aux_head is not None:
                with torch.no_grad():
                    kv = _vmf_aux_head.squeeze(-1) if not vmf_is_log else _vmf_aux_head.exp().squeeze(-1)
                    kv = kv.float()
                    batch_metrics["kappa_mean"] = kv.mean().item()
                    batch_metrics["kappa_std"] = kv.std().item()
                    batch_metrics["kappa_min"] = kv.min().item()
                    batch_metrics["kappa_max"] = kv.max().item()
                    batch_metrics["kappa_q90"] = kv.quantile(0.9).item()

            # --- KL divergence ---
            if kl_scheduler is not None and is_gaussian:
                kl_raw = compute_kl_divergence(pred, aux)
                kl_weight = kl_scheduler.step()
                total_loss = total_loss + kl_weight * kl_raw
                batch_metrics["kl"] = (kl_weight * kl_raw).item()

            if kl_scheduler is not None and is_vmf:
                log_kappa_for_kl = _vmf_aux_head if vmf_is_log else torch.log(_vmf_aux_head.clamp(min=1e-8))
                kl_raw = compute_vmf_kl(_vmf_pred_head, log_kappa_for_kl, _vmf_pred_head.size(-1))
                kl_weight = kl_scheduler.step()
                total_loss = total_loss + kl_weight * kl_raw
                batch_metrics["kl"] = (kl_weight * kl_raw).item()

            # --- MixCo augmentation (second forward pass with soft labels) ---
            if mixco_cfg is not None and mixco_cfg.get("enabled", False):
                fmri_mix, gt_mix, soft_labels = mixco_augment(
                    fmri, gt_embedding, alpha=mixco_cfg.get("alpha", 0.2),
                    use_slerp=mixco_cfg.get("use_slerp", False),
                )
                pred_mix = model(fmri_mix, subject_ids=subject_ids)
                if isinstance(pred_mix, tuple):
                    pred_mix = pred_mix[0]
                pred_mix_c = _proj_head(pred_mix) if _proj_head is not None else pred_mix
                mc_loss = mixco_nce_loss(
                    pred_mix_c, gt_mix, soft_labels,
                    temperature=mixco_cfg.get("temperature", 0.006),
                )
                total_loss = total_loss + mixco_cfg.get("weight", 1.0) * mc_loss
                batch_metrics["mixco"] = mc_loss.item()

            # Homoscedastic uncertainty regularization (Kendall et al., 2018)
            if log_sigmas is not None:
                for _aw_name, _aw_ls in log_sigmas.items():
                    total_loss = total_loss + 0.5 * _aw_ls.squeeze()

            total_loss = total_loss / grad_accum_steps

        batch_metrics["loss"] = total_loss.item() * grad_accum_steps

        if not torch.isfinite(total_loss):
            logger.warning("NaN/Inf loss at step %d — skipping gradient update", global_step)
            optimizer.zero_grad()
            global_step += 1
            continue

        scaler.scale(total_loss).backward() if scaler is not None else total_loss.backward()

        if (step_in_epoch + 1) % grad_accum_steps == 0 or (step_in_epoch + 1) == len(dataloader):
            if scaler is not None:
                scaler.unscale_(optimizer)
            all_params = list(model.parameters())
            for lm in losses.values():
                if isinstance(lm, nn.Module):
                    all_params.extend(lm.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

        # EMA update after each optimizer step
        if ema is not None and ((step_in_epoch + 1) % grad_accum_steps == 0 or (step_in_epoch + 1) == len(dataloader)):
            ema.update(model)

        pbar.set_postfix({k: f"{v:.4f}" for k, v in batch_metrics.items()})
        for k, v in batch_metrics.items():
            epoch_metrics.setdefault(k, []).append(v)
        global_step += 1

    return {k: float(np.mean(v)) for k, v in epoch_metrics.items()}, global_step


def mc_dropout_tta(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    preprocessor: Optional[EmbeddingPreprocessor],
    n_samples: int = 8,
    vmf_is_log: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """MC-Dropout Test-Time Augmentation for vMF models.

    Runs ``n_samples`` forward passes with dropout enabled, averages the
    mu predictions on the sphere (mean + L2-renorm), and optionally
    returns kappa values per trial for kappa-weighted repetition averaging.

    Uses **online (incremental) averaging** to avoid stacking all K
    forward-pass results in RAM simultaneously.  For 329K-D token targets
    the naive stack would consume ~25 GB RAM at K=8, causing OOM-kill.

    Returns:
        (all_preds, all_gts, all_kappas) where kappas is (N,) or None.
    """
    model.train()  # enable dropout
    model_type = getattr(model, "model_type", "deterministic")

    mu_sum: Optional[np.ndarray] = None      # running sum (N, D)
    kappa_sum: Optional[np.ndarray] = None    # running sum (N,)
    all_gts: List[np.ndarray] = []
    has_kappa = False

    for sample_idx in range(n_samples):
        batch_preds, batch_kappas = [], []
        for batch in dataloader:
            if isinstance(batch, dict):
                fmri = batch["fmri"].to(device, dtype=torch.float32)
                gt_embedding = batch["retrieval_target"].to(device, dtype=torch.float32)
                subject_ids = batch["subject_id"].to(device)
            elif len(batch) == 4:
                fmri, gt_embedding, subject_ids, _ = batch
                subject_ids = subject_ids.to(device)
                fmri = fmri.to(device, dtype=torch.float32)
                gt_embedding = gt_embedding.to(device, dtype=torch.float32)
            elif len(batch) == 3:
                fmri, gt_embedding, subject_ids = batch
                subject_ids = subject_ids.to(device)
                fmri = fmri.to(device, dtype=torch.float32)
                gt_embedding = gt_embedding.to(device, dtype=torch.float32)
            else:
                fmri, gt_embedding = batch
                subject_ids = None
                fmri = fmri.to(device, dtype=torch.float32)
                gt_embedding = gt_embedding.to(device, dtype=torch.float32)

            if preprocessor is not None:
                gt_np = gt_embedding.cpu().numpy()
                gt_embedding = torch.from_numpy(
                    preprocessor.transform(gt_np)
                ).float().to(device)

            with torch.no_grad():
                output = (
                    model(fmri, subject_ids=subject_ids)
                    if subject_ids is not None else model(fmri)
                )
                pred, aux = (output if isinstance(output, tuple) else (output, None))
                batch_preds.append(pred.cpu().numpy())
                if aux is not None and model_type in ("vmf", "vmf_dcf", "vmf_triple"):
                    k = aux.squeeze(-1)
                    if vmf_is_log:
                        k = k.exp()
                    batch_kappas.append(k.cpu().numpy())
                if sample_idx == 0:
                    all_gts.append(gt_embedding.cpu().numpy())

        # Accumulate this sample's predictions into running sum
        sample_mu = np.concatenate(batch_preds, axis=0)  # (N, D)
        if mu_sum is None:
            mu_sum = sample_mu
        else:
            mu_sum += sample_mu
        del sample_mu, batch_preds  # free immediately

        if batch_kappas:
            has_kappa = True
            sample_k = np.concatenate(batch_kappas, axis=0)
            if kappa_sum is None:
                kappa_sum = sample_k
            else:
                kappa_sum += sample_k
            del sample_k, batch_kappas

    model.eval()  # restore

    # Average + renormalise on the sphere
    mu_mean = mu_sum / n_samples
    del mu_sum
    norms = np.linalg.norm(mu_mean, axis=-1, keepdims=True)
    mu_mean = mu_mean / np.maximum(norms, 1e-8)

    gts = np.concatenate(all_gts, axis=0)
    kappas = (kappa_sum / n_samples) if has_kappa else None
    return mu_mean, gts, kappas


# ---------------------------------------------------------------------------
# Shared1000 benchmark evaluation
# ---------------------------------------------------------------------------

def _evaluate_shared1000(
    model: nn.Module,
    subject: str,
    device: str,
    embeddings_df: pd.DataFrame,
    preprocessor: Optional["EmbeddingPreprocessor"] = None,
    vmf_is_log: bool = False,
    zscore_stats_path: Optional[str] = None,
    zscore_mode: str = "global",
    batch_size: int = 64,
    is_multi_subject: bool = False,
    subject_id: int = 0,
    token_cache=None,
    rerank_cache=None,
    retrieval_projector=None,
    legacy_teacher_model: Optional[nn.Module] = None,
) -> Optional[Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
    """Evaluate on NSD shared1000 benchmark for community-standard comparison.

    Loads raw pre-extracted features independently of the training dataset,
    filters to shared1000 trials, averages 3 repetitions per image, applies
    z-scoring with saved training stats, and computes retrieval metrics on a
    ~982-image gallery.

    Returns ``(metrics_dict, predictions, ground_truth)`` or *None* on failure.
    """
    cache_root = os.environ.get("CACHE_ROOT", "cache")
    features_path = Path(cache_root) / "preextracted" / f"subject={subject}" / "fmri_features.npy"
    index_path = Path("data/indices/nsd_index") / f"subject={subject}" / "index.parquet"

    if not features_path.exists():
        logger.warning("Shared1000 eval: features not found at %s — skipping", features_path)
        return None
    if not index_path.exists():
        logger.warning("Shared1000 eval: index not found at %s — skipping", index_path)
        return None

    index_df = pd.read_parquet(index_path)
    if "shared1000" not in index_df.columns:
        logger.warning("Shared1000 eval: no 'shared1000' column in index — skipping")
        return None

    features = np.load(features_path, mmap_mode="r")
    s1000_mask = index_df["shared1000"].fillna(False).astype(bool).values
    n_raw = int(s1000_mask.sum())
    if n_raw == 0:
        logger.warning("Shared1000 eval: zero shared1000 trials found — skipping")
        return None

    s1000_features = np.array(features[s1000_mask], dtype=np.float32)
    s1000_df = index_df[s1000_mask].reset_index(drop=True)

    # --- Z-scoring with saved training stats ---
    if zscore_stats_path is not None:
        zdir = Path(zscore_stats_path)
        if zscore_mode == "per_session" and "session" in s1000_df.columns:
            fb_mean_path = zdir / "global_fallback_mean.npy"
            fb_std_path = zdir / "global_fallback_std.npy"
            fb_mean = np.load(fb_mean_path) if fb_mean_path.exists() else None
            fb_std = np.load(fb_std_path) if fb_std_path.exists() else None
            sessions = s1000_df["session"].values
            _zs_applied = 0
            for sess in np.unique(sessions):
                # Multi-subject saves as {subj}_session_{sess}_*.npy;
                # single-subject saves as session_{sess}_*.npy.  Try both.
                m_path = zdir / f"{subject}_session_{int(sess)}_mean.npy"
                s_path = zdir / f"{subject}_session_{int(sess)}_std.npy"
                if not m_path.exists():
                    m_path = zdir / f"session_{int(sess)}_mean.npy"
                    s_path = zdir / f"session_{int(sess)}_std.npy"
                sess_mask = sessions == sess
                if m_path.exists() and s_path.exists():
                    s_mean = np.load(m_path)
                    s_std = np.load(s_path)
                elif fb_mean is not None and fb_std is not None:
                    s_mean, s_std = fb_mean, fb_std
                else:
                    continue
                s1000_features[sess_mask] = (
                    (s1000_features[sess_mask] - s_mean) / s_std
                ).astype(np.float32)
                _zs_applied += int(sess_mask.sum())
            if _zs_applied == 0:
                logger.warning(
                    "Shared1000 eval: no per-session z-score stats found in %s — using raw features", zdir
                )
            else:
                logger.info("Shared1000 eval: z-scored %d/%d trials (per_session)", _zs_applied, len(s1000_features))
        else:
            # Multi-subject saves as {subj}_mean.npy; single-subject as voxel_mean.npy
            m_path = zdir / f"{subject}_mean.npy"
            s_path = zdir / f"{subject}_std.npy"
            if not m_path.exists():
                m_path = zdir / "voxel_mean.npy"
                s_path = zdir / "voxel_std.npy"
            if m_path.exists() and s_path.exists():
                v_mean = np.load(m_path)
                v_std = np.load(s_path)
                s1000_features = ((s1000_features - v_mean) / v_std).astype(np.float32)
                logger.info("Shared1000 eval: applied global z-scoring from %s", m_path.name)
            else:
                logger.warning("Shared1000 eval: z-score stats not found at %s — using raw features", zdir)

    nsd_ids = s1000_df["nsdId"].values
    unique_ids = np.unique(nsd_ids)
    n_images = len(unique_ids)

    model_type = getattr(model, "model_type", "deterministic")
    is_vmf_model = model_type in ("vmf", "vmf_dcf", "vmf_triple")
    model.eval()

    _uses_compact_cls_space = model_type in ("vmf_triple", "dense_vmf_hybrid")
    _has_component_outputs = model_type == "vmf_triple"
    _all_rich_preds_s1000: List[np.ndarray] = []
    _all_rerank_preds_s1000: List[np.ndarray] = []
    _all_legacy_preds_s1000: List[np.ndarray] = []
    _all_vmf_preds_s1000: List[np.ndarray] = []
    _all_vmf_kappas_s1000: List[np.ndarray] = []
    _all_compact_component_mu_s1000: List[np.ndarray] = []
    _all_compact_component_kappa_s1000: List[np.ndarray] = []
    _all_compact_component_logits_s1000: List[np.ndarray] = []
    _vmf_preds_img: Optional[np.ndarray] = None
    _vmf_kappas_img: Optional[np.ndarray] = None

    if is_vmf_model:
        # --- vMF path: run ALL individual trials, fuse with kappa weights ---
        tensor_ds = torch.utils.data.TensorDataset(torch.from_numpy(s1000_features))
        loader = DataLoader(tensor_ds, batch_size=batch_size, shuffle=False)
        all_preds: List[np.ndarray] = []
        all_kappas: List[np.ndarray] = []
        with torch.no_grad():
            for (batch_fmri,) in loader:
                batch_fmri = batch_fmri.to(device, dtype=torch.float32)
                if is_multi_subject:
                    sid = torch.full((batch_fmri.shape[0],), subject_id,
                                     dtype=torch.long, device=device)
                    out = model(batch_fmri, subject_ids=sid)
                else:
                    out = model(batch_fmri)
                pred, aux = (out if isinstance(out, tuple) else (out, None))
                all_preds.append(pred.cpu().numpy())
                if aux is not None:
                    k = aux.squeeze(-1)
                    if vmf_is_log:
                        k = k.exp()
                    all_kappas.append(k.cpu().numpy())
                if _uses_compact_cls_space:
                    _vmf_pred_head, _vmf_aux_head = _extract_vmf_outputs_for_losses(model_type, pred, aux)
                    if _vmf_pred_head is not None and _vmf_aux_head is not None:
                        _all_vmf_preds_s1000.append(_vmf_pred_head.detach().cpu().numpy())
                        _vk = _vmf_aux_head
                        if vmf_is_log:
                            _vk = _vk.exp()
                        _all_vmf_kappas_s1000.append(_vk.squeeze(-1).detach().cpu().numpy())
                if _uses_compact_cls_space:
                    _rp = getattr(model, "_last_rich_pred", None)
                    if _rp is not None:
                        _all_rich_preds_s1000.append(_rp.detach().cpu().numpy())
                    _rrp = getattr(model, "_last_rerank_pred", None)
                    if _rrp is not None:
                        _all_rerank_preds_s1000.append(_rrp.detach().cpu().numpy())
                    _cmu = getattr(model, "_last_compact_component_mu", None)
                    _ckappa = getattr(model, "_last_compact_component_kappa", None)
                    _clogits = getattr(model, "_last_compact_component_logits", None)
                    if _cmu is not None and _ckappa is not None:
                        _all_compact_component_mu_s1000.append(_cmu.detach().cpu().numpy())
                        _all_compact_component_kappa_s1000.append(_ckappa.detach().cpu().numpy())
                        if _clogits is not None:
                            _all_compact_component_logits_s1000.append(_clogits.detach().cpu().numpy())
                    if legacy_teacher_model is not None:
                        _legacy_out = (
                            legacy_teacher_model(batch_fmri, subject_ids=sid)
                            if is_multi_subject
                            else legacy_teacher_model(batch_fmri)
                        )
                        _legacy_pred = _legacy_out[0] if isinstance(_legacy_out, tuple) else _legacy_out
                        _all_legacy_preds_s1000.append(_legacy_pred.detach().cpu().numpy())

        trial_preds = np.concatenate(all_preds)
        trial_kappas = np.concatenate(all_kappas) if all_kappas else None
        _component_mu_img: Optional[np.ndarray] = None
        _component_kappa_img: Optional[np.ndarray] = None
        _component_logits_img: Optional[np.ndarray] = None
        if _all_compact_component_mu_s1000 and _all_compact_component_kappa_s1000:
            trial_component_mu = np.concatenate(_all_compact_component_mu_s1000)
            trial_component_kappa = np.concatenate(_all_compact_component_kappa_s1000)
            trial_component_logits = (
                np.concatenate(_all_compact_component_logits_s1000)
                if _all_compact_component_logits_s1000 else None
            )
            _component_mu_img, _component_kappa_img, _component_logits_img, _ = _aggregate_component_outputs_by_nsd_id(
                trial_component_mu,
                trial_component_kappa,
                trial_component_logits,
                nsd_ids,
            )

        preds = np.zeros((n_images, trial_preds.shape[1]), dtype=np.float32)
        preds_avg = np.zeros_like(preds)
        for i, uid in enumerate(unique_ids):
            mask = nsd_ids == uid
            preds_avg[i] = trial_preds[mask].mean(axis=0)
            if trial_kappas is not None:
                kw = trial_kappas[mask]
                kw = kw / (kw.sum() + 1e-8)
                preds[i] = (trial_preds[mask] * kw[:, None]).sum(axis=0)
            else:
                preds[i] = preds_avg[i]

        norms = np.linalg.norm(preds, axis=-1, keepdims=True)
        preds = preds / np.maximum(norms, 1e-8)
        norms_avg = np.linalg.norm(preds_avg, axis=-1, keepdims=True)
        preds_avg = preds_avg / np.maximum(norms_avg, 1e-8)
        logger.info("Shared1000: using kappa-weighted spherical Frechet mean (%d trials -> %d images)",
                     len(trial_preds), n_images)
    else:
        # --- Non-vMF path: average fMRI features, single forward pass ---
        avg_features = np.zeros((n_images, s1000_features.shape[1]), dtype=np.float32)
        for i, uid in enumerate(unique_ids):
            avg_features[i] = s1000_features[nsd_ids == uid].mean(axis=0)

        all_preds_list: List[np.ndarray] = []
        tensor_ds = torch.utils.data.TensorDataset(torch.from_numpy(avg_features))
        loader = DataLoader(tensor_ds, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for (batch_fmri,) in loader:
                batch_fmri = batch_fmri.to(device, dtype=torch.float32)
                if is_multi_subject:
                    sid = torch.full((batch_fmri.shape[0],), subject_id,
                                     dtype=torch.long, device=device)
                    out = model(batch_fmri, subject_ids=sid)
                else:
                    out = model(batch_fmri)
                pred, aux = (out if isinstance(out, tuple) else (out, None))
                all_preds_list.append(pred.cpu().numpy())
                if _uses_compact_cls_space:
                    _vmf_pred_head, _vmf_aux_head = _extract_vmf_outputs_for_losses(model_type, pred, aux)
                    if _vmf_pred_head is not None and _vmf_aux_head is not None:
                        _all_vmf_preds_s1000.append(_vmf_pred_head.detach().cpu().numpy())
                        _vk = _vmf_aux_head
                        if vmf_is_log:
                            _vk = _vk.exp()
                        _all_vmf_kappas_s1000.append(_vk.squeeze(-1).detach().cpu().numpy())
                    _rrp = getattr(model, "_last_rerank_pred", None)
                    if _rrp is not None:
                        _all_rerank_preds_s1000.append(_rrp.detach().cpu().numpy())
                    _rp = getattr(model, "_last_rich_pred", None)
                    if _rp is not None:
                        _all_rich_preds_s1000.append(_rp.detach().cpu().numpy())
                    if legacy_teacher_model is not None:
                        _legacy_out = (
                            legacy_teacher_model(batch_fmri, subject_ids=sid)
                            if is_multi_subject
                            else legacy_teacher_model(batch_fmri)
                        )
                        _legacy_pred = _legacy_out[0] if isinstance(_legacy_out, tuple) else _legacy_out
                        _all_legacy_preds_s1000.append(_legacy_pred.detach().cpu().numpy())

        preds = np.concatenate(all_preds_list)
        norms = np.linalg.norm(preds, axis=-1, keepdims=True)
        preds = preds / np.maximum(norms, 1e-8)
        preds_avg = None
        if _all_vmf_preds_s1000:
            _vmf_preds_img = np.concatenate(_all_vmf_preds_s1000)
            _vmf_preds_img = _vmf_preds_img / np.maximum(
                np.linalg.norm(_vmf_preds_img, axis=-1, keepdims=True), 1e-8
            )
        if _all_vmf_kappas_s1000:
            _vmf_kappas_img = np.concatenate(_all_vmf_kappas_s1000)

    # --- Ground-truth CLIP embeddings ---
    # vmf_triple: compact preds are in retrieval_dim space (e.g. 768-D or 1024-D).
    # Use CLS GTs regardless of token_cache, since retrieval is in compact space.
    # Also build rich GTs from token_cache if available for two-stage.
    _rich_gts_s1000: Optional[np.ndarray] = None
    _legacy_preds_img: Optional[np.ndarray] = None
    _legacy_gts_s1000: Optional[np.ndarray] = None
    if _uses_compact_cls_space:
        emb_col = resolve_embedding_column(embeddings_df, _EMBEDDING_COLUMN_OVERRIDE)
        emb_lookup: Dict[int, int] = {}
        if "nsdId" in embeddings_df.columns:
            for i, (_, row) in enumerate(embeddings_df.iterrows()):
                emb_lookup[int(row["nsdId"])] = i
        gts = np.zeros((n_images, preds.shape[1]), dtype=np.float32)
        missing = 0
        for i, uid in enumerate(unique_ids):
            idx = emb_lookup.get(int(uid))
            if idx is not None:
                gts[i] = np.asarray(embeddings_df.iloc[idx][emb_col], dtype=np.float32)
            else:
                missing += 1
        if missing > 0:
            logger.warning("Shared1000 eval (compact GT): %d/%d missing", missing, n_images)

        if token_cache is not None and _all_rich_preds_s1000:
            trial_rich = np.concatenate(_all_rich_preds_s1000)
            _rich_preds_img = np.zeros((n_images, trial_rich.shape[1]), dtype=np.float32)
            _rich_gts_s1000 = np.zeros_like(_rich_preds_img)
            if trial_rich.shape[0] == n_images:
                _rich_preds_img = trial_rich
                for i, uid in enumerate(unique_ids):
                    if int(uid) in token_cache:
                        _rich_gts_s1000[i] = token_cache.get_flat(int(uid))
            else:
                for i, uid in enumerate(unique_ids):
                    mask = nsd_ids == uid
                    _rich_preds_img[i] = trial_rich[mask].mean(axis=0)
                    if int(uid) in token_cache:
                        _rich_gts_s1000[i] = token_cache.get_flat(int(uid))

        # V30d: aggregate rerank predictions per image
        _rerank_preds_img: Optional[np.ndarray] = None
        _rerank_gts_s1000: Optional[np.ndarray] = None
        if _all_rerank_preds_s1000 and rerank_cache is not None:
            trial_rerank = np.concatenate(_all_rerank_preds_s1000)
            _rerank_preds_img = np.zeros((n_images, trial_rerank.shape[1]), dtype=np.float32)
            _rerank_gts_s1000 = np.zeros((n_images, trial_rerank.shape[1]), dtype=np.float32)
            if trial_rerank.shape[0] == n_images:
                _rerank_preds_img = trial_rerank
                for i, uid in enumerate(unique_ids):
                    if int(uid) in rerank_cache:
                        _rerank_gts_s1000[i] = rerank_cache[int(uid)]
            else:
                for i, uid in enumerate(unique_ids):
                    mask = nsd_ids == uid
                    _rerank_preds_img[i] = trial_rerank[mask].mean(axis=0)
                    if int(uid) in rerank_cache:
                        _rerank_gts_s1000[i] = rerank_cache[int(uid)]
            # Re-normalise after averaging
            _rerank_preds_img = _rerank_preds_img / np.maximum(
                np.linalg.norm(_rerank_preds_img, axis=-1, keepdims=True), 1e-8)

        if _all_legacy_preds_s1000 and token_cache is not None:
            trial_legacy = np.concatenate(_all_legacy_preds_s1000)
            _legacy_preds_img = np.zeros((n_images, trial_legacy.shape[1]), dtype=np.float32)
            _legacy_gts_s1000 = np.zeros_like(_legacy_preds_img)
            if trial_legacy.shape[0] == n_images:
                _legacy_preds_img = trial_legacy
                for i, uid in enumerate(unique_ids):
                    if int(uid) in token_cache:
                        _legacy_gts_s1000[i] = token_cache.get_flat(int(uid))
            else:
                for i, uid in enumerate(unique_ids):
                    mask = nsd_ids == uid
                    _legacy_preds_img[i] = trial_legacy[mask].mean(axis=0)
                    if int(uid) in token_cache:
                        _legacy_gts_s1000[i] = token_cache.get_flat(int(uid))
            _legacy_preds_img = _legacy_preds_img / np.maximum(
                np.linalg.norm(_legacy_preds_img, axis=-1, keepdims=True), 1e-8)

    elif token_cache is not None:
        gts = np.zeros((n_images, preds.shape[1]), dtype=np.float32)
        missing = 0
        for i, uid in enumerate(unique_ids):
            if int(uid) in token_cache:
                gts[i] = token_cache.get_flat(int(uid))
            else:
                missing += 1
        if missing > 0:
            logger.warning("Shared1000 eval (token): %d/%d images missing from token cache", missing, n_images)
    else:
        emb_col = resolve_embedding_column(embeddings_df, _EMBEDDING_COLUMN_OVERRIDE)
        emb_lookup: Dict[int, int] = {}
        if "nsdId" in embeddings_df.columns:
            for i, (_, row) in enumerate(embeddings_df.iterrows()):
                emb_lookup[int(row["nsdId"])] = i

        gts = np.zeros((n_images, preds.shape[1]), dtype=np.float32)
        missing = 0
        for i, uid in enumerate(unique_ids):
            idx = emb_lookup.get(int(uid))
            if idx is not None:
                emb = embeddings_df.iloc[idx][emb_col]
                gts[i] = np.asarray(emb, dtype=np.float32)
            else:
                missing += 1
        if missing > 0:
            logger.warning("Shared1000 eval: %d/%d images missing from CLIP cache", missing, n_images)

    if retrieval_projector is not None:
        gts = retrieval_projector.transform(gts)
    if preprocessor is not None:
        gts = preprocessor.transform(gts)

    gt_norms = np.linalg.norm(gts, axis=-1, keepdims=True)
    gts = gts / np.maximum(gt_norms, 1e-8)

    # --- Retrieval metrics (primary: kappa-weighted for vMF, standard otherwise) ---
    retrieval = _compute_retrieval(preds, gts, ks=(1, 5, 10))
    csls_ret = _compute_retrieval_csls(preds, gts, ks=(1, 5, 10), csls_k=10)

    diag_sim = np.sum(preds * gts, axis=-1)
    mean_pos_sim = float(np.mean(diag_sim))

    metrics: Dict[str, Any] = {
        "benchmark": "shared1000",
        "subject": subject,
        "gallery_size": n_images,
        "n_raw_trials": n_raw,
        "clip_model": "auto",  # detected from data, not hardcoded
        "clip_dim": int(preds.shape[1]),
        "r@1": float(retrieval["top1_accuracy"]),
        "r@5": float(retrieval["top5_accuracy"]),
        "r@10": float(retrieval["top10_accuracy"]),
        "median_rank": float(retrieval["median_rank"]),
        "mrr": float(retrieval["mrr"]),
        "csls_r@1": float(csls_ret["top1_accuracy"]),
        "csls_r@5": float(csls_ret["top5_accuracy"]),
        "csls_r@10": float(csls_ret["top10_accuracy"]),
        "mean_pos_sim": mean_pos_sim,
    }

    # Standard-average comparison for vMF models
    if preds_avg is not None:
        ret_avg = _compute_retrieval(preds_avg, gts, ks=(1, 5, 10))
        csls_avg = _compute_retrieval_csls(preds_avg, gts, ks=(1, 5, 10), csls_k=10)
        metrics["r@1_avg"] = float(ret_avg["top1_accuracy"])
        metrics["csls_r@1_avg"] = float(csls_avg["top1_accuracy"])
        logger.info(
            "Shared1000 (kappa-wtd): R@1=%.4f  CSLS_R@1=%.4f  |  "
            "(std-avg): R@1=%.4f  CSLS_R@1=%.4f",
            metrics["r@1"], metrics["csls_r@1"],
            metrics["r@1_avg"], metrics["csls_r@1_avg"],
        )

    if _vmf_preds_img is not None:
        vmf_ret = _compute_retrieval(_vmf_preds_img, gts, ks=(1, 5, 10))
        vmf_csls_ret = _compute_retrieval_csls(_vmf_preds_img, gts, ks=(1, 5, 10), csls_k=10)
        metrics["vmf_r@1"] = float(vmf_ret["top1_accuracy"])
        metrics["vmf_r@5"] = float(vmf_ret["top5_accuracy"])
        metrics["vmf_r@10"] = float(vmf_ret["top10_accuracy"])
        metrics["vmf_median_rank"] = float(vmf_ret["median_rank"])
        metrics["vmf_mrr"] = float(vmf_ret["mrr"])
        metrics["vmf_csls_r@1"] = float(vmf_csls_ret["top1_accuracy"])
        metrics["vmf_csls_r@5"] = float(vmf_csls_ret["top5_accuracy"])
        metrics["vmf_csls_r@10"] = float(vmf_csls_ret["top10_accuracy"])
        metrics["vmf_csls_median_rank"] = float(vmf_csls_ret["median_rank"])
        metrics["vmf_csls_mrr"] = float(vmf_csls_ret["mrr"])
        if _vmf_kappas_img is not None:
            metrics["vmf_kappa_mean_eval"] = float(np.mean(_vmf_kappas_img))
            metrics["vmf_kappa_std_eval"] = float(np.std(_vmf_kappas_img))

    if _has_component_outputs and '_component_mu_img' in locals() and _component_mu_img is not None and _component_kappa_img is not None:
        mix_ret = _compute_mixture_vmf_retrieval(
            _component_mu_img,
            _component_kappa_img,
            gts,
            component_logits=_component_logits_img,
            ks=(1, 5, 10),
            normalize=True,
        )
        mix_csls_ret = _compute_mixture_vmf_retrieval_csls(
            _component_mu_img,
            _component_kappa_img,
            gts,
            component_logits=_component_logits_img,
            ks=(1, 5, 10),
            normalize=True,
            csls_k=10,
        )
        metrics["mixture_r@1"] = float(mix_ret["top1_accuracy"])
        metrics["mixture_r@5"] = float(mix_ret["top5_accuracy"])
        metrics["mixture_r@10"] = float(mix_ret["top10_accuracy"])
        metrics["mixture_median_rank"] = float(mix_ret["median_rank"])
        metrics["mixture_mrr"] = float(mix_ret["mrr"])
        metrics["mixture_csls_r@1"] = float(mix_csls_ret["top1_accuracy"])
        metrics["mixture_csls_r@5"] = float(mix_csls_ret["top5_accuracy"])
        metrics["mixture_csls_r@10"] = float(mix_csls_ret["top10_accuracy"])
        metrics["mixture_csls_median_rank"] = float(mix_csls_ret["median_rank"])
        metrics["mixture_csls_mrr"] = float(mix_csls_ret["mrr"])
        _mix_diag = _compute_mixture_component_diagnostics(
            _component_mu_img,
            _component_kappa_img,
            component_logits=_component_logits_img,
        )
        metrics.update({k: float(v) for k, v in _mix_diag.items()})
        logger.info(
            "Shared1000 mixture diagnostics: pairwise_cos=%.6f  weight_entropy=%.4f  top_weight=%.4f  kappa_across_std=%.6f",
            metrics["component_pairwise_cos_mean"],
            metrics["component_weight_entropy_mean"],
            metrics["component_top_weight_mean"],
            metrics["component_kappa_across_component_std_mean"],
        )
        if (
            metrics["component_pairwise_cos_mean"] > 0.999
            and metrics["component_weight_entropy_norm_mean"] > 0.99
            and metrics["component_kappa_across_component_std_mean"] < 1e-3
        ):
            logger.warning(
                "Shared1000 compact mixture appears collapsed: near-identical component directions, uniform weights, and identical kappas"
            )

    logger.info(
        "Shared1000: R@1=%.4f  R@5=%.4f  R@10=%.4f  CSLS_R@1=%.4f  "
        "MedR=%.1f  MRR=%.4f  pos_sim=%.4f  (N=%d images, %d trials)",
        metrics["r@1"], metrics["r@5"], metrics["r@10"], metrics["csls_r@1"],
        metrics["median_rank"], metrics["mrr"], mean_pos_sim, n_images, n_raw,
    )

    if _uses_compact_cls_space:
        metrics["_space"] = "compact"
        metrics["_nsd_ids"] = unique_ids.astype(np.int32)
        if _rich_gts_s1000 is not None:
            metrics["_rich_preds"] = _rich_preds_img
            metrics["_rich_gts"] = _rich_gts_s1000
        if _rerank_preds_img is not None and _rerank_gts_s1000 is not None:
            metrics["_rerank_preds"] = _rerank_preds_img
            metrics["_rerank_gts"] = _rerank_gts_s1000
        if _legacy_preds_img is not None and _legacy_gts_s1000 is not None:
            metrics["_legacy_preds"] = _legacy_preds_img
            metrics["_legacy_gts"] = _legacy_gts_s1000
        if _vmf_preds_img is not None:
            metrics["_vmf_preds"] = _vmf_preds_img
        if _vmf_kappas_img is not None:
            metrics["_vmf_kappas"] = _vmf_kappas_img
        if '_component_mu_img' in locals() and _component_mu_img is not None and _component_kappa_img is not None:
            metrics["_compact_component_mu"] = _component_mu_img
            metrics["_compact_component_kappa"] = _component_kappa_img
            if _component_logits_img is not None:
                metrics["_compact_component_logits"] = _component_logits_img

    return metrics, preds, gts


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    losses: Dict[str, nn.Module],
    loss_weights: Dict[str, float],
    device: str,
    preprocessor: Optional[EmbeddingPreprocessor],
    queue: Optional[nn.Module],
    vmf_is_log: bool = True,
    current_epoch: int = 0,
    config_ref: Optional[Dict[str, Any]] = None,
    legacy_teacher_model: Optional[nn.Module] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Validate model and collect embeddings for retrieval evaluation."""
    model.eval()
    epoch_metrics: Dict[str, list] = {}
    model_type = getattr(model, "model_type", "deterministic")
    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []
    all_rich_preds: List[np.ndarray] = []
    all_rich_gts: List[np.ndarray] = []
    all_rerank_preds: List[np.ndarray] = []
    all_rerank_gts: List[np.ndarray] = []
    all_legacy_preds: List[np.ndarray] = []
    all_legacy_gts: List[np.ndarray] = []
    all_vmf_preds: List[np.ndarray] = []
    all_vmf_kappas: List[np.ndarray] = []
    all_compact_component_mu: List[np.ndarray] = []
    all_compact_component_kappa: List[np.ndarray] = []
    all_compact_component_logits: List[np.ndarray] = []
    all_nsd_ids: List[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            _rich_target = None
            _rerank_target = None
            if isinstance(batch, dict):
                fmri = batch["fmri"].to(device, dtype=torch.float32)
                gt_embedding = batch["retrieval_target"].to(device, dtype=torch.float32)
                _rich_target = batch["rich_target"].to(device, dtype=torch.float32)
                subject_ids = batch["subject_id"].to(device)
                all_nsd_ids.append(batch["nsd_id"].numpy())
                if "rerank_target" in batch:
                    _rerank_target = batch["rerank_target"].to(device, dtype=torch.float32)
                if batch_idx == 0 and model_type == "vmf_triple":
                    _dec = getattr(model, "decoder", None)
                    if _dec is not None:
                        assert gt_embedding.shape[-1] == _dec.retrieval_dim, (
                            f"retrieval_target dim ({gt_embedding.shape[-1]}) != decoder.retrieval_dim ({_dec.retrieval_dim})"
                        )
                        assert _rich_target.shape[-1] == _dec.token_dim, (
                            f"rich_target dim ({_rich_target.shape[-1]}) != decoder.token_dim ({_dec.token_dim})"
                        )
                        if _rerank_target is not None and getattr(_dec, "has_rerank", False):
                            assert _rerank_target.shape[-1] == _dec.rerank_dim, (
                                f"rerank_target dim ({_rerank_target.shape[-1]}) != decoder.rerank_dim ({_dec.rerank_dim})"
                            )
            elif len(batch) == 4:
                fmri, gt_embedding, subject_ids, _ = batch
                subject_ids = subject_ids.to(device)
                fmri = fmri.to(device, dtype=torch.float32)
                gt_embedding = gt_embedding.to(device, dtype=torch.float32)
            elif len(batch) == 3:
                fmri, gt_embedding, subject_ids = batch
                subject_ids = subject_ids.to(device)
                fmri = fmri.to(device, dtype=torch.float32)
                gt_embedding = gt_embedding.to(device, dtype=torch.float32)
            else:
                fmri, gt_embedding = batch
                subject_ids = None
                fmri = fmri.to(device, dtype=torch.float32)
                gt_embedding = gt_embedding.to(device, dtype=torch.float32)

            fmri_teacher = fmri

            if preprocessor is not None:
                gt_embedding_np = gt_embedding.cpu().numpy()
                gt_embedding_proc = preprocessor.transform(gt_embedding_np)
                gt_embedding = torch.from_numpy(gt_embedding_proc).float().to(device)

            output = model(fmri, subject_ids=subject_ids) if subject_ids is not None else model(fmri)
            if isinstance(output, tuple):
                pred, aux = output
            else:
                pred, aux = output, None
            _vmf_pred_head, _vmf_aux_head = _extract_vmf_outputs_for_losses(model_type, pred, aux)

            all_preds.append(pred.detach().cpu().numpy())
            all_gts.append(gt_embedding.detach().cpu().numpy())
            if _vmf_pred_head is not None and _vmf_aux_head is not None:
                all_vmf_preds.append(_vmf_pred_head.detach().cpu().numpy())
                _vk = _vmf_aux_head
                if vmf_is_log:
                    _vk = _vk.exp()
                all_vmf_kappas.append(_vk.squeeze(-1).detach().cpu().numpy())
            _component_mu_val = getattr(model, "_last_compact_component_mu", None)
            _component_kappa_val = getattr(model, "_last_compact_component_kappa", None)
            _component_logits_val = getattr(model, "_last_compact_component_logits", None)
            if _component_mu_val is not None and _component_kappa_val is not None:
                all_compact_component_mu.append(_component_mu_val.detach().cpu().numpy())
                all_compact_component_kappa.append(_component_kappa_val.detach().cpu().numpy())
                if _component_logits_val is not None:
                    all_compact_component_logits.append(_component_logits_val.detach().cpu().numpy())

            # Collect rich-space predictions when triple-head active
            _reg_pred_for_rich = getattr(model, "_last_rich_pred", None)
            if _reg_pred_for_rich is not None:
                all_rich_preds.append(_reg_pred_for_rich.detach().cpu().numpy())
            if _rich_target is not None:
                all_rich_gts.append(_rich_target.detach().cpu().numpy())

            # Collect rerank-head predictions (V30d+)
            _rerank_pred_val = getattr(model, "_last_rerank_pred", None)
            if _rerank_pred_val is not None:
                all_rerank_preds.append(_rerank_pred_val.detach().cpu().numpy())
            if _rerank_target is not None:
                all_rerank_gts.append(_rerank_target.detach().cpu().numpy())
            _legacy_teacher_mask, _legacy_teacher_voxels = _get_legacy_teacher_mask_and_voxels(model, subject_ids)
            _legacy_teacher_pred = None
            if legacy_teacher_model is not None and _rich_target is not None:
                if _legacy_teacher_mask is not None:
                    if bool(_legacy_teacher_mask.any().item()):
                        _teacher_fmri = fmri_teacher[_legacy_teacher_mask]
                        if _legacy_teacher_voxels > 0:
                            _teacher_fmri = _teacher_fmri[:, :_legacy_teacher_voxels]
                        _legacy_out = legacy_teacher_model(_teacher_fmri)
                        _legacy_teacher_pred = _legacy_out[0] if isinstance(_legacy_out, tuple) else _legacy_out
                else:
                    _legacy_out = (
                        legacy_teacher_model(fmri_teacher, subject_ids=subject_ids)
                        if subject_ids is not None
                        else legacy_teacher_model(fmri_teacher)
                    )
                    _legacy_teacher_pred = _legacy_out[0] if isinstance(_legacy_out, tuple) else _legacy_out
                if _legacy_teacher_pred is not None:
                    all_legacy_preds.append(_legacy_teacher_pred.detach().cpu().numpy())
                    all_legacy_gts.append(_maybe_mask_legacy_tensor(_rich_target, _legacy_teacher_mask).detach().cpu().numpy())

            total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            bm: Dict[str, float] = {}
            is_gaussian = model_type == "gaussian" and aux is not None
            is_vmf = _vmf_pred_head is not None and _vmf_aux_head is not None
            _pred_for_legacy = _maybe_mask_legacy_tensor(pred, _legacy_teacher_mask)
            _gt_for_legacy = _maybe_mask_legacy_tensor(gt_embedding, _legacy_teacher_mask)
            _rich_target_for_legacy = _maybe_mask_legacy_tensor(_rich_target, _legacy_teacher_mask)
            _component_mu_val_for_legacy = _maybe_mask_legacy_tensor(_component_mu_val, _legacy_teacher_mask)
            _component_kappa_val_for_legacy = _maybe_mask_legacy_tensor(_component_kappa_val, _legacy_teacher_mask)
            _component_logits_val_for_legacy = _maybe_mask_legacy_tensor(_component_logits_val, _legacy_teacher_mask)

            if "mse" in losses and not is_gaussian:
                l = losses["mse"](pred, gt_embedding)
                total_loss = total_loss + loss_weights.get("mse", 1.0) * l
                bm["mse"] = l.item()

            if "infonce" in losses and not is_gaussian and not is_vmf:
                l = losses["infonce"](pred, gt_embedding, queue=None)
                total_loss = total_loss + loss_weights.get("infonce", 1.0) * l
                bm["infonce"] = l.item()

            if "softclip" in losses:
                if isinstance(losses["softclip"], VMFSoftCLIPLoss) and is_vmf:
                    l = losses["softclip"](_vmf_pred_head, _vmf_aux_head, gt_embedding, queue=None)
                else:
                    l = losses["softclip"](pred, gt_embedding, queue=None)
                total_loss = total_loss + loss_weights.get("softclip", 1.0) * l
                bm["softclip"] = l.item()

            if "gaussian_nll" in losses and is_gaussian:
                l = losses["gaussian_nll"](pred, aux, gt_embedding)
                total_loss = total_loss + loss_weights.get("gaussian_nll", 1.0) * l
                bm["nll"] = l.item()

            if "gaussian_nce" in losses and is_gaussian:
                l = losses["gaussian_nce"](pred, aux, gt_embedding, queue=None)
                total_loss = total_loss + loss_weights.get("gaussian_nce", 1.0) * l
                bm["gnce"] = l.item()

            if "vmf_nll" in losses and is_vmf:
                l = losses["vmf_nll"](_vmf_pred_head, _vmf_aux_head, gt_embedding)
                total_loss = total_loss + loss_weights.get("vmf_nll", 1.0) * l
                bm["vmf_nll"] = l.item()

            if "vmf_nce" in losses and is_vmf:
                l = losses["vmf_nce"](_vmf_pred_head, _vmf_aux_head, gt_embedding, queue=None)
                total_loss = total_loss + loss_weights.get("vmf_nce", 1.0) * l
                bm["vmf_nce"] = l.item()

            _mix_cfg = (config_ref or {}).get("loss", {}).get("vmf_nce_mixture", {})
            _mix_start_epoch = int(_mix_cfg.get("start_epoch", 0))
            if (
                "vmf_nce_mixture" in losses
                and is_vmf
                and _component_mu_val is not None
                and _component_kappa_val is not None
            ):
                if current_epoch > _mix_start_epoch:
                    l = losses["vmf_nce_mixture"](
                        _component_mu_val,
                        _component_kappa_val,
                        gt_embedding,
                        component_logits=_component_logits_val,
                        queue=None,
                    )
                    if torch.isfinite(l):
                        total_loss = total_loss + loss_weights.get("vmf_nce_mixture", 1.0) * l
                        bm["vmf_nce_mixture"] = l.item()
                    else:
                        logger.warning("Non-finite vmf_nce_mixture in val epoch %d -- skipping mixture loss", current_epoch)
                        bm["vmf_nce_mixture"] = 0.0
                else:
                    bm["vmf_nce_mixture"] = 0.0

            _mix_div_cfg = (config_ref or {}).get("loss", {}).get("mixture_diversity", {})
            _mix_div_start_epoch = int(_mix_div_cfg.get("start_epoch", 0))
            if (
                "mixture_diversity" in losses
                and is_vmf
                and _component_mu_val is not None
                and _component_kappa_val is not None
                and _component_mu_val.ndim == 3
                and _component_mu_val.shape[1] > 1
            ):
                if current_epoch > _mix_div_start_epoch:
                    _mix_div_l, _mix_div_stats = losses["mixture_diversity"](
                        _component_mu_val,
                        _component_kappa_val,
                        component_logits=_component_logits_val,
                    )
                    if torch.isfinite(_mix_div_l):
                        total_loss = total_loss + loss_weights.get("mixture_diversity", 1.0) * _mix_div_l
                        bm["mixture_diversity"] = _mix_div_l.item()
                        bm["mixture_pairwise_cos"] = _mix_div_stats["pairwise_cos_mean"]
                        bm["mixture_weight_entropy_norm"] = _mix_div_stats["weight_entropy_norm"]
                        bm["mixture_top_weight_mean"] = _mix_div_stats["top_weight_mean"]
                        bm["mixture_kappa_std_mean"] = _mix_div_stats["kappa_std_mean"]
                    else:
                        logger.warning("Non-finite mixture_diversity in val epoch %d -- skipping diversity loss", current_epoch)
                        bm["mixture_diversity"] = 0.0
                else:
                    bm["mixture_diversity"] = 0.0

            if "vmf_nce_spcl" in losses and is_vmf:
                spcl_kwargs = dict(queue=None)
                if isinstance(losses["vmf_nce_spcl"], DeltaSPCLVMFNCELoss):
                    dcf_ex = getattr(model, "_last_dcf_extras", {})
                    spcl_kwargs["delta"] = dcf_ex.get("delta")
                l = losses["vmf_nce_spcl"](_vmf_pred_head, _vmf_aux_head, gt_embedding, **spcl_kwargs)
                total_loss = total_loss + loss_weights.get("vmf_nce_spcl", 1.0) * l
                bm["vmf_nce_spcl"] = l.item()

            if "vmf_nce_multitask" in losses and is_vmf:
                dcf_extras = getattr(model, "_last_dcf_extras", {})
                mt_total, mt_fused, mt_aux = losses["vmf_nce_multitask"](
                    _vmf_pred_head, _vmf_aux_head, gt_embedding,
                    per_roi_mus=dcf_extras.get("per_roi_mus"),
                    per_roi_kappas=dcf_extras.get("per_roi_kappas"),
                    queue=None,
                )
                total_loss = total_loss + loss_weights.get("vmf_nce_multitask", 1.0) * mt_total
                bm["mt_fused"] = mt_fused.item()
                bm["mt_aux"] = mt_aux.item()

            # --- Dual-head / triple-head regression MSE ---
            _reg_pred_val = getattr(model, "_last_reg_pred", None)
            _reg_mse_cfg = (config_ref or {}).get("loss", {}).get("regression_mse", {})
            if _reg_mse_cfg.get("enabled", False) and _reg_pred_val is not None:
                _reg_mse_start_epoch = int(_reg_mse_cfg.get("start_epoch", 0))
                if current_epoch > _reg_mse_start_epoch:
                    _reg_mse_w = loss_weights.get("regression_mse", _reg_mse_cfg.get("weight", 1.0))
                    _val_reg_target = _rich_target if _rich_target is not None else gt_embedding
                    _reg_l = F.mse_loss(_reg_pred_val, _val_reg_target, reduction="mean")
                    total_loss = total_loss + _reg_mse_w * _reg_l
                    bm["reg_mse"] = _reg_l.item()
                else:
                    bm["reg_mse"] = 0.0

            # --- V30d: Rerank SoftCLIP val loss ---
            _rerank_pred_val2 = getattr(model, "_last_rerank_pred", None)
            _rerank_cfg = (config_ref or {}).get("loss", {}).get("rerank_softclip", {})
            if "rerank_softclip" in losses and _rerank_pred_val2 is not None and _rerank_target is not None:
                _rerank_start_epoch = int(_rerank_cfg.get("start_epoch", 0))
                if current_epoch > _rerank_start_epoch:
                    _rr_l = losses["rerank_softclip"](_rerank_pred_val2, _rerank_target, queue=None)
                    total_loss = total_loss + loss_weights.get("rerank_softclip", _rerank_cfg.get("weight", 1.0)) * _rr_l
                    bm["rerank_softclip"] = _rr_l.item()
                else:
                    bm["rerank_softclip"] = 0.0

            _std_cfg = (config_ref or {}).get("loss", {}).get("shortlist_teacher_distill", {})
            if (
                "shortlist_teacher_distill" in losses
                and _rerank_pred_val2 is not None
                and _rerank_target is not None
            ):
                _std_l, _std_stats = losses["shortlist_teacher_distill"](
                    compact_pred=pred,
                    retrieval_target=gt_embedding,
                    rerank_pred=_rerank_pred_val2,
                    rerank_target=_rerank_target,
                    return_stats=True,
                )
                _std_start_epoch = int(_std_cfg.get("start_epoch", 10))
                if current_epoch > _std_start_epoch:
                    total_loss = total_loss + loss_weights.get(
                        "shortlist_teacher_distill",
                        _std_cfg.get("weight", 0.15),
                    ) * _std_l
                    bm["shortlist_teacher_distill"] = _std_l.item()
                else:
                    bm["shortlist_teacher_distill"] = 0.0
                bm["shortlist_teacher_gate_frac"] = _std_stats["gate_frac"]
                bm["shortlist_teacher_pos_rank_mean"] = _std_stats["teacher_pos_rank_mean"]
                bm["shortlist_teacher_pos_rank_median"] = _std_stats["teacher_pos_rank_median"]
                bm["shortlist_student_pos_rank_mean"] = _std_stats["student_pos_rank_mean"]
                bm["shortlist_student_pos_rank_median"] = _std_stats["student_pos_rank_median"]

            _ltd_cfg = (config_ref or {}).get("loss", {}).get("legacy_teacher_distill", {})
            if (
                "legacy_teacher_distill" in losses
                and _legacy_teacher_pred is not None
                and _rich_target is not None
            ):
                _ltd_l, _ltd_stats = losses["legacy_teacher_distill"](
                    compact_pred=_pred_for_legacy,
                    retrieval_target=_gt_for_legacy,
                    teacher_pred=_legacy_teacher_pred,
                    teacher_target=_rich_target_for_legacy,
                    return_stats=True,
                )
                _ltd_start_epoch = int(_ltd_cfg.get("start_epoch", 10))
                if current_epoch > _ltd_start_epoch:
                    total_loss = total_loss + loss_weights.get(
                        "legacy_teacher_distill",
                        _ltd_cfg.get("weight", 0.15),
                    ) * _ltd_l
                    bm["legacy_teacher_distill"] = _ltd_l.item()
                else:
                    bm["legacy_teacher_distill"] = 0.0
                bm["legacy_teacher_gate_frac"] = _ltd_stats["gate_frac"]
                bm["legacy_teacher_pos_rank_mean"] = _ltd_stats["teacher_pos_rank_mean"]
                bm["legacy_teacher_pos_rank_median"] = _ltd_stats["teacher_pos_rank_median"]
                bm["legacy_student_pos_rank_mean"] = _ltd_stats["student_pos_rank_mean"]
                bm["legacy_student_pos_rank_median"] = _ltd_stats["student_pos_rank_median"]

            if (
                "legacy_compact_distill" in losses
                and _legacy_teacher_pred is not None
                and _rich_target is not None
            ):
                _lcd_cfg = (config_ref or {}).get("loss", {}).get("legacy_compact_distill", {})
                _lcd_l, _lcd_stats = losses["legacy_compact_distill"](
                    compact_pred=_pred_for_legacy,
                    retrieval_target=_gt_for_legacy,
                    teacher_pred=_legacy_teacher_pred,
                    teacher_target=_rich_target_for_legacy,
                    return_stats=True,
                )
                _lcd_start_epoch = int(_lcd_cfg.get("start_epoch", 10))
                if current_epoch > _lcd_start_epoch:
                    total_loss = total_loss + loss_weights.get(
                        "legacy_compact_distill",
                        _lcd_cfg.get("weight", 0.15),
                    ) * _lcd_l
                    bm["legacy_compact_distill"] = _lcd_l.item()
                else:
                    bm["legacy_compact_distill"] = 0.0
                bm["legacy_compact_teacher_topk_hit_frac"] = _lcd_stats["teacher_topk_hit_frac"]
                bm["legacy_compact_teacher_pos_rank_mean"] = _lcd_stats["teacher_pos_rank_mean"]
                bm["legacy_compact_teacher_pos_rank_median"] = _lcd_stats["teacher_pos_rank_median"]
                bm["legacy_compact_student_pos_rank_mean"] = _lcd_stats["student_pos_rank_mean"]
                bm["legacy_compact_student_pos_rank_median"] = _lcd_stats["student_pos_rank_median"]
                bm["legacy_compact_active_candidate_size_mean"] = _lcd_stats["active_candidate_size_mean"]

            _clcd_cfg = (config_ref or {}).get("loss", {}).get("component_legacy_compact_distill", {})
            if (
                "component_legacy_compact_distill" in losses
                and _legacy_teacher_pred is not None
                and _rich_target is not None
                and _component_mu_val is not None
                and _component_kappa_val is not None
            ):
                _clcd_l, _clcd_stats = losses["component_legacy_compact_distill"](
                    compact_component_mu=_component_mu_val_for_legacy,
                    compact_component_kappa=_component_kappa_val_for_legacy,
                    retrieval_target=_gt_for_legacy,
                    teacher_pred=_legacy_teacher_pred,
                    teacher_target=_rich_target_for_legacy,
                    component_logits=_component_logits_val_for_legacy,
                    return_stats=True,
                )
                _clcd_start_epoch = int(_clcd_cfg.get("start_epoch", 10))
                if current_epoch > _clcd_start_epoch:
                    total_loss = total_loss + loss_weights.get(
                        "component_legacy_compact_distill",
                        _clcd_cfg.get("weight", 0.10),
                    ) * _clcd_l
                    bm["component_legacy_compact_distill"] = _clcd_l.item()
                else:
                    bm["component_legacy_compact_distill"] = 0.0
                bm["component_legacy_teacher_topk_hit_frac"] = _clcd_stats["teacher_topk_hit_frac"]
                bm["component_legacy_teacher_pos_rank_mean"] = _clcd_stats["teacher_pos_rank_mean"]
                bm["component_legacy_teacher_pos_rank_median"] = _clcd_stats["teacher_pos_rank_median"]
                bm["component_legacy_student_pos_rank_mean"] = _clcd_stats["student_pos_rank_mean"]
                bm["component_legacy_student_pos_rank_median"] = _clcd_stats["student_pos_rank_median"]
                bm["component_legacy_active_candidate_size_mean"] = _clcd_stats["active_candidate_size_mean"]
                bm["component_legacy_responsible_component_entropy"] = _clcd_stats["responsible_component_entropy"]
                bm["component_legacy_responsible_component_top_rate"] = _clcd_stats["responsible_component_top_rate"]
                bm["component_legacy_responsible_component_mean"] = _clcd_stats["responsible_component_mean"]

            _tri_cfg = (config_ref or {}).get("loss", {}).get("tri_teacher_distill", {})
            if (
                "tri_teacher_distill" in losses
                and _rerank_pred_val2 is not None
                and _rerank_target is not None
                and _legacy_teacher_pred is not None
                and _rich_target is not None
            ):
                _tri_l, _tri_stats = losses["tri_teacher_distill"](
                    compact_pred=_pred_for_legacy,
                    retrieval_target=_gt_for_legacy,
                    rerank_pred=_maybe_mask_legacy_tensor(_rerank_pred_val2, _legacy_teacher_mask),
                    rerank_target=_maybe_mask_legacy_tensor(_rerank_target, _legacy_teacher_mask),
                    legacy_pred=_legacy_teacher_pred,
                    legacy_target=_rich_target_for_legacy,
                    return_stats=True,
                )
                _tri_start_epoch = int(_tri_cfg.get("start_epoch", 10))
                if current_epoch > _tri_start_epoch:
                    total_loss = total_loss + loss_weights.get(
                        "tri_teacher_distill",
                        _tri_cfg.get("weight", 0.20),
                    ) * _tri_l
                    bm["tri_teacher_distill"] = _tri_l.item()
                else:
                    bm["tri_teacher_distill"] = 0.0
                bm["tri_teacher_gate_frac"] = _tri_stats["gate_frac"]
                bm["tri_teacher_candidate_size_mean"] = _tri_stats["candidate_size_mean"]
                bm["tri_rerank_teacher_pos_rank_mean"] = _tri_stats["rerank_teacher_pos_rank_mean"]
                bm["tri_rerank_teacher_pos_rank_median"] = _tri_stats["rerank_teacher_pos_rank_median"]
                bm["tri_legacy_teacher_pos_rank_mean"] = _tri_stats["legacy_teacher_pos_rank_mean"]
                bm["tri_legacy_teacher_pos_rank_median"] = _tri_stats["legacy_teacher_pos_rank_median"]
                bm["tri_combined_teacher_pos_rank_mean"] = _tri_stats["combined_teacher_pos_rank_mean"]
                bm["tri_combined_teacher_pos_rank_median"] = _tri_stats["combined_teacher_pos_rank_median"]
                bm["tri_student_pos_rank_mean"] = _tri_stats["student_pos_rank_mean"]
                bm["tri_student_pos_rank_median"] = _tri_stats["student_pos_rank_median"]

            # --- V11 val losses ---
            if "direct_alignment" in losses and is_vmf:
                da_l = losses["direct_alignment"](_vmf_pred_head, gt_embedding)
                total_loss = total_loss + loss_weights.get("direct_alignment", 0.5) * da_l
                bm["direct_align"] = da_l.item()
            if "uniformity" in losses:
                uni_l = losses["uniformity"](pred)
                total_loss = total_loss + loss_weights.get("uniformity", 0.1) * uni_l
                bm["uniformity"] = uni_l.item()

            if is_gaussian:
                bm["kl"] = compute_kl_divergence(pred, aux).item()
            if is_vmf:
                log_kappa_for_kl = _vmf_aux_head if vmf_is_log else torch.log(_vmf_aux_head.clamp(min=1e-8))
                bm["kl"] = compute_vmf_kl(_vmf_pred_head, log_kappa_for_kl, _vmf_pred_head.size(-1)).item()

            bm["loss"] = total_loss.item()
            for k, v in bm.items():
                epoch_metrics.setdefault(k, []).append(v)

    loss_metrics = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}
    val_preds = np.concatenate(all_preds)
    val_gts = np.concatenate(all_gts)

    _extras: Dict[str, np.ndarray] = {}
    if all_rich_preds:
        _extras["rich_preds"] = np.concatenate(all_rich_preds)
    if all_rich_gts:
        _extras["rich_gts"] = np.concatenate(all_rich_gts)
    if all_rerank_preds:
        _extras["rerank_preds"] = np.concatenate(all_rerank_preds)
    if all_rerank_gts:
        _extras["rerank_gts"] = np.concatenate(all_rerank_gts)
    if all_legacy_preds:
        _extras["legacy_preds"] = np.concatenate(all_legacy_preds)
    if all_legacy_gts:
        _extras["legacy_gts"] = np.concatenate(all_legacy_gts)
    if all_vmf_preds:
        _extras["vmf_preds"] = np.concatenate(all_vmf_preds)
    if all_vmf_kappas:
        _extras["vmf_kappas"] = np.concatenate(all_vmf_kappas)
    if all_compact_component_mu:
        _extras["compact_component_mu"] = np.concatenate(all_compact_component_mu)
        _extras["compact_component_kappa"] = np.concatenate(all_compact_component_kappa)
        if all_compact_component_logits:
            _extras["compact_component_logits"] = np.concatenate(all_compact_component_logits)
    if all_nsd_ids:
        _extras["nsd_ids"] = np.concatenate(all_nsd_ids)
    loss_metrics["_val_extras"] = _extras

    return loss_metrics, val_preds, val_gts


def _get_eval_fusion_cfg(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve optional fixed fusion evaluation config."""
    from fmri2img.eval.two_stage_retrieval import DEFAULT_FUSION_CONFIG

    fusion_cfg = config.get("evaluation", {}).get("fusion", {})
    if not fusion_cfg.get("enabled", False):
        return None
    resolved = dict(DEFAULT_FUSION_CONFIG)
    resolved.update(fusion_cfg)
    return resolved


def _get_eval_tri_fusion_cfg(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve optional fixed tri-fusion evaluation config."""
    tri_cfg = config.get("evaluation", {}).get("tri_fusion", {})
    if not tri_cfg.get("enabled", False):
        return None
    resolved = {
        "compact_score": "csls",
        "legacy_score": "csls",
        "family": "normalized_weighted",
        "normalization": "zscore",
        "shortlist_k": 150,
        "alpha": 0.3,
        "beta": 0.0,
        "gamma": 0.7,
        "csls_k": 10,
        "rerank_mode": "cosine",
    }
    resolved.update(tri_cfg)
    return resolved


def _compute_fusion_report(
    compact_preds: np.ndarray,
    compact_gts: np.ndarray,
    rerank_preds: np.ndarray,
    rerank_gts: np.ndarray,
    fusion_cfg: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Compute fused shortlist metrics when the config enables them."""
    if fusion_cfg is None:
        return None
    from fmri2img.eval.two_stage_retrieval import fusion_metrics

    return fusion_metrics(
        compact_preds,
        compact_gts,
        rerank_preds,
        rerank_gts,
        shortlist_k=int(fusion_cfg.get("shortlist_k", 50)),
        ks=(1, 5, 10),
        compact_score=str(fusion_cfg.get("compact_score", "csls")),
        family=str(fusion_cfg.get("family", "normalized_weighted")),
        normalization=str(fusion_cfg.get("normalization", "zscore")),
        alpha=float(fusion_cfg.get("alpha", 0.8)),
        csls_k=int(fusion_cfg.get("csls_k", 10)),
        rerank_mode="cosine",
    )


def _compute_tri_fusion_report(
    compact_preds: np.ndarray,
    compact_gts: np.ndarray,
    rerank_preds: np.ndarray,
    rerank_gts: np.ndarray,
    legacy_preds: np.ndarray,
    legacy_gts: np.ndarray,
    tri_fusion_cfg: Optional[Dict[str, Any]],
    compact_component_mu: Optional[np.ndarray] = None,
    compact_component_kappa: Optional[np.ndarray] = None,
    compact_component_logits: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Compute fixed tri-expert fusion metrics inside a compact shortlist."""
    if tri_fusion_cfg is None:
        return None

    from fmri2img.eval.two_stage_retrieval import (
        _cosine_sim,
        _csls_scores,
        _gt_rank_from_scores,
        _metrics_from_gt_rank,
        _normalize_shortlist_scores,
        _prefix_metrics,
        _rank_within_shortlist,
    )

    compact_score = str(tri_fusion_cfg.get("compact_score", "csls"))
    legacy_score = str(tri_fusion_cfg.get("legacy_score", "csls"))
    family = str(tri_fusion_cfg.get("family", "normalized_weighted"))
    normalization = str(tri_fusion_cfg.get("normalization", "zscore"))
    shortlist_k = int(tri_fusion_cfg.get("shortlist_k", 150))
    alpha = float(tri_fusion_cfg.get("alpha", 0.3))
    beta = float(tri_fusion_cfg.get("beta", 0.0))
    gamma = float(tri_fusion_cfg.get("gamma", 0.7))
    csls_k = int(tri_fusion_cfg.get("csls_k", 10))
    rerank_mode = str(tri_fusion_cfg.get("rerank_mode", "cosine"))

    n = compact_preds.shape[0]
    gt_indices = np.arange(n)

    compact_raw_scores = _cosine_sim(compact_preds, compact_gts)
    compact_csls_scores = _csls_scores(compact_preds, compact_gts, k=csls_k)
    mixture_raw_scores = None
    mixture_csls_scores = None
    if compact_component_mu is not None and compact_component_kappa is not None:
        mixture_raw_scores = _score_mixture_vmf_gallery(
            compact_component_mu,
            compact_component_kappa,
            compact_gts,
            component_logits=compact_component_logits,
            normalize=True,
        )
        mixture_csls_scores = _csls_from_score_matrix(mixture_raw_scores, k=csls_k)

    if compact_score == "csls":
        compact_scores = compact_csls_scores
    elif compact_score == "raw_cosine":
        compact_scores = compact_raw_scores
    elif compact_score == "mixture_raw":
        if mixture_raw_scores is None:
            raise ValueError("tri_fusion compact_score='mixture_raw' requires compact component arrays")
        compact_scores = mixture_raw_scores
    elif compact_score == "mixture_csls":
        if mixture_csls_scores is None:
            raise ValueError("tri_fusion compact_score='mixture_csls' requires compact component arrays")
        compact_scores = mixture_csls_scores
    else:
        raise ValueError(f"Unknown tri compact_score variant: {compact_score}")
    compact_order, compact_gt_rank = _gt_rank_from_scores(compact_scores)

    if rerank_mode == "cosine":
        rerank_scores = _cosine_sim(rerank_preds, rerank_gts)
    elif rerank_mode == "dot":
        rerank_scores = rerank_preds @ rerank_gts.T
    else:
        raise ValueError(f"Unknown rerank_mode: {rerank_mode}")

    legacy_raw_scores = _cosine_sim(legacy_preds, legacy_gts)
    legacy_csls_scores = _csls_scores(legacy_preds, legacy_gts, k=csls_k)
    legacy_scores = legacy_csls_scores if legacy_score == "csls" else legacy_raw_scores

    shortlist_k_eff = min(shortlist_k, compact_scores.shape[1])
    shortlist = compact_order[:, :shortlist_k_eff]
    row_idx = np.arange(n)[:, None]
    compact_sl = compact_scores[row_idx, shortlist]
    rerank_sl = rerank_scores[row_idx, shortlist]
    legacy_sl = legacy_scores[row_idx, shortlist]

    shortlist_recall = {
        f"r@{shortlist_k_eff}": float(np.mean(np.any(shortlist == gt_indices[:, None], axis=1)))
    }

    if family == "weighted":
        fused_scores = alpha * compact_sl + beta * rerank_sl + gamma * legacy_sl
    elif family == "normalized_weighted":
        fused_scores = (
            alpha * _normalize_shortlist_scores(compact_sl, normalization)
            + beta * _normalize_shortlist_scores(rerank_sl, normalization)
            + gamma * _normalize_shortlist_scores(legacy_sl, normalization)
        )
    elif family == "rrf":
        fused_scores = (
            alpha / (60.0 + _rank_within_shortlist(compact_sl))
            + beta / (60.0 + _rank_within_shortlist(rerank_sl))
            + gamma / (60.0 + _rank_within_shortlist(legacy_sl))
        )
    elif family == "rank_average":
        fused_scores = -(
            alpha * _rank_within_shortlist(compact_sl)
            + beta * _rank_within_shortlist(rerank_sl)
            + gamma * _rank_within_shortlist(legacy_sl)
        )
    else:
        raise ValueError(f"Unknown tri fusion family: {family}")

    fused_order_local = np.argsort(-fused_scores, axis=1)
    fused_shortlist = shortlist[row_idx, fused_order_local]
    fused_gt_rank = compact_gt_rank.copy()
    hit_rows = np.where(np.any(shortlist == gt_indices[:, None], axis=1))[0]
    if hit_rows.size > 0:
        local_gt_rank = np.argmax(
            fused_shortlist[hit_rows] == hit_rows[:, None],
            axis=1,
        ) + 1
        fused_gt_rank[hit_rows] = local_gt_rank.astype(np.int32)

    compact_raw_metrics = _metrics_from_gt_rank(_gt_rank_from_scores(compact_raw_scores)[1], (1, 5, 10))
    compact_csls_metrics = _metrics_from_gt_rank(_gt_rank_from_scores(compact_csls_scores)[1], (1, 5, 10))
    rerank_only_metrics = _metrics_from_gt_rank(_gt_rank_from_scores(rerank_scores)[1], (1, 5, 10))
    legacy_raw_metrics = _metrics_from_gt_rank(_gt_rank_from_scores(legacy_raw_scores)[1], (1, 5, 10))
    legacy_csls_metrics = _metrics_from_gt_rank(_gt_rank_from_scores(legacy_csls_scores)[1], (1, 5, 10))
    fused_metrics = _metrics_from_gt_rank(fused_gt_rank, (1, 5, 10))

    report: Dict[str, Any] = {
        "config": {
            "compact_score": compact_score,
            "legacy_score": legacy_score,
            "family": family,
            "normalization": normalization,
            "shortlist_k": int(shortlist_k_eff),
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "csls_k": csls_k,
            "rerank_mode": rerank_mode,
        },
        "shortlist_recall": shortlist_recall,
        "compact_raw": _prefix_metrics(compact_raw_metrics, "compact"),
        "compact_csls": _prefix_metrics(compact_csls_metrics, "compact_csls"),
        "rerank_only": _prefix_metrics(rerank_only_metrics, "rerank"),
        "legacy_raw": _prefix_metrics(legacy_raw_metrics, "legacy"),
        "legacy_csls": _prefix_metrics(legacy_csls_metrics, "legacy_csls"),
        "fused": _prefix_metrics(fused_metrics, "fused"),
        "fused_gain_over_compact_csls": round(
            fused_metrics.get("r@1", 0.0) - compact_csls_metrics.get("r@1", 0.0), 4
        ),
        "fused_gain_over_legacy_csls": round(
            fused_metrics.get("r@1", 0.0) - legacy_csls_metrics.get("r@1", 0.0), 4
        ),
        "fused_gain_over_rerank_only": round(
            fused_metrics.get("r@1", 0.0) - rerank_only_metrics.get("r@1", 0.0), 4
        ),
    }
    if mixture_raw_scores is not None:
        mixture_raw_metrics = _metrics_from_gt_rank(_gt_rank_from_scores(mixture_raw_scores)[1], (1, 5, 10))
        mixture_csls_metrics = _metrics_from_gt_rank(_gt_rank_from_scores(mixture_csls_scores)[1], (1, 5, 10))
        report["mixture_raw"] = _prefix_metrics(mixture_raw_metrics, "mixture")
        report["mixture_csls"] = _prefix_metrics(mixture_csls_metrics, "mixture_csls")
        report["fused_gain_over_mixture_csls"] = round(
            fused_metrics.get("r@1", 0.0) - mixture_csls_metrics.get("r@1", 0.0), 4
        )

    logger.info(
        "Tri-fusion eval: compact=%s legacy=%s family=%s norm=%s k=%d "
        "alpha/beta/gamma=%.2f/%.2f/%.2f fused_R@1=%.1f%% compact_csls_R@1=%.1f%% "
        "legacy_csls_R@1=%.1f%%",
        compact_score,
        legacy_score,
        family,
        normalization,
        shortlist_k_eff,
        alpha,
        beta,
        gamma,
        report["fused"].get("fused_r@1", 0.0) * 100,
        report["compact_csls"].get("compact_csls_r@1", 0.0) * 100,
        report["legacy_csls"].get("legacy_csls_r@1", 0.0) * 100,
    )
    return report


def _fusion_scalar_metrics(
    report: Optional[Dict[str, Any]],
    prefix: str = "fused",
    include_legacy_alias: bool = False,
) -> Dict[str, float]:
    """Flatten fused retrieval metrics for CSV logging and checkpointing."""
    if report is None:
        return {}
    fused = report.get("fused", {})
    out = {
        f"{prefix}_r@1": float(fused.get("fused_r@1", 0.0)),
        f"{prefix}_r@5": float(fused.get("fused_r@5", 0.0)),
        f"{prefix}_r@10": float(fused.get("fused_r@10", 0.0)),
        f"{prefix}_median_rank": float(fused.get("fused_median_rank", 0.0)),
        f"{prefix}_mrr": float(fused.get("fused_mrr", 0.0)),
    }
    if include_legacy_alias and prefix != "fused":
        out.update({
            "fused_r@1": out[f"{prefix}_r@1"],
            "fused_r@5": out[f"{prefix}_r@5"],
            "fused_r@10": out[f"{prefix}_r@10"],
            "fused_median_rank": out[f"{prefix}_median_rank"],
            "fused_mrr": out[f"{prefix}_mrr"],
        })
    return out


def _run_post_training_shared1000_eval(
    output_dir: Path,
    model: nn.Module,
    config: Dict[str, Any],
    subject: str,
    device: str,
    embeddings_df: pd.DataFrame,
    preprocessor: Optional[EmbeddingPreprocessor],
    vmf_is_log: bool,
    zscore_stats_path: Optional[str],
    is_multi_subject: bool,
    token_cache=None,
    rerank_cache=None,
    retrieval_projector=None,
    legacy_teacher_model: Optional[nn.Module] = None,
) -> bool:
    """Run shared1000 evaluation once from the currently loaded model checkpoint."""
    _metrics_save_dir = output_dir / "metrics"
    _metrics_save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Running shared1000 benchmark evaluation...")
    _s1000_zscore_mode = config["data"].get("zscore_mode", "global")
    _s1000_result = _evaluate_shared1000(
        model=model,
        subject=subject,
        device=device,
        embeddings_df=embeddings_df,
        preprocessor=preprocessor,
        vmf_is_log=vmf_is_log,
        zscore_stats_path=zscore_stats_path,
        zscore_mode=_s1000_zscore_mode,
        batch_size=config["training"]["batch_size"],
        is_multi_subject=is_multi_subject,
        token_cache=token_cache,
        rerank_cache=rerank_cache,
        retrieval_projector=retrieval_projector,
        legacy_teacher_model=legacy_teacher_model,
    )
    if _s1000_result is None:
        logger.info("Shared1000 evaluation skipped (data unavailable)")
        logger.info("=" * 60)
        return False

    _s1000_metrics, _s1000_preds, _s1000_gts = _s1000_result

    _s1000_rich_preds = _s1000_metrics.pop("_rich_preds", None)
    _s1000_rich_gts = _s1000_metrics.pop("_rich_gts", None)
    _s1000_rerank_preds = _s1000_metrics.pop("_rerank_preds", None)
    _s1000_rerank_gts = _s1000_metrics.pop("_rerank_gts", None)
    _s1000_legacy_preds = _s1000_metrics.pop("_legacy_preds", None)
    _s1000_legacy_gts = _s1000_metrics.pop("_legacy_gts", None)
    _s1000_vmf_preds = _s1000_metrics.pop("_vmf_preds", None)
    _s1000_vmf_kappas = _s1000_metrics.pop("_vmf_kappas", None)
    _s1000_component_mu = _s1000_metrics.pop("_compact_component_mu", None)
    _s1000_component_kappa = _s1000_metrics.pop("_compact_component_kappa", None)
    _s1000_component_logits = _s1000_metrics.pop("_compact_component_logits", None)
    _s1000_space = _s1000_metrics.pop("_space", None)
    _s1000_nsd_ids = _s1000_metrics.pop("_nsd_ids", None)

    _s1000_json_path = _metrics_save_dir / "shared1000_metrics.json"
    with open(_s1000_json_path, "w") as _jf:
        json.dump(_s1000_metrics, _jf, indent=2)
    np.save(_metrics_save_dir / "shared1000_predictions.npy", _s1000_preds)
    np.save(_metrics_save_dir / "shared1000_ground_truth.npy", _s1000_gts)
    logger.info("Saved shared1000 metrics to %s", _s1000_json_path)
    logger.info(
        "Shared1000 benchmark:  R@1=%.1f%%  CSLS_R@1=%.1f%%  (gallery=%d)",
        _s1000_metrics["r@1"] * 100,
        _s1000_metrics["csls_r@1"] * 100,
        _s1000_metrics["gallery_size"],
    )

    if _s1000_space == "compact":
        _s1000_compact_path = _metrics_save_dir / "shared1000_metrics_compact.json"
        with open(_s1000_compact_path, "w") as _jf:
            json.dump(_s1000_metrics, _jf, indent=2)
        np.save(_metrics_save_dir / "shared1000_predictions_compact.npy", _s1000_preds)
        np.save(_metrics_save_dir / "shared1000_ground_truth_compact.npy", _s1000_gts)
        _save_aux_vmf_arrays(_metrics_save_dir, "shared1000", _s1000_vmf_preds, _s1000_vmf_kappas)
        if _s1000_vmf_kappas is not None:
            np.save(_metrics_save_dir / "shared1000_kappas.npy", np.asarray(_s1000_vmf_kappas, dtype=np.float32))
        if _s1000_nsd_ids is not None:
            np.save(_metrics_save_dir / "shared1000_nsd_ids.npy", _s1000_nsd_ids)
        _save_compact_component_arrays(
            _metrics_save_dir,
            "shared1000",
            _s1000_component_mu,
            _s1000_component_kappa,
            _s1000_component_logits,
        )
        logger.info("Saved compact shared1000 metrics to %s", _s1000_compact_path)
        if _s1000_vmf_preds is not None:
            _vmf_metrics = {
                "benchmark": "shared1000_vmf_aux",
                "gallery_size": _s1000_metrics["gallery_size"],
                "vmf_r@1": float(_s1000_metrics.get("vmf_r@1", 0.0)),
                "vmf_r@5": float(_s1000_metrics.get("vmf_r@5", 0.0)),
                "vmf_r@10": float(_s1000_metrics.get("vmf_r@10", 0.0)),
                "vmf_csls_r@1": float(_s1000_metrics.get("vmf_csls_r@1", 0.0)),
                "vmf_csls_r@5": float(_s1000_metrics.get("vmf_csls_r@5", 0.0)),
                "vmf_csls_r@10": float(_s1000_metrics.get("vmf_csls_r@10", 0.0)),
                "vmf_kappa_mean_eval": float(_s1000_metrics.get("vmf_kappa_mean_eval", 0.0)),
                "vmf_kappa_std_eval": float(_s1000_metrics.get("vmf_kappa_std_eval", 0.0)),
            }
            with open(_metrics_save_dir / "shared1000_metrics_vmf.json", "w") as _jf:
                json.dump(_vmf_metrics, _jf, indent=2)

        if _s1000_rich_preds is not None and _s1000_rich_gts is not None:
            np.save(_metrics_save_dir / "shared1000_predictions_rich.npy", _s1000_rich_preds)
            np.save(_metrics_save_dir / "shared1000_ground_truth_rich.npy", _s1000_rich_gts)
            _rich_ret = _compute_retrieval(
                _s1000_rich_preds / np.maximum(
                    np.linalg.norm(_s1000_rich_preds, axis=-1, keepdims=True), 1e-8),
                _s1000_rich_gts / np.maximum(
                    np.linalg.norm(_s1000_rich_gts, axis=-1, keepdims=True), 1e-8),
                ks=(1, 5, 10),
            )
            _rich_metrics = {
                "benchmark": "shared1000_rich",
                "gallery_size": _s1000_metrics["gallery_size"],
                "rich_r@1": float(_rich_ret["top1_accuracy"]),
                "rich_r@5": float(_rich_ret["top5_accuracy"]),
                "rich_r@10": float(_rich_ret["top10_accuracy"]),
                "rich_median_rank": float(_rich_ret["median_rank"]),
            }
            with open(_metrics_save_dir / "shared1000_metrics_rich.json", "w") as _jf:
                json.dump(_rich_metrics, _jf, indent=2)
            logger.info("Saved rich shared1000 metrics: R@1=%.1f%%",
                        _rich_metrics["rich_r@1"] * 100)

        if _s1000_rerank_preds is not None and _s1000_rerank_gts is not None:
            np.save(_metrics_save_dir / "shared1000_predictions_rerank.npy", _s1000_rerank_preds)
            np.save(_metrics_save_dir / "shared1000_ground_truth_rerank.npy", _s1000_rerank_gts)

            from fmri2img.eval.two_stage_retrieval import two_stage_metrics
            _ts_s1000 = two_stage_metrics(
                _s1000_preds, _s1000_gts,
                _s1000_rerank_preds, _s1000_rerank_gts,
                shortlist_k=100, ks=(1, 5, 10),
            )
            _ts_s1000_path = _metrics_save_dir / "shared1000_two_stage_rerank.json"
            with open(_ts_s1000_path, "w") as _jf:
                json.dump(_ts_s1000, _jf, indent=2, default=str)
            logger.info(
                "Shared1000 two-stage (rerank head): reranked_R@1=%.1f%%  "
                "compact_R@1=%.1f%%  gain=%.1f pp",
                _ts_s1000["reranked"].get("reranked_r@1", 0) * 100,
                _ts_s1000["compact_raw"].get("compact_r@1", 0) * 100,
                _ts_s1000.get("rerank_gain_over_compact_raw", 0) * 100,
            )

            _fusion_cfg = _get_eval_fusion_cfg(config)
            _tri_fusion_cfg = _get_eval_tri_fusion_cfg(config)
            _fusion_report = None
            if (
                _tri_fusion_cfg is not None
                and _s1000_legacy_preds is not None
                and _s1000_legacy_gts is not None
            ):
                _fusion_report = _compute_tri_fusion_report(
                    _s1000_preds,
                    _s1000_gts,
                    _s1000_rerank_preds,
                    _s1000_rerank_gts,
                    _s1000_legacy_preds,
                    _s1000_legacy_gts,
                    _tri_fusion_cfg,
                    compact_component_mu=_s1000_component_mu,
                    compact_component_kappa=_s1000_component_kappa,
                    compact_component_logits=_s1000_component_logits,
                )
            elif _fusion_cfg is not None:
                _fusion_report = _compute_fusion_report(
                    _s1000_preds,
                    _s1000_gts,
                    _s1000_rerank_preds,
                    _s1000_rerank_gts,
                    _fusion_cfg,
                )
            if _fusion_report is not None:
                _fusion_path = _metrics_save_dir / "shared1000_fused_metrics.json"
                with open(_fusion_path, "w") as _jf:
                    json.dump(_fusion_report, _jf, indent=2, default=str)
                _fused = _fusion_report.get("fused", {})
                logger.info(
                    "Shared1000 fused retrieval: fused_R@1=%.1f%%  R@5=%.1f%%  "
                    "R@10=%.1f%%  MedR=%.1f  MRR=%.4f",
                    _fused.get("fused_r@1", 0.0) * 100,
                    _fused.get("fused_r@5", 0.0) * 100,
                    _fused.get("fused_r@10", 0.0) * 100,
                    _fused.get("fused_median_rank", 0.0),
                    _fused.get("fused_mrr", 0.0),
                )
        if _s1000_legacy_preds is not None and _s1000_legacy_gts is not None:
            np.save(_metrics_save_dir / "shared1000_predictions_legacy.npy", _s1000_legacy_preds)
            np.save(_metrics_save_dir / "shared1000_ground_truth_legacy.npy", _s1000_legacy_gts)
            logger.info("Saved legacy shared1000 predictions %s", _s1000_legacy_preds.shape)

    logger.info("=" * 60)
    return True


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    val_loss: float,
    config: Dict[str, Any],
    global_step: int,
    subject: str = "",
    roi_mask_path: str = "",
    losses: Optional[Dict[str, nn.Module]] = None,
    meta: Optional[Dict[str, Any]] = None,
    ema: Optional["ModelEMA"] = None,
) -> None:
    loss_states = {}
    if losses:
        for k, v in losses.items():
            if isinstance(v, nn.Module):
                loss_states[k] = v.state_dict()
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "loss_states": loss_states,
        "val_loss": val_loss,
        "config": config,
        "model_config": config.get("model", {}),
        "subject": subject,
        "roi_mask_path": roi_mask_path,
        "_meta": meta or {},
    }
    if ema is not None:
        payload["ema_shadow"] = {k: v.cpu() for k, v in ema.shadow.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    for _attempt in range(3):
        _local_tmp: Optional[str] = None
        _remote_tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f"{path.stem}.",
                suffix=".pt",
                dir="/tmp",
                delete=False,
            ) as _tf:
                _local_tmp = _tf.name

            try:
                torch.save(payload, _local_tmp)
            except RuntimeError:
                # Older serialization is slower but can be more robust on flaky filesystems.
                torch.save(payload, _local_tmp, _use_new_zipfile_serialization=False)

            shutil.copyfile(_local_tmp, _remote_tmp)
            os.replace(_remote_tmp, path)
            return
        except (RuntimeError, OSError) as exc:
            for _stale in (_remote_tmp,):
                try:
                    if os.path.exists(_stale):
                        os.remove(_stale)
                except OSError:
                    pass
            if _attempt < 2:
                logger.warning(
                    "Checkpoint save to %s failed (attempt %d/3, NFS?), "
                    "retrying in 2s … [%s: %s]", path, _attempt + 1,
                    type(exc).__name__, exc,
                )
                time.sleep(2)
            else:
                raise
        finally:
            if _local_tmp is not None:
                try:
                    if os.path.exists(_local_tmp):
                        os.remove(_local_tmp)
                except OSError:
                    pass


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    lr_scheduler: Any, scaler: Optional[torch.amp.GradScaler],
                    device: str,
                    losses: Optional[Dict[str, nn.Module]] = None,
                    ema: Optional["ModelEMA"] = None) -> Tuple[int, float, int]:
    """Load checkpoint and restore state. Returns (start_epoch, best_val_loss, global_step)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if lr_scheduler and ckpt.get("lr_scheduler_state_dict"):
        lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if losses and ckpt.get("loss_states"):
        for k, state in ckpt["loss_states"].items():
            if k in losses and isinstance(losses[k], nn.Module):
                losses[k].load_state_dict(state)
                logger.info("Restored loss state: %s", k)
    if ema is not None and ckpt.get("ema_shadow"):
        for name, val in ckpt["ema_shadow"].items():
            if name in ema.shadow:
                ema.shadow[name] = val.to(device)
        logger.info("Restored EMA shadow state (%d params)", len(ckpt["ema_shadow"]))
    logger.info("Resumed from checkpoint %s (epoch %d, val_loss=%.4f)",
                path, ckpt["epoch"], ckpt["val_loss"])
    return ckpt["epoch"] + 1, ckpt["val_loss"], ckpt.get("global_step", 0)


# ---------------------------------------------------------------------------
# Model soup (Wortsman et al., NeurIPS 2022)
# ---------------------------------------------------------------------------

def model_soup(
    checkpoint_dir: Path,
    model: nn.Module,
    top_k: int = 5,
    device: str = "cpu",
) -> bool:
    """Average weights of periodic checkpoints, load into model.

    Finds all ``checkpoint_epoch_*.pt`` files in *checkpoint_dir*, ranks
    them by ``val_loss`` (lower is better), and averages the top-k model
    state dicts.  The averaged weights are loaded into *model* in-place.

    Returns True if soup was applied, False if not enough checkpoints.
    """
    import glob as _glob
    ckpt_paths = sorted(_glob.glob(str(checkpoint_dir / "checkpoint_epoch_*.pt")))
    if len(ckpt_paths) < 2:
        logger.info("Model soup: fewer than 2 periodic checkpoints found, skipping")
        return False

    ranked: List[Tuple[float, str]] = []
    for p in ckpt_paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        val_loss = ckpt.get("val_loss", float("inf"))
        ranked.append((val_loss, p))
    ranked.sort(key=lambda x: x[0])
    selected = ranked[:top_k]
    logger.info(
        "Model soup: averaging %d / %d checkpoints (val_loss range %.4f – %.4f)",
        len(selected), len(ranked), selected[0][0], selected[-1][0],
    )

    avg_state: Dict[str, torch.Tensor] = {}
    n = len(selected)
    for _, p in selected:
        state = torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"]
        for k, v in state.items():
            if k in avg_state:
                avg_state[k] = avg_state[k] + v.float() / n
            else:
                avg_state[k] = v.float() / n

    model.load_state_dict(avg_state)
    model.to(device)
    logger.info("Model soup: loaded averaged weights into model")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified training script")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--subject", type=str, default=None, help="Override subject (e.g. subj02)")
    parser.add_argument("--save-checkpoints", type=str, default="all",
                        choices=["all", "best", "none"],
                        help="all=save last+best+periodic, best=best only, none=skip all")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="(deprecated) alias for --save-checkpoints none")
    parser.add_argument("--post-eval-shared1000-only", action="store_true",
                        help="Skip training and run shared1000 evaluation from a saved checkpoint")
    parser.add_argument("--save-train-preds", action="store_true",
                        help="Skip training; load best checkpoint and save train-split predictions "
                             "(for V39 union-shortlist reranker)")
    args = parser.parse_args()
    if args.no_checkpoints:
        args.save_checkpoints = "none"

    config = load_config(Path(args.config))
    subject = resolve_subject(args, config)
    seed = config.get("data", {}).get("seed", 42)
    set_seed(seed)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s | Subject: %s | Seed: %d", device, subject, seed)

    # --- Output directory: experimental_results/{exp_name}/{subject}/ ---
    exp_name = config["experiment"]["name"]
    output_dir = Path("experimental_results") / exp_name / subject
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    # Add file handler for training log
    fh = logging.FileHandler(output_dir / "logs" / "train.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    # Save frozen config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # --- Embedding column override from config ---
    global _EMBEDDING_COLUMN_OVERRIDE
    _EMBEDDING_COLUMN_OVERRIDE = config.get("data", {}).get("embedding_column")
    if _EMBEDDING_COLUMN_OVERRIDE:
        logger.info("Embedding column override: %s", _EMBEDDING_COLUMN_OVERRIDE)

    # --- Load embeddings ---
    embeddings_path = find_embeddings_path()
    if embeddings_path is None:
        logger.warning("No embedding cache found — generating dummy embeddings for testing")
        n_samples, edim = 1000, 512
        dummy = torch.randn(n_samples, edim)
        dummy = dummy / dummy.norm(dim=1, keepdim=True)
        embeddings_df = pd.DataFrame({"nsdId": range(n_samples)})
        embeddings_df["embedding"] = [dummy[i].numpy().tolist() for i in range(n_samples)]
    else:
        logger.info("Embeddings: %s", embeddings_path)
        embeddings_df = pd.read_parquet(embeddings_path)
        logger.info("Loaded %d embedding rows", len(embeddings_df))

    # --- Preprocessing ---
    preprocessor, preproc_needs_fit = setup_preprocessing(config, subject, device)

    logger.info("=" * 80)
    logger.info("Experiment: %s", config["experiment"]["name"])
    logger.info("Description: %s", config["experiment"]["description"])
    logger.info("Subject: %s", subject)
    logger.info("=" * 80)

    # --- Index ---
    index_path = resolve_index_path(subject)
    if not index_path.exists():
        legacy = index_path.parent / "index_full.parquet"
        if legacy.exists():
            index_path = legacy
            logger.warning(
                "Primary index.parquet not found — falling back to legacy %s. "
                "If this was generated by build_full_subj01_index.py (cycling nsdIds), "
                "fMRI-CLIP alignment will be WRONG. Rebuild with: "
                "python scripts/build/build_full_index.py --subject %s",
                index_path, subject,
            )
        else:
            logger.error("Index not found: %s", index_path)
            logger.error("Run: python scripts/build/build_full_index.py --subject %s", subject)
            sys.exit(1)

    index_df = pd.read_parquet(index_path)
    logger.info("Index: %d trials, %d unique stimuli", len(index_df), index_df["nsdId"].nunique())

    # Detect buggy cycling-index pattern from build_full_subj01_index.py:
    # that script assigns nsdIds by cycling through a sorted list, producing
    # monotonically sorted nsdIds in the first N_unique rows with no repeats.
    _first_ids = index_df["nsdId"].values[:200]
    if len(_first_ids) >= 200 and np.all(np.diff(_first_ids) >= 0) and len(set(_first_ids)) == len(_first_ids):
        logger.error(
            "INDEX INTEGRITY CHECK FAILED: The first 200 nsdIds are monotonically "
            "sorted with no repeats — this is the signature of the buggy cycling "
            "index (build_full_subj01_index.py). fMRI-CLIP alignment is WRONG. "
            "Rebuild with: python scripts/build/build_full_index.py --subject %s",
            subject,
        )
        sys.exit(1)

    if "nsdId" not in embeddings_df.columns:
        embeddings_df["nsdId"] = range(len(embeddings_df))
        logger.warning("Added sequential nsdId column to embeddings")

    _retrieval_projector = build_retrieval_target_projector(config, embeddings_df)

    # --- Dataset (prefer pre-extracted features for speed) ---
    cache_root = os.environ.get("CACHE_ROOT", "cache")
    preextracted_path = Path(cache_root) / "preextracted" / f"subject={subject}" / "fmri_features.npy"
    roi_mask_path = resolve_roi_mask_path(subject)

    # --- Token mode: load HDF5 token-level CLIP cache (MindEye-style) ---
    _token_cache_path = config.get("data", {}).get("token_cache_path", "")
    _token_cache = None
    _multi_subjects = config.get("data", {}).get("subjects", [])
    if _token_cache_path:
        from fmri2img.data.token_clip_cache import TokenCLIPCache
        _token_cache = TokenCLIPCache(_token_cache_path)
        # Use mmap (lazy HDF5) for multi-subject to avoid 29+ GB RAM usage.
        # Single-subject (~7 GB) loads into RAM for speed.
        _use_mmap = len(_multi_subjects) > 1
        _token_cache.load(mmap=_use_mmap)
        logger.info(
            "TOKEN MODE: Loaded %d images from %s (%d tokens × %d dim, mmap=%s)",
            len(_token_cache), _token_cache_path,
            _token_cache.num_tokens, _token_cache.token_dim, _use_mmap,
        )

    # --- Rerank cache (V30d+): compressed token targets ---
    _rerank_cache_path = config.get("data", {}).get("rerank_cache_path", "")
    _rerank_cache = None
    if _rerank_cache_path:
        from fmri2img.data.compressed_target_cache import CompressedTargetCache
        _rerank_cache = CompressedTargetCache(_rerank_cache_path)
        logger.info(
            "RERANK CACHE: %d images, rerank_dim=%d, variance=%.4f, from %s",
            _rerank_cache.n_images, _rerank_cache.rerank_dim,
            _rerank_cache.explained_variance or -1,
            _rerank_cache_path,
        )

    _encoder_type_check = config.get("model", {}).get("encoder", {}).get("encoder_type", "mlp")
    _cross_subject_cfg = config.get("model", {}).get("cross_subject", {})
    _cross_subject_enabled = _cross_subject_cfg.get("enabled", False)

    # Load multi-subject dataset for either multi_subject_roi_transformer
    # OR cross-subject adapter mode (V25b — MLP with per-subject Linear adapters)
    _use_multi_subject_dataset = (
        (_encoder_type_check == "multi_subject_roi_transformer" and len(_multi_subjects) > 1)
        or (_cross_subject_enabled and len(_multi_subjects) > 1)
    )

    if _use_multi_subject_dataset:
        from fmri2img.data.multi_subject_dataset import MultiSubjectPreextractedDataset

        _emb_col_override = config.get("data", {}).get("embedding_column")
        _exclude_s1000 = config["data"].get("exclude_shared1000", True)
        _split_img = config["data"].get("split_by_image", True)
        _val_ratio = config["data"].get("val_split", 0.10)
        _data_seed = config["data"].get("seed", 42)

        _avg_reps_multi = config["data"].get("average_repetitions", False)
        _dual_target = config.get("data", {}).get("dual_target", False)
        full_dataset = MultiSubjectPreextractedDataset(
            subjects=_multi_subjects,
            cache_root=Path(cache_root) / "preextracted",
            index_root=Path("data/indices/nsd_index"),
            embeddings_df=embeddings_df,
            embedding_column=_emb_col_override,
            exclude_shared1000=_exclude_s1000,
            split_by_image=_split_img,
            val_ratio=_val_ratio,
            seed=_data_seed,
            average_repetitions=_avg_reps_multi,
            token_cache=_token_cache,
            rerank_cache=_rerank_cache,
            dual_target=_dual_target,
            retrieval_projector=_retrieval_projector,
        )
        logger.info("Multi-subject dataset: %d subjects, %d total trials",
                     full_dataset.n_subjects, len(full_dataset))

    elif preextracted_path.exists():
        avg_reps = config["data"].get("average_repetitions", False)
        _dual_target = config.get("data", {}).get("dual_target", False)
        logger.info("Using pre-extracted features: %s (avg_reps=%s, dual_target=%s)",
                     preextracted_path, avg_reps, _dual_target)
        full_dataset = PreextractedNSDDataset(
            preextracted_path, index_df, embeddings_df,
            average_repetitions=avg_reps,
            token_cache=_token_cache,
            rerank_cache=_rerank_cache,
            dual_target=_dual_target,
            retrieval_projector=_retrieval_projector,
        )
    else:
        logger.warning(
            "Pre-extracted features not found at %s — falling back to NIfTI loading (slow). "
            "Run: make preextract SUBJECT=%s",
            preextracted_path, subject,
        )
        if roi_mask_path.exists():
            logger.info("ROI mask: %s", roi_mask_path)
            full_dataset = NSDDataset(index_df, embeddings_df, roi_mask_path=roi_mask_path)
        else:
            logger.warning("ROI mask not found: %s — using full brain volume", roi_mask_path)
            full_dataset = NSDDataset(index_df, embeddings_df)

    sample = full_dataset[0]
    _dual_target_mode = isinstance(sample, dict)
    if _dual_target_mode:
        sample_fmri = sample["fmri"]
        fmri_dim = sample_fmri.shape[0]
        embedding_dim = sample["retrieval_target"].shape[0]
        _rich_dim = sample["rich_target"].shape[0]
        _rerank_dim_sample = sample["rerank_target"].shape[0] if "rerank_target" in sample else 0
        logger.info("Dimensions: fMRI=%d, Retrieval=%d, Rich=%d, Rerank=%d (dual_target)",
                     fmri_dim, embedding_dim, _rich_dim, _rerank_dim_sample)
    else:
        sample_fmri = sample[0]
        sample_emb = sample[1]
        fmri_dim = sample_fmri.shape[0]
        embedding_dim = sample_emb.shape[0]
        logger.info("Dimensions: fMRI=%d, Embedding=%d", fmri_dim, embedding_dim)

    # --- Model ---
    model_config = config["model"]
    model_config["encoder"]["input_dim"] = fmri_dim
    model_config["decoder"]["output_dim"] = embedding_dim

    # Auto-fill token decoder config from token cache
    if _token_cache is not None:
        model_config["decoder"]["num_tokens"] = _token_cache.num_tokens
        model_config["decoder"]["token_dim"] = _token_cache.token_dim
        logger.info(
            "Token decoder config: output_dim=%d, num_tokens=%d, token_dim=%d",
            embedding_dim, _token_cache.num_tokens, _token_cache.token_dim,
        )

    _roi_indices = None
    encoder_type = model_config.get("encoder", {}).get("encoder_type", "mlp")
    _is_multi_subject = (
        encoder_type == "multi_subject_roi_transformer"
        or _cross_subject_enabled
    )
    _roi_patch_size = model_config.get("encoder", {}).get("roi_patch_size", 0)
    if encoder_type == "roi_transformer":
        from fmri2img.data.roi_utils import build_roi_index
        roi_names = list(model_config["encoder"].get("roi_dims", {}).keys())
        if roi_names:
            actual_dims, _roi_indices = build_roi_index(subject, roi_names)
            # --- V25c sub-ROI patching ---
            if _roi_patch_size and _roi_patch_size > 0:
                from fmri2img.data.roi_utils import subdivide_rois
                actual_dims, _roi_indices = subdivide_rois(
                    actual_dims, _roi_indices,
                    max_voxels_per_token=_roi_patch_size,
                )
            model_config["encoder"]["roi_dims"] = dict(actual_dims)
            logger.info("ROI dims overridden from NSD masks (total=%d)", sum(actual_dims.values()))
    elif _is_multi_subject:
        from fmri2img.data.roi_utils import build_roi_index
        roi_names = list(model_config["encoder"].get("roi_dims", {}).keys())
        subjects_list = config.get("data", {}).get("subjects", [subject])
        if roi_names and subjects_list:
            # --- Pass 1: build raw ROI indices for all subjects ---
            raw_roi_dims = {}
            raw_roi_indices = {}
            for subj in subjects_list:
                dims, indices = build_roi_index(subj, roi_names)
                raw_roi_dims[subj] = dims
                raw_roi_indices[subj] = indices
                logger.info("ROI dims for %s: total=%d", subj, sum(dims.values()))

            # --- Pass 2: harmonized sub-ROI patching ---
            if _roi_patch_size and _roi_patch_size > 0:
                from fmri2img.data.roi_utils import harmonize_multi_subject_subdivisions
                h_dims, h_indices = harmonize_multi_subject_subdivisions(
                    raw_roi_dims, raw_roi_indices,
                    max_voxels_per_token=_roi_patch_size,
                )
                subject_roi_dims = {s: dict(d) for s, d in h_dims.items()}
                subject_roi_indices = h_indices
            else:
                subject_roi_dims = {s: dict(d) for s, d in raw_roi_dims.items()}
                subject_roi_indices = raw_roi_indices

            model_config["encoder"]["subject_roi_dims"] = subject_roi_dims
            _roi_indices = subject_roi_indices

    # --- Optional NCSNR loading for voxel attention (V11) ---
    _ncsnr_array = None
    if model_config.get("ncsnr_attention", {}).get("enabled", False):
        try:
            from fmri2img.reliability.noise_ceiling import load_ncsnr
            _ncsnr_subject = config.get("data", {}).get("subject", "subj01")
            _ncsnr_root = os.environ.get("NSD_DATA_ROOT", "data")
            _ncsnr_array = load_ncsnr(_ncsnr_subject, data_root=_ncsnr_root)
            if _ncsnr_array is not None:
                logger.info("Loaded NCSNR for %s: %d voxels, mean=%.2f",
                            _ncsnr_subject, len(_ncsnr_array), _ncsnr_array.mean())
            else:
                logger.warning("NCSNR not found — NCSnrAttention will use uniform init")
        except Exception as e:
            logger.warning("Failed to load NCSNR: %s — using uniform init", e)

    # --- V25b: Populate cross-subject adapter dims from dataset ---
    if _cross_subject_enabled and hasattr(full_dataset, "voxel_counts"):
        _cs_canonical = _cross_subject_cfg.get("canonical_subject", "subj01")
        _cs_voxel_dims = dict(full_dataset.voxel_counts)
        model_config.setdefault("cross_subject", {})
        model_config["cross_subject"]["enabled"] = True
        model_config["cross_subject"]["canonical_subject"] = _cs_canonical
        model_config["cross_subject"]["subject_voxel_dims"] = _cs_voxel_dims
        # For MLP: encoder input_dim must match canonical subject's voxels
        if encoder_type == "mlp":
            model_config["encoder"]["input_dim"] = _cs_voxel_dims[_cs_canonical]
        logger.info(
            "Cross-subject adapters: canonical=%s, voxel_dims=%s",
            _cs_canonical, _cs_voxel_dims,
        )

    model = create_model(model_config, roi_indices=_roi_indices, ncsnr=_ncsnr_array).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{n_params:,}")
    _resume_ckpt_exists = bool(args.resume) and Path(args.resume).exists()

    # --- Optional full-model initialization from a prior checkpoint ---
    _pm_path = model_config.get("pretrained_model_path")
    _require_pm = bool(model_config.get("require_pretrained_model", False))
    _require_parent_split_match = bool(model_config.get("require_parent_split_match", False))
    _pm_loaded_ok = False
    if _pm_path and os.path.isfile(_pm_path):
        _pm_ckpt = torch.load(_pm_path, map_location=device, weights_only=False)
        _pm_sd = _pm_ckpt.get("model_state_dict", _pm_ckpt.get("state_dict", {}))
        if not _pm_sd:
            if _require_pm:
                raise KeyError(
                    "require_pretrained_model=true but checkpoint does not contain "
                    f"model_state_dict/state_dict: {_pm_path}"
                )
            logger.warning(
                "pretrained_model_path has no model_state_dict/state_dict: %s", _pm_path,
            )
        else:
            _model_sd = model.state_dict()
            _extra_source_model_keys = sorted(set(_pm_sd).difference(_model_sd))
            _pm_compatible_sd, _pm_matched_keys, _pm_adapted_keys, _pm_initialized_keys, _pm_skipped_shape_keys = (
                _prepare_compatible_pretrained_state_dict(model, _pm_sd)
            )
            _missing_model_keys = sorted(set(_model_sd).difference(_pm_compatible_sd))
            _pm_missing, _pm_unexpected = model.load_state_dict(_pm_compatible_sd, strict=False)
            logger.info(
                "Pretrained full-model init from %s: source_keys=%d, model_keys=%d, "
                "matched=%d, adapted=%d, initialized=%d, skipped_shape=%d, missing_in_source=%d, extra_in_source=%d",
                _pm_path,
                len(_pm_sd),
                len(_model_sd),
                len(_pm_matched_keys),
                len(_pm_adapted_keys),
                len(_pm_initialized_keys),
                len(_pm_skipped_shape_keys),
                len(_missing_model_keys),
                len(_extra_source_model_keys),
            )
            if _pm_adapted_keys:
                logger.info(
                    "Pretrained adaptation applied to %d key(s): %s",
                    len(_pm_adapted_keys),
                    _pm_adapted_keys[:8],
                )
            if _pm_initialized_keys:
                logger.info(
                    "Pretrained init created %d new key(s): %s",
                    len(_pm_initialized_keys),
                    _pm_initialized_keys[:8],
                )
            if _pm_skipped_shape_keys:
                logger.info(
                    "Skipped %d incompatible pretrained key(s) due to shape mismatch: %s",
                    len(_pm_skipped_shape_keys),
                    _pm_skipped_shape_keys[:8],
                )
            if _pm_missing or _pm_unexpected:
                logger.info(
                    "Full-model load_state_dict(strict=False) summary: missing=%d unexpected=%d",
                    len(_pm_missing),
                    len(_pm_unexpected),
                )
            if _require_pm and (len(_pm_matched_keys) + len(_pm_adapted_keys) + len(_pm_initialized_keys) == 0):
                raise RuntimeError(
                    "require_pretrained_model=true but zero compatible model keys matched from "
                    f"{_pm_path}"
                )
            if _require_pm:
                logger.info(
                    "Verified pretrained full-model initialization is active: matched=%d adapted=%d initialized=%d from %s",
                    len(_pm_matched_keys),
                    len(_pm_adapted_keys),
                    len(_pm_initialized_keys),
                    _pm_path,
                )
            _pm_loaded_ok = True
    elif _pm_path:
        if _require_pm and _resume_ckpt_exists:
            logger.info(
                "require_pretrained_model=true but pretrained_model_path is missing; "
                "continuing because --resume=%s will restore model weights from checkpoint",
                args.resume,
            )
        elif _require_pm:
            raise FileNotFoundError(
                "require_pretrained_model=true but pretrained_model_path was not found: "
                f"{_pm_path}"
            )
        logger.warning(
            "pretrained_model_path not found: %s — model starts from scratch", _pm_path,
        )

    # --- V29b: Load pretrained encoder from a prior run (e.g. cross-subject) ---
    _pe_path = model_config.get("pretrained_encoder_path")
    _require_pe = bool(model_config.get("require_pretrained_encoder", False))
    _pe_loaded_ok = False
    _matched_encoder_key_count = 0
    if _pe_path and os.path.isfile(_pe_path):
        _pe_ckpt = torch.load(_pe_path, map_location=device, weights_only=False)
        _pe_sd = _pe_ckpt.get("model_state_dict", _pe_ckpt.get("state_dict", {}))
        _pe_keys = {k: v for k, v in _pe_sd.items() if k.startswith("encoder.")}
        _model_sd = model.state_dict()
        _model_encoder_keys = {k for k in _model_sd if k.startswith("encoder.")}
        _matched_encoder_keys = sorted(_model_encoder_keys.intersection(_pe_keys))
        _missing_encoder_keys = sorted(_model_encoder_keys.difference(_pe_keys))
        _extra_source_encoder_keys = sorted(set(_pe_keys).difference(_model_encoder_keys))
        _pe_missing, _pe_unexpected = model.load_state_dict(_pe_keys, strict=False)
        logger.info(
            "Pretrained encoder transfer from %s: source_encoder_keys=%d, "
            "model_encoder_keys=%d, matched=%d, missing_in_source=%d, extra_in_source=%d",
            _pe_path,
            len(_pe_keys),
            len(_model_encoder_keys),
            len(_matched_encoder_keys),
            len(_missing_encoder_keys),
            len(_extra_source_encoder_keys),
        )
        if _missing_encoder_keys:
            logger.info("Encoder keys missing from source (first 8): %s", _missing_encoder_keys[:8])
        if _extra_source_encoder_keys:
            logger.info("Extra source encoder keys ignored (first 8): %s", _extra_source_encoder_keys[:8])
        logger.info(
            "Decoder remains randomly initialized after encoder transfer: "
            "decoder.shared_backbone + retrieval/regression/rerank/perceptual heads"
        )
        if _pe_missing or _pe_unexpected:
            logger.info(
                "load_state_dict(strict=False) summary: missing=%d unexpected=%d",
                len(_pe_missing),
                len(_pe_unexpected),
            )
        _matched_encoder_key_count = len(_matched_encoder_keys)
        if _require_pe and _matched_encoder_key_count == 0:
            raise RuntimeError(
                "require_pretrained_encoder=true but zero encoder keys matched from "
                f"{_pe_path}"
            )
        if _require_pe:
            logger.info(
                "Verified pretrained encoder initialization is active: matched_encoder_keys=%d from %s",
                _matched_encoder_key_count,
                _pe_path,
            )
        _pe_loaded_ok = True
    elif _pe_path:
        if _require_pe and _resume_ckpt_exists:
            logger.info(
                "require_pretrained_encoder=true but pretrained_encoder_path is missing; "
                "continuing because --resume=%s will restore model weights from checkpoint",
                args.resume,
            )
        elif _require_pe:
            raise FileNotFoundError(
                "require_pretrained_encoder=true but pretrained_encoder_path was not found: "
                f"{_pe_path}"
            )
        logger.warning(
            "pretrained_encoder_path not found: %s — encoder starts random", _pe_path,
        )

    # --- V35: Load frozen legacy teacher checkpoint for compact-head distillation ---
    legacy_teacher_model = None
    _legacy_teacher_cfg = config.get("loss", {}).get("legacy_teacher_distill", {})
    _legacy_compact_cfg = config.get("loss", {}).get("legacy_compact_distill", {})
    _tri_teacher_cfg = config.get("loss", {}).get("tri_teacher_distill", {})
    _tri_mode = _tri_teacher_cfg.get("mode", "weighted_logits")
    _tri_needs_legacy = _tri_teacher_cfg.get("enabled", False) and _tri_mode != "rerank_only"
    if (
        _legacy_teacher_cfg.get("enabled", False)
        or _legacy_compact_cfg.get("enabled", False)
        or _tri_needs_legacy
    ):
        _legacy_teacher_path = _legacy_teacher_cfg.get("teacher_checkpoint_path", "")
        _legacy_compact_path = _legacy_compact_cfg.get("teacher_checkpoint_path", "")
        _tri_teacher_path = _tri_teacher_cfg.get("teacher_checkpoint_path", "")
        _teacher_paths = {
            "legacy_teacher_distill": _legacy_teacher_path,
            "legacy_compact_distill": _legacy_compact_path,
            "tri_teacher_distill": _tri_teacher_path,
        }
        _teacher_paths = {k: v for k, v in _teacher_paths.items() if v}
        if len(set(_teacher_paths.values())) > 1:
            _teacher_details = "\n".join(f"  {k}: {v}" for k, v in sorted(_teacher_paths.items()))
            raise RuntimeError(
                "legacy distillation losses specify different teacher checkpoints; refuse to continue:\n"
                f"{_teacher_details}"
            )
        _legacy_teacher_path = _tri_teacher_path or _legacy_compact_path or _legacy_teacher_path
        if not _legacy_teacher_path:
            raise RuntimeError(
                "legacy teacher distillation requires teacher_checkpoint_path, but none was provided"
            )
        if not os.path.isfile(_legacy_teacher_path):
            raise FileNotFoundError(
                "legacy teacher checkpoint was not found: "
                f"{_legacy_teacher_path}"
            )
        _legacy_ckpt = torch.load(_legacy_teacher_path, map_location=device, weights_only=False)
        _legacy_model_cfg = _legacy_ckpt.get("model_config")
        if _legacy_model_cfg is None:
            _legacy_root_cfg = _legacy_ckpt.get("config", {})
            _legacy_model_cfg = _legacy_root_cfg.get("model")
        if not isinstance(_legacy_model_cfg, dict):
            raise KeyError(
                "Legacy teacher checkpoint does not contain a usable model_config: "
                f"{_legacy_teacher_path}"
            )
        _legacy_state = _legacy_ckpt.get("model_state_dict", _legacy_ckpt.get("state_dict"))
        if _legacy_state is None:
            raise KeyError(
                "Legacy teacher checkpoint does not contain model_state_dict/state_dict: "
                f"{_legacy_teacher_path}"
            )
        legacy_teacher_model = create_model(_legacy_model_cfg).to(device)
        legacy_teacher_model.load_state_dict(_legacy_state, strict=True)
        legacy_teacher_model.eval()
        for param in legacy_teacher_model.parameters():
            param.requires_grad = False
        _legacy_n_params = sum(p.numel() for p in legacy_teacher_model.parameters())
        logger.info(
            "Loaded frozen legacy teacher from %s (epoch=%s, type=%s, params=%s)",
            _legacy_teacher_path,
            _legacy_ckpt.get("epoch", "unknown"),
            getattr(legacy_teacher_model, "model_type", "unknown"),
            f"{_legacy_n_params:,}",
        )
        logger.info("Legacy teacher params frozen and model set to eval()")

    # --- V25b: Load pretrained backbone + freeze for adapter warm-up ---
    _cs_freeze_epochs = 0
    _cs_backbone_lr_factor = 0.1
    _cs_backbone_frozen = False  # tracks whether freeze actually happened
    if _cross_subject_enabled:
        _cs_ckpt_path = _cross_subject_cfg.get("pretrained_checkpoint")
        _cs_freeze_epochs = _cross_subject_cfg.get("freeze_epochs", 30)
        _cs_backbone_lr_factor = _cross_subject_cfg.get("backbone_lr_factor", 0.1)
        if _cs_ckpt_path and os.path.isfile(_cs_ckpt_path):
            ckpt = torch.load(_cs_ckpt_path, map_location=device, weights_only=False)
            _sd = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
            # Load only encoder/decoder weights (skip subject_adapters, projection_head)
            _backbone_keys = {
                k: v for k, v in _sd.items()
                if k.startswith(("encoder.", "decoder."))
            }
            missing, unexpected = model.load_state_dict(_backbone_keys, strict=False)
            logger.info(
                "Loaded pretrained backbone from %s: %d keys loaded, "
                "%d missing (adapters), %d unexpected",
                _cs_ckpt_path, len(_backbone_keys),
                len(missing), len(unexpected),
            )
            # Freeze backbone for first _cs_freeze_epochs epochs
            for name, param in model.named_parameters():
                if name.startswith(("encoder.", "decoder.")):
                    param.requires_grad = False
            n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
            n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
            logger.info(
                "Cross-subject freeze: %d params frozen (encoder+decoder), "
                "%d trainable (adapters) for %d epochs",
                n_frozen, n_trainable, _cs_freeze_epochs,
            )
            _cs_backbone_frozen = True
        elif _pe_loaded_ok and _cs_freeze_epochs > 0:
            # Encoder was pre-loaded via pretrained_encoder_path (V29a);
            # freeze backbone so adapters can learn subject alignment first.
            for name, param in model.named_parameters():
                if name.startswith(("encoder.", "decoder.")):
                    param.requires_grad = False
            n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
            n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
            logger.info(
                "Cross-subject freeze (encoder from pretrained_encoder_path): "
                "%d params frozen (encoder+decoder), %d trainable (adapters) "
                "for %d epochs",
                n_frozen, n_trainable, _cs_freeze_epochs,
            )
            _cs_backbone_frozen = True
        elif _cs_ckpt_path:
            logger.warning(
                "Cross-subject pretrained checkpoint not found: %s "
                "— training from scratch (adapters + backbone)",
                _cs_ckpt_path,
            )

    # --- EMA ---
    _ema_cfg = config.get("training", {}).get("ema", {})
    ema = None
    if _ema_cfg.get("enabled", False):
        ema = ModelEMA(model, decay=_ema_cfg.get("decay", 0.999))
        logger.info("EMA enabled: decay=%.4f", _ema_cfg.get("decay", 0.999))

    # --- Memory queue ---
    queue = None
    if config.get("queue", {}).get("enabled", False):
        queue = create_memory_queue(config=config["queue"], embedding_dim=embedding_dim)
        if queue is not None:
            queue = queue.to(device)
            logger.info("Memory queue: size=%d", config["queue"].get("size", 8192))

    # --- Losses ---
    losses = setup_losses(config, device, queue)
    for _lk in losses:
        if isinstance(losses[_lk], nn.Module):
            losses[_lk] = losses[_lk].to(device)
    loss_weights = {
        k: v.get("weight", 1.0) for k, v in config.get("loss", {}).items()
        if isinstance(v, dict)
    }

    kl_scheduler = setup_kl_scheduler(config)

    # --- Optimizer ---
    opt_cfg = config["training"]["optimizer"]
    # When cross-subject freeze is active, only include trainable params initially
    _opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        _opt_params,
        lr=float(opt_cfg.get("lr", 1e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
        betas=opt_cfg.get("betas", [0.9, 0.999]),
    )

    loss_params = []
    for loss_mod in losses.values():
        if isinstance(loss_mod, nn.Module):
            loss_params.extend(loss_mod.parameters())
    if loss_params:
        optimizer.add_param_group({"params": loss_params, "lr": float(opt_cfg.get("lr", 1e-4))})
        logger.info("Added %d loss parameter(s) to optimizer", len(loss_params))

    # --- Data split ---
    split_by_image = config["data"].get("split_by_image", False)
    exclude_shared1000 = config["data"].get("exclude_shared1000", False)
    data_seed = config["data"].get("seed", 42)
    _split_file = config["data"].get("split_file")
    _split_nsd_ids_used: dict | None = None

    # --- Try loading a pre-computed split file (cross-phase consistency) ---
    if _split_file and os.path.isfile(_split_file) and hasattr(full_dataset, "index_df"):
        with open(_split_file) as _sf:
            _loaded_split = json.load(_sf)
        _ls_train = set(int(x) for x in _loaded_split["train_nsd_ids"])
        _ls_val = set(int(x) for x in _loaded_split["val_nsd_ids"])
        _idx_df = full_dataset.index_df
        _nsd_col = _idx_df["nsdId"].values

        train_indices = [i for i, nid in enumerate(_nsd_col) if int(nid) in _ls_train]
        val_indices = [i for i, nid in enumerate(_nsd_col) if int(nid) in _ls_val]

        logger.info(
            "Loaded split from %s: %d train nsdIds (%d trials), %d val nsdIds (%d trials)",
            _split_file, len(_ls_train), len(train_indices),
            len(_ls_val), len(val_indices),
        )
        _split_nsd_ids_used = {
            "train_nsd_ids": sorted(_ls_train),
            "val_nsd_ids": sorted(_ls_val),
        }
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
    elif _split_file:
        logger.warning("split_file specified but not found: %s — falling back to computed split", _split_file)

    if _split_nsd_ids_used is None and hasattr(full_dataset, "train_indices") and full_dataset.train_indices is not None:
        train_indices = full_dataset.train_indices.tolist()
        val_indices = full_dataset.val_indices.tolist()
        logger.info(
            "Using dataset's built-in split: %d train, %d val trials",
            len(train_indices), len(val_indices),
        )
        if hasattr(full_dataset, "index_df"):
            _nsd_col = full_dataset.index_df["nsdId"].values
            _train_nsd = set(int(_nsd_col[i]) for i in train_indices)
            _val_nsd = set(int(_nsd_col[i]) for i in val_indices)
            _split_nsd_ids_used = {
                "train_nsd_ids": sorted(_train_nsd),
                "val_nsd_ids": sorted(_val_nsd),
            }
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
    elif _split_nsd_ids_used is None and split_by_image and hasattr(full_dataset, "index_df"):
        _idx_df = full_dataset.index_df
        has_shared = "shared1000" in _idx_df.columns

        if exclude_shared1000 and has_shared:
            pool_mask = ~_idx_df["shared1000"].fillna(False).astype(bool)
            pool_indices = _idx_df.index[pool_mask].tolist()
            n_excluded = len(_idx_df) - len(pool_indices)
            logger.info("Excluded %d shared1000 trials from training pool", n_excluded)
        else:
            pool_indices = list(range(len(_idx_df)))

        pool_nsd_ids = _idx_df.iloc[pool_indices]["nsdId"].values
        unique_images = np.unique(pool_nsd_ids)  # sorted by np.unique
        rng = np.random.default_rng(data_seed)
        rng.shuffle(unique_images)

        train_ratio = config["data"]["train_split"]
        val_ratio = config["data"]["val_split"]
        n_val_images = max(1, int(len(unique_images) * val_ratio / (train_ratio + val_ratio)))
        val_image_set = set(unique_images[:n_val_images])
        train_image_set = set(unique_images[n_val_images:])

        train_indices = [i for i in pool_indices if _idx_df.iloc[i]["nsdId"] in train_image_set]
        val_indices = [i for i in pool_indices if _idx_df.iloc[i]["nsdId"] in val_image_set]

        logger.info(
            "Image-level split: %d unique images -> %d train (%d trials), %d val (%d trials)",
            len(unique_images), len(train_image_set), len(train_indices),
            len(val_image_set), len(val_indices),
        )
        _split_nsd_ids_used = {
            "train_nsd_ids": sorted(int(x) for x in train_image_set),
            "val_nsd_ids": sorted(int(x) for x in val_image_set),
        }
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
    elif _split_nsd_ids_used is None:
        train_split = config["data"]["train_split"]
        val_split = config["data"]["val_split"]
        n_total = len(full_dataset)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)

        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(data_seed)).tolist()
        train_dataset = Subset(full_dataset, indices[:n_train])
        val_dataset = Subset(full_dataset, indices[n_train : n_train + n_val])

    # --- Persist split for cross-phase reproducibility ---
    if _split_nsd_ids_used is not None:
        _split_save_path = output_dir / "split.json"
        _split_payload = {
            "seed": data_seed,
            "split_by_image": split_by_image,
            "exclude_shared1000": exclude_shared1000,
            "n_train_images": len(_split_nsd_ids_used["train_nsd_ids"]),
            "n_val_images": len(_split_nsd_ids_used["val_nsd_ids"]),
            **_split_nsd_ids_used,
        }
        if _split_file and os.path.isfile(_split_file):
            _split_payload["loaded_from"] = _split_file
        with open(_split_save_path, "w") as _sf:
            json.dump(_split_payload, _sf)
        logger.info("Saved split assignments (%d train / %d val images) to %s",
                     len(_split_nsd_ids_used["train_nsd_ids"]),
                     len(_split_nsd_ids_used["val_nsd_ids"]),
                     _split_save_path)

    # --- Split overlap diagnostic (detect cross-phase leakage) ---
    if _split_nsd_ids_used is not None and _pm_path:
        _pm_parent_dir = Path(_pm_path).parent
        _pm_split_path = _pm_parent_dir / "split.json"
        if _pm_split_path.exists():
            with open(_pm_split_path) as _psf:
                _parent_split = json.load(_psf)
            _parent_train = set(int(x) for x in _parent_split["train_nsd_ids"])
            _parent_val = set(int(x) for x in _parent_split["val_nsd_ids"])
            _cur_train = set(int(x) for x in _split_nsd_ids_used["train_nsd_ids"])
            _cur_val = set(int(x) for x in _split_nsd_ids_used["val_nsd_ids"])
            _leaked = _cur_val & _parent_train
            _exact_match = _cur_train == _parent_train and _cur_val == _parent_val
            if _leaked:
                _msg = (
                    "SPLIT OVERLAP: current val images overlap with parent train split "
                    f"({_pm_split_path}); leakage_count={len(_leaked)}"
                )
                if _require_parent_split_match:
                    raise RuntimeError(_msg)
                logger.warning("%s", _msg)
            elif _require_parent_split_match and not _exact_match:
                raise RuntimeError(
                    "Parent split mismatch for pretrained_model_path. "
                    f"Current split differs from {_pm_split_path}"
                )
            else:
                logger.info(
                    "Parent split check PASSED for pretrained model: exact_match=%s leakage=0 (%s)",
                    _exact_match,
                    _pm_split_path,
                )
        elif _require_parent_split_match:
            raise FileNotFoundError(
                f"require_parent_split_match=true but parent split.json was not found: {_pm_split_path}"
            )
        else:
            logger.info(
                "No parent split.json at %s — cannot check pretrained_model split alignment",
                _pm_split_path,
            )

    if _split_nsd_ids_used is not None and _pe_path:
        _pe_parent_dir = Path(_pe_path).parent
        _pe_split_path = _pe_parent_dir / "split.json"
        if _pe_split_path.exists():
            with open(_pe_split_path) as _psf:
                _parent_split = json.load(_psf)
            _parent_train = set(int(x) for x in _parent_split["train_nsd_ids"])
            _cur_val = set(_split_nsd_ids_used["val_nsd_ids"])
            _leaked = _cur_val & _parent_train
            _pct = 100.0 * len(_leaked) / max(len(_cur_val), 1)
            if _leaked:
                logger.warning(
                    "SPLIT OVERLAP: %d / %d current val images (%.1f%%) were in "
                    "parent encoder's train set (%s). This indicates data leakage.",
                    len(_leaked), len(_cur_val), _pct, _pe_split_path,
                )
            else:
                logger.info(
                    "Split overlap check PASSED: 0 / %d val images overlap "
                    "with parent train set (%s)",
                    len(_cur_val), _pe_split_path,
                )
        else:
            logger.info(
                "No parent split.json at %s — cannot check for leakage "
                "(re-run parent experiment to generate it)", _pe_split_path,
            )

    # --- fMRI per-voxel z-scoring (computed on training split only) ---
    normalize_fmri = config["data"].get("normalize_fmri", False)
    zscore_mode = config["data"].get("zscore_mode", "global")
    _zscore_stats_path = None
    if normalize_fmri and isinstance(full_dataset, PreextractedNSDDataset):
        _train_idx = train_dataset.indices if hasattr(train_dataset, "indices") else list(range(len(train_dataset)))
        zscore_dir = output_dir / "zscore_stats"
        zscore_dir.mkdir(parents=True, exist_ok=True)

        if zscore_mode == "per_session" and "session" in full_dataset.index_df.columns:
            # Per-session z-scoring: remove session-level drift (scanner drift,
            # head position changes) that spans ~1 year of data collection.
            # Stats are computed from training samples within each session only.
            sessions = full_dataset.index_df["session"].values
            unique_sessions = np.unique(sessions)
            train_set = set(_train_idx)

            # Pre-compute global training stats as fallback
            _all_train_feat = full_dataset.features[_train_idx]
            _global_mean = _all_train_feat.mean(axis=0, keepdims=True)
            _global_std = _all_train_feat.std(axis=0, keepdims=True)
            _global_std[_global_std < 1e-6] = 1.0
            del _all_train_feat

            n_fallback = 0
            for sess in unique_sessions:
                sess_mask = sessions == sess
                sess_indices = np.where(sess_mask)[0]
                sess_train_indices = [i for i in sess_indices if i in train_set]

                if len(sess_train_indices) >= 2:
                    sess_train_feat = full_dataset.features[sess_train_indices]
                    sess_mean = sess_train_feat.mean(axis=0, keepdims=True)
                    sess_std = sess_train_feat.std(axis=0, keepdims=True)
                    sess_std[sess_std < 1e-6] = 1.0
                else:
                    sess_mean = _global_mean
                    sess_std = _global_std
                    n_fallback += 1

                full_dataset.features[sess_mask] = (
                    (full_dataset.features[sess_mask] - sess_mean) / sess_std
                ).astype(np.float32)

                np.save(zscore_dir / f"session_{int(sess)}_mean.npy", sess_mean.astype(np.float32))
                np.save(zscore_dir / f"session_{int(sess)}_std.npy", sess_std.astype(np.float32))

            np.save(zscore_dir / "global_fallback_mean.npy", _global_mean.astype(np.float32))
            np.save(zscore_dir / "global_fallback_std.npy", _global_std.astype(np.float32))
            logger.info(
                "Applied per-session z-scoring (%d sessions, %d fallback, %d train trials, %d voxels)",
                len(unique_sessions), n_fallback, len(_train_idx), full_dataset.features.shape[1],
            )
            del _global_mean, _global_std
        else:
            if zscore_mode == "per_session":
                logger.warning(
                    "zscore_mode='per_session' requested but no 'session' column in index_df — "
                    "falling back to global z-scoring"
                )
            _train_feat = full_dataset.features[_train_idx]
            _voxel_mean = _train_feat.mean(axis=0, keepdims=True)
            _voxel_std = _train_feat.std(axis=0, keepdims=True)
            _voxel_std[_voxel_std < 1e-6] = 1.0
            full_dataset.features = ((full_dataset.features - _voxel_mean) / _voxel_std).astype(np.float32)
            logger.info(
                "Applied global per-voxel z-scoring (%d train trials, %d voxels)",
                len(_train_idx), full_dataset.features.shape[1],
            )
            np.save(zscore_dir / "voxel_mean.npy", _voxel_mean.astype(np.float32))
            np.save(zscore_dir / "voxel_std.npy", _voxel_std.astype(np.float32))
            del _train_feat, _voxel_mean, _voxel_std

        _zscore_stats_path = str(zscore_dir)
        logger.info("Saved z-scoring stats to %s", zscore_dir)

    elif normalize_fmri and _is_multi_subject:
        _train_idx = train_dataset.indices if hasattr(train_dataset, "indices") else list(range(len(train_dataset)))
        zscore_dir = output_dir / "zscore_stats"
        zscore_dir.mkdir(parents=True, exist_ok=True)
        train_set = set(_train_idx)

        subj_ints = full_dataset.index_df["_subject_int"].values
        local_idxs = full_dataset._feat_local_idx
        has_session = "session" in full_dataset.index_df.columns
        use_per_session = zscore_mode == "per_session" and has_session

        if not use_per_session and zscore_mode == "per_session":
            logger.warning(
                "zscore_mode='per_session' but no 'session' column in multi-subject index_df "
                "— falling back to global z-scoring per subject"
            )

        total_sessions_done = 0
        total_fallback = 0

        for s_idx, subj_name in enumerate(full_dataset.subjects):
            subj_mask = subj_ints == s_idx
            subj_rows = np.where(subj_mask)[0]
            subj_local = local_idxs[subj_rows]
            subj_train_rows = np.array([r for r in subj_rows if r in train_set])
            feats = full_dataset.features_list[s_idx]

            if len(subj_train_rows) == 0:
                logger.warning("Subject %s: no training trials — skipping z-score", subj_name)
                continue

            subj_train_local = local_idxs[subj_train_rows]

            if use_per_session:
                sessions = full_dataset.index_df["session"].values
                subj_sessions = sessions[subj_rows]
                unique_sess = np.unique(subj_sessions)

                global_train_feat = feats[subj_train_local]
                g_mean = global_train_feat.mean(axis=0, keepdims=True)
                g_std = global_train_feat.std(axis=0, keepdims=True)
                g_std[g_std < 1e-6] = 1.0
                del global_train_feat

                n_fb = 0
                for sess in unique_sess:
                    sess_row_mask = (subj_ints == s_idx) & (sessions == sess)
                    sess_local = local_idxs[np.where(sess_row_mask)[0]]
                    sess_train_mask = sess_row_mask & np.isin(np.arange(len(full_dataset.index_df)), subj_train_rows)
                    sess_train_local = local_idxs[np.where(sess_train_mask)[0]]

                    if len(sess_train_local) >= 2:
                        s_mean = feats[sess_train_local].mean(axis=0, keepdims=True)
                        s_std = feats[sess_train_local].std(axis=0, keepdims=True)
                        s_std[s_std < 1e-6] = 1.0
                    else:
                        s_mean, s_std = g_mean, g_std
                        n_fb += 1

                    feats[sess_local] = (
                        (feats[sess_local] - s_mean) / s_std
                    ).astype(np.float32)

                    np.save(
                        zscore_dir / f"{subj_name}_session_{int(sess)}_mean.npy",
                        s_mean.astype(np.float32),
                    )
                    np.save(
                        zscore_dir / f"{subj_name}_session_{int(sess)}_std.npy",
                        s_std.astype(np.float32),
                    )

                total_sessions_done += len(unique_sess)
                total_fallback += n_fb
                del g_mean, g_std
                logger.info(
                    "  %s: per-session z-scored %d sessions (%d fallback), %d voxels",
                    subj_name, len(unique_sess), n_fb, feats.shape[1],
                )
            else:
                s_mean = feats[subj_train_local].mean(axis=0, keepdims=True)
                s_std = feats[subj_train_local].std(axis=0, keepdims=True)
                s_std[s_std < 1e-6] = 1.0
                all_local = local_idxs[subj_rows]
                feats[all_local] = (
                    (feats[all_local] - s_mean) / s_std
                ).astype(np.float32)
                np.save(zscore_dir / f"{subj_name}_mean.npy", s_mean.astype(np.float32))
                np.save(zscore_dir / f"{subj_name}_std.npy", s_std.astype(np.float32))
                logger.info(
                    "  %s: global z-scored %d train trials, %d voxels",
                    subj_name, len(subj_train_local), feats.shape[1],
                )

        logger.info(
            "Multi-subject z-scoring complete: %d subjects, mode=%s",
            len(full_dataset.subjects), "per_session" if use_per_session else "global",
        )
        _zscore_stats_path = str(zscore_dir)

    if preprocessor is not None and preproc_needs_fit:
        n_train = len(train_dataset)
        logger.info("Auto-fitting embedding preprocessor on %d training samples...", n_train)
        try:
            emb_col = resolve_embedding_column(embeddings_df, _EMBEDDING_COLUMN_OVERRIDE)
        except ValueError:
            emb_col = None
        if emb_col is not None:
            _train_idx_fit = train_dataset.indices if hasattr(train_dataset, "indices") else list(range(len(train_dataset)))
            if hasattr(full_dataset, "index_df") and "nsdId" in full_dataset.index_df.columns:
                train_nsd_ids = set(full_dataset.index_df.iloc[_train_idx_fit]["nsdId"].values)
                train_emb_mask = embeddings_df["nsdId"].isin(train_nsd_ids)
                train_embeddings = np.stack(embeddings_df.loc[train_emb_mask, emb_col].values)
                logger.info("Fitting preprocessor on %d train-only embeddings (not all %d)",
                            len(train_embeddings), len(embeddings_df))
            else:
                train_embeddings = np.stack(embeddings_df[emb_col].values)
        else:
            logger.warning("No embedding column found, fitting on first %d samples via dataset", min(n_train, 1000))
            train_embeddings = np.stack([full_dataset[i][1].numpy() for i in range(min(n_train, 1000))])
        if _retrieval_projector is not None:
            train_embeddings = _retrieval_projector.transform(train_embeddings)
        preprocessor.fit(train_embeddings)
        artifact_path = Path(resolve_preproc_artifact(subject, config))
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        preprocessor.save(artifact_path)
        logger.info("Saved fitted preprocessor to %s", artifact_path)

    batch_size = config["training"]["batch_size"]
    use_preextracted = isinstance(full_dataset, PreextractedNSDDataset) or _is_multi_subject
    dl_workers = 2 if use_preextracted else 0
    dl_pin = device.startswith("cuda")

    _collate_fn = None
    if _is_multi_subject:
        import torch.nn.functional as F

        def _multi_subject_collate(batch):
            # Handles:
            #   - 3-element tuples: (fmri, emb, subj_id)
            #   - 4-element tuples: (fmri, emb, subj_id, hier_targets)
            #   - dual_target dict samples from MultiSubjectPreextractedDataset
            first = batch[0]

            if isinstance(first, dict):
                fmri_list = [sample["fmri"] for sample in batch]
                max_v = max(f.shape[0] for f in fmri_list)
                padded = [F.pad(f, (0, max_v - f.shape[0])) for f in fmri_list]
                result = {
                    "fmri": torch.stack(padded),
                    "retrieval_target": torch.stack([sample["retrieval_target"] for sample in batch]),
                    "rich_target": torch.stack([sample["rich_target"] for sample in batch]),
                    "subject_id": torch.stack([sample["subject_id"] for sample in batch]).to(dtype=torch.long),
                    "nsd_id": torch.stack([sample["nsd_id"] for sample in batch]).to(dtype=torch.long),
                }
                if "rerank_target" in first:
                    result["rerank_target"] = torch.stack([sample["rerank_target"] for sample in batch])
                return result

            has_hier = len(first) == 4
            if has_hier:
                fmri_list, emb_list, subj_ids, hier_list = zip(*batch)
            else:
                fmri_list, emb_list, subj_ids = zip(*batch)
                hier_list = None
            max_v = max(f.shape[0] for f in fmri_list)
            padded = [F.pad(f, (0, max_v - f.shape[0])) for f in fmri_list]
            result = (
                torch.stack(padded),
                torch.stack(emb_list),
                torch.tensor(subj_ids, dtype=torch.long),
            )
            if has_hier and hier_list:
                # Stack each hierarchical column across the batch
                keys = hier_list[0].keys()
                hier_stacked = {
                    k: torch.stack([h[k] for h in hier_list])
                    for k in keys
                }
                result = result + (hier_stacked,)
            return result

        _collate_fn = _multi_subject_collate

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=dl_workers, pin_memory=dl_pin,
        persistent_workers=(dl_workers > 0),
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=dl_workers, pin_memory=dl_pin,
        persistent_workers=(dl_workers > 0),
        collate_fn=_collate_fn,
    )
    logger.info("Train: %d | Val: %d", len(train_dataset), len(val_dataset))

    # Build val nsdId array for image-level retrieval (dedup across repetitions)
    _val_nsd_ids = None
    if hasattr(val_dataset, "indices") and hasattr(full_dataset, "index_df"):
        _val_nsd_ids = full_dataset.index_df.iloc[list(val_dataset.indices)]["nsdId"].values

    # --- AMP ---
    grad_accum_steps = config["training"].get("gradient_accumulation_steps", 16)
    use_amp = config["training"].get("mixed_precision", False) and device.startswith("cuda")
    _amp_dtype_str = config["training"].get("mixed_precision_dtype", "fp16")
    _amp_dtype = torch.bfloat16 if _amp_dtype_str == "bf16" else torch.float16
    _use_grad_scaler = use_amp and _amp_dtype != torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=_use_grad_scaler)
    logger.info("Grad accum: %d (effective batch %d) | AMP: %s (%s)",
                grad_accum_steps, batch_size * grad_accum_steps, use_amp,
                _amp_dtype_str if use_amp else "off")

    # --- LR scheduler ---
    num_epochs = config["training"]["num_epochs"]
    total_steps = num_epochs * len(train_loader) // grad_accum_steps
    warmup_steps = config["training"].get("warmup_epochs", 5) * len(train_loader) // grad_accum_steps
    min_lr = float(config["training"].get("min_lr", 1e-6))
    base_lr = float(opt_cfg.get("lr", 1e-4))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Resume ---
    start_epoch = 1
    best_val_loss = float("inf")
    best_r1 = 0.0
    best_metric_val = 0.0
    global_step = 0
    best_epoch = 0

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_epoch, best_val_loss, global_step = load_checkpoint(
                resume_path, model, optimizer, lr_sched, scaler, device,
                losses=losses, ema=ema,
            )
        else:
            logger.warning("Resume path not found: %s — training from scratch", resume_path)

    # --- Manifest ---
    manifest = build_manifest(config, args, model, device)
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    metrics_logger = MetricsLogger(output_dir)
    early_stop_patience = config["training"].get("early_stop_patience", 15)
    early_stop_min_delta = config["training"].get("early_stop_min_delta", 0.001)
    save_frequency = config["training"].get("save_frequency", 0)
    patience_counter = 0

    _ckpt_metric_name = (
        config.get("evaluation", {}).get("checkpoint_metric")
        or config.get("training", {}).get("checkpoint_metric", "r@1")
    )
    _ckpt_lower_is_better = _metric_lower_is_better(_ckpt_metric_name)
    _extra_best_metric_names = [
        m for m in config.get("evaluation", {}).get("best_checkpoints", [])
        if isinstance(m, str) and m and m != _ckpt_metric_name
    ]
    _extra_best_trackers: Dict[str, Dict[str, Any]] = {
        m: {
            "best_value": float("inf") if _metric_lower_is_better(m) else 0.0,
            "best_epoch": 0,
            "lower_is_better": _metric_lower_is_better(m),
        }
        for m in _extra_best_metric_names
    }
    if _ckpt_metric_name != "r@1":
        logger.info("Checkpoint metric: %s (lower_is_better=%s)",
                     _ckpt_metric_name, _ckpt_lower_is_better)
    if _extra_best_trackers:
        logger.info(
            "Additional best-checkpoint metrics: %s",
            ", ".join(
                f"{name}(lower={tracker['lower_is_better']})"
                for name, tracker in _extra_best_trackers.items()
            ),
        )
    if _ckpt_lower_is_better:
        best_metric_val = float("inf")

    _vmf_is_log = getattr(model, "vmf_output_is_log", True)
    _fusion_eval_cfg = _get_eval_fusion_cfg(config)
    _tri_fusion_eval_cfg = _get_eval_tri_fusion_cfg(config)
    if _fusion_eval_cfg is not None:
        logger.info(
            "Fusion eval enabled: compact=%s family=%s norm=%s shortlist_k=%d alpha=%.2f",
            _fusion_eval_cfg.get("compact_score", "csls"),
            _fusion_eval_cfg.get("family", "normalized_weighted"),
            _fusion_eval_cfg.get("normalization", "zscore"),
            int(_fusion_eval_cfg.get("shortlist_k", 50)),
            float(_fusion_eval_cfg.get("alpha", 0.8)),
        )
    if _tri_fusion_eval_cfg is not None:
        logger.info(
            "Tri-fusion eval enabled: compact=%s legacy=%s family=%s norm=%s "
            "shortlist_k=%d alpha/beta/gamma=%.2f/%.2f/%.2f",
            _tri_fusion_eval_cfg.get("compact_score", "csls"),
            _tri_fusion_eval_cfg.get("legacy_score", "csls"),
            _tri_fusion_eval_cfg.get("family", "normalized_weighted"),
            _tri_fusion_eval_cfg.get("normalization", "zscore"),
            int(_tri_fusion_eval_cfg.get("shortlist_k", 150)),
            float(_tri_fusion_eval_cfg.get("alpha", 0.3)),
            float(_tri_fusion_eval_cfg.get("beta", 0.0)),
            float(_tri_fusion_eval_cfg.get("gamma", 0.7)),
        )

    if args.post_eval_shared1000_only:
        _eval_ckpt_path = Path(args.resume) if args.resume else (output_dir / "checkpoint_best.pt")
        if not _eval_ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint for post-training shared1000 evaluation not found: {_eval_ckpt_path}"
            )
        _eval_ckpt = torch.load(_eval_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(_eval_ckpt["model_state_dict"])
        logger.info(
            "Loaded checkpoint for post-training shared1000 evaluation: %s (epoch=%s)",
            _eval_ckpt_path,
            _eval_ckpt.get("epoch", "unknown"),
        )
        _run_post_training_shared1000_eval(
            output_dir=output_dir,
            model=model,
            config=config,
            subject=subject,
            device=device,
            embeddings_df=embeddings_df,
            preprocessor=preprocessor,
            vmf_is_log=_vmf_is_log,
            zscore_stats_path=_zscore_stats_path,
            is_multi_subject=_is_multi_subject,
            token_cache=_token_cache,
            rerank_cache=_rerank_cache,
            retrieval_projector=_retrieval_projector,
            legacy_teacher_model=legacy_teacher_model,
        )
        return

    # --- Save train-split predictions (for V39 reranker cache building) ---
    if args.save_train_preds:
        _eval_ckpt_path = Path(args.resume) if args.resume else (output_dir / "checkpoint_best.pt")
        if not _eval_ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {_eval_ckpt_path}"
            )
        _eval_ckpt = torch.load(_eval_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(_eval_ckpt["model_state_dict"])
        logger.info(
            "Loaded checkpoint for train prediction saving: %s (epoch=%s)",
            _eval_ckpt_path, _eval_ckpt.get("epoch", "unknown"),
        )
        model.eval()

        # Build train nsd_ids for image-level averaging
        _train_nsd_ids = None
        if hasattr(train_dataset, "indices") and hasattr(full_dataset, "index_df"):
            _train_nsd_ids = full_dataset.index_df.iloc[list(train_dataset.indices)]["nsdId"].values

        # Create non-shuffled train loader for deterministic predictions
        _train_pred_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False,
            num_workers=dl_workers, pin_memory=dl_pin,
            collate_fn=_collate_fn,
        )

        # Run validate() on train set to get predictions
        _train_metrics, _train_preds, _train_gts = validate(
            model, _train_pred_loader, losses, loss_weights, device, preprocessor, queue,
            vmf_is_log=_vmf_is_log,
            current_epoch=_eval_ckpt.get("epoch", 0),
            config_ref=config,
            legacy_teacher_model=legacy_teacher_model,
        )
        _train_extras = _train_metrics.pop("_val_extras", {})

        # Collect kappas
        _save_model_type_tr = getattr(model, "model_type", "deterministic")
        _train_kappas = None
        if _save_model_type_tr in ("vmf", "vmf_dcf", "vmf_triple"):
            _kappa_list_tr: list = []
            with torch.no_grad():
                for _sb in _train_pred_loader:
                    if isinstance(_sb, dict):
                        _s_fmri = _sb["fmri"].to(device, dtype=torch.float32)
                        _s_sid = _sb.get("subject_id")
                        if _s_sid is not None:
                            _s_sid = _s_sid.to(device)
                    else:
                        _s_fmri = _sb[0].to(device, dtype=torch.float32)
                        _s_sid = _sb[2].to(device) if _is_multi_subject and len(_sb) >= 3 else None
                    _s_out = model(_s_fmri, subject_ids=_s_sid) if _s_sid is not None else model(_s_fmri)
                    if isinstance(_s_out, tuple) and len(_s_out) >= 2:
                        _s_aux = _s_out[1]
                        if isinstance(_s_aux, dict):
                            _s_k = _s_aux.get("kappa", _s_aux.get("concentration"))
                        elif torch.is_tensor(_s_aux):
                            _s_k = _s_aux
                        else:
                            _s_k = None
                        if _s_k is not None:
                            if _vmf_is_log:
                                _s_k = _s_k.exp()
                            _kappa_list_tr.append(_s_k.squeeze(-1).detach().cpu().numpy())
            if _kappa_list_tr:
                _train_kappas = np.concatenate(_kappa_list_tr)
        elif _save_model_type_tr == "dense_vmf_hybrid" and "vmf_kappas" in _train_extras:
            _train_kappas = np.asarray(_train_extras["vmf_kappas"], dtype=np.float32)

        # Image-level averaging
        if _train_nsd_ids is not None:
            _train_preds, _train_nsd_ids_save = _aggregate_rows_by_nsd_id(
                _train_preds, _train_nsd_ids, reduce="mean", normalize=True,
            )
            _train_gts, _ = _aggregate_rows_by_nsd_id(
                _train_gts, _train_nsd_ids, reduce="first", normalize=False,
            )
            if _train_kappas is not None:
                _train_kappas, _ = _aggregate_rows_by_nsd_id(
                    _train_kappas, _train_nsd_ids[: len(_train_kappas)], reduce="mean", normalize=False,
                )
        else:
            _train_nsd_ids_save = None

        # Save
        _metrics_save_dir = output_dir / "metrics"
        _metrics_save_dir.mkdir(parents=True, exist_ok=True)
        np.save(_metrics_save_dir / "train_predictions.npy", _train_preds)
        np.save(_metrics_save_dir / "train_ground_truth.npy", _train_gts)
        logger.info("Saved train predictions %s and ground truth %s to %s",
                     _train_preds.shape, _train_gts.shape, _metrics_save_dir)
        if _train_kappas is not None:
            np.save(_metrics_save_dir / "train_kappas.npy", _train_kappas)
            logger.info("Saved train kappas %s", _train_kappas.shape)
        if _train_nsd_ids_save is not None:
            np.save(_metrics_save_dir / "train_nsd_ids.npy", _train_nsd_ids_save)
            logger.info("Saved train nsd_ids %s", _train_nsd_ids_save.shape)

        # compact-family export: save dense/compact aliases and auxiliary heads
        if _save_model_type_tr in ("vmf_triple", "dense_vmf_hybrid"):
            np.save(_metrics_save_dir / "train_predictions_compact.npy", _train_preds)
            np.save(_metrics_save_dir / "train_ground_truth_compact.npy", _train_gts)
            logger.info("Saved train compact predictions %s", _train_preds.shape)
            if _save_model_type_tr == "dense_vmf_hybrid":
                _train_vmf_preds = _train_extras.get("vmf_preds")
                _train_vmf_kappas = _train_extras.get("vmf_kappas")
                if _train_vmf_preds is not None:
                    if _train_nsd_ids is not None:
                        _train_vmf_preds, _ = _aggregate_rows_by_nsd_id(
                            _train_vmf_preds, _train_nsd_ids, reduce="mean", normalize=True,
                        )
                    else:
                        _train_vmf_preds = _train_vmf_preds / np.maximum(
                            np.linalg.norm(_train_vmf_preds, axis=-1, keepdims=True), 1e-8,
                        )
                if _train_vmf_kappas is not None and _train_nsd_ids is not None:
                    _train_vmf_kappas, _ = _aggregate_rows_by_nsd_id(
                        _train_vmf_kappas, _train_nsd_ids[: len(_train_vmf_kappas)], reduce="mean",
                    )
                _save_aux_vmf_arrays(
                    _metrics_save_dir,
                    "train",
                    _train_vmf_preds,
                    _train_vmf_kappas,
                )
                if _train_vmf_kappas is not None and _train_kappas is None:
                    np.save(_metrics_save_dir / "train_kappas.npy", _train_vmf_kappas.astype(np.float32))
                    logger.info("Saved train kappas alias from aux-vMF %s", _train_vmf_kappas.shape)
            for _extra_name, _pred_key, _gt_key in [
                ("rich", "rich_preds", "rich_gts"),
                ("legacy", "legacy_preds", "legacy_gts"),
                ("rerank", "rerank_preds", "rerank_gts"),
            ]:
                if _pred_key in _train_extras:
                    _ep = _train_extras[_pred_key]
                    _eg = _train_extras.get(_gt_key)
                    if _train_nsd_ids is not None:
                        _ep, _ = _aggregate_rows_by_nsd_id(_ep, _train_nsd_ids, reduce="mean")
                        if _eg is not None:
                            _eg, _ = _aggregate_rows_by_nsd_id(_eg, _train_nsd_ids, reduce="first")
                    if _extra_name in ("legacy", "rerank"):
                        _ep = _ep / np.maximum(np.linalg.norm(_ep, axis=-1, keepdims=True), 1e-8)
                    np.save(_metrics_save_dir / f"train_predictions_{_extra_name}.npy", _ep)
                    if _eg is not None:
                        np.save(_metrics_save_dir / f"train_ground_truth_{_extra_name}.npy", _eg)
                    logger.info("Saved train %s predictions %s", _extra_name, _ep.shape)

        logger.info("Train prediction saving complete.")
        return

    # --- MixCo config ---
    _mixco_cfg = config.get("training", {}).get("mixco", {})
    if _mixco_cfg.get("enabled", False):
        logger.info("MixCo enabled: alpha=%.2f, temp=%.4f, weight=%.2f",
                     _mixco_cfg.get("alpha", 0.2),
                     _mixco_cfg.get("temperature", 0.006),
                     _mixco_cfg.get("weight", 1.0))

    # --- SoftCLIP / MixCo phase schedule ---
    _softclip_cfg = config.get("loss", {}).get("softclip", {})
    _has_softclip = "softclip" in losses
    _has_mixco = _mixco_cfg.get("enabled", False)
    _softclip_from_start = config.get("training", {}).get("softclip_from_start", False)
    _softclip_transition_epoch = int(num_epochs / 3) + 1
    if _softclip_from_start and _has_softclip and _has_mixco:
        logger.info(
            "SoftCLIP + MixCo: both active from epoch 1 (softclip_from_start=true)",
        )
    elif _has_softclip and _has_mixco:
        logger.info(
            "SoftCLIP + MixCo schedule: MixCo epochs 1-%d, SoftCLIP epochs %d-%d",
            _softclip_transition_epoch - 1, _softclip_transition_epoch, num_epochs,
        )
    _softclip_loss_obj = losses.pop("softclip", None)

    # --- SPCL curriculum schedule ---
    spcl_cfg = config.get("loss", {}).get("vmf_nce_spcl", {})
    spcl_t_start = spcl_cfg.get("initial_curriculum_t", 100.0)
    spcl_t_end = spcl_cfg.get("final_curriculum_t", 1.0)
    spcl_warmup = spcl_cfg.get("warmup_epochs", 10)

    # --- Kappa curriculum schedule (V18) ---
    _kappa_cur_cfg = config.get("training", {}).get("kappa_curriculum", {})
    _kappa_cur_enabled = _kappa_cur_cfg.get("enabled", False)
    _kappa_cur_start = _kappa_cur_cfg.get("start", 10.0)
    _kappa_cur_end = _kappa_cur_cfg.get("end", 50.0)
    _kappa_cur_epochs = _kappa_cur_cfg.get("anneal_epochs", 60)
    if _kappa_cur_enabled:
        logger.info(
            "Kappa curriculum enabled: %.1f -> %.1f over %d epochs",
            _kappa_cur_start, _kappa_cur_end, _kappa_cur_epochs,
        )

    _ckpt_meta = {
        "normalize_fmri": normalize_fmri,
        "zscore_mode": zscore_mode,
        "preprocessing_enabled": config.get("preprocessing", {}).get("enabled", False),
        "zscore_stats_path": _zscore_stats_path,
        "average_repetitions": config["data"].get("average_repetitions", False),
        "split_by_image": config["data"].get("split_by_image", False),
    }

    # --- Two-stage training (V12): contrastive -> NLL fine-tuning ---
    _stage2_cfg = config.get("training", {}).get("stage2", {})
    _stage2_enabled = _stage2_cfg.get("enabled", False)
    _stage2_start = _stage2_cfg.get("start_epoch", 150)
    _stage2_activated = False
    _stage1_loss_weights = dict(loss_weights)

    if _stage2_enabled:
        logger.info(
            "Two-stage training ENABLED: Stage 1 (contrastive) epochs 1-%d, "
            "Stage 2 (NLL fine-tuning) epochs %d+",
            _stage2_start - 1, _stage2_start,
        )

    # --- Homoscedastic uncertainty weighting (Kendall et al., 2018) ---
    _auto_weight_cfg = config.get("loss", {}).get("auto_weight", {})
    _auto_weight_enabled = _auto_weight_cfg.get("enabled", False)
    _log_sigmas = None
    if _auto_weight_enabled:
        _weightable = [k for k in losses if k != "kappa_reg"]
        _log_sigmas = nn.ParameterDict(
            {k: nn.Parameter(torch.zeros(1, device=device)) for k in _weightable}
        )
        _log_sigmas = _log_sigmas.to(device)
        optimizer.add_param_group({"params": list(_log_sigmas.parameters()), "lr": 1e-3})
        lr_sched.base_lrs.append(1e-3)
        lr_sched.lr_lambdas.append(lambda _: 1.0)
        logger.info("Homoscedastic auto-weighting enabled for: %s", _weightable)

    wall_start = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        # --- Two-stage transition (V12) ---
        if _stage2_enabled and epoch >= _stage2_start and not _stage2_activated:
            _stage2_activated = True
            _s2_lr_factor = _stage2_cfg.get("lr_factor", 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = pg["lr"] * _s2_lr_factor
            _s2_mse_w = _stage2_cfg.get("mse_weight", loss_weights.get("mse", 1.0))
            _s2_nll_w = _stage2_cfg.get("vmf_nll_weight", 0.0)
            _s2_sc_w = _stage2_cfg.get("softclip_weight", 0.3)
            _s2_nce_w = _stage2_cfg.get("vmf_nce_weight", 0.0)
            _s2_spcl_w = _stage2_cfg.get("vmf_nce_spcl_weight", 0.0)
            loss_weights["mse"] = _s2_mse_w
            loss_weights["vmf_nll"] = _s2_nll_w
            loss_weights["softclip"] = _s2_sc_w
            loss_weights["vmf_nce"] = _s2_nce_w
            loss_weights["vmf_nce_spcl"] = _s2_spcl_w
            loss_weights["vmf_nce_multitask"] = 0.0
            _s2_hier_w = _stage2_cfg.get("hierarchical_clip_weight",
                                          loss_weights.get("hierarchical_clip", 0.0))
            loss_weights["hierarchical_clip"] = _s2_hier_w
            _s2_da_w = _stage2_cfg.get("direct_alignment_weight",
                                        loss_weights.get("direct_alignment", 0.0))
            loss_weights["direct_alignment"] = _s2_da_w
            if not _stage2_cfg.get("mixco_enabled", False):
                _has_mixco = False
            patience_counter = 0
            best_r1 = 0.0
            best_metric_val = float("inf") if _ckpt_lower_is_better else 0.0
            early_stop_patience = _stage2_cfg.get("patience", 20)
            logger.info(
                "[STAGE 2] Activated at epoch %d: LR *= %.2f, "
                "mse=%.1f, vmf_nll=%.1f, softclip=%.1f, vmf_nce=%.1f, spcl=%.1f, "
                "hier=%.1f, da=%.1f, mixco=%s, patience=%d",
                epoch, _s2_lr_factor, _s2_mse_w, _s2_nll_w, _s2_sc_w, _s2_nce_w,
                _s2_spcl_w, _s2_hier_w, _s2_da_w, _has_mixco, early_stop_patience,
            )

        # --- V25b: Unfreeze backbone after adapter warm-up ---
        if (_cross_subject_enabled and _cs_backbone_frozen
                and epoch == _cs_freeze_epochs + 1):
            for name, param in model.named_parameters():
                if name.startswith(("encoder.", "decoder.")):
                    param.requires_grad = True
            # Add only backbone params NOT already tracked by the optimizer
            _existing_param_ids = {
                id(p) for group in optimizer.param_groups for p in group["params"]
            }
            _backbone_params = [
                p for n, p in model.named_parameters()
                if n.startswith(("encoder.", "decoder."))
                and id(p) not in _existing_param_ids
            ]
            if _backbone_params:
                _backbone_lr = float(opt_cfg.get("lr", 1e-4)) * _cs_backbone_lr_factor
                optimizer.add_param_group({
                    "params": _backbone_params,
                    "lr": _backbone_lr,
                })
                logger.info(
                    "[CROSS-SUBJECT] Unfreezing backbone at epoch %d: "
                    "%d params added at lr=%.2e (%.1f× base)",
                    epoch, len(_backbone_params), _backbone_lr,
                    _cs_backbone_lr_factor,
                )
            else:
                logger.warning(
                    "[CROSS-SUBJECT] Unfreeze at epoch %d: all backbone params "
                    "already in optimizer — skipping add_param_group",
                    epoch,
                )
            _cs_backbone_frozen = False  # prevent re-triggering

        _stage_prefix = "[STAGE 2] " if _stage2_activated else ""
        logger.info("\n%sEpoch %d/%d | lr=%.2e", _stage_prefix, epoch, num_epochs, optimizer.param_groups[0]["lr"])

        # Phase-switch: MixCo warmup -> SoftCLIP distillation
        if _softclip_from_start and _softclip_loss_obj is not None:
            if "softclip" not in losses:
                losses["softclip"] = _softclip_loss_obj
            _epoch_mixco = _mixco_cfg if _has_mixco else None
        elif _softclip_loss_obj is not None and _has_mixco:
            if epoch < _softclip_transition_epoch:
                losses.pop("softclip", None)
                _epoch_mixco = _mixco_cfg
            else:
                if "softclip" not in losses:
                    losses["softclip"] = _softclip_loss_obj
                    logger.info("Epoch %d: switching from MixCo to SoftCLIP", epoch)
                _epoch_mixco = None
        elif _softclip_loss_obj is not None:
            if "softclip" not in losses:
                losses["softclip"] = _softclip_loss_obj
            _epoch_mixco = None
        else:
            _epoch_mixco = _mixco_cfg if _has_mixco else None

        # Update SPCL curriculum temperature
        if "vmf_nce_spcl" in losses:
            if epoch <= spcl_warmup:
                cur_t = spcl_t_start
            else:
                progress = (epoch - spcl_warmup) / max(num_epochs - spcl_warmup, 1)
                cur_t = spcl_t_start + (spcl_t_end - spcl_t_start) * 0.5 * (1 + math.cos(math.pi * (1 - progress)))
            losses["vmf_nce_spcl"].set_curriculum_temperature(cur_t)

        # Update kappa_max curriculum (V18)
        if _kappa_cur_enabled:
            progress = min(epoch / max(_kappa_cur_epochs, 1), 1.0)
            _cur_kappa_max = _kappa_cur_start + (_kappa_cur_end - _kappa_cur_start) * progress
            _mt = getattr(model, "model_type", "deterministic")
            if _mt == "vmf_dcf" and hasattr(model, "decoder"):
                if hasattr(model.decoder, "roi_heads"):
                    model.decoder.roi_heads.kappa_max = _cur_kappa_max
                elif hasattr(model.decoder, "kappa_max"):
                    model.decoder.kappa_max = _cur_kappa_max
            elif _mt == "vmf" and hasattr(model, "decoder"):
                model.decoder.kappa_max = _cur_kappa_max
            if epoch % 10 == 1:
                logger.info("Kappa curriculum: kappa_max=%.1f (epoch %d/%d)",
                            _cur_kappa_max, epoch, _kappa_cur_epochs)

        # Update auto-weights from log_sigmas for this epoch
        if _log_sigmas is not None:
            import math as _m
            for _aw_n, _aw_p in _log_sigmas.items():
                _prec = _m.exp(-_aw_p.item())
                loss_weights[_aw_n] = max(0.05, min(10.0, 0.5 * _prec))
            if epoch % 10 == 1:
                logger.info("Auto-weights: %s",
                            {k: f"{loss_weights[k]:.3f}" for k in _log_sigmas})

        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, losses, loss_weights,
            device, kl_scheduler, queue, preprocessor, global_step,
            grad_accum_steps=grad_accum_steps, scaler=scaler,
            lr_scheduler=lr_sched, config_ref=config, vmf_is_log=_vmf_is_log,
            mixco_cfg=_epoch_mixco if _epoch_mixco and _epoch_mixco.get("enabled", False) else None,
            ema=ema, amp_dtype=_amp_dtype, current_epoch=epoch,
            log_sigmas=_log_sigmas, legacy_teacher_model=legacy_teacher_model,
        )
        logger.info("Train: %s", " | ".join(f"{k}={v:.4f}" for k, v in train_metrics.items()))

        if "kappa_std" in train_metrics:
            if train_metrics["kappa_std"] < 0.01:
                logger.warning("kappa collapsed (std < 0.01)")
            kappa_upper = getattr(model.decoder, "kappa_max", None)
            kq90 = train_metrics.get("kappa_q90")
            if kappa_upper and kq90 and kq90 > 0.99 * kappa_upper:
                logger.warning("kappa saturating at upper bound (%.1f / %.1f)", kq90, kappa_upper)

        # --- Training R@1 monitoring (periodic) ---
        _train_r1_interval = config.get("training", {}).get("train_r1_interval", 10)
        if _train_r1_interval > 0 and epoch % _train_r1_interval == 0:
            model.eval()
            if ema is not None:
                ema.apply_shadow(model)
            _tr_preds_buf: List[np.ndarray] = []
            _tr_gts_buf: List[np.ndarray] = []
            _tr_max = 1024
            _tr_n = 0
            with torch.no_grad():
                for _tr_b in train_loader:
                    if _tr_n >= _tr_max:
                        break
                    if isinstance(_tr_b, dict):
                        _tr_fmri = _tr_b["fmri"].to(device, dtype=torch.float32)
                        _tr_gt = _tr_b["retrieval_target"].to(device, dtype=torch.float32)
                        _tr_sid = _tr_b["subject_id"].to(device)
                    else:
                        _tr_fmri = _tr_b[0].to(device, dtype=torch.float32)
                        _tr_gt = _tr_b[1].to(device, dtype=torch.float32)
                        _tr_sid = (
                            _tr_b[2].to(device)
                            if _is_multi_subject and len(_tr_b) >= 3
                            else None
                        )
                    if preprocessor is not None:
                        _tr_gt = preprocessor.transform_torch(_tr_gt)
                    _tr_out = (
                        model(_tr_fmri, subject_ids=_tr_sid)
                        if _tr_sid is not None
                        else model(_tr_fmri)
                    )
                    if isinstance(_tr_out, tuple):
                        _tr_out = _tr_out[0]
                    _tr_preds_buf.append(_tr_out.cpu().numpy())
                    _tr_gts_buf.append(_tr_gt.cpu().numpy())
                    _tr_n += len(_tr_fmri)
            if ema is not None:
                ema.restore(model)
            model.train()
            if _tr_preds_buf:
                _tr_p = np.concatenate(_tr_preds_buf)
                _tr_g = np.concatenate(_tr_gts_buf)
                _tr_ret = _compute_retrieval(_tr_p, _tr_g, ks=(1, 5))
                train_metrics["train_r@1"] = _tr_ret["top1_accuracy"]
                train_metrics["train_r@5"] = _tr_ret["top5_accuracy"]
                logger.info(
                    "Train retrieval: R@1=%.4f  R@5=%.4f  (N=%d)",
                    _tr_ret["top1_accuracy"], _tr_ret["top5_accuracy"],
                    len(_tr_p),
                )

        if ema is not None:
            ema.apply_shadow(model)
        val_metrics, val_preds, val_gts = validate(
            model, val_loader, losses, loss_weights, device, preprocessor, queue,
            vmf_is_log=_vmf_is_log,
            current_epoch=epoch,
            config_ref=config,
            legacy_teacher_model=legacy_teacher_model,
        )
        _epoch_val_extras = val_metrics.pop("_val_extras", {})
        if ema is not None:
            ema.restore(model)
        logger.info("Val:   %s", " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

        # --- MC-Dropout TTA (V9) ---
        _mc_tta_cfg = config.get("evaluation", {})
        _mc_tta_n = _mc_tta_cfg.get("mc_tta_samples", 0)
        if _mc_tta_n > 1 and getattr(model, "model_type", "") in ("vmf", "vmf_dcf", "vmf_triple"):
            if ema is not None:
                ema.apply_shadow(model)
            mc_preds, _, mc_kappas = mc_dropout_tta(
                model, val_loader, device, preprocessor,
                n_samples=_mc_tta_n, vmf_is_log=_vmf_is_log,
            )
            if ema is not None:
                ema.restore(model)
            val_preds = mc_preds
            logger.info("MC-TTA: averaged %d forward passes", _mc_tta_n)
        else:
            mc_kappas = None

        retrieval = _compute_retrieval(val_preds, val_gts, ks=(1, 5, 10))
        val_metrics["r@1_trial"] = retrieval["top1_accuracy"]
        val_metrics["r@5_trial"] = retrieval["top5_accuracy"]
        val_metrics["r@10_trial"] = retrieval["top10_accuracy"]

        # Image-level retrieval: average predictions per unique nsdId.
        # With kappa-weighted averaging (V9), repetitions with higher
        # confidence contribute more to the image-level prediction.
        if _val_nsd_ids is not None:
            unique_ids = np.unique(_val_nsd_ids)
            img_preds = np.zeros((len(unique_ids), val_preds.shape[1]), dtype=np.float32)
            img_gts = np.zeros((len(unique_ids), val_gts.shape[1]), dtype=np.float32)
            _use_kappa_avg = _mc_tta_cfg.get("kappa_weighted_avg", False)
            for i, uid in enumerate(unique_ids):
                mask = _val_nsd_ids == uid
                if _use_kappa_avg and mc_kappas is not None:
                    k_w = mc_kappas[mask]
                    k_w = k_w / (k_w.sum() + 1e-8)
                    img_preds[i] = (val_preds[mask] * k_w[:, None]).sum(axis=0)
                else:
                    img_preds[i] = val_preds[mask].mean(axis=0)
                img_gts[i] = val_gts[mask][0]
            # Re-normalise after averaging
            norms = np.linalg.norm(img_preds, axis=-1, keepdims=True)
            img_preds = img_preds / np.maximum(norms, 1e-8)
            img_retrieval = _compute_retrieval(img_preds, img_gts, ks=(1, 5, 10))
            val_metrics["r@1"] = img_retrieval["top1_accuracy"]
            val_metrics["r@5"] = img_retrieval["top5_accuracy"]
            val_metrics["r@10"] = img_retrieval["top10_accuracy"]
            val_metrics["median_rank"] = img_retrieval["median_rank"]
            val_metrics["mrr"] = img_retrieval["mrr"]
            logger.info(
                "Retrieval: R@1=%.4f  R@5=%.4f  R@10=%.4f  MedR=%.1f  MRR=%.4f  (N=%d img, %d trial)",
                img_retrieval["top1_accuracy"], img_retrieval["top5_accuracy"],
                img_retrieval["top10_accuracy"], img_retrieval["median_rank"],
                img_retrieval["mrr"], len(unique_ids), len(val_preds),
            )
        else:
            val_metrics["r@1"] = retrieval["top1_accuracy"]
            val_metrics["r@5"] = retrieval["top5_accuracy"]
            val_metrics["r@10"] = retrieval["top10_accuracy"]
            val_metrics["median_rank"] = retrieval["median_rank"]
            val_metrics["mrr"] = retrieval["mrr"]
            logger.info(
                "Retrieval: R@1=%.4f  R@5=%.4f  R@10=%.4f  MedR=%.1f  MRR=%.4f  (N=%d)",
                retrieval["top1_accuracy"], retrieval["top5_accuracy"],
                retrieval["top10_accuracy"], retrieval["median_rank"],
                retrieval["mrr"], len(val_preds),
            )

        # --- CSLS-corrected retrieval (V9) ---
        _use_csls = config.get("evaluation", {}).get("use_csls", False)
        if _use_csls:
            _csls_k = config.get("evaluation", {}).get("csls_k", 10)
            _csls_src = img_preds if _val_nsd_ids is not None else val_preds
            _csls_tgt = img_gts if _val_nsd_ids is not None else val_gts
            csls_ret = _compute_retrieval_csls(_csls_src, _csls_tgt, ks=(1, 5, 10), csls_k=_csls_k)
            val_metrics["csls_r@1"] = csls_ret["top1_accuracy"]
            val_metrics["csls_r@5"] = csls_ret["top5_accuracy"]
            val_metrics["csls_r@10"] = csls_ret["top10_accuracy"]
            logger.info(
                "CSLS Retrieval: R@1=%.4f  R@5=%.4f  R@10=%.4f",
                csls_ret["top1_accuracy"], csls_ret["top5_accuracy"],
                csls_ret["top10_accuracy"],
            )

        if "vmf_preds" in _epoch_val_extras and "vmf_kappas" in _epoch_val_extras:
            _vmf_preds_val = _epoch_val_extras["vmf_preds"]
            _vmf_kappas_val = _epoch_val_extras["vmf_kappas"]
            if _val_nsd_ids is not None:
                _u_ids_v = np.unique(_val_nsd_ids)
                _vmf_img = np.zeros((len(_u_ids_v), _vmf_preds_val.shape[1]), dtype=np.float32)
                _vmf_kappa_img = np.zeros(len(_u_ids_v), dtype=np.float32)
                for i, uid in enumerate(_u_ids_v):
                    _m = _val_nsd_ids == uid
                    _vmf_img[i] = _vmf_preds_val[_m].mean(axis=0)
                    _vmf_kappa_img[i] = _vmf_kappas_val[_m[:len(_vmf_kappas_val)]].mean()
                _vmf_preds_eval = _vmf_img / np.maximum(np.linalg.norm(_vmf_img, axis=-1, keepdims=True), 1e-8)
                _vmf_gts_eval = img_gts
            else:
                _vmf_preds_eval = _vmf_preds_val / np.maximum(np.linalg.norm(_vmf_preds_val, axis=-1, keepdims=True), 1e-8)
                _vmf_kappa_img = _vmf_kappas_val
                _vmf_gts_eval = val_gts
            _vmf_ret = _compute_retrieval(_vmf_preds_eval, _vmf_gts_eval, ks=(1, 5, 10))
            _vmf_csls_ret = _compute_retrieval_csls(
                _vmf_preds_eval,
                _vmf_gts_eval,
                ks=(1, 5, 10),
                csls_k=_csls_k if _use_csls else 10,
            )
            val_metrics["vmf_r@1"] = _vmf_ret["top1_accuracy"]
            val_metrics["vmf_r@5"] = _vmf_ret["top5_accuracy"]
            val_metrics["vmf_r@10"] = _vmf_ret["top10_accuracy"]
            val_metrics["vmf_median_rank"] = _vmf_ret["median_rank"]
            val_metrics["vmf_mrr"] = _vmf_ret["mrr"]
            val_metrics["vmf_csls_r@1"] = _vmf_csls_ret["top1_accuracy"]
            val_metrics["vmf_csls_r@5"] = _vmf_csls_ret["top5_accuracy"]
            val_metrics["vmf_csls_r@10"] = _vmf_csls_ret["top10_accuracy"]
            val_metrics["vmf_csls_median_rank"] = _vmf_csls_ret["median_rank"]
            val_metrics["vmf_csls_mrr"] = _vmf_csls_ret["mrr"]
            val_metrics["vmf_kappa_mean_eval"] = float(np.mean(_vmf_kappa_img))
            val_metrics["vmf_kappa_std_eval"] = float(np.std(_vmf_kappa_img))
            logger.info(
                "Aux-vMF Retrieval: R@1=%.4f  CSLS_R@1=%.4f  kappa_mean=%.3f",
                val_metrics["vmf_r@1"],
                val_metrics["vmf_csls_r@1"],
                val_metrics["vmf_kappa_mean_eval"],
            )


        _val_component_mu_img = None
        _val_component_kappa_img = None
        _val_component_logits_img = None
        if (
            "compact_component_mu" in _epoch_val_extras
            and "compact_component_kappa" in _epoch_val_extras
        ):
            _cmu = _epoch_val_extras["compact_component_mu"]
            _ckappa = _epoch_val_extras["compact_component_kappa"]
            _clogits = _epoch_val_extras.get("compact_component_logits")
            _mix_gts = img_gts if _val_nsd_ids is not None else val_gts
            if _val_nsd_ids is not None:
                _cmu, _ckappa, _clogits, _ = _aggregate_component_outputs_by_nsd_id(
                    _cmu, _ckappa, _clogits, _val_nsd_ids,
                )
            _val_component_mu_img = _cmu
            _val_component_kappa_img = _ckappa
            _val_component_logits_img = _clogits
            mix_ret = _compute_mixture_vmf_retrieval(
                _cmu,
                _ckappa,
                _mix_gts,
                component_logits=_clogits,
                ks=(1, 5, 10),
                normalize=True,
            )
            mix_csls_ret = _compute_mixture_vmf_retrieval_csls(
                _cmu,
                _ckappa,
                _mix_gts,
                component_logits=_clogits,
                ks=(1, 5, 10),
                normalize=True,
                csls_k=_csls_k if _use_csls else 10,
            )
            val_metrics["mixture_r@1"] = mix_ret["top1_accuracy"]
            val_metrics["mixture_r@5"] = mix_ret["top5_accuracy"]
            val_metrics["mixture_r@10"] = mix_ret["top10_accuracy"]
            val_metrics["mixture_median_rank"] = mix_ret["median_rank"]
            val_metrics["mixture_mrr"] = mix_ret["mrr"]
            val_metrics["mixture_csls_r@1"] = mix_csls_ret["top1_accuracy"]
            val_metrics["mixture_csls_r@5"] = mix_csls_ret["top5_accuracy"]
            val_metrics["mixture_csls_r@10"] = mix_csls_ret["top10_accuracy"]
            val_metrics["mixture_csls_median_rank"] = mix_csls_ret["median_rank"]
            val_metrics["mixture_csls_mrr"] = mix_csls_ret["mrr"]
            _mix_diag = _compute_mixture_component_diagnostics(
                _cmu,
                _ckappa,
                component_logits=_clogits,
            )
            val_metrics.update(_mix_diag)
            logger.info(
                "Mixture-vMF Retrieval: R@1=%.4f  R@5=%.4f  R@10=%.4f  MedR=%.1f  MRR=%.4f",
                mix_ret["top1_accuracy"], mix_ret["top5_accuracy"], mix_ret["top10_accuracy"],
                mix_ret["median_rank"], mix_ret["mrr"],
            )
            logger.info(
                "Mixture-vMF CSLS: R@1=%.4f  R@5=%.4f  R@10=%.4f  MedR=%.1f  MRR=%.4f",
                mix_csls_ret["top1_accuracy"], mix_csls_ret["top5_accuracy"], mix_csls_ret["top10_accuracy"],
                mix_csls_ret["median_rank"], mix_csls_ret["mrr"],
            )
            logger.info(
                "Mixture-vMF diagnostics: pairwise_cos=%.6f  weight_entropy=%.4f  top_weight=%.4f  kappa_across_std=%.6f",
                val_metrics["component_pairwise_cos_mean"],
                val_metrics["component_weight_entropy_mean"],
                val_metrics["component_top_weight_mean"],
                val_metrics["component_kappa_across_component_std_mean"],
            )
            if (
                val_metrics["component_pairwise_cos_mean"] > 0.999
                and val_metrics["component_weight_entropy_norm_mean"] > 0.99
                and val_metrics["component_kappa_across_component_std_mean"] < 1e-3
            ):
                logger.warning(
                    "Compact mixture appears collapsed: near-identical component directions, uniform weights, and identical kappas"
                )

        # --- V30d per-validation rerank and two-stage diagnostics ---
        _two_stage_cfg = config.get("evaluation", {}).get("two_stage", {})
        _two_stage_enabled = _two_stage_cfg.get("enabled", False)
        if _two_stage_enabled and "rerank_preds" in _epoch_val_extras and "rerank_gts" in _epoch_val_extras:
            from fmri2img.eval.two_stage_retrieval import two_stage_metrics

            _rrp = _epoch_val_extras["rerank_preds"]
            _rrg = _epoch_val_extras["rerank_gts"]
            if _val_nsd_ids is not None:
                _u_ids_rr = np.unique(_val_nsd_ids)
                _rrp_img = np.zeros((len(_u_ids_rr), _rrp.shape[1]), dtype=np.float32)
                _rrg_img = np.zeros((len(_u_ids_rr), _rrg.shape[1]), dtype=np.float32)
                for i, uid in enumerate(_u_ids_rr):
                    _m = _val_nsd_ids == uid
                    _rrp_img[i] = _rrp[_m].mean(axis=0)
                    _rrg_img[i] = _rrg[_m][0]
                _rrp = _rrp_img
                _rrg = _rrg_img
            _rrp = _rrp / np.maximum(np.linalg.norm(_rrp, axis=-1, keepdims=True), 1e-8)

            _compact_eval_preds = img_preds if _val_nsd_ids is not None else val_preds
            _compact_eval_gts = img_gts if _val_nsd_ids is not None else val_gts
            _ts_val = two_stage_metrics(
                _compact_eval_preds,
                _compact_eval_gts,
                _rrp,
                _rrg,
                shortlist_k=100,
                ks=(1, 5, 10),
            )
            _rr_only = _ts_val.get("rich_only", {})
            _rr_oracle = _ts_val.get("oracle_rerank", {})
            _rr_diag = _ts_val.get("rich_diagnostics", {})
            _rr_reranked = _ts_val.get("reranked", {})
            _rr_shortlist = _ts_val.get("shortlist_recall", {})

            val_metrics["rerank_r@1"] = float(_rr_only.get("rich_r@1", 0.0))
            val_metrics["rerank_r@5"] = float(_rr_only.get("rich_r@5", 0.0))
            val_metrics["rerank_r@10"] = float(_rr_only.get("rich_r@10", 0.0))
            val_metrics["oracle_rerank_r@1"] = float(_rr_oracle.get("oracle_r@1", 0.0))
            val_metrics["rerank_separability"] = float(_rr_diag.get("separability", 0.0))
            val_metrics["rerank_inter_pred_cosine"] = float(_rr_diag.get("inter_pred_cosine_mean", 0.0))
            val_metrics["shortlist_r@100"] = float(_rr_shortlist.get("r@100", 0.0))
            val_metrics["reranked_r@1"] = float(_rr_reranked.get("reranked_r@1", 0.0))
            val_metrics["reranked_r@5"] = float(_rr_reranked.get("reranked_r@5", 0.0))
            val_metrics["reranked_r@10"] = float(_rr_reranked.get("reranked_r@10", 0.0))
            _fusion_report_val = _compute_fusion_report(
                _compact_eval_preds,
                _compact_eval_gts,
                _rrp,
                _rrg,
                _fusion_eval_cfg,
            )
            _used_tri_fusion_val = False
            if _tri_fusion_eval_cfg is not None and "legacy_preds" in _epoch_val_extras and "legacy_gts" in _epoch_val_extras:
                _lp = _epoch_val_extras["legacy_preds"]
                _lg = _epoch_val_extras["legacy_gts"]
                if _val_nsd_ids is not None:
                    _u_ids_l = np.unique(_val_nsd_ids)
                    _lp_img = np.zeros((len(_u_ids_l), _lp.shape[1]), dtype=np.float32)
                    _lg_img = np.zeros((len(_u_ids_l), _lg.shape[1]), dtype=np.float32)
                    for i, uid in enumerate(_u_ids_l):
                        _m = _val_nsd_ids == uid
                        _lp_img[i] = _lp[_m].mean(axis=0)
                        _lg_img[i] = _lg[_m][0]
                    _lp, _lg = _lp_img, _lg_img
                _lp = _lp / np.maximum(np.linalg.norm(_lp, axis=-1, keepdims=True), 1e-8)
                _fusion_report_val = _compute_tri_fusion_report(
                    _compact_eval_preds,
                    _compact_eval_gts,
                    _rrp,
                    _rrg,
                    _lp,
                    _lg,
                    _tri_fusion_eval_cfg,
                    compact_component_mu=_val_component_mu_img,
                    compact_component_kappa=_val_component_kappa_img,
                    compact_component_logits=_val_component_logits_img,
                )
                _used_tri_fusion_val = True
            if _used_tri_fusion_val:
                val_metrics.update(_fusion_scalar_metrics(_fusion_report_val, prefix="tri_fused", include_legacy_alias=True))
            else:
                val_metrics.update(_fusion_scalar_metrics(_fusion_report_val))
            logger.info(
                "Rerank val: rerank_R@1=%.4f  oracle_R@1=%.4f  "
                "sep=%.3f  inter_pred=%.4f  shortlist@100=%.4f  reranked_R@1=%.4f",
                val_metrics["rerank_r@1"],
                val_metrics["oracle_rerank_r@1"],
                val_metrics["rerank_separability"],
                val_metrics["rerank_inter_pred_cosine"],
                val_metrics["shortlist_r@100"],
                val_metrics["reranked_r@1"],
            )
            if _fusion_report_val is not None:
                logger.info(
                    "Fused val: fused_R@1=%.4f  fused_R@5=%.4f  fused_R@10=%.4f  "
                    "MedR=%.1f  MRR=%.4f",
                    val_metrics.get("fused_r@1", 0.0),
                    val_metrics.get("fused_r@5", 0.0),
                    val_metrics.get("fused_r@10", 0.0),
                    val_metrics.get("fused_median_rank", 0.0),
                    val_metrics.get("fused_mrr", 0.0),
                )

        metrics_logger.log_epoch(epoch, optimizer.param_groups[0]["lr"], train_metrics, val_metrics)

        # Free large val arrays before checkpoint save to reduce RAM peak.
        # With 329K-D token targets, val_preds alone is ~3 GB RAM.
        del val_preds, val_gts
        try:
            del img_preds, img_gts
        except NameError:
            pass
        try:
            del mc_kappas
        except NameError:
            pass
        gc.collect()

        # --- Checkpointing (early-stop on configurable metric) ---
        val_loss = val_metrics.get("loss", val_metrics.get("mse", float("inf")))
        val_r1 = val_metrics["r@1"]

        _cur_metric = val_metrics.get(_ckpt_metric_name)
        if _cur_metric is None:
            _cur_metric = val_r1
        best_r1 = max(best_r1, val_r1)

        if args.save_checkpoints == "all":
            save_checkpoint(
                output_dir / "checkpoint_last.pt", model, optimizer, lr_sched,
                scaler, epoch, val_loss, config, global_step,
                subject=subject, roi_mask_path=str(roi_mask_path),
                losses=losses, meta=_ckpt_meta, ema=ema,
            )

        if _ckpt_lower_is_better:
            _improved = _cur_metric < best_metric_val - early_stop_min_delta
        else:
            _improved = _cur_metric > best_metric_val + early_stop_min_delta

        if _improved:
            best_metric_val = _cur_metric
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            if ema is not None:
                ema.apply_shadow(model)
            if args.save_checkpoints in ("all", "best"):
                save_checkpoint(
                    output_dir / "checkpoint_best.pt", model, optimizer, lr_sched,
                    scaler, epoch, val_loss, config, global_step,
                    subject=subject, roi_mask_path=str(roi_mask_path),
                    losses=losses, meta=_ckpt_meta, ema=ema,
                )
            if ema is not None:
                ema.restore(model)
            logger.info("New best: %s=%.4f (R@1=%.4f, val_loss=%.4f)",
                        _ckpt_metric_name, _cur_metric, val_r1, val_loss)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, early_stop_patience)
                break

        for _extra_metric_name, _extra_tracker in _extra_best_trackers.items():
            _extra_cur_metric = val_metrics.get(_extra_metric_name)
            if _extra_cur_metric is None:
                continue
            if _extra_tracker["lower_is_better"]:
                _extra_improved = _extra_cur_metric < _extra_tracker["best_value"] - early_stop_min_delta
            else:
                _extra_improved = _extra_cur_metric > _extra_tracker["best_value"] + early_stop_min_delta
            if not _extra_improved:
                continue
            _extra_tracker["best_value"] = float(_extra_cur_metric)
            _extra_tracker["best_epoch"] = int(epoch)
            if ema is not None:
                ema.apply_shadow(model)
            if args.save_checkpoints in ("all", "best"):
                save_checkpoint(
                    output_dir / f"checkpoint_best_{_sanitize_metric_name(_extra_metric_name)}.pt",
                    model, optimizer, lr_sched,
                    scaler, epoch, val_loss, config, global_step,
                    subject=subject, roi_mask_path=str(roi_mask_path),
                    losses=losses, meta=_ckpt_meta, ema=ema,
                )
            if ema is not None:
                ema.restore(model)
            logger.info(
                "New metric-specific best: %s=%.4f (epoch=%d)",
                _extra_metric_name,
                _extra_cur_metric,
                epoch,
            )

        if args.save_checkpoints == "all" and save_frequency > 0 and epoch % save_frequency == 0:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch}.pt", model, optimizer,
                lr_sched, scaler, epoch, val_loss, config, global_step,
                subject=subject, roi_mask_path=str(roi_mask_path),
                losses=losses, meta=_ckpt_meta, ema=ema,
            )

        # Free fragmented GPU memory before next epoch.  Adam's
        # _multi_tensor_adam allocates large temporaries (exp_avg_sq_sqrt)
        # that can OOM on fragmented heaps.  gc.collect() drops Python
        # ref-cycles so the caching allocator can reclaim blocks.
        gc.collect()
        torch.cuda.empty_cache()

    wall_time = time.time() - wall_start
    metrics_logger.write_summary(best_epoch, best_val_loss, wall_time, manifest,
                                 best_r1=best_r1, best_metric=best_metric_val,
                                 checkpoint_metric=_ckpt_metric_name)
    if _extra_best_trackers:
        _summary_path = output_dir / "metrics" / "summary.json"
        _summary = _load_json_if_exists(_summary_path) or {}
        _summary["best_checkpoints"] = {
            name: {
                "best_value": float(tracker["best_value"]),
                "best_epoch": int(tracker["best_epoch"]),
                "lower_is_better": bool(tracker["lower_is_better"]),
                "checkpoint_path": f"checkpoint_best_{_sanitize_metric_name(name)}.pt",
            }
            for name, tracker in _extra_best_trackers.items()
        }
        with open(_summary_path, "w") as _sf:
            json.dump(_summary, _sf, indent=2, default=str)

    # --- Model soup post-training (V10) ---
    _eval_cfg = config.get("evaluation", {})
    if _eval_cfg.get("model_soup", False) and args.save_checkpoints == "all":
        soup_top_k = _eval_cfg.get("soup_top_k", 5)
        logger.info("Running model soup (top_k=%d)...", soup_top_k)
        soup_applied = model_soup(output_dir, model, top_k=soup_top_k, device=device)
        if soup_applied:
            if ema is not None:
                ema.apply_shadow(model)
            val_metrics_soup, soup_preds, soup_gts = validate(
                model, val_loader, losses, loss_weights, device, preprocessor, queue,
                vmf_is_log=_vmf_is_log,
                current_epoch=best_epoch,
                config_ref=config,
                legacy_teacher_model=legacy_teacher_model,
            )
            val_metrics_soup.pop("_val_extras", None)
            if ema is not None:
                ema.restore(model)

            soup_retrieval = _compute_retrieval(soup_preds, soup_gts, ks=(1, 5, 10))
            soup_r1 = soup_retrieval["top1_accuracy"]

            if _val_nsd_ids is not None:
                unique_ids = np.unique(_val_nsd_ids)
                img_preds_s = np.zeros((len(unique_ids), soup_preds.shape[1]), dtype=np.float32)
                img_gts_s = np.zeros((len(unique_ids), soup_gts.shape[1]), dtype=np.float32)
                for i, uid in enumerate(unique_ids):
                    mask = _val_nsd_ids == uid
                    img_preds_s[i] = soup_preds[mask].mean(axis=0)
                    img_gts_s[i] = soup_gts[mask][0]
                norms_s = np.linalg.norm(img_preds_s, axis=-1, keepdims=True)
                img_preds_s = img_preds_s / np.maximum(norms_s, 1e-8)
                soup_img_ret = _compute_retrieval(img_preds_s, img_gts_s, ks=(1, 5, 10))
                soup_r1 = soup_img_ret["top1_accuracy"]

            logger.info("Model soup R@1: %.4f (best single: %.4f)", soup_r1, best_r1)

            if soup_r1 > best_r1:
                logger.info("Model soup improved R@1 by +%.4f — saving as best", soup_r1 - best_r1)
                best_r1 = soup_r1
                save_checkpoint(
                    output_dir / "checkpoint_soup.pt", model, optimizer, lr_sched,
                    scaler, best_epoch, best_val_loss, config, global_step,
                    subject=subject, roi_mask_path=str(roi_mask_path),
                    losses=losses, meta={**_ckpt_meta, "model_soup": True}, ema=ema,
                )
            else:
                logger.info("Model soup did not improve R@1 (%.4f vs %.4f)", soup_r1, best_r1)

    # --- Save validation predictions for diagnostics (V15) ---
    _best_ckpt_path = output_dir / "checkpoint_best.pt"
    _soup_ckpt_path = output_dir / "checkpoint_soup.pt"
    _loaded_ckpt_for_save = False
    if _soup_ckpt_path.exists():
        _ckpt_data = torch.load(_soup_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(_ckpt_data["model_state_dict"])
        _loaded_ckpt_for_save = True
        logger.info("Loaded soup checkpoint for prediction saving")
    elif _best_ckpt_path.exists():
        _ckpt_data = torch.load(_best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(_ckpt_data["model_state_dict"])
        _loaded_ckpt_for_save = True
        logger.info("Loaded best checkpoint (epoch %d) for prediction saving",
                     _ckpt_data.get("epoch", -1))

    if ema is not None and _loaded_ckpt_for_save:
        ema.apply_shadow(model)

    _val_metrics_extra, _save_preds, _save_gts = validate(
        model, val_loader, losses, loss_weights, device, preprocessor, queue,
        vmf_is_log=_vmf_is_log,
        current_epoch=best_epoch,
        config_ref=config,
        legacy_teacher_model=legacy_teacher_model,
    )
    _val_extras = _val_metrics_extra.pop("_val_extras", {})

    _save_model_type = getattr(model, "model_type", "deterministic")
    _save_kappas = None
    if _save_model_type in ("vmf", "vmf_dcf", "vmf_triple"):
        model.eval()
        _kappa_list: List[np.ndarray] = []
        with torch.no_grad():
            for _sb in val_loader:
                if isinstance(_sb, dict):
                    _s_fmri = _sb["fmri"].to(device, dtype=torch.float32)
                    _s_sid = _sb["subject_id"].to(device)
                else:
                    _s_fmri = _sb[0].to(device, dtype=torch.float32)
                    _s_sid = _sb[2].to(device) if _is_multi_subject and len(_sb) >= 3 else None
                _s_out = model(_s_fmri, subject_ids=_s_sid) if _s_sid is not None else model(_s_fmri)
                if isinstance(_s_out, tuple) and len(_s_out) >= 2:
                    _s_aux = _s_out[1]
                    if isinstance(_s_aux, dict):
                        _s_k = _s_aux.get("kappa", _s_aux.get("concentration"))
                    elif torch.is_tensor(_s_aux):
                        _s_k = _s_aux
                    else:
                        _s_k = None
                    if _s_k is not None:
                        if _vmf_is_log:
                            _s_k = _s_k.exp()
                        _kappa_list.append(_s_k.squeeze(-1).detach().cpu().numpy())
        if _kappa_list:
            _save_kappas = np.concatenate(_kappa_list)
    elif _save_model_type == "dense_vmf_hybrid" and "vmf_kappas" in _val_extras:
        _save_kappas = np.asarray(_val_extras["vmf_kappas"], dtype=np.float32)

    if ema is not None and _loaded_ckpt_for_save:
        ema.restore(model)

    if _val_nsd_ids is not None:
        _save_preds, _val_img_ids = _aggregate_rows_by_nsd_id(
            _save_preds, _val_nsd_ids, reduce="mean", normalize=True,
        )
        _save_gts, _ = _aggregate_rows_by_nsd_id(
            _save_gts, _val_nsd_ids, reduce="first", normalize=False,
        )
        if _save_kappas is not None:
            _save_kappas, _ = _aggregate_rows_by_nsd_id(
                _save_kappas, _val_nsd_ids[: len(_save_kappas)], reduce="mean", normalize=False,
            )
    else:
        _val_img_ids = None

    _metrics_save_dir = output_dir / "metrics"
    _metrics_save_dir.mkdir(parents=True, exist_ok=True)
    np.save(_metrics_save_dir / "val_predictions.npy", _save_preds)
    np.save(_metrics_save_dir / "val_ground_truth.npy", _save_gts)
    logger.info("Saved val predictions %s and ground truth %s to %s",
                _save_preds.shape, _save_gts.shape, _metrics_save_dir)
    if _save_kappas is not None:
        np.save(_metrics_save_dir / "val_kappas.npy", _save_kappas)
        logger.info("Saved val kappas %s to %s", _save_kappas.shape, _metrics_save_dir)

    # --- Compact-family separated outputs for vmf_triple / dense_vmf_hybrid ---
    if _save_model_type in ("vmf_triple", "dense_vmf_hybrid"):
        np.save(_metrics_save_dir / "val_predictions_compact.npy", _save_preds)
        np.save(_metrics_save_dir / "val_ground_truth_compact.npy", _save_gts)
        logger.info("Saved compact val predictions %s", _save_preds.shape)
        if _val_img_ids is not None:
            np.save(_metrics_save_dir / "val_nsd_ids.npy", _val_img_ids)
            logger.info("Saved val nsd_ids %s", _val_img_ids.shape)
        if _save_model_type == "dense_vmf_hybrid":
            _vmf_preds_val = _val_extras.get("vmf_preds")
            _vmf_kappas_val = _val_extras.get("vmf_kappas")
            if _vmf_preds_val is not None:
                if _val_nsd_ids is not None:
                    _vmf_preds_val, _ = _aggregate_rows_by_nsd_id(
                        _vmf_preds_val, _val_nsd_ids, reduce="mean", normalize=True,
                    )
                else:
                    _vmf_preds_val = _vmf_preds_val / np.maximum(
                        np.linalg.norm(_vmf_preds_val, axis=-1, keepdims=True), 1e-8,
                    )
            if _vmf_kappas_val is not None and _val_nsd_ids is not None:
                _vmf_kappas_val, _ = _aggregate_rows_by_nsd_id(
                    _vmf_kappas_val,
                    _val_nsd_ids[: len(_vmf_kappas_val)],
                    reduce="mean",
                    normalize=False,
                )
            _save_aux_vmf_arrays(_metrics_save_dir, "val", _vmf_preds_val, _vmf_kappas_val)
            if _vmf_kappas_val is not None and _save_kappas is None:
                np.save(_metrics_save_dir / "val_kappas.npy", _vmf_kappas_val.astype(np.float32))
                logger.info("Saved val kappas alias from aux-vMF %s", _vmf_kappas_val.shape)
        if "rich_preds" in _val_extras and "rich_gts" in _val_extras:
            _rp = _val_extras["rich_preds"]
            _rg = _val_extras["rich_gts"]
            if _val_nsd_ids is not None:
                _rp, _ = _aggregate_rows_by_nsd_id(_rp, _val_nsd_ids, reduce="mean", normalize=False)
                _rg, _ = _aggregate_rows_by_nsd_id(_rg, _val_nsd_ids, reduce="first", normalize=False)
            np.save(_metrics_save_dir / "val_predictions_rich.npy", _rp)
            np.save(_metrics_save_dir / "val_ground_truth_rich.npy", _rg)
            logger.info("Saved rich val predictions %s", _rp.shape)
        if "legacy_preds" in _val_extras and "legacy_gts" in _val_extras:
            _lp = _val_extras["legacy_preds"]
            _lg = _val_extras["legacy_gts"]
            if _val_nsd_ids is not None:
                _lp, _ = _aggregate_rows_by_nsd_id(_lp, _val_nsd_ids, reduce="mean", normalize=False)
                _lg, _ = _aggregate_rows_by_nsd_id(_lg, _val_nsd_ids, reduce="first", normalize=False)
            _lp = _lp / np.maximum(np.linalg.norm(_lp, axis=-1, keepdims=True), 1e-8)
            np.save(_metrics_save_dir / "val_predictions_legacy.npy", _lp)
            np.save(_metrics_save_dir / "val_ground_truth_legacy.npy", _lg)
            logger.info("Saved legacy val predictions %s", _lp.shape)

        _cmu = None
        _ck = None
        _cl = None
        if "compact_component_mu" in _val_extras and "compact_component_kappa" in _val_extras:
            _cmu = _val_extras["compact_component_mu"]
            _ck = _val_extras["compact_component_kappa"]
            _cl = _val_extras.get("compact_component_logits")
            if _val_nsd_ids is not None:
                _cmu, _ck, _cl, _ = _aggregate_component_outputs_by_nsd_id(_cmu, _ck, _cl, _val_nsd_ids)
            _save_compact_component_arrays(_metrics_save_dir, "val", _cmu, _ck, _cl)
            logger.info("Saved compact component val predictions %s", _cmu.shape)
        # --- V30d: Save rerank head predictions ---
        if "rerank_preds" in _val_extras and "rerank_gts" in _val_extras:
            _rrp = _val_extras["rerank_preds"]
            _rrg = _val_extras["rerank_gts"]
            if _val_nsd_ids is not None:
                _rrp, _ = _aggregate_rows_by_nsd_id(_rrp, _val_nsd_ids, reduce="mean", normalize=False)
                _rrg, _ = _aggregate_rows_by_nsd_id(_rrg, _val_nsd_ids, reduce="first", normalize=False)
            # L2-normalise after averaging (rerank head outputs are L2-normed per-sample,
            # but averaging denormalises them)
            _rrp = _rrp / np.maximum(np.linalg.norm(_rrp, axis=-1, keepdims=True), 1e-8)
            np.save(_metrics_save_dir / "val_predictions_rerank.npy", _rrp)
            np.save(_metrics_save_dir / "val_ground_truth_rerank.npy", _rrg)
            logger.info("Saved rerank val predictions %s", _rrp.shape)

            # Two-stage retrieval using rerank head instead of regression head
            from fmri2img.eval.two_stage_retrieval import two_stage_metrics
            _ts = two_stage_metrics(_save_preds, _save_gts, _rrp, _rrg,
                                     shortlist_k=100, ks=(1, 5, 10))
            _ts_path = _metrics_save_dir / "val_two_stage_rerank.json"
            with open(_ts_path, "w") as _jf:
                json.dump(_ts, _jf, indent=2, default=str)
            logger.info(
                "Val two-stage (rerank head): reranked_R@1=%.1f%%  compact_R@1=%.1f%%  gain=%.1f pp",
                _ts["reranked"].get("reranked_r@1", 0) * 100,
                _ts["compact_raw"].get("compact_r@1", 0) * 100,
                _ts.get("rerank_gain_over_compact_raw", 0) * 100,
            )
            _fusion_report = _compute_fusion_report(
                _save_preds,
                _save_gts,
                _rrp,
                _rrg,
                _fusion_eval_cfg,
            )
            if (
                _tri_fusion_eval_cfg is not None
                and "legacy_preds" in _val_extras
                and "legacy_gts" in _val_extras
            ):
                _lp = _val_extras["legacy_preds"]
                _lg = _val_extras["legacy_gts"]
                if _val_nsd_ids is not None:
                    _u_ids_l = np.unique(_val_nsd_ids)
                    _lp_img = np.zeros((len(_u_ids_l), _lp.shape[1]), dtype=np.float32)
                    _lg_img = np.zeros((len(_u_ids_l), _lg.shape[1]), dtype=np.float32)
                    for i, uid in enumerate(_u_ids_l):
                        _m = _val_nsd_ids == uid
                        _lp_img[i] = _lp[_m].mean(axis=0)
                        _lg_img[i] = _lg[_m][0]
                    _lp, _lg = _lp_img, _lg_img
                _lp = _lp / np.maximum(np.linalg.norm(_lp, axis=-1, keepdims=True), 1e-8)
                _fusion_report = _compute_tri_fusion_report(
                    _save_preds,
                    _save_gts,
                    _rrp,
                    _rrg,
                    _lp,
                    _lg,
                    _tri_fusion_eval_cfg,
                    compact_component_mu=_cmu,
                    compact_component_kappa=_ck,
                    compact_component_logits=_cl,
                )
            if _fusion_report is not None:
                _fusion_path = _metrics_save_dir / "val_fused_metrics.json"
                with open(_fusion_path, "w") as _jf:
                    json.dump(_fusion_report, _jf, indent=2, default=str)
                _fused = _fusion_report.get("fused", {})
                logger.info(
                    "Saved val fused metrics: fused_R@1=%.1f%%  fused_R@5=%.1f%%  "
                    "fused_R@10=%.1f%%  MedR=%.1f  MRR=%.4f",
                    _fused.get("fused_r@1", 0.0) * 100,
                    _fused.get("fused_r@5", 0.0) * 100,
                    _fused.get("fused_r@10", 0.0) * 100,
                    _fused.get("fused_median_rank", 0.0),
                    _fused.get("fused_mrr", 0.0),
                )

        if "nsd_ids" in _val_extras:
            _nids = _val_extras["nsd_ids"]
            if _val_nsd_ids is not None:
                _nids = np.unique(_val_nsd_ids)
            np.save(_metrics_save_dir / "val_nsd_ids.npy", _nids)
            logger.info("Saved val nsd_ids %s", _nids.shape)

    # --- Shared1000 benchmark evaluation ---
    _do_s1000 = config.get("evaluation", {}).get("eval_shared1000", True)
    if _do_s1000:
        _run_post_training_shared1000_eval(
            output_dir=output_dir,
            model=model,
            config=config,
            subject=subject,
            device=device,
            embeddings_df=embeddings_df,
            preprocessor=preprocessor,
            vmf_is_log=_vmf_is_log,
            zscore_stats_path=_zscore_stats_path,
            is_multi_subject=_is_multi_subject,
            token_cache=_token_cache,
            rerank_cache=_rerank_cache,
            retrieval_projector=_retrieval_projector,
            legacy_teacher_model=legacy_teacher_model,
        )
    _merge_summary_metrics(_metrics_save_dir)

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(
        "Best checkpoint metric: %s=%.4f (epoch %d) | best compact R@1=%.4f | best_val_loss=%.4f",
        _ckpt_metric_name,
        best_metric_val,
        best_epoch,
        best_r1,
        best_val_loss,
    )
    logger.info("Wall time: %.1f min", wall_time / 60)
    logger.info("Outputs: %s", output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
