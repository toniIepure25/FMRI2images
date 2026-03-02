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
import os
import platform
import random
import subprocess
import sys
import time
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
    VonMisesFisherNLLLoss,
    KappaSPCLVMFNCELoss,
    DeltaSPCLVMFNCELoss,
    MultiTaskVMFNCELoss,
    kappa_regularizer,
)
from fmri2img.training.kl_schedule import KLScheduler
from fmri2img.losses.mixco import mixco_augment, mixco_nce_loss
from fmri2img.losses.softclip import SoftCLIPLoss, VMFSoftCLIPLoss
from fmri2img.losses.hierarchical_clip_loss import HierarchicalCLIPLoss
from fmri2img.losses.cka_loss import CKALoss
from fmri2img.eval.embedding_eval import compute_retrieval_metrics as _compute_retrieval

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
                      best_r1: float = 0.0) -> None:
        summary = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_r@1": best_r1,
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
    """

    def __init__(
        self,
        features_path: Path,
        index_df: pd.DataFrame,
        embeddings_df: pd.DataFrame,
        average_repetitions: bool = False,
    ):
        self.features = np.load(features_path, mmap_mode=None)  # (N, V) float32
        self.index_df = index_df.reset_index(drop=True)
        self.embeddings_df = embeddings_df

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
            "PreextractedNSDDataset: %d trials, %d voxels (%.2f GB in RAM)",
            *self.features.shape, self.features.nbytes / 1e9,
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fmri = self.features[idx]

        nsdId = self.index_df.iloc[idx]["nsdId"]
        emb_idx = self.embedding_lookup.get(nsdId)
        if emb_idx is None:
            raise KeyError(
                f"nsdId={nsdId} not found in CLIP cache "
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
        losses["mse"] = nn.MSELoss()
        logger.info("MSE loss enabled")

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
        losses["vmf_nll"] = VonMisesFisherNLLLoss(
            dim=c.get("dim", 768), kappa_is_log=vmf_kappa_is_log,
        )
        logger.info("vMF-NLL loss enabled (kappa_is_log=%s)", vmf_kappa_is_log)

    if loss_cfg.get("vmf_nce", {}).get("enabled", False):
        c = loss_cfg["vmf_nce"]
        use_q = c.get("use_queue", False) and queue is not None
        _learnable_tau = c.get("learnable_temperature", False)
        losses["vmf_nce"] = VonMisesFisherNCELoss(
            tau=c.get("tau", 0.07), use_queue=use_q, kappa_is_log=vmf_kappa_is_log,
            learnable_temperature=_learnable_tau,
        )
        logger.info("vMF-NCE loss enabled (queue=%s, tau=%s, learnable_tau=%s)",
                     use_q, c.get("tau", 0.07), _learnable_tau)

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
            )
            logger.info("vMF-NCE-SPCL loss enabled (curriculum_t=%.1f)", c.get("initial_curriculum_t", 100.0))

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
        logger.info("CKA loss enabled (weight=%.3f)", c.get("weight", 0.5))

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
) -> Tuple[Dict[str, float], int]:
    """Train for one epoch with gradient accumulation and optional AMP."""
    model.train()
    epoch_metrics: Dict[str, list] = {}
    use_amp = scaler is not None and scaler.is_enabled()
    model_type = getattr(model, "model_type", "deterministic")

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc="Training")
    for step_in_epoch, batch in enumerate(pbar):
        hier_targets = None
        if len(batch) == 4:
            fmri, gt_embedding, subject_ids, hier_targets = batch
            subject_ids = subject_ids.to(device)
        elif len(batch) == 3:
            fmri, gt_embedding, subject_ids = batch
            subject_ids = subject_ids.to(device)
        else:
            fmri, gt_embedding = batch
            subject_ids = None
        fmri = fmri.to(device, dtype=torch.float32)
        gt_embedding = gt_embedding.to(device, dtype=torch.float32)
        if hier_targets is not None:
            hier_targets = {k: v.to(device, dtype=torch.float32)
                           for k, v in hier_targets.items()}

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

        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(fmri, subject_ids=subject_ids) if subject_ids is not None else model(fmri)
            if isinstance(output, tuple):
                pred, aux = output
            else:
                pred, aux = output, None

            total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            batch_metrics: Dict[str, float] = {}

            is_gaussian = model_type == "gaussian" and aux is not None
            is_vmf = model_type in ("vmf", "vmf_dcf") and aux is not None

            # --- Deterministic losses ---
            if "mse" in losses and not is_gaussian and not is_vmf:
                l = losses["mse"](pred, gt_embedding)
                total_loss = total_loss + loss_weights.get("mse", 1.0) * l
                batch_metrics["mse"] = l.item()

            if "infonce" in losses and not is_gaussian and not is_vmf:
                l = losses["infonce"](pred, gt_embedding, queue=queue)
                total_loss = total_loss + loss_weights.get("infonce", 1.0) * l
                batch_metrics["infonce"] = l.item()

            # --- SoftCLIP knowledge distillation (works for all model types) ---
            if "softclip" in losses:
                if isinstance(losses["softclip"], VMFSoftCLIPLoss) and is_vmf:
                    l = losses["softclip"](pred, aux, gt_embedding, queue=queue)
                else:
                    l = losses["softclip"](pred, gt_embedding, queue=queue)
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
                l = losses["vmf_nll"](pred, aux, gt_embedding)
                total_loss = total_loss + loss_weights.get("vmf_nll", 1.0) * l
                batch_metrics["vmf_nll"] = l.item()

            if "vmf_nce" in losses and is_vmf:
                l = losses["vmf_nce"](pred, aux, gt_embedding, queue=queue)
                total_loss = total_loss + loss_weights.get("vmf_nce", 1.0) * l
                batch_metrics["vmf_nce"] = l.item()

            # --- vMF-NCE-SPCL (N4) or Delta-SPCL ---
            if "vmf_nce_spcl" in losses and is_vmf:
                spcl_kwargs = dict(queue=queue)
                if isinstance(losses["vmf_nce_spcl"], DeltaSPCLVMFNCELoss):
                    dcf_ex = getattr(model, "_last_dcf_extras", {})
                    spcl_kwargs["delta"] = dcf_ex.get("delta")
                l = losses["vmf_nce_spcl"](pred, aux, gt_embedding, **spcl_kwargs)
                total_loss = total_loss + loss_weights.get("vmf_nce_spcl", 1.0) * l
                batch_metrics["vmf_nce_spcl"] = l.item()

            # --- MultiTask vMF-NCE (N3/N4) ---
            if "vmf_nce_multitask" in losses and is_vmf:
                dcf_extras = getattr(model, "_last_dcf_extras", {})
                mt_total, mt_fused, mt_aux = losses["vmf_nce_multitask"](
                    pred, aux, gt_embedding,
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
                cka_loss = losses["cka"](pred, gt_embedding)
                total_loss = total_loss + loss_weights.get("cka", 0.5) * cka_loss
                batch_metrics["cka"] = cka_loss.item()

            # --- Single queue enqueue (after all contrastive losses read the queue) ---
            if queue is not None:
                queue.enqueue(gt_embedding.detach())

            # --- Kappa regularizer ---
            kappa_reg_cfg = config_ref.get("loss", {}).get("kappa_reg", {}) if config_ref else {}
            if kappa_reg_cfg.get("enabled", False) and is_vmf and aux is not None:
                kappa_vals = aux.squeeze(-1) if not vmf_is_log else aux.exp().squeeze(-1)
                kr = kappa_regularizer(kappa_vals, kappa_reg_cfg.get("lambda_kappa", 0.01))
                total_loss = total_loss + kr
                batch_metrics["kappa_reg"] = kr.item()

            # --- Kappa statistics ---
            if is_vmf and aux is not None:
                with torch.no_grad():
                    kv = aux.squeeze(-1) if not vmf_is_log else aux.exp().squeeze(-1)
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
                log_kappa_for_kl = aux if vmf_is_log else torch.log(aux.clamp(min=1e-8))
                kl_raw = compute_vmf_kl(pred, log_kappa_for_kl, pred.size(-1))
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
                mc_loss = mixco_nce_loss(
                    pred_mix, gt_mix, soft_labels,
                    temperature=mixco_cfg.get("temperature", 0.006),
                )
                total_loss = total_loss + mixco_cfg.get("weight", 1.0) * mc_loss
                batch_metrics["mixco"] = mc_loss.item()

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


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    losses: Dict[str, nn.Module],
    loss_weights: Dict[str, float],
    device: str,
    preprocessor: Optional[EmbeddingPreprocessor],
    queue: Optional[nn.Module],
    vmf_is_log: bool = True,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Validate model and collect embeddings for retrieval evaluation."""
    model.eval()
    epoch_metrics: Dict[str, list] = {}
    model_type = getattr(model, "model_type", "deterministic")
    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                fmri, gt_embedding, subject_ids = batch
                subject_ids = subject_ids.to(device)
            else:
                fmri, gt_embedding = batch
                subject_ids = None
            fmri = fmri.to(device, dtype=torch.float32)
            gt_embedding = gt_embedding.to(device, dtype=torch.float32)

            if preprocessor is not None:
                gt_embedding_np = gt_embedding.cpu().numpy()
                gt_embedding_proc = preprocessor.transform(gt_embedding_np)
                gt_embedding = torch.from_numpy(gt_embedding_proc).float().to(device)

            output = model(fmri, subject_ids=subject_ids) if subject_ids is not None else model(fmri)
            if isinstance(output, tuple):
                pred, aux = output
            else:
                pred, aux = output, None

            all_preds.append(pred.detach().cpu().numpy())
            all_gts.append(gt_embedding.detach().cpu().numpy())

            total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            bm: Dict[str, float] = {}
            is_gaussian = model_type == "gaussian" and aux is not None
            is_vmf = model_type in ("vmf", "vmf_dcf") and aux is not None

            if "mse" in losses and not is_gaussian and not is_vmf:
                l = losses["mse"](pred, gt_embedding)
                total_loss = total_loss + loss_weights.get("mse", 1.0) * l
                bm["mse"] = l.item()

            if "infonce" in losses and not is_gaussian and not is_vmf:
                l = losses["infonce"](pred, gt_embedding, queue=None)
                total_loss = total_loss + loss_weights.get("infonce", 1.0) * l
                bm["infonce"] = l.item()

            if "softclip" in losses:
                if isinstance(losses["softclip"], VMFSoftCLIPLoss) and is_vmf:
                    l = losses["softclip"](pred, aux, gt_embedding, queue=None)
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
                l = losses["vmf_nll"](pred, aux, gt_embedding)
                total_loss = total_loss + loss_weights.get("vmf_nll", 1.0) * l
                bm["vmf_nll"] = l.item()

            if "vmf_nce" in losses and is_vmf:
                l = losses["vmf_nce"](pred, aux, gt_embedding, queue=None)
                total_loss = total_loss + loss_weights.get("vmf_nce", 1.0) * l
                bm["vmf_nce"] = l.item()

            if "vmf_nce_spcl" in losses and is_vmf:
                spcl_kwargs = dict(queue=None)
                if isinstance(losses["vmf_nce_spcl"], DeltaSPCLVMFNCELoss):
                    dcf_ex = getattr(model, "_last_dcf_extras", {})
                    spcl_kwargs["delta"] = dcf_ex.get("delta")
                l = losses["vmf_nce_spcl"](pred, aux, gt_embedding, **spcl_kwargs)
                total_loss = total_loss + loss_weights.get("vmf_nce_spcl", 1.0) * l
                bm["vmf_nce_spcl"] = l.item()

            if "vmf_nce_multitask" in losses and is_vmf:
                dcf_extras = getattr(model, "_last_dcf_extras", {})
                mt_total, mt_fused, mt_aux = losses["vmf_nce_multitask"](
                    pred, aux, gt_embedding,
                    per_roi_mus=dcf_extras.get("per_roi_mus"),
                    per_roi_kappas=dcf_extras.get("per_roi_kappas"),
                    queue=None,
                )
                total_loss = total_loss + loss_weights.get("vmf_nce_multitask", 1.0) * mt_total
                bm["mt_fused"] = mt_fused.item()
                bm["mt_aux"] = mt_aux.item()

            if is_gaussian:
                bm["kl"] = compute_kl_divergence(pred, aux).item()
            if is_vmf:
                log_kappa_for_kl = aux if vmf_is_log else torch.log(aux.clamp(min=1e-8))
                bm["kl"] = compute_vmf_kl(pred, log_kappa_for_kl, pred.size(-1)).item()

            bm["loss"] = total_loss.item()
            for k, v in bm.items():
                epoch_metrics.setdefault(k, []).append(v)

    loss_metrics = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}
    return loss_metrics, np.concatenate(all_preds), np.concatenate(all_gts)


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
    torch.save(payload, path)


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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified training script")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--subject", type=str, default=None, help="Override subject (e.g. subj02)")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="Skip saving checkpoint files (saves disk space)")
    args = parser.parse_args()

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

    # --- Dataset (prefer pre-extracted features for speed) ---
    cache_root = os.environ.get("CACHE_ROOT", "cache")
    preextracted_path = Path(cache_root) / "preextracted" / f"subject={subject}" / "fmri_features.npy"
    roi_mask_path = resolve_roi_mask_path(subject)

    _encoder_type_check = config.get("model", {}).get("encoder", {}).get("encoder_type", "mlp")
    _multi_subjects = config.get("data", {}).get("subjects", [])

    if _encoder_type_check == "multi_subject_roi_transformer" and len(_multi_subjects) > 1:
        from fmri2img.data.multi_subject_dataset import MultiSubjectPreextractedDataset

        _emb_col_override = config.get("data", {}).get("embedding_column")
        _exclude_s1000 = config["data"].get("exclude_shared1000", True)
        _split_img = config["data"].get("split_by_image", True)
        _val_ratio = config["data"].get("val_split", 0.10)
        _data_seed = config["data"].get("seed", 42)

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
        )
        logger.info("Multi-subject dataset: %d subjects, %d total trials",
                     full_dataset.n_subjects, len(full_dataset))

    elif preextracted_path.exists():
        avg_reps = config["data"].get("average_repetitions", False)
        logger.info("Using pre-extracted features: %s (avg_reps=%s)", preextracted_path, avg_reps)
        full_dataset = PreextractedNSDDataset(
            preextracted_path, index_df, embeddings_df,
            average_repetitions=avg_reps,
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
    sample_fmri = sample[0]
    sample_emb = sample[1]
    fmri_dim = sample_fmri.shape[0]
    embedding_dim = sample_emb.shape[0]
    logger.info("Dimensions: fMRI=%d, Embedding=%d", fmri_dim, embedding_dim)

    # --- Model ---
    model_config = config["model"]
    model_config["encoder"]["input_dim"] = fmri_dim
    model_config["decoder"]["output_dim"] = embedding_dim

    _roi_indices = None
    encoder_type = model_config.get("encoder", {}).get("encoder_type", "mlp")
    _is_multi_subject = encoder_type == "multi_subject_roi_transformer"
    if encoder_type == "roi_transformer":
        from fmri2img.data.roi_utils import build_roi_index
        roi_names = list(model_config["encoder"].get("roi_dims", {}).keys())
        if roi_names:
            actual_dims, _roi_indices = build_roi_index(subject, roi_names)
            model_config["encoder"]["roi_dims"] = dict(actual_dims)
            logger.info("ROI dims overridden from NSD masks (total=%d)", sum(actual_dims.values()))
    elif _is_multi_subject:
        from fmri2img.data.roi_utils import build_roi_index
        roi_names = list(model_config["encoder"].get("roi_dims", {}).keys())
        subjects_list = config.get("data", {}).get("subjects", [subject])
        if roi_names and subjects_list:
            subject_roi_dims = {}
            subject_roi_indices = {}
            for subj in subjects_list:
                dims, indices = build_roi_index(subj, roi_names)
                subject_roi_dims[subj] = dict(dims)
                subject_roi_indices[subj] = indices
                logger.info("ROI dims for %s: total=%d", subj, sum(dims.values()))
            model_config["encoder"]["subject_roi_dims"] = subject_roi_dims
            _roi_indices = subject_roi_indices

    model = create_model(model_config, roi_indices=_roi_indices).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{n_params:,}")

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
    optimizer = torch.optim.AdamW(
        model.parameters(),
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

    if hasattr(full_dataset, "train_indices") and full_dataset.train_indices is not None:
        train_indices = full_dataset.train_indices.tolist()
        val_indices = full_dataset.val_indices.tolist()
        logger.info(
            "Using dataset's built-in split: %d train, %d val trials",
            len(train_indices), len(val_indices),
        )
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
    elif split_by_image and hasattr(full_dataset, "index_df"):
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
        unique_images = np.unique(pool_nsd_ids)
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
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
    else:
        train_split = config["data"]["train_split"]
        val_split = config["data"]["val_split"]
        n_total = len(full_dataset)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)

        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(data_seed)).tolist()
        train_dataset = Subset(full_dataset, indices[:n_train])
        val_dataset = Subset(full_dataset, indices[n_train : n_train + n_val])

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
            # Handles both 3-element (fmri, emb, subj_id) and
            # 4-element (fmri, emb, subj_id, hier_targets) tuples
            has_hier = len(batch[0]) == 4
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger.info("Grad accum: %d (effective batch %d) | AMP: %s",
                grad_accum_steps, batch_size * grad_accum_steps, use_amp)

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
    save_frequency = config["training"].get("save_frequency", 0)
    patience_counter = 0

    _vmf_is_log = getattr(model, "vmf_output_is_log", True)

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

    _ckpt_meta = {
        "normalize_fmri": normalize_fmri,
        "zscore_mode": zscore_mode,
        "preprocessing_enabled": config.get("preprocessing", {}).get("enabled", False),
        "zscore_stats_path": _zscore_stats_path,
        "average_repetitions": config["data"].get("average_repetitions", False),
        "split_by_image": config["data"].get("split_by_image", False),
    }

    wall_start = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        logger.info("\nEpoch %d/%d | lr=%.2e", epoch, num_epochs, optimizer.param_groups[0]["lr"])

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

        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, losses, loss_weights,
            device, kl_scheduler, queue, preprocessor, global_step,
            grad_accum_steps=grad_accum_steps, scaler=scaler,
            lr_scheduler=lr_sched, config_ref=config, vmf_is_log=_vmf_is_log,
            mixco_cfg=_epoch_mixco if _epoch_mixco and _epoch_mixco.get("enabled", False) else None,
            ema=ema,
        )
        logger.info("Train: %s", " | ".join(f"{k}={v:.4f}" for k, v in train_metrics.items()))

        if "kappa_std" in train_metrics:
            if train_metrics["kappa_std"] < 0.01:
                logger.warning("kappa collapsed (std < 0.01)")
            kappa_upper = getattr(model.decoder, "kappa_max", None)
            kq90 = train_metrics.get("kappa_q90")
            if kappa_upper and kq90 and kq90 > 0.99 * kappa_upper:
                logger.warning("kappa saturating at upper bound (%.1f / %.1f)", kq90, kappa_upper)

        if ema is not None:
            ema.apply_shadow(model)
        val_metrics, val_preds, val_gts = validate(
            model, val_loader, losses, loss_weights, device, preprocessor, queue,
            vmf_is_log=_vmf_is_log,
        )
        if ema is not None:
            ema.restore(model)
        logger.info("Val:   %s", " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

        retrieval = _compute_retrieval(val_preds, val_gts, ks=(1, 5, 10))
        val_metrics["r@1_trial"] = retrieval["top1_accuracy"]
        val_metrics["r@5_trial"] = retrieval["top5_accuracy"]
        val_metrics["r@10_trial"] = retrieval["top10_accuracy"]

        # Image-level retrieval: average predictions per unique nsdId to remove
        # repetition-induced ties and produce metrics comparable to MindEye.
        if _val_nsd_ids is not None:
            unique_ids = np.unique(_val_nsd_ids)
            img_preds = np.zeros((len(unique_ids), val_preds.shape[1]), dtype=np.float32)
            img_gts = np.zeros((len(unique_ids), val_gts.shape[1]), dtype=np.float32)
            for i, uid in enumerate(unique_ids):
                mask = _val_nsd_ids == uid
                img_preds[i] = val_preds[mask].mean(axis=0)
                img_gts[i] = val_gts[mask][0]
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

        metrics_logger.log_epoch(epoch, optimizer.param_groups[0]["lr"], train_metrics, val_metrics)

        # --- Checkpointing (early-stop on R@1, higher is better) ---
        val_loss = val_metrics.get("loss", val_metrics.get("mse", float("inf")))
        val_r1 = val_metrics["r@1"]

        if not args.no_checkpoints:
            save_checkpoint(
                output_dir / "checkpoint_last.pt", model, optimizer, lr_sched,
                scaler, epoch, val_loss, config, global_step,
                subject=subject, roi_mask_path=str(roi_mask_path),
                losses=losses, meta=_ckpt_meta, ema=ema,
            )

        if val_r1 > best_r1:
            best_r1 = val_r1
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            if ema is not None:
                ema.apply_shadow(model)
            if not args.no_checkpoints:
                save_checkpoint(
                    output_dir / "checkpoint_best.pt", model, optimizer, lr_sched,
                    scaler, epoch, val_loss, config, global_step,
                    subject=subject, roi_mask_path=str(roi_mask_path),
                    losses=losses, meta=_ckpt_meta, ema=ema,
                )
            if ema is not None:
                ema.restore(model)
            logger.info("New best: R@1=%.4f (val_loss=%.4f)", val_r1, val_loss)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, early_stop_patience)
                break

        if not args.no_checkpoints and save_frequency > 0 and epoch % save_frequency == 0:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch}.pt", model, optimizer,
                lr_sched, scaler, epoch, val_loss, config, global_step,
                subject=subject, roi_mask_path=str(roi_mask_path),
                losses=losses, meta=_ckpt_meta, ema=ema,
            )

    wall_time = time.time() - wall_start
    metrics_logger.write_summary(best_epoch, best_val_loss, wall_time, manifest, best_r1=best_r1)

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("Best R@1: %.4f | val_loss: %.4f (epoch %d)", best_r1, best_val_loss, best_epoch)
    logger.info("Wall time: %.1f min", wall_time / 60)
    logger.info("Outputs: %s", output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
