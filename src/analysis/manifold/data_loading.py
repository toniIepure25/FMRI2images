"""Artifact discovery and loading for manifold analysis."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.manifold.config import ManifoldAnalysisConfig
from analysis.manifold.utils import safe_normalize

logger = logging.getLogger(__name__)


@dataclass
class AnalysisData:
    """Container for all loaded analysis artifacts."""

    z_pred: np.ndarray  # (N, D) predicted embeddings
    z_target: np.ndarray  # (N, D) target CLIP embeddings
    gallery: Optional[np.ndarray] = None  # (G, D) gallery embeddings
    target_ids: Optional[np.ndarray] = None  # (N,) nsdId per sample
    target_gallery_indices: Optional[np.ndarray] = None  # (N,) index into gallery
    kappa: Optional[np.ndarray] = None  # (N,) vMF concentration
    metadata: Optional[pd.DataFrame] = None  # trial-level metadata
    per_trial_z_pred: Optional[np.ndarray] = None  # (T, D) per-trial preds
    per_trial_nsd_ids: Optional[np.ndarray] = None  # (T,) per-trial nsdIds
    text_embeddings: Optional[np.ndarray] = None  # (C, D) text probe embeds
    text_concepts: Optional[List[str]] = None
    text_probe_metadata: Dict[str, Any] = field(default_factory=dict)
    semantic_axes: Optional[np.ndarray] = None  # (A, D) axis directions
    semantic_axis_names: Optional[List[str]] = None
    experiment_id: Optional[str] = None
    subject: Optional[str] = None
    embedding_dim: int = 768
    availability: Dict[str, bool] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.z_pred.shape[0]

    @property
    def n_gallery(self) -> int:
        return self.gallery.shape[0] if self.gallery is not None else self.n_samples


def _try_load_npy(path: Optional[str | Path], name: str) -> Optional[np.ndarray]:
    """Load a .npy file if it exists, else return None."""
    if path is None:
        return None
    p = Path(path)
    if p.exists():
        arr = np.load(p)
        logger.info("Loaded %s: shape=%s dtype=%s", name, arr.shape, arr.dtype)
        return arr
    logger.info("Not found: %s (%s)", name, p)
    return None


def _try_load_text_embeddings(
    path: Optional[str | Path],
) -> tuple[Optional[np.ndarray], Optional[List[str]], Dict[str, Any]]:
    """Load text probe embeddings from .npy or .npz cache."""
    if path is None:
        return None, None, {}
    p = Path(path)
    if not p.exists():
        logger.info("Not found: text_embeddings (%s)", p)
        return None, None, {"path": str(p), "status": "missing"}
    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=True)
        if "concept_embeddings" in data:
            key = "concept_embeddings"
        elif "text_embeddings" in data:
            key = "text_embeddings"
        elif "embeddings" in data:
            key = "embeddings"
        else:
            raise KeyError(
                f"Text embedding cache {p} must contain concept_embeddings, "
                "text_embeddings, or embeddings."
            )
        embeddings = data[key].astype(np.float32)
        concepts = data["concepts"].astype(str).tolist() if "concepts" in data else None
        meta: Dict[str, Any] = {"path": str(p), "embedding_key": key}
        for mk in [
            "model_name",
            "pretrained_name",
            "embedding_dim",
            "normalized",
            "created_at",
            "notes",
        ]:
            if mk in data:
                val = data[mk]
                if getattr(val, "shape", ()) == ():
                    val = val.item()
                meta[mk] = val
        if "prompts" in data:
            meta["n_prompts"] = int(len(data["prompts"]))
        if concepts is not None:
            meta["n_concepts"] = int(len(concepts))
        logger.info(
            "Loaded text_embeddings cache from %s: key=%s shape=%s",
            p,
            key,
            embeddings.shape,
        )
        return embeddings, concepts, meta
    arr = np.load(p).astype(np.float32)
    logger.info("Loaded text_embeddings: shape=%s", arr.shape)
    return arr, None, {"path": str(p), "embedding_key": "npy_array"}


def _try_load_parquet_embeddings(
    path: Optional[str | Path],
    name: str,
    embedding_col: str = "embedding",
    id_col: str = "nsdId",
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load embeddings from a parquet file.

    Tries common column names: ``embedding``, ``fused``, ``final``, or
    any column starting with ``emb_``.
    """
    if path is None:
        return None, None
    p = Path(path)
    if not p.exists():
        logger.info("Not found: %s (%s)", name, p)
        return None, None
    df = pd.read_parquet(p)
    ids = df[id_col].values if id_col in df.columns else None

    for col in [embedding_col, "fused", "final", "clip_embedding"]:
        if col in df.columns:
            emb = np.stack(df[col].values).astype(np.float32)
            logger.info("Loaded %s from col '%s': shape=%s", name, col, emb.shape)
            return emb, ids

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if emb_cols:
        emb = df[emb_cols].values.astype(np.float32)
        logger.info("Loaded %s from %d emb_* cols: shape=%s", name, len(emb_cols), emb.shape)
        return emb, ids

    logger.warning("No embedding column found in %s. Columns: %s", p, list(df.columns))
    return None, ids


def discover_artifacts(experiment_dir: str | Path) -> Dict[str, Any]:
    """Scan an experiment result directory and return best-guess artifact paths."""
    d = Path(experiment_dir)
    found: Dict[str, Any] = {"experiment_dir": str(d)}

    metrics_dir = d / "metrics"
    if not metrics_dir.is_dir():
        for sub in d.iterdir():
            if sub.is_dir() and (sub / "metrics").is_dir():
                metrics_dir = sub / "metrics"
                break

    mappings = {
        "z_pred_path": [
            "val_predictions.npy",
            "val_predictions_compact.npy",
            "shared1000_predictions_mctta16.npy",
            "shared1000_predictions.npy",
        ],
        "z_target_path": [
            "val_ground_truth.npy",
            "val_ground_truth_compact.npy",
            "shared1000_ground_truth.npy",
        ],
        "kappa_path": [
            "val_kappas.npy",
            "val_kappas_vmf.npy",
            "shared1000_kappas.npy",
        ],
        "target_ids_path": [
            "val_nsd_ids.npy",
            "shared1000_nsd_ids.npy",
        ],
    }

    for key, candidates in mappings.items():
        for cand in candidates:
            p = metrics_dir / cand
            if p.exists():
                found[key] = str(p)
                break

    return found


def _build_gallery_index(
    target_ids: np.ndarray,
    gallery_ids: np.ndarray,
) -> Optional[np.ndarray]:
    """Map each target id to its position in the gallery."""
    gallery_map = {int(gid): i for i, gid in enumerate(gallery_ids)}
    indices = np.array([gallery_map.get(int(tid), -1) for tid in target_ids])
    if (indices == -1).any():
        n_miss = int((indices == -1).sum())
        logger.warning("%d target ids not found in gallery", n_miss)
    return indices


def _require_gallery_index(target_ids: np.ndarray, gallery_ids: np.ndarray) -> np.ndarray:
    """Map target ids to gallery positions and fail if any id is absent."""
    indices = _build_gallery_index(target_ids, gallery_ids)
    if indices is None or (indices == -1).any():
        missing = int((indices == -1).sum()) if indices is not None else len(target_ids)
        raise ValueError(f"{missing} target_ids were not found in the gallery ids.")
    return indices


def load_analysis_data(config: ManifoldAnalysisConfig) -> AnalysisData:
    """Load all artifacts specified in *config* and return an AnalysisData."""
    art = config.artifacts
    analysis = config.analysis
    do_norm = analysis.get("normalize_embeddings", True)
    gallery_mode = str(analysis.get("gallery_mode", "full"))
    allow_diagonal_fallback = bool(analysis.get("allow_diagonal_fallback", False))
    valid_gallery_modes = {"full", "target_ids", "target_embeddings"}
    if gallery_mode not in valid_gallery_modes:
        raise ValueError(
            "analysis.gallery_mode must be one of: "
            f"{', '.join(sorted(valid_gallery_modes))}"
        )
    logger.info(
        "Gallery mode: %s (allow_diagonal_fallback=%s)",
        gallery_mode,
        allow_diagonal_fallback,
    )

    z_pred = _try_load_npy(art.get("z_pred_path"), "z_pred")
    z_target = _try_load_npy(art.get("z_target_path"), "z_target")

    if z_pred is None or z_target is None:
        raise FileNotFoundError(
            "z_pred and z_target are required. Set artifacts.z_pred_path and "
            "artifacts.z_target_path in the config."
        )

    if do_norm:
        z_pred = safe_normalize(z_pred)
        z_target = safe_normalize(z_target)

    if z_pred.shape != z_target.shape:
        raise ValueError(
            f"z_pred shape {z_pred.shape} must match z_target shape {z_target.shape}. "
            "Do not mix token-space predictions with compact CLIP targets."
        )

    kappa = _try_load_npy(art.get("kappa_path"), "kappa")
    if kappa is not None:
        kappa = kappa.ravel()

    target_ids = _try_load_npy(art.get("target_ids_path"), "target_ids")

    gallery = None
    gallery_ids = None
    gallery_path = art.get("gallery_embeddings_path")
    token_h5 = art.get("gallery_token_h5_path")

    if gallery_mode == "target_embeddings":
        gallery = z_target
        gallery_ids = target_ids
        target_gallery_indices = np.arange(z_target.shape[0], dtype=np.int64)
        logger.info(
            "Using z_target as target_embeddings gallery: shape=%s; target_indices=arange(N)",
            gallery.shape,
        )
    else:
        target_gallery_indices = None

    if gallery_mode != "target_embeddings" and token_h5 and gallery_path:
        gp = Path(gallery_path)
        th = Path(str(token_h5))
        if gp.suffix == ".parquet" and gp.exists() and th.exists():
            from analysis.manifold.token_gallery import build_flat_token_gallery_from_parquet

            mmap_tc = bool(art.get("gallery_token_mmap", False))
            id_col = str(art.get("gallery_id_column", "nsdId"))
            logger.info(
                "Building token-space gallery from parquet=%s token_h5=%s",
                gp,
                th,
            )
            gallery, gallery_ids = build_flat_token_gallery_from_parquet(
                parquet_path=gp,
                token_h5_path=th,
                id_column=id_col,
                mmap_token_cache=mmap_tc,
            )
        elif not th.exists():
            logger.warning("gallery_token_h5_path not found: %s", th)
        elif gp.suffix != ".parquet" or not gp.exists():
            logger.warning(
                "gallery_embeddings_path must be a parquet file when "
                "gallery_token_h5_path is set; got %s",
                gallery_path,
            )

    if gallery_mode != "target_embeddings" and gallery is None and gallery_path is not None:
        p = Path(gallery_path)
        if p.suffix == ".npy" and p.exists():
            gallery = np.load(p).astype(np.float32)
        elif p.suffix == ".parquet" and p.exists():
            id_col = str(art.get("gallery_id_column", "nsdId"))
            gallery, gallery_ids = _try_load_parquet_embeddings(p, "gallery", id_col=id_col)
    if gallery is None:
        logger.info("No separate gallery provided; using z_target as gallery")
        gallery = z_target
        gallery_ids = target_ids
        target_gallery_indices = np.arange(z_target.shape[0], dtype=np.int64)

    if gallery_mode == "target_ids":
        if target_ids is None:
            raise ValueError("analysis.gallery_mode='target_ids' requires artifacts.target_ids_path")
        if gallery_ids is None:
            raise ValueError("analysis.gallery_mode='target_ids' requires gallery ids, e.g. nsdId in clip.parquet")
        target_order = _require_gallery_index(target_ids, gallery_ids)
        gallery = gallery[target_order]
        gallery_ids = target_ids.copy()
        target_gallery_indices = np.arange(len(target_ids), dtype=np.int64)
        logger.info(
            "Filtered gallery to target_ids order: shape=%s; target_indices=arange(N)",
            gallery.shape,
        )

    if gallery_mode == "target_embeddings":
        if z_target is None:
            raise ValueError("analysis.gallery_mode='target_embeddings' requires z_target")
        if gallery.shape[0] != z_pred.shape[0]:
            raise ValueError(
                "analysis.gallery_mode='target_embeddings' requires gallery size "
                f"to equal N; got gallery={gallery.shape[0]}, N={z_pred.shape[0]}"
            )

    if do_norm and gallery is not None:
        gallery = safe_normalize(gallery)

    if gallery is not None and gallery.shape[1] != z_pred.shape[1]:
        raise ValueError(
            f"Gallery dimension {gallery.shape[1]} does not match prediction "
            f"dimension {z_pred.shape[1]}. Refusing cross-space manifold analysis."
        )

    if target_gallery_indices is None and target_ids is not None and gallery_ids is not None:
        target_gallery_indices = _build_gallery_index(target_ids, gallery_ids)
        if target_gallery_indices is not None and (target_gallery_indices == -1).any():
            if not allow_diagonal_fallback:
                missing = int((target_gallery_indices == -1).sum())
                raise ValueError(f"{missing} target_ids were not found in the gallery ids.")
            logger.warning("Proceeding despite missing gallery ids because allow_diagonal_fallback=true")

    if gallery_mode == "target_ids" and gallery.shape[0] != z_pred.shape[0]:
        raise ValueError(
            "analysis.gallery_mode='target_ids' must produce an N-row gallery; "
            f"got gallery={gallery.shape[0]}, N={z_pred.shape[0]}"
        )

    if (
        target_gallery_indices is None
        and gallery_mode == "full"
        and not allow_diagonal_fallback
        and z_pred.shape[0] != gallery.shape[0]
    ):
        raise ValueError(
            "Target gallery indices are missing while N != gallery size. "
            "Provide target_ids_path with gallery ids, set gallery_mode='target_ids', "
            "or explicitly set analysis.allow_diagonal_fallback=true for diagnostics."
        )

    if gallery_mode == "full" and target_gallery_indices is not None:
        logger.info("Using full gallery mapping: gallery shape=%s", gallery.shape)

    if target_gallery_indices is not None:
        is_arange = (
            len(target_gallery_indices) == z_pred.shape[0]
            and np.array_equal(target_gallery_indices, np.arange(z_pred.shape[0]))
        )
        logger.info(
            "Final gallery shape=%s; target_indices_arange=%s; target_indices_shape=%s",
            gallery.shape,
            bool(is_arange),
            target_gallery_indices.shape,
        )

    metadata = None
    meta_path = art.get("metadata_path")
    if meta_path is not None:
        mp = Path(meta_path)
        if mp.exists():
            metadata = pd.read_parquet(mp)
            logger.info("Loaded metadata: %d rows, cols=%s", len(metadata), list(metadata.columns))

    per_trial_z_pred = _try_load_npy(art.get("per_trial_pred_path"), "per_trial_z_pred")
    per_trial_nsd_ids = _try_load_npy(art.get("per_trial_nsd_ids_path"), "per_trial_nsd_ids")
    if per_trial_z_pred is not None and do_norm:
        per_trial_z_pred = safe_normalize(per_trial_z_pred)

    text_embeddings, text_concepts, text_probe_metadata = _try_load_text_embeddings(
        art.get("text_embeddings_path")
    )
    if text_embeddings is not None and do_norm:
        text_embeddings = safe_normalize(text_embeddings)
    semantic_axes_arr = _try_load_npy(art.get("semantic_axes_path"), "semantic_axes")

    if text_embeddings is not None and text_embeddings.shape[1] != z_pred.shape[1]:
        raise ValueError(
            f"Text embedding dimension {text_embeddings.shape[1]} does not match "
            f"prediction dimension {z_pred.shape[1]}."
        )
    if semantic_axes_arr is not None and semantic_axes_arr.shape[1] != z_pred.shape[1]:
        raise ValueError(
            f"Semantic axis dimension {semantic_axes_arr.shape[1]} does not match "
            f"prediction dimension {z_pred.shape[1]}."
        )

    avail = {
        "has_gallery": gallery is not None and gallery is not z_target,
        "has_token_gallery": bool(token_h5 and gallery is not None and gallery is not z_target),
        "gallery_mode": gallery_mode,
        "allow_diagonal_fallback": allow_diagonal_fallback,
        "has_kappa": kappa is not None,
        "has_target_ids": target_ids is not None,
        "has_gallery_indices": target_gallery_indices is not None,
        "has_metadata": metadata is not None,
        "has_per_trial": per_trial_z_pred is not None,
        "has_text_embeddings": text_embeddings is not None,
        "has_semantic_axes": semantic_axes_arr is not None,
    }
    logger.info("Data availability: %s", avail)

    return AnalysisData(
        z_pred=z_pred,
        z_target=z_target,
        gallery=gallery,
        target_ids=target_ids,
        target_gallery_indices=target_gallery_indices,
        kappa=kappa,
        metadata=metadata,
        per_trial_z_pred=per_trial_z_pred,
        per_trial_nsd_ids=per_trial_nsd_ids,
        text_embeddings=text_embeddings,
        text_concepts=text_concepts,
        text_probe_metadata=text_probe_metadata,
        semantic_axes=semantic_axes_arr,
        experiment_id=art.get("experiment_id"),
        subject=art.get("subject"),
        embedding_dim=z_pred.shape[1],
        availability=avail,
    )
