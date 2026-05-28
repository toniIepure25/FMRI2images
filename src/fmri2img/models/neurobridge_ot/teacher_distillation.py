"""
Teacher-Student Distillation — Module F
=========================================

Teacher registry and distillation losses for transferring knowledge from
existing subject-specific experts (V61a, V62a, V66a) into the shared
NeuroBridge-OT model.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class TeacherRegistry:
    """Registry mapping (subject, model_name) to teacher prediction artifacts.

    Validates shape and nsdId alignment before training. Refuses to run
    if teacher artifacts do not match trial/image IDs.
    """

    def __init__(self):
        self._teachers: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        subject: str,
        model_name: str,
        predictions_path: Optional[Path] = None,
        scores_path: Optional[Path] = None,
        nsd_ids_path: Optional[Path] = None,
        embedding_dim: int = 768,
    ) -> None:
        """Register a teacher model's artifacts.

        Args:
            subject: Subject identifier (e.g., 'subj01').
            model_name: Teacher model name (e.g., 'V61a', 'V62a', 'V66a').
            predictions_path: Path to predicted embeddings .npy file.
            scores_path: Path to score matrices .npy file.
            nsd_ids_path: Path to nsdId alignment .npy file.
            embedding_dim: Expected embedding dimensionality.
        """
        key = f"{subject}/{model_name}"
        self._teachers[key] = {
            "subject": subject,
            "model_name": model_name,
            "predictions_path": predictions_path,
            "scores_path": scores_path,
            "nsd_ids_path": nsd_ids_path,
            "embedding_dim": embedding_dim,
            "loaded": False,
            "predictions": None,
            "scores": None,
            "nsd_ids": None,
        }
        logger.info("Registered teacher: %s", key)

    def load(self, subject: str, model_name: str) -> Dict[str, Any]:
        """Load teacher artifacts into memory and validate shapes.

        Raises:
            FileNotFoundError: If prediction files don't exist.
            ValueError: If shapes are inconsistent.
        """
        key = f"{subject}/{model_name}"
        if key not in self._teachers:
            raise KeyError(f"Teacher not registered: {key}")

        entry = self._teachers[key]
        if entry["loaded"]:
            return entry

        if entry["predictions_path"] is not None:
            path = Path(entry["predictions_path"])
            if not path.exists():
                raise FileNotFoundError(f"Teacher predictions not found: {path}")
            preds = np.load(path)
            if preds.ndim != 2 or preds.shape[1] != entry["embedding_dim"]:
                raise ValueError(
                    f"Teacher predictions shape {preds.shape} does not match "
                    f"expected (N, {entry['embedding_dim']})"
                )
            entry["predictions"] = torch.from_numpy(preds).float()

        if entry["scores_path"] is not None:
            path = Path(entry["scores_path"])
            if path.exists():
                entry["scores"] = torch.from_numpy(np.load(path)).float()

        if entry["nsd_ids_path"] is not None:
            path = Path(entry["nsd_ids_path"])
            if path.exists():
                entry["nsd_ids"] = np.load(path)

        entry["loaded"] = True
        logger.info("Loaded teacher %s: preds=%s", key,
                    entry["predictions"].shape if entry["predictions"] is not None else "None")
        return entry

    def get_predictions(
        self, subject: str, model_name: str, nsd_ids: np.ndarray
    ) -> Optional[Tensor]:
        """Get teacher predictions aligned to given nsdIds.

        Args:
            subject: Subject identifier.
            model_name: Teacher model name.
            nsd_ids: Array of nsdIds for the current batch/dataset.

        Returns:
            Aligned predictions tensor or None if unavailable.
        """
        key = f"{subject}/{model_name}"
        if key not in self._teachers:
            return None

        entry = self._teachers[key]
        if not entry["loaded"]:
            try:
                self.load(subject, model_name)
            except (FileNotFoundError, ValueError) as e:
                logger.warning("Cannot load teacher %s: %s", key, e)
                return None

        if entry["predictions"] is None:
            return None

        if entry["nsd_ids"] is not None:
            # Build index map for alignment
            teacher_ids = entry["nsd_ids"]
            id_to_idx = {int(nid): i for i, nid in enumerate(teacher_ids)}
            indices = [id_to_idx.get(int(nid), -1) for nid in nsd_ids]
            valid_mask = [i >= 0 for i in indices]
            if not all(valid_mask):
                logger.warning(
                    "Teacher %s: %d/%d nsdIds not found in teacher predictions",
                    key, sum(1 for v in valid_mask if not v), len(nsd_ids),
                )
            valid_indices = [i for i in indices if i >= 0]
            if len(valid_indices) == 0:
                return None
            return entry["predictions"][valid_indices]

        # Assume direct index alignment
        n = min(len(nsd_ids), entry["predictions"].shape[0])
        return entry["predictions"][:n]

    @property
    def registered_teachers(self) -> List[str]:
        return list(self._teachers.keys())


class EmbeddingDistillLoss(nn.Module):
    """Distillation loss in embedding space (cosine or MSE).

    Args:
        loss_type: 'cosine' or 'mse'.
        temperature: Softening temperature for cosine loss.
    """

    def __init__(self, loss_type: str = "cosine", temperature: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature

    def forward(
        self, student_emb: Tensor, teacher_emb: Tensor, weights: Optional[Tensor] = None
    ) -> Tensor:
        """Compute embedding distillation loss.

        Args:
            student_emb: Student predictions, shape (B, D).
            teacher_emb: Teacher predictions, shape (B, D).
            weights: Optional per-sample weights, shape (B,).

        Returns:
            Scalar loss.
        """
        if self.loss_type == "cosine":
            cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=-1)
            loss = 1.0 - cos_sim  # cosine distance
        elif self.loss_type == "mse":
            loss = F.mse_loss(student_emb, teacher_emb, reduction="none").mean(dim=-1)
        else:
            raise ValueError(f"Unknown distill loss type: {self.loss_type}")

        if weights is not None:
            loss = loss * weights

        return loss.mean()


class ScoreDistillLoss(nn.Module):
    """Score-distribution distillation over a gallery.

    Minimizes KL divergence between teacher and student retrieval
    score distributions (softmax over gallery similarities).

    Args:
        temperature: Temperature for softmax smoothing.
    """

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_emb: Tensor,
        teacher_scores: Tensor,
        gallery: Tensor,
    ) -> Tensor:
        """Compute score-distribution KL divergence.

        Args:
            student_emb: Student embeddings, shape (B, D).
            teacher_scores: Teacher similarity scores over gallery, shape (B, G).
            gallery: Gallery embeddings, shape (G, D).

        Returns:
            Scalar KL divergence loss.
        """
        # Student scores
        student_scores = torch.mm(
            F.normalize(student_emb, dim=-1), F.normalize(gallery, dim=-1).T
        )  # (B, G)

        # Softmax distributions
        teacher_probs = F.softmax(teacher_scores / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_scores / self.temperature, dim=-1)

        # KL(teacher || student)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        return kl * (self.temperature ** 2)


class RankAwareDistillLoss(nn.Module):
    """Rank-aware distillation emphasizing top-ranked items.

    Weights the distillation loss by the teacher's rank position,
    focusing learning on getting the top retrievals correct.

    Args:
        top_k: Number of top items to focus on.
        temperature: Score temperature.
    """

    def __init__(self, top_k: int = 10, temperature: float = 1.0):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature

    def forward(
        self,
        student_emb: Tensor,
        teacher_scores: Tensor,
        gallery: Tensor,
    ) -> Tensor:
        """Compute rank-weighted distillation loss.

        Args:
            student_emb: Student embeddings, shape (B, D).
            teacher_scores: Teacher scores over gallery, shape (B, G).
            gallery: Gallery embeddings, shape (G, D).

        Returns:
            Scalar loss focused on top-K teacher predictions.
        """
        B, G = teacher_scores.shape

        # Get teacher top-K indices
        _, top_indices = teacher_scores.topk(self.top_k, dim=-1)  # (B, top_k)

        # Student similarities to gallery
        student_norm = F.normalize(student_emb, dim=-1)
        gallery_norm = F.normalize(gallery, dim=-1)
        student_scores = torch.mm(student_norm, gallery_norm.T)  # (B, G)

        # Gather top-K student scores
        student_topk = student_scores.gather(1, top_indices)  # (B, top_k)
        teacher_topk = teacher_scores.gather(1, top_indices)  # (B, top_k)

        # Rank weights: exponentially decaying importance
        rank_weights = torch.exp(
            -torch.arange(self.top_k, dtype=torch.float, device=student_emb.device) * 0.3
        )  # (top_k,)

        # Weighted MSE on top-K scores
        diff = (student_topk - teacher_topk) ** 2
        loss = (diff * rank_weights.unsqueeze(0)).mean()
        return loss
