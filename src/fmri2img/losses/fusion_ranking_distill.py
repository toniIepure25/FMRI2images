"""Ranking-topology distillation from a frozen fusion recipe over target spaces.

This loss is designed for the all-8 fusion-distillation student wave (V45).
Instead of running brittle subject-mismatched frozen fMRI teachers, it applies
the *validated frozen fusion recipe* to the available image target spaces
inside the batch:

- compact teacher space: retrieval targets
- legacy teacher space: rich targets
- optional rerank teacher space: PCA rerank targets

The student compact head is then trained to match the fused teacher
distribution on a small shortlist-local candidate set.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _positive_rank(scores: torch.Tensor, positive_index: int) -> int:
    """Return the 1-based descending rank of the positive index."""
    order = torch.argsort(scores, descending=True)
    pos = torch.where(order == positive_index)[0]
    return int(pos[0].item()) + 1 if len(pos) > 0 else scores.numel()


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1)


def _cosine_scores(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _l2_normalize(a) @ _l2_normalize(b).T


def _csls_from_scores(scores: torch.Tensor, k: int = 10) -> torch.Tensor:
    scores = scores.float()
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got {tuple(scores.shape)}")
    k_eff = min(int(k), scores.shape[0] - 1, scores.shape[1] - 1)
    if k_eff < 1:
        return scores
    top_k_pred = torch.topk(scores, k=k_eff, dim=1).values.mean(dim=1, keepdim=True)
    top_k_gal = torch.topk(scores.T, k=k_eff, dim=1).values.mean(dim=1).unsqueeze(0)
    return 2.0 * scores - top_k_pred - top_k_gal


def _normalize_shortlist_scores(scores: torch.Tensor, mode: str) -> torch.Tensor:
    scores = scores.float()
    if mode == "none":
        return scores
    if mode == "zscore":
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True, unbiased=False)
        return (scores - mean) / torch.clamp(std, min=1e-8)
    if mode == "minmax":
        s_min = scores.min(dim=1, keepdim=True).values
        s_max = scores.max(dim=1, keepdim=True).values
        return (scores - s_min) / torch.clamp(s_max - s_min, min=1e-8)
    if mode == "stdscale":
        std = scores.std(dim=1, keepdim=True, unbiased=False)
        return scores / torch.clamp(std, min=1e-8)
    raise ValueError(f"Unknown normalization mode: {mode}")


class FusionRankingDistillLoss(nn.Module):
    """Distil a frozen fusion recipe into the student's compact ranking.

    The teacher is a fixed score-combination recipe derived from the frozen best
    system. We apply that recipe to the batch-local image target spaces, then
    match the student compact logits to the fused teacher distribution.
    """

    VALID_SCORE_MODES = {"raw_cosine", "csls"}
    VALID_FAMILIES = {"weighted", "normalized_weighted"}
    VALID_NORMALIZATION = {"none", "zscore", "minmax", "stdscale"}

    def __init__(
        self,
        *,
        compact_score: str = "csls",
        legacy_score: str = "csls",
        rerank_score: str = "raw_cosine",
        family: str = "normalized_weighted",
        normalization: str = "zscore",
        alpha: float = 0.3,
        beta: float = 0.0,
        gamma: float = 0.7,
        csls_k: int = 10,
        compact_topk: int = 16,
        teacher_topk: int = 32,
        teacher_temperature: float = 0.07,
        student_temperature: float = 0.07,
        teacher_rank_gate: int = 20,
        topk_overlap_k: int = 10,
        symmetric: bool = False,
    ) -> None:
        super().__init__()
        if compact_score not in self.VALID_SCORE_MODES:
            raise ValueError(f"Unknown compact_score mode: {compact_score}")
        if legacy_score not in self.VALID_SCORE_MODES:
            raise ValueError(f"Unknown legacy_score mode: {legacy_score}")
        if rerank_score not in self.VALID_SCORE_MODES:
            raise ValueError(f"Unknown rerank_score mode: {rerank_score}")
        if family not in self.VALID_FAMILIES:
            raise ValueError(f"Unknown fusion family: {family}")
        if normalization not in self.VALID_NORMALIZATION:
            raise ValueError(f"Unknown normalization mode: {normalization}")
        self.compact_score = str(compact_score)
        self.legacy_score = str(legacy_score)
        self.rerank_score = str(rerank_score)
        self.family = str(family)
        self.normalization = str(normalization)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.csls_k = int(csls_k)
        self.compact_topk = int(compact_topk)
        self.teacher_topk = int(teacher_topk)
        self.teacher_temperature = float(teacher_temperature)
        self.student_temperature = float(student_temperature)
        self.teacher_rank_gate = int(teacher_rank_gate)
        self.topk_overlap_k = int(topk_overlap_k)
        self.symmetric = bool(symmetric)
        if self.alpha < 0 or self.beta < 0 or self.gamma < 0:
            raise ValueError("Fusion weights alpha/beta/gamma must be non-negative")
        if (self.alpha + self.beta + self.gamma) <= 0:
            raise ValueError("Fusion weights alpha/beta/gamma must sum to a positive value")

    def _apply_score_mode(self, scores: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "raw_cosine":
            return scores
        if mode == "csls":
            return _csls_from_scores(scores, k=self.csls_k)
        raise ValueError(f"Unknown score mode: {mode}")

    def _combine_teacher_scores(
        self,
        compact_scores: torch.Tensor,
        legacy_scores: Optional[torch.Tensor],
        rerank_scores: Optional[torch.Tensor],
    ) -> torch.Tensor:
        compact_term = compact_scores
        legacy_term = legacy_scores
        rerank_term = rerank_scores
        if self.family == "normalized_weighted":
            compact_term = _normalize_shortlist_scores(compact_term, self.normalization)
            if legacy_term is not None:
                legacy_term = _normalize_shortlist_scores(legacy_term, self.normalization)
            if rerank_term is not None:
                rerank_term = _normalize_shortlist_scores(rerank_term, self.normalization)
        fused = self.alpha * compact_term
        if self.beta > 0.0:
            if rerank_term is None:
                raise ValueError("Fusion recipe requires rerank scores, but rerank targets were not provided")
            fused = fused + self.beta * rerank_term
        if self.gamma > 0.0:
            if legacy_term is None:
                raise ValueError("Fusion recipe requires legacy scores, but rich targets were not provided")
            fused = fused + self.gamma * legacy_term
        return fused

    def forward(
        self,
        compact_pred: torch.Tensor,
        retrieval_target: torch.Tensor,
        *,
        legacy_target: Optional[torch.Tensor] = None,
        rerank_target: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        if compact_pred.ndim != 2 or retrieval_target.ndim != 2:
            raise ValueError("compact_pred and retrieval_target must be 2D tensors")
        batch_size = compact_pred.shape[0]
        if retrieval_target.shape[0] != batch_size:
            raise ValueError("compact_pred and retrieval_target must share batch size")
        if legacy_target is not None and legacy_target.ndim != 2:
            raise ValueError("legacy_target must be 2D")
        if rerank_target is not None and rerank_target.ndim != 2:
            raise ValueError("rerank_target must be 2D")
        if legacy_target is not None and legacy_target.shape[0] != batch_size:
            raise ValueError("legacy_target must share batch size with compact_pred")
        if rerank_target is not None and rerank_target.shape[0] != batch_size:
            raise ValueError("rerank_target must share batch size with compact_pred")

        if batch_size <= 1:
            zero = compact_pred.new_zeros(())
            stats = {
                "gate_frac": 0.0,
                "candidate_size_mean": 1.0,
                "teacher_pos_rank_mean": 1.0,
                "teacher_pos_rank_median": 1.0,
                "student_pos_rank_mean": 1.0,
                "student_pos_rank_median": 1.0,
                "teacher_student_topk_overlap": 1.0,
                "teacher_top1_agreement": 1.0,
                "teacher_coverage": 1.0,
            }
            return (zero, stats) if return_stats else zero

        compact_pred_f = compact_pred.float()
        retrieval_target_f = retrieval_target.detach().float()
        compact_teacher_raw = _cosine_scores(retrieval_target_f, retrieval_target_f)
        compact_teacher_scores = self._apply_score_mode(compact_teacher_raw, self.compact_score)

        legacy_teacher_scores = None
        if legacy_target is not None:
            legacy_teacher_raw = _cosine_scores(legacy_target.detach().float(), legacy_target.detach().float())
            legacy_teacher_scores = self._apply_score_mode(legacy_teacher_raw, self.legacy_score)

        rerank_teacher_scores = None
        if rerank_target is not None:
            rerank_teacher_raw = _cosine_scores(rerank_target.detach().float(), rerank_target.detach().float())
            rerank_teacher_scores = self._apply_score_mode(rerank_teacher_raw, self.rerank_score)

        teacher_scores_det = self._combine_teacher_scores(
            compact_teacher_scores,
            legacy_teacher_scores,
            rerank_teacher_scores,
        )
        student_scores_det = _cosine_scores(compact_pred_f.detach(), retrieval_target_f)
        student_scores = _cosine_scores(compact_pred_f, retrieval_target_f)

        teacher_positive_ranks = []
        student_positive_ranks = []
        topk_overlaps = []
        top1_agreements = []
        candidate_sizes = []
        valid_losses = []
        n_gated = 0

        compact_top_k = min(self.compact_topk, batch_size)
        teacher_top_k = min(self.teacher_topk, batch_size)
        overlap_k = min(self.topk_overlap_k, batch_size)

        for row_idx in range(batch_size):
            student_top = torch.topk(student_scores_det[row_idx], k=compact_top_k).indices.tolist()
            teacher_top = torch.topk(teacher_scores_det[row_idx], k=teacher_top_k).indices.tolist()

            teacher_rank = _positive_rank(teacher_scores_det[row_idx], row_idx)
            student_rank = _positive_rank(student_scores_det[row_idx], row_idx)
            teacher_positive_ranks.append(teacher_rank)
            student_positive_ranks.append(student_rank)

            if overlap_k > 0:
                teacher_overlap_top = set(torch.topk(teacher_scores_det[row_idx], k=overlap_k).indices.tolist())
                student_overlap_top = set(torch.topk(student_scores_det[row_idx], k=overlap_k).indices.tolist())
                topk_overlaps.append(float(len(teacher_overlap_top.intersection(student_overlap_top))) / float(overlap_k))
                top1_agreements.append(
                    1.0
                    if int(torch.argmax(teacher_scores_det[row_idx]).item()) == int(torch.argmax(student_scores_det[row_idx]).item())
                    else 0.0
                )
            else:
                topk_overlaps.append(1.0)
                top1_agreements.append(1.0)

            candidate_indices = []
            seen = set()
            for idx in [row_idx, *student_top, *teacher_top]:
                idx_int = int(idx)
                if idx_int not in seen:
                    seen.add(idx_int)
                    candidate_indices.append(idx_int)

            cand_idx = torch.tensor(candidate_indices, device=compact_pred.device, dtype=torch.long)
            gt_local_index = candidate_indices.index(row_idx)
            candidate_sizes.append(len(candidate_indices))

            if self.teacher_rank_gate > 0 and teacher_rank > self.teacher_rank_gate:
                continue

            student_logits = student_scores[row_idx].index_select(0, cand_idx) / self.student_temperature
            teacher_logits = teacher_scores_det[row_idx].index_select(0, cand_idx) / self.teacher_temperature
            if student_logits.shape != teacher_logits.shape:
                raise AssertionError(
                    "fusion_ranking_distill student/teacher logits shape mismatch: "
                    f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
                )
            teacher_dist = F.softmax(teacher_logits, dim=-1)
            student_log_dist = F.log_softmax(student_logits, dim=-1)
            loss = F.kl_div(student_log_dist, teacher_dist, reduction="sum")

            if self.symmetric:
                student_dist = student_log_dist.exp()
                teacher_log_dist = F.log_softmax(teacher_logits, dim=-1)
                reverse_kl = torch.sum(student_dist * (student_log_dist - teacher_log_dist), dim=-1)
                loss = 0.5 * (loss + reverse_kl)

            valid_losses.append(loss)
            n_gated += 1

        if valid_losses:
            loss = torch.stack(valid_losses).mean()
        else:
            loss = compact_pred.sum() * 0.0

        teacher_coverage = 1.0
        if self.gamma > 0.0 and legacy_target is None:
            teacher_coverage = 0.0
        if self.beta > 0.0 and rerank_target is None:
            teacher_coverage = 0.0

        stats = {
            "gate_frac": float(n_gated / max(batch_size, 1)),
            "candidate_size_mean": float(torch.tensor(candidate_sizes, dtype=torch.float32).mean().item()),
            "teacher_pos_rank_mean": float(torch.tensor(teacher_positive_ranks, dtype=torch.float32).mean().item()),
            "teacher_pos_rank_median": float(torch.tensor(teacher_positive_ranks, dtype=torch.float32).median().item()),
            "student_pos_rank_mean": float(torch.tensor(student_positive_ranks, dtype=torch.float32).mean().item()),
            "student_pos_rank_median": float(torch.tensor(student_positive_ranks, dtype=torch.float32).median().item()),
            "teacher_student_topk_overlap": float(torch.tensor(topk_overlaps, dtype=torch.float32).mean().item()),
            "teacher_top1_agreement": float(torch.tensor(top1_agreements, dtype=torch.float32).mean().item()),
            "teacher_coverage": float(teacher_coverage),
        }
        return (loss, stats) if return_stats else loss
