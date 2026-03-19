"""Shortlist-local distillation from combined rerank and legacy teachers."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _positive_rank(scores: torch.Tensor, positive_index: int) -> int:
    """Return 1-based descending rank of the positive index within a score row."""
    order = torch.argsort(scores, descending=True)
    pos = torch.where(order == positive_index)[0]
    return int(pos[0].item()) + 1 if len(pos) > 0 else scores.numel()


class TriTeacherDistillLoss(nn.Module):
    """Distil combined rerank and legacy teacher preferences into the compact head."""

    VALID_MODES = {"rerank_only", "legacy_only", "average_logits", "weighted_logits"}

    def __init__(
        self,
        compact_topk: int = 16,
        teacher_topk: int = 16,
        teacher_tau: float = 0.07,
        student_tau: float = 0.07,
        gate_max_rank: int = 20,
        mode: str = "weighted_logits",
        rerank_teacher_weight: float = 0.35,
        legacy_teacher_weight: float = 0.65,
        symmetric: bool = False,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown tri-teacher distillation mode: {mode}")
        self.compact_topk = int(compact_topk)
        self.teacher_topk = int(teacher_topk)
        self.teacher_tau = float(teacher_tau)
        self.student_tau = float(student_tau)
        self.gate_max_rank = int(gate_max_rank)
        self.mode = str(mode)
        self.rerank_teacher_weight = float(rerank_teacher_weight)
        self.legacy_teacher_weight = float(legacy_teacher_weight)
        self.symmetric = bool(symmetric)

    def _combine_scores(
        self,
        rerank_scores: torch.Tensor,
        legacy_scores: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "rerank_only":
            return rerank_scores
        if self.mode == "legacy_only":
            return legacy_scores
        if self.mode == "average_logits":
            return 0.5 * (rerank_scores + legacy_scores)
        weight_sum = self.rerank_teacher_weight + self.legacy_teacher_weight
        if weight_sum <= 0:
            raise ValueError("tri_teacher_distill weighted_logits requires positive teacher weights")
        rerank_w = self.rerank_teacher_weight / weight_sum
        legacy_w = self.legacy_teacher_weight / weight_sum
        return rerank_w * rerank_scores + legacy_w * legacy_scores

    def forward(
        self,
        compact_pred: torch.Tensor,
        retrieval_target: torch.Tensor,
        rerank_pred: torch.Tensor,
        rerank_target: torch.Tensor,
        legacy_pred: torch.Tensor,
        legacy_target: torch.Tensor,
        return_stats: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        if compact_pred.ndim != 2 or retrieval_target.ndim != 2:
            raise ValueError("compact_pred and retrieval_target must be 2D tensors")
        if rerank_pred.ndim != 2 or rerank_target.ndim != 2:
            raise ValueError("rerank_pred and rerank_target must be 2D tensors")
        if legacy_pred.ndim != 2 or legacy_target.ndim != 2:
            raise ValueError("legacy_pred and legacy_target must be 2D tensors")

        batch_size = compact_pred.shape[0]
        if retrieval_target.shape[0] != batch_size:
            raise ValueError("compact_pred and retrieval_target must share batch size")
        if rerank_pred.shape[0] != batch_size or rerank_target.shape[0] != batch_size:
            raise ValueError("rerank teacher tensors must share student batch size")
        if legacy_pred.shape[0] != batch_size or legacy_target.shape[0] != batch_size:
            raise ValueError("legacy teacher tensors must share student batch size")

        if batch_size <= 1:
            zero = compact_pred.new_zeros(())
            stats = {
                "gate_frac": 0.0,
                "candidate_size_mean": 1.0,
                "rerank_teacher_pos_rank_mean": 1.0,
                "rerank_teacher_pos_rank_median": 1.0,
                "legacy_teacher_pos_rank_mean": 1.0,
                "legacy_teacher_pos_rank_median": 1.0,
                "combined_teacher_pos_rank_mean": 1.0,
                "combined_teacher_pos_rank_median": 1.0,
                "student_pos_rank_mean": 1.0,
                "student_pos_rank_median": 1.0,
            }
            return (zero, stats) if return_stats else zero

        compact_pred_f = compact_pred.float()
        retrieval_target_f = retrieval_target.detach().float()
        rerank_pred_f = rerank_pred.detach().float()
        rerank_target_f = rerank_target.detach().float()
        legacy_pred_f = legacy_pred.detach().float()
        legacy_target_f = legacy_target.detach().float()

        student_scores_det = compact_pred_f.detach() @ retrieval_target_f.T
        student_scores = compact_pred_f @ retrieval_target_f.T
        rerank_scores_det = rerank_pred_f @ rerank_target_f.T
        legacy_scores_det = legacy_pred_f @ legacy_target_f.T
        combined_scores_det = self._combine_scores(rerank_scores_det, legacy_scores_det)

        rerank_ranks = []
        legacy_ranks = []
        combined_ranks = []
        student_ranks = []
        candidate_sizes = []
        valid_losses = []
        n_gated = 0

        compact_top_k = min(self.compact_topk, batch_size)
        teacher_top_k = min(self.teacher_topk, batch_size)

        for row_idx in range(batch_size):
            compact_top = torch.topk(student_scores_det[row_idx], k=compact_top_k).indices.tolist()
            rerank_top = torch.topk(rerank_scores_det[row_idx], k=teacher_top_k).indices.tolist()
            legacy_top = torch.topk(legacy_scores_det[row_idx], k=teacher_top_k).indices.tolist()

            candidate_indices = []
            seen = set()
            for idx in [row_idx, *compact_top, *rerank_top, *legacy_top]:
                idx_int = int(idx)
                if idx_int not in seen:
                    seen.add(idx_int)
                    candidate_indices.append(idx_int)

            cand_idx = torch.tensor(candidate_indices, device=compact_pred.device, dtype=torch.long)
            gt_local_index = candidate_indices.index(row_idx)
            candidate_sizes.append(len(candidate_indices))

            rerank_rank = _positive_rank(rerank_scores_det[row_idx], row_idx)
            legacy_rank = _positive_rank(legacy_scores_det[row_idx], row_idx)
            combined_rank = _positive_rank(combined_scores_det[row_idx], row_idx)
            rerank_ranks.append(rerank_rank)
            legacy_ranks.append(legacy_rank)
            combined_ranks.append(combined_rank)

            student_logits = student_scores[row_idx].index_select(0, cand_idx) / self.student_tau
            teacher_logits = combined_scores_det[row_idx].index_select(0, cand_idx) / self.teacher_tau

            if student_logits.shape != teacher_logits.shape:
                raise AssertionError(
                    "tri_teacher_distill student/teacher logits shape mismatch: "
                    f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
                )

            student_rank = _positive_rank(student_logits.detach(), gt_local_index)
            student_ranks.append(student_rank)

            if self.gate_max_rank > 0 and combined_rank > self.gate_max_rank:
                continue

            teacher_dist = F.softmax(teacher_logits, dim=-1)
            student_log_dist = F.log_softmax(student_logits, dim=-1)
            loss = F.kl_div(student_log_dist, teacher_dist, reduction="sum")

            if self.symmetric:
                student_dist = student_log_dist.exp()
                teacher_log_dist = F.log_softmax(teacher_logits, dim=-1)
                reverse_kl = torch.sum(
                    student_dist * (student_log_dist - teacher_log_dist),
                    dim=-1,
                )
                loss = 0.5 * (loss + reverse_kl)

            valid_losses.append(loss)
            n_gated += 1

        if valid_losses:
            loss = torch.stack(valid_losses).mean()
        else:
            loss = compact_pred.sum() * 0.0

        stats = {
            "gate_frac": float(n_gated / max(batch_size, 1)),
            "candidate_size_mean": float(torch.tensor(candidate_sizes, dtype=torch.float32).mean().item()),
            "rerank_teacher_pos_rank_mean": float(torch.tensor(rerank_ranks, dtype=torch.float32).mean().item()),
            "rerank_teacher_pos_rank_median": float(torch.tensor(rerank_ranks, dtype=torch.float32).median().item()),
            "legacy_teacher_pos_rank_mean": float(torch.tensor(legacy_ranks, dtype=torch.float32).mean().item()),
            "legacy_teacher_pos_rank_median": float(torch.tensor(legacy_ranks, dtype=torch.float32).median().item()),
            "combined_teacher_pos_rank_mean": float(torch.tensor(combined_ranks, dtype=torch.float32).mean().item()),
            "combined_teacher_pos_rank_median": float(torch.tensor(combined_ranks, dtype=torch.float32).median().item()),
            "student_pos_rank_mean": float(torch.tensor(student_ranks, dtype=torch.float32).mean().item()),
            "student_pos_rank_median": float(torch.tensor(student_ranks, dtype=torch.float32).median().item()),
        }
        return (loss, stats) if return_stats else loss
