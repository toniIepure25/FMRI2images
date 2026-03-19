"""Top-k-aware distillation from a frozen legacy teacher into the compact head."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _positive_rank(scores: torch.Tensor, positive_index: int) -> int:
    order = torch.argsort(scores, descending=True)
    pos = torch.where(order == positive_index)[0]
    return int(pos[0].item()) + 1 if len(pos) > 0 else scores.numel()


class LegacyCompactDistillLoss(nn.Module):
    """Distil full-batch legacy ranking signal into compact retrieval logits.

    The student distribution lives in compact retrieval space over the in-batch
    retrieval targets. The teacher distribution lives in the frozen legacy space
    over the aligned in-batch legacy/rich targets. To stay shortlist-aware
    without introducing a second learned gate, the teacher distribution is
    sparsified to its top-k candidates plus the ground-truth item.
    """

    def __init__(
        self,
        teacher_temperature: float = 0.07,
        student_temperature: float = 0.07,
        topk: int = 12,
    ) -> None:
        super().__init__()
        self.teacher_temperature = float(teacher_temperature)
        self.student_temperature = float(student_temperature)
        self.topk = int(topk)

    def forward(
        self,
        compact_pred: torch.Tensor,
        retrieval_target: torch.Tensor,
        teacher_pred: torch.Tensor,
        teacher_target: torch.Tensor,
        return_stats: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        if compact_pred.ndim != 2 or retrieval_target.ndim != 2:
            raise ValueError("compact_pred and retrieval_target must be 2D tensors")
        if teacher_pred.ndim != 2 or teacher_target.ndim != 2:
            raise ValueError("teacher_pred and teacher_target must be 2D tensors")
        if compact_pred.shape[0] != retrieval_target.shape[0]:
            raise ValueError("compact_pred and retrieval_target must share batch size")
        if teacher_pred.shape[0] != teacher_target.shape[0]:
            raise ValueError("teacher_pred and teacher_target must share batch size")
        if compact_pred.shape[0] != teacher_pred.shape[0]:
            raise ValueError("student and teacher tensors must share batch size")

        batch_size = compact_pred.shape[0]
        if batch_size <= 1:
            zero = compact_pred.new_zeros(())
            stats = {
                "teacher_topk_hit_frac": 1.0,
                "teacher_pos_rank_mean": 1.0,
                "teacher_pos_rank_median": 1.0,
                "student_pos_rank_mean": 1.0,
                "student_pos_rank_median": 1.0,
                "active_candidate_size_mean": 1.0,
            }
            return (zero, stats) if return_stats else zero

        compact_pred_f = compact_pred.float()
        retrieval_target_f = retrieval_target.detach().float()
        teacher_pred_f = teacher_pred.detach().float()
        teacher_target_f = teacher_target.detach().float()

        student_logits = (compact_pred_f @ retrieval_target_f.T) / self.student_temperature
        teacher_logits = (teacher_pred_f @ teacher_target_f.T) / self.teacher_temperature

        teacher_positive_ranks = []
        student_positive_ranks = []
        teacher_topk_hits = []
        active_sizes = []
        losses = []

        topk = min(max(self.topk, 1), batch_size)
        neg_inf = torch.finfo(teacher_logits.dtype).min

        for row_idx in range(batch_size):
            teacher_row = teacher_logits[row_idx]
            student_row = student_logits[row_idx]

            teacher_rank = _positive_rank(teacher_row.detach(), row_idx)
            student_rank = _positive_rank(student_row.detach(), row_idx)
            teacher_positive_ranks.append(teacher_rank)
            student_positive_ranks.append(student_rank)

            top_idx = torch.topk(teacher_row, k=topk).indices
            mask = torch.zeros(batch_size, dtype=torch.bool, device=teacher_row.device)
            mask[top_idx] = True
            mask[row_idx] = True

            teacher_topk_hits.append(float(bool(mask[row_idx].item())))
            active_sizes.append(float(mask.sum().item()))

            masked_teacher_logits = teacher_row.masked_fill(~mask, neg_inf)
            teacher_dist = F.softmax(masked_teacher_logits, dim=-1)
            student_log_dist = F.log_softmax(student_row, dim=-1)
            losses.append(F.kl_div(student_log_dist, teacher_dist, reduction="sum"))

        loss = torch.stack(losses).mean() if losses else compact_pred.sum() * 0.0
        stats = {
            "teacher_topk_hit_frac": float(torch.tensor(teacher_topk_hits, dtype=torch.float32).mean().item()),
            "teacher_pos_rank_mean": float(torch.tensor(teacher_positive_ranks, dtype=torch.float32).mean().item()),
            "teacher_pos_rank_median": float(torch.tensor(teacher_positive_ranks, dtype=torch.float32).median().item()),
            "student_pos_rank_mean": float(torch.tensor(student_positive_ranks, dtype=torch.float32).mean().item()),
            "student_pos_rank_median": float(torch.tensor(student_positive_ranks, dtype=torch.float32).median().item()),
            "active_candidate_size_mean": float(torch.tensor(active_sizes, dtype=torch.float32).mean().item()),
        }
        return (loss, stats) if return_stats else loss
