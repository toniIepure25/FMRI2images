"""Shortlist-aware distillation from a frozen legacy retrieval teacher."""

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


class LegacyTeacherDistillLoss(nn.Module):
    """Distil a frozen legacy retrieval expert into the compact retrieval head."""

    def __init__(
        self,
        compact_k: int = 16,
        teacher_k: int = 16,
        teacher_temperature: float = 0.07,
        student_temperature: float = 0.07,
        teacher_rank_gate: int = 20,
    ) -> None:
        super().__init__()
        self.compact_k = int(compact_k)
        self.teacher_k = int(teacher_k)
        self.teacher_temperature = float(teacher_temperature)
        self.student_temperature = float(student_temperature)
        self.teacher_rank_gate = int(teacher_rank_gate)

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
                "gate_frac": 0.0,
                "teacher_pos_rank_mean": 1.0,
                "teacher_pos_rank_median": 1.0,
                "student_pos_rank_mean": 1.0,
                "student_pos_rank_median": 1.0,
            }
            return (zero, stats) if return_stats else zero

        compact_pred_f = compact_pred.float()
        retrieval_target_f = retrieval_target.detach().float()
        teacher_pred_f = teacher_pred.detach().float()
        teacher_target_f = teacher_target.detach().float()

        compact_scores_det = compact_pred_f.detach() @ retrieval_target_f.T
        compact_scores_student = compact_pred_f @ retrieval_target_f.T
        teacher_scores_det = teacher_pred_f @ teacher_target_f.T

        teacher_positive_ranks = []
        student_positive_ranks = []
        valid_losses = []
        n_gated = 0

        compact_top_k = min(self.compact_k, batch_size)
        teacher_top_k = min(self.teacher_k, batch_size)
        gate_limit = self.teacher_rank_gate

        for row_idx in range(batch_size):
            compact_top = torch.topk(compact_scores_det[row_idx], k=compact_top_k).indices.tolist()
            teacher_top = torch.topk(teacher_scores_det[row_idx], k=teacher_top_k).indices.tolist()

            candidate_indices = []
            seen = set()
            for idx in [row_idx, *compact_top, *teacher_top]:
                idx_int = int(idx)
                if idx_int not in seen:
                    seen.add(idx_int)
                    candidate_indices.append(idx_int)

            cand_idx = torch.tensor(candidate_indices, device=compact_pred.device, dtype=torch.long)
            gt_local_index = candidate_indices.index(row_idx)

            teacher_row = teacher_scores_det[row_idx]
            teacher_rank = _positive_rank(teacher_row, row_idx)
            teacher_positive_ranks.append(teacher_rank)

            student_logits = compact_scores_student[row_idx].index_select(0, cand_idx) / self.student_temperature
            student_rank = _positive_rank(student_logits.detach(), gt_local_index)
            student_positive_ranks.append(student_rank)

            if gate_limit > 0 and teacher_rank > gate_limit:
                continue

            teacher_logits = teacher_scores_det[row_idx].index_select(0, cand_idx) / self.teacher_temperature
            teacher_dist = F.softmax(teacher_logits, dim=-1)
            student_log_dist = F.log_softmax(student_logits, dim=-1)
            valid_losses.append(F.kl_div(student_log_dist, teacher_dist, reduction="sum"))
            n_gated += 1

        if valid_losses:
            loss = torch.stack(valid_losses).mean()
        else:
            loss = compact_pred.sum() * 0.0

        stats = {
            "gate_frac": float(n_gated / max(batch_size, 1)),
            "teacher_pos_rank_mean": float(torch.tensor(teacher_positive_ranks, dtype=torch.float32).mean().item()),
            "teacher_pos_rank_median": float(torch.tensor(teacher_positive_ranks, dtype=torch.float32).median().item()),
            "student_pos_rank_mean": float(torch.tensor(student_positive_ranks, dtype=torch.float32).mean().item()),
            "student_pos_rank_median": float(torch.tensor(student_positive_ranks, dtype=torch.float32).median().item()),
        }
        return (loss, stats) if return_stats else loss
