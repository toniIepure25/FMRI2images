"""Component-responsibility legacy -> compact distillation for multi-hypothesis vMF heads."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _positive_rank(scores: torch.Tensor, positive_index: int) -> int:
    order = torch.argsort(scores, descending=True)
    pos = torch.where(order == positive_index)[0]
    return int(pos[0].item()) + 1 if len(pos) > 0 else scores.numel()


class ComponentLegacyCompactDistillLoss(nn.Module):
    """Distil a sparse legacy teacher distribution into one responsible compact component.

    The teacher still operates over full in-batch legacy logits, sparsified to top-k plus
    the positive item. The student side evaluates each compact component separately in the
    compact retrieval space, then assigns each query to the component whose distribution is
    closest to the teacher. This creates component-specific pressure instead of only pulling
    the consensus compact head toward the teacher.
    """

    def __init__(
        self,
        teacher_temperature: float = 0.07,
        student_temperature: float = 0.07,
        topk: int = 12,
        use_component_logits_prior: bool = True,
        prior_weight: float = 0.10,
    ) -> None:
        super().__init__()
        self.teacher_temperature = float(teacher_temperature)
        self.student_temperature = float(student_temperature)
        self.topk = int(topk)
        self.use_component_logits_prior = bool(use_component_logits_prior)
        self.prior_weight = float(prior_weight)

    def forward(
        self,
        compact_component_mu: torch.Tensor,
        compact_component_kappa: torch.Tensor,
        retrieval_target: torch.Tensor,
        teacher_pred: torch.Tensor,
        teacher_target: torch.Tensor,
        component_logits: torch.Tensor | None = None,
        return_stats: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        if compact_component_mu.ndim != 3:
            raise ValueError("compact_component_mu must have shape (B, M, D)")
        if compact_component_kappa.ndim == 3 and compact_component_kappa.shape[-1] == 1:
            compact_component_kappa = compact_component_kappa[..., 0]
        if compact_component_kappa.ndim != 2:
            raise ValueError("compact_component_kappa must have shape (B, M) or (B, M, 1)")
        if retrieval_target.ndim != 2 or teacher_pred.ndim != 2 or teacher_target.ndim != 2:
            raise ValueError("retrieval_target, teacher_pred, and teacher_target must be 2D tensors")
        if compact_component_mu.shape[0] != retrieval_target.shape[0]:
            raise ValueError("student components and retrieval_target must share batch size")
        if teacher_pred.shape[0] != teacher_target.shape[0]:
            raise ValueError("teacher_pred and teacher_target must share batch size")
        if teacher_pred.shape[0] != compact_component_mu.shape[0]:
            raise ValueError("teacher and student tensors must share batch size")

        batch_size, num_components, _ = compact_component_mu.shape
        if batch_size <= 1:
            zero = compact_component_mu.new_zeros(())
            stats = {
                "teacher_topk_hit_frac": 1.0,
                "teacher_pos_rank_mean": 1.0,
                "teacher_pos_rank_median": 1.0,
                "student_pos_rank_mean": 1.0,
                "student_pos_rank_median": 1.0,
                "active_candidate_size_mean": 1.0,
                "responsible_component_entropy": 0.0,
                "responsible_component_top_rate": 1.0,
                "responsible_component_mean": 0.0,
            }
            return (zero, stats) if return_stats else zero

        compact_component_mu = F.normalize(compact_component_mu.float(), p=2, dim=-1)
        compact_component_kappa = compact_component_kappa.float()
        retrieval_target = retrieval_target.detach().float()
        teacher_pred = teacher_pred.detach().float()
        teacher_target = teacher_target.detach().float()

        teacher_logits = (teacher_pred @ teacher_target.T) / self.teacher_temperature
        student_logits = torch.einsum(
            "bmd,nd->bmn",
            compact_component_mu,
            retrieval_target,
        )
        student_logits = student_logits * (compact_component_kappa[:, :, None] / self.student_temperature)

        component_prior = None
        if component_logits is not None:
            component_prior = torch.softmax(component_logits.detach().float(), dim=-1)

        teacher_positive_ranks = []
        student_positive_ranks = []
        teacher_topk_hits = []
        active_sizes = []
        assigned_components = []
        losses = []

        topk = min(max(self.topk, 1), batch_size)
        neg_inf = torch.finfo(teacher_logits.dtype).min

        for row_idx in range(batch_size):
            teacher_row = teacher_logits[row_idx]
            student_rows = student_logits[row_idx]  # (M, B)

            teacher_rank = _positive_rank(teacher_row.detach(), row_idx)
            teacher_positive_ranks.append(teacher_rank)

            top_idx = torch.topk(teacher_row, k=topk).indices
            mask = torch.zeros(batch_size, dtype=torch.bool, device=teacher_row.device)
            mask[top_idx] = True
            mask[row_idx] = True
            teacher_topk_hits.append(float(bool(mask[row_idx].item())))
            active_sizes.append(float(mask.sum().item()))

            masked_teacher_logits = teacher_row.masked_fill(~mask, neg_inf)
            teacher_dist = F.softmax(masked_teacher_logits, dim=-1)

            component_kls = []
            component_pos_ranks = []
            for comp_idx in range(num_components):
                student_row = student_rows[comp_idx]
                component_pos_ranks.append(_positive_rank(student_row.detach(), row_idx))
                student_log_dist = F.log_softmax(student_row, dim=-1)
                component_kls.append(F.kl_div(student_log_dist, teacher_dist, reduction="sum"))
            component_kls_t = torch.stack(component_kls)

            if component_prior is not None and self.use_component_logits_prior:
                prior = component_prior[row_idx].clamp_min(1e-8)
                selection_score = component_kls_t - self.prior_weight * torch.log(prior)
            else:
                selection_score = component_kls_t

            assigned = int(torch.argmin(selection_score).item())
            assigned_components.append(assigned)
            student_positive_ranks.append(component_pos_ranks[assigned])
            losses.append(component_kls_t[assigned])

        loss = torch.stack(losses).mean() if losses else compact_component_mu.sum() * 0.0
        assigned_tensor = torch.tensor(assigned_components, dtype=torch.long)
        comp_hist = torch.bincount(assigned_tensor, minlength=num_components).float()
        comp_probs = comp_hist / max(float(comp_hist.sum().item()), 1.0)
        comp_entropy = -(comp_probs * torch.log(comp_probs.clamp_min(1e-8))).sum().item()
        comp_entropy /= max(float(torch.log(torch.tensor(float(num_components))).item()), 1e-8)

        stats = {
            "teacher_topk_hit_frac": float(torch.tensor(teacher_topk_hits, dtype=torch.float32).mean().item()),
            "teacher_pos_rank_mean": float(torch.tensor(teacher_positive_ranks, dtype=torch.float32).mean().item()),
            "teacher_pos_rank_median": float(torch.tensor(teacher_positive_ranks, dtype=torch.float32).median().item()),
            "student_pos_rank_mean": float(torch.tensor(student_positive_ranks, dtype=torch.float32).mean().item()),
            "student_pos_rank_median": float(torch.tensor(student_positive_ranks, dtype=torch.float32).median().item()),
            "active_candidate_size_mean": float(torch.tensor(active_sizes, dtype=torch.float32).mean().item()),
            "responsible_component_entropy": float(comp_entropy),
            "responsible_component_top_rate": float(comp_probs.max().item()),
            "responsible_component_mean": float(assigned_tensor.float().mean().item()),
        }
        return (loss, stats) if return_stats else loss
