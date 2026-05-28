"""
NeuroBridge-OT Loss Module — Module H
========================================

Configurable composite loss combining all NeuroBridge-OT objectives:
    L_total = L_clip_contrastive
            + lambda_reg * L_embedding_regression
            + lambda_token * L_token_regression
            + lambda_teacher * L_teacher_distillation
            + lambda_ot * L_optimal_transport
            + lambda_shared * L_shared_image_consistency
            + lambda_adv * L_subject_adversarial
            + lambda_vmf * L_vMF_uncertainty
            + lambda_cal * L_calibration
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function

logger = logging.getLogger(__name__)


class GradientReversalFunction(Function):
    """Gradient Reversal Layer for adversarial training (Ganin et al., 2016)."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps the gradient reversal function as a module."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class SubjectAdversarialLoss(nn.Module):
    """Subject adversarial classifier with gradient reversal.

    Predicts subject identity from semantic embedding; gradient reversal
    encourages the encoder to remove subject-identifying information.

    Args:
        d_model: Input feature dimensionality.
        n_subjects: Number of subjects to classify.
        hidden_dim: Classifier hidden dimension.
        grl_lambda: Gradient reversal strength.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_subjects: int = 8,
        hidden_dim: int = 256,
        grl_lambda: float = 1.0,
    ):
        super().__init__()
        self.grl = GradientReversalLayer(grl_lambda)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_subjects),
        )

    def forward(self, features: Tensor, subject_labels: Tensor) -> Tensor:
        """Compute adversarial subject classification loss.

        Args:
            features: Semantic embedding, shape (B, d_model).
            subject_labels: Ground truth subject IDs, shape (B,).

        Returns:
            Scalar cross-entropy loss (with reversed gradients to encoder).
        """
        reversed_features = self.grl(features)
        logits = self.classifier(reversed_features)
        return F.cross_entropy(logits, subject_labels.long())


class InfoNCELoss(nn.Module):
    """Batch-wise InfoNCE contrastive loss.

    Args:
        temperature: Softmax temperature.
        symmetric: Whether to compute symmetric (bidirectional) loss.
    """

    def __init__(self, temperature: float = 0.07, symmetric: bool = True):
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        """Compute InfoNCE loss.

        Args:
            query: Query embeddings (B, D), L2-normalized.
            key: Key embeddings (B, D), L2-normalized.

        Returns:
            Scalar contrastive loss.
        """
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * query @ key.T  # (B, B)
        labels = torch.arange(logits.shape[0], device=logits.device)

        loss_q = F.cross_entropy(logits, labels)
        if self.symmetric:
            loss_k = F.cross_entropy(logits.T, labels)
            return (loss_q + loss_k) / 2
        return loss_q


class VMFNLLLoss(nn.Module):
    """von Mises-Fisher negative log-likelihood loss.

    Computes -log p(target | mu, kappa) under vMF distribution.
    The Bessel normalization constant is approximated for stability.

    Args:
        kappa_reg_weight: Weight for kappa regularization.
        target_kappa: Target mean kappa for regularization.
    """

    def __init__(self, kappa_reg_weight: float = 0.1, target_kappa: float = 50.0):
        super().__init__()
        self.kappa_reg_weight = kappa_reg_weight
        self.target_kappa = target_kappa

    def forward(self, mu: Tensor, kappa: Tensor, target: Tensor) -> Tensor:
        """Compute vMF NLL loss.

        Args:
            mu: Predicted mean direction, shape (B, D), unit-normalized.
            kappa: Concentration parameter, shape (B,), positive.
            target: Target embedding, shape (B, D), unit-normalized.

        Returns:
            Scalar loss = -E[kappa * cos(mu, target)] + kappa_regularization.
        """
        cos_sim = (mu * target).sum(dim=-1)  # (B,)
        nll = -kappa * cos_sim  # Higher kappa + high cos_sim = lower loss

        # Kappa regularization to prevent collapse or explosion
        kappa_reg = self.kappa_reg_weight * (
            (kappa - self.target_kappa) ** 2
        ).mean()

        return nll.mean() + kappa_reg


class SharedImageConsistencyLoss(nn.Module):
    """Encourages consistent semantic embeddings across subjects for same nsdId.

    When multiple subjects view the same image, their brain semantic
    embeddings should be similar in the shared space.

    Args:
        margin: Minimum similarity threshold.
    """

    def __init__(self, margin: float = 0.8):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: Tensor,
        nsd_ids: Tensor,
        subject_ids: Tensor,
    ) -> Tensor:
        """Compute shared-image consistency loss.

        Args:
            embeddings: Brain semantic embeddings, shape (B, D).
            nsd_ids: Image IDs per sample, shape (B,).
            subject_ids: Subject IDs per sample, shape (B,).

        Returns:
            Scalar consistency loss.
        """
        B = embeddings.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Find pairs with same nsdId but different subjects
        nsd_ids_expanded = nsd_ids.unsqueeze(0)  # (1, B)
        same_image = (nsd_ids.unsqueeze(1) == nsd_ids_expanded)  # (B, B)
        diff_subject = (subject_ids.unsqueeze(1) != subject_ids.unsqueeze(0))
        valid_pairs = same_image & diff_subject

        if not valid_pairs.any():
            return torch.tensor(0.0, device=embeddings.device)

        # Cosine similarity for valid pairs
        embeddings_norm = F.normalize(embeddings, dim=-1)
        sim_matrix = embeddings_norm @ embeddings_norm.T  # (B, B)

        # Hinge loss: penalize pairs below margin
        pair_losses = F.relu(self.margin - sim_matrix)
        masked_losses = pair_losses * valid_pairs.float()

        n_pairs = valid_pairs.sum().clamp(min=1)
        return masked_losses.sum() / n_pairs


class NeuroBridgeOTLoss(nn.Module):
    """Composite loss for NeuroBridge-OT training.

    All sub-losses are optional and controlled by config weights.

    Args:
        config: Loss configuration dict with weights and sub-loss parameters.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # Weights
        self.w_contrastive = config.get("w_contrastive", 1.0)
        self.w_regression = config.get("w_regression", 0.0)
        self.w_token = config.get("w_token", 0.0)
        self.w_teacher = config.get("w_teacher", 0.0)
        self.w_ot = config.get("w_ot", 0.1)
        self.w_shared = config.get("w_shared", 0.0)
        self.w_adversarial = config.get("w_adversarial", 0.0)
        self.w_vmf = config.get("w_vmf", 0.0)
        self.w_calibration = config.get("w_calibration", 0.0)

        # Sub-losses
        contrastive_cfg = config.get("contrastive", {})
        self.contrastive_loss = InfoNCELoss(
            temperature=contrastive_cfg.get("temperature", 0.07),
            symmetric=contrastive_cfg.get("symmetric", True),
        )

        if self.w_vmf > 0:
            vmf_cfg = config.get("vmf", {})
            self.vmf_loss = VMFNLLLoss(
                kappa_reg_weight=vmf_cfg.get("kappa_reg_weight", 0.1),
                target_kappa=vmf_cfg.get("target_kappa", 50.0),
            )

        if self.w_adversarial > 0:
            adv_cfg = config.get("adversarial", {})
            self.adversarial_loss = SubjectAdversarialLoss(
                d_model=adv_cfg.get("d_model", 768),
                n_subjects=adv_cfg.get("n_subjects", 8),
                hidden_dim=adv_cfg.get("hidden_dim", 256),
                grl_lambda=adv_cfg.get("grl_lambda", 1.0),
            )

        if self.w_shared > 0:
            shared_cfg = config.get("shared_consistency", {})
            self.shared_loss = SharedImageConsistencyLoss(
                margin=shared_cfg.get("margin", 0.8),
            )

    def forward(
        self,
        model_outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        ot_cost: Optional[Tensor] = None,
        teacher_predictions: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute total loss from model outputs and targets.

        Args:
            model_outputs: Dict from NeuroBridgeOTModel.forward containing:
                - 'clip_embedding': (B, D) predicted CLIP embedding.
                - 'vmf_mu': (B, D) vMF mean direction (optional).
                - 'vmf_kappa': (B,) vMF concentration (optional).
                - 'token_predictions': (B, T, D) token preds (optional).
                - 'confidence': (B,) calibration score (optional).
                - 'cls_embedding': (B, D) raw CLS embedding for adversarial.
            targets: Dict containing:
                - 'clip_target': (B, D) ground truth CLIP embedding.
                - 'token_target': (B, T, D) token targets (optional).
                - 'subject_id': (B,) subject labels.
                - 'nsd_id': (B,) image IDs.
            ot_cost: Scalar OT transport cost from alignment module.
            teacher_predictions: (B, D) teacher embeddings (optional).

        Returns:
            Dict with 'total_loss' and per-component losses for logging.
        """
        losses: Dict[str, Tensor] = {}
        device = model_outputs["clip_embedding"].device
        total = torch.tensor(0.0, device=device)

        # 1. Contrastive loss (always active)
        if self.w_contrastive > 0:
            clip_pred = model_outputs["clip_embedding"]
            clip_target = targets["clip_target"]
            l_cont = self.contrastive_loss(clip_pred, clip_target)
            losses["contrastive"] = l_cont
            total = total + self.w_contrastive * l_cont

        # 2. Embedding regression (cosine)
        if self.w_regression > 0:
            clip_pred = model_outputs["clip_embedding"]
            clip_target = targets["clip_target"]
            cos_loss = 1.0 - F.cosine_similarity(clip_pred, clip_target, dim=-1).mean()
            losses["regression"] = cos_loss
            total = total + self.w_regression * cos_loss

        # 3. Token regression
        if self.w_token > 0 and "token_predictions" in model_outputs:
            token_pred = model_outputs["token_predictions"]
            if "token_target" in targets and targets["token_target"] is not None:
                token_target = targets["token_target"]
                token_loss = F.mse_loss(token_pred, token_target)
                losses["token_regression"] = token_loss
                total = total + self.w_token * token_loss

        # 4. Teacher distillation
        if self.w_teacher > 0 and teacher_predictions is not None:
            clip_pred = model_outputs["clip_embedding"]
            teacher_loss = 1.0 - F.cosine_similarity(
                clip_pred, teacher_predictions, dim=-1
            ).mean()
            losses["teacher_distill"] = teacher_loss
            total = total + self.w_teacher * teacher_loss

        # 5. OT alignment cost
        if self.w_ot > 0 and ot_cost is not None:
            losses["ot_cost"] = ot_cost
            total = total + self.w_ot * ot_cost

        # 6. Shared-image consistency
        if self.w_shared > 0 and "nsd_id" in targets and "subject_id" in targets:
            cls_emb = model_outputs.get("cls_embedding", model_outputs["clip_embedding"])
            l_shared = self.shared_loss(
                cls_emb, targets["nsd_id"], targets["subject_id"]
            )
            losses["shared_consistency"] = l_shared
            total = total + self.w_shared * l_shared

        # 7. Subject adversarial
        if self.w_adversarial > 0 and "subject_id" in targets:
            cls_emb = model_outputs.get("cls_embedding", model_outputs["clip_embedding"])
            l_adv = self.adversarial_loss(cls_emb, targets["subject_id"])
            losses["adversarial"] = l_adv
            total = total + self.w_adversarial * l_adv

        # 8. vMF uncertainty
        if self.w_vmf > 0 and "vmf_mu" in model_outputs:
            l_vmf = self.vmf_loss(
                model_outputs["vmf_mu"],
                model_outputs["vmf_kappa"],
                targets["clip_target"],
            )
            losses["vmf_nll"] = l_vmf
            total = total + self.w_vmf * l_vmf

        # 9. Calibration (binary cross-entropy for confidence prediction)
        if self.w_calibration > 0 and "confidence" in model_outputs:
            conf = model_outputs["confidence"]
            clip_pred = model_outputs["clip_embedding"]
            clip_target = targets["clip_target"]
            cos_sim = F.cosine_similarity(clip_pred, clip_target, dim=-1)
            # Target: high confidence for high-quality predictions
            conf_target = (cos_sim > 0.5).float()
            l_cal = F.binary_cross_entropy_with_logits(conf, conf_target)
            losses["calibration"] = l_cal
            total = total + self.w_calibration * l_cal

        losses["total_loss"] = total
        return losses
