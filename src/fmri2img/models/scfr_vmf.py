"""
Subject-Conditioned Factorized Retrieval (SCFR) with a vMF retrieval head.

This module keeps retrieval-first behaviour while explicitly factorizing the
shared encoder representation into:

- z_vis: subject-invariant visual latent used for retrieval
- z_subj: subject-specific latent used for subject classification

The design is intentionally lightweight and warm-start friendly:
- FiLM-style subject conditioning starts as identity
- the factor projection is initialized to copy the encoder latent into z_vis
  when dimensions allow, so pretrained retrieval weights remain meaningful
- the optional adversarial subject branch uses a Gradient Reversal Layer (GRL)
  whose strength is scheduled over epochs
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
from torch.autograd import Function

from fmri2img.models.vmf_decoder import VonMisesFisherDecoder

logger = logging.getLogger(__name__)


def _build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    *,
    activation: str = "gelu",
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    act: nn.Module
    for hidden_dim in hidden_dims:
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        layers.extend([
            nn.Linear(in_dim, int(hidden_dim)),
            act,
            nn.Dropout(float(dropout)),
        ])
        in_dim = int(hidden_dim)
    layers.append(nn.Linear(in_dim, int(output_dim)))
    return nn.Sequential(*layers)


class _GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_scale: float) -> torch.Tensor:
        ctx.lambda_scale = float(lambda_scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_scale * grad_output, None


def gradient_reverse(x: torch.Tensor, lambda_scale: float) -> torch.Tensor:
    return _GradientReversalFn.apply(x, float(lambda_scale))


class SubjectFiLMConditioner(nn.Module):
    """Lightweight subject conditioning that starts as identity."""

    def __init__(
        self,
        feature_dim: int,
        num_subjects: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_subjects = int(num_subjects)
        self.subject_embedding = nn.Embedding(self.num_subjects, int(embedding_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.adapter = _build_mlp(
            int(embedding_dim),
            [int(hidden_dim)] if hidden_dim > 0 else [],
            self.feature_dim * 2,
            activation=activation,
            dropout=dropout,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.subject_embedding.weight, mean=0.0, std=0.02)
        for module in self.adapter.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, h: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        subject_emb = self.dropout(self.subject_embedding(subject_ids))
        gamma_beta = self.adapter(subject_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


class SCFRVMFOutput(NamedTuple):
    mu: torch.Tensor
    kappa: torch.Tensor
    z_vis: torch.Tensor
    z_subj: torch.Tensor
    subject_logits: torch.Tensor
    adv_subject_logits: Optional[torch.Tensor]
    adv_lambda: float


class SubjectConditionedFactorizedVMFDecoder(nn.Module):
    """Factorized retrieval decoder with optional subject-adversarial probing."""

    def __init__(
        self,
        input_dim: int,
        retrieval_dim: int,
        num_subjects: int,
        *,
        total_latent_dim: int = 4096,
        z_vis_dim: int = 2048,
        z_subj_dim: int = 2048,
        subject_embedding_dim: int = 64,
        subject_condition_hidden_dim: int = 128,
        subject_condition_dropout: float = 0.0,
        factor_hidden_dims: Optional[list[int]] = None,
        subject_head_hidden_dims: Optional[list[int]] = None,
        adversarial_enabled: bool = False,
        adversarial_head_hidden_dims: Optional[list[int]] = None,
        grl_start_epoch: int = 10,
        grl_ramp_epochs: int = 8,
        grl_lambda_max: float = 0.2,
        activation: str = "gelu",
        dropout: float = 0.1,
        kappa_min: float = 1e-3,
        kappa_max: float = 500.0,
        kappa_mode: str = "softplus",
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.output_dim = int(retrieval_dim)
        self.num_subjects = int(num_subjects)
        self.total_latent_dim = int(total_latent_dim)
        self.z_vis_dim = int(z_vis_dim)
        self.z_subj_dim = int(z_subj_dim)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.kappa_mode = str(kappa_mode)
        self.adversarial_enabled = bool(adversarial_enabled)
        self.grl_start_epoch = max(int(grl_start_epoch), 0)
        self.grl_ramp_epochs = max(int(grl_ramp_epochs), 1)
        self.grl_lambda_max = float(grl_lambda_max)
        self._current_epoch = 0
        self._current_adv_lambda = 0.0

        if self.z_vis_dim + self.z_subj_dim != self.total_latent_dim:
            raise ValueError(
                "SCFR latent split mismatch: "
                f"z_vis_dim ({self.z_vis_dim}) + z_subj_dim ({self.z_subj_dim}) "
                f"!= total_latent_dim ({self.total_latent_dim})"
            )
        if self.num_subjects < 2:
            logger.warning(
                "SCFR configured with num_subjects=%d. This architecture is designed "
                "for multi-subject training and may not be meaningful here.",
                self.num_subjects,
            )

        self.subject_conditioner = SubjectFiLMConditioner(
            feature_dim=self.input_dim,
            num_subjects=self.num_subjects,
            embedding_dim=subject_embedding_dim,
            hidden_dim=subject_condition_hidden_dim,
            dropout=subject_condition_dropout,
            activation=activation,
        )

        factor_hidden_dims = factor_hidden_dims or []
        if factor_hidden_dims:
            self.factor_backbone = _build_mlp(
                self.input_dim,
                factor_hidden_dims,
                factor_hidden_dims[-1],
                activation=activation,
                dropout=dropout,
            )
            factor_in_dim = factor_hidden_dims[-1]
        else:
            self.factor_backbone = nn.Identity()
            factor_in_dim = self.input_dim

        self.factor_projection = nn.Linear(factor_in_dim, self.total_latent_dim)
        self._reset_factor_projection()

        self.retrieval_decoder = VonMisesFisherDecoder(
            input_dim=self.z_vis_dim,
            output_dim=int(retrieval_dim),
            hidden_dims=[self.z_vis_dim],
            activation=activation,
            dropout=dropout,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            kappa_mode=kappa_mode,
            regression_head=False,
        )

        subject_head_hidden_dims = subject_head_hidden_dims or [512]
        self.subject_head = _build_mlp(
            self.z_subj_dim,
            subject_head_hidden_dims,
            self.num_subjects,
            activation=activation,
            dropout=dropout,
        )

        self.adv_subject_head: Optional[nn.Sequential] = None
        if self.adversarial_enabled:
            self.adv_subject_head = _build_mlp(
                self.z_vis_dim,
                adversarial_head_hidden_dims or [512],
                self.num_subjects,
                activation=activation,
                dropout=dropout,
            )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "SCFR vMF decoder: input=%d total_latent=%d (z_vis=%d, z_subj=%d) "
            "subjects=%d adv=%s grl(start=%d ramp=%d lambda_max=%.3f) params=%.2fM",
            self.input_dim,
            self.total_latent_dim,
            self.z_vis_dim,
            self.z_subj_dim,
            self.num_subjects,
            self.adversarial_enabled,
            self.grl_start_epoch,
            self.grl_ramp_epochs,
            self.grl_lambda_max,
            n_params / 1e6,
        )

    def _reset_factor_projection(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.factor_projection.weight)
            nn.init.zeros_(self.factor_projection.bias)
            if self.input_dim == self.z_vis_dim:
                eye = torch.eye(self.z_vis_dim, dtype=self.factor_projection.weight.dtype)
                self.factor_projection.weight[: self.z_vis_dim, : self.input_dim].copy_(eye)
                logger.info(
                    "SCFR factor projection initialized as z_vis identity copy "
                    "(encoder_dim=%d -> z_vis_dim=%d)",
                    self.input_dim,
                    self.z_vis_dim,
                )
            else:
                nn.init.kaiming_uniform_(self.factor_projection.weight, a=5 ** 0.5)
                logger.info(
                    "SCFR factor projection uses Kaiming init because encoder_dim=%d "
                    "!= z_vis_dim=%d",
                    self.input_dim,
                    self.z_vis_dim,
                )

    def set_current_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)
        if not self.adversarial_enabled:
            self._current_adv_lambda = 0.0
            return
        if self._current_epoch <= self.grl_start_epoch:
            self._current_adv_lambda = 0.0
            return
        progress = min(
            max(self._current_epoch - self.grl_start_epoch, 0) / float(self.grl_ramp_epochs),
            1.0,
        )
        self._current_adv_lambda = float(self.grl_lambda_max) * float(progress)

    @property
    def current_adv_lambda(self) -> float:
        return float(self._current_adv_lambda)

    def forward(self, h: torch.Tensor, subject_ids: torch.Tensor) -> SCFRVMFOutput:
        if subject_ids is None:
            raise ValueError("SCFR requires subject_ids for subject conditioning")
        if subject_ids.ndim != 1:
            raise ValueError(f"SCFR subject_ids must be 1D, got shape {tuple(subject_ids.shape)}")
        if subject_ids.shape[0] != h.shape[0]:
            raise ValueError(
                f"SCFR subject_ids batch mismatch: {subject_ids.shape[0]} ids for {h.shape[0]} samples"
            )
        if subject_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"SCFR subject_ids must be integer typed, got {subject_ids.dtype}")

        subject_ids = subject_ids.to(dtype=torch.long)
        if int(subject_ids.min().item()) < 0 or int(subject_ids.max().item()) >= self.num_subjects:
            raise ValueError(
                f"SCFR subject_ids out of range [0, {self.num_subjects - 1}]: "
                f"min={int(subject_ids.min().item())}, max={int(subject_ids.max().item())}"
            )

        conditioned = self.subject_conditioner(h, subject_ids)
        factor_features = self.factor_backbone(conditioned)
        factor_latent = self.factor_projection(factor_features)
        z_vis, z_subj = torch.split(
            factor_latent,
            [self.z_vis_dim, self.z_subj_dim],
            dim=-1,
        )

        retrieval_out = self.retrieval_decoder(z_vis)
        if len(retrieval_out) != 2:
            raise RuntimeError("SCFR retrieval decoder must return (mu, kappa)")
        mu, kappa = retrieval_out

        subject_logits = self.subject_head(z_subj)

        adv_logits = None
        if self.adv_subject_head is not None:
            adv_input = gradient_reverse(z_vis, self._current_adv_lambda)
            adv_logits = self.adv_subject_head(adv_input)

        return SCFRVMFOutput(
            mu=mu,
            kappa=kappa,
            z_vis=z_vis,
            z_subj=z_subj,
            subject_logits=subject_logits,
            adv_subject_logits=adv_logits,
            adv_lambda=float(self._current_adv_lambda),
        )
