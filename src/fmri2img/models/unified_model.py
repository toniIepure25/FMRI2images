"""
Unified Model Factory for Research-Grade Experiments
===================================================

Creates models based on experiment configuration:
- Deterministic models (single output)
- Gaussian models (mu + logvar outputs)
- vMF models (mu + kappa on the unit hypersphere)
- SCFR vMF models (subject-conditioned factorized retrieval)
- vMF-DCF models (per-ROI vMF experts with spherical consensus fusion)

Supports modular architecture with encoder + decoder.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Literal, Tuple
import logging
from pathlib import Path

from fmri2img.models.vmf_decoder import VonMisesFisherDecoder
from fmri2img.models.roi_transformer import ROITransformerEncoder
from fmri2img.models.roi_dcf import ROIDCFDecoder
from fmri2img.models.multi_subject_encoder import MultiSubjectROITransformer
from fmri2img.models.dense_vmf_hybrid_decoder import DenseVMFHybridDecoder
from fmri2img.models.scfr_vmf import SCFRVMFOutput, SubjectConditionedFactorizedVMFDecoder
from fmri2img.models.projection_head import ContrastiveProjectionHead
from fmri2img.models.ncsnr_attention import NCSnrAttention
from fmri2img.models.encoders import ResidualMLPEncoder

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm -> Linear -> GELU -> Drop -> Linear -> Drop + skip."""

    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MLPEncoder(nn.Module):
    """
    Multi-layer perceptron encoder with optional residual blocks.

    When ``use_residual=True``, a :class:`ResidualBlock` is inserted after
    each projection layer, following the MindEye architecture pattern.

    Args:
        input_dim: Input dimensionality (fMRI voxels)
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ("relu", "gelu")
        dropout: Dropout probability
        use_residual: Insert a ResidualBlock after each projection layer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: str = "relu",
        dropout: float = 0.1,
        use_residual: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]

        layers: list[nn.Module] = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(dropout),
            ])
            if use_residual:
                layers.append(ResidualBlock(hidden_dim, dropout=dropout))
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        logger.info(
            "MLPEncoder: %d → %s (activation=%s, dropout=%.2f, residual=%s)",
            input_dim, hidden_dims, activation, dropout, use_residual,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, input_dim)
        Returns:
            h: Latent (B, hidden_dims[-1])
        """
        return self.encoder(x)


class DeterministicDecoder(nn.Module):
    """
    Deterministic decoder: outputs single prediction.
    
    Args:
        input_dim: Latent dimension from encoder
        output_dim: Output dimension (CLIP embedding size, typically 768)
        hidden_dims: Hidden layer dimensions
        activation: Activation function
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "relu",
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if hidden_dims is None or len(hidden_dims) == 0:
            # Simple linear projection
            self.decoder = nn.Linear(input_dim, output_dim)
        else:
            # MLP decoder
            layers = []
            in_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                in_dim = hidden_dim
            
            # Final projection
            layers.append(nn.Linear(in_dim, output_dim))
            self.decoder = nn.Sequential(*layers)
        
        logger.info(f"DeterministicDecoder: {input_dim} → {output_dim} (hidden={hidden_dims})")
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Latent (B, input_dim)
        Returns:
            pred: Predictions (B, output_dim), L2-normalized
        """
        pred = self.decoder(h)
        # L2 normalize (CLIP embeddings are normalized)
        pred = nn.functional.normalize(pred, p=2, dim=-1)
        return pred


class GaussianDecoder(nn.Module):
    """
    Gaussian decoder: outputs mean (mu) and log-variance (logvar).
    
    Architecture:
        Shared layers → split into mu_head and logvar_head
    
    Args:
        input_dim: Latent dimension from encoder
        output_dim: Output dimension (CLIP embedding size)
        hidden_dims: Hidden layer dimensions for shared backbone
        activation: Activation function
        dropout: Dropout probability
        logvar_min: Minimum log-variance (for clamping)
        logvar_max: Maximum log-variance (for clamping)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "relu",
        dropout: float = 0.1,
        logvar_min: float = -10.0,
        logvar_max: float = 5.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        
        # Shared backbone
        if hidden_dims is None or len(hidden_dims) == 0:
            # No hidden layers - direct projection
            self.shared_backbone = nn.Identity()
            backbone_out_dim = input_dim
        else:
            layers = []
            in_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                in_dim = hidden_dim
            
            self.shared_backbone = nn.Sequential(*layers)
            backbone_out_dim = hidden_dims[-1]
        
        # Prediction heads
        self.mu_head = nn.Linear(backbone_out_dim, output_dim)
        self.logvar_head = nn.Linear(backbone_out_dim, output_dim)
        
        logger.info(f"GaussianDecoder: {input_dim} → (mu, logvar) {output_dim} (hidden={hidden_dims})")
        logger.info(f"  Logvar clamping: [{logvar_min}, {logvar_max}]")
    
    def forward(
        self,
        h: torch.Tensor,
        return_normalized: bool = True,
        clamp_logvar: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Latent (B, input_dim)
            return_normalized: Whether to L2-normalize mu
            clamp_logvar: Whether to clamp logvar
        
        Returns:
            mu: Mean predictions (B, output_dim), L2-normalized if return_normalized=True
            logvar: Log-variance (B, output_dim), clamped if clamp_logvar=True
        """
        # Shared features
        features = self.shared_backbone(h)
        
        # Predict mu and logvar
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        
        # Normalize mu (CLIP embeddings are normalized)
        if return_normalized:
            mu = nn.functional.normalize(mu, p=2, dim=-1)
        
        # Clamp logvar for numerical stability
        if clamp_logvar:
            logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
        
        return mu, logvar


class VonMisesFisherDecoderLegacy(nn.Module):
    """
    Legacy vMF decoder that outputs LOG kappa (clamped).

    Used when config specifies log_kappa_min / log_kappa_max instead of
    kappa_min / kappa_max. The training loop must exponentiate kappa
    before passing it to losses that expect linear-scale concentration.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        log_kappa_min: float = -2.0,
        log_kappa_max: float = 8.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.log_kappa_min = log_kappa_min
        self.log_kappa_max = log_kappa_max

        if hidden_dims is None or len(hidden_dims) == 0:
            self.shared_backbone = nn.Identity()
            backbone_out_dim = input_dim
        else:
            layers: list[nn.Module] = []
            in_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                in_dim = hidden_dim
            self.shared_backbone = nn.Sequential(*layers)
            backbone_out_dim = hidden_dims[-1]

        self.mu_head = nn.Linear(backbone_out_dim, output_dim)
        self.kappa_head = nn.Linear(backbone_out_dim, 1)

        logger.info(
            f"VonMisesFisherDecoderLegacy: {input_dim} -> (mu, log_kappa) {output_dim} "
            f"(hidden={hidden_dims}, log_kappa=[{log_kappa_min}, {log_kappa_max}])"
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared_backbone(h)
        mu = nn.functional.normalize(self.mu_head(features), p=2, dim=-1)
        raw = self.kappa_head(features)
        log_kappa = self.log_kappa_min + (self.log_kappa_max - self.log_kappa_min) * torch.sigmoid(raw)
        return mu, log_kappa


class UnifiedModel(nn.Module):
    """
    Unified model supporting deterministic, Gaussian, vMF, and vMF-DCF outputs.

    Architecture:
        fMRI → Encoder → Latent → Decoder → output

    Supported (type, encoder_type) combinations:
        - deterministic + mlp
        - gaussian + mlp
        - vmf + mlp
        - vmf + residual_mlp  (MindEye-style: project once then N residual blocks)
        - vmf + roi_transformer
        - vmf_dcf + roi_transformer  (raises ValueError with mlp)
    """
    
    def __init__(self, config: Dict[str, Any], roi_indices: Optional[Dict[str, Any]] = None,
                 ncsnr: Optional[Any] = None):
        super().__init__()
        self.model_type = config.get("type", "deterministic")
        self.vmf_output_is_log = False
        self._return_per_roi = False
        self._last_dcf_extras: Dict[str, Any] = {}

        encoder_cfg = config.get("encoder", {})
        decoder_cfg = config.get("decoder", {})
        encoder_type = encoder_cfg.get("encoder_type", "mlp")

        # --- Encoder ---
        if encoder_type == "multi_subject_roi_transformer":
            subject_roi_dims = encoder_cfg.get("subject_roi_dims", {})
            subject_roi_indices = roi_indices  # dict-of-dicts when multi-subject
            if not subject_roi_dims and not subject_roi_indices:
                raise ValueError(
                    "multi_subject_roi_transformer requires "
                    "encoder.subject_roi_dims or roi_indices"
                )
            self.encoder = MultiSubjectROITransformer(
                subject_roi_dims=subject_roi_dims,
                subject_roi_indices=subject_roi_indices,
                d_model=encoder_cfg.get("d_model", 768),
                nhead=encoder_cfg.get("nhead", 12),
                num_layers=encoder_cfg.get("num_layers", 6),
                dim_feedforward=encoder_cfg.get("dim_feedforward"),
                dropout=encoder_cfg.get("dropout", 0.1),
                activation=encoder_cfg.get("activation", "gelu"),
                drop_path_rate=encoder_cfg.get("drop_path_rate", 0.0),
                roi_token_dropout=encoder_cfg.get("roi_token_dropout", 0.0),
            )
        elif encoder_type == "roi_transformer":
            roi_dims = encoder_cfg.get("roi_dims")
            if roi_dims is None and roi_indices is None:
                raise ValueError("roi_transformer encoder requires roi_dims or roi_indices")
            self.encoder = ROITransformerEncoder(
                roi_dims=roi_dims or {},
                d_model=encoder_cfg.get("d_model", 512),
                nhead=encoder_cfg.get("nhead", 8),
                num_layers=encoder_cfg.get("num_layers", 4),
                dropout=encoder_cfg.get("dropout", 0.1),
                activation=encoder_cfg.get("activation", "gelu"),
                roi_indices=roi_indices,
                dim_feedforward=encoder_cfg.get("dim_feedforward"),
                drop_path_rate=encoder_cfg.get("drop_path_rate", 0.0),
                roi_token_dropout=encoder_cfg.get("roi_token_dropout", 0.0),
            )
        elif encoder_type == "residual_mlp":
            input_dim = encoder_cfg.get("input_dim")
            if input_dim is None:
                raise ValueError("encoder.input_dim must be specified")
            self.encoder = ResidualMLPEncoder(
                input_dim=input_dim,
                latent_dim=encoder_cfg.get("latent_dim", 2048),
                n_blocks=encoder_cfg.get("n_blocks", 4),
                dropout=encoder_cfg.get("dropout", 0.15),
            )
        else:
            input_dim = encoder_cfg.get("input_dim")
            if input_dim is None:
                raise ValueError("encoder.input_dim must be specified")
            self.encoder = MLPEncoder(
                input_dim=input_dim,
                hidden_dims=encoder_cfg.get("hidden_dims", [4096, 2048, 1024]),
                activation=encoder_cfg.get("activation", "relu"),
                dropout=encoder_cfg.get("dropout", 0.1),
                use_residual=encoder_cfg.get("use_residual", False),
            )

        # --- Optional NCSNR voxel attention (V11) ---
        ncsnr_cfg = config.get("ncsnr_attention", {})
        if ncsnr_cfg.get("enabled", False) and encoder_type == "mlp":
            input_dim_for_ncsnr = encoder_cfg.get("input_dim")
            self.ncsnr_attention = NCSnrAttention(
                n_voxels=input_dim_for_ncsnr or 0,
                ncsnr=ncsnr,
            )
        else:
            self.ncsnr_attention = None

        latent_dim = self.encoder.output_dim
        output_dim = decoder_cfg.get("output_dim", 768)

        # --- Decoder ---
        if self.model_type == "deterministic":
            self.decoder = DeterministicDecoder(
                input_dim=latent_dim,
                output_dim=output_dim,
                hidden_dims=decoder_cfg.get("hidden_dims", [1536]),
                activation=decoder_cfg.get("activation", "relu"),
                dropout=decoder_cfg.get("dropout", 0.1),
            )
        elif self.model_type == "gaussian":
            self.decoder = GaussianDecoder(
                input_dim=latent_dim,
                output_dim=output_dim,
                hidden_dims=decoder_cfg.get("hidden_dims", [1536]),
                activation=decoder_cfg.get("activation", "relu"),
                dropout=decoder_cfg.get("dropout", 0.1),
                logvar_min=decoder_cfg.get("logvar_min", -10.0),
                logvar_max=decoder_cfg.get("logvar_max", 5.0),
            )
        elif self.model_type == "vmf":
            uses_log_kappa = "log_kappa_min" in decoder_cfg or "log_kappa_max" in decoder_cfg
            if uses_log_kappa:
                self.vmf_output_is_log = True
                self.decoder = VonMisesFisherDecoderLegacy(
                    input_dim=latent_dim,
                    output_dim=output_dim,
                    hidden_dims=decoder_cfg.get("hidden_dims", [1536]),
                    activation=decoder_cfg.get("activation", "gelu"),
                    dropout=decoder_cfg.get("dropout", 0.1),
                    log_kappa_min=decoder_cfg.get("log_kappa_min", -2.0),
                    log_kappa_max=decoder_cfg.get("log_kappa_max", 8.0),
                )
            else:
                self.vmf_output_is_log = False
                kappa_mode = decoder_cfg.get("kappa_mode", "bounded_sigmoid")
                _num_tokens = decoder_cfg.get("num_tokens", 0)
                _token_dim = decoder_cfg.get("token_dim", 768)
                _regression_head = decoder_cfg.get("regression_head", False)
                self.decoder = VonMisesFisherDecoder(
                    input_dim=latent_dim,
                    output_dim=output_dim,
                    hidden_dims=decoder_cfg.get("hidden_dims", [1536]),
                    activation=decoder_cfg.get("activation", "gelu"),
                    dropout=decoder_cfg.get("dropout", 0.1),
                    kappa_min=decoder_cfg.get("kappa_min", 1e-3),
                    kappa_max=decoder_cfg.get("kappa_max", 500.0),
                    kappa_mode=kappa_mode,
                    num_tokens=_num_tokens,
                    token_dim=_token_dim,
                    regression_head=_regression_head,
                )
        elif self.model_type == "vmf_triple":
            from fmri2img.models.triple_head_decoder import TripleHeadVMFDecoder
            self.vmf_output_is_log = False
            _retrieval_dim = decoder_cfg.get("retrieval_dim", 768)
            # Always compute total from num_tokens * per-token dim.
            # decoder_cfg["token_dim"] is the per-token dim (768) set by
            # auto-fill; "token_dim_per" is an explicit alias in V30 configs.
            _n_tok = decoder_cfg.get("num_tokens", 257)
            _d_tok = decoder_cfg.get("token_dim_per",
                                     decoder_cfg.get("token_dim", 768))
            _token_dim_total = _n_tok * _d_tok
            _perc_enabled = decoder_cfg.get("perceptual_enabled", False)
            _perc_dim = decoder_cfg.get("perceptual_dim", 768)
            _rerank_enabled = decoder_cfg.get("rerank_enabled", False)
            _rerank_dim = decoder_cfg.get("rerank_dim", 1024)
            self.decoder = TripleHeadVMFDecoder(
                input_dim=latent_dim,
                retrieval_dim=_retrieval_dim,
                token_dim=_token_dim_total,
                rerank_dim=_rerank_dim,
                rerank_enabled=_rerank_enabled,
                regression_enabled=decoder_cfg.get("regression_enabled", True),
                perceptual_dim=_perc_dim,
                perceptual_enabled=_perc_enabled,
                hidden_dims=decoder_cfg.get("hidden_dims", [2048]),
                activation=decoder_cfg.get("activation", "gelu"),
                dropout=decoder_cfg.get("dropout", 0.1),
                kappa_min=decoder_cfg.get("kappa_min", 1e-3),
                kappa_max=decoder_cfg.get("kappa_max", 500.0),
                kappa_mode=decoder_cfg.get("kappa_mode", "softplus"),
                retrieval_num_hypotheses=decoder_cfg.get("retrieval_num_hypotheses", 1),
            )
        elif self.model_type == "dense_vmf_hybrid":
            self.vmf_output_is_log = False
            _retrieval_dim = decoder_cfg.get("retrieval_dim", 768)
            _n_tok = decoder_cfg.get("num_tokens", 257)
            _d_tok = decoder_cfg.get("token_dim_per", decoder_cfg.get("token_dim", 768))
            _token_dim_total = _n_tok * _d_tok
            self.decoder = DenseVMFHybridDecoder(
                input_dim=latent_dim,
                retrieval_dim=_retrieval_dim,
                token_dim=_token_dim_total,
                rerank_dim=decoder_cfg.get("rerank_dim", 2048),
                rerank_enabled=decoder_cfg.get("rerank_enabled", False),
                regression_enabled=decoder_cfg.get("regression_enabled", False),
                hidden_dims=decoder_cfg.get("hidden_dims", [2048]),
                activation=decoder_cfg.get("activation", "gelu"),
                dropout=decoder_cfg.get("dropout", 0.1),
                kappa_min=decoder_cfg.get("kappa_min", 1e-3),
                kappa_max=decoder_cfg.get("kappa_max", 500.0),
                kappa_mode=decoder_cfg.get("kappa_mode", "softplus"),
            )
        elif self.model_type == "scfr_vmf":
            self.vmf_output_is_log = False
            scfr_cfg = config.get("scfr", {})
            _cross_dims = config.get("cross_subject", {}).get("subject_voxel_dims", {})
            _num_subjects = int(scfr_cfg.get("num_subjects", max(len(_cross_dims), 1)))
            self.decoder = SubjectConditionedFactorizedVMFDecoder(
                input_dim=latent_dim,
                retrieval_dim=decoder_cfg.get("retrieval_dim", 768),
                num_subjects=_num_subjects,
                total_latent_dim=scfr_cfg.get("total_latent_dim", 4096),
                z_vis_dim=scfr_cfg.get("z_vis_dim", 2048),
                z_subj_dim=scfr_cfg.get("z_subj_dim", 2048),
                subject_embedding_dim=scfr_cfg.get("subject_embedding_dim", 64),
                subject_condition_hidden_dim=scfr_cfg.get("subject_condition_hidden_dim", 128),
                subject_condition_dropout=scfr_cfg.get("subject_condition_dropout", 0.0),
                factor_hidden_dims=scfr_cfg.get("factor_hidden_dims", []),
                subject_head_hidden_dims=scfr_cfg.get("subject_head_hidden_dims", [512]),
                adversarial_enabled=scfr_cfg.get("adversarial_enabled", False),
                adversarial_head_hidden_dims=scfr_cfg.get("adversarial_head_hidden_dims", [512]),
                grl_start_epoch=scfr_cfg.get("grl_start_epoch", 10),
                grl_ramp_epochs=scfr_cfg.get("grl_ramp_epochs", 8),
                grl_lambda_max=scfr_cfg.get("grl_lambda_max", 0.2),
                activation=decoder_cfg.get("activation", "gelu"),
                dropout=decoder_cfg.get("dropout", 0.1),
                kappa_min=decoder_cfg.get("kappa_min", 1e-3),
                kappa_max=decoder_cfg.get("kappa_max", 500.0),
                kappa_mode=decoder_cfg.get("kappa_mode", "softplus"),
            )
            self.num_subjects = _num_subjects
        elif self.model_type == "vmf_dcf":
            if encoder_type not in ("roi_transformer", "multi_subject_roi_transformer"):
                raise ValueError(
                    "vmf_dcf model type requires encoder_type='roi_transformer' or "
                    f"'multi_subject_roi_transformer', got '{encoder_type}'"
                )
            self.vmf_output_is_log = False
            dcf_cfg = decoder_cfg.get("dcf", {})
            self._return_per_roi = dcf_cfg.get("return_per_roi", True)
            kappa_mode = decoder_cfg.get("kappa_mode", "bounded_sigmoid")
            n_rois = getattr(self.encoder, "n_rois", 1)
            self.decoder = ROIDCFDecoder(
                d_model=latent_dim,
                output_dim=output_dim,
                n_rois=n_rois,
                shared=dcf_cfg.get("shared_heads", True),
                hidden_dim=dcf_cfg.get("hidden_dim"),
                kappa_min=decoder_cfg.get("kappa_min", 1e-3),
                kappa_max=decoder_cfg.get("kappa_max", 500.0),
                kappa_mode=kappa_mode,
                dropout=decoder_cfg.get("dropout", 0.1),
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # --- Optional contrastive projection head (V9) ---
        proj_cfg = config.get("projection_head", {})
        if proj_cfg.get("enabled", False):
            self.projection_head = ContrastiveProjectionHead(
                d_model=output_dim,
                hidden_dim=proj_cfg.get("hidden_dim", 2048),
                out_dim=proj_cfg.get("out_dim", output_dim),
                dropout=proj_cfg.get("dropout", 0.1),
            )
        else:
            self.projection_head = None

        # --- Cross-subject linear adapters (V25b) ---
        cross_cfg = config.get("cross_subject", {})
        self.subject_adapters: Optional[nn.ModuleDict] = None
        self._canonical_subject: Optional[str] = None
        self._canonical_voxels: int = 0
        self._subj_int_to_id: Dict[int, str] = {}
        self._subject_voxel_dims: Dict[str, int] = {}

        if cross_cfg.get("enabled", False):
            subject_voxel_dims = cross_cfg.get("subject_voxel_dims", {})
            canonical = cross_cfg.get("canonical_subject", "subj01")
            if subject_voxel_dims and canonical in subject_voxel_dims:
                self._canonical_subject = canonical
                self._canonical_voxels = subject_voxel_dims[canonical]
                self._subject_voxel_dims = subject_voxel_dims
                # Build int→id mapping (same order as MultiSubjectPreextractedDataset)
                sorted_subjects = sorted(subject_voxel_dims.keys())
                self._subj_int_to_id = {i: s for i, s in enumerate(sorted_subjects)}

                adapters = {}
                for subj, n_vox in subject_voxel_dims.items():
                    if subj == canonical:
                        continue
                    adapters[subj] = nn.Linear(n_vox, self._canonical_voxels, bias=False)
                self.subject_adapters = nn.ModuleDict(adapters)
                logger.info(
                    "Cross-subject adapters (V25b): canonical=%s (%d vox), "
                    "adapters for %s",
                    canonical, self._canonical_voxels,
                    list(adapters.keys()),
                )

        logger.info(f"UnifiedModel created: type={self.model_type}")
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Keyword Args:
            subject_ids: Required when encoder is ``MultiSubjectROITransformer``.
                         Integer tensor or scalar identifying each sample's subject.

        Returns:
            deterministic: (pred, None)
            gaussian:      (mu, logvar)
            vmf:           (mu, kappa_or_log_kappa)
            vmf_dcf:       (mu_fused, kappa_consensus)
                           Also stores per-ROI extras in self._last_dcf_extras.
        """
        subject_ids = kwargs.pop("subject_ids", None)
        _is_multi = isinstance(self.encoder, MultiSubjectROITransformer)

        # --- Cross-subject adapter routing (V25b) ---
        # Maps non-canonical subjects into canonical voxel space before encoder
        if self.subject_adapters is not None and subject_ids is not None:
            B = x.shape[0]
            adapted = torch.zeros(
                B, self._canonical_voxels, device=x.device, dtype=x.dtype,
            )
            for subj_int, subj_id in self._subj_int_to_id.items():
                mask = (subject_ids == subj_int)
                if not mask.any():
                    continue
                subj_x = x[mask]
                if subj_id == self._canonical_subject:
                    # Canonical subject: slice to canonical dim (strip padding)
                    adapted[mask] = subj_x[:, :self._canonical_voxels]
                else:
                    # Non-canonical: slice to subject's voxel count, project
                    # .to(adapted.dtype) ensures bf16 AMP output matches float32 dest
                    n_vox = self._subject_voxel_dims[subj_id]
                    adapted[mask] = self.subject_adapters[subj_id](
                        subj_x[:, :n_vox]
                    ).to(adapted.dtype)
            x = adapted

        if self.ncsnr_attention is not None:
            x = self.ncsnr_attention(x)

        if self.model_type == "vmf_dcf":
            if _is_multi:
                enc_out = self.encoder(x, subject_ids, return_roi_tokens=True)
            else:
                enc_out = self.encoder(x, return_roi_tokens=True)
            if self._return_per_roi:
                mu_fused, kappa_consensus, per_roi_mus, per_roi_kappas, delta = (
                    self.decoder(enc_out.roi_tokens, enc_out.cls_to_roi_alpha,
                                 return_per_roi=True)
                )
                self._last_dcf_extras = {
                    "per_roi_mus": per_roi_mus,
                    "per_roi_kappas": per_roi_kappas,
                    "delta": delta,
                    "cls_to_roi_alpha": enc_out.cls_to_roi_alpha,
                }
            else:
                mu_fused, kappa_consensus = self.decoder(
                    enc_out.roi_tokens, enc_out.cls_to_roi_alpha,
                )
            return mu_fused, kappa_consensus

        if _is_multi:
            h = self.encoder(x, subject_ids)
        else:
            h = self.encoder(x)

        if self.model_type == "deterministic":
            return self.decoder(h), None
        elif self.model_type == "gaussian":
            return self.decoder(h, **kwargs)
        elif self.model_type == "vmf_triple":
            from fmri2img.models.triple_head_decoder import TripleHeadOutput
            dec_out: TripleHeadOutput = self.decoder(h)
            self._last_compact_pred = dec_out.mu
            self._last_compact_component_mu = dec_out.component_mu
            self._last_compact_component_kappa = dec_out.component_kappa
            self._last_compact_component_logits = dec_out.component_logits
            self._last_rich_pred = dec_out.reg_pred
            self._last_rerank_pred = dec_out.rerank_pred
            self._last_perc_pred = dec_out.perc_pred
            self._last_reg_pred = dec_out.reg_pred
            return dec_out.mu, dec_out.kappa
        elif self.model_type == "dense_vmf_hybrid":
            dec_out = self.decoder(h)
            self._last_dense_pred = dec_out.dense_pred
            self._last_compact_pred = dec_out.dense_pred
            self._last_vmf_pred = dec_out.vmf_mu
            self._last_vmf_kappa = dec_out.vmf_kappa
            self._last_compact_component_mu = None
            self._last_compact_component_kappa = None
            self._last_compact_component_logits = None
            self._last_rich_pred = dec_out.reg_pred
            self._last_rerank_pred = dec_out.rerank_pred
            self._last_reg_pred = dec_out.reg_pred
            return dec_out.dense_pred, {
                "vmf_mu": dec_out.vmf_mu,
                "kappa": dec_out.vmf_kappa,
            }
        elif self.model_type == "scfr_vmf":
            dec_out: SCFRVMFOutput = self.decoder(h, subject_ids=subject_ids)
            self._last_compact_pred = dec_out.mu
            self._last_vmf_pred = dec_out.mu
            self._last_vmf_kappa = dec_out.kappa
            self._last_compact_component_mu = None
            self._last_compact_component_kappa = None
            self._last_compact_component_logits = None
            self._last_rich_pred = None
            self._last_rerank_pred = None
            self._last_reg_pred = None
            self._last_z_vis = dec_out.z_vis
            self._last_z_subj = dec_out.z_subj
            self._last_subject_logits = dec_out.subject_logits
            self._last_adv_subject_logits = dec_out.adv_subject_logits
            self._last_adv_lambda = float(dec_out.adv_lambda)
            return dec_out.mu, dec_out.kappa
        else:  # vmf, vmf_dcf handled above
            dec_out = self.decoder(h)
            if len(dec_out) == 3:
                mu, kappa, reg_pred = dec_out
                self._last_reg_pred = reg_pred
                return mu, kappa
            else:
                self._last_reg_pred = None
                return dec_out

    def set_current_epoch(self, epoch: int) -> None:
        """Forward epoch state to decoders that need scheduled behaviour."""
        if hasattr(self.decoder, "set_current_epoch"):
            self.decoder.set_current_epoch(epoch)
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            "type": self.model_type,
            "encoder": {
                "input_dim": getattr(self.encoder, "input_dim", None),
                "output_dim": self.encoder.output_dim,
            },
            "decoder": {
                "input_dim": getattr(self.decoder, "input_dim", None),
                "output_dim": self.decoder.output_dim,
            },
        }


class PCDModel(nn.Module):
    """Wrapper for PredictiveCorticalDecoder with UnifiedModel-compatible interface.

    Presents as ``model_type="vmf"`` to the training loop so all vMF loss
    infrastructure works without modification.  Stores PCD-specific extras
    (per-level outputs, prediction errors, level kappas) for neuroscience
    analysis.

    Args:
        config: Full model configuration dict (``config["model"]``).
        roi_indices: Per-ROI voxel index arrays (single-subject) or
            dict-of-dicts (multi-subject).
    """

    def __init__(self, config: Dict[str, Any], roi_indices: Optional[Dict[str, Any]] = None):
        super().__init__()
        from fmri2img.models.predictive_cortical_decoder import PredictiveCorticalDecoder

        self.model_type = "vmf"
        self.architecture_type = "pcd"
        self.vmf_output_is_log = False
        self._return_per_roi = False
        self._last_pcd_extras: Dict[str, Any] = {}

        encoder_cfg = config.get("encoder", {})
        decoder_cfg = config.get("decoder", {})
        pcd_cfg = config.get("pcd", {})

        d_model = pcd_cfg.get("d_model", encoder_cfg.get("d_model", 768))
        output_dim = decoder_cfg.get("output_dim", 768)
        ablation_mode = pcd_cfg.get("ablation_mode", "full")
        bottleneck_dim = pcd_cfg.get("bottleneck_dim", encoder_cfg.get("bottleneck_dim", None))

        # Determine single vs multi-subject
        cross_cfg = config.get("cross_subject", {})
        is_multi = cross_cfg.get("enabled", False) and roi_indices is not None

        if is_multi and isinstance(roi_indices, dict):
            first_val = next(iter(roi_indices.values()))
            if isinstance(first_val, dict):
                subject_roi_indices = {
                    subj: OrderedDict(indices) if not isinstance(indices, OrderedDict) else indices
                    for subj, indices in roi_indices.items()
                }
                self.pcd = PredictiveCorticalDecoder(
                    subject_roi_indices=subject_roi_indices,
                    d_model=d_model,
                    nhead=pcd_cfg.get("nhead", encoder_cfg.get("nhead", 12)),
                    layers_per_level=pcd_cfg.get("layers_per_level", 2),
                    dropout=pcd_cfg.get("dropout", encoder_cfg.get("dropout", 0.1)),
                    output_dim=output_dim,
                    kappa_min=decoder_cfg.get("kappa_min", 1e-3),
                    kappa_max=decoder_cfg.get("kappa_max", 500.0),
                    kappa_mode=decoder_cfg.get("kappa_mode", "softplus"),
                    enable_per_level_kappa=pcd_cfg.get("enable_per_level_kappa", True),
                    ablation_mode=ablation_mode,
                    bottleneck_dim=bottleneck_dim,
                )
            else:
                self.pcd = PredictiveCorticalDecoder(
                    roi_indices=OrderedDict(roi_indices) if not isinstance(roi_indices, OrderedDict) else roi_indices,
                    d_model=d_model,
                    nhead=pcd_cfg.get("nhead", encoder_cfg.get("nhead", 12)),
                    layers_per_level=pcd_cfg.get("layers_per_level", 2),
                    dropout=pcd_cfg.get("dropout", encoder_cfg.get("dropout", 0.1)),
                    output_dim=output_dim,
                    kappa_min=decoder_cfg.get("kappa_min", 1e-3),
                    kappa_max=decoder_cfg.get("kappa_max", 500.0),
                    kappa_mode=decoder_cfg.get("kappa_mode", "softplus"),
                    enable_per_level_kappa=pcd_cfg.get("enable_per_level_kappa", True),
                    ablation_mode=ablation_mode,
                    bottleneck_dim=bottleneck_dim,
                )
        else:
            self.pcd = PredictiveCorticalDecoder(
                roi_indices=OrderedDict(roi_indices) if roi_indices and not isinstance(roi_indices, OrderedDict) else roi_indices,
                d_model=d_model,
                nhead=pcd_cfg.get("nhead", encoder_cfg.get("nhead", 12)),
                layers_per_level=pcd_cfg.get("layers_per_level", 2),
                dropout=pcd_cfg.get("dropout", encoder_cfg.get("dropout", 0.1)),
                output_dim=output_dim,
                kappa_min=decoder_cfg.get("kappa_min", 1e-3),
                kappa_max=decoder_cfg.get("kappa_max", 500.0),
                kappa_mode=decoder_cfg.get("kappa_mode", "softplus"),
                enable_per_level_kappa=pcd_cfg.get("enable_per_level_kappa", True),
                ablation_mode=ablation_mode,
                bottleneck_dim=bottleneck_dim,
            )

        # Optional contrastive projection head
        proj_cfg = config.get("projection_head", {})
        if proj_cfg.get("enabled", False):
            self.projection_head = ContrastiveProjectionHead(
                d_model=output_dim,
                hidden_dim=proj_cfg.get("hidden_dim", 2048),
                out_dim=proj_cfg.get("out_dim", output_dim),
                dropout=proj_cfg.get("dropout", 0.1),
            )
        else:
            self.projection_head = None

        logger.info("PCDModel created: ablation=%s, multi_subject=%s", ablation_mode, is_multi)

    @property
    def decoder(self):
        """Proxy to the internal vMF decoder for training-loop compatibility."""
        return self.pcd.vmf_decoder

    @property
    def encoder(self):
        """Proxy to the PCD module itself (acts as the encoder)."""
        return self.pcd

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, kappa) compatible with the vMF training loop."""
        subject_ids = kwargs.pop("subject_ids", None)
        pcd_out = self.pcd(x, subject_ids=subject_ids, return_details=True)

        self._last_pcd_extras = {
            "level_outputs": pcd_out.level_outputs,
            "prediction_errors": pcd_out.prediction_errors,
            "level_kappas": pcd_out.level_kappas,
            "level_weights": pcd_out.level_weights,
        }

        return pcd_out.mu, pcd_out.kappa

    def set_current_epoch(self, epoch: int) -> None:
        pass

    def get_config(self) -> Dict[str, Any]:
        return {"type": "pcd", "architecture": "predictive_cortical_decoder"}


def create_model(
    config: Dict[str, Any],
    roi_indices: Optional[Dict[str, Any]] = None,
    ncsnr: Optional[Any] = None,
) -> "nn.Module":
    """
    Factory function to create model from config.
    
    Args:
        config: Model configuration dict
        roi_indices: Per-ROI voxel index arrays for ROITransformerEncoder.
            When provided, overrides hardcoded ``roi_dims`` from config.
        ncsnr: Per-voxel NCSNR array for NCSnrAttention (V11).
    
    Returns:
        model: UnifiedModel, PCDModel, or NeuroBridgeOTModel instance
    
    Example:
        >>> config = {
        ...     "type": "gaussian",
        ...     "encoder": {"input_dim": 15724, "hidden_dims": [4096, 2048, 1024]},
        ...     "decoder": {"output_dim": 768, "hidden_dims": [1536]}
        ... }
        >>> model = create_model(config)
    """
    if config.get("type") == "neurobridge_ot":
        from fmri2img.models.neurobridge_ot.model import NeuroBridgeOTModel
        return NeuroBridgeOTModel(config)
    if config.get("type") == "pcd":
        return PCDModel(config, roi_indices=roi_indices)
    return UnifiedModel(config, roi_indices=roi_indices, ncsnr=ncsnr)


def load_model(checkpoint_path: Path, device: str = "cpu") -> UnifiedModel:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded UnifiedModel
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config
    if "config" in checkpoint:
        config = checkpoint["config"]
    elif "model_config" in checkpoint:
        config = checkpoint["model_config"]
    else:
        raise KeyError("No config found in checkpoint")
    
    # Create model
    model = create_model(config)
    
    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        raise KeyError("No state_dict found in checkpoint")
    
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path} (type={model.model_type})")
    return model
