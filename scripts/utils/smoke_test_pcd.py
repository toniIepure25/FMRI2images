"""Quick smoke test for the PCD architecture."""
import sys
sys.path.insert(0, "src")

from fmri2img.models.predictive_cortical_decoder import PredictiveCorticalDecoder, HIERARCHY_LEVELS, LEVEL_NAMES
from fmri2img.models.unified_model import create_model, PCDModel
from collections import OrderedDict
import torch
import numpy as np

roi_indices = OrderedDict()
for name in ['V1v','V1d','V2v','V2d','V3v','V3d','V3A','V3B','V4',
             'FFA1','FFA2','PPA','EBA','OFA','OPA','RSC','nsdgeneral_other']:
    roi_indices[name] = np.arange(100 * len(roi_indices), 100 * (len(roi_indices) + 1))

total_vox = sum(len(v) for v in roi_indices.values())
print(f"Total voxels: {total_vox}, ROIs: {len(roi_indices)}")

config = {
    "type": "pcd",
    "encoder": {"roi_dims": {k: len(v) for k, v in roi_indices.items()}},
    "decoder": {"output_dim": 768, "kappa_mode": "softplus"},
    "pcd": {"d_model": 256, "nhead": 8, "layers_per_level": 1, "dropout": 0.1},
    "projection_head": {"enabled": False},
}
model = create_model(config, roi_indices=roi_indices)
print(f"Model type: {model.model_type}, arch: {model.architecture_type}")
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

x = torch.randn(4, total_vox)
mu, kappa = model(x)
print(f"Output mu: {mu.shape}, kappa: {kappa.shape}")
print(f"mu norm: {mu.norm(dim=-1).mean():.4f}")
print(f"kappa range: [{kappa.min():.2f}, {kappa.max():.2f}]")

extras = model._last_pcd_extras
print(f"Level weights: {extras['level_weights'].shape}")
print(f"Level kappas: {extras['level_kappas'].shape}")
print(f"Prediction errors: {[e.shape if e is not None else None for e in extras['prediction_errors']]}")
print("SMOKE TEST PASSED")
