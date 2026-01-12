from __future__ import annotations
import numpy as np
import torch
from typing import List, Dict, Any

def fmri_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Batch fmri volumes into (B, H, W, D) then move to (B, 1, H, W, D) for 3D convs, 
    or stack 1D features to (B, k) for MLP if PCA preprocessing was applied.
    """
    vols_np = [x["fmri"] for x in batch]
    import numpy as _np
    if vols_np[0].ndim == 1:
        # PCA features -> (B, k)
        x = torch.from_numpy(_np.stack(vols_np, axis=0)).to(torch.float32)
    else:
        # 3D volumes -> (B,1,H,W,D)
        vols = [torch.from_numpy(v).to(torch.float32) for v in vols_np]
        x = torch.stack(vols, dim=0).unsqueeze(1)
        
    nsd_ids = torch.tensor([x["nsdId"] for x in batch], dtype=torch.long)
    return {"fmri": x, "nsdId": nsd_ids}

class SimpleDataModule:
    def __init__(self, ds_train, ds_val=None, batch_size=2, num_workers=0, pin_memory=True, prefetch_factor=2):
        from torch.utils.data import DataLoader
        # Only use prefetch_factor if num_workers > 0
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": fmri_collate,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            
        self.train_loader = DataLoader(ds_train, **loader_kwargs)
        self.val_loader = None
        if ds_val is not None:
            self.val_loader = DataLoader(ds_val, **loader_kwargs)