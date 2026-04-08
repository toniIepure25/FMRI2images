"""Loss functions for fMRI-to-image reconstruction."""
from .infonce_queue import InfoNCEQueueLoss, HardNegativeInfoNCE, create_infonce_loss
from .scfr_losses import CrossCovarianceOrthogonalityLoss
from .softclip import SoftCLIPLoss, VMFSoftCLIPLoss
from .mixco import slerp

__all__ = [
    "InfoNCEQueueLoss", "HardNegativeInfoNCE", "create_infonce_loss",
    "SoftCLIPLoss", "VMFSoftCLIPLoss", "CrossCovarianceOrthogonalityLoss", "slerp",
]
