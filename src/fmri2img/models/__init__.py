"""fmri2img.models

Model implementations for fMRI-to-CLIP encoding.

Some modules depend on optional heavy dependencies (notably ``torchvision``).
To keep lightweight tooling and unit tests runnable in minimal environments,
we intentionally avoid importing those submodules at import-time.

Import what you need from the concrete module, for example::

    from fmri2img.models.ridge import RidgeEncoder
    from fmri2img.models.unified_model import create_model
    from fmri2img.training.losses import compose_loss
"""

from __future__ import annotations

__all__: list[str] = []
