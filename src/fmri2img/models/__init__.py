"""fmri2img.models

This package contains model implementations used across the project.

Some modules depend on optional heavy dependencies (notably `torchvision`).
To keep lightweight tooling and unit tests runnable in minimal environments,
we intentionally avoid importing those submodules at import-time.

Import what you need from the concrete module, for example:

    from fmri2img.models.ridge import RidgeEncoder
    from fmri2img.models.losses import mse_loss
"""

from __future__ import annotations

__all__: list[str] = []
