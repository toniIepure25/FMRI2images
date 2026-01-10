#!/usr/bin/env python3
"""Fast smoke test for the new end-to-end workflow.

This is intentionally tiny and should run in <10s on most machines.
"""

from __future__ import annotations

import importlib


def main() -> int:
    # Import core package
    import fmri2img  # noqa: F401

    # Try importing a few high-level modules that tend to load deps
    for mod in [
        "fmri2img.utils.manifest",
        "fmri2img.data.nsd_index_builder",
    ]:
        importlib.import_module(mod)

    print("smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
