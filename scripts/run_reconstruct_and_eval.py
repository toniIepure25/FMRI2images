#!/usr/bin/env python3
"""
Backward-compatible wrapper for `fmri2img.eval.run_reconstruct_and_eval`.

This script simply forwards all command-line arguments to the library entrypoint
located at `src/fmri2img/eval/run_reconstruct_and_eval.py`. Keeping this wrapper
in `scripts/` preserves existing Makefile and README invocations.
"""

from fmri2img.eval.run_reconstruct_and_eval import main


if __name__ == "__main__":
    raise SystemExit(main())
