"""MINDIR PROX platform. The distribution is `fmri2img` (unchanged); this is the MINDIR platform layer.

Synthetic-only; no historical N=8 access; no biological claims. Scientific status is gated behind the frozen
O2.16 independent replication (see MINDIR_README.md / MINDIR_ROADMAP.md)."""

__version__ = "0.1.0-rc1"                       # MINDIR platform (not the fmri2img distribution version)
SCHEMA_VERSIONS = {
    "object": "prox-object/1.0.0",
    "metric": "prox-metric/1.0.0",
    "benchmark": "prox-benchmark/1.0.0",
    "artifact": "prox-artifact/1.0.0",
    "protocol": "prox-protocol/1.0.0",
    "provenance": "prox-provenance/1.0.0",
}
