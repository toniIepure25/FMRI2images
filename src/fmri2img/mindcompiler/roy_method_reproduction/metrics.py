"""Structured per-voxel correlation diagnostics.

Receives both ``Y_true`` and ``Y_pred`` so it can distinguish *why* a correlation
is undefined (constant target vs constant prediction) rather than hiding it behind
a lone ``nanmean``. The engineering gate fails if the finite fraction drops below
``FINITE_FRACTION_MIN``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

FINITE_FRACTION_MIN = 0.90


@dataclass(frozen=True)
class MetricReport:
    total: int
    finite: int
    finite_fraction: float
    nan: int
    target_constant: int
    prediction_constant: int
    both_constant: int
    mean: float
    median: float
    std: float
    rmin: float
    rmax: float
    p5: float
    p25: float
    p75: float
    p95: float

    def as_dict(self) -> dict:
        return asdict(self)

    def passes(self, threshold: float = FINITE_FRACTION_MIN) -> bool:
        return self.finite_fraction >= threshold


def per_voxel_report(Y_true: np.ndarray, Y_pred: np.ndarray) -> MetricReport:
    """Per-column Pearson r with explicit constant-voxel accounting.

    Args:
        Y_true: (n, q) measured targets. Y_pred: (n, q) predictions.

    Returns:
        A :class:`MetricReport`. Undefined correlations are NaN and counted by
        cause; they are never replaced by a favorable value.
    """
    if Y_true.shape != Y_pred.shape:
        raise ValueError(f"shape mismatch {Y_true.shape} vs {Y_pred.shape}")
    yt = Y_true - Y_true.mean(axis=0)
    yp = Y_pred - Y_pred.mean(axis=0)
    st = np.sqrt((yt ** 2).sum(axis=0))
    sp = np.sqrt((yp ** 2).sum(axis=0))
    tgt_const = st < 1e-12
    pred_const = sp < 1e-12
    denom = st * sp
    num = (yt * yp).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denom > 1e-12, num / denom, np.nan)
    fin = np.isfinite(r)
    q = r.size

    def pct(p):
        return float(np.nanpercentile(r, p)) if fin.any() else float("nan")

    return MetricReport(
        total=int(q), finite=int(fin.sum()), finite_fraction=float(fin.mean()),
        nan=int((~fin).sum()),
        target_constant=int(tgt_const.sum()),
        prediction_constant=int(pred_const.sum()),
        both_constant=int((tgt_const & pred_const).sum()),
        mean=float(np.nanmean(r)) if fin.any() else float("nan"),
        median=float(np.nanmedian(r)) if fin.any() else float("nan"),
        std=float(np.nanstd(r)) if fin.any() else float("nan"),
        rmin=float(np.nanmin(r)) if fin.any() else float("nan"),
        rmax=float(np.nanmax(r)) if fin.any() else float("nan"),
        p5=pct(5), p25=pct(25), p75=pct(75), p95=pct(95),
    )
