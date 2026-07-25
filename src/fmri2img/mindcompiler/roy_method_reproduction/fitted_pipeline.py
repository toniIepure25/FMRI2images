"""Fitted-pipeline lifecycle object with leakage prevention by construction.

The prior smoke enforced leakage safety by inline caller discipline (fit on train,
select on validation, evaluate test once). This object makes those guarantees
structural: preprocessing can only be fit from train-tagged arrays, hyperparameter
selection cannot see test, and the test set can be evaluated exactly once. Invalid
lifecycle transitions raise ``LifecycleError`` rather than silently proceeding.

State machine::

    CREATED -> POLICIES_RESOLVED -> TRAIN_PREPROCESSING_FITTED
    -> HYPERPARAMETERS_SELECTED -> FINAL_MODEL_FITTED -> TEST_EVALUATED -> SEALED

Reproducing the historical smoke through this object gives the same numbers as the
inline path, which is how it is validated as a drop-in replacement.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from fmri2img.mindcompiler.roy_method_reproduction.metrics import MetricReport, per_voxel_report
from fmri2img.mindcompiler.roy_method_reproduction.reduced_rank_ridge import (
    Standardizer, fit_reduced_rank, select_hyperparameters,
)


class LifecycleError(RuntimeError):
    """Raised on any invalid pipeline lifecycle transition."""


class State(str, Enum):
    CREATED = "CREATED"
    POLICIES_RESOLVED = "POLICIES_RESOLVED"
    TRAIN_PREPROCESSING_FITTED = "TRAIN_PREPROCESSING_FITTED"
    HYPERPARAMETERS_SELECTED = "HYPERPARAMETERS_SELECTED"
    FINAL_MODEL_FITTED = "FINAL_MODEL_FITTED"
    TEST_EVALUATED = "TEST_EVALUATED"
    SEALED = "SEALED"


#: The only preprocessing policy that reproduces the historical smoke.
HISTORICAL_POLICY = "HISTORICAL_SMOKE_TRAIN_ONLY_CENTERING_V1"


@dataclass
class FittedPipeline:
    """Leakage-safe reduced-rank ridge pipeline with an explicit lifecycle.

    Args:
        source_centering / target_centering: "train_only" or "none".
        source_scaling / target_scaling: "none" (historical) only, for now.
        rank_tie_break / ridge_tie_break: explicit selection policies.
        rank_max: cap on rank candidates (None = matrix-supported max).
        refit_policy: "train_only" (historical) or "train_plus_validation".
        voxel_hash / trial_table_hash: provenance the object carries.
    """

    source_centering: str = "train_only"
    source_scaling: str = "none"
    target_centering: str = "train_only"
    target_scaling: str = "none"
    rank_tie_break: str = "smallest_rank_at_threshold"
    ridge_tie_break: str = "argmax_validation"
    rank_max: Optional[int] = None
    refit_policy: str = "train_only"
    voxel_hash: str = ""
    trial_table_hash: str = ""

    state: State = field(default=State.CREATED, init=False)
    _sx: Optional[Standardizer] = field(default=None, init=False)
    _sy: Optional[Standardizer] = field(default=None, init=False)
    _lam: Optional[float] = field(default=None, init=False)
    _rank: Optional[int] = field(default=None, init=False)
    _W: Optional[np.ndarray] = field(default=None, init=False)
    _src_dim: Optional[int] = field(default=None, init=False)
    _tgt_dim: Optional[int] = field(default=None, init=False)
    _test_evals: int = field(default=0, init=False)
    _Xtr: Optional[np.ndarray] = field(default=None, init=False)
    _Ytr: Optional[np.ndarray] = field(default=None, init=False)

    # --- lifecycle ---------------------------------------------------------

    def resolve_policies(self) -> "FittedPipeline":
        if self.state != State.CREATED:
            raise LifecycleError(f"resolve_policies from {self.state}")
        for name, val, allowed in [
            ("source_centering", self.source_centering, {"train_only", "none"}),
            ("target_centering", self.target_centering, {"train_only", "none"}),
            ("source_scaling", self.source_scaling, {"none"}),
            ("target_scaling", self.target_scaling, {"none"}),
            ("refit_policy", self.refit_policy, {"train_only", "train_plus_validation"}),
        ]:
            if val not in allowed:
                raise LifecycleError(f"unresolved/implicit policy {name}={val!r}")
        self.state = State.POLICIES_RESOLVED
        return self

    def fit_preprocessing(self, X_train: np.ndarray, Y_train: np.ndarray) -> "FittedPipeline":
        """Fit centering on TRAIN ONLY. Passing validation/test here is impossible
        by contract: the method name and lifecycle position admit train only."""
        if self.state != State.POLICIES_RESOLVED:
            raise LifecycleError(f"fit_preprocessing from {self.state}")
        self._sx = Standardizer.fit(X_train, with_scaling=(self.source_scaling != "none")) \
            if self.source_centering == "train_only" else Standardizer(
                np.zeros(X_train.shape[1]), np.ones(X_train.shape[1]))
        self._sy = Standardizer.fit(Y_train, with_scaling=(self.target_scaling != "none")) \
            if self.target_centering == "train_only" else Standardizer(
                np.zeros(Y_train.shape[1]), np.ones(Y_train.shape[1]))
        self._src_dim, self._tgt_dim = X_train.shape[1], Y_train.shape[1]
        self._Xtr, self._Ytr = self._sx.transform(X_train), self._sy.transform(Y_train)
        self.state = State.TRAIN_PREPROCESSING_FITTED
        return self

    def _check_dims(self, X, Y=None):
        if X.shape[1] != self._src_dim:
            raise LifecycleError(f"source dim {X.shape[1]} != {self._src_dim}")
        if Y is not None and Y.shape[1] != self._tgt_dim:
            raise LifecycleError(f"target dim {Y.shape[1]} != {self._tgt_dim}")

    def select(self, X_val: np.ndarray, Y_val: np.ndarray) -> "FittedPipeline":
        if self.state != State.TRAIN_PREPROCESSING_FITTED:
            raise LifecycleError(f"select from {self.state}")
        self._check_dims(X_val, Y_val)
        Xva, Yva = self._sx.transform(X_val), self._sy.transform(Y_val)
        sel = select_hyperparameters(self._Xtr, self._Ytr, Xva, Yva,
                                     self.rank_tie_break, self.ridge_tie_break,
                                     r_max=self.rank_max)
        self._lam, self._rank = sel.lam, sel.rank
        self._val_score = sel.val_score
        self.state = State.HYPERPARAMETERS_SELECTED
        return self

    def fit_final(self, X_val=None, Y_val=None) -> "FittedPipeline":
        if self.state != State.HYPERPARAMETERS_SELECTED:
            raise LifecycleError(f"fit_final from {self.state}")
        if self.refit_policy == "train_only":
            Xf, Yf = self._Xtr, self._Ytr
        else:  # train_plus_validation (sensitivity; never the historical path)
            if X_val is None:
                raise LifecycleError("train_plus_validation refit needs validation data")
            Xf = np.vstack([self._Xtr, self._sx.transform(X_val)])
            Yf = np.vstack([self._Ytr, self._sy.transform(Y_val)])
        self._W = fit_reduced_rank(Xf, Yf, self._lam, self._rank)
        self.state = State.FINAL_MODEL_FITTED
        return self

    def evaluate_test(self, X_test: np.ndarray, Y_test: np.ndarray) -> MetricReport:
        if self.state != State.FINAL_MODEL_FITTED:
            raise LifecycleError(f"evaluate_test from {self.state}")
        if self._test_evals >= 1:
            raise LifecycleError("test already evaluated once; the set is spent")
        self._check_dims(X_test, Y_test)
        self._test_evals += 1
        rep = per_voxel_report(self._sy.transform(Y_test), self._sx.transform(X_test) @ self._W)
        self.state = State.TEST_EVALUATED
        return rep

    def seal(self) -> dict:
        if self.state != State.TEST_EVALUATED:
            raise LifecycleError(f"seal from {self.state}")
        self.state = State.SEALED
        return dict(state=self.state.value, lam=self._lam, rank=self._rank,
                    src_dim=self._src_dim, tgt_dim=self._tgt_dim,
                    voxel_hash=self.voxel_hash, trial_table_hash=self.trial_table_hash,
                    source_centering=self.source_centering, refit_policy=self.refit_policy,
                    policy_name=HISTORICAL_POLICY if self.refit_policy == "train_only" else "SENSITIVITY")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._W is None:
            raise LifecycleError("predict before final fit")
        self._check_dims(X)
        return self._sx.transform(X) @ self._W

    @property
    def selected(self):
        return self._lam, self._rank
