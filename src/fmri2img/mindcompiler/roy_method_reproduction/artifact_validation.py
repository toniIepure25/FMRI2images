"""Cross-artifact consistency validator for the Roy S1 smoke outputs.

The S1.8 defect class was an INTERNAL contradiction between artifacts that each
looked fine alone: ``smoke_result.json`` recorded ``pairing_seed=null`` while
``smoke_pairing_manifest.json`` recorded ``1234``. This module checks the smoke
outputs *against each other* and against the trial table, so that class of defect
fails loudly.

It is pure and data-free: it operates on already-loaded dicts / a trial-table
DataFrame (or loads them from a directory), and never touches beta data. Every
check returns a structured :class:`Check`; :meth:`ValidationReport.ok` is the
conjunction. Policy-name/seed rules mirror :mod:`pairing`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from fmri2img.mindcompiler.roy_method_reproduction.metrics import FINITE_FRACTION_MIN
from fmri2img.mindcompiler.roy_method_reproduction.pairing import SEED_CONSUMING, SPLIT_DETERMINISTIC

#: Certified subj01 nsdimagery beta content hashes, per SHA-pinned version.
BETA_PINS = {
    "fithrf": "31485ff0e4cb9e90b6f83f2f550714b1a688993604f5795148a2746df7b42b64",            # B0
    "fithrf_GLMdenoise_RR": "cd42e680617d6564f403c7855140c53c0a2fdb9eb382755b49765bf781c4af2e",  # B1
}
#: Back-compat alias (B0).
B0_SHA = BETA_PINS["fithrf"]


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class ValidationReport:
    checks: List[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(Check(name, bool(ok), detail))

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def failures(self) -> List[Check]:
        return [c for c in self.checks if not c.ok]

    def as_dict(self) -> dict:
        return {"ok": self.ok, "checks": [asdict(c) for c in self.checks]}


def _seed_expected(policy: str) -> bool:
    if policy in SEED_CONSUMING:
        return True
    if policy in SPLIT_DETERMINISTIC:
        return False
    raise ValueError(f"unknown policy {policy!r}")


def check_seed_provenance(result: dict, pairing_manifest: dict, rep: ValidationReport) -> None:
    """The S1.8 guard: a seed is recorded iff the policy consumes one, and the
    result-level seed agrees with every per-model manifest entry."""
    seeded_any = False
    for model in ("vis2vis", "vis2img"):
        m = pairing_manifest[model]
        policy = m["policy"]
        want_seed = _seed_expected(policy)
        seeded_any = seeded_any or want_seed
        has_seed = m.get("pairing_seed") is not None
        rep.add(f"manifest.{model}.seed_matches_policy", has_seed == want_seed,
                f"policy={policy} seed={m.get('pairing_seed')} expected_seeded={want_seed}")
        src = m.get("randomness_source")
        want_src = "child_seed(pairing_seed)" if want_seed else "none_after_split"
        rep.add(f"manifest.{model}.randomness_source", src == want_src,
                f"got={src!r} expected={want_src!r}")
        if want_seed:
            rep.add(f"manifest.{model}.seed_equals_result",
                    m.get("pairing_seed") == result.get("pairing_seed"),
                    f"manifest={m.get('pairing_seed')} result={result.get('pairing_seed')}")
    # result-level seed present iff any model consumes one
    rep.add("result.pairing_seed_matches_any_seeded",
            (result.get("pairing_seed") is not None) == seeded_any,
            f"result.pairing_seed={result.get('pairing_seed')} seeded_any={seeded_any}")
    want_src = "child_seed(pairing_seed)" if seeded_any else "split_seed"
    rep.add("result.pairing_randomness_source",
            result.get("pairing_randomness_source") == want_src,
            f"got={result.get('pairing_randomness_source')!r} expected={want_src!r}")


def check_hashes(result: dict, trial_table_csv: Optional[bytes], rep: ValidationReport) -> None:
    # Certify the beta content hash against its pinned value. New artifacts carry
    # beta_version + beta_sha; legacy (fithrf-only) artifacts carry just b0_sha.
    bv = result.get("beta_version")
    pin = BETA_PINS.get(bv)
    if result.get("beta_sha") is not None:
        rep.add("result.beta_sha_is_certified",
                pin is not None and result["beta_sha"] == pin,
                f"beta_version={bv} beta_sha={result.get('beta_sha')}")
    elif result.get("b0_sha") is not None:
        rep.add("result.b0_sha_is_certified", result["b0_sha"] == (pin or B0_SHA),
                f"beta_version={bv} b0_sha={result.get('b0_sha')}")
    else:
        rep.add("result.beta_sha_present", False,
                "neither beta_sha nor b0_sha present")
    if trial_table_csv is not None:
        got = hashlib.sha256(trial_table_csv).hexdigest()
        rep.add("result.trial_table_sha_matches_file", got == result.get("trial_table_sha"),
                f"file={got} result={result.get('trial_table_sha')}")


def check_split_manifest(result: dict, split_manifest: dict, tt, rep: ValidationReport) -> None:
    rep.add("split.seed_matches_result",
            split_manifest.get("split_seed") == result.get("split_seed"),
            f"split={split_manifest.get('split_seed')} result={result.get('split_seed')}")
    trials = split_manifest.get("trials", [])
    if tt is not None:
        rep.add("split.covers_trial_table", len(trials) == len(tt),
                f"manifest={len(trials)} trial_table={len(tt)}")
        mism = [t for t in trials
                if int(tt.iloc[int(t["row"])]["beta_index0"]) != int(t["beta_index0"])]
        rep.add("split.beta_index0_matches_trial_table", not mism,
                f"{len(mism)} mismatched rows")
    parts = {t.get("split") for t in trials}
    rep.add("split.has_train_val_test", {"train", "val", "test"} <= parts,
            f"partitions={sorted(parts)}")


def check_pairing_indices(split_manifest: dict, pairing_manifest: dict, rep: ValidationReport) -> None:
    """Every training pair must reference rows that are in the TRAIN partition."""
    train_rows = {int(t["row"]) for t in split_manifest.get("trials", [])
                  if t.get("split") == "train"}
    for model in ("vis2vis", "vis2img"):
        m = pairing_manifest[model]
        src = set(map(int, m.get("train_src", [])))
        tgt = set(map(int, m.get("train_tgt", [])))
        leaked = (src | tgt) - train_rows
        rep.add(f"pairing.{model}.train_pairs_are_train_rows", not leaked,
                f"{len(leaked)} pair endpoints outside train partition")


def check_metrics_sanity(result: dict, rep: ValidationReport) -> None:
    rep.add("result.finite_fraction_min_is_constant",
            result.get("finite_fraction_min") == FINITE_FRACTION_MIN,
            f"got={result.get('finite_fraction_min')} expected={FINITE_FRACTION_MIN}")
    for model in ("vis2vis", "vis2img"):
        d = result.get(model, {})
        hard = d.get("rank_hard_max")
        rank = d.get("rank")
        if hard is not None and rank is not None:
            rep.add(f"result.{model}.rank_within_hard_max", int(rank) <= int(hard),
                    f"rank={rank} hard_max={hard}")
        rmax = d.get("rank_max")
        if rmax is not None and rank is not None:
            rep.add(f"result.{model}.rank_within_rank_max", int(rank) <= int(rmax),
                    f"rank={rank} rank_max={rmax}")
        ff = d.get("finite_frac")
        if ff is not None:
            rep.add(f"result.{model}.finite_frac_in_unit_interval", 0.0 <= float(ff) <= 1.0,
                    f"finite_frac={ff}")


def validate_artifacts(result: dict, split_manifest: dict, pairing_manifest: dict,
                       trial_table=None, trial_table_csv: Optional[bytes] = None
                       ) -> ValidationReport:
    """Run all cross-artifact checks on already-loaded artifacts."""
    rep = ValidationReport()
    check_hashes(result, trial_table_csv, rep)
    check_seed_provenance(result, pairing_manifest, rep)
    check_split_manifest(result, split_manifest, trial_table, rep)
    check_pairing_indices(split_manifest, pairing_manifest, rep)
    check_metrics_sanity(result, rep)
    return rep


def validate_smoke_dir(dirpath) -> ValidationReport:
    """Load a smoke output directory and validate its artifacts against each other.

    Expects ``smoke_result.json``, ``smoke_split_manifest.json``,
    ``smoke_pairing_manifest.json`` and ``trial_table.csv``.
    """
    d = Path(dirpath)
    result = json.loads((d / "smoke_result.json").read_text())
    split_manifest = json.loads((d / "smoke_split_manifest.json").read_text())
    pairing_manifest = json.loads((d / "smoke_pairing_manifest.json").read_text())
    tt = None
    csv_bytes = None
    csv_path = d / "trial_table.csv"
    if csv_path.exists():
        csv_bytes = csv_path.read_bytes()
        try:
            import pandas as pd
            tt = pd.read_csv(csv_path)
        except Exception:  # noqa: BLE001 - pandas optional for pure-dict validation
            tt = None
    return validate_artifacts(result, split_manifest, pairing_manifest, tt, csv_bytes)
