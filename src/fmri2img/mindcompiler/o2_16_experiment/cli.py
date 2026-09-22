"""CLI for the O2.16 experiment stack. Default mode is SIMULATION. There is no confirmatory-override flag of any
kind; confirmatory requires a real authorization manifest and fails closed. No command displays protected
Stage-B outcomes."""
from __future__ import annotations

import argparse
import json

from . import config as C
from . import protocol as P
from . import simulation as SIM
from . import sites as SITES
from . import replay as RP
from . import stage_a_export as SA


def _p(obj):
    print(json.dumps(obj, indent=2, default=str))


def cmd_validate_config(a):
    cfg = C.ExperimentConfig(mode=C.Mode[a.mode])
    _p({"protocol_version": cfg.protocol_version, "protocol_sha": cfg.protocol_hash(),
        "immutable_hashes": C.IMMUTABLE_HASHES, "mode": cfg.mode.value,
        "confirmatory_timing_missing": cfg.timing.confirmatory_missing()})


def cmd_simulate_session(a):
    _p(SIM.simulate_participant(a.participant or "sub-SYN001", a.session, a.runs))


def cmd_simulate_cohort(a):
    _p(SIM.simulate_cohort(a.n, a.runs))


def cmd_generate_session(a):
    _p(P.prove_all(a.participant or "sub-SYN001", a.session, a.runs))


def cmd_replay(a):
    seq = RP.reconstruct(a.participant or "sub-SYN001", a.session, a.task, a.runs)
    _p({"task": a.task, "n_trials": len(seq), "first": seq[0], "last": seq[-1]})


def cmd_validate_site(a):
    _p(SITES.validate_site(SITES.load_site_yaml(a.site)))


def cmd_export_stage_a(a):
    _p(SA.export_stage_a({"cohort": "synthetic", "participants": ["sub-SYN001"], "n_runs": a.runs}))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mindir-o216")
    ap.add_argument("--mode", default="SIMULATION", choices=[m.value for m in C.Mode])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("validate-config"); s.set_defaults(f=cmd_validate_config)
    s = sub.add_parser("simulate-session"); s.add_argument("--participant"); s.add_argument("--session", default="01"); s.add_argument("--runs", type=int, default=6); s.set_defaults(f=cmd_simulate_session)
    s = sub.add_parser("simulate-cohort"); s.add_argument("--n", type=int, default=12); s.add_argument("--runs", type=int, default=6); s.set_defaults(f=cmd_simulate_cohort)
    s = sub.add_parser("generate-session"); s.add_argument("--participant"); s.add_argument("--session", default="01"); s.add_argument("--runs", type=int, default=6); s.set_defaults(f=cmd_generate_session)
    s = sub.add_parser("replay"); s.add_argument("--participant"); s.add_argument("--session", default="01"); s.add_argument("--task", default="perception", choices=C.TASKS); s.add_argument("--runs", type=int, default=6); s.set_defaults(f=cmd_replay)
    s = sub.add_parser("validate-site"); s.add_argument("--site", required=True); s.set_defaults(f=cmd_validate_site)
    s = sub.add_parser("export-stage-a"); s.add_argument("--runs", type=int, default=6); s.set_defaults(f=cmd_export_stage_a)
    a = ap.parse_args(argv)
    a.f(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
