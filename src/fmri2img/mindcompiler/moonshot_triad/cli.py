"""MINDIR-M0 CLI. All commands operate on SYNTHETIC fixtures (engineering only) or report frozen prereg state.
No historical N=8 loader; no Cohort-B outcome access before the O2.16 unlock chain."""
from __future__ import annotations

import argparse
import json

from . import common as CM
from . import m1_adaptive as M1
from . import m2_zeroshot as M2
from . import m3_crossstate as M3


def _p(o):
    print(json.dumps(o, indent=2, default=str))


def cmd_status(a):
    _p({"program": "MINDIR-M0 three-moonshot preregistration",
        "target": "MINDIR_MOONSHOT_TRIAD_PREREGISTERED_AWAITING_NEW_DATA",
        "M1": "M1_SHADOW_ADAPTIVE_PROTOCOL_READY", "M2": "M2_ZERO_SHOT_PROTOCOL_READY",
        "M3": "M3_CROSS_STATE_PROTOCOL_READY_FOR_HUMAN_REVIEW",
        "immutable": CM.immutability_report(), "sample_size": CM.sample_size_reality()})


def cmd_unlock_check(a):
    st = CM.UnlockState()
    _p({"unlock_order": CM.UNLOCK_ORDER, "cohort_b_unlocked": st.cohort_b_unlocked(),
        "message": "Cohort-B moonshot outcomes stay locked until O2.16 primary + O2.16-SEC are executed and sealed"})


def cmd_sim_m1(a):
    _p(M1.simulate(a.n))


def cmd_sim_m2(a):
    _p(M2.simulate(a.n))


def cmd_sim_m3(a):
    _p(M3.simulate(a.n))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mindir-m0")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(f=cmd_status)
    sub.add_parser("unlock-check").set_defaults(f=cmd_unlock_check)
    s = sub.add_parser("simulate-m1"); s.add_argument("--n", type=int, default=12); s.set_defaults(f=cmd_sim_m1)
    s = sub.add_parser("simulate-m2"); s.add_argument("--n", type=int, default=12); s.set_defaults(f=cmd_sim_m2)
    s = sub.add_parser("simulate-m3"); s.add_argument("--n", type=int, default=12); s.set_defaults(f=cmd_sim_m3)
    a = ap.parse_args(argv); a.f(a); return 0


if __name__ == "__main__":
    raise SystemExit(main())
