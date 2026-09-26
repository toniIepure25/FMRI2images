"""Unified `mindir` CLI namespace for the PROX platform. Discoverable, documented, synthetic-only. No historical
N=8 access; no scientific status change."""
from __future__ import annotations

import argparse
import json

from . import metrics as MX
from . import benchmark as BM
from . import sample_complexity as SC
from . import falsification as FB
from . import governance as GV
from . import synthworld as SW


def _p(o):
    print(json.dumps(o, indent=2, default=str))


def cmd_validate(a):
    _p({"scientific_immutability": GV.scientific_immutability(a.repo), "dependency_env": GV.dependency_env()})


def cmd_metrics(a):
    _p({"n_metrics": len(MX.REGISTRY), "families": sorted(set(m.family for m in MX.REGISTRY.values())),
        "metrics": sorted(MX.REGISTRY)})


def cmd_simulate(a):
    _p(SW.generate(SW.WorldConfig(n_subjects=a.n))["config"])


def cmd_benchmark(a):
    _p(BM.run_benchmark(SW.WorldConfig(n_subjects=a.n)))


def cmd_phase(a):
    _p(SC.phase_diagram(n_obs=a.n_obs))


def cmd_falsify(a):
    _p(FB.battery_manifest())


def cmd_version(a):
    from . import __version__, SCHEMA_VERSIONS
    _p({"mindir_platform_version": __version__, "distribution": "fmri2img", "schema_versions": SCHEMA_VERSIONS})


def cmd_demo(a):
    from . import release as RL
    _p(RL.demo(out_dir=a.out))


def cmd_reproduce(a):
    from . import release as RL
    import json as _j
    manifest = _j.load(open(a.manifest)) if a.manifest else {}
    _p(RL.reproduce(manifest))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mindir", description="MINDIR PROX platform CLI (synthetic; no N=8 access)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version").set_defaults(f=cmd_version)
    s = sub.add_parser("validate"); s.add_argument("--repo", default="."); s.set_defaults(f=cmd_validate)
    sub.add_parser("metrics").set_defaults(f=cmd_metrics)
    s = sub.add_parser("simulate"); s.add_argument("--n", type=int, default=12); s.set_defaults(f=cmd_simulate)
    s = sub.add_parser("demo"); s.add_argument("--out", default=None); s.set_defaults(f=cmd_demo)
    s = sub.add_parser("benchmark"); s.add_argument("mode", nargs="?", default="synthetic", choices=["synthetic"]); s.add_argument("--n", type=int, default=12); s.set_defaults(f=cmd_benchmark)
    s = sub.add_parser("phase"); s.add_argument("--n-obs", type=int, default=8); s.set_defaults(f=cmd_phase)
    sub.add_parser("falsify").set_defaults(f=cmd_falsify)
    s = sub.add_parser("reproduce"); s.add_argument("manifest", nargs="?", default=None); s.set_defaults(f=cmd_reproduce)
    a = ap.parse_args(argv); a.f(a); return 0


if __name__ == "__main__":
    raise SystemExit(main())
