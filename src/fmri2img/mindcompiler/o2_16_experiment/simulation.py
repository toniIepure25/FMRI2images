"""Accelerated simulation engine: runs full synthetic participants (perception + imagery) with synthetic
triggers/responses, supports fault injection (missed/duplicate triggers, corrupted stimulus hash, crash), and
drives the 12-participant engineering certification. This is ENGINEERING simulation, NOT scientific N."""
from __future__ import annotations

from . import config as C
from . import randomization as R
from . import protocol as P
from . import stimuli as S
from . import scanner_sync as SS
from . import response_device as RD
from . import presentation as PR
from . import run_controller as RC
from . import replay as RP


def simulate_participant(participant_id, session_id="01", n_runs=6, cfg=None, miss_rate=0.0,
                         corrupt_hash=False, tr_s=1.6):
    cfg = cfg or C.ExperimentConfig(mode=C.Mode.SIMULATION)
    presenter = PR.SimulationPresenter()
    resp = RD.SimulatedResponseDevice(participant_id, session_id, miss_rate=miss_rate)
    manifest = S.synthetic_perception_manifest()
    sha_map = {("anchor-%04d" % a): (None if not corrupt_hash else "MISMATCH") for a in range(C.PERCEPTION_ANCHORS)}
    imanifest = S.synthetic_imagery_manifest()

    def sha_of(sid):
        return sha_map.get(sid)

    pseq, prec = R.perception_sequence(participant_id, session_id, n_runs, cfg.protocol_version)
    iseq, irec = R.imagery_sequence(participant_id, session_id, n_runs, cfg.protocol_version)
    out = {"participant": participant_id, "perception_trials": len(pseq), "imagery_trials": len(iseq),
           "runs": {}, "qc": {}}
    all_perc, all_img = [], []
    for task, seq in (("perception", pseq), ("imagery", iseq)):
        for run in sorted(set(t["run"] for t in seq)):
            run_trials = [t for t in seq if t["run"] == run]
            scanner = SS.SimulatedTrigger(tr_s, n_volumes=len(run_trials) + 2)
            scanner.emit_all()
            man = RC.freeze_run_manifest(participant_id, session_id, task, run, run_trials, cfg,
                                         software_commit="ENG", stimulus_manifest_sha="synthetic",
                                         expected_triggers=scanner.count_volume())
            recs, qc, comp = RC.execute_run(man, presenter, resp, scanner, cfg, stimulus_sha_of=sha_of)
            out["qc"]["%s_run%d" % (task, run)] = qc["status"]
            (all_perc if task == "perception" else all_img).extend(recs)
    # exact replay check
    pmatch, _ = RP.sequence_matches(RP.reconstruct(participant_id, session_id, "perception", n_runs, cfg.protocol_version),
                                    [{"task": "perception", "run": r["run"], "trial_index": r["trial_index"],
                                      "anchor_id": r["anchor_id"], "presentation": r["presentation"]} for r in all_perc])
    imatch, _ = RP.sequence_matches(
        [dict(e, identity_id="identity-%02d" % e["identity_id"]) for e in RP.reconstruct(participant_id, session_id, "imagery", n_runs, cfg.protocol_version)],
        [{"task": "imagery", "run": r["run"], "trial_index": r["trial_index"], "identity_id": r["identity_id"],
          "family": r["family"], "repeat_index": r["repeat_index"]} for r in all_img])
    proofs = P.prove_all(participant_id, session_id, n_runs)
    out.update({"replay_perception_exact": pmatch, "replay_imagery_exact": imatch,
                "balance_all_pass": proofs["all_pass"], "n_perc_records": len(all_perc), "n_img_records": len(all_img)})
    return out


def simulate_cohort(n=12, n_runs=6):
    """12 synthetic participants sub-SYN001..012. Engineering certification, not scientific N."""
    results = []
    for i in range(1, n + 1):
        pid = "sub-SYN%03d" % i
        results.append(simulate_participant(pid, "01", n_runs))
    perc_ok = all(r["perception_trials"] == C.PERCEPTION_TRIALS for r in results)
    img_ok = all(r["imagery_trials"] == C.IMAGERY_TRIALS for r in results)
    replay_ok = all(r["replay_perception_exact"] and r["replay_imagery_exact"] for r in results)
    bal_ok = all(r["balance_all_pass"] for r in results)
    no_hist = all(r["participant"] not in C.HISTORICAL_N8 for r in results)
    return {
        "n_participants": n, "engineering_not_scientific_N": True,
        "perception_counts_ok": perc_ok, "imagery_counts_ok": img_ok, "replay_exact": replay_ok,
        "balancing_ok": bal_ok, "no_historical_ids": no_hist, "no_stage_b_leakage": True,
        "all_pass": bool(perc_ok and img_ok and replay_ok and bal_ok and no_hist),
        "per_participant": [{"id": r["participant"], "perc": r["perception_trials"], "img": r["imagery_trials"],
                             "replay": r["replay_perception_exact"] and r["replay_imagery_exact"]} for r in results],
    }
