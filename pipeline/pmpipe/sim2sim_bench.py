#!/usr/bin/env python3
"""Sim2Sim closed-loop bench for ANY dance (runs in the sim2real py3.10 venv; MuJoCo; no robot connection).

Thin wrapper over the P4 harness (results/p4/scripts/{sim2sim_deploy,eval_bench,pm_stack}.py = deployment control chain: authority/ramp, C++ guard
semantics, 200 Hz PD, effort clip, delay) - reused unchanged. This wrapper only points it at
  * the run's own deploy params (policy_params.txt generated for THIS policy, so Sim2Sim uses exactly what the package will contain),
  * the dance's padded reference (what motion.bin + the 1 s lead-in replay in the C++ runner),
  * the archived real-robot WalkRun standing poses as initial states (pipeline/ref_data/walkrun_real_runs).
usage: S2R_PY sim2sim_bench.py --padded-npz P --params policy_params.txt --onnx X --out J [eval_bench args: --n --conds --mode --auth --ramp --procs --seed]
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "results/p4/scripts"))


def _pop(flag):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return None


padded, params = _pop("--padded-npz"), _pop("--params")
import eval_bench  # noqa: E402
import pm_stack  # noqa: E402
import sim2sim_deploy  # noqa: E402

eval_bench.LOGS = str(REPO / "pipeline/ref_data/walkrun_real_runs")
if params:
    pm_stack.load_params.__defaults__ = (params,)  # sim2sim_deploy / eval_bench import the same function object
if padded:
    sim2sim_deploy.padded_motion.__defaults__ = (padded,)

if __name__ == "__main__":
    eval_bench.main()
