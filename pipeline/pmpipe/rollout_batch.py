#!/usr/bin/env python3
"""Batch runner for closed-loop deployment-chain rollouts (runs in the sim2real py3.10 venv): executes many `rollout.run_rollout` jobs in ONE process pool, skipping outputs that exist.
usage: rollout_batch.py --jobs jobs.json [--procs 20]   jobs.json = [{"kw": {onnx, params, padded_npz, cond, init, seed}, "out": path}, ...]
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout as RO  # noqa: E402  (also puts results/p4/scripts on sys.path)


def job(a):
    """One rollout in a FRESH sim model: sim2sim_deploy.get_model() caches one MjModel per process and the 'friction' condition mutates it in place, so reusing a worker
    across conditions would leak friction 0.6 into later (nominal) rollouts (found in the V1.6 ftcompare smoke test: identical model+seeds gave bimodal slip)."""
    import sim2sim_deploy as S
    S._MODEL = None
    return RO._job(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--procs", type=int, default=20)
    a = ap.parse_args()
    jobs = [j for j in json.loads(Path(a.jobs).read_text()) if not Path(j["out"]).exists()]
    print(f"{len(jobs)} rollouts to run", flush=True)
    if not jobs:
        return
    for j in jobs:
        Path(j["out"]).parent.mkdir(parents=True, exist_ok=True)
    work = [(j["kw"], j["out"]) for j in jobs]
    from multiprocessing import Pool
    with Pool(min(a.procs, len(work))) as p:
        for i, (o, f) in enumerate(p.imap_unordered(job, work, chunksize=1)):
            if i % 20 == 0:
                print(f"{i + 1}/{len(work)} {Path(o).name} fell_t {f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
