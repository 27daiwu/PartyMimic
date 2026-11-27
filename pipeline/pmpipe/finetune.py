"""PartyMimic V1.6 (B-line) - fine-tune an existing run's weights on a (possibly different) reference-motion directory:  `./pm finetune <dance> ...`.

Why `resume`-style (default) and not `--init`
  train.py `--init F` loads ONLY networks (wm / policy / critic) and starts a new run: optimiser state (Muon momentum / Adam moments) is zeroed, the policy EMA is re-seeded
  from the weights, the iteration counter restarts at 0 (=> lr schedule restarts at its peak 4e-4 and `pi_lr_mult` x4 comes back until the LR trigger fires again),
  the reset-sampler statistics are lost.  `--resume F` (F = last.pt, needs opt_* + ctrl) continues EVERYTHING that train.py can save: weights, optimiser moments, EMA average,
  iteration counter (lr = cosine schedule at it40000 = 3.76e-4, pi_mult already 1: the trigger fired at it1400), reset-sampler statistics.  Replay buffer / simulator state /
  RNG streams are never saved (a resumed run refills the buffer in --warmup_steps, "warm restart").  So `resume` from the it40000 last.pt is the least invasive way of
  "continuing" a trained policy on new data: no optimiser shock, no schedule restart, only the data changes.  `--mode init` is kept for completeness (optimiser reset).
  NOTE: ckpt_<it>.pt of a run is weights-only (pi/pi_live/wm, no critic, no optimiser) -> it can be neither --init'ed nor --resume'd; the full state is last.pt (+ last_bak_*).
  `--from-ckpt` therefore resolves to the full-state file of the same iteration and refuses otherwise.

Run layout: results/<dance>/runs/<run-id>/ (nested, same as every pipeline run).  The source run is READ-ONLY: its last.pt is COPIED into the new run's train/ dir (the supervisor's
crash recovery / rollback works inside the new directory only).  `--iters N` = N ADDITIONAL iterations (target = source iteration + N).
Fairness: control and treatment runs must share every flag except `--motion-dir` (and optionally the seed); all settings are written to pm/finetune.json and the frozen train config.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import yaml

from .core import PY, REPO, Dance, Logger, Run, die, load_cfg, now, pick_gpu, read_json, say, sha256, snapshot_cfgs, tmux_has, tmux_window, tmux_windows, train_procs, write_json

PM_ENV = f"export PYTHONPATH={REPO / 'pipeline'}:{REPO / 'src'}"


def resolve_full_state(base: Run, spec: str | None) -> tuple[Path, Path | None, int]:
    """--from-ckpt (iteration number, ckpt_*.pt path or last.pt path) -> (full-state file, weights-only file or None, iteration)."""
    import torch
    cands: list[Path] = []
    weights_only: Path | None = None
    if spec is None or str(spec).isdigit():
        it = int(spec) if spec else None
        if it is not None:
            weights_only = base.ckpt(it) if base.ckpt(it).exists() else None
            cands = [base.train / "last.pt", base.train / f"last_bak_{it:06d}.pt"]
        else:
            cands = [base.train / "last.pt"]
    else:
        p = Path(spec)
        if not p.exists():
            die(f"--from-ckpt {spec}: file not found")
        if p.name.startswith("ckpt_"):
            weights_only = p
            cands = [p.parent / "last.pt", p.parent / f"last_bak_{int(p.stem.split('_')[1]):06d}.pt"]
        else:
            cands = [p]
    want = None
    if weights_only is not None:
        want = int(weights_only.stem.split("_")[1])
    elif spec and str(spec).isdigit():
        want = int(spec)
    for c in cands:
        if not c.exists():
            continue
        ck = torch.load(c, map_location="cpu", weights_only=False)
        if "opt_pi" in ck and "critic" in ck and (want is None or int(ck["iteration"]) == want):
            return c, weights_only, int(ck["iteration"])
    die(f"--from-ckpt {spec}: no FULL-STATE checkpoint (last.pt with optimiser state + critic) of iteration {want} found (looked at {[str(c) for c in cands]}). "
        "ckpt_*.pt files are weights-only and cannot be resumed/initialised for fine-tuning.", 4)


def make_ft_run(dance: Dance, run_id: str, base: Run, motion_dir: Path, from_ckpt: str | None, iters: int, seed: int, lr: float | None, mode: str,
                ckpt_every: int, last_every: int, note: str = "", train_args: tuple = (), experiment: str = "V1.6 finetune") -> tuple[Run, dict]:
    full, wonly, it0 = resolve_full_state(base, from_ckpt)
    idx = dance.index()
    if run_id in idx["runs"]:
        die(f"run '{run_id}' already exists ({dance.dir / 'runs' / run_id}); choose a new --run-id (finished runs are never overwritten)", 6)
    npzs = sorted(Path(motion_dir).glob("*.npz"))
    if len(npzs) != 1:
        die(f"--motion-dir {motion_dir}: expected exactly one *.npz (the padded clip), found {[p.name for p in npzs]}")
    dflt = idx.get("default_run")
    unpadded = None  # unpadded clip next to the padded one (<name>_padded/<name>_padded.npz -> <name>/<name>.npz): needed by export / package (motion.bin, duration)
    pd = Path(motion_dir).resolve()
    if pd.name.endswith("_padded"):
        cand = pd.parent / pd.name[:-7] / (pd.name[:-7] + ".npz")
        unpadded = str(cand) if cand.exists() else None
    meta = {"layout": "nested", "train_cfg": base.meta.get("train_cfg") or "robust_d_v1", "created": now(), "motion_dir": str(Path(motion_dir).resolve()), **({"motion_npz": unpadded} if unpadded else {}), "experiment": experiment,
            "base_run": base.id, "base_iteration": it0, "note": note}
    idx["runs"][run_id] = meta
    idx["default_run"] = dflt  # never change the default run
    dance.save_index(idx)
    run = Run(dance, run_id, meta)
    run.mk()
    # ---- frozen train config: the base run's frozen recipe + (seed, optional lr) override (later argparse flags win; train_args are passed in order)
    bm = base.manifest()
    src_cfg = Path(bm.get("train_cfg_frozen") or (base.pm / "config_snapshot" / f"train__{meta['train_cfg']}.yaml"))
    cfg = yaml.safe_load(src_cfg.read_text())
    cfg["seed"] = int(seed)
    cfg["name"] = f"{cfg['name']}__ft"
    cfg["description"] = f"V1.6 fine-tune of {base.id} it{it0}: recipe of {src_cfg.name} unchanged" + (f" except --lr={lr}" if lr is not None else "") + f", seed {seed}"
    if lr is not None:
        cfg["train_args"] = [a for a in cfg["train_args"] if not str(a).startswith("--lr=")] + [f"--lr={lr}"]
    cfg["ckpt_every"], cfg["last_every"] = ckpt_every, last_every
    if train_args:  # V1.7: extra train.py flags appended AFTER the recipe (argparse: later flags win); the recipe flags themselves stay untouched
        cfg["train_args"] = list(cfg["train_args"]) + [str(a) for a in train_args]
        cfg["description"] += "; extra train args: " + " ".join(str(a) for a in train_args)
    snapdir = run.pm / "config_snapshot"
    snapdir.mkdir(parents=True, exist_ok=True)
    tpath = snapdir / f"train__{cfg['name']}.yaml"
    tpath.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    ecfg, dcfg = load_cfg("eval", bm.get("eval_cfg") or "balance_first_v1"), load_cfg("deploy", "cpp_g1_v2")
    snap = snapshot_cfgs(run, {"eval": ecfg, "deploy": dcfg})
    snap["train"] = {"name": cfg["name"], "sha256": sha256(tpath), "frozen_copy": str(tpath)}
    # ---- start state
    if mode == "resume":
        shutil.copy2(full, run.train / "last.pt")
        run.init_pt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wonly or full, run.init_pt)  # baseline of the run's own evaluation (it0 weights, evaluated on the run's own reference by the eval daemon)
        hdr = (base.train / "train_log.csv").read_text().splitlines()[0]
        (run.train / "train_log.csv").write_text(hdr + "\n")  # train.py writes the header only when start_it == 0; without it the resumed rows would be headerless
    else:  # init: weights only, optimiser / EMA / iteration reset (iteration counts from 0, so the target is `iters`)
        shutil.copy2(full, run.init_pt)
    target = (it0 + iters) if mode == "resume" else iters
    info = dict(created=now(), dance=dance.name, run=run_id, base_run=base.id, base_full_state=str(full), base_full_state_sha12=sha256(full)[:12], base_weights_only=str(wonly) if wonly else None,
                base_weights_only_sha12=sha256(wonly)[:12] if wonly else None, base_iteration=it0, mode=mode, iters_added=iters, target_iterations=target, seed=seed, lr_override=lr,
                motion_dir=str(Path(motion_dir).resolve()), motion_npz=str(npzs[0]), motion_npz_sha12=sha256(npzs[0])[:12], ckpt_every=ckpt_every, last_every=last_every, note=note,
                train_cfg_frozen=str(tpath), source_run_untouched=True, extra_train_args=[str(a) for a in train_args])
    write_json(run.pm / "finetune.json", info)
    run.update_manifest(dance=dance.name, run=run_id, train_cfg=meta["train_cfg"], train_cfg_frozen=str(tpath), eval_cfg=ecfg["name"], eval_cfg_frozen=snap["eval"]["frozen_copy"],
                        deploy_cfg=dcfg["name"], deploy_cfg_frozen=snap["deploy"]["frozen_copy"], config_snapshot=snap, iterations=target, finetune=info, motion_dir=str(Path(motion_dir).resolve()),
                        padded_npz=str(npzs[0]), repo_head=_git_head(), experiment=experiment)
    return run, info


def _git_head() -> str:
    import subprocess
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def launch(dance: Dance, from_ckpt: str | None, motion_dir: str, run_id: str, iters: int, seed: int = 0, lr: float | None = None, mode: str = "resume", base_run: str | None = None,
           num_envs: int | None = None, gpu: str = "auto", ckpt_every: int = 500, last_every: int = 500, eval_every: int = 2000, eval_also: str = "40500,41000", after: str | None = None,
           no_eval: bool = False, note: str = "", train_args: tuple = (), experiment: str = "V1.6 finetune") -> dict:
    base = dance.run(base_run)
    if train_procs(dance.dir / "runs" / run_id):
        die(f"a trainer is already running for {run_id}")
    run, info = make_ft_run(dance, run_id, base, Path(motion_dir), from_ckpt, iters, seed, lr, mode, ckpt_every, last_every, note, train_args, experiment)
    tcfg = load_cfg("train", info["train_cfg_frozen"])
    g = pick_gpu(gpu, num_envs or tcfg["num_envs"])
    if g["index"] is None:
        die("no CUDA GPU detected")
    for w in g["warnings"]:
        say("[pm] WARNING:", w)
    run.update_manifest(gpu=g)
    try:  # reproducibility snapshot (failure only warns)
        from .repro import snapshot
        snapshot(run, log=Logger(run.logs / "repro.log", echo=False))
    except Exception as e:
        say(f"[pm] WARNING: repro snapshot failed (training continues): {e!r}")
    sess = run.session
    ne = g["num_envs"]
    wait = ""
    if after:
        prev = dance.run(after)
        wait = f"while [ ! -f {prev.state / 'train_done'} ]; do sleep 30; done; echo previous run {after} done; "
    cmd = (f"{PM_ENV}; {wait}{PY} -u -m pmpipe.supervisor --dance {dance.name} --run-id {run_id} --iterations {info['target_iterations']} --num-envs {ne} --gpu {g['index']}"
           f" --ckpt-every {ckpt_every} --last-every {last_every} 2>&1 | tee -a {run.logs}/supervisor.stdout")
    res = tmux_window(sess, "train", cmd)
    out = dict(session=sess, train=res, run=str(run.root), log=str(run.logs / "train_stdout.log"), target=info["target_iterations"], num_envs=ne, gpu=g["index"])
    if not no_eval:
        ev = (f"{PM_ENV}; export CUDA_VISIBLE_DEVICES={g['index']}; {wait}{PY} -u -m pmpipe.evald --dance {dance.name} --run-id {run_id} --motion-dir {info['motion_dir']} "
              f"--every {eval_every} --also {eval_also} 2>&1 | tee -a {run.logs}/evald.stdout")
        out["eval"] = tmux_window(sess, "eval", ev)
    mon = (f"{wait}while true; do echo \"$(date '+%F %T') $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader) | $(tail -n 1 {run.train}/train_log.csv | cut -d, -f1,11,12,13,14,15)\" | tee -a {run.logs}/gpu_monitor.log; sleep 60; done")
    out["monitor"] = tmux_window(sess, "monitor", mon)
    say(f"[pm] finetune started: dance={dance.name} run={run_id} from {run_id and info['base_run']}/it{info['base_iteration']} ({mode}) -> target it{info['target_iterations']}  motion={info['motion_dir']}  seed={seed}"
        f"  GPU{g['index']} envs={ne}  tmux={sess}")
    return out


def status(run: Run) -> str:
    fi = read_json(run.pm / "finetune.json", {})
    rows = run.train_rows(1)
    L = [f"{run.dance.name}/{run.id}: base {fi.get('base_run')}/it{fi.get('base_iteration')} mode={fi.get('mode')} seed={fi.get('seed')} target it{fi.get('target_iterations')} motion={fi.get('motion_dir')}"]
    if rows:
        r = rows[0]
        it, tgt = int(float(r["iteration"])), fi.get("target_iterations") or 0
        ips = float(r["it_per_s"]) or 1
        L.append(f"  it {it}/{tgt}  {ips:.1f} it/s  sim_rew {float(r['sim_rew']):.3f} ep_ret {float(r['ep_ret_norm']):.3f} term {float(r['term_frac']):.4f}  ETA {max(tgt - it, 0) / ips / 60:.0f} min  train process: {'RUNNING' if train_procs(run.train) else 'not running'}")
    return "\n".join(L)
