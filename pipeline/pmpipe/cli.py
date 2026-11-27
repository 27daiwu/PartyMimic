"""PartyMimic Pipeline V1.0 - command line (./pm).  See README_PIPELINE.md."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import core
from .core import (CONFIGS, DATASETS, PY, REPO, RESULTS, S2R_PY, Dance, Logger, Run, die, gpus, load_cfg, now, pick_gpu, read_csv, read_json, say, sh, shell_quote, snapshot_cfgs,
                   tmux, tmux_has, tmux_kill_window, tmux_window, tmux_windows, train_procs, write_json)

PM_ENV = f"export PYTHONPATH={REPO / 'pipeline'}:{REPO / 'src'}" + "".join(f" {k}={shell_quote(os.environ[k])}" for k in ("PM_RESULTS", "PM_DATASETS") if k in os.environ)


# ============================================================================================================================= helpers
def _dance(a) -> Dance:
    return Dance(a.dance)


def _eval_cfg(a, run: Run | None = None) -> dict:
    name = getattr(a, "eval_cfg", None)
    if name:
        return load_cfg("eval", name)
    m = run.manifest() if run else {}
    return load_cfg("eval", m.get("eval_cfg_frozen") or "balance_first_v1")


def _deploy_cfg(a, run: Run | None = None) -> dict:
    name = getattr(a, "deploy_cfg", None)
    if name:
        return load_cfg("deploy", name)
    m = run.manifest() if run else {}
    return load_cfg("deploy", m.get("deploy_cfg") or "cpp_g1_v2")


def smoke_eval_cfg(e: dict) -> dict:
    """--smoke: tiny budgets so the whole chain (not the quality) is exercised in minutes."""
    e = json.loads(json.dumps(e))
    e["periodic"].update(every=50, also=[], reps=2)
    e["hard"].update(reps=2, top_balance=1, top_fidelity=1, top_combined=1)
    e["sim2sim"].update(n=2, conds=["nominal", "delay10", "push_f"], max_candidates=1)
    e["sim2sim"]["gate"].update(core_conds=["delay10", "push_f"])
    return e


def get_run(a, create=False, train_cfg=None) -> Run:
    d = _dance(a)
    return d.run(getattr(a, "run_id", None), train_cfg=train_cfg, create=create)


def live_windows(run: Run) -> list[str]:
    return [w for s in run.session_names for w in tmux_windows(s)]


def live_session(run: Run) -> str | None:
    for s in run.session_names:
        if tmux_has(s):
            return s
    return None


def adopt_or_create(a, tcfg: dict) -> tuple[Run, bool]:
    """Resolve the run to use. A pre-pipeline run (flat layout, e.g. results/ythn) with the same recipe is adopted in place, never migrated or restarted."""
    d = _dance(a)
    idx = d.index()
    rid = getattr(a, "run_id", None)
    if rid and rid in idx["runs"]:
        return d.run(rid), False
    legacy = idx["runs"].get("legacy")
    if legacy and not rid:
        from .supervisor import recipe_diff
        args = read_json(d.dir / "training/run/args.json", {})
        diff = recipe_diff(tcfg, args)
        if diff:
            die(f"a pre-pipeline run exists in {d.dir / 'training/run'} but its recipe differs from '{tcfg['name']}':\n  " + "\n  ".join(diff[:12]) +
                "\n  -> give a new --run-id to start a separate run, or choose the matching --train-cfg", 6)
        idx["runs"].pop("legacy")
        idx["runs"][tcfg["name"]] = dict(layout="flat", train_cfg=tcfg["name"], adopted=True, adopted_at=now(), note="pre-pipeline run adopted in place (same recipe verified against args.json)")
        idx["default_run"] = tcfg["name"]
        d.save_index(idx)
        return d.run(tcfg["name"]), True
    rid = rid or tcfg["name"]
    if rid not in idx["runs"]:
        return d.run(rid, train_cfg=tcfg["name"], create=True), False
    return d.run(rid), False


def eval_daemon_cmd(run: Run, ecfg_name: str, gpu) -> str:
    return f"{PM_ENV}; {'export CUDA_VISIBLE_DEVICES=' + str(gpu) + '; ' if gpu is not None else ''}{PY} -u -m pmpipe.evald --dance {run.dance.name} --run-id {run.id} 2>&1 | tee -a {run.logs}/evald.stdout"


def tb_port_free(p: int) -> bool:
    import socket
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", p)) != 0


# ============================================================================================================================= prepare / preview
def cmd_prepare(a):
    from .prepare import prepare
    d = _dance(a)
    m = prepare(d, a.source, a.trim, a.force)
    print(f"[pm] prepare OK: {d.npz}  (QA {m.get('qa_verdict')})\n     padded training clip: {d.padded_npz}")


def cmd_preview(a):
    from .prepare import preview
    d = _dance(a)
    if not d.npz.exists():
        from .prepare import prepare
        prepare(d)
    out = preview(d, video=a.video)
    if out.get('video'):
        print('  preview video:', out['video'])


# ============================================================================================================================= train / status / resume / stop
def start_training(a, iters: int | None, resume: bool = False) -> Run:
    from .supervisor import ensure_init
    d = _dance(a)
    tname = getattr(a, "train_cfg", None) or "robust_d_v1"
    tcfg = load_cfg("train", tname)
    ecfg = _eval_cfg(a)
    dcfg = _deploy_cfg(a)
    smoke = getattr(a, "smoke", False)
    if smoke:
        ecfg = smoke_eval_cfg(ecfg)
        a.run_id = getattr(a, "run_id", None) or "smoke"
    run, adopted = adopt_or_create(a, tcfg) if not resume else (d.run(getattr(a, "run_id", None)), False)
    if not resume or not run.manifest():
        pass
    run.mk()
    _auto_snapshot(run)
    # ---- data
    from .prepare import prepare
    meta = prepare(d)
    if meta.get("qa_verdict") == "FAIL":
        die("dance data failed QA")
    # ---- iterations
    m = run.manifest()
    target = iters or m.get("iterations") or tcfg["iterations"]
    if smoke:
        target = iters or 100
    # ---- finished?
    if run.done("train_done") and run.iteration() >= target:
        say(f"[pm] training of '{d.name}/{run.id}' already finished (it {run.iteration()} >= {target}); nothing to train")
        if not run.manifest().get("config_snapshot"):
            snap = snapshot_cfgs(run, {"train": tcfg, "eval": ecfg, "deploy": dcfg})
            run.update_manifest(config_snapshot=snap, train_cfg_frozen=snap["train"]["frozen_copy"], eval_cfg_frozen=snap["eval"]["frozen_copy"], adopted_finished=adopted)
        run.update_manifest(dance=d.name, run=run.id, train_cfg=tcfg["name"], eval_cfg=ecfg["name"], deploy_cfg=dcfg["name"], iterations=target, recipe_verified=True, repo_head=_git_head(),
                            source=str(d.source_pkl()), motion_npz=str(d.npz), padded_npz=str(d.padded_npz))
        return run
    live = [w for w in live_windows(run)]
    procs = train_procs(run.train)
    sess = live_session(run)
    # ---- already running (adopt): do not start a second trainer on the same run directory
    if procs or "train" in live:
        args_json = read_json(run.train / "args.json", {})
        running_target = int(args_json.get("iterations", target))
        say(f"[pm] training of '{d.name}/{run.id}' is ALREADY RUNNING (it {run.iteration()}/{running_target}; tmux session '{sess}'): attaching, not starting a second trainer")
        if iters and iters != running_target:
            die(f"requested --iters {iters} differs from the running target {running_target}. To change it: `./pm stop {d.name}` then `./pm resume {d.name} --iters {iters}`", 7)
        snap = snapshot_cfgs(run, {"train": tcfg, "eval": ecfg, "deploy": dcfg}) if not run.manifest().get("config_snapshot") else run.manifest()["config_snapshot"]
        run.update_manifest(dance=d.name, run=run.id, train_cfg=tcfg["name"], eval_cfg=ecfg["name"], deploy_cfg=dcfg["name"], iterations=running_target, config_snapshot=snap,
                            adopted_running=True, attached=now(), recipe_verified=True)
        ensure_support_windows(run, sess or run.session, None, a)
        return run
    # ---- fresh start / resume
    if run.done("train_done") and run.iteration() < target:  # extend a finished run
        for mk in ("train_done", "eval_done", "post_done"):
            if run.marker(mk).exists():
                run.marker(mk).unlink()
    g = pick_gpu(getattr(a, "gpu", "auto"), getattr(a, "num_envs", None) or (256 if smoke else tcfg["num_envs"]))
    if g["index"] is None:
        die("no CUDA GPU detected (nvidia-smi failed)")
    for w in g["warnings"]:
        say("[pm] WARNING:", w)
    others = [p for p in train_procs() if str(run.train) not in p]
    if others and g["util"] >= 60 and not getattr(a, "allow_concurrent", False):
        die("another training process is using this GPU:\n  " + "\n  ".join(x[:140] for x in others) + "\n  start anyway with --allow-concurrent (both will slow down)", 8)
    snap = snapshot_cfgs(run, {"train": tcfg, "eval": ecfg, "deploy": dcfg})
    frozen_train = snap["train"]["frozen_copy"]
    run.update_manifest(dance=d.name, run=run.id, train_cfg=tcfg["name"], train_cfg_frozen=frozen_train, eval_cfg=ecfg["name"], eval_cfg_frozen=snap["eval"]["frozen_copy"],
                        deploy_cfg=dcfg["name"], iterations=target, config_snapshot=snap, gpu=g, smoke=smoke, created=m.get("created", now()), repo_head=_git_head(),
                        source=str(d.source_pkl()), motion_npz=str(d.npz), padded_npz=str(d.padded_npz))
    if smoke:  # smoke runs use their own (tiny) eval budget; write it as the run's eval config
        p = run.pm / "config_snapshot" / "eval__smoke.yaml"
        import yaml
        p.write_text(yaml.safe_dump({k: v for k, v in ecfg.items() if not k.startswith("_")}))
        run.update_manifest(eval_cfg_frozen=str(p))
    ensure_init(run, tcfg, log=say)
    num_envs = g["num_envs"]
    sess = run.session
    ce = 50 if smoke else None
    cmd = (f"{PM_ENV}; {PY} -u -m pmpipe.supervisor --dance {d.name} --run-id {run.id} --iterations {target} --num-envs {num_envs} --gpu {g['index']}"
           + (f" --ckpt-every {ce} --last-every {ce}" if ce else "") + (f" --marker {a.marker}" if getattr(a, "marker", None) else "") + f" 2>&1 | tee -a {run.logs}/supervisor.stdout")
    if getattr(a, "foreground", False):
        ensure_support_windows(run, sess, g["index"], a)
        return_code = subprocess.call(["bash", "-lc", cmd])
        if return_code:
            die(f"supervisor rc={return_code}", return_code)
        return run
    res = tmux_window(sess, "train", cmd)
    say(f"[pm] training started: dance={d.name} run={run.id} cfg={tcfg['name']} iters={target} GPU{g['index']} ({g['name']}, {g['free_gib']} GiB free) num_envs={num_envs} tmux={sess}:train ({res})")
    ensure_support_windows(run, sess, g["index"], a)
    return run


def ensure_support_windows(run: Run, sess: str, gpu, a):
    if getattr(a, "no_eval", False):
        return
    ws = tmux_windows(sess)
    if "eval" not in ws and not run.done("eval_done"):
        say("[pm] eval daemon:", tmux_window(sess, "eval", eval_daemon_cmd(run, "", gpu)))
    if not getattr(a, "no_tb", False) and "tensorboard" not in ws:
        port = int(getattr(a, "tb_port", None) or 6006)
        while not tb_port_free(port) and port < 6100:
            port += 1
        say(f"[pm] tensorboard: http://localhost:{port}  ", tmux_window(sess, "tensorboard", f"{REPO / '.venv/bin/tensorboard'} --logdir {run.tb} --port {port} --bind_all 2>&1 | tee -a {run.logs}/tensorboard.log"))
    if "monitor" not in ws:
        tmux_window(sess, "monitor", f"{REPO / 'pm'} status {run.dance.name} --run-id {run.id} --watch")


def _git_head() -> str:
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def cmd_train(a):
    run = start_training(a, a.iters)
    say(f"[pm] next: ./pm status {a.dance}   |   tmux attach -t {run.session}   |   ./pm stop {a.dance}")


def cmd_resume(a):
    d = _dance(a)
    run = d.run(getattr(a, "run_id", None))
    m = run.manifest()
    if not m:
        die(f"run '{run.id}' has no pm manifest yet: use `./pm train {d.name}`")
    a.train_cfg = m.get("train_cfg")
    a.eval_cfg = a.deploy_cfg = None
    start_training(a, a.iters or m.get("iterations"), resume=True)


def cmd_stop(a):
    d = _dance(a)
    run = d.run(getattr(a, "run_id", None))
    sess = live_session(run)
    if not sess and not train_procs(run.train):
        say(f"[pm] nothing running for {d.name}/{run.id}")
        return
    # graceful: the supervisor traps SIGTERM, terminates trainer and exits; last.pt (every last_every iterations) is the resume point
    sup = subprocess.run(["pgrep", "-f", f"supervisor.*{run.train}|pmpipe.supervisor --dance {d.name} --run-id {run.id}|ythn_supervisor.py --run {run.train}"], capture_output=True, text=True).stdout.split()
    for pid in sup:
        os.kill(int(pid), signal.SIGTERM)
    time.sleep(3)
    for w in ("eval", "tensorboard", "monitor", "pipeline", "train"):
        if sess and w in tmux_windows(sess):
            tmux_kill_window(sess, w)
    time.sleep(2)
    left = train_procs(run.train)
    for l in left:
        os.kill(int(l.split()[0]), signal.SIGTERM)
    say(f"[pm] stopped {d.name}/{run.id} at it {run.iteration()} (resume with `./pm resume {d.name}`; last.pt is saved every 1000 iterations)")


def status_text(run: Run, tcfg_iters=None) -> str:
    L = [f"==== {run.dance.name} / {run.id}   {now()} ====", f"layout: {'legacy-flat (adopted)' if run.meta.get('layout') == 'flat' else 'nested'}   root: {run.root}"]
    sess = live_session(run)
    L.append(f"tmux: {sess + ' windows=' + ','.join(tmux_windows(sess)) if sess else '(no session)'}")
    m = run.manifest()
    args = read_json(run.train / "args.json", {})
    tgt = args.get("iterations") or m.get("iterations")
    rows = run.train_rows(1)
    procs = train_procs(run.train)
    if rows:
        r = rows[0]
        L.append(f"train: it {int(float(r['iteration']))}/{tgt}  {float(r['it_per_s']):.1f} it/s  term {float(r['term_frac']):.3f}  sim_rew {float(r['sim_rew']):.3f}  wall {float(r['wall_h']):.2f} h   process: {'RUNNING' if procs else 'not running'}")
        if procs and tgt:
            it = int(float(r["iteration"]))
            ips = float(r["it_per_s"]) or 1
            L.append(f"eta: ~{(int(tgt) - it) / ips / 60:.0f} min")
    else:
        L.append("train: no log yet")
    L.append(f"checkpoints: {len(run.ckpts())} (latest {run.ckpts()[-1] if run.ckpts() else '-'})   markers: {' '.join(sorted(p.name for p in run.state.glob('*') if p.suffix == '')) if run.state.exists() else '-'}")
    ev = read_csv(run.eval / "summary.csv")
    if ev:
        L.append("eval (it balance fidelity combined surv mpjpe):")
        for r in ev[-5:]:
            L.append(f"   {int(r['iteration']):6d}  bal {float(r['balance']):5.1f}  fid {float(r['fidelity']):5.1f}  comb {float(r['combined']):5.1f}  surv {float(r['survival_robust']):.2f}  mpjpe {float(r['mpjpe_nominal_mm']):.1f}mm")
    b = read_json(run.best / "best_index.json", {})
    if b:
        L.append("selected (balance-first rule): " + (f"it{b['selected']}" if "selected" in b else "-") + "   | raw maxima (diagnostic only): " + ", ".join(f"{k} it{v}" for k, v in b.items() if k != "selected"))
    adm = run.admission()
    if adm:
        L.append(f"admission: {adm.get('real_robot_admission')}  final it{adm.get('final', {}).get('iteration', '-')}  sim2sim {'PASS' if (adm.get('sim2sim') or {}).get('passed') else 'FAIL/none'}")
    dep = read_json(run.deploy / "latest.json")
    if dep:
        L.append(f"package: {dep['zip']}  admitted={dep['admitted']} selfcheck={dep['selfcheck']}")
    sup = (run.logs / "supervisor.log")
    if sup.exists():
        L.append("supervisor (last 3): " + " | ".join(l[11:120] for l in sup.read_text().splitlines()[-3:]))
    g = gpus()
    if g:
        L.append("GPU: " + "; ".join(f"{x['index']} {x['name']} {x['used_mib']}/{x['total_mib']} MiB util {x['util']}%" for x in g))
    return "\n".join(L)


def cmd_status(a):
    d = _dance(a)
    if not d.dir.exists():
        die(f"no results for '{a.dance}' ({d.dir})")
    idx = d.index()
    if not idx["runs"]:
        print(f"{d.name}: data prepared={'yes' if d.npz.exists() else 'no'}, no training run yet")
        return
    run = d.run(getattr(a, "run_id", None))
    if a.watch:
        while True:
            os.system("clear")
            print(status_text(run))
            time.sleep(30)
    print(status_text(run))
    if len(idx["runs"]) > 1:
        print("runs:", ", ".join(idx["runs"]))


# ============================================================================================================================= eval / export / sim2sim / package / analyze
def cmd_eval(a):
    d = _dance(a)
    run = d.run(getattr(a, "run_id", None))
    ecfg = _eval_cfg(a, run)
    if a.ckpt:
        from . import evaluator as E
        out = run.eval / "manual"
        out.mkdir(parents=True, exist_ok=True)
        for c in a.ckpt:
            p = Path(c)
            if not p.exists():
                p = run.ckpt(int(c))
            r = E.evaluate(str(p), str(d.padded_dir), a.reps or ecfg["periodic"]["reps"], a.seed or ecfg["periodic"]["seed"], hard=a.hard)
            write_json(out / f"{p.stem}{'_hard' if a.hard else ''}.json", r)
            s = r["scores"]
            print(f"{p.name}: balance {s['balance']:.1f} fidelity {s['fidelity']:.1f} combined {s['combined']:.1f} surv_robust {s['survival_robust']:.2f} mpjpe {s['mpjpe_nominal_mm']:.1f}mm")
        return
    if live_session(run) and "eval" in tmux_windows(live_session(run)):
        say("[pm] the eval daemon is running (tmux window 'eval'); showing its results. Use --select for the final balance-first selection.")
    else:
        sh([PY, "-m", "pmpipe.evald", "--dance", d.name, "--run-id", run.id, "--once"], env={"PYTHONPATH": f"{REPO / 'pipeline'}:{REPO / 'src'}"}, check=True)
    from .post import candidates, hard_reeval, select
    if a.select:
        log = Logger(run.logs / "post.log")
        cand = candidates(run, ecfg)
        R = hard_reeval(run, cand, ecfg, log)
        sel = select(run, R, ecfg, log)
        print(f"[pm] balance-first selection: primary it{sel['primary']}  order {sel['order']}  roles {sel['roles']}")
    print(status_text(run))


def _final_it(run: Run, a) -> int | None:
    return getattr(a, "it", None)


def cmd_export(a):
    from .post import candidates, export_model, hard_reeval, select
    d = _dance(a)
    run = d.run(getattr(a, "run_id", None))
    ecfg, dcfg = _eval_cfg(a, run), _deploy_cfg(a, run)
    log = Logger(run.logs / "post.log")
    it = a.it
    if it is None:
        sel = read_json(run.final / "final_selection.json")
        if not sel:
            cand = candidates(run, ecfg)
            sel = select(run, hard_reeval(run, cand, ecfg, log), ecfg, log)
        it = sel["primary"]
    art = export_model(run, it, ecfg, dcfg, log)
    print(json.dumps(art, indent=1))
    if not art["parity"]["pass"]:
        die("ONNX parity FAILED", 4)


def cmd_sim2sim(a):
    from .post import export_model, select_and_validate, sim2sim, sim2sim_gate
    d = _dance(a)
    run = d.run(getattr(a, "run_id", None))
    ecfg, dcfg = _eval_cfg(a, run), _deploy_cfg(a, run)
    log = Logger(run.logs / "post.log")
    if a.it:  # one specific checkpoint, ad-hoc (does not change the admission record)
        art = export_model(run, a.it, ecfg, dcfg, log)
        res = sim2sim(run, art, ecfg, log, procs=a.procs, n=a.n)
        g = sim2sim_gate(res, ecfg)
        print(json.dumps(g, indent=1))
        sys.exit(0 if g["passed"] else 1)
    adm = select_and_validate(run, ecfg, dcfg, log, procs=a.procs)
    print(f"[pm] admission: {adm['real_robot_admission']}  final it{adm.get('final', {}).get('iteration')}  -> {run.reports / 'SELECTION_REPORT.md'}")
    sys.exit(0 if adm["real_robot_admission"] == "PASS" else 1)


def cmd_package(a):
    from .package import package
    d = _dance(a)
    run = d.run(getattr(a, "run_id", None))
    # V1.6.1: packaging defaults to the V1.6.1 deploy config (no authorisation gate, READY without timeout, auto-IDLE) regardless of the config the run was TRAINED with (v1/v2 in older run manifests); --deploy-cfg overrides
    dcfg = load_cfg("deploy", a.deploy_cfg) if getattr(a, "deploy_cfg", None) else load_cfg("deploy", "cpp_g1_v3_1")
    out = package(run, dcfg, allow_unadmitted=a.allow_unadmitted, build_check=False if a.no_build_check else None, compare_to=getattr(a, "compare_to", None))
    print(f"[pm] package: {out['zip']}\n     sha256 {out['sha256']}  {out['size'] / 1e6:.1f} MB  admitted={out['admitted']}  dev-selfcheck={out['selfcheck']}")


def cmd_quality(a):
    from .report import run_quality
    d = _dance(a)
    run = d.run(getattr(a, "run_id", None))
    its = [int(x) for x in a.its] if a.its else None
    out = run_quality(run, its=its, n_perturb=a.n_perturb, procs=a.procs, video=not a.no_video, log=Logger(run.logs / "quality.log"))
    print(f"[pm] quality report: {out['report']}\n     quad replay video: {out['video']}\n     models: {out['models']}")


def cmd_replay(a):
    """quad-panel replay video only (rollouts are produced if missing)."""
    from .post import export_model
    from .report import default_models, make_rollouts
    from .replay import render_quad
    d = _dance(a)
    run = d.run(getattr(a, "run_id", None))
    ecfg, dcfg = _eval_cfg(a, run), _deploy_cfg(a, run)
    its = sorted([int(x) for x in a.its] if a.its else default_models(run, ecfg))
    qdir = run.reports / "quality"
    files = make_rollouts(run, its, qdir, 0, a.procs, ecfg, dcfg, Logger(run.logs / "quality.log"))
    out = render_quad(d, [(f"it{i}", files[i][0]) for i in its], qdir / f"{d.name}_quad_replay.mp4", qdir / f"{d.name}_quad_sheet.png")
    print(f"[pm] replay: {out['video']}  sheet: {out['sheet']}")


def cmd_limits(a):
    from .limits import run_limits
    d = _dance(a)
    run = d.run(getattr(a, "run_id", None))
    it = a.it or int((read_json(run.final / "final_selection.json", {}) or {}).get("primary"))
    doc = run_limits(run, it, limits=a.limits, vel_variants=a.vel, n=a.n, procs=a.procs)
    print(f"[pm] limits study it{it}: {run.reports / f'limits_it{it}' / 'LIMITS_REPORT.md'}")


def _auto_snapshot(run: Run):
    """reproducibility snapshot at training start / resume: a failure only warns (never blocks training)."""
    try:
        from .repro import snapshot
        snapshot(run, log=Logger(run.logs / "repro.log", echo=False))
    except Exception as e:
        say(f"[pm] WARNING: repro snapshot failed (training continues): {e!r}")


def cmd_snapshot(a):
    from .repro import format_verify, snapshot, verify
    d = _dance(a)
    run = d.run(getattr(a, "run_id", None))
    if a.verify:
        res = verify(run, keep=a.keep)
        print(format_verify(res))
        sys.exit(0 if res["result"] == "PASS" else 1)
    man = snapshot(run, force=a.force)
    out = run.root / "repro"
    print(f"[pm] repro snapshot: {out}\n     HEAD {man['head'][:12]}  train.patch {man['patches']['train.patch']['sha256'][:12]} ({man['patches']['train.patch']['bytes']} B)  deploy.patch {man['patches']['deploy.patch']['sha256'][:12]}\n"
          f"     untracked files {man['untracked']['copied']} ({man['untracked']['bytes']} B), skipped {len(man['untracked']['skipped'])}; legacy src/tools diff sha12 {man['legacy_src_tools_diff_sha12']}\n"
          f"     rebuild instructions: {out / 'REPRODUCE.md'}   verify: ./pm snapshot {d.name} --verify")


def cmd_repair(a):
    """optional reference repair: rebuild root XY/Z from reliable contact (joints, frame count and timing untouched); --validate = compare with the frozen policy."""
    from . import repair as RP
    d = _dance(a)
    res = RP.repair(d, force=a.force, fc_xy=a.fc_xy)
    i = res["info"]
    print(f"[pm] repair {'(reused) ' if res.get('reused') else ''}OK: {res['dance'].npz}\n     padded training clip: {res['dance'].padded_npz}\n     max root move {i['max_dxy_m']:.2f} m XY / {i['max_dz_mm']:.0f} mm Z; joints/frames/timing/orientation unchanged (asserted)")
    if a.validate:
        from .repairval import validate
        run = d.run(getattr(a, "run_id", None))
        it = a.it or int((read_json(run.final / "final_selection.json", {}) or {}).get("primary"))
        out = validate(run, it, n_perturb=a.n_perturb, video=not a.no_video)
        print(f"[pm] repair report: {out['report']}\n     video: {out['video']}\n     verdict: {out['recommendation']['verdict']}")


def cmd_abtrain(a):
    """V1.3: freeze + launch the two short A/B runs (repaired-data run and original-data noise control) in tmux; evaluate later with `./pm abcompare`."""
    from . import abtest
    d = _dance(a)
    out = abtest.launch(d, a.iters)
    print(f"[pm] V1.3 A/B training started in tmux session {out['session']} (windows: train, monitor): attach with `tmux attach -t {out['session']}`\n     runs: {json.dumps(out['runs'], indent=1)}\n     init sha12 {out['init_sha12']} (== official pretrained widened)")


def cmd_abcompare(a):
    from .abreport import compare
    d = _dance(a)
    out = compare(d, video=not a.no_video)
    print(f"[pm] V1.3 report: {out['report']}\n     video: {out['video']}\n     verdict: {out['verdict']}")


def cmd_scan(a):
    """V1.4: prepare + analyse every dance in the dataset root, rank difficulty, flag data defects, pick ONE low-difficulty dance for the default recipe."""
    from .scan import scan
    out = scan(a.dances or None, force=a.force)
    print(f"[pm] scan report: {out['report']}\n     difficulty order (easy -> hard): {' < '.join(out['order'])}\n     chosen: {out['selection']['chosen']} (runner-up {out['selection']['runner_up']}); start with `./pm run {out['selection']['chosen']} --iters 40000`")


def _cr_params(a):
    from .contact_refine.config import Params
    p = Params()
    for kv in (getattr(a, "set", None) or []):
        k, v = kv.split("=", 1); cur = getattr(p, k); setattr(p, k, (v in ("1", "true", "True")) if isinstance(cur, bool) else type(cur)(v))
    return p


def cmd_contact_refine(a):
    """V1.6 B1: dance-independent contact identification + ground calibration + planted-foot root/leg-IK refinement + pre-registered data QA gate (writes NEW files only)."""
    from .contact_refine import api, report as RPT
    d = _dance(a); p = _cr_params(a); src = a.src
    if not (Path(src) if src else d.npz).exists():
        die(f"{src or d.npz} not found: run `./pm prepare {d.name}` first")
    if a.analyze_only:
        r = api.analyze_cmd(d, p, src, focus=a.focus)
        s = r["summary"]; c = s["contacts"]
        print(f"[pm] contact-refine analyze-only: {r['outdir']}\n     ground curve {s['ground']['range_mm'][0]:.0f}..{s['ground']['range_mm'][1]:.0f} mm; stance L/R {c['stance_frac'][0]:.2f}/{c['stance_frac'][1]:.2f}; true flight frames {c['flight_true_frames']} "
              f"(naive both-feet-gt25mm flight {s['naive_both_feet_gt25mm_over_z0_frac']:.2f}); focus window {s['focus_window']}")
        return
    r = api.refine_dance(d, p, force=a.force, src=src)
    if r.get("reused"):
        print(f"[pm] contact-refine outputs exist (use --force): {r['paths']['padded_npz']}\n     verdict {r['verdict']['verdict']}"); return
    diag, fd = api.diagnose(d, r, a.focus)
    if a.video:
        api.make_videos(d, r["outdir"], r["M"], a.focus, full=not a.no_full_video)
    extra = json.loads((r["outdir"] / "report_extra.json").read_text()) if (r["outdir"] / "report_extra.json").exists() else {}
    rp = RPT.write_report(r["outdir"], d.name, a.focus, extra)
    v = r["verdict"]
    print(f"[pm] contact-refine {v['verdict']} ({v['n_pass']}/{v['n_gates']}; failed {v['failed']})\n     refined: {r['paths']['padded_npz']}\n     qa_verdict: {r['outdir'] / 'qa_verdict.json'}\n     report: {rp}")


def cmd_contact_qa(a):
    """re-judge an existing <dance>_contactfix.npz against the original with the frozen QA gate -> qa_verdict.json (exit code 0 = PASS, 1 = FAIL)."""
    from .contact_refine import api
    d = _dance(a); v, res = api.qa_existing(d, _cr_params(a), a.src, a.fixed)
    for g in v["gates"]:
        print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['id']}: {g['value']} {g['op']} {g['threshold']} {g['unit']}")
    print(f"[pm] contact-qa {v['verdict']} ({v['n_pass']}/{v['n_gates']}) -> {api.out_dir(d.name) / 'qa_verdict.json'}")
    sys.exit(0 if v["verdict"] == "PASS" else 1)


def cmd_finetune(a):
    """V1.6: fine-tune an existing run's full-state checkpoint on another reference-motion directory (tmux + supervisor + eval daemon). See finetune.py."""
    from . import finetune as FT
    d = _dance(a)
    if not a.run_id:
        die("--run-id is required (e.g. ft_orig_v16)")
    out = FT.launch(d, a.from_ckpt, a.motion_dir or str(d.padded_dir), a.run_id, a.iters, seed=a.seed, lr=a.lr, mode=a.mode, base_run=a.base_run, num_envs=a.num_envs, gpu=a.gpu,
                    ckpt_every=a.ckpt_every, last_every=a.last_every, eval_every=a.eval_every, eval_also=a.eval_also, after=a.after, no_eval=a.no_eval, note=a.note or "")
    print(f"[pm] fine-tune started in tmux session {out['session']} (windows train/eval/monitor); target it{out['target']}; log {out['log']}\n     attach: tmux attach -t {out['session']}   status: ./pm ftstatus {a.dance} --run-id {a.run_id}")


def cmd_v17labels(a):
    """V1.7 C2: write the trainable contact-label sidecars <clip>_contact.npz (original + contact-fixed padded clips) into results/<dance>/v17/C_train/labels/. Motion files are not touched."""
    from . import v17ablate as VA
    VA.labels(_dance(a), force=a.force)


def cmd_v17ablate(a):
    """V1.7 C2: short (500-2000 it) resume-style fine-tune of a contact-label / contact-aware-reward variant (A0..A4, see v17ablate.VARIANTS); never touches the source run."""
    from . import v17ablate as VA
    d = _dance(a)
    out = VA.launch(d, a.variant, a.iters, seed=a.seed, run_id=a.run_id, extra=a.extra or [], from_ckpt=a.from_ckpt, base_run=a.base_run, num_envs=a.num_envs, gpu=a.gpu, after=a.after, no_eval=not a.eval, ckpt_every=a.ckpt_every, last_every=a.ckpt_every)
    print(f"[pm] v17-ablate {a.variant} started; log {out['log']}  target it{out['target']}  status: ./pm ftstatus {a.dance} --run-id {a.run_id or 'v17_' + a.variant}")


def cmd_ftstatus(a):
    from . import finetune as FT
    d = _dance(a)
    print(FT.status(d.run(a.run_id)))


def cmd_ftcompare(a):
    """V1.6: closed-loop + training comparison of two fine-tune runs (pre-registered protocol FT_PROTOCOL.md). See ftcompare.py."""
    from . import ftcompare as FC
    d = _dance(a)
    out = FC.compare(d, a.a, a.b, a_seeds=a.a_extra or [], b_seeds=a.b_extra or [], it=a.it, ref_a=a.ref_a, ref_b=a.ref_b, out_dir=a.out, n_perturb=a.n_perturb, procs=a.procs, video=not a.no_video, grids=tuple(a.grids.split(",")), tag=a.tag)
    print(f"[pm] ftcompare: {out['report']}\n     verdict: {out['verdict']}")


def cmd_v16export(a):
    """V1.6: acceptance (ftcompare GO + QA PASS) -> hard re-eval -> ONNX -> Sim2Sim -> C++ package; refuses (rc 5) when the acceptance is not GO. See v16export.py."""
    from . import v16export as VX
    d = _dance(a)
    if not a.run_id:
        die("--run-id is required")
    out = VX.v16_export(d, a.run_id, compare_json=a.compare_json, deploy_cfg=a.deploy_cfg, procs=a.procs, dry_run=a.dry_run)
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    sys.exit(0 if a.dry_run or out.get("admission") == "PASS" else 1)


def cmd_analyze(a):
    from .analyze import analyze_logs, analyze_training
    d = _dance(a)
    run = d.run(getattr(a, "run_id", None))
    if a.logs:
        analyze_logs(run, a.logs, a.onnx, a.mode)
    p = analyze_training(run)
    print("[pm] training analysis:", p)


# ============================================================================================================================= run (one-click offline pipeline)
def cmd_run(a):
    d = _dance(a)
    if not a.foreground:
        # re-exec ourselves inside tmux window `pipeline` so it survives the terminal
        run_id = a.run_id or ("smoke" if a.smoke else None)
        args = " ".join(shell_quote(x) for x in sys.argv[2:]) + " --foreground"
        tcfg = load_cfg("train", getattr(a, "train_cfg", None) or "robust_d_v1")
        rid = run_id or tcfg["name"]
        idx = d.index()
        run = d.run(rid) if rid in idx["runs"] else (d.run("legacy") if ("legacy" in idx["runs"] and not run_id) else None)
        sess = live_session(run) if run else None
        sess = sess or (f"pm-{d.name}" + ("" if rid == tcfg["name"] else f"-{rid}"))
        if "pm-pipeline" in tmux_windows(sess):
            say(f"[pm] pipeline window already running in tmux session {sess}: `tmux attach -t {sess}` (log: results/{d.name}/logs/pipeline.log)")
            return
        res = tmux_window(sess, "pm-pipeline", f"{PM_ENV}; {REPO / 'pm'} run {args} 2>&1 | tee -a {d.dir}/logs/pipeline.stdout")
        say(f"[pm] one-click pipeline started in tmux session '{sess}' window 'pm-pipeline' ({res}).  Follow: tmux attach -t {sess}   |   ./pm status {d.name}")
        (d.dir / "logs").mkdir(parents=True, exist_ok=True)
        return
    log = Logger(d.dir / "logs" / "pipeline.log")
    log(f"=== pm run {d.name} start  iters={a.iters} smoke={a.smoke} ===")
    from .prepare import prepare
    meta = prepare(d, a.source)
    log("prepare done", qa=meta.get("qa_verdict"))
    # ---- V1.5: STAGED training. The short-run gate used to run *while* the full 40000-iteration job was already training (race: a FAIL could arrive after
    # thousands of wasted iterations, and a late/missed check let training continue unchecked). Now stage 1 trains ONLY to the gate iteration and stops
    # (supervisor marker `gate_stage_done`); the gate is evaluated on that checkpoint; stage 2 (resume to the full target) is started ONLY on PASS.
    ecfg0 = _eval_cfg(a)
    gate_it = ecfg0.get("short_gate", {}).get("iters", 2000)
    tcfg0 = load_cfg("train", getattr(a, "train_cfg", None) or "robust_d_v1")
    target0 = a.iters or (100 if a.smoke else tcfg0["iterations"])
    a_train = argparse.Namespace(**{**vars(a), "foreground": False})
    pre = _find_run(a)  # existing run (if any) decides whether a gate is still pending
    gate_pending = not (a.no_gate or a.smoke or target0 <= gate_it or (pre is not None and pre.done("gate_pass")))
    if gate_pending:
        log("V1.5 staged training: stage 1 trains to the gate iteration only", gate_it=gate_it, full_target=target0)
        a_train.marker = "gate_stage_done"
        it_pre = pre.iteration() if pre is not None else 0
        if it_pre >= gate_it and (pre.train / f"ckpt_{gate_it:06d}.pt").exists():
            run = pre  # stage 1 was already done earlier (e.g. an interrupted pipeline): go straight to the gate
            run.mk(); run.mark("gate_stage_done") if not run.done("gate_stage_done") else None
        else:
            run = start_training(a_train, gate_it, resume=it_pre > 0)
            run.mark("gate_stage_started")
            _wait_stage(run, d, "gate_stage_done", log)
        # stage-1 trainer is finished: free its tmux window (a lingering window would make stage 2 think training is "already running")
        s1 = live_session(run)
        if s1 and "train" in tmux_windows(s1):
            tmux_kill_window(s1, "train")
        ok, why = _gate_on_checkpoint(run, d, gate_it, log)
        (run.reports / "SHORT_RUN_GATE.md").write_text(why)
        if not ok:
            run.mark("gate_fail"); log("short-run gate FAIL -> training NOT continued (stays at it%d). see reports/SHORT_RUN_GATE.md" % gate_it)
            cmd_stop(argparse.Namespace(dance=a.dance, run_id=run.id))
            die(f"short-run gate failed at it{gate_it}: no further iterations are spent before the cause is understood", 9)
        run.mark("gate_pass"); log("short-run gate PASS -> auto-continue to full target", gate_it=gate_it, target=target0)
        a_train.marker = None
        run = start_training(a_train, target0, resume=True)
    else:
        run = start_training(a_train, a.iters)
    ecfg = _eval_cfg(a, run)
    log("waiting for training", target=run.manifest().get("iterations"), it=run.iteration())
    while not run.done("train_done"):
        time.sleep(30)
        if run.done("gate_fail"):
            die("short-run gate failed earlier: refusing to continue training", 9)
        sess = live_session(run)
        if not train_procs(run.train) and not _supervisor_alive(run, d) and not (sess and "train" in tmux_windows(sess)) and not run.done("train_done"):
            die("training is neither running nor finished (supervisor window gone). Re-run `./pm resume` / `./pm run`", 10)
    log("training done", it=run.iteration())
    # legacy (pre-pipeline) orchestrator may still be running its own post chain on the same results: wait for it, never run two at once
    sess = live_session(run)
    if run.meta.get("layout") == "flat" and sess and "pipeline" in tmux_windows(sess) and not (os.environ.get("TMUX_PANE") and False):
        legacy = (run.dance.dir / "scripts/ythn_pipeline.py").exists()
        if legacy and not run.done("post_done"):
            log("legacy ythn_pipeline.py is still orchestrating: waiting for its post_done")
            while not run.done("post_done") and "pipeline" in tmux_windows(sess):
                time.sleep(30)
    while not run.done("eval_done"):
        time.sleep(20)
        if live_session(run) is None or "eval" not in tmux_windows(live_session(run)):
            sh([PY, "-m", "pmpipe.evald", "--dance", d.name, "--run-id", run.id, "--once"], env={"PYTHONPATH": f"{REPO / 'pipeline'}:{REPO / 'src'}"})
            run.mark("eval_done")
    log("evaluation done")
    from .post import select_and_validate
    dcfg = _deploy_cfg(a, run)
    adm = select_and_validate(run, ecfg, dcfg, Logger(run.logs / "post.log"), procs=a.procs)
    log("admission", status=adm["real_robot_admission"], final=adm.get("final", {}).get("iteration"))
    from .package import package
    from .analyze import analyze_training
    pk = None
    if adm["real_robot_admission"] == "PASS" or a.smoke:
        pk = package(run, dcfg, allow_unadmitted=a.smoke and adm["real_robot_admission"] != "PASS")
    else:
        log("NOT packaging: admission not PASS (Sim2Sim/parity/eval). Inspect reports/SELECTION_REPORT.md")
    analyze_training(run)
    qual = None
    try:
        from .report import run_quality
        qual = run_quality(run, log=Logger(run.logs / "quality.log"), eval_cfg=ecfg, deploy_cfg=dcfg)
        log("quality report", report=qual["report"])
    except Exception as e:  # the quality report must never block admission / packaging
        log("WARNING: quality report failed (admission/package unaffected):", repr(e))
    summary = [f"== pm run {d.name} finished ==", f"admission : {adm['real_robot_admission']} (final it{adm.get('final', {}).get('iteration')})",
               f"report    : {run.reports / 'SELECTION_REPORT.md'}", f"analysis  : {run.analysis / 'TRAINING_ANALYSIS.md'}", f"quality   : {qual['report'] if qual else '-'}", f"package   : {pk['zip'] if pk else '-'}"]
    log("\n".join(summary))
    print("\n".join(summary))
    sys.exit(0 if (pk and adm["real_robot_admission"] == "PASS") or a.smoke else 5)



def _find_run(a) -> Run | None:
    """The run `pm run` would use, if it already exists (no creation)."""
    d = _dance(a)
    idx = d.index()
    tname = getattr(a, "train_cfg", None) or "robust_d_v1"
    rid = getattr(a, "run_id", None) or ("smoke" if getattr(a, "smoke", False) else None)
    if rid and rid in idx["runs"]:
        return d.run(rid)
    if not rid and (tname in idx["runs"] or "legacy" in idx["runs"]):
        return d.run(tname if tname in idx["runs"] else "legacy")
    return None


def _supervisor_alive(run: Run, d: Dance) -> bool:
    r = subprocess.run(["pgrep", "-f", f"pmpipe.supervisor --dance {d.name} --run-id {run.id}"], capture_output=True, text=True)
    return bool(r.stdout.split())


def _wait_stage(run: Run, d: Dance, marker: str, log, poll: int = 20):
    """Block until the stage-1 supervisor wrote `marker`; abort if the supervisor/trainer vanished without it (never silently proceed)."""
    gone = 0
    while not run.done(marker):
        time.sleep(poll)
        alive = bool(train_procs(run.train)) or _supervisor_alive(run, d)
        gone = 0 if alive else gone + 1
        if gone >= 6:
            die(f"stage-1 training ended without marker '{marker}' (supervisor aborted?). See {run.logs / 'supervisor.log'}; fix and re-run `./pm run`", 10)
    log("stage 1 finished", it=run.iteration())


def _gate_on_checkpoint(run: Run, d: Dance, gate_it: int, log, timeout_s: int = 3 * 3600) -> tuple[bool, str]:
    """Wait for the (daemon's) evaluation of ckpt_<gate_it> + baseline, then apply short_gate(). Missing evaluation => run it here; never default to PASS."""
    need = [run.eval / "ckpt_000000.json", run.eval / f"ckpt_{gate_it:06d}.json"]
    t0 = time.time()
    while not all(p.exists() for p in need):
        sess = live_session(run)
        if time.time() - t0 > 600 and not (sess and "eval" in tmux_windows(sess)):
            log("eval daemon not running: evaluating the gate checkpoint here")
            sh([PY, "-m", "pmpipe.evald", "--dance", d.name, "--run-id", run.id, "--once"], env={"PYTHONPATH": f"{REPO / 'pipeline'}:{REPO / 'src'}"})
            if not all(p.exists() for p in need):
                return False, f"# 短程闸门 it{gate_it}: **FAIL**（评估结果缺失，按失败处理，不继续训练）\n"
            break
        if time.time() - t0 > timeout_s:
            return False, f"# 短程闸门 it{gate_it}: **FAIL**（等待评估超时 {timeout_s}s，按失败处理）\n"
        time.sleep(20)
    return short_gate(run, gate_it)


def short_gate(run: Run, it: int) -> tuple[bool, str]:
    """Cheap early warning after `it` iterations (relative to the widened pretrained baseline, same grid): numerics sane, nominal survives, balance & fidelity improve."""
    e0, e1 = read_json(run.eval / "ckpt_000000.json"), read_json(run.eval / f"ckpt_{it:06d}.json")
    rows = run.train_rows()
    last = rows[-1] if rows else None
    b, s = e0["scores"], e1["scores"]
    fin = all(all(v == v and abs(v) < 1e9 for v in (float(r[k]) for k in ("wm_loss", "pi_loss", "v_loss", "sim_rew"))) for r in rows)
    checks = [("训练日志有限", fin), ("nominal 存活 100%", e1["conditions"]["nominal"]["survival"] >= 1.0),
              ("鲁棒存活 >= 基线-0.02", s["survival_robust"] >= b["survival_robust"] - 0.02), ("综合分 >= 基线+5", s["combined"] >= b["combined"] + 5),
              ("平衡分 >= 基线", s["balance"] >= b["balance"]), ("nominal MPJPE <= 基线", s["mpjpe_nominal_mm"] <= b["mpjpe_nominal_mm"]),
              ("sim_rew < 1.2 (无奖励漏洞)", last is None or float(last["sim_rew"]) < 1.2)]
    ok = all(c for _, c in checks)
    txt = [f"# 短程闸门 it{it}（{now()}）: **{'PASS' if ok else 'FAIL'}**", "", "| 判据 | 结果 |", "|---|---|"] + [f"| {n} | {'PASS' if c else '**FAIL**'} |" for n, c in checks]
    txt += ["", f"基线: balance {b['balance']:.1f} fidelity {b['fidelity']:.1f} combined {b['combined']:.1f}; it{it}: balance {s['balance']:.1f} fidelity {s['fidelity']:.1f} combined {s['combined']:.1f}"]
    return ok, "\n".join(txt) + "\n"


def cmd_hwaudit(a):
    """V1.2: hardware profile (official URDF) + per-joint parameter audit across training / evaluator / Sim2Sim / deployment (read-only; never edits deploy assets)."""
    from . import hwaudit
    if a.write_profile:
        p = hwaudit.write_profile()
        print(f"[pm] hardware profile written: {hwaudit.PROFILE}  ({len(p['joints'])} joints, URDF sha256 {p['urdf']['sha256'][:12]})")
        return
    d = _dance(a)
    run = d.run(getattr(a, "run_id", None))
    it = a.it or (read_json(run.final / "final_selection.json", {}) or {}).get("primary") or run.ckpts()[-1]
    out = run.reports
    res = hwaudit.audit(run.ckpt(int(it)), d.padded_dir, out)
    hwaudit.write_report(res, out / "HARDWARE_AUDIT.md")
    print(f"[pm] hardware audit: {out / 'HARDWARE_AUDIT.md'}\n     findings S1={res['counts']['1']} S2={res['counts']['2']} S3={res['counts']['3']}  profile matches URDF: {res['profile']['matches_urdf']}")


# ============================================================================================================================= doctor / list
def cmd_doctor(a):
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")
    print("PartyMimic pipeline environment check")
    chk("main venv", Path(PY).exists(), PY)
    chk("sim2real venv (Sim2Sim, deploy assets)", Path(S2R_PY).exists(), S2R_PY)
    chk("converter (supermimic)", (core.SUPERMIMIC / "scripts/data/gmr_pkl_to_npz.py").exists())
    chk("C++ framework", (core.CPP / "CMakeLists.txt").exists(), str(core.CPP))
    chk("pretrained checkpoint", (REPO / "checkpoints/g1_mlp.pt").exists())
    chk("dataset root", DATASETS.exists(), str(DATASETS))
    chk("tmux", Path(core.TMUX).exists(), core.TMUX)
    g = gpus()
    chk("GPU", bool(g), "; ".join(f"{x['index']}:{x['name']} free {x['free_mib']} MiB util {x['util']}%" for x in g))
    chk("pip onnxruntime lib (dev C++ selfcheck)", bool(list(core.REPO.glob(".venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.*"))))
    chk("g++/cmake", bool(subprocess.run("which g++ cmake", shell=True, capture_output=True).stdout))
    print("configs:", json.dumps(core.list_cfgs(), ensure_ascii=False))
    print("dances in dataset:", ", ".join(sorted(p.name for p in DATASETS.iterdir() if (p / 'robot_motion.pkl').exists())) if DATASETS.exists() else "-")
    sys.exit(0 if ok else 1)


def cmd_list(a):
    rs = sorted(p.name for p in RESULTS.iterdir() if (p / "pm").exists() or (p.name in core.LEGACY_FLAT_DANCES)) if RESULTS.exists() else []
    for n in rs:
        d = Dance(n)
        idx = d.index()
        print(f"{n}: runs={list(idx['runs'])} default={idx['default_run']}")


# ============================================================================================================================= parser
def build_parser():
    p = argparse.ArgumentParser(prog="pm", description="PartyMimic Pipeline V1.0 - video robot_motion.pkl -> trained policy -> ONNX -> Sim2Sim -> C++ Sim2Real package (offline; never connects to a robot)",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="常用: ./pm run <舞蹈> --iters 40000   (一键离线全流程)\n详见 README_PIPELINE.md")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help, dance=True):
        s = sub.add_parser(name, help=help)
        if dance:
            s.add_argument("dance", help="舞蹈名 = /home/unitree/motion_datasets/video/<舞蹈名>/")
            s.add_argument("--run-id", default=None, help="run 名（默认=训练配置名；同一舞蹈可有多个 run）")
        s.set_defaults(fn=fn)
        return s
    s = add("prepare", cmd_prepare, "pkl -> 50Hz npz/csv + QA + padded 训练片段")
    s.add_argument("--source", default=None, help="换用其它源文件（默认 <舞蹈>/robot_motion.pkl）"); s.add_argument("--trim", default="auto", help="auto | 0 | N: 丢弃首部 N 帧(跟踪器初始化毛刺)"); s.add_argument("--force", action="store_true")
    s = add("preview", cmd_preview, "动作 QA 摘要 + 预览图（--video 额外渲染 mp4）"); s.add_argument("--video", action="store_true")

    def train_opts(s, train=True):
        s.add_argument("--iters", type=int, default=None, help="总训练轮次（默认取训练配置 iterations）")
        s.add_argument("--gpu", default="auto"); s.add_argument("--num-envs", type=int, default=None)
        s.add_argument("--foreground", action="store_true", help="前台运行(不用 tmux)"); s.add_argument("--no-eval", action="store_true"); s.add_argument("--no-tb", action="store_true"); s.add_argument("--tb-port", type=int, default=None)
        s.add_argument("--allow-concurrent", action="store_true", help="允许与已有训练共用同一块忙碌的 GPU")
    s = add("train", cmd_train, "启动训练（tmux + 监督器自动续训 + 评估守护 + TensorBoard）；已在跑的同配方任务会被接管而不是重复启动")
    train_opts(s); s.add_argument("--train-cfg", default="robust_d_v1"); s.add_argument("--eval-cfg", default=None); s.add_argument("--deploy-cfg", default=None); s.add_argument("--smoke", action="store_true")
    s = add("resume", cmd_resume, "断点续训 / 增加总轮次（--iters）"); train_opts(s)
    add("stop", cmd_stop, "停止训练（保留 last.pt，可 resume）")
    s = add("status", cmd_status, "训练/评估/准入/打包状态"); s.add_argument("--watch", action="store_true")
    s = add("eval", cmd_eval, "评估：补评未评估的 checkpoint；--select 做硬复评+平衡优先筛选；--ckpt 评估指定权重")
    s.add_argument("--select", action="store_true"); s.add_argument("--ckpt", nargs="*", default=None, help="路径或迭代数"); s.add_argument("--hard", action="store_true"); s.add_argument("--reps", type=int, default=None); s.add_argument("--seed", type=int, default=None); s.add_argument("--eval-cfg", default=None)
    s = add("export", cmd_export, "导出选定 checkpoint 的 ONNX + 一致性校验 + 部署资产"); s.add_argument("--it", type=int, default=None); s.add_argument("--eval-cfg", default=None); s.add_argument("--deploy-cfg", default=None)
    s = add("sim2sim", cmd_sim2sim, "筛选 + 导出 + Sim2Sim 部署链闭环 + 写准入记录（失败则不准入）"); s.add_argument("--it", type=int, default=None, help="只对该迭代做一次性 Sim2Sim(不改准入记录)")
    s.add_argument("--procs", type=int, default=16); s.add_argument("--n", type=int, default=None, help="Sim2Sim 采样初态数（默认取评估配置 sim2sim.n=14）；每工况总初态 = n + 10 个真机站姿"); s.add_argument("--eval-cfg", default=None); s.add_argument("--deploy-cfg", default=None)
    s = add("package", cmd_package, "生成 C++ Sim2Real 部署包 zip（Sim2Sim 未通过则拒绝）"); s.add_argument("--deploy-cfg", default=None, help="默认 cpp_g1_v3_1（V1.6.1）"); s.add_argument("--compare-to", default=None, help="与哪个旧包 zip 比较框架/模型差异（默认=deploy/latest.json 的 zip）"); s.add_argument("--allow-unadmitted", action="store_true", help="仅供离线检查：产出带 UNADMITTED 标记的包"); s.add_argument("--no-build-check", action="store_true")
    s = add("analyze", cmd_analyze, "训练评估分析；--logs 分析真机/Sim2Real 日志（目录或 export_logs 的 tar.gz）"); s.add_argument("--logs", default=None); s.add_argument("--onnx", default=None); s.add_argument("--mode", default="all")
    s = add("snapshot", cmd_snapshot, "保存复现训练所需的一切（git HEAD + 完整补丁 + 未跟踪文件 + 环境 + 输入哈希 + 时间线 + REPRODUCE.md）；--verify 在临时目录重建并校验")
    s.add_argument("--verify", action="store_true"); s.add_argument("--force", action="store_true"); s.add_argument("--keep", action="store_true", help="--verify: 保留临时重建目录")
    s = add("quality", cmd_quality, "动作质量分析 + 策略闭环回放(四宫格视频) + 模型对比 + 溯源 -> reports/quality/QUALITY_REPORT.md")
    s.add_argument("--its", nargs="*", default=None, help="要对比的迭代数（默认：筛选结果+平衡最高+综合最高，最多3个）"); s.add_argument("--n-perturb", type=int, default=8); s.add_argument("--procs", type=int, default=8); s.add_argument("--no-video", action="store_true")
    s = add("limits", cmd_limits, "冻结 checkpoint 的髋Roll 力矩/速度上限闭环对比（88/110/139 N·m，不重训、不改部署资产）")
    s.add_argument("--it", type=int, default=None); s.add_argument("--limits", type=float, nargs="*", default=[88, 110, 139]); s.add_argument("--vel", type=float, nargs="*", default=[20.0]); s.add_argument("--n", type=int, default=8); s.add_argument("--procs", type=int, default=16)
    s = add("repair", cmd_repair, "可选：按可靠接触重建参考根 XY/Z（关节/帧数/时序不变），--validate 与冻结模型做修复前后对比")
    s.add_argument("--force", action="store_true"); s.add_argument("--fc-xy", type=float, default=6.0); s.add_argument("--validate", action="store_true"); s.add_argument("--it", type=int, default=None); s.add_argument("--n-perturb", type=int, default=8); s.add_argument("--no-video", action="store_true")
    s = add("abtrain", cmd_abtrain, "V1.3：冻结历史成果，独立启动 2000 轮 A/B 短程训练（修复数据 + 原始数据噪声对照），不进入长程")
    s.add_argument("--iters", type=int, default=2000)
    s = add("abcompare", cmd_abcompare, "V1.3：公共（修复）参考上对比 A/B 模型，预登记判据，输出 GO/NO-GO 与对比视频"); s.add_argument("--no-video", action="store_true")
    s = add("finetune", cmd_finetune, "V1.6：基于已训练 run 的完整状态(last.pt)在指定动作目录上微调（resume 式：保留优化器/EMA/迭代数；tmux+监督器+评估守护）")
    s.add_argument("--from-ckpt", default="40000", help="起点：迭代数 | ckpt_*.pt | last.pt 路径（自动解析到同迭代的完整状态文件）"); s.add_argument("--motion-dir", default=None, help="padded 动作目录（默认=原始 <舞蹈>_padded）")
    s.add_argument("--iters", type=int, required=True, help="追加迭代数（目标 = 起点迭代 + N）"); s.add_argument("--seed", type=int, default=0); s.add_argument("--lr", type=float, default=None, help="覆盖配方 --lr（默认不改）")
    s.add_argument("--mode", choices=["resume", "init"], default="resume"); s.add_argument("--base-run", default=None, help="起点所属 run（默认=default run）"); s.add_argument("--num-envs", type=int, default=None); s.add_argument("--gpu", default="auto")
    s.add_argument("--ckpt-every", type=int, default=500); s.add_argument("--last-every", type=int, default=500); s.add_argument("--eval-every", type=int, default=2000); s.add_argument("--eval-also", default="40500,41000")
    s.add_argument("--after", default=None, help="排队：等该 run 的 train_done 后再开始"); s.add_argument("--no-eval", action="store_true"); s.add_argument("--note", default=None)
    s = add("v17-labels", cmd_v17labels, "V1.7 C2：生成 contact_refine 训练标签侧车 <clip>_contact.npz（原始/修正 padded 各一，不改动作文件）"); s.add_argument("--force", action="store_true")
    s = add("v17-ablate", cmd_v17ablate, "V1.7 C2：接触标签/接触感知奖励短程消融（A0..A4；resume 式微调，新 run 目录，默认 1000 轮，单种子）")
    s.add_argument("--variant", required=True); s.add_argument("--iters", type=int, default=1000); s.add_argument("--seed", type=int, default=0); s.add_argument("--from-ckpt", default="40000"); s.add_argument("--base-run", default="robust_d_v1")
    s.add_argument("--num-envs", type=int, default=None); s.add_argument("--gpu", default="auto"); s.add_argument("--ckpt-every", type=int, default=500); s.add_argument("--after", default=None); s.add_argument("--eval", action="store_true", help="启动评估守护（默认关，省 GPU）")
    s.add_argument("--extra", nargs="*", default=None, help="额外 train.py 参数（如 --cr_slip=1.0 --cr_zmp_relax=0.5）")
    s = add("ftstatus", cmd_ftstatus, "V1.6：微调 run 状态")
    s = add("ftcompare", cmd_ftcompare, "V1.6：两个微调 run 的训练与闭环对比（双参考、第三方指标、预登记 GO/NO-GO；协议见 results/<舞蹈>/v16/B_train/FT_PROTOCOL.md）")
    s.add_argument("--a", required=True, help="对照组 run"); s.add_argument("--b", required=True, help="实验组 run"); s.add_argument("--a-extra", nargs="*", default=None, help="对照组的额外重复 run(不同种子)"); s.add_argument("--b-extra", nargs="*", default=None)
    s.add_argument("--it", type=int, default=None, help="比较的迭代（默认=各 run 的最终迭代）"); s.add_argument("--ref-a", default=None, help="参考A（原始）padded 目录，默认 <舞蹈>_padded"); s.add_argument("--ref-b", default=None, help="参考B（修正）padded 目录")
    s.add_argument("--out", default=None, help="输出目录（默认 results/<舞蹈>/v16/B_train）"); s.add_argument("--n-perturb", type=int, default=8); s.add_argument("--procs", type=int, default=16); s.add_argument("--no-video", action="store_true"); s.add_argument("--grids", default="g1234,g4321,hard777", help="评估网格子集（调试用；正式对比用默认）"); s.add_argument("--tag", default="", help="输出文件名后缀，如 _4seed → FT_COMPARE_REPORT_4seed.md / ft_result_4seed.json（不覆盖已有版本）")
    s = add("v16-export", cmd_v16export, "V1.6：验收(ftcompare=GO 且数据 QA=PASS)→硬复评→ONNX→Sim2Sim→C++ 部署包；验收未通过则拒绝(返回码5)")
    s.add_argument("--compare-json", default=None); s.add_argument("--deploy-cfg", default=None, help="默认 cpp_g1_v3_1（若存在）否则 cpp_g1_v3 / cpp_g1_v2"); s.add_argument("--procs", type=int, default=16); s.add_argument("--dry-run", action="store_true", help="只检查验收闸门")
    s = sub.add_parser("scan", help="V1.4：扫描数据集中全部舞蹈（prepare + 质量分析），难度排名/数据缺陷/训练推荐，并选出一支低难度舞蹈"); s.set_defaults(fn=cmd_scan)
    s.add_argument("dances", nargs="*", help="只扫描这些舞蹈（默认全部）"); s.add_argument("--force", action="store_true", help="重新转换")
    s = add("replay", cmd_replay, "只生成四宫格闭环回放视频（参考 + 最多3个模型）"); s.add_argument("--its", nargs="*", default=None); s.add_argument("--procs", type=int, default=8); s.add_argument("--eval-cfg", default=None); s.add_argument("--deploy-cfg", default=None)
    s = add("run", cmd_run, "一键离线全流程：prepare→train→eval→筛选→export→sim2sim→package→analyze")
    train_opts(s); s.add_argument("--train-cfg", default="robust_d_v1"); s.add_argument("--eval-cfg", default=None); s.add_argument("--deploy-cfg", default=None); s.add_argument("--source", default=None)
    s.add_argument("--smoke", action="store_true", help="冒烟模式：100 轮、极小评估预算，只验证流程"); s.add_argument("--no-gate", action="store_true"); s.add_argument("--procs", type=int, default=16)
    s = add("hwaudit", cmd_hwaudit, "硬件 profile(官方URDF) + 29 关节 训练/评估/Sim2Sim/部署 参数审计 -> reports/HARDWARE_AUDIT.md（只读，不改部署）"); s.add_argument("--it", type=int, default=None); s.add_argument("--write-profile", action="store_true", help="仅重新生成 pipeline/configs/hardware/g1_29dof_rev1_0.yaml")
    s = add("contact-refine", cmd_contact_refine, "V1.6 B1：接触识别 + 地面标定 + 支撑脚钉住 + 根XYZ/腿IK 修正 + 预登记数据QA门（只写新文件 results/<舞蹈>/motions/<舞蹈>_contactfix{,_padded}/、results/<舞蹈>/v16/B_contact/）")
    s.add_argument("--analyze-only", action="store_true", help="只诊断原始数据，不修正"); s.add_argument("--force", action="store_true"); s.add_argument("--src", default=None, help="换用其它 50Hz npz（默认 results/<舞蹈>/motions/<舞蹈>/<舞蹈>.npz）")
    s.add_argument("--focus", type=int, default=320, help="重点窗口中心（舞蹈帧；真机/训练帧 = 舞蹈帧+50），默认 320 ≈ 真机 370"); s.add_argument("--video", action="store_true", help="渲染 原始 vs 修正 对照视频（全段 + 焦点慢放）"); s.add_argument("--no-full-video", action="store_true")
    s.add_argument("--set", nargs="*", default=None, help="覆盖算法参数 name=value（见 contact_refine/config.py::Params）")
    s = add("contact-qa", cmd_contact_qa, "V1.6 B1：对已有 <舞蹈>_contactfix.npz 独立重判数据QA门，输出 qa_verdict.json（PASS=0 / FAIL=1），B2 训练闸门"); s.add_argument("--src", default=None); s.add_argument("--fixed", default=None); s.add_argument("--set", nargs="*", default=None)
    sub.add_parser("doctor", help="环境自检").set_defaults(fn=cmd_doctor)
    sub.add_parser("list", help="已有舞蹈/run").set_defaults(fn=cmd_list)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
