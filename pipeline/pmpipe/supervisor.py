"""Training: command construction from the versioned YAML, init-checkpoint surgery, and the supervisor (auto-resume, anomaly rollback, backups).

Generalised from results/ythn/scripts/{train_ythn.sh,ythn_supervisor.py} (validated on YTHN) - same recipe, same supervisor logic.
`python -m pmpipe.supervisor --dance D --run-id R --iterations N` is what runs inside tmux window `train`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .core import PY, REPO, Dance, Logger, Run, die, load_cfg, now, py_env, read_json, say, sh, write_json

# train.py arguments that identify the *recipe* (compared when adopting a run that was started before the pipeline existed)
NON_RECIPE = {"iterations", "resume", "init", "log_dir", "ckpt_every", "last_every", "num_envs", "motion_dir", "seed"}


def ensure_init(run: Run, tcfg: dict, log=print) -> Path:
    """init checkpoint = official pretrained g1_mlp.pt widened to the xi of the robot variant (tools/p4_ckpt_surgery.py)."""
    if run.init_pt.exists():
        return run.init_pt
    src = REPO / tcfg["init"]["checkpoint"]
    if not src.exists():
        die(f"pretrained checkpoint missing: {src}")
    run.init_pt.parent.mkdir(parents=True, exist_ok=True)
    tmp = run.init_pt.with_suffix(".tmp.pt")
    log(f"init surgery {src.name} -> {run.init_pt}  (robot {tcfg['robot']})")
    r = sh([PY, REPO / tcfg["init"]["surgery"], src, tmp, "--robot", tcfg["robot"]], env=py_env(), log=run.logs / "init_surgery.log")
    if r.returncode:
        die(f"init surgery failed (see {run.logs / 'init_surgery.log'})", 3)
    os.replace(tmp, run.init_pt)
    return run.init_pt


def train_argv(run: Run, tcfg: dict, iterations: int, num_envs: int, resume: bool, ckpt_every=None, last_every=None, extra=()) -> list[str]:
    """python -m partymimic.train argv. Resume from last.pt when present, otherwise start from the widened pretrained checkpoint."""
    dance = run.dance
    start = ["--resume", str(run.train / "last.pt")] if resume and (run.train / "last.pt").exists() else ["--init", str(run.init_pt)]
    a = [PY, "-B", "-u", "-m", "partymimic.train", "--robot", tcfg["robot"], "--motion_dir", str(run.meta.get("motion_dir") or dance.padded_dir), "--log_dir", str(run.train),
         "--seed", str(tcfg.get("seed", 0)), *start, "--iterations", str(iterations),
         "--ckpt_every", str(ckpt_every or tcfg["ckpt_every"]), "--last_every", str(last_every or tcfg["last_every"]), "--num_envs", str(num_envs)]
    a += list(tcfg["train_args"]) + list(extra)
    return a


def parse_argv_to_dict(args: list[str]) -> dict:
    """--k=v / --k v / --flag / --no-flag  ->  {k: value} (types guessed) for comparing a recipe with a run's args.json."""
    out, i = {}, 0

    def conv(v):
        for f in (int, float):
            try:
                return f(v)
            except ValueError:
                pass
        return v
    toks = []
    for t in args:
        if t.startswith("--") and "=" in t:
            k, v = t.split("=", 1)
            toks += [k, v]
        else:
            toks.append(t)
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            k = t[2:]
            if k.startswith("no-"):
                out[k[3:]] = False
            elif i + 1 < len(toks) and (not toks[i + 1].startswith("--") or _is_num(toks[i + 1])):
                out[k] = conv(toks[i + 1]); i += 1
            else:
                out[k] = True
        i += 1
    return out


def _is_num(s):
    try:
        float(s); return True
    except ValueError:
        return False


def recipe_diff(tcfg: dict, args_json: dict) -> list[str]:
    """Differences between the YAML recipe and a run's args.json (empty = same recipe). Only keys the YAML sets are compared."""
    want = parse_argv_to_dict(["--robot", tcfg["robot"]] + list(tcfg["train_args"]))
    diffs = []
    for k, v in want.items():
        if k in NON_RECIPE:
            continue
        have = args_json.get(k, "<absent>")
        if isinstance(v, (int, float)) and isinstance(have, (int, float)) and not isinstance(v, bool):
            same = math.isclose(float(v), float(have), rel_tol=1e-9, abs_tol=1e-12)
        else:
            same = (v == have) or (v == "" and have in ("", None))
        if not same:
            diffs.append(f"{k}: yaml={v!r} run={have!r}")
    return diffs


# ----------------------------------------------------------------------------------------------------------------------------- supervisor
def tail_rows(path, n):
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        return rows[-n:], (int(float(rows[-1]["iteration"])) if rows else 0)
    except Exception:
        return [], 0


def anomaly(rows, it):
    """Reason string or None (NaN/inf, loss explosion, survival collapse)."""
    if not rows:
        return None
    for r in rows[-4:]:
        for k in ("wm_loss", "pi_loss", "v_loss", "imag_rew", "sim_rew", "pi_gradnorm"):
            try:
                x = float(r[k])
            except Exception:
                return f"unparsable {k}"
            if not math.isfinite(x):
                return f"non-finite {k} at it {r['iteration']}"
        if float(r["wm_loss"]) > 1e3 or float(r["v_loss"]) > 1e4:
            return f"loss explosion at it {r['iteration']} wm {r['wm_loss']} v {r['v_loss']}"
    if it > 1500 and len(rows) >= 8 and all(float(r["term_frac"]) > 0.6 for r in rows[-8:]):
        return f"survival collapse at it {it} (term_frac>0.6 for 8 rows)"
    return None


def trim_log_to_iteration(run: Run, it: int, log):
    lp = run.train / "train_log.csv"
    if not lp.exists():
        return
    rows = list(csv.reader(open(lp)))
    keep = [rows[0]] + [r for r in rows[1:] if int(float(r[0])) <= it]
    if len(keep) != len(rows):
        csv.writer(open(lp, "w")).writerows(keep)
        log("train_log trimmed", it=it, dropped=len(rows) - len(keep))


def last_pt_iteration(run: Run) -> int | None:
    lp = run.train / "last.pt"
    if not lp.exists():
        return None
    import torch
    return int(torch.load(lp, map_location="cpu", weights_only=False)["iteration"])


def kill_group(p):
    if p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except Exception:
        pass
    for _ in range(40):
        if p.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        pass


def supervise(run: Run, tcfg: dict, iterations: int, num_envs: int, gpu: int | None, marker: str = "train_done",
              ckpt_every=None, last_every=None, stall_s=None) -> int:
    sv = tcfg.get("supervisor", {})
    max_restarts, stall_s = sv.get("max_restarts", 30), stall_s or sv.get("stall_s", 900)
    keep_bak, min_free = sv.get("keep_backups", 3), sv.get("min_free_disk_gib", 15)
    log = Logger(run.logs / "supervisor.log")
    run.mk()
    stdout = run.logs / "train_stdout.log"
    restarts = rollbacks = 0
    last_bak_mtime = 0.0
    env = py_env({"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else {})
    for k in ("PM_HOOK", "PM_HOOK_AT", "PM_PROFILE"):
        os.environ.pop(k, None)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    log("supervisor start", run=str(run.train), iterations=iterations, num_envs=num_envs, gpu=gpu, config=tcfg["name"])
    while restarts <= max_restarts and not stop["flag"]:
        _, it0 = tail_rows(run.train / "train_log.csv", 1)
        if it0 >= iterations and run.ckpt(iterations).exists():
            log("target iterations reached", it=it0)
            break
        free = shutil.disk_usage(str(run.train)).free / 2**30
        if free < min_free:
            log("ABORT: free disk below limit", free_gib=free, min_gib=min_free)
            return 3
        li = last_pt_iteration(run)
        if li is not None:
            trim_log_to_iteration(run, li, log)
        argv = train_argv(run, tcfg, iterations, num_envs, resume=True, ckpt_every=ckpt_every, last_every=last_every)
        with open(stdout, "a") as so:
            so.write(f"\n===== {now()} supervisor launch #{restarts} (log it {it0}, {'RESUME' if li is not None else 'FRESH'}) =====\n")
            so.flush()
            p = subprocess.Popen(argv, cwd=str(REPO), stdout=so, stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, **env))
            log("training launched", pid=p.pid, restart=restarts, resume_it=it0, mode="RESUME" if li is not None else "FRESH")
            reason, t_last, last_it = None, time.time(), it0
            while p.poll() is None:
                for _ in range(20):
                    time.sleep(1)
                    if stop["flag"] or p.poll() is not None:
                        break
                if stop["flag"]:
                    reason = "stop requested"
                    break
                rows, it = tail_rows(run.train / "train_log.csv", 12)
                if it > last_it:
                    last_it, t_last = it, time.time()
                lp = run.train / "last.pt"
                if lp.exists() and lp.stat().st_mtime > last_bak_mtime + 1 and time.time() - lp.stat().st_mtime > 5:
                    last_bak_mtime = lp.stat().st_mtime
                    bak = run.train / f"last_bak_{it:06d}.pt"
                    shutil.copy2(lp, str(bak) + ".tmp")
                    os.replace(str(bak) + ".tmp", bak)
                    for old in sorted(run.train.glob("last_bak_*.pt"))[:-keep_bak]:
                        old.unlink()
                    log("backup", file=bak.name)
                an = anomaly(rows, it)
                if an:
                    reason = "anomaly: " + an
                    break
                if time.time() - t_last > stall_s:
                    reason = f"stall: no progress for {stall_s}s"
                    break
            if reason == "stop requested":
                log("stop requested: terminating training (resume later with `./pm resume`)")
                kill_group(p)
                return 130
            if reason:
                log("TERMINATING training", reason=reason)
                kill_group(p)
                if reason.startswith("anomaly"):
                    baks = sorted(run.train.glob("last_bak_*.pt"))
                    rollbacks += 1
                    if rollbacks > 3 or not baks:
                        log("ABORT: anomaly persists / no backup", rollbacks=rollbacks)
                        return 4
                    use = baks[-2] if len(baks) >= 2 else baks[-1]
                    shutil.copy2(use, str(run.train / "last.pt") + ".tmp")
                    os.replace(str(run.train / "last.pt") + ".tmp", run.train / "last.pt")
                    trim_log_to_iteration(run, int(use.stem.split("_")[-1]), log)
                    log("ROLLBACK", to=use.name, rollbacks=rollbacks)
                restarts += 1
                continue
            rc = p.returncode
            log("training process exited", rc=rc)
            _, it1 = tail_rows(run.train / "train_log.csv", 1)
            if rc == 0 and it1 >= iterations:
                log("training complete", it=it1)
                break
            restarts += 1
            time.sleep(30)
    else:
        if stop["flag"]:
            return 130
        log("ABORT: max restarts exceeded")
        return 5
    _, itf = tail_rows(run.train / "train_log.csv", 1)
    if itf >= iterations:
        run.mark(marker, now() + f" it {itf}\n")
        log("train_done marker written")
        return 0
    return 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dance", required=True); ap.add_argument("--run-id", required=True); ap.add_argument("--iterations", type=int, required=True)
    ap.add_argument("--num-envs", type=int, default=None); ap.add_argument("--gpu", type=int, default=None); ap.add_argument("--marker", default="train_done")
    ap.add_argument("--ckpt-every", type=int, default=None); ap.add_argument("--last-every", type=int, default=None)
    a = ap.parse_args()
    run = Dance(a.dance).run(a.run_id)
    m = run.manifest()
    tcfg = load_cfg("train", m["train_cfg_frozen"] if "train_cfg_frozen" in m else m["train_cfg"])
    sys.exit(supervise(run, tcfg, a.iterations, a.num_envs or tcfg["num_envs"], a.gpu, a.marker, a.ckpt_every, a.last_every))


if __name__ == "__main__":
    main()
