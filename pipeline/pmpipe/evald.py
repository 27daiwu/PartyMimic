"""Evaluation daemon (own process, runs next to the trainer; generalised from results/ythn/scripts/ythn_evald.py).

 * every ckpt_<it>.pt with it % every == 0 (or in `also`) is evaluated once by the independent evaluator (fixed condition grid, fixed seed)
 * eval/ckpt_<it>.json + eval/summary.csv + best/best_{balance,fidelity,combined}.pt (+ .json) + TensorBoard scalars (also copies train_log.csv)
 * the baseline (widened official pretrained policy, iteration 0) is evaluated first for reference
`python -m pmpipe.evald --dance D --run-id R` is what runs in tmux window `eval`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from .core import Dance, Logger, Run, load_cfg, now, read_json


def pick_best(rows, sel=None):
    """balance/fidelity/combined = raw per-score maxima (DIAGNOSTIC only); `selected` = the pipeline rule (pmpipe.selection, balance-first with tolerance)."""
    ok = [r for r in rows if r["nominal_survival"] >= 1.0]
    out = {}
    if sel:
        from .selection import rank_scores
        order = rank_scores({r["iteration"]: dict(scores=r, nominal_survival=r["nominal_survival"]) for r in rows}, sel)
        if order:
            out["selected"] = next(r for r in rows if r["iteration"] == order[0])
    if ok:
        out["balance"] = max(ok, key=lambda r: r["balance"])
        out["fidelity"] = max([r for r in ok if r["survival_robust"] >= 0.8] or ok, key=lambda r: r["fidelity"])
        out["combined"] = max(ok, key=lambda r: r["combined"])
    return out


def summarize(ev: Path) -> list[dict]:
    rows = []
    for j in sorted(ev.glob("ckpt_*.json")):
        r = json.loads(j.read_text())
        s = r["scores"]
        rows.append(dict(iteration=int(j.stem.split("_")[1]), json=j.name, checkpoint=r["checkpoint"], nominal_survival=r["conditions"]["nominal"]["survival"], **s))
    return rows


def write_summary(ev: Path, rows: list[dict]):
    keys = [k for k in rows[0] if k != "checkpoint"]
    with open(ev / "summary.csv.tmp", "w") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        [w.writerow(r) for r in sorted(rows, key=lambda r: r["iteration"])]
    os.replace(ev / "summary.csv.tmp", ev / "summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dance", required=True); ap.add_argument("--run-id", required=True)
    ap.add_argument("--eval-cfg", default=None); ap.add_argument("--every", type=int, default=None); ap.add_argument("--also", default=None)
    ap.add_argument("--once", action="store_true", help="one pass (evaluate what is pending, update summary/best) and exit")
    ap.add_argument("--reps", type=int, default=None); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--motion-dir", default=None, help="evaluate on this padded-motion directory instead of the dance's own (V1.6 fine-tune runs on a corrected reference)")
    a = ap.parse_args()
    run = Dance(a.dance).run(a.run_id)
    m = run.manifest()
    ecfg = load_cfg("eval", a.eval_cfg or m.get("eval_cfg_frozen") or m.get("eval_cfg") or "balance_first_v1")
    per = ecfg["periodic"]
    every = a.every or per["every"]
    also = {int(x) for x in (a.also.split(",") if a.also is not None else per.get("also", [])) if str(x)}
    reps, seed = a.reps or per["reps"], a.seed or per["seed"]
    log = Logger(run.logs / "evald.log")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from . import evaluator as E
    from torch.utils.tensorboard import SummaryWriter
    run.mk()
    tb_eval, tb_train = SummaryWriter(str(run.tb / "eval")), SummaryWriter(str(run.tb / "train"))
    tb_pos = run.state / "tb_train_pos.txt"
    last_tb = int(tb_pos.read_text()) if tb_pos.exists() else 0
    base_dst = run.eval / "ckpt_000000.json"
    motion_dir = str(a.motion_dir or run.dance.padded_dir)
    log("evald start", run=str(run.train), every=every, also=sorted(also), reps=reps, seed=seed)
    while True:
        try:
            lp = run.train / "train_log.csv"
            if lp.exists():
                new = 0
                with open(lp) as f:
                    for row in csv.DictReader(f):
                        try:
                            it = int(float(row["iteration"]))
                        except Exception:
                            continue
                        if it <= last_tb:
                            continue
                        for k, v in row.items():
                            if k != "iteration":
                                try:
                                    tb_train.add_scalar("train/" + k, float(v), it)
                                except Exception:
                                    pass
                        last_tb, new = it, new + 1
                if new:
                    tb_train.flush(); tb_pos.write_text(str(last_tb))
            todo = []
            if not base_dst.exists() and run.init_pt.exists():
                todo.append((0, run.init_pt, base_dst))
            for c in sorted(run.train.glob("ckpt_*.pt")):
                it = int(c.stem.split("_")[1])
                dst = run.eval / f"ckpt_{it:06d}.json"
                if (it % every == 0 or it in also) and it > 0 and not dst.exists():
                    todo.append((it, c, dst))
            for it, c, dst in todo:
                time.sleep(2)
                log("evaluating", c.name)
                r = E.evaluate(str(c), motion_dir, reps, seed)
                run.eval.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(".tmp"); tmp.write_text(json.dumps(r, indent=1)); os.replace(tmp, dst)
                s = r["scores"]
                for k in ("balance", "fidelity", "combined", "survival_robust", "mpjpe_nominal_mm", "mpjpe_robust_mean_mm", "tilt_p95_stress", "slip_mean", "torque_sat_mean", "pen_max_mm"):
                    tb_eval.add_scalar("eval/" + k, s[k], it)
                for cn, cr in r["conditions"].items():
                    tb_eval.add_scalar("eval_survival/" + cn, cr["survival"], it)
                tb_eval.flush()
                log(f"it {it}: balance {s['balance']:.1f} fidelity {s['fidelity']:.1f} combined {s['combined']:.1f} surv_robust {s['survival_robust']:.2f} mpjpe {s['mpjpe_nominal_mm']:.1f}mm ({r['eval_seconds']:.0f}s)")
            rows = summarize(run.eval)
            if rows:
                write_summary(run.eval, rows)
                run.best.mkdir(parents=True, exist_ok=True)
                meta = {}
                for k, r in pick_best([r for r in rows if r["iteration"] > 0], ecfg["selection"]).items():
                    src, dst = run.ckpt(r["iteration"]), run.best / f"best_{k}.pt"
                    cur = read_json(run.best / f"best_{k}.json", {})
                    if src.exists() and cur.get("iteration") != r["iteration"]:
                        shutil.copy2(src, str(dst) + ".tmp"); os.replace(str(dst) + ".tmp", dst)
                        (run.best / f"best_{k}.json").write_text(json.dumps(dict(iteration=r["iteration"], score=r["balance" if k == "selected" else k], rule="balance_first (pmpipe.selection)" if k == "selected" else "raw maximum (diagnostic)", scores={x: r[x] for x in ("balance", "fidelity", "combined")}, source=str(src), eval=r["json"]), indent=1))
                        log(f"NEW BEST {k}: it {r['iteration']} score {r['balance' if k == 'selected' else k]:.2f}")
                    meta[k] = r["iteration"]
                (run.best / "best_index.json").write_text(json.dumps(meta, indent=1))
            if a.once:
                log("--once: pass finished")
                return
            if run.done("train_done") and not todo:
                pend = [it for it in run.ckpts() if (it % every == 0 or it in also) and not (run.eval / f"ckpt_{it:06d}.json").exists()]
                if not pend:
                    log("training finished and all evaluations done")
                    run.mark("eval_done")
                    return
        except Exception:
            log("evald error:\n" + traceback.format_exc())
            time.sleep(20)
        time.sleep(30)


if __name__ == "__main__":
    main()
