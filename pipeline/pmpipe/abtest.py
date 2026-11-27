"""PartyMimic Pipeline V1.3 - short A/B training on original vs repaired reference data (2000 iterations, same recipe), `./pm abtrain` / `./pm abcompare`.

Design rules (to avoid fake improvements)
  * FROZEN: original motions, it40000 checkpoint/ONNX, deploy package, admission records are hashed before the experiment (FREEZE.json) and verified after it.
  * ONE variable: the reference motion directory.  Same official pretrained start (widened g1_mlp), same recipe `robust_d_v1` (frozen YAML copy), seed 0, 4096 envs,
    ckpt/last every 500, 2000 iterations.  The pipeline-launched runs live in results/<dance>/runs/<id>/ (nested layout) - the historical flat run is untouched.
  * NOISE CONTROL: the ORIGINAL-data recipe is trained a second time (same seed; GPU float non-determinism + legacy-script-vs-pipeline launcher) -> an estimate of the
    run-to-run spread that a claimed improvement has to exceed.  It is the only addition to the requested design and costs ~4 min.
  * COMMON EVALUATION: every model (original-data it2000, repaired-data it2000, control it2000, and the frozen it40000 as a scale anchor) is evaluated on the SAME
    reference - the repaired one - with the SAME initial states (seeded perturbations), SAME deployment control chain (deployed effort limits: hip roll 88 N.m kept) and the
    SAME metric code, plus the training-env grid on both references.  Metrics that depend on the reference's floating/ground use the height-aligned variant.
  * PRE-REGISTERED decision rule (written here, before any result existed; see DECISION below).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import quality as Q
from . import repair as P
from .core import PY, REPO, S2R_PY, Dance, Logger, Run, load_cfg, now, py_env, read_json, say, sh, sha256, snapshot_cfgs, tmux_has, tmux_window, tmux_windows, write_json

TAG_REP, TAG_CTRL = "ab_repaired_v1", "ab_origctl_v1"
PAD = Q.PAD
# ---- pre-registered decision rule ------------------------------------------------------------------------------------------------------------------------
# dynamics metrics (actual physics, deployment chain, common repaired reference).  direction: -1 = lower is better, +1 = higher is better
DYNAMICS = {"foot_slip_mps": -1, "pen_dance_max_mm": -1, "cop_margin_p05_mm": +1, "hr_peak_nm": -1, "guard_rewrite": -1, "root_xy_err_mean_m": -1}
FIDELITY = {"mpjpe_zaligned_mm": -1, "jmae_rad": -1}
DECISION = dict(
    rel_min=0.10,            # an improvement must be >= 10% of the baseline level AND larger than the run-to-run spread (|A - C|) AND better than BOTH A and C
    abs_min={"foot_slip_mps": 0.01, "pen_dance_max_mm": 0.3, "cop_margin_p05_mm": 1.0, "hr_peak_nm": 1.0, "guard_rewrite": 0.01, "root_xy_err_mean_m": 0.02, "mpjpe_zaligned_mm": 0.5, "jmae_rad": 0.002},
    need_improved=3,         # >= 3 of the 6 dynamics metrics significantly improved
    max_degraded=0,          # and none of the dynamics / fidelity metrics significantly degraded
    robustness="survival (deployment chain, all conditions) and evaluator-grid robust survival not below min(A, C)",
)


# ------------------------------------------------------------------------------------------------------------------------------------------- freeze
def _h(p):
    p = Path(p)
    return dict(path=str(p), sha12=sha256(p)[:12], size=p.stat().st_size, mtime=p.stat().st_mtime) if p.exists() else dict(path=str(p), missing=True)


def freeze_set(dance: Dance) -> list[Path]:
    r = dance.dir
    files = [dance.npz, dance.csv, dance.padded_npz, r / "training/init_from_g1_mlp.pt", r / "training/run/args.json", r / "training/run/train_log.csv", r / "pm/admission.json", r / "final/final_selection.json"]
    files += [r / f"training/run/ckpt_{i:06d}.pt" for i in (500, 1000, 2000, 4000, 32000, 40000)] + sorted((r / "final/onnx").glob("ythn_it040000*")) + sorted((r / "final/assets/it040000").glob("*"))
    files += sorted((r / "deploy").glob("*.zip")) + sorted((r / "deploy").glob("*.sha256")) + [dance.source_pkl()]
    return [f for f in files if f.exists()]


def freeze(dance: Dance, out: Path) -> dict:
    d = {str(f): _h(f) for f in freeze_set(dance)}
    doc = dict(time=now(), files=d); write_json(out, doc); return doc


def verify_freeze(dance: Dance, fz: Path) -> dict:
    old = read_json(fz)["files"]; bad = []
    for f, h in old.items():
        n = _h(f)
        if n.get("sha12") != h.get("sha12"): bad.append(dict(file=f, before=h.get("sha12"), after=n.get("sha12")))
    return dict(checked=len(old), changed=bad, ok=not bad)


# ------------------------------------------------------------------------------------------------------------------------------------------- runs
def make_run(dance: Dance, rid: str, motion_dir: Path, tcfg_name="robust_d_v1", note="") -> Run:
    idx = dance.index(); dflt = idx.get("default_run")
    idx["runs"].setdefault(rid, {"layout": "nested", "train_cfg": tcfg_name, "created": now(), "motion_dir": str(motion_dir), "experiment": "V1.3 A/B", "note": note}); idx["default_run"] = dflt
    dance.save_index(idx); run = Run(dance, rid, idx["runs"][rid]); run.mk()
    tcfg = load_cfg("train", tcfg_name); ecfg = load_cfg("eval", "balance_first_v1"); dcfg = load_cfg("deploy", "cpp_g1_v1")
    snap = snapshot_cfgs(run, {"train": tcfg, "eval": ecfg, "deploy": dcfg})
    write_json(run.pm / "manifest.json", dict(config_snapshot=snap, train_cfg_frozen=snap["train"]["frozen_copy"], eval_cfg_frozen=snap["eval"]["frozen_copy"], deploy_cfg_frozen=snap["deploy"]["frozen_copy"], dance=dance.name, run=rid, train_cfg=tcfg_name,
                                               iterations=2000, motion_dir=str(motion_dir), experiment="V1.3 A/B short run", updated=now()))
    import shutil
    legacy = dance.dir / "training/init_from_g1_mlp.pt"  # the EXACT file the original-data run started from (a fresh surgery gives identical tensors but different bytes)
    if legacy.exists():
        if not run.init_pt.exists() or sha256(run.init_pt) != sha256(legacy): shutil.copy2(legacy, run.init_pt)
    else:
        from .supervisor import ensure_init
        ensure_init(run, tcfg, Logger(run.logs / "init.log").__call__)
    return run


def launch(dance: Dance, iters=2000, log=print) -> dict:
    rep_dir = P.make_dance_like(dance).padded_dir
    if not (rep_dir / f"{dance.name}_repaired_padded.npz").exists(): raise RuntimeError("run `./pm repair` first")
    fz = dance.dir / "v13/FREEZE.json"
    if not fz.exists(): freeze(dance, fz); log("frozen", fz)
    runs = {TAG_REP: make_run(dance, TAG_REP, rep_dir, note="repaired reference (ythn_repaired_padded)"), TAG_CTRL: make_run(dance, TAG_CTRL, dance.padded_dir, note="noise control: original reference, same recipe, seed 0")}
    init_sha = {k: sha256(r.init_pt)[:12] for k, r in runs.items()}; ref_init = sha256(dance.dir / "training/init_from_g1_mlp.pt")[:12]
    assert all(v == ref_init for v in init_sha.values()), f"init checkpoints differ: {init_sha} vs {ref_init}"
    sess = f"pm-{dance.name}-v13"; env = f"export PYTHONPATH={REPO / 'pipeline'}:{REPO / 'src'}"
    cmds = " ; ".join(f"{PY} -u -m pmpipe.supervisor --dance {dance.name} --run-id {rid} --iterations {iters} --ckpt-every 500 --last-every 500 2>&1 | tee -a {runs[rid].logs}/supervisor.stdout" for rid in (TAG_REP, TAG_CTRL))
    res = tmux_window(sess, "train", f"{env}; {cmds}")
    mon = tmux_window(sess, "monitor", f"{env}; watch -n 20 'for r in {TAG_REP} {TAG_CTRL}; do echo $r; tail -n 1 {dance.dir}/runs/$r/train/train_log.csv | cut -c1-60; done; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'")
    return dict(session=sess, train=res, monitor=mon, init_sha12=init_sha, runs={k: str(v.train) for k, v in runs.items()})


# ------------------------------------------------------------------------------------------------------------------------------------------- export / rollouts
def export_eval_model(ckpt: Path, out: Path, motion_npz: Path, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True); onnx = out / "policy.onnx"
    if not onnx.exists():
        r = sh([PY, REPO / "tools/export_deploy_onnx.py", ckpt, onnx], env=py_env(), capture=True)
        if r.returncode: raise RuntimeError("export failed: " + r.stderr[-500:])
    v = sh([PY, REPO / "results/p0/verify_deploy_onnx.py", onnx], capture=True); ok = "DEPLOY_ONNX_INFERENCE=PASS" in v.stdout
    a = out / "assets"
    if not (a / "policy_params.txt").exists():
        r = sh([S2R_PY, REPO / "pipeline/pmpipe/deploy_assets.py", "--onnx", onnx, "--yaml", onnx.with_suffix(".yaml"), "--motion", motion_npz, "--out", a, "--provenance", f"V1.3 A/B eval export {ckpt}"], capture=True)
        if r.returncode: raise RuntimeError("deploy assets failed: " + r.stdout[-400:] + r.stderr[-400:])
    return dict(onnx=str(onnx), params=str(a / "policy_params.txt"), parity=ok, ckpt_sha12=sha256(ckpt)[:12], onnx_sha12=sha256(onnx)[:12])


CONDS = ["nominal", "push_f", "push_b", "push_l", "delay10", "kp0.85", "fric0.6", "noise"]


def rollouts(model: dict, padded: Path, out: Path, n_nominal=16, n_cond=8, procs=16) -> dict:
    out.mkdir(parents=True, exist_ok=True); files = {}
    for c in CONDS:
        n = n_nominal if c == "nominal" else n_cond; base = [S2R_PY, REPO / "pipeline/pmpipe/rollout.py", "--onnx", model["onnx"], "--params", model["params"], "--padded-npz", padded, "--cond", c]
        ls = []
        if c == "nominal":
            f = out / "nominal_ref.npz"
            if not f.exists(): sh(base + ["--out", f, "--init", "ref"], check=True, capture=True)
            ls.append(f)
        have = sorted(out.glob(f"{c}_perturb_*.npz"))
        if len(have) < n:
            [p.unlink() for p in have]; sh(base + ["--out", out / f"{c}_perturb.npz", "--init", "perturb", "--n", n, "--procs", min(procs, n)], check=True, capture=True); have = sorted(out.glob(f"{c}_perturb_*.npz"))
        files[c] = ls + have[:n]
    return files


# ------------------------------------------------------------------------------------------------------------------------------------------- metrics
_REF = {}


def _ref(padded):
    if padded not in _REF:
        R = Q.load_ref(Path(padded)); _REF[padded] = (R, P.ref_contact_labels(Path(padded)))
    return _REF[padded]


def _metrics(a):
    f, padded = a; R, lab = _ref(padded); ro = Q.load_rollout(f); T = len(ro["t"]); s = slice(PAD, T - PAD); BW = ro["meta"]["mass_kg"] * 9.81
    S = Q.fk_series(ro["root_p"], ro["root_q"], ro["q"], body_names=R["body_names"], want_pts=True)
    rb, rp, rq = R["body_p"][:T], R["root_p"][:T], R["root_q"][:T]; sp, sq = ro["root_p"], ro["root_q"]
    dyaw = Q._wrap(Q._yaw(sq) - Q._yaw(rq)); c, s_ = np.cos(dyaw), np.sin(dyaw); rel = rb - rp[:, None, :]
    rot = np.stack([np.stack([c, -s_], -1), np.stack([s_, c], -1)], -2); rxy = np.einsum("tij,tbj->tbi", rot, rel[..., :2]); dz = float((sp[s, 2] - rp[s, 2]).mean())
    err = lambda zs: float(np.linalg.norm(S["bodies"] - np.concatenate([sp[:, None, :2] + rxy, rp[:, 2:3, None] + rel[..., 2:3] + zs], -1), axis=-1).mean(-1)[s].mean() * 1000)
    glob = float(np.linalg.norm(S["bodies"] - rb, axis=-1).mean(-1)[s].mean() * 1000)
    # actual foot slip: horizontal speed of the contact patch of a LOADED foot that was loaded on the previous tick too
    patch, _ = P.contact_patch(S); BWf = ro["foot_fz"] > 0.10 * BW; v = np.linalg.norm(np.gradient(patch, 0.02, axis=0), axis=-1)
    okm = BWf[1:-1] & BWf[:-2] & BWf[2:]; okm[:PAD - 1] = False; okm[T - PAD - 1:] = False; sl = v[1:-1][okm]  # loaded on 3 consecutive ticks, inside the dance span
    d = Q.contact_dynamics(ro); lb = BWf
    out = dict(file=Path(f).name, survived=ro["meta"]["fell_t"] is None, fell_t=ro["meta"]["fell_t"], global_mpjpe_mm=glob, mpjpe_plain_mm=err(0.0), mpjpe_zaligned_mm=err(dz), root_z_offset_mm=dz * 1000,
               jmae_rad=float(np.abs(ro["q"][s] - R["q"][:T][s]).mean()), root_xy_err_mean_m=float(np.linalg.norm(sp[s, :2] - rp[s, :2], axis=1).mean()), root_xy_err_max_m=float(np.linalg.norm(sp[s, :2] - rp[s, :2], axis=1).max()),
               root_xy_err_end_m=float(np.linalg.norm(sp[T - PAD - 1, :2] - rp[T - PAD - 1, :2])), yaw_err_deg=float(np.degrees(np.abs(dyaw[s]).mean())),
               foot_slip_mps=float(sl.mean()) if len(sl) else 0.0, foot_slip_p95_mps=float(np.percentile(sl, 95)) if len(sl) else 0.0,
               label_agreement=float((lab[:T][s] == lb[s]).mean()), label_ref_swing_sim_loaded=float((~lab[:T][s] & lb[s]).mean()), label_ref_contact_sim_unloaded=float((lab[:T][s] & ~lb[s]).mean()),
               pen_dance_max_mm=d["penetration_mm"]["dance_max"], pen_dance_p95_mm=d["penetration_mm"]["dance_p95"], cop_margin_p05_mm=(d["balance"]["cop_margin_mm"] or {}).get("p05", float("nan")), com_outside_sole=d["balance"]["com_outside_sole_frac"],
               tilt_p95=d["balance"]["tilt_p95_rad"], hr_peak_nm=d["torque"]["hip_roll_peak_nm"], hr_sat=float(np.mean([d["torque"]["hip_roll_sat"]["left"], d["torque"]["hip_roll_sat"]["right"]])), peak_ratio_p95=d["torque"]["peak_ratio_p95"], sat_all=d["torque"]["sat_frac_mean"],
               guard_rewrite=d["guard"]["any_clip_rate"], guard_max_dev=d["guard"]["max_dev_rad"], ds=d["contact"]["double_support"], ss=d["contact"]["single_support"])
    return out


def run_metrics(files: dict, padded: Path, pool) -> dict:
    jobs = [(str(f), str(padded)) for c in files for f in files[c]]; res = pool.map(_metrics, jobs); out = {c: [] for c in files}; i = 0
    for c in files:
        for _ in files[c]: out[c].append(res[i]); i += 1
    return out


def agg(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return dict(mean=float(np.mean(v)), std=float(np.std(v)), n=len(v)) if v else None


# ------------------------------------------------------------------------------------------------------------------------------------------- decision
def decide(M: dict, A="orig", B="rep", C="ctrl") -> dict:
    """M[model] = metrics over the NOMINAL-condition rollouts (reference start + perturbed starts).  Rule: see DECISION (pre-registered)."""
    res = {}
    for name, d in {**DYNAMICS, **FIDELITY}.items():
        a, b, c = (M[m][name]["mean"] for m in (A, B, C)); base = (a + c) / 2; spread = abs(a - c)
        gain = (base - b) if d < 0 else (b - base)  # positive = B better than the mean of A and C
        better_than_both = (b < min(a, c)) if d < 0 else (b > max(a, c)); worse_than_both = (b > max(a, c)) if d < 0 else (b < min(a, c))
        need = max(DECISION["rel_min"] * abs(base), DECISION["abs_min"][name], spread)
        status = "improved" if (better_than_both and gain >= need) else "degraded" if (worse_than_both and -gain >= need) else "unchanged"
        res[name] = dict(A=a, B=b, C=c, spread_AC=spread, gain=gain, need=need, status=status, group="dynamics" if name in DYNAMICS else "fidelity")
    return res


def verdict(rows: dict, robust: dict) -> dict:
    imp = [k for k, v in rows.items() if v["group"] == "dynamics" and v["status"] == "improved"]
    deg = [k for k, v in rows.items() if v["status"] == "degraded"]
    rob_ok = all(robust[k]["B"] >= min(robust[k]["A"], robust[k]["C"]) - 1e-9 for k in robust)
    go = len(imp) >= DECISION["need_improved"] and len(deg) <= DECISION["max_degraded"] and rob_ok
    return dict(go=bool(go), improved=imp, degraded=deg, robustness_ok=bool(rob_ok), rule=DECISION)


# ------------------------------------------------------------------------------------------------------------------------------------------- post-hoc diagnostics
def _slip_diag(f):
    ro = Q.load_rollout(f); T = len(ro["t"]); BW = ro["meta"]["mass_kg"] * 9.81; K = Q.fk_series(ro["root_p"], ro["root_q"], ro["q"], want_pts=True); pts = K["pts"]; patch, _ = P.contact_patch(K)
    s = np.zeros(T, bool); s[PAD:T - PAD] = True; fz = ro["foot_fz"] / BW; vp = np.linalg.norm(np.gradient(patch, 0.02, axis=0), axis=-1); out = {}
    for th in (0.10, 0.30, 0.50):
        lb = (fz > th) & s[:, None]; out[f"patch_gt{int(th*100)}"] = float(vp[lb].mean()) if lb.any() else np.nan
    cp = Q.contact_point_slip(pts, PAD, T - PAD); out["point_mean"], out["point_p95"] = cp["mean"], cp["p95"]  # V1.4 standard definition lives in quality.contact_point_slip
    return out


def slip_diagnostics(dance: Dance, models=("orig", "rep", "ctrl", "anchor")) -> dict:
    cache = dance.dir / "v13/diagnostics.json"
    if cache.exists(): return read_json(cache)
    res = {}
    with Pool(16) as pool:
        for k in models:
            r = pool.map(_slip_diag, sorted((dance.dir / f"v13/traj/{k}").glob("nominal_*.npz"))); res[k] = {n: dict(mean=float(np.nanmean([x[n] for x in r])), std=float(np.nanstd([x[n] for x in r]))) for n in r[0]}
    write_json(cache, res); return res
