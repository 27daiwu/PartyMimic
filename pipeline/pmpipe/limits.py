"""PartyMimic Pipeline V1.2 - frozen-policy actuator-limit study: `./pm limits <dance> --it N --limits 88 110 139`.

Question: how does ONE frozen checkpoint behave in the deployment-chain closed loop (ONNX -> authority/ramp -> C++ guard semantics -> 200 Hz PD -> torque clip,
MuJoCo) when the hip-roll torque limit is 88 (deployed, kept), 110 (training DR lower end 0.8x139) or 139 N.m (URDF Rev 1.0)?  Nothing is retrained and the deploy
assets are never touched: each variant is a COPY of the run's policy_params.txt with only the hip-roll effort (joints 1 and 7) replaced; the guard clamps the
command with the same limit (as the real C++ guard does), so the study includes the guard's reaction. Optional `--vel` adds the hip-roll velocity limit variant.
Metrics come straight from the rollout logs (no FK): survival, joint MAE to the reference, tilt, hip-roll peak pre-clip torque / saturation / peak speed,
guard rewrite rate, penetration; anchored MPJPE is computed (FK) on the reference-start rollouts only.
"""
from __future__ import annotations

import json
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import quality as Q
from .core import REPO, S2R_PY, Logger, Run, now, read_json, sh, write_json

ROLLOUT = REPO / "pipeline/pmpipe/rollout.py"
CONDS = ["nominal", "push_f", "push_b", "push_l", "delay10", "kp0.85", "fric0.6", "noise"]
HIP_ROLL = (1, 7)  # joint-order indices of left/right hip roll


def variant_params(src: Path, dst: Path, hip_roll_nm: float | None, hip_roll_vel: float | None) -> Path:
    lines = []
    for l in src.read_text().splitlines():
        k = l.split()[0] if l.strip() and not l.startswith("#") else None
        if k in ("effort_lim", "vel_lim") and (hip_roll_nm if k == "effort_lim" else hip_roll_vel) is not None:
            p = l.split(); vals = p[2:]
            for i in HIP_ROLL: vals[i] = repr(float(hip_roll_nm if k == "effort_lim" else hip_roll_vel))
            l = " ".join(p[:2] + vals)
        lines.append(l)
    dst.write_text("\n".join(lines) + "\n"); return dst


def _analyse(a):
    f, lim = a; ro = Q.load_rollout(f); T = len(ro["t"]); dsl = slice(Q.PAD, T - Q.PAD); eff = np.array(ro["meta"]["effort_lim"]); hr = list(HIP_ROLL)
    tau = ro["tau_ratio"][dsl][:, hr] * eff[hr]  # pre-clip PD torque magnitude of the hip rolls (N.m)
    ref_q = np.load(ro["meta"]["padded"])["joint_pos"][:T]
    return dict(file=Path(f).name, survived=ro["meta"]["fell_t"] is None, fell_t=ro["meta"]["fell_t"], jmae=float(np.abs(ro["q"][dsl] - ref_q[dsl]).mean()), tilt_p95=float(np.percentile(ro["tilt"][dsl], 95)),
                hr_peak_nm=float(tau.max()), hr_p99_nm=float(np.percentile(tau, 99)), hr_sat=float(ro["sat_frac"][dsl][:, hr].mean()), hr_speed_max=float(np.abs(ro["dq"][dsl][:, hr]).max()), hr_speed_p99=float(np.percentile(np.abs(ro["dq"][dsl][:, hr]), 99)),
                guard_rewrite=float((np.abs(ro["qcmd"][dsl] - ro["qpre"][dsl]) > 1e-6).any(1).mean()), pen_max_mm=float(ro["pen"][dsl].max() * 1000), cop_ok=float(np.isfinite(ro["cop"][dsl, 0]).mean()))


def _mpjpe(a):
    dance_padded, f = a
    from .core import Dance
    ref = Q.load_ref(dance_padded); ro = Q.load_rollout(f)
    d = Q.trajectory_decomposition(ref, dict(ank=None, h=None, cal=None), ro) if False else None
    S = Q.fk_series(ro["root_p"], ro["root_q"], ro["q"], body_names=ref["body_names"]); T = len(ro["t"]); sp, sq = ro["root_p"], ro["root_q"]
    rb, rp, rq = ref["body_p"][:T], ref["root_p"][:T], ref["root_q"][:T]
    dyaw = Q._wrap(Q._yaw(sq) - Q._yaw(rq)); c, s = np.cos(dyaw), np.sin(dyaw); rel = rb - rp[:, None, :]
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2); rxy = np.einsum("tij,tbj->tbi", rot, rel[..., :2])
    anc = np.concatenate([sp[:, None, :2] + rxy, rp[:, 2:3, None] + rel[..., 2:3]], -1)
    return float(np.linalg.norm(S["bodies"] - anc, axis=-1).mean(-1)[Q.PAD:T - Q.PAD].mean() * 1000)


def run_limits(run: Run, it: int, limits=(88, 110, 139), vel_variants=(20.0,), n: int = 8, conds=CONDS, procs: int = 16, log=None) -> dict:
    from .post import export_model
    from .core import load_cfg
    log = log or Logger(run.logs / "limits.log")
    m = run.manifest(); ecfg = load_cfg("eval", m.get("eval_cfg") or "balance_first_v1"); dcfg = load_cfg("deploy", m.get("deploy_cfg") or "cpp_g1_v1")
    art = export_model(run, it, ecfg, dcfg, log)  # frozen checkpoint -> same ONNX / params as the deploy package
    base_params = Path(art["assets"]) / "policy_params.txt"
    out = run.reports / f"limits_it{it}"; out.mkdir(parents=True, exist_ok=True)
    variants = [(f"hr{int(l)}", l, None) for l in limits] + [(f"hr{int(max(limits))}_v{int(v)}", max(limits), v) for v in vel_variants]
    jobs = []
    for tag, lim, vel in variants:
        pp = variant_params(base_params, out / f"policy_params_{tag}.txt", lim, vel)
        for c in conds:
            for init in ("ref", "perturb"):
                if c != "nominal" and init == "ref": continue
                dst = out / "traj" / f"{tag}_{c}_{init}.npz"
                if not dst.exists() or (init == "perturb" and len(list(dst.parent.glob(f"{tag}_{c}_{init}_*.npz"))) < n):
                    jobs.append((tag, c, init, pp, dst))
    (out / "traj").mkdir(exist_ok=True)
    for tag, c, init, pp, dst in jobs:  # each call fans out over `procs` processes (n perturbed starts) - small workloads, sequential calls
        if init == "perturb": [p.unlink() for p in dst.parent.glob(f"{tag}_{c}_{init}_*.npz")]
        r = sh([S2R_PY, ROLLOUT, "--onnx", art["onnx"], "--params", pp, "--padded-npz", run.dance.padded_npz, "--out", dst, "--cond", c, "--init", init, "--n", n if init == "perturb" else 1, "--procs", min(procs, n)], capture=True)
        if r.returncode: raise RuntimeError(f"rollout {tag}/{c}/{init}: {r.stderr[-600:]}")
    log("limit rollouts done", variants=[v[0] for v in variants])
    res = {}
    with Pool(procs) as pool:
        for tag, lim, vel in variants:
            res[tag] = dict(hip_roll_nm=lim, hip_roll_vel=vel if vel is not None else "deployed(vel_lim)", conds={})
            for c in conds:
                files = sorted((out / "traj").glob(f"{tag}_{c}_perturb_*.npz")) + ([out / "traj" / f"{tag}_nominal_ref.npz"] if c == "nominal" else [])
                rows = pool.map(_analyse, [(f, lim) for f in files])
                agg = lambda k: float(np.mean([r[k] for r in rows])); mx = lambda k: float(np.max([r[k] for r in rows]))
                e = dict(n=len(rows), survival=float(np.mean([r["survived"] for r in rows])), jmae_rad=agg("jmae"), tilt_p95=agg("tilt_p95"), hr_peak_nm=mx("hr_peak_nm"), hr_p99_nm=agg("hr_p99_nm"), hr_sat=agg("hr_sat"),
                         hr_speed_max=mx("hr_speed_max"), hr_speed_p99=agg("hr_speed_p99"), guard_rewrite=agg("guard_rewrite"), pen_max_mm=mx("pen_max_mm"))
                if c == "nominal": e["anchored_mpjpe_mm"] = float(np.mean(pool.map(_mpjpe, [(str(run.dance.padded_npz), f) for f in files])))
                res[tag]["conds"][c] = e
    doc = dict(generated=now(), iteration=it, onnx=art["onnx"], onnx_sha12=__import__("hashlib").sha256(Path(art["onnx"]).read_bytes()).hexdigest()[:12], base_params=str(base_params), n_perturbed=n, conds=conds, variants=res)
    write_json(out / "limits.json", doc); _report(run, doc, out); return doc


def _report(run: Run, doc: dict, out: Path):
    V = doc["variants"]; tags = list(V); it = doc["iteration"]
    L = [f"# it{it} 冻结模型：髋 Roll 力矩/速度上限对闭环的影响（V1.2，自动生成 {doc['generated']}）", "",
         f"> 同一个 ONNX（sha12 `{doc['onnx_sha12']}`），不重训、不改部署资产；每个变体只替换 `policy_params.txt` 副本里左右髋 Roll 的 `effort_lim`（及可选的 `vel_lim`）。MuJoCo 闭环（部署控制链）：1 个参考起点 + {doc['n_perturbed']} 个扰动起点 × {len(doc['conds'])} 工况；guard 用同一上限钳制指令。", "",
         "## 总览（所有工况汇总）", "", "| 变体 | 髋Roll N·m | 髋Roll rad/s | 存活 | 关节MAE rad | 力矩饱和 | 峰值力矩 N·m | 峰值速度 rad/s | guard 改写 | nominal 锚定MPJPE mm |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in tags:
        cs = V[t]["conds"]; a = lambda k: np.mean([c[k] for c in cs.values()])
        L.append(f"| {t} | {V[t]['hip_roll_nm']:.0f} | {V[t]['hip_roll_vel']} | {np.mean([c['survival'] for c in cs.values()]):.3f} | {a('jmae_rad'):.4f} | {a('hr_sat'):.4f} | {max(c['hr_peak_nm'] for c in cs.values()):.1f} | {max(c['hr_speed_max'] for c in cs.values()):.1f} | {a('guard_rewrite'):.3f} | {cs['nominal']['anchored_mpjpe_mm']:.1f} |")
    L += ["", "## 分工况存活率 / 髋Roll 饱和占比 / 关节MAE", "", "| 工况 | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    for c in doc["conds"]:
        L.append(f"| {c} | " + " | ".join(f"{V[t]['conds'][c]['survival']:.2f} / {V[t]['conds'][c]['hr_sat']:.4f} / {V[t]['conds'][c]['jmae_rad']:.4f}" for t in tags) + " |")
    (out / "LIMITS_REPORT.md").write_text("\n".join(L) + "\n")
