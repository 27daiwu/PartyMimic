#!/usr/bin/env python3
"""Closed-loop MuJoCo rollout with FULL physical logging (runs in the sim2real py3.10 venv; no robot connection).

Same deployment control chain as the Sim2Sim bench (results/p4/scripts: ONNX policy stack, authority/ramp, C++ guard semantics, 200 Hz PD,
effort clip, optional delay) - reused unchanged - but initialised ON the reference (world frames coincide at t=0, zero velocity) and recording
everything the quality analysis needs, at the 50 Hz control rate:
  root pose / velocity, joint positions + velocities, CoM, per-foot ground-contact normal force + centre of pressure (from MuJoCo contact
  forces = reliable contact), contact point cloud (support polygon), worst ground penetration, PD torque ratio (pre-clip) + saturation fraction
  per joint, guard activity, tilt, fall time.
usage: rollout.py --onnx X.onnx --params policy_params.txt --padded-npz P.npz --out out.npz [--cond nominal] [--init ref|perturb] [--seed 0] [--n 1]
       (--n > 1 writes out_000.npz ... ; the perturbed start follows the evaluator's init_perturb: xy +-2 cm, yaw +-0.05 rad, joints +-0.03 rad)
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "results/p4/scripts"))


def _pop(flag):
    if flag in sys.argv:
        i = sys.argv.index(flag); v = sys.argv[i + 1]; del sys.argv[i:i + 2]; return v
    return None


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True); ap.add_argument("--params", required=True); ap.add_argument("--padded-npz", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--cond", default="nominal"); ap.add_argument("--init", default="ref", choices=["ref", "perturb"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n", type=int, default=1); ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--guard", default="new"); ap.add_argument("--auth", type=float, default=1.0); ap.add_argument("--ramp", type=float, default=0.1)
    return ap


FOOT_L, FOOT_R, OTHER = 0, 1, 2


def run_rollout(onnx, params, padded_npz, cond="nominal", init="ref", seed=0, guard="new", auth=1.0, ramp=0.1):
    import mujoco
    import pm_stack
    pm_stack.load_params.__defaults__ = (params,)
    import sim2sim_deploy as S
    import eval_bench
    S.padded_motion.__defaults__ = (padded_npz,)
    m, qadr, vadr, names = S.get_model(); d = mujoco.MjData(m); P = pm_stack.load_params(params)
    M = S.padded_motion(padded_npz); T = M["T"]
    kw = dict(eval_bench.COND[cond])
    delay_substeps = kw.get("delay_substeps", 0); kp_scale = kw.get("kp_scale", 1.0); friction = kw.get("friction"); push = kw.get("push"); noise = kw.get("noise", 0.0)
    if friction is not None:
        for g in range(m.ngeom): m.geom_friction[g, 0] = friction
    rng = np.random.default_rng(seed)
    q0 = M["jp"][0].astype(np.float64).copy(); xy0 = M["rp"][0][:2].astype(np.float64).copy(); quat0 = M["rq"][0].astype(np.float64).copy()
    if init == "perturb":
        xy0 = xy0 + rng.uniform(-1, 1, 2) * 0.02
        dyaw = rng.uniform(-1, 1) * 0.05
        qy = np.array([np.cos(dyaw / 2), 0, 0, np.sin(dyaw / 2)]); quat0 = pm_stack.qmul(qy, quat0)
        q0 = np.clip(q0 + rng.uniform(-1, 1, 29) * 0.03, P["pos_lo"], P["pos_hi"])
    floor = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    assert floor, "no floor plane in the sim model"
    floor_id = floor[0]
    S.set_state(m, d, qadr, vadr, q0, np.zeros(29), quat0, np.zeros(3), z=1.0)
    d.qpos[0:2] = xy0
    # settle the root height on the FLOOR only (S.settle_height counts self-collisions as "touching": 4 of 8 perturbed starts were left 377 mm in the air)
    lo, hi = 0.2, 1.2
    for _ in range(40):
        z = 0.5 * (lo + hi); d.qpos[2] = z; mujoco.mj_forward(m, d)
        touching = any(floor_id in (d.contact[i].geom1, d.contact[i].geom2) and d.contact[i].dist < 0 for i in range(d.ncon))
        lo, hi = (z, hi) if touching else (lo, z)
    d.qpos[2] = hi - 0.0005; mujoco.mj_forward(m, d)
    body_name = lambda g: m.body(m.geom_bodyid[g]).name
    foot_of = {}
    for g in range(m.ngeom):
        n = body_name(g)
        foot_of[g] = FOOT_L if n == "left_ankle_roll_link" else FOOT_R if n == "right_ankle_roll_link" else OTHER
    lim = P["effort_lim"]; kp, kd = P["kp"] * kp_scale, P["kd"]
    def read():
        q = np.array([d.qpos[a] for a in qadr]); dq = np.array([d.qvel[a] for a in vadr])
        quat = d.qpos[3:7].copy(); gyro = d.qvel[3:6].copy()
        if noise > 0:
            dq = dq + rng.normal(0, noise, 29); gyro = gyro + rng.normal(0, noise * 0.2, 3)
        return q, dq, gyro, quat
    qa, dqa, ga, quata = read(); q_hold = qa.copy()
    st = S.Stack(onnx, P, M); st.reset(qa, dqa, ga, quata, False)
    G = S.Guard(P, guard)
    steps = T - 1
    NP = 48  # contact point cloud rows kept per control tick (last substep)
    R = dict(t=np.zeros(steps), root_p=np.zeros((steps, 3)), root_q=np.zeros((steps, 4)), root_v=np.zeros((steps, 3)), q=np.zeros((steps, 29)), dq=np.zeros((steps, 29)),
             com=np.zeros((steps, 3)), com_v=np.zeros((steps, 3)), tilt=np.zeros(steps), tau_ratio=np.zeros((steps, 29)), sat_frac=np.zeros((steps, 29)), act=np.zeros((steps, 29)),
             foot_fz=np.zeros((steps, 2)), foot_cop=np.full((steps, 2, 2), np.nan), other_fz=np.zeros(steps), cop=np.full((steps, 2), np.nan), fz_total=np.zeros(steps),
             pen=np.zeros(steps), pen_foot=np.zeros(steps), ncon_foot=np.zeros(steps), pts=np.full((steps, NP, 5), np.nan), guard_clip=np.zeros(steps),
             qcmd=np.zeros((steps, 29)), qpre=np.zeros((steps, 29)), guard_pos=np.zeros(steps), guard_slew=np.zeros(steps), guard_tq=np.zeros(steps))
    pipe = [q0.copy()] * delay_substeps; fell = None; f6 = np.zeros(6); prev_guard = 0
    for k in range(steps):
        t = k * 0.02
        q, dq, gyro, quat = read()
        a, qt = st.tick(q, dq, gyro, quat)
        au = auth * S.smoothstep(t / ramp) if ramp > 0 else auth
        qc = q_hold + au * (qt - q_hold)
        gs0 = dict(G.stats); qf = G.filt(qc, q, dq, 0.02)
        R["guard_pos"][k] = G.stats["pos"] - gs0["pos"]; R["guard_slew"][k] = G.stats["slew"] - gs0["slew"]; R["guard_tq"][k] = G.stats["tq"] - gs0["tq"]; R["qpre"][k] = qc
        R["t"][k] = t; R["root_p"][k] = d.qpos[0:3]; R["root_q"][k] = d.qpos[3:7]; R["root_v"][k] = d.qvel[0:3]; R["q"][k] = q; R["dq"][k] = dq; R["act"][k] = a; R["qcmd"][k] = qf
        R["com"][k] = d.subtree_com[1]; R["tilt"][k] = S.tilt_of(d.qpos[3:7])
        fz_f = np.zeros(2); mom = np.zeros((2, 2)); oth = 0.0; sat = np.zeros(29); trmax = np.zeros(29); pen_all = 0.0; pen_foot = 0.0; cnt = 0
        for sub in range(4):
            pipe.append(qf); qcur = pipe.pop(0)
            if push and abs(t + sub * 0.005 - push["t"]) < 0.0025:
                d.qvel[0:3] += np.array(push["v"])
            qq = np.array([d.qpos[x] for x in qadr]); dd = np.array([d.qvel[x] for x in vadr])
            tau_pre = kp * (qcur - qq) - kd * dd
            tau = np.clip(tau_pre, -lim, lim)
            sat += (np.abs(tau_pre) >= 0.98 * lim).astype(float) / 4; trmax = np.maximum(trmax, np.abs(tau_pre) / lim)
            d.ctrl[:29] = tau; mujoco.mj_step(m, d)
            for i in range(d.ncon):
                c = d.contact[i]
                if floor_id not in (c.geom1, c.geom2): continue
                other = c.geom2 if c.geom1 == floor_id else c.geom1
                mujoco.mj_contactForce(m, d, i, f6)
                fw = c.frame.reshape(3, 3).T @ f6[:3]; sgn = 1.0 if c.geom1 == floor_id else -1.0
                fz = max(sgn * fw[2], 0.0); ft = foot_of.get(other, OTHER)
                pen_all = max(pen_all, -c.dist)
                if ft == OTHER: oth += fz / 4
                else:
                    fz_f[ft] += fz / 4; mom[ft] += fz * c.pos[:2] / 4; pen_foot = max(pen_foot, -c.dist)
        R["tau_ratio"][k] = trmax; R["sat_frac"][k] = sat; R["pen"][k] = pen_all; R["pen_foot"][k] = pen_foot
        R["foot_fz"][k] = fz_f; R["other_fz"][k] = oth; R["fz_total"][k] = fz_f.sum() + oth
        for ft in (0, 1):
            if fz_f[ft] > 1e-6: R["foot_cop"][k, ft] = mom[ft] / fz_f[ft]
        if fz_f.sum() > 1e-6: R["cop"][k] = mom.sum(0) / fz_f.sum()
        j = 0
        for i in range(d.ncon):
            c = d.contact[i]
            if floor_id in (c.geom1, c.geom2) and j < NP:
                other = c.geom2 if c.geom1 == floor_id else c.geom1
                mujoco.mj_contactForce(m, d, i, f6); fw = c.frame.reshape(3, 3).T @ f6[:3]; sgn = 1.0 if c.geom1 == floor_id else -1.0
                R["pts"][k, j] = (c.pos[0], c.pos[1], c.pos[2], max(sgn * fw[2], 0.0), foot_of.get(other, OTHER)); j += 1
        R["ncon_foot"][k] = j
        tot = G.stats["pos"] + G.stats["slew"] + G.stats["tq"]; R["guard_clip"][k] = tot - prev_guard; prev_guard = tot
        if fell is None and (R["tilt"][k] > 1.0 or d.qpos[2] < 0.45): fell = t + 0.02
    mass = float(m.body_subtreemass[1])
    R["com_v"][:] = np.gradient(R["com"], 0.02, axis=0)
    meta = dict(cond=cond, init=init, seed=seed, fell_t=fell, mass_kg=mass, joint_names=list(names), effort_lim=lim.tolist(), onnx=str(onnx), padded=str(padded_npz), guard=guard, auth=auth, ramp=ramp, fps=50)
    return R, meta


def _job(a):
    kw, out = a
    R, meta = run_rollout(**kw)
    np.savez_compressed(out, meta=json.dumps(meta), **R)
    return out, meta["fell_t"]


def main():
    a = build_argparser().parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i in range(a.n):
        o = out if a.n == 1 else out.with_name(f"{out.stem}_{i:03d}.npz")
        jobs.append((dict(onnx=a.onnx, params=a.params, padded_npz=a.padded_npz, cond=a.cond, init=a.init, seed=a.seed + i, guard=a.guard, auth=a.auth, ramp=a.ramp), str(o)))
    if a.n == 1:
        res = [_job(jobs[0])]
    else:
        from multiprocessing import Pool
        with Pool(min(a.procs, a.n)) as p: res = p.map(_job, jobs, chunksize=1)
    for o, f in res: print(o, "fell_t", f)


if __name__ == "__main__":
    main()
