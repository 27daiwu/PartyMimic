#!/usr/bin/env python3
"""Metrics of one sim2sim / real run directory (runner_trace.npz, run_report.json, optional host/host_trace.npz).

Root-relative 30-body position error: FK from (joint q, IMU quaternion) only, pelvis at origin, yaw-aligned to the reference.
This is NOT the P2 'XY/yaw anchored 14-link MPJPE incl. root height' (needs root position); it is computed identically for the
P2 in-process EGalahad run (control_end_unique.npz) so the two numbers are comparable.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fasttrack import paths
paths.setup_sys_path(sdk=False); paths.setup_offline_env()
import numpy as np, mujoco


def yaw_of(q):
    w, x, y, z = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def fk_rel(model, root_adr, joint_adr, body_ids, quat, q):
    d = mujoco.MjData(model); out = []
    for qu, qq in zip(quat, q):
        d.qpos[:] = 0; d.qpos[root_adr + 3:root_adr + 7] = qu; d.qpos[joint_adr] = qq
        mujoco.mj_kinematics(model, d); p = d.xpos[body_ids].copy(); out.append(p - p[0])
    return np.asarray(out)


def rel_mpjpe(model, adrs, ref, quat, q, frames):
    root_adr, joint_adr, body_ids = adrs
    rb = ref["body_pos_w"][frames]; rq = ref["body_quat_w"][frames, 0]
    robot = fk_rel(model, root_adr, joint_adr, body_ids, quat, q)
    refrel = rb - rb[:, :1]
    dyaw = yaw_of(quat) - yaw_of(rq)                       # rotate reference by the yaw difference
    c, s = np.cos(dyaw), np.sin(dyaw)
    R = np.zeros((len(dyaw), 3, 3)); R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s; R[:, 1, 1] = c; R[:, 2, 2] = 1
    # reference root orientation is already in world; compare in the robot's yaw frame: (robot_rel) vs R*(ref_rel)
    # robot FK used the IMU quaternion as world orientation; ref_rel is world-frame too => only yaw alignment needed
    ref_al = np.einsum("tij,tnj->tni", R, refrel)
    return np.linalg.norm(robot - ref_al, axis=-1)          # [T, 30]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--json", default=None); a = ap.parse_args()
    case = Path(a.run); run = case / "runner" if (case / "runner").is_dir() else case
    hostdir = case / "host" if (case / "host").is_dir() else run / "host"
    from sim2real.config.robots import get_robot_cfg
    cfg = get_robot_cfg("g1")
    model = mujoco.MjModel.from_xml_path(str(cfg.resolve_mjcf_path()))
    ref = dict(np.load(paths.MOTION_NPZ)); names = list(map(str, ref["joint_names"])); bnames = list(map(str, ref["body_names"]))
    free = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE][0]
    adrs = (int(model.jnt_qposadr[free]), np.array([model.joint(n).qposadr[0] for n in names]), np.array([model.body(n).id for n in bnames]))
    rep = json.loads((run / "run_report.json").read_text())
    out = {"result": rep.get("result"), "fault": rep.get("fault"), "exit_confirmed": rep.get("exit", {}).get("confirmed"),
           "guard_stats": rep.get("guard_stats"), "guard_max_ratio": rep.get("guard_max_ratio"), "frames_sent": rep.get("frames_sent")}
    tpath = run / "runner_trace.npz"
    if tpath.exists():
        t = np.load(tpath); ph = t["phase"]; out["ticks"] = int(len(ph))
        out["phase_ticks"] = {p: int((ph == p).sum()) for p in dict.fromkeys(ph.tolist())}
        out["step_ms"] = {"p50": float(np.percentile(t["step_ms"], 50)), "p99": float(np.percentile(t["step_ms"], 99)), "max": float(t["step_ms"].max())}
        out["state_age_ms"] = {"p50": float(np.percentile(t["state_age"], 50) * 1e3), "p99": float(np.percentile(t["state_age"], 99) * 1e3), "max": float(t["state_age"].max() * 1e3)}
        m = (ph == "DANCE") | (ph == "END_HOLD")
        if m.sum() > 5:
            mt = np.clip(t["motion_t"][m], 0, 353); q, quat = t["q"][m], t["quat"][m]
            out["dance_ticks"] = int(m.sum()); out["motion_t_range"] = [int(mt.min()), int(mt.max())]
            je = q - ref["joint_pos"][mt]; out["joint_mae_rad"] = float(np.abs(je).mean()); out["joint_max_err_rad"] = float(np.abs(je).max())
            e = rel_mpjpe(model, adrs, ref, quat, q, mt)
            out["root_rel_body_err_mm"] = {"mean": float(e.mean() * 1e3), "p95": float(np.percentile(e, 95) * 1e3), "max": float(e.max() * 1e3)}
            lo = np.array([cfg.joint_pos_lower_limit[n] for n in names]); hi = np.array([cfg.joint_pos_upper_limit[n] for n in names])
            out["max_q_outside_limit_rad"] = float(max((lo - q).max(), (q - hi).max(), 0.0))
            out["max_abs_dq_rad_s"] = float(np.abs(t["dq"][m]).max()); out["max_gyro_rad_s"] = float(np.linalg.norm(t["gyro"][m], axis=1).max())
            w, x, y, z = quat.T; out["max_tilt_rad"] = float(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)).max())
            full = t["authority"][m] >= 0.999
            if full.any():
                d = np.abs(t["q_cmd"][m][full] - t["q_policy"][m][full]); out["guard_cmd_change_vs_policy_rad"] = {"mean": float(np.nanmean(d)), "max": float(np.nanmax(d)), "frac_ticks_changed_gt_0.02": float((np.nanmax(d, 1) > 0.02).mean())}
            # P2 in-process EGalahad run on the same metric
            p2 = np.load(paths.P2 / "sim2real_adapter/runs/ssb_seed0_it1500/seed_0_init_zero/control_end_unique.npz")
            fr = np.asarray(p2["frame"], int); pq = p2["robot_joint_pos"]; pu = p2["robot_root_quat_w"]
            e2 = rel_mpjpe(model, adrs, ref, pu, pq, fr)
            out["p2_inprocess_root_rel_body_err_mm"] = {"mean": float(e2.mean() * 1e3), "p95": float(np.percentile(e2, 95) * 1e3), "max": float(e2.max() * 1e3)}
            out["p2_inprocess_joint_mae_rad"] = float(np.abs(pq - ref["joint_pos"][fr]).mean())
    hp = hostdir / "host_trace.npz"
    if hp.exists():
        h = np.load(hp, allow_pickle=True); eff = np.array([cfg.joint_effort_limit[n] for n in names]); tq = np.abs(h["torque"])
        out["host_peak_torque_over_effort"] = float((tq / eff).max()); out["host_saturated_joints"] = [n for n, r in zip(names, (tq / eff).max(0)) if r > 0.98]
        out["host_user_frames_200hz"] = int(h["user"].sum())
        out["host_pelvis_z_min_max"] = [float(h["qpos"][:, 2].min()), float(h["qpos"][:, 2].max())]
    s = json.dumps(out, indent=1); print(s)
    if a.json: Path(a.json).write_text(s)


if __name__ == "__main__":
    main()
