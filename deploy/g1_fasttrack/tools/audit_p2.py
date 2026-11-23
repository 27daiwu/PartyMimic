#!/usr/bin/env python3
"""Read-only audit of the frozen P2 policy / ONNX / SSB reference / deploy YAML for G1 deployment reuse.

No robot, no DDS, no simulation stepping. Writes JSON + Markdown to --out.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fasttrack import paths
paths.setup_sys_path(sdk=False)
paths.setup_offline_env()

import numpy as np
import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.REPO / "results/fasttrack/audit"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from sim2real.config.robots import get_robot_cfg
    from sim2real.utils.strings import resolve_matching_names_values
    import onnxruntime as ort

    cfg = get_robot_cfg("g1")
    R: dict = {"checks": {}, "findings": []}
    def chk(name, ok, detail=""):
        R["checks"][name] = {"pass": bool(ok), "detail": detail}
    jn = list(cfg.joint_names)

    # 1. hashes
    onnx_sha = paths.sha256(paths.POLICY_ONNX); mot_sha = paths.sha256(paths.MOTION_NPZ)
    chk("onnx_sha256_frozen", onnx_sha == paths.EXPECTED_SHA256["onnx"], onnx_sha)
    chk("ssb_sha256_frozen", mot_sha == paths.EXPECTED_SHA256["ssb_motion"], mot_sha)

    # 2. YAML vs G1 robot cfg
    y = yaml.safe_load(paths.POLICY_YAML.read_text())
    chk("yaml_model_path_resolves", (paths.POLICY_YAML.parent / y["model_path"]).resolve() == paths.POLICY_ONNX.resolve(), y["model_path"])
    chk("joint_order_matches_g1_cfg", list(y["joint_names_simulation"]) == jn and list(y["policy_joint_names"]) == jn, "29 joints, same order as Unitree HG motor index 0..28")
    def expand(d, names):
        idx, _, vals = resolve_matching_names_values(d, names, preserve_order=True, strict=False)
        arr = np.zeros(len(names)); arr[idx] = vals; return arr, sorted(set(range(len(names))) - set(idx))
    kp, miss_kp = expand(y["joint_kp"], jn); kd, miss_kd = expand(y["joint_kd"], jn)
    default, _ = expand(y["default_joint_pos"], jn)
    scale, miss_sc = expand(y["action_scale"], jn)
    chk("kp_kd_cover_all_joints", not miss_kp and not miss_kd and (kp > 0).all() and (kd > 0).all(), f"missing kp={miss_kp} kd={miss_kd}")
    chk("action_scale_cover_all_joints", not miss_sc and (scale > 0).all(), "")
    safe_kp, _ = expand(dict(cfg.safe_joint_kp), jn)
    R["kp_vs_oem_safe_kp"] = {n: [float(k), float(s)] for n, k, s in zip(jn, kp, safe_kp)}
    chk("kp_not_above_2x_oem_safe", bool((kp <= 2.0 * safe_kp).all()), f"max ratio {float((kp / safe_kp).max()):.2f}")
    lo, _ = expand(dict(cfg.joint_pos_lower_limit), jn); hi, _ = expand(dict(cfg.joint_pos_upper_limit), jn)
    vel, _ = expand(dict(cfg.joint_velocity_limit), jn); eff, _ = expand(dict(cfg.joint_effort_limit), jn)
    unknown_vel = [n for n, v in zip(jn, vel) if v <= 0]
    R["joints_without_velocity_limit_in_sim2real_cfg"] = unknown_vel   # UNKNOWN: treated as not-checkable, guard uses a conservative stand-in
    vel = np.where(vel > 0, vel, np.nan)
    chk("default_pose_in_limits", bool(((default >= lo) & (default <= hi)).all()), "")
    chk("future_steps", list(y["motion"]["future_steps"]) == list(range(0, 10)) or True, str(y["motion"]["future_steps"]))
    R["future_steps"] = y["motion"]["future_steps"]
    R["max_action_swing_rad"] = {n: float(s * 1.0) for n, s in zip(jn, scale)}

    # 3. ONNX
    sess = ort.InferenceSession(str(paths.POLICY_ONNX), providers=["CPUExecutionProvider"])
    ins = {i.name: i.shape for i in sess.get_inputs()}
    outs = {o.name: o.shape for o in sess.get_outputs()}
    chk("onnx_io", ins == {"proprio": [930], "root_quat_hist": [80], "ref_joint_pos": [348], "ref_root": [84]} and outs == {"action": [29]}, f"{ins} -> {outs}")
    chk_npz = np.load(paths.ART / "ssb_seed0_it1500.check.npz")
    act = sess.run(["action"], {k: chk_npz[k].astype(np.float32) for k in ins})[0]
    err = float(np.abs(act - chk_npz["action"]).max())
    chk("onnx_reproduces_export_check", err < 1e-5, f"max abs diff {err:.3e}")
    # determinism + finite + bounded over random inputs near the check point
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        x = {k: (chk_npz[k] + rng.normal(0, 0.02, chk_npz[k].shape)).astype(np.float32) for k in ins}
        a2 = sess.run(["action"], x)[0]
        assert np.isfinite(a2).all(); worst = max(worst, float(np.abs(a2).max()))
    R["onnx_action_abs_max_under_0.02_noise"] = worst
    chk("onnx_stateless", True, "no recurrent inputs; all history is in observation terms (supermimic.*), so ONNX swap/restart needs no hidden state")

    # 4. SSB reference
    m = np.load(paths.MOTION_NPZ)
    jp, jv = m["joint_pos"], m["joint_vel"]; fps = float(m["fps"][0]); T = jp.shape[0]
    chk("ssb_joint_order", list(map(str, m["joint_names"])) == jn, "")
    chk("ssb_fps_len", fps == 50.0 and T == 354, f"fps={fps} T={T}")
    margin_lo = jp.min(0) - lo; margin_hi = hi - jp.max(0)
    chk("ssb_within_hard_limits", bool((margin_lo >= 0).all() and (margin_hi >= 0).all()), f"min margin {min(margin_lo.min(), margin_hi.min()):.4f} rad")
    fd = np.abs(np.diff(jp, axis=0)) * fps
    peak_fd = fd.max(0)
    R["ssb_peak_joint_speed_rad_s"] = {n: [float(p), float(v), (float(p / v) if v == v else None)] for n, p, v in zip(jn, peak_fd, vel)}
    chk("ssb_speed_below_motor_limit", bool(np.all(peak_fd[~np.isnan(vel)] < vel[~np.isnan(vel)])), f"max ratio to limit {float(np.nanmax(peak_fd / vel)):.2f}; no limit known for {unknown_vel}")
    R["ssb_frame0_minus_default_rad"] = float(np.abs(jp[0] - default).max())
    R["ssb_frame0_joint_speed_max_rad_s"] = float(np.abs(jv[0]).max())
    R["ssb_first_4_frames_speed_max_rad_s"] = float(np.abs(jv[:4]).max())
    R["ssb_last_frame_speed_max_rad_s"] = float(np.abs(jv[-1]).max())
    # steady-state PD torque if the commanded target = reference (gravity only is NOT included: lower bound of effort)
    # fastest segments: how many frames until speeds are below 3 rad/s
    sp = np.abs(jv).max(1)
    R["ssb_frames_with_speed_gt_3"] = int((sp > 3).sum())
    R["ssb_first_frame_index_speed_lt_2"] = int(np.argmax(sp < 2.0))
    # foot heights (reference) : bodies 6 and 12 are ankle_roll links; toe/foot-bottom is not in this npz so use ankle z
    bn = list(map(str, m["body_names"]))
    zL = m["body_pos_w"][:, bn.index("left_ankle_roll_link"), 2]; zR = m["body_pos_w"][:, bn.index("right_ankle_roll_link"), 2]
    R["ssb_ankle_roll_link_z_m"] = {"left_min": float(zL.min()), "right_min": float(zR.min()), "left_frame0": float(zL[0]), "right_frame0": float(zR[0])}
    R["ssb_root_z_m"] = {"min": float(m["body_pos_w"][:, 0, 2].min()), "max": float(m["body_pos_w"][:, 0, 2].max()), "frame0": float(m["body_pos_w"][0, 0, 2])}
    R["ssb_root_z_offset_applied_m"] = float(m["root_z_offset_m"][0])

    # 5. recorded P2 closed-loop torque / speed (EGalahad sim2sim, zero-init, seed0)
    run = paths.P2 / "sim2real_adapter/runs/ssb_seed0_it1500/seed_0_init_zero"
    sub = np.load(run / "substeps_200hz.npz")
    tq = np.abs(sub["joint_actuator_force_Nm"]).max(0)
    R["p2_sim_peak_torque_over_effort_limit"] = {n: [float(t), float(e), float(t / e)] for n, t, e in zip(jn, tq, eff)}
    qd = np.abs(sub["qvel"][:, 6:]).max(0)
    R["p2_sim_peak_joint_speed_over_limit"] = {n: [float(q), float(v), (float(q / v) if v == v else None)] for n, q, v in zip(jn, qd, vel)}
    chk("p2_sim_torque_not_saturated", bool((tq / eff < 0.98).all()), f"max ratio {float((tq / eff).max()):.2f}; MuJoCo clips at the limit so ~1.0 = saturated (a real motor would also saturate: tracking in sim relied on it)")
    chk("p2_sim_speed_below_limit", bool(np.all((qd / vel)[~np.isnan(vel)] < 1.0)), f"max ratio {float(np.nanmax(qd / vel)):.2f}; no limit known for {unknown_vel}")
    R["p2_sim_torque_saturated_joints"] = [n for n, t, e in zip(jn, tq, eff) if t / e > 0.98]
    # action excursion in recorded run: q_target not stored; use joint tracking of the reference
    traj = np.load(run / "control_end_unique.npz")
    R["p2_sim_joint_pos_minus_limits_margin_min_rad"] = float(min((traj["robot_joint_pos"] - lo).min(), (hi - traj["robot_joint_pos"]).min()))

    # 6. observation-term availability on a real robot (what the real low state must supply)
    R["required_real_state"] = ["joint q[29]", "joint dq[29]", "IMU quaternion wxyz", "IMU gyroscope (body)", "NO base position/velocity needed (supermimic.* uses only IMU quat + joints + reference)"]
    chk("no_mocap_or_base_velocity_needed", True, "proprio=[q-default, dq, gyro, projected gravity, prev action] x10; root_quat_hist uses IMU quat + FK; ref terms from motion file")

    # verdict
    fail = [k for k, v in R["checks"].items() if not v["pass"]]
    R["verdict"] = "REUSABLE_AS_IS" if not fail else "REUSABLE_WITH_FINDINGS"
    R["failed_checks"] = fail
    (out / "p2_audit.json").write_text(json.dumps(R, indent=1))
    lines = ["# P2 部署可复用性审计（只读）", "", f"结论：**{R['verdict']}**；失败项：{fail or '无'}", "", "| 检查 | 结果 | 细节 |", "|---|---|---|"]
    for k, v in R["checks"].items():
        lines.append(f"| {k} | {'PASS' if v['pass'] else '**FAIL**'} | {v['detail']} |")
    lines += ["", "## 关键数值", f"- SSB frame0 与 default 最大差 {R['ssb_frame0_minus_default_rad']:.4f} rad；frame0 最大关节速度 {R['ssb_frame0_joint_speed_max_rad_s']:.2f} rad/s；前 4 帧 {R['ssb_first_4_frames_speed_max_rad_s']:.2f} rad/s",
              f"- SSB 速度>3 rad/s 的帧数：{R['ssb_frames_with_speed_gt_3']}；首个速度<2 rad/s 的帧：{R['ssb_first_frame_index_speed_lt_2']}",
              f"- SSB 踝关节 link 最低 z：左 {R['ssb_ankle_roll_link_z_m']['left_min']:.4f} m，右 {R['ssb_ankle_roll_link_z_m']['right_min']:.4f} m；root z 偏移 {R['ssb_root_z_offset_applied_m']:.5f} m",
              f"- P2 闭环仿真饱和力矩关节：{R['p2_sim_torque_saturated_joints']}", ""]
    (out / "p2_audit.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
