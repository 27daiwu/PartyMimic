#!/usr/bin/env python3
"""PartyMimic P5.1 offline analysis of ONE run log directory (dev PC). Uses only the log files (+ the ONNX in the package for `replay`).

    python3 pm_analyze.py <run_dir> all   [--onnx PATH] [--out DIR]
    python3 pm_analyze.py <run_dir> replay | init | tracking | pd | latency | safety | exit

  replay    per-tick policy replay: obs.bin -> ONNX -> action, compared with the logged action (bit-exact check) + reconstruction of obs/action history consistency
  init      Sim2Sim initial-state export (initial_state.json / .npz): q, dq, quat, gyro, yaw offset, q_hold, SSB frame, gains, plus the pre-take-over state window
  tracking  motion-tracking error: q vs SSB reference, q vs commanded target, root orientation vs reference root (yaw aligned), per joint rms / max over time
  pd        PD analysis: logged kp/kd, command vs measured, tau_est vs tau_pd (effective gain / lag), torque saturation, which joints carry the load
  latency   control period, inference, state age, send, DDS loopback, lowstate gaps / loss / transport jitter, command->response delay per joint
  safety    safety gates: clip episodes per joint / reason / duration, raw vs clipped, torque / speed / tilt margins, what the gates changed in the command
  exit      exit diagnostics: timeline detect -> RPC -> FSM -> stable, attitude at each step, hand-over transient (state before / after), WalkRun recovery
Every command prints a compact text report and writes <out>/<name>.json (+ csv where useful). `all` also writes analysis_report.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmlog import N, RunLog  # noqa: E402


def _stats(x, scale=1.0):
    x = np.asarray(x, dtype=np.float64) * scale
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def _qmul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def _qangle(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    d = np.clip(np.abs(np.sum(a * b, -1)), 0, 1)
    return 2 * np.arccos(d)


def _names(r):
    return r.joint_names or [f"j{i}" for i in range(N)]


# --------------------------------------------------------------------------------------------- replay
def cmd_replay(r: RunLog, a) -> dict:
    out = {"available": False}
    if r.obs is None or "action" not in r.trace:
        return out
    import onnxruntime as ort

    onnx = a.onnx or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "policy.onnx")
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    dims = [int(d["dim"]) for d in r.obs_meta["layout"]]
    names = [d["name"] for d in r.obs_meta["layout"]]
    act = r.trace["action"]
    n = min(len(r.obs), len(act))
    errs = np.zeros(n)
    for i in range(n):
        o = r.obs[i]["obs"]
        feeds, off = {}, 0
        for nm, d in zip(names, dims):
            feeds[nm] = o[off : off + d].astype(np.float32)
            off += d
        y = sess.run(["action"], feeds)[0].reshape(-1)
        errs[i] = np.max(np.abs(y.astype(np.float64) - act[i].astype(np.float32).astype(np.float64)))
    # history consistency: the newest proprio frame must contain the logged q/dq (relative to default) and the previous action
    d0 = r.vec("default_pos") if "default_pos" in r.manifest else None
    hist_err = None
    if d0 is not None and "action_prev" in r.trace:
        P = 2 * N + 6 + N
        hp = dims[0] // P
        newest = np.stack([r.obs[i]["obs"][(hp - 1) * P : hp * P] for i in range(n)])
        q_obs = newest[:, :N] + d0.astype(np.float32)
        dq_obs = newest[:, N : 2 * N]
        pa = newest[:, -N:]
        hist_err = {"q_vs_trace": float(np.max(np.abs(q_obs - r.trace["q"][:n]))), "dq_vs_trace": float(np.max(np.abs(dq_obs - r.trace["dq"][:n]))), "prev_action_vs_trace": float(np.max(np.abs(pa - r.trace["action_prev"][:n])))}
    chain = float(np.max(np.abs(r.trace["action"][: n - 1] - r.trace["action_prev"][1:n]))) if n > 1 else 0.0  # action(t) is the previous-action input of tick t+1
    out = {"available": True, "onnx": os.path.abspath(onnx), "ticks": int(n), "max_abs_action_diff": float(errs.max()) if n else None, "ticks_above_1e-6": int((errs > 1e-6).sum()),
           "obs_vs_trace": hist_err, "action_history_chain_max_diff": chain, "pass": bool(n and errs.max() <= 1e-5), "note": "bit-exact on the same CPU architecture; an aarch64 robot log replayed on x86 differs by float32 rounding (<= ~3e-6)"}
    return out


# --------------------------------------------------------------------------------------------- init (Sim2Sim)
def cmd_init(r: RunLog, a) -> dict:
    pr = r.first_ev("policy_reset")
    out = {"available": pr is not None}
    if pr is None:
        return out
    init = {k: pr[k] for k in ("q", "dq", "quat_wxyz", "gyro", "yaw_offset_wxyz", "q_hold", "motion_t", "paused")}
    init["take_state"] = pr.get("take_state")
    for k in ("default_pos", "action_scale", "kp", "kd", "pos_lo", "pos_hi", "vel_lim", "effort_lim", "joint_names"):
        if k in r.manifest:
            init[k] = r.manifest[k]
    init["authority_max"] = r.manifest.get("authority")
    init["ramp_s"] = r.manifest.get("ramp_s")
    init["ref_mode"] = r.manifest.get("ref_mode")
    init["control_dt_s"] = r.manifest.get("control_dt_s")
    init["policy_sha256"] = r.manifest.get("sha256", {}).get("policy_onnx")
    init["motion_sha256"] = r.manifest.get("sha256", {}).get("motion_bin")
    init["t_mono"] = pr["t_mono"]
    # the lowstate window before the take-over: full-rate q/dq/IMU, e.g. to settle a simulator into the same state history
    pre = {}
    if r.state is not None and len(r.state):
        t0 = pr["t_mono"]
        w = r.state[(r.state["t"] <= t0)]
        if len(w):
            pre = {"n": int(len(w)), "span_s": float(w["t"][-1] - w["t"][0]), "q_last": w["q"][-1].tolist(), "dq_last": w["dq"][-1].tolist(), "quat_last": w["quat"][-1].tolist(),
                   "gyro_rms": float(np.sqrt(np.mean(np.sum(w["gyro"].astype(float) ** 2, 1)))), "dq_abs_max": float(np.abs(w["dq"]).max())}
            if a.out:
                np.savez_compressed(os.path.join(a.out, "initial_state.npz"), t=w["t"], q=w["q"], dq=w["dq"], quat=w["quat"], gyro=w["gyro"], tau=w["tau"], q_hold=np.array(init["q_hold"]), yaw_offset_wxyz=np.array(init["yaw_offset_wxyz"]),
                                    default_pos=np.array(init.get("default_pos", [])), kp=np.array(init.get("kp", [])), kd=np.array(init.get("kd", [])))
    init["preroll_window"] = pre
    # trace-time reconstruction check: q at the first tick vs the q handed to the policy reset
    if "q" in r.trace and len(r.trace["q"]):
        init["first_tick_q_minus_reset_q_max"] = float(np.max(np.abs(r.trace["q"][0] - np.array(init["q"]))))
    return {"available": True, "initial_state": init}


# --------------------------------------------------------------------------------------------- tracking
def _up(q):
    """"up" vector (world +z) in the pelvis frame for quaternions (w,x,y,z): yaw-free, angle between two of them = gravity-direction error"""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)


def _segment_tracking(r: RunLog, e, ec, ok) -> dict:
    """tracking statistics per segment: lead-in (frozen first frame) / dance / end-hold / paused reference, with the take-over tick (tick 0) apart.
    Also: measured tilt vs REFERENCE tilt vs gravity-direction error (V1.6: a lean that follows the reference is not an anomaly)."""
    T = r.trace
    out = {"lead_in_frames": r.lead_in_frames, "motion_frames": r.motion_frames, "fps": r.fps, "dance_axis_derived_from_ssb_frame": getattr(r, "dance_axis_derived", False),
           "n_ticks_total": r.n_ticks, "n_ticks_dance": r.n_ticks_dance,
           "note": "trace column ssb_frame = index into the REPLAYED clip (lead-in + dance); dance_frame = ssb_frame - lead_in_frames; 'n_ticks' counts lead-in and end-hold too"}
    have_att = all(k in T for k in ("quat_w", "ref_rq_w"))
    if have_att:
        qm = np.stack([T[k] for k in ("quat_w", "quat_x", "quat_y", "quat_z")], 1)
        qr = np.stack([T[k] for k in ("ref_rq_w", "ref_rq_x", "ref_rq_y", "ref_rq_z")], 1)
        um, ur = _up(qm), _up(qr)
        tilt_m = np.arccos(np.clip(um[:, 2], -1, 1))
        tilt_r = np.arccos(np.clip(ur[:, 2], -1, 1))
        dev = np.arccos(np.clip(np.sum(um * ur, 1), -1, 1))
    for nm in ("lead_in", "dance", "end_hold", "paused_reference", "takeover_first_tick"):
        m = r.seg_mask(nm)
        if not m.any():
            continue
        d = {"ticks": int(m.sum()), "t_rel": [float(T["t"][m][0]), float(T["t"][m][-1])], "joint_rms_vs_ref_rad": float(np.sqrt(np.mean(e[m] ** 2))), "joint_max_vs_ref_rad": float(np.abs(e[m]).max())}
        if "dance_frame" in T:
            d["dance_frame"] = [float(T["dance_frame"][m][0]), float(T["dance_frame"][m][-1])]
        okm = m & ok
        if okm.any():
            d["actual_vs_command_rms_rad"] = float(np.sqrt(np.mean(ec[okm] ** 2)))
        if have_att:
            d.update({"tilt_max_rad": float(tilt_m[m].max()), "ref_tilt_max_rad": float(tilt_r[m].max()), "gravity_err_vs_ref_max_rad": float(dev[m].max()), "gravity_err_vs_ref_mean_rad": float(dev[m].mean()),
                      "tilt_minus_ref_tilt_max_rad": float((tilt_m[m] - tilt_r[m]).max())})
        out[nm] = d
    if have_att:
        out["overall_incl_lead_in_and_hold"] = {"joint_rms_vs_ref_rad": float(np.sqrt(np.mean(e ** 2))), "tilt_max_rad": float(tilt_m.max()), "ref_tilt_max_rad": float(tilt_r.max()), "gravity_err_vs_ref_max_rad": float(dev.max())}
        i = int(np.argmax(tilt_m))
        out["tilt_peak"] = {"t_rel": float(T["t"][i]), "tilt_rad": float(tilt_m[i]), "ref_tilt_rad": float(tilt_r[i]), "gravity_err_rad": float(dev[i]), "segment": r.SEG_NAMES.get(int(T["seg"][i]), "?") if "seg" in T else "?"}
    return out


def cmd_tracking(r: RunLog, a) -> dict:
    if "q" not in r.trace or "ref_jp" not in r.trace or not r.n_ticks:
        return {"available": False}
    T = r.trace
    q, ref, qc, qp = T["q"], T["ref_jp"], T["q_cmd"], T["q_policy"]
    names = _names(r)
    e = q - ref
    ec = q - qc
    ok = np.isfinite(qc).all(1)
    per = {}
    for j, nm in enumerate(names):
        per[nm] = {"ref_rms": float(np.sqrt(np.mean(e[:, j] ** 2))), "ref_max": float(np.abs(e[:, j]).max()), "cmd_rms": float(np.sqrt(np.mean(ec[ok, j] ** 2))) if ok.any() else None,
                   "cmd_max": float(np.abs(ec[ok, j]).max()) if ok.any() else None, "policy_target_vs_ref_rms": float(np.sqrt(np.mean((qp[:, j] - ref[:, j]) ** 2)))}
    # P5.3: the four quantities kept apart: raw policy target / clipped command / actual / reference
    okc = np.isfinite(qc).all(1)
    sep = {"policy_vs_reference_rms": float(np.sqrt(np.mean((qp - ref) ** 2))), "command_vs_reference_rms": float(np.sqrt(np.mean((qc[okc] - ref[okc]) ** 2))),
           "actual_vs_reference_rms": float(np.sqrt(np.mean(e ** 2))), "actual_vs_command_rms": float(np.sqrt(np.mean((q[okc] - qc[okc]) ** 2))), "policy_minus_command_rms": float(np.sqrt(np.mean((qp[okc] - qc[okc]) ** 2)))}
    if "pos_lo" in r.manifest:
        lo_, hi_ = r.vec("pos_lo"), r.vec("pos_hi")
        outm = (qp < lo_) | (qp > hi_)
        sep["raw_target_outside_hard_limits_tick_fraction"] = float(outm.any(1).mean())
        sep["raw_target_outside_fraction_per_joint"] = {names[j]: float(outm[:, j].mean()) for j in np.argsort(-outm.mean(0))[:6]}
        eff_, kp_, kd_ = r.vec("effort_lim"), T["kp"], T["kd"]
        tt = np.clip(kp_ * (qp - q) - kd_ * T["dq"], -eff_, eff_)
        td = kp_ * (qc - q) - kd_ * T["dq"] + (T["tau_ff"] if "tau_ff" in T else 0)
        sh = (np.sign(tt) * (tt - td))[okc]
        sep["torque_shortfall_vs_training_nm_mean_per_joint"] = {names[j]: float(sh[:, j].mean()) for j in np.argsort(-sh.mean(0))[:6]}
    segs = _segment_tracking(r, e, ec, ok)
    out = {"available": True, "separation": sep, "ticks": r.n_ticks, "segments": segs, "legacy_overall_note": "the *_over_joints / separation / error_vs_time numbers below average EVERY tick incl. the frozen lead-in, the end-hold and the take-over tick (V1.5 behaviour); see `segments` for lead-in / dance / end-hold apart", "joint_rms_over_joints_vs_ref_rad": float(np.sqrt(np.mean(e ** 2))), "worst_joints_vs_ref": sorted(per.items(), key=lambda kv: -kv[1]["ref_rms"])[:6], "per_joint": per}
    out["worst_joints_vs_ref"] = [{"joint": k, **v} for k, v in out["worst_joints_vs_ref"]]
    # time resolved: how fast does the tracking error grow
    t = r.trace["t"]
    dt_ = r.trace.get("dance_t", t)
    sg_ = r.trace.get("seg", np.ones(len(t)))
    out["error_vs_time"] = [{"t": float(t[i]), "dance_t": float(dt_[i]), "seg": r.SEG_NAMES.get(int(sg_[i]), "?"), "rms": float(np.sqrt(np.mean(e[i] ** 2))), "max": float(np.abs(e[i]).max())} for i in range(0, len(t), max(1, len(t) // 25))]
    # root orientation vs reference root, yaw aligned with the offset the policy used
    pr = r.first_ev("policy_reset")
    if pr and "quat_w" in r.trace and "ref_rq_w" in r.trace:
        off = np.array(pr["yaw_offset_wxyz"])
        qr = np.stack([r.trace[k] for k in ("quat_w", "quat_x", "quat_y", "quat_z")], 1)
        rr = np.stack([r.trace[k] for k in ("ref_rq_w", "ref_rq_x", "ref_rq_y", "ref_rq_z")], 1)
        rel = _qmul(np.broadcast_to(off, qr.shape), qr)
        ang = _qangle(rel, rr)
        out["root_orientation_error_rad"] = _stats(ang)
        out["tilt_rad"] = _stats(r.trace["tilt"])
        out["root_orientation_error_by_segment_rad"] = {nm: _stats(ang[r.seg_mask(nm)]) for nm in ("lead_in", "dance", "end_hold", "paused_reference")}
        out["yaw_alignment_note"] = "yaw offset = policy_reset yaw_offset (the SSB first frame). During the frozen lead-in the reference IS that first frame, so dance frame 0 is aligned with it: no bias between lead-in and dance (verified: ref_rq is constant over the lead-in)"
        if a.out:
            np.savetxt(os.path.join(a.out, "tracking_error.csv"), np.column_stack([t, dt_, sg_, np.sqrt(np.mean(e ** 2, 1)), np.abs(e).max(1), ang]), delimiter=",", header="t,dance_t,seg,joint_rms_vs_ref,joint_max_vs_ref,root_orient_err_rad", comments="")
    return out


# --------------------------------------------------------------------------------------------- pd
def cmd_pd(r: RunLog, a) -> dict:
    if "tau_pd" not in r.trace or not r.n_ticks:
        return {"available": False}
    names = _names(r)
    q, dq, tau, qc, kp, kd, tpd = (r.trace[k] for k in ("q", "dq", "tau_est", "q_cmd", "kp", "kd", "tau_pd"))
    ok = np.isfinite(qc).all(1) & np.isfinite(tpd).all(1)
    eff = r.vec("effort_lim") if "effort_lim" in r.manifest else np.full(N, np.nan)
    per = {}
    for j, nm in enumerate(names):
        o = ok
        if o.sum() < 5:
            continue
        e = qc[o, j] - q[o, j]
        # regress the measured torque on (position error, -velocity): effective stiffness / damping seen by the joint (tau_est includes friction, load, 1 control-period lag)
        A = np.column_stack([e, -dq[o, j]])
        coef, *_ = np.linalg.lstsq(A, tau[o, j], rcond=None)
        # lag between commanded PD torque and measured torque (ticks): argmax of the cross-correlation over 0..5 ticks
        best, lag = -2, 0
        for L in range(0, 6):
            x, y = tpd[o, j][: len(tpd[o, j]) - L if L else None], tau[o, j][L:]
            m = min(len(x), len(y))
            if m > 8 and np.std(x[:m]) > 1e-9 and np.std(y[:m]) > 1e-9:
                c = np.corrcoef(x[:m], y[:m])[0, 1]
                if c > best:
                    best, lag = c, L
        per[nm] = {"kp_cmd": float(kp[o, j].mean()), "kd_cmd": float(kd[o, j].mean()), "kp_effective": float(coef[0]), "kd_effective": float(coef[1]), "tau_pd_vs_est_corr": float(best), "best_lag_ticks": int(lag),
                   "tau_pd_abs_max": float(np.abs(tpd[o, j]).max()), "tau_est_abs_max": float(np.abs(tau[o, j]).max()), "effort_limit": float(eff[j]),
                   "tau_pd_over_limit_max": float(np.abs(tpd[o, j]).max() / eff[j]) if np.isfinite(eff[j]) else None,
                   "tau_est_over_limit_max": float(np.abs(tau[o, j]).max() / eff[j]) if np.isfinite(eff[j]) else None,
                   "frac_ticks_tau_est_over_90pct": float(np.mean(np.abs(tau[o, j]) > 0.9 * eff[j])) if np.isfinite(eff[j]) else None, "pos_err_rms": float(np.sqrt(np.mean(e ** 2)))}
    worst = sorted(per.items(), key=lambda kv: -(kv[1]["tau_est_over_limit_max"] or 0))[:6]
    return {"available": True, "ticks_with_command": int(ok.sum()), "per_joint": per, "most_loaded_joints": [{"joint": k, "tau_est_over_limit_max": v["tau_est_over_limit_max"], "tau_pd_over_limit_max": v["tau_pd_over_limit_max"]} for k, v in worst],
            "note": "kp_effective/kd_effective = least-squares fit of the measured tau_est on (q_cmd-q, -dq); a large gap to the commanded gains = friction / drive dynamics / tau_est scaling; best_lag_ticks = delay between PD torque and measured torque"}


# --------------------------------------------------------------------------------------------- latency
def cmd_latency(r: RunLog, a) -> dict:
    out = {"available": r.n_ticks > 0}
    if not r.n_ticks:
        return out
    T = r.trace
    out["control_period_ms"] = _stats(T["dt_ms"][1:])
    out["inference_ms"] = _stats(T["infer_ms"])
    out["tick_step_ms"] = _stats(T["step_ms"])
    out["state_age_at_tick_ms"] = _stats(T["state_age_ms"])
    out["state_age_at_send_ms"] = _stats(T["state_age_send_ms"])
    out["state_sync_wait_ms"] = _stats(T["sync_wait_ms"])
    out["send_call_us"] = _stats(T["send_us"])
    out["dds_command_loopback_ms"] = _stats(T["dds_loop_ms"])
    out["end_to_end_obs_to_send_ms"] = _stats(T["state_age_send_ms"])
    fl = T["flags"].astype(int)
    out["flags"] = {"late": int(((fl & 1) != 0).sum()), "repeat_state": int(((fl & 2) != 0).sum()), "sync_timeout": int(((fl & 4) != 0).sum()), "clip": int(((fl & 8) != 0).sum()),
                    "lowstate_gap": int(((fl & 16) != 0).sum()), "slow_infer": int(((fl & 32) != 0).sum())}
    out["lowstate_lost_est_final"] = int(T["ls_lost"][-1])
    if r.state is not None and len(r.state) > 10:
        s = r.state
        gaps = np.diff(s["t"]) * 1e3
        out["lowstate_gap_ms"] = _stats(gaps)
        dtick = np.diff(s["tick"].astype(np.int64))
        nom = int(np.median(dtick)) if len(dtick) else 1
        out["lowstate_nominal_tick_step"] = nom
        out["lowstate_lost_samples"] = int(np.sum(np.maximum(dtick // max(nom, 1) - 1, 0)))
        # transport jitter: arrival time minus robot tick (1 ms counter): drift-removed spread = DDS + scheduling jitter of the state path
        t = s["t"] - s["t"][0]
        off = t - (s["tick"].astype(np.float64) - s["tick"][0]) * 1e-3
        k = np.polyfit(t, off, 1) if len(t) > 3 else [0, 0]
        res = (off - np.polyval(k, t)) * 1e3
        out["lowstate_transport_jitter_ms"] = {"p50_abs": float(np.percentile(np.abs(res), 50)), "p99_abs": float(np.percentile(np.abs(res), 99)), "max_abs": float(np.abs(res).max()), "clock_drift_ppm": float(k[0] * 1e6)}
    # command -> response: per joint cross-correlation lag between the change of q_cmd and the change of q (ticks)
    if "q_cmd" in T:
        q, qc = T["q"], T["q_cmd"]
        ok = np.isfinite(qc).all(1)
        lags = []
        for j in range(N):
            dqc, dqm = np.diff(qc[ok, j]), np.diff(q[ok, j])
            best, bl = -2, 0
            for L in range(0, 6):
                x, y = dqc[: len(dqc) - L if L else None], dqm[L:]
                m = min(len(x), len(y))
                if m > 8 and np.std(x[:m]) > 1e-9 and np.std(y[:m]) > 1e-9:
                    c = np.corrcoef(x[:m], y[:m])[0, 1]
                    if c > best:
                        best, bl = c, L
            lags.append({"joint": _names(r)[j], "lag_ticks": bl, "lag_ms": bl * 20.0, "corr": float(best)})
        out["command_to_response_lag"] = lags
    # RPC latency from the events
    rpcs = r.ev("rpc")
    by = {}
    for e in rpcs:
        by.setdefault(e["api"] + ("(" + e["arg"] + ")" if e.get("arg") else ""), []).append(e["latency_ms"])
    out["rpc_latency_ms"] = {k: _stats(v) for k, v in by.items()}
    lg = r.first_ev("log_stats")
    if lg:
        out["logger"] = {k: lg.get(k) for k in ("writer_max_lag_ms", "dropped_ticks", "dropped_states", "dropped_frames", "dropped_clips", "dropped_events", "write_errors", "lowstate_max_gap_ms", "dds_loopback_max_ms")}
    return out


# --------------------------------------------------------------------------------------------- safety
def cmd_safety(r: RunLog, a) -> dict:
    out = {"available": True}
    names = _names(r)
    rows = r.clips
    h = r.clips_hdr
    ix = {k: i for i, k in enumerate(h)}
    ep = {}
    for row in rows:
        if int(row[ix["tick"]]) < 0:
            continue
        key = (row[ix["joint"]], row[ix["reason"]])
        e = ep.setdefault(key, {"ticks": [], "raw": [], "final": []})
        e["ticks"].append(int(row[ix["tick"]]))
        e["raw"].append(float(row[ix["raw"]]))
        e["final"].append(float(row[ix["final"]]))
    per = []
    for (j, why), e in ep.items():
        t = sorted(e["ticks"])
        runs, run = [], 1
        for a1, b1 in zip(t, t[1:]):
            if b1 == a1 + 1:
                run += 1
            else:
                runs.append(run)
                run = 1
        runs.append(run)
        d = np.abs(np.array(e["raw"]) - np.array(e["final"]))
        per.append({"joint": j, "reason": why, "ticks": len(t), "episodes": len(runs), "longest_episode_ticks": max(runs), "longest_episode_s": max(runs) * 0.02, "max_clip_rad": float(d.max()), "mean_clip_rad": float(d.mean())})
    per.sort(key=lambda x: -x["max_clip_rad"])
    out["per_joint_reason"] = per
    out["exit_frame_clips"] = int(sum(1 for row in rows if int(row[ix["tick"]]) < 0)) if rows else 0
    if r.n_ticks:
        T = r.trace
        out["ticks_with_clip_fraction"] = float(np.mean(T["clip_n"] > 0))
        # effective command change by the gates: |q_pd - q_cmd| per joint
        if "q_pd" in T and "q_cmd" in T:
            ok = np.isfinite(T["q_cmd"]).all(1)
            d = np.abs(T["q_pd"][ok] - T["q_cmd"][ok])
            out["gate_change_rad"] = {"overall_rms": float(np.sqrt(np.mean(d ** 2))), "overall_max": float(d.max()), "worst_joints": [{"joint": names[j], "max": float(d[:, j].max()), "frac_ticks_changed": float(np.mean(d[:, j] > 1e-6))} for j in np.argsort(-d.max(0))[:6]]}
        # margins to hardware limits actually reached by the MEASURED state
        if "pos_lo" in r.manifest:
            lo, hi, vl, ef = r.vec("pos_lo"), r.vec("pos_hi"), r.vec("vel_lim"), r.vec("effort_lim")
            q = T["q"]
            m = np.minimum(q - lo, hi - q)
            out["measured_min_margin_to_joint_limit_rad"] = {"value": float(m.min()), "joint": names[int(np.unravel_index(np.argmin(m), m.shape)[1])]}
            vk = np.array(r.manifest.get("vel_known", [1] * N), dtype=bool)
            ratio = np.abs(T["dq"]) / np.where(vk, vl, 20.0)
            out["measured_max_speed_over_limit"] = {"value": float(ratio.max()), "joint": names[int(np.unravel_index(np.argmax(ratio), ratio.shape)[1])]}
            tr = np.abs(T["tau_est"]) / ef
            out["measured_max_torque_over_limit"] = {"value": float(tr.max()), "joint": names[int(np.unravel_index(np.argmax(tr), tr.shape)[1])]}
            out["raw_policy_target_beyond_limits_fraction"] = float(np.mean((T["q_policy"] < lo) | (T["q_policy"] > hi)))
        out["max_tilt_rad"] = float(T["tilt"].max())
        out["max_gyro_rad_s"] = float(np.max(np.sqrt(T["gyro_x"] ** 2 + T["gyro_y"] ** 2 + T["gyro_z"] ** 2)))
    out["site_gates"] = r.manifest.get("site", {})
    out["fault_events"] = r.report.get("fault_events", [])
    out["guard_stats"] = r.report.get("guard_stats", {})
    return out


# --------------------------------------------------------------------------------------------- exit
def cmd_exit(r: RunLog, a) -> dict:
    ed = r.report.get("exit_detail") or {}
    out = {"available": bool(ed) or bool(r.first_ev("take_over")), "result": r.report.get("result"), "exit_reason": r.report.get("exit_reason"), "final_fsm": r.report.get("summary", {}).get("final_fsm")}
    steps = [e for e in r.events if e.get("kind", "").startswith("exit_") or e["kind"] in ("EXIT_TRIGGER", "take_over", "released", "remote_start_double_click", "remote_l2b_emergency")]
    t0 = None
    for e in steps:
        if e["kind"] in ("exit_submit", "take_over"):
            t0 = e["t_mono"]
    t0 = t0 if t0 is not None else (steps[0]["t_mono"] if steps else 0.0)
    out["timeline_ms_from_submit"] = [{"t_ms": round((e["t_mono"] - t0) * 1e3, 3), "kind": e["kind"], **{k: e[k] for k in ("mode", "rc", "fsm", "result", "fault", "latency_ms", "tilt", "gyro_xy", "why", "detail") if k in e}} for e in sorted(steps, key=lambda e: e["t_mono"])]
    out["exit_detail"] = {k: ed.get(k) for k in ("requested_mode", "final_mode", "final_fsm", "recovery_ok", "confirmed", "rpc_codes", "rpc_modes", "detect_to_rpc_sent_ms", "rpc_latency_ms", "timing_ms_from_detect", "attitude_at_decision", "attitude_at_rpc_sent",
                                                  "attitude_at_fsm_confirmed", "t_frames_stopped", "fsm_trace", "cause_class", "trigger_fault", "trigger_detail", "note", "escalated_to_passive")} if ed else {}
    out["fsm_changes"] = [{"t_mono": e["t_mono"], "from": e["from"], "to": e["to"]} for e in r.ev("fsm_change")]
    out["handover"] = r.report.get("handover")
    # hand-over transient from the full-rate state stream: before / after the exit RPC, and around the take-over
    if r.state is not None and len(r.state):
        s = r.state
        te = t0
        def window(t_a, t_b):
            w = s[(s["t"] >= t_a) & (s["t"] < t_b)]
            if not len(w):
                return None
            q = w["quat"].astype(float)
            tilt = 2 * np.arcsin(np.clip(np.sqrt(q[:, 1] ** 2 + q[:, 2] ** 2), 0, 1))
            return {"n": int(len(w)), "tilt_max": float(tilt.max()), "tilt_mean": float(tilt.mean()), "gyro_max": float(np.max(np.linalg.norm(w["gyro"].astype(float), axis=1))), "dq_abs_max": float(np.abs(w["dq"]).max()),
                    "tau_abs_max": float(np.abs(w["tau"]).max()), "q_end": w["q"][-1].tolist()}
        out["state_before_exit_1s"] = window(te - 1.0, te)
        out["state_after_exit_0_1s"] = window(te, te + 1.0)
        out["state_after_exit_1_3s"] = window(te + 1.0, te + 3.0)
        tt = r.t_takeover()
        if np.isfinite(tt):
            out["state_preroll_before_takeover"] = window(tt - 3.0, tt)
            out["state_first_second_after_takeover"] = window(tt, tt + 1.0)
        # joint jump at hand-back: the last commanded q vs the first OEM-controlled measured q
        if r.frames.get("q_cmd") is not None and len(r.frames["q_cmd"]):
            ft = r.frames["t_mono"]
            tstop = ed.get("t_frames_stopped")
            out["last_exit_frame_t_after_submit_ms"] = float((ft[-1] - t0) * 1e3)
            after = s[s["t"] > ft[-1]]
            if len(after):
                out["handback_joint_jump_rad_max"] = float(np.max(np.abs(after["q"][0].astype(float) - r.frames["q_cmd"][-1])))
            out["exit_frames"] = {"n": int(len(ft)), "kinds": {k: r.frame_kind.count(k) for k in set(r.frame_kind)}}
    return out


# --------------------------------------------------------------------------------------------- driver
CMDS = {"replay": cmd_replay, "init": cmd_init, "tracking": cmd_tracking, "pd": cmd_pd, "latency": cmd_latency, "safety": cmd_safety, "exit": cmd_exit}


def brief(name: str, d: dict) -> str:
    L = [f"## {name}"]
    if not d.get("available", True):
        return "\n".join(L + ["(no data in this run)\n"])
    if name == "replay":
        L.append(f"- {'PASS' if d['pass'] else 'FAIL'}: {d['ticks']} ticks, max |action_replay - action_logged| = {d['max_abs_action_diff']:.3e}; obs-vs-trace {d['obs_vs_trace']}")
    elif name == "init":
        i = d["initial_state"]
        L.append(f"- policy_reset @t_mono {i['t_mono']:.3f}: motion_t {i['motion_t']}, paused {i['paused']}, yaw_offset {np.round(i['yaw_offset_wxyz'], 4).tolist()}, q_hold max |q-default| {np.max(np.abs(np.array(i['q_hold']) - np.array(i.get('default_pos', i['q_hold'])))):.3f}")
        pw = i.get("preroll_window") or {}
        L.append(f"- pre-take-over lowstate window: {pw.get('n', 0)} samples / {pw.get('span_s', 0):.2f} s, max|dq| {pw.get('dq_abs_max', 0):.3f} rad/s -> initial_state.json / .npz")
    elif name == "tracking":
        sp = d["separation"]
        L.append(f"- policy/command/actual vs reference rms: {sp['policy_vs_reference_rms']:.3f} / {sp['command_vs_reference_rms']:.3f} / {sp['actual_vs_reference_rms']:.3f} rad; actual vs command {sp['actual_vs_command_rms']:.3f}; raw target outside limits in {100*sp.get('raw_target_outside_hard_limits_tick_fraction',0):.0f}% of ticks")
        L.append(f"- joint rms error vs reference (ALL ticks incl. lead-in / end-hold): {d['joint_rms_over_joints_vs_ref_rad']:.4f} rad over {d['ticks']} ticks; worst: " + ", ".join(f"{w['joint']} {w['ref_rms']:.3f}" for w in d["worst_joints_vs_ref"][:4]))
        sg = d.get("segments", {})
        for nm in ("lead_in", "dance", "end_hold", "paused_reference"):
            if nm in sg:
                x = sg[nm]
                L.append(f"- [{nm}] {x['ticks']} ticks: joint rms vs ref {x['joint_rms_vs_ref_rad']:.3f} rad (max {x['joint_max_vs_ref_rad']:.3f})" + (f", tilt max {x['tilt_max_rad']:.2f} / ref tilt max {x['ref_tilt_max_rad']:.2f} / gravity err max {x['gravity_err_vs_ref_max_rad']:.2f} rad" if "tilt_max_rad" in x else ""))
        if "tilt_peak" in sg:
            tp = sg["tilt_peak"]
            L.append(f"- peak tilt {tp['tilt_rad']:.2f} rad at t={tp['t_rel']:.2f} s ({tp['segment']}): reference tilt {tp['ref_tilt_rad']:.2f}, gravity error {tp['gravity_err_rad']:.2f}")
        if "root_orientation_error_rad" in d:
            L.append(f"- root orientation error vs reference (yaw aligned): p50 {d['root_orientation_error_rad']['p50']:.3f} / max {d['root_orientation_error_rad']['max']:.3f} rad")
    elif name == "pd":
        L.append("- most loaded joints (tau_est / effort limit): " + ", ".join(f"{m['joint']} {m['tau_est_over_limit_max']:.2f}" for m in d["most_loaded_joints"][:5]))
    elif name == "latency":
        L.append(f"- control period p50/p99/max {d['control_period_ms']['p50']:.2f}/{d['control_period_ms']['p99']:.2f}/{d['control_period_ms']['max']:.2f} ms; inference p99 {d['inference_ms']['p99']:.2f} ms; state age at send p99 {d['state_age_at_send_ms']['p99']:.2f} ms; DDS loopback p99 {d['dds_command_loopback_ms'].get('p99', float('nan')):.3f} ms; lowstate lost {d.get('lowstate_lost_samples', d['lowstate_lost_est_final'])}")
    elif name == "safety":
        L.append(f"- ticks with clip {d.get('ticks_with_clip_fraction', 0) * 100:.1f}%; top: " + ", ".join(f"{p['joint']}[{p['reason']}] {p['max_clip_rad']:.3f}rad x{p['ticks']}" for p in d["per_joint_reason"][:4]))
    elif name == "exit":
        e = d.get("exit_detail", {})
        L.append(f"- result {d['result']} ({d['exit_reason']}), final fsm {d['final_fsm']}; detect->rpc {e.get('detect_to_rpc_sent_ms')} ms, rpc {e.get('rpc_latency_ms')} ms; handback joint jump {d.get('handback_joint_jump_rad_max')}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("what", choices=list(CMDS) + ["all"])
    ap.add_argument("--onnx", default=None, help="policy.onnx for replay (default: ../assets/policy.onnx of the package)")
    ap.add_argument("--out", default=None, help="output directory (default: <run_dir>/analysis)")
    a = ap.parse_args(argv)
    r = RunLog(a.run_dir)
    a.out = a.out or os.path.join(a.run_dir, "analysis")
    os.makedirs(a.out, exist_ok=True)
    todo = list(CMDS) if a.what == "all" else [a.what]
    md = [f"# PartyMimic run analysis: {a.run_dir}", f"mode {r.report.get('mode', r.manifest.get('mode'))}, result {r.report.get('result')}, program {r.manifest.get('program_version')}, policy sha256 {r.manifest.get('sha256', {}).get('policy_onnx', '')[:16]}…, ticks {r.n_ticks}\n"]
    results = {}
    for name in todo:
        try:
            d = CMDS[name](r, a)
        except Exception as ex:  # an analysis that cannot run must say so, not hide the others
            d = {"available": False, "error": f"{type(ex).__name__}: {ex}"}
        results[name] = d
        with open(os.path.join(a.out, f"{name}.json"), "w") as f:
            json.dump(d, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        txt = brief(name, d) if "error" not in d else f"## {name}\nERROR {d['error']}\n"
        md.append(txt)
        print(txt)
    if a.what == "all":
        open(os.path.join(a.out, "analysis_report.md"), "w").write("\n".join(md))
        print(f"analysis written to {a.out}/ (analysis_report.md + one JSON per topic)")
    return 0 if all("error" not in d for d in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
