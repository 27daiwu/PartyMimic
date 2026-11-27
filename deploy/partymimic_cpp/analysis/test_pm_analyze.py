#!/usr/bin/env python3
"""Self-test of the P5.1 analysis kit on a REAL run directory written by this program (a --sim loopback run is enough):
    python3 analysis/test_pm_analyze.py <run_dir> [--crash-dir <salvaged run dir>]
Asserts that every analysis the next optimisation round needs can be computed from the logs alone, and that the replay is bit-exact."""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pm_analyze  # noqa: E402
from pmlog import N, RunLog  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    ok = ok and bool(cond)


def run(r, what, out):
    class A:  # noqa
        onnx = None
    A.out = out
    return pm_analyze.CMDS[what](r, A)


def main():
    d = sys.argv[1]
    out = tempfile.mkdtemp()
    r = RunLog(d)
    T = r.trace
    check("loaded: 29 joints, ticks, obs rows == ticks", len(r.joint_names) == N and r.n_ticks > 20 and r.obs is not None and len(r.obs) == r.n_ticks, f"({r.n_ticks} ticks)")
    check("state stream: rows, strictly ordered times, pre-roll before the first tick, post-exit after the last", r.state is not None and np.all(np.diff(r.state["t"]) >= 0) and r.state["t"][0] < T["t_mono"][0] - 3.0 and r.state["t"][-1] > T["t_mono"][-1] + 1.0)
    check("trace has raw action / history action / reference / PD target / final command / tau_pd / kp / kd", all(k in T for k in ("action", "action_prev", "ref_jp", "ref_jv", "q_policy", "q_pd", "q_cmd", "tau_pd", "kp", "kd", "q", "dq", "tau_est")))
    # identities that must hold by construction
    ok_cmd = np.isfinite(T["q_cmd"]).all(1)
    check("action(t) == action_prev(t+1) (history consistent)", np.max(np.abs(T["action"][:-1] - T["action_prev"][1:])) < 1e-6)
    man = r.manifest
    qp = np.array(man["default_pos"]) + np.array(man["action_scale"]) * T["action"]
    check("q_policy == default + action * scale (float32 action, float64 map)", np.max(np.abs(qp - T["q_policy"])) < 1e-5)
    auth = T["authority"][:, None]
    # q_hold is the measured q at the take-over (policy_reset event)
    qh = np.array(r.first_ev("policy_reset")["q_hold"])
    check("q_pd == q_hold + authority * (q_policy - q_hold)", np.max(np.abs(qh + auth * (T["q_policy"] - qh) - T["q_pd"])) < 1e-6)
    check("tau_pd == kp*(q_cmd - q) - kd*dq", np.max(np.abs(T["kp"][ok_cmd] * (T["q_cmd"][ok_cmd] - T["q"][ok_cmd]) - T["kd"][ok_cmd] * T["dq"][ok_cmd] - T["tau_pd"][ok_cmd])) < 1e-2)
    rp = run(r, "replay", out)
    check("replay: max action diff <= 1e-5 (0 on the same architecture)", rp.get("pass"), f"({rp.get('max_abs_action_diff')})")
    check("replay: observation newest frame == logged q / dq / previous action", rp["obs_vs_trace"]["q_vs_trace"] < 1e-5 and rp["obs_vs_trace"]["prev_action_vs_trace"] < 1e-5)
    ini = run(r, "init", out)
    i = ini["initial_state"]
    check("Sim2Sim initial state: q/dq/quat/gyro/yaw offset/q_hold/gains/limits + pre-take-over window", len(i["q"]) == N and len(i["quat_wxyz"]) == 4 and len(i["yaw_offset_wxyz"]) == 4 and len(i["kp"]) == N and len(i["pos_lo"]) == N and i["preroll_window"]["n"] > 100)
    check("initial state matches the first tick (q at reset == q of tick 0)", i["first_tick_q_minus_reset_q_max"] < 1e-5)
    tr = run(r, "tracking", out)
    check("tracking error: per joint rms/max vs reference and vs command + root orientation + policy/command/actual separation", tr["available"] and len(tr["per_joint"]) == N and "root_orientation_error_rad" in tr and "separation" in tr)
    # ---- V1.6 dance time axis / segments (SSB-first-frame legacy semantics)
    li = r.lead_in_frames
    f = T["ssb_frame"].astype(int)
    check("dance_frame == ssb_frame - lead_in_frames, dance_t == dance_frame / fps; seg column present", "dance_frame" in T and "dance_t" in T and "seg" in T and np.all(T["dance_frame"] == f - li) and np.allclose(T["dance_t"], T["dance_frame"] / r.fps, atol=1e-3), f"(lead_in_frames {li}, derived from ssb_frame: {r.dance_axis_derived})")
    sg = tr["segments"]
    if r.manifest.get("ref_mode") == "running" and li > 0:
        check("segments: lead-in = frozen first frame (ref_rq / ref_jp constant, dance_frame < 0), dance and end-hold apart, take-over tick excluded", "lead_in" in sg and "dance" in sg and np.all(T["dance_frame"][r.seg_mask("lead_in")] < 0) and np.ptp(T["ref_jp"][: max(2, li // 2)], axis=0).max() < 1e-6 and r.seg_mask("lead_in").sum() + r.seg_mask("dance").sum() + r.seg_mask("end_hold").sum() + 1 == r.n_ticks, f"(lead-in {sg['lead_in']['ticks']}, dance {sg['dance']['ticks']}, hold {sg.get('end_hold', {}).get('ticks', 0)} of {r.n_ticks} ticks)")
        check("n_ticks_dance excludes the lead-in and the hold; per-segment tilt / reference tilt / gravity error present", r.n_ticks_dance == sg["dance"]["ticks"] < r.n_ticks and all(k in sg["dance"] for k in ("tilt_max_rad", "ref_tilt_max_rad", "gravity_err_vs_ref_max_rad", "tilt_minus_ref_tilt_max_rad")))
    check("error_vs_time carries dance_t and the segment name; yaw alignment note", all("dance_t" in x and "seg" in x for x in tr["error_vs_time"]) and "root_orientation_error_by_segment_rad" in tr)
    pd = run(r, "pd", out)
    check("PD analysis: effective kp/kd, lag, torque margins per joint", pd["available"] and len(pd["per_joint"]) >= 20)
    la = run(r, "latency", out)
    check("latency: period / inference / state age / send / DDS loopback / lowstate gaps / transport jitter / command lag / RPC", all(k in la for k in ("control_period_ms", "inference_ms", "state_age_at_send_ms", "send_call_us", "dds_command_loopback_ms", "lowstate_gap_ms", "lowstate_transport_jitter_ms", "command_to_response_lag", "rpc_latency_ms")))
    sf = run(r, "safety", out)
    check("safety-gate analysis: per joint/reason episodes, gate change, margins", "per_joint_reason" in sf and "gate_change_rad" in sf and "measured_min_margin_to_joint_limit_rad" in sf)
    ex = run(r, "exit", out)
    check("exit diagnostics: timeline, attitude at decision/rpc/confirmation, state before/after, hand-back jump", ex["available"] and len(ex["timeline_ms_from_submit"]) >= 5 and ex["exit_detail"].get("attitude_at_decision") and ex.get("state_after_exit_0_1s") and "handback_joint_jump_rad_max" in ex)
    kinds = [t["kind"] for t in ex["timeline_ms_from_submit"]]
    check("exit timeline order: submit < rpc attempt < rpc result < fsm matched < stable < result", [k for k in kinds if k in ("exit_submit", "exit_rpc_attempt", "exit_rpc_result", "exit_fsm_matched", "exit_stable_confirmed", "exit_result")] == ["exit_submit", "exit_rpc_attempt", "exit_rpc_result", "exit_fsm_matched", "exit_stable_confirmed", "exit_result"])
    if "--crash-dir" in sys.argv:
        c = RunLog(sys.argv[sys.argv.index("--crash-dir") + 1])
        rp2 = run(c, "replay", out)
        check("crashed run: ticks recorded up to the crash are still replayable bit-exactly", rp2.get("pass") and rp2["ticks"] >= 20, f"({rp2.get('ticks')} ticks)")
        check("crashed run: salvage report + CRASH marker", c.report.get("result") == "CRASH_NO_REPORT" and len(c.ev("CRASH")) >= 1)
    # driver end-to-end
    rc = pm_analyze.main([d, "all", "--out", out])
    check("pm_analyze.py all: exit 0, analysis_report.md + 7 JSON files", rc == 0 and os.path.exists(os.path.join(out, "analysis_report.md")) and len([f for f in os.listdir(out) if f.endswith(".json")]) >= 7)
    print("ANALYSIS KIT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
