"""FastTrack supervisor: runs the unmodified EGalahad `Tracking` policy stack (ONNX + supermimic.* observations + SSB npz) through
ControlSession. Same code path for sim (SDK2 DDS on loopback against tools/sim_host.py) and real robot.

Phases
  WAIT_STATE  read-only: wait for a fresh lowstate
  ARM         ControlSession.acquire(): preflight, lock, damping frame, SwitchToUserCtrl, confirm
  PREPARE     gains ramp 0->1 and position ramp measured q -> SSB frame-0 pose (smoothstep, rate-limited)
  SETTLE      hold frame-0 pose, require convergence and that the commands are EFFECTIVE (else NO_RESPONSE)
  SHADOW      policy runs on real state with the reference paused at frame 0; its output is compared, NOT applied
  WAIT_GO     operator 'g' (or --auto-go)
  RAMP_IN     authority 0->1 while the reference is still paused (q_cmd = q_hold + a (q_policy - q_hold))
  DANCE       reference advances for --segment-s seconds (or whole clip)
  END_HOLD    reference paused at the last frame, policy still holding
  EXIT        ControlSession.finish(): damp -> SwitchToInternalCtrl -> verify
Any guard fault / e-stop / watchdog -> DAMP immediately (inside ControlSession), runner goes to EXIT.
"""
from __future__ import annotations

import argparse, json, os, sys, time, traceback
from pathlib import Path

import numpy as np

from . import paths
paths.setup_sys_path(); paths.setup_offline_env()

from .guard import Fault, GuardConfig, GuardLimits, SafetyGuard
from .session import ControlSession, EstopMonitor, SessionConfig
from .guardian import GUARDIAN_ALIVE
from .profile import get_profile, PROFILES


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (10 + x * (-15 + 6 * x))


class NullController:
    name = "null"
    def get_control_mode(self): return None
    def get_extra_keys(self): return []
    def close(self): pass


def build_policy(args, robot_io):
    """Construct EGalahad's Tracking with our RobotIO injected (no keyboard listener, no ZMQ sockets)."""
    from sim2real.rl_policy import base_policy
    from sim2real.rl_policy.tracking import Tracking, TrackingArgs

    class FtTracking(Tracking):
        def _build_controller(self):
            self.keyboard_controller = self.joystick_controller = self.pico_controller = None
            return NullController()

    orig = base_policy.create_robot_io
    base_policy.create_robot_io = lambda **kw: robot_io
    try:
        targs = TrackingArgs(policy_config=str(paths.POLICY_YAML), robot="g1", rl_rate=50.0, inference_backend="onnx-cpu",
                             robot_io="zmq", controller="keyboard", motion_backend="npz", motion_path=str(paths.MOTION_NPZ))
        return FtTracking(args=targs)
    finally:
        base_policy.create_robot_io = orig


class SessionRobotIO:
    """RobotIO adapter (EGalahad ABI) over ControlSession. write_command is unused: the runner owns command flow."""
    def __init__(self, session: ControlSession):
        self.s = session
    def read_state(self):
        from sim2real.rl_policy.robot_io.base import RobotState
        st = self.s.state()
        if st is None:
            return None
        qpos = np.zeros(36, np.float32); qvel = np.zeros(35, np.float32)
        qpos[3:7] = st.quat_wxyz; qpos[7:] = st.q; qvel[3:6] = st.gyro; qvel[6:] = st.dq
        return RobotState(qpos=qpos, qvel=qvel, joint_torque=st.tau_est.astype(np.float32), tick=int(st.tick))
    def write_command(self, *a, **k):
        raise RuntimeError("direct writes are forbidden; use ControlSession.write")
    def close(self): pass


def make_link(args):
    from .link import DdsUserCtrlLink
    if args.target == "sim":
        if args.interface != "lo":
            sys.exit("--target sim must use --interface lo")
        domain = 77
    else:
        if args.interface == "lo":
            sys.exit("--target real cannot use the loopback interface")
        domain = args.domain
    return DdsUserCtrlLink(args.interface, domain, read_only=False, lowcmd=get_profile(args.profile).lowcmd)


REAL_GATE_ENV = "PARTYMIMIC_G1_ALLOW_MOTION"
REAL_GATE_VALUE = "HANGING_SHORT_SEGMENT_ONLY"


def check_real_gates(args) -> list[str]:
    """Return unmet gates; the runner refuses to create a writer on a real robot while any gate is unmet."""
    miss = []
    if os.environ.get(REAL_GATE_ENV) != REAL_GATE_VALUE:
        miss.append(f"env {REAL_GATE_ENV}={REAL_GATE_VALUE} not set")
    if not args.i_confirm_hanging_and_estop:
        miss.append("--i-confirm-hanging-and-estop not given (robot on a hoist, feet clear of floor, remote damping within reach, 2 people)")
    if args.entry_fsm is None:
        miss.append("--entry-fsm N not given (declare the fsm id you observed with bin/ft diag; it is no longer assumed)")
    pf = Path(args.preflight_pass)
    if not pf.exists():
        miss.append(f"no preflight pass file {pf} (run tools/preflight.py against the real robot, read-only)")
    else:
        try:
            d = json.loads(pf.read_text())
            if time.time() - d["time"] > args.preflight_max_age_s: miss.append("preflight pass file too old")
            if d.get("onnx_sha256") != paths.EXPECTED_SHA256["onnx"]: miss.append("preflight pass file is for a different ONNX")
            if d.get("interface") != args.interface: miss.append("preflight was run on a different interface")
            if d.get("profile") != args.profile: miss.append(f"preflight was run with profile {d.get('profile')} not {args.profile}")
            if d.get("fsm") != args.entry_fsm: miss.append(f"preflight observed fsm {d.get('fsm')} != --entry-fsm {args.entry_fsm}")
            if not d.get("diag_ok"): miss.append("no passing read-only diagnostic (bin/ft diag) recorded in the preflight file")
            if not d.get("all_pass"): miss.append("preflight did not pass")
        except Exception as e:
            miss.append(f"preflight pass file unreadable: {e!r}")
    # independent guardian process must be alive (heartbeat file refreshed every 50 ms) -- R0 claimed this gate but the patch never applied
    try:
        g = json.loads(Path(GUARDIAN_ALIVE).read_text())
        if time.time() - g["t"] > 1.0:
            miss.append("guardian heartbeat stale (>1 s): restart `bin/ft guardian IFACE` in its own terminal")
        elif g.get("interface") != args.interface:
            miss.append(f"guardian is watching interface {g.get('interface')}, not {args.interface}")
    except Exception:
        miss.append("guardian not running (start `bin/ft guardian IFACE` in its own terminal BEFORE this command)")
    hold_only = float(getattr(args, "hold_only_s", 0.0) or 0.0)
    if hold_only > 0:
        if hold_only > 5: miss.append("--hold-only-s must be <= 5 for real runs")
    else:
        if args.segment_s is None or args.segment_s > args.real_max_segment_s:
            miss.append(f"--segment-s must be given and <= {args.real_max_segment_s} s for real runs (hanging short-segment policy)")
        if args.authority_max > args.real_max_authority:
            miss.append(f"--authority-max > {args.real_max_authority} not allowed for real runs")
    return miss


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["sim", "real"], required=True)
    ap.add_argument("--interface", required=True, help="DDS NIC ('lo' for sim; robot NIC for real)")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--mode", choices=["shadow", "live"], default="shadow",
                    help="shadow: arm + hold frame-0 pose, policy computed and logged but NOT applied; live: policy applied")
    ap.add_argument("--segment-s", type=float, default=None, help="dance seconds from frame 0 (default: whole clip, sim only)")
    ap.add_argument("--authority-max", type=float, default=1.0)
    ap.add_argument("--auto-go", action="store_true", help="skip the operator 'g' key (sim only)")
    ap.add_argument("--prepare-s", type=float, default=4.0); ap.add_argument("--settle-s", type=float, default=1.0)
    ap.add_argument("--shadow-s", type=float, default=1.0); ap.add_argument("--ramp-in-s", type=float, default=1.0)
    ap.add_argument("--end-hold-s", type=float, default=1.0)
    ap.add_argument("--shadow-gate-rad", type=float, default=1.2, help="refuse 'go' if the shadow policy target is further than this from the hold pose")
    ap.add_argument("--settle-tol-rad", type=float, default=0.25, help="steady-state |q-q_hold| allowed (low-kp joints droop under gravity)")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="demo_walkrun", help="firmware/protocol profile (see fasttrack/profile.py)")
    ap.add_argument("--entry-fsm", type=int, default=None, help="operator-declared fsm id observed on the robot (bin/ft diag); REQUIRED for --target real")
    ap.add_argument("--exit-mode", choices=["AUTO", "LAST", "PASSIVE", "WALKRUN"], default=None, help="override the profile's exit rule")
    ap.add_argument("--expect-user-fsm", type=int, default=None, help="override the profile's user-control fsm id (demo: 1000)")
    ap.add_argument("--torque-frac", type=float, default=0.85)
    ap.add_argument("--target-speed-frac", type=float, default=0.60)
    ap.add_argument("--max-tilt", type=float, default=1.0)
    ap.add_argument("--pos-margin", type=float, default=0.02, help="command clip distance inside hard joint limits (rad)")
    ap.add_argument("--max-gyro", type=float, default=10.0)
    ap.add_argument("--guard-profile", choices=["strict", "open"], default="strict",
                    help="open = SIM ONLY diagnostic: no command clipping / joint+speed+torque+tilt faults (comm, NaN, e-stop, watchdog stay on)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--lock-path", default="/tmp/partymimic_g1_control.lock")
    ap.add_argument("--estop-file", default="/tmp/partymimic_g1_ESTOP")
    ap.add_argument("--preflight-pass", default=str(paths.PKG_ROOT / "runs/preflight_pass.json"))
    ap.add_argument("--preflight-max-age-s", type=float, default=900.0)
    ap.add_argument("--i-confirm-hanging-and-estop", action="store_true")
    ap.add_argument("--real-max-segment-s", type=float, default=3.0)
    ap.add_argument("--real-max-authority", type=float, default=0.3)
    ap.add_argument("--hold-only-s", type=float, default=0.0, help="STAGE S2: after the switch only hold the measured pose with policy gains for this long, then exit (no PREPARE motion, no policy)")
    ap.add_argument("--silence-test-ms", type=float, default=0.0, help="with --hold-only-s: stop sending for this long (<=200) at mid-hold to MEASURE the firmware's command-timeout behaviour")
    ap.add_argument("--unconfirmed-hold-s", type=float, default=30.0)
    ap.add_argument("--diverge-rad", type=float, default=1.5, help="|q - q_cmd_prev| above this for 8 consecutive ticks => NO_RESPONSE")
    ap.add_argument("--debug-stall-at-s", type=float, default=None, help="TEST ONLY: block the control loop to exercise the watchdog")
    ap.add_argument("--debug-stall-s", type=float, default=0.4)
    ap.add_argument("--debug-nan-action-at-s", type=float, default=None, help="TEST ONLY: policy returns NaN")
    a = ap.parse_args(argv)
    import faulthandler, signal as _sg
    faulthandler.register(_sg.SIGUSR1, all_threads=True)      # kill -USR1 <pid> dumps every thread's stack (post-mortem of hangs)

    run_dir = Path(a.run_dir or paths.PKG_ROOT / "runs" / time.strftime("%Y%m%d_%H%M%S")); run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / "console.log", "a", buffering=1)
    def log(m):
        s = f"{time.strftime('%H:%M:%S')} {m}"; print(s, flush=True); logf.write(s + "\n")

    if a.silence_test_ms > 200 or (a.silence_test_ms > 0 and a.hold_only_s <= 0):
        log("REFUSED: --silence-test-ms needs --hold-only-s and must be <= 200"); return 2
    if a.target == "real":
        miss = check_real_gates(a)
        if miss:
            log("REFUSED: real-robot motion gates not met:\n  - " + "\n  - ".join(miss)); return 2
    elif a.segment_s is None and a.mode == "live":
        pass

    from sim2real.config.robots import get_robot_cfg
    cfg = get_robot_cfg("g1")
    lim = GuardLimits.from_robot_cfg(cfg)
    import yaml
    from sim2real.utils.strings import resolve_matching_names_values
    y = yaml.safe_load(paths.POLICY_YAML.read_text()); names = list(cfg.joint_names)
    def expand(d):
        idx, _, v = resolve_matching_names_values(d, names, preserve_order=True, strict=False)
        o = np.zeros(len(names)); o[idx] = v; return o
    kp_y, kd_y, q_def = expand(y["joint_kp"]), expand(y["joint_kd"]), expand(y["default_joint_pos"])
    gcfg = GuardConfig(torque_cmd_frac=a.torque_frac, target_speed_frac=a.target_speed_frac, max_tilt_rad=a.max_tilt,
                       pos_margin_rad=a.pos_margin, max_gyro_rad_s=a.max_gyro)
    if a.guard_profile == "open":
        if a.target != "sim":
            log("REFUSED: --guard-profile open is simulation-only"); return 2
        gcfg = GuardConfig(torque_cmd_frac=1e3, target_speed_frac=1e3, pos_margin_rad=-10.0, limit_fault_tol_rad=10.0,
                           speed_fault_frac=1e3, torque_meas_fault_frac=1e3, max_tilt_rad=3.2, max_gyro_rad_s=1e3)
    guard = SafetyGuard(lim, gcfg, kp_cap=kp_y, kd_floor=None)
    prof = get_profile(a.profile)
    scfg = SessionConfig(lock_path=a.lock_path, estop_file=a.estop_file, profile=prof, declared_entry_fsm=a.entry_fsm,
                         expect_user_fsm=a.expect_user_fsm, internal_exit_mode=a.exit_mode, unconfirmed_exit_hold_s=a.unconfirmed_hold_s)
    link = make_link(a)
    sess = ControlSession(link, guard, scfg, log=log)
    sess.hold_kp, sess.hold_kd = kp_y.copy(), kd_y.copy()
    log(f"profile={prof.name} (verified_on_firmware={prof.verified_on_firmware}) entry-forbidden={prof.entry_forbidden} user_fsm={scfg.user_fsm} exit={scfg.exit_mode} lowcmd={prof.lowcmd}")
    mon = EstopMonitor(sess)
    rep = {"args": vars(a), "phases": [], "result": None}
    trace = {k: [] for k in ["t", "phase", "q", "dq", "tau_est", "quat", "gyro", "action", "q_policy", "q_hold", "q_cmd", "kp", "authority",
                             "state_age", "step_ms", "motion_t", "fault"]}
    phase = "WAIT_STATE"; phase_t0 = time.monotonic()
    def goto(p):
        nonlocal phase, phase_t0
        rep["phases"].append({"phase": phase, "t_end": round(sess.now(), 3)}); phase, phase_t0 = p, time.monotonic(); log(f"=== phase {p}")
    exit_code = 0
    try:
        # --- policy (heavy imports + ONNX session happen BEFORE we touch the robot)
        rio = SessionRobotIO(sess)
        tr = build_policy(a, rio)
        motion = np.load(paths.MOTION_NPZ); q0 = motion["joint_pos"][0].astype(float)
        n_frames = int(motion["joint_pos"].shape[0]); fps = float(motion["fps"][0])
        seg_frames = n_frames - 1 if a.segment_s is None else min(n_frames - 1, int(round(a.segment_s * fps)))
        dt = 1.0 / 50.0
        log(f"policy ready; segment = {seg_frames} frames ({seg_frames / fps:.2f}s) of {n_frames}; mode={a.mode}; target={a.target}")
        # --- wait for state
        t_w = time.monotonic()
        while sess.state() is None:
            if time.monotonic() - t_w > 5.0:
                log("no lowstate within 5 s; abort"); return 3
            time.sleep(0.05)
        goto("ARM")
        if not sess.acquire():
            log(f"ARM FAILED: fault={guard.fault.value}"); exit_code = 4
            rep["result"] = "ARM_FAILED"
            raise SystemExit
        st = sess.state(); q_start = st.q.copy()
        e0 = float(np.abs(q_start - q0).max()); rep["prepare_initial_err_rad"] = e0
        goto("HOLD" if a.hold_only_s > 0 else "PREPARE")
        sil = {"ms": a.silence_test_ms, "started": False, "done": False, "max_dev_rad": 0.0, "pre_max_dev_rad": 0.0, "fsm_after": None, "fsm_before": None}
        authority = 0.0; q_hold = q0.copy(); n = 0; motion_t = 0
        next_t = time.perf_counter(); stall_done = False; nan_done = False
        div_count = 0; settle_ok_since = None; shadow_dev_max = 0.0; go_logged = False; dance_t0 = None
        t_start = time.monotonic()
        while True:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += dt
            tstep = time.perf_counter()
            sess.heartbeat()
            ts = sess.now()
            if a.debug_stall_at_s is not None and not stall_done and ts - 0 > a.debug_stall_at_s and phase in ("DANCE", "RAMP_IN", "SETTLE", "SHADOW", "WAIT_GO"):
                stall_done = True; log("DEBUG: stalling control loop"); time.sleep(a.debug_stall_s)
            if guard.tripped or mon.quit_requested.is_set():
                break
            st = sess.state()
            f = sess.check_state_once(st)
            if f is not Fault.NONE:
                break
            if sess.last_cmd is not None and phase not in ("PREPARE",):
                dv = float(np.abs(st.q - sess.last_cmd[0]).max())
                div_count = div_count + 1 if dv > a.diverge_rad else 0
                if div_count >= 8:
                    sess.emergency(Fault.NO_RESPONSE, f"|q - q_cmd| {dv:.2f} rad > {a.diverge_rad} for 8 ticks (command not followed)"); break
            el = time.monotonic() - phase_t0
            # ---- policy (runs from SHADOW on; before that only the hold/ramp commands exist)
            q_pol = None; action = np.zeros(29)
            if phase in ("SHADOW", "WAIT_GO", "RAMP_IN", "DANCE", "END_HOLD"):
                try:
                    if not tr.state_processor._prepare_low_state():
                        raise RuntimeError("no state")
                    tr.update()
                    obs, _ = tr.prepare_obs_for_rl()
                    tr.state_dict.update(obs); tr.state_dict["is_init"] = np.zeros(1, bool)
                    action, q_pol, tr.state_dict = tr.policy(tr.state_dict)
                    if a.debug_nan_action_at_s is not None and not nan_done and ts > a.debug_nan_action_at_s and phase == "DANCE":
                        nan_done = True; q_pol = q_pol * np.nan; log("DEBUG: injecting NaN policy output")
                    tr.state_dict["action"] = action; tr.state_dict["q_target"] = q_pol
                    q_pol = np.asarray(q_pol, float)
                    if tr.motion_t[0] >= seg_frames and not tr.state_dict.get("paused", False):
                        tr.state_dict["paused"] = True
                    motion_t = int(tr.motion_t[0])
                except Exception as e:
                    sess.emergency(Fault.INFERENCE, repr(e)); log(traceback.format_exc()); break
            # ---- phase logic -> desired (q_cmd, kp, kd)
            kp, kd = kp_y.copy(), kd_y.copy()
            silent = False
            if phase == "HOLD":
                q_cmd = q_start.copy()
                mid = a.hold_only_s / 2
                if a.silence_test_ms > 0 and mid <= el < mid + a.silence_test_ms / 1e3:
                    silent = True
                    if not sil["started"]:
                        sil["started"] = True; sil["fsm_before"] = sess.link.get_fsm()[1]; log(f"SILENCE TEST: no commands for {a.silence_test_ms:.0f} ms")
                if not sil["started"]:
                    sil["pre_max_dev_rad"] = max(sil["pre_max_dev_rad"], float(np.abs(st.q - q_start).max()))     # baseline sag while still commanded
                if sil["started"]:
                    sil["max_dev_rad"] = max(sil["max_dev_rad"], float(np.abs(st.q - q_start).max()))
                if sil["started"] and not silent and not sil["done"]:
                    sil["done"] = True; sil["fsm_after"] = sess.link.get_fsm()[1]; log(f"silence test over: fsm {sil['fsm_before']} -> {sil['fsm_after']}, max |q-q_hold| {sil['max_dev_rad']:.3f} rad (baseline before silence {sil['pre_max_dev_rad']:.3f})")
                if el >= a.hold_only_s:
                    rep["silence_test"] = sil; break
            elif phase == "PREPARE":
                s = smoothstep(el / a.prepare_s)
                q_cmd = q_start + (q0 - q_start) * s
                if prof.acquire_frame != "hold":
                    kp = kp * smoothstep(el / min(1.0, a.prepare_s))      # gain ramp over the first second (damped entry)
                if el >= a.prepare_s:
                    err_p = float(np.abs(st.q - q0).max())
                    if e0 > 0.2 and err_p > 0.6 * e0:
                        sess.emergency(Fault.NO_RESPONSE, f"after the {a.prepare_s:.1f}s ramp the pose error only went {e0:.2f} -> {err_p:.2f} rad: commands are not effective"); break
                    goto("SETTLE"); settle_ok_since = None
            elif phase == "SETTLE":
                q_cmd = q0.copy()
                err = float(np.abs(st.q - q0).max()); spd = float(np.abs(st.dq).max())
                responsive = err < max(0.5 * e0, a.settle_tol_rad)     # command must have moved the robot most of the way
                if err < a.settle_tol_rad and responsive and spd < 0.5:
                    settle_ok_since = settle_ok_since or time.monotonic()
                    if time.monotonic() - settle_ok_since >= a.settle_s:
                        tr.reset(); tr.state_dict["control_mode"] = "policy"; tr.state_dict["paused"] = True
                        goto("SHADOW")
                else:
                    settle_ok_since = None
                    if el > 3.0:
                        sess.emergency(Fault.NO_RESPONSE, f"pose not reached: max err {err:.2f} rad, max |dq| {spd:.2f} after {el:.1f}s (command path ineffective or blocked)")
                        break
            elif phase in ("SHADOW", "WAIT_GO", "RAMP_IN", "DANCE", "END_HOLD"):
                q_cmd = q_hold.copy()
                if phase == "SHADOW":
                    shadow_dev_max = max(shadow_dev_max, float(np.abs(q_pol - q_hold).max()))
                    if el >= a.shadow_s:
                        log(f"shadow max |q_policy - q_hold| = {shadow_dev_max:.3f} rad (gate {a.shadow_gate_rad})")
                        rep["shadow_dev_max_rad"] = shadow_dev_max
                        rep["settle_err_rad"] = float(np.abs(st.q - q0).max())
                        goto("WAIT_GO")
                elif phase == "WAIT_GO":
                    shadow_dev_max = max(shadow_dev_max, float(np.abs(q_pol - q_hold).max()))
                    if a.mode == "shadow":
                        goto("DANCE")        # shadow runs the clock but never applies authority
                    elif a.auto_go or mon.go_requested.is_set():
                        if shadow_dev_max > a.shadow_gate_rad:
                            log(f"GO REFUSED: shadow policy deviates {shadow_dev_max:.2f} rad > {a.shadow_gate_rad}"); sess.emergency(Fault.NO_RESPONSE, "shadow gate"); break
                        goto("RAMP_IN")
                    elif not go_logged:
                        log("WAIT_GO: press 'g' to apply the policy, 'x'/space/Ctrl-C/ESC = e-stop"); go_logged = True
                if phase in ("RAMP_IN", "DANCE", "END_HOLD"):
                    if phase == "RAMP_IN":
                        authority = a.authority_max * smoothstep(el / a.ramp_in_s)
                        if el >= a.ramp_in_s:
                            authority = a.authority_max
                            tr.state_dict["paused"] = False; goto("DANCE"); dance_t0 = time.monotonic()
                    elif phase == "DANCE":
                        authority = a.authority_max if a.mode == "live" else 0.0
                        if tr.state_dict.get("paused", False):
                            goto("END_HOLD")
                    elif phase == "END_HOLD":
                        authority = a.authority_max if a.mode == "live" else 0.0
                        if el >= a.end_hold_s:
                            break
                    q_cmd = q_hold + authority * (q_pol - q_hold)
                elif phase == "DANCE":
                    pass
                if a.mode == "shadow" and phase == "DANCE":
                    authority = 0.0
            else:
                q_cmd = q0.copy()
            if a.mode == "shadow" and phase == "WAIT_GO":
                pass
            # shadow mode: dance clock must start un-paused
            if a.mode == "shadow" and phase == "DANCE" and tr.state_dict.get("paused", False) and motion_t == 0:
                tr.state_dict["paused"] = False
            sent = True if silent else sess.write(q_cmd, np.zeros(29), np.zeros(29), kp, kd, dt)
            if not sent:
                break
            tq = trace
            tq["t"].append(ts); tq["phase"].append(phase); tq["q"].append(st.q); tq["dq"].append(st.dq); tq["tau_est"].append(st.tau_est)
            tq["quat"].append(st.quat_wxyz); tq["gyro"].append(st.gyro); tq["action"].append(action)
            tq["q_policy"].append(q_pol if q_pol is not None else np.full(29, np.nan)); tq["q_hold"].append(q_hold)
            tq["q_cmd"].append(sess.last_cmd[0]); tq["kp"].append(sess.last_cmd[3]); tq["authority"].append(authority)
            tq["state_age"].append(time.monotonic() - st.arrival_mono); tq["motion_t"].append(motion_t)
            tq["step_ms"].append(1e3 * (time.perf_counter() - tstep)); tq["fault"].append(0)
            if (1e3 * (time.perf_counter() - tstep)) > 3 * 1e3 * dt:
                sess.emergency(Fault.LOOP_OVERRUN, f"control step took {1e3 * (time.perf_counter() - tstep):.0f} ms"); break
        rep["result"] = "FAULT" if guard.tripped else ("OPERATOR_QUIT" if mon.quit_requested.is_set() else "COMPLETED")
    except SystemExit:
        pass
    except BaseException as e:
        log("UNHANDLED " + traceback.format_exc())
        sess.emergency(Fault.INFERENCE, f"unhandled {e!r}"); rep["result"] = "CRASH"; exit_code = 1
    finally:
        goto("EXIT") if phase != "EXIT" else None
        exit_rep = sess.finish(reason=rep.get("result") or "unknown")
        rep["exit"] = exit_rep; rep["fault"] = guard.fault.value
        rep["fault_events"] = [{"t": e.t, "fault": e.fault.value, "detail": e.detail} for e in guard.events]
        rep["guard_stats"] = guard.stats; rep["guard_max_ratio"] = guard.max_ratio
        rep["session_events"] = sess.events; rep["frames_sent"] = sess.n_sent
        arr = {k: np.asarray(v) for k, v in trace.items() if len(v)}
        if arr:
            np.savez_compressed(run_dir / "runner_trace.npz", **arr)
        (run_dir / "run_report.json").write_text(json.dumps(rep, indent=1, default=str))
        link.close()
        log(f"RESULT {rep.get('result')} fault={guard.fault.value} exit={exit_rep.get('confirmed')} -> {run_dir}")
        if exit_rep.get("confirmed") is False:
            exit_code = max(exit_code, 5)
        elif guard.tripped and exit_code == 0:
            exit_code = 10
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
