"""FastTrack runner: loads the frozen P2 Policy, checks the live robot read-only, waits for GO, takes control from WalkRun, runs the SSB window, hands back.

Phases (status file `phase`):
  STARTING -> LOADING_MODEL (ONNX + observation stack + warm-up on live states, NO command) -> CHECKING (real-time admission, read-only)
  -> READY (waits for the GO file written by the service on L1+A; SIGTERM / START x2 / timeout abort WITHOUT ever touching control)
  -> ARM (fast preflight, lock, SwitchToUserCtrl, confirm) -> RUN (Policy drives from the first tick; authority ramp 0->max in 0.1 s)
  -> EXIT (damp -> SwitchToInternalCtrl -> verify by GetFsmId).
Any guard fault / e-stop / START x2 / watchdog -> damping at once (ControlSession), then EXIT.  Guard thresholds are the frozen defaults (not CLI-tunable).
Exit codes: 0 clean | 1 crash | 2 refused | 3 no lowstate | 4 ARM failed | 5 EXIT UNCONFIRMED | 6 BLOCKED by admission | 10 fault/stop, hand-back confirmed
"""
from __future__ import annotations

import argparse, json, os, sys, time, traceback
from pathlib import Path

import numpy as np

from . import paths
paths.setup_sys_path(); paths.setup_offline_env()

from . import checks
from .guard import Fault, GuardConfig, GuardLimits, SafetyGuard
from .session import ControlSession, EstopMonitor, SessionConfig
from .profile import get_profile
from .tracelog import JsonlLog, TraceCsv

DT = 1.0 / 50.0
RAMP_S = 0.1                      # authority 0 -> max, so the first policy frame is not a step


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (10 + x * (-15 + 6 * x))


class NullController:
    name = "null"
    def get_control_mode(self): return None
    def get_extra_keys(self): return []
    def close(self): pass


def build_policy(args, robot_io, motion_npz=None):
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
                             robot_io="zmq", controller="keyboard", motion_backend="npz", motion_path=str(motion_npz or paths.MOTION_NPZ))
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


def policy_tick(tr):
    """One policy step on the current state (the same code path for warm-up, selfcheck and the live loop)."""
    tr.update()
    obs, _ = tr.prepare_obs_for_rl()
    tr.state_dict.update(obs); tr.state_dict["is_init"] = np.zeros(1, bool)
    action, q_pol, tr.state_dict = tr.policy(tr.state_dict)
    tr.state_dict["action"] = action; tr.state_dict["q_target"] = q_pol
    return np.asarray(action, float), np.asarray(q_pol, float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["sim", "real"], required=True, help="sim = loopback test robot (domain 77); real = the G1")
    ap.add_argument("--interface", required=True); ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--entry-fsm", type=int, default=None, help="only this WalkRun fsm id may be the entry (default: config/site.json entry_fsm_allowed)")
    ap.add_argument("--run-s", type=float, default=1.0, help=f"seconds of Policy control after the take-over (real: <= {checks.REAL_MAX_RUN_S})")
    ap.add_argument("--ref-mode", choices=["running", "paused"], default="running", help="running = SSB clock runs from frame 0; paused = hold SSB frame 0")
    ap.add_argument("--authority-max", type=float, default=0.3, help=f"q_cmd = q_start + a*(q_policy-q_start) (real: <= {checks.REAL_MAX_AUTHORITY})")
    ap.add_argument("--i-confirm-protection", action="store_true", help="REAL only: fall protection in place, remote e-stop in hand, >= 2 people (set by start.sh after you confirmed)")
    ap.add_argument("--run-dir", default=None); ap.add_argument("--site", default=None, help="config/site.json override")
    ap.add_argument("--lock-path", default="/tmp/partymimic_g1_control.lock"); ap.add_argument("--estop-file", default="/tmp/partymimic_g1_ESTOP")
    ap.add_argument("--wait-go", default=None, metavar="FILE", help="service mode: report READY, then wait for FILE (written on L1+A); without it GO is immediate")
    ap.add_argument("--status-file", default=None, metavar="FILE"); ap.add_argument("--ready-timeout-s", type=float, default=120.0)
    ap.add_argument("--warmup-n", type=int, default=8); ap.add_argument("--unconfirmed-hold-s", type=float, default=30.0)
    ap.add_argument("--diverge-rad", type=float, default=1.5, help="|q - q_cmd_prev| above this for 8 consecutive ticks => NO_RESPONSE")
    a = ap.parse_args(argv)
    import faulthandler, signal as _sg
    faulthandler.register(_sg.SIGUSR1, all_threads=True)

    run_dir = Path(a.run_dir or paths.LOGS / "runs" / time.strftime("%Y%m%d_%H%M%S")); run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / "console.log", "a", buffering=1); events = JsonlLog(run_dir / "events.jsonl")
    def log(m):
        s = f"{time.strftime('%H:%M:%S')} {m}"; print(s, flush=True); logf.write(s + "\n")
    def status(phase, **kw):
        events.write(kind="phase", phase=phase, t_mono=time.monotonic(), **kw)
        if not a.status_file: return
        try:
            tmp = a.status_file + ".tmp"; Path(tmp).write_text(json.dumps({"phase": phase, "tm": time.monotonic(), "time": time.time(), "pid": os.getpid(), **kw})); os.replace(tmp, a.status_file)
        except Exception:
            pass
    status("STARTING")
    site = checks.load_site(a.site)
    allowed = checks.entry_allowed(site, a.entry_fsm)
    # ---- hard refusals before ANY DDS object exists
    cap = checks.caps_item(a.target, a.run_s, a.authority_max)
    if not cap["pass"]:
        log("REFUSED: " + cap["detail"]); return 2
    if a.target == "real":
        if a.interface == "lo": log("REFUSED: --target real cannot use the loopback interface"); return 2
        if not a.i_confirm_protection: log("REFUSED: real target needs --i-confirm-protection (start.sh asks you; fall protection, e-stop in hand, >= 2 people)"); return 2
        domain = a.domain
    else:
        if a.interface != "lo": log("REFUSED: --target sim must use --interface lo"); return 2
        domain = 77

    from sim2real.config.robots import get_robot_cfg
    cfg = get_robot_cfg("g1"); lim = GuardLimits.from_robot_cfg(cfg); names = list(cfg.joint_names)
    import yaml
    from sim2real.utils.strings import resolve_matching_names_values
    y = yaml.safe_load(paths.POLICY_YAML.read_text())
    def expand(d):
        idx, _, v = resolve_matching_names_values(d, names, preserve_order=True, strict=False)
        o = np.zeros(len(names)); o[idx] = v; return o
    kp_y, kd_y = expand(y["joint_kp"]), expand(y["joint_kd"])
    guard = SafetyGuard(lim, GuardConfig(), kp_cap=kp_y, kd_floor=None)          # frozen strict thresholds
    prof = get_profile("demo_walkrun")
    scfg = SessionConfig(lock_path=a.lock_path, estop_file=a.estop_file, profile=prof, entry_allowed=allowed, unconfirmed_exit_hold_s=a.unconfirmed_hold_s)
    from .link import DdsUserCtrlLink
    link = DdsUserCtrlLink(a.interface, domain, read_only=False, lowcmd=prof.lowcmd)
    sess = ControlSession(link, guard, scfg, log=log)
    sess.on_event = lambda e: events.write(**e)
    sess.hold_kp, sess.hold_kd = kp_y.copy(), kd_y.copy()
    mon = EstopMonitor(sess, tty_keys=False)
    log(f"runner: target={a.target} iface={a.interface} domain={domain} run_s={a.run_s} ref={a.ref_mode} authority<={a.authority_max} entry_allowed={allowed}")
    rep = {"args": vars(a), "result": None, "exit_reason": None, "phases": []}
    trace = TraceCsv(run_dir / "trace.csv", names)
    exit_code = 0; phase = "STARTING"; t_run0 = None; n_ticks = 0
    def goto(p):
        nonlocal phase
        rep["phases"].append({"phase": phase, "t_end": round(sess.now(), 3)}); phase = p; log(f"=== phase {p}"); status(p)
    try:
        # ---- model + warm-up (read-only)
        goto("LOADING_MODEL")
        rio = SessionRobotIO(sess)
        tr = build_policy(a, rio)
        motion = np.load(paths.MOTION_NPZ); n_frames = int(motion["joint_pos"].shape[0]); fps = float(motion["fps"][0])
        seg_frames = min(n_frames - 1, int(round(a.run_s * fps)))
        t_w = time.monotonic()
        while sess.state() is None:
            if time.monotonic() - t_w > 5.0:
                log("no lowstate within 5 s; abort"); rep["result"] = "NO_LOWSTATE"; exit_code = 3; raise SystemExit
            time.sleep(0.05)
        wl = []
        for _ in range(max(0, a.warmup_n)):                  # read-only: warms ONNX / FK so the first live inference is not a 50-100 ms outlier (watchdog 100 ms)
            tw = time.perf_counter()
            if not tr.state_processor._prepare_low_state(): break
            policy_tick(tr); wl.append(1e3 * (time.perf_counter() - tw)); time.sleep(0.02)
        rep["warmup_ms"] = wl
        log(f"warm-up: {len(wl)} read-only inferences, first {wl[0]:.1f} ms, last {wl[-1]:.1f} ms, max {max(wl):.1f} ms" if wl else "warm-up: none (no state)")
        if wl and max(wl[-3:]) > 40: log("WARNING: inference still slow after warm-up (watchdog is 100 ms)")
        # ---- real-time admission (read-only)
        goto("CHECKING")
        def admission():
            pf_ok, pf = sess.preflight()
            try: writers = checks.discover_writers()
            except Exception as e: writers = e
            ready, items = checks.admission(target=a.target, iface=a.interface, run_s=a.run_s, authority=a.authority_max, site=site, pf_results=pf, writers=writers)
            return ready, items, writers
        ready, items, writers = admission()
        st0 = sess.state(); code, fsm = link.get_fsm()
        rep["admission"] = {"ready": ready, "items": items}; rep["entry_fsm_seen"] = fsm
        (run_dir / "admission.json").write_text(json.dumps(rep["admission"], indent=1, default=str))
        (run_dir / "ready.json").write_text(json.dumps({"time": time.time(), "fsm": fsm, "rpc_code": code, "writers": writers if isinstance(writers, dict) else repr(writers), "admission": rep["admission"],
                                                        "state": None if st0 is None else {"q": st0.q.tolist(), "dq": st0.dq.tolist(), "tau_est": st0.tau_est.tolist(), "quat_wxyz": st0.quat_wxyz.tolist(),
                                                                                          "gyro": st0.gyro.tolist(), "rpy": list(st0.rpy), "mode_machine": st0.mode_machine, "wireless_remote": st0.wireless.hex()}}, indent=1, default=str))
        for i in items:
            log(f"[{'PASS' if i['pass'] else 'FAIL'}] {i['item']}: {i['detail']}")
        if not ready:
            bad = [i for i in items if not i["pass"]]
            log("BLOCKED: real-time admission not met (nothing was sent): " + "; ".join(f"{i['item']}" for i in bad)); rep["result"] = "BLOCKED"; exit_code = 6; raise SystemExit
        # ---- READY / GO
        if a.wait_go:
            log("READY: model loaded, checks passed. Waiting for GO (L1+A) / abort (START x2, SIGTERM).")
            status("READY"); t_ready = time.monotonic()
            while not os.path.exists(a.wait_go):
                if mon.quit_requested.is_set():
                    log("ABORTED before ARM (signal): control was never taken."); rep["result"] = "ABORTED_BEFORE_ARM"; rep["exit_reason"] = "signal before ARM"; raise SystemExit
                if sess._remote_fired_tm is not None:
                    log("ABORTED before ARM by remote START double-click: control was never taken."); rep["result"] = "ABORTED_BEFORE_ARM"; rep["exit_reason"] = "START x2 before ARM"; raise SystemExit
                if time.monotonic() - t_ready > a.ready_timeout_s:
                    log(f"READY timed out after {a.ready_timeout_s:.0f} s without GO."); rep["result"] = "READY_TIMEOUT"; rep["exit_reason"] = "READY timeout"; raise SystemExit
                time.sleep(0.02)
            try: os.remove(a.wait_go)
            except OSError: pass
            # state may have changed while waiting: only the state-dependent checks repeat (guardian, OEM writers); sess.acquire() re-runs the fast preflight itself
            g_item = checks.guardian_item(a.interface)
            try: w2 = checks.discover_writers()
            except Exception as e: w2 = e
            w_item = checks.oem_writer_item(a.target, site, w2)
            events.write(kind="go_recheck", items=[g_item, w_item])
            if not (g_item["pass"] and w_item["pass"]):
                log("GO REFUSED: " + "; ".join(f"{i['item']}: {i['detail']}" for i in (g_item, w_item) if not i["pass"])); rep["result"] = "GO_REFUSED"; exit_code = 6; raise SystemExit
            log("GO received -> ARM")
        # ---- ARM
        goto("ARM")
        if not sess.acquire(fast=True):
            if guard.fault is Fault.REMOTE_STOP and sess.mode == sess.OFF:
                log("ABORTED before control by START x2 during ARM"); rep["result"] = "ABORTED_BEFORE_ARM"; rep["exit_reason"] = "START x2 during ARM"; raise SystemExit
            log(f"ARM FAILED: fault={guard.fault.value}: {guard.events[-1].detail if guard.events else ''}"); rep["result"] = "ARM_FAILED"; exit_code = 4; raise SystemExit
        st = sess.state(); q_start = st.q.copy(); q_hold = q_start.copy()
        rep["entry_fsm"] = sess.entry_fsm; rep["user_fsm"] = sess.user_fsm_seen
        goto("RUN")
        if not tr.state_processor._prepare_low_state():
            sess.emergency(Fault.INFERENCE, "no robot state available for policy reset"); raise SystemExit
        tr.reset(); tr.state_dict["control_mode"] = "policy"; tr.state_dict["paused"] = (a.ref_mode == "paused")
        t_take = time.monotonic(); next_t = time.perf_counter(); div_count = 0; wire_seen = set(); first_cmd = True
        while True:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += DT
            seq0 = (lambda s_: None if s_ is None else s_.seq)(sess.link.latest()); t_w = time.perf_counter() + 0.008      # sample-synchronous tick: shave the state age
            while time.perf_counter() < t_w and (lambda s_: s_ is not None and s_.seq == seq0)(sess.link.latest()):
                time.sleep(0.0003)
            tstep = time.perf_counter()
            sess.heartbeat(); ts = sess.now()
            if guard.tripped or mon.quit_requested.is_set():
                break
            st = sess.state()
            if sess.check_state_once(st) is not Fault.NONE:
                break
            if sess.last_cmd is not None:
                dv = float(np.abs(st.q - sess.last_cmd[0]).max()); div_count = div_count + 1 if dv > a.diverge_rad else 0
                if div_count >= 8:
                    sess.emergency(Fault.NO_RESPONSE, f"|q - q_cmd| {dv:.2f} rad > {a.diverge_rad} for 8 ticks (command not followed)"); break
            el = time.monotonic() - t_take
            try:
                if not tr.state_processor._prepare_low_state():
                    raise RuntimeError("no state")
                t_inf = time.perf_counter(); action, q_pol = policy_tick(tr); infer_ms = 1e3 * (time.perf_counter() - t_inf)
                motion_t = int(tr.motion_t[0])
            except Exception as e:
                sess.emergency(Fault.INFERENCE, repr(e)); log(traceback.format_exc()); break
            authority = a.authority_max * smoothstep(el / RAMP_S)
            q_cmd = q_hold + authority * (q_pol - q_hold)
            sent = sess.write(q_cmd, np.zeros(29), np.zeros(29), kp_y.copy(), kd_y.copy(), DT)
            if not sent:
                break
            if first_cmd:
                first_cmd = False; rep["first_cmd_step_rad"] = float(np.abs(sess.last_cmd[0] - st.q).max()); rep["first_cmd_authority"] = float(authority)
            n_ticks += 1; wire_seen.add(bytes(st.wireless))
            step_ms = 1e3 * (time.perf_counter() - tstep)
            from .profile import roll_pitch
            r_, p_ = roll_pitch(st.quat_wxyz)
            trace.row({"t": round(ts, 4), "phase": phase, "ssb_frame": motion_t, "authority": authority, "infer_ms": infer_ms, "step_ms": step_ms,
                       "state_age_ms": 1e3 * (time.monotonic() - st.arrival_mono), "tick": st.tick,
                       "keys_b2": (st.wireless[2] if len(st.wireless) > 3 else None), "keys_b3": (st.wireless[3] if len(st.wireless) > 3 else None),
                       "quat_w": st.quat_wxyz[0], "quat_x": st.quat_wxyz[1], "quat_y": st.quat_wxyz[2], "quat_z": st.quat_wxyz[3],
                       "gyro_x": st.gyro[0], "gyro_y": st.gyro[1], "gyro_z": st.gyro[2], "roll": r_, "pitch": p_, "yaw": st.rpy[2]},
                      q=st.q, dq=st.dq, tau_est=st.tau_est, action=action, q_policy=q_pol, q_cmd=sess.last_cmd[0])
            if step_ms > 3 * 1e3 * DT:
                sess.emergency(Fault.LOOP_OVERRUN, f"control step took {step_ms:.0f} ms"); break
            if (a.ref_mode == "paused" and el >= a.run_s) or (a.ref_mode == "running" and motion_t >= seg_frames) or el > a.run_s + 1.0:
                break                                                                   # run window over -> EXIT hand-back
        rep["result"] = "FAULT" if guard.tripped else ("OPERATOR_QUIT" if mon.quit_requested.is_set() else "COMPLETED")
        rep["wireless_remote_distinct_payloads_in_run"] = len(wire_seen)
    except SystemExit:
        pass
    except BaseException as e:
        log("UNHANDLED " + traceback.format_exc())
        sess.emergency(Fault.INFERENCE, f"unhandled {e!r}"); rep["result"] = "CRASH"; exit_code = 1; rep["exit_reason"] = f"crash: {e!r}"
    finally:
        goto("EXIT") if phase != "EXIT" else None
        exit_rep = sess.finish(reason=rep.get("result") or "unknown")
        rep["exit"] = exit_rep; rep["fault"] = guard.fault.value
        if rep.get("exit_reason") is None:
            rep["exit_reason"] = (f"{guard.fault.value}: {guard.events[0].detail}" if guard.tripped else
                                  ("operator quit (signal)" if rep.get("result") == "OPERATOR_QUIT" else f"{rep.get('result')}"))
        rep["fault_events"] = [{"t": e.t, "fault": e.fault.value, "detail": e.detail} for e in guard.events]
        rep["guard_stats"] = guard.stats; rep["guard_max_ratio"] = guard.max_ratio
        rep["frames_sent"] = sess.n_sent; rep["control_ticks"] = n_ticks
        hb = exit_rep.get("handback") or {}
        rep["restored_fsm"] = exit_rep.get("fsm_after"); rep["restored_to_entry_fsm"] = (exit_rep.get("fsm_after") == rep.get("entry_fsm")) if rep.get("entry_fsm") is not None else None
        rep["session_events"] = sess.events
        trace.close()
        (run_dir / "run_report.json").write_text(json.dumps(rep, indent=1, default=str))
        events.write(kind="result", result=rep.get("result"), exit_reason=rep["exit_reason"], fault=guard.fault.value, exit_confirmed=exit_rep.get("confirmed"),
                     restored_fsm=rep["restored_fsm"], entry_fsm=rep.get("entry_fsm"))
        events.close(); link.close()
        log(f"RESULT {rep.get('result')} reason={rep['exit_reason']} fault={guard.fault.value} exit_confirmed={exit_rep.get('confirmed')} fsm_after={exit_rep.get('fsm_after')} -> {run_dir}")
        if exit_rep.get("confirmed") is False:
            exit_code = max(exit_code, 5)
        elif guard.tripped and exit_code == 0 and rep.get("result") != "ABORTED_BEFORE_ARM":
            exit_code = 10
    return exit_code


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(_rc if isinstance(_rc, int) else 0)        # CycloneDDS reader threads can keep the interpreter alive after main() returned
