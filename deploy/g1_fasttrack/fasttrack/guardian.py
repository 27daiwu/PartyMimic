"""Out-of-process guardian + operator e-stop.

Why: the in-process watchdog dies with the process (kill -9, SIGSTOP, segfault, OOM, Python hang inside inference).
The guardian is a SEPARATE process that:
  * stays read-only until the control lock is held by a runner (pid in the lock file),
  * takes over only if (a) the runner pid is dead while the OEM is still in user control, (b) the runner is alive but rt/user_lowcmd
    has been silent for --silence-s (hung / stopped), (c) the e-stop file appears, (d) it is invoked as `estop --direct`.
  * take-over = kill the runner (SIGKILL, so there is exactly one writer), stream damping (kp=0, kd=1.5), SwitchToInternalCtrl,
    verify with GetFsmId, never retry timeouts blindly, report UNCONFIRMED loudly.
It never writes while a healthy runner owns the lock.
"""
from __future__ import annotations

import argparse, fcntl, json, os, signal, sys, time
from pathlib import Path

import numpy as np

from . import paths
paths.setup_sys_path(); paths.setup_offline_env()
from .session import SessionConfig


GUARDIAN_ALIVE = "/tmp/partymimic_g1_guardian.alive"


def lock_state(path: str):
    """(held_by_live_process, pid, info). Uses a non-blocking flock probe: authoritative, no stale-pid ambiguity."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    except OSError:
        return False, None, {}
    try:
        info = {}
        try:
            raw = os.pread(fd, 400, 0).decode(errors="ignore"); info = json.loads(raw) if raw.strip() else {}
        except Exception:
            pass
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False, info.get("pid"), info
        except OSError:
            return True, info.get("pid"), info
    finally:
        os.close(fd)


def take_over(link, *, reason: str, pid: int | None, exit_mode: str = "AUTO", damp_kd: float = 3.0, entry_fsm: int | None = None,
              kill: bool = True, log=print, confirm_timeout_s: float = 3.0, hold_unconfirmed_s: float = 30.0) -> dict:
    """Kill the runner (single writer), stream damping (demo: kp=0, kd=3), SwitchToInternalCtrl(AUTO rule), verify by GetFsmId.
    Confirmed = RPC code 0 and the fsm moved away from the value seen at take-over (the user-control fsm)."""
    from .profile import resolve_exit_mode
    rep = {"reason": reason, "killed_pid": None, "rpc_code": None, "fsm_before": None, "fsm_after": None, "confirmed": False}
    if kill and pid and pid != os.getpid():
        try:
            os.kill(pid, signal.SIGKILL); rep["killed_pid"] = pid; log(f"[guardian] SIGKILL runner pid {pid}")
            time.sleep(0.15)
        except ProcessLookupError:
            pass
        except PermissionError:
            log(f"[guardian] cannot kill pid {pid}: permission denied (continue: it will be out-voted by 100 Hz damping)")
    link.enable_write()
    def damp():
        st = link.latest(); q = st.q if st is not None else np.zeros(29)
        link.send(q, np.zeros(29), np.zeros(29), np.zeros(29), np.full(29, damp_kd))
    t_end = time.monotonic() + 0.5
    while time.monotonic() < t_end:                    # damping stream first (kp = 0)
        damp(); time.sleep(0.01)
    code0, fsm0 = link.get_fsm(); rep["fsm_before"] = fsm0
    st = link.latest()
    mode = resolve_exit_mode(exit_mode, None if st is None else st.quat_wxyz); rep["exit_mode_used"] = mode
    if code0 == 0 and fsm0 is not None and entry_fsm is not None and fsm0 == entry_fsm and rep["killed_pid"] is None:
        rep.update(confirmed=True, fsm_after=fsm0, note="robot already at entry fsm; no exit RPC"); return rep
    rc = link.switch_to_internal(mode); rep["rpc_code"] = rc
    log(f"[guardian] SwitchToInternalCtrl({mode}) rpc_code={rc} (fsm before={fsm0})")
    t_end = time.monotonic() + confirm_timeout_s; fsm = None
    while time.monotonic() < t_end:
        damp()
        code, fsm = link.get_fsm()
        if rc == 0 and code == 0 and fsm is not None and (fsm0 is None or fsm != fsm0):
            rep["confirmed"] = True; break
        time.sleep(0.1)
    rep["fsm_after"] = fsm
    if not rep["confirmed"]:
        log("[guardian] !!! EXIT UNCONFIRMED: keep damping; use the remote controller damping / emergency stop. Do NOT approach.")
        t_end = time.monotonic() + hold_unconfirmed_s
        while time.monotonic() < t_end:
            damp(); time.sleep(0.02)
    return rep


def guardian_main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interface", required=True); ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--lock-path", default=SessionConfig.lock_path); ap.add_argument("--estop-file", default=SessionConfig.estop_file)
    ap.add_argument("--silence-s", type=float, default=0.30); ap.add_argument("--exit-mode", default=None, help="override profile exit rule")
    ap.add_argument("--profile", default="demo_walkrun"); ap.add_argument("--entry-fsm", type=int, default=None, help="default: fsm observed when the guardian starts")
    ap.add_argument("--alive-file", default=GUARDIAN_ALIVE); ap.add_argument("--max-run-s", type=float, default=3600.0); ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    from .link import DdsUserCtrlLink
    from .profile import get_profile
    prof = get_profile(a.profile)
    link = DdsUserCtrlLink(a.interface, a.domain, read_only=False, rpc_timeout_s=1.0, lowcmd=prof.lowcmd)
    def log(m): print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
    if a.entry_fsm is None:
        time.sleep(1.0); c0, f0 = link.get_fsm(); a.entry_fsm = f0 if c0 == 0 else None
        log(f"[guardian] entry fsm observed at start: {a.entry_fsm}")
    log(f"[guardian] watching lock {a.lock_path}, silence {a.silence_s}s, estop file {a.estop_file}")
    t0 = time.monotonic(); armed_pid = None; last_rx = link.command_topic_rx_count(); last_rx_t = time.monotonic(); seen_user = False; rx_at_arm = last_rx; user_fsm_seen = None; last_poll = 0.0
    result = {"action": None}
    while time.monotonic() - t0 < a.max_run_s:
        time.sleep(0.05)
        try:
            Path(a.alive_file).write_text(json.dumps({"t": time.time(), "pid": os.getpid(), "interface": a.interface}))
        except OSError:
            pass
        held, pid, info = lock_state(a.lock_path)
        rx = link.command_topic_rx_count(); now = time.monotonic()
        if rx != last_rx: last_rx, last_rx_t = rx, now
        trigger = None
        if os.path.exists(a.estop_file):
            trigger = "estop_file"
        elif held:
            if armed_pid != pid:
                armed_pid = pid; last_rx_t = now; rx_at_arm = rx; log(f"[guardian] runner pid {pid} holds the control lock (waiting for its first command)")
            if rx != rx_at_arm and not seen_user:
                seen_user = True; log("[guardian] armed: runner is commanding")
            if seen_user and now - last_poll > 0.5:
                last_poll = now; c1, f1 = link.get_fsm()
                if c1 == 0 and f1 is not None and f1 != a.entry_fsm: user_fsm_seen = f1
            if seen_user and now - last_rx_t > a.silence_s:
                trigger = f"runner silent {now - last_rx_t:.2f}s on rt/user_lowcmd (hung/stopped)"
        elif armed_pid is not None:
            seen_user = False
            # lock released: normal exit -> OEM fsm back at entry; crash -> still in user control
            code, fsm = link.get_fsm()
            if code == 0 and fsm is not None and fsm != user_fsm_seen and user_fsm_seen is not None:
                log(f"[guardian] runner released the lock and fsm moved {user_fsm_seen} -> {fsm}: normal end, guardian done")
                result = {"action": "none", "note": "normal end"}; break
            if code == 0 and fsm is not None:
                trigger = f"runner pid {armed_pid} gone but OEM fsm={fsm} is still the user-control value (crash)"
            elif now - last_rx_t > 1.0:
                trigger = f"runner gone and fsm unreadable (rpc code {code})"
        if trigger:
            log(f"[guardian] TAKE-OVER: {trigger}")
            rep = take_over(link, reason=trigger, pid=armed_pid or pid, exit_mode=a.exit_mode or prof.exit_mode, damp_kd=prof.damp_kd, entry_fsm=a.entry_fsm, log=log)
            result = {"action": "take_over", **rep}
            log(f"[guardian] done: {json.dumps(rep)}")
            break
    try: os.remove(a.alive_file)
    except OSError: pass
    if a.report:
        Path(a.report).write_text(json.dumps(result, indent=1))
    link.close()
    return 0 if result.get("action") in ("none", None) or result.get("confirmed") else 5


def estop_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Operator e-stop: latch the estop file (runner + guardian react) and, with --direct, "
                                             "take over right now from this terminal (kills the runner, damps, hands back to the OEM).")
    ap.add_argument("--interface", default=None); ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--estop-file", default=SessionConfig.estop_file); ap.add_argument("--lock-path", default=SessionConfig.lock_path)
    ap.add_argument("--direct", action="store_true"); ap.add_argument("--exit-mode", default=None); ap.add_argument("--profile", default="demo_walkrun"); ap.add_argument("--entry-fsm", type=int, default=None)
    ap.add_argument("--clear", action="store_true", help="remove the e-stop latch (only after the robot is safe)")
    a = ap.parse_args(argv)
    if a.clear:
        try: os.remove(a.estop_file); print("e-stop latch cleared")
        except FileNotFoundError: print("no latch")
        return 0
    Path(a.estop_file).write_text(f"estop by {os.getpid()} at {time.time()}\n"); print(f"e-stop latched: {a.estop_file}")
    if not a.direct:
        return 0
    if not a.interface:
        print("--direct needs --interface"); return 2
    from .link import DdsUserCtrlLink
    from .profile import get_profile
    prof = get_profile(a.profile)
    link = DdsUserCtrlLink(a.interface, a.domain, read_only=False, lowcmd=prof.lowcmd)
    time.sleep(0.5)
    held, pid, _ = lock_state(a.lock_path)
    rep = take_over(link, reason="operator estop --direct", pid=pid if held else None, exit_mode=a.exit_mode or prof.exit_mode, damp_kd=prof.damp_kd, entry_fsm=a.entry_fsm)
    print(json.dumps(rep)); link.close()
    return 0 if rep["confirmed"] else 5


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "guardian":
        sys.argv.pop(1); sys.exit(guardian_main())
    sys.exit(estop_main())
