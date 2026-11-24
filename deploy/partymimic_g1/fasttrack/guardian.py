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


def read_alive(path=GUARDIAN_ALIVE, tries=4):
    """Heartbeat reader that tolerates a concurrent rewrite (retries instead of reporting 'guardian not running')."""
    last = None
    for _ in range(tries):
        try:
            return json.loads(Path(path).read_text())
        except Exception as e:
            last = e; time.sleep(0.005)
    raise last


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
              kill: bool = True, log=print, confirm_timeout_s: float = 3.0, hold_unconfirmed_s: float = 30.0, damp_s: float = 0.1,
              profile=None, fault: str = "NONE", detected_tm: float | None = None, legacy_kill_sleep: bool = False) -> dict:
    """Kill the runner (single writer), stream damping (demo: kp=0, kd=3), SwitchToInternalCtrl (AUTO rule), verify by GetFsmId (handback.py).
    Confirmed = GetFsmId proves the OEM has control (see handback.py); same semantics as ControlSession.finish()."""
    from .profile import decide_exit_mode, get_profile
    from .handback import handback
    profile = profile or get_profile("demo_walkrun")
    tl = {"detected_tm": detected_tm, "take_over_start_tm": time.monotonic()}
    rep = {"reason": reason, "killed_pid": None, "rpc_code": None, "fsm_before": None, "fsm_after": None, "confirmed": False, "timeline": tl}
    link.enable_write()                                  # no-op when the guardian pre-built the writer at start-up (R2.1)
    if kill and pid and pid != os.getpid():
        try:
            os.kill(pid, signal.SIGKILL); rep["killed_pid"] = pid; tl["kill_tm"] = time.monotonic(); log(f"[guardian] SIGKILL runner pid {pid}")
            if legacy_kill_sleep: time.sleep(0.15)          # R2 behaviour, kept only for A/B latency measurements
            t_gone = time.monotonic() + (0.0 if legacy_kill_sleep else 0.25)                # single-writer guarantee: wait until the kernel reports it dead (R2: fixed 150 ms sleep)
            while time.monotonic() < t_gone:
                try:
                    st = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                except (FileNotFoundError, ProcessLookupError, IndexError):
                    break
                if st in ("Z", "X"): break
                time.sleep(0.001)
            tl["runner_gone_tm"] = time.monotonic()
        except ProcessLookupError:
            tl["runner_gone_tm"] = time.monotonic()
        except PermissionError:
            log(f"[guardian] cannot kill pid {pid}: permission denied (continue: it will be out-voted by 100 Hz damping)")
    def damp():
        st = link.latest(); q = st.q if st is not None else np.zeros(29)
        link.send(q, np.zeros(29), np.zeros(29), np.zeros(29), np.full(29, damp_kd))
        tl.setdefault("first_damp_frame_tm", time.monotonic())
    t_end = time.monotonic() + damp_s
    damp()
    while time.monotonic() < t_end:                    # damping stream first (kp = 0)
        damp(); time.sleep(0.01)
    code0, fsm0 = link.get_fsm(); rep["fsm_before"] = fsm0
    user_ref = profile.user_fsm_expected
    if code0 == 0 and fsm0 is not None and entry_fsm is not None and fsm0 == entry_fsm and fsm0 != user_ref:
        rep.update(confirmed=True, fsm_after=fsm0, note="robot still at the OEM entry fsm; no exit RPC needed"); return rep
    st = link.latest()
    age = None if st is None else time.monotonic() - st.arrival_mono
    mode, why = decide_exit_mode(exit_mode, None if st is None else st.quat_wxyz, age, fault)
    rep["exit_mode_used"], rep["exit_mode_why"] = mode, why
    stop = {"v": False}
    import threading
    def keep_damping():
        while not stop["v"]:
            damp(); time.sleep(0.01)
    th = threading.Thread(target=keep_damping, daemon=True); th.start()
    try:
        hb = handback(link, mode=mode, user_fsm=(fsm0 if (code0 == 0 and fsm0 is not None) else user_ref), expected_fsm=profile.expected_exit_fsm(mode),
                      confirm_timeout_s=confirm_timeout_s, hold_unconfirmed_s=hold_unconfirmed_s, log=log)
    finally:
        time.sleep(0.2); stop["v"] = True; th.join(timeout=1.0)
    tl["rpc_first_sent_tm"], tl["handback_confirmed_tm"] = hb["t_rpc_first_sent"], hb["t_confirmed"]
    rep.update(rpc_code=hb["rpc_code"], fsm_after=hb["fsm_after"], confirmed=bool(hb["confirmed"]), handback=hb)
    if not rep["confirmed"]:
        log("[guardian] !!! EXIT UNCONFIRMED: keep damping; use the remote controller damping / emergency stop. Do NOT approach.")
    return rep


def guardian_main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interface", required=True); ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--lock-path", default=SessionConfig.lock_path); ap.add_argument("--estop-file", default=SessionConfig.estop_file)
    ap.add_argument("--silence-s", type=float, default=0.30); ap.add_argument("--exit-mode", default=None, help="override profile exit rule")
    ap.add_argument("--profile", default="demo_walkrun"); ap.add_argument("--entry-fsm", type=int, default=None, help="default: fsm observed when the guardian starts")
    ap.add_argument("--alive-file", default=GUARDIAN_ALIVE); ap.add_argument("--no-prebuilt-writer", action="store_true", help="R2 behaviour (writer created at take-over time) for latency comparison")
    ap.add_argument("--legacy-kill-sleep", action="store_true", help="R2 behaviour: fixed 150 ms sleep after SIGKILL (A/B latency only)")
    ap.add_argument("--poll-s", type=float, default=0.01, help="guardian loop period (R2: 0.05)")
    ap.add_argument("--remote-grace-s", type=float, default=0.25, help="START double-click: the runner gets this long to stop by itself before the guardian takes over")
    ap.add_argument("--max-run-s", type=float, default=3600.0); ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    from .link import DdsUserCtrlLink
    from .profile import get_profile
    prof = get_profile(a.profile)
    link = DdsUserCtrlLink(a.interface, a.domain, read_only=False, rpc_timeout_s=1.0, lowcmd=prof.lowcmd)
    if not a.no_prebuilt_writer:
        link.prime()                                      # publisher + CRC path built now, silent until a take-over (R2.1)
    def log(m): print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
    if a.entry_fsm is None:
        time.sleep(1.0); c0, f0 = link.get_fsm(); a.entry_fsm = f0 if (c0 == 0 and f0 not in (0, 1)) else None
        log(f"[guardian] entry fsm observed at start: {a.entry_fsm} (the runner publishes the real entry fsm in the lock file when it takes control)")
    log(f"[guardian] watching lock {a.lock_path}, silence {a.silence_s}s, estop file {a.estop_file}")
    t0 = time.monotonic(); armed_pid = None; last_rx = link.command_topic_rx_count(); last_rx_t = time.monotonic(); seen_user = False; rx_at_arm = last_rx; user_fsm_seen = None; last_poll = 0.0
    result = {"action": None}; lock_entry = None
    poll_busy = {"v": False}; poll_res = {"fsm": None}
    from .remote import StartDoubleClick, start_pressed
    det = StartDoubleClick(); remote = {"t": None}
    def _on_state(st):
        if det.feed(st.arrival_mono, start_pressed(st.wireless)) and remote["t"] is None:
            remote["t"] = time.monotonic()
    link.add_state_listener(_on_state)
    while time.monotonic() - t0 < a.max_run_s:
        time.sleep(a.poll_s)
        try:
            _tmp = a.alive_file + ".tmp"; Path(_tmp).write_text(json.dumps({"t": time.time(), "pid": os.getpid(), "interface": a.interface})); os.replace(_tmp, a.alive_file)     # atomic: readers never see a half-written file
        except OSError:
            pass
        held, pid, info = lock_state(a.lock_path)
        if info.get("entry_fsm") is not None: lock_entry = info["entry_fsm"]
        rx = link.command_topic_rx_count(); now = time.monotonic()
        if rx != last_rx: last_rx, last_rx_t = rx, now
        trigger = None
        if os.path.exists(a.estop_file):
            trigger = "estop_file"
        elif remote["t"] is not None and held and now - remote["t"] > a.remote_grace_s and now - link.last_active_rx_tm() < 0.1:
            trigger = f"remote START double-click {now - remote['t']:.2f}s ago and the runner still sends stiff frames (its own stop did not act)"
        elif remote["t"] is not None and not held and armed_pid is None:
            remote["t"] = None            # nobody to stop
        elif held:
            if armed_pid != pid:
                armed_pid = pid; last_rx_t = now; rx_at_arm = rx; log(f"[guardian] runner pid {pid} holds the control lock (waiting for its first command)")
            if rx != rx_at_arm and not seen_user:
                seen_user = True; log("[guardian] armed: runner is commanding")
            if seen_user and now - last_poll > 0.5 and not poll_busy["v"]:
                last_poll = now; poll_busy["v"] = True          # R2.1-service: the fsm query runs in a thread: a stalled RPC (timeout 1 s) must not freeze the trigger loop / alive file
                def _poll():
                    try:
                        c1, f1 = link.get_fsm()
                        if c1 == 0 and f1 is not None and f1 != (lock_entry if lock_entry is not None else a.entry_fsm): poll_res["fsm"] = f1
                    finally:
                        poll_busy["v"] = False
                import threading; threading.Thread(target=_poll, daemon=True, name="ft-guardian-fsm").start()
            if poll_res["fsm"] is not None: user_fsm_seen = poll_res["fsm"]
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
            rep = take_over(link, reason=trigger, pid=armed_pid or pid, exit_mode=a.exit_mode or prof.exit_mode, damp_kd=prof.damp_kd, entry_fsm=lock_entry if lock_entry is not None else a.entry_fsm, log=log,
                            profile=prof, detected_tm=time.monotonic(), legacy_kill_sleep=a.legacy_kill_sleep)
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
    rep = take_over(link, reason="operator estop --direct", pid=pid if held else None, exit_mode=a.exit_mode or prof.exit_mode, damp_kd=prof.damp_kd, entry_fsm=a.entry_fsm,
                    profile=prof, fault="ESTOP", detected_tm=time.monotonic())
    print(json.dumps(rep)); link.close()
    return 0 if rep["confirmed"] else 5


def release_main(argv=None) -> int:
    """Normal (non-emergency) hand-back: ONLY if the robot is positively in user control (GetFsmId == --user-fsm) and no runner holds the lock.
    Reads first; sends nothing when the evidence is missing. Use this after a runner died with no guardian, or to leave user control cleanly."""
    ap = argparse.ArgumentParser(description=release_main.__doc__)
    ap.add_argument("--interface", required=True); ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--lock-path", default=SessionConfig.lock_path); ap.add_argument("--profile", default="demo_walkrun")
    ap.add_argument("--user-fsm", type=int, default=None, help="fsm id that means 'user control' (default: profile value, demo 1000; UNKNOWN on the target firmware)")
    ap.add_argument("--exit-mode", default="AUTO", choices=["AUTO", "WALKRUN", "PASSIVE", "LAST"])
    a = ap.parse_args(argv)
    from .link import DdsUserCtrlLink
    from .profile import get_profile
    prof = get_profile(a.profile); uf = a.user_fsm if a.user_fsm is not None else prof.user_fsm_expected
    held, pid, info = lock_state(a.lock_path)
    if held:
        print(f"REFUSED: a runner (pid {pid}) holds the control lock. Stop it first (Ctrl-C in its terminal, or `bin/ft estop IFACE`)."); return 2
    link = DdsUserCtrlLink(a.interface, a.domain, read_only=False, rpc_timeout_s=1.0, lowcmd=prof.lowcmd)
    time.sleep(0.5); code, fsm = link.get_fsm()
    if code != 0 or fsm is None:
        print(f"NOTHING SENT: GetFsmId unreadable (rpc code {code}). Use the remote controller; `bin/ft estop IFACE` is the emergency path."); link.close(); return 3
    if fsm != uf:
        print(f"NOTHING SENT: fsm={fsm} is not the user-control fsm ({uf}); the robot is not in user control (or the user fsm value differs on this firmware: --user-fsm N)."); link.close(); return 0
    print(f"robot is in user control (fsm {fsm}); handing back ({a.exit_mode})")
    rep = take_over(link, reason="operator release", pid=None, exit_mode=a.exit_mode, damp_kd=prof.damp_kd, entry_fsm=None, profile=prof, detected_tm=time.monotonic())
    print(json.dumps(rep, default=str)); link.close()
    return 0 if rep["confirmed"] else 5


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "release":
        sys.argv.pop(1); _rc = release_main(); sys.stdout.flush(); os._exit(_rc)
    if len(sys.argv) > 1 and sys.argv[1] == "guardian":
        sys.argv.pop(1); _rc = guardian_main(); sys.stdout.flush(); os._exit(_rc)
    _rc = estop_main(); sys.stdout.flush(); os._exit(_rc)
