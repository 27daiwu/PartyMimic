"""PartyMimic FastTrack SERVICE: one process that listens to the remote controller and manages the guardian and the runner.

  F1+F3 (chord)  : load the model + real-time admission (read-only)  -> READY or BLOCKED
  L1+A  (chord)  : (only from READY) hand the robot from the factory WalkRun to PartyMimic, run the SSB motion window, hand back to WalkRun
  START x2       : at ANY stage: abort. Before ARM: nothing was ever sent. After ARM: the runner's own START handler stops + hands back (WalkRun if the
                   attitude is normal, PASSIVE if severe/unknown) and the independent guardian is the backup.
After a clean run the service returns to IDLE and keeps listening.  After any non-clean outcome it latches HOLD until `./ack.sh`.
THIS PROCESS NEVER WRITES TO THE ROBOT: its DDS link is read-only; only the runner and the guardian (separate processes) can publish.
Logs: everything under logs/ (service.log, keys.log, events.jsonl, state.json, guardian/, runs/<time>/).
"""
from __future__ import annotations

import argparse, fcntl, json, os, signal, subprocess, sys, threading, time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty

from . import paths
paths.setup_sys_path(); paths.setup_offline_env()
from .remote import KeyMap, ChordDetector, StartDoubleClick, start_pressed

IDLE, LOADING, READY, RUNNING, HOLD = "IDLE", "LOADING", "READY", "RUNNING", "HOLD"
RC_TEXT = {0: "clean end", 1: "runner crashed", 2: "refused", 3: "no lowstate", 4: "ARM failed (control could not be taken cleanly)",
           5: "EXIT UNCONFIRMED (robot may still be in user control!)", 6: "BLOCKED by real-time admission / GO refused", 10: "fault / operator stop, hand-back confirmed"}


@dataclass
class MachineCfg:
    ready_timeout_s: float = 120.0
    load_timeout_s: float = 90.0
    run_max_s: float = 90.0
    abort_escalate_s: float = 2.5


class Machine:
    """Pure state machine (no I/O): inputs are events, outputs are action tuples. Unit-tested with a fake clock."""

    def __init__(self, cfg: MachineCfg | None = None, armed: bool = False):
        self.cfg, self.armed = cfg or MachineCfg(), armed
        self.state = IDLE; self.since = 0.0; self.hold_reason = ""; self.user_abort = False
        self.deadline = None; self.abort_deadline = None; self.last_result = None; self.escalated = False

    def _go(self, st, now, **kw):
        self.state, self.since = st, now
        for k, v in kw.items(): setattr(self, k, v)

    def chord_load(self, now, guardian_ok=True):
        if self.state != IDLE:
            return [("NOTE", f"F1+F3 ignored in state {self.state}")]
        if not guardian_ok:
            return [("NOTE", "guardian not alive: F1+F3 ignored (it is restarted automatically; try again in a few seconds)")]
        self.user_abort = False; self._go(LOADING, now, deadline=now + self.cfg.load_timeout_s)
        return [("SPAWN",)]

    def chord_go(self, now, guardian_ok=True):
        if self.state != READY:
            return [("NOTE", f"L1+A ignored in state {self.state} (needs READY: press F1+F3 first)")]
        if not self.armed:
            return [("NOTE", "service is not armed (start.sh without confirmation): L1+A ignored")]
        if not guardian_ok:
            self.user_abort = True
            return [("NOTE", "guardian not alive: refusing GO, aborting the prepared runner"), ("TERMINATE", "guardian dead at GO")]
        self._go(RUNNING, now, deadline=now + self.cfg.run_max_s, abort_deadline=None)
        return [("GO",)]

    def start_double(self, now):
        if self.state in (LOADING, READY):
            self.user_abort = True; return [("NOTE", "START x2: abort before control was taken"), ("TERMINATE", "START x2")]
        if self.state == RUNNING:
            self.user_abort = True; self.abort_deadline = now + self.cfg.abort_escalate_s
            return [("NOTE", "START x2 during RUN: the runner stops by itself (guardian is the backup); watching for the hand-back")]
        return [("NOTE", f"START x2 in {self.state}: nothing to stop")]

    def runner_status(self, phase, now):
        if self.state == LOADING and phase == "READY":
            self._go(READY, now, deadline=now + self.cfg.ready_timeout_s)
            return [("NOTE", "READY: model loaded, real-time admission passed. L1+A to go, START x2 to cancel")]
        return []

    def tick(self, now, runner_alive=True, guardian_ok=True):
        a = []
        if self.state in (LOADING, READY) and self.deadline is not None and now > self.deadline and runner_alive:
            a += [("NOTE", f"{self.state} timed out"), ("TERMINATE", "timeout")]; self.deadline = None; self.user_abort = True
        if self.state == RUNNING:
            if self.deadline is not None and now > self.deadline and runner_alive:
                a += [("ESCALATE", "runner exceeded the maximum run time")]; self.deadline = None; self.escalated = True
            if self.abort_deadline is not None and now > self.abort_deadline and runner_alive:
                a += [("ESCALATE", "runner did not finish the START x2 hand-back in time")]; self.abort_deadline = None; self.escalated = True
            if not guardian_ok and runner_alive and not getattr(self, "_g_warned", False):
                self._g_warned = True; a += [("NOTE", "CRITICAL: guardian died while the runner is in control: stopping the runner"), ("TERMINATE", "guardian dead")]
        return a

    def runner_exit(self, rc, now):
        txt = RC_TEXT.get(rc, f"exit code {rc}")
        self.last_result = {"rc": rc, "text": txt, "state_before": self.state, "user_abort": self.user_abort, "time": time.time()}
        was = self.state; self.deadline = None; self.abort_deadline = None; self._g_warned = False
        if was in (LOADING, READY):
            self._go(IDLE, now)
            return [("NOTE", f"runner ended before taking control (rc={rc}: {txt}); service back to IDLE")]
        if was == RUNNING:
            if self.escalated:
                self._go(HOLD, now, hold_reason=f"escalated (runner rc={rc}: {txt})"); self.escalated = False
                return [("NOTE", f"HOLD: {self.hold_reason}. The e-stop latch was set. Check the robot, then `./ack.sh` (clears the latch).")]
            if rc == 6:                                      # GO refused by the real-time re-check: control was never taken, nothing was sent
                self._go(IDLE, now); return [("NOTE", f"GO refused (rc=6: {txt}); nothing was sent; back to IDLE")]
            if rc == 0 or (rc == 10 and self.user_abort):
                self._go(IDLE, now); return [("NOTE", f"RUN FINISHED rc={rc}: {txt}{' (operator START x2, hand-back confirmed)' if rc == 10 else ''}; back to IDLE, listening")]
            self._go(HOLD, now, hold_reason=f"runner rc={rc}: {txt}")
            return [("NOTE", f"HOLD: {self.hold_reason}. Check the robot, read logs/, then `./ack.sh` to re-arm.")]
        return [("NOTE", f"runner exit rc={rc} in {was}")]

    def ack(self, now):
        if self.state == HOLD:
            self._go(IDLE, now, hold_reason=""); return [("NOTE", "ACK: back to IDLE")]
        return [("NOTE", "ack ignored (not in HOLD)")]


# ----------------------------------------------------------------------------------------------------------------------------------
class Service:
    def __init__(self, a):
        self.a = a; self.home = paths.PKG_ROOT
        self.logs = Path(a.logs).resolve(); (self.logs / "runs").mkdir(parents=True, exist_ok=True); (self.logs / "guardian").mkdir(exist_ok=True)
        self.logf = open(self.logs / "service.log", "a", buffering=1); self.evf = open(self.logs / "events.jsonl", "a", buffering=1)
        self.keysf = open(self.logs / "keys.log", "a", buffering=1)
        self.sim = a.sim
        self.iface = "lo" if self.sim else a.iface; self.domain = 77 if self.sim else a.domain
        self.km = KeyMap.load(a.keymap)
        self.armed = bool(a.confirm_protection) or self.sim
        self.m = Machine(MachineCfg(ready_timeout_s=a.ready_timeout_s, run_max_s=a.run_s + 60.0), armed=self.armed)
        self.q: Queue = Queue(); self.runner = None; self.runner_log = None; self.guardian = None; self.guardian_log = None
        self.state_file = self.logs / "state.json"; self.status_file = self.logs / "runner_status.json"; self.go_file = self.logs / "GO"; self.ack_file = self.logs / "ACK"
        self.last_pressed = frozenset(); self.last_lowstate = 0.0; self.runner_dir = None; self.stop = threading.Event()
        self.chord_load = ChordDetector(self.km.combos["load"]); self.chord_go = ChordDetector(self.km.combos["go"]); self.dbl = StartDoubleClick()
        self.guardian_last_restart = 0.0; self.runner_phase = None

    # ---- logging
    def log(self, msg, **kv):
        line = f"{time.strftime('%H:%M:%S')} [{self.m.state}] {msg}"; print(line, flush=True); self.logf.write(line + "\n")
        self.evf.write(json.dumps({"time": time.time(), "state": self.m.state, "msg": msg, **kv}, default=str) + "\n")

    def write_state(self):
        d = {"state": self.m.state, "since": self.m.since, "time": time.time(), "hold_reason": self.m.hold_reason, "armed": self.armed, "target": "sim" if self.sim else "real",
             "interface": self.iface, "guardian_alive": self.guardian_ok(), "lowstate_age_s": (time.monotonic() - self.last_lowstate) if self.last_lowstate else None,
             "last_result": self.m.last_result, "pressed": sorted(self.last_pressed), "pid": os.getpid()}
        tmp = self.state_file.with_suffix(".tmp"); tmp.write_text(json.dumps(d, indent=1, default=str)); os.replace(tmp, self.state_file)

    # ---- remote listener (DDS reader thread: cheap, no I/O)
    def on_state(self, st):
        self.last_lowstate = time.monotonic()
        pressed = self.km.pressed(st.wireless); t = st.arrival_mono
        if pressed != self.last_pressed:
            self.q.put(("keys", t, (pressed, bytes(st.wireless[2:4]), self.km.unmapped_bits(st.wireless)))); self.last_pressed = pressed
        if self.chord_load.feed(t, pressed): self.q.put(("load", t, None))
        if self.chord_go.feed(t, pressed): self.q.put(("go", t, None))
        if self.dbl.feed(t, start_pressed(st.wireless)): self.q.put(("start2", t, None))

    # ---- guardian management
    def guardian_ok(self):
        try:
            from .guardian import read_alive
            g = read_alive()
            return time.time() - g["t"] < 1.0 and (self.guardian is not None and self.guardian.poll() is None) and g.get("interface") == self.iface
        except Exception:
            return False

    def guardian_ok_running(self):
        """Tolerant liveness for the in-control phase: process alive and heartbeat < 3 s old (the strict 1 s freshness is enforced when ARMING)."""
        try:
            from .guardian import read_alive
            g = read_alive()
            return self.guardian is not None and self.guardian.poll() is None and time.time() - g["t"] < 3.0
        except Exception:
            return False

    def ensure_guardian(self):
        if self.guardian is not None and self.guardian.poll() is None: return
        now = time.monotonic()
        if self.guardian is not None:
            self.log(f"guardian ended (rc={self.guardian.returncode}); see logs/guardian/")
            self.guardian = None
        if now - self.guardian_last_restart < 1.0: return
        self.guardian_last_restart = now
        ts = time.strftime("%Y%m%d_%H%M%S"); lf = open(self.logs / "guardian" / f"guardian_{ts}.log", "w", buffering=1)
        cmd = [self.py, "-B", "-m", "fasttrack.guardian", "guardian", "--interface", self.iface, "--domain", str(self.domain),
               "--report", str(self.logs / "guardian" / f"guardian_{ts}.json"), "--max-run-s", "604800"]
        if self.a.entry_fsm is not None: cmd += ["--entry-fsm", str(self.a.entry_fsm)]
        self.guardian = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=self.home, start_new_session=True)
        self.guardian_log = lf; self.log(f"guardian started pid {self.guardian.pid} -> logs/guardian/guardian_{ts}.log")

    # ---- runner
    @property
    def py(self): return sys.executable

    def runner_cmd(self, run_dir):
        a = self.a
        cmd = [self.py, "-B", "-m", "fasttrack.runner", "--target", "sim" if self.sim else "real", "--interface", self.iface, "--domain", str(self.domain),
               "--run-s", str(a.run_s), "--ref-mode", a.ref_mode, "--authority-max", str(a.authority_max), "--run-dir", str(run_dir),
               "--wait-go", str(self.go_file), "--status-file", str(self.status_file), "--ready-timeout-s", str(a.ready_timeout_s)]
        if a.entry_fsm is not None: cmd += ["--entry-fsm", str(a.entry_fsm)]
        if not self.sim: cmd += ["--i-confirm-protection"]
        return cmd

    def spawn_runner(self):
        ts = time.strftime("%Y%m%d_%H%M%S"); self.runner_dir = self.logs / "runs" / ts; self.runner_dir.mkdir(parents=True, exist_ok=True)
        for f in (self.go_file, self.status_file):
            try: f.unlink()
            except FileNotFoundError: pass
        self.runner_log = open(self.runner_dir / "runner.out", "w", buffering=1); self.runner_phase = None
        self.runner = subprocess.Popen(self.runner_cmd(self.runner_dir), stdout=self.runner_log, stderr=subprocess.STDOUT, cwd=self.home, start_new_session=True)
        try:
            latest = self.logs / "latest"
            if latest.is_symlink() or latest.exists(): latest.unlink()
            latest.symlink_to(self.runner_dir)
        except OSError: pass
        self.log(f"runner started pid {self.runner.pid} -> logs/runs/{ts}/")

    def term(self, p, why):
        if p is not None and p.poll() is None:
            try: os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            self.log(f"SIGTERM -> pid {p.pid} ({why})")

    # ---- actions
    def do(self, actions):
        for act in actions:
            k = act[0]
            if k == "NOTE": self.log(act[1])
            elif k == "SPAWN": self.spawn_runner()
            elif k == "GO":
                self.go_file.write_text(str(time.time())); self.log("GO issued (L1+A): the runner now takes control (ARM -> take-over -> motion window -> hand-back)")
            elif k == "TERMINATE": self.term(self.runner, act[1])
            elif k == "ESCALATE":
                self.log(f"ESCALATE: {act[1]} -> e-stop latch + SIGTERM to the runner (guardian backup stays active)", level="CRITICAL")
                try: Path("/tmp/partymimic_g1_ESTOP").write_text(f"service escalation {time.time()}\n")
                except OSError: pass
                self.term(self.runner, act[1])

    def report_summary(self):
        """One line per finished run: result / exit reason / fsm restore (details: logs/runs/<time>/run_report.json)."""
        try:
            res = json.loads((self.runner_dir / "run_report.json").read_text()); ex = res.get("exit") or {}
            self.log(f"run result={res.get('result')} reason={res.get('exit_reason')} fault={res.get('fault')} hand-back confirmed={ex.get('confirmed')} mode={ex.get('internal_mode')} "
                     f"entry_fsm={res.get('entry_fsm')} restored_fsm={res.get('restored_fsm')}")
            bad = [i for i in (res.get("admission") or {}).get("items", []) if not i["pass"]]
            for i in bad: self.log(f"  BLOCKED item: {i['item']}: {i['detail']}")
        except Exception:
            pass

    def tick_children(self):
        if self.runner is not None:
            st = None
            try: st = json.loads(self.status_file.read_text())
            except Exception: pass
            if st:
                self.runner_phase = st.get("phase"); self.do(self.m.runner_status(self.runner_phase, time.monotonic()))
            rc = self.runner.poll()
            if rc is None and self.runner_dir is not None and (self.runner_dir / "run_report.json").exists():      # finished but the process lingers (DDS threads)
                age = time.time() - (self.runner_dir / "run_report.json").stat().st_mtime
                if age > 8.0:
                    self.log(f"runner wrote its report {age:.0f} s ago but did not exit: SIGKILL (its result is in run_report.json)")
                    try: os.killpg(self.runner.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    try: rc = self.runner.wait(timeout=3)
                    except subprocess.TimeoutExpired: rc = None
                    try:
                        res = json.loads((self.runner_dir / "run_report.json").read_text()); ex = (res.get("exit") or {})
                        rc = 5 if ex.get("confirmed") is False else (10 if res.get("fault") not in (None, "NONE") and res.get("result") != "ABORTED_BEFORE_ARM" else 0)
                    except Exception: rc = rc if rc is not None else 1
            if rc is not None:
                self.runner = None
                if self.runner_log: self.runner_log.close()
                self.report_summary()
                self.do(self.m.runner_exit(rc, time.monotonic()))
                try: self.go_file.unlink()
                except FileNotFoundError: pass

    def handle_event(self, ev):
        kind, t, payload = ev; now = time.monotonic(); gok = self.guardian_ok()
        if kind == "keys":
            pressed, raw23, unmapped = payload
            self.keysf.write(f"{time.strftime('%H:%M:%S')} {time.time():.3f} byte2=0x{raw23[0]:02x} byte3=0x{raw23[1]:02x} keys={'+'.join(sorted(pressed)) or '(none)'}"
                             f"{' UNMAPPED_BITS=' + str(unmapped) if unmapped else ''}\n")
        elif kind == "load": self.log("chord F1+F3 detected"); self.do(self.m.chord_load(now, gok))
        elif kind == "go": self.log("chord L1+A detected"); self.do(self.m.chord_go(now, gok))
        elif kind == "start2": self.log("START double-click detected"); self.do(self.m.start_double(now))

    def run(self):
        lockf = open(self.logs / "service.lock", "w")
        try: fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError: print("another service instance holds logs/service.lock"); return 2
        (self.logs / "service.pid").write_text(str(os.getpid()))
        for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, lambda *_: self.stop.set())
        self.log(f"service starting: target={'sim' if self.sim else 'real'} iface={self.iface} domain={self.domain} entry_fsm={self.a.entry_fsm or 'site.json'} armed={self.armed} "
                 f"keymap={self.km.source[:50]} run_s={self.a.run_s} ref={self.a.ref_mode} authority={self.a.authority_max}")
        if not self.armed: self.log("NOT ARMED: F1+F3 runs the load + checks and reports READY/BLOCKED; L1+A does nothing.")
        from .link import DdsUserCtrlLink
        link = DdsUserCtrlLink(self.iface, self.domain, read_only=True)
        link.add_state_listener(self.on_state)
        self.ensure_guardian(); t_state = 0.0; t_g = 0.0
        try:
            while not self.stop.is_set():
                try: self.handle_event(self.q.get(timeout=0.02))
                except Empty: pass
                now = time.monotonic()
                self.tick_children()
                # the guardian ends by itself once the runner has handed control back ("normal end"); that is NOT a failure while the runner is in/after EXIT
                self.do(self.m.tick(now, runner_alive=self.runner is not None, guardian_ok=self.guardian_ok_running() or self.runner_phase == "EXIT"))
                if self.ack_file.exists():
                    try: self.ack_file.unlink()
                    except OSError: pass
                    self.do(self.m.ack(now))
                if now - t_g > 0.5: t_g = now; self.ensure_guardian()
                if now - t_state > 0.5: t_state = now; self.write_state()
        finally:
            self.log("service stopping")
            if self.runner is not None and self.runner.poll() is None:
                self.term(self.runner, "service stop")
                try: self.runner.wait(timeout=25)
                except subprocess.TimeoutExpired: self.log("runner did not exit within 25 s after SIGTERM (guardian still active)", level="CRITICAL")
            if self.guardian is not None and self.m.state in (IDLE, HOLD, READY, LOADING): self.term(self.guardian, "service stop")
            link.close(); self.write_state()
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default=None); ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--entry-fsm", type=int, default=None, help="only this fsm id may be the WalkRun entry (default: config/site.json entry_fsm_allowed)")
    ap.add_argument("--confirm-protection", action="store_true", help="HUMAN statement (start.sh asks): fall protection in place, remote e-stop in hand, >= 2 people (arms L1+A)")
    ap.add_argument("--sim", action="store_true", help="TEST ONLY: simulated robot on loopback domain 77")
    ap.add_argument("--logs", default=str(paths.LOGS)); ap.add_argument("--keymap", default=str(paths.CONFIG / "keymap.json"))
    ap.add_argument("--run-s", type=float, default=1.0); ap.add_argument("--ref-mode", choices=["running", "paused"], default="running")
    ap.add_argument("--authority-max", type=float, default=0.3); ap.add_argument("--ready-timeout-s", type=float, default=120.0)
    a = ap.parse_args(argv)
    if not a.sim and not a.iface: ap.error("--iface IFACE is required for the real target")
    sys.exit(Service(a).run())


if __name__ == "__main__":
    main()
