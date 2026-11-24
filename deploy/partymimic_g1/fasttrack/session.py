"""ControlSession: the only object allowed to talk to the robot.

Responsibilities (the FastTrack minimum gates):
  * exclusive control    : flock on a per-robot lock file + foreign-writer detection on rt/user_lowcmd and rt/lowcmd
  * single logical writer: a mode machine {OFF, NORMAL, DAMP}; in DAMP only the streamer thread publishes
  * guard                : every command passes SafetyGuard.filter_command; every state sample passes check_state
  * comm timeout / stall : watchdog thread (state age AND control-loop heartbeat)
  * e-stop               : emergency() latches a fault and switches to DAMP from any thread / signal handler
  * reliable exit        : finish() = damp -> SwitchToInternalCtrl -> verify by GetFsmId; never blind-retries the RPC
"""
from __future__ import annotations

import fcntl, json, os, signal, sys, threading, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .guard import Fault, GuardReject, SafetyGuard
from .link import Link, LinkState
from .profile import FwProfile, get_profile, entry_check, decide_exit_mode, ATTITUDE_MAX_AGE_S
from .handback import handback
from .remote import StartDoubleClick, start_pressed


@dataclass
class SessionConfig:
    lock_path: str = "/tmp/partymimic_g1_control.lock"
    estop_file: str = "/tmp/partymimic_g1_ESTOP"
    profile: FwProfile = field(default_factory=lambda: get_profile("demo_walkrun"))
    entry_allowed: Optional[tuple] = None     # WalkRun-like fsm ids a take-over may start from (None = any except 0/1); config/site.json
    expect_user_fsm: Optional[int] = None     # override profile.user_fsm_expected (None = use profile)
    internal_exit_mode: Optional[str] = None  # override profile.exit_mode (AUTO|LAST|PASSIVE|WALKRUN)
    damp_hz: float = 100.0
    damp_before_release_s: float = 0.1        # R2: was 0.5 s of kp=0 before the exit RPC (a standing robot falls meanwhile); demo sends the RPC at once
    foreign_writer_listen_s: float = 1.0
    min_state_hz: float = 30.0
    quiescent_dq: float = 0.5                 # rad/s: required before taking control
    max_entry_tilt_rad: float = 0.5
    watchdog_s: float = 0.10                  # control loop heartbeat timeout
    confirm_timeout_s: float = 3.0
    unconfirmed_exit_hold_s: float = 30.0
    attitude_wait_s: float = 2.0              # while damping, wait this long for a FRESH attitude before choosing the exit mode
    handback_dwell_s: float = 0.3
    remote_stop_enabled: bool = True          # START double-click on the remote controller (rt/lowstate.wireless_remote)
    remote_window_s: float = 0.3
    fsm_poll_s: Optional[float] = None        # None -> profile.fsm_recheck_s

    @property
    def damp_kd(self) -> float: return self.profile.damp_kd
    @property
    def user_fsm(self) -> Optional[int]: return self.expect_user_fsm if self.expect_user_fsm is not None else self.profile.user_fsm_expected
    @property
    def exit_mode(self) -> str: return self.internal_exit_mode or self.profile.exit_mode


class ExclusiveLock:
    def __init__(self, path: str):
        self.path, self.fd = path, None

    def acquire(self) -> bool:
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = ""
            try:
                holder = os.pread(self.fd, 200, 0).decode(errors="ignore")
            except Exception:
                pass
            os.close(self.fd); self.fd = None
            self.holder = holder
            return False
        self.info = {"pid": os.getpid(), "t": time.time(), "argv": sys.argv[:6]}
        self._write()
        return True

    def _write(self):
        os.ftruncate(self.fd, 0)
        os.pwrite(self.fd, json.dumps(self.info).encode(), 0)

    def update(self, **kw):
        """Publish extra facts to whoever reads the lock (the guardian reads the entry fsm here to restore the right state)."""
        if self.fd is not None:
            self.info.update(kw); self._write()

    def release(self):
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN); os.close(self.fd)
            finally:
                self.fd = None


class ControlSession:
    OFF, NORMAL, DAMP, ACQUIRE = "OFF", "NORMAL", "DAMP", "ACQUIRE"

    def __init__(self, link: Link, guard: SafetyGuard, cfg: SessionConfig | None = None, log=print):
        self.link, self.guard, self.cfg, self.log = link, guard, cfg or SessionConfig(), log
        self.mode = self.OFF
        self._lk = threading.RLock()
        self._lock = ExclusiveLock(self.cfg.lock_path)
        self.t0 = time.monotonic()
        self._heartbeat = time.monotonic()
        self._stop = threading.Event()
        self._wd: Optional[threading.Thread] = None
        self._streamer: Optional[threading.Thread] = None
        self.entry_fsm: Optional[int] = None
        self.user_fsm_seen: Optional[int] = None
        self.last_cmd = None
        self.events: list[dict] = []
        self.on_event = None                      # optional callback(event dict): the runner streams events to logs/ the moment they happen
        self.exit_report: dict | None = None
        self.n_sent = 0
        self._last_state_seq = -1
        self.hold_kp = np.zeros(29); self.hold_kd = np.full(29, self.cfg.profile.damp_kd)   # set by the runner (policy yaml gains)
        self._acq_q = None
        self._fsm_th: Optional[threading.Thread] = None
        self.fsm_polls = {"ok": 0, "fail": 0}
        self.exit_mode_used: Optional[str] = None
        self.timeline: dict = {}                  # CLOCK_MONOTONIC stamps (comparable across processes on this host)
        self._wake = threading.Event()
        self._remote = StartDoubleClick(window_s=self.cfg.remote_window_s)
        self._remote_fired_tm: Optional[float] = None
        if self.cfg.remote_stop_enabled:
            try:
                self.link.add_state_listener(self._on_remote_sample)
            except Exception:
                pass

    def _on_remote_sample(self, st) -> None:
        """Runs in the DDS reader thread: only sets a flag (no locks, no I/O)."""
        if self._remote.feed(st.arrival_mono, start_pressed(st.wireless)) and self._remote_fired_tm is None:
            self._remote_fired_tm = time.monotonic(); self._wake.set()

    # ------------------------------------------------------------------ utilities
    def now(self) -> float:
        return time.monotonic() - self.t0

    def event(self, kind: str, **kw):
        e = {"t": round(self.now(), 4), "tm": time.monotonic(), "kind": kind, **kw}
        self.events.append(e); self.log(f"[session] {json.dumps(e, default=str)}")
        if self.on_event is not None:
            try: self.on_event(e)
            except Exception: pass

    def _damp_cmd(self, n=29):
        z = np.zeros(n)
        return z.copy() if self.last_cmd is None else self.last_cmd[0].copy(), z, z, z, np.full(n, self.cfg.damp_kd)

    # ------------------------------------------------------------------ state access
    def state(self) -> Optional[LinkState]:
        return self.link.latest()

    def check_state_once(self, st: Optional[LinkState]) -> Fault:
        """Run the guard on the newest sample (or on 'nothing')."""
        t = self.now()
        if st is None:
            return self.guard.check_state(q=np.zeros(29), dq=np.zeros(29), tau_est=np.zeros(29), quat_wxyz=[1, 0, 0, 0], gyro=[0, 0, 0],
                                          tick=None, age_s=1e9, t=t)
        f = self.guard.check_state(q=st.q, dq=st.dq, tau_est=st.tau_est, quat_wxyz=st.quat_wxyz, gyro=st.gyro, tick=st.tick,
                                   age_s=time.monotonic() - st.arrival_mono, t=t)
        if f is not Fault.NONE:
            self.emergency(f, self.guard.events[-1].detail)
        return f

    # ------------------------------------------------------------------ gates before touching the robot
    def preflight(self, listen_s: float | None = None, rate_window_s: float = 1.0) -> tuple[bool, list[dict]]:
        """Read-only checks (no publisher use, no RPC that changes state). Returns (ok, results)."""
        c = self.cfg; res = []
        def add(name, ok, detail=""):
            res.append({"check": name, "pass": bool(ok), "detail": detail})
        # 1. stream alive and fast enough
        t_end = time.monotonic() + rate_window_s; seqs = []
        first = None
        while time.monotonic() < t_end:
            st = self.link.latest()
            if st is not None:
                first = first or st; seqs.append((time.monotonic(), st.seq, st.tick))
            time.sleep(0.005)
        if not seqs:
            add("lowstate_stream", False, "no rt/lowstate sample within 1 s"); return False, res
        rate = (seqs[-1][1] - seqs[0][1]) / max(1e-6, seqs[-1][0] - seqs[0][0])
        ticks = {s[2] for s in seqs}
        add("lowstate_rate_hz", rate >= c.min_state_hz, f"{rate:.0f} Hz (min {c.min_state_hz})")
        add("lowstate_tick_advancing", len(ticks) > 5, f"{len(ticks)} distinct ticks")
        st = self.link.latest()
        # 2. guard on a live sample (does not latch: use a throw-away guard copy semantic via direct checks)
        import copy
        g = copy.deepcopy(self.guard)
        f = g.check_state(q=st.q, dq=st.dq, tau_est=st.tau_est, quat_wxyz=st.quat_wxyz, gyro=st.gyro, tick=None,
                          age_s=time.monotonic() - st.arrival_mono, t=0.0)
        add("state_inside_guard_envelope", f is Fault.NONE, g.events[-1].detail if g.events else "ok")
        L = self.guard.lim
        add("q_within_hard_limits", bool(((st.q >= L.lo) & (st.q <= L.hi)).all()), f"min margin {float(np.minimum(st.q - L.lo, L.hi - st.q).min()):.3f} rad")
        add("robot_quiescent", float(np.abs(st.dq).max()) < c.quiescent_dq, f"max |dq| {float(np.abs(st.dq).max()):.3f} rad/s")
        qn = st.quat_wxyz / np.linalg.norm(st.quat_wxyz); tilt = float(np.arccos(np.clip(1 - 2 * (qn[1] ** 2 + qn[2] ** 2), -1, 1)))
        add("tilt_ok", tilt < c.max_entry_tilt_rad, f"{tilt:.2f} rad")
        # 3. FSM (read-only RPC)
        code, fsm = self.link.get_fsm()
        add("fsm_query", code == 0 and fsm is not None, f"code={code} fsm={fsm}")
        ok_e, why = entry_check(c.profile, fsm, c.entry_allowed)
        add("fsm_entry_allowed", ok_e, f"profile={c.profile.name}: {why}")
        em = c.profile.expected_mode_machine
        add("mode_machine_matches", em is None or st.mode_machine in (0, em), f"lowstate.mode_machine={st.mode_machine} expected {em} (0 = simulation, as in the official check_mode_machine)")
        # 4. foreign writer
        n0 = getattr(self.link, "command_topic_rx_count", lambda: 0)()
        time.sleep(listen_s if listen_s is not None else c.foreign_writer_listen_s)
        n1 = getattr(self.link, "command_topic_rx_count", lambda: 0)()
        add("no_foreign_writer_on_user_lowcmd", n1 == n0, f"{n1 - n0} samples in listen window")
        # 5. exclusive lock availability (test + release)
        probe = ExclusiveLock(c.lock_path); ok = probe.acquire(); probe.release()
        add("exclusive_lock_free", ok, "" if ok else f"held by {getattr(probe, 'holder', '?')}")
        add("estop_file_absent", not Path(c.estop_file).exists(), c.estop_file)
        return all(r["pass"] for r in res), res

    # ------------------------------------------------------------------ acquire / switch
    def acquire(self, fast: bool = False) -> bool:
        """OEM -> USER handover following the official example: damping frame, SwitchToUserCtrl, confirm.
        fast=True (GO after a full READY check): the state-dependent preflight repeats with 0.3 s windows instead of 1 s."""
        c = self.cfg
        ok, res = self.preflight(listen_s=0.3, rate_window_s=0.3) if fast else self.preflight()
        self.event("preflight", ok=ok, failed=[r for r in res if not r["pass"]])
        if not ok:
            self.guard.trip(Fault.SWITCH_FAILED, "preflight failed", self.now()); return False
        if self._remote_fired_tm is not None:                     # START x2 during the preflight: stop BEFORE control is taken
            self.guard.trip(Fault.REMOTE_STOP, "remote START double-click before control was taken", self.now()); return False
        if not self._lock.acquire():
            self.guard.trip(Fault.LOCK_LOST, f"exclusive lock held: {getattr(self._lock, 'holder', '')}", self.now()); self.event("lock_denied"); return False
        self.event("lock_acquired", path=c.lock_path)
        # Re-check the foreign writer window now that we hold the lock (closes the preflight->acquire race).
        n0 = self.link.command_topic_rx_count(); time.sleep(0.3)
        if self.link.command_topic_rx_count() != n0:
            self.guard.trip(Fault.LOCK_LOST, "foreign writer appeared on rt/user_lowcmd", self.now()); self._lock.release(); return False
        code, self.entry_fsm = self.link.get_fsm()
        ok_e, why = entry_check(c.profile, self.entry_fsm, c.entry_allowed)
        if not ok_e:
            self.guard.trip(Fault.SWITCH_FAILED, f"entry fsm rejected after lock: {why}", self.now()); self._lock.release(); return False
        if self._remote_fired_tm is not None:
            self.guard.trip(Fault.REMOTE_STOP, "remote START double-click before control was taken", self.now()); self._lock.release(); return False
        self._lock.update(entry_fsm=self.entry_fsm)
        self.event("entry_checked", entry_fsm=self.entry_fsm, why=why)
        st = self.link.latest()
        self.guard.reset_target(st.q)
        self.link.enable_write()
        # Frame streamed while switching (profile.acquire_frame):
        #   hold: q = measured q, policy gains (kp,kd) -> the pose is held the moment the host starts accepting user commands
        #         (demo: the policy thread publishes stiff commands immediately after SwitchToUserCtrl)
        #   damp: kp=0, kd=profile.damp_kd (sdk2 example's pre-switch frame)
        self._acq_q = st.q.copy()
        if c.profile.acquire_frame == "hold":
            self.last_cmd = (st.q.copy(), np.zeros(29), np.zeros(29), self.hold_kp.copy(), self.hold_kd.copy())
        else:
            self.last_cmd = (st.q.copy(), np.zeros(29), np.zeros(29), np.zeros(29), np.full(29, c.damp_kd))
        self._send_raw(*self.last_cmd)
        self.mode = self.ACQUIRE    # we are the writer from now on: streamer keeps sending damping frames until confirmed
        self._start_threads()
        rc = self.link.switch_to_user()
        self.event("SwitchToUserCtrl", rpc_code=rc)
        if rc != 0:
            self.emergency(Fault.SWITCH_FAILED, f"SwitchToUserCtrl rpc code {rc}")
            return False
        # confirmation: FSM changed (weak) / equals expected user fsm (strong, only if configured)
        t_end = time.monotonic() + c.confirm_timeout_s; fsm = None
        while time.monotonic() < t_end:
            code, fsm = self.link.get_fsm()
            if code == 0 and fsm is not None and ((c.user_fsm is not None and fsm == c.user_fsm) or
                                                  (c.user_fsm is None and fsm != self.entry_fsm)):
                break
            time.sleep(0.1)
        else:
            self.emergency(Fault.SWITCH_FAILED, f"user control not confirmed: fsm entry={self.entry_fsm} now={fsm}")
            return False
        self.user_fsm_seen = fsm
        with self._lk:
            if self.mode == self.ACQUIRE:
                self.heartbeat()          # R2: the watchdog starts judging the moment NORMAL is set; the loop heartbeat is seconds old after ARM
                self.mode = self.NORMAL
        self.event("user_control_confirmed", entry_fsm=self.entry_fsm, user_fsm=fsm, strong=c.user_fsm is not None,
                   note=("fsm == profile value" if c.user_fsm is not None else "weak: fsm id changed (no official value in this profile)"))
        self._fsm_th = threading.Thread(target=self._fsm_poll, daemon=True, name="ft-fsm-poll"); self._fsm_th.start()
        return True

    # ------------------------------------------------------------------ command path
    def _send_raw(self, q, dq, tau, kp, kd):
        self.link.send(q, dq, tau, kp, kd); self.n_sent += 1
        if self.mode in (self.NORMAL, self.ACQUIRE):
            self.timeline["last_stiff_send_tm"] = time.monotonic()      # last policy/hold frame handed to DDS (NORMAL mode only)
        if self.mode == self.DAMP and "first_damp_frame_tm" not in self.timeline:
            self.timeline["first_damp_frame_tm"] = time.monotonic()

    def heartbeat(self):
        self._heartbeat = time.monotonic()

    def write(self, q_t, dq_t, tau_ff, kp, kd, dt: float) -> bool:
        """Main-loop command entry. Returns True if a command was sent in NORMAL mode."""
        with self._lk:
            if self.mode != self.NORMAL:
                return False
            if os.path.exists(self.cfg.estop_file):
                self.emergency(Fault.ESTOP, f"estop file {self.cfg.estop_file}"); return False
            st = self.link.latest()
            if st is None:
                self.emergency(Fault.COMM_TIMEOUT, "no state when sending"); return False
            try:
                cmd = self.guard.filter_command(q_t, dq_t, tau_ff, kp, kd, st.q, st.dq, dt, self.now())
            except GuardReject as e:
                self.emergency(Fault.CMD_INVALID, str(e)); return False
            self.last_cmd = tuple(np.asarray(x, float).copy() for x in cmd)
            self._send_raw(*cmd)
            return True

    # ------------------------------------------------------------------ fault handling
    def emergency(self, fault: Fault, detail: str = "") -> None:
        """Latch a fault and take over writing with damping. Safe from any thread / signal handler. Never blocks on RPC."""
        with self._lk:
            self.guard.trip(fault, detail, self.now())
            self.timeline.setdefault("emergency_tm", time.monotonic())
            if self.mode in (self.NORMAL, self.ACQUIRE):
                self.mode = self.DAMP
                try:                                  # do not wait for the 100 Hz streamer: first damping frame goes out right now
                    q, dq, tau, kp, kd = self._damp_cmd(); st = self.link.latest()
                    if st is not None and np.isfinite(st.q).all(): q = st.q
                    self._send_raw(q, dq, tau, kp, kd)
                except Exception as e:
                    self.log(f"[session] immediate damp frame failed {e!r}")
                self.event("EMERGENCY_DAMP", fault=self.guard.fault.value, trigger=fault.value, detail=detail)
            elif self.mode == self.OFF:
                self.event("fault_before_control", fault=self.guard.fault.value, detail=detail)

    def _start_threads(self):
        if self._wd is None:
            self._stop.clear()
            self._wd = threading.Thread(target=self._watch, daemon=True, name="ft-watchdog"); self._wd.start()
            self._streamer = threading.Thread(target=self._stream, daemon=True, name="ft-damp-streamer"); self._streamer.start()

    def _watch(self):
        c = self.cfg
        while not self._stop.is_set():
            self._wake.wait(0.005); self._wake.clear()
            if self._remote_fired_tm is not None and self.mode in (self.NORMAL, self.ACQUIRE) and not self.guard.tripped:
                self.timeline.setdefault("trigger_seen_tm", self._remote_fired_tm)
                self.emergency(Fault.REMOTE_STOP, "remote controller START double-click"); continue
            if self.mode != self.NORMAL:
                continue
            if time.monotonic() - self._heartbeat > c.watchdog_s:
                self.emergency(Fault.WATCHDOG, f"control loop heartbeat older than {c.watchdog_s * 1e3:.0f} ms"); continue
            st = self.link.latest()
            if st is not None and time.monotonic() - st.arrival_mono > self.guard.cfg.state_stale_s * 2:
                self.emergency(Fault.COMM_TIMEOUT, f"lowstate silent {1e3 * (time.monotonic() - st.arrival_mono):.0f} ms")
            if os.path.exists(c.estop_file):
                self.emergency(Fault.ESTOP, f"estop file {c.estop_file}")
            # another writer on the command topic: our own samples are echoed to our reader, so rx should track n_sent
            if self.n_sent > 20 and self.link.command_topic_rx_count() - self.n_sent > 15:
                self.emergency(Fault.LOCK_LOST, f"foreign writer on command topic (rx={self.link.command_topic_rx_count()} sent={self.n_sent})")

    def _fsm_poll(self):
        """Demo parity: it re-verifies GetFsmId == 1000 every second while the policy runs. Leaving the user fsm = we lost control."""
        c = self.cfg; period = c.fsm_poll_s or c.profile.fsm_recheck_s; bad = 0
        while not self._stop.wait(period):
            if self.mode != self.NORMAL:
                continue
            code, fsm = self.link.get_fsm()
            if code != 0 or fsm is None:
                self.fsm_polls["fail"] += 1; continue          # an unanswered query is not proof of lost control
            self.fsm_polls["ok"] += 1
            if fsm != self.user_fsm_seen:
                bad += 1
                if bad >= 2:
                    self.emergency(Fault.FSM_CHANGED, f"fsm {fsm} != user fsm {self.user_fsm_seen} (control was taken back or lost)")
            else:
                bad = 0

    def _stream(self):
        period = 1.0 / self.cfg.damp_hz
        nxt = time.perf_counter()
        while not self._stop.is_set():
            nxt += period
            if self.mode in (self.DAMP, self.ACQUIRE):
                with self._lk:
                    if self.mode in (self.DAMP, self.ACQUIRE):
                        try:
                            if self.mode == self.ACQUIRE and self.last_cmd is not None:
                                self._send_raw(*self.last_cmd)            # keep streaming the acquire frame until confirmed
                            else:
                                q, dq, tau, kp, kd = self._damp_cmd()
                                st = self.link.latest()
                                if st is not None and np.isfinite(st.q).all():
                                    q = st.q      # kp=0 -> q ignored by PD; keep it finite and near the state
                                self._send_raw(q, dq, tau, kp, kd)
                        except Exception as e:     # never let the streamer die silently
                            self.log(f"[session] damp stream error {e!r}")
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.perf_counter()

    # ------------------------------------------------------------------ exit
    def _remode(self):
        """Re-evaluate the exit mode from the current attitude (used before a retried exit RPC)."""
        stq = self.link.latest(); age = None if stq is None else time.monotonic() - stq.arrival_mono
        m, why = decide_exit_mode(self.cfg.exit_mode, None if stq is None else stq.quat_wxyz, age, self.guard.fault.value)
        self.exit_mode_used = m; self._why_latest = why; self.event("exit_mode_recheck", mode=m, why=why)
        return m, self.cfg.profile.expected_exit_fsm(m)

    def finish(self, reason: str = "normal") -> dict:
        """Damp, hand back to the internal controller, VERIFY. Idempotent. Returns the exit report.
        confirmed=True only when GetFsmId proves the OEM has control (see handback.py); anything else is reported as a FAILED exit."""
        c = self.cfg
        if self.exit_report is not None:
            return self.exit_report
        self.timeline["finish_start_tm"] = time.monotonic()
        rep = {"reason": reason, "fault": self.guard.fault.value, "internal_mode": c.exit_mode, "rpc_code": None,
               "fsm_after": None, "confirmed": None, "streamed_damp": False}
        if self.mode == self.OFF:
            rep["note"] = "never took control; nothing to release"; rep["confirmed"] = True
            self._cleanup(); self.exit_report = rep; return rep
        with self._lk:
            if self.mode in (self.NORMAL, self.ACQUIRE):
                self.mode = self.DAMP
        rep["streamed_damp"] = True
        time.sleep(c.damp_before_release_s)
        if self.user_fsm_seen is None:
            # user control was never confirmed: if the OEM is demonstrably still in its entry state there is nothing to hand back
            code, fsm0 = self.link.get_fsm()
            if code == 0 and fsm0 is not None and fsm0 == self.entry_fsm:
                rep.update(confirmed=True, fsm_after=fsm0, note="user control never confirmed and fsm still at entry value: no exit RPC sent")
                self.event("exit_not_needed", fsm=fsm0); time.sleep(0.2); self._cleanup(); self.exit_report = rep; return rep
        if self.user_fsm_seen is not None:
            code, fsm_now = self.link.get_fsm()
            if code == 0 and fsm_now is not None and fsm_now != self.user_fsm_seen:
                # control already left the user fsm (OEM took it back / firmware watchdog): do NOT send a second switch request
                rep.update(confirmed=True, fsm_after=fsm_now, note=f"fsm already {fsm_now} != user fsm {self.user_fsm_seen}: no exit RPC sent")
                self.event("exit_not_needed_already_internal", fsm=fsm_now); time.sleep(0.2); self._cleanup(); self.exit_report = rep; return rep
        # exit mode: needs a FRESH attitude; while damping we may wait briefly for one (comm glitches), else PASSIVE
        mode, why = None, ""
        t_end = time.monotonic() + c.attitude_wait_s
        while True:
            stq = self.link.latest()
            age = None if stq is None else time.monotonic() - stq.arrival_mono
            mode, why = decide_exit_mode(c.exit_mode, None if stq is None else stq.quat_wxyz, age, self.guard.fault.value)
            if c.exit_mode != "AUTO" or mode == "WALKRUN" or self.guard.fault.value in ("TILT", "STATE_INVALID", "STATE_FROZEN") or time.monotonic() > t_end \
                    or (age is not None and age <= ATTITUDE_MAX_AGE_S):
                break
            time.sleep(0.02)
        self.exit_mode_used = mode; rep["internal_mode"] = mode; rep["exit_mode_rule"] = c.exit_mode; rep["exit_mode_why"] = why
        self.event("exit_mode", mode=mode, why=why)
        hb = handback(self.link, mode=mode, user_fsm=self.user_fsm_seen, expected_fsm=self.cfg.profile.expected_exit_fsm(mode),
                      confirm_timeout_s=c.confirm_timeout_s, dwell_s=c.handback_dwell_s, hold_unconfirmed_s=c.unconfirmed_exit_hold_s, log=self.log,
                      remode=self._remode)
        rep["handback"] = hb
        rep["internal_mode"] = hb["mode_requested"]; self.exit_mode_used = hb["mode_requested"]
        if hb.get("mode_changed_before_retry"): rep["exit_mode_why"] = getattr(self, "_why_latest", why) + " (re-decided before the retried exit RPC)"      # may differ from the first decision if a retry re-decided it
        rep["rpc_code"] = hb["rpc_code"]; rep["fsm_after"] = hb["fsm_after"]; rep["confirmed"] = bool(hb["confirmed"])
        rep["retried_once_after_explicit_rejection"] = hb["retried_once_after_explicit_rejection"]
        self.timeline["rpc_first_sent_tm"] = hb["t_rpc_first_sent"]; self.timeline["handback_confirmed_tm"] = hb["t_confirmed"]
        self.event("SwitchToInternalCtrl", rpc_code=hb["rpc_code"], mode=mode, rule=c.exit_mode, calls=len(hb["rpc_calls"]))
        if not rep["confirmed"]:
            self.event("EXIT_UNCONFIRMED", rpc_code=hb["rpc_code"], fsm=hb["fsm_after"],
                       action=f"kept DAMPING for up to {c.unconfirmed_exit_hold_s:.0f}s; use remote damping / power cut; do NOT approach")
        # tail: a few more damping frames then stop writing (official example stops writing after the switch)
        time.sleep(0.2)
        self.timeline["last_frame_tm"] = time.monotonic()
        self._cleanup()
        self.event("released", confirmed=rep["confirmed"], fsm_after=rep["fsm_after"], via=hb.get("confirmed_via"), mode_matched=hb.get("mode_matched"))
        rep["timeline"] = dict(self.timeline)
        self.exit_report = rep
        return rep

    def _cleanup(self):
        with self._lk:
            self.mode = self.OFF
        self._stop.set()
        for t in (self._wd, self._streamer):
            if t is not None and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=1.0)
        self._lock.release()


class EstopMonitor:
    """Operator inputs: SIGINT/SIGTERM/SIGHUP, 'x' / space / ESC on a tty, or the estop file (watched by the session)."""

    def __init__(self, session: ControlSession, tty_keys: bool = True):
        self.s = session
        self.go_requested = threading.Event()
        self.quit_requested = threading.Event()
        self._old = None
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, self._on_signal)
        self._th = None
        if tty_keys and sys.stdin.isatty():
            self._th = threading.Thread(target=self._keys, daemon=True, name="ft-keys"); self._th.start()

    def _on_signal(self, signum, frame):
        self.quit_requested.set()
        self.s.emergency(Fault.ESTOP, f"signal {signum}")

    def _keys(self):
        import termios, tty, select
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd); tty.setcbreak(fd)
        except Exception:
            return
        try:
            while not self.quit_requested.is_set():
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    if ch in ("x", "X", " ", "\x1b", "q"):
                        self.quit_requested.set(); self.s.emergency(Fault.ESTOP, f"key {ch!r}")
                    elif ch in ("g", "G"):
                        self.go_requested.set()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
