"""Pure-logic safety guard: joint / torque protection, state validity, sustained-fault detection.

No I/O. Every decision is deterministic and unit-tested (tests/test_guard.py).

Layers
  1. state validity   : NaN/inf, stale stream, frozen tick, IMU quaternion norm
  2. state envelope   : joint position beyond hard limit, joint speed, measured torque (sustained), pelvis tilt, gyro
  3. command filter   : NaN reject, kp/kd caps, tau_ff = 0, position clip inside the joint limits with margin,
                        target slew limit, and a PD-torque limit that is applied by shrinking the position error
                        (tau = kp*(q_t-q) + kd*(0-dq)  must stay within frac*effort_limit)
A fault is LATCHED: the guard never clears it; the supervisor decides the (damped) exit and a human restarts.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

import numpy as np


class Fault(str, enum.Enum):
    NONE = "NONE"
    COMM_TIMEOUT = "COMM_TIMEOUT"          # lowstate stream stale
    STATE_FROZEN = "STATE_FROZEN"          # tick not advancing though messages arrive
    STATE_INVALID = "STATE_INVALID"        # NaN / inf / bad quaternion / wrong size
    JOINT_LIMIT = "JOINT_LIMIT"            # measured q beyond hard limit (+tol)
    JOINT_SPEED = "JOINT_SPEED"            # measured |dq| above fraction of motor speed limit
    TORQUE_SUSTAINED = "TORQUE_SUSTAINED"  # measured tau_est above effort limit for several ticks
    TILT = "TILT"                          # pelvis tilt beyond bound
    GYRO = "GYRO"                          # |gyro| beyond bound
    CMD_INVALID = "CMD_INVALID"            # NaN / inf command from policy
    ESTOP = "ESTOP"                        # operator e-stop (key / file / signal)
    WATCHDOG = "WATCHDOG"                  # control loop stalled
    LOOP_OVERRUN = "LOOP_OVERRUN"          # repeated control period overruns
    INFERENCE = "INFERENCE"                # policy inference exception
    NO_RESPONSE = "NO_RESPONSE"            # commanded motion not observed (command path not effective)
    SWITCH_FAILED = "SWITCH_FAILED"        # SwitchToUserCtrl / precondition failure
    LOCK_LOST = "LOCK_LOST"                # exclusive control lost
    FSM_CHANGED = "FSM_CHANGED"            # official fsm left the user-control value (OEM took back / lost)
    SEGMENT_END = "SEGMENT_END"            # not a fault: normal end handled by supervisor (never latched here)


@dataclass
class GuardConfig:
    # --- state envelope
    state_stale_s: float = 0.060            # no new lowstate for this long -> COMM_TIMEOUT (3 policy periods)
    state_frozen_ticks: int = 5             # consecutive identical `tick` values with fresh arrival
    limit_fault_tol_rad: float = 0.05       # measured q may exceed the hard limit by this much before FAULT
    speed_fault_frac: float = 0.90          # of the motor speed limit
    unknown_speed_limit: float = 20.0       # rad/s stand-in where sim2real cfg has no limit (waist roll/pitch)
    torque_meas_fault_frac: float = 1.00    # tau_est/effort above this ...
    torque_meas_fault_ticks: int = 8        # ... for this many consecutive ticks (50 Hz -> 160 ms)
    max_tilt_rad: float = 1.0               # angle between body z and world z (hanging robots swing; ground robots fall)
    max_gyro_rad_s: float = 10.0
    # --- command filter
    pos_margin_rad: float = 0.02            # command clip distance inside hard limits
    target_speed_frac: float = 0.60         # of motor speed limit, slew of the *target*
    torque_cmd_frac: float = 0.85           # PD torque estimate limit as fraction of effort limit
    kp_scale_max: float = 1.0               # multiplies the yaml kp as an upper cap
    kd_min_frac: float = 1.0
    allow_tau_ff: bool = False


@dataclass
class GuardLimits:
    names: tuple
    lo: np.ndarray
    hi: np.ndarray
    vel: np.ndarray        # NaN where unknown
    effort: np.ndarray

    @classmethod
    def from_robot_cfg(cls, cfg) -> "GuardLimits":
        from sim2real.utils.strings import resolve_matching_names_values
        names = tuple(cfg.joint_names)
        def arr(d):
            idx, _, vals = resolve_matching_names_values(dict(d), list(names), preserve_order=True, strict=False)
            a = np.full(len(names), np.nan); a[idx] = vals; return a
        lim = cls(names, arr(cfg.joint_pos_lower_limit), arr(cfg.joint_pos_upper_limit),
                  arr(cfg.joint_velocity_limit), arr(cfg.joint_effort_limit))
        if np.isnan(lim.lo).any() or np.isnan(lim.hi).any() or np.isnan(lim.effort).any():
            raise ValueError("robot cfg lacks position/effort limits for some joints; refusing to build guard")
        return lim


@dataclass
class GuardEvent:
    t: float
    fault: Fault
    detail: str


class SafetyGuard:
    def __init__(self, limits: GuardLimits, cfg: GuardConfig | None = None, kp_cap: np.ndarray | None = None,
                 kd_floor: np.ndarray | None = None):
        self.lim = limits
        self.cfg = cfg or GuardConfig()
        n = len(limits.names)
        self.speed_lim = np.where(np.isnan(limits.vel), self.cfg.unknown_speed_limit, limits.vel)
        self.kp_cap = None if kp_cap is None else np.asarray(kp_cap, float) * self.cfg.kp_scale_max
        self.kd_floor = None if kd_floor is None else np.asarray(kd_floor, float) * self.cfg.kd_min_frac
        self.fault = Fault.NONE
        self.events: list[GuardEvent] = []
        self._tau_count = np.zeros(n, int)
        self._last_tick: int | None = None
        self._same_tick = 0
        self._prev_target: np.ndarray | None = None
        self.stats = {"cmd_pos_clipped": 0, "cmd_slew_clipped": 0, "cmd_torque_clipped": 0, "cmds": 0}
        self.max_ratio = {"q_margin": math.inf, "dq_over_lim": 0.0, "tau_est_over_eff": 0.0, "tilt": 0.0, "gyro": 0.0}

    # ------------------------------------------------------------------ fault latch
    def trip(self, fault: Fault, detail: str, t: float = 0.0) -> Fault:
        if self.fault is Fault.NONE:
            self.fault = fault
            self.events.append(GuardEvent(t, fault, detail))
        return self.fault

    @property
    def tripped(self) -> bool:
        return self.fault is not Fault.NONE

    # ------------------------------------------------------------------ state checks
    def check_state(self, *, q, dq, tau_est, quat_wxyz, gyro, tick: int | None, age_s: float, t: float) -> Fault:
        """Return the (latched) fault after examining one state sample. Call once per control tick."""
        c, L = self.cfg, self.lim
        if self.tripped:
            return self.fault
        if age_s > c.state_stale_s:
            return self.trip(Fault.COMM_TIMEOUT, f"lowstate age {age_s * 1e3:.0f} ms > {c.state_stale_s * 1e3:.0f} ms", t)
        if tick is not None:
            if self._last_tick is not None and tick == self._last_tick:
                self._same_tick += 1
                if self._same_tick >= c.state_frozen_ticks:
                    return self.trip(Fault.STATE_FROZEN, f"tick {tick} unchanged for {self._same_tick} control ticks", t)
            else:
                self._same_tick = 0
            self._last_tick = tick
        arrs = [np.asarray(x, float) for x in (q, dq, tau_est, quat_wxyz, gyro)]
        if any(not np.isfinite(a).all() for a in arrs) or arrs[0].shape != (len(L.names),) or arrs[1].shape != arrs[0].shape:
            return self.trip(Fault.STATE_INVALID, "non-finite or mis-sized state", t)
        q, dq, tau, quat, gyro = arrs
        qn = float(np.linalg.norm(quat))
        if not (0.9 < qn < 1.1):
            return self.trip(Fault.STATE_INVALID, f"IMU quaternion norm {qn:.3f}", t)
        quat = quat / qn
        # joint position beyond hard limit (+tol)
        under, over = L.lo - q, q - L.hi
        worst = np.maximum(under, over)
        self.max_ratio["q_margin"] = min(self.max_ratio["q_margin"], float(-worst.max()))
        if (worst > c.limit_fault_tol_rad).any():
            i = int(np.argmax(worst))
            return self.trip(Fault.JOINT_LIMIT, f"{L.names[i]} q={q[i]:.3f} outside [{L.lo[i]:.3f},{L.hi[i]:.3f}] by {worst[i]:.3f}", t)
        r = np.abs(dq) / self.speed_lim
        self.max_ratio["dq_over_lim"] = max(self.max_ratio["dq_over_lim"], float(r.max()))
        if (r > c.speed_fault_frac).any():
            i = int(np.argmax(r))
            return self.trip(Fault.JOINT_SPEED, f"{L.names[i]} |dq|={abs(dq[i]):.2f} rad/s > {c.speed_fault_frac:.2f}*{self.speed_lim[i]:.1f}", t)
        te = np.abs(tau) / L.effort
        self.max_ratio["tau_est_over_eff"] = max(self.max_ratio["tau_est_over_eff"], float(te.max()))
        hot = te > c.torque_meas_fault_frac
        self._tau_count = np.where(hot, self._tau_count + 1, 0)
        if (self._tau_count >= c.torque_meas_fault_ticks).any():
            i = int(np.argmax(self._tau_count))
            return self.trip(Fault.TORQUE_SUSTAINED, f"{L.names[i]} |tau_est|={abs(tau[i]):.1f} > effort {L.effort[i]:.0f} for {self._tau_count[i]} ticks", t)
        # tilt: angle between body z axis and world z
        w, x, y, z = quat
        zz = 1.0 - 2.0 * (x * x + y * y)          # R[2,2]
        tilt = math.acos(max(-1.0, min(1.0, zz)))
        self.max_ratio["tilt"] = max(self.max_ratio["tilt"], tilt)
        if tilt > c.max_tilt_rad:
            return self.trip(Fault.TILT, f"tilt {tilt:.2f} rad > {c.max_tilt_rad:.2f}", t)
        g = float(np.linalg.norm(gyro))
        self.max_ratio["gyro"] = max(self.max_ratio["gyro"], g)
        if g > c.max_gyro_rad_s:
            return self.trip(Fault.GYRO, f"|gyro| {g:.1f} rad/s > {c.max_gyro_rad_s:.1f}", t)
        return Fault.NONE

    # ------------------------------------------------------------------ command filter
    def reset_target(self, q_now) -> None:
        """Seed the slew limiter (call when taking control, with the measured q)."""
        self._prev_target = np.asarray(q_now, float).copy()

    def filter_command(self, q_t, dq_t, tau_ff, kp, kd, q, dq, dt: float, t: float = 0.0):
        """Return (q_t, dq_t, tau_ff, kp, kd) that are safe to send, or raise GuardReject (and latch CMD_INVALID)."""
        c, L = self.cfg, self.lim
        q_t, dq_t, tau_ff, kp, kd = (np.asarray(a, float).copy() for a in (q_t, dq_t, tau_ff, kp, kd))
        q, dq = np.asarray(q, float), np.asarray(dq, float)
        if not all(np.isfinite(a).all() for a in (q_t, dq_t, tau_ff, kp, kd)):
            self.trip(Fault.CMD_INVALID, "non-finite command", t)
            raise GuardReject("non-finite command")
        self.stats["cmds"] += 1
        if (kp < 0).any() or (kd < 0).any():
            self.trip(Fault.CMD_INVALID, "negative gain", t)
            raise GuardReject("negative gain")
        if self.kp_cap is not None:
            kp = np.minimum(kp, self.kp_cap)
        if self.kd_floor is not None and (kp > 0).any():
            # never lower damping on joints that are position-controlled
            kd = np.where(kp > 0, np.maximum(kd, self.kd_floor), kd)
        if not c.allow_tau_ff:
            tau_ff[:] = 0.0
        dq_t[:] = 0.0
        # 1) clip target inside joint limits with margin
        lo, hi = L.lo + c.pos_margin_rad, L.hi - c.pos_margin_rad
        clipped = np.clip(q_t, lo, hi)
        self.stats["cmd_pos_clipped"] += int((clipped != q_t).any()); q_t = clipped
        # 2) target slew limit
        if self._prev_target is None:
            self._prev_target = q.copy()
        step = self.speed_lim * c.target_speed_frac * dt
        sl = np.clip(q_t, self._prev_target - step, self._prev_target + step)
        self.stats["cmd_slew_clipped"] += int((sl != q_t).any()); q_t = sl
        # 3) PD torque limit by shrinking the position error (only where kp>0)
        Lt = c.torque_cmd_frac * L.effort
        a = kd * dq                                  # tau = kp*e - kd*dq  (dq_t forced 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            e_lo = np.where(kp > 0, (-Lt + a) / kp, -np.inf)
            e_hi = np.where(kp > 0, (Lt + a) / kp, np.inf)
        e = q_t - q
        e_c = np.clip(e, e_lo, e_hi)
        self.stats["cmd_torque_clipped"] += int((e_c != e).any()); q_t = q + e_c
        # torque clipping must not break the slew/limit bounds
        q_t = np.clip(q_t, lo, hi)
        self._prev_target = q_t.copy()
        return q_t, dq_t, tau_ff, kp, kd

    def estimated_torque(self, q_t, kp, kd, q, dq) -> np.ndarray:
        return kp * (np.asarray(q_t) - q) - kd * dq


class GuardReject(RuntimeError):
    pass
