"""Firmware / protocol profile of the official g1_mimic_demo route (unitree-app-templates/g1_mimic_demo@7b3edef) + exit-mode rules.

  entry   : a take-over may only start from a WalkRun-like fsm; 0 (ZeroTorque) and 1 (Damp/PASSIVE) are always refused
  switch  : GetFsmId must become 1000 after SwitchToUserCtrl (strong confirmation; anything else = SWITCH_FAILED)
  stream  : rt/user_lowcmd, motor.mode=1, mode_machine=5, CRC
  exit    : damping frames (kp=0, kd=3) + SwitchToInternalCtrl(WALKRUN, or PASSIVE if |roll|>1.0 or |pitch|>0.8 / attitude unknown)
Nothing here is verified against the target firmware (`verified_on_firmware=False`); the first real run records the observed values in logs/.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass(frozen=True)
class LowCmdFormat:
    topic: str = "rt/user_lowcmd"
    set_motor_mode: bool = True        # motor_cmd[i].mode = motor_mode for the 29 joints
    motor_mode: int = 1
    mode_machine: Optional[int] = 5    # 5 = G1 29dof (demo main.cpp)
    set_crc: bool = True
    mode_pr: int = 0


@dataclass(frozen=True)
class FwProfile:
    name: str = "demo_walkrun"
    entry_forbidden: tuple = (0, 1)           # fsm ids a take-over never starts from
    user_fsm_expected: Optional[int] = 1000   # strong confirmation value
    fsm_recheck_s: float = 1.0
    lowcmd: LowCmdFormat = field(default_factory=LowCmdFormat)
    acquire_frame: str = "hold"               # frame streamed while switching: measured q + policy gains (as the demo)
    exit_mode: str = "AUTO"                   # AUTO | LAST | PASSIVE | WALKRUN
    damp_kd: float = 3.0
    expected_mode_machine: Optional[int] = 5
    verified_on_firmware: bool = False
    # fsm id the OEM should report after SwitchToInternalCtrl(mode) (SDK Damp()=1, Start()=500). Not verified on the target firmware: a mismatch is
    # reported (mode_matched=False), it does not by itself make the hand-back unconfirmed (handback.py).
    exit_fsm_expected: tuple = (("PASSIVE", 1), ("WALKRUN", 500))

    def expected_exit_fsm(self, mode: str) -> Optional[int]:
        return dict(self.exit_fsm_expected).get(mode)

    def to_json(self) -> dict:
        return asdict(self)


PROFILES = {"demo_walkrun": FwProfile()}


def get_profile(name: str = "demo_walkrun") -> FwProfile:
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; known: {sorted(PROFILES)}")
    return PROFILES[name]


def roll_pitch(quat_wxyz) -> tuple[float, float]:
    w, x, y, z = (float(v) for v in quat_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


SEVERE_FAULTS = ("TILT",)                                  # the guard itself says the posture is bad -> PASSIVE regardless of the latest sample
UNTRUSTED_STATE_FAULTS = ("STATE_INVALID", "STATE_FROZEN")  # attitude cannot be trusted -> PASSIVE (unknown = safest)
ATTITUDE_MAX_AGE_S = 0.1                                    # attitude older than this is "unknown"


def decide_exit_mode(rule: str, quat_wxyz=None, quat_age_s: Optional[float] = None, fault: str = "NONE") -> tuple[str, str]:
    """(mode, why).  Normal ends and recoverable faults -> WALKRUN; severe or unknown posture -> PASSIVE.
    rule != AUTO is an explicit operator override (LAST|PASSIVE|WALKRUN) and wins."""
    if rule != "AUTO":
        return rule, f"explicit rule {rule}"
    if fault in SEVERE_FAULTS:
        return "PASSIVE", f"fault {fault} = severe posture"
    if fault in UNTRUSTED_STATE_FAULTS:
        return "PASSIVE", f"fault {fault}: attitude not trustworthy"
    if quat_wxyz is None or quat_age_s is None or quat_age_s > ATTITUDE_MAX_AGE_S:
        return "PASSIVE", f"attitude unknown/stale (age={quat_age_s})"
    r, p = roll_pitch(quat_wxyz)
    if abs(r) > 1.0 or abs(p) > 0.8:
        return "PASSIVE", f"severe attitude roll={r:.2f} pitch={p:.2f} (demo rule |roll|>1.0 or |pitch|>0.8)"
    return "WALKRUN", f"attitude ok roll={r:.2f} pitch={p:.2f}, fault={fault}"


def entry_check(profile: FwProfile, fsm: Optional[int], allowed=None) -> tuple[bool, str]:
    """Entry rule: fsm must be readable, never in profile.entry_forbidden (0/1), and (if given) one of the allowed WalkRun-like ids."""
    if fsm is None:
        return False, "fsm unknown"
    if fsm in profile.entry_forbidden:
        return False, f"fsm {fsm} is forbidden as entry {profile.entry_forbidden} (ZeroTorque/PASSIVE: put the robot in WalkRun first)"
    if allowed is not None and fsm not in allowed:
        return False, f"fsm {fsm} is not an allowed WalkRun-like entry {sorted(allowed)} (config/site.json entry_fsm_allowed / --entry-fsm)"
    return True, f"fsm {fsm} accepted (allowed {sorted(allowed) if allowed is not None else 'any'})"
