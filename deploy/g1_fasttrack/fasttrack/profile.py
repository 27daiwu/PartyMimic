"""Firmware / protocol profiles. Replaces the R0 hard-coded 'entry fsm == 1, user fsm == 6000 placeholder'.

Two OFFICIAL sources disagree about the entry state, so the entry rule is a profile, not a constant:

  demo_walkrun   unitree-app-templates/g1_mimic_demo (commit 7b3edef, 2026-06-23, UniStore app, newer):
                   State_Mimic: `GetFsmId != 1 -> SwitchToUserCtrl()`, else error  (comment: "only SwitchToUserCtrl in walkrun mode")
                   verifies GetFsmId == 1000 after 0.1 s (and re-verifies every second)
                   publishes rt/user_lowcmd with motor.mode=1, mode_machine=5, CRC, ~50 Hz (policy rate)
                   exit (State_Exit): damping frames (kp=0, kd=3) + SwitchToInternalCtrl(WALKRUN, or PASSIVE if |roll|>1.0 or |pitch|>0.8)
  sdk2_passive   unitree_sdk2 example/g1/high_level/g1_userctrl_dds_example.cpp (2026-03-02, older):
                   requires GetFsmId == 1 (PASSIVE), pre-sends a damping frame, SwitchToUserCtrl(), no mode/mode_machine/CRC set,
                   exit SwitchToInternalCtrl(LAST), no confirmation of any step.

Nothing here is verified against the target firmware (`verified_on_firmware=False` until a human records it); the real entry
fsm number is NEVER assumed: the read-only diagnostics record the observed value and the runner requires the operator to
declare the same value (--entry-fsm N).
"""
from __future__ import annotations

import json, math
from dataclasses import dataclass, asdict, field, replace
from typing import Optional


@dataclass(frozen=True)
class LowCmdFormat:
    topic: str = "rt/user_lowcmd"
    set_motor_mode: bool = True        # motor_cmd[i].mode = motor_mode for the 29 joints
    motor_mode: int = 1
    mode_machine: Optional[int] = 5    # 5 = G1 29dof (demo main.cpp); None = copy the value reported in rt/lowstate
    set_crc: bool = True
    mode_pr: int = 0                   # not set by the demo (default 0 = PR)


@dataclass(frozen=True)
class FwProfile:
    name: str
    source: str
    entry_forbidden: tuple = (0, 1)           # fsm ids the runner refuses to switch away from
    entry_allowed: Optional[tuple] = None     # None: any not forbidden (operator must still declare the observed value)
    user_fsm_expected: Optional[int] = 1000   # strong confirmation value; None = weak (fsm changed)
    fsm_recheck_s: float = 1.0
    lowcmd: LowCmdFormat = field(default_factory=LowCmdFormat)
    acquire_frame: str = "hold"               # hold | damp | none   (frame streamed before/while switching)
    exit_mode: str = "AUTO"                   # AUTO | LAST | PASSIVE | WALKRUN
    damp_kd: float = 3.0
    expected_mode_machine: Optional[int] = 5
    verified_on_firmware: bool = False
    notes: str = ""

    def to_json(self) -> dict:
        return asdict(self)


PROFILES = {
    "demo_walkrun": FwProfile(
        name="demo_walkrun", source="unitree-app-templates/g1_mimic_demo@7b3edef (State_Mimic.cpp, State_Exit.h, g1_pub.h, main.cpp)",
        entry_forbidden=(0, 1), user_fsm_expected=1000, acquire_frame="hold", exit_mode="AUTO", damp_kd=3.0,
        notes="entry != PASSIVE(1); we additionally forbid 0 (ZeroTorque: unsupported robot). User fsm 1000 is the demo's check."),
    "sdk2_passive": FwProfile(
        name="sdk2_passive", source="unitree_sdk2@b7fec60 example/g1/high_level/g1_userctrl_dds_example.cpp",
        entry_forbidden=(), entry_allowed=(1,), user_fsm_expected=None, acquire_frame="damp", exit_mode="LAST", damp_kd=1.5,
        lowcmd=LowCmdFormat(set_motor_mode=False, mode_machine=None, set_crc=False), expected_mode_machine=None,
        notes="LEGACY (R0 default). Contradicts the demo on the entry state; fields mode/mode_machine/CRC not set."),
}


def get_profile(name: str) -> FwProfile:
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


def resolve_exit_mode(mode: str, quat_wxyz=None) -> str:
    """AUTO = the demo's State_Exit rule: PASSIVE if |roll|>1.0 or |pitch|>0.8 (or attitude unknown), else WALKRUN."""
    if mode != "AUTO":
        return mode
    if quat_wxyz is None:
        return "PASSIVE"
    r, p = roll_pitch(quat_wxyz)
    return "PASSIVE" if (abs(r) > 1.0 or abs(p) > 0.8) else "WALKRUN"


def entry_check(profile: FwProfile, fsm: Optional[int], declared: Optional[int]) -> tuple[bool, str]:
    if fsm is None:
        return False, "fsm unknown"
    if fsm in profile.entry_forbidden:
        return False, f"fsm {fsm} is forbidden as entry by profile {profile.name} {profile.entry_forbidden}"
    if profile.entry_allowed is not None and fsm not in profile.entry_allowed:
        return False, f"fsm {fsm} not in profile.entry_allowed {profile.entry_allowed}"
    if declared is not None and fsm != declared:
        return False, f"observed fsm {fsm} != operator-declared --entry-fsm {declared}"
    return True, f"fsm {fsm} accepted (declared={declared})"
