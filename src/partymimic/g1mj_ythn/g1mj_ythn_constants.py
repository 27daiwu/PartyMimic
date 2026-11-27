"""YTHN robust-training variant of g1mj (opt-in; the P2 baseline g1mj and the P4 variants are untouched).

Same robot / motion format / action scale as g1mj. On top of P4's robust_d randomization (per-joint + shared PD gain scale,
command delay, WalkRun-standing-pose starts, wide initial pose / velocity perturbation) it adds, per the SSB real-robot falls
(P4 / P5.3 reports: the policy leaned on full ankle torque saturation and on PD targets beyond the joint limits):
  * per-joint torque-limit scale (0.8-1.0) and a whole-robot mass scale (0.92-1.08), both in xi;
  * deployment-guard semantics: PD targets clamped to the hard joint limits (margin 5 mrad), as the C++ runner does;
  * nonzero initial joint-velocity perturbation and a longer command delay range (0-25 ms).
"""
import json, os
from dataclasses import replace

from partymimic.g1mj.g1mj_constants import SPEC as BASE

_poses = os.environ.get("PM_P4_OEM_POSES", "/home/unitree/workwpace/zjy_ws/PartyMimic/results/p4/oem_poses.json")
OEM_POSES = tuple(tuple(float(x) for x in p) for p in json.load(open(_poses))) if os.path.exists(_poses) else ()

_dr = replace(
    BASE.dr,
    joint_offset={".*": 0.015},
    pd_gain_startup=((0.9, 1.1), (0.85, 1.15)),
    pd_gain_global=((0.75, 1.25), (0.7, 1.3)),
    cmd_delay=(0, 5),
    reset_rot=(0.2, 0.2, 0.3),
    reset_lin_vel=(0.8, 0.8, 0.3),
    reset_ang_vel=(0.8, 0.8, 1.0),
    reset_joint_pos=0.15,
    reset_joint_vel=0.3,
    reset_oem_prob=0.8 if OEM_POSES else 0.0,
    reset_oem_poses=OEM_POSES,
    reset_oem_frames=25,
    effort_scale=(0.8, 1.0),
    mass_scale=(0.92, 1.08),
    target_clip=0.005,
)
SPEC = replace(BASE, name="g1mj_ythn", dr=_dr)
