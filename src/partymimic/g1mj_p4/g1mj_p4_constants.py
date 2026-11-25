"""P4 robustness variant of g1mj: identical robot / motion format / action scale; only the domain randomization differs.
Adds (all graph-compatible): per-env PD gain scale, per-env command delay (0-15 ms), WalkRun-standing-pose episode starts, wider start
perturbations (pushes/lean/pose mismatch). The P2 baseline (g1mj) is untouched."""
import json, os
from dataclasses import replace

from partymimic.dr import DomainRandCfg
from partymimic.g1mj.g1mj_constants import SPEC as BASE

_poses = os.environ.get("PM_P4_OEM_POSES", "/home/unitree/workwpace/zjy_ws/PartyMimic/results/p4/oem_poses.json")
OEM_POSES = tuple(tuple(float(x) for x in p) for p in json.load(open(_poses))) if os.path.exists(_poses) else ()

_dr = replace(
    BASE.dr,
    pd_gain_startup=((0.8, 1.2), (0.7, 1.3)),
    cmd_delay=(0, 3),
    reset_rot=(0.2, 0.2, 0.3),
    reset_lin_vel=(0.8, 0.8, 0.3),
    reset_ang_vel=(0.8, 0.8, 1.0),
    reset_joint_pos=0.15,
    reset_oem_prob=0.8 if OEM_POSES else 0.0,
    reset_oem_poses=OEM_POSES,
    reset_oem_frames=25,
)
SPEC = replace(BASE, name="g1mj_p4", dr=_dr)
