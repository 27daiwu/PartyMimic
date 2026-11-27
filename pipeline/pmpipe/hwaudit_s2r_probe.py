#!/usr/bin/env python3
"""Hardware-audit probe, sim2real (py3.10) side: dumps the per-joint numbers of (a) sim2real `G1_CFG` (source of deploy_assets.py effort / velocity / limits)
and (b) the Sim2Sim MuJoCo model (`sim2sim_deploy.get_model`; effort clip taken from policy_params). Read-only, offline, no robot connection.
usage: S2R_PY hwaudit_s2r_probe.py OUT.json [policy_params.txt]"""
import json
import os
import sys

PM = "/home/unitree/workwpace/zjy_ws/PartyMimic"
sys.path.insert(0, PM + "/results/p4/scripts")
import numpy as np  # noqa: E402
import sim2sim_deploy as S  # noqa: E402  (unsets proxies, points HF at the offline cache, adds the sim2real checkout to sys.path)
from sim2real.config.robots import get_robot_cfg  # noqa: E402
from sim2real.utils.strings import resolve_matching_names_values  # noqa: E402

out_path = sys.argv[1]
if len(sys.argv) > 2:
    import pm_stack
    pm_stack.load_params.__defaults__ = (sys.argv[2],)
cfg = get_robot_cfg("g1")
names = list(cfg.joint_names)


def expand(d, fill=np.nan):
    o = np.full(len(names), fill)
    if not d:
        return o
    idx, _, v = resolve_matching_names_values(dict(d), names, preserve_order=True, strict=False)
    o[idx] = v
    return o


lst = lambda a: [None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x) for x in np.asarray(a, float)]
g1 = dict(names=names,
          lower=lst(expand(cfg.joint_pos_lower_limit)), upper=lst(expand(cfg.joint_pos_upper_limit)), velocity=lst(expand(cfg.joint_velocity_limit)), effort=lst(expand(cfg.joint_effort_limit)),
          armature=lst(expand(cfg.joint_armature)), frictionloss=lst(expand(cfg.joint_frictionloss)), safe_kp=lst(expand(cfg.safe_joint_kp)), safe_kd=lst(expand(cfg.safe_joint_kd)),
          default_qpos=lst(cfg.default_qpos[-len(names):]) if cfg.default_qpos else None, mjcf=str(cfg.mjcf_path))
m, qadr, vadr, jn = S.get_model()
sim = dict(joint_names=list(jn), jnt_range=[[float(x) for x in m.jnt_range[m.joint(n).id]] for n in jn], dof_armature=lst([m.dof_armature[m.joint(n).dofadr[0]] for n in jn]),
           dof_damping=lst([m.dof_damping[m.joint(n).dofadr[0]] for n in jn]), dof_frictionloss=lst([m.dof_frictionloss[m.joint(n).dofadr[0]] for n in jn]),
           actuator_ctrlrange_hi=lst([m.actuator_ctrlrange[i][1] for i in range(29)]), actuator_forcerange_hi=lst([m.actuator_forcerange[i][1] for i in range(29)]),
           actuator_joint=[m.joint(int(m.actuator_trnid[i, 0])).name for i in range(29)], timestep=float(m.opt.timestep))
json.dump(dict(g1_cfg=g1, sim2sim_model=sim), open(out_path, "w"), indent=1)
print("wrote", out_path)
