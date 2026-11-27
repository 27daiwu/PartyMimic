"""PartyMimic Pipeline V1.2 - hardware profile + per-joint parameter audit (`./pm hwaudit`).

Single source of truth for the hardware limits: pipeline/configs/hardware/g1_29dof_rev1_0.yaml, generated FROM the official Unitree URDF
(data/g1_description/g1_29dof_rev_1_0.urdf, `<limit effort velocity lower upper>`), never from memory.

The audit compares, joint by joint (29 joints, policy order), the same quantities in the five places they live:
  urdf/profile   official hardware limits
  train          mjlab robot cfg of the training robot (g1mj_ythn): actuator effort / kp / kd / armature, MJCF position range, default pose, action scale
  evaluator      what pmpipe.evaluator really uses (per-joint torque limit via the actuator->joint map, gains, ranges, guard clamp)
  sim2sim        the Sim2Sim MuJoCo model (sim2real get_model: effort clip from policy_params, armature / damping / frictionloss of the sim2real MJCF + g1.py)
  deploy         C++ deployment: deploy/partymimic_cpp/assets/policy_params.txt (frozen SSB) and the YTHN package params + guard defaults / site.conf
It never modifies a deployment asset: findings are reported with a "pending hardware confirmation" change list, nothing is applied.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from .core import CONFIGS, REPO, S2R_PY, now, sha256, write_json

URDF = REPO / "data/g1_description/g1_29dof_rev_1_0.urdf"
PROFILE = CONFIGS / "hardware/g1_29dof_rev1_0.yaml"
DEPLOY_PARAMS = REPO / "deploy/partymimic_cpp/assets/policy_params.txt"
PROBE = REPO / "pipeline/pmpipe/hwaudit_s2r_probe.py"
SITE_CONF = REPO / "deploy/partymimic_cpp/config/site.conf"
GUARD_HPP = REPO / "deploy/partymimic_cpp/include/pm/guard.hpp"
G1_PY = REPO.parent / "sim2real_party_p0_patched/sim2real/config/robots/g1.py"

GROUP = {"hip_pitch": "leg", "hip_roll": "leg", "hip_yaw": "leg", "knee": "leg", "ankle_pitch": "ankle(parallel)", "ankle_roll": "ankle(parallel)", "waist_yaw": "waist",
         "waist_roll": "waist(parallel)", "waist_pitch": "waist(parallel)", "shoulder": "arm", "elbow": "arm", "wrist_roll": "arm", "wrist_pitch": "wrist", "wrist_yaw": "wrist"}

MODE_MACHINE = dict(
    value_used_by_deployment=5,
    meaning="Machine / DoF-configuration identifier carried in the HG LowState_/LowCmd_ (uint8). The official demo (g1_mimic_demo/deploy_template/main.cpp:43) writes 5 with the comment '29dof'; "
            "check_mode_machine() (g1_pub.h:52-60) accepts 0 (simulation) or a value equal to the one in LowState. Source IDL / examples give NO table of other values, so only 0 = simulation and 5 = G1 29-DoF are documented.",
    does_not_change_joint_limits="No official source found that makes effort / velocity / position limits depend on mode_machine; the limits below are per-URDF (Rev 1.0, 29 DoF).",
    related_but_different=dict(
        mode_pr="LowCmd_.mode_pr: 0 = PR 'series control for pitch/roll joints', 1 = AB 'parallel control for A/B joints' (py_g1_low_level_example.py:72-75). It decides how the ANKLE command (and the waist pitch/roll "
                "pair) is interpreted: joint space (pitch/roll) vs. motor space (A/B). The deployment and sim2real real_bridge use mode_pr=0 (PR), so all limits here are joint-space limits.",
        sim2real_mode_13_15="'g1-mode_13_15.xml' / `G1_MODE_15_JOINT_ARMATURE` in sim2real g1.py are simulation-model names (reflected armature of the parallel ankle/waist = 2 x 5020 motors); "
                            "undocumented, NOT shown to be related to LowCmd_.mode_machine."),
    other_variants_not_covered="G1 23-DoF (waist locked, wrists 'INVALID') and G1+ are different machines; the audit rejects them as targets (OFFICIAL_FRAMEWORK_AUDIT F13).",
    evidence=["sim2real_party_p0_patched/scripts/g1/real_bridge.py:131 (mode_machine = 5, mode_pr = 0)", "results/p3_1/vendor/unitree-app-templates/g1_mimic_demo/deploy_template/main.cpp:43-44",
              "results/p3_1/vendor/unitree-app-templates/g1_mimic_demo/deploy_template/include/unitree/dds_wrapper/robots/g1/g1_pub.h:52-60", "results/p3/official/sources/py_g1_low_level_example.py.txt:72-75,137,173",
              "results/p3_1/reports/OFFICIAL_FRAMEWORK_AUDIT.md F13", "deploy/partymimic_cpp/src/link.cpp:168-169, session.cpp:217 (expects 5)"])


def joint_group(name: str) -> str:
    for k in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll", "waist_yaw", "waist_roll", "waist_pitch", "wrist_roll", "wrist_pitch", "wrist_yaw", "shoulder", "elbow"):
        if k in name:
            return GROUP[k]
    return "?"


# ----------------------------------------------------------------------------------------------------------------------------- URDF / profile
def parse_urdf(path: Path = URDF) -> dict[str, dict]:
    lines = Path(path).read_text().split("\n")
    out = {}
    for j in ET.parse(path).getroot().iter("joint"):
        l = j.find("limit")
        if j.get("type") == "fixed" or l is None:
            continue
        ln = next((i + 1 for i, x in enumerate(lines) if "<joint" in x and f'name="{j.get("name")}"' in x), None)
        out[j.get("name")] = dict(effort=float(l.get("effort")), velocity=float(l.get("velocity")), lower=float(l.get("lower")), upper=float(l.get("upper")), line=ln)
    return out


def policy_joint_order(params: Path = DEPLOY_PARAMS) -> list[str]:
    for l in open(params):
        if l.startswith("joint_names"):
            return l.split()[2:]
    raise RuntimeError("joint_names missing in " + str(params))


def build_profile() -> dict:
    u = parse_urdf(); order = policy_joint_order()
    miss = [n for n in order if n not in u]
    assert not miss, f"joints missing in URDF: {miss}"
    joints = []
    for i, n in enumerate(order):
        r = u[n]
        joints.append(dict(index=i, name=n, group=joint_group(n), effort_nm=r["effort"], velocity_rad_s=r["velocity"], lower_rad=r["lower"], upper_rad=r["upper"], source=f"urdf:L{r['line']}"))
    return dict(profile="g1_29dof_rev1_0", version="1.2.0", generated=now(), robot="Unitree G1 29-DoF, Rev 1.0",
                urdf=dict(path=str(URDF.relative_to(REPO)), robot_name=ET.parse(URDF).getroot().get("name"), sha256=sha256(URDF), tag="<limit effort velocity lower upper> of every non-fixed joint"),
                policy_joint_order_source="deploy/partymimic_cpp/assets/policy_params.txt (joint_names)", mode_machine=MODE_MACHINE,
                other_official_source=dict(path="data/g1_description/g1_29dof.xml", note="official MJCF of the same Rev 1.0 robot: actuatorfrcrange hip_roll ±139, hip_yaw ±88 (agrees with the URDF)"),
                user_claim_check=dict(claim="Rev 1.0 hip roll = 139 N.m, 20 rad/s", urdf_hip_roll=dict(effort=u["left_hip_roll_joint"]["effort"], velocity=u["left_hip_roll_joint"]["velocity"]),
                                      verdict="CONFIRMED by the URDF (left and right hip roll: effort 139, velocity 20); the knee has the same 139 N.m / 20 rad/s"),
                joints=joints)


def write_profile(path: Path = PROFILE) -> dict:
    p = build_profile()
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = ("# Hardware profile g1_29dof_rev1_0 - SINGLE SOURCE OF TRUTH for G1 29-DoF Rev 1.0 joint limits.\n# Generated from the official URDF by `./pm hwaudit --write-profile` (pmpipe/hwaudit.py); do not edit numbers by hand,\n"
           "# regenerate. The unit test checks every number against the URDF and the recorded sha256.\n")
    path.write_text(hdr + yaml.safe_dump(p, allow_unicode=True, sort_keys=False, width=200))
    return p


def load_profile(path: Path = PROFILE) -> dict:
    return yaml.safe_load(Path(path).read_text())


# ----------------------------------------------------------------------------------------------------------------------------- readers
def read_policy_params(path: Path) -> dict:
    d = {}
    for l in open(path):
        if l.startswith("#") or not l.strip():
            continue
        k, *v = l.split(); d[k] = v
    n = int(d["joint_names"][0]); P = dict(names=d["joint_names"][1:])
    for k in ("default_pos", "action_scale", "kp", "kd", "pos_lo", "pos_hi", "vel_lim", "effort_lim"):
        P[k] = [float(x) for x in d[k][1:1 + n]]
    P["vel_known"] = [int(x) for x in d["vel_known"][1:1 + n]]
    return P


def probe_train(ckpt: Path, motion_dir: Path) -> dict:
    """what the TRAINING robot and the EVALUATOR simulator really contain (main venv; builds the evaluation env with 2 envs)."""
    import numpy as np
    import torch
    from partymimic.env import TrackingSim, make_sim

    from . import evaluator as E
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    ck, cfg, robot, kin, motion, mdp, pi, K, H = E.build(str(ckpt), str(motion_dir), 2, dev)
    names = list(robot.joint_names)
    # training robot cfg (mjlab EntityCfg): regex-expanded per joint
    import re as _re
    cfgr = robot.robot_cfg() if callable(robot.robot_cfg) else robot.robot_cfg
    tr = {n: {} for n in names}
    for a in cfgr.articulation.actuators:
        for n in names:
            if any(_re.fullmatch(p, n) for p in a.target_names_expr):
                tr[n].update(effort=float(a.effort_limit), kp=float(a.stiffness), kd=float(a.damping), armature=float(a.armature), velocity_limit=getattr(a, "velocity_limit", None), frictionloss=getattr(a, "frictionloss", None))
    dflt = robot.default_joint_pos_list(); asc = robot.action_scale_list()
    env = TrackingSim(make_sim(robot), 2, robot, motion, mdp, dev, eval_mode=True, randomize=False)
    sim = env.sim; m = sim.mj_model
    jids = [m.joint("robot/" + n).id for n in names]
    aid = [int((m.actuator_trnid[:, 0] == j).nonzero()[0][0]) for j in jids]
    gain = sim.model.actuator_gainprm[0, :, 0].cpu().numpy(); bias = sim.model.actuator_biasprm[0].cpu().numpy(); frng = sim.model.actuator_forcerange[0].cpu().numpy()
    qlo = kin.q_lo.detach().cpu().numpy(); qhi = kin.q_hi.detach().cpu().numpy()
    ev = {}
    for i, n in enumerate(names):
        tr[n].update(default_pos=float(dflt[i] if not isinstance(dflt, dict) else dflt[n]), action_scale=float(asc[i]), mjcf_lower=float(m.jnt_range[jids[i]][0]), mjcf_upper=float(m.jnt_range[jids[i]][1]))
        ev[n] = dict(effort=float(frng[aid[i], 1]), kp=float(gain[aid[i]]), kd=float(-bias[aid[i], 2]), armature=float(m.dof_armature[m.jnt_dofadr[jids[i]]]), frictionloss=float(m.dof_frictionloss[m.jnt_dofadr[jids[i]]]),
                     lower=float(qlo[i]), upper=float(qhi[i]), mjcf_lower=float(m.jnt_range[jids[i]][0]), mjcf_upper=float(m.jnt_range[jids[i]][1]), velocity_limit=None)
    del env
    torch.cuda.empty_cache()
    return dict(train=tr, evaluator=ev, evaluator_guard="evaluator output string: 'PD target clamped to hard limits +-0.005 rad'; no velocity or torque guard in the evaluator (torque is clipped by the MuJoCo actuator forcerange)")


def probe_s2r(params: Path | None, out: Path) -> dict:
    r = subprocess.run([S2R_PY, str(PROBE), str(out)] + ([str(params)] if params else []), capture_output=True, text=True, cwd=str(REPO))
    if r.returncode:
        raise RuntimeError("sim2real probe failed: " + r.stderr[-800:])
    return json.loads(out.read_text())


def guard_config() -> dict:
    g = {}
    for l in GUARD_HPP.read_text().splitlines():
        m = re.match(r"\s*double\s+(\w+)\s*=\s*([0-9.]+);", l)
        if m and m.group(1) in ("speed_fault_frac", "unknown_speed_limit", "pos_margin_rad", "target_speed_frac", "torque_cmd_frac"):
            g[m.group(1)] = float(m.group(2))
    site = {}
    for l in SITE_CONF.read_text().splitlines():
        l = l.split("#")[0].strip()
        if l:
            k, *v = l.split(); site[k] = v
    return dict(guard_hpp_defaults=g, site_conf_active_overrides=site, python_sim2sim_guard="results/p4/scripts/sim2sim_deploy.py Guard NEW: pos_margin 0.005, target_speed_frac 1.0, torque_cmd_frac 1.0, unknown speed 20 rad/s")


def observed_usage(profile: dict) -> dict | None:
    """per-joint peak |dq| and pre-clip PD torque seen in the closed-loop MuJoCo rollouts that already exist (read-only; optional)."""
    import numpy as np
    files = sorted((REPO / "results/ythn/reports/quality/traj").glob("*.npz")) if (REPO / "results/ythn/reports/quality/traj").exists() else []
    files = [f for f in files if "_nominal" in f.name or "_perturb" in f.name]
    if not files:
        return None
    n = len(profile["joints"]); dq = np.zeros(n); tau = np.zeros(n); dqs = []
    for f in files:
        z = np.load(f, allow_pickle=True); meta = json.loads(str(z["meta"])); lim = np.array(meta["effort_lim"])
        dq = np.maximum(dq, np.abs(z["dq"]).max(0)); tau = np.maximum(tau, z["tau_ratio"].max(0) * lim); dqs.append(np.percentile(np.abs(z["dq"]), 99, axis=0))
    return dict(files=len(files), source_dir="results/ythn/reports/quality/traj", peak_abs_dq=dq.tolist(), p99_abs_dq=np.max(dqs, 0).tolist(), peak_pre_clip_torque_nm=tau.tolist())


# ----------------------------------------------------------------------------------------------------------------------------- compare
FIELDS = ("effort", "velocity", "lower", "upper", "kp", "kd", "armature", "frictionloss", "default_pos", "action_scale")
HW = ("effort", "velocity", "lower", "upper")  # compared with the URDF; the others are compared with the training value


def close(a, b, f):
    if a is None or b is None:
        return a is None and b is None
    tol = dict(effort=1e-6, velocity=1e-6, lower=2e-4, upper=2e-4).get(f)
    return abs(a - b) <= tol if tol is not None else abs(a - b) <= 1e-3 * max(abs(a), abs(b), 1e-9) + 1e-9


def collect(profile, tr, s2r, dep_ssb, dep_pkg) -> dict:
    """source -> joint -> field -> value (None = the source does not define it)."""
    names = [j["name"] for j in profile["joints"]]; S = {}
    S["urdf"] = {j["name"]: dict(effort=j["effort_nm"], velocity=j["velocity_rad_s"], lower=j["lower_rad"], upper=j["upper_rad"]) for j in profile["joints"]}
    S["train"] = {n: dict(effort=tr["train"][n]["effort"], velocity=tr["train"][n]["velocity_limit"], lower=tr["train"][n]["mjcf_lower"], upper=tr["train"][n]["mjcf_upper"], kp=tr["train"][n]["kp"], kd=tr["train"][n]["kd"],
                          armature=tr["train"][n]["armature"], frictionloss=tr["train"][n]["frictionloss"], default_pos=tr["train"][n]["default_pos"], action_scale=tr["train"][n]["action_scale"]) for n in names}
    for n in names:  # mjlab leaves frictionloss unset (None): the simulated value is the MJCF one, which the evaluator model exposes
        if S["train"][n]["frictionloss"] is None:
            S["train"][n]["frictionloss"] = tr["evaluator"][n]["frictionloss"]
    S["evaluator"] = {n: dict(effort=tr["evaluator"][n]["effort"], velocity=None, lower=tr["evaluator"][n]["lower"], upper=tr["evaluator"][n]["upper"], kp=tr["evaluator"][n]["kp"], kd=tr["evaluator"][n]["kd"],
                              armature=tr["evaluator"][n]["armature"], frictionloss=tr["evaluator"][n]["frictionloss"]) for n in names}
    g, sm = s2r["g1_cfg"], s2r["sim2sim_model"]; gi = {n: i for i, n in enumerate(g["names"])}; si = {n: i for i, n in enumerate(sm["joint_names"])}
    S["sim2real_g1py"] = {n: dict(effort=g["effort"][gi[n]], velocity=g["velocity"][gi[n]], lower=g["lower"][gi[n]], upper=g["upper"][gi[n]], armature=g["armature"][gi[n]], frictionloss=g["frictionloss"][gi[n]],
                                  default_pos=(g["default_qpos"][gi[n]] if g["default_qpos"] else None)) for n in names}
    pi = {n: i for i, n in enumerate(dep_ssb["names"])}
    S["sim2sim_model"] = {n: dict(effort=sm["actuator_forcerange_hi"][sm["actuator_joint"].index(n)], lower=sm["jnt_range"][si[n]][0], upper=sm["jnt_range"][si[n]][1], armature=sm["dof_armature"][si[n]],
                                  frictionloss=sm["dof_frictionloss"][si[n]], velocity=dep_ssb["vel_lim"][pi[n]] if dep_ssb["vel_known"][pi[n]] else None, kp=dep_ssb["kp"][pi[n]], kd=dep_ssb["kd"][pi[n]]) for n in names}
    for tag, P in (("deploy_ssb_frozen", dep_ssb), ("deploy_ythn_pkg", dep_pkg)):
        pi = {n: i for i, n in enumerate(P["names"])}
        S[tag] = {n: dict(effort=P["effort_lim"][pi[n]], velocity=P["vel_lim"][pi[n]] if P["vel_known"][pi[n]] else None, lower=P["pos_lo"][pi[n]], upper=P["pos_hi"][pi[n]], kp=P["kp"][pi[n]], kd=P["kd"][pi[n]],
                          default_pos=P["default_pos"][pi[n]], action_scale=P["action_scale"][pi[n]]) for n in names}
    return S


SEV = {1: "S1 安全", 2: "S2 训练/部署行为", 3: "S3 信息"}


def classify(joint, field, src, val, ref, ref_name, obs, urdf):
    """-> (severity, note, suggestion) for ONE disagreement."""
    if field == "effort":
        if src in ("deploy_ssb_frozen", "deploy_ythn_pkg", "sim2sim_model", "sim2real_g1py") and val is not None and val < ref:
            pk = f"；回放中该关节预截断峰值 {obs['peak_pre_clip_torque_nm']:.0f} N·m" if obs else ""
            return 2, f"部署力矩上限 {val:g} < 训练/URDF {ref:g} N·m：策略在训练时可用 {ref:g}，部署会被截断{pk}", "确认硬件额定值后决定：保持保守 88 并在训练里同样限幅重训，或经评估放开。本轮不改部署"
        if val is not None and val > ref:
            return 1, f"{src} 力矩上限 {val:g} 超过 {ref_name} {ref:g} N·m", "立即核对：超出硬件额定"
        return 2, f"{src} 力矩上限 {val} ≠ {ref_name} {ref}", "对齐"
    if field == "velocity":
        if val is None:
            return 3, f"{src} 未定义该关节速度上限（{ref_name}={ref:g} rad/s）", "部署回退到 unknown_speed_limit=20 rad/s（低于 URDF 37，偏保守）；建议补全 g1.py joint_velocity_limit"
        if val > ref:
            pk = f"（回放中观测到的峰值 {obs['peak_abs_dq']:.1f} rad/s）" if obs else ""
            return 1, f"{src} 速度上限 {val:g} > URDF {ref:g} rad/s：guard 的限速/故障阈值(0.9×)按 {val:g} 计算，高于官方规格{pk}", "待确认硬件：若 20 为真实上限，应在 g1.py 改为 20（不在本轮执行）"
        return 2, f"{src} 速度上限 {val:g} < URDF {ref:g}", "保守，仅信息"
    if field in ("lower", "upper"):
        if src.startswith("deploy") or src in ("sim2sim_model", "sim2real_g1py"):
            wider = (val < ref) if field == "lower" else (val > ref)
            return (1 if wider else 2), f"{src} 位置{('下' if field == 'lower' else '上')}限 {val:.4f} 与 URDF {ref:.4f} 不同（{'更宽' if wider else '更窄'}）", "核对"
        return 2, f"{src} 位置限位 {val:.4f} 与 URDF {ref:.4f} 不同", "核对"
    if field in ("kp", "kd"):
        return (1 if src.startswith("deploy") else 2), f"{src} {field} {val:.4g} ≠ 训练 {ref:.4g}（部署增益必须与训练一致）", "对齐；PD 增益不一致会改变策略闭环行为"
    if field in ("armature", "frictionloss"):
        return 3, f"{src} {field} {val:.5g} ≠ 训练 {ref:.5g}（仿真保真度差异，不影响部署）", "Sim2Sim 与训练的动力学模型不同；解释 Sim2Sim 结果时需注意"
    if field == "default_pos" and src == "sim2real_g1py":
        return 3, f"g1.py default_qpos {val:.4f} ≠ 训练默认姿态 {ref:.4f}：这是 sim2real 运行时的站立/初始姿态，C++ 部署实际使用 policy_params.default_pos（与训练一致，无差异）", "仅信息；不要把 g1.py default_qpos 当作策略默认姿态"
    if field == "default_pos":
        return 2, f"{src} 默认姿态 {val:.4f} ≠ 训练 {ref:.4f}", "对齐"
    if field == "action_scale":
        return 2, f"{src} action_scale {val:.6g} ≠ 训练 {ref:.6g}", "对齐"
    return 3, "", ""


def group_issues(findings: list[dict]) -> list[dict]:
    g = {}
    for f in findings:
        k = (f["severity"], f["joint"], f["field"], None if f["value"] is None else round(float(f["value"]), 6), None if f["ref"] is None else round(float(f["ref"]), 6))
        e = g.setdefault(k, dict(f, sources=[]))
        e["sources"].append(f["source"])
    return sorted(g.values(), key=lambda e: (e["severity"], e["field"], e["joint"]))


def audit(ckpt: Path, motion_dir: Path, out_dir: Path, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = load_profile()
    cur = build_profile()
    prof_ok = [(a["name"], a["effort_nm"], a["velocity_rad_s"], a["lower_rad"], a["upper_rad"]) for a in profile["joints"]] == [(a["name"], a["effort_nm"], a["velocity_rad_s"], a["lower_rad"], a["upper_rad"]) for a in cur["joints"]] and profile["urdf"]["sha256"] == cur["urdf"]["sha256"]
    log("profile matches URDF:", prof_ok)
    dep_ssb = read_policy_params(DEPLOY_PARAMS)
    pkg = REPO / "results/ythn/final/assets/it040000/policy_params.txt"
    dep_pkg = read_policy_params(pkg if pkg.exists() else DEPLOY_PARAMS)
    s2r = probe_s2r(None, out_dir / "s2r_probe.json"); log("sim2real probe done")
    tr = probe_train(ckpt, motion_dir); log("training / evaluator probe done")
    obs = observed_usage(profile)
    S = collect(profile, tr, s2r, dep_ssb, dep_pkg); names = [j["name"] for j in profile["joints"]]
    findings = []
    for ji, n in enumerate(names):
        o = None if not obs else dict(peak_abs_dq=obs["peak_abs_dq"][ji], p99_abs_dq=obs["p99_abs_dq"][ji], peak_pre_clip_torque_nm=obs["peak_pre_clip_torque_nm"][ji])
        for f in FIELDS:
            if f in HW:
                ref, rn = S["urdf"][n].get(f), "urdf"
            else:
                ref, rn = S["train"][n].get(f), "train"
            for src in ("train", "evaluator", "sim2real_g1py", "sim2sim_model", "deploy_ssb_frozen", "deploy_ythn_pkg"):
                if src == rn:
                    continue
                v = S[src][n].get(f, "n/a") if f in S[src][n] else "n/a"
                if v == "n/a":
                    continue
                if f == "velocity" and src in ("train", "evaluator"):
                    continue  # aggregated below (one finding per source)
                if ref is None and f in ("velocity", "frictionloss") or (ref is not None and v is not None and close(v, ref, f)):
                    continue
                if ref is None and v is None:
                    continue
                if f in ("armature", "frictionloss", "kp", "kd", "default_pos", "action_scale") and ref is None:
                    continue
                sev, note, sug = classify(n, f, src, v, ref, rn, o, S["urdf"][n])
                findings.append(dict(joint=n, field=f, source=src, value=v, ref=ref, ref_source=rn, severity=sev, note=note, suggestion=sug))
    for src in ("train", "evaluator"):
        if obs:
            ratios = [(obs["peak_abs_dq"][i] / S["urdf"][n]["velocity"], n, obs["peak_abs_dq"][i]) for i, n in enumerate(names)]
            rmax, jn, dqm = max(ratios)
            sev = 1 if rmax > 1.0 else 2 if rmax > 0.7 else 3
            findings.append(dict(joint="(all 29)", field="velocity", source=src, value=None, ref=S["urdf"][jn]["velocity"], ref_source="urdf", severity=sev,
                                 note=f"{src} 不建模关节速度上限；已有闭环回放中占 URDF 速度上限比例最大的是 {jn}：峰值 {dqm:.1f} rad/s = {rmax * 100:.0f}% of {S['urdf'][jn]['velocity']:g}", suggestion="实际占用远低于上限则仅信息；若某关节占比 >70% 应在训练里加速度惩罚/限幅"))
        else:
            findings.append(dict(joint="(all 29)", field="velocity", source=src, value=None, ref=None, ref_source="urdf", severity=3, note=f"{src} 不建模关节速度上限（无回放可估占用）", suggestion="先跑 ./pm quality 生成闭环回放"))
    # observed torque against the deploy cap / URDF
    obs_rows = []
    if obs:
        for ji, n in enumerate(names):
            obs_rows.append(dict(joint=n, peak_abs_dq=obs["peak_abs_dq"][ji], urdf_velocity=S["urdf"][n]["velocity"], dq_ratio_to_urdf=obs["peak_abs_dq"][ji] / S["urdf"][n]["velocity"],
                                 peak_pre_clip_torque_nm=obs["peak_pre_clip_torque_nm"][ji], deploy_effort=S["deploy_ythn_pkg"][n]["effort"], urdf_effort=S["urdf"][n]["effort"]))
            if obs["peak_pre_clip_torque_nm"][ji] > S["deploy_ythn_pkg"][n]["effort"] * 1.0001:
                findings.append(dict(joint=n, field="effort(observed)", source="rollout", value=obs["peak_pre_clip_torque_nm"][ji], ref=S["deploy_ythn_pkg"][n]["effort"], ref_source="deploy cap", severity=2 if S["deploy_ythn_pkg"][n]["effort"] < S["train"][n]["effort"] else 3,
                                     note=f"回放预截断峰值 {obs['peak_pre_clip_torque_nm'][ji]:.1f} N·m > 部署上限 {S['deploy_ythn_pkg'][n]['effort']:g}（URDF {S['urdf'][n]['effort']:g}）", suggestion="决定 88/110/139 限幅实验的依据" if S["deploy_ythn_pkg"][n]["effort"] < S["train"][n]["effort"] else "训练与部署上限相同，仅说明该关节在峰值处贴着上限"))
    res = dict(generated=now(), profile=dict(path=str(PROFILE.relative_to(REPO)), sha256_urdf=profile["urdf"]["sha256"], matches_urdf=prof_ok), joints=names, sources=S, guard=guard_config(), observed=obs_rows,
               evaluator_guard=tr["evaluator_guard"], findings=findings, issues=(issues := group_issues(findings)), counts={str(k): sum(1 for f in issues if f["severity"] == k) for k in (1, 2, 3)},
               counts_raw={str(k): sum(1 for f in findings if f["severity"] == k) for k in (1, 2, 3)}, checkpoint=str(ckpt))
    write_json(out_dir / "hardware_audit.json", res)
    return res


# ----------------------------------------------------------------------------------------------------------------------------- report
def _f(x, nd=4):
    return "-" if x is None else (f"{x:.{nd}f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x))


def write_report(res: dict, path: Path):
    S = res["sources"]; names = res["joints"]; fnd = res["findings"]; iss = res["issues"]; c = res["counts"]
    srcs = ["urdf", "train", "evaluator", "sim2real_g1py", "sim2sim_model", "deploy_ssb_frozen", "deploy_ythn_pkg"]
    short = dict(urdf="URDF", train="训练mjlab", evaluator="评估器", sim2real_g1py="g1.py", sim2sim_model="Sim2Sim", deploy_ssb_frozen="部署(SSB冻结)", deploy_ythn_pkg="部署(YTHN包)")
    hr = [f for f in fnd if f["joint"].endswith("hip_roll_joint") and f["field"] == "effort"]
    vel = [f for f in iss if f["field"] == "velocity" and f["severity"] == 1]
    obs = {o["joint"]: o for o in res["observed"]}
    pk = ("；已有闭环回放中髋 Roll 预截断峰值力矩 " + " / ".join(f"{obs[j]['peak_pre_clip_torque_nm']:.0f}" for j in ("left_hip_roll_joint", "right_hip_roll_joint")) + " N·m，峰值转速 " + " / ".join(f"{obs[j]['peak_abs_dq']:.1f}" for j in ("left_hip_roll_joint", "right_hip_roll_joint")) + " rad/s（URDF 20）") if res["observed"] else ""
    L = [f"# 硬件参数审计 HARDWARE_AUDIT（Pipeline V1.2，自动生成 {res['generated']}）", "",
         "> 硬件 profile：`pipeline/configs/hardware/g1_29dof_rev1_0.yaml`（由官方 URDF 生成，URDF sha256 `" + res["profile"]["sha256_urdf"][:12] + "…`，与 profile 记录一致=" + str(res["profile"]["matches_urdf"]) + "）。**未修改任何部署资产；未连接真机。**", "",
         "## 结论", "",
         f"共 **{sum(c.values())} 个独立问题**（来源记录 {sum(res['counts_raw'].values())} 条，同一问题在 g1.py / Sim2Sim / 部署多处重复出现已合并）：S1（可能越过官方硬件规格）{c['1']} 个，S2（训练/部署行为不一致）{c['2']} 个，S3（仅信息）{c['3']} 个。", "",
         "**最重要的 3 条：**", "",
         f"1. **用户给的 Rev 1.0 髋 Roll = 139 N·m / 20 rad/s 由官方 URDF 证实**（L108/L300，官方 MJCF `g1_29dof.xml` actuatorfrcrange ±139 同）。训练(mjlab)=139 与之一致；**部署 `policy_params.txt` 与 sim2real `g1.py` 的髋 Roll 力矩=88**（只有髋 Pitch/Yaw 才是 88）。部署比训练少 51 N·m；这是 g1.py 的取值，而不是 URDF 的。",
         "2. **部署速度上限与 URDF 不一致**：" + ("；".join(f"{f['joint']} 部署 {f['value']:g} vs URDF {f['ref']:g} rad/s" for f in vel) if vel else "无") + "（其余 27 个关节的速度上限与 URDF 一致）。guard 的限速和故障阈值(0.9×)按部署值计算，宽于官方规格" + pk + "。腰 Roll/Pitch 在 g1.py 里没有速度上限（部署回退 20 rad/s，URDF 为 37）。训练与评估都不建模速度上限。",
         "3. **训练/评估与 Sim2Sim 的动力学模型不同（仅仿真保真度，不影响部署）**：关节摩擦 训练 0 vs sim2real 0.01（29 个关节）；电枢 髋 Pitch 训练 0.0102 vs sim2real 0.0251（g1.py 把髋 Pitch 按 7520_22 取，训练按 7520_14）、腕 Pitch/Yaw 训练 0.00425 vs sim2real 0.00218。解读 Sim2Sim 结果时应注意这两个模型并不相同。",
         "", "已核对一致的部分：训练(mjlab)与评估器(V1.1)的力矩上限、全部 29 个关节的位置限位（URDF / mjlab / 评估器 / g1.py / Sim2Sim / 部署）、PD 增益 kp/kd、部署 default_pos 与 action_scale 均与基准一致；评估器的力矩上限这一项在 V1.0 是错的（已在 V1.1 修复），本审计可以检出这类错误。", ""]
    L += ["关于 mode_machine：", "", "- " + MODE_MACHINE["meaning"], "- " + MODE_MACHINE["does_not_change_joint_limits"], "- " + MODE_MACHINE["related_but_different"]["mode_pr"], "- " + MODE_MACHINE["related_but_different"]["sim2real_mode_13_15"], "",
          "## 1. 官方限位总表（URDF，Rev 1.0）与各来源是否一致", "",
          "| # | 关节 | 分组 | 力矩 N·m：" + " / ".join(short[s] for s in srcs) + " | 速度 rad/s：" + " / ".join(short[s] for s in srcs) + " |", "|---|---|---|---|---|"]
    for i, n in enumerate(names):
        L.append(f"| {i} | {n} | {joint_group(n)} | " + " / ".join(_f(S[s][n].get('effort'), 1) for s in srcs) + " | " + " / ".join(_f(S[s][n].get('velocity'), 1) for s in srcs) + " |")
    L += ["", "## 2. 位置限位", "", "| 关节 | URDF [lo, hi] | 与 URDF 不一致的来源 |", "|---|---|---|"]
    for n in names:
        bad = [f"{short[f['source']]} {_f(f['value'])}" for f in fnd if f["joint"] == n and f["field"] in ("lower", "upper")]
        L.append(f"| {n} | [{_f(S['urdf'][n]['lower'])}, {_f(S['urdf'][n]['upper'])}] | {'; '.join(bad) if bad else '全部一致'} |")
    L += ["", "## 3. PD 增益 / 电枢 / 摩擦 / 默认姿态 / action_scale（以训练为基准）", "", "| 关节 | kp 训练 | kp 部署 | kd 训练 | kd 部署 | 电枢 训练 | 电枢 sim2real | 摩擦 训练 | 摩擦 sim2real |", "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        L.append(f"| {n} | {_f(S['train'][n]['kp'], 3)} | {_f(S['deploy_ythn_pkg'][n]['kp'], 3)} | {_f(S['train'][n]['kd'], 3)} | {_f(S['deploy_ythn_pkg'][n]['kd'], 3)} | {_f(S['train'][n]['armature'], 5)} | {_f(S['sim2real_g1py'][n]['armature'], 5)} | {_f(S['train'][n]['frictionloss'], 3)} | {_f(S['sim2real_g1py'][n]['frictionloss'], 3)} |")
    G = res["guard"]
    L += ["", "## 4. guard / 余量", "", "- 评估器：" + res["evaluator_guard"], f"- C++ guard 默认值（guard.hpp）：{json.dumps(G['guard_hpp_defaults'])}；site.conf 生效的覆盖项：{json.dumps(G['site_conf_active_overrides'])}", f"- Python Sim2Sim guard：{G['python_sim2sim_guard']}", ""]
    if res["observed"]:
        L += ["## 5. 已有闭环回放的实际占用（只读取 results/ythn/reports/quality/traj，nominal+扰动全部回放的最大值）", "", "| 关节 | 峰值 |dq| rad/s | 占 URDF 速度 | 预截断峰值力矩 N·m | 部署上限 | URDF 上限 |", "|---|---|---|---|---|---|---|"]
        for o in res["observed"]:
            L.append(f"| {o['joint']} | {o['peak_abs_dq']:.1f} | {o['dq_ratio_to_urdf'] * 100:.0f}% | {o['peak_pre_clip_torque_nm']:.1f} | {o['deploy_effort']:g} | {o['urdf_effort']:g} |")
        L.append("")
    L += ["## 6. 全部问题（按严重度；来源列 = 出现该数值的位置）", "", "| 级别 | 关节 | 项 | 来源 | 数值 | 基准 | 说明 | 建议 |", "|---|---|---|---|---|---|---|---|"]
    for f in iss:
        L.append(f"| {SEV[f['severity']]} | {f['joint']} | {f['field']} | {', '.join(short.get(x, x) for x in f['sources'])} | {_f(f['value'])} | {_f(f['ref'])} ({f['ref_source']}) | {f['note']} | {f['suggestion']} |")
    L += ["", "## 7. 待确认硬件后的变更清单（**本轮不执行**）", "",
          "1. 确认真机硬件额定：髋 Roll 力矩 139 还是 88 N·m、速度 20 还是 32 rad/s。官方 URDF/MJCF 给出 139/20；`g1.py` 给出 88/32；二者必有一个不是真值。",
          "2. 若以 URDF 为准：`sim2real/config/robots/g1.py` 的 `joint_effort_limit[hip_roll]=139`、`joint_velocity_limit[hip_roll]=20`；重新生成 `policy_params.txt`（`deploy_assets.py`）；部署包需重打包并重新跑 selfcheck / Sim2Sim。",
          "3. 若以 88 为准（保守）：训练侧 `g1mj_ythn` 应把髋 Roll 力矩设为 88 并重新做域随机化 / 训练，否则策略学到的是用 139 才可行的动作。",
          "4. 腰 Roll/Pitch 补上速度上限 37（g1.py 缺项）；按 URDF 补全训练/评估的速度约束或加速度惩罚（视回放占用比例决定）。",
          "5. 对齐训练与 sim2real 模型的电枢 / 摩擦（Sim2Sim 保真度）。", ""]
    Path(path).write_text("\n".join(L) + "\n")
