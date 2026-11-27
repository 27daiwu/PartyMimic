#!/usr/bin/env python3
"""PartyMimic pipeline evaluator (generalised from results/ythn/scripts/ythn_eval.py, same grid / scores; push time clamped for short clips).
Independent evaluator: a fixed grid of nominal / perturbed / disturbed conditions, batched in ONE native simulator.

Independent of training: its own process, own seed, deterministic condition grid (not the training randomization), the
deployment-guard semantics (PD targets clamped to the hard limits, margin 5 mrad). Never touches the motion files.

For each checkpoint it writes <out>/<name>.json with
  per-condition rows   : survival, fidelity (XY/yaw-anchored 14-link MPJPE incl. root height, joint MAE) over the dance span,
                         balance (tilt, angular velocity), foot slip, torque saturation, penetration, single-support stats
  scores               : balance / fidelity / combined (definitions in SCORE_DEF; raw numbers are kept so any re-ranking is possible)
usage: ythn_eval.py --ckpt a.pt [b.pt ...] --out DIR [--reps 8] [--seed 1234] [--motion-dir DIR]
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

PM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PM / "src"))
import mujoco
import partymimic.networks as networks
from partymimic.dr import XiLayout
from partymimic.env import TrackingSim, make_sim
from partymimic.kinematics import Kinematics
from partymimic.mdp import MDP
from partymimic.motion import MotionLib
from partymimic.networks import Policy, policy_kwargs
from partymimic.robot import get_robot
from partymimic.rotations import mat_to_quat, quat_to_mat, yaw_mat, mm3, mv3

PAD = 50  # hold frames before / after the dance in the padded training clip

# (name, params). delay: physics substeps (5 ms); kp: global PD scale; fric: friction; effort: torque-limit scale; mass: robot mass scale;
# payload: kg per wrist; perturb: initial pose offset scale; push: (speed m/s, heading deg) velocity kick at push_t seconds (control time in the clip)
CONDS = [
    ("nominal", {}),
    ("init_perturb", dict(perturb=1.0)),
    ("delay10ms", dict(delay=2, perturb=0.5)),
    ("delay20ms", dict(delay=4, perturb=0.5)),
    ("kp0.85", dict(kp=0.85, perturb=0.5)),
    ("kp1.15", dict(kp=1.15, perturb=0.5)),
    ("fric0.5", dict(fric=0.5, perturb=0.5)),
    ("effort0.8", dict(effort=0.8, perturb=0.5)),
    ("mass1.08_payload", dict(mass=1.08, payload=1.0, perturb=0.5)),
    ("push_fwd0.5", dict(push=(0.5, 0.0), push_t=3.0)),
    ("push_back0.5", dict(push=(0.5, 180.0), push_t=3.0)),
    ("push_left0.5", dict(push=(0.5, 90.0), push_t=5.0)),
    ("push_right0.5", dict(push=(0.5, -90.0), push_t=5.0)),
    ("stress_combo", dict(kp=0.9, delay=3, fric=0.6, effort=0.9, mass=1.05, payload=0.5, perturb=1.0, push=(0.3, None), push_t=4.0)),
]
# harsher grid, only for the final re-evaluation of the candidates (--hard): beyond the training randomization ranges
CONDS_HARD = [
    ("hard_delay25ms", dict(delay=5, perturb=1.0)),
    ("hard_kp0.8", dict(kp=0.8, perturb=1.0)),
    ("hard_effort0.7", dict(effort=0.7, perturb=0.5)),
    ("hard_fric0.35", dict(fric=0.35, perturb=0.5)),
    ("hard_push_fwd0.8", dict(push=(0.8, 0.0), push_t=3.0)),
    ("hard_push_lat0.7", dict(push=(0.7, 90.0), push_t=5.0)),
    ("hard_push_rand0.7", dict(push=(0.7, None), push_t=4.0)),
    ("hard_combo", dict(kp=0.85, delay=4, fric=0.45, effort=0.8, mass=1.08, payload=1.0, perturb=1.5, push=(0.5, None), push_t=4.0)),
]
CONDS_BASE = list(CONDS)
ROBUST = [c for c, _ in CONDS if c != "nominal"]
SCORE_DEF = {
    "balance": "0.45*survival(all non-nominal conds, mean) + 0.15*(1-clip(tilt_p95_stress/0.5)) + 0.15*(1-clip(slip_mean/0.30 m/s)) "
               "+ 0.15*(1-clip(torque_sat_frac/0.10)) + 0.10*(1-clip(pen_max/0.02 m)); nominal survival must be 1 else balance*=0.5",
    "fidelity": "0.6*(1-clip((mpjpe_nominal_mm-15)/65)) + 0.2*(1-clip(jmae_nominal_rad/0.12)) + 0.2*(1-clip((mpjpe_robust_mean_mm-15)/90))",
    "combined": "0.5*balance + 0.5*fidelity",
}
clip01 = lambda x: float(min(max(x, 0.0), 1.0))


def build(ckpt_path, motion_dir, N, dev):
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = ck["args"]
    robot = get_robot(cfg["robot"])
    import dataclasses
    if cfg.get("wrist_payload", 0) > 0:
        robot = dataclasses.replace(robot, dr=dataclasses.replace(robot.dr, wrist_payload=cfg["wrist_payload"], payload_p0=cfg["payload_p0"]))
    kin = Kinematics(robot, dev)
    motion = MotionLib.from_dir(str(motion_dir), robot, kin, dev)
    motion.set_interface(cfg.get("interface", "joint"), robot.tracked_bodies, cfg.get("obs_err", "none"))
    layout = XiLayout(robot, dev)
    mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), layout.dim, robot.sensing)
    K, H = cfg["goal_frames"], cfg["hist"]
    w = lambda k, d: tuple(map(int, cfg.get(k, d).split(",")))
    pi = Policy(mdp.obs_dim, motion.goal_dim(K), robot.num_joints, hist=H, enc_hidden=w("enc_hidden", "2048,1024,512,256"),
                dec_hidden=w("dec_hidden", "2048,1024,512,256"), **policy_kwargs(cfg, robot)).to(dev).eval()
    pi.load_state_dict(ck["pi"], strict=True)
    networks.AMP_DTYPE = torch.bfloat16 if cfg.get("amp", "none") == "bf16" else None
    return ck, cfg, robot, kin, motion, mdp, pi, K, H


@torch.no_grad()
def evaluate(ckpt_path, motion_dir, reps, seed, dev="cuda:0", log=print, hard=False):
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    global CONDS, ROBUST
    CONDS = CONDS_BASE + (CONDS_HARD if hard else [])
    ROBUST = [c for c, _ in CONDS if c != "nominal"]
    N = len(CONDS) * reps
    ck, cfg, robot, kin, motion, mdp, pi, K, H = build(ckpt_path, motion_dir, N, dev)
    assert len(motion.names) == 1
    env = TrackingSim(make_sim(robot), N, robot, motion, mdp, dev, eval_mode=True, randomize=False)
    J, sim, ent = robot.num_joints, env.sim, env.robot
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    gid = torch.arange(N, device=dev) // reps  # condition index of each env
    P = lambda key, default: torch.tensor([c.get(key, default) if c.get(key, default) is not None else default for _, c in CONDS], device=dev, dtype=torch.float32)[gid]
    # ---- per-env parameters (after the neutral eval startup) ----
    kp, fric, effort, mass, payload, delay, pert = P("kp", 1.0), P("fric", 1.0), P("effort", 1.0), P("mass", 1.0), P("payload", 0.0), P("delay", 0), P("perturb", 0.0)
    rz = env.randomizer
    sim.model.geom_friction[:, rz.ground_geoms, 0] = fric[:, None]
    sim.model.geom_friction[:, rz.friction_geoms, 0] = fric[:, None]
    for act in ent.actuators:
        ids = act.global_ctrl_ids
        sim.model.actuator_gainprm[:, ids, 0] = sim.model.actuator_gainprm[:, ids, 0] * kp[:, None]
        sim.model.actuator_biasprm[:, ids, 1] = sim.model.actuator_biasprm[:, ids, 1] * kp[:, None]
        if "effort" in env.layout.slices:
            dfr = sim.get_default_field("actuator_forcerange")
            sim.model.actuator_forcerange[:, ids, 0] = dfr[ids, 0] * effort[:, None]
            sim.model.actuator_forcerange[:, ids, 1] = dfr[ids, 1] * effort[:, None]
    from mjlab.managers.event_manager import RecomputeLevel
    dm, di = sim.get_default_field("body_mass"), sim.get_default_field("body_inertia")
    sim.model.body_mass[:] = dm.unsqueeze(0) * mass[:, None]
    sim.model.body_inertia[:] = di.unsqueeze(0) * mass[:, None, None]
    if cfg.get("wrist_payload", 0) > 0:
        b = rz.wrist_body_ids
        sim.model.body_mass[:, b] = dm[b] * mass[:, None] + payload[:, None]
    sim.recompute_constants(RecomputeLevel.set_const)
    env.cmd_delay = delay.long()
    # ---- start exactly on the reference, zero velocity, plus the initial perturbation of the condition ----
    ids_all = torch.arange(N, device=dev)
    env.reset(ids_all, motion.clip_start.expand(N).clone())
    p, R, v, w, q, qd = env.state(); quat = mat_to_quat(R)
    v, w, qd = torch.zeros_like(v), torch.zeros_like(w), torch.zeros_like(qd)
    rnd = lambda *s: (torch.rand(*s, device=dev, generator=gen) * 2 - 1)
    p = p.clone(); p[:, :2] += rnd(N, 2) * 0.02 * pert[:, None]
    dyaw = rnd(N) * 0.05 * pert
    qy = torch.stack([torch.cos(dyaw / 2), torch.zeros_like(dyaw), torch.zeros_like(dyaw), torch.sin(dyaw / 2)], -1)
    quat = mat_to_quat(quat_to_mat(qy) @ quat_to_mat(quat))
    q = torch.clamp(q + rnd(N, J) * 0.03 * pert[:, None], kin.q_lo, kin.q_hi)
    ent.write_root_link_pose_to_sim(torch.cat((p + env.scene.env_origins, quat), -1)); ent.write_root_link_velocity_to_sim(torch.cat((v, w), -1))
    ent.write_joint_position_to_sim(q); ent.write_joint_velocity_to_sim(qd); ent.set_joint_position_target(q); env.prev_target[:] = q
    env.done_eval[:] = False; env.failed_eval[:] = False
    # ---- substep observer: foot clearance / slip / torque saturation, on the GPU ----
    m = sim.mj_model
    foot_geoms = [[i for i in range(m.ngeom) if (m.geom(i).name or "").split("/")[-1].startswith(s + "_foot")] for s in ("left", "right")]
    fg = torch.tensor(sum(foot_geoms, []), device=dev); nfg = [len(g) for g in foot_geoms]
    fside = torch.cat([torch.full((n,), k, device=dev) for k, n in enumerate(nfg)])
    gsz = torch.tensor(m.geom_size[sum(foot_geoms, [])], device=dev, dtype=torch.float32)  # capsule: [radius, half length]
    jids = [m.joint("robot/" + n).id for n in robot.joint_names]
    va = torch.tensor(m.jnt_dofadr[jids], device=dev)
    effort_nom = torch.tensor([m.actuator_forcerange[i, 1] for i in range(m.nu)], device=dev, dtype=torch.float32)
    # actuator order == joint order (one position actuator per joint); keep the per-env limit
    # per-JOINT torque limit. NOTE: actuators are created per actuator group in mjlab (shoulder/elbow/wrist first, then hip/knee ...), so actuator index != joint index.
    # (V1.1 fix: V1.0 indexed the limits by joint order, e.g. compared the hip-roll torque with the 25 N.m shoulder limit -> bogus saturation statistics.)
    aid = torch.tensor([int((m.actuator_trnid[:, 0] == j).nonzero()[0][0]) for j in jids], device=dev)
    lim = sim.model.actuator_forcerange[:, :, 1][:, aid].clone().float()
    wtn = [n.replace("_joint", "") for n in robot.joint_names]
    grp = {"ankle": [i for i, n in enumerate(wtn) if "ankle" in n], "hip_roll": [i for i, n in enumerate(wtn) if "hip_roll" in n],
           "knee": [i for i, n in enumerate(wtn) if "knee" in n], "waist": [i for i, n in enumerate(wtn) if "waist" in n]}
    acc = dict(sat=torch.zeros(N, J, device=dev), n=torch.zeros(N, device=dev), taumax=torch.zeros(N, J, device=dev),
               slip_sum=torch.zeros(N, 2, device=dev), slip_cnt=torch.zeros(N, 2, device=dev), pen=torch.zeros(N, device=dev),
               slip_ss_sum=torch.zeros(N, device=dev), slip_ss_cnt=torch.zeros(N, device=dev))
    state = dict(prev_pt=None, active=torch.ones(N, dtype=torch.bool, device=dev), ss=torch.zeros(N, dtype=torch.bool, device=dev), dance=torch.zeros(N, dtype=torch.bool, device=dev))
    real_step = sim.step

    def observed_step():
        real_step()
        d = sim.data
        xp = d.geom_xpos[:, fg]                      # (N, G, 3)
        xm = d.geom_xmat[:, fg].reshape(N, -1, 3, 3)
        ax = xm[..., :, 2]                             # capsule axis (world)
        half, rad = gsz[:, 1], gsz[:, 0]
        # lowest end point of each capsule and the clearance of its surface above the floor
        sgn = torch.where(ax[..., 2] > 0, -1.0, 1.0)
        pt = xp + ax * (sgn * half)[..., None]
        clear = pt[..., 2] - rad
        contact = clear < 0.005
        if state["prev_pt"] is not None:
            sp = (pt[..., :2] - state["prev_pt"][..., :2]).norm(dim=-1) / 0.005
            ok = contact & state["prev_c"] & (sp < 3.0)
            for s in (0, 1):
                mk = ok & (fside == s)[None]
                acc["slip_sum"][:, s] += torch.where(mk & state["dance"][:, None], sp, 0.0).sum(-1)
                acc["slip_cnt"][:, s] += (mk & state["dance"][:, None]).sum(-1)
            mk = ok & state["ss_mask"][:, None] & state["dance"][:, None]
            acc["slip_ss_sum"] += torch.where(mk, sp, 0.0).sum(-1); acc["slip_ss_cnt"] += mk.sum(-1)
        state["prev_pt"], state["prev_c"] = pt, contact
        acc["pen"] = torch.maximum(acc["pen"], torch.where(state["active"] & state["dance"], torch.relu(-clear).amax(-1), torch.zeros(N, device=dev)))
        tau = d.qfrc_actuator[:, va].abs()
        on = (state["active"] & state["dance"]).float()
        acc["sat"] += (tau >= 0.98 * lim).float() * on[:, None]
        acc["n"] += on
        acc["taumax"] = torch.maximum(acc["taumax"], tau / lim * on[:, None])

    sim.step = observed_step
    state["ss_mask"] = torch.zeros(N, dtype=torch.bool, device=dev)
    # ---- rollout ----
    T = int(motion.clip_len[0]); dance_lo, dance_hi = PAD, T - PAD  # control steps whose reference frame is the dance (f - clip_start)
    cs = int(motion.clip_start[0])
    dance_s = max(T - 2 * PAD, 1) / 50.0  # pipeline generalisation: push time stays inside the dance (short clips), unchanged for dances >= 8 s
    ptime = lambda c: min(c.get("push_t", 0), max(0.5, 0.6 * dance_s))
    push_step = torch.tensor([int(ptime(c) * 50) + PAD if "push" in c else -1 for _, c in CONDS], device=dev)[gid]
    push_spd = torch.tensor([c["push"][0] if "push" in c else 0.0 for _, c in CONDS], device=dev)[gid]
    ph = torch.tensor([(c["push"][1] if c["push"][1] is not None else 0.0) if "push" in c else 0.0 for _, c in CONDS], device=dev)[gid]
    rand_head = torch.tensor([("push" in c and c["push"][1] is None) for _, c in CONDS], device=dev)[gid]
    ph = torch.where(rand_head, rnd(N) * 180.0, ph)
    rc = motion.ref_contact if getattr(motion, "ref_contact", None) is not None else None
    if rc is None:
        feet = [list(robot.tracked_bodies).index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
        motion.set_ref_contact(feet); rc = motion.ref_contact
    rec = {k: [] for k in ("mpjpe", "jmae", "tilt", "wrms", "alive", "zerr", "ss", "cagree", "gerr", "vxy")}
    history = None
    push_seen = torch.zeros(N, device=dev)
    nb = len(robot.tracked_bodies)
    for step in range(T - 1):
        o = env.observe()
        history = o[:, None].repeat(1, H, 1) if history is None else torch.cat((history[:, 1:], o[:, None]), 1)
        a = pi(history, motion.goal(env.f, K))
        fi = env.f - cs                                   # frame index of this step's start
        in_dance = (fi >= dance_lo) & (fi < dance_hi)
        state["dance"] = in_dance
        state["ss_mask"] = (rc[env.f].sum(-1) == 1)
        state["active"] = ~env.done_eval
        out = env.step(a)
        p, R, v, w, q, qd = env.state(); f1 = env.f
        bp = kin.forward(p, R, v, w, q, qd)[0]
        rp, rR = motion.root_p[f1], motion.root_R[f1]
        dR = yaw_mat(mm3(R, rR.transpose(-1, -2)))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + mv3(dR.unsqueeze(-3), motion.body_p[f1] - rp.unsqueeze(-2))
        rec["mpjpe"].append((ref_bp - bp).norm(dim=-1).mean(-1))
        rec["gerr"].append((motion.body_p[f1] - bp).norm(dim=-1).mean(-1))
        rec["jmae"].append((q - motion.q[f1]).abs().mean(-1))
        rec["tilt"].append(torch.acos(R[:, 2, 2].clamp(-1, 1)))
        rec["wrms"].append(w.norm(dim=-1))
        rec["vxy"].append(v[:, :2].norm(dim=-1))
        rec["zerr"].append((p[:, 2] - rp[:, 2]).abs())
        rec["alive"].append(~env.done_eval | out["done"] & ~out["term"])
        rec["ss"].append(state["ss_mask"].clone())
        rec["cagree"].append(((state["prev_c"].float() * (fside == 0)[None]).amax(-1) == rc[f1][:, 0]).float() * 0.5 + ((state["prev_c"].float() * (fside == 1)[None]).amax(-1) == rc[f1][:, 1]).float() * 0.5)
        # velocity kick (applied after the step, so it acts on the next control step)
        kick = (push_step == step + 1) & ~env.done_eval
        if bool(kick.any()):
            idk = kick.nonzero().squeeze(-1); yaw = torch.atan2(R[idk, 1, 0], R[idk, 0, 0]); ang = yaw + ph[idk] * math.pi / 180.0
            dv = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1) * push_spd[idk, None]
            ent.write_root_link_velocity_to_sim(torch.cat((v[idk] + dv, w[idk]), -1), env_ids=idk)
            kicked = env.state()[2][idk, :2].norm(dim=-1); push_seen[idk] = kicked
        if bool(env.done_eval.all()):
            break
    sim.step = real_step
    S = {k: torch.stack(v, 0) for k, v in rec.items()}      # (steps, N)
    Tn = S["mpjpe"].shape[0]
    stepi = torch.arange(Tn, device=dev)[:, None]
    fidx = stepi + 1                                      # reference frame index after the step
    dmask = (fidx >= dance_lo) & (fidx <= dance_hi)
    alive_t = torch.ones(Tn, N, dtype=torch.bool, device=dev)
    # a step counts while the env had not terminated before it; failed_eval marks termination
    failed = env.failed_eval
    # survival = never terminated over the full clip; time of failure not tracked per step -> use failed_eval
    surv = ~failed
    # fidelity masks: dance span and (for failed envs) the whole dance anyway (they are scored as failed separately)
    msk = dmask.float()
    avg = lambda x, m_=None: ((x * (msk if m_ is None else m_)).sum(0) / (msk if m_ is None else m_).sum(0).clamp(min=1))
    ssm = S["ss"].float() * msk
    res_env = dict(
        surv=surv.float(), mpjpe=avg(S["mpjpe"]), gerr=avg(S["gerr"]), jmae=avg(S["jmae"]), tilt_max=(S["tilt"] * msk).amax(0), vxy_max=(S["vxy"] * msk).amax(0),
        tilt_p95=torch.quantile(torch.where(msk > 0, S["tilt"], torch.nan).nan_to_num(0.0), 0.95, dim=0), wrms=avg(S["wrms"]),
        zerr=avg(S["zerr"]), ss_tilt=avg(S["tilt"], ssm), ss_mpjpe=avg(S["mpjpe"], ssm), cagree=avg(S["cagree"]),
        slip=(acc["slip_sum"].sum(-1) / acc["slip_cnt"].sum(-1).clamp(min=1)), slip_ss=acc["slip_ss_sum"] / acc["slip_ss_cnt"].clamp(min=1),
        pen=acc["pen"], sat=(acc["sat"] / acc["n"].clamp(min=1)[:, None]),
        taumax=acc["taumax"],
    )
    rows = {}
    for ci, (name, _) in enumerate(CONDS):
        sl = slice(ci * reps, (ci + 1) * reps)
        sv = res_env["surv"][sl]
        ok = sv > 0.5
        def mean_ok(x):  # mean over surviving envs (failed envs have meaningless tracking)
            xs = x[sl]
            return float(xs[ok].mean()) if bool(ok.any()) else None
        sat = res_env["sat"][sl]
        rows[name] = dict(
            survival=float(sv.mean()), n=reps, mpjpe_mm=None if not bool(ok.any()) else 1000 * mean_ok(res_env["mpjpe"]),
            global_mpjpe_mm=None if not bool(ok.any()) else 1000 * mean_ok(res_env["gerr"]), jmae_rad=mean_ok(res_env["jmae"]),
            tilt_max_rad=float(res_env["tilt_max"][sl].max()), tilt_p95_rad=mean_ok(res_env["tilt_p95"]), gyro_rms=mean_ok(res_env["wrms"]), root_speed_xy_max=float(res_env["vxy_max"][sl].mean()), push_speed_after_kick=float(push_seen[sl].mean()),
            root_z_err_mm=None if not bool(ok.any()) else 1000 * mean_ok(res_env["zerr"]), ss_tilt_rad=mean_ok(res_env["ss_tilt"]),
            ss_mpjpe_mm=None if not bool(ok.any()) else 1000 * mean_ok(res_env["ss_mpjpe"]), contact_agreement=mean_ok(res_env["cagree"]),
            foot_slip_mps=mean_ok(res_env["slip"]), foot_slip_single_support_mps=mean_ok(res_env["slip_ss"]), foot_pen_max_mm=1000 * float(res_env["pen"][sl].max()),
            torque_sat_frac=float(sat[ok].mean()) if bool(ok.any()) else None,
            torque_sat_frac_ankle=float(sat[ok][:, grp["ankle"]].mean()) if bool(ok.any()) else None,
            torque_sat_frac_hip_roll=float(sat[ok][:, grp["hip_roll"]].mean()) if bool(ok.any()) else None,
            torque_peak_ratio_p95=float(torch.quantile(res_env["taumax"][sl].amax(-1), 0.95)),
        )
    nom = rows["nominal"]; rob = [rows[c] for c in ROBUST]
    g = lambda k: [r[k] for r in rob if r[k] is not None]
    surv_rob = float(np.mean([r["survival"] for r in rob]))
    tilt_st = float(np.mean(g("tilt_p95_rad"))) if g("tilt_p95_rad") else 1.0
    slip_m = float(np.mean(g("foot_slip_mps"))) if g("foot_slip_mps") else 1.0
    sat_m = float(np.mean(g("torque_sat_frac"))) if g("torque_sat_frac") else 1.0
    pen_max = max([nom["foot_pen_max_mm"]] + [r["foot_pen_max_mm"] for r in rob]) / 1000.0
    bal = 0.45 * surv_rob + 0.15 * (1 - clip01(tilt_st / 0.5)) + 0.15 * (1 - clip01(slip_m / 0.30)) + 0.15 * (1 - clip01(sat_m / 0.10)) + 0.10 * (1 - clip01(pen_max / 0.02))
    if nom["survival"] < 1.0:
        bal *= 0.5
    mp_nom = nom["mpjpe_mm"] if nom["mpjpe_mm"] is not None else 200.0
    jm_nom = nom["jmae_rad"] if nom["jmae_rad"] is not None else 1.0
    mp_rob = float(np.mean(g("mpjpe_mm"))) if g("mpjpe_mm") else 200.0
    fid = 0.6 * (1 - clip01((mp_nom - 15) / 65)) + 0.2 * (1 - clip01(jm_nom / 0.12)) + 0.2 * (1 - clip01((mp_rob - 15) / 90))
    if nom["survival"] < 1.0:
        fid *= 0.5
    scores = dict(balance=100 * bal, fidelity=100 * fid, combined=100 * (0.5 * bal + 0.5 * fid), survival_robust=surv_rob,
                  mpjpe_nominal_mm=mp_nom, mpjpe_robust_mean_mm=mp_rob, tilt_p95_stress=tilt_st, slip_mean=slip_m, torque_sat_mean=sat_m, pen_max_mm=1000 * pen_max)
    out = dict(checkpoint=str(ckpt_path), iteration=int(ck.get("iteration", -1)), motion_dir=str(motion_dir), reps=reps, seed=seed, scores=scores,
               conditions=rows, score_definition=SCORE_DEF, eval_seconds=time.time() - t0, guard="PD target clamped to hard limits +-0.005 rad")
    del env
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=8); ap.add_argument("--hard", action="store_true"); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--motion-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for c in a.ckpt:
        dst = out / (Path(c).parent.name + "__" + Path(c).stem + ".json") if False else out / (Path(c).stem + ".json")
        if dst.exists():
            print("skip", dst); continue
        r = evaluate(c, a.motion_dir, a.reps, a.seed, hard=a.hard)
        tmp = dst.with_suffix(".tmp"); tmp.write_text(json.dumps(r, indent=1)); os.replace(tmp, dst)
        s = r["scores"]
        print(f"{Path(c).stem} it{r['iteration']} balance {s['balance']:.1f} fidelity {s['fidelity']:.1f} combined {s['combined']:.1f} | surv_robust {s['survival_robust']:.2f} "
              f"mpjpe_nom {s['mpjpe_nominal_mm']:.1f}mm tilt95 {s['tilt_p95_stress']:.3f} slip {s['slip_mean']:.3f} sat {s['torque_sat_mean']:.3f} pen {s['pen_max_mm']:.1f}mm ({r['eval_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
