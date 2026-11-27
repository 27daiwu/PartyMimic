"""contact_refine.geometry - g1mj kinematics layer: batched FK, sole sample points, point Jacobians, motion-file I/O helpers.

Sole model: the 14 end points of the 7 foot-collision capsules (g1mj collision group 3) expressed in the ankle-roll-link frame, each a sphere of radius 0.01 m.
The lowest point of a sample is `centre_z - radius`.  Samples are tagged heel / toe / inner / outer (by their position in the sole) so contact state can be
reported per sole region.  Everything here is dance-independent.
"""
from __future__ import annotations

import numpy as np

G = 9.81
SIDES = ("left", "right")
_C: dict = {}


def model(robot: str = "g1mj"):
    import mujoco
    if robot not in _C:
        from partymimic.robot import get_robot
        m = mujoco.MjModel.from_xml_path(get_robot(robot).mjcf_path)
        d = mujoco.MjData(m)
        ank = [m.body(f"{s}_ankle_roll_link").id for s in SIDES]
        loc, rad = [], []
        mujoco.mj_kinematics(m, d)
        d.qpos[:] = 0; d.qpos[3] = 1; mujoco.mj_kinematics(m, d)
        for s, b in zip(SIDES, ank):
            Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
            pts = []
            for g in range(m.ngeom):
                if m.geom_group[g] == 3 and m.geom_bodyid[g] == b:
                    ax = d.geom_xmat[g].reshape(3, 3)[:, 2]; h = m.geom_size[g, 1]; c = d.geom_xpos[g]
                    pts += [Rb.T @ (c + h * ax - pb), Rb.T @ (c - h * ax - pb)]
                    rad.append(m.geom_size[g, 0])
            loc.append(np.array(pts))
        loc = np.array(loc)  # (2,14,3) in ankle frame (left, right)
        r = float(np.mean(rad))
        names = [m.body(i).name for i in range(1, m.nbody)]
        jn = [m.joint(i).name for i in range(1, m.njnt)]
        # region tags from the sole layout (x forward, y lateral); left/right mirrored for inner/outer
        x = loc[0, :, 0]; tag = np.empty(14, "<U5")
        order = np.argsort(x); tag[:] = "mid"; tag[order[:4]] = "heel"; tag[order[-4:]] = "toe"
        _C[robot] = dict(m=m, d=d, ank=ank, loc=loc, radius=r, body_names=names, joint_names=jn, tag=tag,
                         sole_dz=float(loc[0, :, 2].mean() - r), jr=m.jnt_range[1:].copy())
    return _C[robot]


def joint_index(robot="g1mj"):
    jn = model(robot)["joint_names"]; return {n: i for i, n in enumerate(jn)}


LEG_JOINTS = [f"{s}_{j}_joint" for s in SIDES for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]


def leg_idx(side: int, robot="g1mj"):
    ji = joint_index(robot); return [ji[f"{SIDES[side]}_{j}_joint"] for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]


def fk(root_p, root_q, q, robot="g1mj", want_bodies=False):
    """batched FK. returns dict: pelvis (T,3), ank_p (T,2,3), ank_R (T,2,3,3), sole (T,2,14,3) sphere centres, com (T,3), [body_p (T,nb,3)]"""
    import mujoco
    M = model(robot); m, d = M["m"], M["d"]; T = len(q)
    ank_p = np.zeros((T, 2, 3)); ank_R = np.zeros((T, 2, 3, 3)); com = np.zeros((T, 3)); bp = np.zeros((T, m.nbody - 1, 3)) if want_bodies else None
    for t in range(T):
        d.qpos[:3] = root_p[t]; d.qpos[3:7] = root_q[t]; d.qpos[7:] = q[t]
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        com[t] = d.subtree_com[1]
        for k, b in enumerate(M["ank"]):
            ank_p[t, k] = d.xpos[b]; ank_R[t, k] = d.xmat[b].reshape(3, 3)
        if want_bodies: bp[t] = d.xpos[1:]
    sole = ank_p[:, :, None, :] + np.einsum("tfij,fnj->tfni", ank_R, M["loc"])
    return dict(pelvis=np.asarray(root_p), ank_p=ank_p, ank_R=ank_R, sole=sole, com=com, body_p=bp, radius=M["radius"])


def sole_height(F):
    """(T,2,14) lowest-point height of every sole sample above z=0 (no floor calibration)."""
    return F["sole"][..., 2] - F["radius"]


def point_jacobians(root_p, root_q, q, side, cols, robot="g1mj", pts=None):
    """position Jacobian (T, 14, 3, len(cols)) of the sole sample centres of foot `side` wrt the joints `cols` (indices into the 29 joints), and (T,14,3) positions.
    The Jacobian wrt the root translation is the identity (not returned)."""
    import mujoco
    M = model(robot); m, d = M["m"], M["d"]; T = len(q); b = M["ank"][side]; loc = M["loc"][side]; n = len(loc)
    J = np.zeros((T, n, 3, len(cols))); P = np.zeros((T, n, 3)); jp = np.zeros((3, m.nv)); cc = np.array(cols) + 6
    for t in range(T):
        d.qpos[:3] = root_p[t]; d.qpos[3:7] = root_q[t]; d.qpos[7:] = q[t]; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)  # mj_jac needs cdof
        Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
        for i in range(n):
            P[t, i] = pb + Rb @ loc[i]; mujoco.mj_jac(m, d, jp, None, P[t, i], b); J[t, i] = jp[:, cc]
    return J, P


def yaw_of(q):
    return np.arctan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]), 1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2))


def load_motion(path):
    z = np.load(path, allow_pickle=True); zz = {k: z[k] for k in z.files}
    M_ = model()
    if "body_names" not in zz: zz["body_names"] = np.array(M_["body_names"])  # LAFAN-style npz without name fields (pelvis first, g1mj order)
    if "joint_names" not in zz: zz["joint_names"] = np.array(M_["joint_names"])
    if "fps" not in zz: zz["fps"] = np.array([50.0], np.float32)
    z = zz; names = [str(n) for n in z["body_names"]]; r = names.index("pelvis")
    return dict(z=zz, fps=float(np.ravel(z["fps"])[0]), root_p=z["body_pos_w"][:, r].astype(float), root_q=z["body_quat_w"][:, r].astype(float),
                q=z["joint_pos"].astype(float), jv=z["joint_vel"].astype(float), body_names=names, joint_names=[str(n) for n in z["joint_names"]])


def runs(mask, min_len=1):
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None: s = i
        if not v and s is not None:
            if i - s >= min_len: out.append((s, i))
            s = None
    if s is not None and len(mask) - s >= min_len: out.append((s, len(mask)))
    return out
