"""contact_refine.export - write a refined motion as a prepare-compatible SuperMimic npz.

body_pos_w / body_quat_w / body_lin_vel_w / body_ang_vel_w / joint_vel are ALL recomputed with the SAME converter code as `./pm prepare`
(supermimic.data.motion_convert.forward_kinematics, np.gradient velocities, so3 central-difference angular velocity) from the refined root / joints, so every field is
mutually consistent (round-trip of an unmodified file reproduces it to float32 precision: verified in the unit tests).  Frame-0 semantics: joint_vel[0] and
root_lin_vel[0] are the one-sided gradient of the refined trajectory (as prepare does); the padded clip repeats frame 0 with ZERO velocities (prepare.make_padded).
All original fields (fps, joint_names, body_names, source, input_fps, root_z_offset_m ...) are kept; `contactfix_info` (json) is added.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ..core import SUPERMIMIC, sha256

_mc = None


def mc():
    global _mc
    if _mc is None:
        sp = str(SUPERMIMIC / "src")
        if sp not in sys.path: sys.path.insert(0, sp)
        from supermimic.data import motion_convert
        _mc = motion_convert
    return _mc


def recompute_states(root_p, root_q, q, fps=50.0):
    """-> dict of npz arrays (float32) computed exactly like the converter."""
    m = mc(); dt = 1.0 / fps
    mot = m.ResampledMotion(fps=fps, root_pos=root_p, root_quat=root_q, joint_pos=q, root_lin_vel_w=np.gradient(root_p, dt, axis=0), root_ang_vel_w=m._so3_derivative(root_q, dt),
                            joint_vel=np.gradient(q, dt, axis=0))
    B = m.forward_kinematics(m.load_g1_model(), mot)
    return dict(joint_pos=q.astype(np.float32), joint_vel=mot.joint_vel.astype(np.float32), body_pos_w=B.pos_w.astype(np.float32), body_quat_w=B.quat_w.astype(np.float32),
                body_lin_vel_w=B.lin_vel_w.astype(np.float32), body_ang_vel_w=B.ang_vel_w.astype(np.float32))


def sign_align(new_q, ref_q):
    """keep the quaternion hemisphere of the original file (sign flips are cosmetic but make diffs readable)."""
    s = np.sign((new_q * ref_q).sum(-1, keepdims=True)); s[s == 0] = 1; return new_q * s


def write_refined(src: Path, dst: Path, root_p, q, info: dict):
    z = dict(np.load(src, allow_pickle=True)); names = [str(n) for n in z["body_names"]]; r = names.index("pelvis")
    root_q = z["body_quat_w"][:, r].astype(float); fps = float(z["fps"][0]) if "fps" in z else 50.0
    st = recompute_states(np.asarray(root_p, float), root_q, np.asarray(q, float), fps)
    st["body_quat_w"] = sign_align(st["body_quat_w"], z["body_quat_w"]).astype(np.float32)
    out = dict(z); out.update(st)
    for k in z:
        if k in st: assert out[k].shape == z[k].shape and out[k].dtype == z[k].dtype, k
    info = dict(info); info["source_sha256"] = sha256(src)
    out["contactfix_info"] = np.array(json.dumps(info, default=str))
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True); np.savez(dst, **out)
    return out
