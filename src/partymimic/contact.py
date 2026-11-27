"""V1.7 (C2) contact-aware reward options (all OFF by default = the V1.6 reward, bit for bit).

ContactCfg collects the opt-in switches of train.py (--contact_source / --contact_feat / --contact_conf_weight / --cr_*). The per-frame labels come from a
contact_refine sidecar (pmpipe/contact_refine/labels.py) attached to the MotionLib (MotionLib.set_contact_labels).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOLE_OBS_N = 14  # sole sample points per foot (the 14 end points of the 7 foot-collision capsules, ankle frame), sphere radius 0.01 m


@dataclass
class ContactCfg:
    source: str = "legacy"  # c_ref: "legacy" = MotionLib.set_ref_contact (clip-floor quantile + speed); "refine" = sidecar stance / swing / uncertain (soft on uncertain)
    feat: str = "ankle"  # c_sim feature: "ankle" = ankle-roll-link ORIGIN height < CONTACT_Z (V1.6); "sole" = lowest foot-sole point (tilt invariant, no flat-foot bias)
    sole_h: float = 0.012  # --contact_feat sole: sole height at which c_sim = 0.5 [m]
    sole_tau: float = 0.005  # sigmoid width [m]
    conf_weight: float = 0.0  # r_contact *= (1 - cw) + cw * conf  (cw = 0: off)
    hi_conf: float = 0.7  # "high confidence" label threshold for the opt-in terms
    stable_k: int = 3  # stable stance = high-confidence stance for frames t-k .. t+k
    # opt-in terms (weights; 0 = off). Every term is a [0,1] kernel x weight x dt (penalties are negative)
    w_consist: float = 0.0  # (a) conf-weighted stance consistency + swing lift-off
    w_slip: float = 0.0  # (b) narrow slip kernel on stable stance
    slip_std: float = 0.1  # [m/s]
    w_smooth: float = 0.0  # (c) c_sim change-rate (flicker) kernel on stable stance
    smooth_std: float = 2.0  # [1/s]
    w_foot_xy: float = 0.0  # (d1) stable-stance foot world XY vs reference footprint (L1 shape, capped)
    foot_cap: float = 0.5  # [m]
    w_root_l1: float = 0.0  # (d2) root world XY drift, L1 shape (non-vanishing gradient, capped)
    root_cap: float = 1.0  # [m]
    zmp_relax: float = 0.0  # (e) body / root POSITION kernels widen by (1 + zmp_relax * infeasibility[f]) (<= 1 + zmp_relax)
    zmp_lo: float = 0.10  # kinematic ZMP outside the support polygon by more than this [m] starts to relax (C1: kin-ZMP is ~3 cm too strict; only the worst frames count) ...
    zmp_hi: float = 0.25  # ... and fully relaxes at this margin [m]

    @property
    def terms(self) -> bool:
        return any(w != 0 for w in (self.w_consist, self.w_slip, self.w_smooth, self.w_foot_xy, self.w_root_l1))

    @property
    def needs_labels(self) -> bool:
        return self.source == "refine" or self.conf_weight > 0 or self.terms or self.zmp_relax > 0


def sole_points(mjcf_path: str, feet=("left_ankle_roll_link", "right_ankle_roll_link")):
    """(2, 14, 3) sole sample centres in the ankle-roll-link frame and the sphere radius (same construction as contact_refine.geometry.model)."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(mjcf_path)
    d = mujoco.MjData(m)
    d.qpos[:] = 0
    d.qpos[3] = 1
    mujoco.mj_kinematics(m, d)
    loc, rad = [], []
    for n in feet:
        b = m.body(n).id
        Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
        pts = []
        for g in range(m.ngeom):
            if m.geom_group[g] == 3 and m.geom_bodyid[g] == b:
                ax = d.geom_xmat[g].reshape(3, 3)[:, 2]
                h = m.geom_size[g, 1]
                c = d.geom_xpos[g]
                pts += [Rb.T @ (c + h * ax - pb), Rb.T @ (c - h * ax - pb)]
                rad.append(m.geom_size[g, 0])
        loc.append(np.array(pts))
    return np.array(loc), float(np.mean(rad))


# ---------------------------------------------------------------------------------------------------------------- sidecar I/O (writer: pmpipe/contact_refine/labels.py)
_ARRAYS = ("stance_prob", "conf", "state", "inferred", "zmp_margin_m", "ground_mm")


def _digest(d: dict) -> str:
    import hashlib

    h = hashlib.sha256()
    for k in _ARRAYS:
        a = np.ascontiguousarray(d[k])
        h.update(k.encode()), h.update(str(a.dtype).encode()), h.update(str(a.shape).encode()), h.update(np.nan_to_num(a.astype(np.float64), nan=-9e9).tobytes())
    return h.hexdigest()


def _file_sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_sidecar(path: str, motion_npz: str | None = None, expect_frames: int | None = None) -> dict:
    """Read `<clip>_contact.npz`; verifies the payload digest, the sha256 of the motion file it was made for and the frame count."""
    import json

    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in _ARRAYS}
    meta = json.loads(str(z["meta"]))
    assert _digest(d) == meta["payload_sha256"], f"{path}: payload sha256 mismatch"
    if motion_npz is not None:
        assert _file_sha256(motion_npz) == meta["motion_sha256"], f"{path}: made for another motion file ({meta['motion_npz']})"
    if expect_frames is not None:
        assert len(d["state"]) == expect_frames, f"{path}: {len(d['state'])} label frames vs {expect_frames} motion frames"
    d["meta"] = meta
    return d


def attach_labels(motion, label_dir: str, feet: list[int], mjcf_path: str, cfg: ContactCfg, mirror_clips: bool = False) -> dict:
    """Load the sidecars of every clip of `motion` from label_dir (`<clip>_contact.npz`), check alignment (sha256 + frames), attach to the MotionLib.
    Returns {clip: meta} for the run log. Clips resampled to 50 Hz must keep their frame count (labels are made at the motion's own 50 Hz grid)."""
    import os

    per, metas = [], {}
    for i, (name, f) in enumerate(zip(motion.names, motion.files)):
        p = os.path.join(label_dir, f"{name}_contact.npz")
        assert os.path.exists(p), f"--contact_source refine: sidecar {p} missing (make it with `./pm v17-labels`)"
        d = load_sidecar(p, motion_npz=f, expect_frames=int(motion.clip_len[i]))
        per.append(d)
        metas[name] = dict(sidecar=p, label_set=d["meta"]["label_set"], payload_sha256=d["meta"]["payload_sha256"][:12], motion_sha256=d["meta"]["motion_sha256"][:12])
    loc, r = sole_points(mjcf_path)
    motion.sole_loc, motion.sole_r = loc, r
    motion.set_contact_labels(per, feet, loc, r, hi_conf=cfg.hi_conf, stable_k=cfg.stable_k, zmp_lo=cfg.zmp_lo, zmp_hi=cfg.zmp_hi)
    return metas
