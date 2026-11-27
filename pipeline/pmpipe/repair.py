"""PartyMimic Pipeline V1.2 - optional reference-motion repair: `./pm repair <dance>`.

Video-derived references (GVHMR/GMR style retargeting) carry two systematic defects that the joint angles do NOT have:
  * the root translation skates (planted feet slide; the root path is 2x longer than the planted feet allow),
  * the ground is wrong (the whole clip is aligned to its single lowest foot sample, so a static stance floats 20-30 mm and the floor drifts +-20 mm).
This module rebuilds ONLY the root translation (XY and Z) from RELIABLE contact:
  1. foot contact patch = soft-min-weighted centroid of the foot-sole samples (g1mj collision geometry, FK of the original joints/orientation),
  2. contact weights w(h) from the foot height above the CURRENT floor estimate (1 below `lo`, 0 above `hi`, plus static-stance frames),
  3. Z : per frame, lower the root so the weighted planted feet touch z=0 (iterated), interpolate over no-contact gaps, low-pass, never below the no-penetration bound,
  4. XY: root velocity = - d(foot position relative to the root)/dt of the weighted planted feet; gaps are bridged by interpolation; velocity low-passed and integrated
     from the ORIGINAL start position.
Preserved EXACTLY (asserted): frame count, fps, joint positions/velocities, root orientation, body orientations, angular velocities.  Changed: root XY/Z, hence all
body positions (rigid per-frame translation) and linear velocities.  Residual foot slip is whatever the joint motion itself implies (it cannot be removed without
changing the joints) and is reported.  The source npz is never modified; outputs go to results/<dance>/motions/<dance>_repaired{,_padded}/.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import quality as Q
from .core import Dance, now, write_json

VERSION = "repair v1.0"


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0); return x * x * (3 - 2 * x)


def contact_patch(K, tau=0.008):
    """(T,2,2) contact-patch centroid xy per foot: soft-min weighted by sole-sample clearance."""
    pts = K["pts"]; clr = pts[..., 2] - pts[..., 3]; cmin = clr.min(-1, keepdims=True); g = np.exp(-(clr - cmin) / tau)
    return (pts[..., :2] * g[..., None]).sum(-2) / g.sum(-1, keepdims=True), clr.min(-1)


def _fill(x, valid):
    """linear interpolation of x (T,...) over frames where valid is False (hold at the ends)."""
    t = np.arange(len(x)); out = np.array(x, float)
    if valid.sum() == 0: return out
    for c in range(out.shape[1] if out.ndim > 1 else 1):
        col = out[:, c] if out.ndim > 1 else out
        col[:] = np.interp(t, t[valid], col[valid])
    return out


def reconstruct(z: dict, robot="g1mj", lo=0.008, hi=0.030, iters=4, fc_z=4.0, fc_xy=6.0, wmin=0.3) -> dict:
    fps = float(z["fps"][0]) if "fps" in z else 50.0; dt = 1 / fps
    names = [str(n) for n in z["body_names"]]; r = names.index("pelvis")
    root_p, root_q, q, jv = z["body_pos_w"][:, r].astype(float), z["body_quat_w"][:, r].astype(float), z["joint_pos"].astype(float), z["joint_vel"].astype(float)
    T = len(q); K = Q.fk_series(root_p, root_q, q, robot, want_pts=True); patch, clr = contact_patch(K)
    rms = np.sqrt((jv ** 2).mean(1)); rsp = np.linalg.norm(np.gradient(root_p, dt, axis=0), axis=1); static = (rms < 0.25) & (rsp < 0.10)
    static = np.convolve(static.astype(float), np.ones(5) / 5, "same") > 0.99
    cal = Q.calibrate_floor(K["clr"], jv, root_p, fps); dz = np.full(T, -float(np.mean(cal["offset"])))
    cmin_t = clr.min(1)
    for it in range(iters):
        h = clr + dz[:, None]; w = 1 - _smoothstep((h - lo) / (hi - lo)); w = np.where(static[:, None] & (h < 0.06), np.maximum(w, 1.0), w)
        sw = w.sum(1); tgt = -(w * clr).sum(1) / np.maximum(sw, 1e-9); ok = sw >= wmin
        d = _fill(tgt, ok); d = Q.lowpass(d, fc_z, fps); dz = np.maximum(d, -cmin_t)  # never below the no-penetration bound
    h = clr + dz[:, None]; w = 1 - _smoothstep((h - lo) / (hi - lo)); w = np.where(static[:, None] & (h < 0.06), np.maximum(w, 1.0), w)
    # ---- XY from the planted feet
    rel = patch - root_p[:, None, :2]  # contact patch relative to the original root (world axes) - independent of the root translation
    drel = np.diff(rel, axis=0); Wf = np.minimum(w[1:], w[:-1]); sw = Wf.sum(1)
    vraw = -(Wf[..., None] * drel).sum(1) / np.maximum(sw, 1e-9)[:, None] / dt
    okv = sw >= wmin; v = _fill(vraw, okv); v = Q.lowpass(v, fc_xy, fps)
    xy = np.zeros((T, 2)); xy[0] = root_p[0, :2]; xy[1:] = root_p[0, :2] + np.cumsum(v * dt, axis=0)
    new_p = np.concatenate([xy, (root_p[:, 2] + dz)[:, None]], 1)
    return dict(new_root=new_p, old_root=root_p, dz=dz, weights=w, static=static, patch=patch, clr=clr, cal=cal, gap_frac=float((~okv).mean()), K=K,
                params=dict(lo_mm=lo * 1000, hi_mm=hi * 1000, iters=iters, fc_z_hz=fc_z, fc_xy_hz=fc_xy, wmin=wmin))


def repair_npz(src: Path, dst: Path, **kw) -> dict:
    z = dict(np.load(src, allow_pickle=True)); fps = float(z["fps"][0]) if "fps" in z else 50.0; dt = 1 / fps
    names = [str(n) for n in z["body_names"]]; r = names.index("pelvis")
    R = reconstruct(z, **kw); delta = R["new_root"] - R["old_root"]
    out = dict(z)
    out["body_pos_w"] = (z["body_pos_w"] + delta[:, None, :]).astype(z["body_pos_w"].dtype)
    dv = np.gradient(R["new_root"], dt, axis=0) - np.gradient(R["old_root"], dt, axis=0)
    out["body_lin_vel_w"] = (z["body_lin_vel_w"] + dv[:, None, :]).astype(z["body_lin_vel_w"].dtype)
    for k in ("joint_pos", "joint_vel", "body_quat_w", "body_ang_vel_w", "fps", "joint_names", "body_names"):  # exactness asserts
        assert np.array_equal(out[k], z[k]), f"repair changed {k}"
    assert out["body_pos_w"].shape == z["body_pos_w"].shape
    info = dict(version=VERSION, time=now(), source=str(src), params=R["params"], gap_frac=R["gap_frac"], max_dxy_m=float(np.abs(delta[:, :2]).max()), max_dz_mm=float(np.abs(delta[:, 2]).max() * 1000),
                dz_start_mm=float(delta[0, 2] * 1000), dz_end_mm=float(delta[-1, 2] * 1000), preserved=["frame_count", "fps", "joint_pos", "joint_vel", "body_quat_w", "body_ang_vel_w"])
    out["repair_info"] = np.array(json.dumps(info)); dst.parent.mkdir(parents=True, exist_ok=True); np.savez(dst, **out)
    return dict(info=info, R=R)


def make_dance_like(dance: Dance, tag="repaired"):
    name = f"{dance.name}_{tag}"
    return SimpleNamespace(name=name, npz=dance.motions / name / f"{name}.npz", padded_dir=dance.motions / f"{name}_padded", padded_npz=dance.motions / f"{name}_padded" / f"{name}_padded.npz", motions=dance.motions)


def repair(dance: Dance, force=False, log=print, **kw) -> dict:
    kw = {k: v for k, v in kw.items() if v is not None}
    from .prepare import make_padded
    rd = make_dance_like(dance)
    if rd.npz.exists() and rd.padded_npz.exists() and not force:
        return dict(dance=rd, info=json.loads(str(np.load(rd.npz, allow_pickle=True)["repair_info"])), reused=True)
    res = repair_npz(dance.npz, rd.npz, **kw); rd.padded_dir.mkdir(parents=True, exist_ok=True)
    if rd.padded_npz.exists(): rd.padded_npz.unlink()
    make_padded(str(rd.npz), str(rd.padded_npz)); log("repaired", rd.npz, rd.padded_npz)
    return dict(dance=rd, info=res["info"], reused=False)


# ------------------------------------------------------------------------------------------------------------------------------ metrics
def ref_contact_labels(npz, feet=("left_ankle_roll_link", "right_ankle_roll_link"), margin=0.015, vmax=0.5, q=0.05):
    """EXACT numpy replica of MotionLib.set_ref_contact (the labels the trainer's contact reward / evaluator contact_agreement use): ankle-origin height below the
    clip's own q-quantile floor + margin, and horizontal speed < vmax."""
    z = np.load(npz, allow_pickle=True); names = [str(n) for n in z["body_names"]]; idx = [names.index(f) for f in feet]
    h = z["body_pos_w"][:, idx, 2]; vh = np.linalg.norm(z["body_lin_vel_w"][:, idx, :2], axis=-1); fl = np.quantile(h, q, axis=0)
    return (h < fl + margin) & (vh < vmax)


def motion_metrics(npz, robot="g1mj", sim_contact=None, pad=0) -> dict:
    """geometry-only metrics of a reference (no policy): slip of planted feet, root-vs-feet consistency, penetration, floating, CoM acceleration, label statistics.
    sim_contact: optional (T,2) bool force-based contact (MuJoCo rollout, independent of the reference's own heights) to validate contact labels."""
    z = np.load(npz, allow_pickle=True); fps = float(z["fps"][0]); dt = 1 / fps; names = [str(n) for n in z["body_names"]]; r = names.index("pelvis")
    rp, rq, q, jv = z["body_pos_w"][:, r], z["body_quat_w"][:, r], z["joint_pos"], z["joint_vel"]; T = len(q); s = slice(pad, T - pad) if pad else slice(0, T)
    K = Q.fk_series(rp, rq, q, robot, want_pts=True); patch, clr = contact_patch(K)
    cal = Q.calibrate_floor(K["clr"], jv, rp, fps); h = K["clr"] - cal["offset"]  # height above the clip's own static-stance floor
    h_abs = clr  # height above z=0
    sure = h < 0.010; v = np.linalg.norm(np.gradient(patch, dt, axis=0), axis=-1)
    imp = Q.dead_reckon(rp[:, :2], K["ank"][:, :, :2], h < 0.025)
    com = K["com"]; zmp, a = Q.kin_zmp(com, fps, 8.0)
    mn = h_abs.min(1) * 1000; win = [float(np.median(mn[b:b + 50])) for b in range(0, T - 49, 50)]
    lab = ref_contact_labels(npz); cps = Q.contact_point_slip(K["pts"], pad, T - pad, floor_off=cal["offset"], thr=0.010)  # V1.4: contact-point slip on sure-contact frames
    out = dict(slip_point_mps=cps, frames=T, floor_offset_mm=(cal["offset"] * 1000).round(1).tolist(), static_stance_clearance_mm=float(np.median(h_abs[:40].min(1)) * 1000), min_clearance_mm=float(h_abs.min() * 1000),
               frac_below_minus1mm=float((h_abs.min(1) < -0.001).mean()), floating_window_median_range_mm=[round(min(win), 1), round(max(win), 1)],
               slip_sure_contact_mps=dict(mean=float(v[s][sure[s]].mean()), p95=float(np.percentile(v[s][sure[s]], 95)), n=int(sure[s].sum())),
               root_path_m=Q.path_len(rp[s, :2]), implied_path_m=Q.path_len(imp[s]), root_vs_implied=dict(mean_m=float(np.linalg.norm(rp[:, :2] - imp, axis=1)[s].mean()), max_m=float(np.linalg.norm(rp[:, :2] - imp, axis=1)[s].max())),
               com_acc_xy_p95=float(np.percentile(np.linalg.norm(a[s, :2], axis=1), 95)), com_speed_max=float(np.linalg.norm(np.gradient(com[s, :2], dt, axis=0), axis=1).max()),
               root_height_range_m=[float(rp[s, 2].min()), float(rp[s, 2].max())], label_contact_frac=float(lab[s].mean()), label_double=float(lab[s].all(1).mean()), label_none=float((~lab[s].any(1)).mean()))
    if sim_contact is not None:
        n = min(len(sim_contact), T); ss = slice(pad, n - pad) if pad else slice(0, n)
        out["label_vs_sim_force_contact"] = dict(agreement=float((lab[:n][ss] == sim_contact[:n][ss]).mean()), ref_contact_sim_unloaded=float((lab[:n][ss] & ~sim_contact[:n][ss]).mean()), ref_swing_sim_loaded=float((~lab[:n][ss] & sim_contact[:n][ss]).mean()))
        hc = h_abs[:n][ss]; sc = sim_contact[:n][ss]
        out["sim_loaded_foot_height_mm"] = dict(median=float(np.median(hc[sc]) * 1000), p90=float(np.percentile(hc[sc], 90) * 1000))  # how far above z=0 the reference foot is while the robot actually stands on it
        out["slip_sim_loaded_mps"] = dict(mean=float(v[:n][ss][sc].mean()), p95=float(np.percentile(v[:n][ss][sc], 95)))
    return out
