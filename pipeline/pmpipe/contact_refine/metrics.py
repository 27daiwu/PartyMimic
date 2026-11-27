"""contact_refine.metrics - quantitative comparison original vs refined (all numbers feed qa.py, the report and the figures).

Conventions: heights are above z=0 for the REFINED motion and above the estimated ground curve g(t) for the ORIGINAL (the most favourable honest reading of the
original; the plain z=0 reading is reported next to it).  Slip = V1.4 contact-point speed: lowest end point of each of the 7 sole capsules, counted while that point is
< thr over the floor on two consecutive frames and its speed < 3 m/s.
"""
from __future__ import annotations

import numpy as np
import yaml

from .. import quality as Q
from ..core import CONFIGS
from . import geometry as Gm

DEG = 180 / np.pi


def _so3(q, dt):
    from .export import mc
    return mc()._so3_derivative(q, dt)


def hw_limits():
    y = yaml.safe_load(open(CONFIGS / "hardware/g1_29dof_rev1_0.yaml")); J = sorted(y["joints"], key=lambda j: j["index"])
    return dict(vel=np.array([j["velocity_rad_s"] for j in J]), lo=np.array([j["lower_rad"] for j in J]), hi=np.array([j["upper_rad"] for j in J]), names=[j["name"] for j in J])


def slip_series(sole, radius, floor, fps, thr=0.010, vmax=3.0, mask=None):
    """returns (T-1,2,7) speeds [m/s] and boolean validity (contact on two consecutive frames, mask on both frames)"""
    T = len(sole); dt = 1 / fps; sp_all = np.zeros((T - 1, 2, 7)); ok = np.zeros((T - 1, 2, 7), bool)
    for f in (0, 1):
        p = sole[:, f].reshape(T, 7, 2, 3); i = np.argmin(p[..., 2], axis=2); low = np.take_along_axis(p, i[..., None, None], 2)[:, :, 0]
        con = (low[..., 2] - radius - floor[:, None]) < thr; xy = low[..., :2]; sp = np.linalg.norm(xy[1:] - xy[:-1], axis=-1) / dt
        o = con[1:] & con[:-1] & (sp < vmax)
        if mask is not None: o &= (mask[1:, f] & mask[:-1, f])[:, None]
        sp_all[:, f] = sp; ok[:, f] = o
    return sp_all, ok


def _stats(v):
    return dict(mean=float(v.mean()), p50=float(np.median(v)), p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)), max=float(v.max()), n=int(len(v))) if len(v) else dict(mean=float("nan"), p50=float("nan"), p95=float("nan"), max=float("nan"), n=0)


def ground_stats(sole, radius, floor, stance):
    H = sole[..., 2] - radius - floor[:, None, None]; e = H.min(-1)
    se = e[stance]; pen = np.maximum(-H.min((1, 2)), 0)
    return dict(stance_abs_height_p95_mm=float(np.percentile(np.abs(se), 95) * 1000) if len(se) else float("nan"), stance_height_mean_mm=float(se.mean() * 1000) if len(se) else float("nan"),
                stance_height_p50_mm=float(np.median(se) * 1000) if len(se) else float("nan"), penetration_max_mm=float(pen.max() * 1000), penetration_gt2mm_frac_pct=float((pen > 0.002).mean() * 100),
                penetration_gt5mm_frames=int((pen > 0.005).sum()), frac_frames_foot_below_10mm=float((H.min((1, 2)) < -0.010).mean()),
                hover_frac_both_feet_gt10mm=float((e.min(1) > 0.010).mean()))


def switches(e, thr=0.012):
    c = e < thr; return int((np.diff(c.astype(int), axis=0) != 0).sum())


def zmp_margin(F, floor, fps, stance, conf_ok):
    """kinematic-ZMP margin to the stance support polygon (stance feet, sole samples < 15 mm over the floor + sphere radius), only where stance & conf_ok."""
    com = F["com"]; zmp, _ = Q.kin_zmp(com, fps, 8.0, floor_z=0.0); zmp = com[:, :2] - ((com[:, 2:3] - floor[:, None]) / (Q.G + _[:, 2:3])) * _[:, :2]
    T = len(com); m = np.full(T, np.nan); r = F["radius"]
    for t in range(T):
        if not conf_ok[t]: continue
        P = []
        for f in (0, 1):
            if stance[t, f]:
                s = F["sole"][t, f]; h = s[:, 2] - r - floor[t]
                for x, y, _z in s[h < 0.015]: P += [(x + r * np.cos(a), y + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        if len(P) >= 3: m[t] = Q.hull_margin(P, zmp[t])
    return m


def compare(O, N, A0, A1, g0, d, u, p, fps, flights0=None):
    """O: original motion dict (geometry.load_motion), N: refined arrays dict(root_p, q, body_pos_w, joint_vel ...), A0/A1: analyses (dict F, ident) of original / refined."""
    lim = hw_limits(); T = len(O["q"]); res = {}
    F0, F1 = A0["F"], A1["F"]; r1 = A1["ident"]; st = A0["ident"]["stance"]; zero = np.zeros(T)  # the stance mask the solver pinned (original identification): the strict reading, no self-selection by the refined heights
    st1 = r1["stance"]
    # ---- slip (gate: same stance-frame mask for both versions)
    s0, o0 = slip_series(F0["sole"], F0["radius"], g0, fps, 0.010, mask=st); s1, o1 = slip_series(F1["sole"], F1["radius"], zero, fps, 0.010, mask=st)
    s0a, o0a = slip_series(F0["sole"], F0["radius"], g0, fps, 0.010); s1a, o1a = slip_series(F1["sole"], F1["radius"], zero, fps, 0.010)
    s0z, o0z = slip_series(F0["sole"], F0["radius"], zero, fps, 0.010)
    a, b = _stats(s0[o0]), _stats(s1[o1])
    res["slip"] = dict(original=a, refined=b, p95_reduction_pct=float((1 - b["p95"] / a["p95"]) * 100), mean_reduction_pct=float((1 - b["mean"] / a["mean"]) * 100),
                       v14_all_contact_frames=dict(original_vs_ground_curve=_stats(s0a[o0a]), original_vs_z0=_stats(s0z[o0z]), refined=_stats(s1a[o1a])),
                       definition="V1.4 contact-point speed, points <10 mm over the floor on consecutive frames, restricted to the stance frames identified on the ORIGINAL (the frames the solver pinned), same mask for both versions")
    # supplementary slip views (not gated): fixed sample identity, and interior-only (>=3 frames away from a stance-run boundary)
    dist = np.zeros((T, 2), int)
    for f in (0, 1):
        for a_, b_ in Gm.runs(st[:, f]): dist[a_:b_, f] = np.minimum(np.arange(a_, b_) - a_, b_ - 1 - np.arange(a_, b_))
    inter = dist >= 3; si0, oi0 = slip_series(F0["sole"], F0["radius"], g0, fps, 0.010, mask=st & inter); si1, oi1 = slip_series(F1["sole"], F1["radius"], zero, fps, 0.010, mask=st & inter)
    def pp(sole, floor):
        v = []
        for f in (0, 1):
            P = sole[:, f]; h = P[..., 2] - radius_ - floor[:, None]; xy = P[..., :2]; sp_ = np.linalg.norm(xy[1:] - xy[:-1], axis=-1) / (1 / fps)
            ok = (h[1:] < 0.010) & (h[:-1] < 0.010) & (st[1:, f] & st[:-1, f])[:, None] & (sp_ < 3.0); v.append(sp_[ok])
        return np.concatenate(v)
    radius_ = F0["radius"]; pp0, pp1 = pp(F0["sole"], g0), pp(F1["sole"], zero)
    res["slip"]["interior_only"] = dict(original=_stats(si0[oi0]), refined=_stats(si1[oi1]), note=">=3 frames from the start/end of a stance run: excludes touch-down / lift-off frames and pivots")
    res["slip"]["per_point_identity"] = dict(original=_stats(pp0), refined=_stats(pp1), note="each of the 14 sole samples tracked by identity (no lowest-point switching), same stance mask")
    res["ground_refined_mask"] = ground_stats(F1["sole"], F1["radius"], zero, st1)
    res["ground"] = dict(original_vs_ground_curve=ground_stats(F0["sole"], F0["radius"], g0, st), original_vs_z0=ground_stats(F0["sole"], F0["radius"], zero, st), **ground_stats(F1["sole"], F1["radius"], zero, st))
    # ---- joints
    du = np.abs(N["q"] - O["q"]) * DEG; names = O["joint_names"]; legi = np.arange(12); ank = np.array([4, 5, 10, 11]); noank = np.array([0, 1, 2, 3, 6, 7, 8, 9]); up = np.arange(12, 29)
    res["joints"] = dict(upper_max_deg=float(du[:, up].max()), leg_noankle_max_deg=float(du[:, noank].max()), ankle_max_deg=float(du[:, ank].max()), rms_all_deg=float(np.sqrt((du ** 2).mean())),
                         rms_legs_deg=float(np.sqrt((du[:, legi] ** 2).mean())), per_joint_rms_deg={n: float(np.sqrt((du[:, i] ** 2).mean())) for i, n in enumerate(names)},
                         per_joint_max_deg={n: float(du[:, i].max()) for i, n in enumerate(names)}, frames_changed_gt1deg=float((du[:, legi].max(1) > 1).mean()))
    # ---- dynamics
    dt = 1 / fps; jv0 = np.gradient(O["q"], dt, axis=0); jv1 = N["joint_vel"].astype(float); ja0 = np.gradient(jv0, dt, axis=0); ja1 = np.gradient(jv1, dt, axis=0)
    exc0 = np.abs(jv0) > lim["vel"][None]; exc1 = np.abs(jv1) > lim["vel"][None]
    ok_each = [bool(np.abs(jv1[:, j]).max() <= max(lim["vel"][j], 1.05 * np.abs(jv0[:, j]).max()) + 1e-9) for j in range(29)]
    rp0, rp1 = O["root_p"], N["root_p"]; ra0 = np.linalg.norm(np.gradient(np.gradient(rp0, dt, axis=0), dt, axis=0), axis=1); ra1 = np.linalg.norm(np.gradient(np.gradient(rp1, dt, axis=0), dt, axis=0), axis=1)
    lo_v = (jv0[:, legi]); res["dynamics"] = dict(joint_vel_ok=bool(all(ok_each[:12]) and exc1[:, :12].sum() <= exc0[:, :12].sum()), joint_vel_ok_all=bool(all(ok_each)), exceed_frames_legs=[int(exc0[:, :12].any(1).sum()), int(exc1[:, :12].any(1).sum())],
                          exceed_frames_all=[int(exc0.any(1).sum()), int(exc1.any(1).sum())], leg_jv_max=[float(np.abs(jv0[:, legi]).max()), float(np.abs(jv1[:, legi]).max())],
                          joint_acc_p999_ratio=float(np.percentile(np.abs(ja1[:, legi]), 99.9) / np.percentile(np.abs(ja0[:, legi]), 99.9)), joint_acc_p999=[float(np.percentile(np.abs(ja0[:, legi]), 99.9)), float(np.percentile(np.abs(ja1[:, legi]), 99.9))],
                          root_acc_p995_ratio=float(np.percentile(ra1, 99.5) / np.percentile(ra0, 99.5)), root_acc_p995=[float(np.percentile(ra0, 99.5)), float(np.percentile(ra1, 99.5))], root_acc_max=[float(ra0.max()), float(ra1.max())],
                          jerk_root_p99=[float(np.percentile(np.linalg.norm(np.gradient(np.gradient(np.gradient(rp, dt, axis=0), dt, axis=0), dt, axis=0), axis=1), 99)) for rp in (rp0, rp1)],
                          joint_range_new_violations=int((((N["q"] < lim["lo"][None] - 1e-6) | (N["q"] > lim["hi"][None] + 1e-6)) & ~((O["q"] < lim["lo"][None] - 1e-6) | (O["q"] > lim["hi"][None] + 1e-6))).sum()))
    conf_ok = (st & ~r1["inferred"]).any(1) & (r1["conf"].max(1) > 0.6)
    m0 = zmp_margin(F0, g0, fps, st, conf_ok); m1 = zmp_margin(F1, zero, fps, st, conf_ok); okm = np.isfinite(m0) & np.isfinite(m1)
    res["dynamics"].update(zmp_frames=int(okm.sum()), zmp_margin_median_mm=[float(np.median(m0[okm]) * 1000) if okm.any() else None, float(np.median(m1[okm]) * 1000) if okm.any() else None],
                           zmp_margin_median_delta_mm=float((np.median(m1[okm]) - np.median(m0[okm])) * 1000) if okm.any() else 0.0,
                           zmp_outside_frac=[float((m0[okm] < 0).mean()) if okm.any() else None, float((m1[okm] < 0).mean()) if okm.any() else None])
    # ---- root
    dd = rp1 - rp0; dv = np.gradient(dd, dt, axis=0); dxy = np.linalg.norm(dd[:, :2], axis=1)
    res["root"] = dict(dev_xy_max_m=float(dxy.max()), dev_xy_mean_m=float(dxy.mean()), dev_z_range_mm=[float(dd[:, 2].min() * 1000), float(dd[:, 2].max() * 1000)], dz_in_bounds=bool(dd[:, 2].min() >= p.root_z_bound_m[0] - 1e-9 and dd[:, 2].max() <= p.root_z_bound_m[1] + 1e-9),
                       dv_p99_mps=float(np.percentile(np.linalg.norm(dv, axis=1), 99)), dv_max_mps=float(np.linalg.norm(dv, axis=1).max()), path_orig_m=Q.path_len(rp0[:, :2]), path_new_m=Q.path_len(rp1[:, :2]))
    imp0 = Q.dead_reckon(rp0[:, :2], F0["ank_p"][:, :, :2], st); imp1 = Q.dead_reckon(rp1[:, :2], F1["ank_p"][:, :, :2], st)
    res["root"]["consistency"] = dict(implied_path_m=[Q.path_len(imp0), Q.path_len(imp1)], mismatch_mean_m=[float(np.linalg.norm(rp0[:, :2] - imp0, axis=1).mean()), float(np.linalg.norm(rp1[:, :2] - imp1, axis=1).mean())],
                                      mismatch_max_m=[float(np.linalg.norm(rp0[:, :2] - imp0, axis=1).max()), float(np.linalg.norm(rp1[:, :2] - imp1, axis=1).max())], note="same stance mask for both")
    # ---- style (pelvis-relative body positions)
    from scipy.spatial.transform import Rotation as Rot
    def rel(bp, rp, rq): return np.einsum("tji,tbj->tbi", Rot.from_quat(rq[:, [1, 2, 3, 0]]).as_matrix(), bp - rp[:, None])
    r0 = rel(O["z"]["body_pos_w"].astype(float), rp0, O["root_q"]); r1_ = rel(N["body_pos_w"].astype(float), rp1, O["root_q"]); e = np.linalg.norm(r1_ - r0, axis=-1)
    res["style"] = dict(pelvis_rel_mpjpe_mm=float(e.mean() * 1000), leg_pelvis_rel_mpjpe_mm=float(e[:, 1:13].mean() * 1000), upper_pelvis_rel_mpjpe_mm=float(e[:, 13:].mean() * 1000), ankle_pelvis_rel_mm_mean_max=[float(e[:, [6, 12]].mean() * 1000), float(e[:, [6, 12]].max() * 1000)],
                        max_any_body_mm=float(e.max() * 1000), per_body_mean_mm={n: float(e[:, i].mean() * 1000) for i, n in enumerate(O["body_names"])})
    # anchored (XY+yaw anchored, root height kept) MPJPE for reference to the project convention
    sh = np.concatenate([rp1[:, :2] - rp0[:, :2], np.zeros((T, 1))], 1)[:, None]  # root yaw is unchanged, so XY/yaw anchoring = removing the XY translation
    res["style"]["anchored_xy_yaw_mpjpe_mm_includes_root_z_shift"] = float(np.linalg.norm((N["body_pos_w"].astype(float) - sh) - O["z"]["body_pos_w"].astype(float), axis=-1).mean() * 1000)
    # ---- flight / contact semantics
    f0 = A0["ident"]["flight"]; f1 = r1["flight"]; m0f, m1f = f0["mask"], f1["mask"]
    dil = np.convolve(m0f.astype(float), np.ones(5), "same") > 0
    pres = True; chg = []
    for c_ in f0["candidates"]:
        if not c_["true_flight"]: continue
        a_, b_ = c_["start"], c_["end"]; z0 = O["root_p"][a_ - 2:b_ + 2, 2]; z1 = N["root_p"][a_ - 2:b_ + 2, 2]; dz = (z1 - z0); lin = np.linspace(dz[0], dz[-1], len(dz))
        shape = float(np.abs(dz - lin).max() * 1000)
        h0 = F0["com"][a_:b_, 2].max() - F0["com"][a_ - 1, 2]; h1 = F1["com"][a_:b_, 2].max() - F1["com"][a_ - 1, 2]
        chg.append(dict(start=a_, end=b_, root_shape_change_mm=shape, apex_rise_change_mm=float(abs(h1 - h0) * 1000), still_flight=bool(m1f[a_:b_].mean() > 0.6)))
        pres &= shape <= 20 and abs(h1 - h0) * 1000 <= 20 and chg[-1]["still_flight"]
    res["flight"] = dict(n_true_flights_orig=sum(c_["true_flight"] for c_ in f0["candidates"]), n_candidates_orig=len(f0["candidates"]), n_true_flights_refined=sum(c_["true_flight"] for c_ in f1["candidates"]), frames_flight_orig=int(m0f.sum()),
                         frames_flight_refined=int(m1f.sum()), false_flight_frames_orig=int(f0["false_mask"].sum()), preserved=bool(pres), no_new_flight=bool(not (m1f & ~dil).any()), segments=chg,
                         naive_flight_frac_orig_v11=float((F0["sole"][..., 2].min((1, 2)) - F0["radius"] > 0.025).mean()))
    # ---- anomalies
    sw0 = [switches(A0["ident"]["feat"]["e"][:, f]) for f in (0, 1)]; sw1 = [switches(r1["feat"]["e"][:, f]) for f in (0, 1)]
    res["contact_switching"] = dict(raw_threshold_crossings_orig=sw0, raw_threshold_crossings_refined=sw1, hysteresis_toggles_orig=[int((np.diff(A0["ident"]["stance"][:, f].astype(int)) != 0).sum()) for f in (0, 1)],
                                    hysteresis_toggles_refined=[int((np.diff(st[:, f].astype(int)) != 0).sum()) for f in (0, 1)])
    new_pen5 = res["ground"]["penetration_gt5mm_frames"]; new_exc = int(((exc1.any(1)) & ~(exc0.any(1))).sum())
    res["anomalies"] = dict(new_penetration_gt5mm_frames=new_pen5, new_joint_vel_exceed_frames=new_exc, new_joint_range_violations=res["dynamics"]["joint_range_new_violations"],
                            contact_switch_ratio=float(sum(sw1) / max(sum(sw0), 1)),
                            none_new=bool(new_pen5 == 0 and new_exc == 0 and res["dynamics"]["joint_range_new_violations"] == 0 and sum(sw1) <= 1.05 * sum(sw0)))
    # ---- integrity
    z0, z1 = O["z"], N["z"]
    integ = dict(frames=bool(z1["joint_pos"].shape == z0["joint_pos"].shape), fps=bool(np.array_equal(z1["fps"], z0["fps"])), names=bool(np.array_equal(z1["body_names"], z0["body_names"]) and np.array_equal(z1["joint_names"], z0["joint_names"])),
                 root_quat=bool(np.abs(np.abs((z1["body_quat_w"][:, 0] * z0["body_quat_w"][:, 0]).sum(-1)) - 1).max() < 1e-5), upper_quat=bool(np.abs(np.abs((z1["body_quat_w"][:, 13:] * z0["body_quat_w"][:, 13:]).sum(-1)) - 1).max() < 1e-5),
                 finite=bool(all(np.isfinite(z1[k]).all() for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"))), root_ang_vel=bool(np.abs(z1["body_ang_vel_w"][:, 0] - _so3(O["root_q"], 1.0 / fps)).max() < 1e-4))  # orientation untouched => angular velocity = converter recomputation of the ORIGINAL quats (foreign files may use another differencing)
    integ["ok"] = bool(all(integ.values())); res["integrity"] = integ
    res["_series"] = dict(slip0=s0, o0=o0, slip1=s1, o1=o1, zmp0=m0, zmp1=m1, dd=dd)
    return res


def pin_residual(sole, unit, radius, mask_pin=None):
    """planted-foot residual of a trajectory: per pin unit (stance run x sole sample) the xy scatter around the unit mean [mm] (rms / p95 over samples) and the height of the pinned samples above z=0."""
    tt, ff, ii = np.where(unit >= 0); uu = unit[tt, ff, ii]; P = sole[tt, ff, ii]; n = uu.max() + 1; cnt = np.bincount(uu, minlength=n)
    mean = np.stack([np.bincount(uu, weights=P[:, a], minlength=n) / np.maximum(cnt, 1) for a in (0, 1)], 1); r = np.linalg.norm(P[:, :2] - mean[uu], axis=1) * 1000
    return dict(n_units=int(n), n_samples=int(len(uu)), xy_rms_mm=float(np.sqrt((r ** 2).mean())), xy_p95_mm=float(np.percentile(r, 95)), xy_max_mm=float(r.max()))
