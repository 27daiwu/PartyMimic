"""contact_refine.diagnostics - per-frame diagnostics table, window statistics, and consistency of the identified contacts with the training labels and MuJoCo force contacts."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .. import quality as Q
from . import geometry as Gm
from .contacts import STATE_NAMES

PAD = 50


def tilt_series(root_q):
    from scipy.spatial.transform import Rotation as Rot
    return np.arccos(np.clip(Rot.from_quat(root_q[:, [1, 2, 3, 0]]).apply([0, 0, 1])[:, 2], -1, 1))


def frame_table(S, root_q, path, lead=PAD):
    T = len(S["g"]); tilt = tilt_series(root_q); dq = np.degrees(np.abs(S["q1"] - S["q0"])[:, :12]).max(1); d = S["root1"] - S["root0"]
    sl0 = np.where(S["o0"], S["slip0"], 0).max(axis=(1, 2)); sl1 = np.where(S["o1"], S["slip1"], 0).max(axis=(1, 2)); pen0 = np.maximum(-S["low0"].min(1) + S["g"], 0)
    hdr = ["dance_frame", "train_frame_with_50_leadin", "tilt_rad", "ground_mm", "root_z_orig_m", "root_z_new_m", "d_root_x_mm", "d_root_y_mm", "d_root_z_mm", "legs_max_delta_deg",
           "L_h_orig_mm(over ground curve)", "R_h_orig_mm", "L_h_new_mm(over z=0)", "R_h_new_mm", "L_state_orig", "R_state_orig", "L_conf_orig", "R_conf_orig", "L_state_new", "R_state_new", "inferred_support_L", "inferred_support_R",
           "true_flight", "false_flight_hover", "slip_max_orig_mps", "slip_max_new_mps", "zmp_margin_orig_mm", "zmp_margin_new_mm"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr)
        for t in range(T):
            w.writerow([t, t + lead, f"{tilt[t]:.3f}", f"{S['g'][t]*1000:.1f}", f"{S['root0'][t,2]:.4f}", f"{S['root1'][t,2]:.4f}", f"{d[t,0]*1000:.1f}", f"{d[t,1]*1000:.1f}", f"{d[t,2]*1000:.1f}", f"{dq[t]:.2f}",
                        f"{S['e0'][t,0]*1000:.1f}", f"{S['e0'][t,1]*1000:.1f}", f"{S['e1'][t,0]*1000:.1f}", f"{S['e1'][t,1]*1000:.1f}", STATE_NAMES[int(S['state0'][t, 0])], STATE_NAMES[int(S['state0'][t, 1])], f"{S['conf0'][t,0]:.2f}", f"{S['conf0'][t,1]:.2f}",
                        STATE_NAMES[int(S['state1'][t, 0])], STATE_NAMES[int(S['state1'][t, 1])], int(S["inferred0"][t, 0]), int(S["inferred0"][t, 1]), int(S["flight0"][t]), int(S["flight_false0"][t]),
                        f"{sl0[t-1] if t else 0:.3f}", f"{sl1[t-1] if t else 0:.3f}", "" if not np.isfinite(S["zmp0"][t]) else f"{S['zmp0'][t]*1000:.0f}", "" if not np.isfinite(S["zmp1"][t]) else f"{S['zmp1'][t]*1000:.0f}"])


def _nm(x):
    x = x[np.isfinite(x)]; return float(np.median(x) * 1000) if len(x) else None


def window_stats(S, root_q, lo, hi, stance_mask=None):
    """numbers for the window [lo,hi) (dance frames): the 'is the original plausible there?' questions"""
    tilt = tilt_series(root_q); sl = slice(lo, hi); st = S["stance0"][sl]; e0 = S["e0"][sl]; e1 = S["e1"][sl]; low0 = S["low0"][sl]; g = S["g"][sl]
    pen_z0 = np.maximum(-low0.min(1), 0); pen_g = np.maximum(-e0.min(1), 0)
    def slip(key):
        v = S["slip" + key][lo - 1:hi - 1]; o = S["o" + key][lo - 1:hi - 1]; return v[o]
    s0, s1 = slip("0"), slip("1"); d = (S["root1"] - S["root0"])[sl]
    dq = np.degrees(np.abs(S["q1"] - S["q0"])[sl, :12])
    return dict(frames=[lo, hi], train_frames=[lo + PAD, hi + PAD], tilt_peak_rad=float(tilt[sl].max()), tilt_peak_frame=int(lo + np.argmax(tilt[sl])), root_z_orig_m=[float(S["root0"][sl, 2].min()), float(S["root0"][sl, 2].max())],
                root_z_new_m=[float(S["root1"][sl, 2].min()), float(S["root1"][sl, 2].max())], ground_curve_mm=[float(g.min() * 1000), float(g.max() * 1000)],
                hover_of_lowest_foot_over_z0_mm=dict(orig_median=float(np.median(low0.min(1)) * 1000), orig_max=float(low0.min(1).max() * 1000), new_median=float(np.median(e1.min(1)) * 1000), new_max=float(e1.min(1).max() * 1000)),
                stance_foot_height_over_ground_curve_mm=dict(orig_p95=float(np.percentile(np.abs(e0[st]), 95) * 1000) if st.any() else None), stance_foot_height_new_p95_mm=float(np.percentile(np.abs(e1[st]), 95) * 1000) if st.any() else None,
                penetration_orig_vs_ground_curve_max_mm=float(pen_g.max() * 1000), penetration_orig_vs_z0_max_mm=float(pen_z0.max() * 1000), penetration_new_max_mm=float(np.maximum(-e1.min(1), 0).max() * 1000),
                slip_orig=dict(mean=float(s0.mean()) if len(s0) else None, p95=float(np.percentile(s0, 95)) if len(s0) else None, n=int(len(s0))), slip_new=dict(mean=float(s1.mean()) if len(s1) else None, p95=float(np.percentile(s1, 95)) if len(s1) else None, n=int(len(s1))),
                root_correction_mm=dict(dx=[float(d[:, 0].min() * 1000), float(d[:, 0].max() * 1000)], dy=[float(d[:, 1].min() * 1000), float(d[:, 1].max() * 1000)], dz=[float(d[:, 2].min() * 1000), float(d[:, 2].max() * 1000)]),
                legs_delta_max_deg=float(dq.max()), legs_delta_rms_deg=float(np.sqrt((dq ** 2).mean())),
                states_orig={f"{'LR'[f]}": {STATE_NAMES[k]: float((S['state0'][sl, f] == k).mean()) for k in (1, 0, 2)} for f in (0, 1)},
                states_new={f"{'LR'[f]}": {STATE_NAMES[k]: float((S['state1'][sl, f] == k).mean()) for k in (1, 0, 2)} for f in (0, 1)},
                zmp_margin_mm=dict(orig_median=_nm(S["zmp0"][sl]), new_median=_nm(S["zmp1"][sl]), n_reliable=int(np.isfinite(S["zmp0"][sl]).sum())),
                flight_true_frames=int(S["flight0"][sl].sum()), false_flight_frames=int(S["flight_false0"][sl].sum()),
                naive_both_feet_gt25mm_over_z0_frames=int((low0.min(1) > 0.025).sum()))


def label_consistency(dance, S, npz_orig, npz_fix, find_rollouts=True):
    """identified stance vs (a) the TRAINING contact labels (MotionLib.set_ref_contact replica) of the original / refined clip, (b) MuJoCo force-contact of the closed-loop nominal rollout (recorded on the ORIGINAL reference, padded index + 50)."""
    from .. import repair as RP
    out = {}
    lab0 = RP.ref_contact_labels(npz_orig); lab1 = RP.ref_contact_labels(npz_fix) if npz_fix else None; st0, st1 = S["stance0"], S["stance1"]
    ok = (S["state0"] != 2)
    out["training_labels"] = dict(orig_label_contact_frac=float(lab0.mean()), orig_label_vs_identified_agreement=float((lab0 == st0)[ok].mean()), orig_label_stance_but_identified_swing=float((lab0 & (S["state0"] == 0))[ok].mean()),
                                  orig_label_swing_but_identified_stance=float((~lab0 & st0).mean()), orig_label_no_contact_frac=float((~lab0.any(1)).mean()), identified_no_support_frac=float((~st0.any(1)).mean()))
    if lab1 is not None:
        out["training_labels"].update(new_label_contact_frac=float(lab1.mean()), new_label_vs_identified_agreement=float((lab1 == st1)[(S["state1"] != 2)].mean()), new_label_no_contact_frac=float((~lab1.any(1)).mean()),
                                      new_label_double_support_frac=float(lab1.all(1).mean()), orig_label_double_support_frac=float(lab0.all(1).mean()))
    if find_rollouts:
        rr = sorted((dance.dir / "runs").glob("*/reports/quality/traj/*nominal.npz")) if (dance.dir / "runs").exists() else []
        if rr:
            ro = Q.load_rollout(rr[-1]); T = len(S["g"]); n = min(len(ro["t"]) - PAD, T) ; BW = ro["meta"]["mass_kg"] * Q.G; lb = ro["foot_fz"][PAD:PAD + n] > 0.10 * BW
            sel = (S["state0"][:n] != 2); both = lambda a: a[:n]
            out["mujoco_force_contact"] = dict(new_label_vs_loaded_agreement=float((lab1[:n] == lb).mean()) if lab1 is not None else None, rollout=str(rr[-1]), frames=int(n), note="rollout of the ORIGINAL reference (frozen policy); loaded = foot Fz > 10% body weight",
                                               identified_stance_vs_loaded_agreement=float((st0[:n] == lb)[sel].mean()), identified_stance_but_unloaded=float((st0[:n] & ~lb).mean()), identified_swing_but_loaded=float((~st0[:n] & lb)[sel].mean()),
                                               training_label_vs_loaded_agreement=float((lab0[:n] == lb).mean()), training_label_contact_but_unloaded=float((lab0[:n] & ~lb).mean()), training_label_swing_but_loaded=float((~lab0[:n] & lb).mean()),
                                               v11_tier_vs_loaded_agreement_reported=0.69)
    return out


def region_stats(S, tol=0.005):
    """sole-region contact on stance frames: flat (heel and toe samples both within tol of the floor) / heel-only / toe-only / floating, original (over the ground curve) vs refined (over z=0)."""
    tag = Gm.model()["tag"]; out = {}
    for key, H in (("original", S["H0"]), ("refined", S["sole1"][..., 2] - 0.01)):
        st = S["stance0"]; Hh = H if key == "original" else H; res = dict(flat=0, heel_only=0, toe_only=0, floating=0, n=0)
        for f in (0, 1):
            h = Hh[st[:, f], f]  # (n,14)
            heel = (h[:, tag == "heel"].min(1) < tol); toe = (h[:, tag == "toe"].min(1) < tol)
            res["flat"] += int((heel & toe).sum()); res["heel_only"] += int((heel & ~toe).sum()); res["toe_only"] += int((~heel & toe).sum()); res["floating"] += int((~heel & ~toe).sum()); res["n"] += int(len(h))
        out[key] = {k: (v / res["n"] if k != "n" else v) for k, v in res.items()}
    return out
