"""PartyMimic Pipeline V1.1 - motion and policy-rollout quality analysis (main venv: numpy / scipy / mujoco / matplotlib; no robot connection).

Two halves
  A. REFERENCE motion analysis  (analyze_reference): ground calibration, contact reliability tiers, root-translation consistency (planted-foot
     dead reckoning), dynamic-feasibility (kinematic ZMP vs. support polygon on RELIABLE-contact frames only), difficulty vs. LAFAN.
  B. ROLLOUT analysis           (analyze_rollouts): closed-loop MuJoCo rollouts recorded by pmpipe.rollout -> global-MPJPE decomposition
     (frame offset vs. real trajectory difference), contact / penetration / torque saturation / balance from MuJoCo contact forces.

Conventions (kept from earlier phases):  "anchored MPJPE" = per-frame XY + yaw anchoring, root height kept (NOT a global trajectory error);
"global MPJPE" = plain world-frame body error.  Rollout sample k (state at the start of control tick k) corresponds to reference frame k.
Nothing here modifies a motion file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

G = 9.81
PAD = 50  # hold frames at both ends of the padded training clip
_C: dict = {}


# ----------------------------------------------------------------------------------------------------------------------------- model
def g1_model(robot: str = "g1mj"):
    import mujoco
    if robot not in _C:
        from partymimic.robot import get_robot
        m = mujoco.MjModel.from_xml_path(get_robot(robot).mjcf_path)
        foot = {s: [g for g in range(m.ngeom) if m.geom_group[g] == 3 and m.body(m.geom_bodyid[g]).name == f"{s}_ankle_roll_link"] for s in ("left", "right")}
        _C[robot] = (m, mujoco.MjData(m), foot)
    return _C[robot]


def foot_samples(m, d, gids):
    """foot collision geoms -> list of (x, y, z, radius) sample points (capsule end points / sphere centres)."""
    import mujoco
    out = []
    for g in gids:
        r = m.geom_size[g, 0]; c = d.geom_xpos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ax = d.geom_xmat[g].reshape(3, 3)[:, 2]; h = m.geom_size[g, 1]
            out += [(*(c + h * ax), r), (*(c - h * ax), r)]
        else:
            out.append((*c, r))
    return out


def fk_series(root_p, root_q, q, robot: str = "g1mj", want_pts: bool = False, body_names=None):
    """Forward kinematics of a (T,3)/(T,4 wxyz)/(T,29) trajectory on the g1mj model: ankle positions, per-foot lowest point clearance, CoM, optional bodies/foot points."""
    import mujoco
    m, d, foot = g1_model(robot); T = len(q)
    ank = np.zeros((T, 2, 3)); clr = np.zeros((T, 2)); com = np.zeros((T, 3))
    ids = [m.body(n).id for n in body_names] if body_names is not None else None
    bodies = np.zeros((T, len(ids), 3)) if ids else None
    nf = len(foot_samples(m, d, foot["left"])); pts = np.zeros((T, 2, nf, 4)) if want_pts else None
    for t in range(T):
        d.qpos[:3] = root_p[t]; d.qpos[3:7] = root_q[t]; d.qpos[7:] = q[t]
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        com[t] = d.subtree_com[1]
        for k, s in enumerate(("left", "right")):
            P = foot_samples(m, d, foot[s]); clr[t, k] = min(p[2] - p[3] for p in P); ank[t, k] = d.body(f"{s}_ankle_roll_link").xpos
            if want_pts: pts[t, k] = np.array(P)
        if ids: bodies[t] = d.xpos[ids]
    return dict(ank=ank, clr=clr, com=com, bodies=bodies, pts=pts)


def lowpass(x, fc, fs):
    from scipy.signal import butter, filtfilt
    if fc is None or fc >= 0.45 * fs: return x
    b, a = butter(4, fc / (fs / 2)); return filtfilt(b, a, x, axis=0)


def kin_zmp(com, fs, fc=8.0, floor_z=0.0):
    """Kinematic ZMP (cart-table, angular-momentum rate ignored). Validated against the MuJoCo contact-force centre of pressure (see analyze_rollouts)."""
    c = lowpass(com, fc, fs); a = np.gradient(np.gradient(c, 1 / fs, axis=0), 1 / fs, axis=0)
    zmp = com[:, :2] - ((com[:, 2:3] - floor_z) / (G + a[:, 2:3])) * a[:, :2]
    return zmp, a


def hull_margin(P2, q):
    """signed distance of q to the convex hull of P2 (>0 inside = distance to the nearest edge; <0 outside = distance to the hull)."""
    from scipy.spatial import ConvexHull, QhullError
    P2 = np.asarray(P2, float)
    if len(P2) < 3: return -float(np.linalg.norm(q - P2.mean(0))) if len(P2) else np.nan
    try:
        h = ConvexHull(P2)
    except QhullError:
        return -float(np.linalg.norm(q - P2.mean(0)))
    s = h.equations[:, :2] @ q + h.equations[:, 2]  # <=0 inside
    return float(-s.max())


def runs(mask):
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None: s = i
        if not v and s is not None: out.append((s, i)); s = None
    if s is not None: out.append((s, len(mask)))
    return out


# ----------------------------------------------------------------------------------------------------------------------------- reference: floor + tiers
def calibrate_floor(clr, jv, root_p, fps, min_s=0.5, jv_rms_max=0.15):
    """Per-foot floor offset = foot clearance during the longest STATIC stance (both feet certainly loaded). Fallback: 5th percentile (low confidence)."""
    rms = np.sqrt((jv ** 2).mean(1)); rv = np.linalg.norm(np.gradient(root_p, 1 / fps, axis=0), axis=1)
    static = (rms < jv_rms_max) & (rv < 0.05)
    seg = [(a, b) for a, b in runs(static) if (b - a) >= min_s * fps]
    if seg:
        a, b = max(seg, key=lambda s: s[1] - s[0]); a2, b2 = a + 2, b - 2
        off = np.median(clr[a2:b2], axis=0); return dict(offset=off, method="static_stance", frames=[int(a), int(b)], confident=True, spread_mm=float(np.ptp(clr[a2:b2], axis=0).max() * 1000))
    off = np.percentile(clr, 5, axis=0); return dict(offset=off, method="p05_fallback", frames=None, confident=False, spread_mm=None)


def contact_tiers(clr, off, sure=0.010, swing=0.040):
    """per foot: 2 = sure contact (height above the calibrated floor < sure), 0 = sure swing (> swing), 1 = ambiguous in between."""
    h = clr - off
    return np.where(h < sure, 2, np.where(h > swing, 0, 1)).astype(int), h


def dead_reckon(root_xy, ank_xy, contact):
    """Root XY path implied by the joint motion if every in-contact foot is planted (does not move in the world). Starts at root_xy[0]."""
    T = len(root_xy); out = np.zeros((T, 2)); out[0] = root_xy[0]; rel = ank_xy - root_xy[:, None, :]
    for t in range(1, T):
        use = contact[t] & contact[t - 1]
        out[t] = out[t - 1] + (-(rel[t, use] - rel[t - 1, use]).mean(0) if use.any() else 0.0)
    return out


def path_len(xy): return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def load_ref(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    names = [str(n) for n in z["body_names"]]; r = names.index("pelvis")
    return dict(fps=float(z["fps"][0]) if "fps" in z.files else 50.0, root_p=z["body_pos_w"][:, r], root_q=z["body_quat_w"][:, r], q=z["joint_pos"], jv=z["joint_vel"],
                body_p=z["body_pos_w"], body_names=names)


def difficulty_metrics(R, K=None):
    """clip-intrinsic difficulty numbers (comparable between clips; no dependence on a per-clip contact classification)."""
    fps = R["fps"]; K = K or fk_series(R["root_p"], R["root_q"], R["q"])
    zmp, a = kin_zmp(K["com"], fps); comv = np.gradient(lowpass(K["com"], 8.0, fps), 1 / fps, axis=0)
    off = np.percentile(K["clr"], 2, axis=0); tier, h = contact_tiers(K["clr"], off, 0.025, 0.025)
    c = h < 0.025; ss = (c.sum(1) == 1); fl = (c.sum(1) == 0)
    zoff = np.linalg.norm(K["com"][:, :2] - zmp, axis=1)
    jv = R["jv"]; ja = np.gradient(jv, 1 / fps, axis=0); sep = np.linalg.norm(K["ank"][:, 0, :2] - K["ank"][:, 1, :2], axis=1)
    return dict(jv_rms=float(np.sqrt((jv ** 2).mean())), jv_p95=float(np.percentile(np.abs(jv), 95)), jacc_p95=float(np.percentile(np.abs(ja), 95)),
                com_speed_max=float(np.linalg.norm(comv[:, :2], axis=1).max()), com_acc_p95=float(np.percentile(np.linalg.norm(a[:, :2], axis=1), 95)),
                zmp_offset_p95=float(np.percentile(zoff, 95)), zmp_offset_frac_gt_12cm=float((zoff > 0.12).mean()), foot_spread_p95=float(np.percentile(sep, 95)),
                root_h_range=float(np.ptp(R["root_p"][:, 2])), single_support_frac=float(ss.mean()), flight_frac=float(fl.mean()),
                transitions_per_s=float((np.diff(c.astype(int), axis=0) != 0).any(1).sum() / (len(c) / fps)), duration_s=float(len(c) / fps))


def lafan_difficulty(cache: Path, robot="g1mj", procs=16, force=False):
    """difficulty metrics of the 41 LAFAN G1 clips (computed once with the SAME code and cached next to the pipeline)."""
    cache = Path(cache)
    if cache.exists() and not force: return json.loads(cache.read_text())
    from multiprocessing import Pool
    from .core import REPO
    files = sorted((REPO / "data/motions/g1/lafan").glob("*.npz"))
    with Pool(min(procs, len(files))) as p: res = p.map(_lafan_one, [(str(f), robot) for f in files])
    out = {Path(f).stem: r for f, r in zip(files, res) if r}
    cache.parent.mkdir(parents=True, exist_ok=True); cache.write_text(json.dumps(dict(code="quality.difficulty_metrics v1.1", clips=out), indent=1)); return json.loads(cache.read_text())


def _lafan_one(a):
    f, robot = a
    try:
        z = np.load(f); n = len(z["joint_pos"])
        R = dict(fps=float(z["fps"][0]), root_p=z["body_pos_w"][:, 0], root_q=z["body_quat_w"][:, 0], q=z["joint_pos"], jv=z["joint_vel"])
        return difficulty_metrics(R)
    except Exception as e:
        print("lafan skip", f, e); return None


# ----------------------------------------------------------------------------------------------------------------------------- reference analysis
def analyze_reference(dance, out_dir: Path, lafan_cache: Path, rollouts: list[Path] | None = None, robot="g1mj") -> dict:
    """Full reference-motion report data for a Dance (uses the 50 Hz padded training clip so ground calibration sees the static hold).
    `rollouts`: optional pmpipe.rollout npz files used ONLY to validate the kinematic ZMP against MuJoCo contact forces."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    R = load_ref(dance.padded_npz); fps = R["fps"]; T = len(R["q"])
    K = fk_series(R["root_p"], R["root_q"], R["q"], robot, want_pts=True)
    cal = calibrate_floor(K["clr"], R["jv"], R["root_p"], fps); off = cal["offset"]
    tier, h = contact_tiers(K["clr"], off)
    dsl = slice(PAD, T - PAD)  # dance span of the padded clip
    res = dict(dance=dance.name, frames_padded=T, fps=fps, dance_frames=T - 2 * PAD)
    # ---- floating / ground semantics
    stand = K["clr"][cal["frames"][0] + 2: cal["frames"][1] - 2] if cal["frames"] else None
    res["ground"] = dict(method=cal["method"], confident=cal["confident"], static_frames=cal["frames"], floor_offset_mm=(off * 1000).round(1).tolist(), spread_mm=cal["spread_mm"],
                         global_min_clearance_mm=float(K["clr"].min() * 1000), floating=bool((off > 0.008).any()),
                         note="foot clearance above z=0 while standing statically; the conversion aligned the lowest foot sample of the WHOLE clip to z=0, so a static stance floats by this amount")
    mn = h.min(1) * 1000; win = [float(np.median(mn[a:a + 50])) for a in range(0, T - 49, 50)]
    res["ground"]["floor_drift"] = dict(window_median_min_foot_mm=[round(x, 1) for x in win], range_mm=[round(min(win), 1), round(max(win), 1)], frac_foot_below_floor_10mm=float((h < -0.010).any(1).mean()),
                                        min_foot_mm=float(h.min() * 1000), end_hold_above_start_mm=float(np.median(h[-30:]) * 1000 - np.median(h[10:40]) * 1000),
                                        note="height of the lowest foot above the static-stance floor; a rigid floor would read ~0 whenever a foot is planted (video-derived references drift by +-20 mm)")
    # ---- contact reliability tiers
    L, Rr = tier[dsl, 0], tier[dsl, 1]
    sureDS = (L == 2) & (Rr == 2); sureSS = ((L == 2) & (Rr == 0)) | ((L == 0) & (Rr == 2)); sureFl = (L == 0) & (Rr == 0)
    amb = ~(sureDS | sureSS | sureFl)
    res["contact_tiers"] = dict(sure_contact_thr_mm=10, sure_swing_thr_mm=40, double_support=float(sureDS.mean()), single_support=float(sureSS.mean()), flight=float(sureFl.mean()), ambiguous=float(amb.mean()),
                                per_foot_sure_contact=[float((tier[dsl, k] == 2).mean()) for k in (0, 1)], per_foot_sure_swing=[float((tier[dsl, k] == 0).mean()) for k in (0, 1)],
                                max_swing_clearance_mm=[float((h[dsl, k].max()) * 1000) for k in (0, 1)],
                                note="only frames with an unambiguous support configuration enter the ZMP / support-margin statistics")
    # ---- root translation consistency (planted-foot dead reckoning); sensitivity over the contact threshold
    rxy = R["root_p"][:, :2]; cons = {}
    for thr in (0.015, 0.025, 0.040):
        pr = dead_reckon(rxy, K["ank"][:, :, :2], h < thr)
        e = np.linalg.norm(rxy - pr, axis=1)[dsl]
        cons[f"{int(thr * 1000)}mm"] = dict(root_path_m=path_len(rxy[dsl]), implied_path_m=path_len(pr[dsl]), mean_mismatch_m=float(e.mean()), max_mismatch_m=float(e.max()))
    thr = 0.025; c25 = h < thr; v = np.linalg.norm(np.gradient(K["ank"][:, :, :2], 1 / fps, axis=0), axis=-1)
    both = c25.all(1); spread = np.linalg.norm(K["ank"][:, 0, :2] - K["ank"][:, 1, :2], axis=1); dspread = np.abs(np.gradient(spread, 1 / fps))
    res["root_consistency"] = dict(by_threshold=cons, planted_foot_speed_mps=dict(mean=float(v[c25 & True][dsl.start:].mean()) if c25.any() else 0, p95=float(np.percentile(v[c25], 95)) if c25.any() else 0),
                                   double_contact_spread_rate_mps=dict(mean=float(dspread[both].mean()) if both.any() else 0, p95=float(np.percentile(dspread[both], 95)) if both.any() else 0),
                                   verdict="INCONSISTENT" if cons["25mm"]["max_mismatch_m"] > 0.3 else "OK",
                                   note="max_mismatch = largest distance between the reference root XY and the root XY implied by its own joint motion with planted feet (no skating)")
    # ---- dynamics on reliable-contact frames
    zmp, a = kin_zmp(K["com"], fps, 8.0, floor_z=float(np.mean(off)))
    marg = np.full(T, np.nan); cmarg = np.full(T, np.nan)
    for t in range(PAD, T - PAD):
        k = t - PAD
        if not (sureDS[k] or sureSS[k]): continue
        P = []
        for f in (0, 1):
            if tier[t, f] == 2:
                for x, y, z, r in K["pts"][t, f]:
                    if z - r - off[f] < 0.015:
                        P += [(x + r * np.cos(a_), y + r * np.sin(a_)) for a_ in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        if len(P) >= 3:
            marg[t] = hull_margin(P, zmp[t]); cmarg[t] = hull_margin(P, K["com"][t, :2])
    ok = np.isfinite(marg)
    res["dynamics_reliable_frames"] = dict(coverage=float(ok[dsl].mean()), frames=int(ok.sum()), fc_hz=8.0,
                                           zmp_margin_mm=dict(p05=float(np.nanpercentile(marg, 5) * 1000), median=float(np.nanmedian(marg) * 1000), min=float(np.nanmin(marg) * 1000)) if ok.any() else None,
                                           zmp_outside_frac=float(np.nanmean(marg[ok] < 0)) if ok.any() else None, com_outside_frac=float(np.nanmean(cmarg[ok] < 0)) if ok.any() else None,
                                           com_acc_p95=float(np.percentile(np.linalg.norm(a[dsl, :2], axis=1), 95)),
                                           zmp_offset_p95_mm=float(np.percentile(np.linalg.norm(K["com"][dsl, :2] - zmp[dsl], axis=1), 95) * 1000))
    # ---- kinematic-ZMP validation against MuJoCo contact forces
    if rollouts:
        errs = []
        for rp in rollouts:
            s = np.load(rp); z2, _ = kin_zmp(s["com"], 50.0, 8.0); okk = (s["fz_total"] > 100) & np.isfinite(s["cop"][:, 0])
            errs.append(np.linalg.norm(z2 - s["cop"], axis=1)[okk])
        e = np.concatenate(errs); res["zmp_validation"] = dict(median_mm=float(np.median(e) * 1000), p90_mm=float(np.percentile(e, 90) * 1000), n=int(len(e)),
                                                                note="kinematic ZMP (8 Hz) vs force-based centre of pressure in closed-loop MuJoCo rollouts")
    # ---- difficulty vs LAFAN
    lf = lafan_difficulty(lafan_cache, robot)["clips"]; mine = difficulty_metrics(dict(fps=fps, root_p=R["root_p"][dsl], root_q=R["root_q"][dsl], q=R["q"][dsl], jv=R["jv"][dsl]),
                                                                                  {k: (v[dsl] if isinstance(v, np.ndarray) else v) for k, v in K.items()})
    dance_pool = {k: v for k, v in lf.items() if k.startswith("dance")}
    diff = {}
    for k, val in mine.items():
        if k == "duration_s": continue
        for nm, pool in (("all", lf), ("dance", dance_pool)):
            arr = np.array([c[k] for c in pool.values()]); diff[f"{k}@{nm}"] = dict(value=float(val), percentile=float((arr < val).mean() * 100), median=float(np.median(arr)), max=float(arr.max()), n=len(arr))
    res["difficulty"] = dict(metrics=mine, vs_lafan=diff)
    res["series"] = dict(h_mm=(h * 1000).round(1).tolist(), tier=tier.tolist(), zmp_margin_mm=np.where(ok, marg * 1000, np.nan).round(1).tolist())
    _plot_reference(res, R, K, h, tier, marg, cmarg, zmp, dead_reckon(rxy, K["ank"][:, :, :2], h < 0.025), out_dir / f"{dance.name}_reference_quality.png")
    return res


def _plot_reference(res, R, K, h, tier, marg, cmarg, zmp, dr, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    T = len(h); t = np.arange(T) / R["fps"]; fig, ax = plt.subplots(5, 1, figsize=(12, 13), sharex=False)
    col = {2: "#2e7d32", 1: "#f9a825", 0: "#c62828"}
    for f, lab in ((0, "left"), (1, "right")):
        ax[0].plot(t, h[:, f] * 1000, label=f"{lab} foot height above calibrated floor")
    ax[0].axhline(10, c="g", ls=":"); ax[0].axhline(40, c="r", ls=":"); ax[0].set_ylabel("mm"); ax[0].legend(fontsize=7)
    ax[0].set_title(f"{res['dance']}: calibrated foot height (sure contact < 10 mm, sure swing > 40 mm; floating offset {res['ground']['floor_offset_mm']} mm)")
    img = np.array([tier[:, 0], tier[:, 1]]); ax[1].imshow(img, aspect="auto", cmap=matplotlib.colors.ListedColormap([col[0], col[1], col[2]]), extent=[0, t[-1], 0, 2], vmin=0, vmax=2)
    ax[1].set_yticks([0.5, 1.5]); ax[1].set_yticklabels(["R", "L"]); ax[1].set_title("contact tier (green sure contact / yellow ambiguous / red sure swing)")
    ax[2].plot(t, marg * 1000, "m", label="kinematic ZMP margin"); ax[2].plot(t, cmarg * 1000, "k", label="CoM margin"); ax[2].axhline(0, c="r"); ax[2].set_ylabel("mm (<0 outside)"); ax[2].legend(fontsize=7)
    ax[2].set_title(f"support margin on reliable-contact frames only (coverage {res['dynamics_reliable_frames']['coverage']*100:.0f}% of the dance)")
    ax[3].plot(R["root_p"][:, 0], R["root_p"][:, 1], "k", label="reference root XY"); ax[3].plot(dr[:, 0], dr[:, 1], "r", label="root XY implied by planted feet (dead reckoning)")
    ax[3].set_aspect("equal"); ax[3].legend(fontsize=7); ax[3].set_title(f"root translation consistency: {res['root_consistency']['verdict']} (max mismatch {res['root_consistency']['by_threshold']['25mm']['max_mismatch_m']:.2f} m)")
    ax[4].plot(t, np.linalg.norm(R["root_p"][:, :2] - dr, axis=1), "r"); ax[4].set_ylabel("|ref root - implied root| [m]"); ax[4].set_xlabel("time [s]")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------------- rollout analysis
def load_rollout(p):
    z = np.load(p, allow_pickle=True); d = {k: z[k] for k in z.files if k != "meta"}; d["meta"] = json.loads(str(z["meta"])); return d


def _yaw(q): return np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def _wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def trajectory_decomposition(ref: dict, refK: dict, ro: dict, robot="g1mj") -> dict:
    """Split the global MPJPE of one rollout into: pose error (anchored) / yaw / root XY offset, and split the XY offset by planted-foot dead reckoning."""
    T = len(ro["t"]); names = ref["body_names"]
    S = fk_series(ro["root_p"], ro["root_q"], ro["q"], robot, body_names=names)
    rb = ref["body_p"][:T]; rp = ref["root_p"][:T]; rq = ref["root_q"][:T]
    sp, sq = ro["root_p"], ro["root_q"]
    glob = np.linalg.norm(S["bodies"] - rb, axis=-1).mean(-1)
    dyaw = _wrap(_yaw(sq) - _yaw(rq))
    c, s_ = np.cos(dyaw), np.sin(dyaw)
    # anchored (evaluator definition): rotate the reference by the yaw difference about the root, translate to the sim root XY, keep the reference root height
    rel = rb - rp[:, None, :]; rot = np.stack([np.stack([c, -s_], -1), np.stack([s_, c], -1)], -2)
    rel_xy = np.einsum("tij,tbj->tbi", rot, rel[..., :2]); anc = np.concatenate([sp[:, None, :2] + rel_xy, (rp[:, 2:3, None] + rel[..., 2:3]).reshape(T, -1, 1)], -1)
    anch = np.linalg.norm(S["bodies"] - anc, axis=-1).mean(-1)
    # xy-offset-only anchoring (no yaw alignment): how much of the global error is the root XY offset
    g1 = np.linalg.norm(S["bodies"] - (rb + np.concatenate([sp[:, None, :2] - rp[:, None, :2], np.zeros((T, 1, 1))], -1)), axis=-1).mean(-1)
    exy = sp[:, :2] - rp[:, :2]
    dsl = slice(PAD, T - PAD)
    # planted-foot dead reckoning: reference (calibrated contact) vs simulation (load-bearing contact from MuJoCo forces)
    cal = refK["cal"]; hh = refK["h"][:T]
    ref_dr = dead_reckon(rp[:, :2], refK["ank"][:T, :, :2], hh < 0.025)
    BW = ro["meta"]["mass_kg"] * G; lb = ro["foot_fz"] > 0.10 * BW
    sim_dr = dead_reckon(sp[:, :2], S["ank"][:, :, :2], lb)
    n = lambda x: float(np.linalg.norm(x, axis=1)[dsl].mean())
    mx = lambda x: float(np.linalg.norm(x, axis=1)[dsl].max())
    out = dict(global_mpjpe_mm=float(glob[dsl].mean() * 1000), xy_offset_anchored_mpjpe_mm=float(g1[dsl].mean() * 1000), anchored_mpjpe_mm=float(anch[dsl].mean() * 1000),
               initial_global_mpjpe_mm=float(glob[0] * 1000), yaw_offset_deg=dict(mean_abs=float(np.degrees(np.abs(dyaw[dsl]).mean())), max_abs=float(np.degrees(np.abs(dyaw[dsl]).max()))),
               root_xy_offset_m=dict(mean=n(exy), max=mx(exy), end=float(np.linalg.norm(exy[-PAD - 1]))), root_z_offset_mm=float((sp[dsl, 2] - rp[dsl, 2]).mean() * 1000),
               path_m=dict(reference=path_len(rp[dsl, :2]), simulation=path_len(sp[dsl, :2]), reference_implied_by_feet=path_len(ref_dr[dsl]), simulation_implied_by_feet=path_len(sim_dr[dsl])),
               offset_components_m=dict(total=n(exy), policy_slip_drift=n(sp[:, :2] - sim_dr), pose_implied_difference=n(sim_dr - ref_dr), reference_root_inconsistency=n(ref_dr - rp[:, :2]),
                                        policy_slip_drift_max=mx(sp[:, :2] - sim_dr), reference_root_inconsistency_max=mx(ref_dr - rp[:, :2])))
    out["_series"] = dict(glob=glob, anch=anch, g1=g1, exy=exy, sim_dr=sim_dr, ref_dr=ref_dr, sp=sp[:, :2], rp=rp[:, :2], yaw=np.degrees(dyaw))
    return out


def contact_point_slip(pts, lo: int, hi: int, floor_off=None, thr: float = 0.005, vmax: float = 3.0, fps: float = 50.0) -> dict:
    """V1.4 STANDARD foot-slip metric = speed of the CONTACT POINT (evaluator definition): the lowest end point of every foot capsule, counted while its
    sole clearance is < thr on two consecutive samples (speed < vmax, filters re-contact jumps).  pts: fk_series(want_pts=True)['pts'] (T,2,14,4) = 7 capsules x 2 end points.
    floor_off: optional per-foot floor offset (reference motions: foot clearance above the clip's own static floor); None = z=0 (simulation).
    Why not the sole-patch centroid (V1.3 `foot_slip_mps`, kept for history): the centroid moves when the load shifts / the foot rolls, which inflates the number for
    gaits with more double support although nothing slides.  """
    T = pts.shape[0]; vals = []; dt = 1.0 / fps; off = np.zeros(2) if floor_off is None else np.asarray(floor_off, float)
    sel = np.zeros(T, bool); sel[lo:hi] = True
    for ft in (0, 1):
        p = pts[:, ft].reshape(T, -1, 2, 4); i = np.argmin(p[..., 2], axis=2); low = np.take_along_axis(p, i[..., None, None], 2)[:, :, 0]
        con = (low[..., 2] - low[..., 3] - off[ft]) < thr; xy = low[..., :2]; sp = np.linalg.norm(xy[1:] - xy[:-1], axis=-1) / dt
        ok = con[1:] & con[:-1] & (sp < vmax) & sel[1:, None]; vals.append(sp[ok])
    v = np.concatenate(vals)
    return dict(mean=float(v.mean()) if len(v) else float("nan"), p95=float(np.percentile(v, 95)) if len(v) else float("nan"), n=int(len(v)))


def contact_dynamics(ro: dict, robot="g1mj") -> dict:
    """Balance / contact / penetration / torque / guard statistics of ONE rollout, from MuJoCo contact forces (reliable contact).
    Support polygon = convex hull of the sole of every LOADED foot (foot geometry within 15 mm of the floor), so CoP / ZMP margins are comparable with the reference's."""
    T = len(ro["t"]); dsl = slice(PAD, T - PAD); BW = ro["meta"]["mass_kg"] * G; fz = ro["foot_fz"]; names = [n.replace("_joint", "") for n in ro["meta"]["joint_names"]]
    lim = np.array(ro["meta"]["effort_lim"])
    lb = fz > 0.10 * BW; light = fz < 0.05 * BW
    ds = lb.all(1); ss = (lb.sum(1) == 1) & light[np.arange(T), np.where(lb[:, 0], 1, 0)]; fl = (fz.sum(1) < 0.05 * BW)
    sat = ro["sat_frac"][dsl]; jsat = sat.mean(0); idx = lambda key: [i for i, n in enumerate(names) if key in n]
    hr = idx("hip_roll"); hrL, hrR = [i for i in hr if names[i].startswith("left")], [i for i in hr if names[i].startswith("right")]
    share = fz / np.maximum(fz.sum(1, keepdims=True), 1e-6)  # load share per foot
    def sat_when(j, foot, cond):
        mk = cond(share[:, foot]) & (fz.sum(1) > 0.5 * BW); mk[:PAD] = False; mk[-PAD:] = False
        return float(ro["sat_frac"][mk][:, j].mean()) if mk.any() else None
    pf = ro["pen_foot"] * 1000; st = slice(10, PAD - 5)
    K = fk_series(ro["root_p"], ro["root_q"], ro["q"], robot, want_pts=True)
    zk, ak = kin_zmp(ro["com"], 50.0, 8.0); cm = np.full(T, np.nan); zm = np.full(T, np.nan); cc = np.full(T, np.nan)
    for t in range(PAD, T - PAD):
        P = []
        for f in (0, 1):
            if lb[t, f]:
                for x, y, z, r in K["pts"][t, f]:
                    if z - r < 0.015: P += [(x + r * np.cos(a_), y + r * np.sin(a_)) for a_ in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        if len(P) >= 3 and np.isfinite(ro["cop"][t, 0]):
            cm[t] = hull_margin(P, ro["cop"][t]); zm[t] = hull_margin(P, zk[t]); cc[t] = hull_margin(P, ro["com"][t, :2])
    okm = np.isfinite(cm); tilt = ro["tilt"][dsl]; top = np.argsort(-jsat)[:5]
    tau_nm = ro["tau_ratio"][dsl] * lim  # pre-clip PD torque magnitude [N.m] (max over the 4 substeps of each tick)
    dev = np.abs(ro["qcmd"][dsl] - ro["qpre"][dsl]) > 1e-6; topg = np.argsort(-dev.mean(0))[:4]
    return dict(survived=ro["meta"]["fell_t"] is None, fell_t=ro["meta"]["fell_t"],
                contact=dict(double_support=float(ds[dsl].mean()), single_support=float(ss[dsl].mean()), flight=float(fl[dsl].mean()), per_foot_loaded=[float(lb[dsl, k].mean()) for k in (0, 1)],
                             other_geom_contact_frac=float((ro["other_fz"][dsl] > 1).mean())),
                balance=dict(tilt_p95_rad=float(np.percentile(tilt, 95)), tilt_max_rad=float(tilt.max()), com_acc_p95=float(np.percentile(np.linalg.norm(ak[dsl, :2], axis=1), 95)),
                             cop_margin_mm=dict(p05=float(np.nanpercentile(cm, 5) * 1000), median=float(np.nanmedian(cm) * 1000), min=float(np.nanmin(cm) * 1000)) if okm.any() else None,
                             cop_within_10mm_of_edge=float(np.nanmean(cm[okm] < 0.010)) if okm.any() else None, com_outside_sole_frac=float(np.nanmean(cc[okm] < 0)) if okm.any() else None,
                             kin_zmp_outside_sole_frac=float(np.nanmean(zm[okm] < 0)) if okm.any() else None,
                             kin_zmp_vs_cop_median_mm=float(np.nanmedian(np.linalg.norm(zk - ro["cop"], axis=1)[okm]) * 1000) if okm.any() else None),
                penetration_mm=dict(static_stance_max=float(pf[st].max()), static_stance_median=float(np.median(pf[st])), dance_p95=float(np.percentile(pf[dsl], 95)), dance_median=float(np.median(pf[dsl])),
                                    dance_max=float(pf[dsl].max()), any_geom_max=float(ro["pen"][dsl].max() * 1000)),
                torque=dict(sat_frac_mean=float(jsat.mean()), sat_frac_by_joint={names[i]: float(jsat[i]) for i in top}, hip_roll_sat=dict(left=float(jsat[hrL].mean()), right=float(jsat[hrR].mean())),
                            hip_roll_sat_when_stance=dict(loaded_gt70=[sat_when(hrL[0], 0, lambda s: s > 0.7), sat_when(hrR[0], 1, lambda s: s > 0.7)], swing_lt30=[sat_when(hrL[0], 0, lambda s: s < 0.3), sat_when(hrR[0], 1, lambda s: s < 0.3)]),
                            hip_roll_peak_nm=float(tau_nm[:, hr].max()), hip_roll_p99_nm=float(np.percentile(tau_nm[:, hr], 99)), hip_roll_limit_nm=float(lim[hr[0]]),
                            ankle_sat=float(jsat[idx("ankle")].mean()), knee_sat=float(jsat[idx("knee")].mean()), peak_ratio_p95=float(np.percentile(ro["tau_ratio"][dsl].max(1), 95)),
                            hip_roll_mean_abs_angle_rad=float(np.abs(ro["q"][dsl][:, hr]).mean())),
                slip=dict(contact_point=contact_point_slip(K["pts"], PAD, T - PAD)),  # V1.4 standard definition (see contact_point_slip)
                guard=dict(any_clip_rate=float(dev.any(1).mean()), top_joints={names[i]: float(dev.mean(0)[i]) for i in topg}, max_dev_rad=float(np.abs(ro["qcmd"][dsl] - ro["qpre"][dsl]).max()),
                           note="fraction of control ticks where the guard changed the policy target by > 1e-6 rad (the guard's own counters fire on float rounding and are not used)"))


def contact_agreement(ref_tier, ro, frac_bw=0.10):
    """reference sure-contact / sure-swing tiers vs MuJoCo load-bearing contact (Fz > frac_bw*BW) on the dance span; ambiguous reference frames are excluded."""
    T = len(ro["t"]); dsl = slice(PAD, T - PAD); lb = ro["foot_fz"][dsl] > frac_bw * ro["meta"]["mass_kg"] * G; t = ref_tier[:T][dsl]; sure = t != 1; rc = t == 2
    return dict(agreement=float((rc == lb)[sure].mean()), ref_swing_sim_loaded=float(((~rc) & lb)[sure].mean()), ref_contact_sim_unloaded=float((rc & ~lb)[sure].mean()), sure_frac=float(sure.mean()))


def analyze_rollouts(dance, tag_files: dict[str, list[Path]], out_dir: Path, robot="g1mj") -> dict:
    """tag -> list of rollout npz (the first one is the nominal start). Returns per-tag aggregates + the per-rollout decomposition of the first (nominal) one."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ref = load_ref(dance.padded_npz); K = fk_series(ref["root_p"], ref["root_q"], ref["q"], robot)
    cal = calibrate_floor(K["clr"], ref["jv"], ref["root_p"], ref["fps"]); _, h = contact_tiers(K["clr"], cal["offset"])
    refK = dict(ank=K["ank"], h=h, cal=cal); res = {}; series = {}; tier, _ = contact_tiers(K["clr"], cal["offset"])
    for tag, files in tag_files.items():
        decs, dyn = [], []
        for f in files:
            ro = load_rollout(f); dc = trajectory_decomposition(ref, refK, ro, robot); sr = dc.pop("_series"); decs.append(dc); dd = contact_dynamics(ro, robot); dd["contact"]["vs_reference"] = contact_agreement(tier, ro); dyn.append(dd)
            if f is files[0]: series[tag] = sr
        def agg(lst, path):
            v = []
            for x in lst:
                try:
                    y = x
                    for p in path: y = y[p]
                    if y is not None: v.append(float(y))
                except (KeyError, TypeError): pass
            return dict(mean=float(np.mean(v)), std=float(np.std(v)), min=float(np.min(v)), max=float(np.max(v)), n=len(v)) if v else None
        paths_dec = [["global_mpjpe_mm"], ["xy_offset_anchored_mpjpe_mm"], ["anchored_mpjpe_mm"], ["initial_global_mpjpe_mm"], ["yaw_offset_deg", "mean_abs"], ["yaw_offset_deg", "max_abs"], ["root_xy_offset_m", "mean"], ["root_xy_offset_m", "max"],
                     ["root_xy_offset_m", "end"], ["root_z_offset_mm"], ["path_m", "reference"], ["path_m", "simulation"], ["path_m", "reference_implied_by_feet"], ["path_m", "simulation_implied_by_feet"]] + \
                    [["offset_components_m", k] for k in decs[0]["offset_components_m"]]
        paths_dyn = [["contact", "double_support"], ["contact", "single_support"], ["contact", "flight"], ["contact", "vs_reference", "agreement"], ["contact", "vs_reference", "ref_swing_sim_loaded"], ["contact", "vs_reference", "ref_contact_sim_unloaded"], ["balance", "tilt_p95_rad"], ["balance", "tilt_max_rad"], ["balance", "com_acc_p95"], ["balance", "cop_margin_mm", "p05"],
                     ["balance", "cop_margin_mm", "min"], ["balance", "cop_within_10mm_of_edge"], ["balance", "com_outside_sole_frac"], ["balance", "kin_zmp_outside_sole_frac"], ["balance", "kin_zmp_vs_cop_median_mm"], ["penetration_mm", "static_stance_max"], ["penetration_mm", "static_stance_median"],
                     ["penetration_mm", "dance_p95"], ["penetration_mm", "dance_max"], ["penetration_mm", "any_geom_max"], ["torque", "sat_frac_mean"], ["torque", "hip_roll_sat", "left"], ["torque", "hip_roll_sat", "right"], ["torque", "ankle_sat"],
                     ["torque", "knee_sat"], ["torque", "peak_ratio_p95"], ["torque", "hip_roll_peak_nm"], ["torque", "hip_roll_p99_nm"], ["guard", "any_clip_rate"], ["guard", "max_dev_rad"], ["slip", "contact_point", "mean"], ["slip", "contact_point", "p95"]]
        res[tag] = dict(n=len(files), survived=int(sum(d["survived"] for d in dyn)), decomposition={"/".join(p): agg(decs, p) for p in paths_dec}, dynamics={"/".join(p): agg(dyn, p) for p in paths_dyn},
                        nominal=dict(decomposition=decs[0], dynamics=dyn[0]))
    _plot_trajectory(ref, series, res, out_dir / f"{dance.name}_trajectory_decomposition.png")
    return res


def _plot_trajectory(ref, series, res, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tags = list(series); T = len(next(iter(series.values()))["glob"]); t = np.arange(T) / 50; cols = plt.cm.tab10.colors
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    s0 = series[tags[0]]
    ax[0, 0].plot(s0["rp"][:, 0], s0["rp"][:, 1], "k", lw=2, label="reference root"); ax[0, 0].plot(s0["ref_dr"][:, 0], s0["ref_dr"][:, 1], "k--", label="reference root implied by planted feet")
    for i, tg in enumerate(tags): ax[0, 0].plot(series[tg]["sp"][:, 0], series[tg]["sp"][:, 1], c=cols[i], label=f"{tg} root")
    ax[0, 0].set_aspect("equal"); ax[0, 0].legend(fontsize=7); ax[0, 0].set_title("root XY (top view): reference vs closed-loop rollouts"); ax[0, 0].set_xlabel("x [m]"); ax[0, 0].set_ylabel("y [m]")
    for i, tg in enumerate(tags):
        s = series[tg]; ax[0, 1].plot(t, np.linalg.norm(s["exy"], axis=1), c=cols[i], label=f"{tg} |sim - ref| root XY")
        ax[0, 1].plot(t, np.linalg.norm(s["sim_dr"] - s["sp"], axis=1), c=cols[i], ls=":", label=f"{tg} policy slip drift (sim vs its own planted-foot path)")
    ax[0, 1].plot(t, np.linalg.norm(s0["ref_dr"] - s0["rp"], axis=1), "k--", label="reference inconsistency (ref vs its planted-foot path)")
    ax[0, 1].axvspan(0, 1, color="0.9"); ax[0, 1].axvspan(t[-1] - 1, t[-1], color="0.9"); ax[0, 1].legend(fontsize=6); ax[0, 1].set_title("root XY offset decomposition [m]"); ax[0, 1].set_xlabel("time [s]")
    for i, tg in enumerate(tags):
        s = series[tg]; ax[1, 0].plot(t, s["glob"] * 1000, c=cols[i], label=f"{tg} global"); ax[1, 0].plot(t, s["anch"] * 1000, c=cols[i], ls="--", label=f"{tg} XY/yaw-anchored")
    ax[1, 0].set_yscale("log"); ax[1, 0].legend(fontsize=6); ax[1, 0].set_title("MPJPE [mm] over time"); ax[1, 0].set_xlabel("time [s]")
    for i, tg in enumerate(tags): ax[1, 1].plot(t, series[tg]["yaw"], c=cols[i], label=tg)
    ax[1, 1].set_title("yaw offset sim - ref [deg]"); ax[1, 1].legend(fontsize=7); ax[1, 1].set_xlabel("time [s]")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
