"""pmpipe.contact_metrics - PartyMimic V1.7 (C1): ONE contact / slip / root-vs-feet metric vocabulary for references, contact-corrected references and closed-loop rollouts.

Why: V1.1 (`quality`), V1.3 (`abtest`), V1.4 (`quality.contact_point_slip`), B1 (`contact_refine.metrics`) and B2 (`ftcompare`) each defined their own contact mask, slip and
"root vs planted-feet" number; they are NOT comparable (see results/diy/v17/C_metrics/METRIC_UNIFICATION_REPORT.md).  This module defines the single reading and offers adapters;
it never changes the behaviour of the old modules (they are only imported for FK / ground / contact identification).

DEFINITIONS (frame index = padded TRAINING frame = rollout sample k; dance frame = k - PAD)
-----------------------------------------------------------------------------------------
Contact masks (T,2) bool, four sources:
  v11    reference geometry: lowest sole sample < 10 mm over the clip's own static-stance floor (quality.contact_tiers "sure contact"); `v11_25` = same at 25 mm (the V1.1 dead-reckoning mask)
  b1     contact_refine stance/swing/uncertain identification (height+speed+hysteresis+support inference) with confidence `b1_conf`
  train  `repair.ref_contact_labels` = exactly what the trainer's contact reward (MotionLib.set_ref_contact) sees: ankle-origin height < clip q05 floor + 15 mm and horizontal speed < 0.5 m/s
  load   MuJoCo force: foot normal force > 10 % body weight (rollouts only)
MAIN MASK:  reference side = `b1` stance with confidence >= `CONF_MIN` (0.5);  simulation side = `load`.  `common` = ref_main AND sim_main (both agree the foot is planted).
  - foot slip, root-vs-feet drift of a REFERENCE          -> ref_main
  - foot slip, root-vs-feet drift of a ROLLOUT            -> sim_main (own support) and common (also reported)
  - load flicker / confusion matrix                       -> sim load vs ref_main (+ the two other ref masks for the label question)
Foot slip: horizontal speed of the contact point (lowest end point of each of the 7 sole capsules per foot, V1.4), counted on two consecutive frames while the point is <
`THR` (10 mm) over the floor (reference: ground curve of the analysis, = 0 for refined motions; rollout: 0) and speed < 3 m/s.  Reported on
  STABLE      mask on both frames and >= K (3) frames away from a mask toggle  (p50 / p95 / mean)
  TRANSITION  frames within K of a mask toggle (touch-down / lift-off) incl. the K frames just outside the mask, point still < THR      (p50 / p95 / mean)
Root-vs-feet consistency (same algorithm everywhere): increments of the planted-foot dead reckoning  inc[t] = -mean_{feet in mask on t-1 and t}(Δ(ankle_xy - root_xy));
  e_inc = Δroot_xy - inc; C = cumsum(e_inc) (frames with no supported foot contribute 0 and are counted in `coverage`)  = root_xy - implied_xy (starting at the dance start).
  cumulative : mean|C|, max|C|, end|C|            (long-horizon integrated drift; depends on clip length)
  windowed   : for every 5 s window |C(t_end) - C(t_start)| (growth inside the window) mean / p95 / max, and rate [m/s]  (comparable between clips)
Flicker: number of load-state toggles per second per foot; `short` = toggles that belong to runs shorter than `MIN_RUN` frames.
Flight/hover: sim airborne = no foot loaded; reference flight = B1 ballistic-validated flight; hover = both feet clearly off the (ground-curve) floor without a ballistic CoM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PAD = 50
THR = 0.010
VMAX = 3.0
K_TRANS = 3
CONF_MIN = 0.5
LOAD_FRAC = 0.10
MIN_RUN = 5


# ------------------------------------------------------------------------------------------------------------------------------------ pure array layer (no MuJoCo)
def toggle_dist(mask, cap=10 ** 6):
    """(T,) frames to the nearest toggle of a boolean series (a toggle sits between t-1 and t; distance of frame t to it = min(t - a, a - 1 - t) style, 0 = adjacent to a toggle).
    no toggle -> cap"""
    m = np.asarray(mask, bool); T = len(m); d = np.full(T, cap, int)
    idx = np.where(m[1:] != m[:-1])[0] + 1  # toggle between idx-1 and idx
    for a in idx:
        t = np.arange(T); d = np.minimum(d, np.where(t >= a, t - a, a - 1 - t))
    return d


def transition_mask(mask, k=K_TRANS):
    """(T,) frames within k of a toggle (both sides)"""
    return toggle_dist(mask) < k


def contact_points(sole, radius, floor):
    """sole (T,2,14,3) sphere centres (7 capsules x 2 ends) -> lowest end point per capsule: xy (T,2,7,2), h (T,2,7) height of its lowest point over `floor` (T,)"""
    T = len(sole); p = sole.reshape(T, 2, 7, 2, 3); i = np.argmin(p[..., 2], axis=3)
    low = np.take_along_axis(p, i[..., None, None], 3)[:, :, :, 0]
    return low[..., :2], low[..., 2] - radius - np.asarray(floor)[:, None, None]


def foot_slip(xy, h, mask, fps, thr=THR, vmax=VMAX, k=K_TRANS, extra=None):
    """xy (T,2,7,2), h (T,2,7), mask (T,2) bool (the mask that defines `stable`).  extra: optional (T,2) bool that must hold too (e.g. the other side's mask for `common`).
    returns dict(stable, transition: stats dict, series: speeds (T-1,2,7), ok_stable, ok_trans)"""
    T = len(h); dt = 1 / fps; sp = np.linalg.norm(xy[1:] - xy[:-1], axis=-1) / dt; con = h < thr; base = con[1:] & con[:-1] & (sp < vmax)
    m = np.asarray(mask, bool) if extra is None else (np.asarray(mask, bool) & np.asarray(extra, bool))
    ok_s = np.zeros((T - 1, 2, 7), bool); ok_t = np.zeros((T - 1, 2, 7), bool)
    for f in (0, 1):
        d = toggle_dist(m[:, f]); stable = m[1:, f] & m[:-1, f] & (d[1:] >= k) & (d[:-1] >= k); tr = (d[1:] < k) | (d[:-1] < k)
        ok_s[:, f] = base[:, f] & stable[:, None]; ok_t[:, f] = base[:, f] & tr[:, None]
    return dict(stable=stats(sp[ok_s]), transition=stats(sp[ok_t]), series=sp, ok_stable=ok_s, ok_trans=ok_t)


def stats(v):
    v = np.asarray(v, float)
    if len(v) == 0: return dict(mean=float("nan"), p50=float("nan"), p95=float("nan"), max=float("nan"), n=0)
    return dict(mean=float(v.mean()), p50=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(v.max()), n=int(len(v)))


def error_increments(root_xy, ank_xy, mask):
    """per-frame increments (T,2) of the root-vs-planted-feet error: Δroot - inc (inc = dead-reckoned root increment of the planted feet, mean over feet on t-1 and t) and (T,) bool `supported`."""
    T = len(root_xy); rel = ank_xy - root_xy[:, None, :]; inc = np.zeros((T, 2)); sup = np.zeros(T, bool); m = np.asarray(mask, bool)
    for t in range(1, T):
        use = m[t] & m[t - 1]
        if use.any(): inc[t] = -(rel[t, use] - rel[t - 1, use]).mean(0); sup[t] = True
    droot = np.zeros((T, 2)); droot[1:] = root_xy[1:] - root_xy[:-1]
    return droot, inc, sup


def root_feet_consistency(root_xy, ank_xy, mask, fps, span=None, win_s=5.0, unsupported="skip"):
    """see module doc.  span = slice of the evaluated frames (dance span; integration restarts at span.start).  unsupported 'skip' = frames without a planted foot add no error
    (honest: the root is allowed to move in a flight), 'count' = the root motion there is counted as error (the quality.dead_reckon / V1.1 / B1 behaviour)."""
    T = len(root_xy); span = span or slice(0, T); droot, inc, sup = error_increments(root_xy, ank_xy, mask)
    e = droot - inc
    if unsupported == "skip": e = np.where(sup[:, None], e, 0.0)
    e = e[span].copy(); e[0] = 0.0; C = np.cumsum(e, axis=0); n = len(C); w = int(round(win_s * fps)); mag = np.linalg.norm(C, axis=1)
    starts = list(range(0, n - w + 1, w)) or [0]; g = np.array([np.linalg.norm(C[min(s + w, n) - 1] - C[s]) for s in starts]) if n > 1 else np.zeros(1)
    cum_path = float(np.linalg.norm(droot[span], axis=1).sum()); 
    return dict(cum_mean_m=float(mag.mean()), cum_max_m=float(mag.max()), cum_end_m=float(mag[-1]), win_growth_mean_m=float(g.mean()), win_growth_p95_m=float(np.percentile(g, 95)), win_growth_max_m=float(g.max()),
                win_growth_rate_mps=float(g.mean() / win_s), coverage=float(sup[span].mean()), root_path_m=cum_path, n_windows=len(g), C=C, win_growth=g, win_starts=np.array(starts))


def decompose_root_offset(sim_root, sim_ank, sim_mask, ref_root, ref_ank, ref_mask, span=None):
    """exact additive split of the sim-vs-ref root XY offset (vector sums, restarted at span.start):
    total = policy_slip (Δsim_root - inc_sim) + pose_implied (inc_sim - inc_ref) + ref_inconsistency (inc_ref - Δref_root).  returns cumulative vectors (n,2) each."""
    T = len(sim_root); span = span or slice(0, T)
    ds, is_, _ = error_increments(sim_root, sim_ank, sim_mask); dr, ir, _ = error_increments(ref_root, ref_ank, ref_mask)
    f = lambda x: np.cumsum(np.where(np.arange(T)[:, None] == (span.start or 0), 0.0, x)[span], axis=0)
    out = dict(policy_slip=f(ds - is_), pose_implied=f(is_ - ir), ref_inconsistency=f(ir - dr), total=f(ds - dr))
    out["check_max_abs"] = float(np.abs(out["policy_slip"] + out["pose_implied"] + out["ref_inconsistency"] - out["total"]).max())
    return out


def flicker(load, fps, span=None, min_run=MIN_RUN):
    """load (T,2) bool.  toggles/s/foot (both edges), and the part caused by runs shorter than min_run frames"""
    T = len(load); span = span or slice(0, T); L = np.asarray(load, bool)[span]; dur = len(L) / fps; tog = 0; short = 0
    for f in (0, 1):
        x = L[:, f]; idx = np.where(x[1:] != x[:-1])[0] + 1; tog += len(idx); b = np.r_[0, idx, len(x)]
        for i in range(1, len(b) - 1):  # interior runs only (first / last run are cut by the span)
            if b[i + 1] - b[i] < min_run: short += 2
    return dict(toggles_per_s_per_foot=float(tog / dur / 2), short_run_toggles_per_s_per_foot=float(short / dur / 2), n_toggles=int(tog), min_run=min_run)


def confusion(ref, sim, valid=None):
    """2x2 confusion of reference contact (rows) vs sim load (cols) on `valid` (T,2) frames: agreement, ref-contact/sim-unloaded, ref-swing/sim-loaded, cohen kappa"""
    r = np.asarray(ref, bool); s = np.asarray(sim, bool); v = np.ones_like(r) if valid is None else np.asarray(valid, bool)
    r, s = r[v], s[v]; n = max(len(r), 1); tp = (r & s).sum(); tn = (~r & ~s).sum(); fp = (~r & s).sum(); fn = (r & ~s).sum()
    po = (tp + tn) / n; pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / n ** 2
    return dict(agreement=float(po), ref_contact_sim_loaded=float(tp / n), ref_swing_sim_unloaded=float(tn / n), ref_contact_sim_unloaded=float(fn / n), ref_swing_sim_loaded=float(fp / n),
                kappa=float((po - pe) / (1 - pe)) if pe < 1 else float("nan"), n=int(len(r)))


def airborne_split(sim_load, ref_flight=None, ref_hover=None, span=None):
    """sim airborne = no loaded foot.  compared with the reference's validated flight and its un-validated hover (both feet off the floor without a ballistic CoM)."""
    T = len(sim_load); span = span or slice(0, T); air = ~np.asarray(sim_load, bool)[span].any(1); out = dict(sim_airborne_frac=float(air.mean()))
    if ref_flight is not None:
        fl = np.asarray(ref_flight, bool)[span]; hv = np.asarray(ref_hover, bool)[span] if ref_hover is not None else np.zeros_like(fl)
        out.update(ref_flight_frac=float(fl.mean()), ref_hover_frac=float(hv.mean()), sim_airborne_during_ref_flight=float(air[fl].mean()) if fl.any() else float("nan"),
                   sim_airborne_during_ref_hover=float(air[hv].mean()) if hv.any() else float("nan"), sim_airborne_elsewhere=float(air[~fl & ~hv].mean()))
    return out


def windows(n, fps, win_s=5.0):
    w = int(round(win_s * fps)); return [(a, min(a + w, n)) for a in range(0, n, w)]


# ------------------------------------------------------------------------------------------------------------------------------------ adapters (MuJoCo FK)
def _ident_params():
    from .contact_refine.config import Params
    return Params()


def ref_bundle(npz, pad=PAD, refined=None, g=None):
    """Analyse a reference npz (padded or not).  refined=None: auto (file has `contactfix_info` -> floor is z=0).  Returns a dict with all four masks and everything needed for metrics."""
    from . import quality as Q, repair as RP
    from .contact_refine import geometry as Gm, pipeline as PL
    M = Gm.load_motion(npz); fps = M["fps"]; T = len(M["q"]); p = _ident_params()
    if refined is None: refined = "contactfix_info" in M["z"]
    A = PL.analyze_arrays(M["root_p"], M["root_q"], M["q"], M["jv"], fps, p, g=np.zeros(T) if refined else g)
    F, I, gr = A["F"], A["ident"], A["g"]
    clr = F["sole"][..., 2].min(-1) - F["radius"]; cal = Q.calibrate_floor(clr, M["jv"], M["root_p"], fps); h11 = clr - cal["offset"]
    st = I["stance"]; conf = I["conf"]; main = st & (conf >= CONF_MIN)
    xy, hh = contact_points(F["sole"], F["radius"], gr)
    hover = (I["feat"]["e"].min(1) > 0.025) & ~I["flight"]["mask"]
    return dict(kind="ref", path=str(npz), fps=fps, T=T, padded=T > 2 * pad + 10 and "padded" in str(npz), refined=bool(refined), root_p=M["root_p"], root_q=M["root_q"], q=M["q"], ank_xy=F["ank_p"][..., :2], F=F, floor=gr, cp_xy=xy, cp_h=hh,
                masks=dict(main=main, b1=st, v11=h11 < 0.010, v11_25=h11 < 0.025, train=RP.ref_contact_labels(npz)), conf=conf, inferred=I["inferred"], state=I["state"],
                flight=I["flight"]["mask"], false_flight=I["flight"]["false_mask"], hover=hover, ident=I, cal_offset=cal["offset"], h11=h11)


def sim_bundle(rollout_npz):
    """rollout npz (pmpipe.rollout) -> same fields: FK of the closed-loop trajectory on the g1mj model, floor z=0, load mask = foot normal force > LOAD_FRAC * body weight"""
    from . import quality as Q
    from .contact_refine import geometry as Gm
    ro = Q.load_rollout(rollout_npz); T = len(ro["t"]); F = Gm.fk(ro["root_p"], ro["root_q"], ro["q"]); BW = ro["meta"]["mass_kg"] * Q.G
    load = ro["foot_fz"] > LOAD_FRAC * BW; xy, hh = contact_points(F["sole"], F["radius"], np.zeros(T))
    return dict(kind="sim", path=str(rollout_npz), fps=50.0, T=T, root_p=ro["root_p"], root_q=ro["root_q"], q=ro["q"], ank_xy=F["ank_p"][..., :2], F=F, floor=np.zeros(T), cp_xy=xy, cp_h=hh,
                masks=dict(main=load, load=load), fz=ro["foot_fz"], BW=BW, ro=ro)


def dance_span(b, pad=PAD):
    """slice of the dance frames of a bundle (padded clip / rollout: drop the hold frames; unpadded reference: all)"""
    if b["kind"] == "sim" or b.get("padded"): return slice(pad, b["T"] - pad)
    return slice(0, b["T"])


def ref_metrics(b, mask="main", span=None, unsupported="skip"):
    """slip (stable/transition) + root-vs-feet consistency of a reference bundle under one named mask"""
    span = span or dance_span(b); m = b["masks"][mask]
    return dict(mask=mask, slip=_trim(foot_slip(b["cp_xy"], b["cp_h"], m, b["fps"]), span), consistency=_trim(root_feet_consistency(b["root_p"][:, :2], b["ank_xy"], m, b["fps"], span, unsupported=unsupported)),
                mask_frac=float(m[span].mean()))


def sim_metrics(s, ref=None, span=None, unsupported="skip"):
    """full unified metric set of a rollout bundle; `ref` (ref_bundle of the reference it was deployed on, same frame indexing) enables the `common` mask, the confusion matrix and the decomposition"""
    span = span or dance_span(s); m = s["masks"]["main"]; out = {}
    out["slip_own"] = _trim(foot_slip(s["cp_xy"], s["cp_h"], m, s["fps"]), span)
    out["consistency_own"] = _trim(root_feet_consistency(s["root_p"][:, :2], s["ank_xy"], m, s["fps"], span, unsupported=unsupported))
    out["flicker"] = flicker(m, s["fps"], span); out["load_frac"] = [float(m[span, f].mean()) for f in (0, 1)]
    if ref is not None:
        T = min(s["T"], ref["T"]); rm = ref["masks"]["main"][:T]; common = rm & m[:T]
        out["slip_common"] = _trim(foot_slip(s["cp_xy"][:T], s["cp_h"][:T], common, s["fps"]), span)
        out["consistency_common"] = _trim(root_feet_consistency(s["root_p"][:T, :2], s["ank_xy"][:T], common, s["fps"], span, unsupported=unsupported))
        out["confusion"] = {k: confusion(ref["masks"][k][:T][span], m[:T][span]) for k in ("main", "b1", "v11", "v11_25", "train")}
        out["airborne"] = airborne_split(m[:T], ref["flight"][:T], ref["hover"][:T], span)
        out["decomp"] = decompose_root_offset(s["root_p"][:T, :2], s["ank_xy"][:T], m[:T], ref["root_p"][:T, :2], ref["ank_xy"][:T], rm, span)
    return out


def _trim(d, span=None):
    """drop the big series from the slip / consistency dicts (keep stats); optionally recompute slip stats restricted to a span (a frame slice)"""
    if "series" in d:
        if span is not None:
            for key in ("ok_stable", "ok_trans"): d[key] = d[key].copy(); d[key][:span.start] = False; d[key][span.stop - 1:] = False  # speed index i is frames (i, i+1)
            d["stable"] = stats(d["series"][d["ok_stable"]]); d["transition"] = stats(d["series"][d["ok_trans"]])
        return {k: v for k, v in d.items() if k in ("stable", "transition")} | {"_ok_stable": d["ok_stable"], "_ok_trans": d["ok_trans"], "_speed": d["series"]}
    return d
