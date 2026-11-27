"""pmpipe.foot_posture - sole posture / support area / contact-morphology analysis (V1.7 C3).  Pure numpy; dance independent.

Why: the V1.6 contact refinement found that only 4% -> 17% of stance frames are FLAT-foot contact; most are heel-only / toe-only contact.  A heel strike or a tiptoe is a
legitimate dance posture, so the goal here is NOT to force a flat sole but to separate legitimate partial contacts from real artifacts:
  * single-frame / <3-frame morphology flicker                   (FLICKER)
  * a sole that sinks into the floor / screws in while sinking     (PEN, SCREW)
  * repeated heel<->toe rolling on a single support region        (ROLLING)
  * a tiny single-point support carrying a stance foot            (POINT_SUPPORT: CoP feasible length < min_len)
  * a tilted sole touching the floor by one or two edge points    (EDGE_HOVER)
Inputs are sole sphere centres (T, 2, 14, 3) (left/right, 14 collision-capsule end points, `contact_refine.geometry.model()['loc']` layout), the ankle rotation (T,2,3,3) and
a floor height (scalar or (T,)), so the same code runs on reference motions (floor = calibrated ground curve) and on closed-loop rollouts (floor = 0).

Support area = area of the convex hull of the contacting sphere centres inflated by the sphere radius (hull area + perimeter*r + pi r^2); CoP feasible length = x / y extent
of the contacting centres + 2r along the foot axes.  Morphology codes: MORPH = ['none', 'flat', 'heel', 'toe', 'edge'].
"""
from __future__ import annotations

import numpy as np

MORPH = ["none", "flat", "heel", "toe", "edge"]
NONE, FLAT, HEEL, TOE, EDGE = range(5)
F_FLICKER, F_PEN, F_SCREW, F_ROLLING, F_POINT, F_EDGE_HOVER = 1, 2, 4, 8, 16, 32
FLAG_NAMES = {F_FLICKER: "FLICKER", F_PEN: "PEN", F_SCREW: "SCREW", F_ROLLING: "ROLLING", F_POINT: "POINT_SUPPORT", F_EDGE_HOVER: "EDGE_HOVER"}

DEFAULTS = dict(contact_mm=8.0, stance_mm=15.0, pen_mm=5.0, screw_pen_mm=2.0, screw_yaw=1.0, screw_hspeed=0.25, flicker_run=3, min_reasonable_run=8, rolling_window=10, rolling_flips=2,
                point_min_len_mm=30.0, edge_hover_tilt=0.35, edge_hover_pts=2, planted_hspeed=0.15, transition_win=8)


# ------------------------------------------------------------------------------------------------------------------------------ geometry
def sole_layout(robot: str = "g1mj"):
    """(loc (2,14,3) ankle-frame sample points, tags (14,), radius) from the contact_refine geometry model (heel / mid / toe by the sample's x)."""
    from .contact_refine import geometry as Gm
    M = Gm.model(robot)
    return M["loc"], M["tag"], float(M["radius"])


def convex_hull(P):
    """Andrew monotone chain; P (n,2) -> hull vertices (CCW, no repeats). n<3 returns the unique points."""
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in np.asarray(P).reshape(-1, 2)})
    if len(pts) <= 2: return np.array(pts, float).reshape(-1, 2)
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo = []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    up = []
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1], float)


def support_area(P, radius: float = 0.01) -> float:
    """area of the convex hull of sphere centres P (n,2) inflated by `radius` [m^2]; 0 points -> 0, 1 point -> pi r^2."""
    P = np.asarray(P, float).reshape(-1, 2)
    if len(P) == 0: return 0.0
    h = convex_hull(P)
    if len(h) == 1: return float(np.pi * radius ** 2)
    if len(h) == 2: return float(2 * radius * np.linalg.norm(h[1] - h[0]) + np.pi * radius ** 2)
    a = 0.5 * abs(sum(h[i, 0] * h[(i + 1) % len(h), 1] - h[(i + 1) % len(h), 0] * h[i, 1] for i in range(len(h))))
    per = sum(np.linalg.norm(h[(i + 1) % len(h)] - h[i]) for i in range(len(h)))
    return float(a + per * radius + np.pi * radius ** 2)


def morphology_of(contact, loc_xy, tags, side: int):
    """contact (14,) bool, loc_xy (14,2) in the foot frame -> morph code. flat = heel AND toe region touch and the contact is not confined to one lateral side;
    heel / toe = only that region (mid points count to neither); edge = heel & toe both touch but ONE lateral side only (inner/outer edge) or only mid points touch."""
    if not contact.any(): return NONE
    h = bool(contact[tags == "heel"].any()); t = bool(contact[tags == "toe"].any())
    y = loc_xy[contact, 1]; one_side = (y > 0.004).all() or (y < -0.004).all()
    if h and t: return EDGE if one_side else FLAT
    if h: return HEEL
    if t: return TOE
    return EDGE


# ------------------------------------------------------------------------------------------------------------------------------ per-frame series
def foot_posture_series(sole, ank_R, radius, floor, fps, loc=None, tags=None, params=None):
    """sole (T,2,14,3), ank_R (T,2,3,3), floor scalar/(T,) -> dict of (T,2) arrays: morph, n_contact, area, len_x, len_y, tilt, pen_mm, hspeed, yawrate, stance, min_h_mm + H (T,2,14)."""
    p = {**DEFAULTS, **(params or {})}
    if loc is None or tags is None: loc, tags, radius0 = sole_layout()
    sole = np.asarray(sole, float); T = len(sole); fl = np.broadcast_to(np.asarray(floor, float).reshape(-1) if np.ndim(floor) else np.full(T, float(floor)), (T,))
    H = sole[..., 2] - radius - fl[:, None, None]
    ct = H <= p["contact_mm"] / 1000
    out = {k: np.zeros((T, 2)) for k in ("n_contact", "area", "len_x", "len_y", "tilt", "pen_mm", "hspeed", "yawrate", "min_h_mm")}
    morph = np.zeros((T, 2), np.int8)
    R = np.asarray(ank_R, float)
    for f in (0, 1):
        for t in range(T):
            c = ct[t, f]; n = int(c.sum()); out["n_contact"][t, f] = n
            if n:
                xy = loc[f, c][:, :2]
                out["area"][t, f] = support_area(xy, radius)
                out["len_x"][t, f] = np.ptp(xy[:, 0]) + 2 * radius; out["len_y"][t, f] = np.ptp(xy[:, 1]) + 2 * radius
            morph[t, f] = morphology_of(c, loc[f][:, :2], tags, f)
    out["morph"] = morph
    out["tilt"] = np.arccos(np.clip(R[:, :, 2, 2], -1, 1))
    out["min_h_mm"] = H.min(-1) * 1000
    out["pen_mm"] = np.maximum(-H.min(-1), 0) * 1000
    dt = 1.0 / fps
    # lowest-point horizontal speed and foot yaw rate
    for f in (0, 1):
        i = H[:, f].argmin(-1); low = sole[np.arange(T), f, i]
        ar = np.arange(1, T); v = np.zeros(T); v[1:] = np.linalg.norm(sole[ar, f, i[ar], :2] - sole[ar - 1, f, i[ar], :2], axis=1) / dt   # speed of the SAME sample that is lowest now (a change of the lowest sample is not slip)
        out["hspeed"][:, f] = v
        yaw = np.unwrap(np.arctan2(R[:, f, 1, 0], R[:, f, 0, 0])); yr = np.zeros(T); yr[1:] = np.abs(np.diff(yaw)) / dt; out["yawrate"][:, f] = yr
    out["stance"] = out["min_h_mm"] <= p["stance_mm"]
    out["H"] = H
    out["fps"] = fps; out["params"] = p
    return out


def morph_runs(code):
    """[(start, end, code)] runs of equal morphology code of one foot."""
    r, s = [], 0
    for i in range(1, len(code) + 1):
        if i == len(code) or code[i] != code[s]: r.append((s, i, int(code[s]))); s = i
    return r


# ------------------------------------------------------------------------------------------------------------------------------ artifact classification
def classify_artifacts(P, params=None):
    """P = foot_posture_series output.  Returns dict: flags (T,2) int bitmask, label (T,2) str-coded int
    0 none (no contact) / 1 flat (full sole) / 2 reasonable partial contact / 3 uncertain / 4 artifact, plus per-run table `runs`."""
    p = {**DEFAULTS, **(P.get("params") or {}), **(params or {})}
    T = len(P["morph"]); fl = np.zeros((T, 2), np.int32); label = np.zeros((T, 2), np.int8); fps = P["fps"]; runs_out = []
    for f in (0, 1):
        m = P["morph"][:, f]; rr = morph_runs(m)
        for k, (a, b, c) in enumerate(rr):
            prev = rr[k - 1][2] if k > 0 else None; nxt = rr[k + 1][2] if k + 1 < len(rr) else None
            if (b - a) < p["flicker_run"] and prev is not None and nxt is not None and prev == nxt and c != prev: fl[a:b, f] |= F_FLICKER      # blip between two equal neighbours
            elif (b - a) < p["flicker_run"] and prev is not None and nxt is not None and c != NONE and prev != c and nxt != c and b - a == 1: fl[a:b, f] |= F_FLICKER
        # rolling: >= rolling_flips heel<->toe changes (through flat or direct) inside one window, on a foot that stays in stance
        sw = np.where(np.isin(m, (HEEL, TOE)))[0]; ch = []
        last = None
        for t in range(T):
            if m[t] in (HEEL, TOE):
                if last is not None and last[1] != m[t]: ch.append(t)
                last = (t, m[t])
        for t0 in ch:
            grp = [t1 for t1 in ch if 0 <= t1 - t0 < p["rolling_window"]]
            if len(grp) >= p["rolling_flips"]: fl[t0:grp[-1] + 1, f] |= F_ROLLING        # only the frames of the flip cluster, not the long runs around it
        pen = P["pen_mm"][:, f]
        fl[pen > p["pen_mm"], f] |= F_PEN
        fl[(pen > p["screw_pen_mm"]) & ((P["yawrate"][:, f] > p["screw_yaw"]) | (P["hspeed"][:, f] > p["screw_hspeed"])), f] |= F_SCREW
        st = P["stance"][:, f] & (P["morph"][:, f] != NONE)
        fl[st & (P["len_x"][:, f] * 1000 < p["point_min_len_mm"]) & (P["n_contact"][:, f] <= 1), f] |= F_POINT
        fl[st & (P["tilt"][:, f] > p["edge_hover_tilt"]) & (P["n_contact"][:, f] <= p["edge_hover_pts"]), f] |= F_EDGE_HOVER
        # labels: flagged frames (+-1 frame) are artifacts; every maximal un-flagged piece of a morphology run is judged on its own (a long heel strike with one bad frame stays reasonable)
        bad = (fl[:, f] & (F_FLICKER | F_PEN | F_SCREW | F_ROLLING | F_POINT | F_EDGE_HOVER)) > 0
        bad = bad | np.r_[bad[1:], False] | np.r_[False, bad[:-1]]
        for (a, b, c) in rr:
            if c == NONE: continue
            i = a
            while i < b:
                j = i
                while j < b and bad[j] == bad[i]: j += 1
                if bad[i]: label[i:j, f] = 4; runs_out.append((f, i, j, c, "artifact"))
                elif c == FLAT: label[i:j, f] = 1; runs_out.append((f, i, j, c, "flat"))
                else:
                    planted = float(np.mean(P["hspeed"][i:j, f])) < p["planted_hspeed"]
                    if (j - i) >= p["min_reasonable_run"] and planted and P["stance"][i:j, f].mean() >= 0.8: label[i:j, f] = 2; runs_out.append((f, i, j, c, "reasonable"))
                    else: label[i:j, f] = 3; runs_out.append((f, i, j, c, "uncertain"))
                i = j
    return dict(flags=fl, label=label, runs=runs_out)


LABEL_NAMES = {0: "no_contact", 1: "flat", 2: "reasonable_partial", 3: "uncertain", 4: "artifact"}


def label_fractions(A, only_stance=None):
    """fraction of frames per label (over frames of `only_stance` mask if given, else all frames of both feet)."""
    lab = A["label"]; m = np.ones_like(lab, bool) if only_stance is None else only_stance
    n = max(int(m.sum()), 1)
    return {LABEL_NAMES[k]: float(((lab == k) & m).sum() / n) for k in LABEL_NAMES}


# ------------------------------------------------------------------------------------------------------------------------------ transitions
def transitions(P, win=None):
    """touch-down / lift-off events (stance flag edges) per foot and the support area / lowest-point speed / morph in +-win frames around them.
    returns dict(events=[(foot, frame, kind)], area (n, 2*win+1), speed (n, 2*win+1), kind (n,)) ; windows are NaN outside the clip."""
    w = win or P["params"]["transition_win"]; st = P["stance"]; T = len(st); ev = []
    for f in (0, 1):
        d = np.diff(st[:, f].astype(int))
        ev += [(f, int(t + 1), "touchdown") for t in np.where(d == 1)[0]] + [(f, int(t + 1), "liftoff") for t in np.where(d == -1)[0]]
    ev.sort(key=lambda e: e[1])
    area = np.full((len(ev), 2 * w + 1), np.nan); sp = np.full_like(area, np.nan); mo = np.full((len(ev), 2 * w + 1), -1, np.int8)
    for i, (f, t, k) in enumerate(ev):
        lo, hi = max(0, t - w), min(T, t + w + 1)
        area[i, lo - (t - w):hi - (t - w)] = P["area"][lo:hi, f] * 1e4; sp[i, lo - (t - w):hi - (t - w)] = P["hspeed"][lo:hi, f]; mo[i, lo - (t - w):hi - (t - w)] = P["morph"][lo:hi, f]
    return dict(events=ev, area_cm2=area, speed=sp, morph=mo, kind=np.array([e[2] for e in ev]), win=w)


def transition_summary(Tr):
    """max |d area|/frame in the window and corr(area jump, slip) over events."""
    if not len(Tr["events"]): return dict(n=0)
    A, S = Tr["area_cm2"], Tr["speed"]
    jump = np.nanmax(np.abs(np.diff(A, axis=1)), axis=1); smax = np.nanmax(S, axis=1)
    ok = np.isfinite(jump) & np.isfinite(smax)
    c = float(np.corrcoef(jump[ok], smax[ok])[0, 1]) if ok.sum() > 3 and np.std(jump[ok]) > 0 and np.std(smax[ok]) > 0 else float("nan")
    return dict(n=int(len(jump)), max_area_jump_cm2_p50=float(np.nanmedian(jump)), max_area_jump_cm2_p95=float(np.nanpercentile(jump, 95)), max_slip_p50=float(np.nanmedian(smax)),
                max_slip_p95=float(np.nanpercentile(smax, 95)), corr_jump_slip=c)


# ------------------------------------------------------------------------------------------------------------------------------ FK helpers
def motion_posture(root_p, root_q, q, fps, floor=0.0, robot="g1mj", params=None):
    """FK of a motion / rollout (root_p (T,3), root_q wxyz (T,4), q (T,29)) -> foot_posture_series (+artifact classification under key 'A')."""
    from .contact_refine import geometry as Gm
    F = Gm.fk(root_p, root_q, q, robot)
    loc, tags, r = sole_layout(robot)
    P = foot_posture_series(F["sole"], F["ank_R"], r, floor, fps, loc, tags, params)
    P["A"] = classify_artifacts(P)
    P["com"] = F["com"]
    return P


def morph_agreement(m_ref, m_pol):
    """confusion (5x5) of policy vs reference morphology codes per frame/foot (rows = reference) + conversion fractions."""
    C = np.zeros((5, 5), int)
    for a, b in zip(np.asarray(m_ref).ravel(), np.asarray(m_pol).ravel()): C[int(a), int(b)] += 1
    return C
