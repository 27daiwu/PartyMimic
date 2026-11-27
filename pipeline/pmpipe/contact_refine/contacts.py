"""contact_refine.contacts - per-foot contact identification (stance / swing / uncertain) + validated true flight.

Per foot and frame we fuse
  * height of the lowest sole point above the calibrated ground (logistic),
  * speed of the low sole points (horizontal and vertical, Gaussian),
  * static stance (both feet certainly loaded, both lowest points within `swing_h`),
into a stance PROBABILITY, make a hysteresis (Schmitt trigger + minimum run length) decision, and output a state + confidence.
Flight (both feet airborne) is accepted ONLY if the whole-body CoM obeys a ballistic parabola over the candidate run (vertical acceleration ~ -g, small fit
residual, plausible duration); otherwise the candidate is labelled `false_flight` (a hover caused by ground drift / retarget offset) and its frames are NOT flight.
"""
from __future__ import annotations

import numpy as np

from .geometry import G, runs

SWING, STANCE, UNCERTAIN = 0, 1, 2
STATE_NAMES = {0: "swing", 1: "stance", 2: "uncertain"}


def _smooth5(x, axis=0):
    k = np.array([1, 2, 3, 2, 1], float); k /= k.sum()
    pad = [(2, 2)] + [(0, 0)] * (x.ndim - 1); xp = np.pad(x, pad, mode="edge"); out = np.zeros_like(x)
    for i, w in enumerate(k): out += w * xp[i:i + len(x)]
    return out


def features(F, g, fps):
    """per-foot features. e: lowest height above ground (T,2); H: all samples (T,2,14); vh/vz: speed of the low samples; tilt: sole normal vs vertical [rad]"""
    dt = 1.0 / fps; S = F["sole"]; H = S[..., 2] - F["radius"] - g[:, None, None]; e = H.min(-1)
    V = _smooth5(np.gradient(S, dt, axis=0)); low = H <= (e[..., None] + 0.008); n = low.sum(-1)
    vh = (np.linalg.norm(V[..., :2], axis=-1) * low).sum(-1) / n; vz = (np.abs(V[..., 2]) * low).sum(-1) / n
    nz = np.clip(F["ank_R"][:, :, 2, 2], -1, 1); tilt = np.arccos(nz)
    return dict(e=e, H=H, vh=vh, vz=vz, tilt=tilt)


def stance_probability(feat, p, static=None):
    e = feat["e"]; ph = 1 / (1 + np.exp(-(p.stance_h_mm / 1000 - e) / (p.stance_tau_mm / 1000)))
    pv = np.exp(-0.5 * (feat["vh"] / p.v_h_ref) ** 2) * np.exp(-0.5 * (feat["vz"] / p.v_z_ref) ** 2)
    pr = ph * (0.25 + 0.75 * pv)
    if static is not None:  # both feet certainly loaded in a static stance (if they are anywhere near the floor)
        near = e < p.swing_h_mm / 1000
        pr = np.where(static[:, None] & near, np.maximum(pr, 0.9), pr)
    return pr


def _schmitt(pr, hi, lo):
    st = np.zeros(len(pr), bool); cur = pr[0] > hi
    for t in range(len(pr)):
        if cur and pr[t] < lo: cur = False
        elif (not cur) and pr[t] > hi: cur = True
        st[t] = cur
    return st


def _clean(mask, min_run):
    """remove runs shorter than min_run in both polarities (closing first, then opening)"""
    m = mask.copy()
    for a, b in runs(~m, 1):
        if b - a < min_run and a > 0 and b < len(m): m[a:b] = True
    for a, b in runs(m, 1):
        if b - a < min_run: m[a:b] = False
    return m


def classify_feet(feat, p, static=None):
    """returns dict(prob (T,2), state (T,2) int, conf (T,2), stance (T,2) bool)"""
    pr = stance_probability(feat, p, static); T = len(pr); st = np.zeros((T, 2), bool)
    for f in (0, 1): st[:, f] = _clean(_schmitt(pr[:, f], p.p_hi, p.p_lo), p.min_run)
    sure_swing = (feat["e"] > p.swing_h_mm / 1000) | (pr < 0.08)
    state = np.where(st, STANCE, np.where(sure_swing, SWING, UNCERTAIN)).astype(np.int8)
    for f in (0, 1):  # sure-swing must also survive the minimum run length, short blips become uncertain
        sw = _clean(state[:, f] == SWING, p.min_run)
        state[:, f] = np.where(st[:, f], STANCE, np.where(sw, SWING, UNCERTAIN))
    conf = np.where(state == STANCE, pr, np.where(state == SWING, 1 - pr, 0.5 * (1 - np.abs(2 * pr - 1))))
    return dict(prob=pr, state=state, conf=conf, stance=state == STANCE)


def _parabola(z, t):
    A = np.stack([np.ones_like(t), t, t * t], 1); c, *_ = np.linalg.lstsq(A, z, rcond=None); r = z - A @ c
    return 2 * c[2], float(np.sqrt((r ** 2).mean())), c


def detect_flight(feat, com, fps, p, stance):
    """both-feet-airborne candidates -> ballistic validation. returns dict(mask (T,) true flight, candidates [dict], false_flight mask)"""
    T = len(com); e = feat["e"]; cand = (e.min(1) > p.flight_h_mm / 1000) & (~stance.any(1))
    t = np.arange(T) / fps; out = []; true = np.zeros(T, bool); fake = np.zeros(T, bool)
    for a, b in runs(cand, p.flight_min_frames):
        # extend by 2 frames on each side (take-off / landing) for the fit; the CoM is smoothed lightly (the retarget is noisy)
        lo, hi = max(0, a - 2), min(T, b + 2); az, rms, c = _parabola(com[lo:hi, 2], t[lo:hi] - t[lo])
        dur = (b - a) / fps; ok_az = p.flight_az_range[0] <= az <= p.flight_az_range[1]
        ok = ok_az and rms * 1000 <= p.flight_fit_rms_mm and dur <= p.flight_max_s
        rise = float(com[lo:hi, 2].max() - 0.5 * (com[lo, 2] + com[hi - 1, 2]))
        out.append(dict(start=int(a), end=int(b), frames=int(b - a), duration_s=dur, fit_az=float(az), fit_rms_mm=rms * 1000, apex_rise_mm=rise * 1000, min_foot_height_mm=float(e[a:b].min(1).min() * 1000),
                        max_foot_height_mm=float(e[a:b].min(1).max() * 1000), true_flight=bool(ok), reason="ballistic" if ok else ("CoM acceleration %.1f m/s2 not ballistic" % az if not ok_az else "fit residual / duration")))
        (true if ok else fake)[a:b] = True
    return dict(mask=true, false_mask=fake, candidates=out)


def infer_support(c, feat, flight_mask, p):
    """Physical prior: without a validated flight a dancer always has a support foot.  Frames with NO stance foot get the lower, slower foot as an INFERRED stance foot
    (confidence capped at 0.5) provided it is within `infer_max_h_mm` of the ground curve (a foot the legs cannot reach is NOT forced down) and not sliding fast.  Returns the new c plus the boolean mask of inferred (T,2)."""
    st = c["stance"].copy(); T = len(st); inferred = np.zeros_like(st); e = feat["e"]
    need = (~st.any(1)) & (~flight_mask)
    for t in np.where(need)[0]:
        f = int(np.argmin(e[t] + 0.05 * feat["vh"][t]))
        if e[t, f] < p.infer_max_h_mm / 1000 and feat["vh"][t, f] < 0.6: st[t, f] = True; inferred[t, f] = True
    state = np.where(st, STANCE, c["state"]).astype(np.int8); conf = np.where(inferred, np.minimum(0.5, 1 - e / (p.swing_h_mm / 1000)).clip(0.15, 0.5), c["conf"])
    return dict(prob=c["prob"], state=state, conf=conf, stance=st), inferred


def identify(F, g, fps, p, static=None, infer=True):
    """full contact identification on one trajectory (needs FK dict F and ground curve g). One flight-consistency pass: candidates are validated, false flights stay non-flight."""
    feat = features(F, g, fps); c = classify_feet(feat, p, static); fl = detect_flight(feat, F["com"], fps, p, c["stance"])
    inferred = np.zeros_like(c["stance"])
    if infer: c, inferred = infer_support(c, feat, fl["mask"], p)
    reg = {}
    H = feat["H"]
    reg["sample_contact"] = (H < p.pin_h_mm / 1000) & c["stance"][..., None]  # (T,2,14)
    return dict(feat=feat, **c, inferred=inferred, flight=fl, regions=reg)
