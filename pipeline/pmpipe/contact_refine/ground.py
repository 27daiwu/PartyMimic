"""contact_refine.ground - ground calibration.

The retargeted references float / sink by a SLOWLY varying amount (video root-height drift) and the converter aligned only the single lowest sample of the whole
clip to z=0.  We estimate the support height curve g(t) as a smooth LOWER ENVELOPE of the lowest-foot height (asymmetric least squares: points above the curve
are cheap because a foot may be in the air, points below are expensive because the floor is rigid), with
  * a curvature limit (equivalent cut-off `ground_fc_hz`, so stance-foot jitter is NOT absorbed into the ground),
  * static-stance anchoring (both feet loaded, weight x static_boost),
  * zero weight inside validated flights (the ground there is interpolated, a real jump is never pulled to the floor),
  * a drift-rate bound (the fit is re-run with a stiffer curve until max|g'| <= ground_max_rate or 4 doublings),
and a per-frame confidence = density of near-floor stance evidence around the frame.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .geometry import runs


def static_mask(jv, root_p, fps, p):
    rms = np.sqrt((jv ** 2).mean(1)); rv = np.linalg.norm(np.gradient(root_p, 1 / fps, axis=0), axis=1)
    m = (rms < p.static_jv_rms) & (rv < p.static_root_v)
    out = np.zeros_like(m)
    for a, b in runs(m, int(p.static_min_s * fps)): out[a:b] = True
    return out


def _d2(n):
    return sp.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n)).tocsc()


def als_lower_envelope(y, w0, lam, pw, iters=30):
    n = len(y); D = _d2(n); P = lam * (D.T @ D); g = np.full(n, np.nanmin(y)); g = np.minimum.accumulate(y[::-1])[::-1] * 0 + np.median(y)
    for _ in range(iters):
        w = np.where(y > g, pw, 1.0 - pw) * w0
        gn = spsolve((sp.diags(w) + P + 1e-9 * sp.eye(n)).tocsc(), w * y)
        if np.max(np.abs(gn - g)) < 1e-6: g = gn; break
        g = gn
    return g


def calibrate_ground(lowest, jv, root_p, fps, p, flight=None):  # confidence = share of +-0.5 s with the lowest foot slow (|dz/dt|<0.15 m/s) and within 12 mm of the curve
    """lowest: (T,2) lowest sole-point height of each foot above z=0.  returns dict(g (T,), conf (T,), static, info)"""
    T = len(lowest); y = lowest.min(1); stat = static_mask(jv, root_p, fps, p)
    w0 = np.ones(T); w0[stat] = p.static_boost
    if flight is not None: w0[flight] = 0.0
    # anchor the very first / last frames if they are static (clip start / end hold)
    s = 2 * np.sin(np.pi * p.ground_fc_hz / fps); lam = (1.0 / s) ** 4
    rate_ok = False
    for k in range(5):
        g = als_lower_envelope(y, w0, lam, p.als_p); rate = float(np.abs(np.gradient(g, 1 / fps)).max())
        if rate <= p.ground_max_rate: rate_ok = True; break
        lam *= 4.0
    res = y - g
    vz = np.abs(np.gradient(np.convolve(np.pad(y, 2, mode='edge'), np.ones(5) / 5, 'valid'), 1 / fps)); near = (res < 0.012) & (w0 > 0) & (vz < 0.15)
    dens = np.convolve(near.astype(float), np.ones(int(fps)) / fps, "same")  # fraction of the surrounding second with evidence of floor contact
    conf = np.clip(dens / 0.5, 0, 1)
    info = dict(method="ALS lower envelope + static anchor", lam=float(lam), fc_hz=p.ground_fc_hz, rate_max_mps=rate, rate_ok=rate_ok, range_mm=[float(g.min() * 1000), float(g.max() * 1000)],
                static_frames=int(stat.sum()), resid_below_mm_p99=float(np.percentile(np.maximum(-res, 0), 99) * 1000), resid_above_median_mm=float(np.median(res) * 1000),
                frac_below_5mm=float((res < -0.005).mean()), conf_mean=float(conf.mean()))
    return dict(g=g, conf=conf, static=stat, info=info)
