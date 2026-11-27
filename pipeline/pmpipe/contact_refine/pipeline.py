"""contact_refine.pipeline - orchestration: analyse a 50 Hz motion, refine it (root XYZ + leg IK), re-analyse, compare, judge.

Flow  (dance-independent):
  1 FK of the original -> lowest-sole heights
  2 ground curve g(t)  (ALS lower envelope, static anchor)  ->  contact identification (stance/swing/uncertain, support inference)  -> validated flights
     (if a true flight exists the ground is re-fitted with the flight excluded and the identification repeated)
  3 stage 1: root-XYZ optimisation (linear least squares)           -> `root_only` variant (kept for the report as ablation)
  4 stage 2: joint-space IK on the legs + root re-fit (sparse Gauss-Newton, bounded deltas, delta smoothness)  -> final
  5 re-analysis of the result against the true floor z=0, metrics, QA verdict
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import contacts, geometry as Gm, ground, metrics, solver
from .config import Params


def analyze_arrays(root_p, root_q, q, jv, fps, p, g=None, static=None, log=print):
    """FK + ground + identification. g given = use it (e.g. zeros for a refined motion)."""
    F = Gm.fk(root_p, root_q, q); low = Gm.sole_height(F).min(-1)
    if g is None:
        G = ground.calibrate_ground(low, jv, root_p, fps, p); ident = contacts.identify(F, G["g"], fps, p, G["static"])
        if ident["flight"]["mask"].any():  # a validated flight must not drag the ground: refit without it, identify again
            G = ground.calibrate_ground(low, jv, root_p, fps, p, flight=np.convolve(ident["flight"]["mask"].astype(float), np.ones(7), "same") > 0)
            ident = contacts.identify(F, G["g"], fps, p, G["static"])
        g = G["g"]; ginfo = G
    else:
        stat = ground.static_mask(jv, root_p, fps, p) if static is None else static
        ident = contacts.identify(F, g, fps, p, stat); ginfo = dict(g=g, conf=np.ones(len(g)), static=stat, info=dict(method="given"))
    return dict(F=F, low=low, ground=ginfo, g=g, ident=ident)


def lift_to_floor(root_p, root_q, q, p, fps):
    """hard no-penetration guarantee: raise the root by the (smoothed, never smaller) amount the lowest sole sample of either foot sinks below -0.5 mm."""
    F = Gm.fk(root_p, root_q, q); h = (F["sole"][..., 2] - F["radius"]).min((1, 2)); need = np.maximum(-h - 0.0005, 0.0)
    if need.max() <= 0: return root_p
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    lift = uniform_filter1d(maximum_filter1d(need, 5), 5); lift = np.maximum(lift, need)
    out = root_p.copy(); out[:, 2] += lift; return out


def refine(M, p: Params = None, log=print, variants=True):
    """M = geometry.load_motion(...).  returns dict with original analysis, refined arrays, stage-1 variant, solver info."""
    p = p or Params(); fps = M["fps"]; T = len(M["q"]); t0 = time.time()
    log("[1/5] analysis of the original ...")
    A0 = analyze_arrays(M["root_p"], M["root_q"], M["q"], M["jv"], fps, p, log=log)
    st, conf, fl = A0["ident"]["stance"], A0["ident"]["conf"], A0["ident"]["flight"]["mask"]
    log(f"      ground range {A0['ground']['info']['range_mm']} mm, stance frac {st.mean(0).round(2)}, inferred {A0['ident']['inferred'].mean(0).round(3)}, true flight frames {int(fl.sum())}")
    log("[2/5] stage 1: root XYZ optimisation ...")
    s1 = solver.solve(M["root_p"], M["root_q"], M["q"], A0["g"], st, conf, fl, p, ik=False, iters=2, log=log, fps=fps)
    out = dict(A0=A0, s1=s1, params=p)
    root1 = M["root_p"] + s1["d"]
    if p.ik:
        log("[3/5] stage 2: leg IK + root re-fit ...")
        s2 = solver.solve(M["root_p"], M["root_q"], M["q"], np.zeros(T), st, conf, fl, p, ik=True, init=s1, log=log, fps=fps)
        qn = M["q"].copy(); allc = np.array(Gm.leg_idx(0) + Gm.leg_idx(1)); qn[:, allc] += s2["u"]; rootn = M["root_p"] + s2["d"]
    else:
        s2 = s1; qn = M["q"].copy(); rootn = root1
    rootn = lift_to_floor(rootn, M["root_q"], qn, p, fps)
    out.update(s2=s2, root_p=rootn, q=qn, root_only=dict(root_p=root1, q=M["q"].copy()), seconds=time.time() - t0)
    return out
