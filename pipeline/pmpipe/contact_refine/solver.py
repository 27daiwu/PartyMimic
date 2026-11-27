"""contact_refine.solver - one sparse Gauss-Newton solver for BOTH the root-XYZ optimisation (stage 1, ik=False: exactly linear) and the joint-space IK
(stage 2, ik=True).  Variables per frame: root translation delta d_t (3) and the leg-joint deltas u_t (12, left/right hip pitch-roll-yaw / knee / ankle pitch-roll);
plus one planted-anchor xy `c` per (stance run, sole sample).  Nothing else moves: root orientation, waist, arms, head are fixed inputs.

Residuals (all in metres, damped Huber on the planted-xy rows):
  pin xy   : planted sole sample (stance foot, sample lower than pin_h over the ground) stays at its anchor c (no sliding)
  pin z    : those samples lie on the floor (z = 0)           ;  flat : slightly tilted stance soles are weakly flattened (ankle)
  hinge    : NO sole sample of either foot may sink below the floor (active-set, escalating)
  regular. : |d|, |u|/sigma (stance weak / swing strong), 2nd difference of the DELTAS d and u (their smoothness, not the trajectory's -> dance high frequencies
             survive), 1st difference of u (velocity change); inside validated flights the correction of the root is almost a rigid shift (ballistic shape kept).
The Jacobians come from MuJoCo (mj_jac on the g1mj model) and are recomputed each iteration; joint deltas are box-bounded (barrier + final clip) and joint-limit clipped.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from . import geometry as Gm


def _diff_op(T, order):
    if order == 1: return sp.diags([-1, 1], [0, 1], shape=(T - 1, T), dtype=float).tocsr()
    return sp.diags([1, -2, 1], [0, 1, 2], shape=(T - 2, T), dtype=float).tocsr()


def joint_bounds(p):
    b = np.full(12, np.radians(p.leg_bound_deg)); b[[4, 5, 10, 11]] = np.radians(p.ankle_bound_deg); return b


def solve(root_p, root_q, q, ground, stance, conf, flight, p, ik=True, init=None, robot="g1mj", log=print, iters=None, fps=50.0):
    """root_p (T,3) root_q (T,4) q (T,29): ORIGINAL trajectory (never modified).  ground: (T,) floor height at the start of the solve (heights are measured above it).
    stance (T,2) bool, conf (T,2), flight (T,) bool.  Returns dict(d, u, c, ...)."""
    T = len(q); M = Gm.model(robot); R = M["radius"]; nu = 12 if ik else 0; nb = 3 + nu; iters = iters or p.gn_iters
    legc = [Gm.leg_idx(0, robot), Gm.leg_idx(1, robot)]; allc = np.array(legc[0] + legc[1])
    jlo, jhi = M["jr"][allc, 0], M["jr"][allc, 1]; ub = joint_bounds(p)
    d = np.zeros((T, 3)); u = np.zeros((T, 12))
    if init is not None: d, u = init["d"].copy(), init["u"].copy()
    qk = lambda: _qfull(q, u, allc)
    # ---- pin sets from the geometry at the start of this solve (heights above `ground`)
    F0 = Gm.fk(root_p + d, root_q, qk(), robot); H0 = F0["sole"][..., 2] - R - ground[:, None, None]
    pin = stance[..., None] & (H0 < p.pin_h_mm / 1000)  # sole samples that stay put (xy) while the foot is planted
    pinz = pin & (H0 <= H0.min(-1, keepdims=True) + p.pinz_band_mm / 1000)  # only the LOWEST samples are forced onto the floor (a tilted rigid sole must not be pushed through it)
    flat = stance[..., None] & (H0 < p.flat_h_mm / 1000) & ~pinz
    unit = -np.ones((T, 2, 14), int); nC = 0
    for f in (0, 1):
        for a, b in Gm.runs(stance[:, f], 2):
            for i in range(14):
                fr = np.where(pin[a:b, f, i])[0] + a
                if len(fr) >= 2: unit[fr, f, i] = nC; nC += 1
    cxy = np.zeros((max(nC, 1), 2))
    wd2 = np.full(T, p.w_d2)
    if flight is not None and flight.any():
        fl = np.convolve(flight.astype(float), np.ones(5), "same") > 0; wd2[fl] = p.w_d2_flight
    sm = np.stack([np.convolve(stance[:, f].astype(float), np.ones(5) / 5, "same") for f in (0, 1)], 1)  # smoothed stance indicator
    w_u0 = (p.w_u0_swing * (1 - sm) + p.w_u0_stance * sm)  # (T,2)
    sigma = np.radians(p.ik_joint_sigma_deg)
    N = T * nb + 2 * max(nC, 1); info = []
    for it in range(iters):
        qf = qk(); root_k = root_p + d
        P = np.zeros((T, 2, 14, 3)); J = np.zeros((T, 2, 14, 3, 6))
        if ik or it == 0:
            for f in (0, 1): J[:, f], P[:, f] = Gm.point_jacobians(root_k, root_q, qf, f, legc[f], robot)
        else:
            P = Gm.fk(root_k, root_q, qf, robot)["sole"]
        if not ik: J[:] = 0
        Hh = P[..., 2] - R  # absolute height above the FINAL floor z=0 (the estimated floor `ground` only chose the pin sets)
        # c current estimate: mean planted xy of every unit
        if nC:
            tt, ff, ii = np.where(unit >= 0); uu = unit[tt, ff, ii]; cnt = np.bincount(uu, minlength=nC).astype(float)
            for a in (0, 1): cxy[:nC, a] = np.bincount(uu, weights=P[tt, ff, ii, a], minlength=nC) / np.maximum(cnt, 1)
        rows, cols, vals, rhs = [], [], [], []; nrow = 0
        def add(rr, cc, vv, bb):
            nonlocal nrow
            rows.append(rr + nrow); cols.append(cc); vals.append(vv); rhs.append(bb); nrow += int(len(bb))
        def jcols(t, f):  # (n,6) column indices of the leg-f joint deltas at frame t
            return (t * nb + 3 + 6 * f)[:, None] + np.arange(6)[None, :]
        def point_rows(t, f, i, comp, target_resid, w, c_idx=None):
            """rows for the linearised component `comp` of sole sample (t,f,i): resid = base + dd_comp + J.du (- dc)."""
            n = len(t); r = np.arange(n); rr = [r]; cc = [t * nb + comp]; vv = [w.copy()]
            if ik:
                jr_ = np.repeat(r, 6); rr.append(jr_); cc.append(jcols(t, f).ravel()); vv.append((J[t, f, i, comp, :] * w[:, None]).ravel())
            if c_idx is not None: rr.append(r); cc.append(T * nb + 2 * c_idx + comp); vv.append(-w)
            add(np.concatenate(rr), np.concatenate(cc), np.concatenate(vv), -target_resid * w)
        # --- planted xy
        npin_xy = 0
        if nC:
            tt, ff, ii = np.where(unit >= 0); uu = unit[tt, ff, ii]
            rx = P[tt, ff, ii, :2] - cxy[uu]; rn = np.linalg.norm(rx, axis=1); hub = np.minimum(1.0, (p.huber_mm / 1000) / np.maximum(rn, 1e-9)) ** 0.5
            w = p.w_pin_xy * np.sqrt(conf[tt, ff]) * hub; npin_xy = len(tt)
            for a in (0, 1): point_rows(tt, ff, ii, a, rx[:, a], w, uu)
        # --- pin z (floor contact), flatten, hinge
        tt, ff, ii = np.where(pinz)
        if len(tt): point_rows(tt, ff, ii, 2, Hh[tt, ff, ii] - p.pin_z_target_mm / 1000, p.w_pin_z * np.sqrt(conf[tt, ff]), None)
        tt, ff, ii = np.where(flat)
        if len(tt): point_rows(tt, ff, ii, 2, Hh[tt, ff, ii], np.full(len(tt), p.w_flat), None)
        act = (Hh < 0.001) & ~pinz; tt, ff, ii = np.where(act)
        if len(tt): point_rows(tt, ff, ii, 2, Hh[tt, ff, ii] - p.pen_tol_mm / 1000, np.full(len(tt), p.w_pen * (1 + it)), None)
        # --- regularisers on the totals
        Xk = np.zeros(T * nb); Xk.reshape(T, nb)[:, :3] = d
        if ik: Xk.reshape(T, nb)[:, 3:] = u
        wvec = np.zeros(nb); wvec[:3] = 1
        def reg(op, wrow, label):
            nonlocal nrow
            Wm = sp.diags(wrow) @ op; Wm = Wm.tocoo(); bb = -(Wm @ np.pad(Xk, (0, N - T * nb)))
            add(Wm.row, Wm.col, Wm.data, bb)
        I_T = sp.identity(T * nb, format="csr"); Ipad = sp.hstack([I_T, sp.csr_matrix((T * nb, N - T * nb))]).tocsr()
        # |d|, |u|/sigma (+ barrier for the box bound)
        w0 = np.zeros((T, nb)); w0[:, :3] = p.w_d0
        if ik:
            for f in (0, 1): w0[:, 3 + 6 * f: 9 + 6 * f] = (np.sqrt(w_u0[:, f:f + 1]) / sigma)
            ratio = np.abs(u) / ub[None, :]; bar = np.clip((ratio - 0.8) / 0.2, 0, None); w0[:, 3:] = np.sqrt(w0[:, 3:] ** 2 + (30.0 * bar / sigma) ** 2)
        reg(Ipad, w0.ravel(), "0")
        D2 = sp.kron(_diff_op(T, 2), sp.identity(nb), format="csr"); D2 = sp.hstack([D2, sp.csr_matrix((D2.shape[0], N - T * nb))]).tocsr()
        w2 = np.zeros((T - 2, nb)); w2[:, :3] = np.sqrt(wd2[1:-1])[:, None] * fps ** 2 / 10.0  # acceleration of the root correction in units of 10 m/s^2
        if ik: w2[:, 3:] = np.sqrt(p.w_u2) * fps ** 2 / 50.0  # joint-correction acceleration in units of 50 rad/s^2
        reg(D2, w2.ravel(), "d2")
        D1 = sp.kron(_diff_op(T, 1), sp.identity(nb), format="csr"); D1 = sp.hstack([D1, sp.csr_matrix((D1.shape[0], N - T * nb))]).tocsr()
        w1 = np.zeros((T - 1, nb)); w1[:, :3] = np.sqrt(p.w_d1) * fps / 0.3  # velocity of the root correction in units of 0.3 m/s
        if ik: w1[:, 3:] = np.sqrt(p.w_u1) * fps / 2.0  # joint-correction velocity in units of 2 rad/s
        reg(D1, w1.ravel(), "d1")
        add(np.arange(2 * max(nC, 1)), T * nb + np.arange(2 * max(nC, 1)), np.full(2 * max(nC, 1), 1e-3), np.zeros(2 * max(nC, 1)))
        # Levenberg-Marquardt damping of the increments (trust region): keeps the linearisation valid
        lm = np.zeros((T, nb)); lm[:, :3] = p.lm_d
        if ik: lm[:, 3:] = p.lm_u / sigma
        add(np.arange(T * nb), np.arange(T * nb), lm.ravel(), np.zeros(T * nb))
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nrow, N)).tocsr(); b = np.concatenate(rhs)
        AtA = (A.T @ A).tocsc() + 1e-10 * sp.identity(N, format="csc"); x = spsolve(AtA, A.T @ b)
        dX = x[:T * nb].reshape(T, nb); step_d = dX[:, :3]; d = d + step_d
        if ik: u = u + dX[:, 3:]; u = np.clip(u, -ub[None, :], ub[None, :]); u = np.clip(qk()[:, allc], jlo + 1e-3, jhi - 1e-3) - q[:, allc]
        # Gauss-Newton progress record
        res_z = Hh[pinz]; info.append(dict(iter=it, pins=int(pin.sum()), units=nC, step_d_max_mm=float(np.abs(step_d).max() * 1000), rows=int(nrow),
                                          pin_z_rms_mm_before=float(np.sqrt((res_z ** 2).mean()) * 1000) if res_z.size else None, pen_active=int(act.sum())))
        log(f"  GN it{it}: {info[-1]}")
    return dict(d=d, u=u, info=info, pin=pin, unit=unit, nC=nC)


def _qfull(q, u, allc):
    out = q.copy(); out[:, allc] += u; return out
