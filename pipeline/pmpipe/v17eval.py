"""PartyMimic V1.7 (C2) - closed-loop evaluation of the short ablations (protocol: results/<dance>/v17/C_train/ABLATION_PROTOCOL.md).
usage (project python):  python -m pmpipe.v17eval diy --runs v17_A0 v17_A1 ... [--procs 20]
Per run: ONNX export -> deployment-chain rollouts on the run's own training reference (33 rollouts: nominal_ref + 16 perturbed nominal + push_f/push_b/delay10/noise x4) -> contact_metrics (unified,
main mask = MuJoCo load) + ftcompare._one (penetration, torque, guard, anchored MPJPE) + fixed-sample-identity slip + sole-tilt posture. Noise band = V1.6 ft_orig_v16 vs ft_orig_v16_s1 (same metrics, same 17 nominal rollouts).
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import abtest as AB, ftcompare as FC, contact_metrics as CM
from .core import Dance, Logger, read_json, write_json

CONDS4 = ["push_f", "push_b", "delay10", "noise"]
_REF: dict = {}


def _ref(p):
    if p not in _REF:
        _REF[p] = CM.ref_bundle(Path(p))
    return _REF[p]


def fixed_point_slip(s, k=CM.K_TRANS):
    """slip on FIXED sole sample identities (each of the 14 sample points of a foot is tracked as itself, not the lowest-point switching of contact_points): horizontal speed while the point is <10 mm
    and its foot is in stable load."""
    F = s["F"]; xy = F["sole"][..., :2]; h = F["sole"][..., 2] - F["radius"]; sp = np.linalg.norm(xy[1:] - xy[:-1], axis=-1) * s["fps"]; con = (h[1:] < CM.THR) & (h[:-1] < CM.THR) & (sp < CM.VMAX)
    m = s["masks"]["main"]; vals = []
    for f in (0, 1):
        d = CM.toggle_dist(m[:, f]); st = m[1:, f] & m[:-1, f] & (d[1:] >= k) & (d[:-1] >= k)
        v = sp[:, f][(con[:, f] & st[:, None])]; vals.append(v)
    return CM.stats(np.concatenate(vals))


def one(a):
    f, Pm, PF, PO = a
    out = FC._one((f, Pm, PF, PO))
    s = CM.sim_bundle(f); ref = _ref(Pm); sp = CM.dance_span(s)
    sm = CM.sim_metrics(s, ref=ref)
    cons = sm["consistency_own"]; g = cons["win_growth"]
    tilt = np.arccos(np.clip(s["F"]["ank_R"][:, :, 2, 2], -1, 1)); L = s["masks"]["main"]
    rl = [np.arccos(np.clip(ref["F"]["ank_R"][:, i, 2, 2], -1, 1))[CM.dance_span(ref)][ref["masks"]["main"][CM.dance_span(ref)][:, i]] for i in (0, 1)]
    tl = np.concatenate([tilt[sp][:, i][L[sp][:, i]] for i in (0, 1)])
    out.update(
        slip_stable_mean=sm["slip_own"]["stable"]["mean"], slip_stable_p95=sm["slip_own"]["stable"]["p95"], slip_trans_mean=sm["slip_own"]["transition"]["mean"], slip_trans_p95=sm["slip_own"]["transition"]["p95"],
        slip_fixed_mean=fixed_point_slip(s)["mean"], slip_fixed_p95=fixed_point_slip(s)["p95"],
        flicker=sm["flicker"]["toggles_per_s_per_foot"], flicker_short=sm["flicker"]["short_run_toggles_per_s_per_foot"],
        drift_win_mean=float(g.mean()), drift_win_25_45=float(g[5:].mean()) if len(g) > 5 else float("nan"), drift_win_max=float(g.max()), drift_cum_end=cons["cum_end_m"],
        agree_main=sm["confusion"]["main"]["agreement"], kappa_main=sm["confusion"]["main"]["kappa"], load_frac=float(np.mean(sm["load_frac"])),
        tilt_loaded_deg=float(np.degrees(tl).mean()), tilt_loaded_gt10=float((np.degrees(tl) > 10).mean()), ref_tilt_stance_deg=float(np.degrees(np.concatenate(rl)).mean()), ref_tilt_stance_gt10=float((np.degrees(np.concatenate(rl)) > 10).mean()))
    return out


KEYS = ["slip_stable_mean", "slip_stable_p95", "slip_fixed_mean", "slip_trans_p95", "flicker", "flicker_short", "drift_win_25_45", "drift_win_mean", "drift_cum_end", "agree_main", "D2_pen_p95_mm", "pen_max_mm", "D3_hr_peak_nm",
        "D4_guard_rewrite", "tilt_loaded_deg", "tilt_loaded_gt10", "ref_tilt_stance_deg", "ref_tilt_stance_gt10", "mpjpe", "survived"]
DIRN = {"agree_main": +1}
ABS_MIN = {"slip_stable_mean": 0.01, "slip_stable_p95": 0.02, "slip_fixed_mean": 0.01, "slip_trans_p95": 0.1, "flicker": 0.2, "flicker_short": 0.1, "drift_win_25_45": 0.02, "drift_win_mean": 0.02, "drift_cum_end": 0.05, "agree_main": 0.02,
           "D2_pen_p95_mm": 0.3, "pen_max_mm": 0.5, "D3_hr_peak_nm": 1.0, "D4_guard_rewrite": 0.01, "mpjpe": 2.0}
KEY3 = ["slip_stable_mean", "flicker", "drift_win_25_45"]
GUARD = ["D2_pen_p95_mm", "D3_hr_peak_nm", "D4_guard_rewrite", "slip_trans_p95", "slip_stable_p95", "drift_win_mean"]


def agg(rows):
    nom = [r for r in rows if r["file"].startswith("nominal")]
    m = {k: float(np.nanmean([r[k] for r in nom])) for k in KEYS if k not in ("mpjpe", "survived") and k in nom[0]}
    m["mpjpe"] = float(np.mean([r["onF" if r["_ref"] == "F" else "onO"]["mpjpe_plain_mm"] for r in nom]))
    m["survived"] = float(np.mean([r["survived"] for r in rows])); m["nominal_all_survive"] = bool(all(r["survived"] for r in nom)); m["n_nominal"] = len(nom); m["n_all"] = len(rows)
    return m


def judge(ctrl, var, noise):
    """rule of ABLATION_PROTOCOL §1: need = max(10% |ctrl|, noise band, abs floor)."""
    res = {}
    for k in ABS_MIN:
        if k not in ctrl: continue
        d = DIRN.get(k, -1); need = max(0.10 * abs(ctrl[k]), noise.get(k, 0.0), ABS_MIN[k]); gain = (ctrl[k] - var[k]) * (-d)  # d=-1: lower is better -> gain = ctrl - var
        gain = (ctrl[k] - var[k]) if d < 0 else (var[k] - ctrl[k])
        st = "improved" if gain >= need else "degraded" if -gain >= need else "unchanged"
        if k == "mpjpe": st = "degraded" if (var[k] > 1.25 * ctrl[k] or var[k] > 50) else ("improved" if gain >= need else "unchanged")
        res[k] = dict(ctrl=ctrl[k], var=var[k], gain=gain, need=need, status=st)
    surv_bad = (ctrl["survived"] - var["survived"] >= 0.02) or (not var["nominal_all_survive"] and ctrl["nominal_all_survive"])
    imp = [k for k in KEY3 if res[k]["status"] == "improved"]; deg = [k for k, v in res.items() if v["status"] == "degraded" and k in set(KEY3) | set(GUARD) | {"mpjpe"}] + (["survival"] if surv_bad else [])
    return dict(metrics=res, improved_key=imp, degraded=deg, clear_benefit=bool(len(imp) >= 2 and not deg))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dance"); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--procs", type=int, default=20); ap.add_argument("--it", type=int, default=41000)
    ap.add_argument("--out", default=None); ap.add_argument("--noise-only", action="store_true"); a = ap.parse_args()
    d = Dance(a.dance); out = Path(a.out) if a.out else d.dir / "v17/C_train/eval"; out.mkdir(parents=True, exist_ok=True); log = Logger(out / "v17eval.log")
    PO = FC.padded_npz_of(d.padded_dir); PF = FC.padded_npz_of(d.motions / f"{d.name}_contactfix_padded"); refs = {"O": PO, "F": PF}
    res = read_json(out / "metrics.json", {}) or {}
    with Pool(a.procs) as pool:
        # noise band: V1.6 seeds on ref-O, E1, nominal class
        if "noise_V16" not in res:
            rows = {}
            for k in ("A0", "A1"):
                files = sorted((d.dir / f"v16/B_train/ftc/traj/{k}/O/E1").glob("nominal*.npz")); rr = pool.map(one, [(str(f), str(PO), str(PF), str(PO)) for f in files], chunksize=1)
                for r in rr: r["_ref"] = "O"
                rows[k] = agg(rr)
            res["noise_V16"] = dict(seed0=rows["A0"], seed1=rows["A1"], band={k: abs(rows["A0"][k] - rows["A1"][k]) for k in rows["A0"] if isinstance(rows["A0"][k], float)}); write_json(out / "metrics.json", res); log("noise band done")
        if a.noise_only: return
        for rid0 in a.runs:
            rid = rid0 if rid0.startswith("v17_") and a.it == 41000 else f"{rid0}@{a.it}"  # v17 runs keep their name; other runs (e.g. V1.6 ft_orig_v16 at it41000 = same 1000-it horizon) are tagged
            if rid in res: continue
            run = d.run(rid0); fi = read_json(run.pm / "finetune.json", {}); ck = run.ckpt(a.it)
            ref = "F" if "contactfix" in fi["motion_dir"] else "O"; pz = refs[ref]
            exp = AB.export_eval_model(ck, out / "models" / rid, d.npz, log); log("exported", rid, exp["parity"])
            jobs = []
            for bt in ("E1",):
                jobs += FC.build_jobs(exp, pz, out / "traj" / rid, bt, n_cond=0)  # nominal_ref + 16 nominal perturb
                for c in CONDS4:
                    for i in range(4): jobs.append(dict(kw=dict(onnx=exp["onnx"], params=exp["params"], padded_npz=str(pz), cond=c, init="perturb", seed=FC.BATCHES[bt] + i), out=str(out / "traj" / rid / f"{c}_perturb_{i:03d}.npz")))
            FC.run_rollouts(jobs, out, a.procs, log)
            files = sorted((out / "traj" / rid).glob("*.npz")); rr = pool.map(one, [(str(f), str(pz), str(PF), str(PO)) for f in files], chunksize=1)
            for r in rr: r["_ref"] = ref
            res[rid] = dict(ref=ref, ckpt_sha12=exp["ckpt_sha12"], parity=exp["parity"], agg=agg(rr), rows=rr); write_json(out / "metrics.json", res); log("metrics", rid, {k: round(v, 3) for k, v in res[rid]["agg"].items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
