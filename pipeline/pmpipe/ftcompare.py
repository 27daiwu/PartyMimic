"""PartyMimic V1.6 (B-line) - fine-tune comparison `./pm ftcompare`: control (original reference) vs treatment (contact-corrected reference), both fine-tuned from the same it40000 state.

Protocol (pre-registered BEFORE any comparison data existed): results/<dance>/v16/B_train/FT_PROTOCOL.md  - the thresholds below (DECISION) are the same numbers.
  * 2 seeds per group (run-to-run noise) x 2 evaluation batches (E1/E2 = different perturbed-start seeds) x 2 references (ref-O original, ref-F corrected) = 2x2 deployment matrix.
  * DECISION = matched deployment (A on ref-O, B on ref-F) with reference-independent THIRD-PARTY metrics; anchored/global tracking errors are never compared across references
    (they only appear as a sanity ceiling G3 and as side-by-side information).
Everything is cached under <out>/ftc/ so an interrupted comparison resumes.
"""
from __future__ import annotations

import csv
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import abtest as AB
from . import quality as Q
from . import repair as P
from .core import PY, REPO, S2R_PY, Dance, Logger, Run, die, load_cfg, now, py_env, read_json, say, sh, sha256, write_json

PAD = Q.PAD
BATCHES = {"E1": 0, "E2": 1000}   # perturbed-start seed offsets
COND_SET = ["push_f", "push_b", "push_l", "delay10", "kp0.85", "fric0.6", "noise"]
# ---------------------------------------------------------------------------------------------------------------------------------- pre-registered rule (see FT_PROTOCOL.md §3-4)
DIR = {  # -1: lower is better, +1: higher is better
    "D1_slip_mps": -1, "D2_pen_p95_mm": -1, "D3_hr_peak_nm": -1, "D4_guard_rewrite": -1, "D5_tilt_dev_rad": -1, "D6_root_feet_drift_m": -1, "D7_root_vs_fixed_feet_m": -1,
    "sat_all": -1, "flicker_per_s": -1, "tilt_p95": -1, "cop_p05_mm": +1, "peak_ratio_p95": -1}
DECISION_KEYS = ["D1_slip_mps", "D2_pen_p95_mm", "D3_hr_peak_nm", "D4_guard_rewrite", "D5_tilt_dev_rad", "D6_root_feet_drift_m", "D7_root_vs_fixed_feet_m"]
TARGETED = ["D1_slip_mps", "D2_pen_p95_mm", "D6_root_feet_drift_m", "D7_root_vs_fixed_feet_m"]
DECISION = dict(
    rel_min=0.10,
    abs_min={"D1_slip_mps": 0.01, "D2_pen_p95_mm": 0.3, "D3_hr_peak_nm": 1.0, "D4_guard_rewrite": 0.01, "D5_tilt_dev_rad": 0.01, "D6_root_feet_drift_m": 0.01, "D7_root_vs_fixed_feet_m": 0.02,
             "sat_all": 5e-4, "flicker_per_s": 0.2, "tilt_p95": 0.01, "cop_p05_mm": 1.0, "peak_ratio_p95": 0.02},
    need_improved=3, need_targeted=1, surv_margin=0.02, robust_min=0.90, mpjpe_ratio_max=1.25, mpjpe_abs_max_mm=50.0)
LABEL = {"D1_slip_mps": "D1 足滑(接触点) m/s", "D2_pen_p95_mm": "D2 穿透 p95 mm", "D3_hr_peak_nm": "D3 髋Roll峰值 N·m", "D4_guard_rewrite": "D4 护栏改写率", "D5_tilt_dev_rad": "D5 倾角偏差(对匹配参考) rad",
         "D6_root_feet_drift_m": "D6 根-落脚一致性(仿真内部) m", "D7_root_vs_fixed_feet_m": "D7 根相对修正落脚隐含根 m", "sat_all": "全关节饱和占比", "flicker_per_s": "承重翻转 次/s/脚", "tilt_p95": "倾角 p95 rad",
         "cop_p05_mm": "CoP 余量 p05 mm(↑)", "peak_ratio_p95": "力矩峰值比 p95"}


LABEL_EN = {"D1_slip_mps": "D1 contact-point slip m/s", "D2_pen_p95_mm": "D2 penetration p95 mm", "D3_hr_peak_nm": "D3 hip-roll peak Nm", "D4_guard_rewrite": "D4 guard rewrite rate", "D5_tilt_dev_rad": "D5 tilt dev vs ref rad",
            "D6_root_feet_drift_m": "D6 root-vs-planted-feet drift m", "D7_root_vs_fixed_feet_m": "D7 root vs corrected-feet root m", "sat_all": "torque saturation frac", "flicker_per_s": "load flicker /s/foot", "tilt_p95": "tilt p95 rad",
            "cop_p05_mm": "CoP margin p05 mm (higher better)", "peak_ratio_p95": "torque peak ratio p95"}


# ---------------------------------------------------------------------------------------------------------------------------------- freeze (historical results must not change)
def freeze_set(dance: Dance) -> list[Path]:
    b = dance.dir / "runs/robust_d_v1"
    fs = [b / "train/last.pt", b / "train/ckpt_040000.pt", b / "train/args.json", b / "train/train_log.csv", b / "pm/admission.json", b / "final/final_selection.json", dance.npz, dance.csv, dance.padded_npz, b / "init.pt"]
    fs += sorted((b / "final/onnx").glob("*")) + sorted((b / "final/assets").glob("*/*")) + sorted((b / "deploy").glob("*"))
    return [f for f in fs if f.is_file()]


def freeze(dance: Dance, out: Path, force=False) -> dict:
    if out.exists() and not force:
        return read_json(out)
    doc = dict(time=now(), files={str(f): AB._h(f) for f in freeze_set(dance)})
    write_json(out, doc)
    return doc


POINTERS = ("latest.json",)  # mutable "latest package" pointers: legitimately rewritten when ANY colleague builds a new package of that run (e.g. cpp_g1_v3 at 13:42); the zips/ONNX they point to stay frozen


def verify_freeze(fz: Path) -> dict:
    old = read_json(fz)["files"]
    bad, ign = [], []
    for f, h in old.items():
        n = AB._h(f).get("sha12")
        if n != h.get("sha12"):
            (ign if Path(f).name in POINTERS else bad).append(dict(file=f, before=h.get("sha12"), after=n))
    return dict(checked=len(old), changed=bad, ignored_mutable_pointers=ign, ok=not bad)


# ---------------------------------------------------------------------------------------------------------------------------------- models / references
def padded_npz_of(d: Path | str) -> Path:
    d = Path(d); z = sorted(d.glob("*.npz"))
    if len(z) != 1:
        die(f"{d}: expected exactly one *.npz, found {[p.name for p in z]}")
    return z[0]


def collect_models(dance: Dance, a: str, b: str, a_extra, b_extra, it: int | None) -> dict:
    """key -> dict(group, run, ckpt, seed, motion_dir).  A* = control, B* = treatment, anchor = frozen it40000."""
    ms = {}
    for grp, ids in (("A", [a] + list(a_extra)), ("B", [b] + list(b_extra))):
        for i, rid in enumerate(ids):
            r = dance.run(rid); fi = read_json(r.pm / "finetune.json", {}); itn = it or fi.get("target_iterations") or r.iteration()
            ck = r.ckpt(itn)
            if not ck.exists():
                die(f"{rid}: {ck} missing (training not finished?)")
            ms[f"{grp}{i}"] = dict(group=grp, run=rid, ckpt=str(ck), seed=fi.get("seed"), motion_dir=fi.get("motion_dir"), iteration=itn)
    base = dance.run("robust_d_v1")
    ms["anchor"] = dict(group="anchor", run="robust_d_v1", ckpt=str(base.ckpt(40000)), seed=0, motion_dir=str(dance.padded_dir), iteration=40000)
    return ms


# ---------------------------------------------------------------------------------------------------------------------------------- rollouts
def build_jobs(exp: dict, padded: Path, out: Path, batch: str, n_nominal=16, n_cond=8, nominal_ref=True) -> list[dict]:
    s0 = BATCHES[batch]; jobs = []
    base = dict(onnx=exp["onnx"], params=exp["params"], padded_npz=str(padded))
    if nominal_ref:
        jobs.append(dict(kw=dict(base, cond="nominal", init="ref", seed=0), out=str(out / "nominal_ref.npz")))
    for i in range(n_nominal):
        jobs.append(dict(kw=dict(base, cond="nominal", init="perturb", seed=s0 + i), out=str(out / f"nominal_perturb_{i:03d}.npz")))
    for c in COND_SET:
        for i in range(n_cond):
            jobs.append(dict(kw=dict(base, cond=c, init="perturb", seed=s0 + i), out=str(out / f"{c}_perturb_{i:03d}.npz")))
    return jobs


def run_rollouts(jobs: list[dict], workdir: Path, procs: int, log) -> None:
    pend = [j for j in jobs if not Path(j["out"]).exists()]
    if not pend:
        return
    jf = workdir / f"jobs_{int(time.time())}.json"; write_json(jf, pend)
    log(f"rollouts: {len(pend)} to run on {procs} procs")
    r = sh([S2R_PY, REPO / "pipeline/pmpipe/rollout_batch.py", "--jobs", jf, "--procs", procs], log=workdir / "rollout_batch.log")
    if r.returncode:
        die(f"rollout batch failed rc={r.returncode} (see {workdir / 'rollout_batch.log'})", 3)


# ---------------------------------------------------------------------------------------------------------------------------------- metrics (one rollout)
_RI: dict = {}


def refinfo(p: str):
    if p not in _RI:
        R = Q.load_ref(Path(p)); K = Q.fk_series(R["root_p"], R["root_q"], R["q"]); cal = Q.calibrate_floor(K["clr"], R["jv"], R["root_p"], R["fps"]); _, h = Q.contact_tiers(K["clr"], cal["offset"])
        _RI[p] = dict(R=R, K=K, refK=dict(ank=K["ank"], h=h, cal=cal), lab=P.ref_contact_labels(Path(p)))
    return _RI[p]


def _tilt_of_quat(q):
    return np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1))


def _gvec(q):
    """gravity direction in the body frame = third row of R(q), q = (w,x,y,z)"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)


SEG = (310, 430)  # padded-clip frames of the deep forward lean (dance frames ~260-380); reported separately (B1 request)


def _seg_metrics(ro, R, T):
    gs, gr = _gvec(ro["root_q"]), _gvec(R["root_q"][:T])
    gerr = np.arccos(np.clip((gs * gr).sum(-1), -1, 1)); ts, tr = _tilt_of_quat(ro["root_q"]), _tilt_of_quat(R["root_q"][:T])
    a, b = SEG, slice(SEG[0], min(SEG[1], T))
    return dict(gerr_max_rad=float(gerr[PAD:T - PAD].max()), gerr_mean_rad=float(gerr[PAD:T - PAD].mean()), tilt_max_rad=float(ts[PAD:T - PAD].max()), ref_tilt_max_rad=float(tr[PAD:T - PAD].max()),
                seg_gerr_max_rad=float(gerr[b].max()), seg_gerr_mean_rad=float(gerr[b].mean()), seg_tilt_max_rad=float(ts[b].max()), seg_ref_tilt_max_rad=float(tr[b].max()),
                seg_tilt_margin_rad=float(tr[b].max() - ts[b].max()))


def _one(a):
    f, Pm, PF, PO = a
    ro = Q.load_rollout(f); T = len(ro["t"]); s = slice(PAD, T - PAD); BW = ro["meta"]["mass_kg"] * Q.G
    im, iF, iO = refinfo(Pm), refinfo(PF), refinfo(PO)
    d = Q.contact_dynamics(ro)
    dc = Q.trajectory_decomposition(im["R"], im["refK"], ro); dc.pop("_series")
    sp = ro["root_p"][:, :2]
    drF = Q.dead_reckon(iF["R"]["root_p"][:T, :2], iF["refK"]["ank"][:T, :, :2], iF["refK"]["h"][:T] < 0.025)
    d7 = np.linalg.norm((sp - sp[0]) - (drF - drF[0]), axis=1)[s].mean()
    lb = ro["foot_fz"] > 0.10 * BW
    flick = float(np.abs(np.diff(lb[s].astype(int), axis=0)).sum() / 2 / ((T - 2 * PAD) / 50.0))
    tilt_ref = _tilt_of_quat(im["R"]["root_q"][:T])
    out = dict(file=Path(f).name, survived=ro["meta"]["fell_t"] is None, fell_t=ro["meta"]["fell_t"],
               D1_slip_mps=d["slip"]["contact_point"]["mean"], D2_pen_p95_mm=d["penetration_mm"]["dance_p95"], pen_max_mm=d["penetration_mm"]["dance_max"], D3_hr_peak_nm=d["torque"]["hip_roll_peak_nm"],
               D4_guard_rewrite=d["guard"]["any_clip_rate"], D5_tilt_dev_rad=float(np.abs(ro["tilt"][s] - tilt_ref[s]).mean()), D6_root_feet_drift_m=dc["offset_components_m"]["policy_slip_drift"], D7_root_vs_fixed_feet_m=float(d7),
               sat_all=d["torque"]["sat_frac_mean"], flicker_per_s=flick, tilt_p95=d["balance"]["tilt_p95_rad"], cop_p05_mm=(d["balance"]["cop_margin_mm"] or {}).get("p05", float("nan")), peak_ratio_p95=d["torque"]["peak_ratio_p95"],
               ds=d["contact"]["double_support"], ss=d["contact"]["single_support"], ref_root_inconsistency_m=dc["offset_components_m"]["reference_root_inconsistency"],
               lab_agree_F=float((iF["lab"][:T][s] == lb[s]).mean()), lab_agree_O=float((iO["lab"][:T][s] == lb[s]).mean()))
    for tag, pp in (("F", PF), ("O", PO)):
        m = AB._metrics((f, pp))
        out[f"seg{tag}"] = _seg_metrics(ro, {"F": iF, "O": iO}[tag]["R"], T)
        out[f"on{tag}"] = {k: m[k] for k in ("global_mpjpe_mm", "mpjpe_plain_mm", "mpjpe_zaligned_mm", "jmae_rad", "root_xy_err_mean_m", "root_xy_err_max_m", "root_xy_err_end_m", "yaw_err_deg")}
    return out


def metrics_for(files: list[Path], Pm: Path, PF: Path, PO: Path, pool) -> list[dict]:
    return pool.map(_one, [(str(f), str(Pm), str(PF), str(PO)) for f in files], chunksize=1)


# ---------------------------------------------------------------------------------------------------------------------------------- aggregation + decision
def _mean(rows, key, sub=None):
    v = [(r[sub][key] if sub else r[key]) for r in rows]; v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def decide(runs: dict, se_eval: dict) -> dict:
    """runs[key][metric] = run-level value (nominal-class rollouts, E1+E2); keys A0.. B0..  Returns per-metric status per the pre-registered rule."""
    Ak = [k for k in runs if k.startswith("A")]; Bk = [k for k in runs if k.startswith("B")]
    res = {}
    for m, dr in DIR.items():
        a = [runs[k][m] for k in Ak]; b = [runs[k][m] for k in Bk]; ma, mb = float(np.mean(a)), float(np.mean(b))
        gain = (ma - mb) if dr < 0 else (mb - ma)
        spread = max([abs(x - y) for g in (a, b) for x in g for y in g] or [0.0])
        need = max(DECISION["rel_min"] * abs(ma), DECISION["abs_min"][m], spread, se_eval.get(m, 0.0))
        better = (lambda x, y: x < y) if dr < 0 else (lambda x, y: x > y)
        dom_b = all(better(y, x) for x in a for y in b); dom_a = all(better(x, y) for x in a for y in b)
        status = "improved" if (gain >= need and dom_b) else "degraded" if (-gain >= need and dom_a) else "unchanged"
        res[m] = dict(A=a, B=b, mean_A=ma, mean_B=mb, gain=gain, need=need, spread_run=spread, se_eval=se_eval.get(m, 0.0), strict_dominance_B=dom_b, strict_dominance_A=dom_a, status=status, decision=m in DECISION_KEYS)
    return res


def verdict(rows: dict, gates: dict) -> dict:
    imp = [m for m in DECISION_KEYS if rows[m]["status"] == "improved"]; deg = [m for m, v in rows.items() if v["status"] == "degraded"]
    tgt = [m for m in imp if m in TARGETED]
    g = dict(gates)
    g["G2_no_degradation"] = dict(passed=not deg, degraded=deg)
    g["G4_benefit"] = dict(passed=len(imp) >= DECISION["need_improved"] and len(tgt) >= DECISION["need_targeted"], improved=imp, targeted_improved=tgt, need=f">={DECISION['need_improved']} improved, >={DECISION['need_targeted']} targeted")
    harm = [k for k in ("G1_safety", "G2_no_degradation", "G3_fidelity_ceiling") if not g[k]["passed"]]
    ok0 = g["G0_integrity"]["passed"]
    go = ok0 and not harm and g["G4_benefit"]["passed"]
    kind = "GO" if go else ("NO-GO (G0 完整性未通过)" if not ok0 else "NO-GO (有害: " + ",".join(harm) + ")" if harm else "NO-GO (未证明收益)")
    return dict(go=bool(go), verdict=kind, gates=g, improved=imp, degraded=deg, rule=DECISION)


# ---------------------------------------------------------------------------------------------------------------------------------- training curves
def train_summary(run: Run) -> dict:
    rows = list(csv.DictReader(open(run.train / "train_log.csv")))
    if not rows:
        return {}
    f = lambda r, k: float(r[k]); last = [r for r in rows if f(r, "iteration") > f(rows[-1], "iteration") - 1000]; first = rows[:4]
    sup = (run.logs / "supervisor.log").read_text() if (run.logs / "supervisor.log").exists() else ""
    mean = lambda rs, k: float(np.mean([f(r, k) for r in rs])); std = lambda rs, k: float(np.std([f(r, k) for r in rs]))
    return dict(iterations=int(f(rows[-1], "iteration")), rows=len(rows), first100=dict(sim_rew=mean(first, "sim_rew"), ep_ret=mean(first, "ep_ret_norm"), term=mean(first, "term_frac"), wm_err8=mean(first, "wm_err8")),
                last1000={k: dict(mean=mean(last, k), std=std(last, k)) for k in ("sim_rew", "ep_ret_norm", "term_frac", "wm_err8", "v_loss", "it_per_s")},
                nonfinite=int(sum(f(r, "nonfinite") for r in rows)), skipped_wm=int(sum(f(r, "wm_skip") for r in rows if "wm_skip" in r)), skipped_v=int(sum(f(r, "v_skip") for r in rows if "v_skip" in r)),
                rollbacks=sup.count("ROLLBACK"), restarts=sup.count("training launched") - 1, wall_h=f(rows[-1], "wall_h") - f(rows[0], "wall_h"))


def eval_curve(run: Run) -> list[dict]:
    out = []
    for p in sorted(run.eval.glob("ckpt_*.json")):
        r = read_json(p); s = r["scores"]; out.append(dict(iteration=int(p.stem.split("_")[1]), balance=s["balance"], fidelity=s["fidelity"], survival_robust=s["survival_robust"], mpjpe_nominal_mm=s["mpjpe_nominal_mm"], slip_mean=s["slip_mean"], pen_max_mm=s["pen_max_mm"]))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- main
def GRIDS(ecfg):
    return {"g1234": dict(reps=ecfg["periodic"]["reps"], seed=1234, hard=False), "g4321": dict(reps=ecfg["periodic"]["reps"], seed=4321, hard=False), "hard777": dict(reps=ecfg["hard"]["reps"], seed=ecfg["hard"]["seed"], hard=True)}


def compare(dance: Dance, a: str, b: str, a_seeds=(), b_seeds=(), it=None, ref_a=None, ref_b=None, out_dir=None, n_perturb=8, procs=16, video=False, log=None, grids=("g1234", "g4321", "hard777"), tag: str = "") -> dict:
    from . import evaluator as E
    out = Path(out_dir) if out_dir else dance.dir / "v16/B_train"; W = out / "ftc"; W.mkdir(parents=True, exist_ok=True)
    log = log or Logger(W / "ftcompare.log")
    ms = collect_models(dance, a, b, a_seeds, b_seeds, it)
    RO_dir = Path(ref_a) if ref_a else dance.padded_dir
    RF_dir = Path(ref_b) if ref_b else Path(ms["B0"]["motion_dir"])
    PO, PF = padded_npz_of(RO_dir), padded_npz_of(RF_dir)
    refs = {"O": PO, "F": PF}
    log("models", {k: (v["run"], Path(v["ckpt"]).name) for k, v in ms.items()}, refs={k: str(v) for k, v in refs.items()}, same_reference=(PO == PF))
    fz = out / "FREEZE.json"; freeze(dance, fz)
    # ---- export (ONNX + parity + assets; motion.bin source = original unpadded dance: params do not depend on the reference)
    exp = {k: AB.export_eval_model(Path(m["ckpt"]), W / "models" / k, dance.npz, log) for k, m in ms.items()}
    log("exported", {k: (v["ckpt_sha12"], v["parity"]) for k, v in exp.items()})
    # ---- rollouts: model x ref x batch
    jobs, dirs = [], {}
    for k in ms:
        for rk, pz in refs.items():
            for bt in BATCHES:
                d = W / "traj" / k / rk / bt; dirs[(k, rk, bt)] = d
                jobs += build_jobs(exp[k], pz, d, bt, nominal_ref=(bt == "E1"), n_cond=n_perturb)
    run_rollouts(jobs, W, procs, log); log("rollouts done")
    # ---- metrics
    mf = W / "metrics_all.json"
    M = read_json(mf, {}) or {}   # incremental: cells already computed (e.g. by the 2-seed comparison) are reused, only new models are evaluated
    todo = [(k, rk, bt, d) for (k, rk, bt), d in dirs.items() if f"{k}|{rk}|{bt}" not in M]
    if todo:
        with Pool(min(procs, 24)) as pool:
            for k, rk, bt, d in todo:
                files = sorted(d.glob("*.npz"))
                rows = metrics_for(files, refs[rk], PF, PO, pool)
                M[f"{k}|{rk}|{bt}"] = {Path(f).stem: r for f, r in zip(files, rows)}
        write_json(mf, M)
    log("metrics done")
    # ---- matched-deployment run-level values (nominal-class rollouts, E1+E2) and evaluation-batch noise
    def cell(k, rk, bt=None, cond="nominal"):
        rs = []
        for b2 in ([bt] if bt else list(BATCHES)):
            rs += [r for n, r in M[f"{k}|{rk}|{b2}"].items() if n.startswith(cond)]
        return rs
    match = lambda k: "O" if (ms[k]["group"] in ("A", "anchor")) else "F"
    runs_v = {k: {m: _mean(cell(k, match(k)), m) for m in DIR} for k in ms}
    se_eval = {m: max(abs(_mean(cell(k, match(k), "E1"), m) - _mean(cell(k, match(k), "E2"), m)) for k in ms if k != "anchor") for m in DIR}
    rows = decide({k: v for k, v in runs_v.items() if k != "anchor"}, se_eval)
    # ---- survival (deployment chain, all conditions)
    surv = {k: float(np.mean([r["survived"] for c in ["nominal"] + COND_SET for r in cell(k, match(k), None, c)])) for k in ms}
    nom_all = {k: all(r["survived"] for r in cell(k, match(k))) for k in ms}
    # ---- evaluator grids (training-env): both seeds, hard (reps 24, seed 777), on both references
    ecfg = load_cfg("eval", "balance_first_v1"); grid = {}
    for k, m in ms.items():
        for rk, pz in refs.items():
            for tag, kw in GRIDS(ecfg).items():
                if tag not in grids:
                    continue
                f = W / "evalgrid" / f"{k}_{rk}_{tag}.json"
                if not f.exists():
                    f.parent.mkdir(exist_ok=True); write_json(f, E.evaluate(m["ckpt"], str(pz.parent), **kw)); log("grid", k, rk, tag)
                grid[f"{k}|{rk}|{tag}"] = read_json(f)
    # ---- gates
    own = lambda k: [r for r in cell(k, match(k))]
    fid = {k: dict(anch=_mean(own(k), "mpjpe_plain_mm", "onF" if match(k) == "F" else "onO"), glob=_mean(own(k), "global_mpjpe_mm", "onF" if match(k) == "F" else "onO")) for k in ms}
    Ak = [k for k in ms if k.startswith("A")]; Bk = [k for k in ms if k.startswith("B")]
    fid_ok = {k: (fid[k]["anch"] <= DECISION["mpjpe_ratio_max"] * np.mean([fid[x]["anch"] for x in Ak]) and fid[k]["anch"] <= DECISION["mpjpe_abs_max_mm"]
                  and fid[k]["glob"] <= DECISION["mpjpe_ratio_max"] * np.mean([fid[x]["glob"] for x in Ak])) for k in Bk}
    gsc = lambda k, rk, tag: grid[f"{k}|{rk}|{tag}"]
    grid_ok = {}
    for k in Bk:
        for tag in grids:
            g = gsc(k, "F", tag); amin = min(gsc(x, "O", tag)["scores"]["survival_robust"] for x in Ak)
            grid_ok[f"{k}|{tag}"] = dict(nominal=g["conditions"]["nominal"]["survival"], robust=g["scores"]["survival_robust"], A_min=amin,
                                         passed=bool(g["conditions"]["nominal"]["survival"] >= 1.0 and g["scores"]["survival_robust"] >= DECISION["robust_min"] and g["scores"]["survival_robust"] >= amin - DECISION["surv_margin"]))
    fzv = verify_freeze(fz); tsum = {k: train_summary(dance.run(m["run"])) for k, m in ms.items() if k != "anchor"}
    qa = read_json(Path(RF_dir) / "qa_verdict.json", None) if PO != PF else None
    parity_ok = all(v["parity"] for v in exp.values())
    g0 = dict(passed=bool(fzv["ok"] and parity_ok and all(t["rollbacks"] == 0 and t["nonfinite"] == 0 for t in tsum.values()) and (PO == PF or (qa or {}).get("verdict", (qa or {}).get("qa_verdict")) in ("PASS", "pass"))),
              freeze=fzv, onnx_parity=parity_ok, rollbacks={k: t["rollbacks"] for k, t in tsum.items()}, nonfinite={k: t["nonfinite"] for k, t in tsum.items()}, qa_verdict=qa, reference_same=(PO == PF))
    g1 = dict(passed=bool(all(nom_all[k] for k in Bk) and min(surv[k] for k in Bk) >= min(surv[k] for k in Ak) - DECISION["surv_margin"] and all(v["passed"] for v in grid_ok.values())),
              deploy_chain_survival=surv, nominal_all_survive=nom_all, grid=grid_ok)
    g3 = dict(passed=bool(all(fid_ok.values())), fidelity=fid, per_B=fid_ok)
    vd = verdict(rows, dict(G0_integrity=g0, G1_safety=g1, G3_fidelity_ceiling=g3))
    doc = dict(generated=now(), models={k: dict(m, sha12=exp[k]["ckpt_sha12"], onnx_sha12=exp[k]["onnx_sha12"]) for k, m in ms.items()}, refs={k: str(v) for k, v in refs.items()},
               runs=runs_v, rows=rows, verdict=vd, survival=surv, se_eval=se_eval, train=tsum, fid=fid)
    write_json(W / f"ft_result{tag}.json", doc); write_json(out / f"ft_result{tag}.json", doc)
    from .ftreport import write_report
    rep = write_report(dance, doc, M, grid, ms, out, cell, match, tag=tag)
    log("report", str(rep), verdict=vd["verdict"])
    return dict(report=str(rep), verdict=vd["verdict"], go=vd["go"])
