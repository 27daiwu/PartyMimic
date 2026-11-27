"""contact_refine.api - the public entry points used by the CLI:  analyze / refine / qa for one dance.

Outputs (all NEW files, the original data is never touched):
  results/<d>/motions/<d>_contactfix/<d>_contactfix.npz   +  <d>_contactfix_padded/<d>_contactfix_padded.npz     (prepare-compatible, usable as PM_MOTION_DIR)
  results/<d>/motions/<d>_contactfix_padded/qa_verdict.json (copy next to the data) ; results/<d>/v16/B_contact/{qa_verdict.json, metrics.json, analysis.json, series.npz, frame_diagnostics.csv, figures/}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .. import quality as Q
from ..core import RESULTS, Dance, now, sha256, write_json
from . import geometry as Gm, export, metrics as MT, pipeline as PL, qa as QA
from .config import GATES_VERSION, VERSION, Params, gates_digest

PAD = 50


def out_dir(dance_name: str, root: Path | None = None) -> Path:
    return Path(root or RESULTS) / dance_name / "v16" / "B_contact"


def fix_paths(dance: Dance):
    n = f"{dance.name}_contactfix"
    return dict(npz=dance.motions / n / f"{n}.npz", padded_dir=dance.motions / f"{n}_padded", padded_npz=dance.motions / f"{n}_padded" / f"{n}_padded.npz")


def _seg_summary(ident, g, fps):
    st, state = ident["stance"], ident["state"]; T = len(st)
    return dict(frames=T, stance_frac=[float(st[:, f].mean()) for f in (0, 1)], swing_frac=[float((state[:, f] == 0).mean()) for f in (0, 1)], uncertain_frac=[float((state[:, f] == 2).mean()) for f in (0, 1)],
                double_support=float(st.all(1).mean()), single_support=float((st.sum(1) == 1).mean()), no_support=float((~st.any(1)).mean()), inferred_support_frac=float(ident["inferred"].any(1).mean()),
                mean_confidence=float(ident["conf"].mean()), flight_true_frames=int(ident["flight"]["mask"].sum()), flight_candidates=ident["flight"]["candidates"])


def analyze_only(M, p: Params, log=print):
    """analysis of the ORIGINAL (no change): ground, contact states, flight validation, tiers; returns (A0, summary)"""
    A0 = PL.analyze_arrays(M["root_p"], M["root_q"], M["q"], M["jv"], M["fps"], p, log=log)
    fps = M["fps"]; low = A0["low"]; g = A0["g"]
    summ = dict(frames=len(low), fps=fps, ground=A0["ground"]["info"], ground_curve_mm=dict(min=float(g.min() * 1000), max=float(g.max() * 1000), start=float(g[0] * 1000), end=float(g[-1] * 1000)),
                min_foot_height_over_z0_mm=dict(p05=float(np.percentile(low.min(1), 5) * 1000), median=float(np.median(low.min(1)) * 1000), p95=float(np.percentile(low.min(1), 95) * 1000)),
                contacts=_seg_summary(A0["ident"], g, fps),
                naive_both_feet_gt25mm_over_z0_frac=float((low.min(1) > 0.025).mean()),
                note="naive flight = both feet >25 mm over z=0 (V1.1 style). validated flight needs a ballistic CoM parabola; the difference is the hover caused by ground drift")
    return A0, summ


def candidate_metrics(M, A0, root_p, q, p, src_npz, tag="", log=print):
    """Re-compute states for (root_p, q), analyse against z=0 and compare with the original."""
    z = dict(np.load(src_npz, allow_pickle=True)); st = export.recompute_states(root_p, M["root_q"], q, M["fps"]); st["body_quat_w"] = export.sign_align(st["body_quat_w"], z["body_quat_w"]).astype(np.float32)
    zz = dict(z); zz.update(st)
    N = dict(root_p=np.asarray(root_p), q=np.asarray(q), z=zz, joint_vel=zz["joint_vel"], body_pos_w=zz["body_pos_w"])
    A1 = PL.analyze_arrays(root_p, M["root_q"], q, zz["joint_vel"].astype(float), M["fps"], p, g=np.zeros(len(q)))
    res = MT.compare(M, N, A0, A1, A0["g"], root_p - M["root_p"], None, p, M["fps"])
    return res, A1, N


def refine_dance(dance: Dance, p: Params | None = None, force=False, src: Path | None = None, results_root: Path | None = None, log=print, baselines=True, write_motion=True):
    p = p or Params(); src = Path(src) if src else dance.npz; od = out_dir(dance.name, results_root); od.mkdir(parents=True, exist_ok=True)
    fp = fix_paths(dance)
    if results_root is not None:
        rr = Path(results_root) / dance.name / "motions"; n = f"{dance.name}_contactfix"
        fp = dict(npz=rr / n / f"{n}.npz", padded_dir=rr / f"{n}_padded", padded_npz=rr / f"{n}_padded" / f"{n}_padded.npz")
    if write_motion and fp["npz"].exists() and fp["padded_npz"].exists() and (od / "qa_verdict.json").exists() and not force:
        log("[contact-refine] outputs exist (use --force to recompute):", fp["npz"]); return dict(reused=True, paths=fp, verdict=json.loads((od / "qa_verdict.json").read_text()))
    M = Gm.load_motion(src); t0 = time.time()
    R = PL.refine(M, p, log=log)
    A0 = R["A0"]; fps = M["fps"]
    info = dict(version=VERSION, time=now(), params=p.as_dict(), gates_version=GATES_VERSION, source=str(src), pad=PAD,
                algorithm=("1) FK of the 14 sole sample spheres/foot; 2) ground curve = ALS lower envelope (fc %.1f Hz, static anchor, flight excluded); 3) stance/swing/uncertain from height+speed+hysteresis, "
                           "support inference, flights accepted only if the CoM is ballistic; 4) stage-1 root XYZ sparse LS (pins xy per stance run & sample, z on floor, hinge, smoothness of the correction); "
                           "5) stage-2 bounded leg IK + root re-fit (sparse Gauss-Newton, LM damping, Huber); 6) hard lift to the floor; 7) all body states recomputed with the prepare converter") % p.ground_fc_hz)
    out = export.write_refined(src, od / "_tmp_contactfix.npz", R["root_p"], R["q"], info) if not write_motion else None
    if write_motion:
        out = export.write_refined(src, fp["npz"], R["root_p"], R["q"], info)
        from ..prepare import make_padded
        fp["padded_dir"].mkdir(parents=True, exist_ok=True)
        if fp["padded_npz"].exists(): fp["padded_npz"].unlink()
        make_padded(str(fp["npz"]), str(fp["padded_npz"]))
    else:
        (od / "_tmp_contactfix.npz").unlink(missing_ok=True)
    log("[4/5] metrics ...")
    res, A1, N = candidate_metrics(M, A0, R["root_p"], R["q"], p, src, log=log)
    series = res.pop("_series")
    res["ik_residual"] = dict(original_pin_xy=MT.pin_residual(A0["F"]["sole"], R["s2"]["unit"], A0["F"]["radius"]), refined_pin_xy=MT.pin_residual(A1["F"]["sole"], R["s2"]["unit"], A1["F"]["radius"]),
                              note="scatter of every planted sole sample (stance run x sample, the units the solver pinned) around its run mean: the sliding that remains after the optimisation")
    variants = {}
    if baselines:
        res_ro, _, _ = candidate_metrics(M, A0, R["root_only"]["root_p"], R["root_only"]["q"], p, src, log=log); res_ro.pop("_series"); variants["root_only_stage1"] = res_ro
        try:
            from .. import repair as RP
            tmp = od / "baselines"; tmp.mkdir(exist_ok=True); RP.repair_npz(src, tmp / f"{dance.name}_repaired_v12.npz")
            Z = Gm.load_motion(tmp / f"{dance.name}_repaired_v12.npz"); res_v12, _, _ = candidate_metrics(M, A0, Z["root_p"], Z["q"], p, src, log=log); res_v12.pop("_series"); variants["repair_v12_root_only"] = res_v12
        except Exception as e:
            log("[warn] V1.2 baseline failed:", e)
    verdict = QA.evaluate(res, dict(dance=dance.name, source=str(src), source_sha256=sha256(src), output=str(fp["npz"]) if write_motion else None, output_padded=str(fp["padded_npz"]) if write_motion else None,
                                    output_sha256=sha256(fp["npz"]) if write_motion else None, params=p.as_dict()))
    write_json(od / "metrics.json", dict(main=res, variants=variants, analysis=_seg_summary(A0["ident"], A0["g"], fps), refined_analysis=_seg_summary(A1["ident"], np.zeros(len(A0["g"])), fps), solver=dict(s1=R["s1"]["info"], s2=R["s2"]["info"]), seconds=time.time() - t0))
    QA.write(verdict, od / "qa_verdict.json")
    if write_motion: QA.write(verdict, fp["padded_dir"] / "qa_verdict.json"); QA.write(verdict, fp["npz"].parent / "qa_verdict.json")
    np.savez(od / "series.npz", fps=fps, g=A0["g"], low0=A0["low"], e0=A0["ident"]["feat"]["e"], e1=A1["ident"]["feat"]["e"], state0=A0["ident"]["state"], state1=A1["ident"]["state"], conf0=A0["ident"]["conf"], conf1=A1["ident"]["conf"],
             stance0=A0["ident"]["stance"], stance1=A1["ident"]["stance"], inferred0=A0["ident"]["inferred"], prob0=A0["ident"]["prob"], flight0=A0["ident"]["flight"]["mask"], flight_false0=A0["ident"]["flight"]["false_mask"],
             root0=M["root_p"], root1=R["root_p"], q0=M["q"], q1=R["q"], H0=A0["ident"]["feat"]["H"], com0=A0["F"]["com"], com1=A1["F"]["com"], sole0=A0["F"]["sole"], sole1=A1["F"]["sole"],
             slip0=series["slip0"], o0=series["o0"], slip1=series["slip1"], o1=series["o1"], zmp0=series["zmp0"], zmp1=series["zmp1"], tilt0=A0["ident"]["feat"]["tilt"], tilt1=A1["ident"]["feat"]["tilt"],
             vh0=A0["ident"]["feat"]["vh"], pin=R["s2"]["pin"])
    log(f"[contact-refine] QA verdict: {verdict['verdict']} ({verdict['n_pass']}/{verdict['n_gates']}); failed: {verdict['failed']}")
    return dict(reused=False, paths=fp, verdict=verdict, metrics=res, variants=variants, M=M, R=R, A0=A0, A1=A1, series=series, outdir=od, src=src)


def _series_from_analysis(A0, M):
    I = A0["ident"]; T = len(A0["g"]); z = np.zeros(T)
    return dict(g=A0["g"], low0=A0["low"], e0=I["feat"]["e"], state0=I["state"], conf0=I["conf"], stance0=I["stance"], inferred0=I["inferred"], flight0=I["flight"]["mask"], flight_false0=I["flight"]["false_mask"], root0=M["root_p"], H0=I["feat"]["H"], sole0=A0["F"]["sole"])


def analyze_cmd(dance: Dance, p: Params | None = None, src: Path | None = None, focus: int = 320, results_root: Path | None = None, log=print):
    """--analyze-only: diagnose the ORIGINAL data (ground, contacts, flights, whole clip + a window around `focus`); writes analysis.json / frame_diagnostics_original.csv / figures; changes nothing."""
    p = p or Params(); src = Path(src) if src else dance.npz; od = out_dir(dance.name, results_root); (od / "figures").mkdir(parents=True, exist_ok=True)
    M = Gm.load_motion(src); A0, summ = analyze_only(M, p, log=log); T = len(M["q"]); lo, hi = max(1, focus - 60), min(T - 1, focus + 60)
    from . import viz, diagnostics as D
    S = _series_from_analysis(A0, M)
    # window numbers on the original only (reuse window_stats by mirroring original into the refined slots; refined fields are NOT reported here)
    tilt = D.tilt_series(M["root_q"]); sl = slice(lo, hi); e = S["e0"][sl]; low = S["low0"][sl]
    summ["focus_window"] = dict(dance_frames=[lo, hi], train_frames=[lo + PAD, hi + PAD], tilt_peak_rad=float(tilt[sl].max()), tilt_peak_dance_frame=int(lo + np.argmax(tilt[sl])), root_z_range_m=[float(M["root_p"][sl, 2].min()), float(M["root_p"][sl, 2].max())],
                                ground_curve_mm=[float(A0["g"][sl].min() * 1000), float(A0["g"][sl].max() * 1000)], lowest_foot_over_z0_mm=dict(median=float(np.median(low.min(1)) * 1000), max=float(low.min(1).max() * 1000)),
                                min_foot_height_over_ground_curve_mm=float(e.min() * 1000), states={f"{'LR'[f]}": {n: float((S["state0"][sl, f] == k).mean()) for k, n in ((1, "stance"), (0, "swing"), (2, "uncertain"))} for f in (0, 1)},
                                true_flight_frames=int(S["flight0"][sl].sum()), naive_flight_frames_over_z0_gt25mm=int((low.min(1) > 0.025).sum()))
    write_json(od / "analysis.json", dict(version=VERSION, params=p.as_dict(), source=str(src), source_sha256=sha256(src), summary=summ))
    viz.analysis_overview(A0, M, dance.name, od / "figures" / "analysis_overview.png", focus=(lo, hi))
    # per-frame table (original columns only)
    import csv
    I = A0["ident"]; from .contacts import STATE_NAMES
    with open(od / "frame_diagnostics_original.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dance_frame", "train_frame_with_50_leadin", "tilt_rad", "ground_mm", "root_z_m", "L_h_over_ground_mm", "R_h_over_ground_mm", "L_h_over_z0_mm", "R_h_over_z0_mm", "L_state", "R_state", "L_conf", "R_conf", "inferred_L", "inferred_R", "true_flight", "false_flight"])
        for t in range(T):
            w.writerow([t, t + PAD, f"{tilt[t]:.3f}", f"{A0['g'][t]*1000:.1f}", f"{M['root_p'][t,2]:.4f}", f"{I['feat']['e'][t,0]*1000:.1f}", f"{I['feat']['e'][t,1]*1000:.1f}", f"{A0['low'][t,0]*1000:.1f}", f"{A0['low'][t,1]*1000:.1f}", STATE_NAMES[int(I['state'][t, 0])], STATE_NAMES[int(I['state'][t, 1])],
                        f"{I['conf'][t,0]:.2f}", f"{I['conf'][t,1]:.2f}", int(I["inferred"][t, 0]), int(I["inferred"][t, 1]), int(I["flight"]["mask"][t]), int(I["flight"]["false_mask"][t])])
    log(f"[contact-refine] analysis written: {od}")
    return dict(summary=summ, outdir=od, A0=A0, M=M)


def diagnose(dance: Dance, r: dict, focus: int, results_root: Path | None = None):
    """after a refine: diagnostics.json (ground, regions, label consistency, windows), frame_diagnostics.csv, figures, report skeleton inputs"""
    from . import diagnostics as D, report as RPT
    od = r["outdir"]; M = r["M"]; S = dict(np.load(od / "series.npz")); T = len(M["q"])
    D.frame_table(S, M["root_q"], od / "frame_diagnostics.csv")
    win = lambda a, b: D.window_stats(S, M["root_q"], max(1, a), min(T - 1, b))
    tilt = D.tilt_series(M["root_q"]); lean = (311, 338)
    focus_d = dict(focus=focus, deep_lean=win(*lean), peak=win(focus - 2, focus + 4), pm60=win(focus - 60, focus + 60))
    fp = r["paths"]; ok = fp["npz"].exists()
    lc = D.label_consistency(dance, S, dance.npz if r.get("src") is None else r["src"], fp["npz"] if ok else None)
    diag = dict(analysis=dict(ground=r["A0"]["ground"]["info"]), regions=D.region_stats(S), label_consistency=lc, focus=focus_d)
    write_json(od / "diagnostics.json", diag)
    fd = RPT.build_figures(od, S, M, r["metrics"], r["variants"], focus)
    return diag, fd


def qa_existing(dance: Dance, p: Params | None = None, src: Path | None = None, fixed: Path | None = None, results_root: Path | None = None, log=print):
    """`contact-qa`: independently re-judge an EXISTING <dance>_contactfix.npz against the original (nothing is recomputed from the refiner's own metrics) and rewrite qa_verdict.json."""
    p = p or Params(); src = Path(src) if src else dance.npz; fp = fix_paths(dance); fixed = Path(fixed) if fixed else fp["npz"]
    if not fixed.exists(): raise FileNotFoundError(f"{fixed} not found: run `./pm contact-refine {dance.name}` first")
    M = Gm.load_motion(src); N = Gm.load_motion(fixed); A0 = PL.analyze_arrays(M["root_p"], M["root_q"], M["q"], M["jv"], M["fps"], p, log=log)
    res, A1, _ = candidate_metrics(M, A0, N["root_p"], N["q"], p, src)
    res.pop("_series"); zf = N["z"]
    # the file itself must be internally consistent: recompute its velocities/body states from its own root+joints and compare
    st = export.recompute_states(N["root_p"], N["root_q"], N["q"], N["fps"]); cons = {k: float(np.abs(st[k].astype(float) - zf[k].astype(float)).max()) for k in ("joint_vel", "body_pos_w", "body_lin_vel_w", "body_ang_vel_w")}
    ci = zf.get("contactfix_info"); info = json.loads(str(ci)) if ci is not None else {}
    res["integrity"]["self_consistency_max_abs"] = cons; res["integrity"]["source_sha256_matches"] = bool(info.get("source_sha256") == sha256(src))
    if max(cons.values()) > 1e-3: res["integrity"]["ok"] = False
    v = QA.evaluate(res, dict(dance=dance.name, source=str(src), source_sha256=sha256(src), output=str(fixed), output_sha256=sha256(fixed), re_judged_independently=True, file_metadata=info.get("version")))
    od = out_dir(dance.name, results_root); QA.write(v, od / "qa_verdict.json")
    if fixed.parent.name.endswith("_contactfix"): QA.write(v, fixed.parent / "qa_verdict.json")
    pd_ = fixed.parent.parent / (fixed.parent.name + "_padded")
    if pd_.exists(): QA.write(v, pd_ / "qa_verdict.json")
    return v, res


def make_videos(dance: Dance, od: Path, M, focus: int = 320, out: Path | None = None, full=True, log=print):
    """ORIGINAL vs REFINED kinematic replay: whole clip (real time, every 2nd frame at 25 fps) + zoom around `focus` (all frames at 10 fps = 5x slow motion)."""
    from . import video, diagnostics as D
    S = dict(np.load(od / "series.npz")); out = Path(out) if out else od / "videos"; out.mkdir(parents=True, exist_ok=True); T = len(M["q"]); tilt = D.tilt_series(M["root_q"]); res = {}
    lo, hi = max(1, focus - 60), min(T - 1, focus + 60)
    note = f"   | {dance.name}: original vs contact_refine"
    p1 = out / f"{dance.name}_contactfix_zoom_dance{focus}_train{focus + PAD}_slow5x.mp4"
    video.render(S, M["root_q"], S["q0"], S["q1"], p1, lo, hi, 1, 10, f"zoom: dance frames {lo}-{hi} (train/real frames {lo + PAD}-{hi + PAD}), 5x slow motion", tilt=tilt, label_note=note); res["zoom"] = str(p1); log("video", p1)
    if full:
        p0 = out / f"{dance.name}_contactfix_full.mp4"
        video.render(S, M["root_q"], S["q0"], S["q1"], p0, 1, T - 1, 2, 25, "full clip, real time", tilt=tilt, label_note=note); res["full"] = str(p0); log("video", p0)
    return res
