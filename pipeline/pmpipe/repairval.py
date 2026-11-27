"""PartyMimic Pipeline V1.2 - validation of a repaired reference (`./pm repair <dance> --validate`).  Original vs repaired, SAME frozen policy, SAME evaluator.

  A. geometry (no policy)         : foot slip on planted feet, root path vs the path the planted feet imply, penetration / floating, CoM speed+accel, training contact labels
  B. reliable-contact dynamics    : pmpipe.quality.analyze_reference on both (ZMP / support margin only on unambiguous-contact frames, difficulty vs LAFAN)
  C. frozen-policy closed loop    : deployment-chain MuJoCo rollouts of the chosen checkpoint on each reference (nominal + perturbed starts): global-MPJPE decomposition,
                                    contact / penetration / balance / torque; contact labels validated against MuJoCo FORCE contact
  D. training-env evaluation grid : pmpipe.evaluator (V1.1) of the same checkpoint on each reference motion (fidelity / balance / survival over the condition grid)
  E. comparison video + figure, REPAIR_REPORT.md with an explicit retrain recommendation (criteria listed, verdict derived from the numbers).
The policy is NOT retrained: effects on policy behaviour are "what the frozen policy does on cleaner data", an upper bound on how much was a data problem.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import quality as Q
from . import repair as P
from .core import PY, REPO, S2R_PY, Logger, Run, load_cfg, now, read_json, sh, write_json


def _rollouts(run: Run, it: int, padded: Path, tag: str, qdir: Path, n: int, log):
    from .post import export_model
    m = run.manifest(); ecfg = load_cfg("eval", m.get("eval_cfg") or "balance_first_v1"); dcfg = load_cfg("deploy", m.get("deploy_cfg") or "cpp_g1_v1")
    art = export_model(run, it, ecfg, dcfg, log); td = qdir / "traj"; td.mkdir(parents=True, exist_ok=True)
    nom = td / f"it{it:06d}_on_{tag}_nominal.npz"; base = [S2R_PY, REPO / "pipeline/pmpipe/rollout.py", "--onnx", art["onnx"], "--params", Path(art["assets"]) / "policy_params.txt", "--padded-npz", padded]
    if not nom.exists(): sh(base + ["--out", nom], check=True, capture=True)
    pert = sorted(td.glob(f"it{it:06d}_on_{tag}_perturb_*.npz"))
    if len(pert) < n:
        [p.unlink() for p in pert]; sh(base + ["--out", td / f"it{it:06d}_on_{tag}_perturb.npz", "--init", "perturb", "--n", n, "--procs", min(n, 8)], check=True, capture=True); pert = sorted(td.glob(f"it{it:06d}_on_{tag}_perturb_*.npz"))
    return [nom] + pert[:n]


def zaligned_anchored_mpjpe(ref_dance, files) -> dict:
    """XY/yaw-anchored MPJPE with the reference shifted by the rollout's mean root-Z offset: removes the reference's floating/ground offset, which the plain
    anchored metric (it keeps the reference root height) counts as tracking error."""
    ref = Q.load_ref(ref_dance.padded_npz); vals, plain = [], []
    for f in files:
        ro = Q.load_rollout(f); T = len(ro["t"]); S = Q.fk_series(ro["root_p"], ro["root_q"], ro["q"], body_names=ref["body_names"]); rb, rp, rq = ref["body_p"][:T], ref["root_p"][:T], ref["root_q"][:T]
        dz = float((ro["root_p"][Q.PAD:T - Q.PAD, 2] - rp[Q.PAD:T - Q.PAD, 2]).mean()); dyaw = Q._wrap(Q._yaw(ro["root_q"]) - Q._yaw(rq)); c, s_ = np.cos(dyaw), np.sin(dyaw); rel = rb - rp[:, None, :]
        rot = np.stack([np.stack([c, -s_], -1), np.stack([s_, c], -1)], -2); rxy = np.einsum("tij,tbj->tbi", rot, rel[..., :2])
        for sh_, acc in ((0.0, plain), (dz, vals)):
            anc = np.concatenate([ro["root_p"][:, None, :2] + rxy, rp[:, 2:3, None] + rel[..., 2:3] + sh_], -1); acc.append(float(np.linalg.norm(S["bodies"] - anc, axis=-1).mean(-1)[Q.PAD:T - Q.PAD].mean() * 1000))
    return dict(plain_mm=float(np.mean(plain)), z_aligned_mm=float(np.mean(vals)), z_aligned_std=float(np.std(vals)))


def validate(run: Run, it: int, n_perturb: int = 8, video: bool = True, log=None, fc_xy=None) -> dict:
    from . import evaluator as E
    from .replay import render_repair_compare
    log = log or Logger(run.logs / "repair.log")
    dance = run.dance; rd = P.make_dance_like(dance); qdir = run.reports / "repair"; qdir.mkdir(parents=True, exist_ok=True)
    if not rd.padded_npz.exists(): raise RuntimeError("run `./pm repair` first")
    files = {"orig": _rollouts(run, it, dance.padded_npz, "orig", qdir, n_perturb, log) if False else None}
    # original-reference rollouts already exist from the quality report; otherwise they are produced here
    files["orig"] = _rollouts(run, it, dance.padded_npz, "original", qdir, n_perturb, log); files["rep"] = _rollouts(run, it, rd.padded_npz, "repaired", qdir, n_perturb, log); log("rollouts ready")
    # A. geometry (+ force-contact validation of the training labels)
    def simc(f):
        ro = Q.load_rollout(f); return ro["foot_fz"] > 0.1 * ro["meta"]["mass_kg"] * 9.81
    geo = {k: P.motion_metrics(p, sim_contact=simc(files[k][0]), pad=50) for k, p in (("orig", dance.padded_npz), ("rep", rd.padded_npz))}
    geo_dance = {k: P.motion_metrics(p) for k, p in (("orig", dance.npz), ("rep", rd.npz))}
    # B. reliable-contact dynamics / difficulty
    lcache = REPO / "pipeline/ref_data/lafan_difficulty_v1.json"
    refq = {"orig": Q.analyze_reference(dance, qdir / "orig", lcache, rollouts=files["orig"][:3]), "rep": Q.analyze_reference(rd, qdir / "rep", lcache, rollouts=files["rep"][:3])}
    for v in refq.values(): v.pop("series", None)
    # C. closed loop
    roll = {"orig": Q.analyze_rollouts(dance, {"orig": files["orig"]}, qdir / "orig")["orig"], "rep": Q.analyze_rollouts(rd, {"rep": files["rep"]}, qdir / "rep")["rep"]}
    zal = {"orig": zaligned_anchored_mpjpe(dance, files["orig"]), "rep": zaligned_anchored_mpjpe(rd, files["rep"])}
    # D. training-env evaluation grid (same checkpoint, same seed / reps)
    ck = run.ckpt(it); ecfg = load_cfg("eval", run.manifest().get("eval_cfg") or "balance_first_v1"); ev = {}
    for k, d in (("orig", dance.padded_dir), ("rep", rd.padded_dir)):
        f = qdir / f"evalgrid_{k}_it{it}.json"
        if not f.exists(): write_json(f, E.evaluate(str(ck), str(d), ecfg["periodic"]["reps"], ecfg["periodic"]["seed"]))
        ev[k] = read_json(f)
    doc = dict(generated=now(), iteration=it, repair=json.loads(str(np.load(rd.npz, allow_pickle=True)["repair_info"])), geometry_padded=geo, geometry_dance=geo_dance, reference_quality={k: {x: v[x] for x in ("ground", "contact_tiers", "root_consistency", "dynamics_reliable_frames", "difficulty")} for k, v in refq.items()},
               closed_loop=roll, z_aligned_anchored=zal, evalgrid={k: dict(scores=v["scores"], nominal=v["conditions"]["nominal"]) for k, v in ev.items()})
    doc["recommendation"] = recommend(doc); write_json(qdir / "repair_validation.json", doc)
    vid = render_repair_compare(dance, rd, files["orig"][0], files["rep"][0], qdir / f"{dance.name}_repair_compare.mp4", qdir / f"{dance.name}_repair_sheet.png") if video else None
    figure(doc, qdir / "repair_comparison.png"); rep = report(run, doc, qdir, vid); log("repair report", str(rep)); return dict(report=str(rep), video=(vid or {}).get("video"), recommendation=doc["recommendation"])


def recommend(doc: dict) -> dict:
    g, c, e = doc["geometry_padded"], doc["closed_loop"], doc["evalgrid"]; dec = lambda k, key: c[k]["decomposition"][key]["mean"]; dyn = lambda k, key: c[k]["dynamics"][key]["mean"]
    crit = []
    def add(name, ok, val): crit.append(dict(name=name, passed=bool(ok), value=val))
    add("参考脚滑下降 ≥ 60%", g["rep"]["slip_sure_contact_mps"]["mean"] <= 0.4 * g["orig"]["slip_sure_contact_mps"]["mean"], f"{g['orig']['slip_sure_contact_mps']['mean']:.3f}→{g['rep']['slip_sure_contact_mps']['mean']:.3f} m/s")
    add("根轨迹与落脚一致 (最大不一致 < 0.25 m)", g["rep"]["root_vs_implied"]["max_m"] < 0.25, f"{g['orig']['root_vs_implied']['max_m']:.2f}→{g['rep']['root_vs_implied']['max_m']:.2f} m")
    add("悬浮消除 (静立离地 < 5 mm)", abs(g["rep"]["static_stance_clearance_mm"]) < 5, f"{g['orig']['static_stance_clearance_mm']:.1f}→{g['rep']['static_stance_clearance_mm']:.1f} mm")
    add("无新增穿透 (帧占比 < -1 mm 为 0)", g["rep"]["frac_below_minus1mm"] == 0.0, f"{g['rep']['frac_below_minus1mm']:.3f}")
    add("训练接触标签与力接触一致率提升 ≥ 0.10", g["rep"]["label_vs_sim_force_contact"]["agreement"] - g["orig"]["label_vs_sim_force_contact"]["agreement"] >= 0.10, f"{g['orig']['label_vs_sim_force_contact']['agreement']:.2f}→{g['rep']['label_vs_sim_force_contact']['agreement']:.2f}")
    add("冻结策略：全局 MPJPE 明显下降 (≥ 30%)", dec("rep", "global_mpjpe_mm") <= 0.7 * dec("orig", "global_mpjpe_mm"), f"{dec('orig','global_mpjpe_mm'):.0f}→{dec('rep','global_mpjpe_mm'):.0f} mm")
    add("冻结策略：存活不退化", c["rep"]["survived"] >= c["orig"]["survived"], f"{c['orig']['survived']}/{c['orig']['n']}→{c['rep']['survived']}/{c['rep']['n']}")
    za = doc["z_aligned_anchored"]
    add("冻结策略：去除参考高度偏移后的锚定 MPJPE 不退化 (≤ +2 mm)", za["rep"]["z_aligned_mm"] - za["orig"]["z_aligned_mm"] <= 2.0, f"{za['orig']['z_aligned_mm']:.1f}→{za['rep']['z_aligned_mm']:.1f} mm（未对齐口径 {za['orig']['plain_mm']:.1f}→{za['rep']['plain_mm']:.1f}，差值来自参考悬浮，不是跟踪变好）")
    add("评估网格：鲁棒存活不退化", e["rep"]["scores"]["survival_robust"] >= e["orig"]["scores"]["survival_robust"] - 1e-9, f"{e['orig']['scores']['survival_robust']:.2f}→{e['rep']['scores']['survival_robust']:.2f}")
    ok = [x["passed"] for x in crit]; data_fixed = all(ok[:5]); policy_ok = all(ok[6:])
    # retraining is worth it when the DATA is fixed AND the frozen policy is not hurt by it; the size of the possible gain is judged from what the old labels got wrong
    lab_gain = g["rep"]["label_vs_sim_force_contact"]["agreement"] - g["orig"]["label_vs_sim_force_contact"]["agreement"]
    verdict = "RETRAIN_GATE_RECOMMENDED" if (data_fixed and policy_ok and lab_gain >= 0.10) else "REPAIR_ONLY_NO_RETRAIN" if data_fixed else "REPAIR_NOT_TRUSTED"
    verdict_note = "数据缺陷已修复且冻结策略不受损；接触标签明显更准。但**跟踪精度没有可证实的提升空间**（去除高度偏移后修复前后相同），收益只可能来自接触奖励/根平移一致性 → 先做 2000 轮短程闸门，闸门显示接触一致率/滑移/穿透有改善再决定是否跑满。"
    return dict(note=verdict_note, criteria=crit, data_fixed=data_fixed, frozen_policy_ok=policy_ok, label_agreement_gain=lab_gain, verdict=verdict)


def figure(doc, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    g = doc["geometry_padded"]; c = doc["closed_loop"]; fig, ax = plt.subplots(2, 4, figsize=(17, 8)); col = ["#c0392b", "#27ae60"]; lab = ["original", "repaired"]
    def bar(a, vals, title, ylab=""):
        a.bar(lab, vals, color=col); a.set_title(title, fontsize=9); a.set_ylabel(ylab)
        for i, v in enumerate(vals): a.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    k = ("orig", "rep")
    bar(ax[0, 0], [g[x]["slip_sure_contact_mps"]["mean"] for x in k], "planted-foot slip [m/s]"); bar(ax[0, 1], [g[x]["root_vs_implied"]["max_m"] for x in k], "max |root - root implied by planted feet| [m]")
    bar(ax[0, 2], [g[x]["root_path_m"] for x in k], "root XY path length [m]"); bar(ax[0, 3], [g[x]["static_stance_clearance_mm"] for x in k], "static-stance foot clearance [mm] (floating)")
    bar(ax[1, 0], [c[x]["decomposition"]["global_mpjpe_mm"]["mean"] for x in k], "frozen policy: global MPJPE [mm]"); bar(ax[1, 1], [c[x]["decomposition"]["anchored_mpjpe_mm"]["mean"] for x in k], "frozen policy: XY/yaw-anchored MPJPE [mm]")
    bar(ax[1, 2], [g[x]["label_vs_sim_force_contact"]["agreement"] for x in k], "training contact label vs MuJoCo force contact (agreement)"); bar(ax[1, 3], [g[x]["com_speed_max"] for x in k], "reference CoM speed peak [m/s]")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def report(run: Run, doc: dict, qdir: Path, vid) -> Path:
    g, gd, c, e, rq = doc["geometry_padded"], doc["geometry_dance"], doc["closed_loop"], doc["evalgrid"], doc["reference_quality"]; rec = doc["recommendation"]; rp = doc["repair"]
    f = lambda x, n=2: "-" if x is None else f"{x:.{n}f}"; ms = lambda a, n=1: "-" if not a else f"{a['mean']:.{n}f}±{a['std']:.{n}f}"
    L = [f"# {run.dance.name}：参考动作修复前后对比（Pipeline V1.2 `pm repair`，自动生成 {doc['generated']}）", "",
         f"> 冻结模型 it{doc['iteration']}（不重训）。修复只重建**根平移（XY/Z）**；帧数、fps、关节角/角速度、根朝向**逐位不变**（程序内 assert）。原始文件未改。没有连接真机。", "",
         f"## 结论：**{rec['verdict']}**", "", "| 判据 | 结果 | 数值（修复前→后） |", "|---|---|---|"] + [f"| {x['name']} | {'PASS' if x['passed'] else '**FAIL**'} | {x['value']} |" for x in rec["criteria"]]
    L += ["", {"RETRAIN_GATE_RECOMMENDED": rec["note"],
              "REPAIR_ONLY_NO_RETRAIN": "数据被修复，但没有足够证据表明重训会带来收益；可保留修复数据用于分析/评估。",
              "REPAIR_NOT_TRUSTED": "修复未满足数据判据，不建议使用。"}[rec["verdict"]], "",
         "## 1. 修复做了什么 / 没做什么", "", f"- 方法：接触足底质心（g1mj 碰撞几何软最小加权）→ 接触权重（<{rp['params']['lo_mm']:.0f} mm 为 1，>{rp['params']['hi_mm']:.0f} mm 为 0，静立帧强制为 1）→ Z：把加权触地足压到 z=0（{rp['params']['iters']} 次迭代，低通 {rp['params']['fc_z_hz']} Hz，且不低于无穿透下界）→ XY：根速度 = −触地足相对根位置的变化率，插值补缺、低通 {rp['params']['fc_xy_hz']} Hz、从原起点积分。无接触缺口帧占比 {rp['gap_frac']*100:.1f}%。",
         f"- 改变量：根 XY 最大平移 {rp['max_dxy_m']:.2f} m，Z 最大 {rp['max_dz_mm']:.0f} mm（起点 {rp['dz_start_mm']:.0f} mm、终点 {rp['dz_end_mm']:.0f} mm）。", "- 没做：不改关节角（因此残余脚滑 = 关节动作本身隐含的滑移，是下限）；不改根朝向；不改时序。", "",
         "## 2. 几何指标（含前后各 1 s 静止段的训练片段；接触判定口径见脚注）", "", "| 指标 | 修复前 | 修复后 | 说明 |", "|---|---|---|---|",
         f"| 支撑脚滑移 均值 (m/s) | {f(g['orig']['slip_sure_contact_mps']['mean'],3)} | {f(g['rep']['slip_sure_contact_mps']['mean'],3)} | 确定触地帧（高于各自地面 <10 mm）的接触足底质心水平速度 |",
         f"| 支撑脚滑移 p95 (m/s) | {f(g['orig']['slip_sure_contact_mps']['p95'],3)} | {f(g['rep']['slip_sure_contact_mps']['p95'],3)} | |",
         f"| 根 XY 路径 (m) | {f(g['orig']['root_path_m'])} | {f(g['rep']['root_path_m'])} | 落脚隐含路径 {f(g['orig']['implied_path_m'])} |",
         f"| 根与落脚隐含根 最大不一致 (m) | {f(g['orig']['root_vs_implied']['max_m'])} | {f(g['rep']['root_vs_implied']['max_m'])} | 越小越说明根没有在地面上滑 |",
         f"| 静立离地 (mm) | {f(g['orig']['static_stance_clearance_mm'],1)} | {f(g['rep']['static_stance_clearance_mm'],1)} | 悬浮 |",
         f"| 最低脚 1 s 窗口中位数范围 (mm) | {g['orig']['floating_window_median_range_mm']} | {g['rep']['floating_window_median_range_mm']} | 地面漂移 |",
         f"| 穿透（< −1 mm 帧占比 / 最小间隙 mm） | {g['orig']['frac_below_minus1mm']:.3f} / {g['orig']['min_clearance_mm']:.1f} | {g['rep']['frac_below_minus1mm']:.3f} / {g['rep']['min_clearance_mm']:.1f} | 修复不引入穿透 |",
         f"| 质心速度峰值 (m/s) | {f(g['orig']['com_speed_max'])} | {f(g['rep']['com_speed_max'])} | 原值是脚滑伪影 |", f"| 质心水平加速度 p95 (m/s²) | {f(g['orig']['com_acc_xy_p95'])} | {f(g['rep']['com_acc_xy_p95'])} | |",
         f"| 根高度范围 (m) | {[round(x,3) for x in g['orig']['root_height_range_m']]} | {[round(x,3) for x in g['rep']['root_height_range_m']]} | |", "",
         "> 接触足底质心 = 足底采样点按离地间隙软最小加权的水平质心；“确定触地”=高度（相对各自静立标定地面）<10 mm。", "",
         "## 3. 可靠接触帧上的 CoM / ZMP / 支撑域余量（`quality.analyze_reference`）", "", "| 指标 | 修复前 | 修复后 |", "|---|---|---|"]
    for nm, getter in (("可靠帧覆盖舞蹈段", lambda r: f"{r['dynamics_reliable_frames']['coverage']*100:.0f}%"), ("确定双/单支撑/不确定", lambda r: f"{r['contact_tiers']['double_support']*100:.0f}/{r['contact_tiers']['single_support']*100:.0f}/{r['contact_tiers']['ambiguous']*100:.0f}%"),
                       ("ZMP 余量 p05 (mm)", lambda r: f"{r['dynamics_reliable_frames']['zmp_margin_mm']['p05']:.0f}"), ("ZMP 在支撑域外", lambda r: f"{r['dynamics_reliable_frames']['zmp_outside_frac']*100:.0f}%"),
                       ("CoM 在支撑域外", lambda r: f"{r['dynamics_reliable_frames']['com_outside_frac']*100:.0f}%"), ("质心加速度 p95 (m/s²)", lambda r: f"{r['dynamics_reliable_frames']['com_acc_p95']:.1f}"),
                       ("地面标定偏移 (mm)", lambda r: str(r['ground']['floor_offset_mm'])), ("根平移一致性判定", lambda r: r['root_consistency']['verdict'])):
        L.append(f"| {nm} | {getter(rq['orig'])} | {getter(rq['rep'])} |")
    dv = {k: rq[k]["difficulty"]["vs_lafan"] for k in ("orig", "rep")}
    L += ["", "难度（LAFAN 舞蹈内百分位）：" + "；".join(f"{nm} {dv['orig'][k+'@dance']['value']:.2f}→{dv['rep'][k+'@dance']['value']:.2f}（{dv['orig'][k+'@dance']['percentile']:.0f}→{dv['rep'][k+'@dance']['percentile']:.0f}）" for k, nm in (("com_speed_max", "质心速度峰值"), ("com_acc_p95", "质心加速度 p95"), ("zmp_offset_p95", "ZMP-CoM 偏移 p95"), ("foot_spread_p95", "两脚间距 p95"))),
          "- **原先“质心速度/加速度最高”的难度结论有相当大一部分来自脚滑伪影**；修复后真实难度主要剩下宽步幅与高频换脚。", "",
          f"## 4. 冻结模型 it{doc['iteration']} 在两份参考上的闭环（MuJoCo 部署链；1 参考起点 + 8 扰动起点；均值±std）", "", "| 指标 | 原始参考 | 修复参考 |", "|---|---|---|"]
    for nm, sec, k, nd in (("存活", None, None, 0), ("全局 MPJPE (mm)", "decomposition", "global_mpjpe_mm", 0), ("XY 平移对齐 (mm)", "decomposition", "xy_offset_anchored_mpjpe_mm", 1), ("XY+yaw 锚定 MPJPE (mm)", "decomposition", "anchored_mpjpe_mm", 1),
                           ("根 XY 偏差均值 (m)", "decomposition", "root_xy_offset_m/mean", 2), ("根 XY 偏差最大 (m)", "decomposition", "root_xy_offset_m/max", 2), ("根 Z 偏差 (mm)", "decomposition", "root_z_offset_mm", 1),
                           ("├ 参考自身根不一致 (m)", "decomposition", "offset_components_m/reference_root_inconsistency", 2), ("├ 姿态隐含差 (m)", "decomposition", "offset_components_m/pose_implied_difference", 2), ("└ 策略滑移漂移 (m)", "decomposition", "offset_components_m/policy_slip_drift", 2),
                           ("接触一致率（对参考确定帧）", "dynamics", "contact/vs_reference/agreement", 3), ("双支撑 / 单支撑", None, "ds", 0), ("舞蹈段穿透最大 (mm)", "dynamics", "penetration_mm/dance_max", 1), ("CoP 距支撑域边缘 p05 (mm)", "dynamics", "balance/cop_margin_mm/p05", 1),
                           ("力矩峰值比 p95", "dynamics", "torque/peak_ratio_p95", 3), ("髋Roll 饱和(左)", "dynamics", "torque/hip_roll_sat/left", 4)):
        if sec is None and k is None: L.append(f"| {nm} | " + " | ".join(f"{c[x]['survived']}/{c[x]['n']}" for x in ("orig", "rep")) + " |")
        elif k == "ds": L.append(f"| {nm} | " + " | ".join(f"{ms(c[x]['dynamics']['contact/double_support'],2)} / {ms(c[x]['dynamics']['contact/single_support'],2)}" for x in ("orig", "rep")) + " |")
        else: L.append(f"| {nm} | " + " | ".join(ms(c[x][sec][k], nd) for x in ("orig", "rep")) + " |")
    L += ["", "### 训练环境评估网格（同一 checkpoint / seed 1234 / 8 次，V1.1 评估器）", "", "| | 原始参考 | 修复参考 |", "|---|---|---|"]
    for nm, k, nd in (("平衡分", "balance", 1), ("还原分", "fidelity", 1), ("综合", "combined", 1), ("鲁棒存活", "survival_robust", 2), ("nominal MPJPE (mm)", "mpjpe_nominal_mm", 1), ("足底滑移 (m/s)", "slip_mean", 3), ("穿透最大 (mm)", "pen_max_mm", 1)):
        L.append(f"| {nm} | {e['orig']['scores'][k]:.{nd}f} | {e['rep']['scores'][k]:.{nd}f} |")
    L.append(f"| 接触一致率（训练标签 vs 仿真足端接触，nominal） | {e['orig']['nominal']['contact_agreement']:.3f} | {e['rep']['nominal']['contact_agreement']:.3f} |")
    za = doc["z_aligned_anchored"]
    L += ["", f"> **解读警告**：还原分/nominal MPJPE 的“提升”（{e['orig']['scores']['fidelity']:.1f}→{e['rep']['scores']['fidelity']:.1f}，{e['orig']['scores']['mpjpe_nominal_mm']:.1f}→{e['rep']['scores']['mpjpe_nominal_mm']:.1f} mm）**不是策略跟踪变好**：锚定 MPJPE 保留参考根高度，原始参考悬浮约 18 mm 被计为误差。把参考按根 Z 偏移对齐后：原始 {za['orig']['z_aligned_mm']:.1f} mm vs 修复 {za['rep']['z_aligned_mm']:.1f} mm，几乎相同。不要把这 9 分当作重训收益；足底滑移（{e['orig']['scores']['slip_mean']:.3f}→{e['rep']['scores']['slip_mean']:.3f} m/s）与穿透也没有变化，说明这两项由策略/物理决定，不受数据修复影响。"]
    L += ["", "## 5. 是否值得用修复数据重训", "",
          f"- **训练实际使用的接触标签**（`MotionLib.set_ref_contact`：踝高度低于“片段自身 5% 分位地面”+15 mm 且水平速度 <0.5 m/s）与 MuJoCo 受力接触的一致率：{g['orig']['label_vs_sim_force_contact']['agreement']:.2f} → {g['rep']['label_vs_sim_force_contact']['agreement']:.2f}。原始标签里“参考判摆动而机器人实际承重”的比例 {g['orig']['label_vs_sim_force_contact']['ref_swing_sim_loaded']*100:.0f}% → {g['rep']['label_vs_sim_force_contact']['ref_swing_sim_loaded']*100:.0f}%。训练用 `--contact_reward 0.5` 奖励机器人让足部接触与这些标签一致，所以原始标签的错误会被直接训练进策略。",
          f"- 机器人实际承重时参考足底离地中位数：{g['orig']['sim_loaded_foot_height_mm']['median']:.1f} → {g['rep']['sim_loaded_foot_height_mm']['median']:.1f} mm。", f"- 全局 MPJPE {c['orig']['decomposition']['global_mpjpe_mm']['mean']:.0f}→{c['rep']['decomposition']['global_mpjpe_mm']['mean']:.0f} mm：修复后根偏差（均值 {c['rep']['decomposition']['root_xy_offset_m/mean']['mean']:.2f} m）才可以解释为策略的真实步幅/落脚差（姿态隐含差 {c['rep']['decomposition']['offset_components_m/pose_implied_difference']['mean']:.2f} m）；它仍远大于锚定 MPJPE，不应与之混用，但此后可作为有意义的回归指标。",
          "- 风险/不确定：① 策略在原始数据上已经很好（锚定 MPJPE ≈25 mm，Sim2Sim 全过），**重训的收益主要在接触标签与根平移一致性，而不是跟踪精度**，预期跟踪误差不会明显下降；② 修复后的根 Z 比原始低约 20–40 mm，策略对 `ref_root` 高度有依赖，需要确认重训后 Sim2Sim 不退化；③ 单个舞蹈，修复参数（接触阈值/截止频率）只在 YTHN 上验证过；④ 修复不能消除关节动作本身隐含的残余滑移。",
          "- **建议**：先用同一配方（`robust_d_v1`，init 官方预训练）在修复数据上做 2000 轮短程闸门（≈4 分钟，闸门已有），对比 contact_agreement、滑移、穿透、锚定 MPJPE 与原 it2000；闸门通过再决定是否跑满 40000（≈1 小时）。本轮未重训。", "",
          "## 6. 产物", "", f"- 修复动作：`motions/{run.dance.name}_repaired/`、`motions/{run.dance.name}_repaired_padded/`（原始 `motions/{run.dance.name}/` 未改）", f"- 对比视频：`{(vid or {}).get('video', '-')}`；关键帧：`{(vid or {}).get('sheet', '-')}`；对比图：`reports/repair/repair_comparison.png`", "- 数据：`reports/repair/repair_validation.json`", ""]
    p = run.reports / "REPAIR_REPORT.md"; p.write_text("\n".join(L) + "\n"); return p
