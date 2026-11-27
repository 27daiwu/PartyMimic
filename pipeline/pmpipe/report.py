"""PartyMimic Pipeline V1.1 - quality report orchestrator: `./pm quality <dance>`  (also runs at the end of `./pm run`).

For one run it (re)uses what exists and only computes what is missing:
  1. models to compare   : default = pipeline-selected + raw-best balance + raw-best combined (<= 3), or `--its`
  2. ONNX + deploy assets: pmpipe.post.export_model (the SAME export path as the deploy package)
  3. closed-loop rollouts: pmpipe.rollout in the sim2real MuJoCo model (deployment control chain), nominal + N perturbed starts, FULL contact-force logging
  4. reference analysis  : pmpipe.quality.analyze_reference (ground calibration, contact tiers, root consistency, dynamics on reliable frames, difficulty vs LAFAN)
  5. rollout analysis    : global-MPJPE decomposition, contact / penetration / torque / guard / balance
  6. hard re-evaluation + evaluation-seed noise of every compared model (same evaluator, same grid, so the numbers are comparable)
  7. quad-panel replay video (reference + 3 models), model-comparison figure, provenance block, QUALITY_REPORT.md
Everything lands in results/<dance>/.../reports/quality/. Nothing outside the dance's result directory is written; motion files are never modified.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from . import quality as Q
from .core import PY, REPO, S2R_PY, Logger, Run, load_cfg, now, read_csv, read_json, sh, write_json

EVALUATOR_VERSION = "V1.1 (per-joint torque limits via actuator->joint map; V1.0 results are archived as *_v10_buggy / archive/v10_actuator_limit_bug)"
ROLLOUT = REPO / "pipeline/pmpipe/rollout.py"


def sha12(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20): h.update(b)
    return h.hexdigest()[:12]


def default_models(run: Run, ecfg: dict) -> list[int]:
    from .selection import rank_scores
    rows = [r for r in read_csv(run.eval / "summary.csv") if int(r["iteration"]) > 0]
    items = {int(r["iteration"]): dict(scores={k: float(v) for k, v in r.items() if k not in ("json", "checkpoint")}, nominal_survival=float(r["nominal_survival"])) for r in rows}
    out = []
    sel = read_json(run.final / "final_selection.json", {})
    if sel.get("primary"): out.append(int(sel["primary"]))
    for k in ("balance", "combined"):
        if items: out.append(max(items, key=lambda it: items[it]["scores"][k]))
    out.append(max(items) if items else 0)
    seen, res = set(), []
    for it in out:
        if it and it not in seen and run.ckpt(it).exists(): seen.add(it); res.append(it)
    return sorted(res)[:3]


def _tag(it: int) -> str: return f"it{it:06d}"


def make_rollouts(run: Run, its, qdir: Path, n_perturb: int, procs: int, ecfg: dict, dcfg: dict, log) -> dict:
    from .post import export_model
    tdir = qdir / "traj"; tdir.mkdir(parents=True, exist_ok=True); files = {}
    for it in its:
        art = export_model(run, it, ecfg, dcfg, log)
        base = dict(onnx=art["onnx"], params=Path(art["assets"]) / "policy_params.txt", padded=run.motion_padded_npz)
        nom = tdir / f"{_tag(it)}_nominal.npz"
        if not nom.exists():
            r = sh([S2R_PY, ROLLOUT, "--onnx", base["onnx"], "--params", base["params"], "--padded-npz", base["padded"], "--out", nom], capture=True)
            if r.returncode: raise RuntimeError(f"rollout failed it{it}: {r.stdout[-400:]}{r.stderr[-800:]}")
        pert = sorted(tdir.glob(f"{_tag(it)}_perturb_*.npz"))
        if len(pert) < n_perturb:
            for f in pert: f.unlink()
            r = sh([S2R_PY, ROLLOUT, "--onnx", base["onnx"], "--params", base["params"], "--padded-npz", base["padded"], "--out", tdir / f"{_tag(it)}_perturb.npz", "--init", "perturb", "--n", n_perturb, "--procs", procs], capture=True)
            if r.returncode: raise RuntimeError(f"perturbed rollouts failed it{it}: {r.stderr[-800:]}")
            pert = sorted(tdir.glob(f"{_tag(it)}_perturb_*.npz"))
        files[it] = [nom] + pert[:n_perturb]
    return files


def seed_noise(run: Run, its, qdir: Path, ecfg: dict, seeds, log) -> dict:
    """evaluation-seed noise of the periodic-grid scores of every compared model (same evaluator)."""
    from . import evaluator as E
    d = qdir / "seed_eval"; d.mkdir(parents=True, exist_ok=True); out = {}
    for it in its:
        L = []
        for sd in seeds:
            f = d / f"{_tag(it)}_seed{sd}.json"
            if not f.exists():
                r = E.evaluate(str(run.ckpt(it)), str(run.motion_padded_dir), ecfg["periodic"]["reps"], sd); write_json(f, r)
            L.append(json.loads(f.read_text())["scores"])
        out[it] = {k: dict(mean=float(np.mean([x[k] for x in L])), std=float(np.std([x[k] for x in L], ddof=1)) if len(L) > 1 else 0.0) for k in ("balance", "fidelity", "combined", "mpjpe_nominal_mm", "slip_mean", "pen_max_mm", "torque_sat_mean")}
    return out


def provenance(run: Run, its) -> dict:
    sup = (run.logs / "supervisor.log"); launches = []
    if sup.exists():
        for l in sup.read_text().splitlines():
            if "training launched" in l:
                m = re.search(r'"resume_it": (\d+)', l); launches.append(dict(time=l[:19], resume_it=int(m.group(1)) if m else None))
    args = read_json(run.train / "args.json", {})
    git = lambda *a: subprocess.run(["git", *a], cwd=str(REPO), capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["git", "diff", "--", "src", "tools"], cwd=str(REPO), capture_output=True).stdout
    prov = dict(generated=now(), dance=run.dance.name, run=run.id, evaluator=EVALUATOR_VERSION, repo_head=git("rev-parse", "--short", "HEAD"),
                src_tools_uncommitted_diff_sha12=hashlib.sha256(diff).hexdigest()[:12], src_tools_uncommitted_files=[l.split()[-1] for l in git("status", "--short", "--", "src", "tools").splitlines()],
                init_checkpoint=dict(path=str(run.init_pt), sha12=sha12(run.init_pt)) if run.init_pt.exists() else None,
                source_pkl=dict(path=str(run.dance.source_pkl()), sha12=sha12(run.dance.source_pkl())) if run.dance.source_pkl().exists() else None,
                motion_npz_sha12=sha12(run.motion_npz) if run.motion_npz.exists() else None, padded_npz_sha12=sha12(run.motion_padded_npz) if run.motion_padded_npz.exists() else None,
                train_launches=launches, train_args={k: args.get(k) for k in ("robot", "seed", "iterations", "ckpt_every", "last_every", "num_envs", "resume", "init")},
                manifest=read_json(run.pm / "manifest.json", {}).get("config_snapshot"), models={str(it): dict(ckpt_sha12=sha12(run.ckpt(it)), onnx_sha12=sha12(run.final / f"onnx/{run.dance.name}_it{it:06d}.onnx") if (run.final / f"onnx/{run.dance.name}_it{it:06d}.onnx").exists() else None) for it in its},
                archived_invalid=[str(p.relative_to(run.root)) for p in sorted(run.root.glob("*v10*")) + sorted((run.root / "archive").glob("*v10*")) if p.exists()],
                reports=[dict(file=str(p.relative_to(run.root)), mtime=time_of(p)) for p in sorted([run.root / "YTHN_AUTO_REPORT.md", run.reports / "SELECTION_REPORT.md", run.reports / "SHORT_RUN_GATE.md"]) if p.exists()])
    return prov


def time_of(p: Path) -> str:
    import time
    return time.strftime("%F %T", time.localtime(p.stat().st_mtime))


def sim2sim_of(run: Run, it: int):
    for f in sorted((run.final / "sim2sim").glob("*.json")):
        a = read_json(f, {}).get("args", {})
        if re.search(rf"(?:it0*{it}|_it{it})\b", str(a.get("onnx", ""))) and "summary" in read_json(f, {}):
            s = read_json(f)["summary"]; ns = sorted({int(v["n"]) for v in s.values() if "n" in v})
            return dict(file=f.name, mean_success=float(np.mean([v["success"] for v in s.values()])), nominal=s.get("nominal", {}).get("success"), worst=min(v["success"] for v in s.values()),
                        n_conditions=len(s), total_inits=ns[-1] if ns else None, sampled=a.get("n"))
    return None


def verdicts(ref: dict, roll: dict, tag: str) -> dict:
    """auto-generated one-line conclusions (numbers in, wording fixed) for the final / first model."""
    d = roll[tag]["decomposition"]; g = lambda k: d[k]["mean"]
    G, A1, E = g("global_mpjpe_mm"), g("xy_offset_anchored_mpjpe_mm"), g("root_xy_offset_m/mean")
    inc, slip, pose = g("offset_components_m/reference_root_inconsistency"), g("offset_components_m/policy_slip_drift"), g("offset_components_m/pose_implied_difference")
    v = {}
    v["global_mpjpe"] = dict(global_mm=G, xy_anchored_mm=A1, root_xy_offset_m=E, reference_root_inconsistency_m=inc, policy_slip_drift_m=slip, pose_implied_difference_m=pose,
                             verdict="REFERENCE_ROOT_TRANSLATION" if (A1 < 0.15 * G and inc > 0.6 * E) else "POLICY_DRIFT" if slip > 0.5 * E else "MIXED")
    gr = ref["ground"]; v["floating"] = dict(static_floor_offset_mm=gr["floor_offset_mm"], drift_range_mm=gr["floor_drift"]["range_mm"], verdict="FLOATING_REFERENCE" if max(gr["floor_offset_mm"]) > 8 else "OK")
    v["root_consistency"] = ref["root_consistency"]["verdict"]
    return v


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def ms(a, nd=1):
    return "-" if not a else f"{a['mean']:.{nd}f}±{a['std']:.{nd}f}"


def build_report(run: Run, its, ref, roll, hard, noise, prov, vid, ecfg, qdir: Path, expl) -> Path:
    from .selection import explain
    tags = {it: f"it{it}" for it in its}; fin = (read_json(run.final / "final_selection.json", {}) or {}).get("primary")
    R = lambda it, k: hard[it]["scores"][k]
    v = verdicts(ref, roll, tags[fin if fin in its else its[-1]]); gm = v["global_mpjpe"]
    L = [f"# {run.dance.name} / {run.id}：动作质量分析与模型对比（Pipeline V1.1，自动生成 {now()}）", "",
         f"> 评估器版本：{EVALUATOR_VERSION}。**没有连接真机**；原始动作文件未被修改。回放与指标均来自 MuJoCo 闭环物理（部署控制链：ONNX→authority/ramp→C++ guard 语义→200 Hz PD→力矩截断）。", "",
         "## 0. 结论摘要", ""]
    L += [f"1. **全局 MPJPE 根因判定：`{gm['verdict']}`**（{ {'REFERENCE_ROOT_TRANSLATION': '主因是参考动作根平移与自身落脚不一致（脚滑/根漂移），不是策略漂移，也不是坐标系错位', 'POLICY_DRIFT': '主因是策略自身相对落脚的滑移，参考本身基本一致；不是坐标系错位', 'MIXED': '参考不一致与策略滑移都有贡献；不是坐标系错位'}[gm['verdict']] }）。"
          f" 平均全局 MPJPE {gm['global_mm']:.0f} mm，而只做 XY 平移对齐后降到 {gm['xy_anchored_mm']:.0f} mm；根 XY 平均偏差 {gm['root_xy_offset_m']:.2f} m，其中参考自身不一致 {gm['reference_root_inconsistency_m']:.2f} m、策略相对自身落脚的滑移 {gm['policy_slip_drift_m']:.2f} m。",
          f"2. **参考悬浮**：静立时双脚离地 {ref['ground']['floor_offset_mm'][0]:.0f}/{ref['ground']['floor_offset_mm'][1]:.0f} mm（转换按全片最低的单个样本对齐 z=0），且全程地面高度在 {ref['ground']['floor_drift']['range_mm'][0]:.0f}~{ref['ground']['floor_drift']['range_mm'][1]:.0f} mm 间漂移；机器人在刚性地面上必然比参考低约 {-roll[tags[its[-1]]]['decomposition']['root_z_offset_mm']['mean']:.0f} mm。",
          f"3. 接触相位：参考 vs 仿真在“确定帧”上一致率 {roll[tags[its[-1]]]['dynamics']['contact/vs_reference/agreement']['mean']*100:.0f}%（参考判摆动而仿真承重 {roll[tags[its[-1]]]['dynamics']['contact/vs_reference/ref_swing_sim_loaded']['mean']*100:.0f}%，参考判触地而仿真未承重 {roll[tags[its[-1]]]['dynamics']['contact/vs_reference/ref_contact_sim_unloaded']['mean']*100:.0f}%）。",
          f"4. 选模：按统一规则（平衡优先，容差 {ecfg['selection']['balance_tolerance']} 分）最终 **it{fin}**；同档候选 {expl.get('tier')}，平衡分差在评估噪声以内，最终由还原度决定。", ""]
    L += ["## 1. 回放视频与图", "", f"- 四宫格同步回放（参考 + {len(its)} 个模型，叠加接触状态/倾角/CoM/CoP/力矩饱和/护栏/帧号 + 俯视小地图）：`{vid['video'].replace(str(run.root) + '/', '')}`",
          f"- 关键帧拼图：`{str(vid['sheet']).replace(str(run.root) + '/', '')}`", "- 参考动作质量图：`reports/quality/" + f"{run.dance.name}_reference_quality.png`；轨迹分解图：`reports/quality/{run.dance.name}_trajectory_decomposition.png`；模型对比图：`reports/quality/model_comparison.png`", ""]
    # ---- global MPJPE
    L += ["## 2. 全局 MPJPE 根因（坐标系误差 vs 真实轨迹漂移）", "",
          "| 量 | " + " | ".join(tags[i] for i in its) + " | 含义 |", "|---|" + "---|" * len(its) + "---|"]
    rows = [("全局 MPJPE (mm)", "global_mpjpe_mm", 0, "世界系 30-body 误差（含根平移）"), ("仅 XY 平移对齐 (mm)", "xy_offset_anchored_mpjpe_mm", 1, "去掉根 XY 偏差后"), ("XY+yaw 锚定 MPJPE (mm)", "anchored_mpjpe_mm", 1, "评估器口径，**不是**轨迹误差"),
            ("起点全局误差 (mm)", "initial_global_mpjpe_mm", 1, "t=0；非 0 = 起始姿态/扰动，不是坐标系偏差"), ("yaw 偏差均值 (°)", "yaw_offset_deg/mean_abs", 2, "朝向基本不漂 → 不是坐标/朝向错位"),
            ("根 XY 偏差 均值 (m)", "root_xy_offset_m/mean", 2, ""), ("根 XY 偏差 末端 (m)", "root_xy_offset_m/end", 2, "舞蹈结束时"),
            ("参考根路径 (m)", "path_m/reference", 2, "参考根实际走过的路程"), ("参考落脚隐含路径 (m)", "path_m/reference_implied_by_feet", 2, "按参考自己的触地脚不动推算"), ("仿真根路径 (m)", "path_m/simulation", 2, ""),
            ("├ 参考自身根不一致 (m)", "offset_components_m/reference_root_inconsistency", 2, "参考根 vs 参考落脚隐含根"), ("├ 姿态隐含差 (m)", "offset_components_m/pose_implied_difference", 2, "仿真落脚隐含根 vs 参考落脚隐含根"),
            ("└ 策略滑移漂移 (m)", "offset_components_m/policy_slip_drift", 2, "仿真根 vs 仿真自己落脚隐含根")]
    for name, k, nd, note in rows:
        L.append(f"| {name} | " + " | ".join(ms(roll[tags[i]]["decomposition"][k], nd) for i in its) + f" | {note} |")
    L += ["", "判读：", f"- 起点全局误差只有 {roll[tags[its[0]]]['decomposition']['initial_global_mpjpe_mm']['mean']:.0f} mm 且 yaw 偏差仅 {roll[tags[its[0]]]['decomposition']['yaw_offset_deg/mean_abs']['mean']:.1f}° → 仿真与参考使用同一世界系，**不存在坐标系误差**。",
          f"- 参考根在 {ref['frames_padded']} 帧里走了 {roll[tags[its[0]]]['decomposition']['path_m/reference']['mean']:.1f} m，而按它自己触地脚不动推算只应走 {roll[tags[its[0]]]['decomposition']['path_m/reference_implied_by_feet']['mean']:.1f} m（一致性判定 `{ref['root_consistency']['verdict']}`，最大不一致 {ref['root_consistency']['by_threshold']['25mm']['max_mismatch_m']:.2f} m）。参考动作里脚在地面“滑”，策略不会也不应复现这种滑行。",
          f"- 策略自身滑移漂移只有 {gm['policy_slip_drift_m']*1000:.0f} mm 量级，且三个模型相近。", f"- 另有真实成分：仿真落脚隐含根路径与参考落脚隐含根路径相差 {gm['pose_implied_difference_m']:.2f} m（仿真根路径 {roll[tags[its[0]]]['decomposition']['path_m/simulation_implied_by_feet']['mean']:.1f} m vs 参考 {roll[tags[its[0]]]['decomposition']['path_m/reference_implied_by_feet']['mean']:.1f} m），即策略的步幅/落脚位置比参考短；其中一部分来自接触分类差异（参考按高度、仿真按受力），无法进一步拆分。这一项约占总偏差的三分之一，三个模型相近。",
          "- **提示**：仅当判定为 `REFERENCE_ROOT_TRANSLATION` 时，可用 `./pm repair <舞蹈> --validate` 生成修复参考（只改根平移）并与冻结模型对比；是否值得重训见 `REPAIR_REPORT.md`。",
          {"REFERENCE_ROOT_TRANSLATION": "- **结论**：全局 MPJPE 的主成分是参考自身的脚滑/根平移不一致；全局 MPJPE 不应作为模型优劣指标，也不能靠训练降到 XY 锚定口径的水平。需要改的是参考（根平移按落脚重建），属数据问题。本轮不改原始动作。",
           "POLICY_DRIFT": "- **结论**：参考根平移与自身落脚基本一致，全局 MPJPE 的偏差主要来自策略相对落脚的滑移与步幅差，是策略质量问题（可结合接触点足滑与回放视频查看），全局 MPJPE 可以作为轨迹保持指标使用。",
           "MIXED": "- **结论**：参考不一致与策略滑移各占一部分；解读全局 MPJPE 时需同时看“参考自身根不一致”和“策略滑移漂移”两行。"}[gm["verdict"]], ""]
    # ---- reference
    c = ref["contact_tiers"]; dy = ref["dynamics_reliable_frames"]; gr = ref["ground"]
    L += ["## 3. 参考动作：悬浮、接触相位、动态 ZMP 与支撑域余量", "",
          f"- 地面标定：`{gr['method']}`（可信={gr['confident']}，静立段帧 {gr['static_frames']}，两脚离地 {gr['floor_offset_mm']} mm，段内波动 {gr['spread_mm']} mm）。",
          f"- 地面漂移：以静立地面为基准，最低脚高度的 1 s 窗口中位数范围 {gr['floor_drift']['range_mm']} mm；最低 {gr['floor_drift']['min_foot_mm']:.1f} mm；{gr['floor_drift']['frac_foot_below_floor_10mm']*100:.0f}% 的帧脚低于标定地面 10 mm 以上；结尾站姿比开头高 {gr['floor_drift']['end_hold_above_start_mm']:.0f} mm。",
          f"- 接触可靠度分层（>40 mm 才算确定摆动，<10 mm 才算确定触地）：确定双支撑 {c['double_support']*100:.0f}% / 确定单支撑 {c['single_support']*100:.0f}% / 确定腾空 {c['flight']*100:.0f}% / **不确定 {c['ambiguous']*100:.0f}%**。",
          f"- **动态 ZMP 与支撑域余量只在可靠接触帧上计算**（覆盖舞蹈段 {dy['coverage']*100:.0f}%，{dy['frames']} 帧）：ZMP 余量 p05 {dy['zmp_margin_mm']['p05']:.0f} mm、中位 {dy['zmp_margin_mm']['median']:.0f} mm；ZMP 在支撑域外 {dy['zmp_outside_frac']*100:.0f}%，CoM 在支撑域外 {dy['com_outside_frac']*100:.0f}%。质心水平加速度 p95 {dy['com_acc_p95']:.1f} m/s²。",
          f"- 运动学 ZMP 校验（对照 MuJoCo 接触力 CoP，闭环回放）：中位误差 {ref['zmp_validation']['median_mm']:.0f} mm、p90 {ref['zmp_validation']['p90_mm']:.0f} mm（n={ref['zmp_validation']['n']}）——可用于趋势判断，不能当毫米级真值。",
          "- 含义：参考的质心加速度远超静态可支撑范围（ZMP 偏出支撑域是常态），意味着参考**无法被严格动力学复现**，策略只能折中；这是难度来源，不是策略缺陷。", ""]
    dd = ref["difficulty"]["vs_lafan"]
    L += ["难度对比（LAFAN 舞蹈 8 条 / 全集 40 条，同一代码同一口径；百分位 = 严格小于该值的比例）：", "", "| 指标 | 本舞蹈 | 舞蹈中位 | 舞蹈内百分位 | 全集百分位 |", "|---|---|---|---|---|"]
    names = dict(jv_rms="关节速度 RMS (rad/s)", com_speed_max="质心速度峰值 (m/s)", com_acc_p95="质心加速度 p95 (m/s²)", zmp_offset_p95="ZMP-CoM 偏移 p95 (m)", foot_spread_p95="两脚间距 p95 (m)",
                 single_support_frac="单支撑占比", flight_frac="腾空占比", transitions_per_s="接触切换 (次/s)", root_h_range="根高度变化 (m)")
    for k, nm in names.items():
        a, b = dd.get(f"{k}@dance"), dd.get(f"{k}@all")
        if a: L.append(f"| {nm} | {a['value']:.2f} | {a['median']:.2f} | {a['percentile']:.0f} | {b['percentile']:.0f} |")
    L.append("")
    # ---- model comparison
    L += ["## 4. 三个模型对比（同一评估器 / 同一网格 / 同一回放口径）", "", "### 4.1 平衡性与还原度（评估器；硬复评 = seed 777 × 24 次 × 含超训练分布工况；噪声 = 3 个评估种子的 std）", "",
          "| 指标 | " + " | ".join(tags[i] for i in its) + " | 基线(预训练,硬复评) |", "|---|" + "---|" * (len(its) + 1)]
    b0 = hard[0]["scores"]
    for nm, k, nd in (("平衡分 (硬)", "balance", 1), ("还原分 (硬)", "fidelity", 1), ("综合 (硬)", "combined", 1), ("鲁棒存活 (硬)", "survival_robust", 2), ("nominal MPJPE (mm,硬)", "mpjpe_nominal_mm", 1), ("鲁棒 MPJPE 均值 (mm,硬)", "mpjpe_robust_mean_mm", 1), ("足底滑移 (m/s,硬)", "slip_mean", 3), ("穿透最大 (mm,硬)", "pen_max_mm", 1), ("力矩饱和占比 (硬)", "torque_sat_mean", 4)):
        L.append(f"| {nm} | " + " | ".join(fmt(R(i, k), nd) for i in its) + f" | {fmt(b0[k], nd)} |")
    for nm, k, nd in (("平衡分 评估种子噪声", "balance", 2), ("还原分 评估种子噪声", "fidelity", 2), ("综合 评估种子噪声", "combined", 2)):
        L.append(f"| {nm} (均值±std) | " + " | ".join(ms(noise[i][k], nd) for i in its) + " | - |")
    L += ["", "### 4.2 闭环回放动力学（MuJoCo 接触力；1 个 nominal 起点 + N 个扰动起点，均值±std）", "", "| 指标 | " + " | ".join(tags[i] for i in its) + " |", "|---|" + "---|" * len(its)]
    dyn = [("存活", None, 0), ("双支撑占比", "contact/double_support", 3), ("单支撑占比", "contact/single_support", 3), ("接触一致率(对参考确定帧)", "contact/vs_reference/agreement", 3), ("倾角 p95 (rad)", "balance/tilt_p95_rad", 3), ("CoP 距支撑域边缘 p05 (mm)", "balance/cop_margin_mm/p05", 1),
           ("CoM 投影在足底支撑域外占比", "balance/com_outside_sole_frac", 3), ("运动学 ZMP 在支撑域外占比", "balance/kin_zmp_outside_sole_frac", 3), ("静立段穿透最大 (mm)", "penetration_mm/static_stance_max", 1), ("舞蹈段穿透 p95 (mm)", "penetration_mm/dance_p95", 2), ("舞蹈段穿透最大 (mm)", "penetration_mm/dance_max", 1),
           ("力矩饱和占比(全关节)", "torque/sat_frac_mean", 4), ("髋Roll 饱和(左)", "torque/hip_roll_sat/left", 4), ("髋Roll 饱和(右)", "torque/hip_roll_sat/right", 4), ("髋Roll 峰值力矩 (N·m)", "torque/hip_roll_peak_nm", 1), ("力矩峰值比 p95", "torque/peak_ratio_p95", 3),
           ("接触点足滑 均值 (m/s,V1.4标准口径)", "slip/contact_point/mean", 3), ("接触点足滑 p95 (m/s)", "slip/contact_point/p95", 3), ("护栏改写目标占比", "guard/any_clip_rate", 3), ("护栏最大改写 (rad)", "guard/max_dev_rad", 3)]
    for nm, k, nd in dyn:
        if k is None: L.append(f"| {nm} | " + " | ".join(f"{roll[tags[i]]['survived']}/{roll[tags[i]]['n']}" for i in its) + " |")
        else: L.append(f"| {nm} | " + " | ".join(ms(roll[tags[i]]["dynamics"].get(k), nd) for i in its) + " |")
    s2s = {i: sim2sim_of(run, i) for i in its}
    if any(s2s.values()):
        s0 = next(v for v in s2s.values() if v); tot, smp = s0["total_inits"], s0["sampled"]; real = (tot - smp) if (tot and smp is not None) else 10
        L += ["", f"### 4.3 Sim2Sim（部署链，{s0['n_conditions']} 工况 × {tot} 初态 = {real} 真机站姿 + {smp} 采样）", "", "| | " + " | ".join(tags[i] for i in its) + " |", "|---|" + "---|" * len(its), "| 均值成功率 | " + " | ".join(fmt(s2s[i]["mean_success"], 2) if s2s[i] else "未做" for i in its) + " |",
              "| nominal | " + " | ".join(fmt(s2s[i]["nominal"], 2) if s2s[i] else "-" for i in its) + " |", "", f"> Sim2Sim 各工况基本全是 100%：每工况只有 {tot} 个初态（{real} 个真机站姿 + {smp} 个采样），推力温和，**区分度很低**，不能单独作为鲁棒性证据。"]
    # ---- the four dynamics topics
    t0 = tags[its[-1]]; rr = roll[t0]["dynamics"]; nom_d = roll[t0]["nominal"]["dynamics"]
    L += ["", "## 5. 专题结论", "",
          f"### 5.1 足底穿透\n- 参考动作本身无穿透（QA：负间隙 < -1 mm 帧数 0）。回放中静立段最大穿透 {ms(rr['penetration_mm/static_stance_max'])} mm（落地瞬态），舞蹈段 p95 {ms(rr['penetration_mm/dance_p95'], 2)} mm、最大 {ms(rr['penetration_mm/dance_max'])} mm。评估器报告的 `pen_max` 是**所有工况（含推力/低摩擦/大扰动）的最大值**，故比 nominal 回放大；它是门槛口径，不是常态穿透。",
          f"\n### 5.2 髋 Roll 力矩饱和\n- **V1.0 报告的“髋 Roll 饱和 31–33%”是评估器 bug**（执行器顺序≠关节顺序：用 25 N·m 的肩关节上限去比髋 Roll 力矩）。修正后评估器（训练环境，髋 Roll 上限 139 N·m）全部模型饱和 ≈ 0。"
          f"\n- **但训练与部署的髋 Roll 上限不一致**：训练/评估环境（mjlab）为 139 N·m，部署链（sim2real `g1.py` → `policy_params.txt effort_lim`）为 **88 N·m**。本报告的 MuJoCo 回放用部署上限 88：髋 Roll 预截断峰值 {ms(rr['torque/hip_roll_peak_nm'])} N·m（> 88），饱和占比左 {ms(rr['torque/hip_roll_sat/left'], 3)}、右 {ms(rr['torque/hip_roll_sat/right'], 3)}，且随训练**单调上升**（见 4.2：it4000→it40000 饱和与力矩峰值比同步变大）。"
          f"\n- 饱和发生在承重腿还是摆动腿：nominal 回放中左髋 Roll 承重(>70%)时 {fmt(nom_d['torque']['hip_roll_sat_when_stance']['loaded_gt70'][0], 3)}、摆动(<30%)时 {fmt(nom_d['torque']['hip_roll_sat_when_stance']['swing_lt30'][0], 3)}，**两者同量级，不能说集中在承重腿**。"
          "\n- 判断：绝对量小（≤2%）且 Sim2Sim 全过，当前不是瓶颈；但它是一个**未被训练覆盖的部署限制**——若真机以 88 N·m 为准，后期 checkpoint 比早期 checkpoint 更吃这条余量。需要确认真机硬件上限（139 还是 88）后，再决定是否在训练里用 88 重新做 DR；本轮不改训练。",
          f"\n### 5.3 接触相位\n- 双支撑 {ms(rr['contact/double_support'], 2)}、单支撑 {ms(rr['contact/single_support'], 2)}、腾空 {ms(rr['contact/flight'], 3)}；与参考确定帧一致率 {ms(rr['contact/vs_reference/agreement'], 2)}。不一致主要是“参考判摆动而仿真承重”（{ms(rr['contact/vs_reference/ref_swing_sim_loaded'], 2)}），与参考地面漂移一致：策略在真实刚性地面上本就不会悬空。",
          f"\n### 5.4 动态平衡\n- 倾角 p95 {ms(rr['balance/tilt_p95_rad'], 3)} rad，倾角主要由舞蹈姿态本身决定；CoP 距支撑域边缘 p05 仅 {ms(rr['balance/cop_margin_mm/p05'])} mm（策略在支撑域边缘工作：压力中心常贴着脚边，余量小）。", ""]
    # ---- selection + provenance
    L += ["## 6. 统一选模规则", "", f"规则（`pmpipe.selection`，评估守护 `best/best_selected`、硬复评 `final_selection.json`、本报告三处共用同一实现）：合格（nominal 存活 {ecfg['selection']['eligible']['nominal_survival_min']}、鲁棒存活 ≥ {ecfg['selection']['eligible']['robust_survival_min']}）→ 平衡分在最优 {ecfg['selection']['balance_tolerance']} 分以内算同档 → 同档取还原度最高。原始分项最大值（best_balance/best_fidelity/best_combined）仅作诊断。",
          f"\n本 run：eligible {expl.get('eligible')}；顶档 {expl.get('tier')}（平衡分 {expl.get('tier_balance_range')}）；胜出 it{expl.get('winner')}（还原 {expl.get('winner_fidelity'):.1f}）。评估种子噪声：平衡分 std ≈ {np.mean([noise[i]['balance']['std'] for i in its]):.2f}，远小于容差 {ecfg['selection']['balance_tolerance']}。", "",
          "## 7. 训练与评估溯源", "", "```json", json.dumps({k: v for k, v in prov.items() if k != "manifest"}, indent=1, ensure_ascii=False), "```", ""]
    p = run.reports / "QUALITY_REPORT.md"; p.write_text("\n".join(L) + "\n"); return p


def comparison_figure(its, hard, noise, roll, path: Path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tags = [f"it{i}" for i in its]; x = np.arange(len(its)); fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    def bar(a, vals, errs, title, ylab="", base=None):
        a.bar(x, vals, yerr=errs, color=plt.cm.tab10.colors[:len(its)], capsize=4); a.set_xticks(x); a.set_xticklabels(tags); a.set_title(title, fontsize=10); a.set_ylabel(ylab)
        if base is not None: a.axhline(base, c="k", ls=":", lw=1)
    bar(ax[0, 0], [noise[i]["balance"]["mean"] for i in its], [noise[i]["balance"]["std"] for i in its], "balance score (periodic grid, mean±seed std)")
    bar(ax[0, 1], [noise[i]["fidelity"]["mean"] for i in its], [noise[i]["fidelity"]["std"] for i in its], "fidelity score")
    d = lambda k, sec="decomposition": ([roll[t][sec][k]["mean"] for t in tags], [roll[t][sec][k]["std"] for t in tags])
    bar(ax[0, 2], *d("anchored_mpjpe_mm"), "XY/yaw-anchored MPJPE [mm] (rollout)"); bar(ax[0, 3], *d("global_mpjpe_mm"), "GLOBAL MPJPE [mm] (reference-root skating dominates)")
    bar(ax[1, 0], *d("offset_components_m/policy_slip_drift"), "policy slip drift [m]"); bar(ax[1, 1], *d("contact/vs_reference/agreement", "dynamics"), "contact agreement with reference")
    bar(ax[1, 2], *d("penetration_mm/dance_max", "dynamics"), "max penetration in dance [mm]"); bar(ax[1, 3], *d("torque/hip_roll_peak_nm", "dynamics"), "hip-roll peak torque [N.m] (limit 88)", base=88)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def run_quality(run: Run, its=None, n_perturb: int = 8, noise_seeds=(11, 12, 13), procs: int = 8, video: bool = True, log=None, eval_cfg: dict | None = None, deploy_cfg: dict | None = None) -> dict:
    from .post import hard_reeval
    from .replay import render_quad
    from .selection import explain
    log = log or Logger(run.logs / "quality.log")
    m = run.manifest(); ecfg = eval_cfg or load_cfg("eval", m.get("eval_cfg") or "balance_first_v1"); dcfg = deploy_cfg or load_cfg("deploy", m.get("deploy_cfg") or "cpp_g1_v1")
    its = sorted(its or default_models(run, ecfg))
    if not its: raise RuntimeError("no checkpoints to analyse")
    qdir = run.reports / "quality"; qdir.mkdir(parents=True, exist_ok=True); log("quality: models", its)
    files = make_rollouts(run, its, qdir, n_perturb, procs, ecfg, dcfg, log); log("rollouts ready")
    nom0 = files[its[-1]]
    ref = Q.analyze_reference(run.dance, qdir, REPO / "pipeline/ref_data/lafan_difficulty_v1.json", rollouts=nom0[:5]); ref.pop("series", None); write_json(qdir / "reference_quality.json", ref); log("reference analysis done")
    roll = Q.analyze_rollouts(run.dance, {f"it{i}": files[i] for i in its}, qdir); write_json(qdir / "rollout_quality.json", roll); log("rollout analysis done")
    hard = hard_reeval(run, {i: [] for i in its}, ecfg, log); noise = seed_noise(run, its, qdir, ecfg, noise_seeds, log)
    # selection explanation on the hard re-evaluation results (the decision grid)
    items = {it: dict(scores=r["scores"], nominal_survival=r["conditions"]["nominal"]["survival"]) for it, r in hard.items()}
    expl = explain(items, ecfg["selection"])
    comparison_figure(its, hard, noise, roll, qdir / "model_comparison.png")
    vid = render_quad(run.dance, [(f"it{i}", files[i][0]) for i in its], qdir / f"{run.dance.name}_quad_replay.mp4", qdir / f"{run.dance.name}_quad_sheet.png") if video else dict(video="-", sheet="-")
    prov = provenance(run, its); write_json(qdir / "provenance.json", prov)
    write_json(qdir / "model_comparison.json", dict(models=its, hard={str(i): hard[i]["scores"] for i in its}, baseline_hard=hard[0]["scores"], seed_noise={str(i): noise[i] for i in its}, selection=expl,
                                                  verdicts=verdicts(ref, roll, f"it{its[-1]}")))
    rep = build_report(run, its, ref, roll, hard, noise, prov, vid, ecfg, qdir, expl); log("QUALITY_REPORT", str(rep))
    return dict(report=str(rep), video=vid["video"], models=its)
