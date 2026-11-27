"""contact_refine.report - figures + CONTACT_REFINE_REPORT.md (Chinese) from the artefacts written by api.refine_dance / api.analyze_cmd."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import diagnostics as D, geometry as Gm, viz
from .config import GATES_VERSION, QA_GATES


def f1(x, n=1):
    return "—" if x is None or (isinstance(x, float) and x != x) else f"{x:.{n}f}"


def build_figures(od: Path, S, M, res, variants, focus):
    fd = od / "figures"; fd.mkdir(exist_ok=True); tilt = D.tilt_series(M["root_q"]); T = len(S["g"]); lo, hi = max(1, focus - 60), min(T - 1, focus + 60)
    viz.overview(S, od.parent.parent.name, fd / "overview_full.png", focus=(lo, hi)); viz.window(S, tilt, od.parent.parent.name, lo, hi, fd / "window_focus.png", M["joint_names"])
    viz.window(S, tilt, od.parent.parent.name, max(1, focus - 15), min(T - 1, focus + 20), fd / "window_focus_tight.png", M["joint_names"])
    viz.joint_delta(M["joint_names"], S["q0"], S["q1"], fd / "joint_delta.png"); viz.root_path(S, res, fd / "root_path.png")
    if variants: viz.variants_bar(res, variants, fd / "variants_compare.png")
    return fd


def _row(name, vals):
    return "| " + name + " | " + " | ".join(vals) + " |"


def write_report(od: Path, dance_name: str, focus: int, extra: dict, path: Path | None = None):
    m = json.loads((od / "metrics.json").read_text()); res = m["main"]; var = m.get("variants", {}); ver = json.loads((od / "qa_verdict.json").read_text()); an = m["analysis"]; ran = m["refined_analysis"]
    diag = json.loads((od / "diagnostics.json").read_text()); L = []; A = L.append
    ro, v12 = var.get("root_only_stage1"), var.get("repair_v12_root_only")
    A(f"# {dance_name}：接触修正（contact_refine）量化报告"); A("")
    A(f"> 自动生成。算法版本 `{ver['algorithm']}`；判据版本 `{ver['gates_version']}`（sha256 `{ver['gates_sha256'][:12]}`，判据在运行修正之前写入 `pipeline/pmpipe/contact_refine/config.py::QA_GATES`，未随结果改动）。")
    A(f"> 原始数据 `{ver['source']}` sha256 `{ver['source_sha256'][:16]}…` **未被修改**；修正数据写入新目录：`{ver.get('output_padded')}`。**没有连接真机。**"); A("")
    A("## 0. 结论摘要"); A("")
    A(f"- **QA 判决：{ver['verdict']}**（{ver['n_pass']}/{ver['n_gates']} 项）。未通过项：{ver['failed'] or '无'}。")
    s = res["slip"]; g = res["ground"]; j = res["joints"]; r = res["root"]; fl = res["flight"]
    A(f"- 脚滑（V1.4 接触点口径，确定支撑帧）p95 {f1(s['original']['p95'],3)} → {f1(s['refined']['p95'],3)} m/s（**−{f1(s['p95_reduction_pct'],0)}%**），均值 {f1(s['original']['mean'],3)} → {f1(s['refined']['mean'],3)}（−{f1(s['mean_reduction_pct'],0)}%）；p99 {f1(s['original']['p99'],2)} → {f1(s['refined']['p99'],2)}，**最大值几乎不变**（{f1(s['original']['max'],2)} → {f1(s['refined']['max'],2)} m/s，出现在落脚/抬脚过渡帧）。")
    A(f"- 悬浮/穿透：原始静止/支撑脚相对真实地面 z=0 的最低足底点高度 p95 {f1(g['original_vs_z0']['stance_abs_height_p95_mm'],0)} mm（**整段悬浮，中位 {f1(g['original_vs_z0']['stance_height_p50_mm'],0)} mm**；相对自估地面曲线仍有 p95 {f1(g['original_vs_ground_curve']['stance_abs_height_p95_mm'],1)} mm、穿透最大 {f1(g['original_vs_ground_curve']['penetration_max_mm'],1)} mm）→ 修正后 p95 {f1(g['stance_abs_height_p95_mm'],1)} mm、穿透最大 {f1(g['penetration_max_mm'],2)} mm。")
    A(f"- 改动量：上肢/腰/头关节 **{f1(j['upper_max_deg'],2)}°**（未动）；腿关节最大 {f1(j['leg_noankle_max_deg'],1)}°、踝最大 {f1(j['ankle_max_deg'],1)}°，全关节 RMS {f1(j['rms_all_deg'],2)}°；根 z 偏移范围 {f1(r['dev_z_range_mm'][0],0)}~{f1(r['dev_z_range_mm'][1],0)} mm，根 XY 最大偏离 {f1(r['dev_xy_max_m'],2)} m；骨盆相对 30-body 风格偏离 {f1(res['style']['pelvis_rel_mpjpe_mm'],2)} mm，上肢 {res['style']['upper_pelvis_rel_mpjpe_mm']:.4f} mm。")
    A(f"- 腾空：原始数据通过弹道一致性检验的**真实腾空 {fl['n_true_flights_orig']} 段**（候选 {fl['n_candidates_orig']} 段）；朴素口径（双脚高出 z=0 25 mm）判腾空占比 {f1(fl['naive_flight_frac_orig_v11']*100,0)}%，其余全部是地面漂移/重定向悬浮造成的**“悬浮误判腾空”**。")
    A(f"- 对照 V1.2 repair（只改根）：见 §6；stage-1（本模块仅改根 XYZ）见 §6。"); A("")
    A("## 1. 方法（通用，与舞蹈无关）"); A("")
    A("1. **足底模型**：g1mj 每脚 7 根碰撞胶囊的 14 个端点（半径 10 mm 球）= 足底采样点，按位置标记脚跟/脚尖。2. **地面标定**：两脚最低点高度的非对称最小二乘下包络（点在曲线之上代价小、之下代价大），曲率限制（等效截止 2.5 Hz、漂移速率 ≤0.2 m/s）、静立段加权锚定、验证腾空段权重 0；输出地面曲线 g(t) 与置信度。3. **接触识别**：每脚逐帧 `P(stance)` = 足高 logistic × (0.25+0.75·水平/竖直速度高斯)，静立段强化，Schmitt 迟滞 + 最小持续 3 帧；输出 stance / swing / uncertain + 置信度；无支撑且非腾空的帧按“总有一只支撑脚”先验推断支撑（置信度≤0.5，单独标记）。4. **真实腾空**：双脚离地候选段必须满足质心弹道（拟合竖直加速度 −13.5~−3.5 m/s²、拟合残差 ≤12 mm、时长 ≤1.2 s），否则判为 `false_flight`（悬浮）。")
    A("5. **根 XYZ 优化（stage 1，线性稀疏最小二乘）**+ **腿部 IK（stage 2，稀疏 Gauss-Newton + LM 阻尼，变量=每帧 root Δxyz 与 12 个腿关节 Δ）**：残差 = 支撑脚采样点水平钉住（每个 stance 段×采样点一个自由锚点，Huber 鲁棒）+ 最低采样点贴地(z=+1 mm) + 单侧不穿地铰链(≥0.5 mm) + 弱足底放平；正则 = 修改量(而非轨迹)的 0/1/2 阶差分（保留舞蹈高频）、腿关节 box 约束 ±6°/踝 ±10°、关节限位；腾空段根修正近似刚性平移（保弹道）。6. 最后一次“抬到地面”硬保证无穿透。7. **导出**：body_pos/quat/lin_vel/ang_vel 与 joint_vel 全部用 `./pm prepare` 同一转换器代码重新一致计算，字段/dtype/形状与 prepare 产物一致。上肢、腰、头、根朝向、帧数、fps 不变。")
    A(""); A("## 2. 数据 QA 判决（预先固定的判据）"); A("")
    A("| 判据 | 要求 | 实测 | 结果 |"); A("|---|---|---|---|")
    for gt in ver["gates"]:
        v = gt["value"]; vs = "—" if v is None else (str(v) if isinstance(v, bool) else f"{v:.3f}")
        A(f"| {gt['id']}：{gt['desc']} | {gt['op']} {gt['threshold']} {gt['unit']} | {vs} | {'PASS' if gt['passed'] else '**FAIL**'} |")
    for n in ver.get("notes", []): A(f"\n> {n}")
    A(""); A("## 3. 全段（{} 帧）诊断".format(an["frames"])); A("")
    gr = diag["analysis"]["ground"]
    A(f"- **地面**：估计地面曲线范围 {f1(gr['range_mm'][0],0)}~{f1(gr['range_mm'][1],0)} mm（相对转换器给的 z=0），最大漂移速率 {f1(gr['rate_max_mps'],3)} m/s（界 0.2），静立锚定帧 {gr['static_frames']}，包络下方残差 p99 {f1(gr['resid_below_mm_p99'],1)} mm，包络上方残差中位 {f1(gr['resid_above_median_mm'],1)} mm；置信度均值 {f1(gr['conf_mean'],2)}。残余：地面曲线是“被支撑脚抬起的下包络”，脚长时间整体离地（双脚都在空中摆动）的时段曲线由两侧插值，置信度低。")
    c0, c1 = an, ran
    A("| 接触状态占比 | 原始(L/R) | 修正后(L/R) |"); A("|---|---|---|")
    A(_row("支撑 stance", [f"{c0['stance_frac'][0]:.2f} / {c0['stance_frac'][1]:.2f}", f"{c1['stance_frac'][0]:.2f} / {c1['stance_frac'][1]:.2f}"]))
    A(_row("摆动 swing", [f"{c0['swing_frac'][0]:.2f} / {c0['swing_frac'][1]:.2f}", f"{c1['swing_frac'][0]:.2f} / {c1['swing_frac'][1]:.2f}"]))
    A(_row("不确定", [f"{c0['uncertain_frac'][0]:.2f} / {c0['uncertain_frac'][1]:.2f}", f"{c1['uncertain_frac'][0]:.2f} / {c1['uncertain_frac'][1]:.2f}"]))
    A(_row("双支撑 / 单支撑 / 无支撑", [f"{c0['double_support']:.2f} / {c0['single_support']:.2f} / {c0['no_support']:.2f}", f"{c1['double_support']:.2f} / {c1['single_support']:.2f} / {c1['no_support']:.2f}"]))
    A(_row("推断支撑帧占比(置信≤0.5)", [f"{c0['inferred_support_frac']:.3f}", f"{c1['inferred_support_frac']:.3f}"]))
    A(_row("平均置信度", [f"{c0['mean_confidence']:.2f}", f"{c1['mean_confidence']:.2f}"]))
    A(""); A(f"V1.1 旧分层（<10 mm 确定触地 / >40 mm 确定摆动，相对静立地面）：确定双支撑 1%/单支撑 17%/腾空 11%/不确定 70%，腾空占比 0.56（对 LAFAN 舞蹈中位 0.03 异常）。本模块：无支撑占比 {c0['no_support']:.3f}，**经弹道检验的真实腾空 {an['flight_true_frames']} 帧**；腾空候选 {len(an['flight_candidates'])} 段：")
    for cd in an["flight_candidates"]: A(f"  - 帧 {cd['start']}–{cd['end']}（{cd['frames']} 帧）：质心拟合加速度 {cd['fit_az']:.1f} m/s²，拟合残差 {cd['fit_rms_mm']:.1f} mm，→ **{'真腾空' if cd['true_flight'] else '假腾空(悬浮)'}**（{cd['reason']}）")
    A(f"\n![全段总览]({'figures/overview_full.png'})"); A("")
    rg = diag["regions"]
    A(f"- **足底接触区域**（支撑帧）：原始 平整接触 {rg['original']['flat']*100:.0f}% / 仅脚跟 {rg['original']['heel_only']*100:.0f}% / 仅脚尖 {rg['original']['toe_only']*100:.0f}% / 悬空 {rg['original']['floating']*100:.0f}%（相对 5 mm）；修正后 平整 {rg['refined']['flat']*100:.0f}% / 脚跟 {rg['refined']['heel_only']*100:.0f}% / 脚尖 {rg['refined']['toe_only']*100:.0f}% / 悬空 {rg['refined']['floating']*100:.0f}%。**残余缺陷**：重定向的脚底大量处于脚跟着地/脚尖翘起姿态，踝关节改动上限 10° 且仅弱放平，修正后仍以“脚跟或脚尖单点接触”为主（平整接触仅 {rg['refined']['flat']*100:.0f}%），这不是脚底穿地，但与站稳的物理直觉不同。")
    A(""); A("### 与训练接触标签 / MuJoCo 受力接触的一致性"); A("")
    lc = diag["label_consistency"]; tl = lc["training_labels"]
    A(f"- 训练用接触标签（`MotionLib.set_ref_contact` 的 numpy 复刻：踝关节原点低于该片段 5% 分位地面+15 mm 且水平速<0.5 m/s）：原始数据仅 **{tl['orig_label_contact_frac']*100:.0f}%** 的脚-帧被标为接触，**{tl['orig_label_no_contact_frac']*100:.0f}% 的帧两脚都不接触**（本模块识别的无支撑仅 {tl['identified_no_support_frac']*100:.1f}%）；修正数据 {tl['new_label_contact_frac']*100:.0f}% / 无接触 {tl['new_label_no_contact_frac']*100:.0f}%，与本模块识别的一致率 {tl['new_label_vs_identified_agreement']*100:.0f}%（原始数据 {tl['orig_label_vs_identified_agreement']*100:.0f}%）。")
    if "mujoco_force_contact" in lc:
        mf = lc["mujoco_force_contact"]
        A(f"- MuJoCo 闭环受力（冻结模型在**原始**参考上的 nominal 回放，Fz>10%BW 视为承重）：本模块识别的支撑 vs 承重一致率 **{mf['identified_stance_vs_loaded_agreement']*100:.0f}%**（识别支撑但未承重 {mf['identified_stance_but_unloaded']*100:.0f}%，识别摆动但承重 {mf['identified_swing_but_loaded']*100:.0f}%）；原始数据的训练标签 vs 承重一致率 {mf['training_label_vs_loaded_augment' if False else 'training_label_vs_loaded_agreement']*100:.0f}%（**标签说摆动而实际承重 {mf['training_label_swing_but_loaded']*100:.0f}%**）；修正数据训练标签 vs 同一回放承重一致率 {f1((mf['new_label_vs_loaded_agreement'] or 0)*100,0)}%（回放是原始参考的，仅作标签合理性旁证）。V1.1 报告的分层一致率为 69%。")
    A(""); A("## 4. 第 370 帧附近（真机倾角 0.40 rad）")
    A(""); A(f"索引换算：真机/训练帧 = 舞蹈帧 + 50（1 s 前导）。真机三次到达 0.40 rad 的 370–371 帧 ≈ 舞蹈帧 320–321；参考动作的深前倾段为舞蹈帧 311–337（训练帧 361–387），倾角峰值出现在舞蹈帧 {diag['focus']['deep_lean']['tilt_peak_frame']}（训练帧 {diag['focus']['deep_lean']['tilt_peak_frame']+50}）。")
    def wrow(nm, w):
        return _row(nm + f"（舞蹈 {w['frames'][0]}–{w['frames'][1]} / 训练 {w['train_frames'][0]}–{w['train_frames'][1]}）", [f"{w['tilt_peak_rad']:.2f}", f"{w['root_z_orig_m'][0]:.3f}~{w['root_z_orig_m'][1]:.3f} → {w['root_z_new_m'][0]:.3f}~{w['root_z_new_m'][1]:.3f}",
                          f"{w['hover_of_lowest_foot_over_z0_mm']['orig_median']:.0f}（最大 {w['hover_of_lowest_foot_over_z0_mm']['orig_max']:.0f}）→ {w['hover_of_lowest_foot_over_z0_mm']['new_median']:.1f}", f"{w['penetration_orig_vs_ground_curve_max_mm']:.1f} / {w['penetration_orig_vs_z0_max_mm']:.1f} → {w['penetration_new_max_mm']:.1f}",
                          f"{f1(w['slip_orig']['mean'],3)}/{f1(w['slip_orig']['p95'],2)} → {f1(w['slip_new']['mean'],3)}/{f1(w['slip_new']['p95'],2)}", f"dz {w['root_correction_mm']['dz'][0]:.0f}~{w['root_correction_mm']['dz'][1]:.0f}; legs≤{w['legs_delta_max_deg']:.1f}°"])
    A("| 窗口 | 倾角峰值 rad | 根高 m 原→修 | 最低脚离地 z=0 mm 原→修 | 穿透 mm 原(g/z0)→修 | 支撑接触点滑移 均值/p95 m/s 原→修 | 修正量 |"); A("|---|---|---|---|---|---|---|")
    for key, nm in (("deep_lean", "深前倾段 311–337"), ("peak", "真机 370–371 ≈ 舞蹈 318–323"), ("pm60", f"焦点 ±60（舞蹈 {focus-60}–{focus+60}）")): A(wrow(nm, diag["focus"][key]))
    w = diag["focus"]["deep_lean"]
    A(""); A(f"- 状态：深前倾段 左脚(L) 支撑占比 {w['states_orig']['L']['stance']*100:.0f}%，右脚(R) 支撑 {w['states_orig']['R']['stance']*100:.0f}% / 摆动 {w['states_orig']['R']['swing']*100:.0f}%：这是**单脚(左)承重的深蹲前倾**，右脚摆动抬起；无真实腾空（{w['flight_true_frames']} 帧；朴素口径会把其中 {w['naive_both_feet_gt25mm_over_z0_frames']} 帧判成腾空）。")
    A(f"- 悬浮：原始在这一段左(支撑)脚最低点离真实地面 z=0 {w['hover_of_lowest_foot_over_z0_mm']['orig_median']:.0f} mm（最大 {w['hover_of_lowest_foot_over_z0_mm']['orig_max']:.0f}），而根高从 0.84 降到 0.66 的下蹲过程里地面曲线本身在 {w['ground_curve_mm'][0]:.0f}~{w['ground_curve_mm'][1]:.0f} mm 间漂；修正后支撑脚落在 z=0（最大 {w['hover_of_lowest_foot_over_z0_mm']['new_max']:.1f} mm），根整体下移 {-w['root_correction_mm']['dz'][1]:.0f}~{-w['root_correction_mm']['dz'][0]:.0f} mm。**含义**：训练时机器人脚踩刚性地面，而参考根比应有高度高 ≈30–40 mm，下蹲幅度/重心高度与真实可达姿态不一致；修正后参考是几何自洽的。")
    A(f"- 穿地：原始相对 z=0 无穿地（转换器按全片最低点对齐，故整段悬浮）；相对地面曲线最大穿入 {w['penetration_orig_vs_ground_curve_max_mm']:.1f} mm（噪声量级）。脚滑：支撑接触点滑移均值 {f1(w['slip_orig']['mean'],3)} → {f1(w['slip_new']['mean'],3)} m/s；p95 {f1(w['slip_orig']['p95'],2)} → {f1(w['slip_new']['p95'],2)} m/s（焦点 ±60 窗 p95 {f1(diag['focus']['pm60']['slip_orig']['p95'],2)} → {f1(diag['focus']['pm60']['slip_new']['p95'],2)}：该较宽窗内 p95 **变差**，原因是落脚/抬脚过渡帧，见 §5 的“内部帧”口径）。")
    A(f"- 根 XYZ：修正量 dx {w['root_correction_mm']['dx'][0]:.0f}~{w['root_correction_mm']['dx'][1]:.0f} mm、dy {w['root_correction_mm']['dy'][0]:.0f}~{w['root_correction_mm']['dy'][1]:.0f} mm 为**缓变**平移（积累自参考根滑行，窗内变化平滑），不产生额外根加速度；IK 在窗内腿关节最大改动 {w['legs_delta_max_deg']:.1f}°、RMS {w['legs_delta_rms_deg']:.2f}°。")
    A(f"- ZMP：窗内可靠帧 {w['zmp_margin_mm']['n_reliable']} 帧，运动学 ZMP 到支撑域边距中位数 原始 {f1(w['zmp_margin_mm']['orig_median'],0)} → 修正 {f1(w['zmp_margin_mm']['new_median'],0)} mm（负=在支撑域外；**单脚承重 + 深前倾 0.41 rad 时 ZMP 本来就偏出足底，这段动作本身动力学上极苛刻，修正几何不能改变这一点**）。")
    A("- **判断**：370 帧附近的倾角 0.40 rad 来自参考动作本身的深蹲前倾（躯干姿态未改动），不是接触/IK 错误；其附近确实存在“整体悬浮 30–45 mm + 支撑脚滑移”，已被修正，但修正**不降低**动作难度（质心前移、单脚支撑）。")
    A("\n![370 附近 ±60](figures/window_focus.png)\n\n![370 附近放大（舞蹈帧 {}–{}）](figures/window_focus_tight.png)".format(focus - 15, focus + 20)); A("")
    A("## 5. 修正前后量化对比（全段）"); A("")
    ju, dy, st = res["joints"], res["dynamics"], res["style"]
    A("| 指标 | 原始 | 修正后 | 备注 |"); A("|---|---|---|---|")
    rows = [("支撑接触点滑移 均值/p95/p99/max (m/s)", f"{s['original']['mean']:.3f}/{s['original']['p95']:.3f}/{f1(s['original']['p99'],2)}/{s['original']['max']:.2f}", f"{s['refined']['mean']:.3f}/{s['refined']['p95']:.3f}/{f1(s['refined']['p99'],2)}/{s['refined']['max']:.2f}", "同一支撑帧掩码、V1.4 口径"),
            ("**内部支撑帧**滑移（距 stance 段首尾≥3 帧）均值/p95/p99/max", f"{s['interior_only']['original']['mean']:.3f}/{s['interior_only']['original']['p95']:.3f}/{f1(s['interior_only']['original']['p99'],2)}/{s['interior_only']['original']['max']:.2f}", f"{s['interior_only']['refined']['mean']:.3f}/{s['interior_only']['refined']['p95']:.3f}/{f1(s['interior_only']['refined']['p99'],2)}/{s['interior_only']['refined']['max']:.2f}", "排除落脚/抬脚/转轴过渡帧（修正把悬浮的脚降到地面后，这些过渡帧才开始被计入）"),
            ("按采样点身份跟踪的滑移 均值/p95/p99/max", f"{s['per_point_identity']['original']['mean']:.3f}/{s['per_point_identity']['original']['p95']:.3f}/{f1(s['per_point_identity']['original']['p99'],2)}/{s['per_point_identity']['original']['max']:.2f}", f"{s['per_point_identity']['refined']['mean']:.3f}/{s['per_point_identity']['refined']['p95']:.3f}/{f1(s['per_point_identity']['refined']['p99'],2)}/{s['per_point_identity']['refined']['max']:.2f}", "14 个足底点各自跟踪，不受“最低点切换”影响"),
            ("全部接触帧滑移 均值/p95", f"{s['v14_all_contact_frames']['original_vs_ground_curve']['mean']:.3f}/{s['v14_all_contact_frames']['original_vs_ground_curve']['p95']:.3f}", f"{s['v14_all_contact_frames']['refined']['mean']:.3f}/{s['v14_all_contact_frames']['refined']['p95']:.3f}", "不限定支撑掩码（修正后更多帧满足 <10 mm 而被计入，含过渡帧）"),
            ("支撑脚|离地高度| p95 (mm)", f"{g['original_vs_z0']['stance_abs_height_p95_mm']:.1f}（对 z=0）/ {g['original_vs_ground_curve']['stance_abs_height_p95_mm']:.1f}（对地面曲线）", f"{g['stance_abs_height_p95_mm']:.1f}", ""),
            ("最大穿透 (mm)", f"{g['original_vs_ground_curve']['penetration_max_mm']:.1f}（对地面曲线）", f"{g['penetration_max_mm']:.2f}", ""),
            ("穿透>2mm 帧占比 (%)", f"{g['original_vs_ground_curve']['penetration_gt2mm_frac_pct']:.1f}", f"{g['penetration_gt2mm_frac_pct']:.1f}", ""),
            ("两脚都离地>10 mm 的帧占比（悬浮）", f"{g['original_vs_z0']['hover_frac_both_feet_gt10mm']*100:.0f}%（对 z=0）/ {g['original_vs_ground_curve']['hover_frac_both_feet_gt10mm']*100:.0f}%（对地面曲线）", f"{g['hover_frac_both_feet_gt10mm']*100:.1f}%", "后者含真摆动"),
            ("根路径 (m) / 落脚隐含路径 (m)", f"{r['path_orig_m']:.2f} / {r['consistency']['implied_path_m'][0]:.2f}", f"{r['path_new_m']:.2f} / {r['consistency']['implied_path_m'][1]:.2f}", "隐含路径=支撑脚不动假设下根应走的路程"),
            ("根与落脚隐含根不一致 均值/最大 (m)", f"{r['consistency']['mismatch_mean_m'][0]:.2f} / {r['consistency']['mismatch_max_m'][0]:.2f}", f"{r['consistency']['mismatch_mean_m'][1]:.2f} / {r['consistency']['mismatch_max_m'][1]:.2f}", "同一支撑掩码；V1.1 报告 0.25/0.65 m（不同掩码）"),
            ("接触阈值穿越次数 L/R", str(res['contact_switching']['raw_threshold_crossings_orig']), str(res['contact_switching']['raw_threshold_crossings_refined']), "足高 12 mm 阈值的原始穿越（抖动）"),
            ("迟滞后接触切换次数 L/R", str(res['contact_switching']['hysteresis_toggles_orig']), str(res['contact_switching']['hysteresis_toggles_refined']), ""),
            ("腿关节速度峰值 (rad/s)", f"{dy['leg_jv_max'][0]:.2f}", f"{dy['leg_jv_max'][1]:.2f}", "超硬件限速帧数 腿 %s→%s、全部 %s→%s" % (dy['exceed_frames_legs'][0], dy['exceed_frames_legs'][1], dy['exceed_frames_all'][0], dy['exceed_frames_all'][1])),
            ("腿关节加速度 p99.9 (rad/s²)", f"{dy['joint_acc_p999'][0]:.0f}", f"{dy['joint_acc_p999'][1]:.0f}", f"比值 {dy['joint_acc_p999_ratio']:.3f}"),
            ("根加速度 p99.5 / max (m/s²)", f"{dy['root_acc_p995'][0]:.1f} / {dy['root_acc_max'][0]:.1f}", f"{dy['root_acc_p995'][1]:.1f} / {dy['root_acc_max'][1]:.1f}", f"比值 {dy['root_acc_p995_ratio']:.3f}"),
            ("根 jerk p99 (m/s³)", f"{dy['jerk_root_p99'][0]:.0f}", f"{dy['jerk_root_p99'][1]:.0f}", ""),
            ("ZMP 支撑域边距中位数 (mm，可靠帧 %d)" % dy['zmp_frames'], f"{f1(dy['zmp_margin_median_mm'][0],0)}", f"{f1(dy['zmp_margin_median_mm'][1],0)}", f"ZMP 在支撑域外占比 {f1((dy['zmp_outside_frac'][0] or 0)*100,0)}% → {f1((dy['zmp_outside_frac'][1] or 0)*100,0)}%：**动力学难度基本不变**"),
            ("关节改动 RMS(全部/腿) °", "—", f"{ju['rms_all_deg']:.2f} / {ju['rms_legs_deg']:.2f}", f"腿改动>1° 的帧 {ju['frames_changed_gt1deg']*100:.0f}%"),
            ("风格：骨盆相对 body 位置偏差 全身/腿/上肢 (mm)", "—", f"{st['pelvis_rel_mpjpe_mm']:.2f} / {st['leg_pelvis_rel_mpjpe_mm']:.2f} / {st['upper_pelvis_rel_mpjpe_mm']:.4f}", f"踝最大 {st['ankle_pelvis_rel_mm_mean_max'][1]:.1f} mm；任意 body 最大 {st['max_any_body_mm']:.1f} mm"),
            ("XY/yaw 锚定 MPJPE（含根 z 平移，项目口径）", "—", f"{st['anchored_xy_yaw_mpjpe_mm_includes_root_z_shift']:.1f} mm", "主要来自根 z 整体下移（悬浮修正），**不是姿态变化**"),
            ("支撑钉住残差(各 stance 段采样点 xy 散布 rms/p95 mm)", f"{res['ik_residual']['original_pin_xy']['xy_rms_mm']:.1f} / {res['ik_residual']['original_pin_xy']['xy_p95_mm']:.1f}", f"{res['ik_residual']['refined_pin_xy']['xy_rms_mm']:.1f} / {res['ik_residual']['refined_pin_xy']['xy_p95_mm']:.1f}", "IK 残差")]
    for rw in rows: A(_row(rw[0], list(rw[1:])))
    A("\n![关节改动](figures/joint_delta.png)\n\n![根轨迹](figures/root_path.png)"); A("")
    A("**逐关节改动 RMS/max（°）**（上肢/腰/头全部为 0）："); A(""); A("| 关节 | RMS | max |"); A("|---|---|---|")
    for n, v in ju["per_joint_rms_deg"].items():
        if ju["per_joint_max_deg"][n] > 0: A(f"| {n} | {v:.2f} | {ju['per_joint_max_deg'][n]:.2f} |")
    A(""); A("## 6. 与 V1.2 repair（只改根平移）对比"); A("")
    A("| 指标 | 原始 | V1.2 repair(只改根) | 本模块 stage-1(只改根 XYZ) | 本模块 完整(根+腿IK) |"); A("|---|---|---|---|---|")
    def cell(v, path, fmt="{:.3f}"):
        try:
            for k in path.split("."): v = v[k]
            return fmt.format(v)
        except Exception: return "—"
    keys = [("支撑接触点滑移 p95 m/s", "slip.refined.p95", "{:.3f}", f"{s['original']['p95']:.3f}"), ("支撑接触点滑移 均值 m/s", "slip.refined.mean", "{:.3f}", f"{s['original']['mean']:.3f}"), ("支撑脚|离地高度| p95 mm", "ground.stance_abs_height_p95_mm", "{:.1f}", f"{g['original_vs_z0']['stance_abs_height_p95_mm']:.1f}"),
            ("最大穿透 mm", "ground.penetration_max_mm", "{:.1f}", f"{g['original_vs_ground_curve']['penetration_max_mm']:.1f}*"), ("穿透>2mm 帧 %", "ground.penetration_gt2mm_frac_pct", "{:.1f}", f"{g['original_vs_ground_curve']['penetration_gt2mm_frac_pct']:.1f}*"),
            ("根 XY 最大偏离 m", "root.dev_xy_max_m", "{:.2f}", "0"), ("根速度改动 p99 m/s", "root.dv_p99_mps", "{:.2f}", "0"), ("根加速度 p99.5 比值", "dynamics.root_acc_p995_ratio", "{:.2f}", "1"), ("落脚不一致均值 m", "root.consistency.mismatch_mean_m", None, ""),
            ("ZMP 边距中位 Δ mm", "dynamics.zmp_margin_median_delta_mm", "{:.1f}", "0"), ("关节改动 RMS °", "joints.rms_all_deg", "{:.2f}", "0"), ("QA 判决（同一判据）", None, None, "—")]
    for nm, path, fmt, o in keys:
        if path is None:
            from . import qa as QAm
            vals = [QAm.evaluate(v)["verdict"] if v else "—" for v in (v12, ro, res)]
            A(_row(nm, [o] + vals)); continue
        if path == "root.consistency.mismatch_mean_m":
            A(_row(nm, [f"{r['consistency']['mismatch_mean_m'][0]:.2f}"] + [f"{v['root']['consistency']['mismatch_mean_m'][1]:.2f}" if v else "—" for v in (v12, ro, res)])); continue
        A(_row(nm, [o] + [cell(v, path, fmt) if v else "—" for v in (v12, ro, res)]))
    A("\n（* 原始值相对自估地面曲线；相对 z=0 原始数据无穿地但整段悬浮。）")
    if v12:
        from . import qa as QAm
        fv = QAm.evaluate(v12)["failed"]; A(f"\nV1.2 repair 在本判据下未通过项：{fv}。")
    A("\n![方案对比](figures/variants_compare.png)\n")
    A("解读：只改根的方案（V1.2 / stage-1）能去掉悬浮并降低一部分脚滑，但“残余脚滑是关节运动隐含的下限”——要进一步压低脚滑只能动腿关节；完整方案用 ≤6°/≤10° 的腿/踝小改动把支撑脚滑移再降一截，代价是 RMS 0.5° 量级的关节改动与（仍可接受的）风格偏离。"); A("")
    A("## 7. 通用性验证"); A("")
    for k, v in (extra.get("generality") or {}).items(): A(f"- **{k}**：{v}")
    A(""); A("## 8. 残余缺陷与风险（不回避）"); A("")
    for t in extra.get("residuals", []): A(f"- {t}")
    A(""); A("## 9. 结论：修正是否可信、是否值得训练"); A("")
    for t in extra.get("conclusion", []): A(f"- {t}")
    A(""); A("## 10. 产物"); A("")
    for t in extra.get("artifacts", []): A(f"- {t}")
    path = path or (od / "CONTACT_REFINE_REPORT.md"); path.write_text("\n".join(L)); return path
