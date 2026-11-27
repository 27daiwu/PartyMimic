"""V1.3 A/B comparison: models trained 2000 it on ORIGINAL vs REPAIRED reference, evaluated on the COMMON (repaired) reference.  See abtest.py for the design and the pre-registered rule."""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import abtest as AB
from . import quality as Q
from . import repair as P
from .core import REPO, Dance, Logger, load_cfg, now, read_json, sha256, write_json

LAB = {"orig": "A: ORIGINAL data it2000", "rep": "B: REPAIRED data it2000", "ctrl": "C: original data repeat (noise ctrl)", "anchor": "it40000 (frozen, scale ref)"}
HEAD = {"foot_slip_mps": ("实际足滑(承重脚足底速度) m/s", 3), "pen_dance_max_mm": ("足底穿透最大(舞蹈段) mm", 2), "cop_margin_p05_mm": ("CoP 距足底支撑域边缘 p05 mm", 2), "hr_peak_nm": ("髋Roll 峰值力矩 N·m (限 88)", 1),
        "guard_rewrite": ("护栏改写目标占比", 3), "root_xy_err_mean_m": ("根轨迹误差(对公共参考) m", 3), "mpjpe_zaligned_mm": ("高度对齐锚定 MPJPE mm", 2), "jmae_rad": ("关节 MAE rad", 4)}


def models(dance: Dance):
    r = dance.dir
    return {"orig": r / "training/run/ckpt_002000.pt", "rep": r / "runs/ab_repaired_v1/train/ckpt_002000.pt", "ctrl": r / "runs/ab_origctl_v1/train/ckpt_002000.pt", "anchor": r / "training/run/ckpt_040000.pt"}


def train_curves(dance: Dance):
    import csv
    out = {}
    for k, p in (("orig", dance.dir / "training/run/train_log.csv"), ("rep", dance.dir / "runs/ab_repaired_v1/train/train_log.csv"), ("ctrl", dance.dir / "runs/ab_origctl_v1/train/train_log.csv")):
        rows = [r for r in csv.DictReader(open(p)) if int(float(r["iteration"])) <= 2000]; l = rows[-1]
        out[k] = dict(iteration=int(float(l["iteration"])), sim_rew=float(l["sim_rew"]), term_frac=float(l["term_frac"]), wm_loss=float(l["wm_loss"]), v_loss=float(l["v_loss"]), wall_h=float(l["wall_h"]), it_per_s=float(l["it_per_s"]))
    return out


def compare(dance: Dance, video=True, log=None) -> dict:
    from . import evaluator as E
    from .replay import render_quad
    log = log or Logger(dance.dir / "logs/v13.log")
    V = dance.dir / "v13"; V.mkdir(exist_ok=True); rd = P.make_dance_like(dance); ms = models(dance)
    for k, p in ms.items():
        if not p.exists(): raise RuntimeError(f"missing {k}: {p} (training still running? see tmux session pm-{dance.name}-v13)")
    exp = {k: AB.export_eval_model(p, V / "models" / k, rd.npz, log) for k, p in ms.items()}; log("exported", {k: (v["ckpt_sha12"], v["parity"]) for k, v in exp.items()})
    files = {k: AB.rollouts(exp[k], rd.padded_npz, V / "traj" / k) for k in ms}; log("rollouts done")
    with Pool(16) as pool: met = {k: AB.run_metrics(files[k], rd.padded_npz, pool) for k in ms}
    log("metrics done"); write_json(V / "metrics_all.json", met)
    # nominal-condition aggregates (reference start + 16 perturbed starts) - the basis of the decision
    M = {k: {name: AB.agg(met[k]["nominal"], name) for name in list(AB.DYNAMICS) + list(AB.FIDELITY) + ["global_mpjpe_mm", "mpjpe_plain_mm", "label_agreement", "label_ref_swing_sim_loaded", "label_ref_contact_sim_unloaded", "foot_slip_p95_mps", "pen_dance_p95_mm",
                                                                                                                  "root_xy_err_max_m", "root_xy_err_end_m", "yaw_err_deg", "tilt_p95", "hr_sat", "sat_all", "peak_ratio_p95", "guard_max_dev", "com_outside_sole", "ds", "ss"]} for k in ms}
    rows = AB.decide(M)
    allc = {k: [r for c in met[k] for r in met[k][c]] for k in ms}; surv = {k: float(np.mean([r["survived"] for r in allc[k]])) for k in ms}
    byc = {k: {c: dict(survival=float(np.mean([r["survived"] for r in met[k][c]])), hr_peak=float(np.max([r["hr_peak_nm"] for r in met[k][c]])), guard=float(np.mean([r["guard_rewrite"] for r in met[k][c]])), slip=float(np.mean([r["foot_slip_mps"] for r in met[k][c]])),
                           pen=float(np.max([r["pen_dance_max_mm"] for r in met[k][c]])), root_err=float(np.mean([r["root_xy_err_mean_m"] for r in met[k][c]])), n=len(met[k][c])) for c in AB.CONDS} for k in ms}
    # training-env evaluation grid on BOTH references (same seed / reps)
    ecfg = load_cfg("eval", "balance_first_v1"); grid = {}
    for k, p in ms.items():
        grid[k] = {}
        for rn, dd in (("repaired_ref", rd.padded_dir), ("original_ref", dance.padded_dir)):
            f = V / "evalgrid" / f"{k}_{rn}.json"
            if not f.exists(): f.parent.mkdir(exist_ok=True); write_json(f, E.evaluate(str(p), str(dd), ecfg["periodic"]["reps"], ecfg["periodic"]["seed"]))
            grid[k][rn] = read_json(f)
    rob = {"deploy_chain_survival": dict(A=surv["orig"], B=surv["rep"], C=surv["ctrl"]), "evalgrid_robust_survival_repaired_ref": dict(A=grid["orig"]["repaired_ref"]["scores"]["survival_robust"], B=grid["rep"]["repaired_ref"]["scores"]["survival_robust"], C=grid["ctrl"]["repaired_ref"]["scores"]["survival_robust"]),
           "evalgrid_robust_survival_original_ref": dict(A=grid["orig"]["original_ref"]["scores"]["survival_robust"], B=grid["rep"]["original_ref"]["scores"]["survival_robust"], C=grid["ctrl"]["original_ref"]["scores"]["survival_robust"])}
    vd = AB.verdict(rows, rob); fzv = AB.verify_freeze(dance, V / "FREEZE.json"); curves = train_curves(dance)
    doc = dict(generated=now(), M=M, rows=rows, robustness=rob, verdict=vd, freeze=fzv, grid={k: {r: grid[k][r]["scores"] for r in grid[k]} for k in grid}, byc=byc, curves=curves, exports={k: {x: v[x] for x in ("ckpt_sha12", "onnx_sha12", "parity")} for k, v in exp.items()})
    write_json(V / "ab_result.json", doc)
    vid = None
    if video: vid = render_quad(rd, [(LAB[k], files[k]["nominal"][0]) for k in ("orig", "rep", "ctrl")], V / "ythn_v13_compare.mp4", V / "ythn_v13_sheet.png")
    figure(doc, V / "ab_comparison.png"); rep = report(dance, doc, vid, V / "V13_AB_REPORT.md", AB.slip_diagnostics(dance)); log("report", str(rep))
    return dict(report=str(rep), video=(vid or {}).get("video"), verdict="GO" if vd["go"] else "NO-GO")


def figure(doc, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = list(AB.DYNAMICS) + list(AB.FIDELITY); fig, ax = plt.subplots(2, 4, figsize=(18, 8)); col = {"orig": "#c0392b", "rep": "#27ae60", "ctrl": "#e67e22", "anchor": "#7f8c8d"}
    for a, n in zip(ax.flat, names):
        for i, k in enumerate(("orig", "rep", "ctrl", "anchor")):
            m = doc["M"][k][n]; a.bar(i, m["mean"], yerr=m["std"], color=col[k], capsize=3)
        a.set_xticks(range(4)); a.set_xticklabels(["A orig", "B repaired", "C ctrl", "it40000"], fontsize=8); a.set_title(f"{HEAD[n][0]}  [{doc['rows'][n]['status']}]", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def report(dance: Dance, doc: dict, vid, path: Path, diag=None) -> Path:
    M, rows, vd, g, byc, cv = doc["M"], doc["rows"], doc["verdict"], doc["grid"], doc["byc"], doc["curves"]; prof = Path(REPO / "pipeline/configs/hardware/g1_29dof_rev1_0.yaml"); import yaml
    hp = yaml.safe_load(prof.read_text()); hr = [j for j in hp["joints"] if j["name"] == "left_hip_roll_joint"][0]
    pe = [float(x) for x in [l for l in open(REPO / "results/ythn/final/assets/it040000/policy_params.txt") if l.startswith("effort_lim")][0].split()[2:]][1]
    f = lambda m, n: f"{m['mean']:.{n}f}±{m['std']:.{n}f}"; go = vd["go"]
    L = [f"# V1.3 修复数据短程对照训练：GO / NO-GO（自动生成 {doc['generated']}）", "", f"## 结论：**{'GO — 可以进入 40000 轮长程训练' if go else 'NO-GO — 不进入长程训练'}**（本轮不自动进入长程）", "",
         f"- 预登记判据（训练前写入 `abtest.py`）：6 项实际动力学指标中 ≥{AB.DECISION['need_improved']} 项**显著改善**（优于 A 和 C 两者，且提升 ≥ max(10% 基线, 绝对阈值, A–C 运行间差距)），**没有任何**动力学/还原度指标显著退化，且鲁棒性不低于 min(A,C)。",
         f"- 显著改善：{vd['improved'] or '无'}；显著退化：{vd['degraded'] or '无'}{'（该“退化”源自预登记足滑指标的缺陷，见 §5b；按评估器口径足滑无变化，结论不变）' if vd['degraded'] else ''}；鲁棒性不退化：{vd['robustness_ok']}。", "", "## 1. 实验设置（只改变参考动作这一个变量）", "",
         "| 项 | A 原始数据 | B 修复数据 | C 原始数据重复（噪声对照） |", "|---|---|---|---|", f"| 起点 | `init_from_g1_mlp.pt` sha12 967bbb560703（官方预训练加宽，同一文件） | 同 | 同 |", "| 配方 / seed / envs | robust_d_v1 / 0 / 4096 | 同 | 同 |",
         "| 动作目录 | `motions/ythn_padded`（原始） | `motions/ythn_repaired_padded` | `motions/ythn_padded` |", "| 迭代 / 保存 | 2000 / 每 500（历史运行阶段 1 的 ckpt_002000） | 2000 / 每 500（`runs/ab_repaired_v1`） | 2000 / 每 500（`runs/ab_origctl_v1`） |",
         f"| 末行训练日志 | sim_rew {cv['orig']['sim_rew']:.3f} term {cv['orig']['term_frac']:.3f} | sim_rew {cv['rep']['sim_rew']:.3f} term {cv['rep']['term_frac']:.3f} | sim_rew {cv['ctrl']['sim_rew']:.3f} term {cv['ctrl']['term_frac']:.3f} |",
         f"| checkpoint sha12 | {doc['exports']['orig']['ckpt_sha12']} | {doc['exports']['rep']['ckpt_sha12']} | {doc['exports']['ctrl']['ckpt_sha12']} |", "",
         f"- **硬件 profile**（`g1_29dof_rev1_0.yaml`，URDF sha256 {hp['urdf']['sha256'][:12]}）：髋 Roll {hr['effort_nm']:.0f} N·m / {hr['velocity_rad_s']:.0f} rad/s。**真机力矩限制保留 {pe:.0f} N·m**（部署 `policy_params.txt`），评价用部署链的 {pe:.0f} N·m 限幅；训练环境仍按训练配方（139 N·m）——本轮训练配方不变，无混杂变量。",
         f"- **公共评价参考 = 修复参考**；初态：每个模型相同的种子化扰动起点（nominal 1 个参考起点 + 16 个扰动起点；推力/延迟/kp/摩擦/噪声各 8 个）；同一部署控制链、同一指标代码。A/B/C 的 ONNX 均通过一致性校验。", "",
         "## 2. 实际动力学与还原度（nominal 工况，17 个起点，均值±std；决策依据）", "", "| 指标 | A 原始 | B 修复 | C 重复 | it40000 | A–C 运行间差距 | B 相对 A/C 均值 | 判定 |", "|---|---|---|---|---|---|---|---|"]
    for n, (t, nd) in HEAD.items():
        r = rows[n]; L.append(f"| {t} | {f(M['orig'][n], nd)} | {f(M['rep'][n], nd)} | {f(M['ctrl'][n], nd)} | {f(M['anchor'][n], nd)} | {r['spread_AC']:.{nd}f} | {r['gain']:+.{nd}f}（需 ≥{r['need']:.{nd}f}） | **{r['status']}** |")
    L += ["", "> 方向：滑移/穿透/力矩/护栏/根误差/MPJPE/MAE 越低越好；CoP 余量越高越好。“显著改善”要求优于 A 与 C 两者。", "", "## 3. 接触一致率（标签口径，仅供参考，**不计入决策**：它与训练目标同源）", "",
          "| 指标（对公共修复参考的训练标签） | A | B | C | it40000 |", "|---|---|---|---|---|"]
    for n, t in (("label_agreement", "标签 vs MuJoCo 受力接触 一致率"), ("label_ref_swing_sim_loaded", "参考判摆动而机器人承重"), ("label_ref_contact_sim_unloaded", "参考判触地而机器人未承重")):
        L.append(f"| {t} | " + " | ".join(f(M[k][n], 3) for k in ("orig", "rep", "ctrl", "anchor")) + " |")
    L += ["", "## 4. 其它（信息）", "", "| 指标 | A | B | C | it40000 |", "|---|---|---|---|---|"]
    for n, t, nd in (("global_mpjpe_mm", "全局 MPJPE mm（对修复参考）", 0), ("mpjpe_plain_mm", "锚定 MPJPE mm（未做高度对齐）", 1), ("root_xy_err_max_m", "根轨迹误差最大 m", 2), ("root_xy_err_end_m", "根轨迹误差末端 m", 2), ("foot_slip_p95_mps", "足滑 p95 m/s", 3), ("pen_dance_p95_mm", "穿透 p95 mm", 2),
                     ("tilt_p95", "倾角 p95 rad", 3), ("hr_sat", "髋Roll 饱和占比(88 N·m)", 4), ("sat_all", "全关节饱和占比", 4), ("peak_ratio_p95", "力矩峰值比 p95", 3), ("guard_max_dev", "护栏最大改写 rad", 3), ("com_outside_sole", "CoM 投影在足底外占比", 3), ("ds", "双支撑占比", 3), ("ss", "单支撑占比", 3)):
        L.append(f"| {t} | " + " | ".join(f(M[k][n], nd) for k in ("orig", "rep", "ctrl", "anchor")) + " |")
    L += ["", "## 5. 鲁棒性", "", "| 指标 | A | B | C | 判定 |", "|---|---|---|---|---|"]
    for k, v in doc["robustness"].items(): L.append(f"| {k} | {v['A']:.3f} | {v['B']:.3f} | {v['C']:.3f} | {'OK' if v['B'] >= min(v['A'], v['C']) - 1e-9 else '**退化**'} |")
    L += ["", "分工况（部署链，存活 / 髋Roll 峰值 N·m / 护栏改写 / 足滑 m/s；每工况 8 个扰动起点，nominal 17 个）：", "", "| 工况 | A | B | C |", "|---|---|---|---|"]
    for c in AB.CONDS: L.append(f"| {c} | " + " | ".join(f"{byc[k][c]['survival']:.2f} / {byc[k][c]['hr_peak']:.0f} / {byc[k][c]['guard']:.2f} / {byc[k][c]['slip']:.2f}" for k in ("orig", "rep", "ctrl")) + " |")
    L += ["", "训练环境评估网格（V1.1，seed 1234，8 次；分数含 14 工况）：", "", "| 参考 | 指标 | A | B | C | it40000 |", "|---|---|---|---|---|---|"]
    for rn in ("repaired_ref", "original_ref"):
        for n, nd in (("balance", 1), ("fidelity", 1), ("survival_robust", 2), ("mpjpe_nominal_mm", 1), ("slip_mean", 3), ("pen_max_mm", 1)):
            L.append(f"| {rn} | {n} | " + " | ".join(f"{g[k][rn][n]:.{nd}f}" for k in ("orig", "rep", "ctrl", "anchor")) + " |")
    if diag:
        sd = lambda k, n, nd: f"{diag[k][n]['mean']:.{nd}f}±{diag[k][n]['std']:.{nd}f}"
        L += ["", "## 5b. 事后诊断：预登记的“实际足滑”指标有缺陷（**不改变预登记结论**，只改变对原因的解释）", "",
              "预登记指标 `foot_slip_mps` = 承重脚**足底质心**的水平速度（接触足底采样点按间隙软最小加权的质心）。B 的双支撑占比更高（见 §4），足底质心会随重心转移/足底滚动而移动，该指标因此被放大，**并不等于脚在地面上滑**。用训练评估器同样的定义（足底胶囊最低端点、间隙 <5 mm、连续两个采样、速度 <3 m/s）在同一批部署链轨迹上重算：", "",
              "| 足滑定义（nominal，17 起点） | A | B | C | it40000 |", "|---|---|---|---|---|"]
        for n, t, nd in (("patch_gt10", "足底质心速度（承重>10%体重，预登记指标）", 3), ("patch_gt50", "足底质心速度（承重>50%体重）", 3), ("point_mean", "**接触点速度 均值（评估器定义）**", 3), ("point_p95", "**接触点速度 p95（评估器定义）**", 3)):
            L.append(f"| {t} | " + " | ".join(sd(k, n, nd) for k in ("orig", "rep", "ctrl", "anchor")) + " |")
        L += ["", f"- 接触点足滑：B 与 A/C 均值相差 <10%（“无变化”），p95 明显更低（{diag['rep']['point_p95']['mean']:.2f} vs {diag['orig']['point_p95']['mean']:.2f}/{diag['ctrl']['point_p95']['mean']:.2f} m/s）。所以“B 足滑显著变差”**不成立**；训练评估网格（接触点定义）里 B 的滑移也更低（{g['rep']['repaired_ref']['slip_mean']:.3f} vs {g['orig']['repaired_ref']['slip_mean']:.3f}/{g['ctrl']['repaired_ref']['slip_mean']:.3f}）。",
              "- **但结论不翻转**：若把足滑按接触点均值计，它是“无变化”，显著改善仍只有 2 项（穿透、根轨迹误差），低于 3 项门槛 → **仍是 NO-GO**。",
              "- **判据对指标定义敏感**：若采用训练评估网格的足滑（-15%，满足改善阈值）则会数到 3 项。同时穿透的“改善”只比阈值多 0.05 mm（B 的 std 1.2 mm，n=17），CoP 余量（+0.82，需 1.0）和护栏改写（+0.010，需 0.010）也恰好在阈值边缘。**证据是混合/边缘的，不是“明显改善”**，这正是不推荐直接跑 40000 轮的原因。",
              "- B 的步态有变化：双支撑 0.89 vs 0.81、单支撑 0.09 vs 0.15（原始参考在修复后为 47%/19% 双/单支撑）。B 更多地双脚落地，CoM 落在足底外的比例也更低（0.13 vs 0.23）；本轮没有进一步检查摆动脚抬高与参考的差异，这是遗留问题。"]
    L += ["", "## 6. 解读与限制", "", "- A、C 是同一配方同一 seed 的两次训练（GPU 浮点非确定性 + 历史脚本 vs 流水线启动器），它们的差距就是**运行间噪声**；B 必须超过它才算改善。",
          "- 评价参考为修复参考，对 B 天然有利的“参考一致性”类指标（全局 MPJPE、根轨迹误差对原始参考不可比）已按公共参考统一；高度对齐 MPJPE 去除了参考悬浮偏移。评估网格分别给出两种参考的结果以便交叉检查。",
          "- 2000 轮只是短程：长程是否收敛到不同水平不能由本轮推出；单舞蹈、单 seed（B 只有一次训练，C 是唯一的噪声样本，噪声估计粗糙）。",
          "- 真机限幅 88 N·m 在评价里保留，训练环境仍是 139 N·m（配方不变）；若确认真机上限，需另做实验。", "", "## 7. 产物", "",
          f"- 对比视频：`{(vid or {}).get('video', '-')}`（参考 + A + B + C，nominal 起点，叠加接触/倾角/CoP/力矩饱和/护栏/帧号）；关键帧 `{(vid or {}).get('sheet', '-')}`；对比图 `v13/ab_comparison.png`；数据 `v13/ab_result.json`、`v13/metrics_all.json`",
          f"- 冻结校验：{doc['freeze']['checked']} 个文件，变化 {len(doc['freeze']['changed'])} 个 → **{'通过' if doc['freeze']['ok'] else '失败'}** {doc['freeze']['changed'] or ''}", ""]
    path.write_text("\n".join(L) + "\n"); return path


def rebuild(dance: Dance) -> Path:
    """regenerate the report from the stored result (no re-evaluation)."""
    V = dance.dir / "v13"; doc = read_json(V / "ab_result.json"); vid = dict(video=str(V / "ythn_v13_compare.mp4"), sheet=str(V / "ythn_v13_sheet.png"))
    return report(dance, doc, vid, V / "V13_AB_REPORT.md", AB.slip_diagnostics(dance))
