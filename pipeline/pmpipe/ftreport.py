"""V1.6 report writer for `./pm ftcompare` (figures + FT_COMPARE_REPORT.md).  Numbers come from ftcompare.compare(); nothing is re-judged here."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .core import Dance, now, read_json
from . import ftcompare as FC

GRP_COL = {"A": "#c0392b", "B": "#27ae60", "anchor": "#7f8c8d"}


def _train_rows(run):
    return list(csv.DictReader(open(run.train / "train_log.csv")))


def _smooth(x, k=20):
    x = np.asarray(x, float)
    if len(x) < k:
        return x
    c = np.convolve(x, np.ones(k) / k, mode="valid")
    return np.concatenate([x[:k - 1] * np.nan, c])


def figures(dance: Dance, doc: dict, ms: dict, M: dict, cell, match, out: Path, tag: str = ""):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    figs = {}
    # ---- training curves
    keys = [("sim_rew", "sim_rew (own reference)"), ("ep_ret_norm", "ep_ret_norm"), ("term_frac", "term_frac"), ("wm_err8", "wm_err8")]
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    base = dance.run("robust_d_v1"); br = [r for r in _train_rows(base) if float(r["iteration"]) >= 37000]
    for a, (k, t) in zip(ax, keys):
        a.plot([float(r["iteration"]) for r in br], _smooth([float(r[k]) for r in br]), color="#7f8c8d", lw=1, label="robust_d_v1 (base)")
        for mk, m in ms.items():
            if mk == "anchor":
                continue
            rr = _train_rows(dance.run(m["run"]))
            a.plot([float(r["iteration"]) for r in rr], _smooth([float(r[k]) for r in rr]), color=GRP_COL[m["group"]], ls="-" if mk.endswith("0") else "--", lw=1, label=f"{m['run']}")
        a.set_title(t, fontsize=10); a.set_xlabel("iteration"); a.axvline(40000, color="k", lw=0.5)
    ax[0].legend(fontsize=7); fig.tight_layout(); p = out / f"ft_train_curves{tag}.png"; fig.savefig(p, dpi=100); plt.close(fig); figs["train"] = p
    # ---- eval-grid curves (own references)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for mk, m in ms.items():
        if mk == "anchor":
            continue
        ev = FC.eval_curve(dance.run(m["run"]))
        for a, k in zip(ax, ("balance", "fidelity", "survival_robust")):
            a.plot([e["iteration"] for e in ev], [e[k] for e in ev], marker="o", ms=3, color=GRP_COL[m["group"]], ls="-" if mk.endswith("0") else "--", label=m["run"])
            a.set_title(f"evaluator grid: {k} (own ref; not comparable across groups)", fontsize=9)
    ax[0].legend(fontsize=7); fig.tight_layout(); p = out / f"ft_eval_curves{tag}.png"; fig.savefig(p, dpi=100); plt.close(fig); figs["eval"] = p
    # ---- decision metrics, seed points
    names = list(FC.DIR); fig, ax = plt.subplots(3, 4, figsize=(18, 10))
    for a, n in zip(ax.flat, names):
        r = doc["rows"][n]
        for i, (g, vals) in enumerate((("A", r["A"]), ("B", r["B"]))):
            a.bar(i, np.mean(vals), color=GRP_COL[g], alpha=0.6); a.scatter([i] * len(vals), vals, color="k", s=14, zorder=3)
        a.set_xticks([0, 1]); a.set_xticklabels(["A orig (ref-O)", "B fixed (ref-F)"], fontsize=8); a.set_title(f"{FC.LABEL_EN[n]} [{r['status']}]", fontsize=9)
    fig.tight_layout(); p = out / f"ft_metrics{tag}.png"; fig.savefig(p, dpi=100); plt.close(fig); figs["metrics"] = p
    return figs


def write_report(dance: Dance, doc: dict, M: dict, grid: dict, ms: dict, out: Path, cell, match, tag: str = "") -> Path:
    figs = figures(dance, doc, ms, M, cell, match, out, tag)
    vd, rows, g = doc["verdict"], doc["rows"], doc["verdict"]["gates"]
    Ak = [k for k in ms if k.startswith("A")]; Bk = [k for k in ms if k.startswith("B")]
    f = lambda x, nd=3: f"{x:.{nd}f}"
    L = [f"# V1.6 B 线：原始数据 vs 修正数据（接触修正）微调对比报告（自动生成 {doc['generated']}）", "",
         f"## 结论：**{vd['verdict']}**", "",
         f"- 判据在训练/对比前写定：`results/{dance.name}/v16/B_train/FT_PROTOCOL.md`（同值写入 `pipeline/pmpipe/ftcompare.py:DECISION`）。显著改善项：{vd['improved'] or '无'}；显著退化项：{vd['degraded'] or '无'}。",
         "- 闸门：" + "；".join(f"{k} = {'PASS' if v['passed'] else '**FAIL**'}" for k, v in g.items()), ""]
    L += ["## 1. 实验设置", "", "| 键 | run | 组 | 种子 | checkpoint | ckpt sha12 | 动作目录 |", "|---|---|---|---|---|---|---|"]
    for k, m in doc["models"].items():
        L.append(f"| {k} | {m['run']} | {m['group']} | {m['seed']} | it{m['iteration']} | {m['sha12']} | `{Path(m['motion_dir']).name}` |")
    L += ["", f"- ref-O = `{doc['refs']['O']}`；ref-F = `{doc['refs']['F']}`" + ("（**注意：本次两参考相同，仅为流程冒烟，不构成原始/修正对比**）" if doc['refs']['O'] == doc['refs']['F'] else ""),
          "- 起点均为 `robust_d_v1` it40000 完整状态，resume 式微调，配方/种子集合/envs 相同，唯一差别 = 动作目录；评估 = 部署控制链闭环（全段 45 s，nominal 17 起点 + 7 种工况×8 扰动起点，E1/E2 两批不同种子），2×2（策略×参考）矩阵；决策只用匹配部署格（A↔ref-O，B↔ref-F）。", ""]
    L += ["## 2. 决策指标（第三方、与参考无关；匹配部署；nominal 类起点 E1+E2；每个 run 取均值）", "",
          "| 指标 | A（各 seed） | B（各 seed） | A 均值 | B 均值 | 改善量 | 需要 ≥ | 严格支配 | 判定 |", "|---|---|---|---|---|---|---|---|---|"]
    for n in FC.DIR:
        r = rows[n]; nd = 4 if "rad" in n or "_m" in n or "rewrite" in n else 3
        L.append(f"| {'**' if r['decision'] else ''}{FC.LABEL[n]}{'**' if r['decision'] else ''} | {', '.join(f(x, nd) for x in r['A'])} | {', '.join(f(x, nd) for x in r['B'])} | {f(r['mean_A'], nd)} | {f(r['mean_B'], nd)} | {r['gain']:+.{nd}f} | {f(r['need'], nd)} (run间 {f(r['spread_run'], nd)}, 评估间 {f(r['se_eval'], nd)}) | {'B' if r['strict_dominance_B'] else 'A' if r['strict_dominance_A'] else '-'} | **{r['status']}** |")
    L += ["", "> 粗体 = 计入 GO 计数的 D1–D7；其余仅参与“无退化”检查。方向：除 CoP 余量外越低越好。显著 = 改善量 ≥ max(10% 基线, 绝对阈值, 种子间差, 评估批次间差) 且所有 B 种子严格优于所有 A 种子。", "",
          f"![decision metrics]({figs['metrics'].name})", ""]
    L += ["## 3. 2×2 矩阵（策略组 × 部署参考；nominal 类起点均值，两个种子平均）— 只有匹配格参与判定", "",
          "| 指标 | A→ref-O（匹配） | A→ref-F（交叉） | B→ref-O（交叉） | B→ref-F（匹配） |", "|---|---|---|---|---|"]
    def gm(grp, rk, key, sub=None):
        ks = Ak if grp == "A" else Bk
        return float(np.mean([FC._mean(cell(k, rk), key, sub) for k in ks]))
    for n in list(FC.DIR) + ["lab_agree_F", "lab_agree_O", "ref_root_inconsistency_m", "ds", "ss"]:
        nd = 4 if ("rad" in n or "_m" in n or "rewrite" in n or "lab" in n) else 3
        L.append(f"| {FC.LABEL.get(n, n)} | " + " | ".join(f(gm(gp, rk, n), nd) for gp, rk in (("A", "O"), ("A", "F"), ("B", "O"), ("B", "F"))) + " |")
    L += ["", "> `ref_root_inconsistency_m` = 该参考自身的根轨迹与其落脚隐含根轨迹的偏差（参考质量指标，不是策略指标；ref-O 应明显大于 ref-F）；接触一致率与训练目标同源，仅供参考。", ""]
    L += ["## 4. 跟踪误差（**不跨参考比高低**；锚定 MPJPE = XY/yaw 锚定 14-link，保留根高度，不能代替全局轨迹误差）", "",
          "| 策略组→参考 | 对 ref-O：锚定 mm / 全局 mm / 根XY偏差 m | 对 ref-F：锚定 mm / 全局 mm / 根XY偏差 m |", "|---|---|---|"]
    for gp, rk in (("A", "O"), ("A", "F"), ("B", "O"), ("B", "F")):
        ks = Ak if gp == "A" else Bk
        cellv = lambda tag, key: float(np.mean([FC._mean(cell(k, rk), key, tag) for k in ks]))
        L.append(f"| {gp}→ref-{rk}{'（匹配）' if (gp, rk) in (('A', 'O'), ('B', 'F')) else ''} | " + " | ".join(f"{cellv(t, 'mpjpe_plain_mm'):.1f} / {cellv(t, 'global_mpjpe_mm'):.1f} / {cellv(t, 'root_xy_err_mean_m'):.3f}" for t in ("onO", "onF")) + " |")
    L += ["", "### 4b. 深前倾段（训练帧 310–430）与全程倾角/重力方向误差（B1 要求；gravity_err = 机体系重力方向与参考的夹角，A1 口径；两参考都给）", "",
          "| 策略组→参考 | 全程 gravity_err 最大 rad | 全程最大倾角 rad（参考最大） | 段内 gravity_err 最大/均值 rad | 段内最大倾角 rad（该段参考最大） | 段内倾角余量 rad |", "|---|---|---|---|---|---|"]
    for gp, rk in (("A", "O"), ("A", "F"), ("B", "O"), ("B", "F")):
        ks = Ak if gp == "A" else Bk
        sv = lambda key: float(np.mean([FC._mean(cell(k, rk), key, f"seg{rk}") for k in ks]))
        mx = lambda key: float(np.max([max(r[f"seg{rk}"][key] for r in cell(k, rk)) for k in ks]))
        L.append(f"| {gp}→ref-{rk}{'（匹配）' if (gp, rk) in (('A', 'O'), ('B', 'F')) else ''} | {mx('gerr_max_rad'):.3f} | {mx('tilt_max_rad'):.3f}（{sv('ref_tilt_max_rad'):.3f}） | {mx('seg_gerr_max_rad'):.3f} / {sv('seg_gerr_mean_rad'):.3f} | {mx('seg_tilt_max_rad'):.3f}（{sv('seg_ref_tilt_max_rad'):.3f}） | {sv('seg_tilt_margin_rad'):+.3f} |")
    L += ["", "> 最大值取所有起点/工况的最大（最坏情形），其余为均值。余量 = 参考段内最大倾角 − 闭环段内最大倾角（>0 表示闭环未超过参考；与 A1 的 gravity_err 阈值比较请用该表的最大值）。B1 报告：`results/diy/v16/B_contact/CONTACT_REFINE_REPORT.md`（QA 23/23；接触标签语义变化：58%→14% 两脚均不接触；动力学难度基本不变，ZMP 支撑域外 85%→81%；脚滑 p95 −57% 但 p99/max 变差；根 z 下移 1~83 mm）。", ""]
    fid = g["G3_fidelity_ceiling"]
    L += ["", f"- G3 保真上限（只防学崩）：" + "；".join(f"{k}: 锚定 {v['anch']:.1f} mm / 全局 {v['glob']:.1f} mm" for k, v in fid["fidelity"].items()) + f" → {'PASS' if fid['passed'] else '**FAIL**'}", ""]
    L += ["## 5. 鲁棒性 / 存活", "", "| 键 | 部署链全工况存活 | nominal 全起点存活 |", "|---|---|---|"]
    for k in ms:
        L.append(f"| {k} ({ms[k]['run']}) | {doc['survival'][k]:.3f} | {g['G1_safety']['nominal_all_survive'][k]} |")
    L += ["", "训练评估网格 / 硬复评（`evaluator.py`：14 工况×8 重复 seed 1234/4321；硬复评 22 工况×24 重复 seed 777），在两参考上：", "", "| 键 | 参考 | 评估 | balance | fidelity | 鲁棒存活 | nominal 存活 | mpjpe_nom mm | 足滑 | 穿透 mm |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in grid.items():
        k, rk, gtag = key.split("|"); s = r["scores"]
        L.append(f"| {k} | ref-{rk} | {gtag} | {s['balance']:.1f} | {s['fidelity']:.1f} | {s['survival_robust']:.3f} | {r['conditions']['nominal']['survival']:.2f} | {s['mpjpe_nominal_mm']:.1f} | {s['slip_mean']:.3f} | {s['pen_max_mm']:.1f} |")
    L += ["", "G1 网格闸门（B 在 ref-F 上；阈值 nominal=1、鲁棒≥0.90 且 ≥ min(A 在 ref-O)−0.02）：" + "；".join(f"{k}: {'PASS' if v['passed'] else 'FAIL'}({v['robust']:.3f} vs A_min {v['A_min']:.3f})" for k, v in g["G1_safety"]["grid"].items()), ""]
    L += ["## 6. 训练指标", "", "| run | 迭代 | 起点 100 轮 sim_rew/ep_ret/term | 末 1000 轮 sim_rew | ep_ret | term | wm_err8 | it/s | NaN | 跳步 wm/v | 回滚 | 重启 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, t in doc["train"].items():
        l = t["last1000"]
        L.append(f"| {ms[k]['run']} | {t['iterations']} | {t['first100']['sim_rew']:.3f}/{t['first100']['ep_ret']:.3f}/{t['first100']['term']:.3f} | {l['sim_rew']['mean']:.3f}±{l['sim_rew']['std']:.3f} | {l['ep_ret_norm']['mean']:.3f}±{l['ep_ret_norm']['std']:.3f} | {l['term_frac']['mean']:.4f} | {l['wm_err8']['mean']:.4f} | {l['it_per_s']['mean']:.1f} | {t['nonfinite']} | {t['skipped_wm']}/{t['skipped_v']} | {t['rollbacks']} | {t['restarts']} |")
    L += ["", "> sim_rew / ep_ret 是对**各自参考**的奖励，不跨组比高低；这里只看起点冲击与恢复、数值健康、末段稳定性。", "", f"![train curves]({figs['train'].name})", "", f"![eval curves]({figs['eval'].name})", ""]
    neg = [f"{FC.LABEL[n]}（{r['status']}）" for n, r in rows.items() if r["status"] != "improved"]
    L += ["## 7. 负面结果与局限（如实）", "", "- 未显著改善的指标：" + ("；".join(neg) if neg else "无"), "- 每组只有 2 个种子，运行间噪声估计粗糙；判定同时要求严格支配以抵消。单舞蹈、单起点权重（it40000），10000 轮追加训练。",
          "- 评估为 MuJoCo 部署控制链（非真机）；足滑/穿透等是仿真接触量，数值上的“改善”不等于真机改善。", f"- 冻结校验：{g['G0_integrity']['freeze']['checked']} 个历史文件，变化 {len(g['G0_integrity']['freeze']['changed'])} 个。", ""]
    p = out / f"FT_COMPARE_REPORT{tag}.md"
    p.write_text("\n".join(L) + "\n")
    return p
