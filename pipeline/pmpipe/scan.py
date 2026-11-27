"""PartyMimic Pipeline V1.4 - multi-dance scan: `./pm scan [<dance> ...]`.

Reuses the existing stages only: `prepare` (conversion + kinematic QA), `quality.analyze_reference` (ground calibration, contact tiers, root consistency, ZMP on reliable
frames, difficulty vs LAFAN) and `repair.motion_metrics` (geometry defects: slip, root-vs-feet consistency).  Adds the ranking / defect flags / training-recommendation table
and a deterministic selection of ONE low-difficulty dance for the default recipe (robust_d_v1 on the ORIGINAL data - V1.3 found no evidence that repaired data helps).
Output: results/_scan/{V14_SCAN_REPORT.md, scan.json, selection.json, scan_ranking.png}.  Nothing outside results/<dance>/ and results/_scan/ is written.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np

from . import quality as Q
from . import repair as P
from .core import DATASETS, REPO, RESERVED, RESULTS, Dance, Logger, now, read_json, write_json

# composite difficulty = weighted mean of min-max normalised metrics over the scanned dances (0 = easiest of the set, 1 = hardest)
METRICS = [  # key, label, weight, getter(dict)
    ("single_support_frac", "单脚支撑占比", 0.10, lambda r: r["difficulty"]["metrics"]["single_support_frac"]),
    ("flight_frac", "腾空占比", 0.08, lambda r: r["difficulty"]["metrics"]["flight_frac"]),
    ("transitions_per_s", "接触切换 次/s", 0.08, lambda r: r["difficulty"]["metrics"]["transitions_per_s"]),
    ("foot_spread_p95", "步幅(两脚间距p95) m", 0.10, lambda r: r["difficulty"]["metrics"]["foot_spread_p95"]),
    ("com_acc_p95", "质心加速度p95 m/s²", 0.12, lambda r: r["difficulty"]["metrics"]["com_acc_p95"]),
    ("zmp_offset_p95", "ZMP-CoM 偏移p95 m", 0.12, lambda r: r["difficulty"]["metrics"]["zmp_offset_p95"]),
    ("slip_point_mps", "参考脚滑(接触点) m/s", 0.10, lambda r: r["geo"]["slip_point_mps"]["mean"]),
    ("jv_p95", "关节速度p95 rad/s", 0.10, lambda r: r["difficulty"]["metrics"]["jv_p95"]),
    ("jacc_p95", "关节加速度p95 rad/s²", 0.05, lambda r: r["difficulty"]["metrics"]["jacc_p95"]),
    ("root_vs_implied_max", "根轨迹与落脚不一致(最大) m", 0.15, lambda r: r["geo"]["root_vs_implied"]["max_m"]),
]
NOT_SELECTABLE = {"ythn": "V1.3 已封存（NO-GO，不再追加训练）", "ssb": "SSB 历史对象（results/p0..p5 已有大量成果，避免混淆）"}
MAX_SECONDS = 12.0  # default validated run window of the C++ framework (longer dances need an explicit, unvalidated override)


def discover(names=None) -> list[str]:
    out = []
    for p in sorted(DATASETS.iterdir()):
        if p.is_dir() and (p / "robot_motion.pkl").exists() and p.name not in RESERVED and not p.name.startswith("_") and p.name != "tmp":
            out.append(p.name)
    return [n for n in out if not names or n in names]


def analyse_one(name: str, force=False, log=print) -> dict:
    from .prepare import prepare
    d = Dance(name)
    rec = dict(dance=name, source=str(d.source_pkl()))
    try:
        meta = prepare(d, force=force, log=Logger(d.dir / "logs" / "prepare.log", echo=False))
    except SystemExit as e:  # prepare() dies on a failed QA / converter
        rec["error"] = f"prepare failed (exit {e.code}); see results/{name}/logs"; return rec
    qa = read_json(d.metrics / "qa_kinematic.json", {}) or {}
    rec["qa"] = dict(verdict=qa.get("verdict", meta.get("qa_verdict")), warnings=qa.get("warnings", []), duration_s=(qa.get("clip") or {}).get("duration_s"), frames=(qa.get("clip") or {}).get("frames"), trimmed=meta.get("trimmed"))
    if rec["qa"]["duration_s"] is None:
        z = np.load(d.npz); rec["qa"]["duration_s"] = float(len(z["joint_pos"]) / float(z["fps"][0])); rec["qa"]["frames"] = int(len(z["joint_pos"]))
    ref = Q.analyze_reference(d, RESULTS / "_scan" / name, REPO / "pipeline/ref_data/lafan_difficulty_v1.json")
    ref.pop("series", None); rec.update({k: ref[k] for k in ("ground", "contact_tiers", "root_consistency", "dynamics_reliable_frames", "difficulty")})
    g = P.motion_metrics(d.padded_npz, pad=Q.PAD); rec["geo"] = {k: g[k] for k in ("slip_point_mps", "root_path_m", "implied_path_m", "root_vs_implied", "static_stance_clearance_mm", "floating_window_median_range_mm", "min_clearance_mm", "label_contact_frac", "label_none", "com_speed_max", "com_acc_xy_p95")}
    return rec


def defects(r: dict) -> list[str]:
    f = []
    if r.get("error"): return ["PREPARE_FAILED"]
    if r["qa"]["verdict"] not in ("PASS", "ADOPTED"): f.append(f"QA_{r['qa']['verdict']}")
    if any("hardware limit" in w for w in r["qa"]["warnings"]): f.append("JOINT_VEL_OVER_LIMIT")
    if r["qa"]["duration_s"] > MAX_SECONDS: f.append(f"LONG({r['qa']['duration_s']:.1f}s>12s)")
    if r["qa"].get("trimmed"): f.append(f"LEADING_GLITCH_TRIMMED({r['qa']['trimmed']})")
    gr = r["ground"]
    if max(gr["floor_offset_mm"]) > 8: f.append(f"FLOATING({max(gr['floor_offset_mm']):.0f}mm)")
    if not gr["confident"]: f.append("NO_STATIC_STANCE")
    if r["geo"]["root_vs_implied"]["max_m"] > 0.30: f.append(f"ROOT_SKATING({r['geo']['root_vs_implied']['max_m']:.2f}m)")
    return f


def rank(recs: list[dict]) -> list[dict]:
    ok = [r for r in recs if not r.get("error")]
    for key, _, _, get in METRICS:
        v = np.array([get(r) for r in ok], float); lo, hi = v.min(), v.max()
        for r, x in zip(ok, v): r.setdefault("norm", {})[key] = float((x - lo) / (hi - lo)) if hi > lo else 0.0; r.setdefault("raw", {})[key] = float(x)
    for r in ok:
        r["difficulty_score"] = float(sum(w * r["norm"][k] for k, _, w, _ in METRICS))
    order = sorted(ok, key=lambda r: r["difficulty_score"])
    for i, r in enumerate(order): r["rank_min_max"] = i + 1
    # robustness check: the same composite from per-metric RANKS
    for key, _, _, _ in METRICS:
        v = np.array([r["raw"][key] for r in ok]); rk = v.argsort().argsort()
        for r, x in zip(ok, rk): r.setdefault("rankn", {})[key] = float(x) / max(len(ok) - 1, 1)
    for r in ok: r["difficulty_rank_score"] = float(sum(w * r["rankn"][k] for k, _, w, _ in METRICS))
    for i, r in enumerate(sorted(ok, key=lambda r: r["difficulty_rank_score"])): r["rank_by_rank"] = i + 1
    return order


def choose(order: list[dict]) -> dict:
    """lowest composite difficulty among the dances that are selectable: not sealed/historic, QA PASS, <= 12 s, no joint-velocity violation, no missing static stance."""
    cands = []
    for r in order:
        why = []
        if r["dance"] in NOT_SELECTABLE: why.append(NOT_SELECTABLE[r["dance"]])
        d = defects(r)
        why += [x for x in d if x.startswith(("QA_FAIL", "JOINT_VEL", "LONG", "NO_STATIC", "PREPARE"))]
        r["selectable"] = not why; r["excluded_because"] = why
        if not why: cands.append(r)
    pick = cands[0] if cands else None
    return dict(chosen=pick["dance"] if pick else None, rule="min composite difficulty among selectable dances (not sealed/historic, QA PASS, <=12 s, no joint-velocity violation, static stance present)",
                runner_up=cands[1]["dance"] if len(cands) > 1 else None, agrees_with_rank_composite=bool(pick and pick["rank_by_rank"] == min(c["rank_by_rank"] for c in cands)) if pick else None)


def recommendation(r: dict, sel: dict) -> str:
    if r.get("error"): return "不可用（转换/QA 失败）"
    if r["dance"] == sel["chosen"]: return "**首选：原始数据 + robust_d_v1，2000 轮闸门后续训 40000**"
    if r["dance"] in NOT_SELECTABLE: return NOT_SELECTABLE[r["dance"]]
    if r["excluded_because"]: return "暂不推荐：" + "；".join(r["excluded_because"])
    if r["dance"] == sel["runner_up"]: return "备选（原始数据 + robust_d_v1）"
    return "可训练（难度较高，排在后面；原始数据 + robust_d_v1）" + ("；有参考缺陷，可用 `pm repair --validate` 诊断，但 V1.3 未证明修复数据更好" if any(x.startswith(("ROOT_SKATING", "FLOATING")) for x in defects(r)) else "")


def scan(names=None, force=False, log=print) -> dict:
    out = RESULTS / "_scan"; out.mkdir(parents=True, exist_ok=True); names = discover(names); log("scanning", names)
    recs = []
    for n in names:
        log("  ", n)
        try: recs.append(analyse_one(n, force=force))
        except Exception as e: recs.append(dict(dance=n, error=repr(e), trace=traceback.format_exc()[-600:]))
    order = rank(recs); sel = choose(order)
    for r in recs: r["defects"] = defects(r); r["recommendation"] = recommendation(r, sel) if not r.get("error") else "不可用"
    write_json(out / "scan.json", dict(generated=now(), weights={k: w for k, _, w, _ in METRICS}, dances=recs)); write_json(out / "selection.json", dict(generated=now(), **sel))
    _figure(order, out / "scan_ranking.png"); _report(recs, order, sel, out / "V14_SCAN_REPORT.md"); return dict(selection=sel, report=str(out / "V14_SCAN_REPORT.md"), order=[r["dance"] for r in order])


def _figure(order, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5)); left = np.zeros(len(order)); cols = plt.cm.tab10.colors
    for i, (k, lab, w, _) in enumerate(METRICS):
        v = np.array([r["norm"][k] * w for r in order]); ax.barh([r["dance"] for r in order], v, left=left, color=cols[i % 10], label=lab); left += v
    ax.invert_yaxis(); ax.set_xlabel("composite difficulty (0 easiest of the set .. 1 hardest)"); ax.legend(fontsize=7, ncol=2, loc="lower right"); fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def _report(recs, order, sel, path):
    f = lambda x, n=2: f"{x:.{n}f}"
    L = [f"# V1.4 多舞蹈扫描：难度排名 / 数据缺陷 / 训练推荐（自动生成 {now()}）", "",
         f"> 复用 `./pm prepare` + `quality.analyze_reference` + `repair.motion_metrics`（只读分析，不改任何动作文件）。难度 = 10 项指标在被扫描舞蹈内 min-max 归一后的加权平均（0=最易，1=最难）；另用“按排名”的同权重复合作稳健性检查。足滑采用 V1.4 标准的**接触点**口径。", "",
         f"## 结论：选择 **{sel['chosen']}**" + (f"（备选 {sel['runner_up']}）" if sel["runner_up"] else ""), "",
         f"- 选择规则：{sel['rule']}。按排名复合的排序{'与之一致' if sel['agrees_with_rank_composite'] else '**与之不一致**，见下表“排名复合名次”'}。",
         "- 默认使用**原始参考数据**（V1.3：修复数据 2000 轮对照为 NO-GO），配方 `robust_d_v1`，起点官方预训练；`./pm run` 先 2000 轮闸门，通过后自动续训至 40000。", "",
         "## 1. 排名与推荐", "", "| 名次 | 舞蹈 | 难度 | 排名复合名次 | 时长 s | QA | 缺陷 | 推荐 |", "|---|---|---|---|---|---|---|---|"]
    for r in order: L.append(f"| {r['rank_min_max']} | {r['dance']} | {f(r['difficulty_score'])} | {r['rank_by_rank']} | {f(r['qa']['duration_s'], 1)} | {r['qa']['verdict']} | {', '.join(r['defects']) or '-'} | {r['recommendation']} |")
    for r in recs:
        if r.get("error"): L.append(f"| - | {r['dance']} | - | - | - | - | PREPARE_FAILED | {r['error'][:80]} |")
    L += ["", "## 2. 指标明细（原值；括号内 = 归一值）", "", "| 舞蹈 | " + " | ".join(lab for _, lab, _, _ in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for r in order: L.append(f"| {r['dance']} | " + " | ".join(f"{r['raw'][k]:.3g} ({r['norm'][k]:.2f})" for k, _, _, _ in METRICS) + " |")
    L += ["", "权重：" + "，".join(f"{lab} {w:.2f}" for _, lab, w, _ in METRICS), "", "## 3. 数据缺陷（参考动作本身）", "",
          "| 舞蹈 | 静立离地 mm | 地面漂移范围 mm | 根路径 / 落脚隐含路径 m | 根与落脚最大不一致 m | 接触可靠度（双/单/不确定） | 可靠帧覆盖 | ZMP 在支撑域外 | 接触点足滑 m/s | 训练标签触地占比 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in order:
        g, c, dy = r["geo"], r["contact_tiers"], r["dynamics_reliable_frames"]
        L.append(f"| {r['dance']} | {r['ground']['floor_offset_mm']} | {g['floating_window_median_range_mm']} | {f(g['root_path_m'])} / {f(g['implied_path_m'])} | {f(g['root_vs_implied']['max_m'])} | {c['double_support']*100:.0f}/{c['single_support']*100:.0f}/{c['ambiguous']*100:.0f}% | {dy['coverage']*100:.0f}% | "
                 f"{('%.0f%%' % (dy['zmp_outside_frac']*100)) if dy['zmp_outside_frac'] is not None else '-'} | {f(g['slip_point_mps']['mean'], 3)} | {g['label_contact_frac']*100:.0f}% |")
    L += ["", "- 视频提取的参考普遍存在“悬浮 + 根平移脚滑”（YTHN 同样）；V1.3 表明用修复数据重训在 2000 轮上没有明确收益，因此扫描只**记录**缺陷，默认不修复。", "- 接触可靠度中“不确定”帧不进入 ZMP/支撑域统计；可靠帧覆盖率低的舞蹈，其 ZMP 结论可信度低。",
          "", "## 4. 与 LAFAN 舞蹈（8 条）的相对位置（百分位，越高越难）", "", "| 舞蹈 | 质心加速度 | ZMP-CoM 偏移 | 步幅 | 单脚支撑 | 腾空 | 切换频率 | 关节速度 RMS |", "|---|---|---|---|---|---|---|---|"]
    for r in order:
        v = r["difficulty"]["vs_lafan"]; L.append(f"| {r['dance']} | " + " | ".join(f"{v[k+'@dance']['percentile']:.0f}" for k in ("com_acc_p95", "zmp_offset_p95", "foot_spread_p95", "single_support_frac", "flight_frac", "transitions_per_s", "jv_rms")) + " |")
    L += ["", "![排名](scan_ranking.png)", "", f"数据：`results/_scan/scan.json`、`selection.json`；每支舞蹈的参考质量图在 `results/_scan/<舞蹈>/`。", ""]
    path.write_text("\n".join(L) + "\n")
