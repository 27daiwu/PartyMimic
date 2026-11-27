"""V1.6 deploy-oriented engineering verification of the fine-tuned candidate (`python -m pmpipe.ftdeploy <dance> --b ft_fix_v16 ...`).
Uses the closed-loop deployment-chain rollouts that `./pm ftcompare` already produced (results/<dance>/v16/B_train/ftc/traj/<key>/<ref>/<batch>/*.npz); no new simulation.
Outputs (results/<dance>/v16/B_train/deploy_eval/): FULL_TRACK_REPORT.md, time-series figures, replay video of the candidate on the corrected reference.
Questions answered: full 45 s playback + survival, second-half tracking vs first half, root-XY drift growth, dynamics safety (penetration / torque / guard / tilt / slip) - all per matched deployment AND on the common corrected reference.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import ftcompare as FC
from . import quality as Q
from .core import Dance, now, read_json, write_json

PAD = Q.PAD


def load_set(W: Path, key: str, ref: str, batches=("E1", "E2"), prefix="nominal"):
    fs = []
    for b in batches:
        fs += sorted((W / "traj" / key / ref / b).glob(f"{prefix}_*.npz"))
    return sorted(fs, key=lambda f: (f.name != "nominal_ref.npz", f.parent.name, f.name))  # the reference start (nominal_ref, E1) first


def series(f, PM, PF):
    ro = Q.load_rollout(f); T = len(ro["t"]); out = dict(T=T, survived=ro["meta"]["fell_t"] is None, fell_t=ro["meta"]["fell_t"])
    ref = FC.refinfo(str(PM))["R"]; Rf = FC.refinfo(str(PF))["R"]
    S = Q.fk_series(ro["root_p"], ro["root_q"], ro["q"], body_names=ref["body_names"])
    sp = ro["root_p"]; sq = ro["root_q"]
    for tag, R in (("own", ref), ("F", Rf)):
        rb, rp, rq = R["body_p"][:T], R["root_p"][:T], R["root_q"][:T]
        glob = np.linalg.norm(S["bodies"] - rb, axis=-1).mean(-1)
        dyaw = Q._wrap(Q._yaw(sq) - Q._yaw(rq)); c, s_ = np.cos(dyaw), np.sin(dyaw); rel = rb - rp[:, None, :]
        rot = np.stack([np.stack([c, -s_], -1), np.stack([s_, c], -1)], -2); rxy = np.einsum("tij,tbj->tbi", rot, rel[..., :2])
        anc = np.concatenate([sp[:, None, :2] + rxy, rp[:, 2:3, None] + rel[..., 2:3]], -1)
        out[f"anch_{tag}"] = np.linalg.norm(S["bodies"] - anc, axis=-1).mean(-1) * 1000
        out[f"glob_{tag}"] = glob * 1000
        out[f"exy_{tag}"] = np.linalg.norm(sp[:, :2] - rp[:, :2], axis=1)
    out["tilt"] = ro["tilt"]; out["xy"] = sp[:, :2]
    return out, ro


def half_stats(ser, key, T):
    a, b = PAD, T - PAD; mid = (a + b) // 2
    x = ser[key]
    return float(x[a:mid].mean()), float(x[mid:b].mean()), float(x[a:b].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dance"); ap.add_argument("--a", default="A0"); ap.add_argument("--b", default="B0"); ap.add_argument("--b2", default="B1"); ap.add_argument("--a2", default="A1")
    ap.add_argument("--out", default=None); ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    d = Dance(a.dance); B = d.dir / "v16/B_train"; W = B / "ftc"; out = Path(a.out) if a.out else B / "deploy_eval"; out.mkdir(parents=True, exist_ok=True)
    res = read_json(B / "ft_result.json"); PO, PF = Path(res["refs"]["O"]), Path(res["refs"]["F"])
    keys = {"A": [k for k in res["models"] if k.startswith("A")], "B": [k for k in res["models"] if k.startswith("B")]}
    match = lambda k: "O" if k.startswith("A") else "F"
    pm = lambda k: PO if match(k) == "O" else PF
    # ---- per-rollout series for nominal-class (E1+E2, 33 starts incl. the reference start)
    data = {}
    for k in keys["A"] + keys["B"]:
        fs = load_set(W, k, match(k)); data[k] = [series(f, pm(k), PF)[0] for f in fs]
    T = data[keys["B"][0]][0]["T"]; fps = 50.0; t = (np.arange(T) - PAD) / fps
    # ---- tables: first vs second half (own reference) and on the common reference F
    L = [f"# V1.6 候选模型部署向工程验证（修正数据微调 {a.b} / {a.b2}；自动生成 {now()}）", "",
         "数据来源：`ftcompare` 已有的部署控制链闭环（全段 45 s，含前后各 1 s 静止；nominal 类 33 个起点 = 1 参考起点 + 16 扰动起点 × E1/E2 两批种子）。**不含任何新仿真。** 注意：每组只有 2 个训练种子（4 种子扩展已取消），下列均为描述性对比，不构成统计显著性结论。", ""]
    # playback + survival
    surv = {k: (sum(r["survived"] for r in data[k]), len(data[k])) for k in data}
    L += ["## 1. 全段播放与存活", "", "| 模型 | 播放帧数 / 完整片段 | 存活起点 |", "|---|---|---|"] + [f"| {k} ({res['models'][k]['run']}) | {data[k][0]['T']} / {T}（{data[k][0]['T'] * 0.02:.1f} s，覆盖整段） | {s}/{n} |" for k, (s, n) in surv.items()] + [""]
    # first vs second half
    def grp_stats(ks, key):
        v = np.array([[half_stats(r, key, r["T"]) for r in data[k]] for k in ks])  # (models, starts, 3)
        return v.mean((0, 1)), v.mean(1)  # mean over models+starts ; per model (3)
    L += ["## 2. 前半程 / 后半程跟踪与根 XY 漂移（匹配部署：A→ref-O，B→ref-F；均值，括号内为两个种子各自的均值）", "",
          "| 指标 | A 前半 | A 后半 | B 前半 | B 后半 | A 后/前 | B 后/前 |", "|---|---|---|---|---|---|---|"]
    names = {"anch_own": "XY/yaw 锚定 MPJPE mm（对自己的参考）", "glob_own": "全局 MPJPE mm（对自己的参考）", "exy_own": "根 XY 偏差 m（对自己的参考）"}
    for key, nm in names.items():
        mA, pA = grp_stats(keys["A"], key); mB, pB = grp_stats(keys["B"], key)
        fmt = lambda p, i: "(" + ", ".join(f"{x[i]:.3f}" for x in p) + ")"
        nd = 3 if "exy" in key else 1
        L.append(f"| {nm} | {mA[0]:.{nd}f} {fmt(pA, 0) if nd == 3 else ''} | {mA[1]:.{nd}f} | {mB[0]:.{nd}f} | {mB[1]:.{nd}f} | {mA[1] / mA[0]:.2f}× | {mB[1] / mB[0]:.2f}× |")
    L += ["", "同一指令（公共参考 ref-F）下的对比——B→ref-F（匹配）与 A→ref-F（交叉：原始数据策略接收修正参考指令；ftcompare 的 A→ref-F 格）：", "",
          "| 指标 | A→F 前半 | A→F 后半 | B→F 前半 | B→F 后半 |", "|---|---|---|---|---|"]
    dataAF = {k: [series(f, PF, PF)[0] for f in load_set(W, k, "F")] for k in keys["A"]}
    for key, nm in (("anch_F", "锚定 MPJPE mm"), ("glob_F", "全局 MPJPE mm"), ("exy_F", "根 XY 偏差 m")):
        vA = np.array([[half_stats(r, key, r["T"]) for r in dataAF[k]] for k in keys["A"]]).mean((0, 1)); vB = np.array([[half_stats(r, key, r["T"]) for r in data[k]] for k in keys["B"]]).mean((0, 1))
        nd = 3 if "exy" in key else 1
        L.append(f"| {nm} | {vA[0]:.{nd}f} | {vA[1]:.{nd}f} | {vB[0]:.{nd}f} | {vB[1]:.{nd}f} |")
    # windows of 5 s
    L += ["", "### 5 s 分窗（B 组，匹配 ref-F；A 组匹配 ref-O 作对照）：根 XY 偏差 m（窗口均值 / 窗口末）与全局 MPJPE mm", "", "| 窗口 s | A 根偏差 均值/末 | B 根偏差 均值/末 | A 全局 mm | B 全局 mm | B 锚定 mm | B 倾角 p95 rad |", "|---|---|---|---|---|---|---|"]
    n = 250
    for a0 in range(PAD, T - PAD, n):
        b0 = min(a0 + n, T - PAD)
        def win(ks, key, how):
            v = np.array([[(r[key][a0:b0].mean() if how == "m" else r[key][b0 - 1]) for r in data[k]] for k in ks]); return v.mean()
        tp = np.mean([np.percentile(r["tilt"][a0:b0], 95) for k in keys["B"] for r in data[k]])
        L.append(f"| {(a0 - PAD) / fps:.0f}–{(b0 - PAD) / fps:.0f} | {win(keys['A'], 'exy_own', 'm'):.3f} / {win(keys['A'], 'exy_own', 'e'):.3f} | {win(keys['B'], 'exy_own', 'm'):.3f} / {win(keys['B'], 'exy_own', 'e'):.3f} | {win(keys['A'], 'glob_own', 'm'):.0f} | {win(keys['B'], 'glob_own', 'm'):.0f} | {win(keys['B'], 'anch_own', 'm'):.1f} | {tp:.3f} |")
    # drift growth: slope of root xy offset
    sl = lambda ks: float(np.mean([np.polyfit(t[PAD:T - PAD], r["exy_own"][PAD:T - PAD], 1)[0] for k in ks for r in data[k]]))
    L += ["", f"- 根 XY 偏差线性增长斜率（m/s，对自己的参考）：A {sl(keys['A']):.4f}，B {sl(keys['B']):.4f}；末端偏差（舞蹈结束时刻）：A {np.mean([r['exy_own'][T - PAD - 1] for k in keys['A'] for r in data[k]]):.3f} m，B {np.mean([r['exy_own'][T - PAD - 1] for k in keys['B'] for r in data[k]]):.3f} m。",
          "- 注意：根 XY 偏差是闭环根位置相对**各自参考**的世界系差；A 的原始参考根轨迹与落脚不一致 0.25 m（B1 报告），B 的修正参考一致，所以 B 的偏差更小**一部分来自参考本身更自洽**，不是纯粹的策略更好——请与 ftcompare 的 D6/D7（参考无关）一起读。", ""]
    # ---- dynamics safety (from metrics_all, matched, nominal + disturbance conds)
    M = read_json(W / "metrics_all.json")
    def cond_rows(k, ref, cond):
        return [r for b in ("E1", "E2") for n, r in M[f"{k}|{ref}|{b}"].items() if n.startswith(cond)]
    L += ["## 3. 动力学安全（部署控制链，匹配部署；每工况各组 2 种子 × 2 批 × 8 起点；nominal 33 起点）", "", "| 工况 | 组 | 存活 | 穿透 p95 mm / max mm | 髋Roll峰值 N·m | 力矩饱和占比 | 护栏改写率 | 倾角 p95 rad | 足滑 m/s |", "|---|---|---|---|---|---|---|---|---|"]
    for cond in ["nominal"] + FC.COND_SET:
        for g in ("A", "B"):
            rs = [r for k in keys[g] for r in cond_rows(k, match(k), cond)]
            L.append(f"| {cond} | {g} | {sum(r['survived'] for r in rs)}/{len(rs)} | {np.mean([r['D2_pen_p95_mm'] for r in rs]):.2f} / {np.max([r['pen_max_mm'] for r in rs]):.1f} | {np.max([r['D3_hr_peak_nm'] for r in rs]):.1f} | {np.mean([r['sat_all'] for r in rs]):.4f} | {np.mean([r['D4_guard_rewrite'] for r in rs]):.3f} | {np.mean([r['tilt_p95'] for r in rs]):.3f} | {np.mean([r['D1_slip_mps'] for r in rs]):.3f} |")
    L += ["", "> 髋 Roll 峰值被部署链 88 N·m 限幅钳住（≈88 即触顶，两组相同，无信息量）；穿透最大值取最坏起点。", ""]
    # ---- figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(20, 9))
    col = {"A": "#c0392b", "B": "#27ae60"}
    for g in ("A", "B"):
        for k in keys[g]:
            ls = "-" if k.endswith("0") else "--"
            r0 = data[k][0]  # reference start
            ax[0, 0].plot(t, r0["exy_own"], color=col[g], ls=ls, lw=1, label=f"{g}: {res['models'][k]['run']} (own ref)")
            ax[0, 1].plot(t, r0["glob_own"], color=col[g], ls=ls, lw=1)
            ax[0, 2].plot(t, np.degrees(r0["tilt"]), color=col[g], ls=ls, lw=1)
            ax[1, 0].plot(t, r0["anch_own"], color=col[g], ls=ls, lw=1)
    refO = FC.refinfo(str(PO))["R"]; refF = FC.refinfo(str(PF))["R"]
    ax[0, 2].plot(t, np.degrees(FC._tilt_of_quat(refO["root_q"][:T - 1])[:len(t)] if False else FC._tilt_of_quat(refO["root_q"][:len(t)])), color="#e67e22", lw=0.8, label="ref-O tilt")
    ax[0, 2].plot(t, np.degrees(FC._tilt_of_quat(refF["root_q"][:len(t)])), color="#2980b9", lw=0.8, label="ref-F tilt")
    r0 = data[keys["B"][0]][0]; ax[1, 1].plot(refO["root_p"][:len(t), 0], refO["root_p"][:len(t), 1], color="#e67e22", label="ref-O root"); ax[1, 1].plot(refF["root_p"][:len(t), 0], refF["root_p"][:len(t), 1], color="#2980b9", label="ref-F root")
    for g in ("A", "B"):
        k = keys[g][0]; xy = data[k][0]["xy"]; ax[1, 1].plot(xy[:, 0], xy[:, 1], color=col[g], ls="--", label=f"{g} {res['models'][k]['run']} sim root")
    ax[1, 1].set_aspect("equal"); ax[1, 1].legend(fontsize=7); ax[1, 1].set_title("root XY trajectory (top view)")
    # torque / z
    for g in ("A", "B"):
        k = keys[g][0]; ro = Q.load_rollout(load_set(W, k, match(k))[0]); ax[1, 2].plot(t[:len(ro['t'])], ro["root_p"][:, 2], color=col[g], lw=1, label=f"{g} sim root z")
    ax[1, 2].plot(t, refO["root_p"][:len(t), 2], color="#e67e22", lw=0.8, label="ref-O z"); ax[1, 2].plot(t, refF["root_p"][:len(t), 2], color="#2980b9", lw=0.8, label="ref-F z"); ax[1, 2].legend(fontsize=7); ax[1, 2].set_title("root height")
    for a_, ttl in zip(ax.flat, ["root XY error vs own ref (m)", "global MPJPE vs own ref (mm)", "tilt (deg)", "anchored MPJPE vs own ref (mm)"]):
        a_.set_title(ttl) if ttl else None
    ax[0, 0].legend(fontsize=7); ax[0, 2].legend(fontsize=7)
    for a_ in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0]):
        a_.set_xlabel("time in dance (s)"); a_.axvline(t[len(t) // 2], color="k", lw=0.4)
    fig.tight_layout(); fp = out / "deploy_timeseries.png"; fig.savefig(fp, dpi=100); plt.close(fig)
    L += ["## 4. 时间序列图（参考起点；实线=种子0，虚线=种子1；红=A，绿=B；竖线=动作中点）", "", f"![timeseries]({fp.name})", ""]
    if a.video:
        from .replay import render_quad
        from types import SimpleNamespace
        refd = SimpleNamespace(name=a.dance + "_contactfix", padded_npz=PF, padded_dir=PF.parent, npz=PF)
        panels = [(f"B0 {res['models'][keys['B'][0]]['run']} -> ref-F", load_set(W, keys["B"][0], "F", ("E1",))[0] if False else sorted((W / "traj" / keys["B"][0] / "F" / "E1").glob("nominal_ref.npz"))[0]),
                  (f"B1 {res['models'][keys['B'][1]]['run']} -> ref-F", sorted((W / "traj" / keys["B"][1] / "F" / "E1").glob("nominal_ref.npz"))[0]),
                  (f"A0 {res['models'][keys['A'][0]]['run']} (orig data) given same ref-F", sorted((W / "traj" / keys["A"][0] / "F" / "E1").glob("nominal_ref.npz"))[0])]
        v = render_quad(refd, panels, out / "ft_fix_candidate_full.mp4", out / "ft_fix_candidate_sheet.png")
        L += [f"- 全段回放视频（参考 ref-F + 候选 B0/B1 + 原始数据策略同指令）：`{v['video']}`；关键帧 `{v['sheet']}`", ""]
    (out / "FULL_TRACK_REPORT.md").write_text("\n".join(L) + "\n")
    print(out / "FULL_TRACK_REPORT.md")


if __name__ == "__main__":
    main()
