"""V1.5: long-horizon (time-segmented) tracking analysis of the closed-loop rollouts already produced by `./pm quality` (reports/quality/traj/*.npz).

Per ~5 s window of the dance (padding excluded): XY/yaw-anchored 14... MPJPE (evaluator definition), global MPJPE, root XY offset (mean / end of window), growth of the
offset inside the window, yaw offset, root-height offset, joint-angle RMSE, guard rewrite share, torque saturation, penetration, tilt. Also whole-dance cumulative-drift
numbers (offset at 25/50/75/100 %, linear drift slope, first/second half comparison) and a full-playback check (rollout covers the whole padded clip, no fall).
usage: ./pm-segments <dance> [--run-id R] [--it N] [--win 5.0]   (python -m pmpipe.segments)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .core import Dance, read_json, write_json
from . import quality as Q

PAD = Q.PAD


def analyze(run, it: int, win_s: float = 5.0):
    qd = run.reports / "quality"
    files = sorted((qd / "traj").glob(f"it{it:06d}_*.npz"))
    files = [f for f in files if "nominal" in f.name] + [f for f in files if "nominal" not in f.name]
    ref = Q.load_ref(run.dance.padded_npz); K = Q.fk_series(ref["root_p"], ref["root_q"], ref["q"])
    cal = Q.calibrate_floor(K["clr"], ref["jv"], ref["root_p"], ref["fps"]); _, h = Q.contact_tiers(K["clr"], cal["offset"])
    refK = dict(ank=K["ank"], h=h, cal=cal)
    Tpad = len(ref["q"]); fps = ref["fps"]
    rows_all = []; play = []
    for f in files:
        ro = Q.load_rollout(f); T = len(ro["t"]); dc = Q.trajectory_decomposition(ref, refK, ro); s = dc["_series"]
        meta = ro["meta"]
        play.append(dict(file=f.name, frames=T, padded_frames=Tpad, covers_full_clip=bool(T >= Tpad - 1), fall_time=meta.get("fall_time"), alive_at_end=bool(meta.get("fall_time") in (None, -1, "none"))))
        rq = ref["q"][:T]; jerr = np.sqrt(((ro["q"] - rq) ** 2).mean(-1))
        exy = np.linalg.norm(s["exy"], axis=1); slip = np.linalg.norm(s["sp"] - s["sim_dr"], axis=1); rinc = np.linalg.norm(s["ref_dr"] - s["rp"], axis=1)
        dz = ro["root_p"][:, 2] - ref["root_p"][:T, 2]
        n = int(round(win_s * fps)); a0, a1 = PAD, T - PAD; edges = list(range(a0, a1, n)) + [a1]
        rows = []
        for i in range(len(edges) - 1):
            a, b = edges[i], edges[i + 1]
            if b - a < n // 2 and rows:  # merge a short tail into the previous window
                rows[-1]["t1"] = (b - PAD) / fps; continue
            sl = slice(a, b)
            rows.append(dict(t0=(a - PAD) / fps, t1=(b - PAD) / fps, anchored_mpjpe_mm=float(s["anch"][sl].mean() * 1000), global_mpjpe_mm=float(s["glob"][sl].mean() * 1000),
                             root_xy_offset_mean_m=float(exy[sl].mean()), root_xy_offset_end_m=float(exy[b - 1]), root_xy_offset_growth_m=float(exy[b - 1] - exy[a]),
                             policy_slip_drift_end_m=float(slip[b - 1]), ref_inconsistency_end_m=float(rinc[b - 1]),
                             yaw_off_mean_deg=float(np.abs(s["yaw"][sl]).mean()), root_z_off_mm=float(dz[sl].mean() * 1000), joint_rmse_deg=float(np.degrees(jerr[sl].mean())),
                             guard_rewrite_share=float((np.abs(ro["qcmd"][sl] - ro["qpre"][sl]) > 1e-6).any(1).mean()), torque_sat=float(ro["sat_frac"][sl].mean()), pen_max_mm=float(ro["pen"][sl].max() * 1000), tilt_p95_rad=float(np.percentile(ro["tilt"][sl], 95))))
        rows_all.append(dict(file=f.name, rows=rows, drift=dict(
            offset_at_pct_m={str(p): float(exy[PAD + int((T - 2 * PAD) * p / 100) - 1]) for p in (25, 50, 75, 100)},
            policy_slip_at_pct_m={str(p): float(slip[PAD + int((T - 2 * PAD) * p / 100) - 1]) for p in (25, 50, 75, 100)},
            linear_slope_m_per_min=float(np.polyfit(np.arange(a0, a1) / fps, exy[a0:a1], 1)[0] * 60),
            mpjpe_first_half_mm=float(s["anch"][a0:(a0 + a1) // 2].mean() * 1000), mpjpe_second_half_mm=float(s["anch"][(a0 + a1) // 2:a1].mean() * 1000),
            mpjpe_worst_window_mm=float(max(r["anchored_mpjpe_mm"] for r in rows)), max_root_xy_offset_m=float(exy[a0:a1].max()))))
    return dict(iteration=it, win_s=win_s, playback=play, rollouts=rows_all)


def write_report(res, run, path: Path, deploy: dict | None = None):
    nom = res["rollouts"][0]; R = res["rollouts"]
    L = [f"# {run.dance.name} / {run.id} it{res['iteration']}：长时序分时段跟踪分析（V1.5，自动生成）", "",
         "数据：`reports/quality/traj/*.npz`（部署控制链 MuJoCo 闭环，1 个参考起点 + 扰动起点）。锚定 MPJPE = XY/yaw 锚定、保留 root 高度（评估器口径，**不是**全局轨迹误差）；根 XY 偏差 = 仿真根 − 参考根（含参考自身的根平移不一致）。", "",
         "## 1. 完整播放核对", "| rollout | 帧数 | 训练片段帧数 | 覆盖全片 | 倒地时间 |", "|---|---|---|---|---|"]
    L += [f"| {p['file']} | {p['frames']} | {p['padded_frames']} | {'是' if p['covers_full_clip'] else '**否**'} | {p['fall_time']} |" for p in res["playback"]]
    L += ["", f"## 2. 标称起点分时段（{res['win_s']:.0f} s 窗）", "| 时段 s | 锚定MPJPE mm | 全局MPJPE mm | 根XY偏差均值 m | 窗末 m | 窗内增长 m | 策略滑移累计(窗末) m | 参考自身不一致(窗末) m | yaw° | 根高度偏差 mm | 关节RMSE° | 护栏改写 | 力矩饱和 | 穿透max mm | 倾角p95 |", "|" + "---|" * 15]
    for r in nom["rows"]:
        L.append(f"| {r['t0']:.1f}–{r['t1']:.1f} | {r['anchored_mpjpe_mm']:.1f} | {r['global_mpjpe_mm']:.0f} | {r['root_xy_offset_mean_m']:.2f} | {r['root_xy_offset_end_m']:.2f} | {r['root_xy_offset_growth_m']:+.2f} | {r['policy_slip_drift_end_m']:.2f} | {r['ref_inconsistency_end_m']:.2f} | {r['yaw_off_mean_deg']:.1f} | {r['root_z_off_mm']:.0f} | {r['joint_rmse_deg']:.1f} | {r['guard_rewrite_share']:.2f} | {r['torque_sat']:.4f} | {r['pen_max_mm']:.1f} | {r['tilt_p95_rad']:.2f} |")
    nw = len(nom["rows"]); L += ["", f"## 3. 各 rollout（{len(R)} 个）的累计漂移与前后半程", "| rollout | 偏差@25% | @50% | @75% | @100% m | 线性漂移斜率 m/min | 最大偏差 m | MPJPE 前半 | 后半 | 最差窗 mm | 策略滑移@100% m |", "|" + "---|" * 11]
    for r in R:
        d = r["drift"]; o = d["offset_at_pct_m"]
        L.append(f"| {r['file']} | {o['25']:.2f} | {o['50']:.2f} | {o['75']:.2f} | {o['100']:.2f} | {d['linear_slope_m_per_min']:+.2f} | {d['max_root_xy_offset_m']:.2f} | {d['mpjpe_first_half_mm']:.1f} | {d['mpjpe_second_half_mm']:.1f} | {d['mpjpe_worst_window_mm']:.1f} | {d['policy_slip_at_pct_m']['100']:.2f} |")
    ms = np.array([[w["anchored_mpjpe_mm"] for w in r["rows"]] for r in R if len(r["rows"]) == nw])
    L += ["", f"跨 rollout 分时段锚定 MPJPE（mm，均值±std）：" + "，".join(f"{r['t0']:.0f}–{r['t1']:.0f}s {ms[:, i].mean():.1f}±{ms[:, i].std():.1f}" for i, r in enumerate(nom['rows'])), ""]
    if deploy:
        L += ["## 4. 部署包运行窗口", "```json", json.dumps(deploy, ensure_ascii=False, indent=1), "```", ""]
    path.write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dance"); ap.add_argument("--run-id", default=None); ap.add_argument("--it", type=int, default=None); ap.add_argument("--win", type=float, default=5.0)
    a = ap.parse_args()
    run = Dance(a.dance).run(a.run_id)
    it = a.it or int(read_json(run.final / "final_selection.json", {})["primary"])
    res = analyze(run, it, a.win)
    qd = run.reports / "quality"; write_json(qd / "segments.json", res)
    write_report(res, run, run.reports / "SEGMENT_REPORT.md")
    print(run.reports / "SEGMENT_REPORT.md")


if __name__ == "__main__":
    main()
