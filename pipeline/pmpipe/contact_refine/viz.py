"""contact_refine.viz - figures for the report (matplotlib, Agg)."""
from __future__ import annotations

import numpy as np

from .. import quality as Q
from . import geometry as Gm


def _plt():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "Droid Sans Fallback", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
    return plt


def _states(ax, state, st, T, fps, label, x0=0):
    col = {0: "#c62828", 1: "#2e7d32", 2: "#f9a825"}
    img = np.array([[[int(col[s][i:i + 2], 16) / 255 for i in (1, 3, 5)] for s in state[:, f]] for f in (0, 1)])
    ax.imshow(img, aspect="auto", extent=[x0, x0 + T, 0, 2], interpolation="nearest"); ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["R", "L"]); ax.set_title(label, fontsize=9)


def overview(S, name, path, focus=None):
    plt = _plt(); T = len(S["g"]); t = np.arange(T); fig, ax = plt.subplots(7, 1, figsize=(17, 20), sharex=True)
    ax[0].plot(t, S["low0"].min(1) * 1000, "k", lw=0.8, label="原始：两脚最低点高度（相对 z=0）"); ax[0].plot(t, S["g"] * 1000, "r", lw=1.5, label="估计地面曲线 g(t)"); ax[0].legend(fontsize=8); ax[0].set_ylabel("mm"); ax[0].set_title(f"{name}: 悬浮 / 地面漂移")
    for f, c, l in ((0, "tab:blue", "L"), (1, "tab:orange", "R")):
        ax[1].plot(t, S["e0"][:, f] * 1000, c=c, lw=0.8, label=f"{l} 原始(相对 g)"); ax[2].plot(t, S["e1"][:, f] * 1000, c=c, lw=0.8, label=f"{l} 修正后(相对 z=0)")
    ax[1].set_ylim(-20, 100); ax[2].set_ylim(-20, 100); ax[1].legend(fontsize=7, ncol=2); ax[2].legend(fontsize=7, ncol=2); ax[1].set_ylabel("mm"); ax[2].set_ylabel("mm"); ax[1].set_title("脚最低点高度 原始(相对估计地面)"); ax[2].set_title("修正后(相对真实地面 z=0)")
    ax[3].plot(t, S["root0"][:, 2], "k", label="原始根 z"); ax[3].plot(t, S["root1"][:, 2], "r", label="修正后根 z"); ax[3].legend(fontsize=8); ax[3].set_ylabel("m")
    _states(ax[4], S["state0"], S["stance0"], T, 50, "原始接触状态(绿=支撑 红=摆动 黄=不确定)"); _states(ax[5], S["state1"], S["stance1"], T, 50, "修正后接触状态")
    sp0 = np.where(S["o0"], S["slip0"], np.nan); sp1 = np.where(S["o1"], S["slip1"], np.nan)
    v0 = np.nanmax(np.where(S["o0"], S["slip0"], 0), axis=(1, 2)); v1 = np.nanmax(np.where(S["o1"], S["slip1"], 0), axis=(1, 2))
    ax[6].plot(t[1:], v0, "k", lw=0.7, label="原始 支撑接触点滑移(帧内最大)"); ax[6].plot(t[1:], v1, "r", lw=0.7, label="修正后"); ax[6].set_ylim(0, 1.5); ax[6].legend(fontsize=8); ax[6].set_ylabel("m/s"); ax[6].set_xlabel("舞蹈帧(50Hz，不含前导；含 50 帧前导的训练帧号 = 此值 + 50)")
    for a in ax:
        if focus is not None: a.axvspan(focus[0], focus[1], color="b", alpha=0.08)
    fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)


def window(S, M_tilt, name, lo, hi, path, jn=None):
    plt = _plt(); t = np.arange(lo, hi); fig, ax = plt.subplots(6, 1, figsize=(15, 17))
    for f, c, l in ((0, "tab:blue", "L"), (1, "tab:orange", "R")):
        ax[0].plot(t, S["e0"][lo:hi, f] * 1000, c=c, ls="--", label=f"{l} 原始(相对 g)"); ax[0].plot(t, S["e1"][lo:hi, f] * 1000, c=c, label=f"{l} 修正后(z=0)")
    ax[0].plot(t, S["g"][lo:hi] * 1000, "r:", label="地面曲线 g (原始坐标系 z)"); ax[0].legend(fontsize=7, ncol=3); ax[0].set_ylim(-15, 90); ax[0].set_ylabel("脚最低点高度 mm"); ax[0].grid(alpha=0.3)
    ax[1].plot(t, S["root0"][lo:hi, 2], "k--", label="原始根 z"); ax[1].plot(t, S["root1"][lo:hi, 2], "r", label="修正后根 z"); ax[1].plot(t, M_tilt[lo:hi], "b", label="躯干倾角 rad (未改)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    d = (S["root1"] - S["root0"])[lo:hi]; ax[2].plot(t, d[:, 0] * 1000, label="Δx"); ax[2].plot(t, d[:, 1] * 1000, label="Δy"); ax[2].plot(t, d[:, 2] * 1000, label="Δz"); ax[2].legend(fontsize=8); ax[2].set_ylabel("根修正 mm"); ax[2].grid(alpha=0.3)
    dq = np.degrees(S["q1"] - S["q0"])[lo:hi, :12]
    for j in range(12): ax[3].plot(t, dq[:, j], lw=1, label=(jn[j].replace("_joint", "") if jn else str(j)))
    ax[3].legend(fontsize=6, ncol=6); ax[3].set_ylabel("腿关节改动 °"); ax[3].grid(alpha=0.3)
    for a_, key, c in ((ax[4], "0", "k"), (ax[4], "1", "r")):
        s = np.where(S["o" + key], S["slip" + key], 0).max(axis=(1, 2)); a_.plot(t, s[lo - 1:hi - 1], c=c, label="原始" if key == "0" else "修正后")
    ax[4].set_ylim(0, 1.2); ax[4].legend(fontsize=8); ax[4].set_ylabel("支撑接触点滑移 m/s"); ax[4].grid(alpha=0.3)
    _states(ax[5], np.concatenate([S["state0"][lo:hi], S["state1"][lo:hi]], 1)[:, [0, 1]] if False else S["state0"][lo:hi], None, hi - lo, 50, "接触状态（原始识别；修正后识别几乎相同，见全段图）", x0=lo); ax[5].set_xlim(lo, hi)
    for a in ax[:5]: a.set_xlim(lo, hi)
    ax[0].set_title(f"{name}: 舞蹈帧 {lo}–{hi}（含前导 50 帧的训练帧号 {lo + 50}–{hi + 50}）")
    ax[5].set_xlabel("舞蹈帧")
    fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)


def joint_delta(M_names, q0, q1, path):
    plt = _plt(); d = np.degrees(np.abs(q1 - q0)); fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].bar(range(29), np.sqrt((d ** 2).mean(0)), label="RMS"); ax[0].plot(range(29), d.max(0), "r_", ms=10, label="max"); ax[0].set_xticks(range(29)); ax[0].set_xticklabels([n.replace("_joint", "") for n in M_names], rotation=90, fontsize=7); ax[0].legend(); ax[0].set_ylabel("°"); ax[0].set_title("逐关节改动量（上肢/腰 = 0）")
    ax[1].hist(d[:, :12].max(1), bins=40); ax[1].set_title("每帧腿关节最大改动分布 °"); fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def root_path(S, res, path):
    plt = _plt(); fig, ax = plt.subplots(1, 2, figsize=(14, 6)); r0, r1 = S["root0"], S["root1"]
    ax[0].plot(r0[:, 0], r0[:, 1], "k", lw=1, label="原始根 XY"); ax[0].plot(r1[:, 0], r1[:, 1], "r", lw=1, label="修正后根 XY"); ax[0].set_aspect("equal"); ax[0].legend(); ax[0].set_title("根水平轨迹（俯视）")
    d = r1 - r0; ax[1].plot(np.linalg.norm(d[:, :2], axis=1), label="|Δxy| m"); ax[1].plot(d[:, 2], label="Δz m"); ax[1].legend(); ax[1].set_title("根修正量"); fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def variants_bar(res_main, variants, path):
    plt = _plt(); names = ["原始", "V1.2 repair(仅根)", "stage1 根XYZ(仅根)", "contactfix(根+腿IK)"]
    def get(v, k): 
        for kk in k.split("."): v = v[kk]
        return v
    sets = [("slip.refined.p95", "支撑接触点滑移 p95 m/s"), ("ground.stance_abs_height_p95_mm", "支撑脚|离地高度| p95 mm"), ("ground.penetration_max_mm", "最大穿透 mm"), ("root.dv_p99_mps", "根速度改动 p99 m/s")]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    for a, (k, ttl) in zip(ax, sets):
        vals = [None, get(variants["repair_v12_root_only"], k) if "repair_v12_root_only" in variants else None, get(variants["root_only_stage1"], k), get(res_main, k)]
        if k == "slip.refined.p95": vals[0] = res_main["slip"]["original"]["p95"]
        elif k == "ground.stance_abs_height_p95_mm": vals[0] = res_main["ground"]["original_vs_z0"]["stance_abs_height_p95_mm"]
        elif k == "ground.penetration_max_mm": vals[0] = res_main["ground"]["original_vs_z0"]["penetration_max_mm"]
        else: vals[0] = 0.0
        a.bar(range(4), [0 if v is None else v for v in vals], color=["0.6", "tab:orange", "tab:blue", "tab:green"]); a.set_xticks(range(4)); a.set_xticklabels(names, rotation=20, fontsize=7); a.set_title(ttl, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def analysis_overview(A0, M, name, path, focus=None):
    """analyze-only figure (original data only)"""
    plt = _plt(); I = A0["ident"]; T = len(A0["g"]); t = np.arange(T); fig, ax = plt.subplots(5, 1, figsize=(17, 14), sharex=True)
    ax[0].plot(t, A0["low"].min(1) * 1000, "k", lw=0.8, label="两脚最低点高度(相对 z=0)"); ax[0].plot(t, A0["g"] * 1000, "r", lw=1.5, label="估计地面曲线 g(t)"); ax[0].plot(t, A0["ground"]["conf"] * 20, "b", lw=0.6, label="地面置信度×20"); ax[0].legend(fontsize=8); ax[0].set_ylabel("mm"); ax[0].set_title(f"{name}: 原始数据诊断")
    for f, c, l in ((0, "tab:blue", "L"), (1, "tab:orange", "R")): ax[1].plot(t, I["feat"]["e"][:, f] * 1000, c=c, lw=0.8, label=f"{l} 最低足底点相对地面曲线")
    ax[1].set_ylim(-20, 100); ax[1].legend(fontsize=8); ax[1].set_ylabel("mm")
    _states(ax[2], I["state"], I["stance"], T, 50, "接触状态(绿=支撑 红=摆动 黄=不确定)"); ax[3].plot(t, M["root_p"][:, 2], "k"); ax[3].set_ylabel("根 z m")
    fl = np.where(I["flight"]["mask"], 1, 0) - np.where(I["flight"]["false_mask"], 1, 0); ax[4].plot(t, fl, "m"); ax[4].set_yticks([-1, 0, 1]); ax[4].set_yticklabels(["假腾空(悬浮)", "—", "真腾空"]); ax[4].set_xlabel("舞蹈帧")
    if focus:
        for a in ax: a.axvspan(focus[0], focus[1], color="b", alpha=0.08)
    fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)
