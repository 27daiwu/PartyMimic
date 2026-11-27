"""V1.7 C2: judge + figures of the short ablation -> results/<dance>/v17/C_train/eval/{judge.json, fig_*.png}"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import v17eval as VE
from .core import Dance

d = Dance(sys.argv[1]); ev = d.dir / "v17/C_train/eval"; M = json.load(open(ev / "metrics.json")); P = json.load(open(ev / "posture.json"))
ctrl3 = [M[k]["agg"] for k in ("v17_A0", "ft_orig_v16@41000", "ft_orig_v16_s1@41000")]  # three same-data, same-length (1000 it) control samples: noise band = their range
noise = {k: max(max(c[k] for c in ctrl3) - min(c[k] for c in ctrl3), M["noise_V16"]["band"].get(k, 0.0)) for k in ctrl3[0] if isinstance(ctrl3[0][k], float)}
agg = {k: v["agg"] for k, v in M.items() if k.startswith("v17_") or k.startswith("ft_fix")}
pairs = [("A1 vs A0 (same ref-O: labels+feature)", "v17_A0", "v17_A1"), ("A2 vs A0 (matched deployment: +data)", "v17_A0", "v17_A2"), ("A2 vs A1 (same labels, data O->F)", "v17_A1", "v17_A2"),
         ("A3 vs A2 (reward terms a,b,c)", "v17_A2", "v17_A3"), ("A4 vs A3 (terms d,e)", "v17_A3", "v17_A4"), ("A4 vs A0 (all)", "v17_A0", "v17_A4"), ("A2 vs ftfix@41000 (same fixed data+1000it; legacy->refine labels+sole feature)", "ft_fix_v16@41000", "v17_A2"), ("ftfix@41000 vs A0 (data only: legacy labels)", "v17_A0", "ft_fix_v16@41000"), ("A4 vs ftfix@41000", "ft_fix_v16@41000", "v17_A4")]
CROSS = {"v17_A0|v17_A2", "v17_A0|v17_A4", "v17_A1|v17_A2", "v17_A0|ft_fix_v16@41000"}  # different references: anchored MPJPE / label agreement are not comparable (V1.6 rule)
out = {}
for name, c, v in pairs:
    j = VE.judge(agg[c], agg[v], noise)
    if f"{c}|{v}" in CROSS:
        for k in ("mpjpe", "agree_main"):
            j["metrics"][k]["status"] = "n/a (different references)"
        j["degraded"] = [k for k in j["degraded"] if k not in ("mpjpe", "agree_main")]; j["clear_benefit"] = bool(len(j["improved_key"]) >= 2 and not j["degraded"])
    out[name] = j
json.dump(dict(pairs=out, noise=noise, agg=agg, posture=P), open(ev / "judge.json", "w"), indent=1)
for n, j in out.items():
    print(n, "| improved key:", j["improved_key"], "| degraded:", j["degraded"], "| CLEAR BENEFIT:", j["clear_benefit"])
    print("   ", {k: (round(v["ctrl"], 3), round(v["var"], 3), v["status"][:4]) for k, v in j["metrics"].items() if k in VE.KEY3 + ["slip_stable_p95", "slip_fixed_mean", "slip_trans_p95", "drift_win_mean", "D2_pen_p95_mm", "mpjpe"]})
json.dump(noise, open(ev / "noise_band.json", "w"), indent=1)
# figures
runs = ["v17_A0", "v17_A1", "ft_fix_v16@41000", "v17_A2", "v17_A3", "v17_A4"]; lab = [r[4:] if r.startswith("v17") else "ftfix\n(legacy,F)" for r in runs]; col = ["gray", "tab:blue", "tab:olive", "tab:green", "tab:orange", "tab:red"]
fig, ax = plt.subplots(2, 4, figsize=(17, 7.5))
for a, (k, t) in zip(ax.ravel(), [("slip_stable_mean", "stable slip mean [m/s] (key)"), ("slip_fixed_mean", "stable slip, fixed sample id [m/s]"), ("flicker", "load flips /s/foot (key)"), ("drift_win_25_45", "root drift growth / 5 s window, 25-45 s [m] (key)"),
                                            ("slip_trans_p95", "transition slip p95 [m/s]"), ("D2_pen_p95_mm", "penetration p95 [mm]"), ("tilt_loaded_deg", "loaded-foot sole tilt [deg]"), ("agree_main", "load vs ref contact agreement")]):
    v = [agg[r][k] for r in runs]; a.bar(lab, v, color=col); n0 = noise.get(k, 0); a.axhspan(agg["v17_A0"][k] - n0, agg["v17_A0"][k] + n0, color="k", alpha=0.12)
    a.axhline(agg["v17_A0"][k], color="k", lw=0.6, ls="--"); a.set_title(t, fontsize=9)
    for i, x in enumerate(v): a.text(i, x, f"{x:.3g}", ha="center", va="bottom", fontsize=7)
fig.suptitle("V1.7 C2 short ablation (1000 it, seed 0; 17 nominal rollouts, deployment chain; dashed = A0, grey band = range of 3 same-config controls (A0, ft_orig_v16, ft_orig_v16_s1 @it41000); A0/A1 on ref-O, A2-A4 on ref-F)"); fig.tight_layout(); fig.savefig(ev / "fig_metrics.png", dpi=110)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
keys = [r if r.startswith("v17") else "ft_fix_v16@41000" for r in runs] + ["V16_A0", "V16_B0"]; l2 = lab + ["ft_orig\n@50k", "ft_fix\n@50k"]
ax[0].bar(l2, [P[k]["partial_to_flat"] for k in keys], color=col + ["lightgray", "lightgreen"]); ax[0].set_title("share of reference heel-only / toe-only stance frames\nwhere the policy sole is FLAT (flattening bias)"); 
ax[1].bar(l2, [P[k]["area_ratio_med"] for k in keys], color=col + ["lightgray", "lightgreen"]); ax[1].set_title("median support-area ratio policy / reference")
for a in ax: a.tick_params(axis="x", labelsize=7)
fig.tight_layout(); fig.savefig(ev / "fig_posture.png", dpi=110)
