"""analyze: (a) real-robot / Sim2Real run logs of the C++ package -> the framework's own analysis tool; (b) training + evaluation summary of a run.

(a) reuses deploy/partymimic_cpp/analysis/pm_analyze.py unchanged (replay parity, tracking, PD, latency, safety clips, exit diagnostics) on every run directory found in
    --logs (a run dir, a logs/ tree, or the tar.gz written by the package's export_logs.sh). Nothing is ever sent to a robot.
(b) summary of train_log.csv + eval/summary.csv + selection/admission, written to <run>/analysis/TRAINING_ANALYSIS.md (+ curve figure).
"""
from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

from .core import CPP, PY, Run, die, now, read_csv, read_json, sh, write_json


def find_run_dirs(p: Path) -> list[Path]:
    if (p / "run_report.json").exists() or (p / "trace.csv").exists():
        return [p]
    return sorted({r.parent for r in p.rglob("run_report.json")} | {r.parent for r in p.rglob("trace.csv")})


def analyze_logs(run: Run, logs: str, onnx: str | None = None, modes: str = "all") -> dict:
    src = Path(logs).expanduser()
    if not src.exists():
        die(f"--logs path not found: {src}")
    out_root = run.analysis / "robot_logs"
    out_root.mkdir(parents=True, exist_ok=True)
    if src.is_file() and (src.suffix == ".gz" or src.suffix == ".tar" or src.name.endswith(".tar.gz")):
        dst = out_root / "_extracted" / src.stem
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)
        with tarfile.open(src) as t:
            t.extractall(dst, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(dst)
        src = dst
    runs = find_run_dirs(src)
    if not runs:
        die(f"no run directories (run_report.json / trace.csv) under {src}")
    adm = run.admission()
    onnx = onnx or (adm.get("final") or {}).get("onnx")
    results = []
    for rd in runs:
        o = out_root / rd.name
        args = [PY, CPP / "analysis/pm_analyze.py", rd, modes, "--out", o]
        if onnx and modes in ("all", "replay"):
            args += ["--onnx", onnx]
        r = sh(args, capture=True)
        (o.parent / f"{rd.name}.analyze.log").write_text(r.stdout + r.stderr)
        rep = read_json(rd / "run_report.json", {})
        results.append(dict(run=rd.name, rc=r.returncode, result=rep.get("result"), exit_reason=rep.get("exit_reason"), report=str(o / "analysis_report.md")))
        print(f"[analyze] {rd.name}: rc={r.returncode} result={rep.get('result')} -> {o}")
    write_json(out_root / "index.json", dict(time=now(), source=str(logs), runs=results))
    return dict(out=str(out_root), runs=results)


def analyze_training(run: Run) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read_csv(run.train / "train_log.csv")
    ev = read_csv(run.eval / "summary.csv")
    adm = run.admission()
    run.analysis.mkdir(parents=True, exist_ok=True)
    L = [f"# {run.dance.name} / {run.id} 训练与评估分析（{now()}）", ""]
    if rows:
        f = lambda k: [float(r[k]) for r in rows]
        it = f("iteration")
        L += [f"- 训练到 it {int(it[-1])}，墙钟 {float(rows[-1]['wall_h']):.2f} h，末段 {float(rows[-1]['it_per_s']):.1f} it/s；末行 sim_rew {float(rows[-1]['sim_rew']):.3f}，term_frac {float(rows[-1]['term_frac']):.3f}，"
              f"非有限值次数 {int(sum(f('nonfinite')))}"]
        fig, ax = plt.subplots(1, 3 if ev else 2, figsize=(15 if ev else 10, 4))
        ax[0].plot(it, f("sim_rew"), label="sim_rew"); ax[0].plot(it, f("imag_survival"), label="imag_survival"); ax[0].legend(); ax[0].set_title("training reward / survival")
        ax[1].semilogy(it, f("wm_loss"), label="wm"); ax[1].semilogy(it, f("v_loss"), label="v"); ax[1].legend(); ax[1].set_title("losses")
        if ev:
            e_it = [int(r["iteration"]) for r in ev]
            for k in ("balance", "fidelity", "combined"):
                ax[2].plot(e_it, [float(r[k]) for r in ev], "o-", label=k)
            ax[2].legend(); ax[2].set_title("independent evaluation (0 = pretrained baseline)")
        for a in ax:
            a.set_xlabel("iteration"); a.grid(alpha=.3)
        fig.tight_layout(); png = run.analysis / "training_curves.png"; fig.savefig(png, dpi=100); plt.close(fig)
        L += ["", f"![curves]({png.name})", ""]
    if ev:
        L += ["## 周期评估", "", "| it | 平衡 | 还原 | 综合 | 鲁棒存活 | nominal MPJPE mm | 力矩饱和 | 穿透 mm |", "|---|---|---|---|---|---|---|---|"]
        for r in ev:
            L.append(f"| {r['iteration']} | {float(r['balance']):.1f} | {float(r['fidelity']):.1f} | {float(r['combined']):.1f} | {float(r['survival_robust']):.2f} | {float(r['mpjpe_nominal_mm']):.1f} | {float(r['torque_sat_mean']):.3f} | {float(r['pen_max_mm']):.1f} |")
    if adm:
        L += ["", "## 准入", "", f"- 真机准入（离线门槛）：**{adm.get('real_robot_admission')}**；最终模型 it{adm.get('final', {}).get('iteration', '-')}",
              f"- Sim2Sim：{'PASS' if (adm.get('sim2sim') or {}).get('passed') else 'FAIL/无'}；详见 `reports/SELECTION_REPORT.md`"]
    p = run.analysis / "TRAINING_ANALYSIS.md"
    p.write_text("\n".join(L) + "\n")
    return p
