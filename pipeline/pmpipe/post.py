"""Post-training: candidate screening -> hard re-evaluation -> balance-first selection -> ONNX export + parity -> deploy assets -> Sim2Sim -> admission.

Generalised from results/ythn/scripts/ythn_post.py / ythn_gate.py (same re-evaluation protocol and file formats: final/reeval/ckpt_<it>.json),
with the selection rule of the project: DYNAMIC BALANCE FIRST, tracking fidelity second.

Rule (eval config `selection`):
  eligible  = nominal survival == 1.0 and robust survival >= robust_survival_min   (on the hard re-evaluation)
  among eligible: best balance score B*; every candidate with balance >= B* - balance_tolerance is "equally balanced";
  the winner is the equally-balanced candidate with the highest fidelity (ties: higher balance).
  The same rule applied to the remaining candidates gives the fall-back order used when Sim2Sim rejects the winner.
Sim2Sim never gets overruled: admission (pm/admission.json) is only 'PASS' when the winner's Sim2Sim passes the gate.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from .core import PY, REPO, S2R_PY, Logger, Run, die, now, py_env, read_csv, read_json, sh, write_json

SIM2SIM_BENCH = REPO / "pipeline/pmpipe/sim2sim_bench.py"
DEPLOY_ASSETS = REPO / "pipeline/pmpipe/deploy_assets.py"


# ----------------------------------------------------------------------------------------------------------------------------- selection
def candidates(run: Run, ecfg: dict) -> dict[int, list[str]]:
    h = ecfg["hard"]
    rows = [r for r in read_csv(run.eval / "summary.csv") if int(r["iteration"]) > 0]
    if not rows:
        die(f"no evaluation summary in {run.eval}: run `./pm eval {run.dance.name}` first")
    ok = [r for r in rows if float(r["nominal_survival"]) >= 1.0] or rows
    c: dict[int, list[str]] = {}
    for k, n in (("combined", h["top_combined"]), ("balance", h["top_balance"]), ("fidelity", h["top_fidelity"])):
        for r in sorted(ok, key=lambda r: float(r[k]), reverse=True)[:n]:
            c.setdefault(int(r["iteration"]), []).append(k)
    last = max(int(r["iteration"]) for r in rows)
    c.setdefault(last, []).append("last")
    return dict(sorted(c.items()))


def hard_reeval(run: Run, cand: dict, ecfg: dict, log) -> dict[int, dict]:
    from . import evaluator as E
    h = ecfg["hard"]
    (run.final / "reeval").mkdir(parents=True, exist_ok=True)
    motion_dir = str(run.motion_padded_dir)
    res = {}
    for it in [0] + list(cand):
        dst = run.final / f"reeval/ckpt_{it:06d}.json"
        if not dst.exists():
            src = run.init_pt if it == 0 else run.ckpt(it)
            r = E.evaluate(str(src), motion_dir, h["reps"], h["seed"], hard=True)
            write_json(dst, r)
            s = r["scores"]
            log(f"re-eval it {it}: balance {s['balance']:.1f} fidelity {s['fidelity']:.1f} combined {s['combined']:.1f} surv {s['survival_robust']:.2f}")
        res[it] = json.loads(dst.read_text())
    return res


def rank(R: dict[int, dict], sel: dict, exclude=()) -> list[int]:
    """Balance-first ordering of the hard re-evaluation results (the single rule lives in pmpipe.selection)."""
    from .selection import rank_scores
    return rank_scores({it: dict(scores=r["scores"], nominal_survival=r["conditions"]["nominal"]["survival"]) for it, r in R.items() if it not in exclude}, sel)


def keep_legacy(path: Path, marker: str = "rule"):
    """A pre-pipeline file with the same name but another format is kept next to the new one (never destroyed)."""
    if path.exists() and marker not in path.read_text()[:2000]:
        dst = path.with_name(path.stem + ".legacy" + path.suffix)
        if not dst.exists():
            shutil.copy2(path, dst)


def select(run: Run, R: dict[int, dict], ecfg: dict, log) -> dict:
    sel = ecfg["selection"]
    order = rank(R, sel)
    eligible = bool(order)
    if not order:  # nobody passes the eligibility floor: still report the best-balanced one, flagged NOT eligible (admission will fail)
        order = sorted([it for it in R if it > 0], key=lambda it: (R[it]["scores"]["survival_robust"], R[it]["scores"]["balance"]), reverse=True)
    roles = {}
    for k in ("balance", "fidelity", "combined"):
        roles[k] = max((it for it in R if it > 0), key=lambda it: R[it]["scores"][k])
    out = dict(rule="balance_first", tolerance=sel["balance_tolerance"], eligible_floor=sel["eligible"], any_eligible=eligible, primary=order[0], order=order,
               roles=roles, scores={str(it): R[it]["scores"] for it in R}, time=now())
    keep_legacy(run.final / "final_selection.json")
    write_json(run.final / "final_selection.json", out)
    log("selection", primary=order[0], order=order, roles=roles, eligible=eligible)
    return out


# ----------------------------------------------------------------------------------------------------------------------------- export / sim2sim
def export_model(run: Run, it: int, ecfg: dict, dcfg: dict, log) -> dict:
    """ONNX export of ckpt `it` + ONNX parity + deploy assets (policy_params.txt / motion.bin ...). Returns the artifact dict."""
    d = run.final / "onnx"
    d.mkdir(parents=True, exist_ok=True)
    onnx = d / f"{run.dance.name}_it{it:06d}.onnx"
    src = run.ckpt(it)
    if not src.exists():
        die(f"checkpoint missing: {src}")
    (run.final / "models").mkdir(exist_ok=True)
    mdl = run.final / f"models/{run.dance.name}_it{it:06d}.pt"
    if not mdl.exists():
        shutil.copy2(src, mdl)
    info_p = d / f"{onnx.stem}.export.json"
    if not onnx.exists() or not info_p.exists():
        r = sh([PY, REPO / "tools/export_deploy_onnx.py", mdl, onnx], env=py_env(), capture=True)
        (d / f"{onnx.stem}.export.log").write_text(r.stdout + r.stderr)
        if r.returncode:
            die(f"ONNX export failed (see {d / (onnx.stem + '.export.log')})", 3)
        last = [l for l in r.stdout.splitlines() if l.startswith("{")][-1]
        write_json(info_p, json.loads(last))
    info = read_json(info_p)
    v = sh([PY, REPO / "results/p0/verify_deploy_onnx.py", onnx], capture=True)
    (d / f"{onnx.stem}.verify.log").write_text(v.stdout + v.stderr)
    md = [float(l.split()[1]) for l in v.stdout.splitlines() if l.startswith("max_abs_diff")]
    tol = dcfg.get("onnx_parity_max_abs", 1e-3)
    parity = dict(pt_vs_onnx_max_abs=md[0] if md else None, verify_pass="DEPLOY_ONNX_INFERENCE=PASS" in v.stdout,
                  policy_vs_training_max_err=info.get("policy_vs_training_max_err"), geometry_max_err=info.get("geometry_max_err"), tol=tol)
    parity["pass"] = bool(parity["verify_pass"] and (md and md[0] <= tol) and (info.get("policy_vs_training_max_err") or 0) <= tol * 10)
    # deploy assets from exactly this onnx + the unpadded dance (what the package will contain)
    a = run.final / f"assets/it{it:06d}"
    r = sh([S2R_PY, DEPLOY_ASSETS, "--onnx", onnx, "--yaml", onnx.with_suffix(".yaml"), "--motion", run.motion_npz, "--out", a,
            "--provenance", f"run {run.dance.name}/{run.id} ckpt it{it} sha256 {mdl.name}; yaml+motion {run.motion_npz.name}"], capture=True)
    if r.returncode:
        die("deploy asset generation failed:\n" + r.stdout[-1500:] + r.stderr[-1500:], 3)
    log(f"export it{it}: onnx parity {parity}")
    return dict(iteration=it, checkpoint=str(mdl), onnx=str(onnx), yaml=str(onnx.with_suffix(".yaml")), assets=str(a), parity=parity)


REAL_POSES = 10  # eval_bench.sample_inits(n) always returns the 10 archived real-robot standing poses (5 ready.json + 5 first-tick rows) + n sampled states


def sim2sim_counts(s: dict, n: int | None = None) -> dict:
    """Sim2Sim initial-state accounting. `n` (CLI --n / config sim2sim.n) = number of SAMPLED states; the bench adds the real poses => total = n + real_poses (V1.0/V1.1 wrongly called this '14')."""
    real = int(s.get("real_poses", REAL_POSES))
    sampled = int(n if n is not None else s.get("sampled", s["n"]))
    if n is None and "total" in s and int(s["total"]) != real + sampled:
        die(f"eval config sim2sim: total={s['total']} != real_poses({real}) + sampled({sampled})", 2)
    return dict(real_poses=real, sampled=sampled, total=real + sampled)


def sim2sim(run: Run, art: dict, ecfg: dict, log, procs: int = 16, n: int | None = None, conds: list | None = None) -> dict:
    s = ecfg["sim2sim"]
    cnt = sim2sim_counts(s, n)
    it = art["iteration"]
    out = run.final / f"sim2sim/{run.dance.name}_it{it:06d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    nn, cc = cnt["sampled"], conds or s["conds"]
    if out.exists():
        old = read_json(out)
        if old and old.get("args", {}).get("n") == nn and old.get("args", {}).get("conds") == ",".join(cc):
            return old
    r = sh([S2R_PY, SIM2SIM_BENCH, "--padded-npz", run.motion_padded_npz, "--params", Path(art["assets"]) / "policy_params.txt",
            "--onnx", art["onnx"], "--out", out, "--n", nn, "--conds", ",".join(cc), "--padded", "--mode", "running", "--auth", "1.0", "--ramp", "0.1", "--procs", procs],
           capture=True)
    (out.with_suffix(".log")).write_text(r.stdout + r.stderr[-3000:])
    if r.returncode or not out.exists():
        die(f"Sim2Sim failed (rc={r.returncode}): see {out.with_suffix('.log')}", 3)
    log(f"sim2sim it{it} done ({cnt['total']} initial states per condition = {cnt['real_poses']} real standing poses + {cnt['sampled']} sampled)")
    return read_json(out)


def sim2sim_gate(res: dict, ecfg: dict) -> dict:
    g = ecfg["sim2sim"]["gate"]
    S = res["summary"]
    succ = {c: v["success"] for c, v in S.items()}
    checks = {
        f"nominal >= {g['nominal_success_min']}": succ.get("nominal", 0) >= g["nominal_success_min"],
        f"each core cond >= {g['core_success_min']}": all(succ.get(c, 0) >= g["core_success_min"] for c in g["core_conds"] if c in succ),
        f"mean over all conds >= {g['mean_success_min']}": (sum(succ.values()) / max(len(succ), 1)) >= g["mean_success_min"],
    }
    sampled = int((res.get("args") or {}).get("n", ecfg["sim2sim"].get("sampled", ecfg["sim2sim"]["n"])))
    real = int(ecfg["sim2sim"].get("real_poses", REAL_POSES))
    observed = sorted({int(v["n"]) for v in S.values() if "n" in v})
    warns = []
    if observed != [real + sampled]:
        warns.append(f"Sim2Sim summary.n per condition = {observed} but declared total = {real} real + {sampled} sampled = {real + sampled}: initial-state accounting is inconsistent (failed jobs or changed real-pose set?)")
    return dict(passed=all(checks.values()), checks=checks, success=succ, mean=sum(succ.values()) / max(len(succ), 1),
                worst=min(succ.items(), key=lambda kv: kv[1]) if succ else None,
                inits=dict(real_poses=real, sampled=sampled, total=real + sampled, observed_n_per_condition=observed, consistent=not warns), warnings=warns)


# ----------------------------------------------------------------------------------------------------------------------------- orchestration
def admission_provenance(run: Run, R: dict, primary: int, ecfg: dict) -> dict:
    """What the admission numbers were measured with (V1.2): evaluator version, hard re-evaluation seed/reps/grid, Sim2Sim initial-state accounting and the training source state."""
    try:
        from .report import EVALUATOR_VERSION
    except Exception:
        EVALUATOR_VERSION = "unknown"
    r = R.get(primary) or {}
    out = dict(evaluator_version=EVALUATOR_VERSION,
               hard_reeval=dict(seed=r.get("seed", ecfg["hard"]["seed"]), reps=r.get("reps", ecfg["hard"]["reps"]), conditions=len(r.get("conditions", {})), grid="CONDS + CONDS_HARD (beyond the training randomisation)"),
               sim2sim_inits=sim2sim_counts(ecfg["sim2sim"]), sim2sim_inits_note="total = real_poses + sampled; `--n` is the sampled part only")
    try:
        from .core import read_json as _rj
        from .repro import GROUPS, compute_state
        man = _rj(run.root / "repro/MANIFEST.json", {})
        if man:
            live = compute_state(REPO)
            out["train_source"] = dict(git_head=man["head"], src_tools_diff_sha12=man["legacy_src_tools_diff_sha12"], train_patch_sha256=man["patches"]["train.patch"]["sha256"], train_tree_sha256=man["trees"]["train_tree"]["sha256"],
                                       repro_snapshot=str(run.root / "repro"), snapshot_created=man["created"], source_unchanged_since_first_launch=man.get("train_source_unchanged_since_first_launch"),
                                       work_tree_train_tree_matches_snapshot=live["trees"]["train_tree"]["sha256"] == man["trees"]["train_tree"]["sha256"])
        else:
            out["train_source"] = dict(repro_snapshot=None, note="no repro snapshot: run `./pm snapshot <dance>`")
    except Exception as e:
        out["train_source"] = dict(error=repr(e))
    return out


def select_and_validate(run: Run, ecfg: dict, dcfg: dict, log=None, procs: int = 16, fast: bool = False, only_it: int | None = None) -> dict:
    """The whole offline post chain. Idempotent (every result file is reused). Writes pm/admission.json."""
    log = log or Logger(run.logs / "post.log")
    run.mk()
    cand = candidates(run, ecfg) if only_it is None else {int(only_it): ["accepted (V1.6 ftcompare)"]}  # V1.6: export exactly the checkpoint that was accepted, no re-selection
    write_json(run.final / "candidates.json", cand)
    log("candidates", cand)
    R = hard_reeval(run, cand, ecfg, log)
    sel = select(run, R, ecfg, log)
    adm = run.admission()
    adm.update(dance=run.dance.name, run=run.id, selection=dict(primary=sel["primary"], order=sel["order"], any_eligible=sel["any_eligible"]),
               eval_gate=dict(passed=bool(sel["any_eligible"]), scores=R[sel["primary"]]["scores"]), sim2sim=None, real_robot_admission="PENDING", note="PASS = all offline gates (eval, ONNX parity, Sim2Sim) passed = allowed to enter the supervised real-robot trial; it is NOT a real-robot result",
               updated=now())
    tried = []
    chosen = None
    for it in sel["order"][: ecfg["sim2sim"].get("max_candidates", 3)]:
        art = export_model(run, it, ecfg, dcfg, log)
        res = sim2sim(run, art, ecfg, log, procs=procs)
        gate = sim2sim_gate(res, ecfg)
        tried.append(dict(iteration=it, parity=art["parity"], sim2sim=gate))
        log(f"sim2sim gate it{it}: {'PASS' if gate['passed'] else 'FAIL'}", mean=round(gate["mean"], 3), worst=gate["worst"])
        for w in gate.get("warnings", []):
            log("WARNING:", w)
        if gate["passed"] and art["parity"]["pass"] and sel["any_eligible"]:
            chosen = (it, art, gate)
            break
    adm["provenance"] = admission_provenance(run, R, sel["primary"], ecfg)
    adm["candidates_tried"] = tried
    if chosen:
        it, art, gate = chosen
        adm.update(final=dict(iteration=it, **{k: art[k] for k in ("checkpoint", "onnx", "yaml", "assets")}), onnx_parity=art["parity"], sim2sim=gate)
        adm["real_robot_admission"] = "PASS"  # offline gates only: precondition for a supervised real-robot trial, NOT a real-robot result
        log("ADMISSION (offline): PASS", final=it)
    else:
        first = tried[0] if tried else None
        if first:
            art0 = export_model(run, first["iteration"], ecfg, dcfg, log)
            adm.update(final=dict(iteration=first["iteration"], **{k: art0[k] for k in ("checkpoint", "onnx", "yaml", "assets")}), onnx_parity=art0["parity"], sim2sim=first["sim2sim"])
        adm["real_robot_admission"] = "FAILED" if sel["any_eligible"] else "NOT_ELIGIBLE"
        log("ADMISSION: FAIL (Sim2Sim / parity / eligibility not satisfied) - package will refuse without --allow-unadmitted")
    run.save_admission(adm)
    run.mark("post_done")
    write_selection_report(run, R, sel, adm, ecfg)
    return adm


def write_selection_report(run: Run, R, sel, adm, ecfg):
    L = [f"# {run.dance.name} / {run.id}: 筛选 · 导出 · Sim2Sim 准入（自动生成 {now()}）", "",
         "> 规则：**动态平衡优先，其次还原度**。硬复评（独立种子、更多重复、超出训练分布的工况）上：nominal 全存活且鲁棒存活≥"
         f"{ecfg['selection']['eligible']['robust_survival_min']}为合格；合格者中平衡分在最优 {ecfg['selection']['balance_tolerance']} 分以内视为同档，同档取还原度最高。",
         "> **Sim2Sim 未通过 = 不得标记真机准入通过**。这里的 PASS 只代表离线门槛（评估+ONNX一致性+Sim2Sim）全部通过，允许进入有保护的真机试验，不是真机结论。", "",
         f"## 结论：`{adm['real_robot_admission']}`  最终模型 it{adm.get('final', {}).get('iteration', '-')}", "",
         "| it | 平衡 | 还原 | 综合 | 鲁棒存活 | nominal MPJPE mm | 力矩饱和 | 穿透 mm | 角色 |", "|---|---|---|---|---|---|---|---|---|"]
    roles = {}
    for k, it in sel["roles"].items():
        roles.setdefault(it, []).append(k)
    for it in sorted(R):
        s = R[it]["scores"]
        tag = ("基线(预训练)" if it == 0 else "") + ("★最终 " if it == adm.get("final", {}).get("iteration") else "") + ",".join(roles.get(it, []))
        L.append(f"| {it} | {s['balance']:.1f} | {s['fidelity']:.1f} | {s['combined']:.1f} | {s['survival_robust']:.2f} | {s['mpjpe_nominal_mm']:.1f} | {s['torque_sat_mean']:.3f} | {s['pen_max_mm']:.1f} | {tag} |")
    L += ["", "## Sim2Sim（部署链闭环）", ""]
    for t in adm.get("candidates_tried", []):
        g = t["sim2sim"]
        ini = g.get("inits")
        if ini:
            L.append(f"- 初态口径：每工况 **{ini['total']} 个初态 = {ini['real_poses']} 个真机 WalkRun 站姿 + {ini['sampled']} 个插值采样**（`--n` 只是采样数；V1.0/V1.1 文案写作“14 初态”是误写）" + ("" if ini.get("consistent") else "  ⚠ summary.n 与声明不一致：" + "; ".join(g.get("warnings", []))))
        L.append(f"- it{t['iteration']}: **{'PASS' if g['passed'] else 'FAIL'}**，均值成功率 {g['mean']:.2f}，最差工况 {g['worst']}；ONNX 一致性 {'PASS' if t['parity']['pass'] else 'FAIL'} (max|Δ|={t['parity']['pt_vs_onnx_max_abs']})")
        L += ["  - " + ", ".join(f"{c} {v:.2f}" for c, v in g["success"].items())]
    (run.reports).mkdir(parents=True, exist_ok=True)
    (run.reports / "SELECTION_REPORT.md").write_text("\n".join(L) + "\n")
