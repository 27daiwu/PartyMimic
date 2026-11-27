"""PartyMimic V1.6 - acceptance -> export chain:  `./pm v16-export <dance> --run-id ft_fix_v16`.

The chain REFUSES (exit code 5, no model/ONNX/package is produced) unless ALL of these hold:
  1. results/<dance>/v16/B_train/ft_result.json (written by `./pm ftcompare`) has verdict GO (pre-registered rule, FT_PROTOCOL.md);
  2. the requested run is in the treatment group (B) of that comparison and the checkpoint on disk is byte-identical (sha12) to the one that was compared;
  3. the run's reference motion carries a qa_verdict.json == PASS (contact_refine QA) next to the padded npz;
then: hard re-evaluation of THAT checkpoint (no re-selection) -> ONNX export + parity -> Sim2Sim (admission rules of the eval config, never bypassed) -> only on admission PASS: C++ package
(deploy config cpp_g1_v3_1 if it exists, else cpp_g1_v3 / cpp_g1_v2; the run id is part of the package name, historical packages are never overwritten).  No manual override exists.
"""
from __future__ import annotations

import json
from pathlib import Path

from .core import CONFIGS, Dance, Logger, die, load_cfg, now, read_json, sha256, write_json

REFUSED = 5


def check_acceptance(dance: Dance, run_id: str, compare_json: Path | None = None) -> dict:
    run = dance.run(run_id)
    cj = Path(compare_json) if compare_json else dance.dir / "v16/B_train/ft_result.json"
    if not cj.exists():
        die(f"REFUSED: no comparison result {cj}. Run `./pm ftcompare {dance.name} --a <control> --b {run_id}` first.", REFUSED)
    res = read_json(cj)
    vd = res["verdict"]
    if not vd["go"]:
        write_json(cj.parent / "V16_EXPORT_REFUSED.json", dict(time=now(), run=run_id, verdict=vd["verdict"], compare=str(cj)))
        die(f"REFUSED: acceptance verdict is '{vd['verdict']}' (not GO) -> no ONNX / Sim2Sim / package is produced for {run_id}. See {cj.parent / 'FT_COMPARE_REPORT.md'}", REFUSED)
    ent = [(k, m) for k, m in res["models"].items() if m["run"] == run_id and m["group"] == "B"]
    if not ent:
        die(f"REFUSED: run '{run_id}' is not in the treatment group (B) of the accepted comparison ({[m['run'] for m in res['models'].values() if m['group'] == 'B']})", REFUSED)
    k, m = ent[0]
    ck = Path(m["ckpt"])
    if not ck.exists() or sha256(ck)[:12] != m["sha12"]:
        die(f"REFUSED: checkpoint {ck} changed since the comparison (sha12 {m['sha12']}): re-run ftcompare", REFUSED)
    qa = read_json(run.motion_padded_dir / "qa_verdict.json", None)
    if qa is None or str(qa.get("verdict")).upper() != "PASS":
        die(f"REFUSED: reference motion {run.motion_padded_dir} has no qa_verdict.json == PASS ({qa and qa.get('verdict')})", REFUSED)
    return dict(compare=str(cj), verdict=vd["verdict"], model=m, key=k, qa=dict(verdict=qa["verdict"], n_pass=qa.get("n_pass"), n_gates=qa.get("n_gates"), gates_sha256=qa.get("gates_sha256")), iteration=int(m["iteration"]))


def default_deploy_cfg() -> str:
    return "cpp_g1_v3_1" if (CONFIGS / "deploy/cpp_g1_v3_1.yaml").exists() else ("cpp_g1_v3" if (CONFIGS / "deploy/cpp_g1_v3.yaml").exists() else "cpp_g1_v2")


def v16_export(dance: Dance, run_id: str, compare_json=None, deploy_cfg: str | None = None, procs: int = 16, dry_run: bool = False) -> dict:
    run = dance.run(run_id)
    acc = check_acceptance(dance, run_id, compare_json)
    log = Logger(run.logs / "v16_export.log")
    dname = deploy_cfg or default_deploy_cfg()
    dcfg = load_cfg("deploy", dname)
    ecfg = load_cfg("eval", run.manifest().get("eval_cfg") or "balance_first_v1")
    log("acceptance OK", verdict=acc["verdict"], run=run_id, it=acc["iteration"], deploy_cfg=dname)
    if dry_run:
        return dict(acceptance=acc, deploy_cfg=dname, dry_run=True)
    from .post import select_and_validate
    adm = select_and_validate(run, ecfg, dcfg, log, procs=procs, only_it=acc["iteration"])
    out = dict(acceptance=acc, deploy_cfg=dname, admission=adm["real_robot_admission"], final=adm.get("final", {}).get("iteration"), onnx=(adm.get("final") or {}).get("onnx"),
               sim2sim_report=str(run.reports / "SELECTION_REPORT.md"), package=None)
    if adm["real_robot_admission"] == "PASS":
        from .package import package
        pk = package(run, dcfg)
        out["package"] = dict(zip=pk["zip"], sha256=pk["sha256"], size=pk["size"], selfcheck=pk["selfcheck"])
    else:
        log("NOT packaging: admission is not PASS", admission=adm["real_robot_admission"])
    write_json(dance.dir / f"v16/B_train/V16_EXPORT_{run_id}.json", dict(out, time=now()))
    return out
