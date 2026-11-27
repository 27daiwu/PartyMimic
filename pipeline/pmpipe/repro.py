"""PartyMimic Pipeline V1.2 - reproducibility snapshot: `./pm snapshot <dance> [--verify]`.

Saves EVERYTHING needed to rebuild the code state a training run used, into results/<dance>/.../repro/ :
  git.json / status.txt          HEAD, branch, `git status --porcelain`
  train.patch / deploy.patch     full tracked diff vs HEAD (`git diff HEAD --binary --full-index`, binary safe) for the training/pipeline tree and the C++ deploy tree
  untracked/ + untracked_manifest.sha256   every non-ignored untracked file (src/, tools/, pipeline/, deploy/, root files ...) copied with its relative path + sha256
  config/                        pm config snapshot, manifest, args.json, the dance's own training scripts (legacy flat runs)
  external/                      small git-ignored helpers the pipeline imports from results/p0, results/p4 (sha256 recorded)
  env.json / pip_freeze.txt      python / torch / mujoco / mjlab / warp / onnxruntime versions, GPU + driver, OS, CPU, sim2real venv
  inputs.json                    sha256 of init checkpoint, source pkl, motion npz/csv/padded, uv.lock, pyproject.toml, final models
  timeline.json                  training start / resume / rollback timeline parsed from logs/supervisor.log
  MANIFEST.json                  all hashes (patch sha256, tree hashes of the code groups, legacy diff hash) + REPRODUCE.md (generated step list)
`snapshot --verify` rebuilds the tree in a temp dir from `git archive HEAD` + the patches + the untracked copies and checks that
  (1) the patches apply, (2) the re-derived `git diff` hashes equal the recorded ones, (3) every untracked copy matches its sha256,
  (4) the tree hash of each code group equals the recorded one, (5) the rebuilt `partymimic` package imports (robot spec of the run loads).
Nothing outside results/<dance>/ is written; nothing is modified in the repo.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from .core import PY, REPO, S2R, S2R_PY, Logger, Run, now, read_json, write_json

# code groups: the tree hash of each is recorded and re-checked by --verify
GROUPS = {
    "train_tree": ["src", "tools", "scripts", "manifests", "pyproject.toml", "uv.lock"],        # what `partymimic.train` + the export tools run
    "pipeline_tree": ["pipeline", "pm", "pm-test", "README_PIPELINE.md"],                      # orchestration (./pm ...)
    "deploy_tree": ["deploy/partymimic_cpp"],                                                  # the C++ framework a package is built from
}
PATCHES = {"train.patch": ["src", "tools", "scripts", "manifests", "pyproject.toml", "uv.lock", "pipeline", "pm", "pm-test", "README_PIPELINE.md", "README.md"],
           "deploy.patch": ["deploy"]}
MAX_COPY_BYTES = 8 << 20          # a single untracked file above this is recorded (sha256) but not copied
SKIP_RE = re.compile(r"(^|/)(__pycache__|\.venv|\.git|results)(/|$)|\.pyc$")
EXTERNAL = ["results/p4/scripts/sim2sim_deploy.py", "results/p4/scripts/eval_bench.py", "results/p4/scripts/pm_stack.py", "results/p0/verify_deploy_onnx.py"]
LEGACY_DIFF = ["src", "tools"]    # the V1.1 provenance hash ("src_tools_uncommitted_diff_sha12") was `git diff -- src tools`


# ----------------------------------------------------------------------------------------------------------------------------- helpers
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def git(root: Path, *args: str, check: bool = True) -> bytes:
    r = subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=str(root), capture_output=True)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.decode(errors='replace')[:400]}")
    return r.stdout


def list_files(root: Path, paths: list[str]) -> list[str]:
    """tracked + untracked-not-ignored files under `paths` that exist in the work tree (deleted tracked files are carried by the patch)."""
    out = git(root, "ls-files", "-z", "-c", "-o", "--exclude-standard", "--", *paths).decode().split("\0")
    return sorted({f for f in out if f and not SKIP_RE.search(f) and (root / f).is_file()})


def tree_hash(root: Path, files: list[str]) -> tuple[str, int]:
    """sha256 over `sha256 x|- path` lines (x = executable bit) of the sorted file list."""
    lines = []
    for f in files:
        p = root / f
        lines.append(f"{sha256_file(p)} {'x' if os.access(p, os.X_OK) else '-'} {f}")
    return sha256_bytes(("\n".join(lines) + "\n").encode()), len(lines)


def repro_dir(run: Run) -> Path:
    return run.root / "repro"


# ----------------------------------------------------------------------------------------------------------------------------- environment
def _venv_versions(py: str, names: list[str]) -> dict:
    code = ("import json,sys,platform\nfrom importlib import metadata as m\nout={'python':sys.version.split()[0],'executable':sys.executable}\n"
            "for n in %r:\n    try: out[n]=m.version(n)\n    except Exception: out[n]=None\n"
            "try:\n    import torch; out['torch_cuda']=torch.version.cuda; out['cuda_available']=bool(torch.cuda.is_available()); out['device']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None\nexcept Exception as e: out['torch_import']=repr(e)\n"
            "out['freeze']=sorted(f'{d.metadata[\"Name\"]}=={d.version}' for d in m.distributions())\nprint(json.dumps(out))") % (names,)
    r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=180)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return dict(error=(r.stderr or r.stdout)[-300:])


def capture_env() -> dict:
    main = _venv_versions(PY, ["torch", "mujoco", "mujoco-warp", "mjlab", "warp-lang", "numpy", "scipy", "onnx", "onnxruntime", "tensorboard", "pyyaml", "matplotlib"])
    s2r = _venv_versions(S2R_PY, ["mujoco", "onnxruntime", "numpy", "scipy"]) if Path(S2R_PY).exists() else dict(error="sim2real venv missing")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    freeze_main, freeze_s2r = main.pop("freeze", []), s2r.pop("freeze", [])
    return dict(main_venv=main, sim2real_venv=s2r, gpu=smi, os=platform.platform(), cpu=cpu, nproc=os.cpu_count(), omp_num_threads_in_train_script=4,
                note="CUDA/driver/GPU model and the float atomics order of the mujoco-warp solver decide the exact numbers (see REPRODUCE.md section 'what is NOT bit-reproducible')"), freeze_main, freeze_s2r


def parse_timeline(run: Run) -> dict:
    ev = []
    for name in ("supervisor.log", "pipeline.log"):
        p = run.logs / name
        if not p.exists():
            continue
        for l in p.read_text(errors="replace").splitlines():
            if any(k in l for k in ("supervisor start", "training launched", "ROLLBACK", "TERMINATING", "training process exited", "training complete", "ABORT", "train_done")):
                ev.append(dict(file=name, line=l[:240]))
    rows = []
    try:
        import csv
        rows = list(csv.DictReader(open(run.train / "train_log.csv")))
    except Exception:
        pass
    launches = [e for e in ev if "training launched" in e["line"]]
    first = launches[0]["line"][:19] if launches else None
    return dict(events=ev, first_launch=first, n_launches=len(launches), n_rollbacks=sum("ROLLBACK" in e["line"] for e in ev),
                train_log=dict(rows=len(rows), first_it=int(float(rows[0]["iteration"])) if rows else None, last_it=int(float(rows[-1]["iteration"])) if rows else None, wall_h_last=float(rows[-1]["wall_h"]) if rows else None),
                archived_runs=sorted(str(p.relative_to(run.dance.dir)) for p in (run.dance.dir / "archive").glob("*")) if (run.dance.dir / "archive").exists() else [])


# ----------------------------------------------------------------------------------------------------------------------------- snapshot
def _copy_untracked(root: Path, out: Path) -> tuple[list[dict], list[dict]]:
    files = [f.decode() for f in git(root, "ls-files", "-z", "-o", "--exclude-standard").split(b"\0") if f]
    copied, skipped = [], []
    shutil.rmtree(out, ignore_errors=True)
    for f in sorted(files):
        p = root / f
        if SKIP_RE.search(f) or not p.is_file():
            continue
        size = p.stat().st_size
        if size > MAX_COPY_BYTES:
            skipped.append(dict(path=f, size=size, sha256=sha256_file(p), reason=f"> {MAX_COPY_BYTES >> 20} MiB"))
            continue
        dst = out / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        copied.append(dict(path=f, size=size, sha256=sha256_file(dst), exec=bool(os.access(dst, os.X_OK))))
    return copied, skipped


def compute_state(root: Path) -> dict:
    """code-state fingerprint of a work tree: patch hashes + group tree hashes (used by snapshot and by --verify's live-drift check)."""
    st = dict(head=git(root, "rev-parse", "HEAD").decode().strip())
    st["patches"] = {}
    for name, paths in PATCHES.items():
        d = git(root, "diff", "HEAD", "--binary", "--full-index", "--", *paths)
        st["patches"][name] = dict(sha256=sha256_bytes(d), bytes=len(d), _data=d)
    st["legacy_src_tools_diff_sha256"] = sha256_bytes(git(root, "diff", "--", *LEGACY_DIFF))
    st["trees"] = {}
    for g, paths in GROUPS.items():
        fs = list_files(root, paths)
        h, n = tree_hash(root, fs)
        st["trees"][g] = dict(sha256=h, files=n)
    return st


def snapshot(run: Run, log=None, force: bool = False) -> dict:
    log = log or Logger(run.logs / "repro.log")
    out = repro_dir(run)
    st = compute_state(REPO)
    old = read_json(out / "MANIFEST.json", {})
    if old and not force and old.get("head") == st["head"] and all(old["patches"].get(k, {}).get("sha256") == v["sha256"] for k, v in st["patches"].items()) \
            and all(old["trees"].get(k, {}).get("sha256") == v["sha256"] for k, v in st["trees"].items()):
        log("repro snapshot up to date", dir=str(out))
        return old
    if old:  # keep the previous snapshot instead of overwriting history
        prev = out.with_name(f"repro_prev_{old.get('created', 'old').replace(' ', '_').replace(':', '')}")
        if not prev.exists():
            shutil.move(str(out), str(prev))
            log("previous repro snapshot kept", dir=str(prev))
    out.mkdir(parents=True, exist_ok=True)
    # ---- git
    for name, v in st["patches"].items():
        (out / name).write_bytes(v.pop("_data"))
    (out / "status.txt").write_bytes(git(REPO, "status", "--porcelain"))
    log_line = git(REPO, "log", "-1", "--format=%H %cd %s").decode().strip()
    branch = git(REPO, "rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    # ---- untracked
    copied, skipped = _copy_untracked(REPO, out / "untracked")
    (out / "untracked_manifest.sha256").write_text("".join(f"{c['sha256']}  {c['path']}\n" for c in copied))
    # ---- config / scripts
    cfg = out / "config"
    shutil.rmtree(cfg, ignore_errors=True)
    cfg.mkdir()
    for p in sorted((run.pm / "config_snapshot").glob("*")):
        shutil.copy2(p, cfg / p.name)
    for p in (run.pm / "manifest.json", run.train / "args.json"):
        if p.exists():
            shutil.copy2(p, cfg / p.name)
    sdir = run.dance.dir / "scripts"
    if sdir.exists():
        (cfg / "dance_scripts").mkdir()
        for p in sorted(sdir.glob("*")):
            if p.is_file():
                shutil.copy2(p, cfg / "dance_scripts" / p.name)
    ext = out / "external"
    shutil.rmtree(ext, ignore_errors=True)
    external = []
    for rel in EXTERNAL:
        p = REPO / rel
        if p.exists():
            (ext / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, ext / rel)
            external.append(dict(path=rel, sha256=sha256_file(p)))
    s2r = dict(path=str(S2R), exists=S2R.exists())
    if (S2R / ".git").exists():
        s2r.update(head=git(S2R, "rev-parse", "HEAD", check=False).decode().strip(), dirty_diff_sha256=sha256_bytes(git(S2R, "diff", "HEAD", check=False)), status=git(S2R, "status", "--porcelain", check=False).decode()[:600])
    # ---- environment
    env, fm, fs = capture_env()
    write_json(out / "env.json", env)
    (out / "pip_freeze.txt").write_text("# main venv\n" + "\n".join(fm) + "\n\n# sim2real venv\n" + "\n".join(fs) + "\n")
    # ---- inputs
    d = run.dance
    inputs = {}
    for k, p in (("init_checkpoint", run.init_pt), ("source_pkl", d.source_pkl()), ("motion_npz", d.npz), ("motion_csv", d.csv), ("padded_npz", d.padded_npz), ("pretrained_g1_mlp", REPO / "checkpoints/g1_mlp.pt"),
                 ("uv.lock", REPO / "uv.lock"), ("pyproject.toml", REPO / "pyproject.toml"), ("train_args.json", run.train / "args.json")):
        if Path(p).exists():
            inputs[k] = dict(path=str(p), sha256=sha256_file(p), bytes=Path(p).stat().st_size)
    inputs["final_models"] = {p.name: sha256_file(p) for p in sorted((run.final / "models").glob("*.pt"))}
    write_json(out / "inputs.json", inputs)
    tl = parse_timeline(run)
    write_json(out / "timeline.json", tl)
    # ---- evidence that the training source did not change after training started
    train_files = list_files(REPO, GROUPS["train_tree"][:2])
    newest = max((((REPO / f).stat().st_mtime, f) for f in train_files), default=(0, ""))
    unchanged = None
    if tl["first_launch"]:
        t0 = time.mktime(time.strptime(tl["first_launch"], "%Y-%m-%d %H:%M:%S"))
        unchanged = newest[0] <= t0
    man = dict(created=now(), dance=d.name, run=run.id, head=st["head"], branch=branch, head_log=log_line, patches={k: dict(v) for k, v in st["patches"].items()}, trees=st["trees"],
               legacy_src_tools_diff_sha256=st["legacy_src_tools_diff_sha256"], legacy_src_tools_diff_sha12=st["legacy_src_tools_diff_sha256"][:12],
               untracked=dict(copied=len(copied), bytes=sum(c["size"] for c in copied), skipped=skipped), external=external, sim2real_repo=s2r,
               train_source_newest_mtime=dict(file=newest[1], time=time.strftime("%F %T", time.localtime(newest[0]))), train_source_unchanged_since_first_launch=unchanged,
               first_launch=tl["first_launch"], inputs={k: (v["sha256"][:12] if isinstance(v, dict) and "sha256" in v else None) for k, v in inputs.items() if k != "final_models"})
    write_json(out / "MANIFEST.json", man)
    (out / "REPRODUCE.md").write_text(reproduce_md(run, man, env, tl, inputs))
    log("repro snapshot written", dir=str(out), patches={k: v["sha256"][:12] for k, v in st["patches"].items()}, untracked=len(copied), skipped=len(skipped))
    return man


def reproduce_md(run: Run, man: dict, env: dict, tl: dict, inputs: dict) -> str:
    d = run.dance.name
    t = man["trees"]
    mv = env["main_venv"]
    L = [f"# 复现 {d}/{run.id} 的训练（自动生成 {man['created']}）", "",
         f"- 代码基线：git `{man['head']}`（{man['branch']}）；训练用工作区在其上有**未提交改动**，已完整保存为补丁 + 未跟踪文件（下文）。",
         f"- 训练源码（src/tools）自首次启动（{man['first_launch']}）后是否改动：{'**未改动**（最新修改时间 ' + man['train_source_newest_mtime']['time'] + '）' if man['train_source_unchanged_since_first_launch'] else '**可能改动过，不能保证补丁即训练时代码**' if man['train_source_unchanged_since_first_launch'] is False else '未知'}。",
         f"- 旧版 V1.1 溯源哈希 `src_tools_uncommitted_diff_sha12` = `{man['legacy_src_tools_diff_sha12']}`（`git diff -- src tools`），与此处一致即同一份源码改动。", "",
         "## 1. 重建源码", "```bash",
         f"git clone <repo-url> PartyMimic && cd PartyMimic && git checkout {man['head']}",
         f"git apply --binary <SNAP>/train.patch          # sha256 {man['patches']['train.patch']['sha256'][:16]}…  ({man['patches']['train.patch']['bytes']} B)",
         f"git apply --binary <SNAP>/deploy.patch         # sha256 {man['patches']['deploy.patch']['sha256'][:16]}…  ({man['patches']['deploy.patch']['bytes']} B；C++ 部署框架，只影响打包，不影响训练)",
         "cp -a <SNAP>/untracked/. .                       # 未跟踪文件（含 src/partymimic/g1mj_ythn/、pipeline/、pm、pm-test、deploy 新增文件）",
         "(cd <SNAP>/untracked && sha256sum -c ../untracked_manifest.sha256)   # 逐文件校验",
         "```", f"`<SNAP>` = 本目录（`results/{d}/…/repro`）。\n", "代码组哈希（`./pm snapshot --verify` 会重建并比对）：", "",
         "| 组 | 文件数 | tree sha256 |", "|---|---|---|"] + [f"| {g} | {v['files']} | `{v['sha256'][:24]}…` |" for g, v in t.items()]
    if man["untracked"]["skipped"]:
        L += ["", "**未复制（过大）的未跟踪文件**：" + ", ".join(f"`{s['path']}`" for s in man["untracked"]["skipped"]) + "（sha256 见 MANIFEST.json）。"]
    L += ["", "## 2. 环境", "", f"- main venv：python {mv.get('python')}，torch {mv.get('torch')}（CUDA {mv.get('torch_cuda')}），mujoco {mv.get('mujoco')}，mujoco-warp {mv.get('mujoco-warp')}，mjlab {mv.get('mjlab')}，warp-lang {mv.get('warp-lang')}，onnxruntime {mv.get('onnxruntime')}。",
          f"- GPU/驱动：{env['gpu']}；OS {env['os']}；CPU {env['cpu']}（{env['nproc']} 线程，训练脚本固定 OMP_NUM_THREADS=4）。",
          f"- 依赖锁：`uv.lock` sha256 `{inputs.get('uv.lock', {}).get('sha256', '?')[:16]}…`；`uv sync` 复原 main venv，完整包清单见 `pip_freeze.txt`；sim2real venv（Sim2Sim/部署资产）版本见 `env.json`。",
          "- 仓库外的依赖（git 忽略）：`external/` 下 4 个脚本（results/p0、results/p4）和 sim2real 仓库（`MANIFEST.json: sim2real_repo`）。", "",
          "## 3. 输入数据（sha256 前 16 位）", ""]
    for k, v in inputs.items():
        if k == "final_models":
            L += [f"- final_models：" + ", ".join(f"{n} `{h[:12]}`" for n, h in v.items())]
        else:
            L += [f"- {k}：`{v['sha256'][:16]}…`  ({v['path']})"]
    L += ["", "## 4. 训练", "", "与本次训练同配方（`robust_d_v1`，官方预训练 `checkpoints/g1_mlp.pt` 加宽后微调）：", "```bash",
          f"mkdir -p /home/unitree/motion_datasets/video/{d} && cp <源 robot_motion.pkl> /home/unitree/motion_datasets/video/{d}/   # sha256 必须等于上表 source_pkl",
          f"./pm prepare {d}", f"./pm train {d} --iters {tl['train_log']['last_it'] or 40000} --train-cfg robust_d_v1      # 配置见 config/（sha256 已冻结）", "```",
          f"本次实际过程：{tl['n_launches']} 次启动（{', '.join(e['line'][:19] + ' ' + re.sub(r'.*training launched ', '', e['line']) for e in tl['events'] if 'training launched' in e['line'])}），{tl['n_rollbacks']} 次回滚；train_log 共 {tl['train_log']['rows']} 行，末次迭代 {tl['train_log']['last_it']}，墙钟 {tl['train_log']['wall_h_last']} h。"
          "起点在第一段（it0→2000）；第二段从 `last.pt@2000` 续训，故最终 `args.json` 的 `init=null/resume=last.pt` 不代表没有预训练起点。",
          "该次训练由 `config/dance_scripts/ythn_pipeline.py + ythn_supervisor.py + train_ythn.sh` 编排（pm 随后以 `adopt` 接管，配方逐项核对一致）；`./pm train` 用同一配方。" if (run.dance.dir / "scripts").exists() else "",
          "", "## 5. 哪些部分做不到逐位复现（请勿期望）", "",
          "1. **GPU 浮点原子累加**：mujoco-warp 求解器与 PyTorch 的 CUDA 归约使用原子加，累加顺序不确定；同一代码、同一随机种子在同一台机器上跑两次，轨迹也会在首次碰撞后分歧，并被闭环放大。可复现的是**配方与统计结果**（评估分数在噪声范围内），不是逐位权重。",
          "2. **CUDA / 驱动 / GPU 型号**：不同 SM 架构、cuDNN/cuBLAS 内核选择、TF32/bf16 路径都会改变数值；`env.json` 记录了本次的组合。",
          "3. **`torch.compile` / inductor / triton 缓存**：缓存不同可能选择不同融合与内核（训练脚本把缓存目录指向 results/p2/…/cache，不在快照内）。",
          "4. **并行环境数与多线程**：`num_envs=4096`、`OMP_NUM_THREADS=4` 已记录；改变会改变随机数消耗顺序。",
          "5. **评估**：评估器网格与种子固定，但评估本身同样在 GPU 上跑（多种子 std 约 0.1–0.3 分，见 QUALITY_REPORT）。",
          "6. **取回的外部状态**：sim2real 仓库如有未提交改动，只记录了 diff 哈希。", ""]
    return "\n".join(x for x in L if x is not None) + "\n"


# ----------------------------------------------------------------------------------------------------------------------------- verify
def verify(run: Run, keep: bool = False, log=None) -> dict:
    log = log or Logger(run.logs / "repro.log")
    out = repro_dir(run)
    man = read_json(out / "MANIFEST.json")
    if not man:
        raise RuntimeError(f"no snapshot in {out}: run `./pm snapshot {run.dance.name}` first")
    checks: list[tuple[str, bool, str]] = []
    add = lambda name, ok, detail="": checks.append((name, bool(ok), detail))
    tmp = Path(tempfile.mkdtemp(prefix="pm_repro_verify_"))
    try:
        # the snapshot's own files are intact
        bad = [l for l in (out / "untracked_manifest.sha256").read_text().splitlines() if l and sha256_file(out / "untracked" / l.split("  ", 1)[1]) != l.split("  ", 1)[0]]
        add("untracked copies match untracked_manifest.sha256", not bad, f"{len(bad)} mismatching" if bad else f"{man['untracked']['copied']} files")
        for name, v in man["patches"].items():
            add(f"{name} sha256 == manifest", sha256_file(out / name) == v["sha256"], v["sha256"][:12])
        # rebuild: HEAD content -> temp git repo (baseline) -> patches -> untracked copies
        top = sorted({p for ps in PATCHES.values() for p in ps} | {p for ps in GROUPS.values() for p in ps})
        in_head = [p for p in top if git(REPO, "ls-tree", "--name-only", man["head"], "--", p).strip()]
        tar_p = tmp / "head.tar"
        git(REPO, "archive", "-o", str(tar_p), man["head"], "--", *in_head)
        rb = tmp / "rebuild"
        rb.mkdir()
        with tarfile.open(tar_p) as tf:
            tf.extractall(rb, filter="data")
        tar_p.unlink()
        g = lambda *a, check=True: git(rb, "-c", "user.name=pm", "-c", "user.email=pm@local", *a, check=check)
        g("init", "-q")
        g("add", "-A", "-f")
        g("commit", "-q", "-m", "HEAD baseline")
        head_c = g("rev-parse", "HEAD").decode().strip()
        ap_ok = True
        for name in man["patches"]:
            if man["patches"][name]["bytes"] == 0:
                continue
            r = subprocess.run(["git", "apply", "--binary", str(out / name)], cwd=str(rb), capture_output=True, text=True)
            ap_ok &= r.returncode == 0
            add(f"git apply {name} onto a clean HEAD checkout", r.returncode == 0, r.stderr.strip()[:200] if r.returncode else "applied cleanly")
        n = 0
        for line in (out / "untracked_manifest.sha256").read_text().splitlines():
            rel = line.split("  ", 1)[1]
            dst = rb / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out / "untracked" / rel, dst)
            n += 1
        add("untracked files restored", True, f"{n} files")
        # re-derive the diff hashes in the rebuilt repo (tracked changes only)
        for name, paths in PATCHES.items():
            d = g("diff", "HEAD", "--binary", "--full-index", "--", *paths)
            add(f"re-derived {name} hash == recorded", sha256_bytes(d) == man["patches"][name]["sha256"], f"{sha256_bytes(d)[:12]} vs {man['patches'][name]['sha256'][:12]}")
        leg = sha256_bytes(g("diff", "--", *LEGACY_DIFF))
        add("re-derived `git diff -- src tools` hash == recorded legacy hash", leg == man["legacy_src_tools_diff_sha256"], f"{leg[:12]} vs {man['legacy_src_tools_diff_sha256'][:12]}")
        for grp, paths in GROUPS.items():
            h, nf = tree_hash(rb, list_files(rb, paths))
            add(f"tree hash {grp} == recorded", h == man["trees"][grp]["sha256"] and nf == man["trees"][grp]["files"], f"{nf} files {h[:12]} vs {man['trees'][grp]['sha256'][:12]}")
        # the rebuilt package is importable and the run's robot spec loads
        args = read_json(out / "config/args.json", {}) or {}
        robot = args.get("robot", "g1mj")
        code = ("import sys; sys.path.insert(0, %r)\nimport partymimic, partymimic.dr, partymimic.mdp\nfrom partymimic.robot import get_robot\nr = get_robot(%r)\n"
                "assert partymimic.__file__.startswith(%r), partymimic.__file__\nprint('OK', r.name, r.num_joints)") % (str(rb / "src"), robot, str(rb))
        r = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=300)
        add(f"rebuilt src imports and loads robot '{robot}' (from the rebuilt tree)", r.returncode == 0 and "OK" in r.stdout, (r.stdout or r.stderr).strip()[-160:])
        # informational: has the live work tree drifted since the snapshot?
        live = compute_state(REPO)
        drift = [grp for grp in GROUPS if live["trees"][grp]["sha256"] != man["trees"][grp]["sha256"]]
        for v in live["patches"].values():
            v.pop("_data", None)
        info = dict(live_tree_drift_since_snapshot=drift, live_head=live["head"], snapshot_head=man["head"])
    finally:
        if keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    ok = all(c[1] for c in checks)
    res = dict(verified=now(), result="PASS" if ok else "FAIL", checks=[dict(name=n, ok=o, detail=d) for n, o, d in checks], info=info, snapshot_created=man["created"], rebuilt_baseline_commit=head_c)
    write_json(out / "VERIFY.json", res)
    log("repro verify", result=res["result"], failed=[n for n, o, _ in checks if not o], drift=drift)
    return res


def format_verify(res: dict) -> str:
    L = [f"[pm] snapshot verify: {res['result']}"]
    for c in res["checks"]:
        L.append(f"   [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}" + (f"  ({c['detail']})" if c["detail"] else ""))
    d = res["info"]["live_tree_drift_since_snapshot"]
    L.append("   [info] live work tree vs snapshot: " + ("identical" if not d else f"changed since the snapshot in {d} (expected if you kept developing; re-run `./pm snapshot` to refresh)"))
    return "\n".join(L)
