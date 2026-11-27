"""PartyMimic Pipeline V1.0 - shared helpers: paths, per-dance run layout, versioned configs, state markers, tmux, GPU detection.

Design rules
  * One output directory per dance: results/<dance>/ (override the root with PM_RESULTS). Nothing outside results/<dance>/ is ever written
    (SSB history under results/p0..p5, deploy/partymimic_cpp, src/, tools/ are read-only inputs).
  * Run layout. A dance can hold several runs (one per training-config version / experiment). The legacy YTHN run that predates this
    pipeline lives in the *flat* layout (results/ythn/training/run, eval, best, final ...); new runs use the *nested* layout
    results/<dance>/runs/<run_id>/{train,eval,best,final,state,logs,tb,gate,reports,deploy,analysis}. Both are addressed through `Run`.
  * Configs are versioned YAML files (pipeline/configs/{train,deploy,eval}/<name>.yaml). A copy + sha256 is frozen into every run
    (pm/config_snapshot) so a later edit of the YAML can never silently change a finished run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PIPE = REPO / "pipeline"
CONFIGS = PIPE / "configs"
RESULTS = Path(os.environ.get("PM_RESULTS", REPO / "results"))
DATASETS = Path(os.environ.get("PM_DATASETS", "/home/unitree/motion_datasets/video"))
PY = str(REPO / ".venv/bin/python")  # main env: torch / mjlab / mujoco-warp (training, evaluation, ONNX export)
S2R = Path(os.environ.get("PM_SIM2REAL", REPO.parent / "sim2real_party_p0_patched"))
S2R_PY = str(S2R / ".venv/bin/python")  # py3.10 env with the sim2real G1 model (Sim2Sim, deploy asset export)
SUPERMIMIC = Path(os.environ.get("PM_SUPERMIMIC", REPO.parent / "unitree_rl_mjlab/extensions/supermimic"))
CPP = REPO / "deploy/partymimic_cpp"  # latest C++ deployment framework (read-only input)
TMUX = shutil.which("tmux") or os.path.expanduser("~/.local/bin/tmux")
# directories of SSB / earlier projects that a dance name must never collide with
RESERVED = {"p0", "p1", "p2", "p3", "p3_1", "p4", "p5", "fasttrack", "fasttrack_r1", "fasttrack_r2", "fasttrack_r2_1", "fasttrack_r2_1_fast", "_smoke", "tmp"}
LEGACY_FLAT_DANCES = {"ythn"}  # dances whose pre-pipeline run uses the flat layout (adopted, not migrated)


# ----------------------------------------------------------------------------------------------------------------------------- logging
def now() -> str:
    return time.strftime("%F %T")


def say(*a):
    print(now(), *a, flush=True)


def die(msg: str, code: int = 2):
    print(f"[pm] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


class Logger:
    def __init__(self, path: Path, echo: bool = True):
        self.path, self.echo = Path(path), echo
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, *a, **kw):
        line = now() + " " + " ".join(str(x) for x in a) + ((" " + json.dumps(kw, default=str)) if kw else "")
        if self.echo:
            print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


# ----------------------------------------------------------------------------------------------------------------------------- files
def sha256(path, n: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(n):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj, **kw):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str, **kw))
    os.replace(tmp, p)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def read_csv(path) -> list[dict]:
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def sh(cmd, *, log=None, env=None, cwd=None, check=False, capture=False, timeout=None):
    """subprocess.run with all arguments stringified; stdout+stderr appended to `log` (a path) when given."""
    cmd = [str(c) for c in cmd]
    e = dict(os.environ, **{k: str(v) for k, v in (env or {}).items()})
    if log is not None:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write(f"\n$ {' '.join(cmd)}   [{now()}]\n"); f.flush()
            r = subprocess.run(cmd, cwd=str(cwd or REPO), stdout=f, stderr=subprocess.STDOUT, env=e, timeout=timeout)
    else:
        r = subprocess.run(cmd, cwd=str(cwd or REPO), env=e, capture_output=capture, text=True, timeout=timeout)
    if check and r.returncode:
        die(f"command failed rc={r.returncode}: {' '.join(cmd)}" + (f" (see {log})" if log else ""), 3)
    return r


def py_env(extra: dict | None = None) -> dict:
    """Environment of the training / evaluation / export processes (same as the validated train_ythn.sh)."""
    cache = REPO / "results/p2/run_20261119/cache"
    e = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": f"{REPO / 'src'}", "OMP_NUM_THREADS": "4"}
    if cache.exists():
        e.update(TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), TRITON_CACHE_DIR=str(cache / "triton"))
    e.update(extra or {})
    return e


# ----------------------------------------------------------------------------------------------------------------------------- configs
def load_cfg(kind: str, name: str) -> dict:
    p = Path(name)
    if not p.suffix:
        p = CONFIGS / kind / f"{name}.yaml"
    if not p.exists():
        avail = ", ".join(sorted(x.stem for x in (CONFIGS / kind).glob("*.yaml")))
        die(f"{kind} config '{name}' not found ({p}). available: {avail}")
    cfg = yaml.safe_load(p.read_text())
    cfg["_path"], cfg["_sha256"] = str(p), sha256(p)
    cfg.setdefault("name", p.stem)
    return cfg


def list_cfgs() -> dict:
    return {k: sorted(x.stem for x in (CONFIGS / k).glob("*.yaml")) for k in ("train", "deploy", "eval")}


# ----------------------------------------------------------------------------------------------------------------------------- dance / run layout
def valid_dance(name: str):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]{0,40}", name):
        die(f"dance name '{name}' must match [A-Za-z0-9_-]+")
    if name in RESERVED:
        die(f"dance name '{name}' is reserved (SSB / historic result directory)")


class Dance:
    """results/<dance>/ + the source data in the dataset root."""

    def __init__(self, name: str):
        valid_dance(name)
        self.name = name
        self.dir = RESULTS / name
        self.src_dir = DATASETS / name
        self.pm = self.dir / "pm"
        self.motions = self.dir / "motions"
        self.metrics = self.dir / "metrics"

    # source files (never written)
    def source_pkl(self, override: str | None = None) -> Path:
        return Path(override) if override else self.src_dir / "robot_motion.pkl"

    # derived motion files (same names as the validated YTHN layout)
    @property
    def npz(self) -> Path: return self.motions / self.name / f"{self.name}.npz"
    @property
    def csv(self) -> Path: return self.motions / self.name / f"{self.name}.csv"
    @property
    def padded_dir(self) -> Path: return self.motions / f"{self.name}_padded"
    @property
    def padded_npz(self) -> Path: return self.padded_dir / f"{self.name}_padded.npz"

    @property
    def index_path(self) -> Path: return self.pm / "index.json"

    def index(self) -> dict:
        idx = read_json(self.index_path, None)
        if idx is None:
            idx = {"dance": self.name, "runs": {}, "default_run": None}
            legacy = self.dir / "training/run/args.json"
            if self.name in LEGACY_FLAT_DANCES and legacy.exists():  # pre-pipeline run, addressed in place
                idx["runs"]["legacy"] = {"layout": "flat", "train_cfg": None, "adopted": False, "created": "pre-pipeline"}
                idx["default_run"] = "legacy"
        return idx

    def save_index(self, idx: dict):
        write_json(self.index_path, idx)

    def run(self, run_id: str | None = None, train_cfg: str | None = None, create: bool = False) -> "Run":
        idx = self.index()
        rid = run_id or idx.get("default_run") or train_cfg
        if rid is None:
            die(f"no run for dance '{self.name}' yet: run `./pm train {self.name}` (or give --run-id)")
        if rid not in idx["runs"]:
            if not create:
                die(f"run '{rid}' of dance '{self.name}' does not exist (known: {', '.join(idx['runs']) or 'none'})")
            idx["runs"][rid] = {"layout": "nested", "train_cfg": train_cfg, "created": now()}
            idx["default_run"] = rid
            self.save_index(idx)
        return Run(self, rid, idx["runs"][rid])


class Run:
    """One training run of a dance (all paths of the run in one object; flat or nested layout)."""

    def __init__(self, dance: Dance, run_id: str, meta: dict):
        self.dance, self.id, self.meta = dance, run_id, meta
        d = dance.dir
        if meta.get("layout") == "flat":
            self.root = d
            self.train, self.init_pt = d / "training/run", d / "training/init_from_g1_mlp.pt"
        else:
            self.root = d / "runs" / run_id
            self.train, self.init_pt = self.root / "train", self.root / "init.pt"
        # reference motion of THIS run (V1.6): a run trained on another motion directory (fine-tune on corrected data, V1.3 repaired runs) points to it through meta["motion_dir"] (padded)
        # and meta["motion_npz"] (unpadded); every post-processing step (hard re-eval, export, Sim2Sim, package) must use these instead of the dance's original files
        self.motion_padded_dir = Path(meta["motion_dir"]) if meta.get("motion_dir") else dance.padded_dir
        _pz = sorted(self.motion_padded_dir.glob("*.npz")) if meta.get("motion_dir") else []
        self.motion_padded_npz = _pz[0] if len(_pz) == 1 else dance.padded_npz
        self.motion_npz = Path(meta["motion_npz"]) if meta.get("motion_npz") else dance.npz
        r = self.root
        self.eval, self.best, self.final = r / "eval", r / "best", r / "final"
        self.state, self.logs, self.tb = r / "state", r / "logs", r / "tb"
        self.gate, self.reports = r / "gate", r / "reports"
        self.deploy, self.analysis = r / "deploy", r / "analysis"
        self.pm = r / "pm" if meta.get("layout") != "flat" else d / "pm"
        self.session_names = [f"pm-{dance.name}", dance.name] if meta.get("layout") == "flat" else [f"pm-{dance.name}" + ("" if run_id == meta.get("train_cfg") else f"-{run_id}")]

    @property
    def session(self) -> str:
        """tmux session used by this run (the live one if it exists, else the first candidate)."""
        for s in self.session_names:
            if tmux_has(s):
                return s
        return self.session_names[0]

    def mk(self):
        for p in (self.train, self.eval, self.best, self.final, self.state, self.logs, self.tb, self.gate, self.reports, self.pm):
            p.mkdir(parents=True, exist_ok=True)

    # state markers (same names as the validated YTHN pipeline)
    def marker(self, name: str) -> Path: return self.state / name
    def done(self, name: str) -> bool: return self.marker(name).exists()
    def mark(self, name: str, text: str = ""): self.state.mkdir(parents=True, exist_ok=True); self.marker(name).write_text(text or now())

    # training log
    def train_rows(self, n: int | None = None) -> list[dict]:
        rows = read_csv(self.train / "train_log.csv")
        return rows[-n:] if n else rows

    def iteration(self) -> int:
        r = self.train_rows(1)
        return int(float(r[0]["iteration"])) if r else 0

    def manifest(self) -> dict: return read_json(self.pm / "manifest.json", {})
    def save_manifest(self, m: dict): write_json(self.pm / "manifest.json", m)
    def update_manifest(self, **kw) -> dict:
        m = self.manifest(); m.update(kw); m["updated"] = now(); self.save_manifest(m); return m

    def admission(self) -> dict: return read_json(self.pm / "admission.json", {})
    def save_admission(self, a: dict): write_json(self.pm / "admission.json", a)

    def ckpt(self, it: int) -> Path: return self.train / f"ckpt_{it:06d}.pt"
    def ckpts(self) -> list[int]: return sorted(int(p.stem.split("_")[1]) for p in self.train.glob("ckpt_*.pt"))


def snapshot_cfgs(run: Run, cfgs: dict[str, dict]):
    """Freeze the YAML files a run was created with (never overwritten once present)."""
    d = run.pm / "config_snapshot"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind, c in cfgs.items():
        dst = d / f"{kind}__{c['name']}.yaml"
        if not dst.exists():
            shutil.copy2(c["_path"], dst)
        out[kind] = {"name": c["name"], "sha256": sha256(dst), "frozen_copy": str(dst)}
    return out


# ----------------------------------------------------------------------------------------------------------------------------- tmux
def tmux(*args, check=False):
    return subprocess.run([TMUX, *map(str, args)], capture_output=True, text=True, check=check)


def tmux_has(session: str) -> bool:
    return tmux("has-session", "-t", session).returncode == 0


def tmux_windows(session: str) -> list[str]:
    r = tmux("list-windows", "-t", session, "-F", "#W")
    return r.stdout.split() if r.returncode == 0 else []


def tmux_window(session: str, name: str, shell_cmd: str, cwd: Path | str = REPO) -> str:
    """Start `shell_cmd` in tmux window `name` of `session` (session created on demand). Idempotent: an existing window is left alone."""
    wrapped = f"bash -lc {shell_quote(f'set -o pipefail; export PATH=$HOME/.local/bin:$PATH; cd {cwd}; {shell_cmd}; rc=$?; echo; echo [window {name} finished rc=$rc]; sleep 86400')}"
    if not tmux_has(session):
        r = tmux("new-session", "-d", "-s", session, "-n", name, wrapped)
        return "created session" if r.returncode == 0 else f"FAILED: {r.stderr}"
    if name in tmux_windows(session):
        return "exists"
    r = tmux("new-window", "-d", "-t", session, "-n", name, wrapped)
    return "started" if r.returncode == 0 else f"FAILED: {r.stderr}"


def tmux_kill_window(session: str, name: str):
    tmux("kill-window", "-t", f"{session}:{name}")


def shell_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)


# ----------------------------------------------------------------------------------------------------------------------------- GPU
def gpus() -> list[dict]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    res = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 6:
            res.append(dict(index=int(p[0]), name=p[1], total_mib=int(p[2]), used_mib=int(p[3]), free_mib=int(p[4]), util=int(p[5])))
    return res


def need_gib(num_envs: int) -> float:
    """Rough peak GPU memory of the trainer (measured: 4096 envs + evaluator daemon ~ 8.7 GiB)."""
    return 3.0 + 1.4 * num_envs / 1024.0


def pick_gpu(spec: str = "auto", want_envs: int = 4096) -> dict:
    """Returns dict(index, num_envs, free_gib, util, name, warnings=[...]). spec = 'auto' | '<index>'."""
    g = gpus()
    if not g:
        return dict(index=None, num_envs=want_envs, warnings=["nvidia-smi not available: GPU not detected (training needs a CUDA GPU)"], name=None, free_gib=0, util=0)
    if spec != "auto":
        g = [x for x in g if x["index"] == int(spec)] or die(f"GPU {spec} not found") or g
    best = max(g, key=lambda x: (x["free_mib"] - 20 * x["util"]))  # most free memory, penalise a busy GPU
    free = best["free_mib"] / 1024
    n, warn = want_envs, []
    for cand in sorted({want_envs, 4096, 2048, 1024, 512}, reverse=True):
        if cand <= want_envs and need_gib(cand) <= free * 0.92:
            n = cand
            break
    else:
        n = 512
        warn.append(f"only {free:.1f} GiB free: even 512 envs may not fit")
    if n != want_envs:
        warn.append(f"num_envs reduced {want_envs} -> {n} to fit {free:.1f} GiB free GPU memory (recipe differs from the config default)")
    if best["util"] >= 60:
        warn.append(f"GPU {best['index']} is {best['util']}% busy (another job?): both jobs will slow down")
    return dict(index=best["index"], name=best["name"], num_envs=n, free_gib=round(free, 1), util=best["util"], warnings=warn)


def train_procs(run_dir: Path | None = None) -> list[str]:
    """Live partymimic.train processes (optionally those writing to run_dir)."""
    r = subprocess.run(["pgrep", "-af", "partymimic.train"], capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if "-m partymimic.train" in l and "pgrep" not in l]
    if run_dir is not None:
        lines = [l for l in lines if str(run_dir) in l]
    return lines
