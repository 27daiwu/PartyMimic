"""package: build the C++ Sim2Real deployment package for a dance (no robot connection).

Reuses the latest C++ framework (deploy/partymimic_cpp) UNCHANGED: whitelist + denylist are read from its own dev/make_zip.sh, so the package contains
exactly what the validated framework ships (remote-controller chords, SETTLE/HOLD exit-recovery state machine, guardian, logging, analysis hooks).
Only the *assets* are replaced (policy.onnx / policy_params.txt / motion.bin / fk_chain.txt + the sha256 constants compiled into the binary + the
golden self-check actions + the mode table), all derived from the run's admitted final model and the deploy config (versioned, decoupled from training).

Safety gate: unless Sim2Sim (and ONNX parity, and the evaluation eligibility) passed, packaging refuses. `--allow-unadmitted` builds a clearly marked
UNADMITTED package for offline inspection only.
Verification on the dev PC (x86): the staged sources are compiled with the pip onnxruntime, gen_golden rebases the golden actions onto the new
policy, then `partymimic selfcheck` runs the whole offline self-check (numerics, timing, CRC layout, exit/guard logic) - none of it touches a robot.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat as _stat
import subprocess
import tempfile
import zipfile
from pathlib import Path

from .core import CPP, PY, REPO, Logger, Run, die, now, py_env, read_json, sh, sha256, write_json

ORT_PY = REPO / ".venv/lib/python3.12/site-packages/onnxruntime/capi"


def parse_make_zip() -> tuple[list[str], str]:
    """(whitelist files, denylist regex) from the framework's own make_zip.sh (single source of truth)."""
    t = (CPP / "dev/make_zip.sh").read_text()
    wl_block = re.search(r"WL=\((.*?)\)\n", t, re.S).group(1)
    wl = [x for x in re.findall(r"[\w./+-]+", wl_block) if "/" in x or "." in x]
    wl += [str(p.relative_to(CPP)) for p in sorted((CPP / "src").glob("*.cpp"))] + [str(p.relative_to(CPP)) for p in sorted((CPP / "include/pm").glob("*.hpp"))]
    bad = re.search(r"^BAD='(.*)'$", t, re.M).group(1)
    return sorted(set(wl)), bad


def ceil1(x: float) -> float:
    return math.ceil(x * 10 - 1e-9) / 10


def _cpp_const(name: str) -> float:
    """A `constexpr double NAME = x;` of the framework's app.hpp (single source of truth: nothing is duplicated here)."""
    return float(re.search(rf"constexpr double {name} = ([0-9.]+);", (CPP / "include/pm/app.hpp").read_text()).group(1))


def _site_default(key: str) -> float:
    return float(re.search(rf"^{key}\s+([0-9.]+)", (CPP / "config/site.conf").read_text(), re.M).group(1))


def run_caps(dur_s: float, site_overrides: dict | None = None) -> dict:
    """Run-window caps for a dance of `dur_s` seconds. The `dance` mode window is the WHOLE dance (rounded up to 0.1 s; the runner's own 1 s lead-in is on top and is not
    counted in run_s). The framework refuses run_s > site.conf max_run_s (default 12 s, deliberately conservative) and max_run_s > ABS_MAX_RUN_S (compile-time ceiling).
    A dance longer than the default cap therefore gets an explicit, recorded max_run_s override in the package's site.conf (never larger than the dance window); a dance
    longer than the compile-time ceiling cannot be packaged (it would be cut) - raise ABS_MAX_RUN_S in app.hpp after review instead."""
    need = ceil1(dur_s)
    ceiling, default = _cpp_const("ABS_MAX_RUN_S"), _site_default("max_run_s")
    have = float((site_overrides or {}).get("max_run_s", default))
    out = dict(dance_seconds=dur_s, dance_window_s=need, site_default_max_run_s=default, compile_time_ceiling_s=ceiling, fits_default=need <= default + 1e-9, truncated=need > ceiling + 1e-9)
    out["max_run_s"] = have if have >= need - 1e-9 else need   # explicit overrides larger than the dance are kept; smaller ones are raised to the dance window
    out["override"] = None if out["max_run_s"] <= default + 1e-9 else dict(max_run_s=out["max_run_s"], reason=f"dance window {need:g} s > framework default max_run_s {default:g} s; compile-time ceiling {ceiling:g} s",
                                                                            still_needed="nothing (V1.6.1: no authorisation exists; only this recorded site.conf cap)")
    return out


def mode_seconds(dur_s: float, m: dict) -> dict:
    """Window of every mode in seconds: configured value, never longer than the dance (rounded up to 0.1 s) nor than the compile-time ceiling (ABS_MAX_RUN_S, 60 s)."""
    cap = min(ceil1(dur_s), _cpp_const("ABS_MAX_RUN_S"))
    return dict(hold=m["hold_s"], exit=m["exit_s"], clip2=min(m["clip2_s"], cap), clip4=min(m["clip4_s"], cap), dance=cap)


def patch_selfcheck_modes(stage: Path, sec: dict, max_run_s: float | None = None):
    """The framework's selfcheck 7f hard-codes the SSB mode windows (3 / 8 / 2 / 4 / 7.1 s). A package for another dance has other windows: the staged copy's
    expectation is rewritten to the generated table (one line; nothing else in the check changes). Recorded in PACKAGE_INFO.md."""
    f = stage / "src/selfcheck.cpp"
    t = f.read_text()
    g = lambda x: repr(float(x))
    t2 = t.replace("ms[1].run_s == 3.0", f"ms[1].run_s == {g(sec['hold'])}").replace("ms[3].run_s == 2.0", f"ms[3].run_s == {g(sec['clip2'])}") \
          .replace("ms[4].run_s == 4.0", f"ms[4].run_s == {g(sec['clip4'])}").replace("ms[5].run_s == 7.1", f"ms[5].run_s == {g(sec['dance'])}").replace("ms[2].run_s == 8.0", f"ms[2].run_s == {g(sec['exit'])}")
    assert t2 != t or sec == dict(hold=3.0, exit=8.0, clip2=2.0, clip4=4.0, dance=7.1), "selfcheck 7f expectation not found"
    if max_run_s is not None and abs(max_run_s - 12.0) > 1e-9:  # 7d asserts the shipped site.conf cap; a package with its own max_run_s override expects that value
        t3 = t2.replace("st.max_run_s == 12.0", f"st.max_run_s == {g(max_run_s)}")
        assert t3 != t2, "selfcheck 7d max_run_s expectation not found"
        t2 = t3
    f.write_text(t2)


def gen_modes(base: str, dance: str, dur_s: float, m: dict) -> str:
    """modes.conf of the framework with the dance-specific durations / texts. Mode names and columns are unchanged (selfcheck 7f parses them)."""
    sec = mode_seconds(dur_s, m)
    run_dance = sec["dance"]
    lines = []
    for ln in base.splitlines():
        mm = re.match(r"mode_(\w+)\s+(\S+)\s+(\S+)\s+(\d)\s+(\S+)\s+(.*)", ln)
        if not mm:
            lines.append(ln)
            continue
        name, ref, rs, ev, voice, desc = mm.groups()
        if name == "hold":
            rs, ev, desc = f"{sec['hold']}", "0", f"接管后保持「{dance}」动作第 0 帧（动作首帧）{sec['hold']:g} 秒，再交还 WalkRun（该动作首帧尚无真机证据；沿用框架的已验证包络门槛，请按未验证对待）"
        elif name == "exit":
            rs, desc = f"{sec['exit']}", f"保持「{dance}」第 0 帧最长 {sec['exit']:g} 秒；运动中双击 START 验证主动退出 -> WalkRun 恢复"
        elif name == "clip2":
            rs, desc = f"{sec['clip2']}", f"播放「{dance}」前 {sec['clip2']:g} 秒，冻结 0.5 秒后交还"
        elif name == "clip4":
            rs, desc = f"{sec['clip4']}", f"播放「{dance}」前 {sec['clip4']:g} 秒，冻结 0.5 秒后交还"
        elif name == "dance":
            rs, desc = f"{run_dance}", f"整段「{dance}」{dur_s:.2f} 秒（取 {run_dance:g} 秒窗口）+ 1 秒前导冻结 + 0.5 秒末帧保持后交还"
        lines.append(f"mode_{name}  {ref:<8s} {rs:>4s}  {ev}  {voice}  {desc}")
    return "\n".join(["# GENERATED by ./pm package from the framework's config/modes.conf (dance-specific durations / texts only; same mode names and columns)."] + lines) + "\n"


def start_sh_prompts(text: str) -> list[str]:
    """non-comment lines of start.sh that read input. The ONLY allowed `read` is the mode-table parser (`while read -r name ref rs val voice desc ... < <(...)`), fed from config/modes.conf."""
    out = []
    for l in text.splitlines():
        if l.lstrip().startswith("#"):
            continue
        if re.search(r"(^|[^\w-])read\s+-", l) and "while read -r name ref rs val voice desc" not in l:
            out.append(l.strip())
    return out


def v16_package_rules(stage: Path, dcfg: dict) -> dict:
    """Packaging rules that cannot be switched off (a violation aborts packaging).
    V1.6.1 (`no_authorisation`): the manual-authorisation design is GONE: no authorize.sh / config/authz.conf / src/authz.cpp, no authorisation / confirmation wording or interactive input in the scripts,
    no READY timeout in the runner/service, ack.sh does not prompt, announce_wait_s in [0, 5] and equal to the deploy config.
    The old V1.6 config (cpp_g1_v3, empty register + authorize.sh) cannot be built from this framework any more."""
    v = dcfg.get("v16") or {}
    out = {}
    if not v.get("no_authorisation"):
        die(f"deploy config {dcfg.get('name')} still describes the V1.6 manual-authorisation design (authorize.sh / authz.conf), which the framework no longer has: use --deploy-cfg cpp_g1_v3_1", 6)
    gone = [f for f in ("authorize.sh", "config/authz.conf", "src/authz.cpp") if (stage / f).exists()]
    if gone:
        die("V1.6.1 package rule: removed authorisation files are present: " + ", ".join(gone), 6)
    out["authorisation_files"] = "authorize.sh / config/authz.conf / src/authz.cpp: absent"
    words = re.compile(r"I-ACCEPT|--phrase|--confirm-protection|--i-confirm-protection|authz|authorize\.sh|--revoke|--authorize")
    hits = []
    for f in ("start.sh", "build.sh", "ack.sh", "stop.sh"):
        for i, l in enumerate((stage / f).read_text().splitlines(), 1):
            if words.search(l):
                hits.append(f"{f}:{i}")
    if hits:
        die("V1.6.1 package rule: authorisation / confirmation wording in the scripts: " + ", ".join(hits[:6]), 6)
    for f in ("start.sh", "ack.sh"):
        prompts = start_sh_prompts((stage / f).read_text())
        if v.get("start_sh_no_prompt", True) and prompts:
            die(f"V1.6.1 package rule: {f} must not read any interactive input: " + "; ".join(prompts[:3]), 6)
    out["scripts"] = "start.sh / ack.sh: no interactive read (the only `read` is the mode-table parser fed from config/modes.conf); no authorisation or confirmation wording in start/build/ack/stop"
    if v.get("ready_no_timeout"):
        src = (stage / "src/runner.cpp").read_text() + (stage / "src/service.cpp").read_text()
        if re.search(r"READY_TIMEOUT|READY timed out|ready_timeout", src):
            die("V1.6.1 package rule: the runner/service still has a READY timeout", 6)
        out["ready"] = "READY has no timeout (held until L1+A / START x2)"
    site = (stage / "config/site.conf").read_text()
    m = re.search(r"^announce_wait_s\s+([0-9.]+)", site, re.M)
    if not m or not (0 <= float(m.group(1)) <= 5):
        die("V1.6.1 package rule: site.conf announce_wait_s missing or outside [0, 5] s", 6)
    if v.get("announce_wait_s") is not None and abs(float(m.group(1)) - float(v["announce_wait_s"])) > 1e-9:
        die(f"V1.6.1 package rule: site.conf announce_wait_s {m.group(1)} != deploy config {v['announce_wait_s']}", 6)
    out["announce_wait_s"] = float(m.group(1))
    return out


def apply_site_overrides(base: str, ov: dict) -> str:
    out, done = [], set()
    for ln in base.splitlines():
        k = ln.split()[0].lstrip("#") if ln.split() else ""
        if k in ov and (not ln.startswith("#") or ln.startswith("#" + k)):
            if k not in done:
                out.append(f"{k} {ov[k]}")
                done.add(k)
            continue
        out.append(ln)
    out += [f"{k} {v}" for k, v in ov.items() if k not in done]
    return "\n".join(out) + "\n"


def write_frozen(stage: Path, A: Path, policy_name: str):
    f = stage / "include/pm/frozen.hpp"
    t = f.read_text()
    h = lambda n: sha256(A / n)
    t = re.sub(r'(FROZEN_SHA256_POLICY = ")[0-9a-f]{64}(";)[^\n]*', lambda m: f'{m.group(1)}{h("policy.onnx")}{m.group(2)}  // {policy_name} (generated by ./pm package)', t)
    t = re.sub(r'(FROZEN_SHA256_POLICY_PARAMS = ")[0-9a-f]{64}"', lambda m: f'{m.group(1)}{h("policy_params.txt")}"', t)
    t = re.sub(r'(FROZEN_SHA256_MOTION = ")[0-9a-f]{64}"', lambda m: f'{m.group(1)}{h("motion.bin")}"', t)
    t = "// GENERATED by ./pm package: sha256 of this package's deployed artifacts (checked at every F1+F3 and by selfcheck). P2 baseline hash kept for provenance.\n" + t
    f.write_text(t)


def patch_golden(hpp: Path, line: str):
    t = hpp.read_text()
    t2 = re.sub(r"constexpr double GOLDEN_ACTION\[\] = \{[^}]*\};", lambda m: line.strip(), t)
    assert t2 != t or line.strip() in t
    hpp.write_text(t2)


def dev_build_check(stage: Path, work: Path, A: Path, log) -> dict:
    """Compile the staged package on this x86 PC, rebase the golden onto the new policy, run the offline selfcheck. Returns the report dict."""
    rep = dict(attempted=True, passed=False)
    if not (ORT_PY / "libonnxruntime.so.1.32.0").exists() and not list(ORT_PY.glob("libonnxruntime.so.*")):
        rep.update(attempted=False, reason="pip onnxruntime shared library not found")
        return rep
    shutil.rmtree(work, ignore_errors=True)
    work.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(stage, work)
    shutil.copytree(CPP / "dev", work / "dev")  # dev tools only exist in this temporary copy (gen_golden); never in the package
    ort = work.parent / "_ort"
    shutil.rmtree(ort, ignore_errors=True)
    (ort / "lib").mkdir(parents=True)
    shutil.copytree(CPP / "third_party/onnxruntime/include", ort / "include")
    so = sorted(ORT_PY.glob("libonnxruntime.so.*"))[0]
    shutil.copy2(so, ort / "lib" / so.name)
    (ort / "lib/libonnxruntime.so.1").symlink_to(so.name)
    (ort / "lib/libonnxruntime.so").symlink_to(so.name)
    lg = work.parent / "devbuild.log"
    lg.write_text("")
    env = {"LD_LIBRARY_PATH": f"/usr/local/lib:{ort / 'lib'}"}

    def build(targets):
        r = sh(["cmake", "-S", work, "-B", work / "build", "-DCMAKE_BUILD_TYPE=Release", f"-DPM_ORT_ROOT={ort}"], log=lg)
        if r.returncode:
            return False
        return sh(["cmake", "--build", work / "build", "-j8", "--target", *targets], log=lg).returncode == 0
    if not build(["partymimic", "gen_golden"]):
        rep["reason"] = f"dev build failed (see {lg})"
        return rep
    g = sh([work / "build/dev/gen_golden" if (work / "build/dev/gen_golden").exists() else work / "build/gen_golden", work], env=env, capture=True)
    if g.returncode or "GOLDEN_ACTION" not in g.stdout:
        rep["reason"] = "gen_golden failed: " + (g.stderr or g.stdout)[-300:]
        return rep
    line = [l for l in g.stdout.splitlines() if "GOLDEN_ACTION" in l][0]
    for h in (stage / "include/pm/selfcheck_golden.hpp", work / "include/pm/selfcheck_golden.hpp"):
        patch_golden(h, line)
    if not build(["partymimic"]):
        rep["reason"] = f"rebuild failed (see {lg})"
        return rep
    sc = sh([work / "build/partymimic", "selfcheck"], env=env, capture=True, timeout=900)
    (work.parent / "selfcheck.txt").write_text(sc.stdout + sc.stderr)
    txt = sc.stdout
    rep.update(passed=(sc.returncode == 0 and "SELFCHECK PASS" in txt), rc=sc.returncode, summary=[l for l in txt.splitlines() if l.startswith("SELFCHECK")][-1:],
               n_pass=txt.count("[PASS]"), n_warn=txt.count("[WARN]"), n_fail=txt.count("[FAIL]"), log=str(work.parent / "selfcheck.txt"),
               ort=so.name, note="x86 dev PC with pip onnxruntime; the robot's own build.sh re-runs the selfcheck on aarch64")
    # independent cross-check: the Python replica of the deployment stack (used by Sim2Sim) vs the C++ golden actions
    try:
        rep["python_replica_vs_cpp_max_abs"] = replica_vs_golden(A, stage / "include/pm/selfcheck_golden.hpp")
    except Exception as e:  # informational only
        rep["python_replica_vs_cpp_max_abs"] = f"n/a ({type(e).__name__}: {e})"
    return rep


def replica_vs_golden(A: Path, hpp: Path) -> float:
    import sys
    import numpy as np
    sys.path.insert(0, str(REPO / "results/p4/scripts"))
    from pm_stack import Stack, load_motion, load_params
    t = hpp.read_text()
    G = {k: np.array([float(x) for x in re.search(rf"GOLDEN_{k}\[\] = \{{([^}}]*)\}}", t).group(1).split(",")]) for k in ("Q", "DQ", "QUAT", "GYRO", "ACTION")}
    st = Stack(str(A / "policy.onnx"), load_params(str(A / "policy_params.txt")), load_motion(str(A / "motion.bin")))
    s = lambda k: (G["Q"][k * 29:(k + 1) * 29], G["DQ"][k * 29:(k + 1) * 29], G["GYRO"][k * 3:(k + 1) * 3], G["QUAT"][k * 4:(k + 1) * 4])
    q, dq, g, qu = s(0)
    st.reset(q, dq, g, qu, paused=False)
    d = 0.0
    for k in range(3):
        q, dq, g, qu = s(k)
        a, _ = st.tick(q, dq, g, qu)
        d = max(d, float(np.abs(a - G["ACTION"][k * 29:(k + 1) * 29]).max()))
    return d


GENERATED_PER_PACKAGE = {"ADMISSION.json", "PACKAGE_INFO.md", "include/pm/frozen.hpp", "include/pm/version.hpp", "include/pm/selfcheck_golden.hpp"}  # rewritten for every package by design


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def previous_package(run: Run, compare_to: str | Path | None = None) -> dict | None:
    """The package this one replaces (default: results/<dance>/.../deploy/latest.json's zip; or an explicit zip). Read BEFORE the old files are overwritten."""
    lj = read_json(run.deploy / "latest.json", {}) or {}
    zp = Path(compare_to) if compare_to else Path(lj.get("zip", ""))
    if not zp.is_file():
        return None
    files = {}
    with zipfile.ZipFile(zp) as zf:
        for n in zf.namelist():
            if not n.endswith("/"):
                files[n.split("/", 1)[1]] = _sha_bytes(zf.read(n))
        app = zf.read("partymimic_cpp/include/pm/app.hpp").decode(errors="replace") if "partymimic_cpp/include/pm/app.hpp" in zf.namelist() else ""
    m = re.search(r"ABS_MAX_RUN_S\s*=\s*([0-9.]+)", app)
    info = (lj.get("info") or {}) if str(lj.get("zip", "")) == str(zp) else {}
    return dict(zip=zp.name, zip_sha256=sha256(zp), built=info.get("built"), onnx_sha256=files.get("assets/policy.onnx"), files=files, abs_max_run_s=float(m.group(1)) if m else None)


def framework_changes(prev: dict | None, stage: Path) -> dict:
    """What changed in the shipped package vs the previous one: framework source files (the interesting part) / per-package generated files / assets."""
    if not prev:
        return dict(previous=None, note="no previous package to compare with")
    new = {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file() and str(p.relative_to(stage)) not in ("ADMISSION.json", "PACKAGE_INFO.md")}
    old = {k: v for k, v in prev["files"].items() if k not in ("ADMISSION.json", "PACKAGE_INFO.md")}
    changed = sorted(k for k in new if k in old and new[k] != old[k]); added = sorted(set(new) - set(old)); removed = sorted(set(old) - set(new))
    bucket = lambda ks, pred: [k for k in ks if pred(k)]
    is_asset = lambda k: k.startswith("assets/")
    is_gen = lambda k: k in GENERATED_PER_PACKAGE
    fw = lambda k: not is_asset(k) and not is_gen(k)
    app = (stage / "include/pm/app.hpp").read_text(errors="replace") if (stage / "include/pm/app.hpp").exists() else ""
    m = re.search(r"ABS_MAX_RUN_S\s*=\s*([0-9.]+)", app)
    new_onnx = new.get("assets/policy.onnx")
    return dict(previous=dict(zip=prev["zip"], zip_sha256=prev["zip_sha256"], built=prev.get("built")),
                model=dict(onnx_sha256_previous=prev.get("onnx_sha256"), onnx_sha256_new=new_onnx, onnx_unchanged=bool(new_onnx and new_onnx == prev.get("onnx_sha256"))),
                framework_files_changed=bucket(changed, fw), framework_files_added=bucket(added, fw), framework_files_removed=bucket(removed, fw),
                asset_files_changed=bucket(changed + added, is_asset), generated_files_changed=bucket(changed, is_gen),
                compile_time_ceiling_s=dict(previous=prev.get("abs_max_run_s"), new=float(m.group(1)) if m else None),
                note="framework_* = C++ sources/scripts/config of deploy/partymimic_cpp that differ from the previous package (policy logic unchanged unless listed); generated_* = rewritten per package by design (hash constants, policy name, golden actions)")


# ---- V1.5.1: zip with REAL permission bits (scripts 0755 + regular-file type bit + directory entries) and a verification by independent extractors ---------------------------------------
def _is_script(rel: str, data: bytes) -> bool:
    return rel.endswith((".sh", ".so")) or ".so." in rel or data[:2] == b"#!"


def write_zip(zpath: Path, stage: Path, files: list[Path], script_mode: int = 0o755, data_mode: int = 0o644) -> list[str]:
    """Deterministic zip: every entry gets an explicit Unix mode (file type bit + permission bits, create_system = Unix), directories get explicit entries. Returns the *.sh script list."""
    scripts: list[str] = []
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        dirs = {"partymimic_cpp"}
        for p in files:
            for q in p.relative_to(stage).parents:
                if str(q) != ".":
                    dirs.add(str(Path("partymimic_cpp") / q))
        for d in sorted(dirs):
            zi = zipfile.ZipInfo(d.rstrip("/") + "/", date_time=(2026, 1, 1, 0, 0, 0))
            zi.create_system = 3
            zi.external_attr = ((_stat.S_IFDIR | 0o755) << 16) | 0x10
            zf.writestr(zi, b"")
        for p in files:
            rel = str(p.relative_to(stage))
            data = p.read_bytes()
            exe = _is_script(rel, data)
            zi = zipfile.ZipInfo(f"partymimic_cpp/{rel}", date_time=(2026, 1, 1, 0, 0, 0))
            zi.create_system = 3
            zi.external_attr = (_stat.S_IFREG | (script_mode if exe else data_mode)) << 16
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, data)
            if exe and rel.endswith(".sh"):
                scripts.append(rel)
    return scripts


def verify_zip_permissions(zpath: Path, scripts: list[str], extractors=("unzip", "bsdtar", "python")) -> dict:
    """Check the finished zip: recorded modes, then extraction with every available extractor (every script must be executable right after extraction, no chmod). Failure aborts packaging."""
    res: dict = dict(scripts=scripts, extractors={})
    with zipfile.ZipFile(zpath) as zf:
        bad = []
        for rel in scripts:
            a = zf.getinfo(f"partymimic_cpp/{rel}").external_attr >> 16
            if not (_stat.S_ISREG(a) and (a & 0o111) == 0o111):
                bad.append(f"{rel}: {oct(a)}")
        for i in zf.infolist():
            if i.is_dir():
                continue
            a = i.external_attr >> 16
            if i.create_system != 3 or not _stat.S_ISREG(a) or not a & 0o444:
                bad.append(f"{i.filename}: create_system {i.create_system} mode {oct(a)}")
        if bad:
            die("zip permission check FAILED (recorded modes): " + "; ".join(bad[:6]), 6)
        res["recorded_modes"] = f"OK: {len(scripts)} scripts 0755 + regular-file bit, all other files 0644, create_system=Unix, {sum(1 for i in zf.infolist() if i.is_dir())} directory entries"
    for ex in extractors:
        if ex == "python":  # zipfile.extractall() ignores the Unix bits (a naive extractor): the package must still start with `bash start.sh` (it chmods its siblings)
            res["extractors"][ex] = "ignores unix modes by design; `bash start.sh` / `bash build.sh` self-heal (chmod +x of the sibling scripts)"
            continue
        exe = shutil.which("unzip") if ex == "unzip" else shutil.which("bsdtar")
        if not exe:
            res["extractors"][ex] = "not available on this machine (skipped)"
            continue
        with tempfile.TemporaryDirectory() as td:
            cmd = [exe, "-q", str(zpath), "-d", td] if ex == "unzip" else [exe, "-xf", str(zpath), "-C", td]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                die(f"zip extraction with {ex} failed: {r.stderr[-200:]}", 6)
            miss = [rel for rel in scripts if not os.access(Path(td) / "partymimic_cpp" / rel, os.X_OK)]
            if miss:
                die(f"zip permission check FAILED: after `{ex}` these scripts are not executable: {miss}", 6)
            res["extractors"][ex] = f"OK: {len(scripts)} scripts executable straight after extraction (no chmod)"
    return res


def timeline_check(dance_frames: int, fps: float, run_s: float, lead_in_s: float, end_hold_s: float, min_margin: float) -> dict:
    """Python twin of the framework's plan_timeline() (selfcheck 7k runs the C++ one): lead-in + WHOLE clip + last-frame hold must fit inside the runner fail-safe and reach the last frame."""
    lead = int(round(lead_in_s * fps))
    clip = dance_frames + lead
    seg = min(clip - 1, lead + int(round(run_s * fps)))
    over_tick = max(0, seg - 1)
    planned = over_tick / fps + end_hold_s
    failsafe = lead / fps + run_s + end_hold_s + 1.0
    out = dict(dance_frames=dance_frames, fps=fps, dance_s=dance_frames / fps, lead_in_s=lead / fps, lead_in_frames=lead, window_s=run_s, seg_frames=seg, last_frame_index=clip - 1,
               covers_whole_dance=seg == clip - 1, last_frame_hold_s=end_hold_s, planned_total_s=planned, failsafe_s=failsafe, margin_s=failsafe - planned)
    if not out["covers_whole_dance"] or out["margin_s"] < min_margin:
        die(f"run timeline check FAILED: window {run_s} s does not reach the last of the {dance_frames} frames (covers_whole_dance={out['covers_whole_dance']}) or the runner fail-safe margin "
            f"{out['margin_s']:.2f} s < {min_margin} s", 6)
    return out


def package(run: Run, dcfg: dict, allow_unadmitted: bool = False, build_check: bool | None = None, log=None, compare_to: str | Path | None = None) -> dict:
    log = log or Logger(run.logs / "package.log")
    adm = run.admission()
    ok = adm.get("real_robot_admission") == "PASS"
    if not ok and not allow_unadmitted:
        why = adm.get("real_robot_admission", "no admission record (run `./pm sim2sim`)")
        die(f"admission is '{why}': Sim2Sim / ONNX parity / eval gate not all passed -> refusing to build a deployment package.\n"
            f"  see {run.reports / 'SELECTION_REPORT.md'}.  (offline inspection only: add --allow-unadmitted -> package is marked UNADMITTED)", 5)
    fin = adm.get("final")
    if not fin:
        die("no exported final model: run `./pm export` / `./pm sim2sim` first", 5)
    A = Path(fin["assets"])
    it = fin["iteration"]
    dance = run.dance.name
    tag = f"{dance}_{run.id}_it{it:06d}_{dcfg['name']}" + ("" if ok else "_UNADMITTED")
    pname = f"{dcfg.get('package_prefix', 'partymimic_cpp')}_{tag}"
    base = run.deploy / pname
    stage = base / "partymimic_cpp"
    prev_pkg = previous_package(run, compare_to)  # read before anything of the old package is overwritten
    shutil.rmtree(base, ignore_errors=True)
    stage.mkdir(parents=True)
    wl, bad = parse_make_zip()
    for f in wl:
        if re.search(bad, f, re.I):
            die(f"framework whitelist entry denied by its own denylist: {f}")
        if not (CPP / f).is_file():
            die(f"framework whitelist file missing: {f}")
        (stage / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CPP / f, stage / f)
    for n in ("policy.onnx", "policy_params.txt", "motion.bin", "fk_chain.txt", "SHA256SUMS"):
        shutil.copy2(A / n, stage / "assets" / n)
    # hash constants compiled into the binary + policy name in the run logs
    write_frozen(stage, stage / "assets", f"{dance}_it{it}")
    v = stage / "include/pm/version.hpp"
    v.write_text(re.sub(r'(PM_POLICY_NAME = ")[^"]*(";)', lambda m: f'{m.group(1)}{dance}_it{it}{m.group(2)}', v.read_text()))
    # modes + site config from the versioned deploy config
    import numpy as np
    z = np.load(run.motion_npz, allow_pickle=True)
    dur = len(z["joint_pos"]) / float(z["fps"][0])
    caps = run_caps(dur, dcfg.get("site_overrides"))
    if caps["truncated"]:
        die(f"dance is {dur:.1f} s: its window {caps['dance_window_s']:g} s exceeds the framework's compile-time ceiling ABS_MAX_RUN_S={caps['compile_time_ceiling_s']:g} s, so it would be cut. "
            f"Refusing to package a truncated dance (raise ABS_MAX_RUN_S in deploy/partymimic_cpp/include/pm/app.hpp after review).", 6)
    site_ov = dict(dcfg.get("site_overrides") or {})
    if caps["override"]:
        site_ov["max_run_s"] = f"{caps['max_run_s']:g}"
    patch_selfcheck_modes(stage, mode_seconds(dur, dcfg["modes"]), caps["max_run_s"])
    (stage / "config/modes.conf").write_text(gen_modes((CPP / "config/modes.conf").read_text(), dance, dur, dcfg["modes"]))
    (stage / "config/site.conf").write_text(apply_site_overrides((CPP / "config/site.conf").read_text(), site_ov))
    v16 = v16_package_rules(stage, dcfg) if dcfg.get("v16") else None
    bc = dcfg.get("dev_build_check", True) if build_check is None else build_check
    rep = dev_build_check(stage, base / "devbuild/partymimic_cpp", stage / "assets", log) if bc else dict(attempted=False, reason="disabled")
    log("dev build / selfcheck", **{k: v for k, v in rep.items() if k in ("attempted", "passed", "summary", "n_pass", "n_warn", "n_fail", "reason", "python_replica_vs_cpp_max_abs")})
    adm_rules = dcfg.get("admission", {})
    if adm_rules.get("require_cpp_selfcheck", True) and bc and not rep.get("passed") and not allow_unadmitted:
        die(f"C++ dev build / selfcheck failed: {rep.get('reason') or rep.get('summary')} (see {base})", 6)
    # sanity: shipped assets == compiled constants == SHA256SUMS
    frozen = (stage / "include/pm/frozen.hpp").read_text()
    for n, key in (("policy.onnx", "FROZEN_SHA256_POLICY"), ("policy_params.txt", "FROZEN_SHA256_POLICY_PARAMS"), ("motion.bin", "FROZEN_SHA256_MOTION")):
        assert sha256(stage / "assets" / n) in frozen, f"{n}: hash not compiled into frozen.hpp"
    tc = dcfg.get("timeline") or {}
    tl = timeline_check(len(z["joint_pos"]), float(z["fps"][0]), caps["dance_window_s"], _site_default("lead_in_s"), _site_default("end_hold_s"), float(tc.get("min_failsafe_margin_s", 0.5)))
    info = dict(package=pname, run_timeline=tl, v16_rules=v16, dance=dance, run=run.id, iteration=it, built=now(), admitted_offline=ok, unadmitted_override=bool(allow_unadmitted and not ok),
                deploy_config=dict(name=dcfg["name"], sha256=dcfg["_sha256"]), framework=str(dcfg.get("framework", "deploy/partymimic_cpp")),
                model=dict(checkpoint=fin["checkpoint"], onnx_sha256=sha256(stage / "assets/policy.onnx"), params_sha256=sha256(stage / "assets/policy_params.txt"), motion_sha256=sha256(stage / "assets/motion.bin")),
                dance_seconds=dur, run_caps=caps, sim2sim=adm.get("sim2sim"), onnx_parity=adm.get("onnx_parity"), eval_gate=adm.get("eval_gate"), cpp_dev_selfcheck=rep,
                real_robot_status="NOT TESTED ON THE ROBOT. This package never connects to a robot by itself; ./start.sh only listens, the robot moves only after F1+F3 (checks) and L1+A on the remote (V1.6.1: no authorisation or confirmation step; see README.md).")
    prov = adm.get("provenance") or dict(note="admission record predates V1.2 (no provenance block): re-run `./pm sim2sim <dance>`")
    fc = framework_changes(prev_pkg, stage)
    info.update(evaluator_version=prov.get("evaluator_version"), sim2sim_inits=(adm.get("sim2sim") or {}).get("inits") or prov.get("sim2sim_inits"), hard_reeval=prov.get("hard_reeval"), train_source=prov.get("train_source"),
                admission_provenance=prov, framework_changes_since_previous_package=fc)
    write_json(stage / "ADMISSION.json", info)
    note = [f"# PartyMimic 部署包：{dance}（{run.id} it{it}）", "", f"- 生成 {now()}；部署配置 `{dcfg['name']}`（sha256 {dcfg['_sha256'][:12]}）；C++ 框架 `deploy/partymimic_cpp`（{'V1.6.1' if dcfg.get('v16') else 'V1.5.1'}：模型/动作资产与冻结哈希按本舞蹈替换；selfcheck 7f 对 SSB 模式时长(3/8/2/4/7.1 s)的期望改成本舞蹈的模式表）",
            f"- **准入状态：{'离线准入 PASS（Sim2Sim/ONNX 一致性/评估均通过）——仅允许进入有保护的真机试验，尚无任何真机结论' if ok else '❌ UNADMITTED：离线准入未通过，禁止上机，仅供离线检查'}**",
            f"- 开发机 x86 构建 + selfcheck：{'PASS' if rep.get('passed') else ('跳过' if not rep.get('attempted') else 'FAIL')}（机载 aarch64 仍需 `./build.sh`，其内含同一套 selfcheck）",
            f"- 动作时长 {dur:.2f} s；`dance` 模式窗口 {caps['dance_window_s']:g} s（整段；runner 另在前面补 1 s 冻结首帧，不计入 run_s）。" + (f"**本包 site.conf 覆盖 `max_run_s {caps['max_run_s']:g}`（框架默认 {caps['site_default_max_run_s']:g} s，编译期上限 {caps['compile_time_ceiling_s']:g} s）：动作超过默认上限，否则会被截断。**" if caps["override"] else f"未超过框架默认 max_run_s {caps['site_default_max_run_s']:g} s，site.conf 未改。"),
            f"- 测试模式见 `./start.sh --list`（hold 的“首帧”是 {dance} 的第 0 帧，尚无真机证据）", "",
            f"## 上机（{'V1.6.1' if dcfg.get('v16') else 'V1.5.1'}）", "",
            f"1. `unzip partymimic_cpp_*.zip && cd partymimic_cpp && ./build.sh`（脚本自带可执行权限；若解压工具丢了权限位，`bash build.sh` / `bash start.sh` 会给同目录脚本补 +x）。",
            f"2. `./start.sh` **默认加载完整舞蹈（dance，{caps['dance_window_s']:g} s 窗口）**，没有测试类型菜单。服务启动即后台预热 runner（模型加载、推理预热、DDS 监视），F1+F3 只做实时检查。",
            "3. **没有任何人工授权 / 确认 / 审批步骤（V1.6.1）**：`./start.sh` 不读取输入（`< /dev/null` 不阻塞），直接启动完整舞蹈服务（除只读 `--mode check` 外全部武装）。旧的 `authorize.sh` / `config/authz.conf` / `--yes` / `--accept-unvalidated` / 授权有效期已删除（旧参数被忽略并提示）。保护全部自动：F1+F3 的实时准入、guardian、运行时失稳退出、双击 START、L2+B、e-stop 文件。",
            f"3b. **L1+A 后先播报“开始执行”，播报完成后再等待 {v16['announce_wait_s'] if v16 else 0.5:g} s（site.conf `announce_wait_s`，范围 0–5 s）才接管**；播报时长 = TtsMaker 返回 + 估算语音时长，上限 announce_max_s（默认 4 s，绝不无限等待）。等待期间尚未接管：双击 START / L2+B / e-stop 文件 / SIGTERM / guardian 异常 / lowstate 超时 / 倾角异常 / FSM 变化都只会取消（不下发任何指令，事件日志 cancel 记录原因），接管前重新确认全部前提。",
            "4. 遥控器：F1+F3 加载并检查策略 -> 准备就绪，**持续保持 READY（无超时）**直到 L1+A（或双击 START 取消）；L1+A 播报后再等待 announce_wait_s 接管、执行完整舞蹈；双击 START 交还 WalkRun；L2+B 紧急 Passive。**正常或异常结束后服务自行回到 IDLE（SETTLE -> IDLE，无 HOLD、无 ack）**，可重复 F1+F3 / L1+A；机器人若仍倒地/不在 WalkRun，下一次 F1+F3 的实时准入会自行拒绝（语音“检查失败”）。",
            f"5. 动作计时：前导冻结 {tl['lead_in_s']:g} s + 整段动作 {tl['dance_s']:.2f} s（{tl['dance_frames']} 帧，窗口覆盖最后一帧 #{tl['last_frame_index']}）+ 末帧保持 {tl['last_frame_hold_s']:g} s，规划总时长 {tl['planned_total_s']:.2f} s，runner 保险时限 {tl['failsafe_s']:.2f} s（余量 {tl['margin_s']:.2f} s）；`site.conf max_run_s` ≥ 窗口，保留 88 N·m 限幅与全部保护。", "",
            "上机步骤、遥控器操作、异常退出恢复、日志导出：见同目录 `README.md` / `EXIT_POLICY.md` / `LOGGING.md`（框架原文，未改）。", ""]
    ini = info.get("sim2sim_inits") or {}
    ts = info.get("train_source") or {}
    note += ["## 评估口径与溯源（V1.2）", "", f"- 评估器：{info.get('evaluator_version')}；硬复评 seed {((info.get('hard_reeval') or {}).get('seed'))} × {((info.get('hard_reeval') or {}).get('reps'))} 次，{((info.get('hard_reeval') or {}).get('conditions'))} 个工况（含超训练分布工况）。",
             f"- Sim2Sim：每工况 {ini.get('total', '?')} 个初态 = {ini.get('real_poses', '?')} 个真机 WalkRun 站姿 + {ini.get('sampled', '?')} 个插值采样" + ("" if ini.get("consistent", True) else "  ⚠ summary.n 与声明不一致，见 ADMISSION.json"),
             f"- 训练源码：git `{str(ts.get('git_head', '?'))[:12]}` + 未提交改动（`git diff -- src tools` sha12 `{ts.get('src_tools_diff_sha12', '?')}`）；复现快照 `{ts.get('repro_snapshot', '无')}`（`./pm snapshot <舞蹈> --verify`）。", ""]
    if prev_pkg:
        fcm = fc["model"]
        note += ["## 与上一版包的差异（framework_changes_since_previous_package）", "", f"- 上一版：`{fc['previous']['zip']}`（zip sha256 `{fc['previous']['zip_sha256'][:16]}…`，生成 {fc['previous'].get('built')}）。",
                 f"- **模型 `policy.onnx` {'未变' if fcm['onnx_unchanged'] else '已变化 ⚠'}**（{str(fcm['onnx_sha256_previous'])[:16]}… → {str(fcm['onnx_sha256_new'])[:16]}…）。",
                 f"- 编译期 ABS_MAX_RUN_S：{fc['compile_time_ceiling_s']['previous']} → {fc['compile_time_ceiling_s']['new']} s；框架文件变化 {len(fc['framework_files_changed'])} 个、新增 {len(fc['framework_files_added'])} 个、删除 {len(fc['framework_files_removed'])} 个"
                 + ("：`" + "`、`".join((fc['framework_files_changed'] + fc['framework_files_added'])[:25]) + "`" if fc['framework_files_changed'] or fc['framework_files_added'] else "") + "。", ""]
    (stage / "PACKAGE_INFO.md").write_text("\n".join(note))
    if not ok:
        (stage / "UNADMITTED_DO_NOT_DEPLOY.txt").write_text("Offline admission (Sim2Sim / ONNX parity / eval gate) did NOT pass. Do not put this package on a robot.\n")
    # zip (explicit file list = whitelist + generated; denylist scan)
    zpath = run.deploy / f"{pname}.zip"
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    badre = re.compile(bad, re.I)
    allowed_extra = {"ADMISSION.json", "PACKAGE_INFO.md", "UNADMITTED_DO_NOT_DEPLOY.txt"}
    for p in files:
        rel = str(p.relative_to(stage))
        if rel.startswith("assets/") and rel.endswith(".onnx"):
            continue
        if badre.search(rel) and rel not in allowed_extra:
            die(f"denylist hit in package: {rel}")
    if zpath.exists():
        zpath.unlink()
    zc = dcfg.get("zip") or {}
    mode_of = lambda v, d: int(v, 8) if isinstance(v, str) else int(v if v is not None else d)
    scripts = write_zip(zpath, stage, files, mode_of(zc.get("script_mode"), 0o755), mode_of(zc.get("data_mode"), 0o644))
    zperm = verify_zip_permissions(zpath, scripts, tuple(zc.get("verify_extractors", ("unzip", "bsdtar", "python"))))
    log("zip permissions verified", **{k: v for k, v in zperm["extractors"].items()})
    (run.deploy / f"{pname}.zip.sha256").write_text(f"{sha256(zpath)}  {zpath.name}\n")
    out = dict(zip=str(zpath), sha256=sha256(zpath), size=zpath.stat().st_size, stage=str(stage), admitted=ok, selfcheck=rep.get("passed"), zip_permissions=zperm, run_timeline=info.get("run_timeline"))
    write_json(run.deploy / "latest.json", dict(out, info=info))
    log("package built", **out)
    return out
