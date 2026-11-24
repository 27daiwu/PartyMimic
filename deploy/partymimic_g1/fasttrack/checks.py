"""Real-time admission for a take-over: ONE list of checks, all evaluated by code on the live robot, none of them paperwork.

Merged (previously spread over admission.py, tools/diag.py, runner gates and the preflight file):
  artifacts      frozen P2 ONNX / SSB / yaml hashes                       (the policy that was evaluated is the one that runs)
  caps           real runs: run window <= 1.0 s, authority <= 0.3          (not relaxed)
  guardian       independent guardian alive on this interface (<1 s old)   (backup stop path)
  session.preflight (control authority + state), see ControlSession.preflight:
                 lowstate rate / tick, state inside the guard envelope, q inside the HARD joint limits (zero tolerance), robot quiescent, tilt,
                 fsm readable + WalkRun-like entry (never 0/1), mode_machine, NO foreign writer on rt/user_lowcmd, exclusive lock free, no e-stop latch
  oem_writer     rt/lowcmd writers == the count the OEM confirmed (config/site.json). UNCONFIRMED = BLOCKED. This is the R1 diagnostic blocker
                 `no_writer_on_rt/lowcmd` kept as a measured check: in WalkRun the OEM service writes rt/lowcmd, which software cannot judge.
Documentary / research evidence (sim-evidence, OEM questionnaire, site sign-off, unit-test counts) is NOT checked here; it lives in the dev project and README.
"""
from __future__ import annotations

import json, time
from pathlib import Path

from . import paths

REAL_MAX_RUN_S = 1.0
REAL_MAX_AUTHORITY = 0.3
GUARDIAN_MAX_AGE_S = 1.0


def item(name, ok, detail="", kind="realtime"):
    return {"item": name, "pass": bool(ok), "detail": detail, "kind": kind}


def load_site(path=None) -> dict:
    p = Path(path) if path else paths.CONFIG / "site.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def entry_allowed(site: dict, override=None):
    if override is not None:
        return (int(override),)
    v = site.get("entry_fsm_allowed")
    return tuple(int(x) for x in v) if v else None


def artifact_items() -> list:
    out = []
    for key, path in (("onnx", paths.POLICY_ONNX), ("ssb_motion", paths.MOTION_NPZ), ("yaml", paths.POLICY_YAML)):
        try:
            ok = paths.sha256(path) == paths.EXPECTED_SHA256[key]
            out.append(item(f"artifact_{key}_frozen", ok, "sha256 == frozen P2 value" if ok else f"MISMATCH {Path(path).name}"))
        except Exception as e:
            out.append(item(f"artifact_{key}_frozen", False, repr(e)))
    return out


def caps_item(target: str, run_s: float, authority: float) -> dict:
    if target != "real":
        return item("run_caps", True, f"sim target: run {run_s} s authority {authority}")
    bad = []
    if not (0 < run_s <= REAL_MAX_RUN_S): bad.append(f"run_s {run_s} not in (0, {REAL_MAX_RUN_S}]")
    if not (0 < authority <= REAL_MAX_AUTHORITY): bad.append(f"authority {authority} not in (0, {REAL_MAX_AUTHORITY}]")
    return item("run_caps", not bad, "; ".join(bad) if bad else f"run {run_s} s <= {REAL_MAX_RUN_S}, authority {authority} <= {REAL_MAX_AUTHORITY}")


def guardian_item(iface: str, alive_file=None) -> dict:
    try:
        from .guardian import read_alive
        g = read_alive(alive_file) if alive_file else read_alive()
        age = time.time() - g["t"]
        if age > GUARDIAN_MAX_AGE_S:
            return item("guardian_alive", False, f"heartbeat {age:.1f} s old (> {GUARDIAN_MAX_AGE_S} s)")
        if g.get("interface") != iface:
            return item("guardian_alive", False, f"guardian watches {g.get('interface')}, not {iface}")
        return item("guardian_alive", True, f"heartbeat {age * 1e3:.0f} ms old, pid {g.get('pid')}")
    except Exception as e:
        return item("guardian_alive", False, f"no guardian heartbeat ({e!r})")


def discover_writers(topics=("rt/lowcmd", "rt/arm_sdk", "rt/user_lowcmd"), wait_s: float = 0.6) -> dict:
    """Read-only DDS built-in-topic discovery: distinct writers per command topic. Needs an initialised ChannelFactory (the link made it)."""
    from unitree_sdk2py.core.channel import ChannelFactory
    from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication
    rdr = BuiltinDataReader(ChannelFactory()._ChannelFactory__participant, BuiltinTopicDcpsPublication)
    time.sleep(wait_s)
    ws = {t: set() for t in topics}
    for smp in rdr.take(N=500):
        if smp.topic_name in ws:
            ws[smp.topic_name].add((str(smp.participant_key), str(smp.key)))
    return {t: sorted(v) for t, v in ws.items()}


def oem_writer_item(target: str, site: dict, writers) -> dict:
    """writers: {topic: [(participant, key), ...]} or an Exception/None when discovery was unavailable."""
    if target != "real":
        return item("oem_rt_lowcmd_writer", True, "sim target: no OEM service")
    cfg = site.get("oem_lowcmd_writer") or {}
    if not isinstance(writers, dict):
        return item("oem_rt_lowcmd_writer", False, f"writer discovery unavailable ({writers!r}): cannot verify who writes rt/lowcmd")
    n = len(writers.get("rt/lowcmd", []))
    seen = f"{n} writer(s) on rt/lowcmd in {len({w[0] for w in writers.get('rt/lowcmd', [])})} participant(s)"
    extra = {t: len(v) for t, v in writers.items() if t != "rt/lowcmd" and v}
    if extra:
        return item("oem_rt_lowcmd_writer", False, f"{seen}; UNEXPECTED writers on {extra} (another controller / debug mode is active)")
    exp = cfg.get("expected_writers")
    if cfg.get("confirmed") is not True or not isinstance(exp, int):
        return item("oem_rt_lowcmd_writer", False, f"UNCONFIRMED: {seen}. The OEM must confirm in writing that its motion service is the only writer; then set "
                                                  "config/site.json oem_lowcmd_writer.confirmed=true and expected_writers=<count>")
    return item("oem_rt_lowcmd_writer", n == exp, f"{seen}; OEM-confirmed count {exp}" + ("" if n == exp else " -> MISMATCH (another writer appeared or the service changed)"))


def merge_preflight(pf_results) -> list:
    return [item("pf_" + r["check"], r["pass"], r["detail"]) for r in pf_results]


def admission(*, target: str, iface: str, run_s: float, authority: float, site: dict, pf_results, writers) -> tuple[bool, list]:
    """The complete real-time admission verdict (ready, items). Pure given its inputs (unit-tested)."""
    items = artifact_items() + [caps_item(target, run_s, authority), guardian_item(iface)] + merge_preflight(pf_results) + [oem_writer_item(target, site, writers)]
    return all(i["pass"] for i in items), items
