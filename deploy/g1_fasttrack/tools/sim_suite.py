#!/usr/bin/env python3
"""Run sim2sim cases: one tools/sim_host.py + one fasttrack.runner per case, same code path as the real robot (SDK2 DDS on lo)."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, signal
from pathlib import Path
PKG = Path(__file__).resolve().parents[1]
PY = os.environ.get("FT_PY", "/home/unitree/workwpace/zjy_ws/sim2real_party_p0_patched/.venv/bin/python")

def stale_processes():
    """Leftover sim participants on the loopback domain disturb real-time sims (found in R1: a hung diag.py skewed results)."""
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()
    me = os.getpid()
    return [l.strip() for l in out if any(k in l for k in ("tools/sim_host.py", "tools/diag.py", "fasttrack.runner", "fasttrack.guardian", "tools/foreign_writer.py"))
            and "grep" not in l and str(me) != l.split()[0] and "fault_suite" not in l and "sim_suite" not in l]


def run_case(name, host_args, runner_args, out_root, *, timeout=150, side=None, signal_at=None, actions=()):
    st = stale_processes()
    if st:
        raise RuntimeError("stale sim processes running (kill them first): " + "; ".join(x[:90] for x in st))
    out = Path(out_root) / name; out.mkdir(parents=True, exist_ok=True)
    for f in ("/tmp/partymimic_g1_ESTOP",):
        if os.path.exists(f): os.remove(f)
    hlog = open(out / "host.log", "w")
    host = subprocess.Popen([PY, "-B", str(PKG / "tools/sim_host.py"), "--out", str(out / "host"), "--exit-after-user-done", "--max-runtime-s", str(timeout), *host_args],
                            stdout=hlog, stderr=subprocess.STDOUT, cwd=PKG)
    t0 = time.time()
    while time.time() - t0 < 60:
        if "host ready" in (out / "host.log").read_text(): break
        time.sleep(0.5)
    else:
        host.kill(); return {"name": name, "error": "host not ready"}
    rlog = open(out / "runner.out", "w")
    env = dict(os.environ)
    runner = subprocess.Popen([PY, "-B", "-m", "fasttrack.runner", "--target", "sim", "--interface", "lo", "--run-dir", str(out / "runner"), *runner_args],
                              stdout=rlog, stderr=subprocess.STDOUT, cwd=PKG, env=env)
    sidep = None
    if side:   # a second process to test lock / foreign writer
        time.sleep(side.get("delay", 8)); sidep = subprocess.Popen(side["cmd"], stdout=open(out / "side.out", "w"), stderr=subprocess.STDOUT, cwd=PKG)
    if signal_at is not None:
        time.sleep(signal_at["delay"]); runner.send_signal(signal_at["sig"])
    t_act = time.time()
    pend = sorted(actions, key=lambda x: x[0])
    try:
        while runner.poll() is None and time.time() - t_act < timeout:
            while pend and time.time() - t_act >= pend[0][0]:
                _, fn = pend.pop(0); fn(host=host, runner=runner)
            time.sleep(0.05)
        rc = runner.wait(timeout=5)
    except subprocess.TimeoutExpired:
        runner.kill(); rc = -9
    time.sleep(1.0)
    try:
        host.wait(timeout=15)
    except subprocess.TimeoutExpired:
        host.send_signal(signal.SIGTERM)
        try: host.wait(timeout=10)
        except subprocess.TimeoutExpired: host.kill(); host.wait()
    if sidep and sidep.poll() is None: sidep.kill()
    rep = {}
    try: rep = json.loads((out / "runner/run_report.json").read_text())
    except Exception: pass
    return {"name": name, "runner_rc": rc, "result": rep.get("result"), "fault": rep.get("fault"), "exit": rep.get("exit"),
            "fault_events": rep.get("fault_events"), "out": str(out)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--out", default=str(PKG / "runs/sim"))
    ap.add_argument("--host", default=""); ap.add_argument("--runner", default="")
    a = ap.parse_args()
    import shlex
    print(json.dumps(run_case(a.name, shlex.split(a.host), shlex.split(a.runner), a.out), indent=1, default=str))
