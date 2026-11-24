#!/usr/bin/env python3
"""DEV-ONLY loopback smoke of the lean service (not shipped, not run on site): virtual remote -> supervisor+guardian+runner against the simulated robot
of deploy/g1_fasttrack/tools/sim_host.py.   tests/smoke_sim.py [--only A,B,C]
  A  F1+F3 -> READY -> L1+A -> clean run, WalkRun hand-back confirmed, logs complete
  B  START x2 while READY -> aborted before control
  C  START x2 during the run -> REMOTE_STOP, hand-back confirmed, service back to IDLE
  D  L1+A in IDLE ignored
  E  runner kill -9 during the run -> guardian take-over -> HOLD -> ack
"""
import argparse, csv, json, os, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path
PKG = Path(__file__).resolve().parents[1]; DEV = PKG.parent / "g1_fasttrack"
PY = os.environ.get("FT_PY") or "/home/unitree/workwpace/zjy_ws/sim2real_party_p0_patched/.venv/bin/python"
F1F3 = (0xC0, 0x00); L1A = (0x02, 0x01); NONE = (0, 0)


class Rig:
    def __init__(self, tmp):
        self.tmp, self.keys, self.logs, self.reset_n = tmp, tmp / "keys.json", tmp / "logs", 0; self.host = self.sup = None; self.set_keys(*NONE)
    def set_keys(self, b2, b3):
        t = self.keys.with_suffix(".t"); t.write_text(json.dumps({"byte2": b2, "byte3": b3, "reset": self.reset_n})); os.replace(t, self.keys)
    def press(self, c, dur=0.45): self.set_keys(*c); time.sleep(dur); self.set_keys(*NONE); time.sleep(0.3)
    def start_double(self):
        for _ in range(2): self.set_keys(0x04, 0); time.sleep(0.08); self.set_keys(0, 0); time.sleep(0.06)
    def reset_robot(self): self.reset_n += 1; self.set_keys(0, 0); time.sleep(1.0)
    def start_host(self):
        self.host = subprocess.Popen([PY, "-B", str(DEV / "tools/sim_host.py"), "--out", str(self.tmp / "host"), "--init", "stand", "--stand-pose", "oem_fixstand", "--support", "none",
                                      "--oem-hold", "surrogate", "--entry-fsm", "500", "--keys-file", str(self.keys), "--max-runtime-s", "900"], stdout=open(self.tmp / "host.log", "w"), stderr=subprocess.STDOUT, cwd=DEV)
        t0 = time.time()
        while time.time() - t0 < 60 and "host ready" not in (self.tmp / "host.log").read_text(): time.sleep(0.5)
    def start_service(self, *extra):
        shutil.rmtree(self.logs, ignore_errors=True)
        for f in ("/tmp/partymimic_g1_ESTOP", "/tmp/partymimic_g1_control.lock", "/tmp/partymimic_g1_guardian.alive"):
            try: os.remove(f)
            except FileNotFoundError: pass
        self.sup = subprocess.Popen([PY, "-B", "-m", "fasttrack.supervisor", "--sim", "--logs", str(self.logs), "--ready-timeout-s", "60", *extra], stdout=open(self.tmp / "service.out", "a"),
                                    stderr=subprocess.STDOUT, cwd=PKG, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < 40:
            if self.state().get("guardian_alive"): return
            time.sleep(0.3)
        raise RuntimeError("service/guardian did not come up")
    def stop_service(self):
        if self.sup and self.sup.poll() is None:
            self.sup.send_signal(signal.SIGTERM)
            try: self.sup.wait(timeout=40)
            except subprocess.TimeoutExpired: os.killpg(self.sup.pid, signal.SIGKILL)
        subprocess.run(["pkill", "-f", "fasttrack.(guardian|runner).*" + str(self.logs)], capture_output=True)
    def stop_all(self):
        self.stop_service()
        if self.host and self.host.poll() is None:
            self.host.send_signal(signal.SIGTERM)
            try: self.host.wait(timeout=15)
            except subprocess.TimeoutExpired: self.host.kill()
    def state(self):
        try: return json.loads((self.logs / "state.json").read_text())
        except Exception: return {}
    def wait_state(self, want, t):
        t0 = time.time()
        while time.time() - t0 < t:
            if self.state().get("state") == want: return True
            time.sleep(0.15)
        return False
    def runs(self): return sorted((self.logs / "runs").glob("*"))
    def wait_report(self, n_before, t):
        t0 = time.time()
        while time.time() - t0 < t:
            r = self.runs()
            if len(r) > n_before and (r[-1] / "run_report.json").exists(): return json.loads((r[-1] / "run_report.json").read_text())
            time.sleep(0.3)
        return {}
    def svc_log(self): return (self.logs / "service.log").read_text() if (self.logs / "service.log").exists() else ""


def logs_complete(r):
    d = r.runs()[-1]; need = ["trace.csv", "events.jsonl", "console.log", "ready.json", "admission.json", "run_report.json"]
    miss = [f for f in need if not (d / f).exists()]
    rows = list(csv.DictReader(open(d / "trace.csv"))) if (d / "trace.csv").exists() else []
    ev = {json.loads(l).get("kind") for l in (d / "events.jsonl").read_text().splitlines()}
    want = {"phase", "lock_acquired", "SwitchToUserCtrl", "user_control_confirmed", "SwitchToInternalCtrl", "released", "result"}
    ok = not miss and len(rows) >= 10 and not (want - ev) and rows[0]["infer_ms"] != "" and rows[0]["q:left_hip_pitch_joint"] != "" and rows[0]["action:left_hip_pitch_joint"] != ""
    return ok, f"missing files {miss}; trace rows {len(rows)}; missing events {sorted(want - ev)}"


def A(r):
    r.reset_robot(); n0 = len(r.runs()); r.press(F1F3); out = [("F1+F3 -> READY", r.wait_state("READY", 60))]
    r.press(L1A); rep = r.wait_report(n0, 90); r.wait_state("IDLE", 30); ex = rep.get("exit", {})
    out += [(f"clean run: result={rep.get('result')} reason={rep.get('exit_reason')} exit_confirmed={ex.get('confirmed')} mode={ex.get('internal_mode')} entry={rep.get('entry_fsm')} restored={rep.get('restored_fsm')}",
             rep.get("result") == "COMPLETED" and ex.get("confirmed") is True and ex.get("internal_mode") == "WALKRUN"),
            ("service back to IDLE rc=0", r.state().get("state") == "IDLE" and r.state().get("last_result", {}).get("rc") == 0)]
    ok, d = logs_complete(r); out.append((f"logs complete: {d}", ok))
    out.append(("keys.log has the chords", "keys=F1+F3" in (r.logs / "keys.log").read_text() and "keys=A+L1" in (r.logs / "keys.log").read_text()))
    return out


def B(r):
    r.reset_robot(); n0 = len(r.runs()); r.press(F1F3); out = [("F1+F3 -> READY", r.wait_state("READY", 60))]
    r.start_double(); ok = r.wait_state("IDLE", 30); rep = r.wait_report(n0, 20)
    ev = [e["kind"] for e in rep.get("session_events", [])]
    return out + [("START x2 in READY -> IDLE", ok), (f"aborted before ARM, never switched: {rep.get('result')}", rep.get("result") == "ABORTED_BEFORE_ARM" and "SwitchToUserCtrl" not in ev)]


def C(r):
    r.reset_robot(); n0 = len(r.runs()); r.press(F1F3); out = [("F1+F3 -> READY", r.wait_state("READY", 60))]
    r.press(L1A); out.append(("RUNNING", r.wait_state("RUNNING", 20))); time.sleep(4.2); r.start_double()
    rep = r.wait_report(n0, 40); r.wait_state("IDLE", 30); ex = rep.get("exit", {})
    return out + [(f"START x2 in run: fault={rep.get('fault')} reason={rep.get('exit_reason')} mode={ex.get('internal_mode')} confirmed={ex.get('confirmed')}",
                   rep.get("fault") == "REMOTE_STOP" and ex.get("confirmed") is True and ex.get("internal_mode") == "WALKRUN"),
                  ("service back to IDLE", r.state().get("state") == "IDLE")]


def D(r):
    r.reset_robot(); n0 = len(r.runs()); r.press(L1A); time.sleep(0.8)
    return [("L1+A in IDLE ignored", len(r.runs()) == n0 and "ignored in state IDLE" in r.svc_log())]


def E(r):
    r.reset_robot(); r.press(F1F3); out = [("F1+F3 -> READY", r.wait_state("READY", 60))]
    r.press(L1A); out.append(("RUNNING", r.wait_state("RUNNING", 20))); time.sleep(3.5)
    ps = subprocess.run(["pgrep", "-f", "fasttrack.runner.*" + str(r.logs)], capture_output=True, text=True).stdout.split()
    if ps: os.kill(int(ps[0]), signal.SIGKILL)
    out += [("runner killed (kill -9)", bool(ps)), ("service -> HOLD", r.wait_state("HOLD", 30))]
    time.sleep(1.5); gl = "".join(p.read_text() for p in (r.logs / "guardian").glob("*.log"))
    out.append(("guardian TAKE-OVER confirmed (entry fsm from the lock file)", "TAKE-OVER" in gl and '"confirmed": true' in gl))
    r.press(F1F3); out.append(("F1+F3 ignored while HOLD", r.state().get("state") == "HOLD"))
    (r.logs / "ACK").write_text("x"); out.append(("ack -> IDLE", r.wait_state("IDLE", 10)))
    return out


def F(r):
    """service started with --entry-fsm 505 while the robot is at 500: admission must BLOCK, nothing may be sent, service back to IDLE."""
    r.reset_robot(); n0 = len(r.runs()); r.press(F1F3); ok = r.wait_state("IDLE", 60) and r.svc_log().count("runner started") == 1; time.sleep(1.0); r.press(L1A); time.sleep(0.8)
    rep = json.loads((r.runs()[-1] / "run_report.json").read_text()); ev = [e["kind"] for e in rep.get("session_events", [])]
    log = r.svc_log()
    return [("BLOCKED by admission, back to IDLE", ok and rep.get("result") == "BLOCKED" and r.state().get("last_result", {}).get("rc") == 6),
            ("no lock / no switch / no frame sent", not ev and rep.get("frames_sent") == 0), ("failing item printed in service.log", "BLOCKED item: pf_fsm_entry_allowed" in log),
            ("L1+A after BLOCKED does nothing", len(r.runs()) == n0 + 1 and "ignored in state IDLE" in log)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); a = ap.parse_args(); only = [x for x in a.only.split(",") if x]
    tmp = Path(tempfile.mkdtemp(prefix="lean_smoke_")); r = Rig(tmp); res = []
    def grp(name, fn, *args, unarmed=False):
        if only and name not in only: return
        r.start_service(*args)
        try:
            for d, ok in fn(r): res.append((name, d, bool(ok))); print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {d}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); res.append((name, repr(e), False)); print(f"  [FAIL] {name}: {e!r}", flush=True)
        finally:
            r.stop_service(); shutil.copytree(r.logs, tmp / f"logs_{name}", dirs_exist_ok=True)
    print("artifacts:", tmp, flush=True); r.start_host()
    try:
        grp("A", A, "--run-s", "1.0", "--ref-mode", "running", "--authority-max", "0.3")
        grp("B", B, "--run-s", "1.0"); grp("D", D, "--run-s", "1.0")
        grp("C", C, "--run-s", "6.0", "--ref-mode", "paused", "--authority-max", "1.0")
        grp("F", F, "--run-s", "1.0", "--entry-fsm", "505")
        grp("E", E, "--run-s", "6.0", "--ref-mode", "paused", "--authority-max", "1.0")
    finally:
        r.stop_all()
    bad = [x for x in res if not x[2]]; print(f"\nLEAN SMOKE: {len(res) - len(bad)}/{len(res)} PASS (artifacts {tmp})"); sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
