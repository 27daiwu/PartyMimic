#!/usr/bin/env python3
"""Sim2Sim safety-gate matrix: each case = host (fault injected) + the SAME runner/session/guard code used on the robot.
Timeline (FAST profile): user control starts at rel 0; PREPARE 1.5 s, SETTLE+SHADOW+RAMP ~1.5 s, DANCE from rel ~3 s to ~10 s."""
from __future__ import annotations
import json, os, shlex, signal, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_suite import run_case, PKG, PY

OUT = PKG / "runs/faults"
FAST = "--entry-fsm 500 --prepare-s 1.5 --settle-s 0.3 --shadow-s 0.3 --ramp-in-s 0.5 --settle-tol-rad 0.3 --auto-go --mode live --unconfirmed-hold-s 3"
HANG = "--init default --support hang --oem-hold pd"
F = 5        # fault time, seconds after user control starts (mid-dance, before the natural ankle-limit event at ~9 s)
EST = "/tmp/partymimic_g1_ESTOP"
BG: list = []

def sig(s): return lambda host, runner, out: runner.send_signal(s)
def touch(f): return lambda host, runner, out: Path(f).touch()
def kill_host(host, runner, out): host.kill()
def kill9(host, runner, out): runner.kill()
def second_runner(host, runner, out):
    BG.append(subprocess.Popen([PY, "-B", "-m", "fasttrack.runner", "--target", "sim", "--interface", "lo", "--mode", "shadow", "--run-dir", str(out / "runner2")],
                               stdout=open(out / "second.out", "w"), stderr=subprocess.STDOUT, cwd=PKG))
def foreign_writer(host, runner, out):
    BG.append(subprocess.Popen([PY, "-B", str(PKG / "tools/foreign_writer.py")], stdout=open(out / "foreign.out", "w"), stderr=subprocess.STDOUT, cwd=PKG))
def guardian(host, runner, out):
    BG.append(subprocess.Popen([PY, "-B", "-m", "fasttrack.guardian", "guardian", "--interface", "lo", "--domain", "77", "--report", str(out / "guardian.json")],
                               stdout=open(out / "guardian.out", "w"), stderr=subprocess.STDOUT, cwd=PKG))
def estop_direct(host, runner, out):
    BG.append(subprocess.Popen([PY, "-B", "-m", "fasttrack.guardian", "--interface", "lo", "--domain", "77", "--direct"],
                               stdout=open(out / "estop_direct.out", "w"), stderr=subprocess.STDOUT, cwd=PKG))

def host_final(out):
    try: return json.loads((out / "host/host_info.json").read_text())
    except Exception: return {}

def std_check(exp, expc):
    def f(res, out):
        flt = res.get("fault"); ex = res.get("exit") or {}
        ok = (exp == "INFO") or (exp == "ANY" and flt not in (None, "NONE")) or flt == exp
        return bool(ok and ex.get("confirmed") is expc), f"fault={flt} exit_confirmed={ex.get('confirmed')}"
    return f

def guardian_check(res, out):
    """Runner died or hung: the guardian alone must hand the robot back (host final fsm == entry fsm, exit RPC issued)."""
    try: g = json.loads((out / "guardian.json").read_text())
    except Exception: g = {}
    h = host_final(out)
    ok = g.get("action") == "take_over" and g.get("confirmed") is True and h.get("final_fsm") not in (1000, None) and h.get("exit_calls", 0) >= 1
    return ok, f"guardian={ {k: g.get(k) for k in ('action','confirmed','killed_pid','reason')} } host_final_fsm={h.get('final_fsm')}"

def retry_check(res, out):
    """Whatever ended the run (normal end or a physical limit trip), the exit RPC was rejected once and retried exactly once, then confirmed."""
    ex = res.get('exit') or {}; h = host_final(out)
    ok = ex.get('retried_once_after_explicit_rejection') is True and ex.get('confirmed') is True and h.get('exit_calls') == 2
    return ok, f"retried={ex.get('retried_once_after_explicit_rejection')} host_exit_calls={h.get('exit_calls')} confirmed={ex.get('confirmed')} (run ended with fault={res.get('fault')})"

def guardian_idle_check(res, out):
    try: g = json.loads((out / "guardian.json").read_text())
    except Exception: g = {}
    ok = g.get("action") == "none" and (res.get("exit") or {}).get("confirmed") is True      # guardian never interferes with a runner that exits by itself
    return ok, f"guardian={g.get('action')} runner={res.get('result')}/{res.get('fault')} runner_exit_confirmed={(res.get('exit') or {}).get('confirmed')}"

def second_check(res, out):
    try: r2 = json.loads((out / "runner2/run_report.json").read_text())
    except Exception: r2 = {}
    ok = res.get("result") == "COMPLETED" and r2.get("result") == "ARM_FAILED"
    return ok, f"first={res.get('result')} second={r2.get('result')} second_fault={r2.get('fault')}"

def estop_direct_check(res, out):
    h = host_final(out)
    ok = h.get("final_fsm") not in (1000, None) and h.get("exit_calls", 0) >= 1
    return ok, f"runner={res.get('result')}/{res.get('fault')} host_final_fsm={h.get('final_fsm')} exit_calls={h.get('exit_calls')}"

CASES = {
 "baseline_hang":        (HANG, FAST, [], std_check("NONE", True), "no fault: completes, exits to OEM"),
 "comm_stall":           (HANG + f" --fault state_stall@{F}", FAST, [], std_check("COMM_TIMEOUT", True), "lowstate stops during dance"),
 "state_freeze":         (HANG + f" --fault state_freeze@{F}", FAST, [], std_check("STATE_FROZEN", True), "stream alive but frozen"),
 "state_nan":            (HANG + f" --fault state_nan@{F}", FAST, [], std_check("STATE_INVALID", True), "NaN joint position"),
 "bad_quat":             (HANG + f" --fault state_quat_bad@{F}", FAST, [], std_check("STATE_INVALID", True), "IMU quaternion norm 3"),
 "tau_hot":              (HANG + f" --fault tau_hot@{F}", FAST, [], std_check("TORQUE_SUSTAINED", True), "measured torque 1.5x effort sustained"),
 "cmd_drop":             (HANG + f" --fault cmd_drop@{F}", FAST, [], std_check("ANY", True), "host ignores commands (command path ineffective)"),
 "cmd_drop_at_entry":    (HANG + " --fault cmd_drop@0", FAST, [], std_check("NO_RESPONSE", True), "commands never take effect: caught in SETTLE"),
 "cmd_delay_20ms":       (HANG + f" --fault cmd_delay_20ms@{F}", FAST, [], std_check("INFO", True), "20 ms command latency (informational)"),
 "cmd_delay_40ms":       (HANG + f" --fault cmd_delay_40ms@{F}", FAST, [], std_check("INFO", True), "40 ms command latency (informational)"),
 "rpc_switch_fail":      (HANG + " --fault rpc_switch_fail@0", FAST, [], std_check("SWITCH_FAILED", True), "SwitchToUserCtrl rejected"),
 "rpc_exit_reject_once": (HANG + " --fault rpc_exit_fail_once@0", FAST, [], retry_check, "exit RPC explicitly rejected once -> exactly one retry"),
 "rpc_exit_never":       (HANG + " --fault rpc_exit_never@0", FAST, [], std_check("NONE", False), "exit RPC times out forever -> UNCONFIRMED reported, exit code 5"),
 "foreign_writer":       (HANG, FAST, [(F + 4.5, foreign_writer)], std_check("LOCK_LOST", True), "another process writes rt/user_lowcmd"),
 "loop_stall":           (HANG, FAST + " --debug-stall-at-s 7 --debug-stall-s 0.5", [], std_check("WATCHDOG", True), "control loop blocked 0.5 s"),
 "nan_policy":           (HANG, FAST + " --debug-nan-action-at-s 7", [], std_check("CMD_INVALID", True), "policy emits NaN"),
 "sigint":               (HANG, FAST, [(10, sig(signal.SIGINT))], std_check("ESTOP", True), "operator Ctrl-C"),
 "sigterm":              (HANG, FAST, [(10, sig(signal.SIGTERM))], std_check("ESTOP", True), "kill -TERM"),
 "estop_file":           (HANG, FAST, [(10, touch(EST))], std_check("ESTOP", True), "touch /tmp/partymimic_g1_ESTOP from another shell"),
 "host_dies":            (HANG, FAST, [(10, kill_host)], std_check("COMM_TIMEOUT", False), "robot side vanishes mid-dance: exit cannot be confirmed"),
 "second_instance":      (HANG, FAST, [(8, second_runner)], second_check, "2nd runner started during the run: refused, 1st unaffected"),
 "entry_fsm_passive_refused":(HANG + " --entry-fsm 1", FAST, [], std_check("SWITCH_FAILED", True), "demo profile: OEM in PASSIVE(1) -> refused before any RPC"),
 "entry_fsm_undeclared_mismatch":(HANG, FAST.replace("--entry-fsm 500", "--entry-fsm 501"), [], std_check("SWITCH_FAILED", True), "operator declares fsm 501 but robot reports 500 -> refused"),
 "mode_machine_mismatch":(HANG + " --fault mode_machine_mismatch@0", FAST, [], std_check("SWITCH_FAILED", True), "lowstate.mode_machine=7 (not a 29-dof G1) -> refused"),
 "fsm_takeback":         (HANG + f" --fault fsm_takeback@{F}", FAST, [], std_check("FSM_CHANGED", True), "OEM silently leaves user fsm while commands flow: detected by the 1 Hz fsm re-check, no exit RPC re-sent"),
 "legacy_profile_ignored_by_strict_host": ("--init default --support hang --oem-hold pd --entry-fsm 1 --host-profile sdk2", FAST.replace("--entry-fsm 500", "--entry-fsm 1 --profile sdk2_passive"), [], std_check("NO_RESPONSE", True), "R0 frame format (no mode/mode_machine/CRC) against a host that enforces the demo fields: commands ignored -> caught as NO_RESPONSE"),
 "sdk2_profile_passive_ok": ("--init default --support hang --oem-hold pd --entry-fsm 1 --host-profile sdk2 --no-strict-frames --user-fsm-id 6001", FAST.replace("--entry-fsm 500", "--entry-fsm 1 --profile sdk2_passive"), [], std_check("INFO", True), "legacy sdk2 profile still works against a lenient sdk2-style host (weak confirmation, LAST exit)"),
 "push_200N":            (HANG + f" --fault push@{F}", FAST, [], std_check("INFO", True), "200 N torso push for 0.2 s (informational)"),
 "guardian_idle":        (HANG, FAST, [(0, guardian)], guardian_idle_check, "guardian watches a healthy run and does nothing"),
 "runner_sigkill":       (HANG, FAST, [(0, guardian), (10, kill9)], guardian_check, "runner kill -9 mid-dance: only the guardian can save it"),
 "runner_sigstop":       (HANG, FAST, [(0, guardian), (10, sig(signal.SIGSTOP))], guardian_check, "runner frozen (SIGSTOP, like a hang): guardian kills it and takes over"),
 "estop_direct":         (HANG, FAST, [(10, estop_direct)], estop_direct_check, "operator runs the estop tool with --direct"),
}

def main():
    only = [a for a in sys.argv[1:]]
    OUT.mkdir(parents=True, exist_ok=True); res = []
    for n, (h, r, acts, chk, note) in CASES.items():
        if only and n not in only: continue
        d = OUT / n
        if d.exists(): subprocess.run(["rm", "-rf", str(d)])
        d.mkdir(parents=True)
        if os.path.exists(EST): os.remove(EST)
        t0 = time.time()
        wrapped = [(t, (lambda fn: (lambda host, runner: fn(host, runner, d)))(fn)) for t, fn in acts]
        try:
            out = run_case(n, shlex.split(h), shlex.split(r), OUT, timeout=120, actions=wrapped)
        except Exception as e:
            out = {"name": n, "error": repr(e)}
        for p in BG:
            try:
                p.wait(timeout=12)
            except Exception:
                p.kill()
        BG.clear()
        if os.path.exists(EST): os.remove(EST)
        try: ok, detail = chk(out, d)
        except Exception as e: ok, detail = False, f"check error {e!r}"
        out.update(note=note, check=detail, pass_=bool(ok), secs=round(time.time() - t0, 1))
        res.append(out); print(f"{n:24s} {'PASS' if ok else 'FAIL'}  {detail}  rc={out.get('runner_rc')}  ({note})", flush=True)
        (OUT / "matrix_partial.json").write_text(json.dumps(res, indent=1, default=str))
    (OUT / ("matrix.json" if not only else "matrix_subset.json")).write_text(json.dumps(res, indent=1, default=str))

if __name__ == "__main__":
    main()
