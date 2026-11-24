"""Remote-controller service: the pure state machine (fake clock)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fasttrack import paths
paths.setup_sys_path(sdk=False); paths.setup_offline_env()
import pytest
from fasttrack.supervisor import Machine, MachineCfg, IDLE, LOADING, READY, RUNNING, HOLD


def mk(armed=True, **kw): return Machine(MachineCfg(**kw), armed=armed)
def kinds(acts): return [a[0] for a in acts]

def test_full_clean_cycle_and_repeat():
    m = mk(); assert m.state == IDLE
    assert kinds(m.chord_load(1.0)) == ["SPAWN"] and m.state == LOADING
    assert m.runner_status("LOADING_MODEL", 2.0) == [] and m.state == LOADING
    m.runner_status("READY", 5.0); assert m.state == READY
    assert kinds(m.chord_go(6.0)) == ["GO"] and m.state == RUNNING
    m.runner_exit(0, 9.0); assert m.state == IDLE and m.last_result["rc"] == 0
    assert kinds(m.chord_load(10.0)) == ["SPAWN"]                                         # repeat

def test_go_only_from_ready():
    m = mk()
    for st in (IDLE, LOADING, RUNNING, HOLD):
        m.state = st; assert kinds(m.chord_go(1.0)) == ["NOTE"] and m.state == st
    m.state = READY; assert kinds(m.chord_go(1.0)) == ["GO"]

def test_unarmed_loads_and_checks_but_never_takes_control():
    m = mk(armed=False); assert kinds(m.chord_load(1.0)) == ["SPAWN"] and m.state == LOADING
    m.runner_status("READY", 2.0); assert m.state == READY
    assert kinds(m.chord_go(3.0)) == ["NOTE"] and m.state == READY            # L1+A does nothing

def test_blocked_or_refused_go_returns_to_idle_without_hold():
    m = mk(); m.chord_load(0.0); m.runner_exit(6, 2.0); assert m.state == IDLE
    m = mk(); m.state = RUNNING; m.runner_exit(6, 2.0); assert m.state == IDLE   # GO refused by the re-check: nothing was sent

def test_guardian_dead_blocks_load_and_go():
    m = mk(); assert kinds(m.chord_load(1.0, guardian_ok=False)) == ["NOTE"] and m.state == IDLE
    m.state = READY; a = m.chord_go(1.0, guardian_ok=False); assert "GO" not in kinds(a) and "TERMINATE" in kinds(a)

def test_start_x2_in_every_stage():
    for st, expect in ((IDLE, []), (LOADING, ["TERMINATE"]), (READY, ["TERMINATE"]), (RUNNING, []), (HOLD, [])):
        m = mk(); m.state = st; a = m.start_double(1.0); assert [k for k in kinds(a) if k != "NOTE"] == expect, st
    m = mk(); m.state = RUNNING; m.start_double(1.0); assert m.user_abort and m.abort_deadline == 1.0 + m.cfg.abort_escalate_s

def test_abort_before_arm_returns_to_idle_without_hold():
    m = mk(); m.chord_load(0.0); m.start_double(1.0); m.runner_exit(0, 2.0); assert m.state == IDLE
    m = mk(); m.chord_load(0.0); m.runner_exit(6, 2.0); assert m.state == IDLE                # BLOCKED by admission

def test_run_outcomes_hold_vs_idle():
    def run(rc, abort=False, escalated=False):
        m = mk(); m.state = RUNNING; m.user_abort = abort; m.escalated = escalated; m.runner_exit(rc, 5.0); return m.state
    assert run(0) == IDLE and run(10, abort=True) == IDLE
    assert run(10, abort=False) == HOLD            # a fault the operator did not ask for
    assert run(5) == HOLD and run(5, abort=True) == HOLD and run(1) == HOLD and run(-9) == HOLD and run(4) == HOLD
    assert run(0, abort=True, escalated=True) == HOLD

def test_hold_ignores_chords_until_ack():
    m = mk(); m.state = HOLD; assert kinds(m.chord_load(1.0)) == ["NOTE"] and m.state == HOLD
    assert m.ack(2.0) and m.state == IDLE

def test_timeouts_and_escalation():
    m = mk(ready_timeout_s=10, load_timeout_s=5); m.chord_load(0.0)
    assert m.tick(4.0) == [] and "TERMINATE" in kinds(m.tick(6.0))
    m = mk(ready_timeout_s=10); m.state = READY; m.deadline = 10.0; assert "TERMINATE" in kinds(m.tick(11.0))
    m = mk(); m.state = RUNNING; m.start_double(0.0); assert m.tick(1.0) == [] and "ESCALATE" in kinds(m.tick(3.0))
    m = mk(); m.state = RUNNING; assert "TERMINATE" in kinds(m.tick(1.0, guardian_ok=False))

def test_runner_ready_only_counts_while_loading():
    m = mk(); m.state = IDLE; m.runner_status("READY", 1.0); assert m.state == IDLE
