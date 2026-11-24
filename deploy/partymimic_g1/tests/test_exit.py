"""Exit-mode decision and hand-back confirmation semantics (pure logic, fake link)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fasttrack import paths
paths.setup_sys_path(sdk=False); paths.setup_offline_env()
import pytest
from fasttrack.profile import decide_exit_mode, get_profile
from fasttrack.handback import handback


OK = [1, 0, 0, 0]
def test_exit_mode_normal_and_recoverable_go_walkrun():
    for f in ("NONE", "ESTOP", "REMOTE_STOP", "WATCHDOG", "COMM_TIMEOUT", "CMD_INVALID", "NO_RESPONSE", "JOINT_LIMIT"):
        assert decide_exit_mode("AUTO", OK, 0.01, f)[0] == "WALKRUN", f

def test_exit_mode_severe_or_unknown_posture_is_passive():
    assert decide_exit_mode("AUTO", OK, 0.01, "TILT")[0] == "PASSIVE"
    assert decide_exit_mode("AUTO", OK, 0.01, "STATE_INVALID")[0] == "PASSIVE"
    assert decide_exit_mode("AUTO", OK, 0.01, "STATE_FROZEN")[0] == "PASSIVE"
    assert decide_exit_mode("AUTO", None, None, "NONE")[0] == "PASSIVE"
    assert decide_exit_mode("AUTO", OK, 5.0, "COMM_TIMEOUT")[0] == "PASSIVE"          # stale attitude = unknown
    import math; s, c = math.sin(0.6), math.cos(0.6)
    assert decide_exit_mode("AUTO", [c, s, 0, 0], 0.01, "NONE")[0] == "PASSIVE"         # roll 1.2 rad
    assert decide_exit_mode("LAST", OK, 0.01, "TILT")[0] == "LAST"                      # explicit override wins

class FakeLink:
    """fsm script: fn(t_since_first_switch, calls) -> fsm; rpc script: list of codes."""
    def __init__(self, codes, fsm_fn, get_fail=False):
        self.codes, self.fsm_fn, self.calls, self.t0, self.get_fail = list(codes), fsm_fn, 0, None, get_fail
    def switch_to_internal(self, mode):
        self.calls += 1; self.t0 = self.t0 or time.monotonic(); return self.codes.pop(0)
    def get_fsm(self):
        if self.get_fail: return 3104, None
        return 0, self.fsm_fn(self, 0.0 if self.t0 is None else time.monotonic() - self.t0)

def hb(link, **kw):
    return handback(link, mode="WALKRUN", user_fsm=1000, expected_fsm=500, confirm_timeout_s=0.6, dwell_s=0.05, hold_unconfirmed_s=0.3, poll_s=0.02, log=lambda *_: None, **kw)

def test_handback_ok():
    r = hb(FakeLink([0], lambda l, t: 500 if l.calls else 1000))
    assert r["confirmed"] and r["confirmed_via"] == "rpc_ok+fsm" and r["mode_matched"]

def test_handback_rpc_ok_but_fsm_never_changes_is_failure():
    r = hb(FakeLink([0], lambda l, t: 1000)); assert not r["confirmed"]

def test_handback_fsm_unreadable_is_failure():
    r = hb(FakeLink([0], lambda l, t: 500, get_fail=True)); assert not r["confirmed"]

def test_handback_rpc_timeout_but_expected_fsm_is_success_via_fsm():
    r = hb(FakeLink([3104], lambda l, t: 500 if l.calls else 1000))
    assert r["confirmed"] and r["confirmed_via"] == "fsm_only(rpc_failed)" and r["rpc_calls"][0]["code"] == 3104

def test_handback_rpc_failed_and_unexpected_fsm_is_not_success():
    r = hb(FakeLink([3104], lambda l, t: 7 if l.calls else 1000)); assert not r["confirmed"]

def test_handback_timeout_not_executed_gets_one_verified_retry():
    link = FakeLink([3104, 0], lambda l, t: 500 if l.calls >= 2 else 1000)
    r = hb(link); assert r["confirmed"] and r["verified_retry_after_timeout"] and link.calls == 2

def test_handback_timeout_never_executes_gives_up_after_one_retry():
    link = FakeLink([3104, 3104], lambda l, t: 1000); r = hb(link)
    assert not r["confirmed"] and link.calls == 2

def test_handback_explicit_reject_retried_once():
    link = FakeLink([3204, 0], lambda l, t: 500 if l.calls >= 2 else 1000); r = hb(link)
    assert r["confirmed"] and r["retried_once_after_explicit_rejection"] and link.calls == 2

def test_handback_bounce_back_into_user_fsm_not_confirmed():
    r = hb(FakeLink([0], lambda l, t: (500 if t < 0.03 else 1000) if l.calls else 1000)); assert not r["confirmed"]

def test_handback_other_oem_state_reported_not_hidden():
    r = hb(FakeLink([0], lambda l, t: 1 if l.calls else 1000))
    assert r["confirmed"] and r["mode_matched"] is False        # control returned, but PASSIVE instead of the requested WALKRUN

def test_profile_expected_exit_fsm():
    p = get_profile("demo_walkrun"); assert p.expected_exit_fsm("WALKRUN") == 500 and p.expected_exit_fsm("PASSIVE") == 1 and p.expected_exit_fsm("LAST") is None


def test_handback_retry_rechecks_mode_when_attitude_got_worse():
    link = FakeLink([3104, 0], lambda l, t: 1 if l.calls >= 2 else 1000)
    modes = []
    orig = link.switch_to_internal
    def sw(mode): modes.append(mode); return orig(mode)
    link.switch_to_internal = sw
    r = hb(link, remode=lambda: ("PASSIVE", 1))
    assert modes == ["WALKRUN", "PASSIVE"] and r["confirmed"] and r["mode_matched"] and r["mode_changed_before_retry"] == [{"from": "WALKRUN", "to": "PASSIVE"}]
