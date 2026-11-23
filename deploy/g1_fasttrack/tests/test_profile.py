import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fasttrack import paths
paths.setup_sys_path(sdk=False); paths.setup_offline_env()
import math
from fasttrack.profile import get_profile, entry_check, resolve_exit_mode, roll_pitch, PROFILES

def q_rp(roll, pitch):
    cr, sr, cp, sp = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return [cr * cp, sr * cp, cr * sp, -sr * sp]      # yaw 0, order roll then pitch ~ small-angle exact enough

def test_demo_rejects_passive_and_zero_torque():
    p = get_profile("demo_walkrun")
    assert not entry_check(p, 1, None)[0] and not entry_check(p, 0, None)[0]

def test_demo_has_no_hardcoded_entry_number():
    p = get_profile("demo_walkrun"); assert p.entry_allowed is None
    assert entry_check(p, 500, None)[0] and entry_check(p, 777, None)[0]

def test_declared_must_match_observed():
    p = get_profile("demo_walkrun"); assert entry_check(p, 500, 500)[0]; assert not entry_check(p, 500, 501)[0]

def test_unknown_fsm_refused(): assert not entry_check(get_profile("demo_walkrun"), None, None)[0]
def test_sdk2_profile_is_passive_only():
    p = get_profile("sdk2_passive"); assert entry_check(p, 1, None)[0] and not entry_check(p, 500, None)[0]

def test_demo_user_fsm_and_fields():
    p = get_profile("demo_walkrun"); assert p.user_fsm_expected == 1000
    assert p.lowcmd.motor_mode == 1 and p.lowcmd.mode_machine == 5 and p.lowcmd.set_crc and p.lowcmd.topic == "rt/user_lowcmd"
    assert p.exit_mode == "AUTO" and p.damp_kd == 3.0 and not p.verified_on_firmware

def test_auto_exit_rule_matches_demo():
    assert resolve_exit_mode("AUTO", [1, 0, 0, 0]) == "WALKRUN"
    assert resolve_exit_mode("AUTO", q_rp(1.1, 0.0)) == "PASSIVE"      # |roll| > 1.0
    assert resolve_exit_mode("AUTO", q_rp(0.0, 0.9)) == "PASSIVE"      # |pitch| > 0.8
    assert resolve_exit_mode("AUTO", q_rp(0.5, 0.5)) == "WALKRUN"
    assert resolve_exit_mode("AUTO", None) == "PASSIVE"                # attitude unknown -> safest
    assert resolve_exit_mode("LAST", [1, 0, 0, 0]) == "LAST"

def test_roll_pitch_roundtrip():
    r, p = roll_pitch(q_rp(0.3, -0.2)); assert abs(r - 0.3) < 0.02 and abs(p + 0.2) < 0.02
