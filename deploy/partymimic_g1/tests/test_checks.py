"""Real-time admission + the safety invariants that must not be relaxed (guard thresholds, frozen artifacts, entry rule, caps, writer ack, refusals before DDS)."""
import dataclasses, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fasttrack import paths
paths.setup_sys_path(sdk=False); paths.setup_offline_env()
import pytest
from fasttrack import checks, runner
from fasttrack.guard import GuardConfig
from fasttrack.profile import get_profile, entry_check

GUARD_R1 = {"state_stale_s": 0.06, "state_frozen_ticks": 5, "limit_fault_tol_rad": 0.05, "speed_fault_frac": 0.9, "unknown_speed_limit": 20.0,
            "torque_meas_fault_frac": 1.0, "torque_meas_fault_ticks": 8, "max_tilt_rad": 1.0, "max_gyro_rad_s": 10.0, "pos_margin_rad": 0.02,
            "target_speed_frac": 0.6, "torque_cmd_frac": 0.85, "kp_scale_max": 1.0, "kd_min_frac": 1.0, "allow_tau_ff": False}
PF_OK = [{"check": "q_within_hard_limits", "pass": True, "detail": ""}]
W1 = {"rt/lowcmd": [("p1", "k1")], "rt/arm_sdk": [], "rt/user_lowcmd": []}


def test_guard_thresholds_not_relaxed(): assert dataclasses.asdict(GuardConfig()) == GUARD_R1

def test_p2_artifacts_frozen():
    assert all(i["pass"] for i in checks.artifact_items())

def test_artifact_tamper_is_detected(monkeypatch, tmp_path):
    f = tmp_path / "x.onnx"; f.write_bytes(b"x"); monkeypatch.setattr(paths, "POLICY_ONNX", f)
    assert not [i for i in checks.artifact_items() if i["item"] == "artifact_onnx_frozen"][0]["pass"]

def test_entry_rule_forbids_zero_torque_and_passive_and_unknown_ids():
    p = get_profile(); allowed = (500, 505, 801)
    assert not entry_check(p, 0, allowed)[0] and not entry_check(p, 1, allowed)[0] and not entry_check(p, None, allowed)[0]
    assert not entry_check(p, 1, None)[0] and not entry_check(p, 1000, allowed)[0]
    assert all(entry_check(p, f, allowed)[0] for f in allowed)

def test_site_config_entry_allowed_and_override():
    site = checks.load_site(); assert checks.entry_allowed(site) == (500, 505, 801) and checks.entry_allowed(site, 801) == (801,)

def test_real_caps_not_relaxed():
    assert checks.caps_item("real", 1.0, 0.3)["pass"] and checks.caps_item("real", 0.5, 0.1)["pass"]
    for rs, au in ((1.01, 0.3), (7.06, 0.3), (1.0, 0.31), (1.0, 1.0), (0, 0.3), (1.0, 0)): assert not checks.caps_item("real", rs, au)["pass"], (rs, au)
    assert checks.caps_item("sim", 6.0, 1.0)["pass"]

def test_oem_writer_blocks_until_confirmed_with_matching_count():
    site = {"oem_lowcmd_writer": {"confirmed": False, "expected_writers": None}}
    r = checks.oem_writer_item("real", site, W1); assert not r["pass"] and "UNCONFIRMED" in r["detail"]          # the R1 blocker stays a blocker
    site = {"oem_lowcmd_writer": {"confirmed": True, "expected_writers": 1}}
    assert checks.oem_writer_item("real", site, W1)["pass"]
    assert not checks.oem_writer_item("real", site, {**W1, "rt/lowcmd": [("p1", "k1"), ("p2", "k2")]})["pass"]      # a second writer appeared
    assert not checks.oem_writer_item("real", site, {**W1, "rt/arm_sdk": [("p3", "k")]})["pass"]                    # other controller active
    assert not checks.oem_writer_item("real", site, RuntimeError("no discovery"))["pass"]                          # cannot verify -> blocked
    assert not checks.oem_writer_item("real", {}, W1)["pass"]
    assert checks.oem_writer_item("sim", {}, None)["pass"]

def test_shipped_site_config_is_blocked_by_default():
    assert not checks.oem_writer_item("real", checks.load_site(), W1)["pass"]

def test_any_failed_preflight_item_blocks_motion(monkeypatch):
    monkeypatch.setattr(checks, "guardian_item", lambda iface: checks.item("guardian_alive", True))
    site = {"oem_lowcmd_writer": {"confirmed": True, "expected_writers": 1}}
    ok, items = checks.admission(target="real", iface="e", run_s=1.0, authority=0.3, site=site, pf_results=PF_OK, writers=W1); assert ok, items
    bad = PF_OK + [{"check": "no_foreign_writer_on_user_lowcmd", "pass": False, "detail": "3 samples"}]
    ok, items = checks.admission(target="real", iface="e", run_s=1.0, authority=0.3, site=site, pf_results=bad, writers=W1); assert not ok
    monkeypatch.setattr(checks, "guardian_item", lambda iface: checks.item("guardian_alive", False, "stale"))
    ok, _ = checks.admission(target="real", iface="e", run_s=1.0, authority=0.3, site=site, pf_results=PF_OK, writers=W1); assert not ok

def test_guardian_heartbeat_gate(tmp_path):
    f = tmp_path / "alive"
    assert not checks.guardian_item("eth0", str(f))["pass"]                                                            # no heartbeat
    f.write_text(json.dumps({"t": time.time(), "pid": 1, "interface": "eth0"})); assert checks.guardian_item("eth0", str(f))["pass"]
    assert not checks.guardian_item("eth1", str(f))["pass"]                                                            # watching another interface
    f.write_text(json.dumps({"t": time.time() - 5, "pid": 1, "interface": "eth0"})); assert not checks.guardian_item("eth0", str(f))["pass"]

def test_runner_refuses_before_creating_any_dds_object(tmp_path, monkeypatch):
    import fasttrack.link as L
    monkeypatch.setattr(L, "DdsUserCtrlLink", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no DDS before refusal")))
    base = ["--target", "real", "--interface", "eth9", "--run-dir", str(tmp_path)]
    assert runner.main(base) == 2                                                # no --i-confirm-protection
    assert runner.main(base + ["--i-confirm-protection", "--run-s", "7.06"]) == 2   # full dance: refused
    assert runner.main(base + ["--i-confirm-protection", "--authority-max", "1.0"]) == 2
    assert runner.main(["--target", "real", "--interface", "lo", "--run-dir", str(tmp_path), "--i-confirm-protection"]) == 2
    assert runner.main(["--target", "sim", "--interface", "eth9", "--run-dir", str(tmp_path)]) == 2

def test_runner_has_no_guard_relaxing_options():
    src = (paths.PKG_ROOT / "fasttrack/runner.py").read_text()
    for opt in ("--guard-profile", "--guard-impl", "--torque-frac", "--max-tilt", "--pos-margin", "--max-gyro", "--target-speed-frac", "--debug-"):
        assert opt not in src, opt
