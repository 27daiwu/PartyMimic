"""Real-robot motion gates: every unmet gate must make the runner REFUSE (exit 2) before any DDS object exists."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fasttrack import paths
paths.setup_sys_path(sdk=False); paths.setup_offline_env()
import pytest
from fasttrack import runner

def args(tmp, **kw):
    a = ["--target", "real", "--interface", "eth9", "--run-dir", str(tmp), "--preflight-pass", str(tmp / "pf.json"), "--segment-s", "2"]
    for k, v in kw.items(): a += [k] if v is True else [k, str(v)]
    return a

@pytest.fixture(autouse=True)
def no_dds(monkeypatch):
    monkeypatch.setattr(runner, "make_link", lambda a: (_ for _ in ()).throw(AssertionError("DDS link must not be created when gates fail")))
    monkeypatch.delenv(runner.REAL_GATE_ENV, raising=False)

def test_refuse_all_missing(tmp_path): assert runner.main(args(tmp_path)) == 2
def test_refuse_env_only(tmp_path, monkeypatch):
    monkeypatch.setenv(runner.REAL_GATE_ENV, runner.REAL_GATE_VALUE); assert runner.main(args(tmp_path, **{"--i-confirm-hanging-and-estop": True})) == 2
def test_missing_gates_listed(tmp_path, monkeypatch):
    a = runner.argparse.Namespace(i_confirm_hanging_and_estop=False, preflight_pass=str(tmp_path / "x"), preflight_max_age_s=900, interface="eth9",
                                  segment_s=None, real_max_segment_s=3, authority_max=1.0, real_max_authority=0.3, entry_fsm=None, profile="demo_walkrun")
    m = runner.check_real_gates(a); assert len(m) >= 5
def test_long_segment_refused(tmp_path):
    a = runner.argparse.Namespace(i_confirm_hanging_and_estop=True, preflight_pass=str(tmp_path / "x"), preflight_max_age_s=900, interface="eth9",
                                  segment_s=10, real_max_segment_s=3, authority_max=0.3, real_max_authority=0.3, entry_fsm=500, profile="demo_walkrun")
    assert any("segment" in x for x in runner.check_real_gates(a))
def test_open_guard_refused_on_real(tmp_path):
    assert runner.main(args(tmp_path, **{"--guard-profile": "open"})) == 2


def _ns(tmp, **kw):
    d = dict(i_confirm_hanging_and_estop=True, preflight_pass=str(tmp / "pf.json"), preflight_max_age_s=900, interface="eth9", segment_s=2,
             real_max_segment_s=3, authority_max=0.3, real_max_authority=0.3, entry_fsm=500, profile="demo_walkrun", hold_only_s=0.0)
    d.update(kw); return runner.argparse.Namespace(**d)

def _write_pf(tmp, **kw):
    d = {"time": time.time(), "all_pass": True, "diag_ok": True, "interface": "eth9", "profile": "demo_walkrun", "fsm": 500,
         "onnx_sha256": paths.EXPECTED_SHA256["onnx"]}; d.update(kw); (tmp / "pf.json").write_text(json.dumps(d))

def _write_guardian(age=0.0, iface="eth9"):
    Path(runner.GUARDIAN_ALIVE).write_text(json.dumps({"t": time.time() - age, "pid": 1, "interface": iface}))

@pytest.fixture
def env_ok(monkeypatch):
    monkeypatch.setenv(runner.REAL_GATE_ENV, runner.REAL_GATE_VALUE)
    bak = Path(runner.GUARDIAN_ALIVE); old = bak.read_text() if bak.exists() else None
    yield
    if old is None:
        bak.unlink(missing_ok=True)
    else:
        bak.write_text(old)

def test_all_gates_met_passes(tmp_path, env_ok):
    _write_pf(tmp_path); _write_guardian(); assert runner.check_real_gates(_ns(tmp_path)) == []

def test_missing_guardian_refuses(tmp_path, env_ok):
    _write_pf(tmp_path); Path(runner.GUARDIAN_ALIVE).unlink(missing_ok=True)
    assert any("guardian not running" in m for m in runner.check_real_gates(_ns(tmp_path)))

def test_stale_guardian_refuses(tmp_path, env_ok):
    _write_pf(tmp_path); _write_guardian(age=5.0)
    assert any("guardian heartbeat stale" in m for m in runner.check_real_gates(_ns(tmp_path)))

def test_guardian_other_interface_refuses(tmp_path, env_ok):
    _write_pf(tmp_path); _write_guardian(iface="eth1")
    assert any("not eth9" in m for m in runner.check_real_gates(_ns(tmp_path)))

def test_entry_fsm_must_match_preflight(tmp_path, env_ok):
    _write_pf(tmp_path, fsm=501); _write_guardian()
    assert any("observed fsm 501" in m for m in runner.check_real_gates(_ns(tmp_path)))

def test_entry_fsm_required(tmp_path, env_ok):
    _write_pf(tmp_path); _write_guardian(); assert any("--entry-fsm" in m for m in runner.check_real_gates(_ns(tmp_path, entry_fsm=None)))

def test_preflight_without_diag_ok_refused(tmp_path, env_ok):
    _write_pf(tmp_path, diag_ok=False); _write_guardian(); assert any("diagnostic" in m for m in runner.check_real_gates(_ns(tmp_path)))

def test_preflight_profile_mismatch(tmp_path, env_ok):
    _write_pf(tmp_path, profile="sdk2_passive"); _write_guardian(); assert any("profile" in m for m in runner.check_real_gates(_ns(tmp_path)))

def test_hold_only_limited_and_authority_irrelevant(tmp_path, env_ok):
    _write_pf(tmp_path); _write_guardian()
    assert runner.check_real_gates(_ns(tmp_path, hold_only_s=3.0, segment_s=None, authority_max=1.0)) == []
    assert any("<= 5" in m for m in runner.check_real_gates(_ns(tmp_path, hold_only_s=9.0)))
