import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fasttrack import paths
paths.setup_sys_path(sdk=False); paths.setup_offline_env()
import numpy as np, pytest
from fasttrack.guard import Fault, GuardLimits, SafetyGuard, GuardReject, GuardConfig
from sim2real.config.robots import get_robot_cfg

@pytest.fixture
def g():
    lim = GuardLimits.from_robot_cfg(get_robot_cfg("g1")); return SafetyGuard(lim, GuardConfig(), kp_cap=np.full(29, 100.0))

def ok_state(**kw):
    d = dict(q=np.zeros(29), dq=np.zeros(29), tau_est=np.zeros(29), quat_wxyz=[1, 0, 0, 0], gyro=[0, 0, 0], tick=None, age_s=0.005, t=0.0); d.update(kw); return d

def test_nominal(g): assert g.check_state(**ok_state()) is Fault.NONE
def test_stale(g): assert g.check_state(**ok_state(age_s=0.2)) is Fault.COMM_TIMEOUT
def test_nan(g):
    q = np.zeros(29); q[3] = np.nan; assert g.check_state(**ok_state(q=q)) is Fault.STATE_INVALID
def test_quat(g): assert g.check_state(**ok_state(quat_wxyz=[3, 0, 0, 0])) is Fault.STATE_INVALID
def test_limit(g):
    q = np.zeros(29); q[3] = 3.5; assert g.check_state(**ok_state(q=q)) is Fault.JOINT_LIMIT
def test_speed(g):
    dq = np.zeros(29); dq[3] = 30; assert g.check_state(**ok_state(dq=dq)) is Fault.JOINT_SPEED
def test_torque_sustained(g):
    tau = np.zeros(29); tau[3] = 200
    r = [g.check_state(**ok_state(tau_est=tau, tick=i)) for i in range(10)]
    assert r[0] is Fault.NONE and r[-1] is Fault.TORQUE_SUSTAINED
def test_torque_single_spike_ok(g):
    tau = np.zeros(29); tau[3] = 200
    g.check_state(**ok_state(tau_est=tau, tick=1)); assert g.check_state(**ok_state(tick=2)) is Fault.NONE
def test_tilt(g): assert g.check_state(**ok_state(quat_wxyz=[0, 1, 0, 0])) is Fault.TILT
def test_frozen_tick(g):
    r = [g.check_state(**ok_state(tick=7)) for _ in range(8)]; assert r[-1] is Fault.STATE_FROZEN
def test_latched(g):
    g.check_state(**ok_state(age_s=1)); assert g.check_state(**ok_state()) is Fault.COMM_TIMEOUT

def cmd(g, qt, kp=50.0, kd=2.0, q=None, dq=None, dt=0.02):
    z = np.zeros(29); q = z if q is None else q; dq = z if dq is None else dq
    g.reset_target(q)
    return g.filter_command(qt, z, z + 5, np.full(29, kp), np.full(29, kd), q, dq, dt)

def test_cmd_nan(g):
    q = np.zeros(29); q[0] = np.nan
    with pytest.raises(GuardReject): cmd(g, q)
    assert g.fault is Fault.CMD_INVALID
def test_cmd_tau_ff_zeroed(g): assert (cmd(g, np.zeros(29))[2] == 0).all()
def test_cmd_kp_cap(g): assert (cmd(g, np.zeros(29), kp=500.0)[3] <= 100.0).all()
def test_cmd_limit_clip(g):
    qt = np.full(29, 10.0); o = cmd(g, qt, kp=1.0)[0]; assert (o <= g.lim.hi + 1e-9).all()
def test_cmd_slew(g):
    o = cmd(g, np.full(29, 0.3), kp=1.0)[0]; assert (np.abs(o) <= g.speed_lim * 0.6 * 0.02 + 1e-9).all()
def test_cmd_torque_bound(g):
    g.reset_target(np.zeros(29)); z = np.zeros(29)
    for _ in range(200):
        o = g.filter_command(np.full(29, 0.5), z, z, np.full(29, 100.0), np.full(29, 2.0), z, z, 0.02)
    tau = g.estimated_torque(o[0], o[3], o[4], z, z); assert (np.abs(tau) <= 0.85 * g.lim.effort + 1e-6).all()
def test_cmd_pos_inside_limits_after_torque(g):
    o = cmd(g, np.full(29, 99.0), kp=1.0); assert (o[0] >= g.lim.lo).all() and (o[0] <= g.lim.hi).all()
