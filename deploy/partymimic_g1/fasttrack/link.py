"""Transport to a Unitree G1 through the OFFICIAL SDK2 user-control route.

    rt/lowstate        (HG LowState_)  <- robot          read-only
    rt/user_lowcmd     (HG LowCmd_)    -> robot          ONLY after SwitchToUserCtrl succeeded
    LocoClient (sport service): GetFsmId (7001), SwitchToUserCtrl (7110), SwitchToInternalCtrl (7111)

This mirrors unitree_sdk2/example/g1/high_level/g1_userctrl_dds_example.cpp (see results/p3/official).
It deliberately does NOT use MotionSwitcher.ReleaseMode / rt/lowcmd (EGalahad's debug-mode route): that is a different
route and cannot hand control back to the OEM controller.

Read-only mode (``read_only=True``): no publisher is created, no switching RPC is exposed.
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

N_JOINTS = 29
N_SLOTS = 35


@dataclass
class LinkState:
    q: np.ndarray
    dq: np.ndarray
    tau_est: np.ndarray
    quat_wxyz: np.ndarray
    gyro: np.ndarray
    tick: int
    mode_machine: int
    arrival_mono: float
    seq: int            # local receive counter
    mode_pr: int = 0
    version: tuple = (0, 0)
    rpy: tuple = (0.0, 0.0, 0.0)
    motor_mode: tuple = ()      # per-joint motor_state.mode (29)
    motor_flags: tuple = ()     # per-joint motor_state.motorstate (29)
    wireless: bytes = b""       # rt/lowstate.wireless_remote (40 bytes): remote controller keys


class Link:
    """Interface used by the runner / guardian / session. Implemented by DdsUserCtrlLink and test fakes."""

    def latest(self) -> Optional[LinkState]: ...
    def add_state_listener(self, fn) -> None: ...
    def enable_write(self) -> None: ...
    def send(self, q, dq, tau, kp, kd) -> None: ...
    def get_fsm(self) -> tuple[int, Optional[int]]: ...           # (rpc_code, fsm_id)
    def switch_to_user(self) -> int: ...
    def switch_to_internal(self, mode: str) -> int: ...
    def command_topic_rx_count(self) -> int: ...
    def close(self) -> None: ...


class DdsUserCtrlLink(Link):
    CMD_TOPIC = "rt/user_lowcmd"
    STATE_TOPIC = "rt/lowstate"

    def __init__(self, interface: str, domain_id: int = 0, *, read_only: bool = False, rpc_timeout_s: float = 2.0,
                 lowcmd=None):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        from unitree_sdk2py.g1.loco.g1_loco_api import InternalFsmMode

        self.read_only = read_only
        self.interface, self.domain_id = interface, domain_id
        from .profile import LowCmdFormat
        self.fmt = lowcmd or LowCmdFormat()
        self.CMD_TOPIC = self.fmt.topic
        self._InternalFsmMode = InternalFsmMode
        self._state_lock = threading.Lock()
        self._latest: Optional[LinkState] = None
        self._rx_seq = 0
        self._own_writes = 0
        self._cmd_rx = 0
        self._active_rx_tm = 0.0
        self._cmd_rx_t0 = time.monotonic()
        self._closed = False
        self._listeners: list = []

        ChannelFactoryInitialize(domain_id, interface)
        self._sub_state = ChannelSubscriber(self.STATE_TOPIC, LowState_)
        self._sub_state.Init(self._on_state, 5)
        # foreign-writer detection: count samples seen on the command topic (own writes are echoed to local readers)
        # NOTE: polling reader, NOT a listener callback. A listener on a topic we also write deadlocks CycloneDDS+GIL when a
        # second writer's samples arrive concurrently with our own write (found by the foreign_writer sim case).
        self._sub_cmd = ChannelSubscriber(self.CMD_TOPIC, LowCmd_)
        self._sub_cmd.Init(None, 0)
        self._cmd_poll_stop = threading.Event()
        threading.Thread(target=self._poll_cmd, daemon=True, name="ft-cmd-observer").start()
        self._pub = None
        self._cmd_msg = None
        self._mk_pub = lambda: (ChannelPublisher(self.CMD_TOPIC, LowCmd_), unitree_hg_msg_dds__LowCmd_())
        self._loco = LocoClient()
        self._loco.SetTimeout(rpc_timeout_s)
        self._loco.Init()

    # ---- subscribers (SDK reader threads)
    def _on_state(self, msg) -> None:
        try:
            ms = msg.motor_state
            q = np.fromiter((ms[i].q for i in range(N_JOINTS)), np.float64, N_JOINTS)
            dq = np.fromiter((ms[i].dq for i in range(N_JOINTS)), np.float64, N_JOINTS)
            tau = np.fromiter((ms[i].tau_est for i in range(N_JOINTS)), np.float64, N_JOINTS)
            st = LinkState(q, dq, tau, np.asarray(msg.imu_state.quaternion, np.float64),
                           np.asarray(msg.imu_state.gyroscope, np.float64), int(msg.tick), int(msg.mode_machine),
                           time.monotonic(), 0, int(msg.mode_pr), tuple(int(v) for v in msg.version),
                           tuple(float(v) for v in msg.imu_state.rpy),
                           tuple(int(ms[i].mode) for i in range(N_JOINTS)), tuple(int(ms[i].motorstate) for i in range(N_JOINTS)),
                           bytes(int(b) & 0xFF for b in msg.wireless_remote))
        except Exception:
            return
        with self._state_lock:
            self._rx_seq += 1
            st.seq = self._rx_seq
            self._latest = st
        for fn in self._listeners:          # cheap, non-blocking callbacks only (e.g. the START double-click detector)
            try:
                fn(st)
            except Exception:
                pass

    def add_state_listener(self, fn) -> None:
        self._listeners.append(fn)

    def _poll_cmd(self) -> None:
        while not self._cmd_poll_stop.is_set():
            n = 0
            while n < 200:
                try:
                    smp = self._sub_cmd.Read()
                except Exception:
                    smp = None
                if smp is None:
                    break
                self._cmd_rx += 1; n += 1
                try:                       # stiff (policy-like) frame = any kp > 0; damping/zero frames do not count
                    if any(smp.motor_cmd[i].kp > 0.0 for i in range(N_JOINTS)):
                        self._active_rx_tm = time.monotonic()
                except Exception:
                    pass
            time.sleep(0.005)

    def latest(self) -> Optional[LinkState]:
        with self._state_lock:
            return self._latest

    # ---- publisher (created lazily: read-only sessions never own a writer)
    def enable_write(self) -> None:
        if self.read_only:
            raise RuntimeError("read-only link cannot enable write")
        if self._pub is None:
            self._pub, self._cmd_msg = self._mk_pub()
            self._pub.Init()

    def prime(self) -> None:
        """Pre-build everything a take-over needs (publisher, message, CRC path) WITHOUT putting a sample on the wire."""
        self.enable_write()
        from .crc import lowcmd_crc
        lowcmd_crc(self._cmd_msg)

    def send(self, q, dq, tau, kp, kd) -> None:
        if self.read_only or self._pub is None:
            raise RuntimeError("read-only link cannot send")
        f = self.fmt
        m = self._cmd_msg
        mc = m.motor_cmd
        for i in range(N_JOINTS):
            c = mc[i]
            c.q, c.dq, c.tau, c.kp, c.kd = float(q[i]), float(dq[i]), float(tau[i]), float(kp[i]), float(kd[i])
            if f.set_motor_mode:
                c.mode = f.motor_mode
        # slots 29..34 are never used: they stay at their zero defaults (demo: mode 0, kp=kd=0)
        if f.mode_machine is not None:
            m.mode_machine = f.mode_machine
        elif f.set_crc or f.set_motor_mode:
            st = self.latest()
            if st is not None:
                m.mode_machine = st.mode_machine
        m.mode_pr = f.mode_pr
        if f.set_crc:
            from .crc import lowcmd_crc
            m.crc = lowcmd_crc(m)
        self._pub.Write(m)
        self._own_writes += 1

    # ---- RPC (sport service)
    def get_fsm(self):
        try:
            code, data = self._loco.GetFsmId()
        except Exception:
            return -1, None
        return int(code), (None if data is None else int(data))

    def switch_to_user(self) -> int:
        if self.read_only:
            raise RuntimeError("read-only link cannot switch")
        try:
            return int(self._loco.SwitchToUserCtrl())
        except Exception:
            return -1

    def switch_to_internal(self, mode: str = "PASSIVE") -> int:   # mode in LAST|PASSIVE|WALKRUN (AUTO is resolved by the caller)
        if self.read_only:
            raise RuntimeError("read-only link cannot switch")
        try:
            return int(self._loco.SwitchToInternalCtrl(getattr(self._InternalFsmMode, mode)))
        except Exception:
            return -1

    def command_topic_rx_count(self) -> int:
        return self._cmd_rx

    def last_active_rx_tm(self) -> float:
        return self._active_rx_tm

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cmd_poll_stop.set()
        for s in (self._sub_state, self._sub_cmd):
            try:
                s.Close()
            except Exception:
                pass
