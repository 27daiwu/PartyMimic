#!/usr/bin/env python3
"""Simulated Unitree G1 that speaks the SAME SDK2 wire protocol as the real robot (rt/lowstate, rt/user_lowcmd, 'sport' RPC).

It is NOT the real OEM controller. What is modelled / assumed (all configurable, all stated in the test report):
  * `--entry-fsm` (default 500 = WalkRun-like) until SwitchToUserCtrl; then `--user-fsm-id` (default 1000 = value checked by the official
    g1_mimic_demo; the real value on a given firmware is still to be verified).
  * `--host-profile demo|sdk2` : demo = SwitchToUserCtrl rejected when fsm==1 (demo: "only in walkrun"); sdk2 = only accepted from fsm==1.
  * `--strict-frames` (default on): user_lowcmd frames are ACCEPTED ONLY if every one of the 29 motors has mode==1, mode_machine==5 and the
    CRC is valid (assumed enforcement, modelled on the demo's behaviour of always sending these; the real robot's enforcement is UNKNOWN).
    Rejected frames are counted (host_info.bad_frames) and ignored.  Use --no-strict-frames to model a host that ignores the fields.
  * in OEM hold the robot is either kinematically frozen (`--oem-hold frozen`, an idealised gantry) or PD-held at the
    default pose (`--oem-hold pd`).
  * user_lowcmd is applied as tau = tau_ff + kp(q_t-q) + kd(dq_t-dq), clipped to the effort limit (same law as sim2real
    SimulationBridge).  Frames received before the switch are stored but ignored (like the official example's pre-send).
  * `--user-watchdog-ms N`: user commands older than N ms are dropped and the host falls back to damping.
    Default 0 = NO watchdog (pessimistic: the real watchdog behaviour is UNKNOWN, so we do not rely on it).
  * SwitchToInternalCtrl(LAST|PASSIVE|WALKRUN): back to OEM hold (fsm 1 / 1 / 500) starting from the CURRENT state.
Fault injection (`--fault KIND@T`, T = seconds after user control starts): see FAULTS.
Support: `--support none|hang`; `hang` applies a vertical+damped point force on the torso (a hoist; feet clear of floor).
"""
from __future__ import annotations

import argparse, json, math, os, sched, signal, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fasttrack import paths
paths.setup_sys_path(); paths.setup_offline_env()

import numpy as np
import mujoco

FAULTS = {
    "state_stall": "stop publishing lowstate (comm loss)",
    "state_freeze": "keep publishing but freeze tick+content",
    "state_nan": "publish NaN joint positions",
    "state_quat_bad": "publish a non-unit IMU quaternion",
    "cmd_drop": "silently ignore user_lowcmd (command path ineffective)",
    "cmd_delay_20ms": "delay applying user commands by 20 ms",
    "cmd_delay_40ms": "delay applying user commands by 40 ms",
    "rpc_switch_fail": "SwitchToUserCtrl returns error 3104 and does not switch",
    "rpc_exit_fail_once": "first SwitchToInternalCtrl returns 3104 and does nothing",
    "rpc_exit_never": "SwitchToInternalCtrl never succeeds (tests UNCONFIRMED exit reporting)",
    "foreign_writer": "another process starts writing rt/user_lowcmd",
    "push": "200 N horizontal push on the torso for 0.2 s",
    "tau_hot": "report tau_est = 1.5x effort on the left knee (sensor over-torque)",
    "fsm_takeback": "the OEM silently leaves the user fsm (fsm -> 1) while user commands keep arriving",
    "mode_machine_mismatch": "lowstate reports mode_machine=7 (not a 29-dof G1)",
}


class Host:
    def __init__(self, a):
        from sim2real.config.robots import get_robot_cfg
        from sim2real.sim_env.utils.mjcf import load_sim_model
        self.a = a
        self.cfg = get_robot_cfg("g1")
        self.model = load_sim_model(self.cfg)
        self.model.opt.timestep = a.sim_dt
        self.data = mujoco.MjData(self.model)
        names = list(self.cfg.joint_names)
        self.names = names
        jn = [self.model.joint(i).name for i in range(self.model.njnt)]
        self.qadr = np.array([self.model.jnt_qposadr[jn.index(n)] for n in names])
        self.vadr = np.array([self.model.jnt_dofadr[jn.index(n)] for n in names])
        an = [self.model.actuator(i).name for i in range(self.model.nu)]
        self.aadr = np.array([an.index(n) for n in names])
        free = [i for i in range(self.model.njnt) if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE][0]
        self.radr, self.rvadr = int(self.model.jnt_qposadr[free]), int(self.model.jnt_dofadr[free])
        from sim2real.utils.strings import resolve_matching_names_values
        def arr(d):
            idx, _, v = resolve_matching_names_values(dict(d), names, preserve_order=True, strict=False)
            o = np.zeros(len(names)); o[idx] = v; return o
        self.effort = arr(self.cfg.joint_effort_limit)
        self.hold_kp = arr(self.cfg.safe_joint_kp); self.hold_kd = arr(self.cfg.safe_joint_kd)
        self.torso = self.model.body("torso_link").id
        self.pelvis = self.model.body("pelvis").id
        import yaml
        y = yaml.safe_load(paths.POLICY_YAML.read_text())
        idx, _, dv = resolve_matching_names_values(y["default_joint_pos"], names, preserve_order=True, strict=False)
        self.default_q = np.zeros(29); self.default_q[idx] = dv
        # initial state
        m = np.load(paths.MOTION_NPZ)
        if a.init == "frame0":
            self.data.qpos[self.radr:self.radr + 3] = m["body_pos_w"][0, 0]
            self.data.qpos[self.radr + 3:self.radr + 7] = m["body_quat_w"][0, 0]
            self.data.qpos[self.qadr] = m["joint_pos"][0]
        else:   # default pose hanging
            self.data.qpos[self.radr:self.radr + 3] = [0, 0, a.hang_z if a.support == 'hang' else 0.80]
            self.data.qpos[self.radr + 3:self.radr + 7] = [1, 0, 0, 0]
            self.data.qpos[self.qadr] = self.default_q
        mujoco.mj_forward(self.model, self.data)
        self.frozen_q = self.data.qpos.copy()
        self.bodies = list(map(str, m["body_names"]))
        self.body_ids = np.array([self.model.body(n).id for n in self.bodies])

        # state
        self._off_crc = None
        self.fsm = a.entry_fsm
        self.bad_frames = {"mode": 0, "mode_machine": 0, "crc": 0}; self.good_frames = 0
        self.user_active = False
        self.user_t0 = None
        self.lock = threading.Lock()
        self.cmd = None; self.cmd_t = 0.0; self.cmd_queue = []
        self.cmd_rx = 0
        self.hold_q = None        # OEM pd-hold target
        self.exit_calls = 0
        self.faults = {}
        for f in a.fault or []:
            k, t = f.split("@"); assert k in FAULTS, k
            self.faults[k] = float(t)
        self.fault_state = {"stall": False, "freeze": None}
        self.trace = {k: [] for k in ["t", "tick", "qpos", "qvel", "torque", "user", "fsm", "root_xz", "bodies", "cmd_age", "ctrl_clipped"]}
        self.stop = threading.Event()
        self._init_dds()

    # ------------------------------------------------------------------ DDS / RPC
    def _init_dds(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_, unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.rpc.server import Server
        from unitree_sdk2py.g1.loco.g1_loco_api import (LOCO_SERVICE_NAME, LOCO_API_VERSION, ROBOT_API_ID_LOCO_GET_FSM_ID,
            ROBOT_API_ID_LOCO_SWITCH_TO_USER_CTRL, ROBOT_API_ID_LOCO_SWITCH_TO_INTERNAL_CTRL)
        ChannelFactoryInitialize(self.a.domain, self.a.interface)
        self.pub = ChannelPublisher("rt/lowstate", LowState_); self.pub.Init()
        self.msg = unitree_hg_msg_dds__LowState_(); self.msg.mode_machine = 5; self.msg.mode_pr = 0
        from fasttrack.crc import lowcmd_crc
        self._crc = lowcmd_crc
        self.sub = ChannelSubscriber("rt/user_lowcmd", LowCmd_); self.sub.Init(self._on_cmd, 5)
        host = self
        class Loco(Server):
            def __init__(s): super().__init__(LOCO_SERVICE_NAME)
            def Init(s):
                s._SetApiVersion(LOCO_API_VERSION)
                s._RegistHandler(ROBOT_API_ID_LOCO_GET_FSM_ID, lambda p: (0, json.dumps({"data": host.fsm})), False)
                s._RegistHandler(ROBOT_API_ID_LOCO_SWITCH_TO_USER_CTRL, host._rpc_user, False)
                s._RegistHandler(ROBOT_API_ID_LOCO_SWITCH_TO_INTERNAL_CTRL, host._rpc_internal, False)
        self.srv = Loco(); self.srv.Init(); self.srv.Start()
        self.foreign_pub = None
        self._LowCmd = LowCmd_; self._mk = unitree_hg_msg_dds__LowCmd_; self._ChannelPublisher = ChannelPublisher

    def _on_cmd(self, m):
        try:
            mc = m.motor_cmd
            c = np.array([[mc[i].q, mc[i].dq, mc[i].tau, mc[i].kp, mc[i].kd] for i in range(29)], dtype=np.float64)
            modes_ok = all(int(mc[i].mode) == 1 for i in range(29)); mm_ok = int(m.mode_machine) == 5
            crc_ok = (int(m.crc) == self._crc(m))
        except Exception:
            return
        now = time.monotonic()
        with self.lock:
            self.cmd_rx += 1
            if self.a.strict_frames:
                if not (modes_ok and mm_ok and crc_ok):
                    self.bad_frames["mode"] += int(not modes_ok); self.bad_frames["mode_machine"] += int(not mm_ok); self.bad_frames["crc"] += int(not crc_ok)
                    return
            self.good_frames += 1
            self.cmd_queue.append((now, c))
            self.cmd_queue = self.cmd_queue[-50:]

    def _rel_t(self):
        return None if self.user_t0 is None else time.monotonic() - self.user_t0

    def _rpc_user(self, p):
        if "rpc_switch_fail" in self.faults:
            return 3104, ""
        with self.lock:
            if not self.user_active:
                if self.a.host_profile == "sdk2" and self.fsm != 1:
                    return 3204, ""
                if self.a.host_profile == "demo" and self.fsm == 1:
                    return 3204, ""
            if not self.user_active:
                self.user_active = True; self.user_t0 = time.monotonic()
                self.fsm = self.a.user_fsm_id
                self.hold_q = None
        return 0, ""

    def _rpc_internal(self, p):
        mode = json.loads(p).get("data", 0) if p else 0
        self.exit_calls += 1
        if "rpc_exit_never" in self.faults:
            return 3104, ""          # timeout-like: must NOT be retried blindly
        if "rpc_exit_fail_once" in self.faults and self.exit_calls == 1:
            return 3204, ""          # explicit server rejection: one retry is allowed
        with self.lock:
            self.user_active = False
            self.fsm = {0: self.a.entry_fsm, 1: 1, 2: 500}.get(mode, 1)   # LAST->previous entry fsm, PASSIVE->1, WALKRUN->500
            self.hold_q = self.data.qpos[self.qadr].copy()      # OEM resumes from the CURRENT pose
            self.frozen_q = self.data.qpos.copy() if self.a.oem_hold == "frozen" else self.frozen_q
        return 0, ""

    # ------------------------------------------------------------------ physics
    def _applied_cmd(self, now):
        """Return (cmd array 29x5 or None, age) honouring delay / drop faults and the optional user watchdog."""
        d = {"cmd_delay_20ms": 0.020, "cmd_delay_40ms": 0.040}
        delay = 0.0
        for k, v in d.items():
            if k in self.faults and (self._rel_t() or 0) >= self.faults[k]:
                delay = v
        if "cmd_drop" in self.faults and (self._rel_t() or 0) >= self.faults["cmd_drop"]:
            return None, math.inf
        with self.lock:
            q = [(t, c) for (t, c) in self.cmd_queue if t <= now - delay]
        if not q:
            return None, math.inf
        t, c = q[-1]
        age = now - t
        if self.a.user_watchdog_ms > 0 and age > self.a.user_watchdog_ms / 1e3:
            return None, age
        return c, age

    def step(self, tsim):
        d, m = self.data, self.model
        now = time.monotonic()
        rel = self._rel_t()
        q, dq = d.qpos[self.qadr], d.qvel[self.vadr]
        tau = np.zeros(29); clipped = 0
        if self.user_active:
            c, age = self._applied_cmd(now)
            if c is None and self.a.user_watchdog_mode == "revert" and self.a.user_watchdog_ms > 0 and age != math.inf:
                with self.lock:
                    self.user_active = False; self.fsm = self.a.entry_fsm; self.hold_q = d.qpos[self.qadr].copy()
            if not self.user_active:
                pass
            elif c is None:
                tau = -2.0 * dq                            # damping fallback (host-side, no command / watchdog)
            else:
                tau = c[:, 2] + c[:, 3] * (c[:, 0] - q) + c[:, 4] * (c[:, 1] - dq)
        else:
            if self.a.oem_hold == "pd":
                tgt = self.hold_q if self.hold_q is not None else self.default_q
                tau = self.hold_kp * (tgt - q) - self.hold_kd * dq
            age = 0.0
        t2 = np.clip(tau, -self.effort, self.effort); clipped = int((t2 != tau).any()); tau = t2
        d.ctrl[:] = 0.0; d.ctrl[self.aadr] = tau
        d.xfrc_applied[:] = 0.0
        if self.a.support == "hang":
            p, v = d.xpos[self.torso], d.cvel[self.torso, 3:6]
            fz = self.a.hang_kz * (self.a.hang_z + 0.25 - p[2]) - self.a.hang_cz * v[2]
            fxy = -self.a.hang_kxy * (p[:2]) - self.a.hang_cxy * v[:2]
            d.xfrc_applied[self.torso, :3] = [fxy[0], fxy[1], max(fz, 0.0)]
        if "push" in self.faults and rel is not None and self.faults["push"] <= rel < self.faults["push"] + 0.2:
            d.xfrc_applied[self.torso, 0] += 200.0
        if self.user_active is False and self.a.oem_hold == "frozen":
            d.qpos[:] = self.frozen_q; d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
        else:
            mujoco.mj_step(m, d)
        self._publish(tsim, rel, tau)
        self.trace["t"].append(tsim)
        if len(self.trace["t"]) % self.a.trace_every == 0 or True:
            pass
        return tau, age, clipped

    def _publish(self, tsim, rel, tau):
        f = self.faults
        if "fsm_takeback" in f and rel is not None and rel >= f["fsm_takeback"] and self.user_active:
            with self.lock:
                self.user_active = False; self.fsm = 1; self.hold_q = self.data.qpos[self.qadr].copy()
        if "state_stall" in f and rel is not None and rel >= f["state_stall"]:
            return
        if "state_freeze" in f and rel is not None and rel >= f["state_freeze"]:
            if self.fault_state["freeze"] is None:
                self.fault_state["freeze"] = (self.msg.tick,)
            self.pub.Write(self.msg); return
        d = self.data
        msg = self.msg
        for i in range(29):
            ms = msg.motor_state[i]
            ms.q = float(d.qpos[self.qadr[i]]); ms.dq = float(d.qvel[self.vadr[i]])
            ms.tau_est = float(d.actuator_force[self.aadr[i]])
        if "tau_hot" in f and rel is not None and rel >= f["tau_hot"]:
            msg.motor_state[3].tau_est = float(1.5 * self.effort[3])
        if "state_nan" in f and rel is not None and rel >= f["state_nan"]:
            msg.motor_state[5].q = float("nan")
        q = d.qpos[self.radr + 3:self.radr + 7]
        if "state_quat_bad" in f and rel is not None and rel >= f["state_quat_bad"]:
            q = q * 3.0
        msg.imu_state.quaternion = [float(x) for x in q]
        from fasttrack.profile import roll_pitch
        rr, pp = roll_pitch(q)
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        msg.imu_state.rpy = [float(rr), float(pp), float(yaw)]
        msg.imu_state.gyroscope = [float(x) for x in d.qvel[self.rvadr + 3:self.rvadr + 6]]
        msg.tick = int(tsim * 1000) & 0xFFFFFFFF
        msg.mode_machine = 7 if "mode_machine_mismatch" in f else 5
        if self._off_crc is None:
            from unitree_sdk2py.utils.crc import CRC
            self._off_crc = CRC().Crc          # OFFICIAL library (independent of fasttrack.crc)
        msg.crc = self._off_crc(msg)
        self.pub.Write(msg)

    def maybe_foreign_writer(self, rel):
        if rel is None or "foreign_writer" not in self.faults or rel < self.faults["foreign_writer"] or self.foreign_pub is not None:
            return
        self.foreign_pub = True
        def run():
            pub = self._ChannelPublisher("rt/user_lowcmd", self._LowCmd); pub.Init(); msg = self._mk()
            while not self.stop.is_set():
                pub.Write(msg); time.sleep(0.01)
        threading.Thread(target=run, daemon=True, name="foreign-writer").start()

    def record(self, tsim, tau, age, clipped):
        d = self.data
        tr = self.trace
        tr["tick"].append(int(tsim * 1000)); tr["qpos"].append(d.qpos.copy()); tr["qvel"].append(d.qvel.copy())
        tr["torque"].append(tau.copy()); tr["user"].append(int(self.user_active)); tr["fsm"].append(self.fsm)
        tr["cmd_age"].append(age); tr["ctrl_clipped"].append(clipped)
        tr["bodies"].append(d.xpos[self.body_ids].copy())

    def run(self):
        sim_dt = self.a.sim_dt
        sch = sched.scheduler(time.perf_counter, time.sleep)
        t0 = time.perf_counter(); n = 0
        for k in self.trace:
            if k not in ("t",): self.trace[k] = []
        self.trace["t"] = []
        overruns = 0
        stop_at = self.a.max_runtime_s
        while not self.stop.is_set():
            target = t0 + n * sim_dt
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)
            elif now - target > 0.02:
                overruns += 1
            tsim = n * sim_dt
            self.maybe_foreign_writer(self._rel_t())
            tau, age, clipped = self.step(tsim)
            self.record(tsim, tau, age, clipped)
            n += 1
            if stop_at and tsim > stop_at:
                break
            if self.a.exit_after_user_done and self.user_t0 is not None and not self.user_active and self.exit_calls > 0 and \
               (time.monotonic() - self._exit_t0()) > self.a.post_exit_s:
                break
        self.overruns = overruns
        self.save()

    _et = None
    def _exit_t0(self):
        if self._et is None:
            self._et = time.monotonic()
        return self._et

    def save(self):
        out = Path(self.a.out); out.mkdir(parents=True, exist_ok=True)
        tr = {k: np.asarray(v) for k, v in self.trace.items() if len(v)}
        np.savez_compressed(out / "host_trace.npz", body_names=np.array(self.bodies), joint_names=np.array(self.names),
                            sim_dt=self.a.sim_dt, **tr)
        info = {"args": vars(self.a), "overruns": getattr(self, "overruns", None), "exit_calls": self.exit_calls,
                "cmd_rx": self.cmd_rx, "good_frames": self.good_frames, "bad_frames": self.bad_frames, "faults": self.faults, "final_fsm": self.fsm, "user_active": self.user_active}
        (out / "host_info.json").write_text(json.dumps(info, indent=1, default=str))
        print("host saved", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interface", default="lo"); ap.add_argument("--domain", type=int, default=77)
    ap.add_argument("--init", choices=["frame0", "default"], default="default")
    ap.add_argument("--oem-hold", choices=["frozen", "pd"], default="pd")
    ap.add_argument("--support", choices=["none", "hang"], default="hang")
    ap.add_argument("--hang-z", type=float, default=1.05)
    ap.add_argument("--hang-kz", type=float, default=3000.0); ap.add_argument("--hang-cz", type=float, default=400.0)
    ap.add_argument("--hang-kxy", type=float, default=300.0); ap.add_argument("--hang-cxy", type=float, default=80.0)
    ap.add_argument("--entry-fsm", type=int, default=500)
    ap.add_argument("--host-profile", choices=["demo", "sdk2"], default="demo")
    ap.add_argument("--strict-frames", dest="strict_frames", action="store_true", default=True)
    ap.add_argument("--no-strict-frames", dest="strict_frames", action="store_false")
    ap.add_argument("--user-fsm-id", type=int, default=1000, help="value checked by the official demo; unverified on a given firmware")
    ap.add_argument("--user-watchdog-ms", type=float, default=0.0)
    ap.add_argument("--user-watchdog-mode", choices=["damp", "revert"], default="damp",
                    help="what the modelled firmware watchdog does after --user-watchdog-ms of command silence: damp = host damps, fsm unchanged; revert = host also leaves the user fsm (returns to the entry fsm)")
    ap.add_argument("--fault", action="append", help="KIND@T seconds after user control starts; kinds: " + ",".join(FAULTS))
    ap.add_argument("--sim-dt", type=float, default=0.005)
    ap.add_argument("--max-runtime-s", type=float, default=120.0)
    ap.add_argument("--exit-after-user-done", action="store_true"); ap.add_argument("--post-exit-s", type=float, default=3.0)
    ap.add_argument("--trace-every", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.interface != "lo":
        sys.exit("sim_host refuses to run on a non-loopback interface (it must never share a network with a real robot)")
    h = Host(a)
    signal.signal(signal.SIGTERM, lambda *_: h.stop.set())
    print("host ready", flush=True)
    try:
        h.run()
    except KeyboardInterrupt:
        h.stop.set(); h.save()
    sys.stdout.flush(); os._exit(0)      # DDS reader threads can keep the interpreter alive


if __name__ == "__main__":
    main()
