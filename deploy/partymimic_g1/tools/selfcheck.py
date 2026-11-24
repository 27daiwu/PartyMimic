#!/usr/bin/env python3
"""Self-test of the DEPLOYED stack on THIS machine (robot computer or dev PC).  Read-only w.r.t. any robot: DDS is used on loopback `lo`, domain 78 only.

  1  platform / package versions (printed, informational) and package MANIFEST (sha256 of every shipped file) if present
  2  frozen artifacts: ONNX / SSB / yaml sha256
  3  ONNX parity with the export fixture (ssb_seed0_it1500.check.npz)                          tol 1e-5
  4  motion-loader fingerprint (torch/any4hdmi version drift detector): sha256 of the loaded float arrays == reference
  5  state replay: the P2 golden states fed through the deployed adapter -> actions vs reference computed on the dev PC   tol 1e-3
  6  per-tick compute time (update + observations + ONNX): p50/p99/max.  PASS p99<=15 ms, WARN <=30 ms, FAIL >30 ms (period 20 ms, runner watchdog 100 ms)
  7  DDS loopback: LowState_ round trip through DdsUserCtrlLink (incl. START bit), LowCmd_ frame fields + CRC vs the OFFICIAL lib
Reference data: assets/expected/selfcheck_ref.npz (+ .json).  Make it on the dev PC:  tools/selfcheck.py --make-reference
Exit code 0 = no FAIL (WARN allowed), 1 = at least one FAIL.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2"); os.environ.setdefault("SIM2REAL_ORT_NUM_THREADS", "2")
from fasttrack import paths
paths.setup_sys_path(); paths.setup_offline_env()
import numpy as np

REF_DIR = paths.PKG_ROOT / "assets/expected"
REF_NPZ, REF_JSON = REF_DIR / "selfcheck_ref.npz", REF_DIR / "selfcheck_ref.json"
GOLD = paths.P2 / "sim2real_adapter/runs/ssb_seed0_it1500/seed_0_init_zero/control_end_all.npz"
FIXTURE = paths.ART / "ssb_seed0_it1500.check.npz"
RES = []


def rec(name, status, detail=""):
    RES.append((name, status, detail)); print(f"[{status:4s}] {name}: {detail}", flush=True)


class _FakeSession:
    def __init__(self): self.st = None
    def state(self): return self.st


def _link_state(q, dq, quat, gyro, tick):
    from fasttrack.link import LinkState
    return LinkState(np.asarray(q, np.float64).copy(), np.asarray(dq, np.float64).copy(), np.zeros(29), np.asarray(quat, np.float64).copy(),
                     np.asarray(gyro, np.float64).copy(), int(tick), 5, time.monotonic(), int(tick))


class Deployed:
    """fasttrack.runner.build_policy + SessionRobotIO over a fake session; tick() == the runner's policy step."""
    def __init__(self):
        from fasttrack.runner import build_policy, SessionRobotIO
        import contextlib, io
        self.sess = _FakeSession(); self.rio = SessionRobotIO(self.sess); self.n = 0
        with contextlib.redirect_stdout(io.StringIO()):
            self.tr = build_policy(argparse.Namespace(), self.rio)

    def feed(self, q, dq, quat, gyro):
        self.n += 1; self.sess.st = _link_state(q, dq, quat, gyro, self.n)
        assert self.tr.state_processor._prepare_low_state()

    def reset(self):
        self.tr.reset(); self.tr.state_dict["control_mode"] = "policy"; self.tr.state_dict["paused"] = True

    def tick(self):
        from fasttrack.runner import policy_tick
        a, q = policy_tick(self.tr)
        return np.array(a, np.float32), q


def replay(states, n=None):
    d = Deployed(); q, dq, quat, gyro = states
    d.feed(q[0], dq[0], quat[0], gyro[0]); d.reset(); d.tr.state_dict["paused"] = False
    A, QT, T = [], [], []
    for k in range(len(q) if n is None else n):
        d.feed(q[k], dq[k], quat[k], gyro[k]); t0 = time.perf_counter(); a, qt = d.tick(); T.append((time.perf_counter() - t0) * 1e3); A.append(a); QT.append(qt)
    return d, np.array(A), np.array(QT), np.array(T)


def fingerprint(d):
    ds = d.tr.motion_dataset; T = int(ds.num_steps)
    md = ds.get_slice(np.zeros(T, dtype=int), np.arange(T), np.array([0]))
    h = hashlib.sha256()
    for name in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w"):
        h.update(np.ascontiguousarray(np.asarray(getattr(md, name), np.float32)).tobytes())
    return h.hexdigest(), T


def make_reference():
    with np.load(GOLD) as z:
        q, dq, quat = z["robot_joint_pos"], z["robot_joint_vel"], z["robot_root_quat_w"]; gyro = z["robot_root_ang_vel_b"]
    d, A, QT, T = replay((q, dq, quat, gyro))
    fp, steps = fingerprint(d)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REF_NPZ, q=q, dq=dq, quat=quat, gyro=gyro, actions=A, q_target=QT)
    REF_JSON.write_text(json.dumps({"loader_fingerprint": fp, "motion_steps": steps, "made_on": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0],
                                    "onnxruntime": _ver("onnxruntime"), "torch": _ver("torch"), "frames": int(len(q)), "dev_tick_ms_p50": float(np.percentile(T, 50))}, indent=1))
    print("reference written:", REF_NPZ, REF_JSON)


def _ver(mod):
    try:
        import importlib.metadata as md
        return md.version({"yaml": "pyyaml", "zmq": "pyzmq"}.get(mod, mod))
    except Exception:
        return None


def check_platform():
    vs = {m: _ver(m) for m in ("numpy", "onnxruntime", "mujoco", "torch", "tensordict", "scipy", "pyyaml", "pyzmq", "cyclonedds", "tyro", "loguru")}
    rec("1.platform", "INFO", f"{platform.machine()} {platform.system()} {platform.release()} python {sys.version.split()[0]}; " + ", ".join(f"{k} {v}" for k, v in vs.items()))
    if platform.machine() not in ("aarch64", "arm64"):
        rec("1.arch", "INFO", f"not aarch64 ({platform.machine()}): fine on a dev PC, but this is NOT the robot computer")
    for m in ("numpy", "onnxruntime", "mujoco", "torch", "cyclonedds"):
        if vs[m] is None: rec(f"1.import_{m}", "FAIL", "not installed")
    try:
        import unitree_sdk2py; from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient  # noqa
        rec("1.unitree_sdk2py", "PASS", f"{Path(unitree_sdk2py.__file__).parent}")
    except Exception as e:
        rec("1.unitree_sdk2py", "FAIL", repr(e))
    man = paths.PKG_ROOT / "MANIFEST.sha256"
    if man.exists():
        bad = []
        for line in man.read_text().splitlines():
            if not line.strip(): continue
            h, rel = line.split("  ", 1); f = paths.PKG_ROOT / rel
            if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != h: bad.append(rel)
        rec("1.manifest", "PASS" if not bad else "FAIL", f"{len(man.read_text().splitlines())} files verified" if not bad else f"{len(bad)} changed/missing: {bad[:5]}")
    else:
        rec("1.manifest", "INFO", "no MANIFEST.sha256 (development repo)")


def check_frozen():
    ok = paths.sha256(paths.POLICY_ONNX) == paths.EXPECTED_SHA256["onnx"] and paths.sha256(paths.MOTION_NPZ) == paths.EXPECTED_SHA256["ssb_motion"]
    y = paths.sha256(paths.POLICY_YAML) == "1b52fe3069476ddb6947a198854faf33b6af0a25efee46d8b8859ea7dbbfebac"
    rec("2.frozen_P2_artifacts", "PASS" if (ok and y) else "FAIL", f"onnx+ssb {'==' if ok else '!='} frozen; yaml {'==' if y else '!='} frozen")


def check_onnx():
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(str(paths.POLICY_ONNX), providers=["CPUExecutionProvider"])
        fx = np.load(FIXTURE); out = np.asarray(s.run(None, {i.name: fx[i.name].astype(np.float32) for i in s.get_inputs()})[0]).reshape(-1)
        d = float(np.abs(out - fx["action"]).max()); rec("3.onnx_fixture_parity", "PASS" if d < 1e-5 else "FAIL", f"max abs diff {d:.2e} (tol 1e-5), providers CPU")
    except Exception as e:
        rec("3.onnx_fixture_parity", "FAIL", repr(e))


def check_policy(args):
    if not (REF_NPZ.exists() and REF_JSON.exists()):
        rec("4-6.reference", "FAIL", f"{REF_NPZ} missing (dev PC: tools/selfcheck.py --make-reference)"); return
    ref = json.loads(REF_JSON.read_text()); z = np.load(REF_NPZ)
    n = args.frames
    try:
        d, A, QT, T = replay((z["q"], z["dq"], z["quat"], z["gyro"]), n)
    except Exception as e:
        import traceback; rec("4-6.policy_stack", "FAIL", repr(e)); traceback.print_exc(); return
    fp, steps = fingerprint(d)
    rec("4.loader_fingerprint", "PASS" if (fp == ref["loader_fingerprint"] and steps == ref["motion_steps"] == 354) else "FAIL",
        f"{'==' if fp == ref['loader_fingerprint'] else '!='} reference ({ref['made_on']}); steps {steps}; torch {_ver('torch')} (ref {ref.get('torch')})")
    da = float(np.abs(A - z["actions"][:n]).max()); dq = float(np.abs(QT - z["q_target"][:n]).max())
    rec("5.state_replay_actions", "PASS" if da < 1e-3 else "FAIL", f"{len(A)} frames: max |action - ref| {da:.2e}, max |q_target - ref| {dq:.2e} (tol 1e-3; x86 vs aarch64 float noise ~1e-6)")
    T = T[5:]; p50, p99, mx = (float(np.percentile(T, 50)), float(np.percentile(T, 99)), float(T.max()))
    st = "PASS" if p99 <= 15 else ("WARN" if p99 <= 30 else "FAIL")
    rec("6.tick_compute_time", st, f"p50 {p50:.1f} / p99 {p99:.1f} / max {mx:.1f} ms over {len(T)} ticks (PASS p99<=15, WARN<=30; control period 20 ms, watchdog 100 ms; dev PC p50 {ref.get('dev_tick_ms_p50', 0):.1f} ms)")


def check_dds():
    try:
        import threading
        from fasttrack.link import DdsUserCtrlLink
        from fasttrack.profile import get_profile
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
        prof = get_profile("demo_walkrun")
        link = DdsUserCtrlLink("lo", 78, read_only=False, lowcmd=prof.lowcmd)         # loopback + domain 78 ONLY
        pub = ChannelPublisher("rt/lowstate", LowState_); pub.Init(); m = unitree_hg_msg_dds__LowState_(); m.mode_machine = 5
        for i in range(29): m.motor_state[i].q = 0.01 * (i + 1); m.motor_state[i].dq = -0.02 * (i + 1)
        m.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]; m.imu_state.gyroscope = [0.1, 0.2, 0.3]
        got_cmd = []
        sub = ChannelSubscriber("rt/user_lowcmd", LowCmd_); sub.Init(lambda x: got_cmd.append(x), 5)
        t_end = time.monotonic() + 3.0; ok_state = False
        while time.monotonic() < t_end and not ok_state:
            m.wireless_remote = [0, 0, 0x04] + [0] * 37; m.tick += 1; pub.Write(m); time.sleep(0.02)
            st = link.latest()
            ok_state = st is not None and abs(st.q[28] - 0.29) < 1e-6 and abs(st.dq[0] + 0.02) < 1e-6 and abs(st.gyro[2] - 0.3) < 1e-6
        from fasttrack.remote import start_pressed
        rec("7a.dds_lowstate_roundtrip", "PASS" if (ok_state and start_pressed(st.wireless)) else "FAIL", f"q/dq/gyro land in the right slots: {ok_state}; START bit decoded: {start_pressed(st.wireless) if ok_state else None}")
        link.enable_write(); q = np.linspace(-0.1, 0.1, 29); t_end = time.monotonic() + 3.0
        while time.monotonic() < t_end and not got_cmd:
            link.send(q, np.zeros(29), np.zeros(29), np.full(29, 20.0), np.full(29, 1.0)); time.sleep(0.05)
        if not got_cmd:
            rec("7b.dds_lowcmd_roundtrip", "FAIL", "no rt/user_lowcmd sample received on lo/78"); return
        c = got_cmd[-1]; fields = all(int(c.motor_cmd[i].mode) == 1 for i in range(29)) and int(c.mode_machine) == 5 and abs(c.motor_cmd[28].q - 0.1) < 1e-6
        zero = all(float(c.motor_cmd[i].kp) == 0 and float(c.motor_cmd[i].kd) == 0 for i in range(29, 35))
        try:
            from unitree_sdk2py.utils.crc import CRC
            crc_ok = int(c.crc) == CRC().Crc(c); how = "official unitree_sdk2py CRC lib"
        except Exception as e:
            crc_ok, how = None, f"official CRC lib unavailable ({e!r})"
        from fasttrack.crc import lowcmd_crc
        own = int(c.crc) == lowcmd_crc(c)
        st_ = "PASS" if (fields and zero and own and crc_ok is not False) else "FAIL"
        if st_ == "PASS" and crc_ok is None: st_ = "WARN"
        rec("7b.dds_lowcmd_roundtrip", st_, f"mode=1/mode_machine=5/q slot ok: {fields}; slots 29-34 zero: {zero}; CRC == pure-python: {own}; CRC == {how}: {crc_ok}")
        link.close()
    except Exception as e:
        import traceback; traceback.print_exc(); rec("7.dds_loopback", "FAIL", repr(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-reference", action="store_true", help="DEV PC only: write assets/expected/selfcheck_ref.*")
    ap.add_argument("--frames", type=int, default=354); ap.add_argument("--skip-dds", action="store_true")
    a = ap.parse_args()
    if a.make_reference:
        make_reference(); return 0
    check_platform(); check_frozen(); check_onnx(); check_policy(a)
    if not a.skip_dds: check_dds()
    fails = [r for r in RES if r[1] == "FAIL"]; warns = [r for r in RES if r[1] == "WARN"]
    print(f"\nSELFCHECK {'FAIL' if fails else 'PASS'}: {len(RES) - len(fails) - len(warns)} ok/info, {len(warns)} warn, {len(fails)} fail")
    sys.stdout.flush(); os._exit(1 if fails else 0)      # DDS reader threads can keep the interpreter alive


if __name__ == "__main__":
    sys.exit(main())
