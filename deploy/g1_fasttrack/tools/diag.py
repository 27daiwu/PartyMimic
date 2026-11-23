#!/usr/bin/env python3
"""READ-ONLY real-robot diagnostics for the G1 user-control deployment.

Guarantees (enforced in code + tests/test_diag_readonly.py):
  * the link is created with read_only=True -> no publisher can ever exist in this process;
  * the only RPCs issued are the whitelisted getters (7001 GetFsmId, 7002 GetFsmMode, 7003 GetBalanceMode, 7005 GetStandHeight),
    the generic server-api-version query and MotionSwitcher.CheckMode;  SwitchToUserCtrl / SwitchToInternalCtrl / SetFsmId /
    SetVelocity / SelectMode / ReleaseMode are never called (and are unreachable through Link.rpc_read);
  * the robot is never commanded; the operator may move a joint BY HAND for --probe-joint.

Collects: network interface state, rt/lowstate stream quality + header fields (version, mode_machine, mode_pr, tick), IMU/joint
sanity and limits, per-motor flags/temperature/voltage, joint-configuration hints (23-dof / locked waist), writers on
rt/user_lowcmd | rt/lowcmd | rt/arm_sdk, official FSM id/mode, RPC latency + loss, MotionSwitcher mode, lowstate CRC validity.
Result: runs/diag_<time>.json (+ .md). With --write-pass (and --entry-fsm N) it also writes runs/preflight_pass.json.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fasttrack import paths
paths.setup_sys_path(); paths.setup_offline_env()
import numpy as np

JOINTS = None


def sh(cmd, timeout=5):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:
        return f"ERR {e!r}"


def net_info(iface):
    r = {"interface": iface}
    p = Path("/sys/class/net") / iface
    if not p.exists():
        r["exists"] = False; return r
    r["exists"] = True
    for k in ("operstate", "mtu", "speed", "address", "carrier"):
        try: r[k] = (p / k).read_text().strip()
        except Exception: r[k] = None
    r["ipv4"] = [l.split()[3] for l in sh(["ip", "-4", "-o", "addr", "show", iface]).splitlines() if "inet" in l]
    return r


class Diag:
    def __init__(self, a):
        self.a = a; self.res = []; self.info = {}
    def add(self, name, ok, detail="", level="check"):
        self.res.append({"check": name, "pass": (None if ok is None else bool(ok)), "detail": detail, "level": level})
        tag = "INFO" if ok is None else ("PASS" if ok else "FAIL")
        print(f"[{tag}] {name}: {detail}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interface", required=True); ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--sim", action="store_true", help="self-test against tools/sim_host.py (interface lo, domain 77)")
    ap.add_argument("--profile", default="demo_walkrun")
    ap.add_argument("--entry-fsm", type=int, default=None, help="operator-declared fsm; required with --write-pass")
    ap.add_argument("--write-pass", action="store_true", help="also write runs/preflight_pass.json when everything passes")
    ap.add_argument("--listen-s", type=float, default=3.0); ap.add_argument("--rpc-n", type=int, default=20)
    ap.add_argument("--probe-joint", type=float, default=0.0, metavar="SECONDS", help="read-only: for SECONDS watch which joint index you move by hand")
    ap.add_argument("--out-dir", default=str(paths.PKG_ROOT / "runs"))
    a = ap.parse_args(argv)
    if a.sim:
        a.interface, a.domain = "lo", 77
    elif a.interface == "lo":
        print("loopback is the simulator: use --sim"); return 2
    D = Diag(a)
    from fasttrack.profile import get_profile, entry_check
    from fasttrack.link import DdsUserCtrlLink
    from fasttrack.guard import GuardLimits, SafetyGuard, GuardConfig
    from fasttrack.session import ControlSession, SessionConfig
    from sim2real.config.robots import get_robot_cfg
    prof = get_profile(a.profile)
    cfg = get_robot_cfg("g1"); lim = GuardLimits.from_robot_cfg(cfg); names = list(cfg.joint_names)

    # ---- 0. static
    D.add("onnx_sha256_frozen", paths.sha256(paths.POLICY_ONNX) == paths.EXPECTED_SHA256["onnx"], "P2 it1500")
    D.add("ssb_sha256_frozen", paths.sha256(paths.MOTION_NPZ) == paths.EXPECTED_SHA256["ssb_motion"], "SSB 354 frames")
    D.info["profile"] = prof.to_json()
    D.add("profile", None, f"{prof.name}: verified_on_firmware={prof.verified_on_firmware}; source={prof.source}", "info")
    # ---- 1. network
    ni = net_info(a.interface); D.info["network"] = ni
    D.add("nic_up", ni.get("exists") and ni.get("operstate") in ("up", "unknown") and ni.get("carrier") in ("1", None), json.dumps(ni))
    if not a.sim:
        D.add("nic_has_ipv4", bool(ni.get("ipv4")), str(ni.get("ipv4")))
    # ---- 2. DDS + read-only link
    try:
        link = DdsUserCtrlLink(a.interface, a.domain, read_only=True, rpc_timeout_s=1.5, lowcmd=prof.lowcmd)
    except BaseException as e:
        D.add("dds_init", False, f"could not create the DDS participant on {a.interface}: {e!r} (NIC down / no IPv4 / wrong name?)")
        class _Null:
            def close(self): pass
        return finish(D, _Null(), a, prof, names)
    D.add("link_read_only", link.read_only and link._pub is None, "no publisher exists in this process")
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    other = {}
    def counter(topic):
        box = other[topic]
        def cb(msg): box[1] += 1
        return cb
    for topic in ("rt/lowcmd", "rt/arm_sdk"):
        other[topic] = [None, 0]
        sub = ChannelSubscriber(topic, LowCmd_); sub.Init(counter(topic), 0); other[topic][0] = sub     # read-only callback counter (this process never writes)
    raw = {"state": None}
    sub_raw = ChannelSubscriber("rt/lowstate", LowState_); sub_raw.Init(lambda m: raw.__setitem__("state", m), 0)
    time.sleep(0.5)
    # ---- 3. lowstate stream
    t_end = time.monotonic() + a.listen_s; seqs = []; ticks = []
    while time.monotonic() < t_end:
        st = link.latest()
        if st is not None and (not seqs or st.seq != seqs[-1][1]):
            seqs.append((time.monotonic(), st.seq)); ticks.append(st.tick)
        time.sleep(0.002)
    st = link.latest()
    if st is None:
        D.add("lowstate_stream", False, "no rt/lowstate received (wrong NIC / domain / robot off / not on 192.168.123.x?)")
        return finish(D, link, a, prof, names)
    arr = np.array([s[0] for s in seqs]); dt = np.diff(arr) if len(arr) > 3 else np.array([1.0])
    rate = (len(arr) - 1) / max(1e-6, arr[-1] - arr[0])
    D.info["lowstate"] = {"rate_hz": rate, "dt_ms_p50": float(np.percentile(dt, 50) * 1e3), "dt_ms_p99": float(np.percentile(dt, 99) * 1e3),
                          "dt_ms_max": float(dt.max() * 1e3), "samples": len(arr)}
    D.add("lowstate_rate", rate >= 30, f"{rate:.0f} Hz; dt p50/p99/max {np.percentile(dt, 50) * 1e3:.1f}/{np.percentile(dt, 99) * 1e3:.1f}/{dt.max() * 1e3:.1f} ms (sampled by a 2 ms poll)")
    tk = np.array(ticks, dtype=np.int64); D.add("tick_monotonic", bool((np.diff(tk) >= 0).all() and tk[-1] > tk[0]), f"tick {tk[0]} -> {tk[-1]}")
    D.info["header"] = {"version": list(st.version), "mode_machine": st.mode_machine, "mode_pr": st.mode_pr, "tick": st.tick}
    D.add("lowstate_header", None, f"version={list(st.version)} mode_machine={st.mode_machine} mode_pr={st.mode_pr}  (version words are NOT a decoded firmware number; record them for OEM)", "info")
    em = prof.expected_mode_machine
    D.add("mode_machine_expected", em is None or st.mode_machine == em, f"{st.mode_machine} (profile expects {em}; demo main.cpp: 5 = G1 29dof)")
    # lowstate CRC validity with the official lib (offline check; independent of our own CRC)
    try:
        from unitree_sdk2py.utils.crc import CRC
        from unitree_sdk2py.core.channel import ChannelSubscriber as CS
        smp = raw["state"]
        if smp is not None:
            D.add("lowstate_crc_valid", int(smp.crc) == CRC().Crc(smp), f"msg.crc={int(smp.crc):#x}", "check")
            ms = smp.motor_state; imu = smp.imu_state
            D.info["motors"] = [{"i": i, "name": names[i], "mode": int(ms[i].mode), "q": float(ms[i].q), "dq": float(ms[i].dq), "tau": float(ms[i].tau_est),
                                 "temp": list(map(int, ms[i].temperature)), "vol": float(ms[i].vol), "flags": int(ms[i].motorstate)} for i in range(29)]
            D.info["unused_slots_29_34"] = [{"i": i, "mode": int(ms[i].mode), "q": float(ms[i].q), "flags": int(ms[i].motorstate)} for i in range(29, 35)]
            D.info["imu"] = {"quat": list(map(float, imu.quaternion)), "rpy": list(map(float, imu.rpy)), "gyro": list(map(float, imu.gyroscope)), "acc": list(map(float, imu.accelerometer))}
            flags = [m for m in D.info["motors"] if m["flags"] != 0]
            D.add("motor_error_flags", not flags, f"{len(flags)} motors with non-zero motorstate: {[(m['name'], m['flags']) for m in flags][:6]}")
            temps = [max(m["temp"]) for m in D.info["motors"]]; D.add("motor_temperature", max(temps) < 60, f"max {max(temps)} C (limit 60 used for this check)")
            vols = [m["vol"] for m in D.info["motors"]]; D.add("motor_voltage", None, f"min/max {min(vols):.1f}/{max(vols):.1f} V", "info")
            D.add("motors_report_enabled_mode", None, f"mode values: {sorted(set(m['mode'] for m in D.info['motors']))} (info: OEM controller state, not a check)", "info")
        else:
            D.add("lowstate_crc_valid", None, "could not take a raw sample", "info")
    except Exception as e:
        D.add("lowstate_crc_valid", None, f"skipped: {e!r}", "info")
    # ---- 4. joint configuration
    q = st.q; qv = st.dq
    D.add("q_within_hard_limits", bool(((q >= lim.lo) & (q <= lim.hi)).all()), f"min margin {float(np.minimum(q - lim.lo, lim.hi - q).min()):.3f} rad; worst joint {names[int(np.argmin(np.minimum(q - lim.lo, lim.hi - q)))]}")
    wrist = [names.index(n) for n in names if "wrist" in n]
    wrist_dead = bool(np.all(np.abs(q[wrist]) < 1e-9) and np.all(np.abs(qv[wrist]) < 1e-9))
    waist_rp = [names.index("waist_roll_joint"), names.index("waist_pitch_joint")]
    waist_dead = bool(np.all(np.abs(q[waist_rp]) < 1e-9))
    D.info["config_hints"] = {"wrist_all_exactly_zero": wrist_dead, "waist_roll_pitch_exactly_zero": waist_dead}
    D.add("full_29dof_hint", not wrist_dead, "wrist q/dq not all exactly 0 (all exactly 0 suggests a 23-dof body: NOT the P2 29-dof policy's target)")
    D.add("waist_roll_pitch_hint", not waist_dead, "waist roll/pitch not exactly 0 (exactly 0 may mean locked waist: policy commands to those joints would be ignored/invalid)")
    D.add("quiescent", float(np.abs(qv).max()) < 0.5, f"max |dq| {float(np.abs(qv).max()):.3f} rad/s")
    qn = st.quat_wxyz / np.linalg.norm(st.quat_wxyz); tilt = float(np.arccos(np.clip(1 - 2 * (qn[1] ** 2 + qn[2] ** 2), -1, 1)))
    D.add("imu_quaternion_norm", abs(float(np.linalg.norm(st.quat_wxyz)) - 1) < 0.02, f"|q|={float(np.linalg.norm(st.quat_wxyz)):.4f}")
    D.add("tilt", None, f"{tilt:.2f} rad (hanging robots may show large values depending on pose)", "info")
    from fasttrack.profile import roll_pitch
    r, p = roll_pitch(st.quat_wxyz)
    D.add("imu_rpy_vs_quaternion", max(abs(r - st.rpy[0]), abs(p - st.rpy[1])) < 0.05, f"quat->(roll,pitch)=({r:.3f},{p:.3f}) vs msg.rpy=({st.rpy[0]:.3f},{st.rpy[1]:.3f}); the demo's exit rule uses msg.rpy")
    # ---- 5. writers
    for t, (_, n) in other.items():
        D.add(f"no_writer_on_{t}", n == 0, f"{n} samples in {a.listen_s:.0f} s (any writer there means another controller or debug mode is active)")
    n0 = link.command_topic_rx_count(); time.sleep(1.0); n1 = link.command_topic_rx_count()
    D.add("no_writer_on_rt/user_lowcmd", n1 == n0, f"{n1 - n0} samples in 1 s")
    # ---- 6. RPC (whitelisted getters)
    codes = {}
    for api in sorted(link.READ_ONLY_API_IDS):
        if api in (7004,): continue
        c, d = link.rpc_read(api); codes[link.READ_ONLY_API_IDS[api]] = {"code": c, "data": (d if isinstance(d, str) else str(d))[:200]}
    D.info["rpc"] = codes
    for k, v in codes.items():
        D.add(f"rpc_{k}", None if k != "GetFsmId" else v["code"] == 0, f"code={v['code']} data={v['data']}", "check" if k == "GetFsmId" else "info")
    c, fsm = link.get_fsm(); D.info["fsm"] = fsm
    ok_e, why = entry_check(prof, fsm, a.entry_fsm)
    D.add("fsm_entry_rule", ok_e, f"profile {prof.name}: {why}   [OBSERVED fsm id = {fsm}; pass --entry-fsm {fsm} to confirm it]")
    lat = []; fail = 0
    for _ in range(a.rpc_n):
        t0 = time.perf_counter(); c2, f2 = link.get_fsm(); lat.append((time.perf_counter() - t0) * 1e3); fail += int(c2 != 0 or f2 is None); time.sleep(0.05)
    D.info["rpc_latency_ms"] = {"p50": float(np.percentile(lat, 50)), "p99": float(np.percentile(lat, 99)), "max": float(max(lat)), "fail": fail, "n": a.rpc_n}
    D.add("rpc_reliable", fail == 0, f"{a.rpc_n} GetFsmId: {fail} failed, latency p50/p99/max {np.percentile(lat, 50):.1f}/{np.percentile(lat, 99):.1f}/{max(lat):.1f} ms")
    sv = link.server_api_version(); D.info["server_api_version_sport"] = str(sv)
    D.add("sport_server_api_version", None, f"{sv} (client expects 1.0.0.0)", "info")
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        msc = MotionSwitcherClient(); msc.SetTimeout(1.5); msc.Init(); code, res = msc.CheckMode()      # CheckMode only: read-only
        D.info["motion_switcher"] = {"code": code, "result": res}
        D.add("motion_switcher_CheckMode", None, f"code={code} result={res} (name empty = no OEM motion service selected; the user-control route expects the OEM service to stay selected)", "info")
    except Exception as e:
        D.add("motion_switcher_CheckMode", None, f"unavailable: {e!r}", "info")
    # ---- 7. joint index probe (operator moves a joint by hand)
    if a.probe_joint > 0:
        print(f"PROBE: move ONE joint by hand within {a.probe_joint:.0f} s (e.g. left elbow) ...", flush=True)
        q0 = link.latest().q.copy(); best = np.zeros(29); t_end = time.monotonic() + a.probe_joint
        while time.monotonic() < t_end:
            best = np.maximum(best, np.abs(link.latest().q - q0)); time.sleep(0.02)
        i = int(np.argmax(best)); D.info["probe_joint"] = {"index": i, "name": names[i], "delta_rad": float(best[i])}
        D.add("probe_joint", None, f"largest change: index {i} = {names[i]} ({best[i]:.2f} rad). Compare with the joint you moved.", "info")
    return finish(D, link, a, prof, names)


def finish(D, link, a, prof, names):
    link.close()
    fails = [r for r in D.res if r["pass"] is False]
    ok = not fails
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    doc = {"time": time.time(), "stamp": stamp, "interface": a.interface, "profile": prof.name, "all_pass": ok, "failed": [r["check"] for r in fails],
           "checks": D.res, "info": D.info, "read_only": True}
    (out / f"diag_{stamp}.json").write_text(json.dumps(doc, indent=1, default=str))
    lines = [f"# G1 read-only diagnostic {stamp}", f"interface={a.interface} profile={prof.name} result={'PASS' if ok else 'FAIL'}", ""]
    lines += [f"- {'INFO' if r['pass'] is None else ('PASS' if r['pass'] else '**FAIL**')} `{r['check']}`: {r['detail']}" for r in D.res]
    (out / f"diag_{stamp}.md").write_text("\n".join(lines))
    print(f"\nDIAG {'PASS' if ok else 'FAIL'}  failed={doc['failed']}\nreport: {out / f'diag_{stamp}.json'}")
    if a.write_pass:
        if a.entry_fsm is None:
            print("--write-pass needs --entry-fsm N (the fsm id observed above)"); return 3
        if ok and D.info.get("fsm") == a.entry_fsm:
            name = "preflight_selftest.json" if a.sim else "preflight_pass.json"
            (out / name).write_text(json.dumps({"time": time.time(), "all_pass": True, "diag_ok": True, "interface": a.interface, "profile": prof.name,
                                                "fsm": D.info.get("fsm"), "mode_machine": D.info.get("header", {}).get("mode_machine"),
                                                "header_version": D.info.get("header", {}).get("version"), "onnx_sha256": paths.EXPECTED_SHA256["onnx"],
                                                "diag_report": f"diag_{stamp}.json"}, indent=1))
            print("wrote", out / name)
        else:
            print("preflight pass file NOT written"); return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
