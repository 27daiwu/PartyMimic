"""Streaming run logs (flushed per line, so a crash / kill -9 keeps everything up to the last tick).

Every run writes into logs/runs/<time>/ :
  trace.csv      one row per 50 Hz tick: time, phase, IMU (quat, gyro, rpy), joint q/dq/tau_est, Policy action + q_policy, commanded q_cmd,
                 authority, inference time, loop time, state age, SSB frame, remote key bytes
  events.jsonl   control-authority transitions and faults (lock, SwitchToUserCtrl, user control confirmed, EMERGENCY_DAMP, SwitchToInternalCtrl, released ...)
  console.log    human readable log
  ready.json     the read-only snapshot at READY (state, fsm, writers, every admission item)
  run_report.json  result, exit reason, fault, hand-back report, guard statistics
"""
from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np


class JsonlLog:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def write(self, **kw):
        self.f.write(json.dumps({"wall": round(time.time(), 4), **kw}, default=_jd) + "\n")

    def close(self):
        try: self.f.close()
        except Exception: pass


def _jd(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (bytes, bytearray)): return o.hex()
    return str(o)


class TraceCsv:
    SCALARS = ["t", "phase", "ssb_frame", "authority", "infer_ms", "step_ms", "state_age_ms", "tick", "keys_b2", "keys_b3",
               "quat_w", "quat_x", "quat_y", "quat_z", "gyro_x", "gyro_y", "gyro_z", "roll", "pitch", "yaw"]
    GROUPS = ["q", "dq", "tau_est", "action", "q_policy", "q_cmd"]

    def __init__(self, path, joint_names):
        self.f = open(path, "w", buffering=1)
        cols = list(self.SCALARS)
        for g in self.GROUPS:
            cols += [f"{g}:{n}" for n in joint_names]
        self.f.write(",".join(cols) + "\n")

    def row(self, scalars: dict, **groups):
        parts = [_fmt(scalars.get(k)) for k in self.SCALARS]
        for g in self.GROUPS:
            v = groups.get(g)
            parts += [_fmt(x) for x in v] if v is not None else [""] * 29
        self.f.write(",".join(parts) + "\n")

    def close(self):
        try: self.f.close()
        except Exception: pass


def _fmt(x):
    if x is None: return ""
    if isinstance(x, str): return x
    if isinstance(x, (int, np.integer)): return str(int(x))
    return f"{float(x):.6g}"
