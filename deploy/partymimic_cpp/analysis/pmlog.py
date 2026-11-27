"""PartyMimic P5.1 run-log reader (dev PC only, numpy). One RunLog = one logs/runs/<time>/ directory (or a guardian take-over directory).

    from pmlog import RunLog
    r = RunLog("logs/runs/20261125_134911")
    r.trace["q"] (T,29) ... r.obs (T,obs_dim) ... r.state (S,) structured array ... r.events (list of dicts) ... r.manifest (dict) ... r.report (dict)

All times are mono() seconds (steady clock of the robot computer); r.wall(t_mono) converts with manifest wall_minus_mono. Formats: LOGGING.md.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

N = 29
STATE_DT_CACHE: Dict[int, np.dtype] = {}


def _read_bin(path: str, magic: bytes):
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or not head.startswith(magic):
            raise ValueError(f"{path}: bad magic {head[:6]!r}")
        jl = int.from_bytes(head[8:12], "little")
        meta = json.loads(f.read(jl).decode())
        body = f.read()
    return meta, body


def state_dtype() -> np.dtype:
    return np.dtype([("t", "<f8"), ("seq", "<u8"), ("tick", "<u4"), ("mode_machine", "<i4"), ("q", "<f4", N), ("dq", "<f4", N), ("tau", "<f4", N),
                     ("quat", "<f4", 4), ("gyro", "<f4", 3), ("rpy", "<f4", 3), ("wireless", "u1", 40)])


def load_state_stream(path: str) -> np.ndarray:
    meta, body = _read_bin(path, b"PMST1")
    dt = state_dtype()
    assert dt.itemsize == meta["record_bytes"], (dt.itemsize, meta)
    n = len(body) // dt.itemsize
    return np.frombuffer(body[: n * dt.itemsize], dtype=dt)


def load_obs(path: str):
    meta, body = _read_bin(path, b"PMOB1")
    d = int(meta["obs_dim"])
    dt = np.dtype([("n", "<u4"), ("t", "<f8"), ("obs", "<f4", d)])
    n = len(body) // dt.itemsize
    return meta, np.frombuffer(body[: n * dt.itemsize], dtype=dt)


def load_csv(path: str):
    with open(path, newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        rows = [r for r in rd if len(r) == len(hdr)]
    return hdr, rows


def load_events(path: str) -> List[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _group(hdr, rows, prefix):
    idx = [i for i, h in enumerate(hdr) if h.startswith(prefix + ":")]
    names = [hdr[i].split(":", 1)[1] for i in idx]
    if not idx:
        return None, []
    a = np.array([[float(r[i]) if r[i] not in ("nan", "-nan") else np.nan for i in idx] for r in rows], dtype=np.float64)
    return a, names


@dataclass
class RunLog:
    path: str
    manifest: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    events: List[dict] = field(default_factory=list)
    trace: Dict[str, np.ndarray] = field(default_factory=dict)
    joint_names: List[str] = field(default_factory=list)
    obs: Optional[np.ndarray] = None
    obs_meta: dict = field(default_factory=dict)
    state: Optional[np.ndarray] = None
    clips_hdr: List[str] = field(default_factory=list)
    clips: List[List[str]] = field(default_factory=list)
    frames: Dict[str, np.ndarray] = field(default_factory=dict)
    frame_kind: List[str] = field(default_factory=list)

    def __post_init__(self):
        p = self.path
        rj = lambda n: json.load(open(os.path.join(p, n))) if os.path.exists(os.path.join(p, n)) else {}
        self.manifest = rj("manifest.json")
        self.report = rj("run_report.json")
        self.events = load_events(os.path.join(p, "events.jsonl"))
        if os.path.exists(os.path.join(p, "trace.csv")):
            hdr, rows = load_csv(os.path.join(p, "trace.csv"))
            col = lambda name: np.array([float(r[hdr.index(name)]) for r in rows]) if name in hdr and rows else np.zeros(0)
            for g in ("q", "dq", "tau_est", "action", "action_prev", "q_policy", "q_pd", "q_cmd", "kp", "kd", "tau_pd", "ref_jp", "ref_jv"):
                a, names = _group(hdr, rows, g)
                if a is not None:
                    self.trace[g] = a
                    if g == "q":
                        self.joint_names = names
            for c in ("t", "n", "t_mono", "dt_ms", "infer_ms", "step_ms", "sync_wait_ms", "send_us", "state_age_ms", "state_age_send_ms", "dds_loop_ms", "seq", "seq_delta", "tick", "tick_delta", "flags",
                      "ls_rx", "ls_lost", "ls_gap_ms", "clip_n", "clip_mask", "tilt", "authority", "ssb_frame", "ref_frame", "arrival", "roll", "pitch", "yaw", "quat_w", "quat_x", "quat_y", "quat_z",
                      "gyro_x", "gyro_y", "gyro_z", "ref_rq_w", "ref_rq_x", "ref_rq_y", "ref_rq_z", "ref_rp_x", "ref_rp_y", "ref_rp_z", "dance_frame", "dance_t", "seg"):
                if c in hdr:
                    self.trace[c] = col(c)
        self._derive_dance_axis()
        if os.path.exists(os.path.join(p, "obs.bin")) and os.path.getsize(os.path.join(p, "obs.bin")) > 12:
            self.obs_meta, self.obs = load_obs(os.path.join(p, "obs.bin"))
        if os.path.exists(os.path.join(p, "state_stream.bin")) and os.path.getsize(os.path.join(p, "state_stream.bin")) > 12:
            self.state = load_state_stream(os.path.join(p, "state_stream.bin"))
        if os.path.exists(os.path.join(p, "safety_clips.csv")):
            self.clips_hdr, self.clips = load_csv(os.path.join(p, "safety_clips.csv"))
        if os.path.exists(os.path.join(p, "exit_frames.csv")):
            hdr, rows = load_csv(os.path.join(p, "exit_frames.csv"))
            if rows:
                self.frame_kind = [r[hdr.index("kind")] for r in rows]
                self.frames["t_mono"] = np.array([float(r[0]) for r in rows])
                for g in ("q_meas", "q_cmd", "kp", "kd"):
                    a, _ = _group(hdr, rows, g)
                    if a is not None:
                        self.frames[g] = a

    # ---- V1.6 dance time axis.  trace `ssb_frame` is NOT the dance frame: it indexes the REPLAYED clip = lead_in_frames of frozen first frame + the dance. dance_frame = ssb_frame - lead_in_frames
    #      (negative while the lead-in is frozen; 0 = the first frame of the dance = SSB first frame), dance_t = dance_frame / fps, seg = 0 lead-in | 1 dance | 2 end-hold | 3 paused reference.
    #      V1.6 logs carry the columns; for older logs they are derived here from ssb_frame + manifest (lead_in_frames, motion_frames, ref_mode). self.dance_axis_derived tells which.
    def _derive_dance_axis(self):
        T = self.trace
        self.dance_axis_derived = False
        if "ssb_frame" not in T or not len(T["ssb_frame"]) or ("dance_frame" in T and "seg" in T):
            return
        li, mf = self.lead_in_frames, self.motion_frames
        f = T["ssb_frame"].astype(int)
        T["dance_frame"] = (f - li).astype(float)
        T["dance_t"] = T["dance_frame"] / self.fps
        seg = np.ones(len(f), dtype=int)
        if self.manifest.get("ref_mode") == "paused":
            seg[:] = 3
        else:
            seg[f < li] = 0
            if mf > 0:
                last = li + mf - 1
                hit = np.where(f >= last)[0]
                if len(hit):
                    seg[hit[1:]] = 2  # the tick that FIRST reaches the last frame is the last dance frame; the ticks after it are the held frame
        T["seg"] = seg.astype(float)
        self.dance_axis_derived = True

    @property
    def lead_in_frames(self) -> int:
        return int(self.manifest.get("lead_in_frames", 0) or 0)

    @property
    def motion_frames(self) -> int:
        return int(self.manifest.get("motion_frames", 0) or 0)

    @property
    def fps(self) -> float:
        return float(self.manifest.get("motion_fps", 50.0) or 50.0)

    SEG_NAMES = {0: "lead_in", 1: "dance", 2: "end_hold", 3: "paused_reference"}

    def seg_mask(self, name: str, exclude_first_tick: bool = True) -> np.ndarray:
        """boolean mask over trace rows for 'lead_in' | 'dance' | 'end_hold' | 'paused_reference' | 'takeover_first_tick'."""
        n = self.n_ticks
        if name == "takeover_first_tick":
            m = np.zeros(n, dtype=bool)
            if n:
                m[0] = True
            return m
        ids = {v: k for k, v in self.SEG_NAMES.items()}
        m = (self.trace["seg"] == ids[name]) if "seg" in self.trace and n else np.zeros(n, dtype=bool)
        if exclude_first_tick and n:
            m = m.copy()
            m[0] = False
        return m

    # ---- helpers
    @property
    def n_ticks(self) -> int:
        """control ticks in the trace INCLUDING the frozen lead-in and the end-hold (the old, ambiguous meaning); see n_ticks_dance"""
        return len(self.trace.get("t", []))

    @property
    def n_ticks_dance(self) -> int:
        return int(self.seg_mask("dance").sum())

    def wall(self, t_mono):
        return t_mono + float(self.manifest.get("wall_minus_mono", 0.0))

    def ev(self, kind: str) -> List[dict]:
        return [e for e in self.events if e.get("kind") == kind]

    def first_ev(self, kind: str) -> Optional[dict]:
        l = self.ev(kind)
        return l[0] if l else None

    def vec(self, key: str) -> np.ndarray:
        return np.array(self.manifest[key], dtype=np.float64)

    def t_takeover(self) -> float:
        """mono() time of the first control tick (the take-over), falling back to the user_control_confirmed event."""
        if "t_mono" in self.trace and len(self.trace["t_mono"]):
            return float(self.trace["t_mono"][0])
        e = self.first_ev("user_control_confirmed")
        return float(e["t_mono"]) if e else float("nan")

    def t_exit(self) -> float:
        e = self.first_ev("exit_submit") or self.first_ev("EXIT_TRIGGER") or self.first_ev("take_over")
        return float(e["t_mono"]) if e else float("nan")
