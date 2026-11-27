"""contact_refine.labels - trainable per-frame contact labels (V1.7 C2): `<motion>_contact.npz` SIDECAR next to nothing - the motion npz is never touched.

For a (padded) training clip the sidecar holds, frame-aligned with the padded clip (50 lead-in + T + 50 lead-out; pad frames copy the edge labels, exactly like
prepare.make_padded copies the edge pose):
  stance_prob (T,2) float32   fused stance probability per foot            (foot order = [left_ankle_roll_link, right_ankle_roll_link])
  conf        (T,2) float32   confidence of the decided state (inferred support <= 0.5)
  state       (T,2) int8      0 swing / 1 stance / 2 uncertain
  inferred    (T,2) bool      support foot inferred by the physical prior (not measured)
  zmp_margin_m (T,)  float32  kinematic-ZMP margin to the stance support polygon [m], negative = ZMP outside, NaN = undefined (no support / flight)
  ground_mm   (T,)  float32   calibrated ground curve (original data) / 0 (refined data, floor z=0)
  meta (json str): motion sha256 (the clip the labels belong to), source sha256, label_set, contact_refine VERSION, params, pad, fps, payload sha256
`write_sidecar` refuses to overwrite an existing different file unless force. `load_sidecar` verifies the sha256 of the motion it is attached to.
`mirror_labels` swaps left/right (the labels of a left-right mirrored clip).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..core import now, sha256
from . import geometry as Gm, metrics as MT, pipeline as PL
from .config import VERSION, Params

PAD = 50
ARRAYS = ("stance_prob", "conf", "state", "inferred", "zmp_margin_m", "ground_mm")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def sidecar_path(motion_npz: Path, out_dir: Path | None = None) -> Path:
    motion_npz = Path(motion_npz)
    return Path(out_dir or motion_npz.parent) / f"{motion_npz.stem}_contact.npz"


def _payload_digest(d: dict) -> str:
    h = hashlib.sha256()
    for k in ARRAYS:
        a = np.ascontiguousarray(d[k]); h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(np.nan_to_num(a.astype(np.float64), nan=-9e9).tobytes())
    return h.hexdigest()


def _pad(a, n=PAD):
    return np.concatenate([np.repeat(a[:1], n, 0), a, np.repeat(a[-1:], n, 0)], 0)


def analyse_labels(unpadded_npz: Path, label_set: str, p: Params | None = None, log=print) -> dict:
    """label_set = 'original' (ground curve calibrated from the data, A0) or 'refined' (floor z=0, the contactfix output). Returns the UNPADDED label arrays."""
    p = p or Params(); M = Gm.load_motion(unpadded_npz); fps = M["fps"]
    A = PL.analyze_arrays(M["root_p"], M["root_q"], M["q"], M["jv"], fps, p, g=(np.zeros(len(M["q"])) if label_set == "refined" else None), log=log)
    I = A["ident"]; g = A["g"]
    zm = MT.zmp_margin(A["F"], g, fps, I["stance"], np.ones(len(g), bool))
    return dict(stance_prob=I["prob"].astype(np.float32), conf=I["conf"].astype(np.float32), state=I["state"].astype(np.int8), inferred=I["inferred"].astype(bool),
                zmp_margin_m=zm.astype(np.float32), ground_mm=(g * 1000).astype(np.float32), fps=fps, params=p.as_dict())


def write_sidecar(padded_npz: Path, unpadded_npz: Path, label_set: str, out_dir: Path, p: Params | None = None, force=False, log=print) -> Path:
    padded_npz, unpadded_npz = Path(padded_npz), Path(unpadded_npz)
    dst = sidecar_path(padded_npz, out_dir)
    L = analyse_labels(unpadded_npz, label_set, p, log)
    Tp = int(np.load(padded_npz, allow_pickle=True)["joint_pos"].shape[0]); T = len(L["state"])
    assert Tp == T + 2 * PAD, f"padded clip has {Tp} frames, expected {T}+2*{PAD}"
    d = {k: _pad(L[k]) for k in ARRAYS}
    meta = dict(version=VERSION, label_set=label_set, created=now(), pad=PAD, fps=L["fps"], frames_padded=Tp, feet_order=list(FEET), state_codes={"0": "swing", "1": "stance", "2": "uncertain"},
                motion_npz=str(padded_npz), motion_sha256=sha256(padded_npz), source_npz=str(unpadded_npz), source_sha256=sha256(unpadded_npz), params=L["params"], payload_sha256=_payload_digest(d),
                note="sidecar only: the motion npz bytes are unchanged; zmp_margin_m NaN where no support foot")
    if dst.exists() and not force:
        old = json.loads(str(np.load(dst, allow_pickle=True)["meta"]))
        if old.get("payload_sha256") == meta["payload_sha256"] and old.get("motion_sha256") == meta["motion_sha256"]:
            return dst
        raise FileExistsError(f"{dst} exists with different content (use force=True)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **d, meta=np.array(json.dumps(meta, default=str)))
    return dst


def load_sidecar(path: Path, motion_npz: Path | None = None, expect_frames: int | None = None) -> dict:
    z = np.load(path, allow_pickle=True); d = {k: z[k] for k in ARRAYS}; meta = json.loads(str(z["meta"]))
    assert _payload_digest(d) == meta["payload_sha256"], f"{path}: payload sha256 mismatch (file corrupted / edited)"
    if motion_npz is not None:
        assert sha256(Path(motion_npz)) == meta["motion_sha256"], f"{path}: labels belong to a different motion file ({meta['motion_npz']})"
    if expect_frames is not None:
        assert len(d["state"]) == expect_frames, f"{path}: {len(d['state'])} label frames vs {expect_frames} motion frames"
    d["meta"] = meta
    return d


def mirror_labels(d: dict) -> dict:
    """labels of the left-right mirrored clip: swap the foot axis (zmp margin / ground are mirror-invariant)."""
    out = dict(d)
    for k in ("stance_prob", "conf", "state", "inferred"):
        out[k] = np.ascontiguousarray(d[k][:, ::-1])
    return out
