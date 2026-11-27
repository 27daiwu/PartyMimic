"""Reference motion library: retargeted clips resampled to 50 Hz, with FK of tracked bodies."""

from __future__ import annotations

import glob
import os
from typing import Callable

import numpy as np
import torch

from .kinematics import Kinematics
from .robot import MotionArrays, RobotSpec
from .rotations import mat_to_6d, mm3, mv3, quat_slerp, quat_to_mat, so3_log, yaw_mat

CONTROL_DT = 0.02
BIN_FRAMES = 50  # 1 s bins for return-based reset sampling


GOAL_Q_OFFSET = 15  # index of the joint positions q(f+1) in the goal vector (first future frame: dp 3, rot 6, v 3, w 3, then q)

# --interface: what the policy is told about the reference. "joint": root motion + joint positions / velocities per goal
# frame (GOAL_Q_OFFSET layout). "sparse5": poses of five keypoints per goal frame in the reference's own current pelvis
# heading frame, pelvis position + 6D (9), head 6D (6), then left wrist, right wrist, left ankle, right ankle position
# + 6D (9 each); no joint or velocity channel.
INTERFACES = ("joint", "sparse5")
SPARSE5_FRAME = 9 + 6 + 4 * 9
# sparse5 observation block (mdp.MDP.kp_err): wrist / ankle position (4 x 3) and orientation (4 x 3, so3_log) errors and
# the head orientation error (3) of the robot against reference frame f, both relative to their own pelvis heading frame
SPARSE5_ERR = 4 * 3 + 4 * 3 + 3
# --obs_err (joint interface only): "kp" adds sparse5's keypoint-error block for the decoder
OBS_ERRS = ("none", "kp")


def sparse5_bodies(tracked: tuple[str, ...]) -> list[int]:
    """Tracked-body indices of pelvis, head, left / right wrist, left / right ankle. Neither robot has a head link: the
    head is rigid on the torso (G1 torso_link, RP1 waist_yaw_link, whose mesh carries the head)."""
    head = next(n for n in ("waist_yaw_link", "torso_link") if n in tracked)
    wrist = [n for n in tracked if "_wrist_" in n]
    names = [tracked[0], head, *(next(n for n in wrist if n.startswith(s)) for s in ("left", "right")),
             "left_ankle_roll_link", "right_ankle_roll_link"]  # fmt: skip
    return [tracked.index(n) for n in names]


class MotionLib:
    def __init__(self, files: list[str], kin: Kinematics, device: str, load_motion: Callable[[str], MotionArrays]):
        self.device = device
        self.files = list(files)
        self.names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        ps, qs, js = [], [], []
        lens = []
        for f in files:
            p, quat, j, src_fps = load_motion(f)
            p, quat, j = (torch.tensor(x, dtype=torch.float32) for x in (p, quat, j))
            # resample to 50 Hz
            T_src = p.shape[0]
            dur = (T_src - 1) / src_fps
            n = int(np.floor(dur / CONTROL_DT)) + 1
            t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * src_fps
            i0 = torch.clamp(t.floor().long(), max=T_src - 2)
            a = (t - i0.float()).clamp(0, 1)
            p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
            j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
            quat = quat_slerp(quat[i0], quat[i0 + 1], a)
            ps.append(p), qs.append(quat), js.append(j)
            lens.append(n)
        self.clip_len = torch.tensor(lens, device=device)
        self.clip_start = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.long), torch.tensor(lens[:-1], dtype=torch.long))), 0).to(device)  # dtype: one clip gives an empty (float) tensor
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(sum(lens))
        self.clip_id = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lens)]).to(device)

        root_p, root_R, root_v, root_w, jq, jqd = [], [], [], [], [], []
        for p, quat, j in zip(ps, qs, js):
            R = quat_to_mat(quat)
            v = _gradient(p, CONTROL_DT)
            qd = _gradient(j, CONTROL_DT)
            # angular velocity (world) by central differences of rotations
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
            w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
            w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
            root_p.append(p), root_R.append(R), root_v.append(v), root_w.append(w), jq.append(j), jqd.append(qd)
        cat = lambda xs: torch.cat(xs).to(device)  # noqa: E731
        self.root_p, self.root_R, self.root_v, self.root_w = cat(root_p), cat(root_R), cat(root_v), cat(root_w)
        self.q, self.qd = cat(jq), cat(jqd)
        with torch.no_grad():
            bp, bR, bv, bw = [], [], [], []
            for s in range(0, self.num_frames, 65536):
                e = min(s + 65536, self.num_frames)
                out = kin.forward(self.root_p[s:e], self.root_R[s:e], self.root_v[s:e], self.root_w[s:e], self.q[s:e], self.qd[s:e])
                bp.append(out[0]), bR.append(out[1]), bv.append(out[2]), bw.append(out[3])
            self.body_p, self.body_R, self.body_v, self.body_w = torch.cat(bp), torch.cat(bR), torch.cat(bv), torch.cat(bw)
        # per-frame static goal parts: q, qd, root velocities in the root frame
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w.unsqueeze(-1)).squeeze(-1)
        self.root_gz = self.root_R[:, 2, 2]  # -(projected gravity z) of the reference root: g_b = R^T (0,0,-1)
        # 1 s bins
        bin_clip, bin_start = [], []
        for c in range(len(lens)):
            nb = int(np.ceil(lens[c] / BIN_FRAMES))
            for b in range(nb):
                bin_clip.append(c)
                bin_start.append(int(self.clip_start[c]) + b * BIN_FRAMES)
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.zeros(self.num_frames, dtype=torch.long, device=device)
        for b in range(self.num_bins):
            c = bin_clip[b]
            s = bin_start[b]
            e = min(s + BIN_FRAMES, int(self.clip_end[c]) + 1)
            self.frame_bin[s:e] = b

    def bin_features(self) -> torch.Tensor:
        """(num_bins, 5) per 1 s bin of the reference: root height mean and std, log(1 + horizontal root speed),
        |yaw rate|, log(1 + joint speed norm)."""
        fb, nb = self.frame_bin, self.num_bins
        cnt = torch.zeros(nb, device=self.device).index_add_(0, fb, torch.ones_like(fb, dtype=torch.float32))
        mean = lambda x: torch.zeros(nb, device=self.device).index_add_(0, fb, x) / cnt  # noqa: E731
        z = self.root_p[:, 2]
        mz = mean(z)
        sz = (mean(z * z) - mz**2).clamp(min=0).sqrt()
        sp = torch.log1p(mean(self.root_v[:, :2].norm(dim=-1)))
        return torch.stack((mz, sz, sp, mean(self.root_w[:, 2].abs()), torch.log1p(mean(self.qd.norm(dim=-1)))), -1)

    def bin_clusters(self, k: int, iters: int = 50, seed: int = 0) -> torch.Tensor:
        """k-means labels (num_bins,) of the standardized bin_features."""
        x = self.bin_features()
        x = (x - x.mean(0)) / x.std(0).clamp(min=1e-6)
        g = torch.Generator(device="cpu").manual_seed(seed)
        c = x[torch.randperm(len(x), generator=g)[:k].to(x.device)]
        for _ in range(iters):
            lab = torch.cdist(x, c).argmin(1)
            for j in range(k):
                m = lab == j
                if m.any():
                    c[j] = x[m].mean(0)
        return torch.cdist(x, c).argmin(1)

    @staticmethod
    def from_dir(path: str, robot: RobotSpec, kin: Kinematics, device: str) -> MotionLib:
        files = sorted(glob.glob(os.path.join(path, robot.motion_glob)))
        if not files:  # humanoid_s0 motion library: npz clips of world body states, pelvis first
            return MotionLib(sorted(glob.glob(os.path.join(path, "*.npz"))), kin, device, load_npz)
        return MotionLib(files, kin, device, robot.load_motion)

    def set_ref_contact(self, feet: list[int], margin: float = 0.015, vmax: float = 0.5, q: float = 0.05) -> None:
        """Speed-branch reference foot contact (num_frames, len(feet)): height below the clip's own floor (the
        q-quantile of that foot's height in the clip) + margin, and horizontal speed below vmax. A fixed 4.5 cm
        threshold misses most stance frames of clips whose retargeted feet sit ~1 cm higher (LAFAN1 subject1:
        floor ~4.5 cm against ~3.5 cm for the others)."""
        z, vh = self.body_p[:, feet, 2], self.body_v[:, feet, :2].norm(dim=-1)
        floor = torch.zeros_like(z)
        for c in range(len(self.names)):
            s = self.clip_id == c
            floor[s] = z[s].quantile(q, dim=0)
        self.ref_contact = ((z < floor + margin) & (vh < vmax)).float()

    ref_conf = None  # V1.7: contact_refine labels (set_contact_labels); None = legacy labels only
    ref_contact_refine = None

    def set_contact_labels(self, per_clip: list[dict], feet: list[int], sole_loc, sole_r: float, hi_conf: float = 0.7, stable_k: int = 3,
                           zmp_lo: float = 0.02, zmp_hi: float = 0.10) -> None:
        """V1.7: attach contact_refine sidecar labels (one dict per clip in clip order: state (T,2) 0 swing / 1 stance / 2 uncertain, stance_prob, conf, zmp_margin_m, ground_mm), already
        frame-aligned with the (padded) clip. Sets, all (num_frames, 2) float unless noted:
          ref_contact_refine  hard 1 / 0 on certain stance / swing, the stance probability on uncertain frames (soft label)
          ref_conf            confidence of the decided state
          ref_hi_stance / ref_hi_swing   stance / swing with conf >= hi_conf
          ref_stable          high-confidence stance for frames t-k .. t+k within the clip (no transition nearby)
          ref_sole_h          lowest foot-sole point height of the reference [m] (tilt invariant)
          ref_up              reference foot clearly lifted (sole > 27 mm)
          ref_infeas (N,)     0..1 dynamic-infeasibility from the kinematic-ZMP margin (negative margin = ZMP outside the support polygon), 0.2 s box smoothed"""
        dev = self.device
        cat = lambda k, dt=torch.float32: torch.cat([torch.as_tensor(d[k]).to(dt) for d in per_clip]).to(dev)  # noqa: E731
        assert sum(len(d["state"]) for d in per_clip) == self.num_frames, "label frames != motion frames"
        state, prob, conf = cat("state", torch.int64), cat("stance_prob"), cat("conf")
        # label weight for --contact_conf_weight: confidence, x0.3 on uncertain frames, 0 within +-3 frames of a stance<->non-stance toggle of the foot (C1: those frames carry most label errors)
        st_b = (state == 1).float()
        near = torch.zeros_like(st_b)
        for c in range(len(self.names)):
            s0, e0 = int(self.clip_start[c]), int(self.clip_end[c]) + 1
            x = st_b[s0:e0].t().unsqueeze(0)
            tog = (x[..., 1:] != x[..., :-1]).float()
            tog = torch.nn.functional.pad(tog, (1, 0))
            near[s0:e0] = torch.nn.functional.max_pool1d(tog, 7, 1, 3).squeeze(0).t()
        self.ref_conf = conf * torch.where(state == 2, 0.3, 1.0) * (1 - near)
        self.ref_contact_refine = torch.where(state == 1, torch.ones_like(prob), torch.where(state == 0, torch.zeros_like(prob), prob))
        hs, hw = ((state == 1) & (conf >= hi_conf)).float(), ((state == 0) & (conf >= hi_conf)).float()
        self.ref_hi_stance, self.ref_hi_swing = hs, hw
        stable = torch.zeros_like(hs)
        for c in range(len(self.names)):
            s0, e0 = int(self.clip_start[c]), int(self.clip_end[c]) + 1
            x = hs[s0:e0].t().unsqueeze(0)  # (1,2,T)
            stable[s0:e0] = (-torch.nn.functional.max_pool1d(-x, 2 * stable_k + 1, 1, stable_k, ceil_mode=False)).squeeze(0).t()  # erosion with the clip edges treated as stance (+inf pad)
        self.ref_stable = stable
        # reference foot-sole height: ankle position + R sole_loc - r (lowest of the 14 points)
        loc = torch.as_tensor(sole_loc, dtype=torch.float32, device=dev)  # (2,14,3)
        fp, fR = self.body_p[:, feet], self.body_R[:, feet]  # (N,2,3), (N,2,3,3)
        z = fp[..., 2].unsqueeze(-1) + torch.einsum("nfk,fpk->nfp", fR[..., 2, :], loc)
        self.ref_sole_h = z.min(-1).values - sole_r - cat("ground_mm").unsqueeze(-1) / 1000  # over the calibrated ground curve (0 for the contact-fixed data)
        self.ref_up = (self.ref_sole_h > 0.027).float()
        zm = torch.nan_to_num(cat("zmp_margin_m"), nan=1.0)  # undefined (no support / flight) = not infeasible
        inf = ((-zm - zmp_lo) / (zmp_hi - zmp_lo)).clamp(0, 1)
        out = torch.zeros_like(inf)
        for c in range(len(self.names)):
            s0, e0 = int(self.clip_start[c]), int(self.clip_end[c]) + 1
            x = inf[s0:e0].view(1, 1, -1)
            out[s0:e0] = torch.nn.functional.avg_pool1d(torch.nn.functional.pad(x, (5, 5), mode="replicate"), 11, 1).view(-1)
        self.ref_infeas = out

    interface = "joint"
    obs_err = "none"  # --obs_err: error block of the joint interface (mdp.obs_err_dim)
    kp: list[int] | None = None  # sparse5 / --obs_err kp: tracked-body indices of the keypoints (sparse5_bodies)

    def set_interface(self, interface: str, tracked: tuple[str, ...], obs_err: str = "none") -> None:
        assert interface in INTERFACES, interface
        assert obs_err in OBS_ERRS and (obs_err == "none" or interface == "joint"), (interface, obs_err)
        self.interface, self.obs_err = interface, obs_err
        if interface == "sparse5" or obs_err == "kp":
            self.kp = sparse5_bodies(tuple(tracked))
            self.kp_p, self.kp_R = self.body_p[:, self.kp].contiguous(), self.body_R[:, self.kp].contiguous()

    def goal_dim(self, K: int) -> int:
        if self.interface == "sparse5":
            return K * SPARSE5_FRAME
        return K * (3 + 6 + 3 + 3 + 2 * self.q.shape[-1])

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Frames f+1..f+K clamped to the clip's last frame, shape (B, K)."""
        end = self.clip_end[self.clip_id[f]]
        fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Goal features from the reference only: K future frames relative to the current reference frame."""
        fut = self.future_frames(f, K)  # (B,K)
        H = yaw_mat(self.root_R[f])  # (B,3,3)
        Ht = H.transpose(-1, -2).unsqueeze(1)
        if self.interface == "sparse5":
            Hk = Ht.unsqueeze(2)
            p = mv3(Hk, self.kp_p[fut] - self.root_p[f][:, None, None])  # (B,K,6,3)
            r = mat_to_6d(mm3(Hk, self.kp_R[fut]))  # (B,K,6,6)
            feats = torch.cat((p[:, :, 0], r[:, :, 0], r[:, :, 1], torch.cat((p[:, :, 2:], r[:, :, 2:]), -1).flatten(-2)), -1)
            return feats.reshape(f.shape[0], -1)
        dp = mv3(Ht, self.root_p[fut] - self.root_p[f].unsqueeze(1))
        rot = mat_to_6d(mm3(Ht, self.root_R[fut]))
        feats = torch.cat(
            (dp, rot, self.root_v_local[fut], self.root_w_local[fut], self.q[fut], self.qd[fut] * 0.1), -1
        )  # (B,K,15+2J)
        return feats.reshape(f.shape[0], -1)

    def ref_state(self, f: torch.Tensor):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]


def load_npz(path: str):
    d = np.load(path)
    return d["body_pos_w"][:, 0], d["body_quat_w"][:, 0], d["joint_pos"], float(d["fps"].reshape(-1)[0])


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g
