"""P4: widen a PartyMimic checkpoint's world model / critic input for the extended sim context xi of robot g1mj_p4
(+per-joint kp/kd scale, +command delay, and the world model's a_{t-1} input). New input columns start with ZERO weights, so the
widened networks reproduce the old outputs exactly; --init then fine-tunes under the new randomization.
usage: PYTHONPATH=src python tools/p4_ckpt_surgery.py <last.pt> <out.pt> [--robot g1mj_p4]"""
import argparse, torch
from partymimic.dr import XiLayout
from partymimic.mdp import inv_dim
from partymimic.robot import get_robot

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--robot", default="g1mj_p4"); a = ap.parse_args()
ck = torch.load(a.src, map_location="cpu", weights_only=False)
from dataclasses import replace
def with_payload(r):  # train.py --wrist_payload 2 --payload_p0 0.25 adds the payload slice of xi
    return replace(r, dr=replace(r.dr, wrist_payload=ck["args"]["wrist_payload"], payload_p0=ck["args"]["payload_p0"]))
old_robot, new_robot = with_payload(get_robot(ck["args"]["robot"])), with_payload(get_robot(a.robot))
J = old_robot.num_joints
xo, xn = XiLayout(old_robot, "cpu").dim, XiLayout(new_robot, "cpu").dim
add = xn - xo
print("xi dim", xo, "->", xn, "(+%d)" % add)
iv = inv_dim(J)

def widen(w, mean, var, at, n, copy_from=None):
    z = torch.zeros(w.shape[0], n, dtype=w.dtype)
    m = torch.zeros(n, dtype=mean.dtype) if copy_from is None else mean[copy_from].clone()
    v = torch.ones(n, dtype=var.dtype) if copy_from is None else var[copy_from].clone()
    return torch.cat((w[:, :at], z, w[:, at:]), 1), torch.cat((mean[:at], m, mean[at:])), torch.cat((var[:at], v, var[at:]))

# world model: [inv, a, (a_prev), xi]
wm = ck["wm"]
w, mu, var = wm["net.0.weight"], wm["in_norm.mean"], wm["in_norm.var"]
assert w.shape[1] == iv + J + xo, (w.shape, iv, J, xo)
w, mu, var = widen(w, mu, var, iv + J, J, copy_from=slice(iv, iv + J))       # a_prev: statistics of a
w, mu, var = widen(w, mu, var, iv + 2 * J + xo, add)                          # new xi columns at the end of xi
wm["net.0.weight"], wm["in_norm.mean"], wm["in_norm.var"] = w, mu, var
# critic: [inv, J + 3*n_tracked + 3 + 6, xi, goal]
cr = ck["critic"]
w, mu, var = cr["net.0.weight"], cr["norm.mean"], cr["norm.var"]
from partymimic.kinematics import Kinematics
n_tr = len(Kinematics(old_robot, "cpu").tracked_idx)
pos = iv + J + n_tr * 3 + 3 + 6 + xo
w, mu, var = widen(w, mu, var, pos, add)
cr["net.0.weight"], cr["norm.mean"], cr["norm.var"] = w, mu, var
ck["args"]["robot"] = a.robot
for k in ("opt_wm", "opt_pi", "opt_v", "ctrl"): ck.pop(k, None)
torch.save(ck, a.out); print("wrote", a.out, "wm in", wm["net.0.weight"].shape[1], "critic in", cr["net.0.weight"].shape[1])
