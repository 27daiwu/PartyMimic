"""V1.7 C2: foot-posture check of the ablation rollouts (uses C3's foot_posture.py): is the sole pushed toward flat-foot by the new c_sim (lowest sole point)?
usage: python -m pmpipe.v17posture diy --runs v17_A0 ... [--base-also]   -> results/diy/v17/C_train/eval/posture.json
Per nominal-class rollout (matched reference, dance span): on frames where the REFERENCE foot is stance-with-heel-only / toe-only contact and the policy foot is in contact (n_contact>0):
share of frames the policy sole is FLAT ('flattened'), heel/toe kept; policy flat share overall; median support-area ratio policy/ref; SCREW and EDGE_HOVER frame rates (policy)."""
import argparse, json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from . import foot_posture as FP, ftcompare as FC, quality as Q
from .core import Dance, read_json, write_json

_R = {}


def ref_post(p):
    if p not in _R:
        from .contact_refine import geometry as Gm
        M = Gm.load_motion(p); refined = "contactfix_info" in M["z"]
        floor = 0.0
        if not refined:  # original: calibrated ground curve of contact_refine
            from .contact_refine import pipeline as PL
            from .contact_refine.config import Params
            floor = PL.analyze_arrays(M["root_p"], M["root_q"], M["q"], M["jv"], M["fps"], Params(), log=lambda *a: None)["g"]
        _R[p] = FP.motion_posture(M["root_p"], M["root_q"], M["q"], M["fps"], floor=floor)
    return _R[p]


def one(a):
    f, refp = a
    R = ref_post(refp); ro = Q.load_rollout(f); T = len(ro["t"]); P = FP.motion_posture(ro["root_p"], ro["root_q"], ro["q"], 50.0, floor=0.0)
    s = slice(50, T - 50); m = np.minimum(len(R["morph"]), T)
    rm, pm = R["morph"][:m][s], P["morph"][:m][s]; st = R["stance"][:m][s]; ct = P["n_contact"][:m][s] > 0
    out = {}
    for name, code in (("heel", FP.HEEL), ("toe", FP.TOE)):
        sel = (rm == code) & st & ct; out[f"{name}_to_flat"] = float((pm[sel] == FP.FLAT).mean()) if sel.any() else float("nan"); out[f"{name}_kept"] = float((pm[sel] == code).mean()) if sel.any() else float("nan")
    both = ((rm == FP.HEEL) | (rm == FP.TOE)) & st & ct; out["partial_to_flat"] = float((pm[both] == FP.FLAT).mean()) if both.any() else float("nan"); out["partial_n"] = int(both.sum())
    out["pol_flat_share"] = float((pm[ct] == FP.FLAT).mean()); out["pol_heel_share"] = float((pm[ct] == FP.HEEL).mean()); out["pol_toe_share"] = float((pm[ct] == FP.TOE).mean())
    ra = R["area"][:m][s]; pa = P["area"][:m][s]; sel = (ra > 0) & (pa > 0) & st; out["area_ratio_med"] = float(np.median(pa[sel] / ra[sel])) if sel.any() else float("nan")
    A = P["A"]["flags"][:m][s]; out["screw_rate"] = float(((A & FP.F_SCREW) > 0).mean()); out["edge_hover_rate"] = float(((A & FP.F_EDGE_HOVER) > 0).mean()); out["flicker_rate"] = float(((A & FP.F_FLICKER) > 0).mean())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dance"); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--procs", type=int, default=16); a = ap.parse_args()
    d = Dance(a.dance); ev = d.dir / "v17/C_train/eval"; PO = FC.padded_npz_of(d.padded_dir); PF = FC.padded_npz_of(d.motions / f"{d.name}_contactfix_padded"); res = read_json(ev / "posture.json", {}) or {}
    jobs = {}
    for rid in a.runs:
        ref = read_json(ev / "metrics.json")[rid]["ref"]; jobs[rid] = [(str(f), str(PF if ref == "F" else PO)) for f in sorted((ev / "traj" / rid).glob("nominal*.npz"))]
    for k in ("A0", "A1"):  # V1.6 seeds (noise band) + frozen-ish anchors, ref-O
        jobs[f"V16_{k}"] = [(str(f), str(PO)) for f in sorted((d.dir / f"v16/B_train/ftc/traj/{k}/O/E1").glob("nominal*.npz"))]
    for k in ("B0", "B1"):
        jobs[f"V16_{k}"] = [(str(f), str(PF)) for f in sorted((d.dir / f"v16/B_train/ftc/traj/{k}/F/E1").glob("nominal*.npz"))]
    with Pool(a.procs) as pool:
        for k, js in jobs.items():
            if k in res: continue
            rr = pool.map(one, js, chunksize=1); res[k] = {m: float(np.nanmean([r[m] for r in rr])) for m in rr[0]}; res[k]["n"] = len(rr); write_json(ev / "posture.json", res); print(k, {m: round(v, 3) for m, v in res[k].items()}, flush=True)


if __name__ == "__main__":
    main()
