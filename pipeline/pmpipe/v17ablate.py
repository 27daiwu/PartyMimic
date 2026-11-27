"""PartyMimic V1.7 (C2) - contact-label / contact-aware-reward short ablations: `./pm v17-labels` (sidecars) and `./pm v17-ablate` (resume-style fine-tune with extra train.py flags).

Every variant = `finetune` from robust_d_v1 it40000 (resume: optimiser/EMA/iteration kept), SAME recipe, only the data directory and the V1.7 flags below differ.
Run dirs: results/<dance>/runs/v17_<variant>[_sN] (new); sidecars: results/<dance>/v17/C_train/labels/.
"""
from __future__ import annotations

from pathlib import Path

from .core import Dance, die, say

LABELS = lambda d: d.dir / "v17/C_train/labels"
DATA_O = lambda d: d.padded_dir
DATA_F = lambda d: d.motions / f"{d.name}_contactfix_padded"

# flags are appended to the recipe's train_args; {L} = label directory
REFINE = ["--contact_source=refine", "--contact_feat=sole", "--contact_conf_weight=1.0", "--contact_labels={L}"]
VARIANTS = {
    "A0": dict(data="O", flags=[], desc="control: legacy labels + ankle-height c_sim, original data (V1.6 recipe, new short run)"),
    "A1": dict(data="O", flags=REFINE, desc="refine labels + sole-height c_sim + confidence weight, ORIGINAL data (only the label/feature changes)"),
    "A2": dict(data="F", flags=REFINE, desc="refine labels + sole c_sim + confidence weight, CONTACT-FIXED data"),
    "A3": dict(data="F", flags=REFINE + ["--cr_slip=0.5", "--cr_smooth=0.5", "--cr_consist=0.25"], desc="A2 + narrow stable-stance slip kernel (b) + load-flicker smoothing (c) + stance/swing consistency (a)"),
    "A4": dict(data="F", flags=REFINE + ["--cr_slip=0.5", "--cr_smooth=0.5", "--cr_consist=0.25", "--cr_foot_xy=1.0", "--cr_root_l1=0.5", "--cr_zmp_relax=0.5"], desc="A3 + world foot-footprint / root-drift L1 (d) + ZMP-infeasible position relaxation (e)"),
}


def labels(dance: Dance, force=False):
    from .contact_refine import labels as LB
    out = LABELS(dance)
    res = []
    for padded_dir, unp, ls in ((DATA_O(dance), dance.npz, "original"), (DATA_F(dance), dance.motions / f"{dance.name}_contactfix" / f"{dance.name}_contactfix.npz", "refined")):
        pz = sorted(Path(padded_dir).glob("*.npz"))
        if len(pz) != 1 or not Path(unp).exists():
            die(f"v17-labels: need exactly one padded npz in {padded_dir} and {unp}")
        res.append(str(LB.write_sidecar(pz[0], unp, ls, out, force=force, log=lambda *a, **k: None)))
    say("[pm] sidecars:", *res)
    return res


def launch(dance: Dance, variant: str, iters: int, seed: int = 0, run_id: str | None = None, extra=(), from_ckpt="40000", base_run="robust_d_v1", **kw):
    from . import finetune as FT
    v = VARIANTS.get(variant)
    if v is None:
        die(f"unknown variant {variant}; known: {sorted(VARIANTS)}")
    L = LABELS(dance)
    if any("{L}" in f for f in v["flags"]) and not (L / f"{DATA_O(dance).name}_contact.npz").exists():
        labels(dance)
    md = DATA_O(dance) if v["data"] == "O" else DATA_F(dance)
    flags = [f.format(L=L) for f in v["flags"]] + list(extra)
    rid = run_id or f"v17_{variant}" + (f"_s{seed}" if seed else "")
    return FT.launch(dance, from_ckpt, str(md), rid, iters, seed=seed, base_run=base_run, train_args=tuple(flags), experiment=f"V1.7 C2 {variant}", note=v["desc"], **kw)
