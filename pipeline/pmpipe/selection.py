"""PartyMimic Pipeline V1.1 - THE model-selection rule (single implementation; evald best/, post.select, compare reports all call it).

Rule "balance_first" (config eval/*.yaml: selection):
  1. eligible  : nominal survival >= nominal_survival_min  AND  robust survival >= robust_survival_min (on the scores being ranked)
  2. tier      : among the eligible, every candidate whose BALANCE score is within `balance_tolerance` of the best balance is "the same tier"
  3. winner    : inside the tier, the highest FIDELITY (tie: higher balance); the winner is removed and the procedure repeats -> an ordered list
Why a tolerance: measured noise of the balance score is ~0.13 (std over 5 evaluation seeds of one checkpoint) and ~0.4 between neighbouring
plateau checkpoints, so differences below ~1-2 points are not evidence of a better balance (V1.0 reported "balance-best it4000" from such a tie).
The raw per-score maxima (best_balance / best_fidelity / best_combined) are kept only as DIAGNOSTICS, never as the selection.
Scores must come from the SAME evaluator version and grid (periodic grid for the daemon, hard grid for the final decision).
"""
from __future__ import annotations


def eligible(scores: dict, nominal_survival: float, el: dict) -> bool:
    return nominal_survival >= el["nominal_survival_min"] and scores["survival_robust"] >= el["robust_survival_min"]


def rank_scores(items: dict[int, dict], sel: dict) -> list[int]:
    """items: iteration -> dict(scores=..., nominal_survival=...). Returns the ordered eligible iterations (best first). Iteration 0 never competes."""
    el = sel["eligible"]
    left = {it: v for it, v in items.items() if it > 0 and eligible(v["scores"], v["nominal_survival"], el)}
    order = []
    while left:
        bmax = max(v["scores"]["balance"] for v in left.values())
        tier = {it: v for it, v in left.items() if v["scores"]["balance"] >= bmax - sel["balance_tolerance"]}
        win = max(tier, key=lambda it: (tier[it]["scores"]["fidelity"], tier[it]["scores"]["balance"]))
        order.append(win)
        left.pop(win)
    return order


def explain(items: dict[int, dict], sel: dict) -> dict:
    """tier structure of the FIRST decision: who is in the top balance tier, who wins and by what margin (for reports)."""
    el = sel["eligible"]
    pool = {it: v for it, v in items.items() if it > 0 and eligible(v["scores"], v["nominal_survival"], el)}
    if not pool:
        return dict(eligible=[], tier=[], winner=None)
    bmax = max(v["scores"]["balance"] for v in pool.values())
    tier = sorted(it for it, v in pool.items() if v["scores"]["balance"] >= bmax - sel["balance_tolerance"])
    win = rank_scores(items, sel)[0]
    return dict(eligible=sorted(pool), best_balance=bmax, tier=tier, winner=win, tier_balance_range=[min(pool[i]["scores"]["balance"] for i in tier), max(pool[i]["scores"]["balance"] for i in tier)],
                winner_fidelity=pool[win]["scores"]["fidelity"], tolerance=sel["balance_tolerance"])
