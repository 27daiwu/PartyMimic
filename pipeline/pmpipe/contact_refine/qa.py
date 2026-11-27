"""contact_refine.qa - evaluate the pre-registered gate table (config.QA_GATES) on a metrics dict and write qa_verdict.json."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import GATES_VERSION, QA_GATES, VERSION, gates_digest


def _get(d, key):
    for k in key.split("."): d = d[k]
    return d


def evaluate(metrics: dict, extra: dict | None = None) -> dict:
    rows = []; ok_all = True
    for g in QA_GATES:
        try:
            v = _get(metrics, g["key"])
        except Exception as e:
            v = None
        if v is None or (isinstance(v, float) and v != v): passed = False
        elif g["op"] == "<=": passed = bool(v <= g["value"])
        elif g["op"] == ">=": passed = bool(v >= g["value"])
        else: passed = bool(v == g["value"])
        ok_all &= passed
        rows.append(dict(id=g["id"], desc=g["desc"], op=g["op"], threshold=g["value"], unit=g["unit"], value=v, passed=passed))
    notes = []
    if metrics["flight"]["n_true_flights_orig"] == 0: notes.append("原始数据中没有通过弹道一致性检验的真实腾空：flight_preserved / no_new_flight 为空真（vacuous）。")
    return dict(verdict="PASS" if ok_all else "FAIL", n_pass=sum(r["passed"] for r in rows), n_gates=len(rows), gates_version=GATES_VERSION, gates_sha256=gates_digest(), algorithm=VERSION,
                time=time.strftime("%F %T"), gates=rows, failed=[r["id"] for r in rows if not r["passed"]], notes=notes, **(extra or {}))


def write(verdict: dict, path: Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(verdict, indent=1, ensure_ascii=False, default=float))
