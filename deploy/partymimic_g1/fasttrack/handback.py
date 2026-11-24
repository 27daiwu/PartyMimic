"""Hand control back to the OEM controller and PROVE it happened.

Shared by ControlSession.finish() (in-process) and guardian.take_over() (out-of-process), so both have identical semantics.

Success (`confirmed=True`) needs ALL of
  * GetFsmId answers (code 0) with a value different from the user-control fsm, and
  * either the exit RPC returned 0, or the fsm equals the id the requested mode must produce (profile.exit_fsm_expected);
    an RPC failure with an unrelated fsm is NOT success,
  * the same holds again after `dwell_s` (no bounce back into user control).
Anything else is `confirmed=False`: the caller keeps damping, reports EXIT_UNCONFIRMED and exits non-zero.
`mode_matched` additionally says whether the OEM ended up in the mode we asked for (a confirmed hand-back into a different OEM state,
e.g. PASSIVE instead of WALKRUN, is reported, not hidden).
Retry policy: explicit rejection (3203/3204/3102) -> retry once. Timeout/unknown (e.g. 3104): never blind-retried; only if GetFsmId
still shows the USER fsm after the whole confirm window (so the request demonstrably did not take effect) one verified retry is made.
"""
from __future__ import annotations
import time
from typing import Callable, Optional

EXPLICIT_REJECT = (3203, 3204, 3102)


def handback(link, *, mode: str, user_fsm: Optional[int], expected_fsm: Optional[int], confirm_timeout_s: float = 3.0,
             dwell_s: float = 0.3, hold_unconfirmed_s: float = 30.0, poll_s: float = 0.1,
             on_wait: Optional[Callable[[], None]] = None, log=print,
             remode: Optional[Callable[[], tuple]] = None) -> dict:
    rep = {"mode_requested": mode, "expected_fsm": expected_fsm, "user_fsm": user_fsm, "rpc_calls": [], "rpc_code": None,
           "fsm_after": None, "confirmed": False, "confirmed_via": None, "mode_matched": None, "retried_once_after_explicit_rejection": False,
           "verified_retry_after_timeout": False, "t_rpc_first_sent": None, "t_confirmed": None, "fsm_polls": 0, "fsm_poll_failures": 0}

    def maybe_remode():
        """R2.1: a retry can come seconds after the first request (2 s RPC timeout) during which the robot keeps collapsing under damping:
        re-decide the mode from the CURRENT attitude before sending the retry."""
        nonlocal mode, expected_fsm
        if remode is None: return
        try:
            new_mode, new_expected = remode()
        except Exception:
            return
        if new_mode != mode:
            rep.setdefault("mode_changed_before_retry", []).append({"from": mode, "to": new_mode}); log(f"[handback] attitude changed: retry as {new_mode} instead of {mode}")
            mode, expected_fsm = new_mode, new_expected; rep["mode_requested"], rep["expected_fsm"] = mode, expected_fsm

    def wait(dt):
        t_end = time.monotonic() + dt
        while time.monotonic() < t_end:
            if on_wait: on_wait()
            time.sleep(min(0.01, max(0.0, t_end - time.monotonic())))

    def call():
        ts = time.monotonic(); rc = link.switch_to_internal(mode); tr = time.monotonic()
        rep["rpc_calls"].append({"code": rc, "t_sent": ts, "t_returned": tr})
        if rep["t_rpc_first_sent"] is None: rep["t_rpc_first_sent"] = ts
        rep["rpc_code"] = rc
        return rc

    def fsm_ok(code, fsm, rc):
        if code != 0 or fsm is None: return False
        if user_fsm is not None and fsm == user_fsm: return False
        if rc == 0: return True
        return expected_fsm is not None and fsm == expected_fsm

    def poll_until(rc, timeout):
        t_end = time.monotonic() + timeout; fsm = None
        while time.monotonic() < t_end:
            code, fsm = link.get_fsm(); rep["fsm_polls"] += 1
            if code != 0 or fsm is None: rep["fsm_poll_failures"] += 1
            if fsm_ok(code, fsm, rc):
                wait(dwell_s)                                      # must still hold after the dwell
                code2, fsm2 = link.get_fsm(); rep["fsm_polls"] += 1
                if fsm_ok(code2, fsm2, rc) and fsm2 == fsm:
                    return True, fsm2
                rep["bounce_or_flap"] = {"first": fsm, "second": fsm2}
                fsm = fsm2
            wait(poll_s)
        return False, fsm

    rc = call(); log(f"[handback] SwitchToInternalCtrl({mode}) rc={rc}")
    if rc in EXPLICIT_REJECT:
        wait(0.3); maybe_remode(); rc = call(); rep["retried_once_after_explicit_rejection"] = True
        log(f"[handback] retry after explicit rejection rc={rc}")
    ok, fsm = poll_until(rc, confirm_timeout_s)
    if not ok and rc != 0 and rc not in EXPLICIT_REJECT and not rep["retried_once_after_explicit_rejection"]:
        code, f = link.get_fsm()
        if code == 0 and f is not None and user_fsm is not None and f == user_fsm:
            rep["verified_retry_after_timeout"] = True
            log(f"[handback] rc={rc} and GetFsmId still {f}: request did not take effect -> ONE verified retry")
            maybe_remode(); rc = call(); ok, fsm = poll_until(rc, confirm_timeout_s)
    rep["fsm_after"] = fsm
    if ok:
        rep["confirmed"] = True; rep["t_confirmed"] = time.monotonic()
        rep["confirmed_via"] = "rpc_ok+fsm" if rc == 0 else "fsm_only(rpc_failed)"
        rep["mode_matched"] = (expected_fsm is None) or fsm == expected_fsm
        return rep
    log(f"[handback] !!! EXIT UNCONFIRMED rc={rc} fsm={fsm}: keep damping; remote damping / power cut; do NOT approach")
    t_end = time.monotonic() + hold_unconfirmed_s
    while time.monotonic() < t_end:                                  # keep damping, keep watching: a late execution still counts
        wait(0.2)
        code, f2 = link.get_fsm(); rep["fsm_polls"] += 1
        if fsm_ok(code, f2, rc):
            wait(dwell_s); c3, f3 = link.get_fsm()
            if fsm_ok(c3, f3, rc) and f3 == f2:
                rep.update(confirmed=True, fsm_after=f2, t_confirmed=time.monotonic(), confirmed_via="late(" + ("rpc_ok+fsm" if rc == 0 else "fsm_only") + ")",
                           mode_matched=(expected_fsm is None) or f2 == expected_fsm)
                break
    return rep
