#!/usr/bin/env bash
# Stop the service (a running motion is stopped: damping + hand-back).  ./stop.sh --estop also latches the software e-stop (/tmp/partymimic_g1_ESTOP; clear with ./ack.sh).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
[ "${1:-}" = "--estop" ] && { echo "estop $(date +%s)" > /tmp/partymimic_g1_ESTOP; echo "e-stop latched"; }
P="$(cat logs/service.pid 2>/dev/null || true)"
[ -n "$P" ] && kill -0 "$P" 2>/dev/null || { echo "service is not running"; exit 0; }
grep -q '"state": "RUNNING"' logs/state.json 2>/dev/null && echo "WARNING: a run is in progress: it is stopped now (damping + hand-back). Keep the remote e-stop in hand."
kill -TERM "$P"; for i in $(seq 1 40); do kill -0 "$P" 2>/dev/null || { echo "service stopped"; exit 0; }; sleep 1; done
echo "service did not exit within 40 s (pid $P). Check logs/service.log. NOT killing it: a runner may still be handing control back."; exit 1
