#!/usr/bin/env bash
# Stop the service (a running motion is stopped: immediate hand-back to WalkRun, confirmed by GetFsmId). ./stop.sh --estop also latches the software e-stop = emergency PASSIVE channel (/tmp/partymimic_cpp_ESTOP; remove with ./ack.sh).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
[ "${1:-}" = "--estop" ] && { echo "estop $(date +%s)" > /tmp/partymimic_cpp_ESTOP; echo "e-stop latched"; }
LOGS="${PM_LOGS:-logs}"
P="$(cat "$LOGS/service.pid" 2>/dev/null || true)"
[ -n "$P" ] && kill -0 "$P" 2>/dev/null || { echo "service is not running"; exit 0; }
grep -q '"state":"RUNNING"' "$LOGS/state.json" 2>/dev/null && echo "WARNING: a run is in progress: it is stopped now (immediate hand-back to WalkRun). Keep the remote e-stop in hand."
kill -TERM "$P"; for i in $(seq 1 60); do kill -0 "$P" 2>/dev/null || { echo "service stopped"; exit 0; }; sleep 1; done
echo "service did not exit within 60 s (pid $P). Check $LOGS/service.log. NOT killing it: a runner may still be handing control back."; exit 1
