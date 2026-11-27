#!/usr/bin/env bash
# One-screen status: service state, mode, guardian, last run result and anomaly summary, mode pass table.   ./status.sh [--follow]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
LOGS="${PM_LOGS:-logs}"
show() {
  P="$(cat "$LOGS/service.pid" 2>/dev/null || true)"
  if [ -n "$P" ] && kill -0 "$P" 2>/dev/null; then echo "service: RUNNING pid $P"; else echo "service: NOT running (./start.sh)"; fi
  if command -v jq >/dev/null 2>&1 && [ -f "$LOGS/state.json" ]; then
    jq -r '"state \(.state)  mode \(.mode)  armed \(.armed)  target \(.target)/\(.interface)  guardian_alive \(.guardian_alive)  lowstate_age \(.lowstate_age_s)s\nlast: \(.last_result) final_fsm \(.last_final_fsm) recovery_ok \(.last_recovery_ok)  settle: \(.settle_why)"' "$LOGS/state.json"
  elif [ -f "$LOGS/state.json" ]; then cat "$LOGS/state.json"; echo; fi
  L="$(ls -dt "$LOGS"/runs/*/ 2>/dev/null | head -1)"
  [ -n "$L" ] && { echo "latest run: $L"; [ -f "$L/anomaly_summary.txt" ] && sed 's/^/  /' "$L/anomaly_summary.txt" | head -14; }
  [ -f "$LOGS/mode_pass.jsonl" ] && { echo "mode records (logs/mode_pass.jsonl):"; { command -v jq >/dev/null 2>&1 && jq -r '"  \(.mode)\t\(if .pass then "PASS" else "NOT passed" end)\t\(.result)\t\(.anomaly)\t\(.run_dir)"' "$LOGS/mode_pass.jsonl" | tail -12; } || tail -n 6 "$LOGS/mode_pass.jsonl"; }
  tail -n 5 "$LOGS/service.log" 2>/dev/null | sed 's/^/  | /'
}
if [ "${1:-}" = "--follow" ]; then while true; do clear; show; sleep 1; done; else show; fi
