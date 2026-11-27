#!/usr/bin/env bash
# One-key export of ALL test logs into ONE archive:  ./export_logs.sh [OUT_DIR] [--light]        (never connects to the robot; safe to run while the service is idle)
#   result: <OUT_DIR>/partymimic_logs_<host>_<time>.tar.gz  (+ .sha256)   default OUT_DIR = ./export
#   contains: logs/runs/* (every test run: trace.csv, obs.bin, state_stream.bin, events.jsonl, safety_clips.csv, exit_frames.csv, manifest.json, run_report.json, anomaly_summary.*),
#             logs/guardian/* (take-over logs), service.log, service_events.jsonl, keys.log, mode_pass.jsonl, state.json, a copy of config/ and assets/SHA256SUMS, INDEX.txt (one line per run) and system info
#   --light  leaves out the big binary streams (obs.bin, state_stream.bin) for a quick look; the default keeps EVERYTHING (needed for replay / Sim2Sim initial states)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
LOGS="${PM_LOGS:-logs}"; OUT="export"; LIGHT=0
for a in "$@"; do case "$a" in --light) LIGHT=1;; -h|--help) sed -n 2,8p "$0"; exit 0;; *) OUT="$a";; esac; done
[ -d "$LOGS" ] || { echo "no $LOGS directory: nothing to export"; exit 1; }
mkdir -p "$OUT"; TS="$(date +%Y%m%d_%H%M%S)"; NAME="partymimic_logs_$(hostname)_$TS"; ST="$(mktemp -d)"; D="$ST/$NAME"; mkdir -p "$D"
BIN="${PM_BIN:-$HERE/build/partymimic}"
# 1) make sure every run has a finalized report (a crashed run is salvaged here too) - offline, no DDS
if [ -x "$BIN" ]; then
  [ -f build/env.sh ] && . build/env.sh
  for r in "$LOGS"/runs/*/; do [ -d "$r" ] || continue
    [ -f "$r/anomaly_summary.txt" ] && grep -q '"timing"' "$r/run_report.json" 2>/dev/null && continue
    "$BIN" logtool finalize "$r" >/dev/null 2>&1 && echo "finalized $r"
  done
fi
# 2) index
{ printf "%-17s %-8s %-24s %-8s %-6s %s\n" run mode result anomaly rows dir
  for r in "$LOGS"/runs/*/; do [ -d "$r" ] || continue; n="$(basename "$r")"
    if command -v jq >/dev/null 2>&1 && [ -f "$r/run_report.json" ]; then
      printf "%-17s %-8s %-24s %-8s %-6s %s\n" "$n" "$(jq -r '.mode // "-"' "$r/run_report.json")" "$(jq -r '.result // "-"' "$r/run_report.json")" "$(jq -r '.anomaly.level // "-"' "$r/run_report.json")" "$(jq -r '.logging.rows.trace // "-"' "$r/run_report.json")" "$r"
    else printf "%-17s %-8s %-24s %-8s %-6s %s\n" "$n" "-" "(no run_report.json)" "-" "-" "$r"; fi
  done; } > "$D/INDEX.txt"
# 3) copy
mkdir -p "$D/logs"
if [ $LIGHT -eq 1 ]; then ( cd "$LOGS" && tar --exclude='obs.bin' --exclude='state_stream.bin' --exclude='latest' -cf - . ) | ( cd "$D/logs" && tar -xf - )
else ( cd "$LOGS" && tar --exclude='latest' -cf - . ) | ( cd "$D/logs" && tar -xf - ); fi
mkdir -p "$D/config" && cp -p config/* "$D/config/" 2>/dev/null; cp -p assets/SHA256SUMS "$D/config/assets_SHA256SUMS" 2>/dev/null
{ echo "exported $(date -Is) by $(whoami)@$(hostname)  light=$LIGHT"; uname -a; df -h . | tail -1; echo; [ -x "$BIN" ] && sha256sum "$BIN"; sha256sum assets/policy.onnx assets/policy_params.txt assets/motion.bin 2>/dev/null; ls "$LOGS"/runs 2>/dev/null | wc -l | sed 's/$/ runs/'; } > "$D/SYSTEM.txt"
( cd "$D" && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256 )
tar -C "$ST" -czf "$OUT/$NAME.tar.gz" "$NAME" && ( cd "$OUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" )
rm -rf "$ST"
echo "exported: $OUT/$NAME.tar.gz  ($(du -h "$OUT/$NAME.tar.gz" | cut -f1), $(ls "$LOGS"/runs 2>/dev/null | wc -l) runs)"; cat "$OUT/$NAME.tar.gz.sha256"
echo "copy to the dev PC:  scp $OUT/$NAME.tar.gz* <user>@<pc>:~/   then:  python3 deploy/partymimic_cpp/analysis/pm_analyze.py <unpacked>/logs/runs/<run>"
