#!/usr/bin/env bash
# Start the PartyMimic remote-controller service. It only LISTENS until you press the chords (robot standing in WalkRun):
#   F1+F3  load the Policy + real-time checks -> READY / BLOCKED (robot does not move)    L1+A  (from READY) take over, run the motion window, hand back to WalkRun
#   START x2 anywhere: safe abort.   Every stage is announced by voice.   Logs: ./logs/   Stop: ./stop.sh
#   ./start.sh [--stage hold|clip|dance] [--iface IFACE] [--entry-fsm N] [--run-s S] [--ref running|paused] [--authority 1.0] [--ramp-s 0.3] [--end-hold-s 0.5] [--yes] [--foreground] [--no-tts]
#   --stage hold  = FIRST TEST: take over at FULL (ramped) authority, hold the first SSB frame (reference paused) for 3 s, hand back to WalkRun.
#   --stage clip  = play the first --run-s seconds of the dance (default 3 s), freeze the reference 0.5 s, hand back.
#   --sim = DEV ONLY (loopback test robot: iface lo, domain 77).   --accept-unvalidated: see config/site.conf (full authority is Sim2Sim-validated only).
#   --stage dance = the whole SSB (7.1 s) + 0.5 s freeze, hand back.   (authority < 0.9 is REFUSED: it is the unstable regime, see config/site.conf)
# Without --yes you are asked to confirm fall protection / e-stop / >=2 people; anything but YES = NOT ARMED (F1+F3 checks only, L1+A does nothing).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
BIN="${PM_BIN:-$HERE/build/partymimic}"; LOGS="${PM_LOGS:-logs}"   # PM_BIN / PM_LOGS: dev/test overrides (same variable the binary reads)
[ -x "$BIN" ] || { echo "not built yet: run ./build.sh first"; exit 1; }
[ -f build/env.sh ] && . build/env.sh
IFACE=""; ENTRY=""; RUN_S=""; REF=""; AUTH=""; RAMP=""; ENDH=""; YES=0; FG=0; NOTTS=""; UNVAL=0; SIM=0
while [ $# -gt 0 ]; do case "$1" in
  --iface) IFACE="$2"; shift;; --entry-fsm) ENTRY="$2"; shift;; --run-s) RUN_S="$2"; shift;; --ref) REF="$2"; shift;; --stage) case "$2" in hold) REF=paused; [ -n "$RUN_S" ] || RUN_S=3.0;; clip) REF=running; [ -n "$RUN_S" ] || RUN_S=3.0;; dance) REF=running; [ -n "$RUN_S" ] || RUN_S=7.1;; *) echo "--stage hold|clip|dance"; exit 2;; esac; shift;; --authority) AUTH="$2"; shift;; --ramp-s) RAMP="$2"; shift;; --end-hold-s) ENDH="$2"; shift;;
  --yes) YES=1;; --accept-unvalidated) UNVAL=1;; --sim) SIM=1;; --foreground) FG=1;; --no-tts) NOTTS="--no-tts";; -h|--help) sed -n 2,11p "$0"; exit 0;; *) echo "unknown argument $1"; exit 2;; esac; shift; done
[ $SIM -eq 1 ] && IFACE=lo
if [ -z "$IFACE" ]; then
  IFACE="$(awk '$1=="iface"&&$2!="auto"{print $2; exit}' config/site.conf 2>/dev/null || true)"
  [ -n "$IFACE" ] || IFACE="$(ip -br -4 addr 2>/dev/null | awk '/192\.168\.123\./{print $1; exit}')"
  [ -n "$IFACE" ] || { echo "no 192.168.123.x interface found: use --iface IFACE"; exit 2; }
  echo "interface: $IFACE"
fi
mkdir -p "$LOGS"
if [ -f "$LOGS/service.pid" ] && kill -0 "$(cat "$LOGS/service.pid")" 2>/dev/null; then echo "service already running (pid $(cat "$LOGS/service.pid")); ./stop.sh first"; exit 1; fi
[ -e /tmp/partymimic_cpp_ESTOP ] && echo "NOTE: e-stop latch /tmp/partymimic_cpp_ESTOP exists: motion is refused until ./ack.sh (after you checked the robot)."
if [ $YES -eq 0 ] && [ -t 0 ]; then
  echo "SAFETY: fall protection in place (gantry/strap or >=2 trained people), remote e-stop/damping in hand, floor clear 2 m."
  read -r -p "Type YES to ARM motion (anything else = NOT ARMED, checks only): " a; [ "$a" = YES ] && YES=1
fi
# P4: the full-authority control law / windows > 1 s are validated in Sim2Sim ONLY (results/p4/P4_REPORT.md). A real run needs an explicit typed acknowledgement every start.
EFF_AUTH="${AUTH:-$(awk '$1=="authority"{print $2; exit}' config/site.conf)}"; EFF_RUN="${RUN_S:-$(awk '$1=="run_s"{print $2; exit}' config/site.conf)}"
if awk -v a="${EFF_AUTH:-0.3}" -v r="${EFF_RUN:-1.0}" 'BEGIN{exit !(a>0.3+1e-9 || r>1.0+1e-9)}' && [ $UNVAL -eq 0 ]; then
  echo "NOTE: authority ${EFF_AUTH:-?} / window ${EFF_RUN:-?} s are beyond the real-robot-validated envelope (<=0.3, <=1.0 s): validated in Sim2Sim only."
  if [ $YES -eq 1 ] || [ -t 0 ]; then u=""; read -r -p "Type I-ACCEPT-UNVALIDATED to arm this run (anything else = NOT ARMED): " u || true; [ "$u" = I-ACCEPT-UNVALIDATED ] && UNVAL=1 || YES=0; fi
  [ $UNVAL -eq 1 ] || { YES=0; echo "NOT ARMED (no acknowledgement)."; }
fi
ARGS=(--iface "$IFACE"); [ -n "$ENTRY" ] && ARGS+=(--entry-fsm "$ENTRY"); [ -n "$RUN_S" ] && ARGS+=(--run-s "$RUN_S"); [ -n "$REF" ] && ARGS+=(--ref-mode "$REF"); [ -n "$AUTH" ] && ARGS+=(--authority-max "$AUTH"); [ -n "$RAMP" ] && ARGS+=(--ramp-s "$RAMP"); [ -n "$ENDH" ] && ARGS+=(--end-hold-s "$ENDH")
[ $SIM -eq 1 ] && ARGS+=(--sim); [ $UNVAL -eq 1 ] && ARGS+=(--accept-unvalidated); [ $YES -eq 1 ] && ARGS+=(--confirm-protection); [ -n "$NOTTS" ] && ARGS+=("$NOTTS")
[ $YES -eq 1 ] && echo "ARMED." || echo "NOT ARMED: F1+F3 runs the checks only; L1+A does nothing."
if [ $FG -eq 1 ]; then exec "$BIN" service "${ARGS[@]}"; fi
nohup "$BIN" service "${ARGS[@]}" >> "$LOGS/service.out" 2>&1 &
for i in $(seq 1 20); do grep -q "service starting" "$LOGS/service.log" 2>/dev/null && kill -0 "$(cat "$LOGS/service.pid" 2>/dev/null || echo 0)" 2>/dev/null && break; sleep 1; done
if ! kill -0 "$(cat "$LOGS/service.pid" 2>/dev/null || echo 0)" 2>/dev/null; then echo "service failed to start:"; tail -n 15 "$LOGS/service.out"; exit 1; fi
echo "service started (pid $(cat "$LOGS/service.pid")).  Watch: tail -f $LOGS/service.log"
# P4.1: the service is only useful with a LIVE guardian (F1+F3 is refused without it). Verify it for real and say exactly why if not (no silent restart loop).
G_OK=0
for i in $(seq 1 15); do
  if grep -q '"guardian_alive":true' "$LOGS/state.json" 2>/dev/null; then G_OK=1; break; fi
  if grep -q '"guardian_fatal":true' "$LOGS/state.json" 2>/dev/null; then break; fi
  sleep 1
done
if [ $G_OK -eq 1 ]; then
  tail -n 4 "$LOGS/service.log"; echo "guardian alive: OK.   Remote (robot in WalkRun): F1+F3 load+check | L1+A run | START x2 abort"
else
  echo "ERROR: the guardian is NOT alive after 15 s - F1+F3 will be refused (voice: 检查失败)."
  echo "  reason: $(sed -n 's/.*"guardian_why":"\([^"]*\)".*/\1/p' "$LOGS/state.json" 2>/dev/null)"
  echo "  last guardian log: $(ls -t "$LOGS"/guardian/*.log 2>/dev/null | head -1)"; tail -n 3 "$(ls -t "$LOGS"/guardian/*.log 2>/dev/null | head -1)" 2>/dev/null | sed 's/^/    /'
  echo "  The service keeps running (read-only); fix the cause, then ./stop.sh && ./start.sh again."; exit 1
fi
