#!/usr/bin/env bash
# Start the PartyMimic remote-controller service. It only LISTENS until you press the chords (robot standing in WalkRun):
#   F1+F3  load model + real-time checks -> READY / BLOCKED      L1+A  (from READY) take over, run the SSB window, hand back to WalkRun
#   START x2 anywhere: safe abort.     Logs: ./logs/   Stop: ./stop.sh
#   ./start.sh [--iface IFACE] [--entry-fsm N] [--run-s 1.0] [--ref running|paused] [--authority 0.3] [--yes] [--foreground]
# Without --yes you are asked to confirm fall protection / e-stop / >=2 people; anything but YES = NOT ARMED (F1+F3 checks only, L1+A does nothing).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
[ -f scripts/ft_env.sh ] && . scripts/ft_env.sh
PY="${FT_PY:-$HERE/.venv/bin/python}"; [ -x "$PY" ] || { echo "not built yet: run ./build.sh first"; exit 1; }
IFACE=""; ENTRY=""; RUN_S="1.0"; REF="running"; AUTH="0.3"; YES=0; FG=0
while [ $# -gt 0 ]; do case "$1" in
  --iface) IFACE="$2"; shift;; --entry-fsm) ENTRY="$2"; shift;; --run-s) RUN_S="$2"; shift;; --ref) REF="$2"; shift;; --authority) AUTH="$2"; shift;;
  --yes) YES=1;; --foreground) FG=1;; -h|--help) sed -n 2,7p "$0"; exit 0;; *) echo "unknown argument $1"; exit 2;; esac; shift; done
if [ -z "$IFACE" ]; then IFACE="$(ip -br -4 addr 2>/dev/null | awk '/192\.168\.123\./{print $1; exit}')"; [ -n "$IFACE" ] || { echo "no 192.168.123.x interface found: use --iface IFACE"; exit 2; }; echo "interface: $IFACE"; fi
mkdir -p logs
if [ -f logs/service.pid ] && kill -0 "$(cat logs/service.pid)" 2>/dev/null; then echo "service already running (pid $(cat logs/service.pid)); ./stop.sh first"; exit 1; fi
[ -e /tmp/partymimic_g1_ESTOP ] && echo "NOTE: e-stop latch /tmp/partymimic_g1_ESTOP exists: motion is refused until ./ack.sh (after you checked the robot)."
ARM=""
if [ $YES -eq 0 ] && [ -t 0 ]; then
  echo "SAFETY: fall protection in place (gantry/strap or >=2 trained people), remote e-stop/damping in hand, floor clear 2 m."
  read -r -p "Type YES to ARM motion (anything else = NOT ARMED, checks only): " a; [ "$a" = YES ] && YES=1
fi
[ $YES -eq 1 ] && ARM="--confirm-protection"
ARGS=(--iface "$IFACE" --run-s "$RUN_S" --ref-mode "$REF" --authority-max "$AUTH" --logs "$HERE/logs" $ARM)
[ -n "$ENTRY" ] && ARGS+=(--entry-fsm "$ENTRY")
[ -n "$ARM" ] && echo "ARMED." || echo "NOT ARMED: F1+F3 runs the checks only; L1+A does nothing."
if [ $FG -eq 1 ]; then exec "$PY" -B -m fasttrack.supervisor "${ARGS[@]}"; fi
nohup "$PY" -B -m fasttrack.supervisor "${ARGS[@]}" >> logs/service.out 2>&1 &
for i in $(seq 1 15); do grep -q "service starting" logs/service.log 2>/dev/null && kill -0 "$(cat logs/service.pid 2>/dev/null || echo 0)" 2>/dev/null && break; sleep 1; done
if kill -0 "$(cat logs/service.pid 2>/dev/null || echo 0)" 2>/dev/null; then echo "service started (pid $(cat logs/service.pid)).  Watch: tail -f logs/service.log"; tail -n 4 logs/service.log
  echo "Remote (robot in WalkRun): F1+F3 load+check | L1+A run | START x2 abort"; else echo "service failed to start:"; tail -n 15 logs/service.out; exit 1; fi
