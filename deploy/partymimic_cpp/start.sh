#!/usr/bin/env bash
# PartyMimic V1.6.1 - start the remote-controller service. It only LISTENS; the robot never moves until you press the chords (robot standing in the factory WalkRun):
#   F1+F3 load the policy + real-time checks -> READY (held until L1+A, NO timeout) | L1+A: voice 开始执行, take-over after the voice finished + 0.5 s (site.conf announce_wait_s)
#   START x2 abort / hand back to WalkRun | L2+B emergency PASSIVE.  After every run (normal or abnormal) the service settles back to IDLE by itself: F1+F3 / L1+A can be repeated.
#   ./start.sh                 DEFAULT = the FULL dance (mode dance), armed, ready to use. NO prompt, NO authorisation, NO confirmation of any kind (never reads stdin: works with `< /dev/null`, systemd, ssh -T).
#   ./start.sh --mode M        another mode: check | hold | exit | clip2 | clip4 | dance        ./start.sh --list   list the modes (EVERY MODE IS INDEPENDENT: any mode, any time)
#                              (mode check = read-only F1+F3 check, never armed; every other mode is armed)
#   --torque-ff : (default OFF, experimental) send the training actuator torque as feed-forward when the target is clipped at a joint limit (see README / P5_3_ANALYSIS)
#   --no-standby : do not pre-warm the runner (model + DDS are then loaded at F1+F3, ~3 s slower; fallback / A-B timing)
#   options: --iface IFACE  --entry-fsm N  --foreground  --no-tts  --dry-run (print what would be started and exit)  --sim (DEV loopback robot, never the real one)
#   expert / legacy: --stage hold|clip|dance  --run-s S  --ref running|paused  --authority A  --ramp-s S  --end-hold-s S   (explicit values leave the named mode)
#   obsolete and ignored (kept so old command lines still work): --yes --accept-unvalidated --skip-order --no-remember
# History of the runs: logs/mode_pass.jsonl (record + hint ONLY, never a start condition).  Logs: ./logs/   Stop: ./stop.sh   Status: ./status.sh   Export all logs: ./export_logs.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
chmod +x start.sh stop.sh ack.sh status.sh export_logs.sh build.sh 2>/dev/null || true   # a package unpacked by a tool that drops permission bits still works after `bash start.sh`
BIN="${PM_BIN:-$HERE/build/partymimic}"; LOGS="${PM_LOGS:-logs}"   # PM_BIN / PM_LOGS: dev/test overrides (same variable the binary reads)
MODES="${PM_MODES:-config/modes.conf}"
[ -x "$BIN" ] || { echo "not built yet: run ./build.sh first"; exit 1; }
[ -f build/env.sh ] && . build/env.sh
IFACE=""; ENTRY=""; RUN_S=""; REF=""; AUTH=""; RAMP=""; ENDH=""; FG=0; NOTTS=""; SIM=0; MODE=""; LIST=0; EXPLICIT=0; DRY=0; FF=0; NOFF=0; NOSB=0
while [ $# -gt 0 ]; do case "$1" in
  --mode) MODE="$2"; shift;; --list) LIST=1;; --dry-run) DRY=1;;
  --iface) IFACE="$2"; shift;; --entry-fsm) ENTRY="$2"; shift;; --run-s) RUN_S="$2"; EXPLICIT=1; shift;; --ref) REF="$2"; EXPLICIT=1; shift;;
  --stage) case "$2" in hold) MODE=hold;; clip) REF=running; [ -n "$RUN_S" ] || RUN_S=3.0; EXPLICIT=1;; dance) MODE=dance;; *) echo "--stage hold|clip|dance"; exit 2;; esac; shift;;
  --authority) AUTH="$2"; EXPLICIT=1; shift;; --ramp-s) RAMP="$2"; shift;; --end-hold-s) ENDH="$2"; shift;;
  --torque-ff) FF=1;; --no-torque-ff) NOFF=1;; --no-standby) NOSB=1;;
  --yes|--accept-unvalidated|--skip-order|--no-remember) echo "NOTE: $1 is obsolete and ignored (V1.6.1: nothing needs a confirmation or an authorisation).";;
  --sim) SIM=1;; --foreground) FG=1;; --no-tts) NOTTS="--no-tts";;
  -h|--help) sed -n 2,15p "$0"; exit 0;; *) echo "unknown argument $1 (./start.sh --help)"; exit 2;; esac; shift; done
mkdir -p "$LOGS"
# ---- mode table (config/modes.conf): name ref run_s evidence voice desc
[ -f "$MODES" ] || { echo "missing $MODES"; exit 1; }
mode_lines() { awk '$1 ~ /^mode_/ {sub(/^mode_/,"",$1); print}' "$MODES"; }
mode_field() { mode_lines | awk -v m="$1" -v f="$2" '$1==m {print $f; exit}'; }
last_of() { [ -f "$LOGS/mode_pass.jsonl" ] || { echo "-"; return; }; local l; l="$(grep "\"mode\":\"$1\"" "$LOGS/mode_pass.jsonl" | tail -1 || true)"; if [ -z "$l" ]; then echo "-"; elif echo "$l" | grep -q '"pass":true'; then echo "上次PASS"; else echo "上次FAIL"; fi; }
show_modes() {
  echo "  #  mode   窗口     时长  历史      说明"; local i=0
  while read -r name ref rs val voice desc; do i=$((i+1))
    local ev=""; [ "$val" = 1 ] && ev="  [真机已有证据]"
    printf "  %d  %-6s %-8s %5ss  %-9s %s%s\n" "$i" "$name" "$ref" "$rs" "$(last_of "$name")" "$desc" "$ev"
  done < <(mode_lines | awk '{d=""; for(i=6;i<=NF;i++) d=d" "$i; print $1,$2,$3,$4,$5,d}')
}
if [ $LIST -eq 1 ]; then show_modes; exit 0; fi
# No interactive test-type menu. The default is the FULL dance (explicit expert options --run-s/--ref/--authority leave the named modes).
if [ -z "$MODE" ] && [ $EXPLICIT -eq 0 ]; then MODE=dance; [ -n "$(mode_field dance 1)" ] || MODE=check; fi
M_REF=""; M_RUN=""
if [ -n "$MODE" ]; then
  [ -n "$(mode_field "$MODE" 1)" ] || { echo "unknown mode '$MODE' (./start.sh --list)"; exit 2; }
  M_REF="$(mode_field "$MODE" 2)"; M_RUN="$(mode_field "$MODE" 3)"
fi
[ $SIM -eq 1 ] && IFACE=lo
if [ -z "$IFACE" ]; then
  IFACE="$(awk '$1=="iface"&&$2!="auto"{print $2; exit}' config/site.conf 2>/dev/null || true)"
  [ -n "$IFACE" ] || IFACE="$(ip -br -4 addr 2>/dev/null | awk '/192\.168\.123\./{print $1; exit}')"
  [ -n "$IFACE" ] || { echo "no 192.168.123.x interface found: use --iface IFACE"; exit 2; }
  echo "interface: $IFACE"
fi
if [ $DRY -eq 0 ]; then
  if [ -f "$LOGS/service.pid" ] && kill -0 "$(cat "$LOGS/service.pid")" 2>/dev/null; then echo "service already running (pid $(cat "$LOGS/service.pid")); ./stop.sh first"; exit 1; fi
  [ -e /tmp/partymimic_cpp_ESTOP ] && echo "NOTE: e-stop latch /tmp/partymimic_cpp_ESTOP exists (set by ./stop.sh --estop): F1+F3 is BLOCKED until ./ack.sh removes it."
fi
# ---- arming (V1.6.1): NO authorisation, NO confirmation. Every mode is armed except `check` (read-only F1+F3 check, never armed). The remaining protections are automatic: the real-time admission at F1+F3,
#      the guardian, the runtime instability exit, START x2 and L2+B.
ARM=1; [ "$M_REF" = check ] && ARM=0
EFF_AUTH="${AUTH:-$(awk '$1=="authority"{print $2; exit}' config/site.conf)}"; EFF_RUN="${RUN_S:-${M_RUN:-$(awk '$1=="run_s"{print $2; exit}' config/site.conf)}}"
EFF_REF="${REF:-${M_REF:-$(awk '$1=="ref_mode"{print $2; exit}' config/site.conf)}}"
ARGS=(--iface "$IFACE"); [ -n "$MODE" ] && ARGS+=(--mode "$MODE"); [ -n "$ENTRY" ] && ARGS+=(--entry-fsm "$ENTRY"); [ -n "$RUN_S" ] && ARGS+=(--run-s "$RUN_S"); [ -n "$REF" ] && ARGS+=(--ref-mode "$REF"); [ -n "$AUTH" ] && ARGS+=(--authority-max "$AUTH"); [ -n "$RAMP" ] && ARGS+=(--ramp-s "$RAMP"); [ -n "$ENDH" ] && ARGS+=(--end-hold-s "$ENDH")
[ $SIM -eq 1 ] && ARGS+=(--sim); [ -n "$NOTTS" ] && ARGS+=("$NOTTS")
[ $FF -eq 1 ] && ARGS+=(--torque-ff); [ $NOFF -eq 1 ] && ARGS+=(--no-torque-ff); [ $NOSB -eq 1 ] && ARGS+=(--no-standby)
[ $ARM -eq 1 ] && echo "READY TO USE (mode ${MODE:-custom}): F1+F3 load + check -> READY (held) -> L1+A run." || echo "CHECK MODE (mode ${MODE:-custom}): F1+F3 runs the read-only check only; L1+A does nothing."
if [ $DRY -eq 1 ]; then echo "DRY-RUN: mode=${MODE:-custom} armed=$ARM ref=${EFF_REF:-?} run_s=${EFF_RUN:-?} service-args: ${ARGS[*]}"; exit 0; fi
if [ $FG -eq 1 ]; then exec "$BIN" service "${ARGS[@]}"; fi
nohup "$BIN" service "${ARGS[@]}" >> "$LOGS/service.out" 2>&1 &
for i in $(seq 1 20); do grep -q "service starting" "$LOGS/service.log" 2>/dev/null && kill -0 "$(cat "$LOGS/service.pid" 2>/dev/null || echo 0)" 2>/dev/null && break; sleep 1; done
if ! kill -0 "$(cat "$LOGS/service.pid" 2>/dev/null || echo 0)" 2>/dev/null; then echo "service failed to start:"; tail -n 15 "$LOGS/service.out"; exit 1; fi
echo "service started (pid $(cat "$LOGS/service.pid")).  Watch: tail -f $LOGS/service.log   or ./status.sh"
# the service is only useful with a LIVE guardian (F1+F3 is refused without it). Verify it for real and say exactly why if not (no silent restart loop).
G_OK=0
for i in $(seq 1 15); do
  if grep -q '"guardian_alive":true' "$LOGS/state.json" 2>/dev/null; then G_OK=1; break; fi
  if grep -q '"guardian_fatal":true' "$LOGS/state.json" 2>/dev/null; then break; fi
  sleep 1
done
if [ $G_OK -eq 1 ]; then
  tail -n 4 "$LOGS/service.log"; echo "guardian alive: OK.   Remote (robot in WalkRun): F1+F3 load+check | L1+A run | START x2 abort | L2+B emergency PASSIVE (during the run)"
else
  echo "ERROR: the guardian is NOT alive after 15 s - F1+F3 will be refused (voice: 检查失败)."
  echo "  reason: $(sed -n 's/.*"guardian_why":"\([^"]*\)".*/\1/p' "$LOGS/state.json" 2>/dev/null)"
  echo "  last guardian log: $(ls -t "$LOGS"/guardian/*.log 2>/dev/null | head -1)"; tail -n 3 "$(ls -t "$LOGS"/guardian/*.log 2>/dev/null | head -1)" 2>/dev/null | sed 's/^/    /'
  echo "  The service keeps running (read-only); fix the cause, then ./stop.sh && ./start.sh again."; exit 1
fi
