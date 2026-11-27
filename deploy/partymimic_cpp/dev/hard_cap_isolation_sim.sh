#!/usr/bin/env bash
# DEV ONLY (never shipped). V1.7 C3: HARD_CAP-only isolation scenarios on the loopback robot (lo, DDS domain 77). Same helpers as exit_paths.sh (copied), serialised on /tmp/pm_exit_paths.lock (WAITS instead of refusing).
#   usage: dev/hard_cap_isolation_sim.sh <build_dir>   (never edits config/site.conf: the isolated site.conf lives in a /tmp copy of the package dir, assets symlinked)
#   needs jq. Exit policy under test: EXIT_POLICY.md (WALKRUN at once / PASSIVE for fallen / emergency; confirmation by GetFsmId + stable window).
set -u
# P4 hardening: (1) only ONE instance may run (the scenarios share /tmp paths and DDS domain 77); (2) NO `pkill -f` on command-line patterns (it also killed the invoking
# shell whenever its command line contained "partymimic service ..."): test processes are found by exact process name AND resolved executable path of THIS build dir.
exec 9>/tmp/pm_exit_paths.lock; echo "waiting for /tmp/pm_exit_paths.lock ..."; flock -w 1500 9 || { echo "lock busy for 25 min"; exit 3; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; B="${1:?build dir}"; shift || true
export PM_ROOT="${PM_ROOT_OVERRIDE:-$HERE}"; EXE="$B/partymimic"; SIM="$B/dev/sim_robot"   # PM_ROOT_OVERRIDE: run the scenarios on another package dir (assets/config), e.g. the DIY package
KEY_LOAD=0xC0; KEY_GO=0x102; KEY_START=0x4; KEY_NONE=0x0
PASS=0; FAILN=0
ok() { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad() { echo "  [FAIL] $1"; FAILN=$((FAILN+1)); }
check() { if ( eval "$2" ); then ok "$1"; else bad "$1  ($2)"; fi; }
SD=""; LG=""; SIMPID=""; SVCPID=""
setkeys() { echo "$1" > "$SD/keys"; }
press() { setkeys "$1"; sleep "${2:-0.5}"; setkeys $KEY_NONE; sleep 0.4; }
dclick() { setkeys $KEY_START; sleep 0.06; setkeys $KEY_NONE; sleep 0.08; setkeys $KEY_START; sleep 0.06; setkeys $KEY_NONE; }
waitfor() { local f="$1" pat="$2" t="${3:-20}"; for _ in $(seq 1 $((t*10))); do grep -q -- "$pat" "$f" 2>/dev/null && return 0; sleep 0.1; done; return 1; }
kill_test_procs() {   # kill only processes whose executable is exactly $EXE or $SIM of this run (never matches the caller's shell / unrelated commands)
  local sig="${1:--9}" p exe
  for p in $(pgrep -x partymimic; pgrep -x sim_robot); do exe=$(readlink -f /proc/$p/exe 2>/dev/null); { [ "$exe" = "$(readlink -f "$EXE")" ] || [ "$exe" = "$(readlink -f "$SIM")" ]; } && kill "$sig" "$p" 2>/dev/null; done; return 0; }
cleanup_procs() { kill_test_procs -9; sleep 0.5; }
start_env() {   # name, sim args..., (service args after --)
  kill_test_procs -9; sleep 0.5
  cleanup_procs
  local name="$1"; shift; SD="/tmp/pm_sim_$name"; LG="/tmp/pm_logs_$name"
  rm -rf "$SD" "$LG" /tmp/partymimic_cpp_ESTOP /tmp/partymimic_cpp_control.lock /tmp/partymimic_cpp_guardian.alive; mkdir -p "$SD" "$LG"; echo 0 > "$SD/keys"
  local simargs=() svcargs=() sw=0
  for a in "$@"; do if [ "$a" = "--" ]; then sw=1; elif [ $sw -eq 0 ]; then simargs+=("$a"); else svcargs+=("$a"); fi; done
  "$SIM" --dir "$SD" "${simargs[@]}" > "$SD/sim.out" 2>&1 & SIMPID=$!
  sleep 1.0
  # P5.2: the service itself applies the motion-authorisation gate (also in sim): scenarios that run beyond the validated hold envelope pass the explicit authorisation, like start.sh does
  PM_LOGS="$LG" "$EXE" service --sim --accept-unvalidated "${svcargs[@]}" > "$LG/service.out" 2>&1 & SVCPID=$!
  waitfor "$LG/service.log" "guardian started" 10 && waitfor "$LG/service.log" "service starting" 10
  sleep 3.0   # guardian reads its entry fsm (1 s) and the heartbeat must be fresh
}
stop_env() {
  kill -TERM "$SVCPID" 2>/dev/null; for _ in $(seq 1 100); do kill -0 "$SVCPID" 2>/dev/null || break; sleep 0.1; done
  kill -9 "$SVCPID" 2>/dev/null; kill_test_procs -TERM
  kill -TERM "$SIMPID" 2>/dev/null; sleep 0.5; kill -9 "$SIMPID" 2>/dev/null; wait 2>/dev/null; rm -f /tmp/partymimic_cpp_guardian.alive
}
waitn() { local f="$1" pat="$2" n="$3" t="${4:-20}"; for _ in $(seq 1 $((t*10))); do [ "$(grep -c -- "$pat" "$f" 2>/dev/null)" -ge "$n" ] && return 0; sleep 0.1; done; return 1; }
sim_has() { grep -q -- "$1" "$SD/sim_events.log"; }
log_has() { grep -q -- "$1" "$LG/service.log"; }
latest_report() { ls -d "$LG"/runs/*/ 2>/dev/null | tail -1; }
rj() { jq -r "$1" "$(latest_report)/run_report.json" 2>/dev/null; }          # rj '.summary.result'
lt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0 < b+0 && a+0 >= 0)}'; }      # lt value limit  (value must be >= 0)
sim_delta() { awk -v a="$1" -v b="$2" '$0 ~ a && !ta {ta=$1} $0 ~ b && ta && !tb {tb=$1} END{ if (ta && tb) printf "%.1f", (tb-ta)*1000; else print -1 }' "$SD/sim_events.log"; }   # ms between the first lines matching a and b
sim_count() { grep -c -- "$1" "$SD/sim_events.log" 2>/dev/null; }
guardian_log() { ls -t "$LG"/guardian/*.log 2>/dev/null | head -1; }
# common prologue: start env, load, GO, wait until the policy is driving the robot. args: sim args... -- service args...
go_run() { local name="$1"; shift; start_env "$name" "$@"; press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5; waitfor "$SD/sim_events.log" "first stiff user_lowcmd frame" 20; sleep 0.7; }
lat() { echo "  [info] detect->rpc_sent = $(rj '.exit_detail.detect_to_rpc_sent_ms') ms, state-event->rpc_sent = $(rj '.exit_detail.event_to_rpc_sent_ms') ms, rpc latency = $(rj '.exit_detail.rpc_latency_ms') ms"; }



# ---- isolated package copy: assets symlinked, config copied and ONLY this copy's site.conf is changed (the shipped site.conf is never touched)
iso_root() {   # iso_root <name> <extra site.conf lines>
  local d=/tmp/pm_c3_root_$1; rm -rf "$d"; mkdir -p "$d"; for f in "$HERE"/*; do [ "$(basename "$f")" = config ] || ln -s "$f" "$d/$(basename "$f")"; done
  cp -r "$HERE/config" "$d/config"; printf '%s\n' "$2" >> "$d/config/site.conf"; echo "$d"
}
# HARD_CAP + FALLEN only (site.conf-valid): REF_DEV can only fire above the cap (ref_dev_min_tilt 0.70), FIXED_LIMIT / soft rule pushed to 0.699, gyro rule at its max (10 rad/s), tilt-rate rule off
ISO_CONF='ref_aware_unstable 1
unstable_hard_tilt_rad 0.70
unstable_tilt_rad 0.699
unstable_soft_tilt_rad 0.699
unstable_gyro_rad_s 10
unstable_tilt_rate_rad_s 0
ref_dev_rad 1.0
ref_dev_min_tilt_rad 0.70
ref_dev_growth_rad_s 10'
tj_ev() { jq -r "select(.kind==\"tilt_judge_trigger\")|$1" "$(latest_report)/events.jsonl" 2>/dev/null | head -1; }
SHIPPED_BEFORE=$(sha256sum "$HERE/config/site.conf" | cut -d' ' -f1)

scn_h1() {
  echo "== H1 HARD_CAP only, reference PAUSED (invalid): tilt 0.69 / gyro 12 rad/s for 3 s does NOT exit; tilt 0.75 -> immediate WALKRUN with rule HARD_CAP"
  PM_ROOT=$(iso_root h1 "$ISO_CONF"); export PM_ROOT
  go_run h1 --oem-fall-tilt 0.9 -- --run-s 20.0 --ref-mode paused
  echo "0.69 12.0" > "$SD/tilt"; sleep 3.0
  check "0.69 rad + gyro 12 rad/s for 3 s: run still in progress, no tilt_judge_trigger, no REQUEST at the sim" '! log_has "RUN FINISHED" && ! sim_has "REQUEST mode" && [ -z "$(tj_ev .rule)" ]'
  echo "0.75 12.0" > "$SD/tilt"
  check "0.75 rad: run ended rc=10 (WalkRun, recoverable)" 'waitfor "$LG/service.log" "RUN FINISHED rc=10" 30'
  rm -f "$SD/tilt"; sleep 1.5
  check "event rule HARD_CAP, ref_valid=false, tilt in (0.70, 0.76], limits.hard_tilt_rad 0.7" '[ "$(tj_ev .rule)" = HARD_CAP ] && [ "$(tj_ev .ref_valid)" = false ] && [ "$(tj_ev .limits.hard_tilt_rad)" = 0.7 ] && awk -v t="$(tj_ev .tilt_rad)" "BEGIN{exit !(t>0.70 && t<=0.76)}"'
  check "report: UNSTABLE -> EXITED_WALKRUN, requested WALKRUN, final_fsm 802, no damping frames" '[ "$(rj .summary.fault)" = UNSTABLE ] && [ "$(rj .summary.result)" = EXITED_WALKRUN ] && [ "$(rj .summary.requested_mode)" = WALKRUN ] && [ "$(rj .summary.final_fsm)" = 802 ] && ! sim_has "first damping user_lowcmd frame"'
  lat; stop_env
}
scn_h2() {
  echo "== H2 HARD_CAP only, reference RUNNING (valid, fresh): same injections; the valid reference must not suppress the cap"
  PM_ROOT=$(iso_root h2 "$ISO_CONF"); export PM_ROOT
  go_run h2 --oem-fall-tilt 0.9 -- --run-s 20.0 --ref-mode running
  sleep 2.5   # past the 1 s lead-in: the reference is running
  echo "0.69 12.0" > "$SD/tilt"; sleep 2.5
  check "0.69 rad + gyro 12 with a running reference: no exit" '! log_has "RUN FINISHED" && ! sim_has "REQUEST mode" && [ -z "$(tj_ev .rule)" ]'
  echo "0.75 12.0" > "$SD/tilt"
  check "0.75 rad: rc=10" 'waitfor "$LG/service.log" "RUN FINISHED rc=10" 30'
  rm -f "$SD/tilt"; sleep 1.5
  check "event rule HARD_CAP with ref_valid=true (reference valid, still capped); report EXITED_WALKRUN fsm 802" '[ "$(tj_ev .rule)" = HARD_CAP ] && [ "$(tj_ev .ref_valid)" = true ] && [ "$(rj .summary.result)" = EXITED_WALKRUN ] && [ "$(rj .summary.final_fsm)" = 802 ]'
  lat; stop_env
}
scn_h3() {
  echo "== H3 HARD_CAP only: ramp 0.3 -> 0.9 rad over 0.8 s (growing tilt): the cap fires at ~0.70, BEFORE fallen 1.0, WalkRun takes it back"
  PM_ROOT=$(iso_root h3 "$ISO_CONF"); export PM_ROOT
  go_run h3 --tilt-ramp 0.3,0.9,0.8 --oem-fall-tilt 0.95 -- --run-s 20.0 --ref-mode paused
  check "rc=10 (EXITED_WALKRUN), not PASSIVE" 'waitfor "$LG/service.log" "RUN FINISHED rc=10" 30'
  sleep 1.5
  check "rule HARD_CAP, tilt at decision in [0.70, 0.85) (< fallen 1.0), final fsm 802" '[ "$(tj_ev .rule)" = HARD_CAP ] && awk -v t="$(rj .exit_detail.attitude_at_decision.tilt_rad)" "BEGIN{exit !(t>=0.70 && t<0.85)}" && [ "$(rj .summary.final_fsm)" = 802 ] && [ "$(rj .summary.requested_mode)" = WALKRUN ]'
  echo "  [info] attitude_at_decision.tilt=$(rj .exit_detail.attitude_at_decision.tilt_rad) trigger tilt=$(tj_ev .tilt_rad) detect->rpc=$(rj .exit_detail.detect_to_rpc_sent_ms) ms"
  lat; stop_env
}
scn_h4() {
  echo "== H4 FALLEN only: tilt steps straight to 1.15 rad (> fallen 1.0) -> PASSIVE (rc=13, FALLEN_PASSIVE), no WalkRun request"
  PM_ROOT=$(iso_root h4 "$ISO_CONF"); export PM_ROOT
  go_run h4 -- --run-s 20.0 --ref-mode paused
  echo "1.15 0.5" > "$SD/tilt"
  check "ends with the PASSIVE class (rc=13) and HOLD" 'waitfor "$LG/service.log" "HOLD: runner rc=13" 30'
  rm -f "$SD/tilt"; sleep 1.5
  check "report: FALLEN_PASSIVE, fault TILT, requested/final PASSIVE, final fsm 1, no WalkRun request" '[ "$(rj .summary.result)" = FALLEN_PASSIVE ] && [ "$(rj .summary.fault)" = TILT ] && [ "$(rj .summary.requested_mode)" = PASSIVE ] && [ "$(rj .summary.final_fsm)" = 1 ] && ! sim_has "REQUEST mode=2"'
  stop_env
}
scn_h5() {
  echo "== H5 CONTROL (product defaults, same injection as H1: 0.69 rad / gyro 12): the OTHER rules DO fire -> proves H1/H2 isolation is real"
  PM_ROOT=$(iso_root h5 "# product defaults"); export PM_ROOT
  go_run h5 --oem-fall-tilt 0.9 -- --run-s 20.0 --ref-mode paused
  echo "0.69 12.0" > "$SD/tilt"
  check "product defaults: run ended rc=10 by a non-HARD_CAP rule" 'waitfor "$LG/service.log" "RUN FINISHED rc=10" 30'
  rm -f "$SD/tilt"; sleep 1.5
  echo "  [info] product-default rule at 0.69/gyro12: $(tj_ev .rule)"
  check "rule is FIXED_LIMIT or ABS_RATE (not HARD_CAP)" '[ "$(tj_ev .rule)" = FIXED_LIMIT ] || [ "$(tj_ev .rule)" = ABS_RATE ]'
  stop_env
}
for s in h1 h2 h3 h4 h5; do "scn_$s"; done
kill_test_procs -9
AFTER=$(sha256sum "$HERE/config/site.conf" | cut -d' ' -f1)
check "shipped config/site.conf unchanged by the scenarios" '[ "$SHIPPED_BEFORE" = "$AFTER" ]'
echo; echo "HARD_CAP isolation (sim loopback): $PASS passed, $FAILN failed"; [ $FAILN -eq 0 ]
