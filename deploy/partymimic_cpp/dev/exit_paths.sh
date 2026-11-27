#!/usr/bin/env bash
# DEV ONLY (never shipped): key control/exit paths of the C++ deployment on a LOOPBACK test robot (iface lo, domain 77). Never touches a real robot.
#   usage: dev/exit_paths.sh <build_dir> [scenario ...]      (build_dir contains partymimic and dev/sim_robot)
#   needs jq. Exit policy under test: EXIT_POLICY.md (WALKRUN at once / PASSIVE for fallen / emergency; confirmation by GetFsmId + stable window).
set -u
# P4 hardening: (1) only ONE instance may run (the scenarios share /tmp paths and DDS domain 77); (2) NO `pkill -f` on command-line patterns (it also killed the invoking
# shell whenever its command line contained "partymimic service ..."): test processes are found by exact process name AND resolved executable path of THIS build dir.
exec 9>/tmp/pm_exit_paths.lock; flock -n 9 || { echo "another dev/exit_paths.sh is already running (lock /tmp/pm_exit_paths.lock): refusing to run concurrently"; exit 3; }
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
  # V1.6.1: no authorisation gate: the sim service is armed by default
  PM_LOGS="$LG" "$EXE" service --sim "${svcargs[@]}" > "$LG/service.out" 2>&1 & SVCPID=$!
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


scn_clean() {
  echo "== S1 clean run: F1+F3 -> READY -> L1+A -> motion window -> immediate WALKRUN (no damping first) -> confirmed -> COMPLETED"
  start_env clean -- --run-s 1.0
  press $KEY_LOAD 0.5
  check "READY reached" 'waitfor "$LG/service.log" "READY: model loaded" 30'
  press $KEY_GO 0.5
  check "run finished, service back to IDLE" 'waitfor "$LG/service.log" "RUN FINISHED rc=0" 30'
  sleep 1
  R=$(latest_report)
  check "SwitchToUserCtrl accepted (fsm 1000)" 'sim_has "SwitchToUserCtrl OK"'
  check "stiff policy frames reached the robot, CRC/mode valid" 'sim_has "first stiff user_lowcmd frame" && ! sim_has "BAD"'
  check "hand-back = SwitchToInternalCtrl(WALKRUN) -> fsm 802" 'sim_has "SwitchToInternalCtrl mode=2(WALKRUN) -> fsm 802"'
  check "no damping frame at all on the WALKRUN path (HOLD frames only)" '! sim_has "first damping user_lowcmd frame"'
  check "run_report: COMPLETED, requested/final WALKRUN, final_fsm 802, recovery_ok" '[ "$(rj .summary.result)" = COMPLETED ] && [ "$(rj .summary.requested_mode)" = WALKRUN ] && [ "$(rj .summary.final_mode)" = WALKRUN ] && [ "$(rj .summary.final_fsm)" = 802 ] && [ "$(rj .summary.recovery_ok)" = true ] && [ "$(rj .summary.restored_to_entry_fsm)" = true ]'
  check "timing: detect->rpc_sent < 20 ms; stable window observed (>= 0.8 s)" 'lt "$(rj .exit_detail.detect_to_rpc_sent_ms)" 20 && awk -v t="$(rj .exit_detail.timing_ms_from_detect.t_stable_confirmed)" "BEGIN{exit !(t>=800)}"'
  lat
  check "motion window = 50 control ticks @50 Hz" 'n=$(tail -n +2 "$R/trace.csv" | wc -l); [ "$n" -ge 45 ] && [ "$n" -le 55 ]'
  check "P5.3 race fix: the guardian ending by itself after the hand-back is NOT a CRITICAL (no SIGTERM to the runner)" '! log_has "CRITICAL" && ! log_has "guardian dead"'
  check "voice: 服务已启动 / 正在加载 / 准备就绪 / 开始执行 / 执行完成" 'for t in 服务已启动 正在加载 准备就绪 开始执行 执行完成; do sim_has "TTS.\"text\":\"$t\"" || exit 1; done'
  stop_env
  check "voice: 已退出 at service stop" 'sim_has "TTS.\"text\":\"已退出\""'
}
scn_abort_ready() {
  echo "== S2 START x2 while READY: abort BEFORE control (nothing may reach the robot)"
  start_env abortready -- --run-s 1.0
  press $KEY_LOAD 0.5
  waitfor "$LG/service.log" "READY: model loaded" 30
  dclick
  check "runner ended before taking control" 'waitfor "$LG/service.log" "runner ended before taking control" 15'
  sleep 1
  check "robot never switched to user control, no user_lowcmd frame" '! sim_has "SwitchToUserCtrl" && ! sim_has "first stiff" && ! sim_has "first damping"'
  check "service back to IDLE and voice says cancelled" 'log_has "back to IDLE" && sim_has "TTS.*已退出"'
  stop_env
}

scn_abort_run() {
  echo "== S3 START x2 during RUN: immediate WALKRUN (HOLD frames, no damping) -> EXITED_WALKRUN"
  go_run abortrun -- --run-s 8.0 --ref-mode paused
  sleep 0.5; dclick
  check "service saw START double-click in RUN" 'waitfor "$LG/service.log" "START double-click detected" 10'
  check "run ended by operator, service IDLE" 'waitfor "$LG/service.log" "RUN FINISHED rc=10" 30'
  sleep 1
  check "NO damping frames (kp=0) on the WALKRUN path; WalkRun restored (fsm 802) and confirmed" '! sim_has "first damping user_lowcmd frame" && sim_has "SwitchToInternalCtrl mode=2(WALKRUN) -> fsm 802"'
  check "report: EXITED_WALKRUN, reason REMOTE_STOP, recovery_ok, final fsm 802" '[ "$(rj .summary.result)" = EXITED_WALKRUN ] && rj .summary.exit_reason | grep -q "^REMOTE_STOP" && [ "$(rj .summary.recovery_ok)" = true ] && [ "$(rj .summary.final_fsm)" = 802 ]'
  lat; check "timing: detect->rpc_sent < 20 ms" 'lt "$(rj .exit_detail.detect_to_rpc_sent_ms)" 20'
  check "voice: 已退出 (operator stop, WalkRun confirmed)" 'sim_has "TTS.*已退出"'
  stop_env
}
scn_guardian() {
  echo "== S4 runner killed (kill -9) while in control: guardian = same ExitPolicy (SIGKILL writer, immediate WALKRUN, HOLD frames)"
  go_run guardian -- --run-s 8.0 --ref-mode paused
  RP=$(pgrep -f "partymimic runner --iface lo" | head -1); echo "  killing runner pid $RP"
  kill -9 "$RP"
  check "guardian took over" 'waitfor "$(guardian_log)" "TAKE-OVER" 15'
  sleep 5
  G=$(ls -t $LG/guardian/*.json | head -1)
  check "guardian: WALKRUN at once, no damping frames, fsm 802, result EXITED_WALKRUN" '! sim_has "first damping user_lowcmd frame" && sim_has "SwitchToInternalCtrl mode=2(WALKRUN) -> fsm 802" && [ "$(jq -r .result "$G")" = EXITED_WALKRUN ] && [ "$(jq -r .recovery_ok "$G")" = true ]'
  echo "  [info] guardian detect->rpc_sent = $(jq -r .exit_detail.detect_to_rpc_sent_ms "$G") ms (includes SIGKILL+reap $(jq -r .kill_ms "$G") ms)"
  check "V1.6.1: abnormal end announced with the REAL result (异常已交还), NO HOLD, and the service settles back to IDLE by itself (no ack)" 'log_has "ABNORMAL end" && ! log_has "HOLD:" && log_has "guardian take-over result=EXITED_WALKRUN" && sim_has "TTS.*异常已交还" && waitfor "$LG/state.json" "\"state\":\"IDLE\"" 40'
  stop_env
}
scn_fault() {
  echo "== S5 robot already DOWN in RUN (roll 1.3 rad): FALLEN -> PASSIVE (damping frames + SwitchToInternalCtrl(PASSIVE)), never WALKRUN"
  go_run fault --flip-after-user-s 0.4 -- --run-s 8.0 --ref-mode paused
  check "fall ends the run with the PASSIVE class (rc=13) and HOLD" 'waitfor "$LG/service.log" "RUN FINISHED rc=13" 40'
  sleep 1
  check "report: FALLEN_PASSIVE, fault TILT, requested/final PASSIVE, final fsm 1, no WalkRun request" '[ "$(rj .summary.result)" = FALLEN_PASSIVE ] && [ "$(rj .summary.fault)" = TILT ] && [ "$(rj .summary.requested_mode)" = PASSIVE ] && [ "$(rj .summary.final_fsm)" = 1 ] && ! sim_has "REQUEST mode=2" && sim_has "(PASSIVE) -> fsm 1" && sim_has "first damping user_lowcmd frame"'
  check "service state.json: last_result FALLEN_PASSIVE, last_final_fsm 1, last_recovery_ok ..." '[ "$(jq -r .last_result "$LG/state.json")" = FALLEN_PASSIVE ] && [ "$(jq -r .last_final_fsm "$LG/state.json")" = 1 ]'
  check "voice: 恢复失败" 'sim_has "TTS.*恢复失败"'
  lat
  stop_env
}
scn_blocked() {
  echo "== S6 admission BLOCKED (robot not in WalkRun, fsm 1): nothing is ever sent"
  start_env blocked --fsm 1 -- --run-s 1.0
  press $KEY_LOAD 0.5
  check "BLOCKED reported (rc=6), service IDLE" 'waitfor "$LG/service.log" "rc=6" 40'
  press $KEY_GO 0.5
  sleep 1
  check "no RPC switch, no user_lowcmd frame" '! sim_has "SwitchToUserCtrl" && ! sim_has "first stiff" && ! sim_has "first damping"'
  check "voice says blocked" 'sim_has "TTS.*检查失败"'
  stop_env
}
scn_term() {
  echo "== S7 service SIGTERM while RUN (./stop.sh path): damp -> hand-back"
  start_env term -- --run-s 8.0 --ref-mode paused
  press $KEY_LOAD 0.5
  waitfor "$LG/service.log" "READY: model loaded" 30
  press $KEY_GO 0.5
  waitfor "$SD/sim_events.log" "first stiff user_lowcmd frame" 20
  sleep 1.0
  kill -TERM "$SVCPID"
  check "robot handed back to WalkRun" 'waitfor "$SD/sim_events.log" "SwitchToInternalCtrl mode=2(WALKRUN) -> fsm 802" 30'
  check "service exited" 'for _ in $(seq 1 300); do kill -0 "$SVCPID" 2>/dev/null || exit 0; sleep 0.1; done; exit 1'
  stop_env
}

scn_rival() {
  echo "== S8 competing controller ACTIVELY writing rt/user_lowcmd (idle factory writers present) / rt/arm_sdk: blocked, nothing is sent"
  for topic in user_lowcmd arm_sdk; do
    start_env rival_$topic --rival $topic -- --run-s 1.0
    press $KEY_LOAD 0.5
    check "[$topic] BLOCKED by the active writer (rc=6)" 'waitfor "$LG/service.log" "rc=6" 40'
    check "[$topic] reason names the competing controller" 'grep -rq "COMPETING CONTROLLER\|no_active_writer_on_user_lowcmd" "$LG"/runs/*/console.log'
    check "[$topic] robot never switched" '! sim_has "SwitchToUserCtrl"'
    stop_env
  done
}
scn_fsm_change() {
  echo "== S9 factory fsm changes between READY and L1+A (factory remote reacted / operator changed mode): GO refused, nothing is sent"
  start_env fsmchange -- --run-s 1.0
  press $KEY_LOAD 0.5
  waitfor "$LG/service.log" "READY: model loaded" 30
  echo 500 > "$SD/fsm"; sleep 0.5
  press $KEY_GO 0.5
  check "GO cancelled: fsm changed since READY (V1.6: caught by the announce_wait re-verification, reason fsm_changed; V1.5.1 text 'GO refused' if the recheck sees it)" 'waitfor "$LG/service.log" "GO refused\\|take-over CANCELLED" 20 && grep -rq "fsm_unchanged_since_ready\\|fsm_changed" "$LG"/runs/*/console.log'
  check "robot never switched" '! sim_has "SwitchToUserCtrl" && ! sim_has "first stiff"'
  stop_env
}


scn_unstable() {
  echo "== S10 instability (roll steps to 0.42 rad with gyro 3 rad/s (> unstable_tilt_rad 0.40), still recoverable): immediate WALKRUN, measure detect->RPC latency"
  go_run unstable -- --run-s 8.0 --ref-mode paused
  echo "0.42 3.0" > "$SD/tilt"     # roll steps to 0.42 rad (> 0.40, < OEM fall tilt 0.45), gyro 3 rad/s: unstable but still recoverable
  check "run ended with the fault exit (rc=10); recoverable (WalkRun confirmed) -> SETTLE, no HOLD" 'waitfor "$LG/service.log" "RUN FINISHED rc=10" 40 && ! log_has "HOLD:"'
  rm -f "$SD/tilt"; sleep 1
  check "report: UNSTABLE -> EXITED_WALKRUN, WalkRun confirmed (fsm 802), recovery_ok" '[ "$(rj .summary.fault)" = UNSTABLE ] && [ "$(rj .summary.result)" = EXITED_WALKRUN ] && [ "$(rj .summary.final_fsm)" = 802 ] && [ "$(rj .summary.recovery_ok)" = true ]'
  check "WALKRUN requested at once: no damping frames, HOLD frames only" '! sim_has "first damping user_lowcmd frame" && sim_has "REQUEST mode=2" && ! sim_has "REQUEST mode=1"'
  sleep 3.5
  check "P5.3 stats: the trigger sample counts - attitude_extremes overall >= 0.41 rad (injected 0.42), guard_stats.max_tilt includes it, anomaly summary reports it" 'R=$(latest_report); awk -v a="$(jq -r .attitude_extremes.max_tilt_rad_overall "${R}run_report.json")" -v g="$(jq -r .guard_stats.max_tilt "${R}run_report.json")" "BEGIN{exit !(a>=0.41 && g>=0.41)}" && grep -q "TILT" "${R}anomaly_summary.txt"'
  lat
  check "timing: detect->rpc_sent < 10 ms" 'lt "$(rj .exit_detail.detect_to_rpc_sent_ms)" 10'
  check "timing: lowstate-sample -> rpc_sent < 20 ms (one control period)" 'lt "$(rj .exit_detail.event_to_rpc_sent_ms)" 20'
  D=$(sim_delta "tilt hook published" "REQUEST mode=2"); echo "  [info] END-TO-END (sim publishes the tilted sample -> sim receives SwitchToInternalCtrl) = $D ms"
  check "end-to-end tilt onset -> RPC arrival at the robot < 20 ms" 'lt "$D" 20'
  check "voice: 异常已交还 (said after the RPC, WalkRun confirmed)" 'sim_has "TTS.*异常已交还"'
  stop_env
}
scn_oem_fall() {
  echo "== S11 field falls reproduced: tilt keeps growing during a slow RPC, OEM cannot recover tilt > 0.45 -> falls into fsm 1: WALKRUN_RECOVERY_FAILED (never COMPLETED)"
  go_run oemfall --tilt-ramp 0.3,0.9,0.8 --internal-delay 0.35 -- --run-s 8.0 --ref-mode paused
  check "run ends, rc=11 HOLD" 'waitfor "$LG/service.log" "RUN FINISHED rc=11" 40'
  sleep 1
  check "sim: WALKRUN request returned 0 but the OEM landed in fsm 1" 'sim_has "REQUEST mode=2" && sim_has "mode=2(WALKRUN) -> fsm 1 .OEM cannot recover"'
  check "HOLD frames (no damping) were streamed during the slow RPC" 'sim_has "mode=2(WALKRUN).*frames during the rpc: stiff +[1-9][0-9]* damp +0"'
  check "report: WALKRUN_RECOVERY_FAILED, rpc_code 0, final_fsm 1, final_mode PASSIVE, recovery_ok=false" '[ "$(rj .summary.result)" = WALKRUN_RECOVERY_FAILED ] && [ "$(rj .exit_detail.rpc_code)" = 0 ] && [ "$(rj .summary.final_fsm)" = 1 ] && [ "$(rj .summary.final_mode)" = PASSIVE ] && [ "$(rj .summary.recovery_ok)" = false ] && [ "$(rj .summary.restored_to_entry_fsm)" = false ]'
  check "no extra RPC needed when fsm is already 1 (final PASSIVE)" '[ "$(sim_count "REQUEST mode=")" = 1 ]'
  check "service: HOLD + state.json shows the failed recovery" '[ "$(jq -r .last_recovery_ok "$LG/state.json")" = false ] && [ "$(jq -r .last_final_fsm "$LG/state.json")" = 1 ] && log_has "RECOVERY NOT CONFIRMED"'
  check "voice: 恢复失败 (not 执行完成)" 'sim_has "TTS.*恢复失败" && ! sim_has "TTS.*执行完成"'
  lat
  stop_env
}
scn_recovery_fail() {
  echo "== S12 WalkRun recovery failure variants: RPC ok but fsm mismatch / unchanged / drops later / nothing works"
  echo "  -- (a) the field-log bug: normal end, rpc 0, OEM lands in fsm 1 -> must NOT be COMPLETED"
  start_env rf_a --walkrun-fsm1 -- --run-s 1.0
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "(a) HOLD rc=11" 'waitfor "$LG/service.log" "RUN FINISHED rc=11" 40'
  sleep 1
  check "(a) result WALKRUN_RECOVERY_FAILED, rpc 0, final fsm 1, mode_matched/recovery false, not COMPLETED" '[ "$(rj .summary.result)" = WALKRUN_RECOVERY_FAILED ] && [ "$(rj .exit_detail.rpc_code)" = 0 ] && [ "$(rj .summary.final_fsm)" = 1 ] && [ "$(rj .summary.recovery_ok)" = false ] && [ "$(rj .summary.exit_confirmed)" = false ] && ! grep -q "\"result\":\"COMPLETED\"" "$(latest_report)/run_report.json"'
  check "(a) voice 恢复失败, never 执行完成" 'sim_has "TTS.*恢复失败" && ! sim_has "TTS.*执行完成"'
  stop_env
  echo "  -- (b) WALKRUN request ignored (rpc 0, fsm stays 1000): PASSIVE request follows with damping frames"
  start_env rf_b --ignore-walkrun -- --run-s 1.0
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "(b) HOLD rc=11" 'waitfor "$LG/service.log" "RUN FINISHED rc=11" 60'
  sleep 1
  check "(b) rpc modes WALKRUN then PASSIVE, final PASSIVE fsm 1, damping frames after the fallback" '[ "$(rj ".exit_detail.rpc_modes|join(\",\")")" = "WALKRUN,PASSIVE" ] && [ "$(rj .summary.final_fsm)" = 1 ] && [ "$(rj .summary.result)" = WALKRUN_RECOVERY_FAILED ] && sim_has "first damping user_lowcmd frame"'
  stop_env
  echo "  -- (c) WalkRun reached, OEM drops into fsm 1 0.5 s later: the stable window must catch it"
  start_env rf_c --walkrun-drop-s 0.5 -- --run-s 1.0
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "(c) HOLD rc=11" 'waitfor "$LG/service.log" "RUN FINISHED rc=11" 40'
  sleep 1
  check "(c) WALKRUN_RECOVERY_FAILED, final fsm 1" '[ "$(rj .summary.result)" = WALKRUN_RECOVERY_FAILED ] && [ "$(rj .summary.final_fsm)" = 1 ] && sim_has "OEM drops into fsm 1"'
  stop_env
  echo "  -- (d) every exit RPC ignored: HANDBACK_UNCONFIRMED, exit code 5"
  start_env rf_d --ignore-internal -- --run-s 1.0 --unconfirmed-hold-s 2
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "(d) HOLD rc=5" 'waitfor "$LG/service.log" "RUN FINISHED rc=5" 60'
  sleep 1
  check "(d) HANDBACK_UNCONFIRMED, voice 检查失败" '[ "$(rj .summary.result)" = HANDBACK_UNCONFIRMED ] && [ "$(rj .summary.recovery_ok)" = false ] && sim_has "TTS.*检查失败"'
  stop_env
}
scn_estop() {
  echo "== S14 emergency PASSIVE by the ESTOP file (stop.sh --estop): damping frame + SwitchToInternalCtrl(PASSIVE), never WALKRUN"
  go_run estop -- --run-s 8.0 --ref-mode paused
  echo "estop" > /tmp/partymimic_cpp_ESTOP
  check "run ends, rc=12 HOLD" 'waitfor "$LG/service.log" "RUN FINISHED rc=12" 40'
  sleep 1
  check "report: EMERGENCY_PASSIVE, fault ESTOP, final fsm 1" '[ "$(rj .summary.result)" = EMERGENCY_PASSIVE ] && [ "$(rj .summary.fault)" = ESTOP ] && [ "$(rj .summary.final_fsm)" = 1 ] && [ "$(rj .summary.requested_mode)" = PASSIVE ]'
  check "sim: PASSIVE request only (no WALKRUN), damping frames" 'sim_has "REQUEST mode=1" && ! sim_has "REQUEST mode=2" && sim_has "first damping user_lowcmd frame"'
  lat; check "timing: detect->rpc_sent < 10 ms" 'lt "$(rj .exit_detail.detect_to_rpc_sent_ms)" 10'
  check "voice: 紧急被动" 'sim_has "TTS.*紧急被动"'
  rm -f /tmp/partymimic_cpp_ESTOP
  stop_env
}
scn_l2b() {
  echo "== S15 remote L2+B (0x220): ignored before control; during the run = emergency PASSIVE (debounced, not triggered by START/F1+F3/L1+A)"
  start_env l2b -- --run-s 8.0 --ref-mode paused
  press $KEY_LOAD 0.5
  waitfor "$LG/service.log" "READY: model loaded" 30
  press 0x220 0.6
  sleep 0.5
  check "L2+B at READY does nothing (still READY, nothing sent)" '! sim_has "SwitchToUserCtrl" && ! sim_has "REQUEST" && pgrep -f "partymimic runner --iface lo" >/dev/null'
  press $KEY_GO 0.5
  waitfor "$SD/sim_events.log" "first stiff user_lowcmd frame" 20
  sleep 0.7
  setkeys 0x20; sleep 0.3; setkeys 0x0; sleep 0.4     # L2 alone (half chord): nothing
  check "L2 alone / short glitch does not exit" '! sim_has "REQUEST"'
  setkeys 0x220; sleep 0.04; setkeys 0x0; sleep 0.4   # 40 ms: below the 100 ms debounce
  check "L2+B shorter than the debounce does not exit" '! sim_has "REQUEST"'
  setkeys 0x220
  check "L2+B held -> run ends, rc=12 HOLD" 'waitfor "$LG/service.log" "RUN FINISHED rc=12" 40'
  setkeys 0x0; sleep 1
  check "report: EMERGENCY_PASSIVE, fault REMOTE_DAMP, final fsm 1, PASSIVE only" '[ "$(rj .summary.result)" = EMERGENCY_PASSIVE ] && [ "$(rj .summary.fault)" = REMOTE_DAMP ] && [ "$(rj .summary.final_fsm)" = 1 ] && sim_has "REQUEST mode=1" && ! sim_has "REQUEST mode=2" && sim_has "first damping user_lowcmd frame"'
  lat; check "timing: detect->rpc_sent < 10 ms" 'lt "$(rj .exit_detail.detect_to_rpc_sent_ms)" 10'
  check "voice: 紧急被动" 'sim_has "TTS.*紧急被动"'
  stop_env
}
scn_race() {
  echo "== S16 runner and guardian trigger together: ONE exit owner (runner), exactly one SwitchToInternalCtrl, no guardian take-over"
  echo "  -- (a) ESTOP file while the exit RPC is slow (0.8 s): runner-owned emergency exit"
  go_run race_a --internal-delay 0.8 -- --run-s 8.0 --ref-mode paused
  echo "estop" > /tmp/partymimic_cpp_ESTOP
  check "(a) HOLD rc=12" 'waitfor "$LG/service.log" "RUN FINISHED rc=12" 40'
  sleep 1.5
  check "(a) exactly one exit RPC (PASSIVE), guardian did not take over" '[ "$(sim_count "REQUEST mode=")" = 1 ] && sim_has "REQUEST mode=1" && ! grep -q "TAKE-OVER" "$(guardian_log)"'
  rm -f /tmp/partymimic_cpp_ESTOP
  stop_env
  echo "  -- (b) START x2 with a 1.0 s RPC: HOLD frames during the RPC, runner silent while confirming (guardian must stay out)"
  go_run race_b --internal-delay 1.0 -- --run-s 8.0 --ref-mode paused
  dclick
  check "(b) run finished rc=10" 'waitfor "$LG/service.log" "RUN FINISHED rc=10" 40'
  sleep 1.5
  check "(b) exactly one exit RPC (WALKRUN), result EXITED_WALKRUN, guardian did not take over" '[ "$(sim_count "REQUEST mode=")" = 1 ] && [ "$(rj .summary.result)" = EXITED_WALKRUN ] && ! grep -q "TAKE-OVER" "$(guardian_log)"'
  stop_env
  echo "  -- (c) runner hung (SIGSTOP): guardian owns the exit, still exactly one RPC"
  go_run race_c -- --run-s 8.0 --ref-mode paused
  RP=$(pgrep -f "partymimic runner --iface lo" | head -1); kill -STOP "$RP"
  check "(c) guardian took over the hung runner" 'waitfor "$(guardian_log)" "TAKE-OVER" 15'
  sleep 4
  check "(c) exactly one exit RPC, WALKRUN, no damping frames" '[ "$(sim_count "REQUEST mode=")" = 1 ] && sim_has "REQUEST mode=2" && ! sim_has "first damping user_lowcmd frame"'
  kill -9 "$RP" 2>/dev/null
  stop_env
}
scn_rpc_delay() {
  echo "== S17 slow RPC (0.4 s) on a normal end: HOLD frames keep streaming (stiff, q snapshot), no damping, still COMPLETED"
  start_env rpcdelay --internal-delay 0.4 -- --run-s 1.0
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "run finished rc=0" 'waitfor "$LG/service.log" "RUN FINISHED rc=0" 40'
  sleep 1
  check "frames during the RPC: stiff >= 20 (100 Hz), damp 0" 'sim_has "mode=2(WALKRUN).*frames during the rpc: stiff +[2-9][0-9]* damp +0"'
  check "report: COMPLETED, rpc latency >= 400 ms measured" '[ "$(rj .summary.result)" = COMPLETED ] && awk -v t="$(rj .exit_detail.rpc_latency_ms)" "BEGIN{exit !(t>=390)}"'
  lat; check "detect->rpc_sent unaffected by the slow RPC (< 20 ms)" 'lt "$(rj .exit_detail.detect_to_rpc_sent_ms)" 20'
  stop_env
}
scn_inject() {
  echo "== S18 policy-side faults (injected in the --sim runner only): inference exception / NaN output / loop overrun / hang(watchdog) -> immediate WALKRUN, EXITED_WALKRUN"
  for spec in inference:INFERENCE nan:CMD_INVALID overrun:LOOP_OVERRUN hang:WATCHDOG; do
    kind=${spec%%:*}; fault=${spec##*:}
    export PM_SIM_INJECT="$kind:15"
    go_run inj_$kind -- --run-s 8.0 --ref-mode paused
    unset PM_SIM_INJECT
    check "[$kind] ends with rc=10 (recoverable: SETTLE, no HOLD)" 'waitfor "$LG/service.log" "RUN FINISHED rc=10" 40 && ! log_has "HOLD:"'
    sleep 1
    check "[$kind] fault $fault -> EXITED_WALKRUN, WalkRun confirmed, no damping frames" '[ "$(rj .summary.fault)" = '$fault' ] && [ "$(rj .summary.result)" = EXITED_WALKRUN ] && [ "$(rj .summary.final_fsm)" = 802 ] && ! sim_has "first damping user_lowcmd frame"'
    echo "  [info] [$kind] detect->rpc_sent = $(rj .exit_detail.detect_to_rpc_sent_ms) ms"
    check "[$kind] voice 异常已交还" 'sim_has "TTS.*异常已交还"'
    stop_env
  done
}
scn_release() {
  echo "== S19 operator release (partymimic release): same ExitPolicy, robot left in user control (fsm 1000) -> WALKRUN confirmed"
  start_env release -- --run-s 1.0
  echo 1000 > "$SD/fsm"; sleep 0.5
  out=$("$EXE" release --iface lo --domain 77 2>&1); echo "$out" | tail -2 | cut -c1-200
  check "release: WALKRUN requested, fsm 802, result EXITED_WALKRUN" 'echo "$out" | grep -q "\"result\":\"EXITED_WALKRUN\"" && sim_has "SwitchToInternalCtrl mode=2(WALKRUN) -> fsm 802"'
  stop_env
}
scn_arm_fail() {
  echo "== S21 ARM fails (SwitchToUserCtrl rejected): the exit must not blindly send a WALKRUN RPC while the OEM never left WalkRun"
  start_env armfail --user-reject -- --run-s 1.0
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "ARM_FAILED (rc=4) HOLD" 'waitfor "$LG/service.log" "RUN FINISHED rc=4" 40'
  sleep 1.5
  check "report ARM_FAILED, no exit RPC (fsm already the entry WalkRun fsm 802), voice 检查失败" '[ "$(rj .summary.result)" = ARM_FAILED ] && [ "$(rj .summary.final_fsm)" = 802 ] && [ "$(rj .exit_detail.skipped_rpc)" = true ] && ! sim_has "REQUEST" && sim_has "TTS.*检查失败"'
  stop_env
}
scn_caps() {
  echo "== S20 caps (P4 defaults max_run_s 12 / max_authority 1.0), compile-time limits (60 s / 1.0), partial-authority refusal, ramp/end-hold plumbing, legacy values flagged"
  start_env caps -- --run-s 2.0 --authority-max 1.0 --ref-mode paused
  press $KEY_LOAD 0.5
  check "READY with run_s 2.0 / authority 1.0 (sim)" 'waitfor "$LG/service.log" "READY: model loaded" 30'
  R=$(latest_report)
  check "WARNING in the service/runner log and ready.json exceeds_legacy_caps=true (run_s 2 > legacy 1.0 s)" 'log_has "EXCEED the legacy real-robot caps" && grep -q "EXCEEDS THE LEGACY CAPS" "$R/console.log" && [ "$(jq -r .exceeds_legacy_caps "$R/ready.json")" = true ]'
  stop_env
  # real-mode refusals happen before any DDS object exists; a non-existent interface name makes a real connection impossible even if the check were broken
  export PM_LOGS=/tmp/pm_logs_caps_refuse; rm -rf $PM_LOGS; mkdir -p $PM_LOGS
  out=$("$EXE" runner --iface nonexistent_pm0 --run-s 13 --authority-max 1.0 2>&1); rc=$?
  check "real run_s 13 > cap 12 refused (rc 2, before any DDS)" '[ $rc -eq 2 ] && echo "$out" | grep -q "run_s .* not in (0, max_run_s=12"'
  out=$("$EXE" runner --iface nonexistent_pm0 --run-s 1.0 --authority-max 1.5 2>&1); rc=$?
  check "real authority 1.5 > cap 1.0 refused" '[ $rc -eq 2 ] && echo "$out" | grep -q "authority .* not in (0, max_authority=1"'
  out=$("$EXE" runner --iface nonexistent_pm0 --run-s 1.0 --authority-max 0.3 --ref-mode paused 2>&1); rc=$?
  check "sustained partial authority 0.3 REFUSED (unstable regime) unless --allow-partial-authority" '[ $rc -eq 2 ] && echo "$out" | grep -q "partial-authority regime"'
  out=$("$EXE" runner --iface nonexistent_pm0 --run-s 1.0 --authority-max 1.0 --ramp-s 3 2>&1); rc=$?
  check "ramp_s 3 > 2 s refused" '[ $rc -eq 2 ] && echo "$out" | grep -q "ramp-s"'
  out=$("$EXE" runner --iface lo --sim --run-s 61 --authority-max 1.0 2>&1); rc=$?
  check "above the compile-time limit (60 s) refused even in --sim" '[ $rc -eq 2 ] && echo "$out" | grep -q "compile-time limits (60 s"'
  out=$("$EXE" runner --iface nonexistent_pm0 --run-s 61 --authority-max 1.0 2>&1); rc=$?
  check "real run_s 61: refused by the site cap (default 12) AND above the 60 s compile-time ceiling, with (rc 2)" '[ $rc -eq 2 ] && echo "$out" | grep -q "not in (0, max_run_s=12"'
  out=$("$EXE" runner --iface nonexistent_pm0 --run-s 20 --authority-max 1.0 --ref-mode running 2>&1); rc=$?
  check "real run_s 20 <= 60 but > site max_run_s 12 (default site.conf) refused: the compile-time ceiling does not grant it" '[ $rc -eq 2 ] && echo "$out" | grep -q "not in (0, max_run_s=12"'
  unset PM_LOGS
}

scn_commloss() {
  echo "== S22 communication loss during RUN: rt/lowstate silent -> COMM_TIMEOUT -> WALKRUN requested at once (OEM has its own estimator), FSM-verified; PASSIVE only if it fails"
  go_run commloss -- --run-s 8.0 --ref-mode paused
  touch "$SD/mute"
  check "run ended with the fault exit and HOLD" 'waitfor "$LG/service.log" "RUN FINISHED rc=" 40'
  rm -f "$SD/mute"; sleep 1
  check "fault COMM_TIMEOUT detected, WALKRUN requested (never PASSIVE first), no damping frames" '[ "$(rj .summary.fault)" = COMM_TIMEOUT ] && sim_has "REQUEST mode=2" && ! sim_has "REQUEST mode=1" && ! sim_has "first damping user_lowcmd frame"'
  check "result EXITED_WALKRUN with FSM-verified WalkRun (final_fsm 802), not COMPLETED" '[ "$(rj .summary.result)" = EXITED_WALKRUN ] && [ "$(rj .summary.final_fsm)" = 802 ] && [ "$(rj .summary.recovery_ok)" = true ]'
  lat
  check "detect->rpc_sent < 20 ms" 'lt "$(rj .exit_detail.detect_to_rpc_sent_ms)" 20'
  stop_env
}
# ---- P4.1: the REAL start entry (start.sh -> service -> guardian/runner command lines). Field bug: start.sh -> guardian "missing value for", 178 restarts.
start_sim_only() { kill_test_procs -9; sleep 0.5; SD="/tmp/pm_sim_$1"; LG="/tmp/pm_logs_$1"; rm -rf "$SD" "$LG" /tmp/partymimic_cpp_ESTOP /tmp/partymimic_cpp_control.lock /tmp/partymimic_cpp_guardian.alive; mkdir -p "$SD" "$LG"; echo 0 > "$SD/keys"
  "$SIM" --dir "$SD" > "$SD/sim.out" 2>&1 & SIMPID=$!; sleep 1.0; }
run_start_sh() { PM_BIN="$EXE" PM_LOGS="$LG" PM_ROOT="$HERE" bash "$HERE/start.sh" --sim "$@" > "$LG/start.out" 2>&1 < /dev/null; }
stop_sh() { PM_LOGS="$LG" PM_ROOT="$HERE" bash "$HERE/stop.sh" > "$LG/stop.out" 2>&1; kill_test_procs -TERM; kill -TERM "$SIMPID" 2>/dev/null; sleep 0.5; kill -9 "$SIMPID" 2>/dev/null; wait 2>/dev/null; rm -f /tmp/partymimic_cpp_guardian.alive; }
alive_age() { echo $(( $(date +%s) - $(stat -c %Y /tmp/partymimic_cpp_guardian.alive 2>/dev/null || echo 0) )); }
scn_start_sh() {
  echo "== S24 real start entry: ./start.sh --stage hold --yes (all P4 flags): guardian starts ONCE and stays alive, F1+F3 -> READY, START x2 cancels, stop.sh"
  start_sim_only startsh
  run_start_sh --stage hold --yes; rc=$?
  check "start.sh returned 0 and verified 'guardian alive: OK'" '[ $rc -eq 0 ] && grep -q "guardian alive: OK" "$LG/start.out"'
  ok=1; for i in 1 2 3 4 5 6 7 8; do [ "$(alive_age)" -le 2 ] || ok=0; sleep 1; done
  check "guardian heartbeat fresh for 8 consecutive seconds after start" '[ $ok -eq 1 ]'
  check "guardian started exactly once, never ended, no 'missing value'" '[ "$(grep -c "guardian started" "$LG/service.log")" -eq 1 ] && ! grep -q "guardian ended" "$LG/service.log" && ! grep -rq "missing value" "$LG/guardian/" "$LG/service.log" "$LG/service.out"'
  check "state.json: guardian_alive true, guardian_fatal false" 'jq -e ".guardian_alive == true and .guardian_fatal == false" "$LG/state.json" >/dev/null'
  press $KEY_LOAD 0.5
  check "F1+F3 -> READY (not refused)" 'waitfor "$LG/service.log" "READY: model loaded" 30 && ! grep -q "F1+F3 ignored" "$LG/service.log"'
  dclick
  check "START x2 in READY cancels before control; service back to IDLE" 'waitfor "$LG/service.log" "back to IDLE" 15 && ! sim_has "first stiff user_lowcmd frame"'
  check "after the run the guardian is STILL alive and was never restarted" '[ "$(grep -c "guardian started" "$LG/service.log")" -eq 1 ] && [ "$(alive_age)" -le 2 ]'
  stop_sh
  check "stop.sh stopped the service" 'grep -q "service stopped" "$LG/stop.out"'
  echo "  -- variant (V1.6.1): the FULL dance is armed by plain ./start.sh (no flag, no authorisation, stdin closed)"
  start_sim_only startsh2
  run_start_sh --mode dance; rc=$?
  check "dance: armed at once, nothing asked, service started" '[ $rc -eq 0 ] && grep -q "READY TO USE (mode dance" "$LG/start.out" && grep -q "armed=1" "$LG/service.log" && ! grep -q "NOT ARMED\|CHECK MODE" "$LG/service.log"'
  stop_sh
}
scn_guardian_fatal() {
  echo "== S25 guardian start-up failure (rc 2 = bad arguments): restart loop STOPPED after the first failure, exact reason in service.log / state.json / F1+F3 refusal / start.sh output"
  start_sim_only gfatal
  export PM_TEST_GUARDIAN_ARGS="--bogus-opt"
  run_start_sh --stage hold --yes; rc=$?
  check "start.sh exits 1 and prints the reason (no silent loop)" '[ $rc -eq 1 ] && grep -q "guardian is NOT alive" "$LG/start.out" && grep -q "missing value for --bogus-opt" "$LG/start.out"'
  sleep 6
  check "guardian started exactly once (no restart loop) in 6+ s" '[ "$(grep -c "guardian started" "$LG/service.log")" -eq 1 ]'
  check "service.log: ERROR guardian cannot start + restart loop STOPPED + exact cause" 'grep -q "ERROR: guardian cannot start" "$LG/service.log" && grep -q "restart loop is STOPPED" "$LG/service.log" && grep -q "missing value for --bogus-opt" "$LG/service.log"'
  check "state.json: guardian_fatal true with the reason" 'jq -e ".guardian_fatal == true and (.guardian_why | contains(\"bogus-opt\"))" "$LG/state.json" >/dev/null'
  press $KEY_LOAD 0.5; sleep 1
  check "F1+F3 refused WITH the reason (not the generic text), no runner spawned" 'grep -q "F1+F3 ignored - guardian exited" "$LG/service.log" && grep -q "bogus-opt" "$LG/service.log" && ! grep -q "runner started" "$LG/service.log"'
  check "voice 检查失败 said (fatal + refusal), robot never touched" 'sim_has "TTS.*检查失败" && ! sim_has "first stiff user_lowcmd frame"'
  unset PM_TEST_GUARDIAN_ARGS
  stop_sh
}


# ---- P5.1: logging / test modes / real start entry (see LOGGING.md). Helpers
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib"
ltool() { "$EXE" logtool "$@"; }
ev_count() { jq -s "[.[] | select(.kind==\"$2\")] | length" "$1/events.jsonl" 2>/dev/null; }
rrj() { jq -r "$1" "$R/run_report.json" 2>/dev/null; }
scn_log_complete() {
  echo "== S26 logging completeness (clean run): every file, every column group, hashes, timing, bit-exact policy replay, pre-take-over + post-exit state"
  start_env logc -- --run-s 2.0
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "run finished" 'waitfor "$LG/service.log" "RUN FINISHED rc=0" 40'
  sleep 3.5   # runner keeps recording the post-exit window (log_post_exit_s 2.0) before it writes the final report
  R=$(latest_report); R=${R%/}
  check "all files exist: trace obs state_stream events exit_frames safety_clips manifest run_report anomaly_summary.{json,txt}" 'for f in trace.csv obs.bin state_stream.bin events.jsonl exit_frames.csv safety_clips.csv manifest.json run_report.json anomaly_summary.json anomaly_summary.txt console.log; do [ -s "$R/$f" ] || { echo "missing $f"; exit 1; }; done'
  check "trace.csv columns: observation-side state, raw action, history action, reference, PD target, final cmd, actual q/dq/tau, IMU, timing" 'h=$(head -1 "$R/trace.csv"); for c in "q:left_hip_pitch_joint" "dq:left_hip_pitch_joint" "tau_est:left_hip_pitch_joint" "action:left_hip_pitch_joint" "action_prev:left_hip_pitch_joint" "q_policy:left_hip_pitch_joint" "q_pd:left_hip_pitch_joint" "q_cmd:left_hip_pitch_joint" "tau_pd:left_hip_pitch_joint" "kp:left_hip_pitch_joint" "ref_jp:left_hip_pitch_joint" "ref_jv:left_hip_pitch_joint" quat_w gyro_x roll pitch yaw rpy_roll dt_ms infer_ms step_ms sync_wait_ms send_us dds_loop_ms state_age_ms state_age_send_ms seq_delta tick_delta ls_lost ls_gap_ms flags clip_n ref_frame ref_rq_w ref_rp_x; do echo ",$h," | grep -q ",$c," || { echo "missing column $c"; exit 1; }; done'
  check "logtool check: files well-formed, rows consistent, time order, frozen hashes" 'ltool check "$R" >"$LG/check.out" 2>&1; grep -q "LOGCHECK PASS" "$LG/check.out"'
  check "bit-exact policy replay: obs.bin through the ONNX model reproduces every logged action (tolerance 1e-6)" 'ltool replay "$R" > "$LG/replay.out" 2>&1; grep -q "REPLAY PASS" "$LG/replay.out"'
  echo "  [info] $(cat "$LG/replay.out" | cut -c1-200)"
  check "manifest hashes == sha256sum of the real files (policy, params, motion, executable) and frozen flags" '[ "$(jq -r .sha256.policy_onnx "$R/manifest.json")" = "$(sha256sum "$HERE/assets/policy.onnx" | cut -d" " -f1)" ] && [ "$(jq -r .sha256.motion_bin "$R/manifest.json")" = "$(sha256sum "$HERE/assets/motion.bin" | cut -d" " -f1)" ] && [ "$(jq -r .sha256.policy_params "$R/manifest.json")" = "$(sha256sum "$HERE/assets/policy_params.txt" | cut -d" " -f1)" ] && [ "$(jq -r .sha256.executable "$R/manifest.json")" = "$(sha256sum "$EXE" | cut -d" " -f1)" ] && [ "$(jq -r .sha256.site_conf "$R/manifest.json")" = "$(sha256sum "$HERE/config/site.conf" | cut -d" " -f1)" ] && [ "$(jq -r ".policy_frozen_ok and .params_frozen_ok and .motion_frozen_ok" "$R/manifest.json")" = true ] && [ "$(jq -r .program_version "$R/manifest.json")" = V1.6 ] && [ "$(jq -r .policy_name "$R/manifest.json")" = robust_d_it12000 ]'
  check "run_report carries the same hashes + version + mode, handover states (pre take-over / take-over)" '[ "$(rrj .sha256.policy_onnx)" = "$(jq -r .sha256.policy_onnx "$R/manifest.json")" ] && [ "$(rrj .program_version)" = V1.6 ] && [ "$(rrj .handover.pre_takeover_state.tilt_rad)" != null ] && [ "$(rrj .handover.take_over_state.tilt_rad)" != null ]'
  check "run_report timing (dt / inference / state age / send / DDS loopback), lowstate loss, log stats present; no dropped log records" '[ "$(rrj .timing.dt_ms.p99)" != null ] && [ "$(rrj .timing.infer_ms.max)" != null ] && [ "$(rrj .timing.state_age_ms.max)" != null ] && [ "$(rrj .timing.dds_loopback_ms.max)" != null ] && [ "$(rrj .timing.lowstate_lost_est)" != null ] && [ "$(rrj .logging.dropped.ticks)" = 0 ] && [ "$(rrj .logging.dropped.states)" = 0 ] && [ "$(rrj .logging.complete)" = true ]'
  check "pre-take-over state >= 3.5 s before the first tick, post-exit state >= 1.0 s after the last tick (state_stream.bin)" 'awk -v a="$(rrj .logging.state_preroll_s)" -v b="$(rrj .logging.state_post_exit_s)" "BEGIN{exit !(a>=3.5 && b>=1.0)}"'
  check "state_stream has the full lowstate rate (> 200 samples / s over the recorded span)" 'n=$(rrj .logging.rows.state_stream); span=$(awk -v a="$(rrj .logging.state_stream_t_first)" -v b="$(rrj .logging.state_stream_t_last)" "BEGIN{print b-a}"); awk -v n=$n -v s=$span "BEGIN{exit !(s>4 && n/s>200)}"'
  check "obs.bin rows == trace rows == control ticks; 1442 inputs per tick" '[ "$(rrj .logging.rows.obs)" = "$(rrj .logging.rows.trace)" ] && [ "$(rrj .logging.rows.trace)" = "$(rrj .control_ticks)" ] && [ "$(jq -r .obs_dim "$R/manifest.json")" = 1442 ]'
  check "events.jsonl: manifest, policy_reset (initial state for Sim2Sim), pre_takeover_state, RPCs, FSM changes, exit steps, post_exit_end, log_stats" 'for k in manifest policy_reset pre_takeover_state rpc fsm_change SwitchToUserCtrl user_control_confirmed EXIT_TRIGGER exit_submit exit_rpc_attempt exit_rpc_result exit_fsm_matched exit_frames_stopped exit_stable_confirmed exit_result released post_exit_end log_stats state_stream_begin state_stream_end; do [ "$(ev_count "$R" $k)" -ge 1 ] || { echo "no event $k"; exit 1; }; done'
  check "every RPC has request AND response: api / rc / fsm / t_req / t_resp / latency (SwitchToUserCtrl, GetFsmId, SwitchToInternalCtrl)" 'jq -s -e "[.[] | select(.kind==\"rpc\")] | (length>5) and all(.t_req and .t_resp and (.latency_ms!=null) and (.rc!=null)) and (map(.api)|unique|contains([\"GetFsmId\",\"SwitchToInternalCtrl\",\"SwitchToUserCtrl\"]))" "$R/events.jsonl" >/dev/null'
  check "policy_reset event holds q / dq / quat / gyro / yaw offset / q_hold (Sim2Sim initial state)" 'jq -s -e "[.[] | select(.kind==\"policy_reset\")][0] | (.q|length)==29 and (.dq|length)==29 and (.quat_wxyz|length)==4 and (.yaw_offset_wxyz|length)==4 and (.q_hold|length)==29" "$R/events.jsonl" >/dev/null'
  check "exit_frames.csv has HOLD frames of the exit (and ACQUIRE frames), with the measured q next to the commanded q" 'grep -q ",HOLD," "$R/exit_frames.csv" && grep -q ",ACQUIRE," "$R/exit_frames.csv" && ! grep -q ",DAMP," "$R/exit_frames.csv"'
  check "exit timeline in exit_detail: attitude at decision / rpc / confirmation, frames stopped, fsm trace" '[ "$(rrj .exit_detail.attitude_at_decision.tilt_rad)" != null ] && [ "$(rrj .exit_detail.attitude_at_rpc_sent.tilt_rad)" != null ] && [ "$(rrj .exit_detail.attitude_at_fsm_confirmed.tilt_rad)" != null ] && [ "$(rrj ".exit_detail.t_frames_stopped")" != null ] && [ "$(rrj ".exit_detail.fsm_trace|length")" -ge 1 ]'
  check "P5.3 stats keep policy / command / actual / reference apart + clip decomposition + torque vs training" '[ "$(rrj .tracking.policy_vs_reference.rms_rad)" != null ] && [ "$(rrj .tracking.command_vs_reference.rms_rad)" != null ] && [ "$(rrj .tracking.actual_vs_reference.rms_rad)" != null ] && [ "$(rrj .tracking.actual_vs_command.rms_rad)" != null ] && [ "$(rrj .tracking.clip.raw_target_outside_fraction)" != null ] && [ "$(rrj ".tracking.torque_vs_training|length")" -ge 1 ] && [ "$(rrj .attitude_extremes.max_tilt_rad_overall)" != null ]'
  check "anomaly summary: JSON level field, text in Chinese, result line" 'jq -e ".level" "$R/anomaly_summary.json" >/dev/null && grep -q "运行异常摘要" "$R/anomaly_summary.txt" && grep -q "COMPLETED" "$R/anomaly_summary.txt"'
  stop_env
}
scn_log_clips() {
  echo "== S27 safety-clip log: injected out-of-limit target for 10 ticks (after the 0.3 s authority ramp) -> joint, reason, raw, clipped, duration in safety_clips.csv / report / anomaly summary"
  export PM_SIM_INJECT="clip:25"; export PM_SIM_INJECT_JOINT=4
  go_run logclip -- --run-s 2.0 --ref-mode paused
  unset PM_SIM_INJECT PM_SIM_INJECT_JOINT
  check "run finished" 'waitfor "$LG/service.log" "RUN FINISHED rc=0" 40'
  sleep 3.5; R=$(latest_report); R=${R%/}
  J=$(jq -r '.joint_names[4]' "$R/manifest.json"); HI=$(jq -r '.pos_hi[4]' "$R/manifest.json")
  check "safety_clips.csv: >= 10 rows for joint 4 ($J) with reason POS_LIMIT, raw = hi + 1.0, final inside the limit" 'n=$(awk -F, -v j="$J" "NR>1 && \$6==j && \$7 ~ /POS_LIMIT/ && \$3>=25 && \$3<35" "$R/safety_clips.csv" | wc -l); [ "$n" -ge 10 ] && awk -F, -v j="$J" -v hi="$HI" "NR>1 && \$6==j && \$3==30 {r=\$8; f=\$12; exit !(r>hi+0.99 && r<hi+1.01 && f<hi)}" "$R/safety_clips.csv"'
  check "report.safety_clips: episode on $J POS_LIMIT with >= 10 ticks and duration >= 0.2 s (0.02 s/tick), raw_at_max/clipped_at_max recorded" 'jq -e --arg j "$J" "[.safety_clips.episodes[] | select(.joint==\$j and (.reason|test(\"POS_LIMIT\")) and .n_ticks>=10 and .duration_s>=0.199 and .max_excess>0.9 and .raw_at_max!=null and .clipped_at_max!=null)] | length>=1" "$R/run_report.json" >/dev/null'
  check "trace flag: clip_n >= 1 on those ticks (flags & 8), report.timing.ticks_with_clip >= 10" 'awk -F, "NR==1{for(i=1;i<=NF;i++) if(\$i==\"clip_n\") c=i; next} \$c>=1 {k++} END{exit !(k>=10)}" "$R/trace.csv" && [ "$(rrj .timing.ticks_with_clip)" -ge 10 ]'
  check "anomaly summary names the clip (SAFETY_CLIP) with the joint" 'grep -q "SAFETY_CLIP" "$R/anomaly_summary.txt" && grep -q "$J" "$R/anomaly_summary.txt"'
  stop_env
}
scn_log_events() {
  echo "== S28 remote-controller events, FSM changes, exit steps and operator stop are on the record (START x2 during RUN)"
  go_run logev -- --run-s 8.0 --ref-mode paused
  sleep 0.5; dclick
  check "run ended by the operator" 'waitfor "$LG/service.log" "RUN FINISHED rc=10" 30'
  sleep 3.5; R=$(latest_report); R=${R%/}
  check "remote_keys events with the key text (START) + remote_start_double_click during control" '[ "$(ev_count "$R" remote_keys)" -ge 4 ] && [ "$(ev_count "$R" remote_start_double_click)" -ge 1 ] && jq -s -e "[.[] | select(.kind==\"remote_keys\" and (.text|test(\"START\")))] | length>=2" "$R/events.jsonl" >/dev/null'
  check "the F1+F3 / L1+A chords before the run are in the same log (keys with control_active false)" 'jq -s -e "[.[] | select(.kind==\"remote_keys\" and .control_active==false)] | length>=2" "$R/events.jsonl" >/dev/null'
  check "exit steps in time order (t_mono): exit_submit -> exit_rpc_attempt -> exit_rpc_result -> exit_fsm_matched -> exit_stable_confirmed -> exit_result; EXIT_TRIGGER names REMOTE_STOP" 'jq -s -r "[.[] | select(.kind|test(\"^(exit_submit|exit_rpc_attempt|exit_rpc_result|exit_fsm_matched|exit_stable_confirmed|exit_result)$\"))] | sort_by(.t_mono) | map(.kind) | join(\" \")" "$R/events.jsonl" | grep -q "^exit_submit exit_rpc_attempt exit_rpc_result exit_fsm_matched exit_stable_confirmed exit_result$" && jq -s -e "[.[]|select(.kind==\"EXIT_TRIGGER\" and .fault==\"REMOTE_STOP\")]|length==1" "$R/events.jsonl" >/dev/null'
  check "FSM changes recorded: 802 -> 1000 (user) -> 802 (WalkRun)" 'jq -s -r "[.[] | select(.kind==\"fsm_change\") | .to] | join(\",\")" "$R/events.jsonl" | grep -q "802,1000,802"'
  check "report: exit reason REMOTE_STOP, WalkRun recovery recorded, the service timeline has the START x2 and the voice" '[ "$(rrj .summary.result)" = EXITED_WALKRUN ] && jq -s -e "[.[]|select(.kind==\"voice\")]|length>=3" "$R/service_events.jsonl" >/dev/null && jq -s -e "[.[]|select(.kind==\"keys\" and (.text|test(\"START\")))]|length>=2" "$R/service_events.jsonl" >/dev/null'
  stop_env
}
scn_log_async() {
  echo "== S29 asynchronous logging: the log writer stalls for 1.5 s (disk stall) in the middle of a run -> control loop unaffected, nothing lost"
  export PM_SIM_LOG_STALL=1.5
  start_env logasync -- --run-s 4.0 --ref-mode paused
  unset PM_SIM_LOG_STALL
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "run finished" 'waitfor "$LG/service.log" "RUN FINISHED rc=0" 40'
  sleep 3.5; R=$(latest_report); R=${R%/}
  check "writer stall happened (log line) while control continued" 'grep -q "writer stalled for 1.5 s" "$R/runner.out" "$R/console.log" 2>/dev/null'
  check "control period unaffected: dt p99 < 22 ms, max < 30 ms, no LATE ticks, no WATCHDOG/LOOP_OVERRUN" 'awk -v p="$(rrj .timing.dt_ms.p99)" -v m="$(rrj .timing.dt_ms.max)" "BEGIN{exit !(p<22 && m<30)}" && [ "$(rrj .timing.late_ticks)" = 0 ] && [ "$(rrj .summary.result)" = COMPLETED ]'
  check "nothing lost: 0 dropped ticks, trace rows == obs rows == ticks" '[ "$(rrj .logging.dropped.ticks)" = 0 ] && [ "$(rrj .logging.rows.trace)" = "$(rrj .control_ticks)" ] && [ "$(rrj .logging.rows.obs)" = "$(rrj .control_ticks)" ]'
  check "the writer lag of the stall is visible in the report (writer_max_lag_ms > 1000)" 'awk -v l="$(rrj .logging.writer_max_lag_ms)" "BEGIN{exit !(l>1000)}"'
  stop_env
}
scn_log_crash() {
  echo "== S30 abnormal exit: runner SIGSEGV / kill -9 in control -> log files preserved + complete, run_report salvaged, CRASH marker, guardian take-over has its own log"
  export PM_SIM_INJECT="segv:30"
  go_run logsegv -- --run-s 8.0 --ref-mode paused
  unset PM_SIM_INJECT
  check "guardian took over the crashed runner" 'waitfor "$(guardian_log)" "TAKE-OVER" 20'
  check "service salvaged the run report" 'waitfor "$LG/service.log" "salvaged run_report.json" 30'
  sleep 5; R=$(latest_report); R=${R%/}
  check "CRASH marker (SIGSEGV) in events.jsonl; ~30 ticks recorded in trace/obs/state_stream up to the crash" '[ "$(ev_count "$R" CRASH)" -ge 1 ] && grep -q SIGSEGV "$R/events.jsonl" && n=$(( $(wc -l < "$R/trace.csv") - 1 )) && [ "$n" -ge 28 ] && [ "$n" -le 31 ] && [ "$(jq -r .logging.rows.obs "$R/run_report.json")" = "$n" ]'
  check "run_report.json: CRASH_NO_REPORT, reconstructed_from_logs, mode/run_s from the manifest, anomaly FAIL (PROCESS_CRASH)" '[ "$(jq -r .result "$R/run_report.json")" = CRASH_NO_REPORT ] && [ "$(jq -r .reconstructed_from_logs "$R/run_report.json")" = true ] && [ "$(jq -r .run_s "$R/run_report.json")" = 8 ] && grep -q PROCESS_CRASH "$R/anomaly_summary.txt"'
  check "logtool check: the surviving files are well-formed (rows equal in trace/obs)" 'ltool check "$R" > "$LG/check.out" 2>&1; ! grep -q "FAIL.*obs_rows_match_trace\|FAIL.*trace" "$LG/check.out"'
  check "replay of the crashed run's ticks is still bit-exact" 'ltool replay "$R" | grep -q "REPLAY PASS"'
  check "guardian take-over log directory with events (take_over, rpc, exit steps), HOLD frames, lowstate stream, report; linked from the crashed run" 'G=$(ls -d "$LG"/guardian/takeover_*/ | tail -1); [ "$(ev_count "${G%/}" take_over)" = 1 ] && [ "$(ev_count "${G%/}" exit_result)" = 1 ] && grep -q ",HOLD," "${G%/}/exit_frames.csv" && [ -s "${G%/}/state_stream.bin" ] && [ -s "${G%/}/run_report.json" ] && [ -e "$R/guardian_takeover" ] && [ -s "$R/guardian_report.json" ]'
  check "WalkRun recovery by the guardian confirmed (fsm 802) and announced" 'sim_has "SwitchToInternalCtrl mode=2(WALKRUN) -> fsm 802" && sim_has "TTS.*异常已交还"'
  stop_env
  echo "  -- kill -9 variant (no handler can run): files up to the last 10 ms survive"
  go_run logkill -- --run-s 8.0 --ref-mode paused
  sleep 0.6
  RP=$(pgrep -f "partymimic runner --iface lo" | head -1); kill -9 "$RP"
  check "salvaged after kill -9" 'waitfor "$LG/service.log" "salvaged run_report.json" 40'
  sleep 4; R=$(latest_report); R=${R%/}
  check "trace/obs/events survived kill -9, rows equal, report salvaged" 'n=$(( $(wc -l < "$R/trace.csv") - 1 )) && [ "$n" -ge 20 ] && [ "$(jq -r .logging.rows.obs "$R/run_report.json")" = "$n" ] && [ "$(jq -r .result "$R/run_report.json")" = CRASH_NO_REPORT ] && ltool replay "$R" | grep -q "REPLAY PASS"'
  stop_env
}
scn_modes() {
  echo "== S31 test modes through the REAL entry ./start.sh --mode: modes are INDEPENDENT (no order), check-only is never armed, voice of the mode, history is record only, status.sh, export_logs.sh"
  start_sim_only modes
  run_start_sh --mode check; rc=$?
  check "mode check: started WITHOUT any authorisation flag, 'never armed', mode named in service.log, voice 仅检查" '[ $rc -eq 0 ] && grep -q "mode check" "$LG/start.out" && grep -q "test mode: check" "$LG/service.log" && sim_has "TTS.*仅检查"'
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30
  check "F1+F3 in check mode: READY reached, runner finished CHECK_PASSED (read-only), voice 准备就绪" 'waitfor "$LG/service.log" "RUN FINISHED\|runner ended before taking control" 20 && sim_has "TTS.*准备就绪"'
  sleep 1; R=$(latest_report)
  check "check report: CHECK_PASSED, took_control false, no CRASH_NO_REPORT, anomaly not FAIL" '[ "$(jq -r .result "${R}run_report.json")" = CHECK_PASSED ] && [ "$(jq -r .summary.took_control "${R}run_report.json")" = false ] && ! grep -q CRASH_NO_REPORT "${R}run_report.json" && [ "$(jq -r .anomaly.level "${R}run_report.json")" != FAIL ]'
  press $KEY_GO 0.5; sleep 1.0
  check "check mode: L1+A does nothing and the robot is NEVER commanded (no SwitchToUserCtrl, no user_lowcmd frame, no exit RPC)" 'log_has "service is in check mode" && ! sim_has "SwitchToUserCtrl" && ! sim_has "first stiff" && ! sim_has "first damping" && ! sim_has "REQUEST"'
  stop_sh
  echo "  -- independence: dance first, on an EMPTY history, no --skip-order, no predecessor"
  start_sim_only modes2
  run_start_sh --mode dance --yes; rc=$?
  check "mode dance starts at once (rc 0), no 'requires' / order refusal anywhere, voice 完整舞蹈" '[ $rc -eq 0 ] && ! grep -qi "requires mode\|skip-order\|PASS record" "$LG/start.out" "$LG/service.log" && sim_has "TTS.*完整舞蹈"'
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "dance run (7.1 s window) completed through the real entry" 'waitfor "$LG/service.log" "RUN FINISHED rc=0" 60'
  sleep 3.5
  check "history record written (pass true) but nothing consumed it" 'jq -s -e "map(select(.mode==\"dance\" and .pass==true)) | length==1" "$LG/mode_pass.jsonl" >/dev/null'
  stop_sh
  echo "  -- history FAIL does not block: a FAIL record for hold, then hold starts"
  start_sim_only modes3
  echo '{"wall":1,"mode":"hold","result":"WALKRUN_RECOVERY_FAILED","pass":false}' > "$LG/mode_pass.jsonl"
  out=$(PM_BIN="$EXE" PM_LOGS="$LG" PM_ROOT="$HERE" bash "$HERE/start.sh" --list < /dev/null 2>&1)
  check "--list shows the history (上次FAIL) for hold, all 6 modes selectable" 'echo "$out" | grep -q "上次FAIL" && [ "$(echo "$out" | grep -c "^  [1-6]  ")" = 6 ]'
  run_start_sh --mode hold --yes; rc=$?
  check "hold starts although its last record is FAIL; armed with YES only (validated envelope: no second phrase)" '[ $rc -eq 0 ] && grep -q "READY TO USE (mode hold" "$LG/start.out" && grep -q "armed=1\|armed true\|NOT ARMED" /dev/null || grep -q "ARMED (mode hold, envelope validated)" "$LG/start.out"'
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "hold run finished COMPLETED (3 s window)" 'waitfor "$LG/service.log" "RUN FINISHED rc=0" 40'
  sleep 3.5
  check "voice 执行完成 said BEFORE the post-exit recording ended (REPORTED phase), run_report names the mode" 'R=$(latest_report); [ "$(jq -r .mode "${R}run_report.json")" = hold ] && jq -s -e "[.[]|select(.kind==\"voice\" and .text==\"执行完成\")]|length==1" "$LG/service_events.jsonl" >/dev/null'
  out=$(PM_LOGS="$LG" PM_ROOT="$HERE" bash "$HERE/status.sh" 2>&1)
  check "status.sh: service RUNNING, last result, mode table" 'echo "$out" | grep -q "service: RUNNING" && echo "$out" | grep -q "COMPLETED" && echo "$out" | grep -q "hold"'
  stop_sh
  echo "  -- export (all logs of the run in one archive)"
  out=$(PM_BIN="$EXE" PM_LOGS="$LG" bash "$HERE/export_logs.sh" "$LG/export" 2>&1); rc=$?
  A=$(ls "$LG"/export/*.tar.gz 2>/dev/null | head -1); tar -tzf "$A" > "$LG/export.list" 2>/dev/null
  check "export_logs.sh: one archive + sha256, INDEX.txt, runs, service log, mode records, config" '[ $rc -eq 0 ] && [ -n "$A" ] && ( cd "$LG/export" && sha256sum -c --quiet *.tar.gz.sha256 ) && grep -q "INDEX.txt" "$LG/export.list" && grep -q "logs/runs/.*/trace.csv" "$LG/export.list" && grep -q "logs/runs/.*/state_stream.bin" "$LG/export.list" && grep -q "service.log" "$LG/export.list" && grep -q "mode_pass.jsonl" "$LG/export.list" && grep -q "config/modes.conf" "$LG/export.list" && grep -q MANIFEST.sha256 "$LG/export.list"'
  out=$(PM_BIN="$EXE" PM_LOGS="$LG" bash "$HERE/export_logs.sh" "$LG/export_l" --light 2>&1); A2=$(ls "$LG"/export_l/*.tar.gz | head -1); tar -tzf "$A2" > "$LG/export_l.list" 2>/dev/null
  check "--light leaves out obs.bin / state_stream.bin but keeps trace + reports" '! grep -q "obs.bin" "$LG/export_l.list" && ! grep -q "state_stream.bin" "$LG/export_l.list" && grep -q "run_report.json" "$LG/export_l.list"'
}

dry() { PM_BIN="$EXE" PM_LOGS=/tmp/pm_logs_dry PM_ROOT="$HERE" bash "$HERE/start.sh" --dry-run --iface eth0 "$@" < /dev/null 2>&1 | grep "^DRY-RUN"; }
scn_settle() {
  echo "== S33 (P5.3) service state machine: recoverable exits (clean end, operator START x2, recoverable fault) settle to IDLE WITHOUT ack.sh, a second F1+F3 / L1+A run works; unrecoverable exits stay in HOLD; no automatic take-over"
  echo "  -- (a) clean end -> SETTLE -> IDLE, then a SECOND run in the same service"
  start_env settle_a -- --run-s 1.0
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "run 1 finished (rc=0), service settles (no ack needed) and says it is ready again" 'waitfor "$LG/service.log" "RUN FINISHED rc=0" 40 && waitfor "$LG/service.log" "SETTLED: WalkRun confirmed" 30 && sim_has "TTS.*可再次开始"'
  check "no HOLD, no CRITICAL, no ack file needed; state.json IDLE" '! log_has "HOLD:" && ! log_has "CRITICAL" && [ ! -e "$LG/ACK" ] && waitfor "$LG/state.json" "\"state\":\"IDLE\"" 5'
  check "guardian of run 1 ended normally, a new guardian is ready before IDLE (heartbeat fresh)" 'jq -e ".guardian_alive == true" "$LG/state.json" >/dev/null'
  sleep 3.5
  R1=$(latest_report); R1=${R1%/}
  check "run 1 report is complete although the guardian ended early (post-exit recording finished: log_stats, post-exit state >= 1.5 s)" '[ "$(ev_count "$R1" log_stats)" = 1 ] && awk -v b="$(jq -r .logging.state_post_exit_s "$R1/run_report.json")" "BEGIN{exit !(b>=1.5)}" && [ "$(jq -r .summary.result "$R1/run_report.json")" = COMPLETED ]'
  press $KEY_LOAD 0.5
  check "second F1+F3 accepted straight away (READY again), no ack" 'waitn "$LG/service.log" "READY: model loaded" 2 30'
  press $KEY_GO 0.5
  check "second run takes control only after the second L1+A and COMPLETES" 'waitn "$LG/service.log" "RUN FINISHED rc=0" 2 40 && [ "$(grep -c "GO issued" "$LG/service.log")" = 2 ]'
  sleep 3.5
  R2=$(latest_report); R2=${R2%/}
  check "run 2 started CLEAN: policy_state_reset clean (reference index 0, previous action 0, history identical), tick 0 identical to run 1 (frame, action_prev = 0)" '[ "$R1" != "$R2" ] && jq -s -e "[.[]|select(.kind==\"policy_state_reset\")][0] | .clean==true and .motion_t==0 and .prev_action_abs_max==0" "$R2/events.jsonl" >/dev/null && [ "$(awk -F, "NR==1{for(i=1;i<=NF;i++){if(\$i==\"ssb_frame\")f=i}} NR==2{print \$f}" "$R1/trace.csv")" = "$(awk -F, "NR==1{for(i=1;i<=NF;i++){if(\$i==\"ssb_frame\")f=i}} NR==2{print \$f}" "$R2/trace.csv")" ] && awk -F, "NR==1{for(i=1;i<=NF;i++) if(\$i ~ /^action_prev:/) c[++n]=i} NR==2{for(k=1;k<=n;k++) if(\$c[k]!=0) bad=1; exit bad}" "$R2/trace.csv"'
  check "run 2 bit-exact replay (no state leaked into the policy)" 'ltool replay "$R2" | grep -q "REPLAY PASS"'
  stop_env
  echo "  -- (b) operator START x2 during the run -> SETTLE -> IDLE, then F1+F3 again"
  go_run settle_b -- --run-s 8.0 --ref-mode paused
  sleep 0.5; dclick
  check "operator exit (rc=10) -> settled without ack" 'waitfor "$LG/service.log" "RUN FINISHED rc=10" 30 && waitfor "$LG/service.log" "SETTLED" 30 && ! log_has "HOLD:" && ! log_has "CRITICAL"'
  press $KEY_LOAD 0.5
  check "F1+F3 after the operator exit loads (READY)" 'waitn "$LG/service.log" "READY: model loaded" 2 30'
  stop_env
  echo "  -- (c) recoverable fault (injected inference exception) -> WalkRun confirmed -> SETTLE -> IDLE (no HOLD)"
  export PM_SIM_INJECT="inference:15"
  go_run settle_c -- --run-s 8.0 --ref-mode paused
  unset PM_SIM_INJECT
  check "fault exit rc=10, recoverable: settled, no HOLD, voice 异常已交还 then 可再次开始" 'waitfor "$LG/service.log" "recoverable fault, WalkRun recovery confirmed" 30 && waitfor "$LG/service.log" "SETTLED" 30 && ! log_has "HOLD:" && sim_has "TTS.*异常已交还" && sim_has "TTS.*可再次开始"'
  press $KEY_LOAD 0.5
  check "F1+F3 loads again after the fault exit (no ack.sh)" 'waitn "$LG/service.log" "READY: model loaded" 2 30'
  stop_env
  echo "  -- (d) the attitude gate: after the run the robot is NOT quiet (tilt 0.3 rad) -> stays in SETTLE (F1+F3 ignored with the reason) until it is quiet"
  go_run settle_d -- --run-s 2.0 --ref-mode paused
  echo "0.20 0.0" > "$SD/tilt"     # 0.20 rad: not unstable (< 0.30) but not quiet (>= 0.15)
  waitfor "$LG/service.log" "RUN FINISHED rc=0" 40; sleep 2.5
  press $KEY_LOAD 0.4
  check "while tilted: stays in SETTLE (no SETTLED), F1+F3 ignored naming the missing condition (attitude quiet)" '! log_has "SETTLED" && log_has "F1+F3 ignored: the last run just ended, waiting until" && grep -q "attitude is quiet" "$LG/service.log" && ! log_has "HOLD:"'
  rm -f "$SD/tilt"
  check "attitude quiet again -> SETTLED -> IDLE (still no HOLD)" 'waitfor "$LG/service.log" "SETTLED" 20 && ! log_has "HOLD:"'
  stop_env
  echo "  -- (e) V1.6.1 UNRECOVERABLE end (fallen robot): no HOLD, no ack: the service settles to IDLE by itself; the next F1+F3 admission decides"
  go_run settle_e --flip-after-user-s 0.4 -- --run-s 8.0 --ref-mode paused
  check "fallen -> rc=13 ABNORMAL end, no HOLD, settled to IDLE by itself" 'waitfor "$LG/service.log" "RUN FINISHED rc=13" 40 && log_has "ABNORMAL end" && waitfor "$LG/service.log" "SETTLED (abnormal end" 40 && ! log_has "HOLD:" && waitfor "$LG/state.json" "\"state\":\"IDLE\"" 5'
  press $KEY_LOAD 0.4
  check "F1+F3 is accepted again in IDLE (no ack); the fallen robot is then judged by the admission" 'waitn "$LG/service.log" "chord F1+F3 detected" 1 10 && ! log_has "F1+F3 ignored in state"'
  stop_env
}
scn_leadin() {
  echo "== S34 (P5.3) running modes replay the clip robust_d was trained on (1 s frozen first frame + SSB): lead-in in memory, motion.bin untouched; --torque-ff option logged and CRC-valid"
  start_env leadin -- --run-s 1.0 --ref-mode running
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "run finished" 'waitfor "$LG/service.log" "RUN FINISHED rc=0" 60'
  sleep 3.5; R=$(latest_report); R=${R%/}
  check "manifest: lead_in_frames 50, motion_frames 354 (file untouched, frozen hash ok), report policy_state_reset clip_frames 404" '[ "$(jq -r .lead_in_frames "$R/manifest.json")" = 50 ] && [ "$(jq -r .motion_frames "$R/manifest.json")" = 354 ] && [ "$(jq -r .motion_frozen_ok "$R/manifest.json")" = true ] && jq -s -e "[.[]|select(.kind==\"policy_state_reset\")][0].clip_frames==404" "$R/events.jsonl" >/dev/null'
  check "reference: frozen first frame during the lead-in (ref_jp identical for ticks 0..40), the dance moves afterwards; window = lead-in 1 s + 1.0 s dance + 0.5 s hold (~125 ticks)" 'awk -F, "NR==1{for(i=1;i<=NF;i++) if(\$i==\"ref_jp:left_knee_joint\") c=i; next} {v[NR-1]=\$c} END{ok=1; for(k=2;k<=41;k++) if(v[k]!=v[1]) ok=0; if(v[90]==v[1]) ok=0; exit !ok}" "$R/trace.csv" && n=$(( $(wc -l < "$R/trace.csv") - 1 )) && [ "$n" -ge 118 ] && [ "$n" -le 132 ]'
  check "bit-exact replay with the lead-in clip" 'ltool replay "$R" | grep -q "REPLAY PASS"'
  stop_env
  echo "  -- torque feed-forward option: --torque-ff (default OFF)"
  start_env ff -- --run-s 1.0 --ref-mode paused --torque-ff
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "run finished with --torque-ff, robot-side CRC/mode valid" 'waitfor "$LG/service.log" "RUN FINISHED rc=0" 60 && ! sim_has "BAD"'
  sleep 3.5; R=$(latest_report); R=${R%/}
  check "manifest guard_torque_ff true; trace tau_ff non-zero where the target was clipped; tau_train logged; q_cmd still inside the joint range" '[ "$(jq -r .guard_torque_ff "$R/manifest.json")" = true ] && awk -F, "NR==1{for(i=1;i<=NF;i++){if(\$i ~ /^tau_ff:/) f[++n]=i; if(\$i==\"clip_n\") cn=i}} NR>1{if(\$cn>0) for(k=1;k<=n;k++) if(\$f[k]!=0) nz=1} END{exit !nz}" "$R/trace.csv" && grep -q "tau_train:left_ankle_pitch_joint" "$R/trace.csv" && ltool replay "$R" | grep -q "REPLAY PASS"'
  stop_env
  start_env noff -- --run-s 1.0 --ref-mode paused
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  waitfor "$LG/service.log" "RUN FINISHED rc=0" 60; sleep 3.5; R=$(latest_report); R=${R%/}
  check "default: guard_torque_ff false and tau_ff all zero" '[ "$(jq -r .guard_torque_ff "$R/manifest.json")" = false ] && awk -F, "NR==1{for(i=1;i<=NF;i++) if(\$i ~ /^tau_ff:/) f[++n]=i} NR>1{for(k=1;k<=n;k++) if(\$f[k]!=0) nz=1} END{exit nz}" "$R/trace.csv"'
  stop_env
}

# ============================================================================================================================ V1.5.1
ev_ms() {   # ms between the first service event matching $1 (state_note msg regex) and the first runner_phase event $2, in service_events.jsonl
  jq -s -r --arg a "$1" --arg b "$2" '([.[]|select(.kind=="state_note" and (.msg|test($a)))]|last|.t_mono) as $x | ([.[]|select(.kind=="runner_phase" and .phase==$b)]|last|.t_mono) as $y | (($y-$x)*1000|floor)' "$LG/service_events.jsonl" 2>/dev/null; }
sb_pid() { grep -o "standby runner started pid [0-9]*" "$LG/service.log" | tail -1 | awk '{print $5}'; }
scn_standby() {
  echo "== S35 (V1.5.1) pre-warmed runner: model + warm-up + DDS monitors at service start-up; F1+F3 only triggers the real-time checks (READY < 1 s instead of ~3 s); every run is still a fresh process"
  start_env standby -- --run-s 1.0 --ref-mode paused
  check "standby runner started at service start-up and reached phase STANDBY BEFORE any key (model loaded, warm-up done)" 'waitfor "$LG/service.log" "standby runner started" 5 && waitfor "$LG/standby.out" "=== phase STANDBY$" 20 && grep -q "warm-up: 8 read-only inferences" "$LG/standby.out"'
  check "standby: no lock, no command writer, no RPC (no SwitchToUserCtrl / user_lowcmd frame / FSM RPC reached the robot), state.json standby=ready" '! sim_has "SwitchToUserCtrl" && ! sim_has "first stiff" && ! sim_has "first damping" && jq -e ".standby==\"ready\"" "$LG/state.json" >/dev/null'
  P1=$(sb_pid)
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30
  L1=$(ev_ms "chord F1\\+F3 detected" READY); echo "  [info] F1+F3 -> READY = ${L1} ms (pre-warmed)"
  check "F1+F3 -> READY in < 1000 ms; the pre-warmed runner (same pid) was triggered, not a new process" '[ "$L1" -ge 0 ] && [ "$L1" -lt 1000 ] && log_has "pre-warmed runner pid $P1 triggered"'
  R=$(latest_report); R=${R%/}
  press $KEY_GO 0.5; waitfor "$LG/service.log" "RUN FINISHED rc=0" 60; sleep 3.5
  check "run through the pre-warmed runner: COMPLETED, manifest standby_runner true, the frozen artifacts re-verified at the trigger (artifact_rehash_at_trigger PASS), policy state clean" '[ "$(jq -r .result "$R/run_report.json")" = COMPLETED ] && [ "$(jq -r .standby_runner "$R/manifest.json")" = true ] && grep -q "artifact_rehash_at_trigger: full sha256" "$R/console.log" && grep -q "\"clean\":true" "$R/events.jsonl"'
  check "run statistics start at the trigger (lowstate monitor reset): rx count in log_stats is the run only, not the minutes of standby" 'jq -r "select(.kind==\"log_stats\")|.lowstate_rx" "$R/events.jsonl" | tail -1 | awk "{exit !(\$1 > 0 && \$1 < 12000)}"'
  check "after the run: SETTLED -> IDLE, a NEW standby runner (different pid) is pre-warmed again" 'waitfor "$LG/service.log" "SETTLED" 20 && waitfor "$LG/service.log" "standby runner started pid" 10 && [ "$(grep -c "standby runner started pid" "$LG/service.log")" -ge 2 ] && [ "$(sb_pid)" != "$P1" ]'
  waitfor "$LG/standby.out" "=== phase STANDBY$" 20; sleep 0.5
  echo "  -- repeated trigger + START x2 in IDLE (stale remote events must not abort the next run)"
  dclick; sleep 0.3; dclick; sleep 0.4
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30
  check "START x2 pressed while IDLE (standby alive): the next F1+F3 still reaches READY and the run is not aborted before ARM" 'log_has "START x2 in IDLE" && [ "$(grep -c "READY: model loaded" "$LG/service.log")" -ge 2 ]'
  press $KEY_GO 0.5
  check "second run COMPLETED too (the trigger -> GO path is repeatable)" 'waitn "$LG/service.log" "RUN FINISHED rc=0" 2 60'
  sleep 3.5; waitn "$LG/service.log" "SETTLED" 2 20; waitn "$LG/service.log" "standby runner started pid" 3 15; waitfor "$LG/standby.out" "=== phase STANDBY$" 20; sleep 0.5
  echo "  -- START x2 in READY cancels (the pre-warmed runner is a fresh process every time)"
  press $KEY_LOAD 0.5; waitn "$LG/service.log" "READY: model loaded" 3 30; ST0=$(sim_count "first stiff user_lowcmd frame"); dclick
  check "START x2 in READY: cancelled before control (no new stiff frame), service back to IDLE, the next standby runner started" 'waitn "$LG/service.log" "ended before taking control" 1 15 && waitn "$LG/service.log" "standby runner started pid" 4 15 && [ "$(sim_count "first stiff user_lowcmd frame")" = "$ST0" ] && ! log_has "HOLD:"'
  waitfor "$LG/standby.out" "=== phase STANDBY$" 20; sleep 0.5
  echo "  -- standby crash: kill -9 while IDLE -> logged, restarted; F1+F3 right after the kill uses the COLD path and still works"
  K=$(sb_pid); kill -9 "$K" 2>/dev/null; sleep 0.3
  press $KEY_LOAD 0.5
  check "standby killed: service logs it (control never touched), F1+F3 -> cold runner -> READY" 'waitfor "$LG/service.log" "standby runner ended after" 5 && waitfor "$LG/service.log" "runner started (cold" 5 && waitfor "$LG/service.log" "READY: model loaded" 30'
  dclick; sleep 1.0
  echo "  -- the triggered runner is killed (abnormal exit before control): service continues, IDLE, F1+F3 allowed again"
  waitfor "$LG/service.log" "standby runner started pid" 15; waitfor "$LG/standby.out" "=== phase STANDBY$" 20; sleep 0.5
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; K=$(sb_pid); ST1=$(sim_count "first stiff user_lowcmd frame"); NE=$(grep -c "runner ended before taking control" "$LG/service.log"); kill -9 "$K" 2>/dev/null
  check "runner killed in READY (rc 137): service back to IDLE (not HOLD), nothing was sent, a new pre-warmed runner follows" 'waitn "$LG/service.log" "runner ended before taking control" $((NE+1)) 15 && ! log_has "HOLD:" && [ "$(sim_count "first stiff user_lowcmd frame")" = "$ST1" ]'
  waitfor "$LG/service.log" "standby runner started pid" 15; waitfor "$LG/standby.out" "=== phase STANDBY$" 25; sleep 0.5
  press $KEY_LOAD 0.5
  check "after the crash F1+F3 works again (new pre-warmed runner -> READY)" 'waitfor "$LG/service.log" "READY: model loaded" 30 && [ "$(ev_ms "chord F1\\+F3 detected" READY)" -lt 1500 ]'
  stop_env
  echo "  -- A/B: --no-standby (cold start at F1+F3: model load + warm-up + 1 s traffic windows)"
  start_env standby_cold -- --run-s 1.0 --ref-mode paused --no-standby
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30
  L2=$(ev_ms "chord F1\\+F3 detected" READY); echo "  [info] F1+F3 -> READY = ${L2} ms (cold, --no-standby) vs ${L1} ms (pre-warmed)"
  check "--no-standby: no standby runner at all, the cold runner works; it is slower than the pre-warmed one by > 500 ms" '! log_has "standby runner started" && log_has "runner started (cold" && [ "$L2" -gt $((L1 + 500)) ]'
  stop_env
}
scn_dds() {
  echo "== S36 (V1.5.1) DDS entity lifecycle: root cause of the field crash reproduced, the fix stress-tested against hot rt/lowcmd + rt/lowstate traffic (loopback robot)"
  kill_test_procs -9; sleep 0.5; SD="/tmp/pm_sim_dds"; rm -rf "$SD"; mkdir -p "$SD"; echo 0 > "$SD/keys"
  "$SIM" --dir "$SD" > "$SD/sim.out" 2>&1 & SIMPID=$!; sleep 1.0
  ST="$B/dev/dds_stress"
  out=$(timeout 120 "$ST" --mechanism 1000 --iters 100 2>&1); rc=$?
  if echo "$out" | grep -q "prevent_callbacks.*Assertion"; then ok "ROOT CAUSE reproduced: destroying a reader with a data_available listener while callbacks are in flight aborts in EntityDelegate::prevent_callbacks() (rc $rc = SIGABRT) - the same assertion as the field log"
  else echo "  [info] mechanism demo did not abort here (rc=$rc): this CycloneDDS build has assertions disabled; the field robot's build has them enabled"; fi
  out=$(timeout 300 "$ST" --iters 600 --window 0.05 --link-cycles 60 2>&1); rc=$?
  check "Link::measure_traffic / discover_writers x600 in two threads + 60 Link create/close/destroy cycles against hot traffic: no crash, entities never destroyed (rc 0)" '[ $rc -eq 0 ] && echo "$out" | grep -q "STRESS OK: 600" && echo "$out" | grep -q "LINK CYCLES OK: 60" && ! echo "$out" | grep -q "Assertion"'
  check "rolling history after warm-up reports the factory rt/lowcmd writer traffic (~500 Hz) and lowstate rate (monitors are live)" 'echo "$out" | grep -q "writers(lowcmd)=1" && echo "$out" | grep -q "lowstate [0-9]* Hz"'
  kill -TERM "$SIMPID" 2>/dev/null; sleep 0.5; kill -9 "$SIMPID" 2>/dev/null; wait 2>/dev/null
}
scn_timeline() {
  echo "== S37 (V1.5.1) full-clip timeline on the loopback robot: lead-in frozen first frame + the WHOLE clip + last frame held; nothing cut by the runner fail-safe (TL_RUN_S / PM_ROOT_OVERRIDE select another package)"
  RS="${TL_RUN_S:-7.1}"
  start_env timeline -- --run-s "$RS" --ref-mode running
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "run finished COMPLETED (window $RS s)" 'waitfor "$LG/service.log" "RUN FINISHED rc=0" $(( ${TL_WAIT:-90} ))'
  sleep 3.5; R=$(latest_report); R=${R%/}
  LI=$(jq -r .lead_in_frames "$R/manifest.json"); MF=$(jq -r .motion_frames "$R/manifest.json"); PL=$(jq -r .planned_total_s "$R/manifest.json"); FS=$(jq -r .failsafe_s "$R/manifest.json")
  MAXF=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="ssb_frame") c=i} NR>1{if($c+0>m)m=$c+0} END{print m}' "$R/trace.csv")
  HOLDN=$(awk -F, -v last=$((LI+MF-1)) 'NR==1{for(i=1;i<=NF;i++) if($i=="ssb_frame") c=i} NR>1 && $c+0==last{n++} END{print n+0}' "$R/trace.csv")
  FIRSTN=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i ~ /^ref_jp:/) f[++n]=i; next} NR==2{for(k=1;k<=n;k++) b[k]=$f[k]} NR>1{for(k=1;k<=n;k++) if (($f[k]-b[k])^2 > 1e-12) {print NR-2; exit}} END{}' "$R/trace.csv")
  TICKS=$(jq -r .control_ticks "$R/run_report.json")
  echo "  [info] lead-in $LI frames, clip $MF frames, last frame index $((LI+MF-1)); reference reached frame $MAXF; last frame held for $HOLDN ticks; first frame held for $FIRSTN ticks; $TICKS ticks, planned $PL s, fail-safe $FS s"
  check "the reference reached the LAST frame of lead-in + clip (index $((LI+MF-1))): the dance was not cut" '[ "$MAXF" = "$((LI+MF-1))" ]'
  check "last frame held ~0.5 s (end_hold_s) before the hand-back: 25 +- 2 ticks at 50 Hz" '[ "$HOLDN" -ge 23 ] && [ "$HOLDN" -le 28 ]'
  check "first frame frozen for the lead-in (>= $LI ticks on frame 0/1)" '[ "$FIRSTN" -ge $((LI - 1)) ]'
  check "total control ticks match the plan (planned $PL s +- 0.3 s) and stay below the fail-safe $FS s" 'awk -v t=$TICKS -v p=$PL -v f=$FS "BEGIN{d=t*0.02; exit !(d > p-0.3 && d < p+0.3 && d < f)}"'
  check "manifest records covers_whole_dance (running full window) and the runner logged the timeline line" '[ "$(jq -r .covers_whole_dance "$R/manifest.json")" = true ] && grep -q "timeline: lead-in" "$R/console.log" || grep -q "timeline: lead-in" "$LG/standby.out"'
  stop_env
}
# ---------------------------------------------------------------- V1.6 announce_wait: the robot is taken over only after "开始执行" finished + announce_wait_s, and the wait can be cancelled
evt() { jq -r "select(.kind==\"$2\") | .t_mono" "$(latest_report)events.jsonl" 2>/dev/null | head -1; }          # evt <report dir/> <kind>  (first event time)
evt_in() { jq -r "select(.kind==\"$1\") | $2" "$(latest_report)events.jsonl" 2>/dev/null | head -1; }              # evt_in kind jq-expr
sv_has() { grep -q -- "$1" "$LG/service_events.jsonl"; }
# GO prologue for the wait scenarios: F1+F3, READY, L1+A pressed; returns when the announcement phase has begun. args: sim args... -- service args...
wait_prologue() { local name="$1"; shift; start_env "$name" "$@"; press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5; waitfor "$LG/service.log" "GO issued" 10; }
wait_after_announce() { waitfor "$LG/service.log" "announcement finished (" 10; sleep "${1:-0.8}"; }   # the runner is now inside announce_wait_s
cancelled_checks() {   # reason
  CR="$1"
  check "[$CR] cancelled BEFORE take-over: rc 7, service back to IDLE, nothing sent (no SwitchToUserCtrl, no user_lowcmd frame)" 'waitfor "$LG/service.log" "take-over CANCELLED" 20 && ! sim_has "SwitchToUserCtrl" && ! sim_has "first stiff" && ! sim_has "first damping"'
  sleep 1
  check "[$CR] run_report: CANCELLED_BEFORE_TAKEOVER, took_control false; events.jsonl has the cancel event with reason $CR" '[ "$(rj .summary.result)" = CANCELLED_BEFORE_TAKEOVER ] && [ "$(rj .summary.took_control)" = false ] && [[ "$(evt_in cancel .reason)" =~ ^($CR)$ ]] && [ "$(evt_in cancel .took_control)" = false ]'
}
scn_v161() {
  echo "== S45 (V1.6.1 hotfix) deployment chain: plain ./start.sh runs the FULL dance with no authorisation / confirmation / prompt; READY is HELD (no timeout); L1+A runs; every end returns to IDLE; repeatable"
  echo "  -- (a) package contents / start.sh options: nothing of the manual-authorisation design is left"
  check "no authorize.sh, no config/authz.conf, no src/authz.cpp in the framework" '[ ! -e "$HERE/authorize.sh" ] && [ ! -e "$HERE/config/authz.conf" ] && [ ! -e "$HERE/src/authz.cpp" ]'
  check "no phrase / --confirm-protection / authz / register wording in start.sh or build.sh; start.sh and ack.sh never read stdin" '! grep -qiE "I-ACCEPT|--phrase|--confirm-protection|authz|authorize\.sh|--revoke|--authorize" "$HERE/start.sh" "$HERE/build.sh" "$HERE/ack.sh" "$HERE/stop.sh" && ! grep -nE "(^|[;&| ])read( |$)" "$HERE/start.sh" "$HERE/ack.sh" | grep -v "while read -r name ref rs val voice desc" | grep -q .'
  out=$(PM_BIN="$EXE" PM_LOGS=/tmp/pm_logs_dry PM_ROOT="$HERE" bash "$HERE/start.sh" --dry-run --iface eth0 < /dev/null 2>&1)
  check "dry-run of plain ./start.sh (real interface name): mode dance, armed=1, service args carry NO authorisation/confirmation flag" 'echo "$out" | grep -q "mode=dance armed=1" && ! echo "$out" | grep -q -- "--accept-unvalidated\|--confirm-protection\|--i-confirm"'
  out=$(PM_BIN="$EXE" PM_LOGS=/tmp/pm_logs_dry PM_ROOT="$HERE" bash "$HERE/start.sh" --dry-run --iface eth0 --yes --accept-unvalidated < /dev/null 2>&1)
  check "old flags --yes / --accept-unvalidated are ignored (NOTE), never forwarded" 'echo "$out" | grep -q "obsolete and ignored" && echo "$out" | grep -q "armed=1" && ! echo "$out" | grep "^DRY-RUN" | grep -q -- "--yes\|--accept-unvalidated"'
  for m in hold exit clip2 clip4 dance; do d=$(dry --mode $m); check "[$m] armed=1 without any flag" 'echo "$d" | grep -q "armed=1"'; done
  d=$(dry --mode check); check "[check] read-only mode: armed=0 (the only unarmed mode)" 'echo "$d" | grep -q "armed=0"'
  out=$(PM_BIN="$EXE" PM_ROOT="$HERE" bash "$HERE/ack.sh" < /dev/null 2>&1); rc=$?
  check "ack.sh needs no confirmation (stdin closed: rc 0)" '[ $rc -eq 0 ]'
  echo "  -- (b) plain ./start.sh --sim (no other flag, stdin closed): F1+F3 -> READY -> held for 135 s (the old 120 s timeout is gone) -> L1+A -> FULL dance -> IDLE"
  start_sim_only v161
  run_start_sh; rc=$?
  check "start.sh (no flag, no prompt): rc 0, mode dance, armed, voice 完整舞蹈" '[ $rc -eq 0 ] && grep -q "READY TO USE (mode dance" "$LG/start.out" && grep -q "armed=1" "$LG/service.log" && sim_has "TTS.*完整舞蹈"'
  press $KEY_LOAD 0.5
  check "F1+F3: model loaded, admission passed, READY, voice 准备就绪" 'waitfor "$LG/service.log" "READY: model loaded" 30 && sim_has "TTS.*准备就绪"'
  sleep 135
  check "still READY after 135 s: runner alive, no timeout, state READY, nothing sent to the robot" 'jq -e ".state == \"READY\"" "$LG/state.json" >/dev/null && ! log_has "timed out" && ! log_has "READY timeout" && ! log_has "ended before taking control" && ! sim_has "SwitchToUserCtrl" && ! sim_has "first stiff"'
  press $KEY_GO 0.5
  check "L1+A after the long READY: voice 开始执行, then the take-over (announcement + 0.5 s wait), the FULL dance (7.1 s window) completes" 'sim_has "TTS.*开始执行" && waitfor "$SD/sim_events.log" "first stiff user_lowcmd frame" 20 && waitfor "$LG/service.log" "RUN FINISHED rc=0" 60'
  sleep 1; R=$(latest_report)
  check "run_report: COMPLETED, mode dance, took control, WalkRun confirmed; announce_wait 0.5 s respected" '[ "$(jq -r .result "${R}run_report.json")" = COMPLETED ] && [ "$(jq -r .mode "${R}run_report.json")" = dance ] && [ "$(jq -r .summary.took_control "${R}run_report.json")" = true ] && W0=$(evt "$R" wait_begin) && W1=$(evt "$R" wait_end) && awk -v a=$W0 -v b=$W1 "BEGIN{exit !(b-a >= 0.5)}"'
  check "settled back to IDLE by itself (no ack, no HOLD)" 'waitfor "$LG/service.log" "SETTLED" 40 && waitfor "$LG/state.json" "\"state\":\"IDLE\"" 10 && ! log_has "HOLD:"'
  echo "  -- (c) repeat: the same service runs the dance again; START x2 during the run hands back; abnormal end (runner killed) returns to IDLE and a further run works"
  press $KEY_LOAD 0.5
  check "second F1+F3 -> READY" 'waitn "$LG/service.log" "READY: model loaded" 2 30'
  press $KEY_GO 0.5
  waitfor "$LG/runner_status.json" "\"phase\":\"RUN\"" 25; sleep 1.5; dclick
  check "START x2 during the run: hand-back to WalkRun (rc=10), settle -> IDLE without ack" 'waitn "$LG/service.log" "RUN FINISHED rc=10" 1 40 && waitn "$LG/service.log" "SETTLED" 2 40 && ! log_has "HOLD:"'
  press $KEY_LOAD 0.5
  check "third F1+F3 -> READY" 'waitn "$LG/service.log" "READY: model loaded" 3 30'
  press $KEY_GO 0.5
  waitfor "$LG/runner_status.json" "\"phase\":\"RUN\"" 25; sleep 1.0
  RP=$(pgrep -f "partymimic runner --iface lo" | head -1); kill -9 "$RP"
  check "abnormal end (runner killed in control): guardian hands back, service returns to IDLE by itself (no HOLD, no ack)" 'waitfor "$LG/service.log" "ABNORMAL end" 40 && waitfor "$LG/service.log" "SETTLED (abnormal end" 40 && waitfor "$LG/state.json" "\"state\":\"IDLE\"" 10 && ! log_has "HOLD:" && [ ! -e "$LG/ACK" ]'
  sleep 2
  press $KEY_LOAD 0.5
  check "after the abnormal end a new F1+F3 is accepted and READY is reached again" 'waitn "$LG/service.log" "READY: model loaded" 4 40'
  dclick
  check "START x2 in READY cancels before control, back to IDLE" 'waitn "$LG/service.log" "back to IDLE" 1 20'
  stop_sh
  echo "  -- (d) emergency / protection paths still active in the same build: L2+B, runtime instability, e-stop file (see S5 fault, S9 unstable, S14 estop, S15 l2b) - run those scenarios too"
}
scn_announce_wait() {
  echo "== S39 (V1.6) announce_wait: L1+A -> announce 开始执行 -> announcement FINISHED -> wait announce_wait_s (0.5 s) -> re-verify -> take-over; every step is logged"
  start_env annwait -- --run-s 1.0
  press $KEY_LOAD 0.5
  waitfor "$LG/service.log" "READY: model loaded" 30
  press $KEY_GO 0.5
  check "run finished normally (rc 0) after the wait" 'waitfor "$LG/service.log" "RUN FINISHED rc=0" 40'
  sleep 1
  R=$(latest_report)
  check "service_events.jsonl: announce_begin / announce_done (via estimated, not the cap)" 'sv_has "\"kind\":\"announce_begin\"" && sv_has "\"kind\":\"announce_done\",\"via\":\"estimated\""'
  check "events.jsonl: announce_begin / announce_done / wait_begin / wait_end all present, in this time order" 'for k in announce_begin announce_done wait_begin wait_end; do [ "$(ev_count "$R" $k)" -eq 1 ] || { echo "event $k"; exit 1; }; done; jq -s -r "[.[]|select(.kind|test(\"^(announce_begin|announce_done|wait_begin|wait_end)$\"))]|sort_by(.t_mono)|map(.kind)|join(\" \")" "$R/events.jsonl" | grep -q "^announce_begin announce_done wait_begin wait_end$"'
  A0=$(evt "$R" announce_begin); A1=$(evt "$R" announce_done); W0=$(evt "$R" wait_begin); W1=$(evt "$R" wait_end)
  ARM=$(jq -r 'select(.kind=="phase" and .phase=="ARM") | .t_mono' "$R/events.jsonl" | head -1)
  echo "  [info] announce_begin->announce_done $(awk -v a=$A0 -v b=$A1 'BEGIN{printf "%.2f",b-a}') s, wait $(awk -v a=$W0 -v b=$W1 'BEGIN{printf "%.2f",b-a}') s, wait_end->ARM $(awk -v a=$W1 -v b=$ARM 'BEGIN{printf "%.2f",b-a}') s"
  check "the announcement took its estimate (4 chars: >= 1.2 s) and did not hit the 4 s cap" 'awk -v a=$A0 -v b=$A1 "BEGIN{exit !(b-a >= 1.2 && b-a < 3.5)}"'
  check "wait after announce_done >= announce_wait_s (0.5 s) and < 0.8 s" 'awk -v a=$W0 -v b=$W1 "BEGIN{exit !(b-a >= 0.5 && b-a < 0.8)}"'
  check "take-over (phase ARM) comes AFTER wait_end: the robot was not taken over during the announcement or the wait" 'awk -v a=$W1 -v b=$ARM "BEGIN{exit !(b >= a)}"'
  check "sim: the first stiff user_lowcmd frame is >= announcement + 0.5 s after the voice request" 'T=$(awk "/TTS.*开始执行/{print \$1; exit}" "$SD/sim_events.log"); F=$(awk "/first stiff user_lowcmd frame/{print \$1; exit}" "$SD/sim_events.log"); awk -v t=$T -v f=$F "BEGIN{exit !(f-t >= 1.9)}"'
  stop_env
  echo "  -- announce_wait_s 0 (site/dev override): no extra wait"
  start_env annwait0 -- --run-s 1.0 --announce-wait-s 0
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  waitfor "$LG/service.log" "RUN FINISHED rc=0" 40; sleep 1
  check "announce_wait_s 0: wait_begin and wait_end within 0.9 s of each other (only the final re-verification: scan window 0.5 s)" 'W0=$(evt "$(latest_report)" wait_begin); W1=$(evt "$(latest_report)" wait_end); awk -v a=$W0 -v b=$W1 "BEGIN{exit !(b-a < 0.9)}"'
  stop_env
  echo "  -- TTS disabled (--no-tts): nothing to wait for, the 0.5 s still applies"
  start_env annnotts -- --run-s 1.0 --no-tts
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  waitfor "$LG/service.log" "RUN FINISHED rc=0" 40; sleep 1
  check "no audio: announce_done via no_audio at once, wait_end >= 0.5 s after wait_begin, run completed" 'sv_has "\"kind\":\"announce_done\",\"via\":\"no_audio\"" && W0=$(evt "$(latest_report)" wait_begin); W1=$(evt "$(latest_report)" wait_end); awk -v a=$W0 -v b=$W1 "BEGIN{exit !(b-a >= 0.5)}"'
  stop_env
}
scn_cancel_wait() {
  echo "== S40 (V1.6) START x2 (double click) while the announcement is playing / while waiting: cancel, no take-over"
  wait_prologue cancelann -- --run-s 1.0 --announce-wait-s 3.0
  sleep 0.4; dclick
  cancelled_checks start_double_click
  check "voice says 已退出 and the service is IDLE again (a new F1+F3 is possible)" 'sim_has "TTS.*已退出" && log_has "back to IDLE"'
  stop_env
  wait_prologue cancelwait -- --run-s 1.0 --announce-wait-s 3.0
  wait_after_announce 0.8; dclick
  cancelled_checks start_double_click
  check "the cancel happened INSIDE the wait: announce_done and wait_begin were logged before the cancel, no wait_end" '[ "$(ev_count "$(latest_report)" announce_done)" -eq 1 ] && [ "$(ev_count "$(latest_report)" wait_begin)" -eq 1 ] && [ "$(ev_count "$(latest_report)" wait_end)" -eq 0 ]'
  stop_env
}
scn_estop_wait() {
  echo "== S41 (V1.6) e-stop (latch file) / remote L2+B while waiting: nothing is sent (not yet taken over), cancel; the latch file is NOT cleared (stays until ./ack.sh)"
  wait_prologue estopwait -- --run-s 1.0 --announce-wait-s 3.0
  wait_after_announce 0.8; echo "estop" > /tmp/partymimic_cpp_ESTOP
  cancelled_checks estop_file
  check "e-stop latch still set (existing semantics: cleared only by ./ack.sh) and the next F1+F3 is BLOCKED by it" '[ -e /tmp/partymimic_cpp_ESTOP ]'
  sleep 4; press $KEY_LOAD 0.5
  check "F1+F3 after the e-stop: BLOCKED (estop_file_absent) - never READY" 'waitfor "$LG/service.log" "rc=6" 40 && grep -rq "estop_file_absent" "$LG"/runs/*/console.log'
  rm -f /tmp/partymimic_cpp_ESTOP
  stop_env
  wait_prologue l2bwait -- --run-s 1.0 --announce-wait-s 3.0
  wait_after_announce 0.8; setkeys 0x220; sleep 0.6; setkeys $KEY_NONE
  cancelled_checks remote_l2b
  stop_env
}
scn_fsm_wait() {
  echo "== S42 (V1.6) FSM leaves the entry state / attitude / lowstate problems while waiting: cancel, never taken over"
  wait_prologue fsmwait -- --run-s 1.0 --announce-wait-s 3.0
  wait_after_announce 0.8; echo 1 > "$SD/fsm"
  cancelled_checks fsm_changed
  check "voice: 检查失败" 'sim_has "TTS.*检查失败"'
  stop_env
  wait_prologue tiltwait -- --run-s 1.0 --announce-wait-s 3.0
  wait_after_announce 0.8; echo "0.7 0" > "$SD/tilt"
  cancelled_checks tilt
  stop_env
  wait_prologue mutewait -- --run-s 1.0 --announce-wait-s 3.0
  wait_after_announce 0.8; touch "$SD/mute"
  cancelled_checks lowstate_timeout
  rm -f "$SD/mute"
  stop_env
}
scn_term_wait() {
  echo "== S43 (V1.6) service SIGTERM while waiting (and guardian killed while waiting): cancel, nothing sent"
  wait_prologue termwait -- --run-s 1.0 --announce-wait-s 3.0
  wait_after_announce 0.8; kill -TERM "$SVCPID"
  check "runner cancelled by the signal, robot never switched, service exited" 'waitfor "$LG/service_events.jsonl" "service_start" 1; for _ in $(seq 1 300); do kill -0 "$SVCPID" 2>/dev/null || break; sleep 0.1; done; ! kill -0 "$SVCPID" 2>/dev/null && ! sim_has "SwitchToUserCtrl" && ! sim_has "first stiff"'
  check "cancel event reason signal" 'R=$(latest_report); [ "$(jq -r "select(.kind==\"cancel\")|.reason" "$R/events.jsonl" | head -1)" = signal ]'
  stop_env
  wait_prologue gdwait -- --run-s 1.0 --announce-wait-s 3.0
  wait_after_announce 0.8; GP=$(awk '$1=="pid"{print $2}' /tmp/partymimic_cpp_guardian.alive); kill -9 "$GP"
  cancelled_checks 'guardian|signal'   # the runner's own guardian check, or the service (CRITICAL guardian dead) SIGTERMs it first: both cancel
  stop_env
}

# ---- V1.6 A1: reference-aware instability judge, end to end on the loopback robot. The loopback robot REPLAYS the pelvis attitude of a MuJoCo rollout of the whole DIY dance (sim_robot --quat-replay);
#      the runner runs the real policy stack, so its reference quaternion is the real one. Replay files: results/diy/v16/A_deploy/a1_make_replay.py (TJ_REPLAY_DIR, default /tmp/a1_replay).
#      Use PM_ROOT_OVERRIDE=<DIY package root> (the DIY dance, 45.5 s): the default package only has the generic clip.
tj_root() {   # tj_root <extra site.conf lines> : a copy of PM_ROOT whose config/site.conf has the extra lines (assets are symlinked, never modified)
  local d=/tmp/pm_tj_root; rm -rf "$d"; mkdir -p "$d"; for f in "$PM_ROOT"/*; do [ "$(basename "$f")" = config ] || ln -s "$f" "$d/$(basename "$f")"; done
  cp -r "$PM_ROOT/config" "$d/config"; printf '%s\n' "$1" >> "$d/config/site.conf"; echo "$d"
}
tj_ev() { jq -r "select(.kind==\"tilt_judge_trigger\")|$1" "$(latest_report)/events.jsonl" 2>/dev/null | head -1; }
scn_tiltjudge() {
  echo "== S44 (V1.6 A1) reference-aware instability judge end to end: the whole dance does NOT trip at 0.40 (legacy would), injected falls still hand back to WalkRun, no valid reference -> fixed 0.40"
  RD="${TJ_REPLAY_DIR:-/tmp/a1_replay}"; OFF="${TJ_OFF:--0.02}"; RS="${TJ_RUN_S:-45.5}"; OEMT=0.9
  if [ ! -f "$RD/normal_real0.txt" ]; then echo "  [SKIP] no replay files in $RD (python results/diy/v16/A_deploy/a1_make_replay.py $RD)"; return; fi
  # (a) whole dance, nominal MuJoCo attitude: no exit before the window ends
  start_env tjnom --quat-replay "$RD/normal_real0.txt,$OFF" --oem-fall-tilt $OEMT -- --run-s "$RS" --ref-mode running
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "(a) whole dance finished COMPLETED (no UNSTABLE exit at tilt 0.40; replayed pelvis leans to ~0.48 rad)" 'waitfor "$LG/service.log" "RUN FINISHED rc=0" 100'
  sleep 3.5
  check "(a) report: COMPLETED, fault NONE, no tilt_judge_trigger event" '[ "$(rj .summary.result)" = COMPLETED ] && [ "$(rj .summary.fault)" = NONE ] && [ -z "$(tj_ev .rule)" ]'
  echo "  [info] tilt_judge: $(rj '.tilt_judge.stats|tostring')"
  check "(a) the robot really leaned beyond the old 0.40 limit (max tilt >= 0.45) while the gravity error stayed < 0.35 (min margin > 0), reference valid for most samples" 'awk -v m="$(rj .tilt_judge.stats.max_tilt_rad)" -v e="$(rj .tilt_judge.stats.max_gravity_err_rad)" -v g="$(rj .tilt_judge.stats.min_margin_gravity_err_rad)" -v n="$(rj .tilt_judge.stats.samples_ref_valid)" -v s="$(rj .tilt_judge.stats.samples)" "BEGIN{exit !(m>=0.45 && e<0.35 && g>0 && n>0.9*s)}"'
  check "(a) the whole clip ran (>= 2300 control ticks) and the judge limits are recorded in run_report and events" '[ "$(rj .control_ticks)" -ge 2300 ] && [ "$(rj .tilt_judge.config.unstable_hard_tilt_rad)" = 0.7 ] && grep -q "\"kind\":\"tilt_judge_config\"" "$(latest_report)/events.jsonl"'
  stop_env
  # (b) the SAME run with the legacy judge (ref_aware_unstable 0): fixed 0.40 trips on the dance
  PM_ROOT_SAVE="$PM_ROOT"; PM_ROOT=$(tj_root "ref_aware_unstable 0")
  start_env tjleg --quat-replay "$RD/normal_real0.txt,$OFF" --oem-fall-tilt $OEMT -- --run-s "$RS" --ref-mode running
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "(b) legacy judge (ref_aware_unstable 0): the dance is cut by UNSTABLE (rc=10), rule FIXED_LIMIT" 'waitfor "$LG/service.log" "RUN FINISHED rc=10" 100'
  sleep 3.5
  check "(b) report: fault UNSTABLE via the fixed limit, tilt just above 0.40" '[ "$(rj .summary.fault)" = UNSTABLE ] && [ "$(tj_ev .rule)" = FIXED_LIMIT ] && awk -v t="$(tj_ev .tilt_rad)" "BEGIN{exit !(t>0.40 && t<0.45)}"'
  stop_env; PM_ROOT="$PM_ROOT_SAVE"
  # (c) injected falls: the judge hands back to WalkRun before tilt 1.0. 0152_kp: gain collapse; 0092_joint_off: leg torque lost; 0098_stall: control stall; 0347_push: impulse
  for F in fault_0152_kp fault_0092_joint_off fault_0098_stall fault_0347_push; do
    [ -f "$RD/$F.txt" ] || { echo "  [SKIP] $F replay missing"; continue; }
    start_env tj_$F --quat-replay "$RD/$F.txt,$OFF" --oem-fall-tilt $OEMT -- --run-s "$RS" --ref-mode running
    press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
    check "(c) $F: run ended by the judge (rc=10 = EXITED_WALKRUN, WalkRun confirmed), not PASSIVE" 'waitfor "$LG/service.log" "RUN FINISHED rc=10" 100'
    sleep 3.5
    check "(c) $F: fault UNSTABLE, requested WALKRUN, final fsm 802, tilt at decision < 1.0 (not fallen), event tilt_judge_trigger has rule + reference tilt + gravity error + limits" '[ "$(rj .summary.fault)" = UNSTABLE ] && [ "$(rj .summary.requested_mode)" = WALKRUN ] && [ "$(rj .summary.final_fsm)" = 802 ] && awk -v t="$(rj .exit_detail.attitude_at_decision.tilt_rad)" "BEGIN{exit !(t<1.0)}" && [ -n "$(tj_ev .rule)" ] && [ "$(tj_ev .ref_valid)" = true ] && [ "$(tj_ev .limits.ref_dev_rad)" = 0.35 ] && [ "$(tj_ev .gravity_err_rad)" != null ]'
    echo "  [info] $F: rule=$(tj_ev .rule) tilt=$(tj_ev .tilt_rad) ref_tilt=$(tj_ev .ref_tilt_rad) gravity_err=$(tj_ev .gravity_err_rad) monitor=$(tj_ev .monitor) exit_reason: $(rj .summary.exit_reason)"
    stop_env
  done
  # (d) reference NOT running (--ref-mode paused: lead-in/hold semantics): the old fixed 0.40 is back; a replayed fall still hands back
  if [ -f "$RD/fault_0152_kp.txt" ]; then
    start_env tjpaused --quat-replay "$RD/fault_0152_kp.txt,$OFF" --oem-fall-tilt $OEMT -- --run-s "$RS" --ref-mode paused
    press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
    check "(d) paused reference: fixed-limit fallback fires (rc=10), rule FIXED_LIMIT (or the absolute rate/cap rules), reference reported not valid" 'waitfor "$LG/service.log" "RUN FINISHED rc=10" 100'
    sleep 3.5
    check "(d) event: ref_valid=false, rule in {FIXED_LIMIT, ABS_RATE, HARD_CAP}" '[ "$(tj_ev .ref_valid)" = false ] && case "$(tj_ev .rule)" in FIXED_LIMIT|ABS_RATE|HARD_CAP) true;; *) false;; esac'
    stop_env
  fi
}

SCN=("$@"); [ ${#SCN[@]} -eq 0 ] && SCN=(clean abort_ready abort_run guardian fault blocked term rival fsm_change unstable oem_fall recovery_fail estop l2b race rpc_delay inject release arm_fail caps commloss start_sh guardian_fatal log_complete log_clips log_events log_async log_crash modes settle leadin standby dds timeline announce_wait cancel_wait estop_wait fsm_wait term_wait tiltjudge v161)
for s in "${SCN[@]}"; do "scn_$s"; done
echo; echo "exit-path checks: $PASS passed, $FAILN failed"; [ $FAILN -eq 0 ]
