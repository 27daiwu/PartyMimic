#!/usr/bin/env bash
# DEV ONLY (never shipped): key control/exit paths of the C++ deployment on a LOOPBACK test robot (iface lo, domain 77). Never touches a real robot.
#   usage: dev/exit_paths.sh <build_dir> [scenario ...]      (build_dir contains partymimic and dev/sim_robot)
#   needs jq. Exit policy under test: EXIT_POLICY.md (WALKRUN at once / PASSIVE for fallen / emergency; confirmation by GetFsmId + stable window).
set -u
# P4 hardening: (1) only ONE instance may run (the scenarios share /tmp paths and DDS domain 77); (2) NO `pkill -f` on command-line patterns (it also killed the invoking
# shell whenever its command line contained "partymimic service ..."): test processes are found by exact process name AND resolved executable path of THIS build dir.
exec 9>/tmp/pm_exit_paths.lock; flock -n 9 || { echo "another dev/exit_paths.sh is already running (lock /tmp/pm_exit_paths.lock): refusing to run concurrently"; exit 3; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; B="${1:?build dir}"; shift || true
export PM_ROOT="$HERE"; EXE="$B/partymimic"; SIM="$B/dev/sim_robot"
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
  PM_LOGS="$LG" "$EXE" service --sim "${svcargs[@]}" > "$LG/service.out" 2>&1 & SVCPID=$!
  waitfor "$LG/service.log" "guardian started" 10 && waitfor "$LG/service.log" "service starting" 10
  sleep 3.0   # guardian reads its entry fsm (1 s) and the heartbeat must be fresh
}
stop_env() {
  kill -TERM "$SVCPID" 2>/dev/null; for _ in $(seq 1 100); do kill -0 "$SVCPID" 2>/dev/null || break; sleep 0.1; done
  kill -9 "$SVCPID" 2>/dev/null; kill_test_procs -TERM
  kill -TERM "$SIMPID" 2>/dev/null; sleep 0.5; kill -9 "$SIMPID" 2>/dev/null; wait 2>/dev/null; rm -f /tmp/partymimic_cpp_guardian.alive
}
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
  check "service latched HOLD (needs ./ack.sh) and announced the REAL result: 异常已交还" 'log_has "HOLD:" && log_has "guardian take-over result=EXITED_WALKRUN" && sim_has "TTS.*异常已交还"'
  touch "$LG/ACK"; check "ack returns to IDLE" 'waitfor "$LG/service.log" "ACK: back to IDLE" 5'
  stop_env
}
scn_fault() {
  echo "== S5 robot already DOWN in RUN (roll 1.3 rad): FALLEN -> PASSIVE (damping frames + SwitchToInternalCtrl(PASSIVE)), never WALKRUN"
  go_run fault --flip-after-user-s 0.4 -- --run-s 8.0 --ref-mode paused
  check "fall ends the run with the PASSIVE class (rc=13) and HOLD" 'waitfor "$LG/service.log" "HOLD: runner rc=13" 40'
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
  check "GO refused: fsm changed since READY" 'waitfor "$LG/service.log" "GO refused" 20 && grep -rq "fsm_unchanged_since_ready" "$LG"/runs/*/console.log'
  check "robot never switched" '! sim_has "SwitchToUserCtrl" && ! sim_has "first stiff"'
  stop_env
}


scn_unstable() {
  echo "== S10 instability (roll steps to 0.42 rad with gyro 3 rad/s (> unstable_tilt_rad 0.40), still recoverable): immediate WALKRUN, measure detect->RPC latency"
  go_run unstable -- --run-s 8.0 --ref-mode paused
  echo "0.42 3.0" > "$SD/tilt"     # roll steps to 0.42 rad (> 0.40, < OEM fall tilt 0.45), gyro 3 rad/s: unstable but still recoverable
  check "run ended with the fault exit (rc=10) and HOLD" 'waitfor "$LG/service.log" "HOLD: runner rc=10" 40'
  rm -f "$SD/tilt"; sleep 1
  check "report: UNSTABLE -> EXITED_WALKRUN, WalkRun confirmed (fsm 802), recovery_ok" '[ "$(rj .summary.fault)" = UNSTABLE ] && [ "$(rj .summary.result)" = EXITED_WALKRUN ] && [ "$(rj .summary.final_fsm)" = 802 ] && [ "$(rj .summary.recovery_ok)" = true ]'
  check "WALKRUN requested at once: no damping frames, HOLD frames only" '! sim_has "first damping user_lowcmd frame" && sim_has "REQUEST mode=2" && ! sim_has "REQUEST mode=1"'
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
  check "run ends, rc=11 HOLD" 'waitfor "$LG/service.log" "HOLD: runner rc=11" 40'
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
  check "(a) HOLD rc=11" 'waitfor "$LG/service.log" "HOLD: runner rc=11" 40'
  sleep 1
  check "(a) result WALKRUN_RECOVERY_FAILED, rpc 0, final fsm 1, mode_matched/recovery false, not COMPLETED" '[ "$(rj .summary.result)" = WALKRUN_RECOVERY_FAILED ] && [ "$(rj .exit_detail.rpc_code)" = 0 ] && [ "$(rj .summary.final_fsm)" = 1 ] && [ "$(rj .summary.recovery_ok)" = false ] && [ "$(rj .summary.exit_confirmed)" = false ] && ! grep -q "\"result\":\"COMPLETED\"" "$(latest_report)/run_report.json"'
  check "(a) voice 恢复失败, never 执行完成" 'sim_has "TTS.*恢复失败" && ! sim_has "TTS.*执行完成"'
  stop_env
  echo "  -- (b) WALKRUN request ignored (rpc 0, fsm stays 1000): PASSIVE request follows with damping frames"
  start_env rf_b --ignore-walkrun -- --run-s 1.0
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "(b) HOLD rc=11" 'waitfor "$LG/service.log" "HOLD: runner rc=11" 60'
  sleep 1
  check "(b) rpc modes WALKRUN then PASSIVE, final PASSIVE fsm 1, damping frames after the fallback" '[ "$(rj ".exit_detail.rpc_modes|join(\",\")")" = "WALKRUN,PASSIVE" ] && [ "$(rj .summary.final_fsm)" = 1 ] && [ "$(rj .summary.result)" = WALKRUN_RECOVERY_FAILED ] && sim_has "first damping user_lowcmd frame"'
  stop_env
  echo "  -- (c) WalkRun reached, OEM drops into fsm 1 0.5 s later: the stable window must catch it"
  start_env rf_c --walkrun-drop-s 0.5 -- --run-s 1.0
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "(c) HOLD rc=11" 'waitfor "$LG/service.log" "HOLD: runner rc=11" 40'
  sleep 1
  check "(c) WALKRUN_RECOVERY_FAILED, final fsm 1" '[ "$(rj .summary.result)" = WALKRUN_RECOVERY_FAILED ] && [ "$(rj .summary.final_fsm)" = 1 ] && sim_has "OEM drops into fsm 1"'
  stop_env
  echo "  -- (d) every exit RPC ignored: HANDBACK_UNCONFIRMED, exit code 5"
  start_env rf_d --ignore-internal -- --run-s 1.0 --unconfirmed-hold-s 2
  press $KEY_LOAD 0.5; waitfor "$LG/service.log" "READY: model loaded" 30; press $KEY_GO 0.5
  check "(d) HOLD rc=5" 'waitfor "$LG/service.log" "HOLD: runner rc=5" 60'
  sleep 1
  check "(d) HANDBACK_UNCONFIRMED, voice 检查失败" '[ "$(rj .summary.result)" = HANDBACK_UNCONFIRMED ] && [ "$(rj .summary.recovery_ok)" = false ] && sim_has "TTS.*检查失败"'
  stop_env
}
scn_estop() {
  echo "== S14 emergency PASSIVE by the ESTOP file (stop.sh --estop): damping frame + SwitchToInternalCtrl(PASSIVE), never WALKRUN"
  go_run estop -- --run-s 8.0 --ref-mode paused
  echo "estop" > /tmp/partymimic_cpp_ESTOP
  check "run ends, rc=12 HOLD" 'waitfor "$LG/service.log" "HOLD: runner rc=12" 40'
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
  check "L2+B held -> run ends, rc=12 HOLD" 'waitfor "$LG/service.log" "HOLD: runner rc=12" 40'
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
  check "(a) HOLD rc=12" 'waitfor "$LG/service.log" "HOLD: runner rc=12" 40'
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
    check "[$kind] ends with rc=10 HOLD" 'waitfor "$LG/service.log" "HOLD: runner rc=10" 40'
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
  check "ARM_FAILED (rc=4) HOLD" 'waitfor "$LG/service.log" "HOLD: runner rc=4" 40'
  sleep 1.5
  check "report ARM_FAILED, no exit RPC (fsm already the entry WalkRun fsm 802), voice 检查失败" '[ "$(rj .summary.result)" = ARM_FAILED ] && [ "$(rj .summary.final_fsm)" = 802 ] && [ "$(rj .exit_detail.skipped_rpc)" = true ] && ! sim_has "REQUEST" && sim_has "TTS.*检查失败"'
  stop_env
}
scn_caps() {
  echo "== S20 caps (P4 defaults max_run_s 12 / max_authority 1.0), compile-time limits, partial-authority refusal, ramp/end-hold plumbing, legacy values flagged"
  start_env caps -- --run-s 2.0 --authority-max 1.0 --ref-mode paused
  press $KEY_LOAD 0.5
  check "READY with run_s 2.0 / authority 1.0 (sim)" 'waitfor "$LG/service.log" "READY: model loaded" 30'
  R=$(latest_report)
  check "WARNING in the service/runner log and ready.json exceeds_legacy_caps=true (run_s 2 > legacy 1.0 s)" 'log_has "EXCEED the legacy real-robot caps" && grep -q "EXCEEDS THE LEGACY CAPS" "$R/console.log" && [ "$(jq -r .exceeds_legacy_caps "$R/ready.json")" = true ]'
  stop_env
  # real-mode refusals happen before any DDS object exists; a non-existent interface name makes a real connection impossible even if the check were broken
  export PM_LOGS=/tmp/pm_logs_caps_refuse; rm -rf $PM_LOGS; mkdir -p $PM_LOGS
  out=$("$EXE" runner --iface nonexistent_pm0 --run-s 13 --authority-max 1.0 --i-confirm-protection 2>&1); rc=$?
  check "real run_s 13 > cap 12 refused (rc 2, before any DDS)" '[ $rc -eq 2 ] && echo "$out" | grep -q "run_s .* not in (0, max_run_s=12"'
  out=$("$EXE" runner --iface nonexistent_pm0 --run-s 1.0 --authority-max 1.5 --i-confirm-protection 2>&1); rc=$?
  check "real authority 1.5 > cap 1.0 refused" '[ $rc -eq 2 ] && echo "$out" | grep -q "authority .* not in (0, max_authority=1"'
  out=$("$EXE" runner --iface nonexistent_pm0 --run-s 1.0 --authority-max 0.3 --i-confirm-protection 2>&1); rc=$?
  check "sustained partial authority 0.3 REFUSED (unstable regime) unless --allow-partial-authority" '[ $rc -eq 2 ] && echo "$out" | grep -q "partial-authority regime"'
  out=$("$EXE" runner --iface nonexistent_pm0 --run-s 1.0 --authority-max 1.0 --ramp-s 3 --accept-unvalidated --i-confirm-protection 2>&1); rc=$?
  check "ramp_s 3 > 2 s refused" '[ $rc -eq 2 ] && echo "$out" | grep -q "ramp-s"'
  out=$("$EXE" runner --iface lo --sim --run-s 13 --authority-max 1.0 2>&1); rc=$?
  check "above the compile-time limit (12 s) refused even in --sim" '[ $rc -eq 2 ] && echo "$out" | grep -q "compile-time limits"'
  unset PM_LOGS
}

scn_commloss() {
  echo "== S22 communication loss during RUN: rt/lowstate silent -> COMM_TIMEOUT -> WALKRUN requested at once (OEM has its own estimator), FSM-verified; PASSIVE only if it fails"
  go_run commloss -- --run-s 8.0 --ref-mode paused
  touch "$SD/mute"
  check "run ended with the fault exit and HOLD" 'waitfor "$LG/service.log" "HOLD: runner rc=" 40'
  rm -f "$SD/mute"; sleep 1
  check "fault COMM_TIMEOUT detected, WALKRUN requested (never PASSIVE first), no damping frames" '[ "$(rj .summary.fault)" = COMM_TIMEOUT ] && sim_has "REQUEST mode=2" && ! sim_has "REQUEST mode=1" && ! sim_has "first damping user_lowcmd frame"'
  check "result EXITED_WALKRUN with FSM-verified WalkRun (final_fsm 802), not COMPLETED" '[ "$(rj .summary.result)" = EXITED_WALKRUN ] && [ "$(rj .summary.final_fsm)" = 802 ] && [ "$(rj .summary.recovery_ok)" = true ]'
  lat
  check "detect->rpc_sent < 20 ms" 'lt "$(rj .exit_detail.detect_to_rpc_sent_ms)" 20'
  stop_env
}
scn_unvalidated_gate() {
  echo "== S23 real-mode gate: full authority / windows > 1 s need --accept-unvalidated (refused before any DDS object); partial authority refused"
  export PM_LOGS=/tmp/pm_logs_gate; rm -rf $PM_LOGS; mkdir -p $PM_LOGS
  out=$("$EXE" runner --iface nonexistent_pm0 --run-s 3.0 --authority-max 1.0 --i-confirm-protection 2>&1); rc=$?
  check "authority 1.0 / 3 s without --accept-unvalidated REFUSED (rc 2, validated envelope <=1 s / <=0.3)" '[ $rc -eq 2 ] && echo "$out" | grep -q "NOT been validated on the real robot"'
  out=$("$EXE" runner --iface nonexistent_pm0 --run-s 3.0 --authority-max 1.0 --accept-unvalidated --i-confirm-protection 2>&1); rc=$?
  check "with --accept-unvalidated the cap/ack checks pass (then fails only because the interface does not exist, no robot involved)" '! echo "$out" | grep -q "REFUSED"'
  out=$("$EXE" runner --iface nonexistent_pm0 --run-s 1.0 --authority-max 0.3 --accept-unvalidated --i-confirm-protection 2>&1); rc=$?
  check "partial authority 0.3 REFUSED even with --accept-unvalidated (unstable regime)" '[ $rc -eq 2 ] && echo "$out" | grep -q "partial-authority regime"'
  unset PM_LOGS
}

# ---- P4.1: the REAL start entry (start.sh -> service -> guardian/runner command lines). Field bug: start.sh --accept-unvalidated -> guardian "missing value for --accept-unvalidated", 178 restarts.
start_sim_only() { kill_test_procs -9; sleep 0.5; SD="/tmp/pm_sim_$1"; LG="/tmp/pm_logs_$1"; rm -rf "$SD" "$LG" /tmp/partymimic_cpp_ESTOP /tmp/partymimic_cpp_control.lock /tmp/partymimic_cpp_guardian.alive; mkdir -p "$SD" "$LG"; echo 0 > "$SD/keys"
  "$SIM" --dir "$SD" > "$SD/sim.out" 2>&1 & SIMPID=$!; sleep 1.0; }
run_start_sh() { PM_BIN="$EXE" PM_LOGS="$LG" PM_ROOT="$HERE" bash "$HERE/start.sh" --sim "$@" > "$LG/start.out" 2>&1; }
stop_sh() { PM_LOGS="$LG" PM_ROOT="$HERE" bash "$HERE/stop.sh" > "$LG/stop.out" 2>&1; kill_test_procs -TERM; kill -TERM "$SIMPID" 2>/dev/null; sleep 0.5; kill -9 "$SIMPID" 2>/dev/null; wait 2>/dev/null; rm -f /tmp/partymimic_cpp_guardian.alive; }
alive_age() { echo $(( $(date +%s) - $(stat -c %Y /tmp/partymimic_cpp_guardian.alive 2>/dev/null || echo 0) )); }
scn_start_sh() {
  echo "== S24 real start entry: ./start.sh --stage hold --accept-unvalidated --yes (all P4 flags): guardian starts ONCE and stays alive, F1+F3 -> READY, START x2 cancels, stop.sh"
  start_sim_only startsh
  run_start_sh --stage hold --accept-unvalidated --yes; rc=$?
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
  echo "  -- variant: authority 1.0 WITHOUT the acknowledgement (non-interactive): start.sh must say NOT ARMED"
  start_sim_only startsh2
  run_start_sh --stage hold --yes; rc=$?
  check "no --accept-unvalidated: 'NOT ARMED (no acknowledgement)' printed" 'grep -q "NOT ARMED (no acknowledgement)" "$LG/start.out"'
  stop_sh
}
scn_guardian_fatal() {
  echo "== S25 guardian start-up failure (rc 2 = bad arguments): restart loop STOPPED after the first failure, exact reason in service.log / state.json / F1+F3 refusal / start.sh output"
  start_sim_only gfatal
  export PM_TEST_GUARDIAN_ARGS="--bogus-opt"
  run_start_sh --stage hold --accept-unvalidated --yes; rc=$?
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

SCN=("$@"); [ ${#SCN[@]} -eq 0 ] && SCN=(clean abort_ready abort_run guardian fault blocked term rival fsm_change unstable oem_fall recovery_fail estop l2b race rpc_delay inject release arm_fail caps commloss unvalidated_gate start_sh guardian_fatal)
for s in "${SCN[@]}"; do "scn_$s"; done
echo; echo "exit-path checks: $PASS passed, $FAILN failed"; [ $FAILN -eq 0 ]
