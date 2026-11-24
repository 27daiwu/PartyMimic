# partymimic_g1 (lean package source) — dev notes (not shipped)

Source of `dist/partymimic_g1.zip` (`deploy/make_zip.sh`). Research / evidence / sim tooling stays in `deploy/g1_fasttrack` (unchanged).

## What was removed from the robot package
sim_host / sim suites / service_e2e / sim-evidence / diag / keytest / fault & estop suites, 133-test suite, pytest wheels, `sim2real/sim_env`, viewer, teleop,
`guard_v2` (sim-only), `standref`, PREPARE/SETTLE/SHADOW runner phases, `--policy-direct`/debug flags, guard-threshold CLI options, `sdk2_passive` profile,
sign-off template, `bin/ft`, `status.sh`.

## Admission: before -> after (all gating items are real-time code; `fasttrack/checks.py`)
| before | after |
|---|---|
| admission 1a frozen ONNX/SSB hash | `artifact_*_frozen` (+ yaml) |
| admission 1b guard thresholds == R1, 2 strict-guard-only | unit test `test_guard_thresholds_not_relaxed`; runner has no guard-relaxing options (test) |
| admission 3 + diag.py (q limits, fsm, rt/lowcmd writer, quiescent, tilt...) + session.preflight + runner preflight file | ONE list: `session.preflight` (stream, envelope, **q in hard limits**, quiescent, tilt, fsm entry, mode_machine, **no foreign writer on user_lowcmd**, lock, estop) |
| diag `no_writer_on_rt/lowcmd` (FAIL in WalkRun) | `oem_rt_lowcmd_writer`: measured by DDS discovery, blocks until OEM-confirmed count is in `config/site.json` |
| runner gates: env var, 15-min preflight file, guardian heartbeat, run<=1 s, authority<=0.3 | `run_caps`, `guardian_alive`; human confirmation = start.sh prompt -> `--i-confirm-protection` |
| admission 4/5 OEM + site sign-off JSON, 6 sim evidence code_sha | removed from gating (documentary / research); listed as residual blockers in README |
| preflight at READY and again at GO | full at READY; at GO only guardian + OEM writers + 0.3 s-window preflight inside `acquire(fast=True)` |
| `--entry-fsm N` mandatory | `config/site.json entry_fsm_allowed` [500,505,801] (0/1 always refused); guardian learns the entry fsm from the lock file |

## Added safety/logging
START x2 during preflight/lock aborts BEFORE SwitchToUserCtrl; exit report has `exit_reason`, `entry_fsm`, `restored_fsm`; streaming `trace.csv`/`events.jsonl`.

## Checks run for this change (no full offline matrix)
`tools/selfcheck.py` (policy parity 0.00e+00, tick p99 1.6 ms, DDS/CRC loopback); `pytest tests` (keys vs gamepad.hpp, chords, START x2, state machine, hand-back/exit mode, guard, CRC, admission/caps/writer ack, refusals before DDS);
`tests/smoke_sim.py` (loopback: clean run, START x2 in READY, START x2 in RUN, L1+A in IDLE, BLOCKED path, runner kill -> guardian take-over).
