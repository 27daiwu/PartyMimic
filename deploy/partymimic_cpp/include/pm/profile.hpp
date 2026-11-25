// Firmware / protocol profile of the official g1_mimic_demo route + exit-mode rules (port of fasttrack/profile.py).
// Nothing here is verified against the target firmware (verified_on_firmware=false): the first real run records the observed values in logs/.
#pragma once
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace pm {

constexpr double DAMP_KD = 3.0;                  // damping frames: kp=0, kd=3 (demo)
constexpr int USER_FSM_EXPECTED = 1000;          // GetFsmId after SwitchToUserCtrl (strong confirmation)
constexpr int EXIT_FSM_PASSIVE = 1, EXIT_FSM_WALKRUN = 500;  // SDK Damp()=1, Start()=500 (a mismatch is reported, not fatal)
// FSM ids of the official walking family. wbc_nx (main.cpp/replay.cpp) manages 801 (kWalkingMode) and 802 (kWalkingMode1) as walking modes and 505 as the
// teleop mode; the robot in the field log reads 802 while standing under the factory WalkRun. After SwitchToInternalCtrl(WALKRUN) any of these is a match.
inline bool is_walkrun_family(int fsm) { return fsm == 500 || fsm == 801 || fsm == 802; }

inline void roll_pitch(const std::array<double, 4>& q, double& roll, double& pitch) {
  double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n == 0) n = 1;
  double w = q[0] / n, x = q[1] / n, y = q[2] / n, z = q[3] / n;
  roll = std::atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  pitch = std::asin(std::max(-1.0, std::min(1.0, 2 * (w * y - z * x))));
}

// The exit-mode decision lives in pm/exit_policy.hpp (decide_exit): there is no second copy of the rules.

inline int expected_exit_fsm(const std::string& mode) { return mode == "PASSIVE" ? EXIT_FSM_PASSIVE : mode == "WALKRUN" ? EXIT_FSM_WALKRUN : -1; }
inline bool exit_fsm_matches(const std::string& mode, int fsm) { return mode == "WALKRUN" ? is_walkrun_family(fsm) : fsm == expected_exit_fsm(mode); }

// Entry rule: fsm readable, never 0 (ZeroTorque) / 1 (Damp/PASSIVE), and one of the allowed WalkRun-like ids.
inline bool entry_check(int fsm, const std::vector<int>& allowed, std::string& why) {
  if (fsm < 0) { why = "fsm unknown"; return false; }
  if (fsm == 0 || fsm == 1) { why = "fsm " + std::to_string(fsm) + " is forbidden as entry (ZeroTorque/PASSIVE: put the robot in WalkRun first)"; return false; }
  if (!allowed.empty()) {
    bool ok = false;
    for (int a : allowed) ok = ok || a == fsm;
    if (!ok) { why = "fsm " + std::to_string(fsm) + " is not an allowed WalkRun-like entry (config/site.conf entry_fsm_allowed)"; return false; }
  }
  why = "fsm " + std::to_string(fsm) + " accepted";
  return true;
}

}  // namespace pm
