// Pure-logic safety guard (port of fasttrack/guard.py): state validity, joint/torque/tilt/gyro protection, command filter. No I/O.
// A fault is LATCHED; the session decides the (damped) exit.
#pragma once
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "pm/policy.hpp"

namespace pm {

enum class Fault {
  NONE, COMM_TIMEOUT, STATE_FROZEN, STATE_INVALID, JOINT_LIMIT, JOINT_SPEED, TORQUE_SUSTAINED, TILT, GYRO, CMD_INVALID, ESTOP, REMOTE_STOP,
  WATCHDOG, LOOP_OVERRUN, INFERENCE, NO_RESPONSE, SWITCH_FAILED, LOCK_LOST, FSM_CHANGED,
  UNSTABLE,     // tilt/gyro instability while still recoverable -> immediate WALKRUN (TILT = already fallen -> PASSIVE)
  SIGNAL,       // SIGINT/SIGTERM (stop.sh, service stop): operator quit -> WALKRUN
  REMOTE_DAMP   // remote L2+B held during the run: emergency PASSIVE channel
};
const char* fault_name(Fault f);

struct GuardConfig {  // defaults = the audited values (SAFETY_AUDIT.md); every field can be overridden in config/site.conf, ranges are enforced by validate()
  // ---- communication / illegal numbers / hardware limits: kept
  double state_stale_s = 0.060;
  int state_frozen_ticks = 5;
  double limit_fault_tol_rad = 0.05;     // measured q beyond the HARD joint limit by more than this = JOINT_LIMIT fault
  double speed_fault_frac = 0.90;        // |dq| > frac * hardware speed limit
  double unknown_speed_limit = 20.0;
  double torque_meas_fault_frac = 1.20;  // |tau_est| > frac * hardware effort limit ...
  int torque_meas_fault_ticks = 10;      // ... for this many ticks
  double max_tilt_rad = 1.0;             // fallen: TILT fault -> PASSIVE (the unstable level lives in ExitConfig)
  double max_gyro_rad_s = 15.0;
  // ---- command filter: relaxed from "clip the Policy" to "stay inside the hardware" (hardware limits themselves unchanged)
  double pos_margin_rad = 0.005;         // was 0.02
  double target_speed_frac = 1.00;       // was 0.60 of the hardware speed limit
  double torque_cmd_frac = 1.00;         // was 0.85 of the hardware effort limit
  // ---- runner loop protections (kept, now configurable)
  double diverge_rad = 1.5;              // |q - q_cmd| above this for diverge_ticks ticks = NO_RESPONSE
  int diverge_ticks = 8;
  double loop_overrun_factor = 3.0;      // control step longer than factor * dt = LOOP_OVERRUN
  // P5.3: TRAINING-EQUIVALENT TORQUE (default OFF). The training actuator takes the policy target unclipped and only saturates the torque at the effort limit
  // (mjlab BuiltinPositionActuator + actfrcrange); the deployed guard clips the target into the joint range, so for a target beyond the limit (robust_d does this on ~55% of the
  // ticks, ankle pitch up to +2.4 rad) the drive receives kp*(limit - q) instead of the saturated torque. With torque_ff the target STAYS inside the range (hardware protection unchanged)
  // and the missing torque is sent as LowCmd.tau (feed-forward): total = clamp(kp*(q_policy - q) - kd*dq, +-effort). Never above the effort limit; hip/ankle may then PRESS on the joint limit as in training.
  bool torque_ff = false;
};
bool validate(const GuardConfig& c, std::string& err);

struct GuardEvent { double t; Fault fault; std::string detail; };
struct GuardReject : std::runtime_error { using std::runtime_error::runtime_error; };

struct CmdOut { Vec q, dq, tau, kp, kd; Vec tau_train; };  // tau_train: the torque the TRAINING actuator would apply for the unclipped target (always filled, for the logs)

// P5.1: which stage of filter_command changed a joint's target (bit mask). The stages run in this order on the raw policy/hold target:
//   POS (inside [lo+margin, hi-margin]) -> SLEW (speed limit x dt around the previous target) -> TORQUE (PD torque limit, by shrinking the position error)
//   -> POS2 (re-clip after the torque stage) ; KP = the stiffness was capped at the frozen yaml gain.
enum GuardClip : uint8_t { CLIP_POS = 1, CLIP_SLEW = 2, CLIP_TORQUE = 4, CLIP_POS2 = 8, CLIP_KP = 16 };
const char* clip_reason_text(uint8_t mask);  // "POS_LIMIT|SLEW|..."
struct FilterTrace {  // optional per-call record of every clip (no cost when not requested)
  Vec raw, after_pos, after_slew, after_torque, final_q, lo, hi, prev_target, slew_step, tau_pre_nm, tau_lim_nm;
  std::array<uint8_t, N> mask{};
  int n_clipped = 0;
  uint32_t any_mask = 0;  // OR of all joints
};

class SafetyGuard {
 public:
  SafetyGuard(const Params& p, const GuardConfig& c = GuardConfig());
  Fault trip(Fault f, const std::string& detail, double t = 0);
  Fault fault() const { return fault_; }
  bool tripped() const { return fault_ != Fault::NONE; }
  const std::vector<GuardEvent>& events() const { return events_; }
  Fault check_state(const Vec& q, const Vec& dq, const Vec& tau, const std::array<double, 4>& quat, const std::array<double, 3>& gyro, bool has_tick,
                    uint32_t tick, double age_s, double t);
  void reset_target(const Vec& q) { prev_target_ = q; have_prev_ = true; }
  CmdOut filter_command(const Vec& q_t, const Vec& kp, const Vec& kd, const Vec& q, const Vec& dq, double dt, double t, FilterTrace* tr = nullptr, bool allow_ff = true);
  const GuardConfig& cfg() const { return c_; }
  const Params& params() const { return p_; }
  struct Stats { long cmds = 0, pos_clipped = 0, slew_clipped = 0, torque_clipped = 0; double max_dq_ratio = 0, max_tau_ratio = 0, max_tilt = 0, max_gyro = 0, min_q_margin = 1e9; } stats;

 private:
  Params p_;
  GuardConfig c_;
  Vec speed_lim_, kp_cap_;
  Fault fault_ = Fault::NONE;
  std::vector<GuardEvent> events_;
  std::array<int, N> tau_count_{};
  bool have_tick_ = false, have_prev_ = false;
  uint32_t last_tick_ = 0;
  int same_tick_ = 0;
  Vec prev_target_;
};

}  // namespace pm
