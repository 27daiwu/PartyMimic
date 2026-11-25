// ExitPolicy: the SINGLE source of truth for "how does control go back to the OEM controller" (runner session, guardian and `release` all use it).
//   * decide_exit()      pure decision table: trigger + attitude -> WALKRUN | PASSIVE (+ class, emergency flag)
//   * StabilityMonitor   pure instability / fall detector (tilt, gyro_xy, tilt rate), runs in the 3 ms watch thread AND in the control loop
//   * ExitExecutor       the timed sequence: RPC at once on a dedicated thread, hold/damp frames on a second thread, FSM confirmation with a stable
//                        observation window, WALKRUN -> PASSIVE fallback, emergency escalation, timestamps for the run report
//   * classify_exit()    pure result classification (COMPLETED / EXITED_WALKRUN / WALKRUN_RECOVERY_FAILED / FALLEN_PASSIVE / EMERGENCY_PASSIVE / ...)
// No decision about the exit lives anywhere else. All thresholds are ExitConfig fields (config/site.conf can override them).
#pragma once
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "pm/guard.hpp"
#include "pm/link.hpp"
#include "pm/policy.hpp"

namespace pm {

struct ExitConfig {
  // ---- attitude levels (angle between body z and world z = "tilt")
  double fallen_tilt_rad = 1.0;           // above: already down -> PASSIVE
  double unstable_tilt_rad = 0.40;        // above: unstable -> immediate WALKRUN request (P4: Sim2Sim calibration, see EXIT_POLICY.md section 10)
  double unstable_soft_tilt_rad = 0.30;   // above this AND (gyro or tilt-rate criterion) -> unstable
  double unstable_gyro_rad_s = 8.0;       // |gyro_xy| (roll/pitch rate) criterion
  double unstable_tilt_rate_rad_s = 3.0;  // d(tilt)/dt criterion (0 = off)
  int unstable_debounce_samples = 2;      // consecutive DISTINCT lowstate samples that must agree
  double attitude_max_age_s = 0.10;       // older attitude = unknown = PASSIVE
  // ---- remote controller
  double l2b_hold_s = 0.10;               // L2+B must be held this long (and re-pressed after control started) -> emergency PASSIVE
  // ---- execution
  double rpc_retry_gap_s = 0.05;          // pause before the single retry after an explicit rejection (3203/3204/3102)
  double confirm_timeout_s = 3.0;         // requested FSM must be seen within this time after the RPC
  double fsm_poll_s = 0.05;
  double dwell_s = 0.3;                   // after the first match the FSM must still match ...
  double stable_window_s = 0.8;           // ... and keep matching for this long (catches OEM dropping into FSM 1 afterwards)
  int min_stable_polls = 3;
  double mismatch_fail_s = 0.3;           // a foreign FSM (not user, not expected) lasting this long = recovery failed
  double unconfirmed_hold_s = 30.0;       // keep damping and watching after an unconfirmed exit
  double hold_gain_scale = 1.0;           // kp/kd of the HOLD frames = policy gains * this
  double exit_owner_timeout_s = 10.0;     // guardian leaves an exit owned by the runner alone for this long
  std::string estop_file = "/tmp/partymimic_cpp_ESTOP";
};
bool validate(const ExitConfig& c, std::string& err);

// ---------------------------------------------------------------- attitude
struct Attitude {
  bool valid = false;  // fresh, finite, normalised quaternion
  double age_s = 1e9, tilt = 0, gyro_xy = 0, roll = 0, pitch = 0;
};
double tilt_of_quat(const std::array<double, 4>& q);  // NaN if the quaternion is not usable
Attitude attitude_of(const StatePtr& st, double now_mono, double max_age_s);

// ---------------------------------------------------------------- stability detector
enum class Stability { OK, UNSTABLE, FALLEN };
class StabilityMonitor {
 public:
  explicit StabilityMonitor(const ExitConfig& c = ExitConfig()) : c_(c) {}
  Stability update(const LinkState& st);  // call with every state; duplicates (same seq) are ignored
  Stability level() const { return level_; }
  const std::string& detail() const { return detail_; }
  double tilt = 0, gyro_xy = 0, tilt_rate = 0;
 private:
  ExitConfig c_;
  uint64_t last_seq_ = ~0ull;
  int n_unstable_ = 0, n_fallen_ = 0;
  Stability level_ = Stability::OK;
  std::string detail_;
  std::deque<std::pair<double, double>> hist_;  // (arrival, tilt)
};

// ---------------------------------------------------------------- decision
struct ExitContext {
  Fault fault = Fault::NONE;
  bool normal_end = false;  // the motion window ended by itself
  Attitude att;
};
struct ExitDecision {
  std::string mode = "WALKRUN";         // WALKRUN | PASSIVE
  std::string cls = "NORMAL";           // NORMAL | OPERATOR | POLICY | UNSTABLE | FALLEN | UNTRUSTED | EMERGENCY
  bool emergency = false, normal_end = false, operator_stop = false;
  std::string why;
};
ExitDecision decide_exit(const ExitContext& ctx, const ExitConfig& cfg);

// ---------------------------------------------------------------- execution
enum class FrameKind { HOLD, DAMP };

struct ExitRequest {
  ExitDecision decision;
  std::string owner = "runner";    // runner | guardian | release
  std::string trigger_fault = "NONE", detail;
  int user_fsm = -1;               // FSM id that means "user control still active" (-1 unknown)
  int entry_fsm = -1;
  bool precheck = false;           // GetFsmId before the RPC (only when the user fsm was never confirmed / control may already be back)
  double t_detect = 0;             // mono() when the decision was made
  double t_event = 0;              // mono() arrival of the state sample that showed the problem (0 = unknown)
};

struct ExitVerdict {
  std::string result = "HANDBACK_UNCONFIRMED", voice, final_mode = "UNKNOWN";
  int exit_code = 5;
  bool recovery_ok = false, walkrun_recovered = false;
};

struct ExitOutcome {
  bool done = false;
  std::string requested_mode, final_mode = "UNKNOWN";
  bool confirmed = false;          // the REQUESTED mode was confirmed (rpc + fsm + dwell + stable window)
  bool settled = false;            // the robot is in a confirmed OEM state (requested mode or the PASSIVE fallback)
  bool escalated_to_passive = false, emergency_escalation = false, skipped_rpc = false, retried_after_rejection = false, verified_retry = false, late_confirm = false;
  int rpc_code = -999, final_fsm = -1, fsm_polls = 0, fsm_poll_failures = 0;
  std::vector<int> rpc_codes;
  std::vector<std::string> rpc_modes;
  double t_detect = 0, t_event = 0, t_rpc_sent = 0, t_rpc_returned = 0, t_fsm_confirmed = 0, t_stable_confirmed = 0, t_done = 0;
  std::string note, confirmed_via, escalation_why;
  std::vector<std::pair<double, int>> fsm_trace;  // (mono, fsm) at every change; -1 = unreadable
  ExitVerdict verdict;
};
ExitVerdict classify_exit(const ExitRequest& req, const ExitOutcome& o);
std::string exit_outcome_json(const ExitRequest& req, const ExitOutcome& o);  // timing / fsm fields shared by run_report.json and the guardian report

class ExitExecutor {
 public:
  // FrameFn is called from the caller of submit() (first frame) and from the executor's own 100 Hz frame thread; it must be thread safe and cheap.
  using FrameFn = std::function<void(FrameKind, const Vec& q_snapshot)>;
  using PhaseFn = std::function<void(const std::string& phase)>;      // EXITING / DONE (lock-file marker)
  using RemodeFn = std::function<std::string()>;                      // current attitude -> "PASSIVE" if fallen/untrusted, "" = unchanged
  using EmergencyFn = std::function<bool()>;                          // true once an emergency passive was requested during the exit
  ExitExecutor(Link& link, const ExitConfig& cfg, FrameFn frames, PhaseFn phase = nullptr, RemodeFn remode = nullptr, EmergencyFn emergency = nullptr);
  ~ExitExecutor();
  void start();                                // spawns the two idle threads (call before control is taken: thread creation is not on the critical path)
  bool submit(const ExitRequest& req);         // first call wins (false = already submitted). Wakes the RPC thread FIRST, then sends the first frame.
  bool submitted() const { return state_.load() != 0; }
  void escalate_passive(const std::string& why);  // emergency during/after submit: damp frames + PASSIVE request as soon as the current RPC returns
  bool wait(double timeout_s);                 // true when the outcome is final
  bool done() const { return state_.load() == 2; }
  const ExitOutcome& outcome() const { return out_; }
  const ExitRequest& request() const { return req_; }
  void stop();                                 // joins the threads (after done, or to abandon)

 private:
  void worker();
  void frame_loop();
  struct Conf;
  int call_rpc(const std::string& mode);
  void poll_record(int code, int fsm);
  int confirm(const std::string& mode, double timeout_s, bool& rejected_foreign, int& last_fsm);
  void finish_outcome();
  Link& link_;
  ExitConfig c_;
  FrameFn frames_;
  PhaseFn phase_;
  RemodeFn remode_;
  EmergencyFn emergency_;
  std::atomic<int> state_{0};  // 0 idle, 1 running, 2 done
  std::atomic<bool> stop_{false}, frames_on_{false}, escalate_{false};
  std::atomic<int> frame_kind_{0};
  std::mutex mtx_;
  std::condition_variable cv_, done_cv_;
  bool pending_ = false;
  std::thread worker_, frames_th_;
  ExitRequest req_;
  ExitOutcome out_;
  Vec snap_q_;
  std::string mode_;  // current requested mode (may change WALKRUN -> PASSIVE)
  bool started_ = false;
  std::mutex trace_m_;
};

}  // namespace pm
