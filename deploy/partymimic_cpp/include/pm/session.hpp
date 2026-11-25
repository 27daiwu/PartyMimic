// ControlSession: the only object allowed to talk to the robot (port of fasttrack/session.py).
//  * exclusive control    : flock on a per-robot lock file + foreign-writer detection on rt/user_lowcmd
//  * single logical writer: mode machine {OFF, ACQUIRE, NORMAL, DAMP}; in DAMP/ACQUIRE only the streamer thread publishes
//  * guard                : every command passes SafetyGuard::filter_command, every state sample passes check_state
//  * comm timeout / stall : watchdog thread (state age AND control-loop heartbeat)
//  * exit trigger         : emergency()/trigger_exit() latch ONE exit from any thread (watch thread 3 ms, control loop, signals, remote, e-stop file) and
//                           hand it to the shared ExitExecutor (pm/exit_policy.hpp): RPC at once on a dedicated thread, HOLD/DAMP frames meanwhile
//  * reliable exit        : finish() = (trigger if not yet) -> wait for the executor outcome (FSM confirmed with a stable window); never blind-retries the RPC
#pragma once
#include <atomic>
#include <memory>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "pm/exit_policy.hpp"
#include "pm/guard.hpp"
#include "pm/link.hpp"
#include "pm/profile.hpp"
#include "pm/remote.hpp"

namespace pm {

struct SessionConfig {
  std::string lock_path = "/tmp/partymimic_cpp_control.lock";
  std::string estop_file = "/tmp/partymimic_cpp_ESTOP";
  std::vector<int> entry_allowed;  // WalkRun-like fsm ids a take-over may start from
  ExitConfig exit;                 // exit decision + timing (single source: pm/exit_policy.hpp)
  double damp_hz = 100.0;          // ACQUIRE-phase frame stream rate
  double foreign_writer_listen_s = 1.0;
  double min_state_hz = 30.0;
  double quiescent_dq = 0.5;
  double max_entry_tilt_rad = 0.5;
  double watchdog_s = 0.10;
  double confirm_timeout_s = 3.0;  // SwitchToUserCtrl confirmation
  double remote_window_s = 0.3;
  double fsm_poll_s = 1.0;
};

// level: 0 PASS, 1 WARN (logged, never blocks), 2 FAIL (blocks). Only safety-critical items may FAIL.
enum Level { L_PASS = 0, L_WARN = 1, L_FAIL = 2 };
inline const char* level_name(int l) { return l == L_PASS ? "PASS" : l == L_WARN ? "WARN" : "FAIL"; }
struct PreflightItem { std::string name; int level; std::string detail; bool pass() const { return level != L_FAIL; } };

struct ExitReport {
  std::string reason, internal_mode, exit_mode_why, confirmed_via, note;
  int rpc_code = -999, fsm_after = -1;
  bool confirmed = false, streamed_damp = false, mode_matched = false;
  bool has_handback = false, took_control = false;
  ExitRequest req;
  ExitOutcome out;
};

class ExclusiveLock {
 public:
  ~ExclusiveLock() { release(); }
  bool acquire(const std::string& path);
  void update_entry_fsm(int fsm);
  void update_exit(const std::string& phase, const std::string& owner);  // exit_phase / exit_owner / exit_t in the lock record (guardian reads them)
  void release();
  std::string holder;
 private:
  void write_record();
  int fd_ = -1;
  int pid_ = 0, entry_fsm_ = -1;
  std::string exit_phase_ = "NONE", exit_owner_ = "none";
  double exit_t_ = 0;
  std::mutex wm_;
};
// authoritative non-blocking flock probe: (held by a live process, pid, entry_fsm or -1)
struct LockInfo { bool held = false; int pid = 0, entry_fsm = -1; std::string exit_phase = "NONE", exit_owner = "none"; double exit_t = 0; };
LockInfo lock_probe(const std::string& path);

class ControlSession {
 public:
  enum Mode { OFF, ACQUIRE, NORMAL, EXITING };
  using EventFn = std::function<void(const std::string& kind, const std::string& json_fields)>;
  ControlSession(Link& link, SafetyGuard& guard, const SessionConfig& cfg, EventFn ev = nullptr);
  ~ControlSession();

  double now() const { return mono() - t0_; }
  StatePtr state() const { return link_.latest(); }
  Fault check_state_once(const StatePtr& st);
  bool preflight(std::vector<PreflightItem>& out, double listen_s = -1, double rate_window_s = 1.0);
  bool acquire(bool fast);
  bool write(const Vec& q_t, const Vec& kp, const Vec& kd, double dt);  // true if a command went out in NORMAL mode
  void heartbeat() { heartbeat_ = mono(); }
  // latch the (single) exit: decision from the CURRENT attitude, RPC request handed to the executor within microseconds. Safe from any thread.
  void emergency(Fault f, const std::string& detail);
  void trigger_exit(Fault f, const std::string& detail, double t_detect, double t_event, bool normal_end = false);
  ExitReport finish(const std::string& reason);  // reason: COMPLETED | OPERATOR_QUIT | FAULT (ignored if an exit was already triggered)
  void request_quit() { quit_ = true; }  // async-signal-safe; the watchdog/main loop act on it
  bool quit_requested() const { return quit_; }
  std::atomic<bool>& quit_flag() { return quit_; }  // for signal handlers (lock-free atomic store is async-signal-safe)
  bool remote_fired() const { return rf_->fired.load() > 0; }
  bool exit_triggered() const { return exec_.submitted(); }
  Mode mode() const { return (Mode)mode_.load(); }
  int entry_fsm() const { return entry_fsm_; }
  int user_fsm_seen() const { return user_fsm_seen_; }
  long n_sent() const { return n_sent_; }
  // fsm seen at READY: if the factory fsm differs when control is about to be taken, the factory remote reacted to the chord / the state changed -> refuse
  void set_expected_entry_fsm(int fsm) { expected_entry_fsm_ = fsm; }
  void set_hold_gains(const Vec& kp, const Vec& kd) { hold_kp_ = kp; hold_kd_ = kd; }
  ExitExecutor& executor() { return exec_; }
  const Vec& last_cmd_q() const { return last_q_; }
  bool has_last_cmd() const { return have_last_; }
  SafetyGuard& guard() { return guard_; }

 private:
  void send_raw(const Vec& q, const Vec& dq, const Vec& tau, const Vec& kp, const Vec& kd);
  void exit_frame(FrameKind k, const Vec& q_snapshot);
  void start_threads();
  void watch_loop();
  void stream_loop();
  void fsm_poll_loop();
  void cleanup();
  void event(const std::string& kind, const std::string& fields = "");

  Link& link_;
  SafetyGuard& guard_;
  SessionConfig cfg_;
  EventFn ev_;
  std::atomic<int> mode_{OFF};
  std::recursive_mutex lk_;
  ExclusiveLock lock_;
  double t0_;
  std::atomic<double> heartbeat_;
  std::atomic<bool> stop_{false}, quit_{false};
  std::atomic<long> n_sent_{0};
  std::thread wd_, streamer_, fsmth_;
  int entry_fsm_ = -1, user_fsm_seen_ = -1, expected_entry_fsm_ = -1;
  Vec last_q_, last_dq_, last_tau_, last_kp_, last_kd_, hold_kp_, hold_kd_;
  bool have_last_ = false;
  struct RemoteFlag {
    std::atomic<int> fired{0}, l2b_fired{0};
    std::atomic<bool> active{false};  // control phase: L2+B counts only then
    StartDoubleClick dbl;
    DampCombo l2b;
    std::atomic<double> tm{0}, t_l2b{0};
    std::atomic<double> arrival{0};
  };
  std::shared_ptr<RemoteFlag> rf_;  // shared with the DDS-thread listener (the Link may outlive the session)
  bool finished_ = false;
  ExitReport exit_report_;
  std::atomic<int> fsm_bad_{0};
  ExitExecutor exec_;
  StabilityMonitor main_mon_, watch_mon_;
  std::atomic<double> last_state_arrival_{0};
};

}  // namespace pm
