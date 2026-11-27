#include <cmath>
// Service: ONE process that listens to the remote controller, speaks, and manages the guardian and the runner.
//   F1+F3 (chord) : load the model + real-time admission (read-only)  -> READY or BLOCKED
//   L1+A  (chord) : (only from READY) hand the robot from the factory WalkRun to PartyMimic, run the dance window, hand back to WalkRun
//   START x2      : at ANY stage: abort. Before ARM: nothing was ever sent. After ARM: the runner's own START handler stops + hands back (WalkRun if the
//                   attitude is normal, PASSIVE if severe/unknown) and the independent guardian is the backup.
// V1.6.1: after EVERY run (clean or not) the service settles back to IDLE by itself and keeps listening; READY has no timeout; no authorisation / acknowledgement exists. The next F1+F3 real-time admission refuses an unsafe state.
// V1.5.1 PRE-WARMED RUNNER: at start-up (and after every finished run) the service starts a runner in STANDBY (model loaded + warm-up + DDS monitors done, no lock / writer / RPC). F1+F3 then only
// writes the trigger file; the runner runs its real-time CHECKING and reports READY at once. If no standby runner is alive (starting up, crashed, disabled) F1+F3 starts a cold runner as before.
// Every stage is announced by voice (G1 audio service, as in wbc_nx/remote_start). THIS PROCESS NEVER WRITES TO THE ROBOT: its DDS link is read-only
// (it only subscribes rt/lowstate and calls the audio service); only the runner and the guardian (separate processes) can publish motion commands.
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

#include "pm/app.hpp"
#include "pm/link.hpp"
#include "pm/logtool.hpp"
#include "pm/remote.hpp"
#include "pm/tts.hpp"
#include "pm/exit_policy.hpp"

extern char** environ;

namespace pm {
namespace {

enum class St { IDLE, LOADING, READY, RUNNING, SETTLE };  // SETTLE: a run ended (normal OR abnormal); waiting for the runner / lock / guardian (+ WalkRun / attitude after a recoverable end) before IDLE. V1.6.1: no HOLD, no ack
const char* st_name(St s) {
  switch (s) { case St::IDLE: return "IDLE"; case St::LOADING: return "LOADING"; case St::READY: return "READY"; case St::RUNNING: return "RUNNING"; case St::SETTLE: return "SETTLE"; }
  return "?";
}
std::string rc_text(int rc) {
  switch (rc) {
    case 0: return "clean end"; case 1: return "runner crashed"; case 2: return "refused"; case 3: return "no lowstate";
    case 4: return "ARM failed (control could not be taken cleanly)"; case 5: return "EXIT UNCONFIRMED (robot may still be in user control!)";
    case 6: return "BLOCKED by real-time admission / GO refused"; case 10: return "fault / operator stop, WalkRun recovery confirmed";
    case 11: return "WALKRUN RECOVERY FAILED (rpc ok but fsm not in the WalkRun family / dropped to fsm 1: robot probably down)";
    case 12: return "EMERGENCY PASSIVE (e-stop file / remote L2+B)"; case 13: return "robot down / state untrusted: PASSIVE";
    case 7: return "CANCELLED while waiting to take over (announce_wait): control was never taken, nothing was sent";
  }
  return "exit code " + std::to_string(rc);
}

struct Act { enum K { NOTE, SAY, SPAWN, GO, TERMINATE, ESCALATE, CLEAN } k; std::string s; };
using Acts = std::vector<Act>;

struct MachineCfg { double load_timeout_s = 90, run_max_s = 90, abort_escalate_s = 2.5, settle_stable_s = 1.0, settle_timeout_s = 20.0; };
// what the service observes after a run (all read-only): fed to Machine::tick every loop
struct Settle { bool fsm_ok = false; double stable_s = 0; bool lock_free = false, guardian_ok = false; };

// Pure state machine (no I/O): inputs are events, outputs are actions. (Port of fasttrack/supervisor.py Machine.)
class Machine {
 public:
  Machine(MachineCfg c, bool armed) : cfg(c), armed(armed) {}
  MachineCfg cfg;
  bool armed;
  St state = St::IDLE;
  double since = 0, deadline = -1, abort_deadline = -1;
  std::string settle_note, detail, guardian_why;  // guardian_why: set by the service (why the guardian is not alive: exact cause, not a generic text)
  bool user_abort = false, escalated = false, g_warned = false;
  bool exit_recoverable = false;  // set by the service from run_report.json before runner_exit(): WalkRun confirmed, no untrusted/emergency/fallen cause
  std::string settle_why;
  int last_rc = -1;
  void go(St s, double now) { state = s; since = now; }

  Acts chord_load(double now, bool guardian_ok) {
    if (state == St::SETTLE) return {{Act::NOTE, "F1+F3 ignored: the last run just ended, waiting until " + settle_why + " (no ack needed, a few seconds)"}};
    if (state != St::IDLE) return {{Act::NOTE, std::string("F1+F3 ignored in state ") + st_name(state)}};
    if (!guardian_ok) return {{Act::NOTE, "guardian not alive: F1+F3 ignored - " + (guardian_why.empty() ? std::string("it is restarted automatically; try again in a few seconds") : guardian_why)}, {Act::SAY, "检查失败"}};
    user_abort = false;
    go(St::LOADING, now);
    deadline = now + cfg.load_timeout_s;
    return {{Act::SAY, "正在加载"}, {Act::SPAWN, ""}};
  }
  Acts chord_go(double now, bool guardian_ok) {
    if (!armed) return {{Act::NOTE, "service is in check mode (read-only): L1+A ignored, the robot is not touched"}, {Act::SAY, "检查失败"}};
    if (state != St::READY) return {{Act::NOTE, std::string("L1+A ignored in state ") + st_name(state) + " (needs READY: press F1+F3 first)"}};
    if (!guardian_ok) { user_abort = true; return {{Act::NOTE, "guardian not alive: refusing GO, aborting the prepared runner"}, {Act::SAY, "检查失败"}, {Act::TERMINATE, "guardian dead at GO"}}; }
    go(St::RUNNING, now);
    deadline = now + cfg.run_max_s;
    abort_deadline = -1;
    return {{Act::GO, ""}};
  }
  Acts start_double(double now) {
    if (state == St::LOADING || state == St::READY) {
      user_abort = true;
      return {{Act::NOTE, "START x2: abort before control was taken"}, {Act::SAY, "已退出"}, {Act::TERMINATE, "START x2"}};
    }
    if (state == St::RUNNING) {
      user_abort = true;
      abort_deadline = now + cfg.abort_escalate_s;
      return {{Act::NOTE, "START x2 during RUN: the runner stops by itself (guardian is the backup); watching for the hand-back"}};
    }
    return {{Act::NOTE, std::string("START x2 in ") + st_name(state) + ": nothing to stop"}};
  }
  Acts runner_status(const std::string& phase, double now) {
    if (state == St::LOADING && phase == "READY") {
      go(St::READY, now);
      deadline = -1;  // V1.6.1: READY has NO timeout: it is held until L1+A (run) or START x2 (cancel)
      return {{Act::NOTE, "READY: model loaded, real-time admission passed. Held until L1+A (go) / START x2 (cancel); no timeout"}, {Act::SAY, "准备就绪"}};
    }
    return {};
  }
  Acts tick(double now, bool runner_alive, bool guardian_ok, const Settle& sx = Settle(), bool reported = false) {
    Acts a;
    if (state == St::SETTLE) {
      // V1.6.1: EVERY run end (normal or abnormal) returns to IDLE by itself, no ack, no confirmation. After a RECOVERABLE end we also wait for WalkRun + a quiet attitude; after any other end
      // (fall / emergency PASSIVE / robot down / unconfirmed recovery) only the process-level conditions count (runner ended, lock released, guardian ready): the robot may legitimately stay down until
      // someone stands it up, and the NEXT F1+F3 real-time admission (FSM in the WalkRun family, attitude, lock, e-stop file, guardian, ...) decides whether a new run may start.
      std::string miss;
      if (runner_alive) miss += "the runner process ends; ";
      if (exit_recoverable) {
        if (!sx.fsm_ok) miss += "WalkRun (fsm 500/801/802) is confirmed; ";
        if (sx.stable_s < cfg.settle_stable_s) { char b[96]; snprintf(b, sizeof b, "the attitude is quiet for %.1f s (now %.1f s); ", cfg.settle_stable_s, std::max(0.0, sx.stable_s)); miss += b; }
      }
      if (!sx.lock_free) miss += "the control lock is released; ";
      if (!sx.guardian_ok) miss += "the guardian is ready; ";
      settle_why = miss.empty() ? "ready" : miss;
      if (miss.empty()) {
        go(St::IDLE, now);
        a.push_back({Act::NOTE, exit_recoverable ? "SETTLED: WalkRun confirmed, attitude quiet, lock released, guardian ready -> IDLE (F1+F3 may be pressed again; control is never taken automatically)"
                                                 : "SETTLED (abnormal end: " + detail + "): runner ended, lock released, guardian ready -> IDLE. The next F1+F3 admission checks the robot state (FSM / attitude / e-stop) by itself."});
        if (exit_recoverable) a.push_back({Act::SAY, "可再次开始"});
        a.push_back({Act::CLEAN, ""});
      } else if (runner_alive && now - since > cfg.settle_timeout_s) {  // never IDLE with a live runner: stop it (it only finishes its post-exit recording) and keep waiting
        a.push_back({Act::NOTE, "SETTLE: the runner process is still alive " + std::to_string((int)cfg.settle_timeout_s) + " s after the run ended: SIGTERM (it only finishes its post-exit recording)"});
        a.push_back({Act::TERMINATE, "settle timeout"});
        since = now;
      } else if (now - since > cfg.settle_timeout_s) {
        go(St::IDLE, now);
        a.push_back({Act::NOTE, "SETTLE timeout after " + std::to_string((int)cfg.settle_timeout_s) + " s, still waiting until: " + miss + "(" + detail + ") -> IDLE anyway (no ack needed); the next F1+F3 admission decides whether a new run may start"});
        a.push_back({Act::CLEAN, ""});
      }
      return a;
    }
    if ((state == St::LOADING || state == St::READY) && deadline >= 0 && now > deadline && runner_alive) {
      a.push_back({Act::NOTE, std::string(st_name(state)) + " timed out"});
      a.push_back({Act::SAY, "已退出"});
      a.push_back({Act::TERMINATE, "timeout"});
      deadline = -1; user_abort = true;
    }
    if (state == St::RUNNING && !reported) {  // P5.3: once the runner REPORTED the hand-back it only finishes its post-exit recording: no escalation / run-time limit applies any more
      if (deadline >= 0 && now > deadline && runner_alive) { a.push_back({Act::ESCALATE, "runner exceeded the maximum run time"}); deadline = -1; escalated = true; }
      if (abort_deadline >= 0 && now > abort_deadline && runner_alive) { a.push_back({Act::ESCALATE, "runner did not finish the START x2 hand-back in time"}); abort_deadline = -1; escalated = true; }
      if (!guardian_ok && runner_alive && !g_warned) {
        g_warned = true;
        a.push_back({Act::NOTE, "CRITICAL: guardian died while the runner is in control: stopping the runner"});
        a.push_back({Act::TERMINATE, "guardian dead"});
      }
    }
    return a;
  }
  Acts runner_exit(int rc, double now) {
    last_rc = rc;
    St was = state;
    deadline = abort_deadline = -1;
    g_warned = false;
    std::string txt = rc_text(rc);
    if (was == St::LOADING || was == St::READY) {
      go(St::IDLE, now);
      return {{Act::NOTE, "runner ended before taking control (rc=" + std::to_string(rc) + ": " + txt + "); service back to IDLE"}, {Act::CLEAN, ""}};
    }
    if (was == St::RUNNING) {
      if (rc == 6) { go(St::IDLE, now); return {{Act::NOTE, "GO refused (rc=6: " + txt + "); nothing was sent; back to IDLE"}, {Act::CLEAN, ""}}; }
      if (rc == 7) { go(St::IDLE, now); return {{Act::NOTE, "take-over CANCELLED before control was taken (rc=7: " + txt + "); the robot was never touched; back to IDLE"}, {Act::CLEAN, ""}}; }  // V1.6 announce_wait
      // V1.6.1: normal end, operator stop, fault, fall, emergency PASSIVE, escalation: ALL go SETTLE -> IDLE without any acknowledgement (no HOLD state any more). Nothing is taken over automatically.
      const bool was_escalated = escalated; escalated = false;
      go(St::SETTLE, now);
      settle_why = "runner / lock / guardian are confirmed";
      const bool normal = (rc == 0 || rc == 10) && exit_recoverable && !was_escalated;
      if (normal)
        return {{Act::NOTE, "RUN FINISHED rc=" + std::to_string(rc) + ": " + txt + (rc == 10 ? (user_abort ? " (operator START x2, hand-back confirmed)" : " (recoverable fault, WalkRun recovery confirmed)") : "") +
                                "; settling (no ack needed): waiting for WalkRun / attitude / lock / guardian"}};
      exit_recoverable = false;  // the recoverable-only conditions (WalkRun / attitude) do not apply to an abnormal end
      return {{Act::NOTE, "RUN FINISHED rc=" + std::to_string(rc) + ": " + txt + (was_escalated ? " [escalated: the e-stop latch written by the service is cleared when settled]" : "") + (detail.empty() ? "" : " [" + detail + "]") +
                              " - ABNORMAL end; settling (no ack needed): waiting for runner / lock / guardian, then IDLE"}};
    }
    return {{Act::NOTE, "runner exit rc=" + std::to_string(rc) + " in " + st_name(was)}};
  }
};

// ------------------------------------------------------------------ child processes (posix_spawn, own process group; remote_start style)
struct Child {
  pid_t pid = -1;
  std::string name;
  int rc = 0;
  bool alive() const { return pid > 0; }
};

bool spawn_child(Child& c, const std::vector<std::string>& args, const std::string& log_path, const std::string& cwd) {
  std::vector<char*> argv;
  for (auto& s : args) argv.push_back(const_cast<char*>(s.c_str()));
  argv.push_back(nullptr);
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, 1, log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0664);
  posix_spawn_file_actions_adddup2(&fa, 1, 2);
  posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
#if defined(__GLIBC__) && (__GLIBC__ > 2 || __GLIBC_MINOR__ >= 29)
  if (!cwd.empty()) posix_spawn_file_actions_addchdir_np(&fa, cwd.c_str());
#endif
  posix_spawnattr_t at;
  posix_spawnattr_init(&at);
  posix_spawnattr_setflags(&at, POSIX_SPAWN_SETPGROUP);
  posix_spawnattr_setpgroup(&at, 0);
  pid_t pid = -1;
  int rc = posix_spawn(&pid, args[0].c_str(), &fa, &at, argv.data(), environ);
  posix_spawnattr_destroy(&at);
  posix_spawn_file_actions_destroy(&fa);
  if (rc != 0) { logf("posix_spawn failed for %s: %s", c.name.c_str(), strerror(rc)); return false; }
  c.pid = pid;
  return true;
}
bool reap(Child& c) {
  if (c.pid <= 0) return false;
  int status = 0;
  pid_t r = waitpid(c.pid, &status, WNOHANG);
  if (r == c.pid) {
    c.rc = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    c.pid = -1;
    return true;
  }
  if (r < 0 && errno == ECHILD) { c.rc = 1; c.pid = -1; return true; }
  return false;
}
void term_child(Child& c, const std::string& why) {
  if (c.pid > 0) { killpg(c.pid, SIGTERM); logf("SIGTERM -> pid %d (%s)", (int)c.pid, why.c_str()); }
}

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

struct Ev { enum K { KEYS, LOAD, GO, START2 } k; double t; uint16_t keys; uint8_t b2, b3; };

class Service {
 public:
  Service(const Args& a, const Paths& P, const Site& site) : a_(a), P_(P), site_(site) {
    sim_ = a.has("--sim");
    iface_ = sim_ ? "lo" : a.get("--iface", site.iface);
    domain_ = sim_ ? SIM_DOMAIN : (int)a.num("--domain", 0);
    armed_ = !(sim_ && a.has("--not-armed"));  // V1.6.1: always armed (the only unarmed case is the read-only check mode / the dev flag --not-armed under --sim)
    run_s_ = a.num("--run-s", site.run_s);
    authority_ = a.num("--authority-max", site.authority);
    ref_mode_ = a.get("--ref-mode", site.ref_mode);
    ramp_s_ = a.num("--ramp-s", site.ramp_s);
    end_hold_s_ = a.num("--end-hold-s", site.end_hold_s);
    allow_partial_ = a.has("--allow-partial-authority") || site.allow_partial_authority;
    torque_ff_ = a.has("--torque-ff"); no_torque_ff_ = a.has("--no-torque-ff");
    standby_disabled_ = a.has("--no-standby");  // V1.5.1: no pre-warmed runner (F1+F3 then loads model + DDS itself)
    if (a.kv.count("--mode")) {  // P5.1: a named test mode (config/modes.conf) sets the window and the reference mode; explicit options override it (and then it is NOT the validated envelope)
      std::string e;
      if (!load_modes(P.config + "/modes.conf", modes_, e)) mode_err_ = e;
      else {
        const std::string want = a.get("--mode");
        for (auto& m : modes_) if (m.name == want) { mode_ = m; have_mode_ = true; }
        if (!have_mode_) { mode_err_ = "unknown mode '" + want + "' (see config/modes.conf)"; }
        else {
          mode_overrides_ = a.kv.count("--run-s") || a.kv.count("--ref-mode") || a.kv.count("--authority-max");
          if (!a.kv.count("--run-s")) run_s_ = mode_.run_s;
          if (!a.kv.count("--ref-mode")) ref_mode_ = mode_.ref_mode;
          if (mode_.check_only) armed_ = false;
        }
      }
    }
    m_ = std::unique_ptr<Machine>(new Machine(MachineCfg{90, run_s_ + 60.0, 2.5, site.settle_stable_s, site.settle_timeout_s}, armed_));
    self_ = P.root + "/build/partymimic";
    char b[4096];
    ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1);
    if (n > 0) self_ = std::string(b, (size_t)n);
  }

  int run();

 private:
  void log_state(const std::string& msg) { logf("[%s] %s", st_name(m_->state), msg.c_str()); svc_event("state_note", jstr("state", st_name(m_->state)) + "," + jstr("msg", msg)); }
  // P5.1: structured service timeline (logs/service_events.jsonl + the current run dir): chords, state changes, voice, child processes
  void svc_event(const std::string& kind, const std::string& fields = "") {
    std::lock_guard<std::mutex> lk(sev_m_);
    char b[160];
    snprintf(b, sizeof b, "{\"t_mono\":%.6f,\"wall\":%.4f,\"kind\":\"%s\"", mono(), wall(), kind.c_str());
    std::string line = b + (fields.empty() ? std::string() : "," + fields) + "}\n";
    for (const std::string& f : {P_.logs + "/service_events.jsonl", run_dir_.empty() || !run_active_ ? std::string() : run_dir_ + "/service_events.jsonl"}) {
      if (f.empty()) continue;
      FILE* fp = fopen(f.c_str(), "a");
      if (fp) { fputs(line.c_str(), fp); fclose(fp); }
    }
  }
  uint64_t say_(const std::string& text, const char* why = "") {
    svc_event("voice", jstr("text", text) + "," + jstr("why", why) + "," + jstr("state", st_name(m_->state)));
    return tts_.say(text);
  }
  void write_state();
  void on_lowstate(const LinkState& st);
  bool guardian_ok();
  bool guardian_ok_running();
  void ensure_guardian();
  void on_guardian_ended();
  void spawn_runner();
  void spawn_standby();
  RunnerCmd runner_cmd() const;
  void do_acts(const Acts& acts);
  void report_summary(int rc, bool user_abort, St was);
  void record_mode_pass();
  void reset_run_state();      // P5.3: clear everything a finished run leaves behind (go / status files, phase, abort + escalation flags, pending announcements)
  Settle settle_status();
  static bool fault_recoverable(const std::string& fault) {  // the exit cause leaves the robot standing under WalkRun and the state trustworthy
    static const char* bad[] = {"COMM_TIMEOUT", "STATE_FROZEN", "STATE_INVALID", "FSM_CHANGED", "LOCK_LOST", "SWITCH_FAILED", "ESTOP", "REMOTE_DAMP", "TILT"};
    for (const char* b : bad) if (fault == b) return false;
    return true;
  }
  void tick_children();
  void handle_event(const Ev& ev);

  Args a_;
  Paths P_;
  Site site_;
  bool sim_ = false, armed_ = false;
  std::string iface_, ref_mode_, self_;
  double ramp_s_ = 0.3, end_hold_s_ = 0.5;
  bool allow_partial_ = false, torque_ff_ = false, no_torque_ff_ = false;
  int domain_ = 0;
  double run_s_, authority_;
  std::unique_ptr<Machine> m_;
  std::mutex sev_m_;
  bool run_active_ = false, reported_voiced_ = false;
  std::vector<ModeSpec> modes_;
  ModeSpec mode_;
  bool have_mode_ = false, mode_overrides_ = false;
  std::string mode_err_;
  Tts tts_;
  std::mutex qm_;
  std::deque<Ev> q_;
  std::condition_variable qcv_;
  Child runner_, guardian_;
  std::string run_dir_, runner_phase_, guardian_report_, last_result_ = "NONE", last_final_mode_ = "NONE";
  int last_final_fsm_ = -1;
  bool last_recovery_ok_ = false, pending_guardian_report_ = false;
  double pending_guardian_t_ = 0;
  double guardian_last_restart_ = 0, guardian_started_t_ = 0, guardian_backoff_s_ = 1.0;
  int guardian_quick_fails_ = 0;
  bool guardian_fatal_ = false, guardian_fatal_said_ = false;
  std::string guardian_why_, guardian_log_;
  std::atomic<double> last_lowstate_{0}, quiet_since_{0};  // quiet_since_: mono() since when tilt < 0.15 rad and |gyro_xy| < 1.5 rad/s (0 = not quiet)
  std::atomic<double> last_tilt_{0}, last_gyro_{0};
  std::atomic<uint16_t> last_keys_{0};
  uint16_t prev_keys_ = 0;
  std::unique_ptr<ChordDetector> chord_load_, chord_go_;
  StartDoubleClick dbl_;
  FILE* keysf_ = nullptr;
  std::string state_file_, status_file_, go_file_, trigger_file_;
  bool escalation_latched_ = false;  // the e-stop latch file was written by THIS service (escalation): it is removed again when the service has settled (V1.6.1)
  bool standby_proc_ = false, standby_disabled_ = false;  // standby_proc_: the live runner process is a pre-warmed one that has not been triggered yet
  double next_standby_t_ = 0, standby_spawn_t_ = 0;
  int standby_quick_fails_ = 0;
  bool pending_go_ = false;
  double pending_go_t0_ = 0, keys_free_since_ = 0;
  void release_gate();
  void announce_gate();   // V1.6: "开始执行" finished? -> ANNOUNCED file (the runner then waits announce_wait_s and takes over)
  // V1.6 announce_wait: GO file written -> runner in phase GO_WAIT (monitoring, NOT in control) -> announcement -> ANNOUNCED file -> runner waits announce_wait_s -> re-verification -> take-over
  struct GoPhase { bool on = false; uint64_t ticket = 0; double t0 = 0; } go_ph_;
  std::string announced_file_;
};

void Service::write_state() {
  char b[2048];
  double age = last_lowstate_.load() > 0 ? mono() - last_lowstate_.load() : -1;
  snprintf(b, sizeof b,
           "{\"state\":\"%s\",\"since\":%.3f,\"time\":%.3f,\"settle_why\":\"%s\",\"armed\":%s,\"target\":\"%s\",\"interface\":\"%s\",\"guardian_alive\":%s,\"lowstate_age_s\":%.3f,\"last_rc\":%d,\"pressed\":\"%s\",\"pid\":%d,"
           "\"last_result\":\"%s\",\"last_final_mode\":\"%s\",\"last_final_fsm\":%d,\"last_recovery_ok\":%s,\"guardian_fatal\":%s,\"guardian_why\":\"%s\",\"mode\":\"%s\",\"run_s\":%.2f,\"ref_mode\":\"%s\",\"standby\":\"%s\"}",
           st_name(m_->state), m_->since, wall(), json_escape(m_->state == St::SETTLE ? m_->settle_why : std::string()).c_str(), armed_ ? "true" : "false", sim_ ? "sim" : "real", iface_.c_str(), guardian_ok() ? "true" : "false", age, m_->last_rc,
           keys_text(last_keys_.load()).c_str(), (int)getpid(), json_escape(last_result_).c_str(), json_escape(last_final_mode_).c_str(), last_final_fsm_, last_recovery_ok_ ? "true" : "false", guardian_fatal_ ? "true" : "false", json_escape(guardian_why_).c_str(),
           have_mode_ ? mode_.name.c_str() : "custom", run_s_, ref_mode_.c_str(),
           !runner_.alive() || !standby_proc_ ? (standby_disabled_ ? "disabled" : "none") : (runner_phase_ == "STANDBY" ? "ready" : "loading"));
  write_atomic(state_file_, b);
}

void Service::on_lowstate(const LinkState& st) {  // DDS reader thread: cheap, no I/O
  last_lowstate_ = mono();
  {  // attitude quietness for the post-run settle (read-only)
    double tilt = tilt_of_quat(st.quat), gxy = std::hypot(st.gyro[0], st.gyro[1]);
    last_tilt_ = std::isfinite(tilt) ? tilt : 9.9; last_gyro_ = gxy;
    const bool quiet = std::isfinite(tilt) && tilt < 0.15 && gxy < 1.5;
    if (quiet) { if (quiet_since_.load() == 0) quiet_since_ = st.arrival; } else quiet_since_ = 0;
  }
  uint16_t k = keys_from_wireless(st.wireless);
  double t = st.arrival;
  std::vector<Ev> evs;
  if (k != prev_keys_) { evs.push_back({Ev::KEYS, t, k, st.wireless[2], st.wireless[3]}); prev_keys_ = k; last_keys_ = k; }
  if (chord_load_->feed(t, k)) evs.push_back({Ev::LOAD, t, k, 0, 0});
  if (chord_go_->feed(t, k)) evs.push_back({Ev::GO, t, k, 0, 0});
  if (dbl_.feed(t, start_pressed(k))) evs.push_back({Ev::START2, t, k, 0, 0});
  if (evs.empty()) return;
  std::lock_guard<std::mutex> lk(qm_);
  for (auto& e : evs) q_.push_back(e);
  qcv_.notify_one();
}

bool Service::guardian_ok() {
  GuardianAlive g;
  if (!GuardianAlive::read(GUARDIAN_ALIVE, g)) return false;
  return wall() - g.t < GUARDIAN_MAX_AGE_S && guardian_.alive() && g.iface == iface_;
}
bool Service::guardian_ok_running() {  // tolerant liveness for the in-control phase: process alive and heartbeat < 3 s old
  GuardianAlive g;
  if (!GuardianAlive::read(GUARDIAN_ALIVE, g)) return false;
  return guardian_.alive() && wall() - g.t < 3.0;
}

// P4.1: the guardian is restarted with a BACKOFF and never in an endless 1 Hz loop. A guardian that exits with rc 2 (bad arguments / configuration: deterministic, a restart
// can never help) latches guardian_fatal_ at once; any other very short life (<5 s) is retried with 1,2,4,8,16 s backoff and latches after 5 quick failures. The reason
// (last line of the guardian log) is logged, stored in state.json ("guardian_fatal") and quoted in the F1+F3 refusal instead of a generic "guardian not alive".
void Service::ensure_guardian() {
  if (guardian_.alive() || guardian_fatal_) return;
  double now = mono();
  if (now - guardian_last_restart_ < std::max(1.0, guardian_backoff_s_)) return;
  guardian_last_restart_ = now;
  guardian_.name = "guardian";
  std::string ts = timestamp_compact();
  guardian_report_ = P_.logs + "/guardian/guardian_" + ts + ".json";
  guardian_log_ = P_.logs + "/guardian/guardian_" + ts + ".log";
  GuardianCmd gc;
  gc.self = self_; gc.iface = iface_; gc.domain = domain_; gc.report = guardian_report_; gc.sim = sim_;
  if (a_.kv.count("--entry-fsm")) gc.entry_fsm = a_.get("--entry-fsm");
  std::vector<std::string> args = guardian_cmdline(gc);
  if (sim_ && getenv("PM_TEST_GUARDIAN_ARGS")) {  // DEV ONLY (--sim): extra raw guardian arguments, to test the failure path of the start-up
    std::string x = getenv("PM_TEST_GUARDIAN_ARGS"); size_t p = 0;
    while (p < x.size()) { size_t q = x.find(' ', p); if (q == std::string::npos) q = x.size(); if (q > p) args.push_back(x.substr(p, q - p)); p = q + 1; }
  }
  if (spawn_child(guardian_, args, guardian_log_, P_.root)) {
    guardian_started_t_ = mono();
    logf("guardian started pid %d -> logs/guardian/guardian_%s.log", (int)guardian_.pid, ts.c_str());
  } else {
    guardian_why_ = "posix_spawn failed for the guardian";
    if (++guardian_quick_fails_ >= 5) { guardian_fatal_ = true; guardian_why_ += " (5 attempts)"; }
    guardian_backoff_s_ = std::min(30.0, std::pow(2.0, guardian_quick_fails_));
  }
}

void Service::on_guardian_ended() {
  double life = mono() - guardian_started_t_;
  std::string last;  // last non-empty line of the guardian's own log (its stderr is redirected there)
  { std::string doc; if (read_file(guardian_log_, doc)) { size_t e = doc.find_last_not_of(" \r\n"); if (e != std::string::npos) { size_t b = doc.find_last_of('\n', e); last = doc.substr(b == std::string::npos ? 0 : b + 1, e - (b == std::string::npos ? 0 : b + 1) + 1); } } }
  if (last.size() > 220) last = last.substr(0, 220);
  if (life >= 5.0) { guardian_quick_fails_ = 0; guardian_backoff_s_ = 1.0; guardian_why_ = "guardian ended after " + std::to_string((int)life) + " s (rc=" + std::to_string(guardian_.rc) + "); restarting"; return; }
  ++guardian_quick_fails_;
  guardian_why_ = "guardian exited after " + std::to_string((int)(life * 1000)) + " ms with rc=" + std::to_string(guardian_.rc) + (last.empty() ? "" : ": " + last);
  if (guardian_.rc == 2 || guardian_quick_fails_ >= 5) {
    guardian_fatal_ = true;
    logf("ERROR: guardian cannot start (%s). The restart loop is STOPPED (%s). Fix the cause and restart the service; F1+F3 stays refused until then.", guardian_why_.c_str(), guardian_.rc == 2 ? "rc 2 = bad arguments/configuration, a restart cannot help" : "5 consecutive quick failures");
    if (!guardian_fatal_said_) { guardian_fatal_said_ = true; say_("检查失败"); }
  } else {
    guardian_backoff_s_ = std::min(30.0, std::pow(2.0, guardian_quick_fails_));
    logf("WARNING: %s; retry %d/5 in %.0f s", guardian_why_.c_str(), guardian_quick_fails_, guardian_backoff_s_);
  }
}

RunnerCmd Service::runner_cmd() const {
  RunnerCmd rc;
  rc.self = self_; rc.iface = iface_; rc.domain = domain_; rc.run_s = run_s_; rc.ref_mode = ref_mode_; rc.authority = authority_; rc.ramp_s = ramp_s_; rc.end_hold_s = end_hold_s_;
  rc.go_file = go_file_; rc.status_file = status_file_; rc.sim = sim_; rc.allow_partial = allow_partial_;
  if (have_mode_) rc.mode_name = mode_.name;
  rc.announced_file = announced_file_;
  if (a_.kv.count("--announce-wait-s")) rc.announce_wait_s = a_.get("--announce-wait-s");  // DEV/test override of site.conf announce_wait_s (the runner range-checks it)
  rc.check_only = !armed_; rc.torque_ff = torque_ff_; rc.no_torque_ff = no_torque_ff_;  // an unarmed service (check mode) only ever runs the read-only check
  if (a_.kv.count("--entry-fsm")) rc.entry_fsm = a_.get("--entry-fsm");
  if (a_.kv.count("--unconfirmed-hold-s")) rc.unconfirmed_hold_s = a_.get("--unconfirmed-hold-s");
  return rc;
}

// V1.5.1: a pre-warmed runner (see the top of this file and runner.cpp). It never takes control by itself: it idles until the trigger file exists.
void Service::spawn_standby() {
  if (standby_disabled_ || runner_.alive() || guardian_fatal_) return;
  const double now = mono();
  if (now < next_standby_t_) return;
  ::unlink(go_file_.c_str());
  ::unlink(announced_file_.c_str());
  ::unlink(trigger_file_.c_str());
  ::unlink(status_file_.c_str());
  runner_phase_.clear();
  runner_.name = "runner";
  RunnerCmd rc = runner_cmd();
  rc.standby = true; rc.trigger_file = trigger_file_;
  std::string out = P_.logs + "/standby.out";
  struct stat sb;
  if (::stat(out.c_str(), &sb) == 0 && sb.st_size > (4 << 20)) ::unlink(out.c_str());  // bounded
  if (!spawn_child(runner_, runner_cmdline(rc), out, P_.root)) { next_standby_t_ = now + 10; return; }
  standby_proc_ = true; run_active_ = false; standby_spawn_t_ = now;
  svc_event("standby_spawn", jnum("pid", runner_.pid));
  logf("standby runner started pid %d (model load + warm-up + DDS monitors happen now, before any F1+F3) -> logs/standby.out", (int)runner_.pid);
}

void Service::spawn_runner() {
  std::string ts = timestamp_compact();
  run_dir_ = P_.logs + "/runs/" + ts;
  ::mkdir(run_dir_.c_str(), 0775);
  ::unlink(go_file_.c_str());
  ::unlink(announced_file_.c_str());
  go_ph_ = GoPhase();
  std::string latest = P_.logs + "/latest";
  if (runner_.alive() && standby_proc_) {  // F1+F3 on a pre-warmed runner: only the trigger (the run directory travels in it)
    ::unlink(latest.c_str());
    if (symlink(run_dir_.c_str(), latest.c_str()) != 0) {}
    reported_voiced_ = false; run_active_ = true; standby_proc_ = false;
    svc_event("runner_spawn", jstr("run_dir", run_dir_) + "," + jstr("mode", have_mode_ ? mode_.name : "custom") + "," + jstr("via", "standby_trigger") + "," + jstr("standby_phase", runner_phase_));
    write_atomic(trigger_file_, "{" + jstr("run_dir", run_dir_) + "," + jnum("wall", wall()) + "," + jnum("t_mono", mono()) + "}\n");
    logf("pre-warmed runner pid %d triggered -> logs/runs/%s/ (standby phase was %s)", (int)runner_.pid, ts.c_str(), runner_phase_.empty() ? "?" : runner_phase_.c_str());
    return;
  }
  ::unlink(status_file_.c_str());
  runner_phase_.clear();
  runner_.name = "runner";
  RunnerCmd rc = runner_cmd();
  rc.run_dir = run_dir_;
  run_active_ = true; reported_voiced_ = false; standby_proc_ = false;
  svc_event("runner_spawn", jstr("run_dir", run_dir_) + "," + jstr("mode", have_mode_ ? mode_.name : "custom") + "," + jstr("via", "cold_start"));
  std::vector<std::string> args = runner_cmdline(rc);
  if (!spawn_child(runner_, args, run_dir_ + "/runner.out", P_.root)) { m_->state = St::IDLE; run_active_ = false; return; }
  ::unlink(latest.c_str());
  if (symlink(run_dir_.c_str(), latest.c_str()) != 0) {}
  logf("runner started (cold: no pre-warmed runner available) pid %d -> logs/runs/%s/", (int)runner_.pid, ts.c_str());
}

void Service::do_acts(const Acts& acts) {
  for (auto& a : acts) {
    switch (a.k) {
      case Act::NOTE: log_state(a.s); break;
      case Act::SAY: say_(a.s); break;
      case Act::SPAWN: spawn_runner(); break;
      case Act::GO:  // the GO file is written only after every key is released (the factory remote also sees L1+A): see release_gate()
        pending_go_ = true; pending_go_t0_ = mono(); keys_free_since_ = 0;
        log_state("GO requested (L1+A): waiting for all keys to be released before the runner takes control");
        break;
      case Act::TERMINATE: term_child(runner_, a.s); break;
      case Act::CLEAN: reset_run_state(); break;
      case Act::ESCALATE:
        logf("CRITICAL ESCALATE: %s -> e-stop latch + SIGTERM to the runner (guardian backup stays active)", a.s.c_str());
        write_atomic(DEFAULT_ESTOP, "service escalation " + std::to_string(wall()) + "\n");
        escalation_latched_ = true;
        say_("检查失败");
        term_child(runner_, a.s);
        break;
    }
  }
}

// The announcement for a finished exit: the voice comes from the REAL result (run_report.json "voice", set by the exit classification AFTER the RPC and the FSM
// confirmation), never from "the process ended normally".
static std::string voice_of(const std::string& doc, const std::string& result) {
  std::string v = json_find_string(doc, "voice");
  if (!v.empty() && tts_known(v)) return v;
  if (result == "COMPLETED") return "执行完成";
  if (result == "EXITED_WALKRUN") return "异常已交还";
  if (result == "WALKRUN_RECOVERY_FAILED" || result == "FALLEN_PASSIVE") return "恢复失败";
  if (result == "EMERGENCY_PASSIVE") return "紧急被动";
  return "检查失败";
}

Settle Service::settle_status() {
  Settle x;
  x.fsm_ok = last_recovery_ok_ && (last_final_fsm_ == 500 || last_final_fsm_ == 801 || last_final_fsm_ == 802) && last_final_mode_ == "WALKRUN";
  const double age = last_lowstate_.load() > 0 ? mono() - last_lowstate_.load() : 1e9;
  x.stable_s = (age < 0.3 && quiet_since_.load() > 0) ? mono() - quiet_since_.load() : 0;
  x.lock_free = !lock_probe(DEFAULT_LOCK).held;
  x.guardian_ok = guardian_ok() && !guardian_fatal_;
  return x;
}

void Service::reset_run_state() {
  ::unlink(go_file_.c_str());
  ::unlink(announced_file_.c_str());
  go_ph_ = GoPhase();
  ::unlink(trigger_file_.c_str());
  ::unlink(status_file_.c_str());
  runner_phase_.clear();
  pending_go_ = false; pending_go_t0_ = 0; keys_free_since_ = 0;
  pending_guardian_report_ = false;
  if (escalation_latched_ && !runner_.alive()) {  // the latch the SERVICE wrote at an escalation is removed once the run has settled; an operator latch (./stop.sh --estop) is never touched
    std::string d;
    if (read_file(DEFAULT_ESTOP, d) && d.rfind("service escalation", 0) == 0) { ::unlink(DEFAULT_ESTOP); logf("e-stop latch written by the service escalation removed (run settled)"); }
    escalation_latched_ = false;
  }
  m_->user_abort = false; m_->escalated = false; m_->g_warned = false; m_->exit_recoverable = false; m_->deadline = m_->abort_deadline = -1;
  svc_event("run_state_cleared", jstr("note", "go/status files removed, phase + abort/escalation flags reset; the next runner is a new process (fresh Policy history, previous action and reference index)"));
}

// one line per finished run + the voice announcement of the outcome (details: logs/runs/<time>/run_report.json)
void Service::report_summary(int rc, bool user_abort, St was) {
  std::string doc;
  m_->detail.clear();
  if (!read_file(run_dir_ + "/run_report.json", doc)) {
    logf("no run_report.json (runner rc=%d)", rc);
    if (rc >= 0 && !run_dir_.empty()) {  // P5.1: the runner died without a report: rebuild one from the log files (CRASH_NO_REPORT) + anomaly summary
      FinalizeResult fr = finalize_run(run_dir_, "", "service salvage (runner rc=" + std::to_string(rc) + ")");
      logf("salvaged run_report.json from the logs: %s (%d fail, %d warn) -> %s/anomaly_summary.txt", fr.level.c_str(), fr.n_fail, fr.n_warn, run_dir_.c_str());
      svc_event("run_salvaged", jstr("run_dir", run_dir_) + "," + jnum("runner_rc", rc));
    }
    if (rc > 0) {
      if (was == St::RUNNING && guardian_.alive() && (runner_phase_ == "ARM" || runner_phase_ == "RUN" || runner_phase_ == "EXIT")) {  // the runner died in control: the guardian runs the SAME exit policy; its result decides what is said
        pending_guardian_report_ = true; pending_guardian_t_ = mono();
        logf("runner ended without a report: waiting for the guardian take-over result (%s)", guardian_report_.c_str());
      } else say_("检查失败");
    }
    return;
  }
  std::string result = json_find_string(doc, "result"), reason = json_find_string(doc, "exit_reason"), fault = json_find_string(doc, "fault"), mode = json_find_string(doc, "requested_mode");
  bool confirmed = json_find_raw(doc, "exit_confirmed") == "true", took = json_find_raw(doc, "took_control") == "true", rec_ok = json_find_raw(doc, "recovery_ok") == "true";
  m_->exit_recoverable = took && rec_ok && (result == "COMPLETED" || result == "EXITED_WALKRUN") && fault_recoverable(fault);
  last_result_ = result; last_final_mode_ = json_find_string(doc, "final_mode"); last_final_fsm_ = atoi(json_find_raw(doc, "final_fsm").c_str()); last_recovery_ok_ = rec_ok;
  logf("run result=%s reason=%s fault=%s requested_mode=%s final_mode=%s final_fsm=%d recovery_ok=%d entry_fsm=%s", result.c_str(), reason.c_str(), fault.c_str(), mode.c_str(), last_final_mode_.c_str(), last_final_fsm_,
       (int)rec_ok, json_find_raw(doc, "entry_fsm").c_str());
  m_->detail = "result " + result + ", final_fsm " + std::to_string(last_final_fsm_) + ", final_mode " + last_final_mode_ + ", recovery_ok " + (rec_ok ? "true" : "false");
  if (took && !rec_ok && result != "EMERGENCY_PASSIVE" && result != "COMPLETED")
    logf("!!! RECOVERY NOT CONFIRMED (%s): the OEM is in fsm %d (%s), not the requested %s. Check the robot, read logs/runs/*/run_report.json, the next F1+F3 admission decides (no ack needed)", result.c_str(), last_final_fsm_, last_final_mode_.c_str(), mode.c_str());
  size_t pos = 0;
  while ((pos = doc.find("\"pass\":false", pos)) != std::string::npos) {  // BLOCKED items
    size_t s = doc.rfind("{\"item\":\"", pos);
    if (s != std::string::npos) logf("  BLOCKED item: %s", doc.substr(s, doc.find('}', pos) - s).c_str());
    pos += 12;
  }
  // voice: short phrases only, the reasons are in the log
  if (rc == 6 || result == "BLOCKED" || result == "GO_REFUSED" || result == "REFUSED") { if (!reported_voiced_) say_("检查失败", "blocked"); reported_voiced_ = true; return; }
  if (!took && result == "CANCELLED_BEFORE_TAKEOVER") {  // V1.6: cancelled while waiting to take over: said once, from the runner's own classification (operator cancel = 已退出, anything else = 检查失败)
    if (!reported_voiced_) say_(json_find_string(doc, "voice") == "已退出" ? "已退出" : "检查失败", "cancel_wait");
    reported_voiced_ = true; return;
  }
  if (!took) return;  // control was never taken: the cancel announcement was already made
  (void)confirmed; (void)user_abort;
  if (!reported_voiced_) say_(voice_of(doc, result), result.c_str());
  reported_voiced_ = true;
}

// logs/mode_pass.jsonl: one line per finished run of a named mode; start.sh refuses a mode whose prerequisite mode has no pass record (the test ORDER).
// PASS = the mode ran to its planned end with a CONFIRMED WalkRun hand-back and the anomaly summary has no FAIL (mode "exit": an operator START x2 stop counts as the planned end).
void Service::record_mode_pass() {
  std::string doc;
  if (!have_mode_ || run_dir_.empty() || !read_file(run_dir_ + "/run_report.json", doc)) return;
  const std::string result = json_find_string(doc, "result"), reason = json_find_string(doc, "exit_reason");
  size_t an = doc.find("\"anomaly\":{\"level\":\"");
  const std::string level = an == std::string::npos ? "UNKNOWN" : doc.substr(an + 20, doc.find('"', an + 20) - an - 20);
  const bool recovery = json_find_raw(doc, "recovery_ok") == "true";
  const bool took = json_find_raw(doc, "took_control") == "true";
  const bool recordable = took || result == "REFUSED" || result == "BLOCKED" || result == "GO_REFUSED" || result == "CHECK_PASSED";
  if (!recordable) return;  // aborted before control: nothing to record
  const bool planned = result == "COMPLETED" || result == "CHECK_PASSED" || (mode_.name == "exit" && result == "EXITED_WALKRUN" && reason.rfind("REMOTE_STOP", 0) == 0);
  const bool pass = planned && (recovery || !took) && level != "FAIL" && !mode_overrides_;
  char b[900];
  snprintf(b, sizeof b, "{\"wall\":%.3f,\"mode\":\"%s\",\"result\":\"%s\",\"exit_reason\":\"%s\",\"recovery_ok\":%s,\"anomaly\":\"%s\",\"pass\":%s,\"overrides\":%s,\"run_dir\":\"%s\"}\n", wall(), mode_.name.c_str(), result.c_str(),
           json_escape(reason).c_str(), recovery ? "true" : "false", level.c_str(), pass ? "true" : "false", mode_overrides_ ? "true" : "false", json_escape(run_dir_).c_str());
  FILE* f = fopen((P_.logs + "/mode_pass.jsonl").c_str(), "a");  // HISTORY for the menu / the record only: it never gates a start (P5.2)
  if (f) { fputs(b, f); fclose(f); }
  logf("mode %s: %s (result %s, anomaly %s) -> logs/mode_pass.jsonl (history only)", mode_.name.c_str(), pass ? "PASS recorded" : "NOT passed", result.c_str(), level.c_str());
}

// GO is written only once every key has been released for 0.25 s: the factory remote receives the same L1+A, and the runner must not take control
// while a factory chord is still being evaluated. Not released within 3 s -> the prepared runner is cancelled.
void Service::release_gate() {
  if (!pending_go_) return;
  double now = mono();
  if (!runner_.alive()) { pending_go_ = false; return; }
  if (last_keys_.load() == 0) {
    if (keys_free_since_ == 0) keys_free_since_ = now;
    if (now - keys_free_since_ >= 0.25) {
      // V1.6: GO now means "keys released, start the announcement": the runner enters GO_WAIT (monitoring, robot NOT taken over), the service says 开始执行, and only after the announcement has FINISHED
      // (+ announce_wait_s, measured by the runner) does the runner re-verify and take over. See announce_gate().
      ::unlink(announced_file_.c_str());
      go_ph_ = GoPhase();
      go_ph_.on = true; go_ph_.t0 = now;
      char b[160];
      snprintf(b, sizeof b, "{\"wall\":%.3f,\"announce\":1,\"announce_max_s\":%.2f}\n", wall(), site_.announce_max_s);
      write_atomic(go_file_, b);
      pending_go_ = false;
      log_state("GO issued: announcing 开始执行; the runner takes control only after the announcement finished + announce_wait_s (ARM -> take-over -> motion window -> hand-back)");
      go_ph_.ticket = say_("开始执行", "go");
      svc_event("announce_begin", jstr("text", "开始执行") + "," + jnum("ticket", (double)go_ph_.ticket) + "," + jnum("announce_max_s", site_.announce_max_s) + "," + jnum("announce_wait_s", site_.announce_wait_s));
    }
  } else keys_free_since_ = 0;
  if (pending_go_ && now - pending_go_t0_ > 3.0) {
    pending_go_ = false;
    m_->user_abort = true;
    log_state("GO cancelled: keys were not released within 3 s");
    say_("已退出");
    term_child(runner_, "GO keys not released");
  }
}

// V1.6: the audio service has no playback-finished callback. Finished = TtsMaker returned + estimated speaking time (Tts::status().est_done), but NEVER later than announce_max_s after the
// announcement began (hard cap: the wait cannot hang). No audio / TTS disabled / RPC rejected -> finished at once. The runner (not the service) then waits announce_wait_s and re-verifies everything.
void Service::announce_gate() {
  if (!go_ph_.on) return;
  const double now = mono();
  if (!runner_.alive()) { go_ph_.on = false; svc_event("announce_abandoned", jstr("why", "runner ended")); return; }
  SayStatus s = tts_.status(go_ph_.ticket);
  const char* via = nullptr;
  if (go_ph_.ticket == 0 || !s.known) via = "no_tts";
  else if (s.dropped) via = "dropped";
  else if (s.rpc_done && now >= s.est_done) via = s.audio && s.rc == 0 ? "estimated" : (s.audio ? "rpc_rejected" : "no_audio");
  if (!via && now - go_ph_.t0 >= site_.announce_max_s) via = "cap";
  if (!via) return;
  char b[320];
  snprintf(b, sizeof b, "{\"wall\":%.3f,\"via\":\"%s\",\"announce_s\":%.3f,\"rpc_rc\":%d,\"est_play_s\":%.3f}\n", wall(), via, now - go_ph_.t0, s.rc, s.est_play_s);
  write_atomic(announced_file_, b);
  svc_event("announce_done", jstr("via", via) + "," + jnum("announce_s", now - go_ph_.t0) + "," + jnum("rpc_rc", s.rc) + "," + jnum("rpc_ms", s.rpc_done ? (s.t_rpc_end - s.t_rpc_start) * 1e3 : -1) + "," + jnum("est_play_s", s.est_play_s) + "," +
                           jnum("announce_wait_s", site_.announce_wait_s));
  log_state(std::string("announcement finished (") + via + ", " + std::to_string(now - go_ph_.t0) + " s): the runner waits announce_wait_s, re-verifies, then takes over");
  go_ph_.on = false;
}

void Service::tick_children() {
  double now = mono();
  if (runner_.alive()) {
    std::string st;
    if (read_file(status_file_, st)) {
      std::string ph = json_find_string(st, "phase");
      if (!ph.empty() && ph != runner_phase_) {
        runner_phase_ = ph;
        svc_event("runner_phase", jstr("phase", ph));
        do_acts(m_->runner_status(ph, now));
        if (ph == "REPORTED" && !reported_voiced_) report_summary(-1, m_->user_abort, m_->state);  // P5.1: announce the REAL result as soon as the report exists (the runner keeps recording the post-exit state for a few more seconds)
      }
    }
    if (!run_active_ && reap(runner_)) {  // V1.5.1: the pre-warmed runner ended before any trigger (signal / no lowstate / refusal / crash): nothing was ever sent
      const double life = now - standby_spawn_t_;
      logf("standby runner ended after %.1f s before any trigger (rc=%d); control was never touched", life, runner_.rc);
      svc_event("standby_exit", jnum("rc", runner_.rc) + "," + jnum("life_s", life));
      standby_proc_ = false; runner_phase_.clear();
      if (life < 10.0) {
        if (++standby_quick_fails_ >= 3) { standby_disabled_ = true; logf("WARNING: the standby runner failed %d times within 10 s: pre-warming is DISABLED (F1+F3 starts a cold runner; see logs/standby.out)", standby_quick_fails_); }
        next_standby_t_ = now + std::min(30.0, std::pow(2.0, standby_quick_fails_));
      } else { standby_quick_fails_ = 0; next_standby_t_ = now + 1.0; }
    } else if (run_active_ && reap(runner_)) {
      int rc = runner_.rc;
      bool ua = m_->user_abort;
      St was = m_->state;
      report_summary(rc, ua, was);
      record_mode_pass();
      svc_event("runner_exit", jnum("rc", rc) + "," + jstr("result", last_result_));
      run_active_ = false;
      Acts acts = m_->runner_exit(rc, now);
      do_acts(acts);
      ::unlink(go_file_.c_str());
    }
  }
  if (guardian_.alive() && reap(guardian_)) {
    logf("guardian ended (rc=%d); see logs/guardian/", guardian_.rc);
    on_guardian_ended();
    std::string gdoc;
    if (!guardian_report_.empty() && read_file(guardian_report_, gdoc) && json_find_string(gdoc, "action") == "take_over") {
      std::string res = json_find_string(gdoc, "result");
      logf("guardian take-over result=%s requested_mode=%s final_mode=%s final_fsm=%s recovery_ok=%s (%s)", res.c_str(), json_find_string(gdoc, "requested_mode").c_str(), json_find_string(gdoc, "final_mode").c_str(),
           json_find_raw(gdoc, "final_fsm").c_str(), json_find_raw(gdoc, "recovery_ok").c_str(), json_find_string(gdoc, "reason").c_str());
      last_result_ = "GUARDIAN_" + res; last_final_mode_ = json_find_string(gdoc, "final_mode"); last_final_fsm_ = atoi(json_find_raw(gdoc, "final_fsm").c_str()); last_recovery_ok_ = json_find_raw(gdoc, "recovery_ok") == "true";
      if (!run_dir_.empty()) {  // keep the guardian's take-over next to the crashed run
        write_atomic(run_dir_ + "/guardian_report.json", gdoc);
        std::string gdir = json_find_string(gdoc, "log_dir");
        if (!gdir.empty()) { ::unlink((run_dir_ + "/guardian_takeover").c_str()); if (symlink(gdir.c_str(), (run_dir_ + "/guardian_takeover").c_str()) != 0) {} }
        svc_event("guardian_takeover", jstr("result", res) + "," + jstr("run_dir", run_dir_) + "," + jstr("log_dir", gdir));
      }
      if (pending_guardian_report_) { pending_guardian_report_ = false; say_(voice_of(gdoc, res), "guardian"); }
      if (m_->state == St::SETTLE) m_->detail += " [guardian: " + res + ", final fsm " + std::to_string(last_final_fsm_) + "]";
    } else if (pending_guardian_report_) { pending_guardian_report_ = false; say_("检查失败"); }
  }
  if (pending_guardian_report_ && now - pending_guardian_t_ > 40.0) { pending_guardian_report_ = false; say_("检查失败"); }
}

void Service::handle_event(const Ev& ev) {
  double now = mono();
  bool gok = guardian_ok();
  switch (ev.k) {
    case Ev::KEYS:
      svc_event("keys", jnum("keys", ev.keys) + "," + jstr("text", keys_text(ev.keys)) + "," + jnum("byte2", ev.b2) + "," + jnum("byte3", ev.b3));
      if (keysf_) {
        fprintf(keysf_, "%s %.3f byte2=0x%02x byte3=0x%02x keys=%s\n", timestamp_compact().c_str(), wall(), ev.b2, ev.b3, keys_text(ev.keys).c_str());
        fflush(keysf_);
      }
      break;
    case Ev::LOAD: log_state("chord F1+F3 detected"); m_->guardian_why = guardian_why_; do_acts(m_->chord_load(now, gok)); break;
    case Ev::GO: log_state("chord L1+A detected"); do_acts(m_->chord_go(now, gok)); break;
    case Ev::START2: log_state("START double-click detected"); do_acts(m_->start_double(now)); break;
  }
}

int Service::run() {
  ::mkdir(P_.logs.c_str(), 0775);
  for (const char* d : {"/runs", "/guardian"}) ::mkdir((P_.logs + d).c_str(), 0775);
  state_file_ = P_.logs + "/state.json"; status_file_ = P_.logs + "/runner_status.json"; go_file_ = P_.logs + "/GO"; trigger_file_ = P_.logs + "/TRIGGER"; announced_file_ = P_.logs + "/ANNOUNCED";
  int lockfd = ::open((P_.logs + "/service.lock").c_str(), O_RDWR | O_CREAT, 0664);
  if (lockfd < 0 || ::flock(lockfd, LOCK_EX | LOCK_NB) != 0) { fprintf(stderr, "another service instance holds logs/service.lock\n"); return 2; }
  { char b[32]; snprintf(b, sizeof b, "%d", (int)getpid()); write_atomic(P_.logs + "/service.pid", b); }
  log_set_file(P_.logs + "/service.log");
  log_set_tag("");
  keysf_ = fopen((P_.logs + "/keys.log").c_str(), "a");
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  for (int s : {SIGTERM, SIGINT, SIGHUP}) sigaction(s, &sa, nullptr);
  signal(SIGPIPE, SIG_IGN);

  if (!mode_err_.empty()) { fprintf(stderr, "service: %s\n", mode_err_.c_str()); logf("service: %s", mode_err_.c_str()); return 2; }
  if (iface_.empty()) { fprintf(stderr, "service: --iface IFACE required\n"); return 2; }
  if (!sim_ && iface_ == "lo") { fprintf(stderr, "service: the loopback interface is only for the test robot (--sim); give the robot network interface\n"); return 2; }
  logf("service starting: target=%s iface=%s domain=%d entry_fsm=%s armed=%d run_s=%.2f ref=%s authority=%.2f", sim_ ? "sim" : "real", iface_.c_str(), domain_,
       a_.kv.count("--entry-fsm") ? a_.get("--entry-fsm").c_str() : "site.conf", (int)armed_, run_s_, ref_mode_.c_str(), authority_);
  if (!site_.err.empty()) logf("CONFIG ERROR in config/site.conf: %s  (every runner will REFUSE until it is fixed)", site_.err.c_str());
  for (auto& w : site_.warnings) logf("WARNING: %s", w.c_str());
  if (run_s_ > LEGACY_MAX_RUN_S + 1e-9 || authority_ > LEGACY_MAX_AUTHORITY + 1e-9)
    logf("WARNING: run_s=%.2f / authority=%.2f EXCEED the legacy real-robot caps (%.1f s / %.1f); allowed only if site.conf max_run_s=%.2f / max_authority=%.2f permit it (compile-time limits %.0f s / %.1f)", run_s_, authority_, LEGACY_MAX_RUN_S, LEGACY_MAX_AUTHORITY, site_.max_run_s, site_.max_authority, ABS_MAX_RUN_S, ABS_MAX_AUTHORITY);
  if (!armed_) logf("CHECK MODE (not armed): F1+F3 runs the read-only load + checks and reports READY/BLOCKED (control is never taken); L1+A does nothing.");

  chord_load_.reset(new ChordDetector(COMBO_LOAD));
  chord_go_.reset(new ChordDetector(COMBO_GO));
  LinkOpts lo;
  lo.rpc = false;          // this process never calls the sport service: it cannot switch control
  lo.observe_cmd = false;  // and never creates a command writer
  Link link(iface_, domain_, lo);
  link.add_state_listener([this](const LinkState& st) { on_lowstate(st); });
  tts_.start(site_.tts && !a_.has("--no-tts"), site_.tts_volume, site_.announce_s_per_char);
  say_("服务已启动", "start");
  if (have_mode_) {
    say_(mode_.voice, "mode");
    logf("test mode: %s (%s) ref=%s run_s=%.2f validated=%d%s%s", mode_.name.c_str(), mode_.desc.c_str(), ref_mode_.c_str(), run_s_, (int)mode_.validated, mode_.check_only ? " CHECK ONLY (never armed)" : "", mode_overrides_ ? " [explicit options override the mode: not the validated envelope]" : "");
  }
  svc_event("service_start", jstr("mode", have_mode_ ? mode_.name : "custom") + "," + jstr("iface", iface_) + "," + jbool("armed", armed_) + "," + jnum("run_s", run_s_) + "," + jstr("ref_mode", ref_mode_));

  double t_state = 0, t_g = 0;
  ensure_guardian();
  while (!g_stop) {
    {
      std::unique_lock<std::mutex> lk(qm_);
      if (q_.empty()) qcv_.wait_for(lk, std::chrono::milliseconds(20));
      while (!q_.empty()) {
        Ev e = q_.front();
        q_.pop_front();
        lk.unlock();
        handle_event(e);
        lk.lock();
      }
    }
    double now = mono();
    tick_children();
    release_gate();
    announce_gate();
    // the guardian ends by itself once the runner has handed control back ("normal end"): that is NOT a failure while the runner is in/after EXIT
    // P5.3 race fix: the guardian ends BY ITSELF as soon as the runner released the lock ("normal end"). That is only a failure while the runner is still IN CONTROL (ARM / RUN); in EXIT, REPORTED
    // (post-exit recording) or later it is expected - the old rule (phase == EXIT only) killed a healthy runner in its post-exit window with a false CRITICAL.
    const bool in_control = runner_phase_ == "ARM" || runner_phase_ == "RUN" || runner_phase_ == "GO_WAIT";  // GO_WAIT (V1.6): not in control yet, but a dead guardian must cancel the take-over
    do_acts(m_->tick(now, runner_.alive(), guardian_ok_running() || !in_control, m_->state == St::SETTLE ? settle_status() : Settle(), runner_phase_ == "REPORTED"));
    if (now - t_g > 0.5) { t_g = now; ensure_guardian(); }
    if (m_->state == St::IDLE && !run_active_ && !runner_.alive() && !pending_go_) spawn_standby();  // V1.5.1: model + DDS pre-warm at start-up and after every finished run
    if (now - t_state > 0.5) { t_state = now; write_state(); }
  }
  logf("service stopping");
  if (runner_.alive()) {
    term_child(runner_, "service stop");
    double t_end = mono() + 25;
    while (runner_.alive() && mono() < t_end) { reap(runner_); sleep_s(0.05); }
    if (runner_.alive()) logf("CRITICAL: runner did not exit within 25 s after SIGTERM (guardian still active)");
  }
  St s = m_->state;
  if (guardian_.alive() && (s == St::IDLE || s == St::SETTLE || s == St::READY || s == St::LOADING)) term_child(guardian_, "service stop");
  say_("已退出");
  tts_.stop(5.0);
  write_state();
  fast_exit(0);
}

}  // namespace

int run_service(int argc, char** argv, const Paths& P) {
  Args a;
  std::string err;
  if (!a.parse(argc, argv, flags_service(), err)) { fprintf(stderr, "service: %s\n", err.c_str()); return 2; }
  Site site = Site::load(P.config + "/site.conf");
  Service s(a, P, site);
  return s.run();
}

}  // namespace pm
