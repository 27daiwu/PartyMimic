#include <cmath>
// Service: ONE process that listens to the remote controller, speaks, and manages the guardian and the runner.
//   F1+F3 (chord) : load the model + real-time admission (read-only)  -> READY or BLOCKED
//   L1+A  (chord) : (only from READY) hand the robot from the factory WalkRun to PartyMimic, run the SSB window, hand back to WalkRun
//   START x2      : at ANY stage: abort. Before ARM: nothing was ever sent. After ARM: the runner's own START handler stops + hands back (WalkRun if the
//                   attitude is normal, PASSIVE if severe/unknown) and the independent guardian is the backup.
// After a clean run the service returns to IDLE and keeps listening. After any non-clean outcome it latches HOLD until `./ack.sh`.
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
#include "pm/remote.hpp"
#include "pm/tts.hpp"

extern char** environ;

namespace pm {
namespace {

enum class St { IDLE, LOADING, READY, RUNNING, HOLD };
const char* st_name(St s) {
  switch (s) { case St::IDLE: return "IDLE"; case St::LOADING: return "LOADING"; case St::READY: return "READY"; case St::RUNNING: return "RUNNING"; case St::HOLD: return "HOLD"; }
  return "?";
}
std::string rc_text(int rc) {
  switch (rc) {
    case 0: return "clean end"; case 1: return "runner crashed"; case 2: return "refused"; case 3: return "no lowstate";
    case 4: return "ARM failed (control could not be taken cleanly)"; case 5: return "EXIT UNCONFIRMED (robot may still be in user control!)";
    case 6: return "BLOCKED by real-time admission / GO refused"; case 10: return "fault / operator stop, WalkRun recovery confirmed";
    case 11: return "WALKRUN RECOVERY FAILED (rpc ok but fsm not in the WalkRun family / dropped to fsm 1: robot probably down)";
    case 12: return "EMERGENCY PASSIVE (e-stop file / remote L2+B)"; case 13: return "robot down / state untrusted: PASSIVE";
  }
  return "exit code " + std::to_string(rc);
}

struct Act { enum K { NOTE, SAY, SPAWN, GO, TERMINATE, ESCALATE } k; std::string s; };
using Acts = std::vector<Act>;

struct MachineCfg { double ready_timeout_s = 120, load_timeout_s = 90, run_max_s = 90, abort_escalate_s = 2.5; };

// Pure state machine (no I/O): inputs are events, outputs are actions. (Port of fasttrack/supervisor.py Machine.)
class Machine {
 public:
  Machine(MachineCfg c, bool armed) : cfg(c), armed(armed) {}
  MachineCfg cfg;
  bool armed;
  St state = St::IDLE;
  double since = 0, deadline = -1, abort_deadline = -1;
  std::string hold_reason, detail, guardian_why;  // guardian_why: set by the service (why the guardian is not alive: exact cause, not a generic text)
  bool user_abort = false, escalated = false, g_warned = false;
  int last_rc = -1;
  void go(St s, double now) { state = s; since = now; }

  Acts chord_load(double now, bool guardian_ok) {
    if (state != St::IDLE) return {{Act::NOTE, std::string("F1+F3 ignored in state ") + st_name(state)}};
    if (!guardian_ok) return {{Act::NOTE, "guardian not alive: F1+F3 ignored - " + (guardian_why.empty() ? std::string("it is restarted automatically; try again in a few seconds") : guardian_why)}, {Act::SAY, "检查失败"}};
    user_abort = false;
    go(St::LOADING, now);
    deadline = now + cfg.load_timeout_s;
    return {{Act::SAY, "正在加载"}, {Act::SPAWN, ""}};
  }
  Acts chord_go(double now, bool guardian_ok) {
    if (state != St::READY) return {{Act::NOTE, std::string("L1+A ignored in state ") + st_name(state) + " (needs READY: press F1+F3 first)"}};
    if (!armed) return {{Act::NOTE, "service is not armed (start.sh without confirmation): L1+A ignored"}, {Act::SAY, "检查失败"}};
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
      deadline = now + cfg.ready_timeout_s;
      return {{Act::NOTE, "READY: model loaded, real-time admission passed. L1+A to go, START x2 to cancel"}, {Act::SAY, "准备就绪"}};
    }
    return {};
  }
  Acts tick(double now, bool runner_alive, bool guardian_ok) {
    Acts a;
    if ((state == St::LOADING || state == St::READY) && deadline >= 0 && now > deadline && runner_alive) {
      a.push_back({Act::NOTE, std::string(st_name(state)) + " timed out"});
      a.push_back({Act::SAY, "已退出"});
      a.push_back({Act::TERMINATE, "timeout"});
      deadline = -1; user_abort = true;
    }
    if (state == St::RUNNING) {
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
      return {{Act::NOTE, "runner ended before taking control (rc=" + std::to_string(rc) + ": " + txt + "); service back to IDLE"}};
    }
    if (was == St::RUNNING) {
      if (escalated) {
        hold_reason = "escalated (runner rc=" + std::to_string(rc) + ": " + txt + ")";
        escalated = false;
        go(St::HOLD, now);
        return {{Act::NOTE, "HOLD: " + hold_reason + ". The e-stop latch was set. Check the robot, then `./ack.sh` (clears the latch)."}};
      }
      if (rc == 6) { go(St::IDLE, now); return {{Act::NOTE, "GO refused (rc=6: " + txt + "); nothing was sent; back to IDLE"}}; }
      if (rc == 0 || (rc == 10 && user_abort)) {
        go(St::IDLE, now);
        return {{Act::NOTE, "RUN FINISHED rc=" + std::to_string(rc) + ": " + txt + (rc == 10 ? " (operator START x2, hand-back confirmed)" : "") + "; back to IDLE, listening"}};
      }
      hold_reason = "runner rc=" + std::to_string(rc) + ": " + txt + (detail.empty() ? "" : " [" + detail + "]");
      go(St::HOLD, now);
      return {{Act::NOTE, "HOLD: " + hold_reason + ". Check the robot, read logs/, then `./ack.sh` to re-arm."}};
    }
    return {{Act::NOTE, "runner exit rc=" + std::to_string(rc) + " in " + st_name(was)}};
  }
  Acts ack(double now) {
    if (state == St::HOLD) { go(St::IDLE, now); hold_reason.clear(); return {{Act::NOTE, "ACK: back to IDLE"}}; }
    return {{Act::NOTE, "ack ignored (not in HOLD)"}};
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
    armed_ = a.has("--confirm-protection") || sim_;
    run_s_ = a.num("--run-s", site.run_s);
    authority_ = a.num("--authority-max", site.authority);
    ref_mode_ = a.get("--ref-mode", site.ref_mode);
    ramp_s_ = a.num("--ramp-s", site.ramp_s);
    end_hold_s_ = a.num("--end-hold-s", site.end_hold_s);
    allow_partial_ = a.has("--allow-partial-authority") || site.allow_partial_authority;
    accept_unvalidated_ = a.has("--accept-unvalidated");
    ready_timeout_ = a.num("--ready-timeout-s", 120.0);
    m_ = std::unique_ptr<Machine>(new Machine(MachineCfg{ready_timeout_, 90, run_s_ + 60.0, 2.5}, armed_));
    self_ = P.root + "/build/partymimic";
    char b[4096];
    ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1);
    if (n > 0) self_ = std::string(b, (size_t)n);
  }

  int run();

 private:
  void log_state(const std::string& msg) { logf("[%s] %s", st_name(m_->state), msg.c_str()); }
  void write_state();
  void on_lowstate(const LinkState& st);
  bool guardian_ok();
  bool guardian_ok_running();
  void ensure_guardian();
  void on_guardian_ended();
  void spawn_runner();
  void do_acts(const Acts& acts);
  void report_summary(int rc, bool user_abort, St was);
  void tick_children();
  void handle_event(const Ev& ev);

  Args a_;
  Paths P_;
  Site site_;
  bool sim_ = false, armed_ = false;
  std::string iface_, ref_mode_, self_;
  double ramp_s_ = 0.3, end_hold_s_ = 0.5;
  bool allow_partial_ = false, accept_unvalidated_ = false;
  int domain_ = 0;
  double run_s_, authority_, ready_timeout_;
  std::unique_ptr<Machine> m_;
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
  std::atomic<double> last_lowstate_{0};
  std::atomic<uint16_t> last_keys_{0};
  uint16_t prev_keys_ = 0;
  std::unique_ptr<ChordDetector> chord_load_, chord_go_;
  StartDoubleClick dbl_;
  FILE* keysf_ = nullptr;
  std::string state_file_, status_file_, go_file_, ack_file_;
  bool pending_go_ = false;
  double pending_go_t0_ = 0, keys_free_since_ = 0;
  void release_gate();
};

void Service::write_state() {
  char b[2048];
  double age = last_lowstate_.load() > 0 ? mono() - last_lowstate_.load() : -1;
  snprintf(b, sizeof b,
           "{\"state\":\"%s\",\"since\":%.3f,\"time\":%.3f,\"hold_reason\":\"%s\",\"armed\":%s,\"target\":\"%s\",\"interface\":\"%s\",\"guardian_alive\":%s,\"lowstate_age_s\":%.3f,\"last_rc\":%d,\"pressed\":\"%s\",\"pid\":%d,"
           "\"last_result\":\"%s\",\"last_final_mode\":\"%s\",\"last_final_fsm\":%d,\"last_recovery_ok\":%s,\"guardian_fatal\":%s,\"guardian_why\":\"%s\"}",
           st_name(m_->state), m_->since, wall(), json_escape(m_->hold_reason).c_str(), armed_ ? "true" : "false", sim_ ? "sim" : "real", iface_.c_str(), guardian_ok() ? "true" : "false", age, m_->last_rc,
           keys_text(last_keys_.load()).c_str(), (int)getpid(), json_escape(last_result_).c_str(), json_escape(last_final_mode_).c_str(), last_final_fsm_, last_recovery_ok_ ? "true" : "false", guardian_fatal_ ? "true" : "false", json_escape(guardian_why_).c_str());
  write_atomic(state_file_, b);
}

void Service::on_lowstate(const LinkState& st) {  // DDS reader thread: cheap, no I/O
  last_lowstate_ = mono();
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
    if (!guardian_fatal_said_) { guardian_fatal_said_ = true; tts_.say("检查失败"); }
  } else {
    guardian_backoff_s_ = std::min(30.0, std::pow(2.0, guardian_quick_fails_));
    logf("WARNING: %s; retry %d/5 in %.0f s", guardian_why_.c_str(), guardian_quick_fails_, guardian_backoff_s_);
  }
}

void Service::spawn_runner() {
  std::string ts = timestamp_compact();
  run_dir_ = P_.logs + "/runs/" + ts;
  ::mkdir(run_dir_.c_str(), 0775);
  ::unlink(go_file_.c_str());
  ::unlink(status_file_.c_str());
  runner_phase_.clear();
  runner_.name = "runner";
  RunnerCmd rc;
  rc.self = self_; rc.iface = iface_; rc.domain = domain_; rc.run_s = run_s_; rc.ref_mode = ref_mode_; rc.authority = authority_; rc.ramp_s = ramp_s_; rc.end_hold_s = end_hold_s_;
  rc.run_dir = run_dir_; rc.go_file = go_file_; rc.status_file = status_file_; rc.ready_timeout_s = ready_timeout_; rc.sim = sim_; rc.allow_partial = allow_partial_; rc.accept_unvalidated = accept_unvalidated_;
  if (a_.kv.count("--entry-fsm")) rc.entry_fsm = a_.get("--entry-fsm");
  if (a_.kv.count("--unconfirmed-hold-s")) rc.unconfirmed_hold_s = a_.get("--unconfirmed-hold-s");
  std::vector<std::string> args = runner_cmdline(rc);
  if (!spawn_child(runner_, args, run_dir_ + "/runner.out", P_.root)) { m_->state = St::IDLE; return; }
  std::string latest = P_.logs + "/latest";
  ::unlink(latest.c_str());
  if (symlink(run_dir_.c_str(), latest.c_str()) != 0) {}
  logf("runner started pid %d -> logs/runs/%s/", (int)runner_.pid, ts.c_str());
}

void Service::do_acts(const Acts& acts) {
  for (auto& a : acts) {
    switch (a.k) {
      case Act::NOTE: log_state(a.s); break;
      case Act::SAY: tts_.say(a.s); break;
      case Act::SPAWN: spawn_runner(); break;
      case Act::GO:  // the GO file is written only after every key is released (the factory remote also sees L1+A): see release_gate()
        pending_go_ = true; pending_go_t0_ = mono(); keys_free_since_ = 0;
        log_state("GO requested (L1+A): waiting for all keys to be released before the runner takes control");
        break;
      case Act::TERMINATE: term_child(runner_, a.s); break;
      case Act::ESCALATE:
        logf("CRITICAL ESCALATE: %s -> e-stop latch + SIGTERM to the runner (guardian backup stays active)", a.s.c_str());
        write_atomic(DEFAULT_ESTOP, "service escalation " + std::to_string(wall()) + "\n");
        tts_.say("检查失败");
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

// one line per finished run + the voice announcement of the outcome (details: logs/runs/<time>/run_report.json)
void Service::report_summary(int rc, bool user_abort, St was) {
  std::string doc;
  m_->detail.clear();
  if (!read_file(run_dir_ + "/run_report.json", doc)) {
    logf("no run_report.json (runner rc=%d)", rc);
    if (rc != 0) {
      if (was == St::RUNNING && guardian_.alive() && (runner_phase_ == "ARM" || runner_phase_ == "RUN" || runner_phase_ == "EXIT")) {  // the runner died in control: the guardian runs the SAME exit policy; its result decides what is said
        pending_guardian_report_ = true; pending_guardian_t_ = mono();
        logf("runner ended without a report: waiting for the guardian take-over result (%s)", guardian_report_.c_str());
      } else tts_.say("检查失败");
    }
    return;
  }
  std::string result = json_find_string(doc, "result"), reason = json_find_string(doc, "exit_reason"), fault = json_find_string(doc, "fault"), mode = json_find_string(doc, "requested_mode");
  bool confirmed = json_find_raw(doc, "exit_confirmed") == "true", took = json_find_raw(doc, "took_control") == "true", rec_ok = json_find_raw(doc, "recovery_ok") == "true";
  last_result_ = result; last_final_mode_ = json_find_string(doc, "final_mode"); last_final_fsm_ = atoi(json_find_raw(doc, "final_fsm").c_str()); last_recovery_ok_ = rec_ok;
  logf("run result=%s reason=%s fault=%s requested_mode=%s final_mode=%s final_fsm=%d recovery_ok=%d entry_fsm=%s", result.c_str(), reason.c_str(), fault.c_str(), mode.c_str(), last_final_mode_.c_str(), last_final_fsm_,
       (int)rec_ok, json_find_raw(doc, "entry_fsm").c_str());
  m_->detail = "result " + result + ", final_fsm " + std::to_string(last_final_fsm_) + ", final_mode " + last_final_mode_ + ", recovery_ok " + (rec_ok ? "true" : "false");
  if (took && !rec_ok && result != "EMERGENCY_PASSIVE" && result != "COMPLETED")
    logf("!!! RECOVERY NOT CONFIRMED (%s): the OEM is in fsm %d (%s), not the requested %s. Check the robot, read logs/runs/*/run_report.json, then ./ack.sh", result.c_str(), last_final_fsm_, last_final_mode_.c_str(), mode.c_str());
  size_t pos = 0;
  while ((pos = doc.find("\"pass\":false", pos)) != std::string::npos) {  // BLOCKED items
    size_t s = doc.rfind("{\"item\":\"", pos);
    if (s != std::string::npos) logf("  BLOCKED item: %s", doc.substr(s, doc.find('}', pos) - s).c_str());
    pos += 12;
  }
  // voice: short phrases only, the reasons are in the log
  if (rc == 6) { tts_.say("检查失败"); return; }
  if (!took) return;  // control was never taken: the cancel announcement was already made
  (void)confirmed; (void)user_abort;
  tts_.say(voice_of(doc, result));
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
      char b[64];
      snprintf(b, sizeof b, "%.3f", wall());
      write_atomic(go_file_, b);
      pending_go_ = false;
      log_state("GO issued: the runner now takes control (ARM -> take-over -> motion window -> hand-back)");
      tts_.say("开始执行");
    }
  } else keys_free_since_ = 0;
  if (pending_go_ && now - pending_go_t0_ > 3.0) {
    pending_go_ = false;
    m_->user_abort = true;
    log_state("GO cancelled: keys were not released within 3 s");
    tts_.say("已退出");
    term_child(runner_, "GO keys not released");
  }
}

void Service::tick_children() {
  double now = mono();
  if (runner_.alive()) {
    std::string st;
    if (read_file(status_file_, st)) {
      std::string ph = json_find_string(st, "phase");
      if (!ph.empty() && ph != runner_phase_) { runner_phase_ = ph; do_acts(m_->runner_status(ph, now)); }
    }
    if (reap(runner_)) {
      int rc = runner_.rc;
      bool ua = m_->user_abort;
      St was = m_->state;
      report_summary(rc, ua, was);
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
      if (pending_guardian_report_) { pending_guardian_report_ = false; tts_.say(voice_of(gdoc, res)); }
      if (m_->state == St::HOLD) m_->hold_reason += " [guardian: " + res + ", final fsm " + std::to_string(last_final_fsm_) + "]";
    } else if (pending_guardian_report_) { pending_guardian_report_ = false; tts_.say("检查失败"); }
  }
  if (pending_guardian_report_ && now - pending_guardian_t_ > 40.0) { pending_guardian_report_ = false; tts_.say("检查失败"); }
}

void Service::handle_event(const Ev& ev) {
  double now = mono();
  bool gok = guardian_ok();
  switch (ev.k) {
    case Ev::KEYS:
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
  state_file_ = P_.logs + "/state.json"; status_file_ = P_.logs + "/runner_status.json"; go_file_ = P_.logs + "/GO"; ack_file_ = P_.logs + "/ACK";
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

  if (iface_.empty()) { fprintf(stderr, "service: --iface IFACE required\n"); return 2; }
  if (!sim_ && iface_ == "lo") { fprintf(stderr, "service: the loopback interface is only for the test robot (--sim); give the robot network interface\n"); return 2; }
  logf("service starting: target=%s iface=%s domain=%d entry_fsm=%s armed=%d run_s=%.2f ref=%s authority=%.2f", sim_ ? "sim" : "real", iface_.c_str(), domain_,
       a_.kv.count("--entry-fsm") ? a_.get("--entry-fsm").c_str() : "site.conf", (int)armed_, run_s_, ref_mode_.c_str(), authority_);
  if (!site_.err.empty()) logf("CONFIG ERROR in config/site.conf: %s  (every runner will REFUSE until it is fixed)", site_.err.c_str());
  for (auto& w : site_.warnings) logf("WARNING: %s", w.c_str());
  if (run_s_ > LEGACY_MAX_RUN_S + 1e-9 || authority_ > LEGACY_MAX_AUTHORITY + 1e-9)
    logf("WARNING: run_s=%.2f / authority=%.2f EXCEED the legacy real-robot caps (%.1f s / %.1f); allowed only if site.conf max_run_s=%.2f / max_authority=%.2f permit it (compile-time limits %.0f s / %.1f)", run_s_, authority_, LEGACY_MAX_RUN_S, LEGACY_MAX_AUTHORITY, site_.max_run_s, site_.max_authority, ABS_MAX_RUN_S, ABS_MAX_AUTHORITY);
  if (!armed_) logf("NOT ARMED: F1+F3 runs the load + checks and reports READY/BLOCKED; L1+A does nothing.");

  chord_load_.reset(new ChordDetector(COMBO_LOAD));
  chord_go_.reset(new ChordDetector(COMBO_GO));
  LinkOpts lo;
  lo.rpc = false;          // this process never calls the sport service: it cannot switch control
  lo.observe_cmd = false;  // and never creates a command writer
  Link link(iface_, domain_, lo);
  link.add_state_listener([this](const LinkState& st) { on_lowstate(st); });
  tts_.start(site_.tts && !a_.has("--no-tts"), site_.tts_volume);
  tts_.say("服务已启动");

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
    // the guardian ends by itself once the runner has handed control back ("normal end"): that is NOT a failure while the runner is in/after EXIT
    do_acts(m_->tick(now, runner_.alive(), guardian_ok_running() || runner_phase_ == "EXIT"));
    if (file_exists(ack_file_)) { ::unlink(ack_file_.c_str()); do_acts(m_->ack(now)); }
    if (now - t_g > 0.5) { t_g = now; ensure_guardian(); }
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
  if (guardian_.alive() && (s == St::IDLE || s == St::HOLD || s == St::READY || s == St::LOADING)) term_child(guardian_, "service stop");
  tts_.say("已退出");
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
