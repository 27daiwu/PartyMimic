// Out-of-process guardian + operator release.
// Why: the in-process watchdog dies with the process (kill -9, SIGSTOP, segfault, OOM, hang inside inference). The guardian is a SEPARATE process that
//  * stays read-only until the control lock is held by a runner (pid in the lock file),
//  * takes over only if (a) the runner pid is dead while the OEM is still in user control, (b) the runner is alive but rt/user_lowcmd has been
//    silent for --silence-s (hung / stopped), (c) the e-stop file appeared and the runner did not start its own emergency exit, (d) START was double-clicked /
//    L2+B was held and the runner still sends stiff frames and is not exiting.
//  * NEVER starts a second exit while the runner is exiting: the lock file carries exit_phase / exit_owner / exit_t; an unexpired runner-owned exit is left alone.
//  * take-over = SIGKILL the runner (exactly one writer), then the SAME ExitPolicy as the runner (pm/exit_policy.hpp): decision from the current attitude,
//    SwitchToInternalCtrl at once on the executor's RPC thread, HOLD frames (q snapshot + policy gains through the guard) meanwhile (DAMP frames on the PASSIVE
//    path), FSM confirmation with a stable window, WALKRUN -> PASSIVE fallback, loud UNCONFIRMED report.
// It never writes while a healthy runner owns the lock.
#include <algorithm>
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <csignal>
#include <cstring>
#include <mutex>
#include <thread>

#include "pm/app.hpp"
#include "pm/exit_policy.hpp"
#include "pm/handback.hpp"
#include "pm/link.hpp"
#include "pm/logtool.hpp"
#include "pm/profile.hpp"
#include "pm/remote.hpp"
#include "pm/session.hpp"

namespace pm {
namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

struct TakeOverReport {
  std::string reason, note, log_dir;
  int killed_pid = 0;
  double kill_ms = 0;
  bool have = false;
  HandbackReport hb;
};

// shared between the main loop, the heartbeat thread and the take-over
struct Ctx {
  Link* link = nullptr;
  const Params* params = nullptr;
  Site site;
  std::unique_ptr<SafetyGuard> guard;
  std::mutex mx;            // guards `ga`, `guard`
  GuardianAlive ga;
  std::string alive_file, logs;   // logs = <root>/logs ("" = no take-over log)
  void set_phase(const std::string& ph, const std::string& owner) {
    std::lock_guard<std::mutex> lk(mx);
    ga.exit_phase = ph; ga.exit_owner = owner; ga.exit_t = wall();
  }
};

// Kill the runner (single writer), then the common ExitPolicy. fault_kind: what the trigger means for the decision table. forced: "" = AUTO | WALKRUN | PASSIVE.
TakeOverReport take_over(Ctx& cx, const std::string& reason, Fault fault_kind, int pid, int entry_fsm, int user_fsm, double t_detect, bool precheck, const std::string& forced = "") {
  TakeOverReport rep;
  rep.reason = reason;
  Link& link = *cx.link;
  link.enable_write();
  // decision first (microseconds): attitude of the newest sample
  ExitContext x;
  x.fault = fault_kind;
  x.normal_end = false;
  auto st0 = link.latest();
  x.att = attitude_of(st0, mono(), cx.site.exit.attitude_max_age_s);
  ExitRequest req;
  req.decision = decide_exit(x, cx.site.exit);
  if (!forced.empty() && forced != "AUTO") { req.decision.mode = forced; req.decision.why += " [operator forced " + forced + "]"; }
  req.owner = "guardian";
  req.trigger_fault = fault_name(fault_kind);
  req.detail = reason;
  req.user_fsm = user_fsm >= 0 ? user_fsm : USER_FSM_EXPECTED;
  req.entry_fsm = entry_fsm;
  req.precheck = precheck;  // only when the runner is already gone (it may have handed back itself): one GetFsmId, no blind RPC
  req.t_detect = t_detect;
  req.t_event = st0 ? st0->arrival : 0;
  req.att = x.att;
  if (pid > 0 && pid != (int)getpid()) {
    double tk = mono();
    if (kill(pid, SIGKILL) == 0) {
      rep.killed_pid = pid;
      logf("[guardian] SIGKILL runner pid %d", pid);
      double t_gone = mono() + 0.25;  // single-writer guarantee: wait until the kernel reports it dead
      while (mono() < t_gone) {
        std::string stt;
        if (!read_file("/proc/" + std::to_string(pid) + "/stat", stt)) break;
        size_t k = stt.rfind(')');
        if (k != std::string::npos && k + 2 < stt.size() && (stt[k + 2] == 'Z' || stt[k + 2] == 'X')) break;
        sleep_s(0.0005);
      }
    }
    rep.kill_ms = (mono() - tk) * 1e3;
  }
  {
    std::lock_guard<std::mutex> lk(cx.mx);
    if (st0) cx.guard->reset_target(st0->q);
  }
  const double scale = cx.site.exit.hold_gain_scale;
  // P5.1: the take-over is recorded like a run: logs/guardian/takeover_<time>/ (events incl. every RPC, exit steps, every HOLD/DAMP frame, lowstate stream from now to +2 s after the exit)
  auto glog = std::make_shared<RunLogger>();
  if (!cx.logs.empty()) {
    LogOptions lo2;
    rep.log_dir = cx.logs + "/guardian/takeover_" + timestamp_compact();
    lo2.dir = rep.log_dir; lo2.joint_names = cx.params->names; lo2.obs_bin = false; lo2.preroll_s = 0; lo2.install_crash_handlers = false; lo2.tag = "guardian";
    if (glog->open(lo2)) {
      link.add_state_listener([glog](const LinkState& s) { glog->state(s); });  // (the shared_ptr keeps the logger alive as long as the Link may call back)
      link.set_rpc_observer([glog](const RpcEvent& e) {
        char b[400];
        snprintf(b, sizeof b, "\"api\":\"%s\",\"arg\":\"%s\",\"rc\":%d,\"fsm\":%d,\"t_req\":%.6f,\"t_resp\":%.6f,\"latency_ms\":%.3f", e.api.c_str(), e.arg.c_str(), e.rc, e.fsm, e.t_req, e.t_resp, (e.t_resp - e.t_req) * 1e3);
        glog->event_at(e.t_resp, "rpc", b);
      });
      glog->begin_stream();
      glog->event("take_over", jstr("reason", reason) + "," + jstr("fault_kind", fault_name(fault_kind)) + "," + jnum("runner_pid", pid) + "," + jnum("killed_pid", rep.killed_pid) + "," + jnum("kill_ms", rep.kill_ms) + "," +
                                   jstr("mode", req.decision.mode) + "," + jstr("class", req.decision.cls) + "," + jstr("why", req.decision.why) + "," + jnum("entry_fsm", entry_fsm) + "," + jnum("user_fsm", user_fsm) + "," + jnum("t_detect", t_detect));
    }
  }
  auto frames = [&cx, scale, glog](FrameKind k, const Vec& q_snap) {
    Link& l = *cx.link;
    auto st = l.latest();
    Vec qm = (st && st->q.allFinite()) ? st->q : q_snap;
    auto log_frame = [&](uint8_t kind, const Vec& q, const Vec& kp, const Vec& kd) {
      if (!glog->is_open()) return;
      FrameRec f;
      f.t_mono = mono(); f.kind = kind; f.send_us = (float)l.last_send_us();
      if (st) { f.state_seq = st->seq; for (int i = 0; i < N; i++) f.q_meas[i] = (float)st->q[i]; }
      for (int i = 0; i < N; i++) { f.q_cmd[i] = (float)q[i]; f.kp[i] = (float)kp[i]; f.kd[i] = (float)kd[i]; }
      glog->frame(f);
    };
    if (k == FrameKind::HOLD && st) {
      try {
        std::lock_guard<std::mutex> lk(cx.mx);
        CmdOut c = cx.guard->filter_command(q_snap, cx.params->kp * scale, cx.params->kd * scale, st->q, st->dq, 0.01, 0, nullptr, false);
        l.send(c.q, c.dq, c.tau, c.kp, c.kd);
        log_frame(FK_HOLD, c.q, c.kp, c.kd);
        return;
      } catch (const GuardReject&) {}
    }
    l.send(qm, Vec::Zero(), Vec::Zero(), Vec::Zero(), Vec::Constant(DAMP_KD));
    log_frame(FK_DAMP, qm, Vec::Zero(), Vec::Constant(DAMP_KD));
  };
  cx.set_phase("EXITING", "guardian");
  const std::string estop = cx.site.exit.estop_file;
  rep.hb = handback(link, req, cx.site.exit, frames, [&cx](const std::string& ph) { cx.set_phase(ph, "guardian"); },
                    [&cx, &link]() -> std::string {
                      ExitContext y;
                      y.att = attitude_of(link.latest(), mono(), cx.site.exit.attitude_max_age_s);
                      return decide_exit(y, cx.site.exit).mode == "PASSIVE" ? "PASSIVE" : "";
                    },
                    [estop]() { return file_exists(estop); },
                    [glog](const std::string& k, const std::string& f) { glog->event(k, f); });
  rep.have = true;
  if (!rep.hb.out.settled) logf("[guardian] !!! EXIT UNCONFIRMED: keep damping; use the remote controller damping / emergency stop. Do NOT approach.");
  if (glog->is_open()) {  // post-exit window, then the same report / anomaly files as a runner
    sleep_s(std::min(2.0, std::max(0.0, cx.site.log_post_exit_s)));
    glog->end_stream();
    LogStats ls = glog->stats();
    glog->event("log_stats", log_stats_fields(ls, glog->lowstate()));
    glog->close();
    // (the report text is built by the caller; the base file is written here from the pieces we have)
    const ExitOutcome& o = rep.hb.out;
    std::string base = "{\"summary\":{" + jstr("result", o.verdict.result) + "," + jstr("exit_reason", reason) + "," + jstr("fault", fault_name(fault_kind)) + "," + jstr("final_mode", o.final_mode) + "," + jnum("final_fsm", o.final_fsm) + "," +
                       jbool("recovery_ok", o.verdict.recovery_ok) + "," + jbool("took_control", true) + "},\"result\":\"" + o.verdict.result + "\"," + jstr("exit_reason", reason) + "," + jstr("owner", "guardian") + "," + jbool("recovery_ok", o.verdict.recovery_ok) +
                       "," + jbool("took_control", true) + "," + jnum("run_s", 0) + "," + jnum("authority", 0) + ",\"exit_detail\":" + exit_outcome_json(rep.hb.req, o) + "}";
    FinalizeResult fr = finalize_run(rep.log_dir, base, "guardian take-over log");
    logf("[guardian] take-over log: %s (anomaly %s: %d fail, %d warn)", rep.log_dir.c_str(), fr.level.c_str(), fr.n_fail, fr.n_warn);
  }
  return rep;
}

int report_rc(const TakeOverReport& r) {
  if (!r.have) return 0;
  const auto& v = r.hb.out.verdict;
  return (v.exit_code == 0 || v.exit_code == 10) ? 0 : v.exit_code;
}

std::string report_json(const TakeOverReport& r, const std::string& action) {
  std::string s = "{" + jstr("action", action) + "," + jstr("reason", r.reason) + "," + jnum("killed_pid", r.killed_pid) + "," + jnum("kill_ms", r.kill_ms) + "," + jstr("log_dir", r.log_dir);
  if (r.have) {
    const ExitOutcome& o = r.hb.out;
    s += "," + jstr("result", o.verdict.result) + "," + jstr("requested_mode", o.requested_mode) + "," + jstr("final_mode", o.final_mode) + "," + jnum("final_fsm", o.final_fsm) + "," + jbool("recovery_ok", o.verdict.recovery_ok) + "," +
         jstr("voice", o.verdict.voice) + "," + jnum("fsm_after", o.final_fsm) + "," + jnum("rpc_code", o.rpc_code) + "," + jbool("confirmed", o.settled) + "," + jstr("exit_mode", o.requested_mode) + "," +
         jstr("exit_mode_why", r.hb.req.decision.why) + ",\"exit_detail\":" + exit_outcome_json(r.hb.req, o);
  }
  return s + "," + jstr("note", r.note) + "}";
}

}  // namespace

int run_guardian(int argc, char** argv, const Paths& P) {
  Args a;
  std::string err;
  if (!a.parse(argc, argv, flags_guardian(), err)) { fprintf(stderr, "guardian: %s\n", err.c_str()); return 2; }
  const bool sim = a.has("--sim");
  const std::string iface = a.get("--iface");
  const int domain = sim ? SIM_DOMAIN : (int)a.num("--domain", 0);
  if (iface.empty() || (sim && iface != "lo") || (!sim && iface == "lo")) { fprintf(stderr, "guardian: --iface required (lo only with --sim)\n"); return 2; }
  const std::string lock_path = a.get("--lock-path", DEFAULT_LOCK), estop_file = a.get("--estop-file", DEFAULT_ESTOP), alive_file = a.get("--alive-file", GUARDIAN_ALIVE);
  const double silence_s = a.num("--silence-s", 0.30), poll_s = a.num("--poll-s", 0.01), remote_grace = a.num("--remote-grace-s", 0.25), max_run_s = a.num("--max-run-s", 3600.0);
  const double estop_grace = a.num("--estop-grace-s", 0.10);
  const std::string forced_mode = a.get("--exit-mode", "AUTO"), report = a.get("--report");
  int entry_fsm = a.kv.count("--entry-fsm") ? (int)a.num("--entry-fsm", -1) : -1;
  log_set_tag("[guardian]");
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  for (int s : {SIGINT, SIGTERM, SIGHUP}) sigaction(s, &sa, nullptr);

  Ctx cx;
  cx.site = Site::load(P.config + "/site.conf");
  if (!cx.site.err.empty()) { fprintf(stderr, "guardian: %s\n", cx.site.err.c_str()); return 2; }
  cx.site.exit.estop_file = estop_file;
  Params params;
  if (!Params::load(P.asset("policy_params.txt"), params, err)) { fprintf(stderr, "guardian: %s\n", err.c_str()); return 2; }
  cx.params = &params;
  cx.guard.reset(new SafetyGuard(params, cx.site.guard));
  cx.alive_file = alive_file;
  cx.logs = P.logs;

  LinkOpts lo;
  lo.rpc_timeout_s = 1.0;
  Link link(iface, domain, lo);
  cx.link = &link;
  link.enable_write();  // publisher built now, silent until a take-over (no latency at take-over time)
  cx.ga.pid = (int)getpid(); cx.ga.iface = iface; cx.ga.user_writers = 1;
  if (entry_fsm < 0) {
    sleep_s(1.0);
    auto c0 = link.get_fsm();
    entry_fsm = (c0.first == 0 && c0.second != 0 && c0.second != 1) ? c0.second : -1;
    logf("entry fsm observed at start: %d (the runner publishes the real entry fsm in the lock file when it takes control)", entry_fsm);
  }
  logf("watching lock %s, silence %.2fs, estop file %s, exit owner timeout %.1fs", lock_path.c_str(), silence_s, estop_file.c_str(), cx.site.exit.exit_owner_timeout_s);
  // heartbeat thread: the alive file keeps being written while a take-over is in progress (the main loop is busy then)
  std::atomic<bool> hb_stop{false};
  std::thread hb([&]() {
    while (!hb_stop) {
      {
        std::lock_guard<std::mutex> lk(cx.mx);
        cx.ga.t = wall();
        GuardianAlive::write(alive_file, cx.ga);
      }
      sleep_s(poll_s);
    }
  });
  StartDoubleClick det;
  DampCombo damp(cx.site.exit.l2b_hold_s);
  std::atomic<double> remote_t{0}, l2b_t{0};
  std::atomic<bool> runner_held{false};
  link.add_state_listener([&](const LinkState& st) {
    uint16_t k = keys_from_wireless(st.wireless);
    if (det.feed(st.arrival, start_pressed(k)) && remote_t.load() == 0) remote_t.store(mono());
    if (damp.feed(st.arrival, k, runner_held.load()) && l2b_t.load() == 0) l2b_t.store(mono());
  });

  double t0 = mono(), last_rx_t = mono(), last_poll = 0, estop_seen = 0;
  int armed_pid = 0, lock_entry = -1, user_fsm_seen = -1;
  uint64_t last_rx = link.cmd_rx_count(), rx_at_arm = last_rx;
  bool seen_user = false;
  std::atomic<bool> poll_busy{false};
  std::atomic<int> poll_fsm{-1};
  std::string result_json = "{\"action\":\"none\"}";
  int rc_out = 0;
  while (mono() - t0 < max_run_s && !g_stop) {
    sleep_s(poll_s);
    LockInfo li = lock_probe(lock_path);
    runner_held = li.held;
    if (li.entry_fsm >= 0) lock_entry = li.entry_fsm;
    uint64_t rx = link.cmd_rx_count();
    double now = mono();
    if (rx != last_rx) { last_rx = rx; last_rx_t = now; }
    std::string trigger;
    Fault tf = Fault::WATCHDOG;
    bool pre = false;
    double t_trig = now;
    int pid = li.pid;
    // a runner-owned exit in progress (unexpired) is NEVER interfered with: single exit owner (no control race)
    const bool runner_exiting = li.held && (li.exit_phase == "EXITING" || li.exit_phase == "DONE") && wall() - li.exit_t < cx.site.exit.exit_owner_timeout_s;
    if (file_exists(estop_file)) {
      if (li.held) {
        if (!runner_exiting) {
          if (estop_seen == 0) estop_seen = now;
          if (now - estop_seen >= estop_grace) { trigger = "estop file and the runner did not start its own emergency exit within " + std::to_string(estop_grace) + " s"; tf = Fault::ESTOP; t_trig = estop_seen; }
        }
      } else if (armed_pid == 0 && now - last_poll > 1.0) {  // no runner: only act if the robot is positively in user control
        last_poll = now;
        auto c = link.get_fsm();
        if (c.first == 0 && c.second == USER_FSM_EXPECTED) { trigger = "estop file and the robot is in user control without a runner"; tf = Fault::ESTOP; }
      }
    } else estop_seen = 0;
    if (!trigger.empty()) {
    } else if (remote_t.load() > 0 && li.held && !runner_exiting && now - remote_t.load() > remote_grace && now - link.last_active_rx() < 0.1) {
      trigger = "remote START double-click " + std::to_string(now - remote_t.load()) + "s ago and the runner still sends stiff frames (its own stop did not act)";
      tf = Fault::REMOTE_STOP; t_trig = remote_t.load();
    } else if (l2b_t.load() > 0 && li.held && !runner_exiting && now - l2b_t.load() > remote_grace && now - link.last_active_rx() < 0.1) {
      trigger = "remote L2+B " + std::to_string(now - l2b_t.load()) + "s ago and the runner still sends stiff frames (its own emergency exit did not act)";
      tf = Fault::REMOTE_DAMP; t_trig = l2b_t.load();
    } else if (remote_t.load() > 0 && !li.held && armed_pid == 0) {
      remote_t.store(0);  // nobody to stop
    } else if (l2b_t.load() > 0 && !li.held && armed_pid == 0) {
      l2b_t.store(0);
    } else if (li.held) {
      if (armed_pid != pid) { armed_pid = pid; last_rx_t = now; rx_at_arm = rx; logf("runner pid %d holds the control lock (waiting for its first command)", pid); }
      if (rx != rx_at_arm && !seen_user) { seen_user = true; logf("armed: runner is commanding"); }
      if (seen_user && now - last_poll > 0.5 && !poll_busy && !runner_exiting) {
        last_poll = now;
        poll_busy = true;  // the fsm query runs in a thread: a stalled RPC (timeout 1 s) must not freeze the trigger loop
        int ref = lock_entry >= 0 ? lock_entry : entry_fsm;
        std::thread([&link, &poll_busy, &poll_fsm, ref]() {
          auto c1 = link.get_fsm();
          if (c1.first == 0 && c1.second >= 0 && c1.second != ref) poll_fsm = c1.second;
          poll_busy = false;
        }).detach();
      }
      if (poll_fsm.load() >= 0) user_fsm_seen = poll_fsm.load();
      if (seen_user && !runner_exiting && now - last_rx_t > silence_s) { trigger = "runner silent " + std::to_string(now - last_rx_t) + "s on rt/user_lowcmd (hung/stopped)"; tf = Fault::WATCHDOG; }
    } else if (armed_pid != 0) {
      seen_user = false;
      // lock released: normal exit -> OEM fsm back at entry; crash -> still in user control
      auto c = link.get_fsm();
      if (c.first == 0 && c.second >= 0 && (li.exit_phase == "DONE" || (user_fsm_seen >= 0 && c.second != user_fsm_seen && li.exit_phase != "EXITING"))) {
        logf("runner released the lock (exit phase %s) and fsm moved %d -> %d: normal end, guardian done%s", li.exit_phase.c_str(), user_fsm_seen, c.second, c.second == 1 ? " (fsm 1 = PASSIVE: see the runner's run_report.json)" : "");
        result_json = "{\"action\":\"none\",\"note\":\"normal end\"}";
        break;
      }
      if (c.first == 0 && c.second >= 0) { trigger = "runner pid " + std::to_string(armed_pid) + " gone but OEM fsm=" + std::to_string(c.second) + " is not confirmed back at the OEM (crash?)"; tf = Fault::WATCHDOG; pre = c.second != USER_FSM_EXPECTED && c.second != user_fsm_seen; }  // still the user fsm: RPC at once; any other fsm: verify first, no blind RPC
      else if (now - last_rx_t > 1.0) { trigger = "runner gone and fsm unreadable (rpc code " + std::to_string(c.first) + ")"; tf = Fault::WATCHDOG; pre = true; }
    }
    if (!trigger.empty()) {
      logf("TAKE-OVER: %s", trigger.c_str());
      TakeOverReport rep = take_over(cx, trigger, tf, armed_pid ? armed_pid : pid, lock_entry >= 0 ? lock_entry : entry_fsm, user_fsm_seen >= 0 ? user_fsm_seen : USER_FSM_EXPECTED, t_trig, pre, forced_mode);
      result_json = report_json(rep, "take_over");
      rc_out = report_rc(rep);
      logf("done: %s", result_json.c_str());
      break;
    }
  }
  if (!report.empty()) write_atomic(report, result_json);
  hb_stop = true;
  hb.join();
  ::unlink(alive_file.c_str());
  fast_exit(rc_out);
}

// Normal (non-emergency) hand-back: ONLY if the robot is positively in user control (GetFsmId == user fsm) and no runner holds the lock. Same ExitPolicy.
int run_release(int argc, char** argv, const Paths& P) {
  Args a;
  std::string err;
  if (!a.parse(argc, argv, {}, err)) { fprintf(stderr, "release: %s\n", err.c_str()); return 2; }
  std::string iface = a.get("--iface");
  if (iface.empty()) { fprintf(stderr, "usage: partymimic release --iface IFACE [--domain 0] [--user-fsm 1000] [--exit-mode AUTO|WALKRUN|PASSIVE]\n"); return 2; }
  int uf = (int)a.num("--user-fsm", USER_FSM_EXPECTED);
  std::string mode = a.get("--exit-mode", "AUTO");
  if (mode != "AUTO" && mode != "WALKRUN" && mode != "PASSIVE") { fprintf(stderr, "release: --exit-mode must be AUTO|WALKRUN|PASSIVE\n"); return 2; }
  LockInfo li = lock_probe(a.get("--lock-path", DEFAULT_LOCK));
  if (li.held) { printf("REFUSED: a runner (pid %d) holds the control lock. Stop it first (./stop.sh).\n", li.pid); return 2; }
  Ctx cx;
  cx.site = Site::load(P.config + "/site.conf");
  if (!cx.site.err.empty()) { fprintf(stderr, "release: %s\n", cx.site.err.c_str()); return 2; }
  Params params;
  if (!Params::load(P.asset("policy_params.txt"), params, err)) { fprintf(stderr, "release: %s\n", err.c_str()); return 2; }
  cx.params = &params;
  cx.logs = P.logs;
  cx.guard.reset(new SafetyGuard(params, cx.site.guard));
  LinkOpts lo;
  lo.rpc_timeout_s = 1.0;
  Link link(iface, (int)a.num("--domain", 0), lo);
  cx.link = &link;
  sleep_s(0.5);
  auto c = link.get_fsm();
  if (c.first != 0 || c.second < 0) { printf("NOTHING SENT: GetFsmId unreadable (rpc code %d). Use the remote controller damping.\n", c.first); return 3; }
  if (c.second != uf) { printf("NOTHING SENT: fsm=%d is not the user-control fsm (%d): the robot is not in user control.\n", c.second, uf); return 0; }
  printf("robot is in user control (fsm %d); handing back (%s)\n", c.second, mode.c_str());
  link.enable_write();  // build the publisher (and let the robot match it) BEFORE the exit clock starts
  sleep_s(0.3);
  TakeOverReport rep = take_over(cx, "operator release", Fault::SIGNAL, 0, -1, uf, mono(), false, mode);
  printf("%s\n", report_json(rep, "release").c_str());
  fast_exit(report_rc(rep));
}

}  // namespace pm
