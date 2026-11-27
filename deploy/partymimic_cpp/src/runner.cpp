// Runner: loads the frozen P2 Policy, checks the live robot read-only, waits for GO, takes control from WalkRun, runs the dance window (lead-in + dance + end hold), hands back.
// Phases (status file `phase`):
//   STARTING -> LOADING_MODEL (ONNX + observation stack + warm-up on live states, NO command) -> CHECKING (real-time admission, read-only)
//   -> READY (waits for the GO file written by the service on L1+A; SIGTERM / START x2 abort WITHOUT ever touching control; V1.6.1: READY has NO timeout, it is held until L1+A / START x2 / SIGTERM)
//   -> ARM (fast preflight, lock, SwitchToUserCtrl, confirm) -> RUN (Policy drives from the first tick; authority ramp 0->max in 0.1 s)
//   -> EXIT (SwitchToInternalCtrl at once on a dedicated thread, HOLD frames meanwhile -> verify by GetFsmId + stable window; see pm/exit_policy.hpp).
// Any guard fault / instability / START x2 / watchdog -> the exit starts IMMEDIATELY (WALKRUN, or PASSIVE if fallen / untrusted / emergency channel).
// V1.5.1 STANDBY (pre-warmed runner): `--standby --trigger-file F` = the service starts a runner at SERVICE START-UP (and after every finished run). It does everything that does not need the
//   operator yet - hard refusals, parameter/motion load, DDS Link (persistent traffic monitors + discovery, created ONCE), ONNX load + warm-up on live states - then idles in phase STANDBY with NO
//   lock, NO command writer, NO RPC. F1+F3 makes the service write the trigger file (run directory inside); the runner then runs CHECKING -> READY exactly as a cold runner would (real-time checks on the
//   rolling history of the last 1 s, GetFsmId, guardian, frozen hashes re-verified in parallel) and the GO / ARM path is unchanged. Every run is still a FRESH PROCESS (new Policy history / previous action).
// Exit codes: 0 clean (COMPLETED) | 1 crash | 2 refused | 3 no lowstate | 4 ARM failed | 5 EXIT UNCONFIRMED | 6 BLOCKED by admission | 10 fault/stop, WalkRun confirmed
//             11 WALKRUN_RECOVERY_FAILED | 12 EMERGENCY_PASSIVE | 13 FALLEN_PASSIVE (PASSIVE decided: robot down / state untrusted)
#include <algorithm>
#include <signal.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <cmath>
#include <csignal>
#include <cstring>
#include <deque>
#include <fstream>
#include <future>
#include <thread>

#include "pm/app.hpp"
#include "pm/frozen.hpp"
#include "pm/guard.hpp"
#include "pm/link.hpp"
#include "pm/logtool.hpp"
#include "pm/policy.hpp"
#include "pm/runlog.hpp"
#include "pm/session.hpp"
#include "pm/version.hpp"

namespace pm {
namespace {

constexpr double DT = 1.0 / 50.0;
// authority 0 -> max over --ramp-s (site.conf ramp_s, default 0.3 s) so the first policy frame is not a step

std::atomic<bool>* g_sess_quit = nullptr;
std::atomic<bool> g_quit{false};
void on_signal(int) {
  g_quit.store(true);
  if (g_sess_quit) g_sess_quit->store(true);
}

double smoothstep(double x) {
  x = std::min(std::max(x, 0.0), 1.0);
  return x * x * x * (10 + x * (-15 + 6 * x));
}

struct Item { std::string name; int level; std::string detail; bool pass() const { return level != L_FAIL; } };

RobotSample to_sample(const LinkState& s) {
  RobotSample r;
  for (int i = 0; i < N; i++) { r.q[i] = (float)s.q[i]; r.dq[i] = (float)s.dq[i]; }
  for (int i = 0; i < 4; i++) r.quat[i] = (float)s.quat[i];
  for (int i = 0; i < 3; i++) r.gyro[i] = (float)s.gyro[i];
  return r;
}

std::string items_json(const std::vector<Item>& items) {
  std::string s = "[";
  for (size_t i = 0; i < items.size(); i++) s += (i ? "," : "") + std::string("{") + jstr("item", items[i].name) + "," + jbool("pass", items[i].pass()) + "," + jstr("level", level_name(items[i].level)) + "," + jstr("detail", items[i].detail) + "}";
  return s + "]";
}

std::string vec_json(const Vec& v) {
  std::string s = "[";
  for (int i = 0; i < N; i++) { char b[32]; snprintf(b, sizeof b, "%s%.6g", i ? "," : "", v[i]); s += b; }
  return s + "]";
}

// compact JSON snapshot of a lowstate sample (hand-over diagnostics)
std::string state_snapshot_json(const LinkState& st) {
  double roll, pitch;
  roll_pitch(st.quat, roll, pitch);
  double dqm = st.dq.cwiseAbs().maxCoeff(), taum = st.tau.cwiseAbs().maxCoeff();
  return "{" + jnum("t_mono", st.arrival) + "," + jnum("lowstate_tick", st.tick) + "," + jnum("tilt_rad", tilt_of_quat(st.quat)) + "," + jnum("roll", roll) + "," + jnum("pitch", pitch) + "," +
         jnum("gyro_x", st.gyro[0]) + "," + jnum("gyro_y", st.gyro[1]) + "," + jnum("gyro_z", st.gyro[2]) + "," + jnum("max_abs_dq", dqm) + "," + jnum("max_abs_tau_est", taum) + "," + jnum("mode_machine", st.mode_machine) + "}";
}
std::string arr_json(const std::vector<std::string>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); i++) s += (i ? ",\"" : "\"") + json_escape(v[i]) + "\"";
  return s + "]";
}

}  // namespace

WriterVerdict oem_writer_verdict(const Writers& w, const std::map<std::string, uint64_t>& traffic, double window_s, int own, int fsm, const Site& site) {
  auto cnt = [&](const char* t) { auto it = w.find(t); return it == w.end() ? 0 : (int)it->second.size(); };
  auto rx = [&](const char* t) { auto it = traffic.find(t); return it == traffic.end() ? (uint64_t)0 : it->second; };
  auto hz = [&](const char* t) { return window_s > 0 ? (double)rx(t) / window_s : 0.0; };
  const int n_low = cnt("rt/lowcmd"), n_arm = cnt("rt/arm_sdk"), n_user = cnt("rt/user_lowcmd");
  const int foreign_user = std::max(0, n_user - own);
  char b[512];
  snprintf(b, sizeof b, "rt/lowcmd: %d writer(s), %.0f Hz [factory motion service]; rt/arm_sdk: %d writer(s), %.0f Hz [%s]; rt/user_lowcmd: %d writer(s) = %d guardian + %d other, %.0f Hz [%s]",
           n_low, hz("rt/lowcmd"), n_arm, hz("rt/arm_sdk"), rx("rt/arm_sdk") ? "ACTIVE" : "idle", n_user, std::min(n_user, own), foreign_user, hz("rt/user_lowcmd"), rx("rt/user_lowcmd") ? "ACTIVE" : "idle");
  std::string seen = b;
  if (rx("rt/user_lowcmd") > 0 || rx("rt/arm_sdk") > 0)
    return {L_FAIL, "COMPETING CONTROLLER: another process is actively commanding " + std::string(rx("rt/user_lowcmd") ? "rt/user_lowcmd" : "rt/arm_sdk") + " (" + seen + ")"};
  if (site.oem_confirmed && site.oem_expected_writers >= 0 && n_low != site.oem_expected_writers)
    return {L_FAIL, "rt/lowcmd writer count " + std::to_string(n_low) + " != OEM-confirmed " + std::to_string(site.oem_expected_writers) + " (another writer appeared or the service changed); " + seen};
  if (site.oem_confirmed && site.oem_expected_writers >= 0) return {L_PASS, "matches OEM baseline " + std::to_string(site.oem_expected_writers) + "; " + seen};
  std::string note;
  if (n_low == 0) note = " NOTE: no rt/lowcmd writer seen (factory motion service not driving?)";
  else if (rx("rt/lowcmd") == 0 && (fsm == 500 || fsm == 801 || fsm == 802)) note = " NOTE: fsm " + std::to_string(fsm) + " but no rt/lowcmd traffic seen";
  return {note.empty() ? L_PASS : L_WARN, "no active competing writer; idle factory writers are not conflicts; rt/lowcmd baseline not set (informational). " + seen + note};
}

int run_runner(int argc, char** argv, const Paths& P) {
  Args a;
  std::string err;
  if (!a.parse(argc, argv, flags_runner(), err)) { fprintf(stderr, "runner: %s\n", err.c_str()); return 2; }
  const bool sim = a.has("--sim");
  const std::string iface = a.get("--iface");
  const int domain = sim ? SIM_DOMAIN : (int)a.num("--domain", 0);
  const double run_s = a.num("--run-s", 1.0), amax = a.num("--authority-max", 0.3);
  const bool cli_diverge = a.kv.count("--diverge-rad") != 0;
  const std::string ref_mode = a.get("--ref-mode", "running");
  const double ramp_s = a.num("--ramp-s", 0.3), end_hold_s = a.num("--end-hold-s", 0.5);
  const bool allow_partial = a.has("--allow-partial-authority");
  const bool standby = a.has("--standby");               // V1.5.1: pre-warmed runner, waits for --trigger-file before CHECKING
  const std::string trigger_file = a.get("--trigger-file");
  const bool check_only = a.has("--check-only");  // P5.2: read-only F1+F3 check: no motion gates apply because this process can NEVER take control
  const std::string lock_path = a.get("--lock-path", DEFAULT_LOCK), estop_file = a.get("--estop-file", DEFAULT_ESTOP);
  const std::string wait_go = a.get("--wait-go"), status_file = a.get("--status-file"), announced_file = a.get("--announced-file");
  const int warmup_n = (int)a.num("--warmup-n", 8);
  std::string run_dir = standby ? std::string() : a.get("--run-dir", P.logs + "/runs/" + timestamp_compact());  // standby: the run directory arrives with the trigger
  ::mkdir(P.logs.c_str(), 0775);
  ::mkdir((P.logs + "/runs").c_str(), 0775);
  if (!standby) {
    ::mkdir(run_dir.c_str(), 0775);
    log_set_file(run_dir + "/console.log");
  }
  log_set_tag("[runner]");
  if (standby && trigger_file.empty()) { fprintf(stderr, "runner: --standby needs --trigger-file\n"); return 2; }
  RunLogger rlog;  // P5.1: opened after the hard refusals (needs the joint names); earlier events wait in `pending_ev`
  std::vector<std::tuple<double, std::string, std::string>> pending_ev;
  auto ev_write = [&](const std::string& kind, const std::string& fields) {
    if (rlog.is_open()) rlog.event(kind, fields);
    else pending_ev.emplace_back(mono(), kind, fields);
  };
  auto status = [&](const std::string& phase) {
    ev_write("phase", jstr("phase", phase) + "," + jnum("t_mono", mono()));
    if (status_file.empty()) return;
    char b[256];
    snprintf(b, sizeof b, "{\"phase\":\"%s\",\"tm\":%.6f,\"time\":%.6f,\"pid\":%d}", phase.c_str(), mono(), wall(), (int)getpid());
    write_atomic(status_file, b);
  };
  status("STARTING");
  const std::string mode_arg = a.get("--mode", "");
  // P5.2: a refusal is a RESULT, not a crash: run_report.json (result REFUSED + the exact reason) + anomaly summary, so the service never has to "salvage" it as CRASH_NO_REPORT
  auto refuse = [&](const std::string& why) {
    logf("REFUSED: %s", why.c_str());
    if (run_dir.empty()) { status("REFUSED"); return 2; }  // standby before the trigger: no run exists yet; F1+F3 then starts a cold runner, which writes the REFUSED report
    std::string rep = "{\"summary\":{" + jstr("result", "REFUSED") + "," + jstr("exit_reason", why) + "," + jbool("took_control", false) + "," + jnum("exit_code", 2) + "}," + jstr("result", "REFUSED") + "," + jstr("exit_reason", why) + "," + jnum("exit_code_planned", 2) + "," +
                      jstr("mode", mode_arg.empty() ? "custom" : mode_arg) + "," + jstr("target", sim ? "sim" : "real") + "," + jbool("check_only", check_only) + "," + jstr("program_version", PM_VERSION) + "," + jbool("took_control", false) + "," + jnum("run_s", run_s) + "," + jnum("authority", amax) + "}";
    finalize_run(run_dir, rep, "runner refusal");
    status("REFUSED");
    return 2;
  };

  Site site = Site::load(P.config + "/site.conf");
  if (!site.err.empty()) return refuse(site.err);
  for (auto& w : site.warnings) logf("WARNING: %s", w.c_str());
  const double unconfirmed_hold = a.num("--unconfirmed-hold-s", site.exit.unconfirmed_hold_s);
  const double announce_wait = a.kv.count("--announce-wait-s") ? a.num("--announce-wait-s", site.announce_wait_s) : site.announce_wait_s;
  if (!(announce_wait >= 0 && announce_wait <= 5)) return refuse("--announce-wait-s must be in [0, 5] s");
  const double diverge_rad = cli_diverge ? a.num("--diverge-rad", 1.5) : site.guard.diverge_rad;
  const bool exceeds_legacy = run_s > LEGACY_MAX_RUN_S + 1e-9 || amax > LEGACY_MAX_AUTHORITY + 1e-9;
  std::vector<int> allowed = site.entry_allowed;
  if (a.kv.count("--entry-fsm")) allowed = {(int)a.num("--entry-fsm", 500)};

  // ---- hard refusals before ANY DDS object exists
  // V1.6.1: no manual authorisation / confirmation gate. What remains are the NUMERIC caps (site.conf max_run_s / max_authority, compile-time ceilings). check-only can never take control: no caps apply to it.
  if (!sim) {
    if (iface.empty() || iface == "lo") return refuse("a real run needs the robot network interface (not lo)");
    if (!check_only) {
      std::string bad;
      if (!(run_s > 0 && run_s <= site.max_run_s)) bad += "run_s " + std::to_string(run_s) + " not in (0, max_run_s=" + std::to_string(site.max_run_s) + "] (site.conf max_run_s, compile-time limit " + std::to_string(ABS_MAX_RUN_S) + "); ";
      if (!(amax > 0 && amax <= site.max_authority)) bad += "authority " + std::to_string(amax) + " not in (0, max_authority=" + std::to_string(site.max_authority) + "] (site.conf max_authority, compile-time limit " + std::to_string(ABS_MAX_AUTHORITY) + "); ";
      if (!bad.empty()) return refuse(bad);
    }
  } else if (iface != "lo") return refuse("--sim must use --iface lo");
  if (sim && !check_only && !(run_s > 0 && run_s <= ABS_MAX_RUN_S && amax > 0 && amax <= ABS_MAX_AUTHORITY)) return refuse("run_s/authority above the compile-time limits (" + std::to_string((int)ABS_MAX_RUN_S) + " s / " + std::to_string(ABS_MAX_AUTHORITY) + ")");
  if (!check_only && exceeds_legacy) logf("WARNING: run_s=%.2f / authority=%.2f EXCEED the legacy real-robot caps (%.1f s / %.1f): allowed by site.conf max_run_s=%.2f max_authority=%.2f (compile-time limits %.0f s / %.1f)", run_s, amax, LEGACY_MAX_RUN_S, LEGACY_MAX_AUTHORITY, site.max_run_s, site.max_authority, ABS_MAX_RUN_S, ABS_MAX_AUTHORITY);
  if (ref_mode != "running" && ref_mode != "paused") return refuse("bad --ref-mode");
  if (!(ramp_s >= 0 && ramp_s <= 2.0) || !(end_hold_s >= 0 && end_hold_s <= 5.0)) return refuse("--ramp-s must be in [0,2] and --end-hold-s in [0,5]");
  if (!check_only && amax > 0.05 && amax < 0.9 && !allow_partial)
    return refuse("sustained authority " + std::to_string(amax) + " is in the partial-authority regime (0.05, 0.9): in Sim2Sim (real-log initial states) q_cmd = q_hold + a*(q_target - q_hold) with a<0.9 does NOT balance (a=0: falls in 1.0-1.7 s; a=0.3: falls in 1.2-2.0 s, reproducing the 2 real falls). "
                  "Use authority 1.0 (ramped by --ramp-s) or pass --allow-partial-authority for a deliberate experiment.");

  Params params;
  Motion motion;
  if (!Params::load(P.asset("policy_params.txt"), params, err) || !Motion::load(P.asset("motion.bin"), motion, err)) return refuse(err);
  const std::string mode_name = mode_arg;
  // ---- P5.1 run log options (async writer, 5 s lowstate pre-roll ring from the moment it is opened; see LOGGING.md). The options are validated NOW (a refusal must come before any DDS object);
  // a cold runner opens the log here, a standby runner opens it when the trigger arrives (the lowstate listener only feeds the ring once it is open).
  LogOptions lopt;
  lopt.dir = run_dir; lopt.joint_names = params.names; lopt.preroll_s = site.log_preroll_s; lopt.obs_bin = site.log_obs; lopt.state_stream = site.log_state_stream;
  lopt.state_cap = 16384;
  if (sim && getenv("PM_SIM_LOG_STALL")) lopt.dev_stall_s = atof(getenv("PM_SIM_LOG_STALL"));  // DEV ONLY (--sim): the writer stalls once (async-logging regression)
  {  // dimensions of the model inputs (history + future slices) without running the model
    const int proprio = params.history * (2 * N + 6 + N), quat = params.history * 8, jp = (int)params.future.size() * N, root = (int)params.future.size() * 7;
    lopt.obs_dim = proprio + quat + jp + root;
    lopt.obs_layout_json = "[{\"name\":\"proprio\",\"dim\":" + std::to_string(proprio) + "},{\"name\":\"root_quat_hist\",\"dim\":" + std::to_string(quat) + "},{\"name\":\"ref_joint_pos\",\"dim\":" + std::to_string(jp) +
                           "},{\"name\":\"ref_root\",\"dim\":" + std::to_string(root) + "}]";
  }
  if (lopt.obs_dim > OBS_MAX) return refuse("obs_dim " + std::to_string(lopt.obs_dim) + " > OBS_MAX " + std::to_string(OBS_MAX));
  auto open_runlog = [&]() {
    lopt.dir = run_dir;
    if (!rlog.open(lopt)) logf("WARNING: run log could not be opened in %s (%s): the run would be UNRECORDED", run_dir.c_str(), strerror(errno));
    for (auto& e : pending_ev) rlog.event_at(std::get<0>(e), std::get<1>(e), std::get<2>(e));
    pending_ev.clear();
  };
  if (!standby) open_runlog();
  GuardConfig gcfg = site.guard;
  gcfg.torque_ff = a.has("--no-torque-ff") ? false : (gcfg.torque_ff || a.has("--torque-ff"));  // P5.3 option (default off): training-equivalent torque as LowCmd.tau
  // P5.3: the clip robust_d was trained on starts with 1 s of frozen first frame (ssb_padded); running modes replay that in memory. motion.bin (frozen hash) is untouched.
  const RunTimeline tl = plan_timeline(motion, ref_mode == "running", run_s, site.lead_in_s, end_hold_s);  // V1.5.1: ONE function plans the window (also checked by selfcheck): lead-in + whole clip + last-frame hold
  const int lead_in_frames = tl.lead_in_frames;
  const Motion motion_run = motion.with_lead_in(lead_in_frames);
  SafetyGuard guard(params, gcfg);
  SessionConfig scfg;
  site.apply_to(scfg);
  scfg.lock_path = lock_path; scfg.estop_file = estop_file; scfg.entry_allowed = allowed;
  scfg.exit.unconfirmed_hold_s = unconfirmed_hold; scfg.exit.estop_file = estop_file;
  Vec kp = params.kp, kd = params.kd;

  LinkOpts lo;
  lo.rpc_timeout_s = 2.0;
  lo.monitor_traffic = true;  // V1.5.1: persistent traffic monitors + discovery reader + rolling history, created ONCE here (never destroyed: see pm/link.hpp)
  Link link(iface, domain, lo);
  // V1.5.1: a standby runner has no run directory yet (the run log opens at the trigger), but the 5 s lowstate PRE-ROLL before the take-over must be as long as with a cold runner: until the trigger
  // the listener keeps a private ring of the last preroll_s + 1 s, replayed into the run log's ring when it opens.
  std::mutex pre_m;
  std::deque<LinkState> pre_ring;
  bool pre_live = !standby;
  const double pre_keep_s = site.log_preroll_s + 1.0;
  link.add_state_listener([&](const LinkState& st) {  // DDS reader thread: ring / stream of every lowstate sample + arrival statistics
    {
      std::lock_guard<std::mutex> lk(pre_m);
      if (!pre_live) {
        pre_ring.push_back(st);
        while (!pre_ring.empty() && pre_ring.front().arrival < st.arrival - pre_keep_s) pre_ring.pop_front();
        return;
      }
    }
    rlog.state(st);
  });
  {
    auto last_fsm = std::make_shared<std::atomic<int>>(-2);
    link.set_rpc_observer([&rlog, last_fsm](const RpcEvent& e) {  // every sport-service request and response (+ FSM changes)
      char b[512];
      snprintf(b, sizeof b, "\"api\":\"%s\",\"arg\":\"%s\",\"rc\":%d,\"fsm\":%d,\"t_req\":%.6f,\"t_resp\":%.6f,\"latency_ms\":%.3f,\"exception\":\"%s\"", e.api.c_str(), e.arg.c_str(), e.rc, e.fsm, e.t_req, e.t_resp, (e.t_resp - e.t_req) * 1e3,
               json_escape(e.exception).c_str());
      rlog.event_at(e.t_resp, "rpc", b);
      if (e.api == "GetFsmId" && e.rc == 0 && e.fsm >= 0) {
        int prev = last_fsm->exchange(e.fsm);
        if (prev != e.fsm) { snprintf(b, sizeof b, "\"from\":%d,\"to\":%d", prev, e.fsm); rlog.event_at(e.t_resp, "fsm_change", b); }
      }
    });
  }
  ControlSession sess(link, guard, scfg, [&](const std::string& k, const std::string& f) { ev_write(k, f); });
  sess.set_logger(&rlog);
  sess.set_hold_gains(kp, kd);
  g_sess_quit = &sess.quit_flag();
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  for (int s : {SIGINT, SIGTERM, SIGHUP}) sigaction(s, &sa, nullptr);

  logf("runner: %s iface=%s domain=%d run_s=%.2f ref=%s authority<=%.2f entry_allowed=%s", sim ? "SIM" : "real", iface.c_str(), domain, run_s, ref_mode.c_str(), amax,
       [&] { std::string s; for (int e : allowed) s += std::to_string(e) + " "; return s; }().c_str());

  std::string result, exit_reason, run_end, inject_kind;
  int inject_tick = -1;
  if (sim && getenv("PM_SIM_INJECT")) {  // "kind:tick" with kind in inference|nan|overrun|hang
    std::string v = getenv("PM_SIM_INJECT");
    size_t c = v.find(':');
    if (c != std::string::npos) { inject_kind = v.substr(0, c); inject_tick = atoi(v.c_str() + c + 1); }
  }
  int exit_code = 0, n_ticks = 0;
  double first_cmd_step = -1, first_cmd_auth = -1;
  std::vector<double> warm_ms;
  std::vector<Item> items;
  std::string phase = "STARTING", writers_dump, ready_state_json;
  int ready_fsm = -1;
  bool go_refused = false;
  std::string cancel_voice;  // V1.6: set when the take-over was cancelled while waiting (announce_wait)
  std::string pre_state_json = "null", take_state_json = "null";
  const double t_start_mono = mono();
  ArtifactHashes hashes;
  std::string stage_json, manifest_extra;
  auto goto_phase = [&](const std::string& p) { phase = p; logf("=== phase %s", p.c_str()); status(p); };
  std::unique_ptr<PolicyStack> ps;

  auto guardian_item = [&]() -> Item {
    GuardianAlive g;
    if (!GuardianAlive::read(GUARDIAN_ALIVE, g)) return {"guardian_alive", L_FAIL, "no guardian heartbeat"};
    double age = wall() - g.t;
    if (age > GUARDIAN_MAX_AGE_S) return {"guardian_alive", L_FAIL, "heartbeat " + std::to_string(age) + " s old (> 1 s)"};
    if (g.iface != iface) return {"guardian_alive", L_FAIL, "guardian watches " + g.iface + ", not " + iface};
    if (g.pid <= 0 || (kill(g.pid, 0) != 0 && errno != EPERM)) return {"guardian_alive", L_FAIL, "guardian pid not running"};
    return {"guardian_alive", L_PASS, "heartbeat " + std::to_string((int)(age * 1e3)) + " ms old, pid " + std::to_string(g.pid)};
  };
  // DDS writer classification (see oem_writer_verdict): discovery + a traffic window, run in parallel with the other read-only checks
  struct WriterScan { Writers w; std::map<std::string, uint64_t> rx; std::string err; double window = 0; };
  auto scan_writers = [&](double window, WriterScan& out) {
    out.window = window;
    try {
      std::thread t([&] { try { out.rx = link.measure_traffic(window); } catch (const std::exception& e) { out.err = e.what(); } });
      out.w = link.discover_writers(std::min(0.6, window));
      t.join();
    } catch (const std::exception& e) { out.err = e.what(); }
  };
  auto writer_item = [&](const WriterScan& sc, int fsm) -> Item {
    if (!sc.err.empty()) return {"dds_command_writers", L_WARN, "writer scan unavailable (" + sc.err + "): active-writer protection falls back to the user_lowcmd traffic check"};
    GuardianAlive g;
    int own = GuardianAlive::read(GUARDIAN_ALIVE, g) ? g.user_writers : 0;
    WriterVerdict v = oem_writer_verdict(sc.w, sc.rx, sc.window, own, fsm, site);
    return {"dds_command_writers", v.level, v.detail};
  };

  sess.set_expected_entry_fsm(-1);
  bool admitted = false;
  std::string fsm_seen_json = "null";
  try {
    // ---- model + warm-up (read-only)
    goto_phase(standby ? "STANDBY_LOADING" : "LOADING_MODEL");
    ps.reset(new PolicyStack(params, motion_run, P.asset("policy.onnx"), (int)(getenv("PM_ORT_THREADS") ? atoi(getenv("PM_ORT_THREADS")) : 2)));
    const int seg_frames = tl.seg_frames;  // run_s = seconds of DANCE; the lead-in comes on top; the window is over when the reference reaches this frame (whole clip = last frame)
    logf("timeline: lead-in %.2f s (%d frames) + dance %.2f s (%d frames) -> window over at tick %d (frame %d%s), then %.2f s last-frame hold; planned %.2f s, runner fail-safe %.2f s (margin %.2f s)", tl.lead_in_s, tl.lead_in_frames, tl.dance_s,
         tl.dance_frames, tl.window_over_tick, tl.seg_frames, tl.covers_whole_dance ? " = LAST frame: whole dance" : " (partial window)", tl.running ? end_hold_s : 0.0, tl.planned_total_s, tl.failsafe_s, tl.margin_s);
    if (tl.running && !tl.covers_whole_dance) logf("NOTE: this window ends BEFORE the last frame of the clip (partial mode, run_s %.2f s < dance %.2f s)", run_s, tl.dance_s);
    if (tl.margin_s < 0.3) { logf("REFUSED: runner fail-safe (%.2f s) would cut the planned window (%.2f s)", tl.failsafe_s, tl.planned_total_s); result = "REFUSED"; exit_reason = "timeline fail-safe margin too small"; exit_code = 2; throw 0; }
    double t_w = mono();
    while (!sess.state()) {
      if (g_quit) { result = "ABORTED_BEFORE_ARM"; exit_reason = "signal while loading"; throw 0; }
      if (mono() - t_w > 5.0) { logf("no lowstate within 5 s; abort"); result = "NO_LOWSTATE"; exit_code = 3; throw 0; }
      sleep_s(0.05);
    }
    {  // warm-up: ONNX/caches are hot before the first live inference (watchdog is 100 ms); live states, no command
      auto st0 = sess.state();
      ps->reset(to_sample(*st0), false);
      for (int i = 0; i < warmup_n; i++) {
        auto st = sess.state();
        warm_ms.push_back(ps->tick(to_sample(*st)).infer_ms);
        sleep_s(0.02);
      }
      if (!warm_ms.empty()) {
        double mx = *std::max_element(warm_ms.begin(), warm_ms.end());
        logf("warm-up: %zu read-only inferences, first %.1f ms, last %.1f ms, max %.1f ms", warm_ms.size(), warm_ms.front(), warm_ms.back(), mx);
        if (warm_ms.back() > 40) logf("WARNING: inference still slow after warm-up (watchdog is 100 ms)");
      }
    }
    // ---- V1.5.1 standby: everything that can be done before the operator asks is done. Wait for the trigger (F1+F3), then continue exactly like a cold runner.
    ArtifactHashes hashes_pre;
    std::future<ArtifactHashes> hash_fut;
    if (standby) {
      hashes_pre = hash_artifacts(P);  // start-up: full sha256 of every artifact (re-verified in parallel with the CHECKING items after the trigger)
      const double t_ready_sb = mono();
      logf("STANDBY: model loaded + warm-up done %.2f s after start, DDS monitors %s (age %.2f s); waiting for the F1+F3 trigger (%s). No lock, no command writer, no RPC in this phase.", t_ready_sb - t_start_mono,
           link.monitors_ready() ? "armed" : "arming", link.monitor_age_s(), trigger_file.c_str());
      goto_phase("STANDBY");
      while (!file_exists(trigger_file)) {
        if (g_quit) { logf("standby: signal received before any trigger: exit (nothing was sent)"); fast_exit(0); }
        sleep_s(0.01);
      }
      const double t_trig = mono();
      std::string tdoc;
      read_file(trigger_file, tdoc);
      ::unlink(trigger_file.c_str());
      run_dir = json_find_string(tdoc, "run_dir");
      if (run_dir.empty()) run_dir = P.logs + "/runs/" + timestamp_compact();
      ::mkdir(run_dir.c_str(), 0775);
      log_set_file(run_dir + "/console.log");
      logf("STANDBY -> TRIGGER: run dir %s (standby %.2f s)", run_dir.c_str(), t_trig - t_ready_sb);
      ev_write("standby_trigger", jnum("t_mono_standby_ready", t_ready_sb) + "," + jnum("t_mono_trigger", t_trig) + "," + jnum("standby_s", t_trig - t_ready_sb) + "," + jnum("model_load_warmup_s", t_ready_sb - t_start_mono) + "," + jstr("note", "model load + warm-up + DDS monitors were done at service start-up"));
      open_runlog();
      {  // replay the private pre-roll ring into the run log (ring), then go live
        std::lock_guard<std::mutex> lk(pre_m);
        for (const LinkState& x : pre_ring) rlog.state(x);
        logf("pre-roll: %zu standby lowstate samples (%.1f s) carried into the run log", pre_ring.size(), pre_ring.empty() ? 0.0 : pre_ring.back().arrival - pre_ring.front().arrival);
        pre_ring.clear();
        pre_live = true;
      }
      rlog.lowstate().reset_window();  // the run statistics start at the trigger, not at process start
      sess.reset_remote_flags();       // a START double-click while idle must not abort the next run
      if (!warm_ms.empty()) logf("warm-up (done at start-up): %zu read-only inferences, first %.1f ms, last %.1f ms", warm_ms.size(), warm_ms.front(), warm_ms.back());
      hash_fut = std::async(std::launch::async, [&P] { return hash_artifacts(P); });
    }
    // ---- real-time admission (read-only)
    goto_phase("CHECKING");
    if (g_quit) { result = "ABORTED_BEFORE_ARM"; exit_reason = "signal before checks"; throw 0; }
    {  // frozen artifacts (+ everything else that identifies this run: executable, config; recorded in manifest.json)
      hashes = standby ? hashes_pre : hash_artifacts(P);  // standby: the verdict below is taken from a FRESH full hash (hash_fut), joined after the other checks
      items.push_back({"artifact_policy.onnx_frozen", hashes.policy_ok ? L_PASS : L_FAIL, hashes.policy_ok ? "sha256 == frozen value (robust_d it12000, see frozen.hpp)" : "MISMATCH policy.onnx " + hashes.policy});
      items.push_back({"artifact_policy_params.txt_frozen", hashes.params_ok ? L_PASS : L_FAIL, hashes.params_ok ? "sha256 == frozen value" : "MISMATCH policy_params.txt"});
      items.push_back({"artifact_motion.bin_frozen", hashes.motion_ok ? L_PASS : L_FAIL, hashes.motion_ok ? "sha256 == frozen value" : "MISMATCH motion.bin"});
      struct statvfs vf;  // the run must be recordable: free space in the log directory
      if (statvfs(run_dir.c_str(), &vf) == 0) {
        double mb = (double)vf.f_bavail * (double)vf.f_frsize / 1048576.0;
        char b[160];
        snprintf(b, sizeof b, "%.0f MB free in %s (min %.0f MB; this run ~%.0f MB at ~0.9 MB/s)", mb, run_dir.c_str(), site.log_min_free_mb, 0.9 * (run_s + lead_in_frames / motion.fps + end_hold_s));
        items.push_back({"log_disk_space", mb >= site.log_min_free_mb ? L_PASS : L_FAIL, b});
      }
      items.push_back({"run_log_open", rlog.is_open() ? L_PASS : L_FAIL, rlog.is_open() ? "events / trace / obs / state_stream / frames / clips files open (async writer)" : "run log could not be opened: the run could not be recorded"});
      // manifest.json: program / policy / config identity + everything an offline tool needs without the assets
      std::string pj;
      {
        auto num = [&](const char* k, const Vec& v) { return std::string("\"") + k + "\":" + vec_json(v) + ","; };
        pj += num("default_pos", params.default_pos) + num("action_scale", params.action_scale) + num("kp", params.kp) + num("kd", params.kd) + num("pos_lo", params.pos_lo) + num("pos_hi", params.pos_hi) +
              num("vel_lim", params.vel_lim) + num("effort_lim", params.effort_lim);
        std::string vk = "[";
        for (int i = 0; i < N; i++) vk += std::string(i ? "," : "") + (params.vel_known[i] ? "1" : "0");
        pj += "\"vel_known\":" + vk + "],";
        pj += "\"joint_names\":" + arr_json(params.names) + ",";
      }
      char sb[2400];
      snprintf(sb, sizeof sb,
               "\"site\":{\"pos_margin_rad\":%.6g,\"target_speed_frac\":%.6g,\"torque_cmd_frac\":%.6g,\"state_stale_s\":%.6g,\"limit_fault_tol_rad\":%.6g,\"speed_fault_frac\":%.6g,\"torque_meas_fault_frac\":%.6g,\"torque_meas_fault_ticks\":%d,"
               "\"max_tilt_rad\":%.6g,\"max_gyro_rad_s\":%.6g,\"diverge_rad\":%.6g,\"loop_overrun_factor\":%.6g,\"unstable_tilt_rad\":%.6g,\"unstable_soft_tilt_rad\":%.6g,\"unstable_gyro_rad_s\":%.6g,\"unstable_tilt_rate_rad_s\":%.6g,"
               "\"fallen_tilt_rad\":%.6g,\"watchdog_s\":%.6g,\"max_entry_tilt_rad\":%.6g,\"exit_dwell_s\":%.6g,\"exit_stable_window_s\":%.6g,\"exit_confirm_timeout_s\":%.6g,\"log_preroll_s\":%.6g,\"log_post_exit_s\":%.6g}",
               gcfg.pos_margin_rad, gcfg.target_speed_frac, gcfg.torque_cmd_frac, gcfg.state_stale_s, gcfg.limit_fault_tol_rad, gcfg.speed_fault_frac, gcfg.torque_meas_fault_frac, gcfg.torque_meas_fault_ticks, gcfg.max_tilt_rad,
               gcfg.max_gyro_rad_s, gcfg.diverge_rad, gcfg.loop_overrun_factor, site.exit.unstable_tilt_rad, site.exit.unstable_soft_tilt_rad, site.exit.unstable_gyro_rad_s, site.exit.unstable_tilt_rate_rad_s, site.exit.fallen_tilt_rad,
               site.watchdog_s, site.max_entry_tilt_rad, site.exit.dwell_s, site.exit.stable_window_s, site.exit.confirm_timeout_s, site.log_preroll_s, site.log_post_exit_s);
      std::string args;
      for (int i = 0; i < argc; i++) args += (i ? " " : "") + std::string(argv[i]);
      std::string lay;
      for (auto& kv : ps->obs_layout()) lay += (lay.empty() ? "" : ",") + std::string("{\"name\":\"") + kv.first + "\",\"dim\":" + std::to_string(kv.second) + "}";
      manifest_extra = jstr("mode", mode_name.empty() ? "custom" : mode_name) + "," + jstr("target", sim ? "sim" : "real") + "," + jstr("iface", iface) + "," + jstr("argv", args) + "," + jnum("run_s", run_s) + "," + jnum("authority", amax) + "," +
                       jstr("ref_mode", ref_mode) + "," + jnum("ramp_s", ramp_s) + "," + jnum("end_hold_s", end_hold_s) + "," + jnum("control_dt_s", DT) + "," + jbool("exceeds_legacy_caps", exceeds_legacy) + "," +
                       jnum("t_mono_runner_start", t_start_mono) + "," + jnum("wall_minus_mono", wall() - mono()) + "," + jnum("obs_dim", ps->obs_dim()) + ",\"obs_layout\":[" + lay + "]," +
                       jnum("motion_frames", motion.T) + "," + jnum("lead_in_frames", lead_in_frames) + "," + jnum("planned_total_s", tl.planned_total_s) + "," + jnum("failsafe_s", tl.failsafe_s) + "," + jbool("covers_whole_dance", tl.covers_whole_dance) + "," + jbool("standby_runner", standby) + "," + jbool("guard_torque_ff", gcfg.torque_ff) + "," + jnum("motion_fps", motion.fps) + "," + jnum("history_steps", params.history) + "," + jstr("onnx_runtime", "linked") + "," + pj + sb;
      write_atomic(run_dir + "/manifest.json", manifest_json(P, hashes, manifest_extra) + "\n");
      rlog.event("manifest", "\"file\":\"manifest.json\",\"policy_sha256\":\"" + hashes.policy + "\",\"exe_sha256\":\"" + hashes.exe + "\",\"program_version\":\"" + PM_VERSION + "\"");
    }
    {
      char b[400];
      snprintf(b, sizeof b, "run %.2f s (cap %.2f), authority %.2f (cap %.2f); legacy caps %.1f s / %.1f; compile-time limits %.0f s / %.1f%s", run_s, site.max_run_s, amax, site.max_authority, LEGACY_MAX_RUN_S,
               LEGACY_MAX_AUTHORITY, ABS_MAX_RUN_S, ABS_MAX_AUTHORITY, exceeds_legacy ? "  *** EXCEEDS THE LEGACY CAPS (1.0 s / 0.3): offline validation required ***" : "");
      items.push_back({"run_caps", exceeds_legacy ? L_WARN : L_PASS, sim ? std::string("sim target; ") + b : std::string(b)});
    }
    items.push_back(guardian_item());
    WriterScan scan;
    std::thread scan_thr([&] { scan_writers(1.0, scan); });  // idle-vs-active DDS writer classification, in parallel with the other read-only checks
    std::vector<PreflightItem> pf;
    sess.preflight(pf, -1, 1.0, true);  // retro: on a warm (standby) Link the 1 s windows are the LAST second of the rolling history (no waiting); a cold Link falls back to forward windows
    scan_thr.join();
    for (auto& r : pf) items.push_back({"pf_" + r.name, r.level, r.detail});
    if (standby) {  // the frozen-artifact verdict at the trigger: a fresh FULL sha256 (run in parallel with the checks above) must equal the start-up hashes and the frozen values
      ArtifactHashes hf = hash_fut.get();
      const bool same = hf.policy == hashes_pre.policy && hf.params == hashes_pre.params && hf.motion == hashes_pre.motion && hf.fk_chain == hashes_pre.fk_chain && hf.site_conf == hashes_pre.site_conf && hf.modes_conf == hashes_pre.modes_conf && hf.exe == hashes_pre.exe;
      const bool ok = same && hf.policy_ok && hf.params_ok && hf.motion_ok;
      items.push_back({"artifact_rehash_at_trigger", ok ? L_PASS : L_FAIL, same ? (ok ? "full sha256 of policy/params/motion/config/executable re-verified at the trigger == start-up hashes == frozen values" : "fresh hashes differ from the frozen values")
                                                                           : "ARTIFACT CHANGED since service start-up (policy/params/motion/config/executable differ from the start-up hashes)"});
      if (!same) hashes = hf;
    }
    {
      double warm_last = warm_ms.empty() ? 0.0 : warm_ms.back();
      items.push_back({"policy_inference_time", warm_last <= 15.0 ? L_PASS : L_WARN, "last warm-up inference " + std::to_string(warm_last) + " ms (control period 20 ms; slow = WARN only, a >60 ms control step still damps in RUN)"});
    }
    auto cf = link.get_fsm();
    ready_fsm = cf.second;
    items.push_back(writer_item(scan, cf.second));
    for (auto& kv : scan.w) {
      writers_dump += "\"" + json_escape(kv.first) + "\":{\"writers\":[";
      for (size_t i = 0; i < kv.second.size(); i++) writers_dump += (i ? ",\"" : "\"") + json_escape(kv.second[i]) + "\"";
      auto it = scan.rx.find(kv.first);
      writers_dump += "],\"rx_samples\":" + std::to_string(it == scan.rx.end() ? 0 : (unsigned long long)it->second) + "},";
    }
    sess.set_expected_entry_fsm(ready_fsm);  // acquire() refuses if the factory fsm changed between READY and the take-over
    auto st_ready = sess.state();
    ready_state_json = "{" + jbool("exceeds_legacy_caps", exceeds_legacy) + "," + jnum("run_s", run_s) + "," + jnum("authority", amax) + "," + jnum("max_run_s", site.max_run_s) + "," + jnum("max_authority", site.max_authority) + "," + jnum("legacy_max_run_s", LEGACY_MAX_RUN_S) + "," + jnum("legacy_max_authority", LEGACY_MAX_AUTHORITY) + "," + jnum("fsm", cf.second) + "," + jnum("rpc_code", cf.first) + "," + jnum("traffic_window_s", scan.window) + ",\"writers\":{" + (writers_dump.empty() ? "" : writers_dump.substr(0, writers_dump.size() - 1)) + "},\"admission\":" + items_json(items) +
                       ",\"state\":" + (st_ready ? "{\"q\":" + vec_json(st_ready->q) + ",\"dq\":" + vec_json(st_ready->dq) + ",\"tau_est\":" + vec_json(st_ready->tau) + "," + jnum("mode_machine", st_ready->mode_machine) + "}" : "null") + "}";
    write_atomic(run_dir + "/ready.json", ready_state_json);
    write_atomic(run_dir + "/admission.json", items_json(items));
    bool all = true;
    for (auto& i : items) { logf("[%s] %s: %s", level_name(i.level), i.name.c_str(), i.detail.c_str()); all = all && i.pass(); }
    if (!all) {
      std::string bad;
      for (auto& i : items) if (!i.pass()) bad += i.name + "; ";
      logf("BLOCKED: real-time admission not met (nothing was sent): %s", bad.c_str());
      result = "BLOCKED"; exit_code = 6; throw 0;
    }
    admitted = true;
    if (check_only) {  // P5.2: read-only check done. This process never reaches ARM: no lock, no command writer, no RPC that moves anything.
      logf("CHECK ONLY: real-time admission passed (nothing was sent, control is never taken in this mode).");
      ev_write("check_only_passed", jstr("note", "read-only check; no motion authorisation needed or used"));
      status("READY");
      sleep_s(1.5);  // the service announces 准备就绪 from this phase
      result = "CHECK_PASSED"; exit_reason = "check only; admission passed; robot untouched"; throw 0;
    }
    // ---- READY / GO
    if (!wait_go.empty()) {
      logf("READY: model loaded, checks passed. Waiting for GO (L1+A) / abort (START x2, SIGTERM).");
      status("READY");
      while (!file_exists(wait_go)) {
        if (g_quit) { logf("ABORTED before ARM (signal): control was never taken."); result = "ABORTED_BEFORE_ARM"; exit_reason = "signal before ARM"; throw 0; }
        if (sess.remote_fired()) { logf("ABORTED before ARM by remote START double-click: control was never taken."); result = "ABORTED_BEFORE_ARM"; exit_reason = "START x2 before ARM"; throw 0; }
        sleep_s(0.02);
      }
      // ---- V1.6 announce_wait: GO = "keys released, the announcement starts". The robot is NOT taken over until the announcement has FINISHED (service: ANNOUNCED file) + announce_wait_s,
      // and every take-over precondition is re-verified at the end. Nothing is sent in this phase: every problem CANCELS (result CANCELLED_BEFORE_TAKEOVER, rc 7).
      std::string gdoc;
      read_file(wait_go, gdoc);
      ::unlink(wait_go.c_str());
      const double t_go = mono();
      const bool svc_announces = !announced_file.empty() && json_find_raw(gdoc, "announce") == "1";
      double ann_max = 0;
      { std::string v = json_find_raw(gdoc, "announce_max_s"); ann_max = v.empty() ? site.announce_max_s : atof(v.c_str()); }
      if (!(ann_max >= 1 && ann_max <= 10)) ann_max = site.announce_max_s;
      const double ann_cap_s = ann_max + 1.5;  // the service's own cap (announce_max_s) + margin; beyond it the service is gone / stuck -> cancel
      goto_phase("GO_WAIT");
      ev_write("announce_begin", jnum("t_mono", t_go) + "," + jbool("service_announces", svc_announces) + "," + jnum("announce_max_s", ann_max) + "," + jnum("announce_wait_s", announce_wait) + "," + jstr("go_file", gdoc.substr(0, gdoc.find('\n'))));
      struct WaitFlags { DampCombo l2b; std::atomic<bool> hit{false}, active{false}; WaitFlags(double h) : l2b(h) {} };
      auto wf = std::make_shared<WaitFlags>(site.exit.l2b_hold_s);
      link.add_state_listener([wf](const LinkState& st) {  // remote L2+B (factory damping chord) during the wait: DampCombo semantics of the run (a press that started while active; debounced)
        if (wf->l2b.feed(st.arrival, keys_from_wireless(st.wireless), wf->active.load())) wf->hit.store(true);
      });
      wf->active = true;
      const uint64_t user_cmd0 = link.cmd_rx_count();
      double t_fsm_poll = 0, t_g_poll = 0;
      int lock_held_n = 0;
      std::string cancel_code, cancel_why, cancel_voice_tmp;
      // one pass of every take-over precondition; the FSM query (RPC) and the guardian heartbeat file are polled at a lower rate. Returns false and fills cancel_* on the first violation.
      auto wait_check = [&](bool force_slow) -> bool {
        auto fail = [&](const char* code, const std::string& why, const char* voice) { cancel_code = code; cancel_why = why; cancel_voice_tmp = voice; return false; };
        if (g_quit) return fail("signal", "SIGTERM/SIGINT (service stop) while waiting to take over", "已退出");
        if (sess.remote_fired()) return fail("start_double_click", "remote START double-click while waiting to take over (operator cancel)", "已退出");
        if (wf->hit.load()) return fail("remote_l2b", "remote L2+B (emergency chord) while waiting to take over: cancelled, nothing was sent (the factory controller reacts to L2+B itself)", "已退出");
        if (file_exists(estop_file)) return fail("estop_file", "e-stop latch file " + estop_file + " appeared while waiting to take over: cancelled, nothing was sent; the latch stays set until ./ack.sh", "已退出");
        auto st = sess.state();
        const double age = st ? mono() - st->arrival : 1e9;
        if (!st || age > 0.25) return fail("lowstate_timeout", "no fresh rt/lowstate (age " + std::to_string(age) + " s > 0.25 s) while waiting to take over", "检查失败");
        const double tilt = tilt_of_quat(st->quat), gxy = std::hypot(st->gyro[0], st->gyro[1]), dqm = st->dq.cwiseAbs().maxCoeff();
        const double entry_tilt = std::min(site.max_entry_tilt_rad, site.exit.unstable_tilt_rad);
        if (!std::isfinite(tilt) || tilt >= entry_tilt) return fail("tilt", "attitude tilt " + std::to_string(tilt) + " rad >= entry limit " + std::to_string(entry_tilt) + " while waiting to take over", "检查失败");
        if (gxy > 1.5 || dqm >= 4 * site.quiescent_dq) return fail("not_quiet", "robot not still while waiting to take over (|gyro_xy| " + std::to_string(gxy) + " rad/s, max|dq| " + std::to_string(dqm) + " rad/s)", "检查失败");
        if (link.cmd_rx_count() != user_cmd0) return fail("competing_writer", "another process started writing rt/user_lowcmd while waiting to take over", "检查失败");
        // the lock probe (flock EX|NB) collides with the guardian's / service's own probes for microseconds: only a lock held over many consecutive passes counts
        lock_held_n = lock_probe(lock_path).held ? lock_held_n + 1 : 0;
        if (lock_held_n >= 8) return fail("lock_held", "the exclusive control lock was taken by another process while waiting", "检查失败");
        const double now_m = mono();
        if (force_slow || now_m - t_g_poll > 0.1) {
          t_g_poll = now_m;
          Item g = guardian_item();
          if (!g.pass()) return fail("guardian", "guardian lost while waiting to take over: " + g.detail, "检查失败");
        }
        if (force_slow || now_m - t_fsm_poll > 0.25) {
          t_fsm_poll = now_m;
          auto cf = link.get_fsm();
          if (cf.first != 0 || cf.second < 0) return fail("fsm_query", "GetFsmId failed (code " + std::to_string(cf.first) + ") while waiting to take over", "检查失败");
          if (cf.second != ready_fsm) return fail("fsm_changed", "FSM changed from " + std::to_string(ready_fsm) + " to " + std::to_string(cf.second) + " while waiting to take over (the factory mode / remote reacted): not taking control", "检查失败");
          std::string why;
          if (!entry_check(cf.second, allowed, why)) return fail("fsm_not_allowed", "FSM no longer in the allowed WalkRun set: " + why, "检查失败");
        }
        return true;
      };
      auto cancel_now = [&]() {
        wf->active = false;
        const double t_c = mono();
        logf("CANCELLED before take-over (%s): %s  [%.2f s after GO; control was never taken, nothing was sent]", cancel_code.c_str(), cancel_why.c_str(), t_c - t_go);
        ev_write("cancel", jstr("reason", cancel_code) + "," + jstr("detail", cancel_why) + "," + jnum("t_mono", t_c) + "," + jnum("waited_since_go_s", t_c - t_go) + "," + jstr("phase", phase) + "," + jbool("took_control", false));
        result = "CANCELLED_BEFORE_TAKEOVER"; exit_reason = "cancel(" + cancel_code + "): " + cancel_why; exit_code = 7; cancel_voice = cancel_voice_tmp;
        throw 0;
      };
      // phase 1: wait for "announcement finished" (service writes ANNOUNCED; it is bounded by announce_max_s there, ann_cap_s here)
      if (svc_announces) {
        while (!file_exists(announced_file)) {
          if (!wait_check(false)) cancel_now();
          if (mono() - t_go > ann_cap_s) { cancel_code = "announce_timeout"; cancel_why = "no 'announcement finished' from the service within " + std::to_string(ann_cap_s) + " s (service gone or stuck)"; cancel_voice_tmp = "检查失败"; cancel_now(); }
          sleep_s(0.01);
        }
        std::string adoc;
        read_file(announced_file, adoc);
        ::unlink(announced_file.c_str());
        ev_write("announce_done", jnum("t_mono", mono()) + "," + jnum("since_go_s", mono() - t_go) + "," + jstr("via", json_find_string(adoc, "via")) + "," + jnum("service_announce_s", atof(json_find_raw(adoc, "announce_s").c_str())) + "," + jnum("rpc_rc", atof(json_find_raw(adoc, "rpc_rc").c_str())));
      } else ev_write("announce_done", jnum("t_mono", mono()) + "," + jnum("since_go_s", mono() - t_go) + "," + jstr("via", "no_service_announcement"));
      // phase 2: announce_wait_s after the announcement finished. The OEM-writer scan of the final re-check (0.5 s window) runs INSIDE the last 0.5 s of the wait.
      const double t_w0 = mono();
      ev_write("wait_begin", jnum("t_mono", t_w0) + "," + jnum("announce_wait_s", announce_wait));
      WriterScan sc2;
      std::thread scan2;
      bool scan2_started = false;
      auto start_scan2 = [&]() { if (!scan2_started) { scan2_started = true; scan2 = std::thread([&] { scan_writers(0.5, sc2); }); } };
      while (mono() - t_w0 < announce_wait) {
        if (!wait_check(false)) { if (scan2_started) scan2.join(); cancel_now(); }
        if (announce_wait - (mono() - t_w0) <= 0.5) start_scan2();
        sleep_s(0.01);
      }
      start_scan2();
      scan2.join();
      ev_write("wait_end", jnum("t_mono", mono()) + "," + jnum("waited_s", mono() - t_w0) + "," + jnum("since_go_s", mono() - t_go));
      if (!wait_check(true)) cancel_now();  // the last word before the take-over: every precondition, FSM and guardian freshly queried
      // state may have changed while waiting: the state-dependent checks repeat (guardian, OEM writers); acquire() re-runs the fast preflight itself
      Item g_item = guardian_item();
      auto cf2 = link.get_fsm();
      Item w_item = writer_item(sc2, cf2.second);
      Item f_item = {"fsm_unchanged_since_ready", (cf2.first == 0 && cf2.second == ready_fsm) ? L_PASS : L_FAIL,
                     "fsm " + std::to_string(ready_fsm) + " at READY, " + std::to_string(cf2.second) + " now (a change means the factory remote reacted to the chord or the mode was changed)"};
      ev_write("go_recheck", "\"items\":" + items_json({g_item, w_item, f_item}));
      if (!(g_item.pass() && w_item.pass() && f_item.pass())) {
        for (const Item* it : {&g_item, &w_item, &f_item}) if (!it->pass()) logf("GO REFUSED: %s: %s", it->name.c_str(), it->detail.c_str());
        result = "GO_REFUSED"; exit_code = 6; go_refused = true; throw 0;
      }
      wf->active = false;
      logf("GO received, announcement finished + %.2f s wait done, preconditions re-verified -> ARM", announce_wait);
    }
    // ---- ARM
    goto_phase("ARM");
    {
      auto sp = sess.state();
      if (sp) { pre_state_json = state_snapshot_json(*sp); ev_write("pre_takeover_state", "\"state\":" + pre_state_json); }
    }
    rlog.begin_stream();  // from here the full-rate lowstate stream (incl. the 5 s pre-roll) goes to state_stream.bin
    if (!sess.acquire(true)) {
      if (guard.fault() == Fault::REMOTE_STOP && sess.mode() == ControlSession::OFF) {
        logf("ABORTED before control by START x2 during ARM");
        result = "ABORTED_BEFORE_ARM"; exit_reason = "START x2 during ARM"; throw 0;
      }
      logf("ARM FAILED: fault=%s: %s", fault_name(guard.fault()), guard.events().empty() ? "" : guard.events().back().detail.c_str());
      result = "ARM_FAILED"; exit_code = 4; throw 0;
    }
    auto st = sess.state();
    Vec q_hold = st->q;
    goto_phase("RUN");
    take_state_json = state_snapshot_json(*st);
    ps->reset(to_sample(*sess.state()), ref_mode == "paused");
    {
      PolicyResetInfo ri = ps->reset_info();
      char b[900];
      snprintf(b, sizeof b, "\"yaw_offset_wxyz\":[%.17g,%.17g,%.17g,%.17g],\"motion_t\":%d,\"paused\":%s,\"quat_wxyz\":[%.9g,%.9g,%.9g,%.9g],\"gyro\":[%.9g,%.9g,%.9g]", ri.yaw_offset[0], ri.yaw_offset[1], ri.yaw_offset[2], ri.yaw_offset[3],
               ri.motion_t, ri.paused ? "true" : "false", ri.sample.quat[0], ri.sample.quat[1], ri.sample.quat[2], ri.sample.quat[3], ri.sample.gyro[0], ri.sample.gyro[1], ri.sample.gyro[2]);
      std::string qs = "[", dqs = "[";
      for (int i = 0; i < N; i++) { char c[48]; snprintf(c, sizeof c, "%s%.9g", i ? "," : "", ri.sample.q[i]); qs += c; snprintf(c, sizeof c, "%s%.9g", i ? "," : "", ri.sample.dq[i]); dqs += c; }
      ev_write("policy_reset", std::string(b) + ",\"q\":" + qs + "],\"dq\":" + dqs + "],\"q_hold\":" + vec_json(q_hold) + ",\"take_state\":" + take_state_json);
    }
    {
      auto dg = ps->state_digest();
      const bool clean = dg.motion_t == 0 && dg.prev_action_abs_max == 0.0 && dg.history_identical && dg.quat_history_identical;
      ev_write("policy_state_reset", jnum("motion_t", dg.motion_t) + "," + jbool("paused", dg.paused) + "," + jnum("prev_action_abs_max", dg.prev_action_abs_max) + "," + jbool("proprio_history_identical", dg.history_identical) + "," +
                                     jbool("quat_history_identical", dg.quat_history_identical) + "," + jbool("clean", clean) + "," + jnum("lead_in_frames", lead_in_frames) + "," + jnum("clip_frames", motion_run.T) + "," + jstr("note", "fresh process: nothing of a previous run survives (history, previous action, reference index)"));
      if (!clean) { logf("REFUSED: policy state after reset is not clean"); sess.emergency(Fault::INFERENCE, "policy state not clean after reset"); }
      ev_write("tilt_judge_config", "\"lead_in_frames\":" + std::to_string(lead_in_frames) + ",\"ref_mode\":\"" + ref_mode + "\",\"judge\":" + sess.tilt_judge_json());  // V1.6 A1: which instability limits this run used
    }
    const double t_take = mono();
    double next_t = mono();
    int div_count = 0;
    double t_window_end = -1;
    bool first_cmd = true;
    std::unique_ptr<TickRec> rec(new TickRec());
    double prev_tstep = 0;
    uint64_t prev_seq = st->seq, prev_lost = 0;
    uint32_t prev_ls_tick = st->tick;
    for (;;) {
      double now = mono();
      if (now < next_t) sleep_s(next_t - now);
      next_t += DT;
      auto s0 = sess.state();
      uint64_t seq0 = s0 ? s0->seq : 0;
      const double t_sw0 = mono();
      double t_wait = t_sw0 + 0.008;  // sample-synchronous tick: shave the state age
      bool got_new = false;
      while (mono() < t_wait) {
        auto s1 = sess.state();
        if (!(s1 && s1->seq == seq0)) { got_new = true; break; }
        sleep_s(0.0003);
      }
      double tstep = mono();
      sess.heartbeat();
      if (guard.tripped() || g_quit || sess.exit_triggered()) break;
      auto stc = sess.state();
      if (sess.check_state_once(stc) != Fault::NONE) break;
      if (sess.has_last_cmd()) {
        double dv = (stc->q - sess.last_cmd_q()).cwiseAbs().maxCoeff();
        div_count = dv > diverge_rad ? div_count + 1 : 0;
        if (div_count >= gcfg.diverge_ticks) { sess.emergency(Fault::NO_RESPONSE, "|q - q_cmd| " + std::to_string(dv) + " rad > " + std::to_string(diverge_rad) + " for 8 ticks (command not followed)"); break; }
      }
      double el = mono() - t_take;
      sess.set_tick(n_ticks, el);
      PolicyOut po;
      try {
        if (sim && inject_tick >= 0 && n_ticks == inject_tick && inject_kind == "inference") throw std::runtime_error("injected inference exception (sim)");
        po = ps->tick(to_sample(*stc));
      } catch (const std::exception& e) {
        sess.emergency(Fault::INFERENCE, e.what());
        break;
      }
      sess.set_reference(ref_mode == "running" && po.motion_t > lead_in_frames && t_window_end < 0, po.ref_rq, (uint64_t)n_ticks);  // V1.6 A1: reference pelvis attitude for the stability judge (valid = dance running, not lead-in / hold)
      if (sim && inject_tick >= 0 && n_ticks == inject_tick) {  // DEV ONLY (--sim, env PM_SIM_INJECT): exercise the fault exits on the loopback robot
        if (inject_kind == "nan") po.q_target[3] = NAN;
        else if (inject_kind == "overrun") sleep_s(0.075);
        else if (inject_kind == "hang") sleep_s(0.3);
        logf("SIM INJECT %s at tick %d", inject_kind.c_str(), n_ticks);
      }
      if (sim && inject_tick >= 0 && n_ticks == inject_tick && inject_kind == "segv") { logf("SIM INJECT segv at tick %d", n_ticks); raise(SIGSEGV); }
      if (sim && inject_kind == "clip" && n_ticks >= inject_tick && n_ticks < inject_tick + 10) {  // DEV ONLY: a target far outside the joint limit for 10 ticks (safety-clip logging test)
        const int j = getenv("PM_SIM_INJECT_JOINT") ? atoi(getenv("PM_SIM_INJECT_JOINT")) : 4;
        po.q_target[j] = params.pos_hi[j] + 1.0;
        if (n_ticks == inject_tick) logf("SIM INJECT clip: joint %d target = hi + 1.0 rad for 10 ticks", j);
      }
      double authority = ramp_s > 0 ? amax * smoothstep(el / ramp_s) : amax;
      Vec q_cmd = q_hold + authority * (po.q_target - q_hold);
      // ---- P5.1 tick record: filled BEFORE the command goes out so that a rejected command is still on record
      TickRec& R = *rec;
      R.n = (uint32_t)n_ticks; R.t_mono = tstep; R.t_rel = el; R.phase = PH_RUN; R.ssb_frame = po.motion_t; R.authority = authority;
      R.dt_ms = prev_tstep > 0 ? (float)((tstep - prev_tstep) * 1e3) : (float)(DT * 1e3);
      R.sync_wait_ms = (float)((tstep - t_sw0) * 1e3);
      R.seq = stc->seq; R.seq_delta = (uint32_t)(stc->seq - prev_seq); R.ls_tick = stc->tick; R.mode_machine = stc->mode_machine; R.arrival = stc->arrival;
      R.state_age_ms = (float)(1e3 * (tstep - stc->arrival));
      memcpy(R.wireless, stc->wireless.data(), 40);
      R.keys = keys_from_wireless(stc->wireless);
      for (int i = 0; i < 4; i++) R.quat[i] = (float)stc->quat[i];
      for (int i = 0; i < 3; i++) { R.gyro[i] = (float)stc->gyro[i]; R.rpy[i] = (float)stc->rpy[i]; }
      { double rr, pp; roll_pitch(stc->quat, rr, pp); R.roll = (float)rr; R.pitch = (float)pp; R.tilt = (float)tilt_of_quat(stc->quat); }
      for (int i = 0; i < N; i++) {
        R.q[i] = (float)stc->q[i]; R.dq[i] = (float)stc->dq[i]; R.tau[i] = (float)stc->tau[i];
        R.action[i] = po.action[i]; R.action_prev[i] = po.action_prev[i]; R.q_policy[i] = po.q_target[i]; R.q_pd[i] = q_cmd[i];
        R.ref_jp[i] = po.ref_jp[i]; R.ref_jv[i] = po.ref_jv[i];
      }
      for (int i = 0; i < 3; i++) R.ref_rp[i] = po.ref_rp[i];
      for (int i = 0; i < 4; i++) R.ref_rq[i] = po.ref_rq[i];
      R.ref_frame = po.ref_idx; R.infer_ms = (float)po.infer_ms;
      // V1.6: dance time axis. ssb_frame counts the replayed clip = lead_in_frames frozen first frames + the dance; dance_frame = ssb_frame - lead_in_frames (<0 in the lead-in); seg: 0 lead-in, 1 dance, 2 end-hold, 3 paused reference
      R.dance_frame = po.motion_t - lead_in_frames; R.dance_t = R.dance_frame / (motion.fps > 0 ? motion.fps : 50.0);
      R.seg = ref_mode == "paused" ? 3 : (t_window_end >= 0 ? 2 : (po.motion_t < lead_in_frames ? 0 : 1));
      ps->copy_obs(R.obs); R.obs_n = ps->obs_dim();
      uint32_t flags = 0;
      if (prev_tstep > 0 && R.dt_ms > 1.5f * (float)(DT * 1e3)) flags |= TF_LATE;
      if (R.seq_delta == 0) flags |= TF_REPEAT_STATE;
      if (!got_new) flags |= TF_SYNC_TIMEOUT;
      if (po.infer_ms > 15.0) flags |= TF_SLOW_INFER;
      const bool wok = sess.write(q_cmd, kp, kd, DT);
      if (wok && first_cmd) { first_cmd = false; first_cmd_step = (sess.last_cmd_q() - stc->q).cwiseAbs().maxCoeff(); first_cmd_auth = authority; }
      if (wok) {
        const CmdOut& co = sess.last_out();
        for (int i = 0; i < N; i++) {
          R.q_cmd[i] = (float)co.q[i]; R.kp[i] = (float)co.kp[i]; R.kd[i] = (float)co.kd[i];
          R.tau_pd[i] = (float)(co.kp[i] * ((double)R.q_cmd[i] - stc->q[i]) - co.kd[i] * stc->dq[i]);
          R.tau_ff[i] = (float)co.tau[i]; R.tau_train[i] = (float)co.tau_train[i];
        }
        R.clip_n = sess.last_clip_n(); R.clip_mask = sess.last_clip_mask();
        if (R.clip_n) flags |= TF_CLIP;
      } else {  // rejected / not sent: q_cmd = NaN marks "no command went out this tick"
        for (int i = 0; i < N; i++) { R.q_cmd[i] = NAN; R.kp[i] = 0; R.kd[i] = 0; R.tau_pd[i] = NAN; }
        R.clip_n = 0; R.clip_mask = 0;
      }
      R.send_us = (float)link.last_send_us();
      R.dds_loop_ms = (float)link.last_loopback_ms();
      {
        auto ms = rlog.lowstate().snap();
        R.ls_rx = ms.rx; R.ls_lost = ms.lost; R.ls_gap_ms = (float)ms.last_gap_ms;
        if (ms.lost > prev_lost) flags |= TF_TICK_GAP;
        prev_lost = ms.lost;
      }
      R.tick_delta = stc->tick - prev_ls_tick;
      R.flags = flags;
      double step_ms = 1e3 * (mono() - tstep);
      R.step_ms = (float)step_ms;
      R.state_age_send_ms = (float)(1e3 * (mono() - stc->arrival));
      rlog.tick(R);
      prev_tstep = tstep; prev_seq = stc->seq; prev_ls_tick = stc->tick;
      if (!wok) break;
      n_ticks++;
      if (step_ms > gcfg.loop_overrun_factor * 1e3 * DT) { sess.emergency(Fault::LOOP_OVERRUN, "control step took " + std::to_string(step_ms) + " ms"); break; }
      // run window over -> (reference stays paused for end_hold_s so the robot settles) -> EXIT hand-back
      const bool window_over = (ref_mode == "paused" && el >= run_s) || (ref_mode == "running" && po.motion_t >= seg_frames);
      if (window_over && t_window_end < 0) { t_window_end = el; if (ref_mode == "running") ps->hold_reference(); }
      if ((t_window_end >= 0 && el - t_window_end >= (ref_mode == "paused" ? 0.0 : end_hold_s)) || el > tl.failsafe_s) break;
    }
    run_end = guard.tripped() ? "FAULT" : (g_quit ? "OPERATOR_QUIT" : "COMPLETED");
  } catch (int) {
    // controlled early exits set result/exit_code
  } catch (const std::exception& e) {
    logf("UNHANDLED %s", e.what());
    sess.emergency(Fault::INFERENCE, std::string("unhandled ") + e.what());
    result = "CRASH"; exit_code = 1; exit_reason = std::string("crash: ") + e.what();
  }

  if (run_dir.empty()) {  // V1.5.1: a standby runner that ended BEFORE any trigger (signal, no lowstate, refusal, exception): no run exists, control was never touched; the service restarts it
    logf("standby runner ended before any trigger (result=%s rc=%d): nothing was sent, control never taken", result.empty() ? "-" : result.c_str(), exit_code ? exit_code : (result == "CRASH" ? 1 : 0));
    fast_exit(exit_code ? exit_code : (result == "CRASH" ? 1 : 0));
  }
  // ---- EXIT (always: releases control if it was ever taken). The exit itself was normally started the instant the trigger fired (session.trigger_exit).
  if (phase != "EXIT") goto_phase("EXIT");
  ExitReport er = sess.finish(run_end.empty() ? (g_quit ? "OPERATOR_QUIT" : (result.empty() ? "FAULT" : result)) : run_end);
  const ExitOutcome& eo = er.out;
  const bool took = er.took_control;
  if (took) {
    if (result.empty() || result == "FAULT" || result == "OPERATOR_QUIT" || result == "COMPLETED") result = eo.verdict.result;  // early-failure results (ARM_FAILED, CRASH, ...) keep their name
    if (exit_code == 0) exit_code = eo.verdict.exit_code;
    else if (!eo.settled) exit_code = 5;
  }
  if (exit_reason.empty()) {
    if (guard.tripped()) exit_reason = std::string(fault_name(guard.fault())) + ": " + guard.events().front().detail;
    else if (run_end == "OPERATOR_QUIT") exit_reason = "operator quit (signal)";
    else exit_reason = result;
  }
  std::string fev = "[";
  for (size_t i = 0; i < guard.events().size(); i++) {
    auto& e = guard.events()[i];
    fev += (i ? "," : "") + std::string("{") + jnum("t", e.t) + "," + jstr("fault", fault_name(e.fault)) + "," + jstr("detail", e.detail) + "}";
  }
  fev += "]";
  const int final_fsm = took ? eo.final_fsm : -1;
  const bool restored_entry = sess.entry_fsm() >= 0 && final_fsm == sess.entry_fsm();
  const std::string final_mode = took ? eo.final_mode : "NONE", requested_mode = took ? eo.requested_mode : "NONE";
  const bool recovery_ok = took ? eo.verdict.recovery_ok : false;
  std::string rep = "{\"summary\":{" + jstr("result", result) + "," + jstr("exit_reason", exit_reason) + "," + jstr("fault", fault_name(guard.fault())) + "," + jbool("exit_confirmed", took && eo.confirmed) + "," +
                    jstr("exit_mode", requested_mode) + "," + jstr("requested_mode", requested_mode) + "," + jstr("final_mode", final_mode) + "," + jnum("final_fsm", final_fsm) + "," + jbool("recovery_ok", recovery_ok) + "," +
                    jnum("entry_fsm", sess.entry_fsm()) + "," + jnum("restored_fsm", final_fsm) + "," + jbool("restored_to_entry_fsm", restored_entry) + "," + jbool("took_control", took) + "," +
                    jstr("exit_class", took ? er.req.decision.cls : "NONE") + "," + jstr("voice", took ? (result == eo.verdict.result ? eo.verdict.voice : std::string("检查失败")) : cancel_voice) + "," + jnum("exit_code", exit_code) + "},";
  rep += jstr("result", result) + "," + jstr("exit_reason", exit_reason) + "," + jnum("exit_code_planned", exit_code) + "," + jnum("user_fsm", sess.user_fsm_seen()) + "," + jstr("run_end", run_end) + ",";
  rep += "\"exit\":{" + jstr("reason", er.reason) + "," + jstr("internal_mode", requested_mode) + "," + jstr("exit_mode_why", er.exit_mode_why) + "," + jnum("rpc_code", er.rpc_code) + "," + jnum("fsm_after", final_fsm) + "," +
         jbool("confirmed", took && eo.confirmed) + "," + jstr("confirmed_via", er.confirmed_via) + "," + jbool("mode_matched", took && eo.confirmed) + "," + jstr("note", er.note) + "},";
  rep += std::string("\"exit_detail\":") + (took ? exit_outcome_json(er.req, eo) : "null") + ",";
  rep += jbool("exceeds_legacy_caps", exceeds_legacy) + "," + jnum("run_s", run_s) + "," + jnum("authority", amax) + ",";
  rep += jstr("schema", PM_LOG_SCHEMA) + "," + jstr("program_version", PM_VERSION) + "," + jstr("policy_name", PM_POLICY_NAME) + "," + jstr("mode", mode_name.empty() ? "custom" : mode_name) + "," + jstr("target", sim ? "sim" : "real") + "," +
         jstr("ref_mode", ref_mode) + "," + jbool("check_only", check_only) + "," + jstr("run_dir", run_dir) + "," +
         "\"sha256\":{" + jstr("executable", hashes.exe) + "," + jstr("policy_onnx", hashes.policy) + "," + jstr("policy_params", hashes.params) + "," + jstr("motion_bin", hashes.motion) + "," + jstr("site_conf", hashes.site_conf) + "," + jstr("modes_conf", hashes.modes_conf) + "}," +
         "\"handover\":{\"pre_takeover_state\":" + pre_state_json + ",\"take_over_state\":" + take_state_json + "}," +
         "\"t_mono\":{" + jnum("runner_start", t_start_mono) + "," + jnum("report", mono()) + "," + jnum("wall_minus_mono", wall() - mono()) + "},";
  rep += "\"fault_events\":" + fev + ",\"admission\":" + items_json(items) + "," + jnum("frames_sent", (double)sess.n_sent()) + "," + jnum("control_ticks", n_ticks) + "," + jnum("first_cmd_step_rad", first_cmd_step) + "," +
         jnum("first_cmd_authority", first_cmd_auth) + "," +
         "\"guard_stats\":{" + jnum("cmds", (double)guard.stats.cmds) + "," + jnum("pos_clipped", (double)guard.stats.pos_clipped) + "," + jnum("slew_clipped", (double)guard.stats.slew_clipped) + "," + jnum("torque_clipped", (double)guard.stats.torque_clipped) + "," +
         jnum("max_dq_ratio", guard.stats.max_dq_ratio) + "," + jnum("max_tau_ratio", guard.stats.max_tau_ratio) + "," + jnum("max_tilt", std::max(guard.stats.max_tilt, sess.max_tilt_seen())) + "," + jnum("max_tilt_ticks_only", guard.stats.max_tilt) + "," + jnum("max_gyro", std::max(guard.stats.max_gyro, sess.max_gyro_xy_seen())) + "," + jnum("max_gyro_ticks_only", guard.stats.max_gyro) + "," + jnum("min_q_margin", guard.stats.min_q_margin) + "},\"tilt_judge\":" + sess.tilt_judge_json() + "}";  // V1.6 A1: tilt_judge = limits in force + max gravity error / min margins of the run
  write_atomic(run_dir + "/run_report.json", rep);  // first version: the service announces the result from it at once (voice does not wait for the post-exit recording)
  ev_write("result", jstr("result", result) + "," + jstr("exit_reason", exit_reason) + "," + jstr("fault", fault_name(guard.fault())) + "," + jbool("exit_confirmed", took && eo.confirmed) + "," + jnum("restored_fsm", final_fsm) + "," +
                         jstr("final_mode", final_mode) + "," + jbool("recovery_ok", recovery_ok));
  logf("RESULT %s reason=%s fault=%s requested=%s final_mode=%s final_fsm=%d recovery_ok=%d detect->rpc_sent=%.2f ms exit_code=%d -> %s", result.c_str(), exit_reason.c_str(), fault_name(guard.fault()), requested_mode.c_str(),
       final_mode.c_str(), final_fsm, (int)recovery_ok, (took && eo.t_rpc_sent > 0) ? (eo.t_rpc_sent - eo.t_detect) * 1e3 : -1.0, exit_code, run_dir.c_str());
  (void)admitted; (void)go_refused;
  status("REPORTED");
  // ---- post-exit recording: the state after the hand-back (the first seconds of the OEM controller) stays in state_stream.bin
  if (took && site.log_post_exit_s > 0) {
    const double t_end = mono() + site.log_post_exit_s;
    while (mono() < t_end && !g_quit) sleep_s(0.05);
  }
  ev_write("post_exit_end", jnum("post_exit_s", site.log_post_exit_s) + "," + jbool("interrupted", g_quit.load()));
  rlog.end_stream();
  {
    LogStats ls = rlog.stats();
    ev_write("log_stats", log_stats_fields(ls, rlog.lowstate()) + "," + jnum("dds_loopback_max_ms", link.max_loopback_ms()));  // V1.6: dropped_* = real losses; state_ring_evicted = pre-roll window rolling (by design)
  }
  rlog.close();
  FinalizeResult fin = finalize_run(run_dir, rep, "runner");
  logf("anomaly summary: %s (%d fail, %d warn) -> %s/anomaly_summary.txt", fin.level.c_str(), fin.n_fail, fin.n_warn, run_dir.c_str());
  fast_exit(exit_code);
}

}  // namespace pm
