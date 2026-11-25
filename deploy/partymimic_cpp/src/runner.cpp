// Runner: loads the frozen P2 Policy, checks the live robot read-only, waits for GO, takes control from WalkRun, runs the SSB window, hands back.
// Phases (status file `phase`):
//   STARTING -> LOADING_MODEL (ONNX + observation stack + warm-up on live states, NO command) -> CHECKING (real-time admission, read-only)
//   -> READY (waits for the GO file written by the service on L1+A; SIGTERM / START x2 / timeout abort WITHOUT ever touching control)
//   -> ARM (fast preflight, lock, SwitchToUserCtrl, confirm) -> RUN (Policy drives from the first tick; authority ramp 0->max in 0.1 s)
//   -> EXIT (SwitchToInternalCtrl at once on a dedicated thread, HOLD frames meanwhile -> verify by GetFsmId + stable window; see pm/exit_policy.hpp).
// Any guard fault / instability / START x2 / watchdog -> the exit starts IMMEDIATELY (WALKRUN, or PASSIVE if fallen / untrusted / emergency channel).
// Exit codes: 0 clean (COMPLETED) | 1 crash | 2 refused | 3 no lowstate | 4 ARM failed | 5 EXIT UNCONFIRMED | 6 BLOCKED by admission | 10 fault/stop, WalkRun confirmed
//             11 WALKRUN_RECOVERY_FAILED | 12 EMERGENCY_PASSIVE | 13 FALLEN_PASSIVE (PASSIVE decided: robot down / state untrusted)
#include <algorithm>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <csignal>
#include <cstring>
#include <fstream>
#include <thread>

#include "pm/app.hpp"
#include "pm/frozen.hpp"
#include "pm/guard.hpp"
#include "pm/link.hpp"
#include "pm/policy.hpp"
#include "pm/session.hpp"

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

struct TraceCsv {
  FILE* f = nullptr;
  explicit TraceCsv(const std::string& path, const std::vector<std::string>& names) {
    f = fopen(path.c_str(), "w");
    if (!f) return;
    fprintf(f, "t,phase,ssb_frame,authority,infer_ms,step_ms,state_age_ms,tick,keys_b2,keys_b3,quat_w,quat_x,quat_y,quat_z,gyro_x,gyro_y,gyro_z,roll,pitch,yaw");
    for (const char* g : {"q", "dq", "tau_est", "action", "q_policy", "q_cmd"})
      for (auto& n : names) fprintf(f, ",%s:%s", g, n.c_str());
    fprintf(f, "\n");
    fflush(f);
  }
  void row(double t, const char* phase, int frame, double auth, double infer, double step, double age, const LinkState& st, const Vec* groups[6]) {
    if (!f) return;
    double roll, pitch;
    roll_pitch(st.quat, roll, pitch);
    fprintf(f, "%.4f,%s,%d,%.6g,%.6g,%.6g,%.6g,%u,%u,%u,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g", t, phase, frame, auth, infer, step, age, st.tick, st.wireless[2], st.wireless[3],
            st.quat[0], st.quat[1], st.quat[2], st.quat[3], st.gyro[0], st.gyro[1], st.gyro[2], roll, pitch, st.rpy[2]);
    for (int g = 0; g < 6; g++)
      for (int i = 0; i < N; i++) fprintf(f, ",%.6g", (*groups[g])[i]);
    fprintf(f, "\n");
    fflush(f);  // flushed per line: a crash / kill -9 keeps everything up to the last tick
  }
  ~TraceCsv() { if (f) fclose(f); }
};

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
  const bool accept_unvalidated = a.has("--accept-unvalidated");
  const std::string lock_path = a.get("--lock-path", DEFAULT_LOCK), estop_file = a.get("--estop-file", DEFAULT_ESTOP);
  const std::string wait_go = a.get("--wait-go"), status_file = a.get("--status-file");
  const double ready_timeout = a.num("--ready-timeout-s", 120.0);
  const int warmup_n = (int)a.num("--warmup-n", 8);
  std::string run_dir = a.get("--run-dir", P.logs + "/runs/" + timestamp_compact());
  ::mkdir(P.logs.c_str(), 0775);
  ::mkdir((P.logs + "/runs").c_str(), 0775);
  ::mkdir(run_dir.c_str(), 0775);
  log_set_file(run_dir + "/console.log");
  log_set_tag("[runner]");
  FILE* evf = fopen((run_dir + "/events.jsonl").c_str(), "a");
  auto ev_write = [&](const std::string& kind, const std::string& fields) {
    if (!evf) return;
    fprintf(evf, "{\"wall\":%.4f,\"kind\":\"%s\"%s%s}\n", wall(), kind.c_str(), fields.empty() ? "" : ",", fields.c_str());
    fflush(evf);
  };
  auto status = [&](const std::string& phase) {
    ev_write("phase", jstr("phase", phase) + "," + jnum("t_mono", mono()));
    if (status_file.empty()) return;
    char b[256];
    snprintf(b, sizeof b, "{\"phase\":\"%s\",\"tm\":%.6f,\"time\":%.6f,\"pid\":%d}", phase.c_str(), mono(), wall(), (int)getpid());
    write_atomic(status_file, b);
  };
  status("STARTING");

  Site site = Site::load(P.config + "/site.conf");
  if (!site.err.empty()) { logf("REFUSED: %s", site.err.c_str()); return 2; }
  for (auto& w : site.warnings) logf("WARNING: %s", w.c_str());
  const double unconfirmed_hold = a.num("--unconfirmed-hold-s", site.exit.unconfirmed_hold_s);
  const double diverge_rad = cli_diverge ? a.num("--diverge-rad", 1.5) : site.guard.diverge_rad;
  const bool exceeds_legacy = run_s > LEGACY_MAX_RUN_S + 1e-9 || amax > LEGACY_MAX_AUTHORITY + 1e-9;
  std::vector<int> allowed = site.entry_allowed;
  if (a.kv.count("--entry-fsm")) allowed = {(int)a.num("--entry-fsm", 500)};

  // ---- hard refusals before ANY DDS object exists
  if (!sim) {
    std::string bad;
    if (!(run_s > 0 && run_s <= site.max_run_s)) bad += "run_s " + std::to_string(run_s) + " not in (0, max_run_s=" + std::to_string(site.max_run_s) + "] (site.conf max_run_s, compile-time limit " + std::to_string(ABS_MAX_RUN_S) + "); ";
    if (!(amax > 0 && amax <= site.max_authority)) bad += "authority " + std::to_string(amax) + " not in (0, max_authority=" + std::to_string(site.max_authority) + "] (site.conf max_authority, compile-time limit " + std::to_string(ABS_MAX_AUTHORITY) + "); ";
    // P4: authority > 0.3 or a window > 1.0 s (the only configurations ever run on the real robot) are validated OFFLINE (Sim2Sim) only. They need an explicit,
    // per-start operator acknowledgement; nothing in site.conf can grant it.
    if (exceeds_legacy && !accept_unvalidated) bad += "run_s " + std::to_string(run_s) + " / authority " + std::to_string(amax) + " exceed the real-robot validated envelope (<= 1.0 s, <= 0.3): the P4 full-authority control law has NOT been validated on the real robot - start with ./start.sh --accept-unvalidated (asks for a typed confirmation); ";
    if (!bad.empty()) { logf("REFUSED: %s", bad.c_str()); return 2; }
    if (iface.empty() || iface == "lo") { logf("REFUSED: a real run needs the robot network interface (not lo)"); return 2; }
    if (!a.has("--i-confirm-protection")) { logf("REFUSED: real target needs --i-confirm-protection (start.sh asks you; fall protection, e-stop in hand, >= 2 people)"); return 2; }
  } else if (iface != "lo") { logf("REFUSED: --sim must use --iface lo"); return 2; }
  if (sim && !(run_s > 0 && run_s <= ABS_MAX_RUN_S && amax > 0 && amax <= ABS_MAX_AUTHORITY)) { logf("REFUSED: run_s/authority above the compile-time limits (%.0f s / %.1f)", ABS_MAX_RUN_S, ABS_MAX_AUTHORITY); return 2; }
  if (exceeds_legacy) logf("WARNING: run_s=%.2f / authority=%.2f EXCEED the legacy real-robot caps (%.1f s / %.1f): allowed by site.conf max_run_s=%.2f max_authority=%.2f (compile-time limits %.0f s / %.1f)", run_s, amax, LEGACY_MAX_RUN_S, LEGACY_MAX_AUTHORITY, site.max_run_s, site.max_authority, ABS_MAX_RUN_S, ABS_MAX_AUTHORITY);
  if (ref_mode != "running" && ref_mode != "paused") { logf("REFUSED: bad --ref-mode"); return 2; }
  if (!(ramp_s >= 0 && ramp_s <= 2.0) || !(end_hold_s >= 0 && end_hold_s <= 5.0)) { logf("REFUSED: --ramp-s must be in [0,2] and --end-hold-s in [0,5]"); return 2; }
  if (amax > 0.05 && amax < 0.9 && !allow_partial) {
    logf("REFUSED: sustained authority %.2f is in the partial-authority regime (0.05, 0.9): in Sim2Sim (real-log initial states) q_cmd = q_hold + a*(q_target - q_hold) with a<0.9 does NOT balance (a=0: falls in 1.0-1.7 s; a=0.3: falls in 1.2-2.0 s, reproducing the 2 real falls). "
         "Use authority 1.0 (ramped by --ramp-s) or pass --allow-partial-authority for a deliberate experiment.", amax);
    return 2;
  }

  Params params;
  Motion motion;
  if (!Params::load(P.asset("policy_params.txt"), params, err) || !Motion::load(P.asset("motion.bin"), motion, err)) { logf("REFUSED: %s", err.c_str()); return 2; }
  const GuardConfig& gcfg = site.guard;
  SafetyGuard guard(params, gcfg);
  SessionConfig scfg;
  site.apply_to(scfg);
  scfg.lock_path = lock_path; scfg.estop_file = estop_file; scfg.entry_allowed = allowed;
  scfg.exit.unconfirmed_hold_s = unconfirmed_hold; scfg.exit.estop_file = estop_file;
  Vec kp = params.kp, kd = params.kd;

  LinkOpts lo;
  lo.rpc_timeout_s = 2.0;
  Link link(iface, domain, lo);
  ControlSession sess(link, guard, scfg, [&](const std::string& k, const std::string& f) { ev_write(k, f); });
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
  TraceCsv trace(run_dir + "/trace.csv", params.names);
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
    goto_phase("LOADING_MODEL");
    ps.reset(new PolicyStack(params, motion, P.asset("policy.onnx"), (int)(getenv("PM_ORT_THREADS") ? atoi(getenv("PM_ORT_THREADS")) : 2)));
    const int seg_frames = std::min(motion.T - 1, (int)std::lround(run_s * motion.fps));
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
    // ---- real-time admission (read-only)
    goto_phase("CHECKING");
    if (g_quit) { result = "ABORTED_BEFORE_ARM"; exit_reason = "signal before checks"; throw 0; }
    {
      struct Art { const char* name; const char* want; };
      for (Art ar : {Art{"policy.onnx", FROZEN_SHA256_POLICY}, Art{"policy_params.txt", FROZEN_SHA256_POLICY_PARAMS}, Art{"motion.bin", FROZEN_SHA256_MOTION}}) {
        std::string h;
        bool ok = sha256_file(P.asset(ar.name), h) && h == ar.want;
        items.push_back({std::string("artifact_") + ar.name + "_frozen", ok ? L_PASS : L_FAIL, ok ? "sha256 == frozen value (policy: P4 candidate, see frozen.hpp)" : "MISMATCH " + std::string(ar.name)});
      }
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
    sess.preflight(pf);
    scan_thr.join();
    for (auto& r : pf) items.push_back({"pf_" + r.name, r.level, r.detail});
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
    // ---- READY / GO
    if (!wait_go.empty()) {
      logf("READY: model loaded, checks passed. Waiting for GO (L1+A) / abort (START x2, SIGTERM).");
      status("READY");
      double t_ready = mono();
      while (!file_exists(wait_go)) {
        if (g_quit) { logf("ABORTED before ARM (signal): control was never taken."); result = "ABORTED_BEFORE_ARM"; exit_reason = "signal before ARM"; throw 0; }
        if (sess.remote_fired()) { logf("ABORTED before ARM by remote START double-click: control was never taken."); result = "ABORTED_BEFORE_ARM"; exit_reason = "START x2 before ARM"; throw 0; }
        if (mono() - t_ready > ready_timeout) { logf("READY timed out after %.0f s without GO.", ready_timeout); result = "READY_TIMEOUT"; exit_reason = "READY timeout"; throw 0; }
        sleep_s(0.02);
      }
      ::unlink(wait_go.c_str());
      // state may have changed while waiting: the state-dependent checks repeat (guardian, OEM writers); acquire() re-runs the fast preflight itself
      Item g_item = guardian_item();
      WriterScan sc2;
      scan_writers(0.5, sc2);
      auto cf2 = link.get_fsm();
      Item w_item = writer_item(sc2, cf2.second);
      Item f_item = {"fsm_unchanged_since_ready", (cf2.first == 0 && cf2.second == ready_fsm) ? L_PASS : L_FAIL,
                     "fsm " + std::to_string(ready_fsm) + " at READY, " + std::to_string(cf2.second) + " now (a change means the factory remote reacted to the chord or the mode was changed)"};
      ev_write("go_recheck", "\"items\":" + items_json({g_item, w_item, f_item}));
      if (!(g_item.pass() && w_item.pass() && f_item.pass())) {
        for (const Item* it : {&g_item, &w_item, &f_item}) if (!it->pass()) logf("GO REFUSED: %s: %s", it->name.c_str(), it->detail.c_str());
        result = "GO_REFUSED"; exit_code = 6; go_refused = true; throw 0;
      }
      logf("GO received -> ARM");
    }
    // ---- ARM
    goto_phase("ARM");
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
    ps->reset(to_sample(*sess.state()), ref_mode == "paused");
    const double t_take = mono();
    double next_t = mono();
    int div_count = 0;
    double t_window_end = -1;
    bool first_cmd = true;
    for (;;) {
      double now = mono();
      if (now < next_t) sleep_s(next_t - now);
      next_t += DT;
      auto s0 = sess.state();
      uint64_t seq0 = s0 ? s0->seq : 0;
      double t_wait = mono() + 0.008;  // sample-synchronous tick: shave the state age
      while (mono() < t_wait) {
        auto s1 = sess.state();
        if (!(s1 && s1->seq == seq0)) break;
        sleep_s(0.0003);
      }
      double tstep = mono();
      sess.heartbeat();
      double ts = sess.now();
      if (guard.tripped() || g_quit || sess.exit_triggered()) break;
      auto stc = sess.state();
      if (sess.check_state_once(stc) != Fault::NONE) break;
      if (sess.has_last_cmd()) {
        double dv = (stc->q - sess.last_cmd_q()).cwiseAbs().maxCoeff();
        div_count = dv > diverge_rad ? div_count + 1 : 0;
        if (div_count >= gcfg.diverge_ticks) { sess.emergency(Fault::NO_RESPONSE, "|q - q_cmd| " + std::to_string(dv) + " rad > " + std::to_string(diverge_rad) + " for 8 ticks (command not followed)"); break; }
      }
      double el = mono() - t_take;
      PolicyOut po;
      try {
        if (sim && inject_tick >= 0 && n_ticks == inject_tick && inject_kind == "inference") throw std::runtime_error("injected inference exception (sim)");
        po = ps->tick(to_sample(*stc));
      } catch (const std::exception& e) {
        sess.emergency(Fault::INFERENCE, e.what());
        break;
      }
      if (sim && inject_tick >= 0 && n_ticks == inject_tick) {  // DEV ONLY (--sim, env PM_SIM_INJECT): exercise the fault exits on the loopback robot
        if (inject_kind == "nan") po.q_target[3] = NAN;
        else if (inject_kind == "overrun") sleep_s(0.075);
        else if (inject_kind == "hang") sleep_s(0.3);
        logf("SIM INJECT %s at tick %d", inject_kind.c_str(), n_ticks);
      }
      double authority = ramp_s > 0 ? amax * smoothstep(el / ramp_s) : amax;
      Vec q_cmd = q_hold + authority * (po.q_target - q_hold);
      if (!sess.write(q_cmd, kp, kd, DT)) break;
      if (first_cmd) { first_cmd = false; first_cmd_step = (sess.last_cmd_q() - stc->q).cwiseAbs().maxCoeff(); first_cmd_auth = authority; }
      n_ticks++;
      double step_ms = 1e3 * (mono() - tstep);
      Vec act, qtv;
      for (int i = 0; i < N; i++) act[i] = po.action[i];
      const Vec* groups[6] = {&stc->q, &stc->dq, &stc->tau, &act, &po.q_target, &sess.last_cmd_q()};
      trace.row(ts, "RUN", po.motion_t, authority, po.infer_ms, step_ms, 1e3 * (mono() - stc->arrival), *stc, groups);
      if (step_ms > gcfg.loop_overrun_factor * 1e3 * DT) { sess.emergency(Fault::LOOP_OVERRUN, "control step took " + std::to_string(step_ms) + " ms"); break; }
      // run window over -> (reference stays paused for end_hold_s so the robot settles) -> EXIT hand-back
      const bool window_over = (ref_mode == "paused" && el >= run_s) || (ref_mode == "running" && po.motion_t >= seg_frames);
      if (window_over && t_window_end < 0) { t_window_end = el; if (ref_mode == "running") ps->hold_reference(); }
      if ((t_window_end >= 0 && el - t_window_end >= (ref_mode == "paused" ? 0.0 : end_hold_s)) || el > run_s + end_hold_s + 1.0) break;
    }
    run_end = guard.tripped() ? "FAULT" : (g_quit ? "OPERATOR_QUIT" : "COMPLETED");
  } catch (int) {
    // controlled early exits set result/exit_code
  } catch (const std::exception& e) {
    logf("UNHANDLED %s", e.what());
    sess.emergency(Fault::INFERENCE, std::string("unhandled ") + e.what());
    result = "CRASH"; exit_code = 1; exit_reason = std::string("crash: ") + e.what();
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
                    jstr("exit_class", took ? er.req.decision.cls : "NONE") + "," + jstr("voice", took ? (result == eo.verdict.result ? eo.verdict.voice : std::string("检查失败")) : "") + "," + jnum("exit_code", exit_code) + "},";
  rep += jstr("result", result) + "," + jstr("exit_reason", exit_reason) + "," + jnum("exit_code_planned", exit_code) + "," + jnum("user_fsm", sess.user_fsm_seen()) + "," + jstr("run_end", run_end) + ",";
  rep += "\"exit\":{" + jstr("reason", er.reason) + "," + jstr("internal_mode", requested_mode) + "," + jstr("exit_mode_why", er.exit_mode_why) + "," + jnum("rpc_code", er.rpc_code) + "," + jnum("fsm_after", final_fsm) + "," +
         jbool("confirmed", took && eo.confirmed) + "," + jstr("confirmed_via", er.confirmed_via) + "," + jbool("mode_matched", took && eo.confirmed) + "," + jstr("note", er.note) + "},";
  rep += std::string("\"exit_detail\":") + (took ? exit_outcome_json(er.req, eo) : "null") + ",";
  rep += jbool("exceeds_legacy_caps", exceeds_legacy) + "," + jnum("run_s", run_s) + "," + jnum("authority", amax) + ",";
  rep += "\"fault_events\":" + fev + ",\"admission\":" + items_json(items) + "," + jnum("frames_sent", (double)sess.n_sent()) + "," + jnum("control_ticks", n_ticks) + "," + jnum("first_cmd_step_rad", first_cmd_step) + "," +
         jnum("first_cmd_authority", first_cmd_auth) + "," +
         "\"guard_stats\":{" + jnum("cmds", (double)guard.stats.cmds) + "," + jnum("pos_clipped", (double)guard.stats.pos_clipped) + "," + jnum("slew_clipped", (double)guard.stats.slew_clipped) + "," + jnum("torque_clipped", (double)guard.stats.torque_clipped) + "," +
         jnum("max_dq_ratio", guard.stats.max_dq_ratio) + "," + jnum("max_tau_ratio", guard.stats.max_tau_ratio) + "," + jnum("max_tilt", guard.stats.max_tilt) + "," + jnum("max_gyro", guard.stats.max_gyro) + "," + jnum("min_q_margin", guard.stats.min_q_margin) + "}}";
  write_atomic(run_dir + "/run_report.json", rep);
  ev_write("result", jstr("result", result) + "," + jstr("exit_reason", exit_reason) + "," + jstr("fault", fault_name(guard.fault())) + "," + jbool("exit_confirmed", took && eo.confirmed) + "," + jnum("restored_fsm", final_fsm) + "," +
                         jstr("final_mode", final_mode) + "," + jbool("recovery_ok", recovery_ok));
  logf("RESULT %s reason=%s fault=%s requested=%s final_mode=%s final_fsm=%d recovery_ok=%d detect->rpc_sent=%.2f ms exit_code=%d -> %s", result.c_str(), exit_reason.c_str(), fault_name(guard.fault()), requested_mode.c_str(),
       final_mode.c_str(), final_fsm, (int)recovery_ok, (took && eo.t_rpc_sent > 0) ? (eo.t_rpc_sent - eo.t_detect) * 1e3 : -1.0, exit_code, run_dir.c_str());
  (void)admitted; (void)go_refused;
  if (evf) fclose(evf);
  fast_exit(exit_code);
}

}  // namespace pm
