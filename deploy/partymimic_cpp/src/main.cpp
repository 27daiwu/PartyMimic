// partymimic: single executable of the C++ PartyMimic G1 deployment.
//   partymimic service   remote-controller listener + voice + guardian/runner supervisor (what ./start.sh runs; READ-ONLY on the robot)
//   partymimic runner    one F1+F3 -> L1+A cycle (spawned by the service): load, check, take over, run the window, hand back
//   partymimic guardian  independent watchdog process (takes over if the runner dies / hangs)
//   partymimic release   operator hand-back of a robot stuck in user control
//   partymimic selfcheck offline checks (no robot connection)
#include <unistd.h>

#include <climits>
#include <cstdio>
#include <cstring>

#include "pm/app.hpp"

namespace pm {

Paths Paths::resolve() {
  Paths p;
  if (const char* e = getenv("PM_ROOT")) p.root = e;
  else {
    char b[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1);
    std::string exe = n > 0 ? std::string(b, (size_t)n) : "";
    std::string dir = exe.substr(0, exe.find_last_of('/'));
    p.root = dir.substr(0, dir.find_last_of('/'));
  }
  p.assets = p.root + "/assets";
  p.config = p.root + "/config";
  p.logs = getenv("PM_LOGS") ? getenv("PM_LOGS") : p.root + "/logs";
  return p;
}

bool Args::parse(int argc, char** argv, const std::set<std::string>& flag_names, std::string& err) {
  for (int i = 0; i < argc; i++) {
    std::string a = argv[i];
    if (a.rfind("--", 0) != 0) { err = "unexpected argument " + a; return false; }
    if (flag_names.count(a)) { flags.insert(a); continue; }
    if (i + 1 >= argc) { err = "missing value for " + a + " (an unknown option is read as \"--key value\"; value-less flags must be listed in flags_service/flags_runner/flags_guardian in app.hpp)"; return false; }
    kv[a] = argv[++i];
  }
  return true;
}

std::vector<std::string> guardian_cmdline(const GuardianCmd& c) {
  std::vector<std::string> a = {c.self, "guardian", "--iface", c.iface, "--domain", std::to_string(c.domain), "--report", c.report, "--max-run-s", "604800"};
  if (c.sim) a.push_back("--sim");
  if (!c.entry_fsm.empty()) { a.push_back("--entry-fsm"); a.push_back(c.entry_fsm); }
  return a;
}
std::vector<std::string> runner_cmdline(const RunnerCmd& c) {
  std::vector<std::string> a = {c.self, "runner", "--iface", c.iface, "--domain", std::to_string(c.domain), "--run-s", std::to_string(c.run_s), "--ref-mode", c.ref_mode, "--authority-max", std::to_string(c.authority),
                                "--ramp-s", std::to_string(c.ramp_s), "--end-hold-s", std::to_string(c.end_hold_s),
                                "--run-dir", c.run_dir, "--wait-go", c.go_file, "--status-file", c.status_file, "--ready-timeout-s", std::to_string(c.ready_timeout_s)};
  if (c.allow_partial) a.push_back("--allow-partial-authority");
  if (c.accept_unvalidated) a.push_back("--accept-unvalidated");
  if (!c.entry_fsm.empty()) { a.push_back("--entry-fsm"); a.push_back(c.entry_fsm); }
  if (!c.unconfirmed_hold_s.empty()) { a.push_back("--unconfirmed-hold-s"); a.push_back(c.unconfirmed_hold_s); }
  a.push_back(c.sim ? "--sim" : "--i-confirm-protection");
  return a;
}

void Site::apply_to(SessionConfig& s) const {
  s.exit = exit;
  s.watchdog_s = watchdog_s;
  s.max_entry_tilt_rad = max_entry_tilt_rad;
  s.quiescent_dq = quiescent_dq;
  s.min_state_hz = min_state_hz;
}

Site Site::load(const std::string& path) {
  Site s;
  KV kv;
  if (!kv.load(path)) return s;
  auto e = kv.ints("entry_fsm_allowed");
  if (!e.empty()) s.entry_allowed = e;
  s.oem_confirmed = kv.num("oem_lowcmd_confirmed", 0) != 0;
  s.oem_expected_writers = (int)kv.num("oem_lowcmd_expected_writers", -1);
  s.iface = kv.str("iface", "");
  if (s.iface == "auto") s.iface = "";
  s.run_s = kv.num("run_s", 1.0);
  s.authority = kv.num("authority", 0.3);
  s.ref_mode = kv.str("ref_mode", "running");
  s.ramp_s = kv.num("ramp_s", s.ramp_s);
  s.end_hold_s = kv.num("end_hold_s", s.end_hold_s);
  s.allow_partial_authority = kv.num("allow_partial_authority", 0) != 0;
  s.tts = kv.num("tts", 1) != 0;
  s.tts_volume = (int)kv.num("tts_volume", 100);
  s.max_run_s = kv.num("max_run_s", s.max_run_s);
  s.max_authority = kv.num("max_authority", s.max_authority);
  s.watchdog_s = kv.num("watchdog_s", s.watchdog_s);
  s.max_entry_tilt_rad = kv.num("max_entry_tilt_rad", s.max_entry_tilt_rad);
  s.quiescent_dq = kv.num("quiescent_dq", s.quiescent_dq);
  s.min_state_hz = kv.num("min_state_hz", s.min_state_hz);
  GuardConfig& g = s.guard;
  g.state_stale_s = kv.num("state_stale_s", g.state_stale_s);
  g.state_frozen_ticks = (int)kv.num("state_frozen_ticks", g.state_frozen_ticks);
  g.limit_fault_tol_rad = kv.num("limit_fault_tol_rad", g.limit_fault_tol_rad);
  g.speed_fault_frac = kv.num("speed_fault_frac", g.speed_fault_frac);
  g.unknown_speed_limit = kv.num("unknown_speed_limit", g.unknown_speed_limit);
  g.torque_meas_fault_frac = kv.num("torque_meas_fault_frac", g.torque_meas_fault_frac);
  g.torque_meas_fault_ticks = (int)kv.num("torque_meas_fault_ticks", g.torque_meas_fault_ticks);
  g.max_tilt_rad = kv.num("max_tilt_rad", g.max_tilt_rad);
  g.max_gyro_rad_s = kv.num("max_gyro_rad_s", g.max_gyro_rad_s);
  g.pos_margin_rad = kv.num("pos_margin_rad", g.pos_margin_rad);
  g.target_speed_frac = kv.num("target_speed_frac", g.target_speed_frac);
  g.torque_cmd_frac = kv.num("torque_cmd_frac", g.torque_cmd_frac);
  g.diverge_rad = kv.num("diverge_rad", g.diverge_rad);
  g.diverge_ticks = (int)kv.num("diverge_ticks", g.diverge_ticks);
  g.loop_overrun_factor = kv.num("loop_overrun_factor", g.loop_overrun_factor);
  ExitConfig& x = s.exit;
  x.fallen_tilt_rad = kv.num("fallen_tilt_rad", x.fallen_tilt_rad);
  x.unstable_tilt_rad = kv.num("unstable_tilt_rad", x.unstable_tilt_rad);
  x.unstable_soft_tilt_rad = kv.num("unstable_soft_tilt_rad", x.unstable_soft_tilt_rad);
  x.unstable_gyro_rad_s = kv.num("unstable_gyro_rad_s", x.unstable_gyro_rad_s);
  x.unstable_tilt_rate_rad_s = kv.num("unstable_tilt_rate_rad_s", x.unstable_tilt_rate_rad_s);
  x.unstable_debounce_samples = (int)kv.num("unstable_debounce_samples", x.unstable_debounce_samples);
  x.attitude_max_age_s = kv.num("attitude_max_age_s", x.attitude_max_age_s);
  x.l2b_hold_s = kv.num("l2b_hold_s", x.l2b_hold_s);
  x.confirm_timeout_s = kv.num("exit_confirm_timeout_s", x.confirm_timeout_s);
  x.fsm_poll_s = kv.num("exit_fsm_poll_s", x.fsm_poll_s);
  x.dwell_s = kv.num("exit_dwell_s", x.dwell_s);
  x.stable_window_s = kv.num("exit_stable_window_s", x.stable_window_s);
  x.min_stable_polls = (int)kv.num("exit_min_stable_polls", x.min_stable_polls);
  x.mismatch_fail_s = kv.num("exit_mismatch_fail_s", x.mismatch_fail_s);
  x.unconfirmed_hold_s = kv.num("exit_unconfirmed_hold_s", x.unconfirmed_hold_s);
  x.hold_gain_scale = kv.num("exit_hold_gain_scale", x.hold_gain_scale);
  x.exit_owner_timeout_s = kv.num("exit_owner_timeout_s", x.exit_owner_timeout_s);
  static const char* known[] = {"entry_fsm_allowed", "oem_lowcmd_confirmed", "oem_lowcmd_expected_writers", "iface", "run_s", "authority", "ref_mode", "tts", "tts_volume", "max_run_s", "max_authority", "ramp_s", "end_hold_s", "allow_partial_authority",
      "watchdog_s", "max_entry_tilt_rad", "quiescent_dq", "min_state_hz", "state_stale_s", "state_frozen_ticks", "limit_fault_tol_rad", "speed_fault_frac", "unknown_speed_limit",
      "torque_meas_fault_frac", "torque_meas_fault_ticks", "max_tilt_rad", "max_gyro_rad_s", "pos_margin_rad", "target_speed_frac", "torque_cmd_frac", "diverge_rad", "diverge_ticks",
      "loop_overrun_factor", "fallen_tilt_rad", "unstable_tilt_rad", "unstable_soft_tilt_rad", "unstable_gyro_rad_s", "unstable_tilt_rate_rad_s", "unstable_debounce_samples", "attitude_max_age_s",
      "l2b_hold_s", "exit_confirm_timeout_s", "exit_fsm_poll_s", "exit_dwell_s", "exit_stable_window_s", "exit_min_stable_polls", "exit_mismatch_fail_s", "exit_unconfirmed_hold_s",
      "exit_hold_gain_scale", "exit_owner_timeout_s"};
  for (auto& kvp : kv.m) {
    bool k = false;
    for (const char* n : known) k = k || kvp.first == n;
    if (!k) s.warnings.push_back("unknown site.conf key '" + kvp.first + "' ignored");
  }
  // range checks (enforced, not advisory)
  auto bad = [&](const std::string& m) { if (s.err.empty()) s.err = m; };
  std::string er;
  if (!validate(s.guard, er)) bad("site.conf: " + er);
  if (!validate(s.exit, er)) bad("site.conf: " + er);
  if (!(s.max_run_s > 0 && s.max_run_s <= ABS_MAX_RUN_S)) bad("site.conf: max_run_s must be in (0, " + std::to_string(ABS_MAX_RUN_S) + "]");
  if (!(s.ramp_s >= 0 && s.ramp_s <= 2.0)) bad("site.conf: ramp_s must be in [0, 2] s");
  if (!(s.end_hold_s >= 0 && s.end_hold_s <= 5.0)) bad("site.conf: end_hold_s must be in [0, 5] s");
  if (!(s.max_authority > 0 && s.max_authority <= ABS_MAX_AUTHORITY)) bad("site.conf: max_authority must be in (0, " + std::to_string(ABS_MAX_AUTHORITY) + "]");
  if (!(s.watchdog_s >= 0.05 && s.watchdog_s <= 0.5)) bad("site.conf: watchdog_s must be in [0.05, 0.5]");
  if (!(s.max_entry_tilt_rad >= 0.1 && s.max_entry_tilt_rad <= 0.8)) bad("site.conf: max_entry_tilt_rad must be in [0.1, 0.8]");
  if (!(s.quiescent_dq >= 0.1 && s.quiescent_dq <= 2.0)) bad("site.conf: quiescent_dq must be in [0.1, 2.0]");
  if (!(s.min_state_hz >= 15 && s.min_state_hz <= 500)) bad("site.conf: min_state_hz must be in [15, 500]");
  return s;
}

bool GuardianAlive::read(const std::string& path, GuardianAlive& g) {
  KV kv;
  for (int i = 0; i < 4; i++) {  // tolerate a concurrent rewrite
    if (kv.load(path) && kv.has("t")) {
      g.t = kv.num("t", 0); g.pid = (int)kv.num("pid", 0); g.user_writers = (int)kv.num("user_writers", 0); g.iface = kv.str("iface", "");
      g.exit_phase = kv.str("exit_phase", "NONE"); g.exit_owner = kv.str("exit_owner", "none"); g.exit_t = kv.num("exit_t", 0);
      return true;
    }
    sleep_s(0.005);
  }
  return false;
}
void GuardianAlive::write(const std::string& path, const GuardianAlive& g) {
  char b[384];
  snprintf(b, sizeof b, "t %.6f\npid %d\niface %s\nuser_writers %d\nexit_phase %s\nexit_owner %s\nexit_t %.3f\n", g.t, g.pid, g.iface.c_str(), g.user_writers, g.exit_phase.c_str(), g.exit_owner.c_str(), g.exit_t);
  write_atomic(path, b);  // atomic: readers never see a half-written file
}

void fast_exit(int rc) {
  fflush(stdout);
  fflush(stderr);
  _exit(rc);
}

std::string jstr(const std::string& k, const std::string& v) { return "\"" + k + "\":\"" + json_escape(v) + "\""; }
std::string jnum(const std::string& k, double v) { char b[64]; snprintf(b, sizeof b, "%.9g", v); return "\"" + k + "\":" + b; }
std::string jbool(const std::string& k, bool v) { return "\"" + k + "\":" + (v ? "true" : "false"); }
std::string json_find_raw(const std::string& doc, const std::string& key) {
  size_t k = doc.find("\"" + key + "\":");
  if (k == std::string::npos) return "";
  k += key.size() + 3;
  while (k < doc.size() && doc[k] == ' ') k++;
  size_t e = doc.find_first_of(",}\n", k);
  return doc.substr(k, e == std::string::npos ? std::string::npos : e - k);
}
std::string json_find_string(const std::string& doc, const std::string& key) {
  std::string raw = json_find_raw(doc, key);
  if (raw.size() >= 2 && raw.front() == '"' && raw.back() == '"') return raw.substr(1, raw.size() - 2);
  return raw;
}

}  // namespace pm

int main(int argc, char** argv) {
  using namespace pm;
  if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
    printf("usage: partymimic service|runner|guardian|release|selfcheck [options]   (normally started by ./start.sh)\n");
    return argc < 2 ? 2 : 0;
  }
  Paths p = Paths::resolve();
  std::string cmd = argv[1];
  int rc;
  if (cmd == "service") rc = run_service(argc - 2, argv + 2, p);
  else if (cmd == "runner") rc = run_runner(argc - 2, argv + 2, p);
  else if (cmd == "guardian") rc = run_guardian(argc - 2, argv + 2, p);
  else if (cmd == "release") rc = run_release(argc - 2, argv + 2, p);
  else if (cmd == "selfcheck") rc = run_selfcheck(argc - 2, argv + 2, p);
  else { fprintf(stderr, "unknown command %s\n", cmd.c_str()); return 2; }
  fflush(stdout);
  fflush(stderr);
  _exit(rc);  // DDS reader threads can keep the process alive after main() returned
}
