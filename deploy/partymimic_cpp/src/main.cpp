// partymimic: single executable of the C++ PartyMimic G1 deployment.
//   partymimic service   remote-controller listener + voice + guardian/runner supervisor (what ./start.sh runs; READ-ONLY on the robot)
//   partymimic runner    one F1+F3 -> L1+A cycle (spawned by the service): load, check, take over, run the window, hand back
//   partymimic guardian  independent watchdog process (takes over if the runner dies / hangs)
//   partymimic release   operator hand-back of a robot stuck in user control
//   partymimic selfcheck offline checks (no robot connection)
//   partymimic logtool   offline log tools: finalize (salvage run_report after a crash) | check (integrity) | replay (bit-exact policy replay of obs.bin)
#include <unistd.h>

#include <climits>
#include <sys/utsname.h>

#include "pm/frozen.hpp"
#include "pm/version.hpp"
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
                                "--wait-go", c.go_file, "--status-file", c.status_file};
  if (c.standby) { a.push_back("--standby"); a.push_back("--trigger-file"); a.push_back(c.trigger_file); }  // V1.5.1: pre-warmed runner; the run directory arrives with the trigger
  else { a.push_back("--run-dir"); a.push_back(c.run_dir); }
  if (c.allow_partial) a.push_back("--allow-partial-authority");
  if (c.check_only) a.push_back("--check-only");
  if (c.torque_ff) a.push_back("--torque-ff");
  if (c.no_torque_ff) a.push_back("--no-torque-ff");
  if (!c.entry_fsm.empty()) { a.push_back("--entry-fsm"); a.push_back(c.entry_fsm); }
  if (!c.mode_name.empty()) { a.push_back("--mode"); a.push_back(c.mode_name); }
  if (!c.announced_file.empty()) { a.push_back("--announced-file"); a.push_back(c.announced_file); }  // V1.6: the service reports "announcement finished" through this file
  if (!c.announce_wait_s.empty()) { a.push_back("--announce-wait-s"); a.push_back(c.announce_wait_s); }
  if (!c.unconfirmed_hold_s.empty()) { a.push_back("--unconfirmed-hold-s"); a.push_back(c.unconfirmed_hold_s); }
  if (c.sim) a.push_back("--sim");
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
  s.lead_in_s = kv.num("lead_in_s", s.lead_in_s);
  s.settle_stable_s = kv.num("settle_stable_s", s.settle_stable_s);
  s.settle_timeout_s = kv.num("settle_timeout_s", s.settle_timeout_s);
  s.log_preroll_s = kv.num("log_preroll_s", s.log_preroll_s);
  s.log_post_exit_s = kv.num("log_post_exit_s", s.log_post_exit_s);
  s.log_obs = kv.num("log_obs", 1) != 0;
  s.log_state_stream = kv.num("log_state_stream", 1) != 0;
  s.log_min_free_mb = kv.num("log_min_free_mb", s.log_min_free_mb);
  s.announce_wait_s = kv.num("announce_wait_s", s.announce_wait_s);
  s.announce_max_s = kv.num("announce_max_s", s.announce_max_s);
  s.announce_s_per_char = kv.num("announce_s_per_char", s.announce_s_per_char);
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
  g.torque_ff = kv.num("guard_torque_ff", 0) != 0;
  ExitConfig& x = s.exit;
  x.fallen_tilt_rad = kv.num("fallen_tilt_rad", x.fallen_tilt_rad);
  x.unstable_tilt_rad = kv.num("unstable_tilt_rad", x.unstable_tilt_rad);
  x.unstable_soft_tilt_rad = kv.num("unstable_soft_tilt_rad", x.unstable_soft_tilt_rad);
  x.unstable_gyro_rad_s = kv.num("unstable_gyro_rad_s", x.unstable_gyro_rad_s);
  x.unstable_tilt_rate_rad_s = kv.num("unstable_tilt_rate_rad_s", x.unstable_tilt_rate_rad_s);
  x.unstable_debounce_samples = (int)kv.num("unstable_debounce_samples", x.unstable_debounce_samples);
  x.ref_aware = kv.num("ref_aware_unstable", x.ref_aware ? 1 : 0) != 0;
  x.unstable_hard_tilt_rad = kv.num("unstable_hard_tilt_rad", x.unstable_hard_tilt_rad);
  x.ref_dev_rad = kv.num("ref_dev_rad", x.ref_dev_rad);
  x.ref_dev_min_tilt_rad = kv.num("ref_dev_min_tilt_rad", x.ref_dev_min_tilt_rad);
  x.ref_dev_growth_rad_s = kv.num("ref_dev_growth_rad_s", x.ref_dev_growth_rad_s);
  x.ref_max_age_s = kv.num("ref_max_age_s", x.ref_max_age_s);
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
      "loop_overrun_factor", "fallen_tilt_rad", "unstable_tilt_rad", "unstable_soft_tilt_rad", "unstable_gyro_rad_s", "unstable_tilt_rate_rad_s", "unstable_debounce_samples", "ref_aware_unstable", "unstable_hard_tilt_rad", "ref_dev_rad", "ref_dev_min_tilt_rad", "ref_dev_growth_rad_s", "ref_max_age_s", "attitude_max_age_s",
      "l2b_hold_s", "exit_confirm_timeout_s", "exit_fsm_poll_s", "exit_dwell_s", "exit_stable_window_s", "exit_min_stable_polls", "exit_mismatch_fail_s", "exit_unconfirmed_hold_s",
      "exit_hold_gain_scale", "exit_owner_timeout_s", "log_preroll_s", "log_post_exit_s", "log_obs", "log_state_stream", "log_min_free_mb", "guard_torque_ff", "lead_in_s", "settle_stable_s", "settle_timeout_s", "announce_wait_s", "announce_max_s", "announce_s_per_char"};
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
  if (!(s.lead_in_s >= 0 && s.lead_in_s <= 3)) bad("site.conf: lead_in_s must be in [0, 3] s");
  if (!(s.settle_stable_s >= 0.3 && s.settle_stable_s <= 10)) bad("site.conf: settle_stable_s must be in [0.3, 10] s");
  if (!(s.settle_timeout_s >= 3 && s.settle_timeout_s <= 120)) bad("site.conf: settle_timeout_s must be in [3, 120] s");
  if (!(s.log_preroll_s >= 0 && s.log_preroll_s <= 10)) bad("site.conf: log_preroll_s must be in [0, 10] s");
  if (!(s.log_post_exit_s >= 0 && s.log_post_exit_s <= 15)) bad("site.conf: log_post_exit_s must be in [0, 15] s");
  if (!(s.log_min_free_mb >= 0 && s.log_min_free_mb <= 100000)) bad("site.conf: log_min_free_mb must be in [0, 100000]");
  if (!(s.announce_wait_s >= 0 && s.announce_wait_s <= 5)) bad("site.conf: announce_wait_s must be in [0, 5] s");
  if (!(s.announce_max_s >= 1 && s.announce_max_s <= 10)) bad("site.conf: announce_max_s must be in [1, 10] s");
  if (!(s.announce_s_per_char >= 0.1 && s.announce_s_per_char <= 1.0)) bad("site.conf: announce_s_per_char must be in [0.1, 1.0] s");
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

bool load_modes(const std::string& path, std::vector<ModeSpec>& out, std::string& err) {
  KV kv;
  out.clear();
  if (!kv.load(path)) { err = "cannot read " + path; return false; }
  std::vector<std::string> order;
  {  // KV is a map: keep the file order for the menu
    std::string doc;
    read_file(path, doc);
    size_t p = 0;
    while (p < doc.size()) {
      size_t e = doc.find('\n', p);
      std::string line = doc.substr(p, e == std::string::npos ? std::string::npos : e - p);
      p = e == std::string::npos ? doc.size() : e + 1;
      if (line.rfind("mode_", 0) == 0) { std::string k = line.substr(0, line.find_first_of(" \t")); order.push_back(k); }
    }
  }
  for (auto& k : order) {
    auto& v = kv.m[k];
    if (v.size() < 5) { err = k + ": need <ref_mode> <run_s> <evidence 0|1> <voice> <description>"; return false; }
    ModeSpec m;
    m.name = k.substr(5);
    m.ref_mode = v[0];
    try { m.run_s = std::stod(v[1]); } catch (...) { err = k + ": bad run_s"; return false; }
    m.validated = v[2] == "1";
    m.voice = v[3];
    for (size_t i = 4; i < v.size(); i++) m.desc += (i > 4 ? " " : "") + v[i];
    m.check_only = m.ref_mode == "check";
    if (m.check_only) m.ref_mode = "paused";
    if (m.ref_mode != "paused" && m.ref_mode != "running") { err = k + ": ref_mode must be paused|running|check"; return false; }
    if (!(m.run_s > 0 && m.run_s <= ABS_MAX_RUN_S)) { err = k + ": run_s out of (0, " + std::to_string((int)ABS_MAX_RUN_S) + "]"; return false; }
    out.push_back(m);
  }
  if (out.empty()) { err = "no mode_* lines in " + path; return false; }
  return true;
}

ArtifactHashes hash_artifacts(const Paths& p) {
  ArtifactHashes h;
  std::string x;
  auto hf = [&](const std::string& path) { return sha256_file(path, x) ? x : std::string("MISSING"); };
  char b[4096];
  ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1);
  h.exe = n > 0 ? hf(std::string(b, (size_t)n)) : "UNKNOWN";
  h.policy = hf(p.asset("policy.onnx")); h.params = hf(p.asset("policy_params.txt")); h.motion = hf(p.asset("motion.bin")); h.fk_chain = hf(p.asset("fk_chain.txt"));
  h.site_conf = hf(p.config + "/site.conf"); h.modes_conf = hf(p.config + "/modes.conf");
  h.policy_ok = h.policy == FROZEN_SHA256_POLICY; h.params_ok = h.params == FROZEN_SHA256_POLICY_PARAMS; h.motion_ok = h.motion == FROZEN_SHA256_MOTION;
  return h;
}
std::string manifest_json(const Paths& p, const ArtifactHashes& h, const std::string& extra) {
  struct utsname u;
  uname(&u);
  char host[128] = {0};
  gethostname(host, sizeof host - 1);
  std::string s = "{" + jstr("schema", PM_LOG_SCHEMA) + "," + jstr("program_version", PM_VERSION) + "," + jstr("build_stamp", PM_BUILD_STAMP) + "," + jstr("policy_name", PM_POLICY_NAME) + "," +
                  jstr("root", p.root) + "," + jstr("host", host) + "," + jstr("machine", u.machine) + "," + jstr("kernel", u.release) + "," + jnum("wall_at_write", wall()) + "," + jnum("mono_at_write", mono()) + "," +
                  "\"sha256\":{" + jstr("executable", h.exe) + "," + jstr("policy_onnx", h.policy) + "," + jstr("policy_params", h.params) + "," + jstr("motion_bin", h.motion) + "," + jstr("fk_chain", h.fk_chain) + "," +
                  jstr("site_conf", h.site_conf) + "," + jstr("modes_conf", h.modes_conf) + "}," +
                  "\"frozen_expected\":{" + jstr("policy_onnx", FROZEN_SHA256_POLICY) + "," + jstr("policy_params", FROZEN_SHA256_POLICY_PARAMS) + "," + jstr("motion_bin", FROZEN_SHA256_MOTION) + "," + jstr("p2_baseline_policy_onnx", P2_BASELINE_SHA256_POLICY) + "}," +
                  jbool("policy_frozen_ok", h.policy_ok) + "," + jbool("params_frozen_ok", h.params_ok) + "," + jbool("motion_frozen_ok", h.motion_ok);
  if (!extra.empty()) s += "," + extra;
  return s + "}";
}

void fast_exit(int rc) {
  RunLogger::flush_all(0.5);  // P5.1: queued log records reach the disk even on the fast exit path
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
  if (k < doc.size() && doc[k] == '"') {  // P5.2: a string value may contain ',' or '}' (refusal reasons do): end at the closing unescaped quote
    size_t i = k + 1;
    while (i < doc.size() && doc[i] != '"') i += doc[i] == '\\' ? 2 : 1;
    return doc.substr(k, std::min(doc.size(), i + 1) - k);
  }
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
  else if (cmd == "logtool") rc = run_logtool(argc - 2, argv + 2, p);
  else { fprintf(stderr, "unknown command %s\n", cmd.c_str()); return 2; }
  fflush(stdout);
  fflush(stderr);
  _exit(rc);  // DDS reader threads can keep the process alive after main() returned
}
