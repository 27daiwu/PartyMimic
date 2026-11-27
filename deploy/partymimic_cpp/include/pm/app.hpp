// Shared glue of the single executable: package paths, site config, argument parsing, guardian heartbeat file, subcommand entry points.
#pragma once
#include <map>
#include <set>
#include <string>
#include <vector>

#include "pm/guard.hpp"
#include "pm/link.hpp"
#include "pm/session.hpp"
#include "pm/util.hpp"

namespace pm {

constexpr const char* DEFAULT_LOCK = "/tmp/partymimic_cpp_control.lock";
constexpr const char* DEFAULT_ESTOP = "/tmp/partymimic_cpp_ESTOP";
constexpr const char* GUARDIAN_ALIVE = "/tmp/partymimic_cpp_guardian.alive";
// Real-robot caps. LEGACY_* = the old hard-coded limits = the DEFAULT of the site.conf caps (max_run_s / max_authority). The site.conf caps can be raised, but never
// above the compile-time ABS_* limits; a run above the legacy values is flagged in the log, ready.json and the admission list.
constexpr double LEGACY_MAX_RUN_S = 1.0;
constexpr double LEGACY_MAX_AUTHORITY = 0.3;
// ABS_MAX_RUN_S: compile-time ceiling for ONE run window (seconds of dance). It was 12 s (the length of the first SSB test clips); a dance longer than that was cut.
// 60 s = a generous bound for a single full dance (~55 MB of run log, all timeouts derive from run_s). It is a CEILING only: the DEFAULT of site.conf max_run_s stays 12 s,
// a longer window needs an explicit site.conf max_run_s (<= ABS_MAX_RUN_S) (V1.6.1: no manual authorisation exists any more).
constexpr double ABS_MAX_RUN_S = 60.0;
constexpr double ABS_MAX_AUTHORITY = 1.0;
constexpr double GUARDIAN_MAX_AGE_S = 1.0;
// V1.6.1: the manual-authorisation gate (validated envelope / --accept-unvalidated / authz.conf) is GONE. The only motion limits left are the numeric caps (site.conf max_run_s / max_authority, compile-time ceilings above).
constexpr int SIM_DOMAIN = 77;                // --sim: loopback test robot only

struct Paths {
  std::string root, assets, config, logs;
  static Paths resolve();  // PM_ROOT env, else <dir of the executable>/..
  std::string asset(const std::string& n) const { return assets + "/" + n; }
};

// P4.1: ONE definition of which options are value-less flags per subcommand, used by BOTH the parser of the subcommand and the code that builds its command line
// (service -> guardian / runner). A flag missing from a parser's list is read as "--key value" and fails with "missing value for --flag": that is exactly the
// P4 field bug (service forwarded --accept-unvalidated to a guardian whose parser did not know it -> 178 restarts, F1+F3 refused).
// The guardian is also given the runner-only flags and IGNORES them (never forwarded any more, but accepted so the two can never disagree again).
inline const std::set<std::string>& flags_service() { static const std::set<std::string> s{"--sim", "--no-tts", "--allow-partial-authority", "--not-armed", "--torque-ff", "--no-torque-ff", "--no-standby"}; return s; }
inline const std::set<std::string>& flags_runner() { static const std::set<std::string> s{"--sim", "--allow-partial-authority", "--check-only", "--torque-ff", "--no-torque-ff", "--standby"}; return s; }
inline const std::set<std::string>& flags_guardian() { static const std::set<std::string> s{"--sim", "--allow-partial-authority"}; return s; }
struct GuardianCmd { std::string self, iface, report, entry_fsm; int domain = 0; bool sim = false; };
struct RunnerCmd {
  std::string self, iface, ref_mode, run_dir, go_file, status_file, entry_fsm, unconfirmed_hold_s, mode_name, trigger_file, announced_file, announce_wait_s;  // trigger_file (V1.5.1): standby runner (run_dir is empty then)
  int domain = 0; double run_s = 1.0, authority = 1.0, ramp_s = 0.3, end_hold_s = 0.5;
  bool sim = false, allow_partial = false, check_only = false, torque_ff = false, no_torque_ff = false, standby = false;  // check_only: read-only F1+F3 check, the runner can never take control
};
std::vector<std::string> guardian_cmdline(const GuardianCmd& c);  // argv incl. argv[0]; only options the guardian uses (runner-only options are NOT forwarded)
std::vector<std::string> runner_cmdline(const RunnerCmd& c);

struct Args {  // "--key value" / "--flag"
  std::map<std::string, std::string> kv;
  std::set<std::string> flags;
  std::string get(const std::string& k, const std::string& def = "") const { auto it = kv.find(k); return it == kv.end() ? def : it->second; }
  double num(const std::string& k, double def) const { auto it = kv.find(k); return it == kv.end() ? def : std::stod(it->second); }
  bool has(const std::string& f) const { return flags.count(f) != 0; }
  // flag_names: options that take no value. Returns false on malformed input.
  bool parse(int argc, char** argv, const std::set<std::string>& flag_names, std::string& err);
};

struct Site {
  std::vector<int> entry_allowed{500, 505, 801};
  GuardConfig guard;          // site.conf overrides (ranges enforced by validate())
  ExitConfig exit;
  double max_run_s = LEGACY_MAX_RUN_S, max_authority = LEGACY_MAX_AUTHORITY;  // caps for run_s / authority (<= ABS_*)
  double watchdog_s = 0.10, max_entry_tilt_rad = 0.5, quiescent_dq = 0.5, min_state_hz = 30.0;
  std::string err;                  // non-empty = site.conf rejected (runner refuses, service logs)
  std::vector<std::string> warnings;  // unknown keys etc.
  void apply_to(SessionConfig& s) const;
  bool oem_confirmed = false;
  int oem_expected_writers = -1;
  std::string iface;  // "" = auto
  double run_s = 1.0, authority = 0.3;
  double ramp_s = 0.3, end_hold_s = 0.5;  // P4: authority ramp 0 -> authority over ramp_s; after the motion window the reference stays paused this long before the hand-back
  bool allow_partial_authority = false;     // P4: sustained authority in (0.05, 0.9) is refused unless set (Sim2Sim: partial authority is the unstable regime)
  std::string ref_mode = "running";
  bool tts = true;
  int tts_volume = 100;
  // P5.1 logging (site.conf log_*): pre-take-over / post-exit state window, binary streams
  // P5.3: lead-in = the clip robust_d was TRAINED on starts with 1 s of frozen first frame (ssb_padded); running modes replay it in memory (motion.bin untouched, hash unchanged)
  double lead_in_s = 1.0;
  // P5.3: after a recoverable exit the service returns to IDLE once WalkRun is confirmed, the attitude has been quiet for settle_stable_s, the lock is free and the guardian is ready
  double settle_stable_s = 1.0, settle_timeout_s = 20.0;
  double log_preroll_s = 5.0, log_post_exit_s = 2.0;
  bool log_obs = true, log_state_stream = true;
  double log_min_free_mb = 100.0;   // admission FAIL below this free space in the log directory (the run could not be recorded)
  // V1.6 announce_wait: after L1+A the service says 开始执行; the runner takes over only after that announcement has FINISHED + announce_wait_s (the audio service gives no completion callback:
  // finish = TtsMaker returned + estimated speaking time, capped at announce_max_s after the announcement began; see EXIT_POLICY.md / README "开始执行播报后等待")
  double announce_wait_s = 0.5, announce_max_s = 4.0, announce_s_per_char = 0.30;
  static Site load(const std::string& path);
};

// P5.1/P5.2 test modes (config/modes.conf, one line per mode: "mode_<name> <ref_mode|check> <run_s> <evidence 0|1> <voice phrase> <description...>"). Modes are INDEPENDENT: no mode requires another.
// ONE definition for start.sh (menu, confirmation, order check), the service (run parameters, voice, validated envelope) and selfcheck.
struct ModeSpec {
  std::string name, ref_mode, voice, desc;
  double run_s = 3.0;
  bool validated = false;   // menu hint only: field evidence exists. It is never a start condition
  bool check_only = false;  // F1+F3 checks only, never armed
};
bool load_modes(const std::string& path, std::vector<ModeSpec>& out, std::string& err);

struct GuardianAlive {
  double t = 0;
  int pid = 0, user_writers = 0;
  std::string iface;
  std::string exit_phase = "NONE", exit_owner = "none";  // guardian-owned exit in progress (EXITING) / finished (DONE); the runner never starts a second one
  double exit_t = 0;
  static bool read(const std::string& path, GuardianAlive& g);
  static void write(const std::string& path, const GuardianAlive& g);
};

// DDS command-writer verdict. Counting writers is NOT a conflict test: the factory services keep idle writers on rt/arm_sdk and rt/user_lowcmd, and
// the factory motion service legitimately writes rt/lowcmd. Classification:
//   guardian (our own pre-built publisher, `own_user_writers`) / factory idle writers (no traffic in the window) / factory motion service (rt/lowcmd)
//   / ACTIVE competing controller = samples seen on rt/user_lowcmd or rt/arm_sdk during the window  -> FAIL (the only blocking outcome by default)
// rt/lowcmd: with a site.conf baseline (oem_lowcmd_confirmed 1 + oem_lowcmd_expected_writers N) a changed writer count FAILs; without a baseline it is a WARN.
struct WriterVerdict { int level; std::string detail; };
WriterVerdict oem_writer_verdict(const Writers& w, const std::map<std::string, uint64_t>& traffic, double window_s, int own_user_writers, int fsm, const Site& site);

// flush and _exit WITHOUT running destructors: tearing down the DDS participant / reader threads can crash or hang the exit path
[[noreturn]] void fast_exit(int rc);

int run_service(int argc, char** argv, const Paths& p);
int run_runner(int argc, char** argv, const Paths& p);
int run_guardian(int argc, char** argv, const Paths& p);
int run_release(int argc, char** argv, const Paths& p);
int run_selfcheck(int argc, char** argv, const Paths& p);
int run_logtool(int argc, char** argv, const Paths& p);  // finalize | check | replay  (offline, never touches DDS / the robot)

// P5.1 manifest: program / config / artifact hashes of one run (also used by the offline tool)
struct ArtifactHashes { std::string exe, policy, params, motion, fk_chain, site_conf, modes_conf; bool policy_ok = false, params_ok = false, motion_ok = false; };
ArtifactHashes hash_artifacts(const Paths& p);
std::string manifest_json(const Paths& p, const ArtifactHashes& h, const std::string& extra_members);  // extra_members: comma-separated JSON members (run parameters)

// small JSON helpers for reports (no JSON library on purpose)
std::string jstr(const std::string& k, const std::string& v);
std::string jnum(const std::string& k, double v);
std::string jbool(const std::string& k, bool v);
std::string json_find_string(const std::string& doc, const std::string& key);  // first "key":"value"
std::string json_find_raw(const std::string& doc, const std::string& key);     // first "key":<raw token>

}  // namespace pm
