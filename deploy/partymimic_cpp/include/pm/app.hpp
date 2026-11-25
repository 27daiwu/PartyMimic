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
constexpr double ABS_MAX_RUN_S = 12.0;
constexpr double ABS_MAX_AUTHORITY = 1.0;
constexpr double GUARDIAN_MAX_AGE_S = 1.0;
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
inline const std::set<std::string>& flags_service() { static const std::set<std::string> s{"--confirm-protection", "--sim", "--no-tts", "--accept-unvalidated", "--allow-partial-authority"}; return s; }
inline const std::set<std::string>& flags_runner() { static const std::set<std::string> s{"--i-confirm-protection", "--sim", "--allow-partial-authority", "--accept-unvalidated"}; return s; }
inline const std::set<std::string>& flags_guardian() { static const std::set<std::string> s{"--sim", "--allow-partial-authority", "--accept-unvalidated"}; return s; }
struct GuardianCmd { std::string self, iface, report, entry_fsm; int domain = 0; bool sim = false; };
struct RunnerCmd {
  std::string self, iface, ref_mode, run_dir, go_file, status_file, entry_fsm, unconfirmed_hold_s;
  int domain = 0; double run_s = 1.0, authority = 1.0, ramp_s = 0.3, end_hold_s = 0.5, ready_timeout_s = 120.0;
  bool sim = false, allow_partial = false, accept_unvalidated = false;
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
  static Site load(const std::string& path);
};

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

// small JSON helpers for reports (no JSON library on purpose)
std::string jstr(const std::string& k, const std::string& v);
std::string jnum(const std::string& k, double v);
std::string jbool(const std::string& k, bool v);
std::string json_find_string(const std::string& doc, const std::string& key);  // first "key":"value"
std::string json_find_raw(const std::string& doc, const std::string& key);     // first "key":<raw token>

}  // namespace pm
