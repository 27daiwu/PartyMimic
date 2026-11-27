// DEV ONLY (never shipped). V1.7 C3: ISOLATED check of the absolute hard tilt cap (unstable_hard_tilt_rad, rule HARD_CAP) and of FALLEN.
// The V1.6 A1 evidence for "the hard cap does not depend on the reference" was indirect (cheating / invalid references inside the REF_DEV test matrix). Here every other rule
// (REF_DEV, ABS_RATE (soft tilt + gyro / tilt rate), FIXED_LIMIT) is switched OFF or made unreachable, and only HARD_CAP + FALLEN remain.
// Two isolation configurations:
//   DIRECT : ExitConfig built in code (bypasses validate(): unstable_tilt_rad = soft = 9, gyro = 1e9, ref_dev_rad = 4)  -> the other rules are truly unreachable
//   SITE   : a site.conf that passes the real range checks: ref_aware_unstable 1, ref_dev_min_tilt_rad = hard cap (REF_DEV can only fire above the cap, where HARD_CAP wins), unstable_tilt_rad = soft = hard - 1e-3,
//            unstable_gyro_rad_s 10, unstable_tilt_rate_rad_s 0.   (documented residual: FIXED_LIMIT / ABS_RATE cannot be removed through site.conf, only pushed to within 1 mrad of the cap)
// No logic, no default value of exit_policy.cpp / site.conf is changed by this file. Usage: hard_cap_isolation <package root>  (needs assets/policy_params.txt for the SafetyGuard independence cases)
#include <unistd.h>

#include <cmath>
#include <cstdarg>
#include <memory>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "pm/app.hpp"
#include "pm/exit_policy.hpp"
#include "pm/guard.hpp"
#include "pm/policy.hpp"
using namespace pm;

static int g_pass = 0, g_fail = 0;
static void rec(const char* id, const char* what, bool ok, const std::string& detail = "") {
  printf("[%s] %-6s %s%s%s\n", ok ? "PASS" : "FAIL", id, what, detail.empty() ? "" : " :: ", detail.c_str());
  (ok ? g_pass : g_fail)++;
}
static std::string fmt(const char* f, ...) { char b[512]; va_list a; va_start(a, f); vsnprintf(b, sizeof b, f, a); va_end(a); return b; }

// ---------------------------------------------------------------- helpers
using Q = std::array<double, 4>;
static Q roll_q(double r) { return {std::cos(r / 2), std::sin(r / 2), 0, 0}; }           // tilt == r (0..pi)
static Q rollyaw_q(double r, double yaw) { double a = r / 2, b = yaw / 2; return {std::cos(b) * std::cos(a), std::cos(b) * std::sin(a), std::sin(b) * std::sin(a), std::sin(b) * std::cos(a)}; }
static std::array<float, 4> f4(const Q& q) { return {(float)q[0], (float)q[1], (float)q[2], (float)q[3]}; }

struct Samp { uint64_t seq; double t; double tilt; double gyro = 0.2; bool nan_quat = false; };
struct Res { Stability last = Stability::OK; int first_idx = -1; int first_fallen_idx = -1; TiltRule rule = TiltRule::NONE; double trig_tilt = NAN; double max_tilt = 0; };

// reference behaviours
enum class RefMode { NONE_CHANNEL, VALID_UPRIGHT, VALID_FOLLOWING_0_5, INVALID_FLAG, NAN_QUAT, STALE, REF_EQUALS_MEAS, REF_ABOVE_CAP_0_8, REF_FLICKER, REF_MIRROR_CHEAT_TILT };
static const char* ref_name(RefMode m) {
  switch (m) {
    case RefMode::NONE_CHANNEL: return "no reference channel"; case RefMode::VALID_UPRIGHT: return "valid fresh ref, upright"; case RefMode::VALID_FOLLOWING_0_5: return "valid fresh ref leaning 0.50 (dance peak)";
    case RefMode::INVALID_FLAG: return "reference marked invalid"; case RefMode::NAN_QUAT: return "reference quaternion NaN"; case RefMode::STALE: return "reference stale (5 s old)";
    case RefMode::REF_EQUALS_MEAS: return "reference == measured every sample (cheat, error 0)"; case RefMode::REF_ABOVE_CAP_0_8: return "reference tilt 0.80 > cap (valid)";
    case RefMode::REF_FLICKER: return "reference valid/invalid flickers every sample"; default: return "";
  }
}

static Res run(const ExitConfig& c, RefMode rm, const std::vector<Samp>& v) {
  RefChannel ch;
  StabilityMonitor m(c, rm == RefMode::NONE_CHANNEL ? nullptr : &ch, "iso");
  Res r;
  for (size_t i = 0; i < v.size(); i++) {
    const Samp& s = v[i];
    double age = rm == RefMode::STALE ? 5.0 : 0.002;
    Q q = s.nan_quat ? Q{NAN, 0, 0, 0} : roll_q(s.tilt);
    switch (rm) {
      case RefMode::VALID_UPRIGHT: ch.set(true, f4(roll_q(0.0)), s.t - age, i); break;
      case RefMode::VALID_FOLLOWING_0_5: ch.set(true, f4(roll_q(0.5)), s.t - age, i); break;
      case RefMode::INVALID_FLAG: ch.set(false, f4(roll_q(0.2)), s.t - age, i); break;
      case RefMode::NAN_QUAT: ch.set(true, std::array<float, 4>{NAN, 0, 0, 0}, s.t - age, i); break;
      case RefMode::STALE: ch.set(true, f4(roll_q(0.0)), s.t - age, i); break;
      case RefMode::REF_EQUALS_MEAS: ch.set(true, f4(rollyaw_q(s.nan_quat ? 0 : s.tilt, 0.3)), s.t - age, i); break;
      case RefMode::REF_ABOVE_CAP_0_8: ch.set(true, f4(roll_q(0.8)), s.t - age, i); break;
      case RefMode::REF_FLICKER: ch.set(i % 2 == 0, f4(roll_q(0.4)), s.t - age, i); break;
      default: break;
    }
    LinkState st; st.seq = s.seq; st.arrival = s.t; st.quat = q; st.gyro = {s.gyro, 0, 0};
    Stability lv = m.update(st);
    r.last = lv;
    if (lv != Stability::OK && r.first_idx < 0) { r.first_idx = (int)i; r.rule = m.trigger().rule; r.trig_tilt = m.trigger().tilt; }
    if (lv == Stability::FALLEN && r.first_fallen_idx < 0) r.first_fallen_idx = (int)i;
  }
  r.max_tilt = m.max_tilt;
  return r;
}
static std::vector<Samp> steady(double tilt, int n, uint64_t seq0 = 1, double t0 = 10.0, double dt = 0.002, double gyro = 0.2) { std::vector<Samp> v; for (int i = 0; i < n; i++) v.push_back({seq0 + (uint64_t)i, t0 + dt * i, tilt, gyro}); return v; }
static std::vector<Samp> cat(std::vector<Samp> a, const std::vector<Samp>& b) { a.insert(a.end(), b.begin(), b.end()); return a; }

static ExitConfig direct_cfg() {
  ExitConfig c;                       // defaults: hard 0.70, fallen 1.0, debounce 2, ref_max_age 0.15
  c.unstable_tilt_rad = 9.0;          // FIXED_LIMIT unreachable
  c.unstable_soft_tilt_rad = 9.0;     // ABS_RATE unreachable (tilt never > 9)
  c.unstable_gyro_rad_s = 1e9;
  c.unstable_tilt_rate_rad_s = 0.0;   // tilt-rate criterion OFF
  c.ref_dev_rad = 4.0;                // acos() <= pi: REF_DEV unreachable
  c.ref_dev_min_tilt_rad = 9.0;
  return c;
}
static const char* TMP_CONF = "/tmp/pm_c3_site.conf";
static Site site_load(const std::string& body) { std::ofstream f(TMP_CONF); f << body; f.close(); return Site::load(TMP_CONF); }
static const char* SITE_ISO_CONF =
    "ref_aware_unstable 1\nunstable_hard_tilt_rad 0.70\nfallen_tilt_rad 1.0\nunstable_tilt_rad 0.699\nunstable_soft_tilt_rad 0.699\nunstable_gyro_rad_s 10\nunstable_tilt_rate_rad_s 0\n"
    "ref_dev_rad 1.0\nref_dev_min_tilt_rad 0.70\nref_dev_growth_rad_s 10\nunstable_debounce_samples 2\n";

static bool dec_is(const ExitDecision& d, const char* mode, const char* cls) { return d.mode == mode && d.cls == cls; }

int main(int argc, char** argv) {
  const std::string root = argc > 1 ? argv[1] : ".";
  const ExitConfig D = direct_cfg();
  Site S = site_load(SITE_ISO_CONF);
  printf("# HARD_CAP isolation. SITE conf load err='%s' warnings=%zu  hard=%.3f fallen=%.3f unstable=%.4f soft=%.4f gyro=%.1f rate=%.1f ref_dev_min=%.2f\n", S.err.c_str(), S.warnings.size(), S.exit.unstable_hard_tilt_rad,
         S.exit.fallen_tilt_rad, S.exit.unstable_tilt_rad, S.exit.unstable_soft_tilt_rad, S.exit.unstable_gyro_rad_s, S.exit.unstable_tilt_rate_rad_s, S.exit.ref_dev_min_tilt_rad);
  rec("I0", "SITE isolation config passes the real site.conf range checks (no err)", S.err.empty(), S.err);
  std::string verr;
  rec("I0b", "DIRECT isolation config is OUT of the validated ranges on purpose (validate rejects it: proves it bypasses the checks)", !validate(D, verr), verr);
  struct Cfg { const char* name; ExitConfig c; double lim; };
  const Cfg cfgs[] = {{"DIRECT", D, 0.70}, {"SITE", S.exit, 0.70}};

  // ====================================================================== (a) threshold + 2-sample debounce
  for (const Cfg& k : cfgs) {
    const ExitConfig& c = k.c;
    auto tag = [&](const char* id) { return std::string(id) + "." + k.name; };
    auto R = [&](const char* id, const char* what, bool ok, const std::string& d = "") { rec(tag(id).c_str(), what, ok, d); };
    // 0.69 never triggers (long steady), also with gyro 100 rad/s (ABS_RATE would fire if it were on) and a 0.69 -> 0.0 -> 0.69 sawtooth
    Res r = run(c, RefMode::NONE_CHANNEL, steady(0.69, 2000));
    R("a1", "steady 0.69 rad x2000 samples: OK, no trigger", r.last == Stability::OK && r.first_idx < 0, fmt("max_tilt=%.4f", r.max_tilt));
    r = run(c, RefMode::NONE_CHANNEL, steady(0.69, 500, 1, 10.0, 0.002, 100.0));
    R("a2", "0.69 with |gyro| 100 rad/s: still OK (the gyro rule is not a cause)", r.last == Stability::OK && r.first_idx < 0);
    const double just_below = std::strcmp(k.name, "SITE") == 0 ? 0.6985 : 0.6999;   // SITE: FIXED_LIMIT sits at 0.699 (cannot be removed via site.conf, see e13)
    r = run(c, RefMode::NONE_CHANNEL, steady(just_below, 500));
    R("a3", "just below the cap (DIRECT 0.6999 / SITE 0.6985): OK", r.last == Stability::OK && r.first_idx < 0);
    r = run(c, RefMode::NONE_CHANNEL, steady(0.7001, 500));
    R("a4", "0.7001 (0.1 mrad above): UNSTABLE on the 2nd sample, rule HARD_CAP", r.first_idx == 1 && r.last == Stability::UNSTABLE && r.rule == TiltRule::HARD, fmt("idx=%d rule=%s", r.first_idx, tilt_rule_name(r.rule)));
    r = run(c, RefMode::NONE_CHANNEL, steady(0.75, 1));
    R("a5", "single 0.75 sample: debounce -> OK (needs 2 distinct samples)", r.last == Stability::OK && r.first_idx < 0);
    r = run(c, RefMode::NONE_CHANNEL, cat(steady(0.3, 5), cat(steady(0.75, 1, 6, 10.01), cat(steady(0.3, 1, 7, 10.012), cat(steady(0.75, 1, 8, 10.014), steady(0.3, 5, 9, 10.016))))));
    R("a6", "isolated 0.75 spikes separated by a 0.30 sample: never triggers (consecutive requirement)", r.first_idx < 0 && r.last == Stability::OK);
    r = run(c, RefMode::NONE_CHANNEL, cat(steady(0.3, 5), steady(0.75, 2, 6, 10.01)));
    R("a7", "two consecutive 0.75: UNSTABLE exactly at the 2nd (index 6 of 7)", r.first_idx == 6 && r.rule == TiltRule::HARD);
    {
      StabilityMonitor m(c, nullptr, "iso");
      std::string j;
      for (int i = 0; i < 4; i++) { LinkState s; s.seq = i + 1; s.arrival = 10.0 + 0.002 * i; s.quat = roll_q(0.75); m.update(s); }
      j = m.trigger_json();
      R("a8", "trigger_json / detail name HARD_CAP and the 0.70 limit (log record for the post-mortem)", j.find("\"rule\":\"HARD_CAP\"") != std::string::npos && j.find("\"hard_tilt_rad\":0.700") != std::string::npos && m.detail().find("unstable[HARD_CAP]") != std::string::npos, m.detail().substr(0, 60));
    }
    // debounce 1 and 3 (range 1..10)
    for (int nd : {1, 3, 10}) {
      ExitConfig cc = c; cc.unstable_debounce_samples = nd;
      Res rr = run(cc, RefMode::NONE_CHANNEL, steady(0.8, 20));
      R(nd == 1 ? "a9" : nd == 3 ? "a10" : "a11", fmt("debounce_samples=%d: trigger at the %d-th distinct sample", nd, nd).c_str(), rr.first_idx == nd - 1 && rr.rule == TiltRule::HARD, fmt("idx=%d", rr.first_idx));
    }
    // legacy ref_aware = false does not change the cap
    { ExitConfig cc = c; cc.ref_aware = false; Res rr = run(cc, RefMode::VALID_FOLLOWING_0_5, steady(0.75, 10)); Res r2 = run(cc, RefMode::VALID_FOLLOWING_0_5, steady(0.69, 100));
      R("a12", "ref_aware=false (legacy): 0.75 -> HARD_CAP, 0.69 -> OK", rr.first_idx == 1 && rr.rule == TiltRule::HARD && r2.first_idx < 0); }
    // yaw does not matter (tilt is yaw free): roll 0.75 with arbitrary yaw
    { StabilityMonitor m(c, nullptr, "iso"); Stability v = Stability::OK; for (int i = 0; i < 3; i++) { LinkState s; s.seq = i + 1; s.arrival = 10 + 0.002 * i; s.quat = rollyaw_q(0.75, 2.0); v = m.update(s); }
      R("a13", "yaw-free: roll 0.75 with yaw 2.0 rad -> HARD_CAP", v == Stability::UNSTABLE && m.trigger().rule == TiltRule::HARD); }
  }

  // ====================================================================== (b) reference channel states cannot change the outcome
  for (const Cfg& k : cfgs) {
    const RefMode modes[] = {RefMode::NONE_CHANNEL, RefMode::VALID_UPRIGHT, RefMode::VALID_FOLLOWING_0_5, RefMode::INVALID_FLAG, RefMode::NAN_QUAT, RefMode::STALE, RefMode::REF_EQUALS_MEAS, RefMode::REF_ABOVE_CAP_0_8, RefMode::REF_FLICKER};
    int n = 0;
    for (RefMode rm : modes) {
      n++;
      char id[32]; snprintf(id, sizeof id, "b%d.%s", n, k.name);
      Res hi = run(k.c, rm, steady(0.75, 400));
      Res lo = run(k.c, rm, steady(0.69, 400));
      Res ramp; { std::vector<Samp> v; for (int i = 0; i < 600; i++) v.push_back({(uint64_t)i + 1, 10.0 + 0.002 * i, 0.3 + 0.001 * i}); ramp = run(k.c, rm, v); }
      const bool okhi = hi.first_idx == 1 && hi.rule == TiltRule::HARD && hi.last == Stability::UNSTABLE;
      const bool oklo = lo.first_idx < 0 && lo.last == Stability::OK;
      const bool okramp = ramp.rule == TiltRule::HARD && ramp.trig_tilt > 0.700 && ramp.trig_tilt < 0.7025 && ramp.first_fallen_idx < 0;
      rec(id, (std::string("ref: ") + ref_name(rm) + " -> 0.75 trips HARD_CAP @2nd sample, 0.69 never trips, ramp trips at 0.70-0.7025 before FALLEN").c_str(), okhi && oklo && okramp,
          fmt("hi idx=%d rule=%s | lo idx=%d | ramp trip tilt=%.4f rule=%s", hi.first_idx, tilt_rule_name(hi.rule), lo.first_idx, ramp.trig_tilt, tilt_rule_name(ramp.rule)));
    }
    // reference quaternion mirrors the measurement (even at 0.69 / 0.75): "cheating" reference gets no leniency
    { Res r = run(k.c, RefMode::REF_EQUALS_MEAS, steady(0.75, 50)); Res r2 = run(k.c, RefMode::REF_EQUALS_MEAS, steady(0.69, 50));
      char id[32]; snprintf(id, sizeof id, "b10.%s", k.name);
      rec(id, "cheating reference == measurement: gravity error 0 yet HARD_CAP still fires at 0.75 and not at 0.69", r.first_idx == 1 && r.rule == TiltRule::HARD && r2.first_idx < 0); }
    // the reference changing validity mid-stream around the trigger sample
    { std::vector<Samp> v = steady(0.75, 6); RefChannel ch; StabilityMonitor m(k.c, &ch, "iso"); Stability vv = Stability::OK; int idx = -1;
      for (int i = 0; i < 6; i++) { ch.set(i < 1, f4(roll_q(0.75)), v[i].t, i); LinkState s; s.seq = v[i].seq; s.arrival = v[i].t; s.quat = roll_q(0.75); vv = m.update(s); if (vv != Stability::OK && idx < 0) idx = i; }
      char id[32]; snprintf(id, sizeof id, "b11.%s", k.name);
      rec(id, "reference valid on the 1st sample only, invalid afterwards: HARD_CAP at the 2nd sample", idx == 1 && m.trigger().rule == TiltRule::HARD); }
  }

  // ====================================================================== (c) hard cap fires before fallen (1.0)
  for (const Cfg& k : cfgs) {
    auto id = [&](const char* s) { return std::string(s) + "." + k.name; };
    // slow ramp 0.3 .. 1.2 rad over 2 s, 500 Hz
    std::vector<Samp> v; for (int i = 0; i < 1000; i++) v.push_back({(uint64_t)i + 1, 10.0 + 0.002 * i, 0.3 + 0.0009 * i});
    Res r = run(k.c, RefMode::NONE_CHANNEL, v);
    rec(id("c1").c_str(), "ramp 0.3->1.2 rad (0.45 rad/s): UNSTABLE first (tilt 0.70+), FALLEN only later (>1.0), UNSTABLE comes >= 0.25 rad earlier",
        r.first_idx >= 0 && r.rule == TiltRule::HARD && r.first_fallen_idx > r.first_idx && r.trig_tilt < 0.71, fmt("unstable idx=%d tilt=%.4f, fallen idx=%d (tilt~%.3f)", r.first_idx, r.trig_tilt, r.first_fallen_idx, 0.3 + 0.0009 * r.first_fallen_idx));
    // fast ramp: 25 rad/s (0.05 rad per 2 ms sample)
    v.clear(); for (int i = 0; i < 40; i++) v.push_back({(uint64_t)i + 1, 10.0 + 0.002 * i, 0.3 + 0.05 * i});
    r = run(k.c, RefMode::NONE_CHANNEL, v);
    rec(id("c2").c_str(), "fast fall 25 rad/s: UNSTABLE(HARD_CAP) is reported on the sample pair straddling 0.70, before the sample pair above 1.0", r.first_idx >= 0 && r.rule == TiltRule::HARD && r.first_fallen_idx > r.first_idx,
        fmt("unstable idx=%d (tilt %.2f), fallen idx=%d", r.first_idx, r.trig_tilt, r.first_fallen_idx));
    // jump 0.5 -> 0.85 -> 1.2 -> 1.2
    r = run(k.c, RefMode::NONE_CHANNEL, {{1, 10.000, 0.5}, {2, 10.002, 0.85}, {3, 10.004, 1.2}, {4, 10.006, 1.2}});
    rec(id("c3").c_str(), "0.5 -> 0.85 -> 1.2 -> 1.2: UNSTABLE at sample 3 (0.85 & 1.2 both above the cap), FALLEN at sample 4", r.first_idx == 2 && r.first_fallen_idx == 3 && r.last == Stability::FALLEN, fmt("unstable idx=%d fallen idx=%d", r.first_idx, r.first_fallen_idx));
    // 0.9999 sustained: UNSTABLE (HARD_CAP) never FALLEN; 1.0001: FALLEN
    r = run(k.c, RefMode::NONE_CHANNEL, steady(0.9999, 50));
    Res r2 = run(k.c, RefMode::NONE_CHANNEL, steady(1.0001, 50));
    rec(id("c4").c_str(), "0.9999 sustained -> UNSTABLE(HARD_CAP) not FALLEN; 1.0001 sustained -> FALLEN at the 2nd sample", r.last == Stability::UNSTABLE && r.first_fallen_idx < 0 && r2.first_fallen_idx == 1 && r2.last == Stability::FALLEN,
        fmt("0.9999: last=%d fallen_idx=%d; 1.0001: fallen_idx=%d", (int)r.last, r.first_fallen_idx, r2.first_fallen_idx));
  }

  // ====================================================================== (d) tilt > 1.0 -> PASSIVE path (decide_exit) ; attitude validity
  {
    const ExitConfig& c = D;
    auto att = [](bool valid, double tilt, double gyro = 0.5) { Attitude a; a.valid = valid; a.tilt = tilt; a.gyro_xy = gyro; a.age_s = valid ? 0.003 : 5.0; return a; };
    auto dec = [&](Fault f, bool valid, double tilt, bool normal_end = false) { ExitContext x; x.fault = f; x.normal_end = normal_end; x.att = att(valid, tilt); return decide_exit(x, c); };
    rec("d1", "Fault::TILT (monitor FALLEN) -> PASSIVE / FALLEN", dec_is(dec(Fault::TILT, true, 1.1), "PASSIVE", "FALLEN"));
    rec("d2", "Fault::UNSTABLE with attitude tilt 0.75 (< fallen) -> WALKRUN / UNSTABLE (hard cap hands back, no damping)", dec_is(dec(Fault::UNSTABLE, true, 0.75), "WALKRUN", "UNSTABLE"), dec(Fault::UNSTABLE, true, 0.75).why);
    rec("d3", "Fault::UNSTABLE but the robot is already at tilt 1.1 (> fallen) when the decision is made -> PASSIVE / FALLEN (decision table is the 2nd line of defence)", dec_is(dec(Fault::UNSTABLE, true, 1.1), "PASSIVE", "FALLEN"));
    rec("d4", "Fault::UNSTABLE with the attitude stale/unknown -> WALKRUN / UNTRUSTED (ask the factory WalkRun anyway)", dec_is(dec(Fault::UNSTABLE, false, 0.0), "WALKRUN", "UNTRUSTED"));
    rec("d5", "COMM_TIMEOUT (lowstate silent) with stale attitude -> WALKRUN / UNTRUSTED; with a valid tilt 1.1 -> PASSIVE", dec_is(dec(Fault::COMM_TIMEOUT, false, 0.0), "WALKRUN", "UNTRUSTED") && dec_is(dec(Fault::COMM_TIMEOUT, true, 1.1), "PASSIVE", "FALLEN"));
    rec("d6", "tilt exactly at fallen (1.0) is NOT yet down (strict >); 1.0001 is", dec_is(dec(Fault::UNSTABLE, true, 1.0), "WALKRUN", "UNSTABLE") && dec_is(dec(Fault::UNSTABLE, true, 1.0001), "PASSIVE", "FALLEN"));
    rec("d7", "emergency channels stay PASSIVE and never WalkRun (ESTOP / REMOTE_DAMP)", dec(Fault::ESTOP, true, 0.1).emergency && dec(Fault::REMOTE_DAMP, true, 0.1).emergency);
    // attitude_of: age gating (lowstate expired)
    auto st = std::make_shared<LinkState>(); st->arrival = 100.0; st->quat = roll_q(0.75); st->gyro = {0.1, 0.1, 0};
    Attitude fresh = attitude_of(st, 100.05, 0.10), old = attitude_of(st, 100.20, 0.10), none = attitude_of(nullptr, 100.0, 0.10);
    rec("d8", "attitude_of: 50 ms old = valid (tilt 0.75), 200 ms old (> attitude_max_age_s 0.10) = NOT valid, null state = not valid", fresh.valid && std::fabs(fresh.tilt - 0.75) < 1e-9 && !old.valid && !none.valid, fmt("tilt=%.4f age=%.3f/%.3f", fresh.tilt, fresh.age_s, old.age_s));
  }

  // ====================================================================== (e) debounce / duplicates / time-stamp / expiry boundaries (monitor level)
  for (const Cfg& k : cfgs) {
    auto id = [&](const char* s) { return std::string(s) + "." + k.name; };
    Res r;
    // duplicates: same seq repeated N times is ONE sample
    r = run(k.c, RefMode::NONE_CHANNEL, {{5, 10.000, 0.75}, {5, 10.002, 0.75}, {5, 10.004, 0.75}, {5, 10.006, 0.75}});
    rec(id("e1").c_str(), "same seq x4 (DDS re-delivery of one sample): counts as ONE sample -> no trigger", r.first_idx < 0 && r.last == Stability::OK);
    r = run(k.c, RefMode::NONE_CHANNEL, {{5, 10.000, 0.75}, {5, 10.002, 0.75}, {6, 10.004, 0.75}});
    rec(id("e2").c_str(), "dup then a new seq: triggers at the 2nd DISTINCT sample (index 2)", r.first_idx == 2 && r.rule == TiltRule::HARD);
    // duplicates in between do not reset the counter
    r = run(k.c, RefMode::NONE_CHANNEL, {{5, 10.000, 0.75}, {5, 10.002, 0.2}, {6, 10.004, 0.75}});
    rec(id("e3").c_str(), "dup of seq 5 carrying a benign payload between two hard samples: ignored (the duplicate is dropped before parsing) -> trigger at index 2", r.first_idx == 2);
    // seq going backwards: counts as distinct (monitor only compares for equality)
    r = run(k.c, RefMode::NONE_CHANNEL, {{9, 10.000, 0.75}, {4, 10.002, 0.75}});
    rec(id("e4").c_str(), "FINDING (benign): seq going backwards (9 -> 4) counts as a new distinct sample -> trigger (monitor uses '!=' not '>')", r.first_idx == 1, "documented; Link assigns monotonically increasing seq");
    // timestamp regression
    r = run(k.c, RefMode::NONE_CHANNEL, {{1, 10.010, 0.75}, {2, 10.000, 0.75}, {3, 9.990, 0.75}});
    rec(id("e5").c_str(), "arrival time going backwards (10.010 -> 10.000 -> 9.990): no crash, trigger at the 2nd sample (hard rule is time independent)", r.first_idx == 1 && r.rule == TiltRule::HARD);
    { std::vector<Samp> v = steady(0.3, 30); for (int i = 0; i < 30; i++) v.push_back({(uint64_t)(100 + i), 5.0 - 0.002 * i, 0.69, 0.2}); Res rr = run(k.c, RefMode::NONE_CHANNEL, v);
      rec(id("e6").c_str(), "regressing clock (jump back 5 s) with 0.69 tilt: no spurious trigger (rate history is time-robust)", rr.first_idx < 0, fmt("idx=%d", rr.first_idx)); }
    // very large gap between the 2 hard samples (lowstate gap 5 s): the monitor has no age gate -> still the 2nd distinct sample
    r = run(k.c, RefMode::NONE_CHANNEL, {{1, 10.0, 0.75}, {2, 15.0, 0.75}});
    rec(id("e7").c_str(), "FINDING (by design): 2 hard samples 5 s apart still debounce (no time window). Lowstate EXPIRY is handled outside the monitor (guard STATE_STALE / COMM_TIMEOUT -> decide_exit UNTRUSTED)", r.first_idx == 1);
    // invalid sample between two hard samples
    r = run(k.c, RefMode::NONE_CHANNEL, {{1, 10.000, 0.75}, {2, 10.002, 0.75, 0.2, true}, {3, 10.004, 0.75}});
    rec(id("e8").c_str(), "FINDING: a NaN-quaternion sample between two hard samples is skipped without resetting the counter -> trigger at the 3rd (still 2 distinct VALID hard samples)", r.first_idx == 2, fmt("idx=%d", r.first_idx));
    r = run(k.c, RefMode::NONE_CHANNEL, steady(0.75, 100, 1, 10.0, 0.002, NAN));
    rec(id("e9").c_str(), "NaN gyro with a hard tilt: sample is treated invalid (monitor does not trigger; STATE_INVALID guard fault + untrusted WalkRun path own this)", r.first_idx < 0);
    { std::vector<Samp> v(10); for (int i = 0; i < 10; i++) v[i] = {(uint64_t)i + 1, 10.0 + 0.002 * i, 0.75, 0.2, true}; r = run(k.c, RefMode::NONE_CHANNEL, v);
      rec(id("e10").c_str(), "all-NaN quaternion stream: monitor stays OK (not its job; guard check_state returns STATE_INVALID, see f5)", r.first_idx < 0 && r.last == Stability::OK); }
    // non-latching: after a trigger a benign sample returns OK (the latch is trigger_exit/ExitExecutor first-wins, not the monitor)
    r = run(k.c, RefMode::NONE_CHANNEL, cat(steady(0.75, 3), steady(0.2, 3, 10, 10.01)));
    rec(id("e11").c_str(), "monitor is non-latching: trigger then benign samples -> level back to OK, trigger() record kept (latching = ExitExecutor::submit first-wins)", r.first_idx == 1 && r.last == Stability::OK);
    // unnormalised quaternion: n in (0.9,1.1) normalised, otherwise invalid
    { StabilityMonitor m(k.c, nullptr, "iso"); Stability v = Stability::OK; for (int i = 0; i < 3; i++) { LinkState s; s.seq = i + 1; s.arrival = 10 + 0.002 * i; Q q = roll_q(0.75); for (auto& x : q) x *= 1.05; s.quat = q; v = m.update(s); }
      Stability v2 = Stability::OK; StabilityMonitor m2(k.c, nullptr, "iso2"); for (int i = 0; i < 3; i++) { LinkState s; s.seq = i + 1; s.arrival = 10 + 0.002 * i; Q q = roll_q(0.75); for (auto& x : q) x *= 1.3; s.quat = q; v2 = m2.update(s); }
      rec(id("e12").c_str(), "quaternion norm 1.05 is renormalised (still trips at 0.75); norm 1.3 is rejected as invalid (no trigger, guard's STATE_INVALID)", v == Stability::UNSTABLE && v2 == Stability::OK); }
    // boundary between the SITE-only residual window: in (0.699,0.70] FIXED_LIMIT may fire
    if (std::strcmp(k.name, "SITE") == 0) {
      r = run(k.c, RefMode::NONE_CHANNEL, steady(0.6995, 50));
      Res r2 = run(k.c, RefMode::VALID_FOLLOWING_0_5, steady(0.6995, 50));
      rec("e13.SITE", "RESIDUAL (site.conf cannot remove FIXED_LIMIT): 0.6995 with no valid reference fires FIXED_LIMIT (<=1 mrad below the cap) but NOT with a valid reference; 0.69 stays OK", r.first_idx == 1 && r.rule == TiltRule::LIMIT && r2.first_idx < 0,
          fmt("no-ref rule=%s, valid-ref idx=%d", tilt_rule_name(r.rule), r2.first_idx));
    }
  }

  // ====================================================================== (f) independence from guard max_tilt_rad / max_gyro_rad_s and from ABS_RATE inputs
  {
    Params p; std::string perr;
    const bool have_params = Params::load(root + "/assets/policy_params.txt", p, perr);
    rec("f0", "policy_params.txt loaded for the SafetyGuard cases", have_params, perr);
    if (have_params) {
      Vec q = p.default_pos, dq = Vec::Zero(), tau = Vec::Zero();
      GuardConfig gdef;
      auto gchk = [&](const GuardConfig& gc, double tilt, double gyro) { SafetyGuard g(p, gc); return g.check_state(q, dq, tau, roll_q(tilt), {gyro, 0, 0}, true, 1, 0.0, 0.0); };
      // guard (default max_tilt 1.0 / max_gyro 15) does not know the 0.70 cap
      rec("f1", "SafetyGuard default (max_tilt_rad 1.0, max_gyro 15): tilt 0.75 -> NONE while the isolated monitor says UNSTABLE (two independent mechanisms)", gchk(gdef, 0.75, 0.2) == Fault::NONE && run(D, RefMode::NONE_CHANNEL, steady(0.75, 5)).first_idx == 1);
      rec("f2", "SafetyGuard: tilt 1.05 -> Fault::TILT (own threshold) ; monitor reports FALLEN as well", gchk(gdef, 1.05, 0.2) == Fault::TILT && run(D, RefMode::NONE_CHANNEL, steady(1.05, 5)).last == Stability::FALLEN);
      rec("f3", "SafetyGuard: |gyro| 16 at tilt 0.1 -> Fault::GYRO; isolated monitor (HARD_CAP only) stays OK", gchk(gdef, 0.1, 16.0) == Fault::GYRO && run(D, RefMode::NONE_CHANNEL, steady(0.1, 20, 1, 10.0, 0.002, 16.0)).first_idx < 0);
      // moving guard thresholds does not move the monitor and vice versa
      GuardConfig g06 = gdef; g06.max_tilt_rad = 0.6; GuardConfig g15 = gdef; g15.max_tilt_rad = 1.5;
      rec("f4", "max_tilt_rad 0.6 / 1.5 only moves the guard: guard(0.65)=TILT@0.6, NONE@1.5; monitor trips at 0.70 in both (config coupling absent)",
          gchk(g06, 0.65, 0.2) == Fault::TILT && gchk(g15, 0.75, 0.2) == Fault::NONE && run(D, RefMode::NONE_CHANNEL, steady(0.65, 100)).first_idx < 0 && run(D, RefMode::NONE_CHANNEL, steady(0.75, 5)).first_idx == 1);
      ExitConfig h50 = D; h50.unstable_hard_tilt_rad = 0.55; ExitConfig h90 = D; h90.unstable_hard_tilt_rad = 0.90;
      rec("f4b", "unstable_hard_tilt_rad 0.55 / 0.90 moves only the monitor: guard(0.75)=NONE in both, monitor trips 0.60@0.55 and not at 0.85@... (0.85 <0.90 OK, 0.95 trips)",
          gchk(gdef, 0.75, 0.2) == Fault::NONE && run(h50, RefMode::NONE_CHANNEL, steady(0.60, 5)).first_idx == 1 && run(h90, RefMode::NONE_CHANNEL, steady(0.85, 50)).first_idx < 0 && run(h90, RefMode::NONE_CHANNEL, steady(0.95, 5)).first_idx == 1);
      SafetyGuard gn(p);
      rec("f5", "SafetyGuard on a NaN quaternion -> STATE_INVALID (the stream the monitor skips is caught here)", gn.check_state(q, dq, tau, {NAN, 0, 0, 0}, {0, 0, 0}, true, 1, 0.0, 0.0) == Fault::STATE_INVALID);
      SafetyGuard gs(p);
      rec("f6", "SafetyGuard: lowstate 0.2 s old -> COMM_TIMEOUT (expiry is the guard's rule, independent of tilt)", gs.check_state(q, dq, tau, roll_q(0.1), {0, 0, 0}, true, 1, 0.2, 0.0) == Fault::COMM_TIMEOUT);
    }
    // ABS_RATE inputs are inert in the isolated config: high gyro / rate below the cap never trigger
    { std::vector<Samp> v; for (int i = 0; i < 100; i++) v.push_back({(uint64_t)i + 1, 10.0 + 0.002 * i, 0.31 + 0.003 * i, 12.0});   // 0.31 -> 0.61 at 1.5 rad/s, gyro 12
      Res r = run(D, RefMode::VALID_UPRIGHT, v); Res r2 = run(S.exit, RefMode::VALID_UPRIGHT, v);
      rec("f7", "high gyro (12 rad/s) + fast tilt rate, tilt 0.31..0.61 (< cap), reference upright (REF_DEV would be 'on' in the product config): no trigger in either isolated config", r.first_idx < 0 && r2.first_idx < 0);
      ExitConfig prod; Res r3 = run(prod, RefMode::VALID_UPRIGHT, v);
      rec("f8", "CONTROL: the same stream under the PRODUCT defaults DOES trigger (so the isolated config really removed the other rules)", r3.first_idx >= 0 && r3.rule != TiltRule::HARD, fmt("rule=%s idx=%d", tilt_rule_name(r3.rule), r3.first_idx)); }
    // the other rules individually do fire in the product config at 0.45 etc. (control for the isolation claim)
    { ExitConfig prod; Res r1 = run(prod, RefMode::NONE_CHANNEL, steady(0.45, 10)); Res r2 = run(prod, RefMode::VALID_UPRIGHT, steady(0.45, 400));
      rec("f9", "CONTROL: product defaults at tilt 0.45: no ref -> FIXED_LIMIT, upright valid ref -> REF_DEV; isolated configs stay OK at 0.45", r1.rule == TiltRule::LIMIT && r2.rule == TiltRule::REF_DEV && run(D, RefMode::NONE_CHANNEL, steady(0.45, 10)).first_idx < 0 && run(S.exit, RefMode::VALID_UPRIGHT, steady(0.45, 400)).first_idx < 0,
          fmt("%s / %s", tilt_rule_name(r1.rule), tilt_rule_name(r2.rule))); }
  }

  // ====================================================================== (g) range checks
  {
    auto V = [&](const std::function<void(ExitConfig&)>& f) { ExitConfig c; f(c); std::string e; return validate(c, e); };
    rec("g1", "default config valid; hard cap = 0.70 (> ref peak 0.498 by 0.20 rad, < fallen 1.0)", V([](ExitConfig&) {}));
    bool allbad = true; std::string det;
    for (double h : {0.0, 0.30, 0.45, 0.497, 0.498, 0.50, 0.54, 0.5499}) { bool rej = !V([&](ExitConfig& c) { c.unstable_hard_tilt_rad = h; }); allbad &= rej; det += fmt("%.4f:%s ", h, rej ? "rej" : "ACCEPTED"); }
    rec("g2", "hard cap <= 0.5499 (incl. the reference peak 0.498 and anything below) is refused: the range floor is 0.55 = peak + 0.052", allbad, det);
    rec("g3", "hard cap 0.55 (floor) and 0.999999 (just below fallen 1.0) accepted", V([](ExitConfig& c) { c.unstable_hard_tilt_rad = 0.55; }) && V([](ExitConfig& c) { c.unstable_hard_tilt_rad = 0.999999; }));
    det.clear(); allbad = true;
    for (double h : {1.0, 1.0000001, 1.2, 1.5, 3.0}) { bool rej = !V([&](ExitConfig& c) { c.unstable_hard_tilt_rad = h; }); allbad &= rej; det += fmt("%.7f:%s ", h, rej ? "rej" : "ACCEPTED"); }
    rec("g4", "hard cap >= fallen (1.0) refused (equal, above, far above)", allbad, det);
    det.clear(); allbad = true;
    for (double h : {(double)NAN, (double)INFINITY, -(double)INFINITY, -0.7}) { bool rej = !V([&](ExitConfig& c) { c.unstable_hard_tilt_rad = h; }); allbad &= rej; det += fmt("%g:%s ", h, rej ? "rej" : "ACCEPTED"); }
    rec("g5", "NaN / +-inf / negative hard cap refused", allbad, det);
    rec("g6", "hard cap must also stay above unstable_tilt_rad: (0.40, hard 0.40) and (unstable 0.60, hard 0.56) refused; (unstable 0.50, hard 0.56) accepted",
        !V([](ExitConfig& c) { c.unstable_hard_tilt_rad = 0.55; c.unstable_tilt_rad = 0.55; }) && !V([](ExitConfig& c) { c.unstable_tilt_rad = 0.60; c.unstable_hard_tilt_rad = 0.56; }) && V([](ExitConfig& c) { c.unstable_tilt_rad = 0.50; c.unstable_hard_tilt_rad = 0.56; }));
    rec("g7", "lowering fallen below the cap is refused (fallen 0.6 with hard 0.70), raising fallen above is fine (fallen 1.2 hard 0.70); fallen range [0.6,1.5]",
        !V([](ExitConfig& c) { c.fallen_tilt_rad = 0.6; }) && V([](ExitConfig& c) { c.fallen_tilt_rad = 1.2; }) && !V([](ExitConfig& c) { c.fallen_tilt_rad = 0.59; }) && !V([](ExitConfig& c) { c.fallen_tilt_rad = 1.6; }));
    rec("g8", "ref_dev_min_tilt_rad must be <= hard cap (0.71 refused with hard 0.70)", !V([](ExitConfig& c) { c.ref_dev_min_tilt_rad = 0.71; }) && V([](ExitConfig& c) { c.ref_dev_min_tilt_rad = 0.70; }));
    // through the real site.conf parser
    struct SC { const char* body; bool expect_err; const char* what; };
    const SC scs[] = {
        {"unstable_hard_tilt_rad 0.45\n", true, "0.45"}, {"unstable_hard_tilt_rad 0.498\n", true, "0.498 (reference peak)"}, {"unstable_hard_tilt_rad 0.5499\n", true, "0.5499"}, {"unstable_hard_tilt_rad 0.55\n", false, "0.55 (floor)"},
        {"unstable_hard_tilt_rad 0.70\n", false, "0.70 (default)"}, {"unstable_hard_tilt_rad 0.99\n", false, "0.99"}, {"unstable_hard_tilt_rad 1.0\n", true, "1.0 (= fallen)"}, {"unstable_hard_tilt_rad 1.2\n", true, "1.2 (> fallen)"},
        {"unstable_hard_tilt_rad nan\n", true, "nan"}, {"unstable_hard_tilt_rad inf\n", true, "inf"}, {"unstable_hard_tilt_rad 0.3\nunstable_tilt_rad 0.2\n", true, "0.3 with unstable 0.2"},
        {"fallen_tilt_rad 0.6\n", true, "fallen 0.6 (default hard 0.70 >= fallen)"}, {"fallen_tilt_rad 0.8\nunstable_hard_tilt_rad 0.75\n", false, "fallen 0.8, hard 0.75"}, {"fallen_tilt_rad 0.8\nunstable_hard_tilt_rad 0.80\n", true, "hard == fallen 0.8"}};
    allbad = true; det.clear();
    for (const SC& s : scs) { Site x = site_load(s.body); bool ok = x.err.empty() != s.expect_err; allbad &= ok; det += fmt("[%s:%s] ", s.what, x.err.empty() ? "accepted" : "refused"); }
    rec("g9", "site.conf parser (Site::load) enforces the same ranges (every line behaves as expected)", allbad, det);
  }

  // ====================================================================== the shipped config is untouched by this test
  {
    Site shipped = Site::load(root + "/config/site.conf");
    ExitConfig d;
    bool same = shipped.err.empty() && shipped.exit.unstable_hard_tilt_rad == d.unstable_hard_tilt_rad && shipped.exit.fallen_tilt_rad == 1.0 && shipped.exit.ref_aware && shipped.exit.unstable_tilt_rad == 0.40 && shipped.exit.unstable_soft_tilt_rad == 0.30 &&
                shipped.exit.unstable_gyro_rad_s == 8.0 && shipped.exit.unstable_tilt_rate_rad_s == 3.0 && shipped.exit.ref_dev_rad == 0.35 && shipped.exit.ref_dev_min_tilt_rad == 0.25 && shipped.exit.ref_dev_growth_rad_s == 0.0 &&
                shipped.exit.ref_max_age_s == 0.15 && shipped.guard.max_tilt_rad == 1.0 && shipped.guard.max_gyro_rad_s == 15.0 && shipped.exit.unstable_debounce_samples == 2;
    rec("z1", "shipped config/site.conf still parses to the V1.6 protection values (hard 0.70, fallen 1.0, REF_DEV 0.35/0.25/0, soft 0.30/gyro 8/rate 3, fixed 0.40, ref age 0.15, guard tilt 1.0 / gyro 15, debounce 2)", same);
  }
  ::remove(TMP_CONF);
  printf("\nHARD_CAP isolation: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
