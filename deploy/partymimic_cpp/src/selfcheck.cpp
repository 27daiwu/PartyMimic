// Offline self-check of the DEPLOYED binary on this machine (robot computer or dev PC). It never opens a DDS connection and never touches the robot.
//   1 platform / ONNX Runtime version        2 frozen artifacts (sha256)        3 policy numerics: golden replay of the Python stack (first 3 frames)
//   4 per-tick compute time                  5 FK chain                         6 LowCmd CRC layout vs the OFFICIAL SDK crc32_core
//   7 control logic: exit-mode rules, START x2, chords, command filter, entry rule
// Exit code 0 = no FAIL (WARN allowed), 1 = at least one FAIL.
#include <onnxruntime_c_api.h>
#include <sys/utsname.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <unitree/idl/hg/LowCmd_.hpp>
#if __has_include(<unitree/dds_wrapper/common/crc.h>)
#include <unitree/dds_wrapper/common/crc.h>
#define PM_HAVE_OFFICIAL_CRC 1
#endif

#include "pm/app.hpp"
#include "pm/fk.hpp"
#include "pm/frozen.hpp"
#include "pm/exit_policy.hpp"
#include "pm/guard.hpp"
#include "pm/link.hpp"
#include "pm/policy.hpp"
#include "pm/profile.hpp"
#include "pm/runlog.hpp"
#include "pm/logtool.hpp"
#include "pm/version.hpp"
#include <sys/stat.h>
#include <unistd.h>
#include "pm/remote.hpp"
#include "pm/selfcheck_golden.hpp"
#include "pm/tts.hpp"

namespace pm {
namespace {
int g_fail = 0, g_warn = 0;
void rec(const char* name, const char* status, const std::string& detail) {
  printf("[%-4s] %s: %s\n", status, name, detail.c_str());
  if (!strcmp(status, "FAIL")) g_fail++;
  if (!strcmp(status, "WARN")) g_warn++;
  fflush(stdout);
}
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
  char b[512];
  va_list ap;
  va_start(ap, f);
  vsnprintf(b, sizeof b, f, ap);
  va_end(ap);
  return b;
}
}  // namespace

int run_selfcheck(int argc, char** argv, const Paths& P) {
  struct utsname u;
  uname(&u);
  rec("1.platform", "INFO", fmt("%s %s %s, ONNX Runtime %s, root %s", u.sysname, u.release, u.machine, OrtGetApiBase()->GetVersionString(), P.root.c_str()));

  // ---- 2 frozen artifacts
  struct Art { const char* name; const char* want; };
  bool frozen = true;
  for (Art a : {Art{"policy.onnx", FROZEN_SHA256_POLICY}, Art{"policy_params.txt", FROZEN_SHA256_POLICY_PARAMS}, Art{"motion.bin", FROZEN_SHA256_MOTION}}) {
    std::string h;
    bool ok = sha256_file(P.asset(a.name), h) && h == a.want;
    frozen = frozen && ok;
    if (!ok) rec("2.frozen_artifact", "FAIL", fmt("%s != frozen value (policy: P4 candidate, see frozen.hpp)", a.name));
  }
  if (frozen) rec("2.frozen_artifacts", "PASS", "policy.onnx, policy_params.txt, motion.bin == frozen sha256");

  // ---- 3/4 policy
  Params params;
  Motion motion;
  std::string err;
  if (!Params::load(P.asset("policy_params.txt"), params, err) || !Motion::load(P.asset("motion.bin"), motion, err)) {
    rec("3.load", "FAIL", err);
    printf("\nSELFCHECK FAIL\n");
    return 1;
  }
  rec("3.assets", "PASS", fmt("29 joints, %d history steps, %zu future steps, %d motion frames @ %.0f Hz", params.history, params.future.size(), motion.T, motion.fps));
  try {
    PolicyStack ps(params, motion, P.asset("policy.onnx"), getenv("PM_ORT_THREADS") ? atoi(getenv("PM_ORT_THREADS")) : 2);
    auto gs = [&](int k) {
      RobotSample s;
      for (int i = 0; i < N; i++) { s.q[i] = (float)GOLDEN_Q[k * N + i]; s.dq[i] = (float)GOLDEN_DQ[k * N + i]; }
      for (int i = 0; i < 4; i++) s.quat[i] = (float)GOLDEN_QUAT[k * 4 + i];
      for (int i = 0; i < 3; i++) s.gyro[i] = (float)GOLDEN_GYRO[k * 3 + i];
      return s;
    };
    ps.reset(gs(0), false);
    double d = 0;
    for (int k = 0; k < GOLDEN_N; k++) {
      PolicyOut o = ps.tick(gs(k));
      for (int i = 0; i < N; i++) d = std::max(d, std::fabs((double)o.action[i] - GOLDEN_ACTION[k * N + i]));
    }
    rec("3.policy_golden_replay", d < 1e-3 ? "PASS" : "FAIL", fmt("%d frames: max |action - Python reference| %.2e (tol 1e-3; same ORT version gives 0)", GOLDEN_N, d));
    // timing on a synthetic standing state
    RobotSample s;
    for (int i = 0; i < N; i++) s.q[i] = (float)params.default_pos[i];
    ps.reset(s, false);
    std::vector<double> ms;
    for (int i = 0; i < 230; i++) {
      double t0 = mono();
      ps.tick(s);
      ms.push_back(1e3 * (mono() - t0));
      if (ps.motion_length() > 0 && i > 300) break;
    }
    ms.erase(ms.begin(), ms.begin() + 5);
    std::sort(ms.begin(), ms.end());
    double p50 = ms[ms.size() / 2], p99 = ms[(size_t)(ms.size() * 0.99)], mx = ms.back();
    rec("4.tick_compute_time", p99 <= 15 ? "PASS" : (p99 <= 30 ? "WARN" : "FAIL"), fmt("p50 %.2f / p99 %.2f / max %.2f ms over %zu ticks (PASS p99<=15, WARN<=30; control period 20 ms, watchdog 100 ms)", p50, p99, mx, ms.size()));
  } catch (const std::exception& e) {
    rec("3.policy", "FAIL", e.what());
  }

  // ---- 5 FK
  {
    std::string txt;
    FkChain c;
    if (!read_file(P.asset("fk_chain.txt"), txt) || !c.parse(txt, params.names, err)) rec("5.fk_chain", "FAIL", "cannot load assets/fk_chain.txt: " + err);
    else {
      std::array<double, 4> r = {0.3, -0.5, 0.2, 0.8};
      double n = std::sqrt(0.09 + 0.25 + 0.04 + 0.64);
      double q[N] = {0};
      auto x = c.anchor_quat(r, q);
      double d = std::fabs(x[0] - r[0] / n) + std::fabs(x[1] - r[1] / n) + std::fabs(x[2] - r[2] / n) + std::fabs(x[3] - r[3] / n);
      bool ok = d < 1e-12 && c.links().empty();
      rec("5.fk_anchor", ok ? "PASS" : "FAIL", fmt("anchor body = floating base: %zu links, FK == normalised root quaternion (err %.1e)", c.links().size(), d));
    }
  }

  // ---- 6 CRC
  {
    using unitree_hg::msg::dds_::LowCmd_;
    LowCmd_ m;
    for (int i = 0; i < N; i++) {
      auto& c = m.motor_cmd()[i];
      c.mode() = 1; c.q() = 0.01f * (i + 1); c.dq() = -0.02f * i; c.tau() = 0.5f * (i % 3); c.kp() = 20.f + i; c.kd() = 1.f + 0.1f * i;
    }
    m.mode_machine() = 5;
    uint32_t mine = lowcmd_crc(m);
#ifdef PM_HAVE_OFFICIAL_CRC
    LowCmd_ m2 = m;
    uint32_t off = crc32_core((uint32_t*)&m2, (sizeof(LowCmd_) >> 2) - 1);
    rec("6.lowcmd_crc", (sizeof(LowCmd_) == 1004 && mine == off) ? "PASS" : "FAIL", fmt("sizeof(LowCmd_)=%zu, field-wise CRC 0x%08x %s official crc32_core 0x%08x", sizeof(LowCmd_), mine, mine == off ? "==" : "!=", off));
#else
    rec("6.lowcmd_crc", "WARN", fmt("official crc.h not installed here; field-wise CRC 0x%08x (sizeof(LowCmd_)=%zu)", mine, sizeof(LowCmd_)));
#endif
  }

  // ---- 7 control logic
  {
    bool ok = true;
    std::string why;
    ExitConfig ec;
    auto att = [&](double tilt, double gyro = 0, double age = 0.01, bool valid = true) { Attitude x; x.valid = valid; x.tilt = tilt; x.gyro_xy = gyro; x.age_s = age; return x; };
    auto dec = [&](Fault f, bool normal, Attitude a) { ExitContext x; x.fault = f; x.normal_end = normal; x.att = a; return decide_exit(x, ec); };
    // decision table (single source: pm/exit_policy.cpp)
    ok &= dec(Fault::NONE, true, att(0.02)).mode == "WALKRUN" && dec(Fault::NONE, true, att(0.02)).cls == "NORMAL";
    ok &= dec(Fault::REMOTE_STOP, false, att(0.1)).mode == "WALKRUN" && dec(Fault::SIGNAL, false, att(0.1)).mode == "WALKRUN";
    for (Fault f : {Fault::INFERENCE, Fault::LOOP_OVERRUN, Fault::WATCHDOG, Fault::NO_RESPONSE, Fault::CMD_INVALID, Fault::JOINT_SPEED, Fault::TORQUE_SUSTAINED, Fault::JOINT_LIMIT, Fault::GYRO, Fault::UNSTABLE})
      ok &= dec(f, false, att(0.1)).mode == "WALKRUN" && !dec(f, false, att(0.1)).emergency;
    ok &= dec(Fault::UNSTABLE, false, att(0.5)).mode == "WALKRUN";                      // unstable but still recoverable: immediate WALKRUN
    ok &= dec(Fault::UNSTABLE, false, att(1.2)).mode == "PASSIVE";                       // already down
    ok &= dec(Fault::TILT, false, att(0.5)).mode == "PASSIVE";
    ok &= dec(Fault::NONE, true, att(1.01)).mode == "PASSIVE" && dec(Fault::NONE, true, att(0.99)).mode == "WALKRUN";
    ok &= dec(Fault::INFERENCE, false, att(0.1, 0, 0.5, false)).mode == "WALKRUN";       // attitude unknown/stale: ask WalkRun anyway (PASSIVE only if it fails)
    ok &= dec(Fault::STATE_INVALID, false, att(0.1)).mode == "WALKRUN" && dec(Fault::STATE_FROZEN, false, att(0.1)).mode == "WALKRUN" && dec(Fault::COMM_TIMEOUT, false, att(0.1)).mode == "WALKRUN";
    ok &= dec(Fault::COMM_TIMEOUT, false, att(1.2)).mode == "PASSIVE";                   // known to be down
    ok &= dec(Fault::ESTOP, false, att(0.0)).mode == "PASSIVE" && dec(Fault::ESTOP, false, att(0.0)).emergency;
    ok &= dec(Fault::REMOTE_DAMP, false, att(0.0)).mode == "PASSIVE" && dec(Fault::REMOTE_DAMP, false, att(0.0)).emergency;
    // attitude helpers
    std::array<double, 4> up = {1, 0, 0, 0}, tilted = {0.7071, 0.7071, 0, 0}, bad = {2, 0, 0, 0};
    ok &= std::fabs(tilt_of_quat(up)) < 1e-9 && std::fabs(tilt_of_quat(tilted) - 1.5708) < 1e-3 && !std::isfinite(tilt_of_quat(bad));
    rec("7a1.logic_exit_decision_table", ok ? "PASS" : "FAIL", "normal/START/policy faults -> WALKRUN; unstable -> WALKRUN at once; fallen/ESTOP/L2+B -> PASSIVE; untrusted state/comm loss -> WALKRUN attempt (PASSIVE only if it fails)");
    // handback judgement: RPC success + FSM mismatch must NEVER be COMPLETED (the field-log bug: rpc 0, fsm 1, mode_matched=false, "COMPLETED")
    {
      bool k = true;
      ExitRequest rq;
      rq.decision = dec(Fault::NONE, true, att(0.02));
      ExitOutcome o;
      o.requested_mode = "WALKRUN"; o.settled = true; o.confirmed = true; o.final_mode = "WALKRUN"; o.final_fsm = 802; o.rpc_code = 0;
      k &= classify_exit(rq, o).result == "COMPLETED" && classify_exit(rq, o).exit_code == 0;
      ExitOutcome o2 = o;  // rpc 0, but the OEM ended in fsm 1
      o2.confirmed = false; o2.final_mode = "PASSIVE"; o2.final_fsm = 1;
      k &= classify_exit(rq, o2).result == "WALKRUN_RECOVERY_FAILED" && !classify_exit(rq, o2).recovery_ok && classify_exit(rq, o2).voice == "恢复失败";
      ExitOutcome o3 = o2;  // never settled
      o3.settled = false; o3.final_mode = "UNKNOWN";
      k &= classify_exit(rq, o3).result == "HANDBACK_UNCONFIRMED";
      ExitRequest rq2 = rq;  // abnormal exit, WalkRun confirmed
      rq2.decision = dec(Fault::INFERENCE, false, att(0.1));
      k &= classify_exit(rq2, o).result == "EXITED_WALKRUN" && classify_exit(rq2, o).voice == "异常已交还";
      ExitRequest rq3 = rq;  // emergency passive
      rq3.decision = dec(Fault::ESTOP, false, att(0.0));
      ExitOutcome o4; o4.requested_mode = "PASSIVE"; o4.settled = true; o4.confirmed = true; o4.final_mode = "PASSIVE"; o4.final_fsm = 1;
      k &= classify_exit(rq3, o4).result == "EMERGENCY_PASSIVE" && classify_exit(rq3, o4).voice == "紧急被动";
      ExitRequest rq4 = rq;  // fallen
      rq4.decision = dec(Fault::TILT, false, att(1.1));
      k &= classify_exit(rq4, o4).result == "FALLEN_PASSIVE";
      k &= is_walkrun_family(500) && is_walkrun_family(801) && is_walkrun_family(802) && !is_walkrun_family(1) && !is_walkrun_family(1000) && !is_walkrun_family(0);
      k &= exit_fsm_matches("WALKRUN", 802) && !exit_fsm_matches("WALKRUN", 1) && exit_fsm_matches("PASSIVE", 1) && !exit_fsm_matches("PASSIVE", 802);
      ok &= k;
      rec("7a2.logic_handback_judgement", k ? "PASS" : "FAIL", "rpc ok + fsm 1 = WALKRUN_RECOVERY_FAILED (not COMPLETED); COMPLETED needs a confirmed WalkRun-family fsm; classes/voices/exit codes");
    }
    // instability detector
    {
      bool k = true;
      auto mk = [&](uint64_t seq, double t, double roll, double gx) { LinkState s; s.seq = seq; s.arrival = t; s.quat = {std::cos(roll / 2), std::sin(roll / 2), 0, 0}; s.gyro = {gx, 0, 0}; return s; };
      StabilityMonitor m1(ec);
      for (int i = 0; i < 10; i++) k &= m1.update(mk(i, 0.002 * i, 0.05, 0.1)) == Stability::OK;           // quiet standing
      Stability v = Stability::OK;
      for (int i = 10; i < 14; i++) v = m1.update(mk(i, 0.002 * i, 0.45, 0.5));                           // tilt 0.45 > 0.40
      k &= v == Stability::UNSTABLE;
      StabilityMonitor m2(ec);
      for (int i = 0; i < 4; i++) v = m2.update(mk(i, 0.002 * i, 0.35, 9.0));                              // tilt 0.35 > 0.30 AND gyro 9.0 > 8.0
      k &= v == Stability::UNSTABLE;
      StabilityMonitor m3(ec);
      for (int i = 0; i < 4; i++) v = m3.update(mk(i, 0.002 * i, 0.35, 0.5));                              // tilt 0.35, calm: OK
      k &= v == Stability::OK;
      StabilityMonitor m4(ec);
      for (int i = 0; i < 4; i++) v = m4.update(mk(i, 0.002 * i, 1.2, 0));
      k &= v == Stability::FALLEN;
      StabilityMonitor m5(ec);
      m5.update(mk(0, 0.0, 0.45, 0)); k &= m5.update(mk(0, 0.0, 0.45, 0)) == Stability::OK;                 // single sample / duplicate: debounced
      StabilityMonitor m6(ec);
      for (int i = 0; i < 20; i++) v = m6.update(mk(i, 0.002 * i, 0.31 + 0.02 * i, 0.2));       // tilt rate 10 rad/s (> 3) once above 0.30
      k &= v == Stability::UNSTABLE;
      ok &= k;
      rec("7a3.logic_stability_detector", k ? "PASS" : "FAIL", "tilt>0.40 | tilt>0.30 & (|gyro_xy|>8 | tilt rate>3) = UNSTABLE; tilt>1.0 = FALLEN; 2-sample debounce");
    }
    // V1.6 A1: reference-aware instability judge (the DIY dance leans the pelvis to 0.50 rad: a fixed 0.40 limit fired on the dance itself)
    {
      bool k = true;
      std::string det;
      auto qxz = [](double roll, double yaw) { double a = roll / 2, b = yaw / 2; return std::array<double, 4>{std::cos(b) * std::cos(a), std::cos(b) * std::sin(a), std::sin(b) * std::sin(a), std::sin(b) * std::cos(a)}; };  // Rz(yaw) * Rx(roll), wxyz
      auto f4 = [](const std::array<double, 4>& q) { return std::array<float, 4>{(float)q[0], (float)q[1], (float)q[2], (float)q[3]}; };
      auto run = [&](const ExitConfig& cfg, bool with_chan, bool ref_valid, double ref_roll, double ref_yaw, double ref_age, auto meas_roll, int n, std::string* rule = nullptr) {
        // n lowstate samples at 500 Hz; the reference was published ref_age seconds before the first sample; meas_roll(i) = measured roll at sample i; the measured yaw is arbitrary (1.0 rad: the judge must be yaw-free)
        RefChannel ch;
        StabilityMonitor m(cfg, with_chan ? &ch : nullptr, "selfcheck");
        const double t0 = 10.0;
        Stability v = Stability::OK;
        for (int i = 0; i < n; i++) {
          ch.set(ref_valid, f4(qxz(ref_roll, ref_yaw)), t0 + 0.002 * i - ref_age, i);   // the control loop re-publishes the (held) reference every tick; its age at the judge stays ref_age
          LinkState s; s.seq = i + 1; s.arrival = t0 + 0.002 * i; s.quat = qxz(meas_roll(i), 1.0); s.gyro = {0.3, 0, 0};
          v = m.update(s);
          if (v != Stability::OK) break;
        }
        if (rule) *rule = tilt_rule_name(m.trigger().rule);
        return v;
      };
      ExitConfig leg = ec; leg.ref_aware = false;
      std::string r1;
      auto steady = [](double r) { return [r](int) { return r; }; };
      // (1) the dance leans 0.50 rad and the robot follows (yaw differs by 0.9 rad between the frames): NOT unstable; the legacy fixed 0.40 limit would fire
      k &= run(ec, true, true, 0.50, 0.1, 0.01, steady(0.50), 100) == Stability::OK;
      k &= run(leg, true, true, 0.50, 0.1, 0.01, steady(0.50), 100) == Stability::UNSTABLE;
      k &= run(ec, true, true, 0.50, 0.1, 0.01, steady(0.46), 100) == Stability::OK;                         // lags by 0.04 rad
      // (2) the robot falls away from a reference that is calm (0.10 rad) / leans the other way (-0.4): gravity error 0.45 / 0.9 rad, growing -> REF_DEV
      k &= run(ec, true, true, 0.10, 0.1, 0.01, [](int i) { return 0.20 + 0.002 * i; }, 400, &r1) == Stability::UNSTABLE && r1 == "REF_DEV";
      k &= run(ec, true, true, -0.40, 0.1, 0.01, [](int i) { return 0.10 + 0.01 * i; }, 400, &r1) == Stability::UNSTABLE && r1 == "REF_DEV";
      k &= run(ec, true, true, 0.10, 0.1, 0.01, steady(0.20), 100) == Stability::OK;                          // 0.10 rad error: fine
      k &= run(ec, true, true, 0.00, 0.1, 0.01, steady(0.30), 100) == Stability::OK;                          // error 0.30 < 0.35: fine
      // growth criterion: a large error that is SHRINKING (robot converging onto the reference) does not trigger (reference updated every sample here: 5 rad/s towards the robot)
      {
        RefChannel ch; StabilityMonitor m(ec, &ch, "shrink"); Stability v = Stability::OK;
        for (int i = 0; i < 100; i++) {
          ch.set(true, f4(qxz(std::min(0.60, 0.01 * i), 0.1)), 10.0 + 0.002 * i, i);
          LinkState s; s.seq = i + 1; s.arrival = 10.0 + 0.002 * i; s.quat = qxz(0.60, 1.0); s.gyro = {0.2, 0, 0};
          v = m.update(s);
          if (v != Stability::OK) break;
        }
        k &= v == Stability::OK && m.max_dev > 0.5;
        // ... while the same error that does NOT shrink (reference frozen at 0.0) trips
        RefChannel ch2; StabilityMonitor m2(ec, &ch2, "frozen"); Stability v2 = Stability::OK;
        for (int i = 0; i < 100; i++) {
          ch2.set(true, f4(qxz(0.0, 0.1)), 10.0 + 0.002 * i, i);
          LinkState s; s.seq = i + 1; s.arrival = 10.0 + 0.002 * i; s.quat = qxz(0.60, 1.0); s.gyro = {0.2, 0, 0};
          v2 = m2.update(s);
          if (v2 != Stability::OK) break;
        }
        k &= v2 == Stability::UNSTABLE && m2.trigger().rule == TiltRule::REF_DEV;
      }
      // (3) absolute hard cap: not lifted by the reference, independent of the reference channel (no channel / invalid / stale / reference itself above the cap)
      k &= run(ec, false, true, 0.0, 0.1, 0.01, steady(0.75), 100, &r1) == Stability::UNSTABLE && r1 == "HARD_CAP";
      k &= run(ec, true, false, 0.0, 0.1, 0.01, steady(0.75), 100, &r1) == Stability::UNSTABLE && r1 == "HARD_CAP";
      k &= run(ec, true, true, 0.0, 0.1, 5.0, steady(0.75), 100, &r1) == Stability::UNSTABLE && r1 == "HARD_CAP";
      k &= run(ec, true, true, 0.80, 0.1, 0.01, steady(0.80), 100, &r1) == Stability::UNSTABLE && r1 == "HARD_CAP";   // a (hypothetical) reference leaning 0.80 and a robot following it exactly is still capped
      k &= run(ec, true, true, 0.50, 0.1, 0.01, steady(0.65), 100) == Stability::OK;                           // 0.15 above a 0.50 reference, below the 0.70 cap: following, OK
      // (4) no valid reference -> the old conservative fixed 0.40
      k &= run(ec, true, false, 0.50, 0.1, 0.01, steady(0.45), 100, &r1) == Stability::UNSTABLE && r1 == "FIXED_LIMIT";   // reference not running (lead-in / paused / hold)
      k &= run(ec, true, true, 0.50, 0.1, 5.0, steady(0.45), 100, &r1) == Stability::UNSTABLE && r1 == "FIXED_LIMIT";     // stale (control loop stalled)
      k &= run(ec, false, true, 0.50, 0.1, 0.01, steady(0.45), 100, &r1) == Stability::UNSTABLE && r1 == "FIXED_LIMIT";   // no reference channel at all
      {
        RefChannel ch; ch.set(true, std::array<float, 4>{NAN, 0, 0, 0}, 10.0, 1);                                // NaN reference quaternion -> invalid
        StabilityMonitor m(ec, &ch, "nan"); Stability v = Stability::OK;
        for (int i = 0; i < 10; i++) { LinkState s; s.seq = i + 1; s.arrival = 10.0 + 0.002 * i; s.quat = qxz(0.45, 0.3); v = m.update(s); }
        k &= v == Stability::UNSTABLE && m.trigger().rule == TiltRule::LIMIT && !m.trigger().ref_used;
      }
      k &= run(ec, true, false, 0.0, 0.1, 0.01, steady(0.35), 100) == Stability::OK;                           // quiet below the fixed limit stays OK
      // (5) the old absolute rate rule and FALLEN are unchanged; the fallen cap does not depend on the reference either
      k &= run(ec, true, true, 0.60, 0.1, 0.01, [](int i) { return 0.31 + 0.02 * i; }, 20, &r1) == Stability::UNSTABLE;  // 10 rad/s rate above 0.30 (follows a 0.60 reference at first: only the absolute rate rule can fire)
      k &= run(ec, true, true, 1.2, 0.1, 0.01, steady(1.2), 20) == Stability::FALLEN;
      // (6) structured trigger record + config checks
      {
        RefChannel ch; StabilityMonitor m(ec, &ch, "w");
        for (int i = 0; i < 400; i++) { ch.set(true, f4(qxz(0.10, 0.1)), 10.0 + 0.002 * i, i); LinkState s; s.seq = i + 1; s.arrival = 10.0 + 0.002 * i; s.quat = qxz(0.20 + 0.002 * i, 1.0); if (m.update(s) != Stability::OK) break; }
        std::string j = m.trigger_json();
        k &= j.find("\"rule\":\"REF_DEV\"") != std::string::npos && j.find("\"ref_tilt_rad\":0.1") != std::string::npos && j.find("\"gravity_err_rad\"") != std::string::npos && j.find("\"hard_tilt_rad\":0.700") != std::string::npos &&
             m.detail().find("gravity err") != std::string::npos && m.max_dev > 0.3;
        std::string jj = tilt_judge_json(ec, m, m);
        k &= jj.find("\"max_gravity_err_rad\"") != std::string::npos && jj.find("\"min_margin_hard_cap_rad\"") != std::string::npos;
      }
      {
        std::string e; ExitConfig b1 = ec, b2 = ec, b3 = ec, b4 = ec, b5 = ec;
        b1.unstable_hard_tilt_rad = 0.35;   // not above the fixed limit
        b2.unstable_hard_tilt_rad = 1.1;    // not below fallen
        b3.ref_dev_rad = 0.05; b4.ref_dev_growth_rad_s = 50; b5.ref_max_age_s = 5;
        k &= validate(ec, e) && !validate(b1, e) && !validate(b2, e) && !validate(b3, e) && !validate(b4, e) && !validate(b5, e);
        k &= ec.ref_aware && ec.unstable_hard_tilt_rad == 0.70 && ec.ref_dev_rad == 0.35 && ec.ref_dev_min_tilt_rad == 0.25 && ec.ref_dev_growth_rad_s == 0.0 && ec.ref_max_age_s == 0.15;
      }
      ok &= k;
      rec("7a5.logic_reference_aware_judge", k ? "PASS" : "FAIL", "dance lean 0.50 followed -> no trip (legacy 0.40 would); gravity error > 0.35 & tilt > 0.25 & growing -> REF_DEV (yaw-free); absolute hard cap 0.70 independent of the reference channel; no/stale/NaN/lead-in reference -> fixed 0.40; ranges checked");
    }
    // L2+B emergency chord: debounce, runtime only, no trip from keys held before control, other chords unaffected
    {
      bool k = true;
      DampCombo c(0.10);
      bool f1 = false, f2 = false, f3 = false, f4 = false;
      for (int i = 0; i < 50; i++) f1 |= c.feed(0.01 * i, K_L2 | K_B, false);          // not running: never
      for (int i = 0; i < 20; i++) f2 |= c.feed(1.0 + 0.01 * i, K_L2 | K_B, true);     // held since before control: ignored until released
      c.feed(1.3, 0, true);
      c.feed(1.4, K_L2, true); c.feed(1.402, 0, true);                                 // glitch / half chord
      for (int i = 0; i < 4; i++) f3 |= c.feed(1.5 + 0.01 * i, K_L2 | K_B, true);      // 40 ms < 0.10 s hold
      c.feed(1.6, 0, true);
      int fires = 0;
      for (int i = 0; i < 50; i++) { bool r = c.feed(2.0 + 0.01 * i, K_L2 | K_B, true); fires += r; f4 |= r; }
      DampCombo c2(0.10);
      c2.feed(0, 0, true);
      bool lg = false;
      for (int i = 0; i < 40; i++) lg |= c2.feed(0.01 * i, K_L1 | K_A, true);          // L1+A / F1+F3 / START never fire it
      for (int i = 0; i < 40; i++) lg |= c2.feed(1 + 0.01 * i, K_F1 | K_F3, true);
      for (int i = 0; i < 40; i++) lg |= c2.feed(2 + 0.01 * i, K_START, true);
      k &= !f1 && !f2 && !f3 && f4 && fires == 1 && !lg;
      ok &= k;
      rec("7a4.logic_l2b_emergency_chord", k ? "PASS" : "FAIL", "L2+B held >= 0.10 s fires once, only while running, ignores a press held before control, ignores F1+F3 / L1+A / START");
    }
    ok &= entry_check(802, {500, 505, 801, 802}, why) && is_walkrun_family(802) && exit_fsm_matches("WALKRUN", 801) && !exit_fsm_matches("WALKRUN", 1) && exit_fsm_matches("PASSIVE", 1);
    ok &= !entry_check(0, {500, 505, 801, 802}, why) && !entry_check(1, {500, 505, 801, 802}, why) && !entry_check(200, {500, 505, 801, 802}, why) && entry_check(505, {500, 505, 801, 802}, why);
    StartDoubleClick d;
    bool f1 = d.feed(0.00, true); d.feed(0.05, false); bool f2 = d.feed(0.15, true);
    StartDoubleClick d2;
    d2.feed(0.0, true); d2.feed(0.05, false); bool f3 = d2.feed(0.6, true);          // too slow: not a double click
    StartDoubleClick d3;
    bool held = false;
    for (int i = 0; i < 100; i++) held |= d3.feed(0.01 * i, true);                    // held down: no auto-repeat
    ok &= !f1 && f2 && !f3 && !held;
    ChordDetector load(COMBO_LOAD), go(COMBO_GO);
    bool a1 = false, a2 = false, a3 = false;
    for (int i = 0; i < 40; i++) a1 |= load.feed(0.01 * i, K_F1 | K_F3);               // F1+F3 held 0.4 s fires once
    for (int i = 0; i < 40; i++) a2 |= go.feed(0.01 * i, K_L1 | K_A | K_B);            // extra key cancels
    for (int i = 0; i < 40; i++) a3 |= go.feed(1.0 + 0.01 * i, K_L1);                  // only half the chord
    int fires = 0;
    ChordDetector once(COMBO_GO);
    for (int i = 0; i < 100; i++) fires += once.feed(0.01 * i, K_L1 | K_A);
    ok &= a1 && !a2 && !a3 && fires == 1;
    rec("7a.logic_exit_keys", ok ? "PASS" : "FAIL", "exit-mode rules (WalkRun / PASSIVE), entry fsm rule, START double-click, F1+F3 / L1+A chords");
    // command filter: position clip inside the limits, slew limit, PD torque limit, NaN reject
    SafetyGuard g(params);
    Vec q = params.default_pos, dq = Vec::Zero();
    g.reset_target(q);
    Vec far = q.array() + 3.0;
    CmdOut o = g.filter_command(far, params.kp, params.kd, q, dq, 0.02, 0);
    bool within = true;
    const GuardConfig gd;
    for (int i = 0; i < N; i++) {
      double lim = (params.vel_known[i] ? params.vel_lim[i] : gd.unknown_speed_limit) * gd.target_speed_frac * 0.02;
      within &= o.q[i] <= params.pos_hi[i] - gd.pos_margin_rad + 1e-12 && o.q[i] >= params.pos_lo[i] + gd.pos_margin_rad - 1e-12 && std::fabs(o.q[i] - q[i]) <= lim + 1e-9;
    }
    bool nan_rej = false;
    try { Vec bad = q; bad[3] = NAN; g.filter_command(bad, params.kp, params.kd, q, dq, 0.02, 0); } catch (const GuardReject&) { nan_rej = true; }
    SafetyGuard g2(params);
    Vec tau = Vec::Zero(), gy3;
    Fault ft = g2.check_state(q, dq, tau, {1, 0, 0, 0}, {0, 0, 0}, true, 1, 0.0, 0);
    SafetyGuard g3(params);
    Fault ft2 = g3.check_state(q, dq, tau, {0.7, 0.7, 0, 0}, {0, 0, 0}, true, 1, 0.0, 0);   // 90 deg tilt
    SafetyGuard g4(params);
    Fault ft3 = g4.check_state(q, dq, tau, {1, 0, 0, 0}, {0, 0, 0}, true, 1, 0.2, 0);       // stale state
    std::string verr;
    GuardConfig bad_cfg = gd;
    bad_cfg.target_speed_frac = 1.5;                       // above the hardware speed limit: refused
    GuardConfig bad_cfg2 = gd;
    bad_cfg2.limit_fault_tol_rad = 0.2;                    // looser than the audited hardware tolerance: refused
    ExitConfig bad_e;
    bad_e.unstable_tilt_rad = 1.2;                         // above the fall threshold: refused
    bool cfg_ok = validate(gd, verr) && !validate(bad_cfg, verr) && !validate(bad_cfg2, verr) && !validate(bad_e, verr) && validate(ExitConfig(), verr);
    bool gok = within && nan_rej && ft == Fault::NONE && ft2 == Fault::TILT && ft3 == Fault::COMM_TIMEOUT && cfg_ok && gd.pos_margin_rad == 0.005 && gd.target_speed_frac == 1.0 && gd.torque_cmd_frac == 1.0;
    rec("7b.logic_guard", gok ? "PASS" : "FAIL", "command clip/slew/torque limits (audited defaults), NaN reject, tilt and stale-state faults, config range checks");
  }
  {  // DDS writer classification, with the numbers of the real robot (field log 2026-11-24: fsm 802)
    Site st;
    Writers w;
    w["rt/lowcmd"] = {"a/1", "b/1"};                      // factory motion service(s)
    w["rt/arm_sdk"] = {"c/1", "d/1"};                     // factory idle writers
    w["rt/user_lowcmd"] = {"e/1", "f/1", "g/1"};          // 2 factory idle writers + our guardian
    std::map<std::string, uint64_t> rx = {{"rt/lowcmd", 1000}, {"rt/arm_sdk", 0}, {"rt/user_lowcmd", 0}};
    bool ok = oem_writer_verdict(w, rx, 1.0, 1, 802, st).level != L_FAIL;           // idle writers + active factory rt/lowcmd: NOT a conflict
    ok = ok && oem_writer_verdict(w, rx, 1.0, 1, 802, st).level == L_PASS;
    auto r2 = rx; r2["rt/user_lowcmd"] = 40;
    ok = ok && oem_writer_verdict(w, r2, 1.0, 1, 802, st).level == L_FAIL;          // an ACTIVE controller on the user command topic
    auto r3 = rx; r3["rt/arm_sdk"] = 25;
    ok = ok && oem_writer_verdict(w, r3, 1.0, 1, 802, st).level == L_FAIL;          // an ACTIVE controller on rt/arm_sdk
    auto r4 = rx; r4["rt/lowcmd"] = 0;
    ok = ok && oem_writer_verdict(w, r4, 1.0, 1, 802, st).level == L_WARN;          // factory not driving: informational
    st.oem_confirmed = true; st.oem_expected_writers = 2;
    ok = ok && oem_writer_verdict(w, rx, 1.0, 1, 802, st).level == L_PASS;          // matches the optional baseline
    st.oem_expected_writers = 3;
    ok = ok && oem_writer_verdict(w, rx, 1.0, 1, 802, st).level == L_FAIL;          // baseline set and the rt/lowcmd writer count changed
    rec("7c.logic_dds_writer_classification", ok ? "PASS" : "FAIL", "idle factory writers / guardian are not conflicts; ACTIVE writers on rt/user_lowcmd or rt/arm_sdk block; rt/lowcmd count blocks only against an OEM baseline");
  }
  {  // site.conf: shipped file is valid and keeps the default caps; overrides parse; out-of-range values are refused; voice phrase list
    bool ok = true;
    Site st = Site::load(P.config + "/site.conf");
    ok &= st.err.empty() && st.run_s == 3.0 && st.authority == 1.0 && st.max_run_s == 12.0 && st.max_authority == 1.0 && st.ramp_s == 0.3 && st.end_hold_s == 0.5 && !st.allow_partial_authority && st.ref_mode == "paused";
    ok &= st.guard.pos_margin_rad == 0.005 && st.guard.target_speed_frac == 1.0 && st.guard.torque_cmd_frac == 1.0 && st.guard.limit_fault_tol_rad == 0.05 && st.guard.max_tilt_rad == 1.0;
    ok &= st.exit.unstable_tilt_rad == 0.40 && st.exit.unstable_soft_tilt_rad == 0.30 && st.exit.unstable_gyro_rad_s == 8.0 && st.exit.fallen_tilt_rad == 1.0 && st.guard.torque_meas_fault_frac == 1.2 && st.guard.torque_meas_fault_ticks == 10 && st.guard.max_gyro_rad_s == 15.0;
    auto tmp = [&](const std::string& body) { std::string f = "/tmp/pm_selfcheck_site.conf"; write_atomic(f, body); return Site::load(f); };
    Site o = tmp("max_run_s 5\nmax_authority 0.6\nrun_s 4\npos_margin_rad 0.01\nunstable_tilt_rad 0.3\nbogus_key 1\n");
    ok &= o.err.empty() && o.max_run_s == 5 && o.max_authority == 0.6 && o.guard.pos_margin_rad == 0.01 && o.exit.unstable_tilt_rad == 0.3 && !o.warnings.empty();
    ok &= !tmp("max_run_s 60.5\n").err.empty() && !tmp("max_run_s 0\n").err.empty() && tmp("max_run_s 13\n").err.empty() && tmp("max_run_s 60\n").err.empty() && ABS_MAX_RUN_S == 60.0 && !tmp("max_authority 1.5\n").err.empty() && !tmp("ramp_s 3\n").err.empty() && !tmp("target_speed_frac 1.2\n").err.empty() && !tmp("torque_cmd_frac 1.01\n").err.empty() &&
          !tmp("limit_fault_tol_rad 0.1\n").err.empty() && !tmp("unstable_tilt_rad 1.2\n").err.empty() && !tmp("fallen_tilt_rad 2.0\n").err.empty();
    {  // V1.6 A1: reference-aware judge keys: shipped defaults, overrides, range checks (hard cap must stay above the fixed limit and below fallen)
      ok &= st.exit.ref_aware && st.exit.unstable_hard_tilt_rad == 0.70 && st.exit.ref_dev_rad == 0.35 && st.exit.ref_dev_min_tilt_rad == 0.25 && st.exit.ref_dev_growth_rad_s == 0.0 && st.exit.ref_max_age_s == 0.15;
      Site r = tmp("ref_dev_rad 0.4\nunstable_hard_tilt_rad 0.8\nref_aware_unstable 0\nref_dev_growth_rad_s 0.5\nref_dev_min_tilt_rad 0.3\nref_max_age_s 0.1\n");
      ok &= r.err.empty() && r.warnings.empty() && r.exit.ref_dev_rad == 0.4 && r.exit.unstable_hard_tilt_rad == 0.8 && !r.exit.ref_aware && r.exit.ref_dev_growth_rad_s == 0.5 && r.exit.ref_dev_min_tilt_rad == 0.3 && r.exit.ref_max_age_s == 0.1;
      ok &= !tmp("unstable_hard_tilt_rad 0.4\n").err.empty() && !tmp("unstable_hard_tilt_rad 1.2\n").err.empty() && !tmp("ref_dev_rad 0.1\n").err.empty() && !tmp("ref_dev_rad 1.5\n").err.empty() &&
            !tmp("ref_dev_growth_rad_s 20\n").err.empty() && !tmp("ref_max_age_s 3\n").err.empty() && !tmp("ref_dev_min_tilt_rad 0.9\n").err.empty();
    }
    ok &= st.announce_wait_s == 0.5 && st.announce_max_s == 4.0 && tmp("announce_wait_s 0\n").err.empty() && tmp("announce_wait_s 5\n").err.empty() && !tmp("announce_wait_s -0.1\n").err.empty() && !tmp("announce_wait_s 5.5\n").err.empty() &&
          !tmp("announce_max_s 0.5\n").err.empty() && !tmp("announce_max_s 11\n").err.empty() && !tmp("announce_s_per_char 0.05\n").err.empty() && tmp("announce_wait_s 1.5\n").warnings.empty();  // V1.6 announce_wait range checks
    ::remove("/tmp/pm_selfcheck_site.conf");
    auto u8len = [](const std::string& x) { size_t n = 0; for (unsigned char c : x) n += (c & 0xC0) != 0x80; return n; };
    bool ph = tts_phrases().size() == 17 && tts_phrases()[0] == "服务已启动" && tts_phrases()[6] == "已退出";
    for (auto& t : tts_phrases()) ph &= u8len(t) <= 6;
    ok &= ph;
    rec("7d.site_config_and_voice", ok ? "PASS" : "FAIL", "shipped site.conf valid, P4 defaults (3 s hold, authority 1.0, caps 12 s default (compile-time ceiling 60 s: 13 and 60 accepted, 60.5 refused) / 1.0, ramp 0.3), audited guard defaults, overrides parsed, out-of-range refused; announce_wait keys range-checked; 17 voice phrases (10 results + 6 mode names + 可再次开始) each <= 6 characters");
  }
  {  // P4.1: the command lines the service builds must be accepted by the parser of the child they are built for (field bug: an unknown flag -> guardian rc 2, 178 restarts)
    bool ok = true; std::string det;
    auto parses = [&](const std::vector<std::string>& v, const std::set<std::string>& fl, const char* what) {
      std::vector<char*> av; for (size_t i = 2; i < v.size(); i++) av.push_back(const_cast<char*>(v[i].c_str()));   // skip argv[0] + subcommand like main does
      Args a; std::string err; bool r = a.parse((int)av.size(), av.data(), fl, err);
      if (!r) det += std::string(what) + ": " + err + "; ";
      return r;
    };
    GuardianCmd g; g.self = "/x/partymimic"; g.iface = "eth0"; g.report = "/tmp/r.json"; g.entry_fsm = "802";
    for (int sim = 0; sim < 2; sim++) { g.sim = sim; if (sim) g.iface = "lo"; ok &= parses(guardian_cmdline(g), flags_guardian(), "guardian"); }
    RunnerCmd r; r.self = "/x/partymimic"; r.iface = "eth0"; r.ref_mode = "paused"; r.run_dir = "/tmp/run"; r.go_file = "/tmp/go"; r.status_file = "/tmp/st"; r.entry_fsm = "802"; r.unconfirmed_hold_s = "30";
    for (int m = 0; m < 4; m++) { r.sim = m & 1; r.allow_partial = m & 2; ok &= parses(runner_cmdline(r), flags_runner(), "runner"); }
    // every value-less flag the service accepts must also be accepted by the guardian (so a forward can never break it) or be a runner-only/service-only flag that is NOT forwarded
    for (const char* f : {"--sim", "--allow-partial-authority"}) { std::vector<char*> av{const_cast<char*>(f)}; Args a; std::string err; bool k = a.parse(1, av.data(), flags_guardian(), err); ok &= k; if (!k) det += std::string("guardian rejects ") + f + "; "; }
    // the real failure text is explicit
    { std::vector<char*> av{const_cast<char*>("--bogus-flag")}; Args a; std::string err; bool k = !a.parse(1, av.data(), std::set<std::string>{}, err) && err.find("missing value for --bogus-flag") != std::string::npos; ok &= k; if (!k) det += "error text; "; }
    // runner-only options are not forwarded to the guardian
    { auto v = guardian_cmdline(g); for (auto& x : v) if (x == "--allow-partial-authority" || x == "--ramp-s" || x == "--end-hold-s") { ok = false; det += "runner-only option forwarded to the guardian; "; } }
    rec("7e.subcommand_argv_consistency", ok ? "PASS" : "FAIL", ok ? "service-built guardian/runner command lines (all flag combinations, sim and real) parse with the child's own flag set; runner-only options are not forwarded to the guardian" : det);
  }

  {  // P5.2 7f: test-mode table (config/modes.conf): valid, modes INDEPENDENT (no prerequisite), one definition for start.sh / service
    std::vector<ModeSpec> ms;
    std::string e;
    bool ok = load_modes(P.config + "/modes.conf", ms, e);
    std::string det = e;
    if (ok) {
      const char* want[] = {"check", "hold", "exit", "clip2", "clip4", "dance"};
      ok = ms.size() == 6;
      for (size_t i = 0; ok && i < ms.size(); i++) ok = ms[i].name == want[i];
      ok = ok && ms[0].check_only && !ms[1].check_only && ms[1].ref_mode == "paused" && ms[1].run_s == 3.0 && ms[3].ref_mode == "running" && ms[3].run_s == 2.0 && ms[4].run_s == 4.0 && ms[5].run_s == 7.1 && ms[2].run_s == 8.0;
      auto u8 = [](const std::string& x) { size_t n = 0; for (unsigned char c : x) n += (c & 0xC0) != 0x80; return n; };
      for (auto& m : ms) { ok = ok && tts_known(m.voice) && u8(m.voice) <= 6 && m.run_s <= ABS_MAX_RUN_S; }
      if (!ok) det = "mode table content differs from the P5.2 modes";
    }
    auto tmpm = [&](const std::string& body) { std::string f = "/tmp/pm_selfcheck_modes.conf"; write_atomic(f, body); std::vector<ModeSpec> v; std::string er; bool r = load_modes(f, v, er); ::remove(f.c_str()); return r; };
    ok = ok && tmpm("mode_a running 13 1 仅检查 x\n") && tmpm("mode_a running 60 1 仅检查 x\n") && !tmpm("mode_a running 60.5 1 仅检查 x\n") && !tmpm("mode_a bogus 3 1 仅检查 x\n") && !tmpm("mode_a paused 3.0\n") && tmpm("mode_a paused 3.0 1 仅检查 x\nmode_b running 2.0 0 保持测试 y\n");
    std::string cfg; read_file(P.config + "/modes.conf", cfg);
    ok = ok && cfg.find("requires") == std::string::npos;  // the dependency column is gone for good
    rec("7f.test_mode_table", ok ? "PASS" : "FAIL", ok ? "6 independent modes check/hold/exit/clip2/clip4/dance (no prerequisite column), voice phrases known, bad tables refused" : det);
  }
  {  // V1.6.1 7i: NO manual gate. Every mode (incl. dance) is armed by start.sh itself; the service-built runner command lines parse for all modes x check/armed x real/sim and carry no authorisation flag
    bool ok = true;
    std::string det;
    std::vector<ModeSpec> ms;
    std::string e;
    load_modes(P.config + "/modes.conf", ms, e);
    for (auto& m : ms)
      for (int armed = 0; armed < 2; armed++)
        for (int sim = 0; sim < 2; sim++) {
          RunnerCmd r; r.self = "/x/partymimic"; r.iface = sim ? "lo" : "eth0"; r.ref_mode = m.ref_mode; r.run_s = m.run_s; r.run_dir = "/tmp/run"; r.go_file = "/tmp/go"; r.status_file = "/tmp/st";
          r.mode_name = m.name; r.sim = sim; r.check_only = !armed;
          auto v = runner_cmdline(r);
          for (auto& x : v) if (x == "--accept-unvalidated" || x == "--i-confirm-protection" || x == "--confirm-protection" || x == "--ready-timeout-s") { ok = false; det += " [" + m.name + ": obsolete flag " + x + "]"; }
          std::vector<char*> av; for (size_t i = 2; i < v.size(); i++) av.push_back(const_cast<char*>(v[i].c_str()));
          Args a; std::string er; bool pr = a.parse((int)av.size(), av.data(), flags_runner(), er);
          pr = pr && a.get("--mode") == m.name && a.has("--check-only") == (armed == 0);
          if (!pr) det += " [" + m.name + " argv: " + er + "]";
          ok = ok && pr;
        }
    ok = ok && flags_service().count("--not-armed") && flags_runner().count("--check-only") && !flags_service().count("--accept-unvalidated") && !flags_service().count("--confirm-protection") && !flags_runner().count("--i-confirm-protection");
    rec("7i.no_manual_gate", ok ? "PASS" : "FAIL", ok ? "no authorisation / confirmation flag exists any more; service-built runner argv parses for all 6 modes x armed/check x real/sim" : det);
  }
  {  // P5.1 7g: safety-clip bookkeeping: per-stage record of the guard, clip episodes (joint, reason, duration, raw/clipped)
    bool ok = true;
    SafetyGuard g(params);
    Vec q = params.default_pos, dq = Vec::Zero();
    g.reset_target(q);
    Vec tgt = q;
    tgt[4] = params.pos_hi[4] + 1.0;        // far beyond the limit: POS clip (and slew follows)
    {  // within the limits, but 1.5 x the per-tick slew step: SLEW
      const double stp = (params.vel_known[7] ? params.vel_lim[7] : 20.0) * 0.02 * 1.5, up = q[7] + stp, dn = q[7] - stp;
      tgt[7] = up < params.pos_hi[7] - 0.01 ? up : dn;
    }
    FilterTrace ft;
    CmdOut o = g.filter_command(tgt, params.kp, params.kd, q, dq, 0.02, 0, &ft);
    ok &= (ft.mask[4] & CLIP_POS) && ft.raw[4] == tgt[4] && ft.final_q[4] == o.q[4] && ft.n_clipped >= 2 && (ft.mask[7] & CLIP_SLEW) && ft.mask[0] == 0 && std::string(clip_reason_text(CLIP_POS | CLIP_SLEW)) == "POS_LIMIT|SLEW";
    ok &= ft.after_pos[4] < tgt[4] && ft.after_pos[4] <= params.pos_hi[4] - 0.005 + 1e-12 && ft.lo[4] == params.pos_lo[4] + 0.005;
    ClipTracker ct(0.02);
    auto row = [&](int tick, int joint, uint8_t mask, double raw, double fin) { ClipRow c; c.tick = tick; c.joint = joint; c.mask = mask; c.raw = raw; c.after_pos = fin; c.final_q = fin; c.t_rel = tick * 0.02; return c; };
    for (int t = 0; t < 10; t++) {
      std::vector<ClipRow> rows;
      if (t >= 2 && t <= 4) rows.push_back(row(t, 4, CLIP_POS, 2.0 + t, 1.0));   // 3 consecutive ticks on joint 4
      if (t == 7) rows.push_back(row(t, 4, CLIP_POS, 2.5, 1.0));                  // separate 1-tick episode
      if (t >= 3 && t <= 4) rows.push_back(row(t, 9, CLIP_POS, 0.9, 0.5));
      ct.tick(t, t * 0.02, rows);
    }
    ct.finish();
    const auto& ep = ct.episodes();
    bool e3 = false, e1 = false, e2 = false;
    for (auto& e : ep) {
      if (e.joint == 4 && e.n_ticks == 3 && std::fabs(e.duration_s - 0.06) < 1e-9 && std::fabs(e.max_excess - 5.0) < 1e-9 && e.first_tick == 2 && e.last_tick == 4) e3 = true;
      if (e.joint == 4 && e.n_ticks == 1 && e.first_tick == 7 && std::fabs(e.duration_s - 0.02) < 1e-9) e1 = true;
      if (e.joint == 9 && e.n_ticks == 2) e2 = true;
    }
    ok &= ep.size() == 3 && e3 && e1 && e2 && ct.ticks_with_clip == 4 && ct.ticks_total == 10;
    std::string sj = ct.summary_json(params.names);
    ok &= sj.find("\"n_episodes\":3") != std::string::npos && sj.find(params.names[4]) != std::string::npos && sj.find("POS_LIMIT") != std::string::npos;
    if (false) fprintf(stderr, "7g debug: mask4=%u mask7=%u mask0=%u n_clipped=%d eps=%zu e3=%d e1=%d e2=%d twc=%ld tt=%ld after_pos4=%g hi4=%g lo4=%g\n", ft.mask[4], ft.mask[7], ft.mask[0], ft.n_clipped, ep.size(), e3, e1, e2, ct.ticks_with_clip, ct.ticks_total, ft.after_pos[4], params.pos_hi[4], ft.lo[4]);
    rec("7g.safety_clip_records", ok ? "PASS" : "FAIL", "guard reports joint / stage / raw / clipped per command; episodes = consecutive ticks per (joint, reason) with duration and worst excess; summary names the joints");
  }
  {  // P5.1 7h: asynchronous run logger: the producer never waits for the disk, nothing is lost, files are well-formed, a stalled disk only drops (counted)
    bool ok = true;
    std::string det;
    const std::string dir = "/tmp/pm_selfcheck_log_" + std::to_string(getpid());
    ::mkdir(dir.c_str(), 0775);
    {
      RunLogger lg;
      LogOptions lo;
      lo.dir = dir; lo.joint_names = params.names; lo.obs_dim = 1442; lo.install_crash_handlers = false; lo.preroll_s = 1.0;
      ok &= lg.open(lo);
      for (int i = 0; i < 100; i++) { LinkState s; s.arrival = mono(); s.seq = (uint64_t)i + 1; s.tick = (uint32_t)i; lg.state(s); }  // pre-roll ring
      lg.begin_stream();
      std::unique_ptr<TickRec> r(new TickRec());
      double worst = 0;
      for (int i = 0; i < 400; i++) {
        r->n = (uint32_t)i; r->t_mono = mono(); r->t_rel = i * 0.02; r->obs_n = 1442;
        for (int j = 0; j < 1442; j++) r->obs[j] = (float)(i + j * 1e-3);
        double t0 = mono();
        lg.tick(*r);
        worst = std::max(worst, mono() - t0);
        lg.event("probe", "\"i\":" + std::to_string(i));
        if (i % 4 == 0) sleep_s(0.001);
      }
      LinkState s; s.arrival = mono(); s.seq = 1000; lg.state(s);
      lg.end_stream();
      lg.event("log_stats", "\"dropped_ticks\":0");
      ok &= lg.flush(3.0);
      LogStats st = lg.stats();
      ok &= st.ticks == 400 && st.dropped_ticks == 0 && st.events >= 401 && worst < 0.005;
      det += fmt("worst tick() call %.1f us, queue lag max %.1f ms; ", worst * 1e6, st.max_queue_lag_ms);
      lg.close();
    }
    std::string trace;
    read_file(dir + "/trace.csv", trace);
    size_t nl = 0;
    for (char c : trace) nl += c == '\n';
    ok &= nl == 401;  // header + 400 rows
    uint64_t obs_sz = 0, st_sz = 0;
    { struct stat sb; if (::stat((dir + "/obs.bin").c_str(), &sb) == 0) obs_sz = (uint64_t)sb.st_size; if (::stat((dir + "/state_stream.bin").c_str(), &sb) == 0) st_sz = (uint64_t)sb.st_size; }
    ok &= obs_sz > 400u * (12 + 1442 * 4) && (obs_sz - 400u * (12 + 1442 * 4)) < 1000;   // header JSON + 400 records of (u32 n, f64 t, f32[1442])
    ok &= st_sz > 12 + 20 * STATE_REC_BYTES;                                               // pre-roll (100 samples, 1 s ring) + the live samples
    { std::string ev; read_file(dir + "/events.jsonl", ev); size_t c = 0; for (size_t p = 0; (p = ev.find("\"kind\":\"probe\"", p)) != std::string::npos; p++) c++; ok &= c == 400; }
    // overload: more ticks than the queue holds while the writer cannot keep up -> counted drops, never blocking
    {
      RunLogger lg2;
      LogOptions lo;
      lo.dir = dir + "/over"; lo.joint_names = params.names; lo.obs_dim = 1442; lo.install_crash_handlers = false; lo.tick_cap = 8; lo.obs_bin = false; lo.state_stream = false;
      ok &= lg2.open(lo);
      std::unique_ptr<TickRec> r(new TickRec());
      double worst = 0;
      for (int i = 0; i < 20000; i++) { r->n = (uint32_t)i; double t0 = mono(); lg2.tick(*r); worst = std::max(worst, mono() - t0); }
      LogStats st = lg2.stats();
      ok &= st.dropped_ticks > 0 && st.dropped_ticks + st.ticks <= 20000 && worst < 0.005;
      det += fmt("overload: %llu dropped (counted), worst call %.1f us", (unsigned long long)st.dropped_ticks, worst * 1e6);
      lg2.close();
    }
    std::string rm = "rm -rf " + dir;
    if (system(rm.c_str()) != 0) {}
    rec("7h.async_run_logger", ok ? "PASS" : "FAIL", ok ? det : "logger output wrong: " + det);
  }
  {  // P5.3 7j: repeated start = nothing of the previous run survives (Policy history, previous action, reference index); lead-in clip; training-equivalent torque
    bool ok = true;
    std::string det;
    try {
      Motion lead = motion.with_lead_in(50);
      ok &= lead.T == motion.T + 50 && std::memcmp(&lead.jp[0], &lead.jp[(size_t)49 * N], N * sizeof(float)) == 0 && std::memcmp(&lead.jp[(size_t)50 * N], &motion.jp[0], N * sizeof(float)) == 0 &&
            std::memcmp(&lead.jp[(size_t)(lead.T - 1) * N], &motion.jp[(size_t)(motion.T - 1) * N], N * sizeof(float)) == 0 && std::memcmp(&lead.rq[0], &lead.rq[(size_t)49 * 4], 4 * sizeof(float)) == 0;
      ok &= motion.with_lead_in(0).T == motion.T;
      RobotSample s0, s1;
      for (int i = 0; i < N; i++) { s0.q[i] = (float)params.default_pos[i]; s0.dq[i] = 0.f; s1.q[i] = (float)params.default_pos[i] + 0.05f * (float)std::sin(i); s1.dq[i] = 0.3f * (float)std::cos(i); }
      s1.quat = {0.99f, 0.05f, 0.02f, 0.01f}; s1.gyro = {0.2f, -0.1f, 0.05f};
      PolicyStack fresh(params, lead, P.asset("policy.onnx"), 2), used(params, lead, P.asset("policy.onnx"), 2);
      fresh.reset(s0, false);
      PolicyOut a0 = fresh.tick(s0);
      used.reset(s1, false);                                      // a "previous run": dirty history, previous action, advanced reference index, paused flag
      for (int i = 0; i < 60; i++) used.tick(i % 2 ? s1 : s0);
      used.hold_reference();
      auto dirty = used.state_digest();
      used.reset(s0, false);                                      // the start of the next run
      auto d = used.state_digest();
      PolicyOut b0 = used.tick(s0);
      bool same = a0.motion_t == b0.motion_t && a0.ref_idx == b0.ref_idx && std::memcmp(a0.action.data(), b0.action.data(), N * sizeof(float)) == 0 && std::memcmp(a0.action_prev.data(), b0.action_prev.data(), N * sizeof(float)) == 0;
      std::vector<float> o1(fresh.obs_dim()), o2(used.obs_dim());
      fresh.copy_obs(o1.data()); used.copy_obs(o2.data());
      same = same && o1 == o2;
      bool dirty_was = dirty.motion_t > 0 && dirty.prev_action_abs_max > 0 && !dirty.history_identical;
      bool clean = d.motion_t == 0 && d.prev_action_abs_max == 0 && d.history_identical && d.quat_history_identical && !d.paused;
      ok &= dirty_was && clean && same && a0.motion_t == 1 && a0.ref_idx == 1;
      det += fmt("dirty (t=%d, |prev a| %.2f) -> reset -> digest clean, first tick bit-identical to a fresh stack (motion_t %d, ref idx %d); ", dirty.motion_t, dirty.prev_action_abs_max, b0.motion_t, b0.ref_idx);
    } catch (const std::exception& e) { ok = false; det = e.what(); }
    {  // training-equivalent torque identity: total drive torque == clamp(kp*(q_policy - q) - kd*dq, +-effort) while the sent target stays inside the joint range
      GuardConfig gc; gc.torque_ff = true;
      SafetyGuard g(params, gc), g0(params);
      bool id = true, inside = true, zero_off = true, exit_noff = true;
      double worst = 0;
      uint32_t seed = 12345;
      auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (double)(seed >> 8) / 16777216.0; };
      for (int it = 0; it < 400; it++) {
        Vec q, dq, qt;
        for (int i = 0; i < N; i++) { q[i] = params.pos_lo[i] + 0.1 + (params.pos_hi[i] - params.pos_lo[i] - 0.2) * rnd(); dq[i] = (rnd() - 0.5) * 6; qt[i] = params.pos_lo[i] - 1.0 + (params.pos_hi[i] - params.pos_lo[i] + 2.0) * rnd(); }
        g.reset_target(q); g0.reset_target(q);
        CmdOut c = g.filter_command(qt, params.kp, params.kd, q, dq, 1.0, 0);   // dt 1 s: no slew limit in play, isolates the position clip + torque path
        CmdOut c0 = g0.filter_command(qt, params.kp, params.kd, q, dq, 1.0, 0);
        CmdOut cx = g.filter_command(qt, params.kp, params.kd, q, dq, 1.0, 0, nullptr, false);  // exit frames: never feed-forward
        for (int i = 0; i < N; i++) {
          const double want = std::max(-params.effort_lim[i], std::min(params.effort_lim[i], params.kp[i] * (qt[i] - q[i]) - params.kd[i] * dq[i]));
          const double tot = params.kp[i] * (c.q[i] - q[i]) - params.kd[i] * dq[i] + c.tau[i];
          const bool sat_cap = std::fabs(c.tau[i]) >= params.effort_lim[i] - 1e-9;  // ff itself capped at the effort limit: then total may fall short, never exceed
          if (!sat_cap) worst = std::max(worst, std::fabs(tot - want));
          if (!sat_cap) id &= std::fabs(tot - want) < 1e-9; else id &= std::fabs(tot) <= std::fabs(want) + 1e-9;
          inside &= c.q[i] >= params.pos_lo[i] && c.q[i] <= params.pos_hi[i] && std::fabs(c.tau_train[i] - want) < 1e-9;
          zero_off &= c0.tau[i] == 0.0 && std::fabs(c0.tau_train[i] - want) < 1e-9;
          exit_noff &= cx.tau[i] == 0.0;
        }
      }
      ok &= id && inside && zero_off && exit_noff;
      det += fmt("torque_ff: total torque == training torque (worst %.1e Nm; where the feed-forward itself hits the effort cap the total only falls short, never exceeds), target inside the range, off = no feed-forward, exit frames never feed-forward", worst);
    }
    rec("7j.repeated_start_and_training_torque", ok ? "PASS" : "FAIL", ok ? det : "FAILED: " + det);
  }
  {  // V1.5.1 7k: the planned timeline of every mode: lead-in + WHOLE clip + last-frame hold, never cut by the runner's own fail-safe
    std::vector<ModeSpec> ms;
    std::string e;
    Site site = Site::load(P.config + "/site.conf");
    bool ok = load_modes(P.config + "/modes.conf", ms, e);
    std::string det = e;
    if (ok) {
      for (auto& m : ms) {
        if (m.check_only) continue;
        const bool running = m.ref_mode == "running";
        RunTimeline t = plan_timeline(motion, running, m.run_s, site.lead_in_s, site.end_hold_s);
        const bool margin = t.margin_s >= 0.5;
        ok = ok && margin && t.seg_frames <= t.clip_frames - 1 && t.window_over_tick >= 0;
        if (running) {
          ok = ok && t.lead_in_frames == (int)std::lround(site.lead_in_s * motion.fps) && t.clip_frames == motion.T + t.lead_in_frames;
          if (m.name == "dance") {  // the FULL dance: the window must reach the LAST frame and hold it
            ok = ok && t.covers_whole_dance && t.last_frame_held && m.run_s * motion.fps >= (motion.T - 1) - 1e-6 && t.seg_frames == t.clip_frames - 1;
            det += fmt("dance: lead-in %.2f s + %d frames (%.2f s) -> last frame at tick %d, hold %.2f s, planned %.2f s, fail-safe %.2f s (margin %.2f s); ", t.lead_in_s, motion.T, t.dance_s, t.window_over_tick, site.end_hold_s, t.planned_total_s, t.failsafe_s, t.margin_s);
          } else ok = ok && !t.covers_whole_dance == (m.run_s * motion.fps < (motion.T - 1) - 1e-6);  // clip modes: partial by design, flagged
        } else ok = ok && std::fabs(t.planned_total_s - m.run_s) < 1e-9;  // paused hold: exactly run_s
      }
      // the generic rule behind it: any window shorter than the clip is flagged as partial, a window >= the clip covers it entirely, a window more than the fail-safe allows is impossible
      RunTimeline a = plan_timeline(motion, true, motion.T / (double)motion.fps, 1.0, 0.5), b = plan_timeline(motion, true, 1.0, 1.0, 0.5);
      ok = ok && a.covers_whole_dance && !b.covers_whole_dance;
    }
    rec("7k.run_timeline", ok ? "PASS" : "FAIL", ok ? det + "every non-check mode: window inside the runner fail-safe (margin >= 0.5 s); clip modes flagged partial; paused hold = run_s" : "timeline check failed: " + det);
  }
  {  // V1.6.1 7l: the manual authorisation design is gone from the package: no authorize.sh / authz.conf / ack prompt, start.sh without any authorisation option, site.conf without a READY timeout
    bool ok = true; std::string det, d;
    for (const char* f : {"/authorize.sh", "/config/authz.conf", "/src/authz.cpp"}) if (file_exists(P.root + f)) { ok = false; det += std::string(f) + " exists; "; }
    if (read_file(P.root + "/start.sh", d)) {
      for (const char* w : {"authz", "I-ACCEPT", "--phrase", "--confirm-protection", "authorize.sh", "PM_AUTHZ_FILE"}) if (d.find(w) != std::string::npos) { ok = false; det += std::string("start.sh mentions ") + w + "; "; }
    } else { ok = false; det += "start.sh unreadable; "; }
    if (read_file(P.root + "/ack.sh", d) && d.find("read -r") != std::string::npos) { ok = false; det += "ack.sh prompts; "; }
    rec("7l.no_authorisation_design", ok ? "PASS" : "FAIL", ok ? "V1.6.1: no authorize.sh / config/authz.conf / authz.cpp, start.sh has no authorisation option, ack.sh does not prompt" : det);
  }
  printf("\nSELFCHECK %s: %d warn, %d fail\n", g_fail ? "FAIL" : "PASS", g_warn, g_fail);
  return g_fail ? 1 : 0;
}

}  // namespace pm
