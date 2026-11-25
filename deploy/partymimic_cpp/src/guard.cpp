#include <algorithm>
#include "pm/guard.hpp"

#include <cmath>

namespace pm {

const char* fault_name(Fault f) {
  switch (f) {
    case Fault::NONE: return "NONE"; case Fault::COMM_TIMEOUT: return "COMM_TIMEOUT"; case Fault::STATE_FROZEN: return "STATE_FROZEN";
    case Fault::STATE_INVALID: return "STATE_INVALID"; case Fault::JOINT_LIMIT: return "JOINT_LIMIT"; case Fault::JOINT_SPEED: return "JOINT_SPEED";
    case Fault::TORQUE_SUSTAINED: return "TORQUE_SUSTAINED"; case Fault::TILT: return "TILT"; case Fault::GYRO: return "GYRO";
    case Fault::CMD_INVALID: return "CMD_INVALID"; case Fault::ESTOP: return "ESTOP"; case Fault::REMOTE_STOP: return "REMOTE_STOP";
    case Fault::WATCHDOG: return "WATCHDOG"; case Fault::LOOP_OVERRUN: return "LOOP_OVERRUN"; case Fault::INFERENCE: return "INFERENCE";
    case Fault::NO_RESPONSE: return "NO_RESPONSE"; case Fault::SWITCH_FAILED: return "SWITCH_FAILED"; case Fault::LOCK_LOST: return "LOCK_LOST";
    case Fault::FSM_CHANGED: return "FSM_CHANGED"; case Fault::UNSTABLE: return "UNSTABLE"; case Fault::SIGNAL: return "SIGNAL"; case Fault::REMOTE_DAMP: return "REMOTE_DAMP";
  }
  return "?";
}

bool validate(const GuardConfig& c, std::string& err) {
  auto bad = [&](const char* n, double v, const char* rng) { err = std::string(n) + " = " + std::to_string(v) + " out of range " + rng; return false; };
  if (!(c.state_stale_s >= 0.02 && c.state_stale_s <= 0.2)) return bad("state_stale_s", c.state_stale_s, "[0.02, 0.2]");
  if (!(c.state_frozen_ticks >= 2 && c.state_frozen_ticks <= 20)) return bad("state_frozen_ticks", c.state_frozen_ticks, "[2, 20]");
  if (!(c.limit_fault_tol_rad >= 0 && c.limit_fault_tol_rad <= 0.05)) return bad("limit_fault_tol_rad", c.limit_fault_tol_rad, "[0, 0.05] (hardware limit: may only be tightened)");
  if (!(c.speed_fault_frac >= 0.5 && c.speed_fault_frac <= 1.0)) return bad("speed_fault_frac", c.speed_fault_frac, "[0.5, 1.0] (hardware limit: never above 1.0)");
  if (!(c.unknown_speed_limit >= 1 && c.unknown_speed_limit <= 20)) return bad("unknown_speed_limit", c.unknown_speed_limit, "[1, 20]");
  if (!(c.torque_meas_fault_frac >= 0.5 && c.torque_meas_fault_frac <= 1.5)) return bad("torque_meas_fault_frac", c.torque_meas_fault_frac, "[0.5, 1.5] (x rated peak torque; measured tau_est may overshoot it briefly)");
  if (!(c.torque_meas_fault_ticks >= 1 && c.torque_meas_fault_ticks <= 50)) return bad("torque_meas_fault_ticks", c.torque_meas_fault_ticks, "[1, 50]");
  if (!(c.max_tilt_rad >= 0.6 && c.max_tilt_rad <= 1.5)) return bad("max_tilt_rad", c.max_tilt_rad, "[0.6, 1.5]");
  if (!(c.max_gyro_rad_s >= 2 && c.max_gyro_rad_s <= 25)) return bad("max_gyro_rad_s", c.max_gyro_rad_s, "[2, 25]");
  if (!(c.pos_margin_rad >= 0 && c.pos_margin_rad <= 0.1)) return bad("pos_margin_rad", c.pos_margin_rad, "[0, 0.1]");
  if (!(c.target_speed_frac >= 0.1 && c.target_speed_frac <= 1.0)) return bad("target_speed_frac", c.target_speed_frac, "[0.1, 1.0] (hardware speed limit)");
  if (!(c.torque_cmd_frac >= 0.1 && c.torque_cmd_frac <= 1.0)) return bad("torque_cmd_frac", c.torque_cmd_frac, "[0.1, 1.0] (hardware effort limit)");
  if (!(c.diverge_rad >= 0.3 && c.diverge_rad <= 3.0)) return bad("diverge_rad", c.diverge_rad, "[0.3, 3.0]");
  if (!(c.diverge_ticks >= 2 && c.diverge_ticks <= 50)) return bad("diverge_ticks", c.diverge_ticks, "[2, 50]");
  if (!(c.loop_overrun_factor >= 1.5 && c.loop_overrun_factor <= 5.0)) return bad("loop_overrun_factor", c.loop_overrun_factor, "[1.5, 5.0]");
  return true;
}

SafetyGuard::SafetyGuard(const Params& p, const GuardConfig& c) : p_(p), c_(c) {
  for (int i = 0; i < N; i++) speed_lim_[i] = p.vel_known[i] ? p.vel_lim[i] : c.unknown_speed_limit;
  kp_cap_ = p.kp;  // never above the frozen yaml gains
  prev_target_.setZero();
}

Fault SafetyGuard::trip(Fault f, const std::string& detail, double t) {
  if (fault_ == Fault::NONE) { fault_ = f; events_.push_back({t, f, detail}); }
  return fault_;
}

Fault SafetyGuard::check_state(const Vec& q, const Vec& dq, const Vec& tau, const std::array<double, 4>& quat, const std::array<double, 3>& gyro, bool has_tick,
                               uint32_t tick, double age_s, double t) {
  char b[256];
  if (tripped()) return fault_;
  if (age_s > c_.state_stale_s) {
    snprintf(b, sizeof b, "lowstate age %.0f ms > %.0f ms", age_s * 1e3, c_.state_stale_s * 1e3);
    return trip(Fault::COMM_TIMEOUT, b, t);
  }
  if (has_tick) {
    if (have_tick_ && tick == last_tick_) {
      if (++same_tick_ >= c_.state_frozen_ticks) {
        snprintf(b, sizeof b, "tick %u unchanged for %d control ticks", tick, same_tick_);
        return trip(Fault::STATE_FROZEN, b, t);
      }
    } else same_tick_ = 0;
    last_tick_ = tick; have_tick_ = true;
  }
  bool finite = q.allFinite() && dq.allFinite() && tau.allFinite();
  for (double v : quat) finite = finite && std::isfinite(v);
  for (double v : gyro) finite = finite && std::isfinite(v);
  if (!finite) return trip(Fault::STATE_INVALID, "non-finite or mis-sized state", t);
  double qn = std::sqrt(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
  if (!(qn > 0.9 && qn < 1.1)) { snprintf(b, sizeof b, "IMU quaternion norm %.3f", qn); return trip(Fault::STATE_INVALID, b, t); }
  double qx = quat[1] / qn, qy = quat[2] / qn;
  // joint position beyond the hard limit (+tol)
  double worst = -1e9; int wi = 0;
  for (int i = 0; i < N; i++) { double w = std::max(p_.pos_lo[i] - q[i], q[i] - p_.pos_hi[i]); if (w > worst) { worst = w; wi = i; } }
  stats.min_q_margin = std::min(stats.min_q_margin, -worst);
  if (worst > c_.limit_fault_tol_rad) {
    snprintf(b, sizeof b, "%s q=%.3f outside [%.3f,%.3f] by %.3f", p_.names[wi].c_str(), q[wi], p_.pos_lo[wi], p_.pos_hi[wi], worst);
    return trip(Fault::JOINT_LIMIT, b, t);
  }
  double rmax = 0; int ri = 0;
  for (int i = 0; i < N; i++) { double r = std::fabs(dq[i]) / speed_lim_[i]; if (r > rmax) { rmax = r; ri = i; } }
  stats.max_dq_ratio = std::max(stats.max_dq_ratio, rmax);
  if (rmax > c_.speed_fault_frac) {
    snprintf(b, sizeof b, "%s |dq|=%.2f rad/s > %.2f*%.1f", p_.names[ri].c_str(), std::fabs(dq[ri]), c_.speed_fault_frac, speed_lim_[ri]);
    return trip(Fault::JOINT_SPEED, b, t);
  }
  double temax = 0;
  for (int i = 0; i < N; i++) {
    double te = std::fabs(tau[i]) / p_.effort_lim[i];
    temax = std::max(temax, te);
    tau_count_[i] = te > c_.torque_meas_fault_frac ? tau_count_[i] + 1 : 0;
  }
  stats.max_tau_ratio = std::max(stats.max_tau_ratio, temax);
  for (int i = 0; i < N; i++)
    if (tau_count_[i] >= c_.torque_meas_fault_ticks) {
      snprintf(b, sizeof b, "%s |tau_est|=%.1f > effort %.0f for %d ticks", p_.names[i].c_str(), std::fabs(tau[i]), p_.effort_lim[i], tau_count_[i]);
      return trip(Fault::TORQUE_SUSTAINED, b, t);
    }
  double zz = 1.0 - 2.0 * (qx * qx + qy * qy);
  double tilt = std::acos(std::max(-1.0, std::min(1.0, zz)));
  stats.max_tilt = std::max(stats.max_tilt, tilt);
  if (tilt > c_.max_tilt_rad) { snprintf(b, sizeof b, "tilt %.2f rad > %.2f", tilt, c_.max_tilt_rad); return trip(Fault::TILT, b, t); }
  double g = std::sqrt(gyro[0] * gyro[0] + gyro[1] * gyro[1] + gyro[2] * gyro[2]);
  stats.max_gyro = std::max(stats.max_gyro, g);
  if (g > c_.max_gyro_rad_s) { snprintf(b, sizeof b, "|gyro| %.1f rad/s > %.1f", g, c_.max_gyro_rad_s); return trip(Fault::GYRO, b, t); }
  return Fault::NONE;
}

CmdOut SafetyGuard::filter_command(const Vec& q_t_in, const Vec& kp_in, const Vec& kd_in, const Vec& q, const Vec& dq, double dt, double t) {
  CmdOut o;
  o.q = q_t_in; o.kp = kp_in; o.kd = kd_in; o.dq.setZero(); o.tau.setZero();  // dq_target forced 0, tau_ff forced 0
  if (!(q_t_in.allFinite() && kp_in.allFinite() && kd_in.allFinite())) { trip(Fault::CMD_INVALID, "non-finite command", t); throw GuardReject("non-finite command"); }
  stats.cmds++;
  if ((kp_in.array() < 0).any() || (kd_in.array() < 0).any()) { trip(Fault::CMD_INVALID, "negative gain", t); throw GuardReject("negative gain"); }
  o.kp = o.kp.cwiseMin(kp_cap_);
  // 1) clip the target inside the joint limits with margin
  Vec lo = p_.pos_lo.array() + c_.pos_margin_rad, hi = p_.pos_hi.array() - c_.pos_margin_rad;
  Vec cl = o.q.cwiseMax(lo).cwiseMin(hi);
  if (cl != o.q) stats.pos_clipped++;
  o.q = cl;
  // 2) target slew limit (seeded with the measured q when taking control)
  if (!have_prev_) { prev_target_ = q; have_prev_ = true; }
  Vec step = speed_lim_ * c_.target_speed_frac * dt;
  Vec sl = o.q.cwiseMax(prev_target_ - step).cwiseMin(prev_target_ + step);
  if (sl != o.q) stats.slew_clipped++;
  o.q = sl;
  // 3) PD torque limit by shrinking the position error: tau = kp*e - kd*dq must stay within frac*effort
  bool tclip = false;
  for (int i = 0; i < N; i++) {
    if (o.kp[i] <= 0) continue;
    double Lt = c_.torque_cmd_frac * p_.effort_lim[i], a = o.kd[i] * dq[i];
    double e_lo = (-Lt + a) / o.kp[i], e_hi = (Lt + a) / o.kp[i], e = o.q[i] - q[i];
    double ec = std::max(e_lo, std::min(e_hi, e));
    if (ec != e) tclip = true;
    o.q[i] = q[i] + ec;
  }
  if (tclip) stats.torque_clipped++;
  o.q = o.q.cwiseMax(lo).cwiseMin(hi);
  prev_target_ = o.q;
  return o;
}

}  // namespace pm
