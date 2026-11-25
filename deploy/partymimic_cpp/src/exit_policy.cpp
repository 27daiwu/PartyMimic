#include "pm/exit_policy.hpp"

#include <algorithm>
#include <cmath>

#include "pm/app.hpp"
#include "pm/profile.hpp"
#include "pm/util.hpp"

namespace pm {

bool validate(const ExitConfig& c, std::string& err) {
  auto bad = [&](const char* n, double v, const char* rng) { err = std::string(n) + " = " + std::to_string(v) + " out of range " + rng; return false; };
  if (!(c.fallen_tilt_rad >= 0.6 && c.fallen_tilt_rad <= 1.5)) return bad("fallen_tilt_rad", c.fallen_tilt_rad, "[0.6, 1.5]");
  if (!(c.unstable_tilt_rad >= 0.1 && c.unstable_tilt_rad < c.fallen_tilt_rad)) return bad("unstable_tilt_rad", c.unstable_tilt_rad, "[0.1, fallen_tilt_rad)");
  if (!(c.unstable_soft_tilt_rad >= 0.05 && c.unstable_soft_tilt_rad <= c.unstable_tilt_rad)) return bad("unstable_soft_tilt_rad", c.unstable_soft_tilt_rad, "[0.05, unstable_tilt_rad]");
  if (!(c.unstable_gyro_rad_s >= 0.3 && c.unstable_gyro_rad_s <= 10)) return bad("unstable_gyro_rad_s", c.unstable_gyro_rad_s, "[0.3, 10]");
  if (!(c.unstable_tilt_rate_rad_s == 0 || (c.unstable_tilt_rate_rad_s >= 0.5 && c.unstable_tilt_rate_rad_s <= 20))) return bad("unstable_tilt_rate_rad_s", c.unstable_tilt_rate_rad_s, "0 or [0.5, 20]");
  if (!(c.unstable_debounce_samples >= 1 && c.unstable_debounce_samples <= 10)) return bad("unstable_debounce_samples", c.unstable_debounce_samples, "[1, 10]");
  if (!(c.attitude_max_age_s >= 0.02 && c.attitude_max_age_s <= 0.5)) return bad("attitude_max_age_s", c.attitude_max_age_s, "[0.02, 0.5]");
  if (!(c.l2b_hold_s >= 0.02 && c.l2b_hold_s <= 1.0)) return bad("l2b_hold_s", c.l2b_hold_s, "[0.02, 1.0]");
  if (!(c.confirm_timeout_s >= 0.5 && c.confirm_timeout_s <= 10)) return bad("exit_confirm_timeout_s", c.confirm_timeout_s, "[0.5, 10]");
  if (!(c.fsm_poll_s >= 0.01 && c.fsm_poll_s <= 0.5)) return bad("exit_fsm_poll_s", c.fsm_poll_s, "[0.01, 0.5]");
  if (!(c.dwell_s >= 0.1 && c.dwell_s <= 2)) return bad("exit_dwell_s", c.dwell_s, "[0.1, 2]");
  if (!(c.stable_window_s >= 0.5 && c.stable_window_s <= 10)) return bad("exit_stable_window_s", c.stable_window_s, "[0.5, 10]");
  if (!(c.min_stable_polls >= 2 && c.min_stable_polls <= 100)) return bad("exit_min_stable_polls", c.min_stable_polls, "[2, 100]");
  if (!(c.mismatch_fail_s >= 0.05 && c.mismatch_fail_s <= 3)) return bad("exit_mismatch_fail_s", c.mismatch_fail_s, "[0.05, 3]");
  if (!(c.hold_gain_scale > 0 && c.hold_gain_scale <= 1.0)) return bad("exit_hold_gain_scale", c.hold_gain_scale, "(0, 1]");
  if (!(c.exit_owner_timeout_s >= 3 && c.exit_owner_timeout_s <= 60)) return bad("exit_owner_timeout_s", c.exit_owner_timeout_s, "[3, 60]");
  if (!(c.unconfirmed_hold_s >= 0 && c.unconfirmed_hold_s <= 120)) return bad("exit_unconfirmed_hold_s", c.unconfirmed_hold_s, "[0, 120]");
  return true;
}

// ------------------------------------------------------------------ attitude
double tilt_of_quat(const std::array<double, 4>& q) {
  for (double v : q) if (!std::isfinite(v)) return NAN;
  double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (!(n > 0.9 && n < 1.1)) return NAN;
  double x = q[1] / n, y = q[2] / n;
  return std::acos(std::max(-1.0, std::min(1.0, 1.0 - 2.0 * (x * x + y * y))));
}

Attitude attitude_of(const StatePtr& st, double now_mono, double max_age_s) {
  Attitude a;
  if (!st) return a;
  a.age_s = now_mono - st->arrival;
  double t = tilt_of_quat(st->quat);
  if (!std::isfinite(t) || !std::isfinite(st->gyro[0]) || !std::isfinite(st->gyro[1])) return a;
  a.tilt = t;
  a.gyro_xy = std::hypot(st->gyro[0], st->gyro[1]);
  roll_pitch(st->quat, a.roll, a.pitch);
  a.valid = a.age_s <= max_age_s;
  return a;
}

// ------------------------------------------------------------------ stability monitor
Stability StabilityMonitor::update(const LinkState& st) {
  if (st.seq == last_seq_) return level_;
  last_seq_ = st.seq;
  double t = tilt_of_quat(st.quat);
  if (!std::isfinite(t) || !std::isfinite(st.gyro[0]) || !std::isfinite(st.gyro[1])) return level_;  // invalid data is the guard's business (STATE_INVALID)
  tilt = t;
  gyro_xy = std::hypot(st.gyro[0], st.gyro[1]);
  hist_.push_back({st.arrival, t});
  while (hist_.size() > 2 && st.arrival - hist_[1].first >= 0.040) hist_.pop_front();  // keep one sample >= 40 ms old, drop older ones
  tilt_rate = 0;
  if (hist_.size() >= 2 && st.arrival - hist_.front().first >= 0.015) tilt_rate = (t - hist_.front().second) / (st.arrival - hist_.front().first);
  bool fallen = t > c_.fallen_tilt_rad;
  bool unstable = t > c_.unstable_tilt_rad || (t > c_.unstable_soft_tilt_rad && (gyro_xy > c_.unstable_gyro_rad_s || (c_.unstable_tilt_rate_rad_s > 0 && tilt_rate > c_.unstable_tilt_rate_rad_s)));
  n_fallen_ = fallen ? n_fallen_ + 1 : 0;
  n_unstable_ = unstable ? n_unstable_ + 1 : 0;
  char b[200];
  if (n_fallen_ >= c_.unstable_debounce_samples) {
    snprintf(b, sizeof b, "fallen: tilt %.2f rad > %.2f", t, c_.fallen_tilt_rad);
    detail_ = b;
    level_ = Stability::FALLEN;
  } else if (n_unstable_ >= c_.unstable_debounce_samples) {
    snprintf(b, sizeof b, "unstable: tilt %.2f rad, |gyro_xy| %.2f rad/s, tilt rate %.2f rad/s (limits tilt>%.2f | tilt>%.2f & gyro>%.2f / rate>%.2f)", t, gyro_xy, tilt_rate, c_.unstable_tilt_rad,
             c_.unstable_soft_tilt_rad, c_.unstable_gyro_rad_s, c_.unstable_tilt_rate_rad_s);
    detail_ = b;
    level_ = Stability::UNSTABLE;
  } else level_ = Stability::OK;
  return level_;
}

// ------------------------------------------------------------------ decision table (single source of truth)
ExitDecision decide_exit(const ExitContext& x, const ExitConfig& c) {
  ExitDecision d;
  char b[220];
  d.normal_end = x.normal_end;
  const Fault f = x.fault;
  if (f == Fault::ESTOP || f == Fault::REMOTE_DAMP) {  // independent emergency channel: never goes through WALKRUN
    d.mode = "PASSIVE"; d.cls = "EMERGENCY"; d.emergency = true;
    d.why = std::string("emergency passive channel (") + fault_name(f) + ")";
    return d;
  }
  if (f == Fault::TILT) {  // the guard already measured tilt > max_tilt: already down
    d.mode = "PASSIVE"; d.cls = "FALLEN";
    d.why = "fault TILT: robot is down (tilt above the fall threshold)";
    return d;
  }
  // P4: a robot that is KNOWN to be down (valid attitude beyond the fall threshold) goes PASSIVE; everything else - including comm glitches and
  // untrustworthy/unknown attitude - first asks the factory WalkRun (it has its own state estimation and strong recovery); the executor verifies the
  // FSM and escalates to PASSIVE if WalkRun does not take it (WALKRUN_RECOVERY_FAILED).
  if (x.att.valid && x.att.tilt > c.fallen_tilt_rad) {
    d.mode = "PASSIVE"; d.cls = "FALLEN";
    snprintf(b, sizeof b, "robot is down: tilt %.2f rad > fall threshold %.2f", x.att.tilt, c.fallen_tilt_rad);
    d.why = b;
    return d;
  }
  if (f == Fault::STATE_INVALID || f == Fault::STATE_FROZEN || f == Fault::COMM_TIMEOUT || !x.att.valid) {
    d.mode = "WALKRUN"; d.cls = "UNTRUSTED";
    snprintf(b, sizeof b, "fault %s / attitude %s: state not trustworthy -> ask WALKRUN anyway (factory estimator); PASSIVE only if it fails", fault_name(f), x.att.valid ? "ok" : "unknown/stale");
    d.why = b;
    return d;
  }
  if (x.att.tilt > c.fallen_tilt_rad) {
    d.mode = "PASSIVE"; d.cls = "FALLEN";
    snprintf(b, sizeof b, "robot is down: tilt %.2f rad > fall threshold %.2f", x.att.tilt, c.fallen_tilt_rad);
    d.why = b;
    return d;
  }
  d.mode = "WALKRUN";
  if (f == Fault::NONE && x.normal_end) d.cls = "NORMAL";
  else if (f == Fault::REMOTE_STOP || f == Fault::SIGNAL || (f == Fault::NONE && !x.normal_end)) { d.cls = "OPERATOR"; d.operator_stop = true; }
  else if (f == Fault::UNSTABLE) d.cls = "UNSTABLE";
  else d.cls = "POLICY";
  snprintf(b, sizeof b, "%s, tilt %.2f rad, |gyro_xy| %.2f rad/s: recoverable -> WALKRUN at once", d.cls.c_str(), x.att.tilt, x.att.gyro_xy);
  d.why = b;
  return d;
}

// ------------------------------------------------------------------ classification
ExitVerdict classify_exit(const ExitRequest& req, const ExitOutcome& o) {
  ExitVerdict v;
  const std::string orig = req.decision.mode;
  v.final_mode = o.final_mode;
  v.walkrun_recovered = o.final_mode == "WALKRUN" && o.settled;
  v.recovery_ok = o.confirmed;
  const bool emergency = req.decision.emergency || o.emergency_escalation;
  if (!o.settled) { v.result = "HANDBACK_UNCONFIRMED"; v.exit_code = 5; v.voice = "检查失败"; v.recovery_ok = false; return v; }
  if (emergency) { v.result = "EMERGENCY_PASSIVE"; v.exit_code = 12; v.voice = "紧急被动"; return v; }
  if (orig == "WALKRUN") {
    if (o.confirmed && o.final_mode == "WALKRUN") {
      if (req.decision.normal_end) { v.result = "COMPLETED"; v.exit_code = 0; v.voice = "执行完成"; }
      else { v.result = "EXITED_WALKRUN"; v.exit_code = 10; v.voice = req.decision.operator_stop ? "已退出" : "异常已交还"; }
    } else { v.result = "WALKRUN_RECOVERY_FAILED"; v.exit_code = 11; v.voice = "恢复失败"; v.recovery_ok = false; }
    return v;
  }
  v.result = "FALLEN_PASSIVE"; v.exit_code = 13; v.voice = "恢复失败";
  return v;
}

static std::string ms_rel(double t, double t0) { char b[48]; snprintf(b, sizeof b, "%.3f", t > 0 ? (t - t0) * 1e3 : -1.0); return b; }

std::string exit_outcome_json(const ExitRequest& r, const ExitOutcome& o) {
  const double t0 = o.t_detect;
  auto num = [&](const char* k, const std::string& v) { return std::string("\"") + k + "\":" + v; };
  std::string s = "{";
  s += jstr("result", o.verdict.result) + "," + jstr("requested_mode", o.requested_mode) + "," + jstr("final_mode", o.final_mode) + "," + jnum("final_fsm", o.final_fsm) + "," +
       jbool("recovery_ok", o.verdict.recovery_ok) + "," + jbool("walkrun_recovered", o.verdict.walkrun_recovered) + "," + jbool("confirmed", o.confirmed) + "," + jbool("settled", o.settled) + "," +
       jbool("escalated_to_passive", o.escalated_to_passive) + "," + jbool("emergency_escalation", o.emergency_escalation) + "," + jbool("skipped_rpc", o.skipped_rpc) + "," +
       jstr("cause_class", r.decision.cls) + "," + jstr("decision_why", r.decision.why) + "," + jstr("trigger_fault", r.trigger_fault) + "," + jstr("trigger_detail", r.detail) + "," + jstr("owner", r.owner) + "," +
       jnum("exit_code", o.verdict.exit_code) + "," + jnum("rpc_code", o.rpc_code) + "," + jstr("confirmed_via", o.confirmed_via) + "," + jstr("note", o.note) + "," +
       jnum("fsm_polls", o.fsm_polls) + "," + jnum("fsm_poll_failures", o.fsm_poll_failures) + ",";
  s += "\"rpc_codes\":[";
  for (size_t i = 0; i < o.rpc_codes.size(); i++) s += (i ? "," : "") + std::to_string(o.rpc_codes[i]);
  s += "],\"rpc_modes\":[";
  for (size_t i = 0; i < o.rpc_modes.size(); i++) s += (i ? ",\"" : "\"") + o.rpc_modes[i] + "\"";
  s += "],\"timing_ms_from_detect\":{";
  s += num("t_event", o.t_event > 0 ? ms_rel(o.t_event, t0) : "-1") + "," + num("t_detect", "0") + "," + num("t_rpc_sent", ms_rel(o.t_rpc_sent, t0)) + "," + num("t_rpc_returned", ms_rel(o.t_rpc_returned, t0)) + "," +
       num("t_fsm_confirmed", ms_rel(o.t_fsm_confirmed, t0)) + "," + num("t_stable_confirmed", ms_rel(o.t_stable_confirmed, t0)) + "," + num("t_done", ms_rel(o.t_done, t0)) + "},";
  s += jnum("detect_to_rpc_sent_ms", o.t_rpc_sent > 0 ? (o.t_rpc_sent - o.t_detect) * 1e3 : -1) + "," + jnum("rpc_latency_ms", o.t_rpc_returned > 0 ? (o.t_rpc_returned - o.t_rpc_sent) * 1e3 : -1) + "," +
       jnum("event_to_rpc_sent_ms", (o.t_event > 0 && o.t_rpc_sent > 0) ? (o.t_rpc_sent - o.t_event) * 1e3 : -1) + ",\"fsm_trace\":[";
  for (size_t i = 0; i < o.fsm_trace.size() && i < 60; i++) {
    char b[64];
    snprintf(b, sizeof b, "%s[%.1f,%d]", i ? "," : "", (o.fsm_trace[i].first - t0) * 1e3, o.fsm_trace[i].second);
    s += b;
  }
  return s + "]}";
}

// ------------------------------------------------------------------ executor
ExitExecutor::ExitExecutor(Link& link, const ExitConfig& cfg, FrameFn frames, PhaseFn phase, RemodeFn remode, EmergencyFn emergency)
    : link_(link), c_(cfg), frames_(std::move(frames)), phase_(std::move(phase)), remode_(std::move(remode)), emergency_(std::move(emergency)) {
  snap_q_.setZero();
}
ExitExecutor::~ExitExecutor() { stop(); }

void ExitExecutor::start() {
  if (started_) return;
  started_ = true;
  stop_ = false;
  worker_ = std::thread([this] { worker(); });
  frames_th_ = std::thread([this] { frame_loop(); });
}

void ExitExecutor::stop() {
  if (!started_) return;
  stop_ = true;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    cv_.notify_all();
  }
  for (std::thread* t : {&worker_, &frames_th_})
    if (t->joinable() && t->get_id() != std::this_thread::get_id()) t->join();
  started_ = false;
}

bool ExitExecutor::submit(const ExitRequest& req) {
  int exp = 0;
  if (!state_.compare_exchange_strong(exp, 1)) return false;
  req_ = req;
  mode_ = req.decision.mode;
  auto st = link_.latest();
  snap_q_ = (st && st->q.allFinite()) ? st->q : Vec(Vec::Zero());  // the measured q at the trigger time = what the HOLD frames pin
  const FrameKind kind = mode_ == "PASSIVE" ? FrameKind::DAMP : FrameKind::HOLD;
  frame_kind_ = (int)kind;
  frames_on_ = true;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    pending_ = true;
  }
  cv_.notify_one();  // 1) wake the RPC thread FIRST
  try { frames_(kind, snap_q_); } catch (const std::exception& e) { logf("[exit] first frame failed: %s", e.what()); }  // 2) first hold/damp frame
  if (phase_) phase_("EXITING");                                                                                           // 3) bookkeeping (lock-file marker)
  return true;
}

void ExitExecutor::escalate_passive(const std::string& why) {
  if (state_.load() == 2) return;
  if (!escalate_.exchange(true)) {
    out_.escalation_why = why;
    frame_kind_ = (int)FrameKind::DAMP;  // damp frames at once; the RPC follows when the current one returns
    logf("[exit] ESCALATE to emergency PASSIVE: %s", why.c_str());
  }
}

bool ExitExecutor::wait(double timeout_s) {
  std::unique_lock<std::mutex> lk(mtx_);
  done_cv_.wait_for(lk, std::chrono::duration<double>(timeout_s), [this] { return state_.load() == 2; });
  return state_.load() == 2;
}

void ExitExecutor::frame_loop() {
  double nxt = mono();
  while (!stop_) {
    nxt += 0.01;
    if (state_.load() == 1 && emergency_ && !escalate_.load() && emergency_()) escalate_passive("emergency channel during exit");
    if (frames_on_.load() && state_.load() != 0) {
      try { frames_((FrameKind)frame_kind_.load(), snap_q_); } catch (const std::exception&) {}
    }
    double d = nxt - mono();
    if (d > 0) sleep_s(d); else nxt = mono();
  }
}

int ExitExecutor::call_rpc(const std::string& mode) {
  double ts = mono();
  if (out_.t_rpc_sent == 0) out_.t_rpc_sent = ts;
  int rc = link_.switch_to_internal(mode);
  out_.t_rpc_returned = mono();
  out_.rpc_codes.push_back(rc);
  out_.rpc_modes.push_back(mode);
  out_.rpc_code = rc;
  logf("[exit] SwitchToInternalCtrl(%s) rc=%d (%.1f ms)", mode.c_str(), rc, (out_.t_rpc_returned - ts) * 1e3);
  return rc;
}

void ExitExecutor::poll_record(int code, int fsm) {
  out_.fsm_polls++;
  if (code != 0 || fsm < 0) { out_.fsm_poll_failures++; fsm = -1; }
  if (out_.fsm_trace.empty() || out_.fsm_trace.back().second != fsm) out_.fsm_trace.push_back({mono(), fsm});
}

static bool is_reject(int rc) { return rc == 3203 || rc == 3204 || rc == 3102; }

// Returns 0 = confirmed (match + dwell + stable window), 1 = mismatch (foreign fsm, last_fsm set), 2 = still user fsm / no answer, 3 = escalation requested,
// 4 = matched but too few GetFsmId answers to confirm (never a mismatch: NO fallback request, keep watching).
int ExitExecutor::confirm(const std::string& mode, double timeout_s, bool& /*unused*/, int& last_fsm) {
  const double t_end = mono() + timeout_s;
  double mismatch_since = 0;
  double t_match = 0;
  last_fsm = -1;
  int ok_polls = 0;
  int last_code = -1;
  for (;;) {
    if (escalate_.load() && mode != "PASSIVE") return 3;
    auto cf = link_.get_fsm();
    poll_record(cf.first, cf.second);
    const double now = mono();
    last_code = cf.first;
    if (cf.first == 0 && cf.second >= 0) {
      last_fsm = cf.second;
      const bool match = exit_fsm_matches(mode, cf.second);
      const bool user = req_.user_fsm >= 0 && cf.second == req_.user_fsm;
      if (mode == "WALKRUN" && frame_kind_.load() == (int)FrameKind::HOLD && req_.user_fsm >= 0 && !user) frames_on_ = false;  // the fsm left the user fsm: the OEM has control again, stop writing (official example does the same)
      if (match) {
        if (t_match == 0) { t_match = now; if (out_.t_fsm_confirmed == 0) out_.t_fsm_confirmed = now; if (mode == "WALKRUN") frames_on_ = false; }
        ok_polls++;
        if (now - t_match >= c_.dwell_s + c_.stable_window_s && ok_polls >= c_.min_stable_polls) { out_.t_stable_confirmed = now; return 0; }
      } else if (t_match > 0) {
        return 1;  // matched before, then fell out of the expected set (e.g. OEM dropped into fsm 1): recovery failed
      } else if (user) {
        mismatch_since = 0;
      } else {
        if (mismatch_since == 0) mismatch_since = now;
        if (now - mismatch_since >= c_.mismatch_fail_s) return 1;
      }
    } else if (t_match > 0 && cf.first != 0) {
      // an unanswered poll inside the stable window is not a mismatch; it just does not count
    }
    if (t_match == 0 && now >= t_end) break;
    if (t_match > 0 && now > t_match + c_.dwell_s + c_.stable_window_s + 3.0) return 4;  // matched, never contradicted, but too few answers to call it stable
    sleep_s(c_.fsm_poll_s);
  }
  (void)last_code;
  if (t_match > 0) return 2;
  return last_fsm >= 0 && !(req_.user_fsm >= 0 && last_fsm == req_.user_fsm) ? 1 : 2;
}

void ExitExecutor::worker() {
  {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait(lk, [this] { return pending_ || stop_.load(); });
    if (!pending_) return;
  }
  ExitOutcome& o = out_;
  o.t_detect = req_.t_detect;
  o.t_event = req_.t_event;
  o.requested_mode = req_.decision.mode;
  std::string mode = mode_;
  bool dummy = false;
  int fsm = -1;
  bool rpc_needed = true;
  auto check_escalation = [&]() {
    if (escalate_.load() && mode != "PASSIVE") {
      o.emergency_escalation = true; o.escalated_to_passive = true;
      mode = "PASSIVE"; frame_kind_ = (int)FrameKind::DAMP; frames_on_ = true;
      o.note += "emergency escalation to PASSIVE (" + o.escalation_why + "); ";
      return true;
    }
    return false;
  };
  // optional pre-check: the OEM may already be in the requested mode (user fsm never confirmed / control already taken back)
  if (req_.precheck) {
    auto c = link_.get_fsm();
    poll_record(c.first, c.second);
    if (c.first == 0 && c.second >= 0 && exit_fsm_matches(mode, c.second)) {
      rpc_needed = false; o.skipped_rpc = true;
      o.note += "fsm already " + std::to_string(c.second) + " (requested mode): no exit RPC sent; ";
      o.t_rpc_sent = o.t_rpc_returned = 0;
    }
  }
  check_escalation();
  int rc = 0;
  if (rpc_needed) {
    rc = call_rpc(mode);
    if (is_reject(rc)) {  // explicit rejection: ONE immediate retry (re-deciding the mode from the current attitude)
      sleep_s(c_.rpc_retry_gap_s);
      if (remode_) { std::string nm; try { nm = remode_(); } catch (...) {} if (!nm.empty() && nm != mode) { o.note += mode + "->" + nm + " before retry; "; mode = nm; if (mode == "PASSIVE") { frame_kind_ = (int)FrameKind::DAMP; o.escalated_to_passive = true; } } }
      o.retried_after_rejection = true;
      rc = call_rpc(mode);
    }
  }
  int status = -1;
  bool verified_retry_done = false;
  for (int round = 0; round < 6; round++) {
    check_escalation();
    status = confirm(mode, c_.confirm_timeout_s, dummy, fsm);
    if (status == 3) {  // emergency while confirming a WALKRUN: PASSIVE request now (damp frames already running)
      check_escalation();
      rc = call_rpc("PASSIVE");
      continue;
    }
    if (status == 0) break;
    // not confirmed
    if (status == 2 && mode == "WALKRUN" && rc != 0 && !is_reject(rc) && !verified_retry_done && !o.retried_after_rejection) {
      verified_retry_done = true; o.verified_retry = true;
      logf("[exit] rc=%d and fsm still not %s: ONE verified retry", rc, mode.c_str());
      rc = call_rpc(mode);
      continue;
    }
    break;
  }
  o.final_fsm = fsm;
  if (status == 0) {
    o.confirmed = !o.escalated_to_passive && mode == req_.decision.mode;
    o.settled = true;
    o.final_mode = mode;
    o.confirmed_via = o.skipped_rpc ? "fsm_already" : (rc == 0 ? "rpc_ok+fsm+stable" : "fsm_only(rpc_failed)+stable");
  } else if (mode == "WALKRUN" && status == 4) {
    o.confirmed = false;
    o.final_mode = "UNKNOWN";
    o.note += "WalkRun-family fsm seen but too few GetFsmId answers for the stable window; no fallback request sent; ";
  } else if (mode == "WALKRUN") {
    // ---- WALKRUN recovery failed
    o.confirmed = false;
    o.note += status == 1 ? "WalkRun recovery failed: fsm=" + std::to_string(fsm) + " is not in the WalkRun family (rpc " + std::to_string(rc) + "); " : "WalkRun recovery failed: fsm never left user control (rpc " + std::to_string(rc) + "); ";
    logf("[exit] !!! WALKRUN RECOVERY FAILED rc=%d fsm=%d", rc, fsm);
    if (status == 1 && fsm == 1) {
      o.settled = true; o.final_mode = "PASSIVE"; o.confirmed_via = "fsm=1 observed (OEM damping)";
      frame_kind_ = (int)FrameKind::DAMP; frames_on_ = true;
    } else if (status == 1 && fsm >= 0) {
      o.settled = true; o.final_mode = fsm == 0 ? "ZERO_TORQUE" : "OTHER"; o.confirmed_via = "fsm=" + std::to_string(fsm) + " observed";
      frame_kind_ = (int)FrameKind::DAMP; frames_on_ = true;
    } else {
      // still the user fsm (or unreadable): keep control damped and ask for PASSIVE
      o.escalated_to_passive = true; mode = "PASSIVE";
      frame_kind_ = (int)FrameKind::DAMP; frames_on_ = true;
      int rc2 = call_rpc("PASSIVE");
      if (is_reject(rc2)) { sleep_s(c_.rpc_retry_gap_s); rc2 = call_rpc("PASSIVE"); }
      int st2 = confirm("PASSIVE", c_.confirm_timeout_s, dummy, fsm);
      o.final_fsm = fsm;
      if (st2 == 0) { o.settled = true; o.final_mode = "PASSIVE"; o.confirmed_via = "PASSIVE fallback confirmed"; }
      else o.note += "PASSIVE fallback not confirmed (fsm " + std::to_string(fsm) + "); ";
    }
  } else {
    // PASSIVE requested (or escalated) and not confirmed
    o.confirmed = false;
    o.note += mode + " not confirmed (fsm " + std::to_string(fsm) + "); ";
    if (status == 1 && fsm >= 0 && fsm != 1) { /* foreign fsm */ }
  }
  if (!o.settled) {  // keep damping and keep watching: a late execution still counts
    frame_kind_ = (int)FrameKind::DAMP; frames_on_ = true;
    logf("[exit] !!! EXIT UNCONFIRMED (mode %s fsm %d): keep damping; use the remote controller damping / power cut; do NOT approach", mode.c_str(), fsm);
    const double t_hold_end = mono() + c_.unconfirmed_hold_s;
    while (mono() < t_hold_end && !stop_) {
      sleep_s(0.2);
      auto cf = link_.get_fsm();
      poll_record(cf.first, cf.second);
      if (cf.first == 0 && cf.second >= 0 && exit_fsm_matches(mode, cf.second)) {
        sleep_s(c_.dwell_s);
        auto c3 = link_.get_fsm();
        poll_record(c3.first, c3.second);
        if (c3.first == 0 && c3.second == cf.second) {
          o.settled = true; o.late_confirm = true; o.final_mode = mode; o.final_fsm = cf.second;
          o.confirmed = !o.escalated_to_passive && mode == req_.decision.mode;
          o.confirmed_via = "late(fsm)";
          break;
        }
      }
    }
  }
  if (escalate_.load()) o.emergency_escalation = true;
  if (o.settled && o.final_mode == "PASSIVE") sleep_s(0.1);  // a few more damping frames
  frames_on_ = false;
  o.t_done = mono();
  o.verdict = classify_exit(req_, o);
  if (o.final_mode == "UNKNOWN" && !o.settled) o.verdict.final_mode = "UNKNOWN";
  o.done = true;
  logf("[exit] RESULT %s requested=%s final_mode=%s final_fsm=%d recovery_ok=%d detect->rpc_sent %.2f ms", o.verdict.result.c_str(), o.requested_mode.c_str(), o.final_mode.c_str(), o.final_fsm,
       (int)o.verdict.recovery_ok, o.t_rpc_sent > 0 ? (o.t_rpc_sent - o.t_detect) * 1e3 : -1.0);
  if (phase_) phase_("DONE");
  {
    std::lock_guard<std::mutex> lk(mtx_);
    state_ = 2;
  }
  done_cv_.notify_all();
}

}  // namespace pm
