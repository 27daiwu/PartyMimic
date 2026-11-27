#include <algorithm>
#include "pm/session.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstring>

namespace pm {

// ------------------------------------------------------------------ lock
bool ExclusiveLock::acquire(const std::string& path) {
  fd_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0666);
  if (fd_ < 0) { holder = "cannot open lock file"; return false; }
  int lk = ::flock(fd_, LOCK_EX | LOCK_NB);
  for (int i = 0; lk != 0 && i < 3; i++) { ::usleep(1000); lk = ::flock(fd_, LOCK_EX | LOCK_NB); }   // a concurrent probe (guardian / service / runner GO_WAIT) holds the flock for microseconds only
  if (lk != 0) {
    char b[300] = {0};
    ssize_t n = ::pread(fd_, b, sizeof b - 1, 0);
    holder = n > 0 ? std::string(b, (size_t)n) : "";
    ::close(fd_); fd_ = -1;
    return false;
  }
  pid_ = (int)getpid();
  entry_fsm_ = -1; exit_phase_ = "NONE"; exit_owner_ = "none"; exit_t_ = 0;
  write_record();
  return true;
}
// ONE fixed-size record written with a single pwrite (no truncate): a concurrent reader (guardian) never sees a half-written or empty file
void ExclusiveLock::write_record() {
  std::lock_guard<std::mutex> lk(wm_);
  if (fd_ < 0) return;
  char b[257];
  int n = snprintf(b, 256, "pid=%d\nentry_fsm=%d\nexit_phase=%s\nexit_owner=%s\nexit_t=%.3f\n", pid_, entry_fsm_, exit_phase_.c_str(), exit_owner_.c_str(), exit_t_);
  if (n < 0) return;
  if (n > 255) n = 255;
  memset(b + n, ' ', 255 - (size_t)n);
  b[255] = '\n';
  if (::pwrite(fd_, b, 256, 0) < 0) return;
  if (::ftruncate(fd_, 256) != 0) return;
}
void ExclusiveLock::update_entry_fsm(int fsm) { entry_fsm_ = fsm; write_record(); }
void ExclusiveLock::update_exit(const std::string& phase, const std::string& owner) {
  exit_phase_ = phase; exit_owner_ = owner; exit_t_ = wall();
  write_record();
}
void ExclusiveLock::release() {
  if (fd_ >= 0) { ::flock(fd_, LOCK_UN); ::close(fd_); fd_ = -1; }
}
LockInfo lock_probe(const std::string& path) {
  LockInfo li;
  int fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0666);
  if (fd < 0) return li;
  char b[300] = {0};
  ssize_t n = ::pread(fd, b, sizeof b - 1, 0);
  if (n > 0) {
    const char* p = strstr(b, "pid=");
    if (p) li.pid = atoi(p + 4);
    p = strstr(b, "entry_fsm=");
    if (p) li.entry_fsm = atoi(p + 10);
    p = strstr(b, "exit_phase=");
    if (p) { char w[32] = {0}; if (sscanf(p + 11, "%31s", w) == 1) li.exit_phase = w; }
    p = strstr(b, "exit_owner=");
    if (p) { char w[32] = {0}; if (sscanf(p + 11, "%31s", w) == 1) li.exit_owner = w; }
    p = strstr(b, "exit_t=");
    if (p) li.exit_t = atof(p + 7);
  }
  // V1.6 integration fix: another PROBE (runner GO_WAIT loop, guardian 10 ms poll, service) holds the flock for microseconds while it tests it -> a single failed try is NOT "held".
  // A false "held" together with the STALE record of the previous run (exit_phase DONE) made a fresh guardian "arm" on the old pid and end as "normal end" -> service: guardian died -> take-over cancelled.
  // A real holder keeps failing: 3 tries 1 ms apart (costs <= 2 ms only while the lock really is held).
  li.held = true;
  for (int i = 0; i < 3; i++) {
    if (::flock(fd, LOCK_EX | LOCK_NB) == 0) { ::flock(fd, LOCK_UN); li.held = false; break; }
    if (i < 2) ::usleep(1000);
  }
  ::close(fd);
  return li;
}

// ------------------------------------------------------------------ session
static std::string fnum(double v) { char b[64]; snprintf(b, sizeof b, "%.6g", v); return b; }

ControlSession::ControlSession(Link& link, SafetyGuard& guard, const SessionConfig& cfg, EventFn ev)
    : link_(link), guard_(guard), cfg_(cfg), ev_(std::move(ev)), t0_(mono()), heartbeat_(mono()),
      exec_(link, cfg.exit, [this](FrameKind k, const Vec& q) { exit_frame(k, q); }, [this](const std::string& ph) { lock_.update_exit(ph, "runner"); },
            [this]() -> std::string {  // re-decide from the CURRENT attitude before a retried RPC: fallen / unknown -> PASSIVE
              ExitContext x;
              x.att = attitude_of(link_.latest(), mono(), cfg_.exit.attitude_max_age_s);
              return decide_exit(x, cfg_.exit).mode == "PASSIVE" ? "PASSIVE" : "";
            },
            [this]() { return file_exists(cfg_.estop_file) || rf_->l2b_fired.load() > 0; }),
      main_mon_(cfg.exit, &ref_, "control_loop"), watch_mon_(cfg.exit, &ref_, "watch_3ms") {
  last_q_.setZero(); last_dq_.setZero(); last_tau_.setZero(); last_kp_.setZero(); last_kd_.setConstant(DAMP_KD);
  hold_kp_.setZero(); hold_kd_.setConstant(DAMP_KD);
  rf_ = std::make_shared<RemoteFlag>();
  rf_->dbl = StartDoubleClick(cfg_.remote_window_s);
  rf_->l2b = DampCombo(cfg_.exit.l2b_hold_s);
  auto rf = rf_;
  link_.add_state_listener([rf](const LinkState& st) {  // DDS reader thread: sets flags (+ P5.1: remote-controller events into the run log, only on a change)
    uint16_t keys = keys_from_wireless(st.wireless);
    RunLogger* lg = rf->log.load();
    if (lg && keys != rf->prev_keys.exchange(keys)) {
      char b[200];
      snprintf(b, sizeof b, "\"keys\":%u,\"text\":\"%s\",\"byte2\":%u,\"byte3\":%u,\"control_active\":%s", (unsigned)keys, keys_text(keys).c_str(), st.wireless[2], st.wireless[3], rf->active.load() ? "true" : "false");
      lg->event_at(st.arrival, "remote_keys", b);
    }
    if (rf->dbl.feed(st.arrival, start_pressed(keys)) && rf->fired.load() == 0) {
      rf->tm = mono(); rf->fired.store(1);
      if (lg) lg->event_at(st.arrival, "remote_start_double_click", std::string("\"control_active\":") + (rf->active.load() ? "true" : "false"));
    }
    if (rf->l2b.feed(st.arrival, keys, rf->active.load()) && rf->l2b_fired.load() == 0) {
      rf->t_l2b = mono(); rf->l2b_fired.store(1);
      if (lg) lg->event_at(st.arrival, "remote_l2b_emergency", "\"control_active\":true");
    }
  });
}
ControlSession::~ControlSession() { cleanup(); rf_->log.store(nullptr); }

void ControlSession::set_logger(RunLogger* l) {
  log_ = l;
  rf_->log.store(l);
  exec_.set_log([this](const std::string& k, const std::string& f) { event(k, f); });
}

void ControlSession::log_frame(uint8_t kind, const Vec& q_cmd, const Vec& kp, const Vec& kd, double send_us) {
  if (!log_) return;
  FrameRec f;
  f.t_mono = mono(); f.kind = kind; f.send_us = (float)send_us;
  auto st = link_.latest();
  if (st) { f.state_seq = st->seq; for (int i = 0; i < N; i++) f.q_meas[i] = (float)st->q[i]; }
  for (int i = 0; i < N; i++) { f.q_cmd[i] = (float)q_cmd[i]; f.kp[i] = (float)kp[i]; f.kd[i] = (float)kd[i]; }
  log_->frame(f);
}

void ControlSession::log_clip_rows(const FilterTrace& ft, const Vec& q_meas, int tick, uint8_t phase, std::vector<ClipRow>* rows) {
  if (!ft.any_mask) return;
  for (int i = 0; i < N; i++) {
    if (!ft.mask[i]) continue;
    ClipRow c;
    c.t_mono = mono(); c.t_rel = cur_t_rel_.load(); c.tick = tick; c.phase = phase; c.joint = i; c.mask = ft.mask[i];
    c.raw = ft.raw[i]; c.after_pos = ft.after_pos[i]; c.after_slew = ft.after_slew[i]; c.after_torque = ft.after_torque[i]; c.final_q = ft.final_q[i];
    c.lo = ft.lo[i]; c.hi = ft.hi[i]; c.prev_target = ft.prev_target[i]; c.slew_step = ft.slew_step[i]; c.tau_pre_nm = ft.tau_pre_nm[i]; c.tau_lim_nm = ft.tau_lim_nm[i]; c.q_meas = q_meas[i];
    if (rows) rows->push_back(c);
    if (log_) log_->clip(c);
  }
}

void ControlSession::event(const std::string& kind, const std::string& fields) {
  logf("[session] {\"t\":%.4f,\"kind\":\"%s\"%s%s}", now(), kind.c_str(), fields.empty() ? "" : ",", fields.c_str());
  if (ev_) ev_(kind, fields);
}

Fault ControlSession::check_state_once(const StatePtr& st) {
  double t = now();
  Fault f;
  if (!st) {
    f = guard_.check_state(Vec::Zero(), Vec::Zero(), Vec::Zero(), {1, 0, 0, 0}, {0, 0, 0}, false, 0, 1e9, t);
  } else {
    f = guard_.check_state(st->q, st->dq, st->tau, st->quat, st->gyro, true, st->tick, mono() - st->arrival, t);
  }
  if (f != Fault::NONE) { emergency(f, guard_.events().back().detail); return f; }
  if (st && (mode_ == NORMAL || mode_ == ACQUIRE)) {  // control-loop copy of the stability detector (the 3 ms watch thread has the same one)
    Stability sv = main_mon_.update(*st);
    if (sv == Stability::FALLEN) { trigger_exit(Fault::TILT, main_mon_.detail(), mono(), st->arrival); event("tilt_judge_trigger", main_mon_.trigger_json()); return Fault::TILT; }
    if (sv == Stability::UNSTABLE) { trigger_exit(Fault::UNSTABLE, main_mon_.detail(), mono(), st->arrival); event("tilt_judge_trigger", main_mon_.trigger_json()); return Fault::UNSTABLE; }
  }
  return f;
}

bool ControlSession::preflight(std::vector<PreflightItem>& res, double listen_s, double rate_window_s, bool retro) {
  res.clear();
  auto add = [&](const std::string& n, int level, const std::string& d = "") { res.push_back({n, level, d}); };
  char b[256];
  const double listen_w = listen_s >= 0 ? listen_s : cfg_.foreign_writer_listen_s;
  // 1. stream alive (FAIL = communication interrupted) and fast enough (WARN: slow but alive; the in-run guard still trips at 60 ms staleness)
  WindowStats rw;
  const bool retro_rate = retro && (rw = link_.window(rate_window_s)).covered;
  const bool retro_listen = retro && link_.window(listen_w).covered;
  const double t_listen0 = mono();
  const uint64_t n_listen0 = link_.cmd_rx_count();  // the foreign-writer window runs CONCURRENTLY with the stream-rate window (same length, ends at item 4 below)
  double rate = 0;
  size_t nd = 0;
  if (retro_rate) {
    if (rw.ls_rx == 0) { add("lowstate_stream", L_FAIL, "no rt/lowstate sample within the window (communication interrupted)"); return false; }
    rate = rw.ls_hz; nd = rw.distinct_ticks;
  } else {
    double t_end = mono() + rate_window_s;
    struct S { double t; uint64_t seq; uint32_t tick; };
    std::vector<S> seqs;
    while (mono() < t_end) {
      auto st = link_.latest();
      if (st) seqs.push_back({mono(), st->seq, st->tick});
      sleep_s(0.005);
    }
    if (seqs.empty()) { add("lowstate_stream", L_FAIL, "no rt/lowstate sample within the window (communication interrupted)"); return false; }
    rate = (double)(seqs.back().seq - seqs.front().seq) / std::max(1e-6, seqs.back().t - seqs.front().t);
    std::vector<uint32_t> ticks;
    for (auto& s : seqs) ticks.push_back(s.tick);
    std::sort(ticks.begin(), ticks.end());
    nd = std::unique(ticks.begin(), ticks.end()) - ticks.begin();
  }
  snprintf(b, sizeof b, "%.0f Hz (min %.0f)%s", rate, cfg_.min_state_hz, retro_rate ? " [rolling history, last window]" : "");
  add("lowstate_rate_hz", rate >= cfg_.min_state_hz ? L_PASS : (rate >= 15 ? L_WARN : L_FAIL), b);
  snprintf(b, sizeof b, "%zu distinct ticks", nd);
  add("lowstate_tick_advancing", nd > 5 ? L_PASS : (nd > 1 ? L_WARN : L_FAIL), b);
  auto st = link_.latest();
  if (!st) { add("lowstate_stream", L_FAIL, "no rt/lowstate sample"); return false; }
  // 2. guard on a live sample (on a throw-away copy: does not latch the real guard)
  SafetyGuard g = guard_;
  Fault f = g.check_state(st->q, st->dq, st->tau, st->quat, st->gyro, false, 0, mono() - st->arrival, 0.0);
  add("state_inside_guard_envelope", f == Fault::NONE ? L_PASS : L_FAIL, g.events().empty() ? "ok" : g.events().back().detail);
  const Params& P = guard_.params();
  double margin = 1e9;
  bool within = true;
  for (int i = 0; i < N; i++) {
    margin = std::min(margin, std::min(st->q[i] - P.pos_lo[i], P.pos_hi[i] - st->q[i]));
    within = within && st->q[i] >= P.pos_lo[i] && st->q[i] <= P.pos_hi[i];
  }
  snprintf(b, sizeof b, "min margin %.3f rad", margin);
  add("q_within_hard_limits", within ? L_PASS : L_FAIL, b);
  double dqmax = st->dq.cwiseAbs().maxCoeff();
  snprintf(b, sizeof b, "max |dq| %.3f rad/s", dqmax);
  add("robot_quiescent", dqmax < cfg_.quiescent_dq ? L_PASS : (dqmax < 4 * cfg_.quiescent_dq ? L_WARN : L_FAIL), b);
  double qn = std::sqrt(st->quat[0] * st->quat[0] + st->quat[1] * st->quat[1] + st->quat[2] * st->quat[2] + st->quat[3] * st->quat[3]);
  double qx = st->quat[1] / qn, qy = st->quat[2] / qn;
  double tilt = std::acos(std::max(-1.0, std::min(1.0, 1 - 2 * (qx * qx + qy * qy))));
  const double entry_tilt = std::min(cfg_.max_entry_tilt_rad, cfg_.exit.unstable_tilt_rad);  // an entry tilt above the instability threshold would trip the exit at once
  snprintf(b, sizeof b, "%.2f rad (limit %.2f = min(max_entry_tilt_rad %.2f, unstable_tilt_rad %.2f))", tilt, entry_tilt, cfg_.max_entry_tilt_rad, cfg_.exit.unstable_tilt_rad);
  add("tilt_ok", tilt < entry_tilt ? L_PASS : L_FAIL, b);  // fall protection
  // 3. FSM (read-only RPC)
  auto cf = link_.get_fsm();
  snprintf(b, sizeof b, "code=%d fsm=%d", cf.first, cf.second);
  add("fsm_query", (cf.first == 0 && cf.second >= 0) ? L_PASS : L_FAIL, b);
  std::string why;
  bool ok_e = entry_check(cf.second, cfg_.entry_allowed, why);
  add("fsm_entry_allowed", ok_e ? L_PASS : L_FAIL, why);
  snprintf(b, sizeof b, "lowstate.mode_machine=%d expected 5 (0 = simulation, as in the official check_mode_machine)", st->mode_machine);
  add("mode_machine_matches", (st->mode_machine == 0 || st->mode_machine == 5) ? L_PASS : L_FAIL, b);
  // 4. a foreign controller ACTIVELY writing the user command topic
  uint64_t n_user;
  if (retro_listen) n_user = link_.window(listen_w).user_cmd;
  else {
    const double rest = listen_w - (mono() - t_listen0);
    if (rest > 0) sleep_s(rest);
    n_user = link_.cmd_rx_count() - n_listen0;
  }
  snprintf(b, sizeof b, "%llu samples in listen window (%.1f s%s)", (unsigned long long)n_user, listen_w, retro_listen ? ", rolling history" : "");
  add("no_active_writer_on_user_lowcmd", n_user == 0 ? L_PASS : L_FAIL, b);
  // 5. exclusive lock availability, e-stop latch
  LockInfo li = lock_probe(cfg_.lock_path);
  add("exclusive_lock_free", li.held ? L_FAIL : L_PASS, li.held ? "held by pid " + std::to_string(li.pid) : "");
  add("estop_file_absent", file_exists(cfg_.estop_file) ? L_FAIL : L_PASS, cfg_.estop_file);
  bool all = true;
  for (auto& r : res) all = all && r.pass();
  return all;
}

bool ControlSession::acquire(bool fast) {
  std::vector<PreflightItem> res;
  bool ok = fast ? preflight(res, 0.3, 0.3) : preflight(res);
  std::string failed;
  for (auto& r : res) if (!r.pass()) failed += (failed.empty() ? "" : ",") + r.name;
  event("preflight", std::string("\"ok\":") + (ok ? "true" : "false") + ",\"failed\":\"" + json_escape(failed) + "\"");
  if (!ok) { guard_.trip(Fault::SWITCH_FAILED, "preflight failed: " + failed, now()); return false; }
  if (remote_fired()) { guard_.trip(Fault::REMOTE_STOP, "remote START double-click before control was taken", now()); return false; }
  if (!lock_.acquire(cfg_.lock_path)) {
    guard_.trip(Fault::LOCK_LOST, "exclusive lock held: " + lock_.holder, now());
    event("lock_denied");
    return false;
  }
  event("lock_acquired", "\"path\":\"" + json_escape(cfg_.lock_path) + "\"");
  // re-check the foreign writer window now that we hold the lock (closes the preflight->acquire race)
  uint64_t n0 = link_.cmd_rx_count();
  sleep_s(0.3);
  if (link_.cmd_rx_count() != n0) {
    guard_.trip(Fault::LOCK_LOST, "foreign writer appeared on rt/user_lowcmd", now());
    lock_.release();
    return false;
  }
  auto cf = link_.get_fsm();
  entry_fsm_ = cf.second;
  std::string why;
  if (!entry_check(entry_fsm_, cfg_.entry_allowed, why)) {
    guard_.trip(Fault::SWITCH_FAILED, "entry fsm rejected after lock: " + why, now());
    lock_.release();
    return false;
  }
  if (expected_entry_fsm_ >= 0 && entry_fsm_ != expected_entry_fsm_) {  // the factory reacted to the chord (or the operator changed mode) since READY
    guard_.trip(Fault::SWITCH_FAILED, "fsm changed since READY (" + std::to_string(expected_entry_fsm_) + " -> " + std::to_string(entry_fsm_) + "): factory remote function active? not taking control", now());
    lock_.release();
    return false;
  }
  if (remote_fired()) {
    guard_.trip(Fault::REMOTE_STOP, "remote START double-click before control was taken", now());
    lock_.release();
    return false;
  }
  lock_.update_entry_fsm(entry_fsm_);  // the guardian reads it to restore the right state
  event("entry_checked", "\"entry_fsm\":" + std::to_string(entry_fsm_) + ",\"why\":\"" + json_escape(why) + "\"");
  auto st = link_.latest();
  guard_.reset_target(st->q);
  link_.enable_write();
  // frame streamed while switching: measured q + policy gains -> the pose is held the moment the host starts accepting user commands
  last_q_ = st->q; last_dq_.setZero(); last_tau_.setZero(); last_kp_ = hold_kp_; last_kd_ = hold_kd_; have_last_ = true;
  send_raw(last_q_, last_dq_, last_tau_, last_kp_, last_kd_, FK_ACQUIRE);
  rf_->active = true;  // L2+B counts from now on
  mode_ = ACQUIRE;  // we are the writer from now on: the streamer keeps sending the acquire frame until confirmed
  start_threads();
  int rc = link_.switch_to_user();
  event("SwitchToUserCtrl", "\"rpc_code\":" + std::to_string(rc));
  if (rc != 0) { emergency(Fault::SWITCH_FAILED, "SwitchToUserCtrl rpc code " + std::to_string(rc)); return false; }
  double t_end = mono() + cfg_.confirm_timeout_s;
  int fsm = -1;
  bool confirmed = false;
  while (mono() < t_end) {
    auto c = link_.get_fsm();
    fsm = c.second;
    if (c.first == 0 && fsm == USER_FSM_EXPECTED) { confirmed = true; break; }
    sleep_s(0.1);
  }
  if (!confirmed) {
    emergency(Fault::SWITCH_FAILED, "user control not confirmed: fsm entry=" + std::to_string(entry_fsm_) + " now=" + std::to_string(fsm));
    return false;
  }
  user_fsm_seen_ = fsm;
  {
    std::lock_guard<std::recursive_mutex> lk(lk_);
    if (mode_ == ACQUIRE) { heartbeat(); mode_ = NORMAL; }  // the watchdog judges from the moment NORMAL is set: the loop heartbeat is seconds old after ARM
  }
  event("user_control_confirmed", "\"entry_fsm\":" + std::to_string(entry_fsm_) + ",\"user_fsm\":" + std::to_string(fsm) + ",\"strong\":true");
  fsmth_ = std::thread([this] { fsm_poll_loop(); });
  return true;
}

void ControlSession::send_raw(const Vec& q, const Vec& dq, const Vec& tau, const Vec& kp, const Vec& kd, uint8_t kind) {
  link_.send(q, dq, tau, kp, kd);
  n_sent_++;
  if (kind != FK_POLICY) log_frame(kind, q, kp, kd, link_.last_send_us());  // policy frames are rows of trace.csv
}

bool ControlSession::write(const Vec& q_t, const Vec& kp, const Vec& kd, double dt) {
  std::lock_guard<std::recursive_mutex> lk(lk_);
  if (mode_ != NORMAL) return false;
  if (file_exists(cfg_.estop_file)) { trigger_exit(Fault::ESTOP, "estop file " + cfg_.estop_file, mono(), 0); return false; }
  auto st = link_.latest();
  if (!st) { emergency(Fault::COMM_TIMEOUT, "no state when sending"); return false; }
  CmdOut c;
  try {
    c = guard_.filter_command(q_t, kp, kd, st->q, st->dq, dt, now(), &ftr_);
  } catch (const GuardReject& e) {
    event("cmd_rejected", "\"tick\":" + std::to_string(cur_tick_.load()) + ",\"why\":\"" + json_escape(e.what()) + "\",\"raw_q_target\":[" + [&] { std::string s; for (int i = 0; i < N; i++) { char b[32]; if (std::isfinite(q_t[i])) snprintf(b, sizeof b, "%s%.9g", i ? "," : "", q_t[i]); else snprintf(b, sizeof b, "%snull", i ? "," : ""); s += b; } return s; }() + "]");
    emergency(Fault::CMD_INVALID, e.what());
    return false;
  }
  log_clip_rows(ftr_, st->q, cur_tick_.load(), PH_RUN, nullptr);
  last_out_ = c;
  last_q_ = c.q; last_dq_ = c.dq; last_tau_ = c.tau; last_kp_ = c.kp; last_kd_ = c.kd; have_last_ = true;
  send_raw(c.q, c.dq, c.tau, c.kp, c.kd, FK_POLICY);
  return true;
}

void ControlSession::emergency(Fault f, const std::string& detail) { trigger_exit(f, detail, mono(), 0.0); }

// The exit trigger. Order matters for latency: decide -> latch mode -> wake the RPC thread + first frame (executor.submit) -> only THEN the bookkeeping
// that takes locks (guard trip, events). Nothing before submit() waits for the 100 Hz frame stream or for the control loop.
void ControlSession::trigger_exit(Fault f, const std::string& detail, double t_detect, double t_event, bool normal_end) {
  int m = mode_.load();
  while (m == NORMAL || m == ACQUIRE) {
    if (mode_.compare_exchange_weak(m, EXITING)) {
      ExitContext x;
      x.fault = f;
      x.normal_end = normal_end;
      x.att = attitude_of(link_.latest(), mono(), cfg_.exit.attitude_max_age_s);
      ExitRequest req;
      req.decision = decide_exit(x, cfg_.exit);
      req.owner = "runner";
      req.trigger_fault = fault_name(f);
      req.detail = detail;
      req.user_fsm = user_fsm_seen_ >= 0 ? user_fsm_seen_ : USER_FSM_EXPECTED;
      req.entry_fsm = entry_fsm_;
      req.precheck = user_fsm_seen_ < 0 || f == Fault::FSM_CHANGED;  // control may never have been taken / may already be back at the OEM
      req.t_detect = t_detect;
      req.t_event = t_event;
      req.att = x.att;
      exec_.submit(req);  // RPC wake + first HOLD/DAMP frame
      {
        std::lock_guard<std::recursive_mutex> lk(lk_);
        if (f != Fault::NONE) guard_.trip(f, detail, now());
      }
      event("EXIT_TRIGGER", "\"fault\":\"" + std::string(fault_name(f)) + "\",\"mode\":\"" + req.decision.mode + "\",\"class\":\"" + req.decision.cls + "\",\"why\":\"" +
                                json_escape(req.decision.why) + "\",\"detail\":\"" + json_escape(detail) + "\",\"detect_to_submit_us\":" + fnum((mono() - t_detect) * 1e6));
      return;
    }
  }
  std::lock_guard<std::recursive_mutex> lk(lk_);
  guard_.trip(f, detail, now());
  if (m == EXITING) {
    if (f == Fault::ESTOP || f == Fault::REMOTE_DAMP) exec_.escalate_passive(std::string(fault_name(f)) + ": " + detail);  // emergency during an exit in progress
  } else {
    event("fault_before_control", "\"fault\":\"" + std::string(fault_name(guard_.fault())) + "\",\"detail\":\"" + json_escape(detail) + "\"");
  }
}

// HOLD = q snapshot taken at the trigger + the policy gains, through the guard (position clip, slew, torque limit); DAMP = kp 0 / kd 3 (PASSIVE path only)
void ControlSession::exit_frame(FrameKind k, const Vec& q_snap) {
  std::lock_guard<std::recursive_mutex> lk(lk_);
  auto st = link_.latest();
  Vec qm = (st && st->q.allFinite()) ? st->q : q_snap;
  if (k == FrameKind::HOLD && st) {
    try {
      FilterTrace ft;
      CmdOut c = guard_.filter_command(q_snap, hold_kp_ * cfg_.exit.hold_gain_scale, hold_kd_ * cfg_.exit.hold_gain_scale, st->q, st->dq, 0.01, now(), log_ ? &ft : nullptr, false);
      if (log_) log_clip_rows(ft, st->q, -1, PH_EXIT, nullptr);
      last_q_ = c.q; have_last_ = true;
      send_raw(c.q, c.dq, c.tau, c.kp, c.kd, FK_HOLD);
      return;
    } catch (const GuardReject&) {}
  }
  send_raw(qm, Vec::Zero(), Vec::Zero(), Vec::Zero(), Vec::Constant(DAMP_KD), FK_DAMP);
}

void ControlSession::start_threads() {
  if (wd_.joinable()) return;
  stop_ = false;
  exec_.start();
  wd_ = std::thread([this] { watch_loop(); });
  streamer_ = std::thread([this] { stream_loop(); });
}

void ControlSession::watch_loop() {
  uint64_t last_seq = ~0ull;
  while (!stop_) {
    sleep_s(0.003);
    Mode m = (Mode)mode_.load();
    if (m != NORMAL && m != ACQUIRE) continue;  // during EXITING the executor itself watches the emergency channel
    double t_now = mono();
    // ---- emergency channel first: never goes through WALKRUN
    if (file_exists(cfg_.estop_file)) { trigger_exit(Fault::ESTOP, "estop file " + cfg_.estop_file, t_now, 0); continue; }
    if (rf_->l2b_fired.load() > 0) { trigger_exit(Fault::REMOTE_DAMP, "remote L2+B held", rf_->t_l2b.load(), 0); continue; }
    // ---- instability / fall detection on every NEW lowstate sample (3 ms poll: <= 3 ms + 2 samples after the sample arrived)
    auto st = link_.latest();
    if (st && st->seq != last_seq) {
      last_seq = st->seq;
      Stability sv = watch_mon_.update(*st);
      if (sv == Stability::FALLEN) { trigger_exit(Fault::TILT, watch_mon_.detail(), mono(), st->arrival); event("tilt_judge_trigger", watch_mon_.trigger_json()); continue; }
      if (sv == Stability::UNSTABLE) { trigger_exit(Fault::UNSTABLE, watch_mon_.detail(), mono(), st->arrival); event("tilt_judge_trigger", watch_mon_.trigger_json()); continue; }
    }
    if (quit_ && !guard_.tripped()) { trigger_exit(Fault::SIGNAL, "signal (SIGINT/SIGTERM)", t_now, 0); continue; }
    if (rf_->fired.load() > 0 && !guard_.tripped()) { trigger_exit(Fault::REMOTE_STOP, "remote controller START double-click", rf_->tm.load(), 0); continue; }
    if (m != NORMAL) continue;
    if (mono() - heartbeat_ > cfg_.watchdog_s) { trigger_exit(Fault::WATCHDOG, "control loop heartbeat older than " + fnum(cfg_.watchdog_s * 1e3) + " ms", mono(), 0); continue; }
    if (st && mono() - st->arrival > guard_.cfg().state_stale_s * 2) trigger_exit(Fault::COMM_TIMEOUT, "lowstate silent " + fnum(1e3 * (mono() - st->arrival)) + " ms", mono(), 0);
    // another writer on the command topic: our own samples are echoed to our reader, so rx should track n_sent
    long ns = n_sent_.load();
    if (ns > 20 && (long)link_.cmd_rx_count() - ns > 15) trigger_exit(Fault::LOCK_LOST, "foreign writer on command topic (rx=" + std::to_string(link_.cmd_rx_count()) + " sent=" + std::to_string(ns) + ")", mono(), 0);
  }
}

void ControlSession::fsm_poll_loop() {
  // re-verifies GetFsmId == user fsm every second while the policy runs. Leaving the user fsm = control was taken back or lost.
  fsm_bad_ = 0;
  double next = mono() + cfg_.fsm_poll_s;
  while (!stop_) {
    sleep_s(0.02);
    if (mono() < next) continue;
    next = mono() + cfg_.fsm_poll_s;
    if (mode_ != NORMAL) continue;
    auto c = link_.get_fsm();
    if (c.first != 0 || c.second < 0) continue;  // an unanswered query is not proof of lost control
    if (c.second != user_fsm_seen_) {
      if (++fsm_bad_ >= 2) emergency(Fault::FSM_CHANGED, "fsm " + std::to_string(c.second) + " != user fsm " + std::to_string(user_fsm_seen_) + " (control was taken back or lost)");
    } else fsm_bad_ = 0;
  }
}

void ControlSession::stream_loop() {
  // ACQUIRE only: keep streaming the acquire frame until user control is confirmed. From the first exit trigger the ExitExecutor owns the frame stream.
  double period = 1.0 / cfg_.damp_hz, nxt = mono();
  while (!stop_) {
    nxt += period;
    if (mode_.load() == ACQUIRE) {
      std::lock_guard<std::recursive_mutex> lk(lk_);
      if (mode_.load() == ACQUIRE && have_last_) {
        try { send_raw(last_q_, last_dq_, last_tau_, last_kp_, last_kd_, FK_ACQUIRE); } catch (const std::exception& e) { logf("[session] acquire stream error %s", e.what()); }
      }
    }
    double d = nxt - mono();
    if (d > 0) sleep_s(d); else nxt = mono();
  }
}

ExitReport ControlSession::finish(const std::string& reason) {
  if (finished_) return exit_report_;
  ExitReport rep;
  rep.reason = reason;
  if (mode_ == OFF && !exec_.submitted()) {
    rep.note = "never took control; nothing to release";
    rep.confirmed = true;
    cleanup();
    finished_ = true;
    return exit_report_ = rep;
  }
  rep.took_control = true;
  if (!exec_.submitted()) {  // the motion window ended by itself / operator quit: same executor, same decision table
    if (reason == "OPERATOR_QUIT") trigger_exit(Fault::SIGNAL, "operator quit", mono(), 0, false);
    else if (reason == "COMPLETED") trigger_exit(Fault::NONE, "motion window completed", mono(), 0, true);
    else trigger_exit(guard_.tripped() ? guard_.fault() : Fault::NONE, "finish(" + reason + ")", mono(), 0, false);
  }
  while (!exec_.wait(1.0)) {}
  const ExitOutcome& o = exec_.outcome();
  rep.req = exec_.request();
  rep.out = o;
  rep.has_handback = true;
  rep.streamed_damp = true;
  rep.internal_mode = o.requested_mode;
  rep.exit_mode_why = rep.req.decision.why;
  rep.rpc_code = o.rpc_code;
  rep.fsm_after = o.final_fsm;
  rep.confirmed = o.confirmed;
  rep.confirmed_via = o.confirmed_via;
  rep.mode_matched = o.confirmed;
  rep.note = o.note;
  event("SwitchToInternalCtrl", "\"rpc_code\":" + std::to_string(o.rpc_code) + ",\"mode\":\"" + o.requested_mode + "\",\"calls\":" + std::to_string(o.rpc_codes.size()));
  event("exit_outcome", "\"outcome\":" + exit_outcome_json(rep.req, o));
  if (!o.settled)
    event("EXIT_UNCONFIRMED", "\"rpc_code\":" + std::to_string(o.rpc_code) + ",\"fsm\":" + std::to_string(o.final_fsm) + ",\"action\":\"kept DAMPING; use remote damping / power cut; do NOT approach\"");
  cleanup();
  event("released", "\"confirmed\":" + std::string(o.confirmed ? "true" : "false") + ",\"fsm_after\":" + std::to_string(o.final_fsm) + ",\"via\":\"" + o.confirmed_via + "\",\"result\":\"" + o.verdict.result + "\"");
  finished_ = true;
  return exit_report_ = rep;
}

void ControlSession::cleanup() {
  {
    std::lock_guard<std::recursive_mutex> lk(lk_);
    mode_ = OFF;
  }
  rf_->active = false;
  stop_ = true;
  exec_.stop();
  for (std::thread* t : {&wd_, &streamer_, &fsmth_})
    if (t->joinable() && t->get_id() != std::this_thread::get_id()) t->join();
  lock_.release();
}

}  // namespace pm
