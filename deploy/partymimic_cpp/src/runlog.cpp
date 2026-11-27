#include "pm/runlog.hpp"

#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <map>

namespace pm {

namespace {

void app(std::string& s, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
void app(std::string& s, const char* fmt, ...) {
  char b[256];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(b, sizeof b, fmt, ap);
  va_end(ap);
  if (n > 0) s.append(b, (size_t)std::min<int>(n, (int)sizeof b - 1));
}
std::string jq(const std::string& s) { return "\"" + json_escape(s) + "\""; }
std::string jn(double v) {
  if (!std::isfinite(v)) return "null";
  char b[48];
  snprintf(b, sizeof b, "%.9g", v);
  return b;
}

// bounded queue, preallocated storage, mutex only around a memcpy-sized critical section
template <class T> class BQ {
 public:
  explicit BQ(size_t cap = 0) { reset(cap); }
  void reset(size_t cap) { buf_.assign(cap, T()); head_ = n_ = 0; }
  // full queue: drop_oldest (rolling pre-roll window, BY DESIGN) overwrites the oldest and counts `evicted`; otherwise the NEW item is lost and counted in `dropped` (a real loss)
  bool push(const T& v, bool drop_oldest, uint64_t& dropped, uint64_t* evicted = nullptr) {
    std::lock_guard<std::mutex> lk(m_);
    if (buf_.empty()) { dropped++; return false; }
    if (n_ == buf_.size()) {
      if (!drop_oldest) { dropped++; return false; }
      if (evicted) (*evicted)++; else dropped++;
      head_ = (head_ + 1) % buf_.size();
      n_--;
    }
    buf_[(head_ + n_) % buf_.size()] = v;
    n_++;
    if (n_ > max_n_) max_n_ = n_;
    return true;
  }
  bool pop(T& out) {
    std::lock_guard<std::mutex> lk(m_);
    if (!n_) return false;
    out = buf_[head_];
    head_ = (head_ + 1) % buf_.size();
    n_--;
    return true;
  }
  template <class F> size_t trim_front(F old) {  // remove the oldest items while old(item) holds (pre-roll window at begin_stream)
    std::lock_guard<std::mutex> lk(m_);
    size_t k = 0;
    while (n_ && old(buf_[head_])) { head_ = (head_ + 1) % buf_.size(); n_--; k++; }
    return k;
  }
  size_t trim_to_free(size_t min_free) {  // headroom: drop the OLDEST items until min_free slots are free
    std::lock_guard<std::mutex> lk(m_);
    size_t k = 0;
    while (n_ && buf_.size() - n_ < min_free) { head_ = (head_ + 1) % buf_.size(); n_--; k++; }
    return k;
  }
  size_t capacity() const { std::lock_guard<std::mutex> lk(m_); return buf_.size(); }
  size_t size() const { std::lock_guard<std::mutex> lk(m_); return n_; }
  size_t max_size() const { std::lock_guard<std::mutex> lk(m_); return max_n_; }
 private:
  mutable std::mutex m_;
  std::vector<T> buf_;
  size_t head_ = 0, n_ = 0, max_n_ = 0;
};

struct EvMsg { std::string line; double t = 0; };

std::mutex g_reg_m;
std::vector<RunLogger*> g_reg;
std::atomic<bool> g_handlers_installed{false};

void crash_marker(const char* what) {
  std::unique_lock<std::mutex> lk(g_reg_m, std::try_to_lock);
  if (!lk.owns_lock()) return;
  for (auto* l : g_reg) if (l->is_open()) l->event("CRASH", std::string("\"signal\":\"") + what + "\"");
}
void crash_handler(int sig) {
  static std::atomic<int> once{0};
  if (once.exchange(1) == 0) {
    alarm(2);  // best effort only: whatever happens below (a lock held by the crashed thread, a dead disk), the process is gone after 2 s
    const char* nm = sig == SIGSEGV ? "SIGSEGV" : sig == SIGABRT ? "SIGABRT" : sig == SIGBUS ? "SIGBUS" : sig == SIGFPE ? "SIGFPE" : "SIGILL";
    crash_marker(nm);
    RunLogger::flush_all(0.4);
  }
  signal(sig, SIG_DFL);
  raise(sig);
}
void terminate_handler() {
  alarm(2);
  crash_marker("std::terminate (uncaught exception)");
  RunLogger::flush_all(0.4);
  std::abort();
}

}  // namespace

// ------------------------------------------------------------------ formatting
std::string trace_csv_header(const std::vector<std::string>& names) {
  std::string h = "t,phase,ssb_frame,authority,infer_ms,step_ms,state_age_ms,tick,keys_b2,keys_b3,quat_w,quat_x,quat_y,quat_z,gyro_x,gyro_y,gyro_z,roll,pitch,yaw";
  for (const char* g : {"q", "dq", "tau_est", "action", "q_policy", "q_cmd"})
    for (auto& n : names) h += std::string(",") + g + ":" + n;
  h += ",n,t_mono,dt_ms,sync_wait_ms,send_us,state_age_send_ms,dds_loop_ms,seq,seq_delta,tick_delta,arrival,flags,ls_rx,ls_lost,ls_gap_ms,clip_n,clip_mask,tilt,keys,mode_machine,ref_frame,"
       "ref_rp_x,ref_rp_y,ref_rp_z,ref_rq_w,ref_rq_x,ref_rq_y,ref_rq_z,rpy_roll,rpy_pitch";
  for (const char* g : {"q_pd", "action_prev", "kp", "kd", "tau_pd", "ref_jp", "ref_jv", "tau_ff", "tau_train"})
    for (auto& n : names) h += std::string(",") + g + ":" + n;
  return h + ",dance_frame,dance_t,seg\n";  // V1.6: appended at the END: every reader that looks columns up by name (logtool, pmlog) is unaffected
}

void trace_csv_row(const TickRec& r, std::string& o) {
  o.reserve(o.size() + 6000);
  app(o, "%.4f,%s,%d,%.6g,%.6g,%.6g,%.6g,%u,%u,%u,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g", r.t_rel, r.phase == PH_RUN ? "RUN" : "EXIT", r.ssb_frame, r.authority, r.infer_ms, r.step_ms,
      r.state_age_ms, r.ls_tick, r.wireless[2], r.wireless[3], r.quat[0], r.quat[1], r.quat[2], r.quat[3], r.gyro[0], r.gyro[1], r.gyro[2], r.roll, r.pitch, r.rpy[2]);
  auto fl = [&](const float* a) { for (int i = 0; i < N; i++) app(o, ",%.9g", a[i]); };
  auto db = [&](const double* a) { for (int i = 0; i < N; i++) app(o, ",%.15g", a[i]); };
  fl(r.q); fl(r.dq); fl(r.tau); fl(r.action); db(r.q_policy); fl(r.q_cmd);
  app(o, ",%u,%.6f,%.4f,%.4f,%.2f,%.4f,%.4f,%llu,%u,%u,%.6f,%u,%llu,%llu,%.4f,%u,%u,%.9g,%u,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g", r.n, r.t_mono, r.dt_ms, r.sync_wait_ms, r.send_us,
      r.state_age_send_ms, r.dds_loop_ms, (unsigned long long)r.seq, r.seq_delta, r.tick_delta, r.arrival, r.flags, (unsigned long long)r.ls_rx, (unsigned long long)r.ls_lost, r.ls_gap_ms, r.clip_n,
      r.clip_mask, r.tilt, r.keys, r.mode_machine, r.ref_frame, r.ref_rp[0], r.ref_rp[1], r.ref_rp[2], r.ref_rq[0], r.ref_rq[1], r.ref_rq[2], r.ref_rq[3], r.rpy[0], r.rpy[1]);
  db(r.q_pd); fl(r.action_prev); fl(r.kp); fl(r.kd); fl(r.tau_pd); fl(r.ref_jp); fl(r.ref_jv); fl(r.tau_ff); fl(r.tau_train);
  app(o, ",%d,%.4f,%u", r.dance_frame, r.dance_t, (unsigned)r.seg);
  o += '\n';
}

std::string clips_csv_header() {
  return "t_mono,t_rel,tick,phase,joint_idx,joint,reason,raw,after_pos,after_slew,after_torque,final,delta,lo,hi,prev_target,slew_step,tau_pre_nm,tau_lim_nm,q_meas\n";
}
std::string frames_csv_header(const std::vector<std::string>& names) {
  std::string h = "t_mono,kind,state_seq,send_us";
  for (const char* g : {"q_meas", "q_cmd", "kp", "kd"})
    for (auto& n : names) h += std::string(",") + g + ":" + n;
  return h + "\n";
}

static void put_u32(std::string& s, uint32_t v) { s.append((const char*)&v, 4); }
static void put_f64(std::string& s, double v) { s.append((const char*)&v, 8); }
static void put_u64(std::string& s, uint64_t v) { s.append((const char*)&v, 8); }
static void put_f32v(std::string& s, const float* v, int n) { s.append((const char*)v, (size_t)n * 4); }

// ------------------------------------------------------------------ lowstate monitor
void LowstateMonitor::feed(const LinkState& s) {
  rx_++;
  double la = last_arrival_.load();
  if (la > 0) {
    double gap = (s.arrival - la) * 1e3;
    last_gap_ms_ = gap;
    if (gap > max_gap_ms_.load()) max_gap_ms_ = gap;
    {
      std::lock_guard<std::mutex> lk(hm_);
      hist_[std::min<size_t>(hist_.size() - 1, (size_t)(gap / 0.5))]++;
    }
    uint32_t lt = last_tick_.load();
    uint32_t dt = s.tick - lt;
    uint32_t nom = nominal_dtick_.load();
    if (dt == 0) dup_++;
    if (nom == 0) {
      std::lock_guard<std::mutex> lk(hm_);
      if (dt > 0 && dt < 1000 && n_first_ < 16) first_ticks_[n_first_++] = dt;
      if (n_first_ >= 16) {
        std::vector<uint32_t> v(first_ticks_, first_ticks_ + 16);
        std::sort(v.begin(), v.end());
        nominal_dtick_ = std::max<uint32_t>(1, v[8]);
      }
    } else if (dt > nom + nom / 2 && dt < 100000) {
      lost_ += (uint64_t)std::max<uint32_t>(1, (dt + nom / 2) / nom) - 1;
    }
  }
  last_arrival_ = s.arrival;
  last_tick_ = s.tick;
}
void LowstateMonitor::reset_window() {
  rx_ = 0; lost_ = 0; dup_ = 0; last_gap_ms_ = 0; max_gap_ms_ = 0;
  std::lock_guard<std::mutex> lk(hm_);
  hist_.fill(0);
}
LowstateMonitor::Snap LowstateMonitor::snap() const {
  Snap r;
  r.rx = rx_.load(); r.lost = lost_.load(); r.last_gap_ms = last_gap_ms_.load(); r.max_gap_ms = max_gap_ms_.load();
  std::lock_guard<std::mutex> lk(hm_);
  uint64_t tot = 0;
  for (auto c : hist_) tot += c;
  if (tot) {
    uint64_t a = 0;
    bool p50 = false;
    for (size_t i = 0; i < hist_.size(); i++) {
      a += hist_[i];
      if (!p50 && a >= tot * 0.5) { r.p50_gap_ms = (i + 0.5) * 0.5; p50 = true; }
      if (a >= tot * 0.99) { r.p99_gap_ms = (i + 0.5) * 0.5; break; }
    }
  }
  return r;
}

// ------------------------------------------------------------------ clip tracker
static bool stage_vals(const ClipRow& r, uint8_t bit, double& in, double& out) {
  switch (bit) {
    case CLIP_POS: in = r.raw; out = r.after_pos; return true;
    case CLIP_SLEW: in = r.after_pos; out = r.after_slew; return true;
    case CLIP_TORQUE: in = r.after_slew; out = r.after_torque; return true;
    case CLIP_POS2: in = r.after_torque; out = r.final_q; return true;
    case CLIP_KP: in = 0; out = 0; return true;
  }
  return false;
}

void ClipTracker::tick(int n, double t, const std::vector<ClipRow>& rows) {
  ticks_total++;
  if (!rows.empty()) ticks_with_clip++;
  for (const ClipRow& r : rows) {
    for (uint8_t bit : {CLIP_POS, CLIP_SLEW, CLIP_TORQUE, CLIP_POS2, CLIP_KP}) {
      if (!(r.mask & bit)) continue;
      double in = 0, out = 0;
      stage_vals(r, bit, in, out);
      double ex = std::fabs(in - out);
      Open* op = nullptr;
      for (auto& o : open_) if (o.e.joint == r.joint && o.e.reason == bit && o.last_seen == n - 1) { op = &o; break; }
      if (!op) {
        Open o;
        o.e.joint = r.joint; o.e.reason = bit; o.e.first_tick = n; o.e.t_start = t; o.e.max_excess = -1; o.last_seen = n - 1;
        open_.push_back(o);
        op = &open_.back();
      }
      op->e.last_tick = n; op->e.t_end = t; op->e.n_ticks++;
      op->last_seen = n;
      if (ex > op->e.max_excess) { op->e.max_excess = ex; op->e.raw_at_max = in; op->e.clipped_at_max = out; }
    }
  }
  for (size_t i = 0; i < open_.size();) {  // not continued this tick -> closed
    if (open_[i].last_seen < n) {
      open_[i].e.duration_s = open_[i].e.t_end - open_[i].e.t_start + dt_;
      done_.push_back(open_[i].e);
      open_.erase(open_.begin() + i);
    } else i++;
  }
}
void ClipTracker::finish() {
  for (auto& o : open_) { o.e.duration_s = o.e.t_end - o.e.t_start + dt_; done_.push_back(o.e); }
  open_.clear();
}
std::string ClipTracker::summary_json(const std::vector<std::string>& names, size_t max_episodes) const {
  std::map<std::string, std::map<std::string, std::array<double, 3>>> pj;  // joint -> reason -> {ticks, max_excess, longest_s}
  for (const auto& e : done_) {
    std::string jn_ = e.joint >= 0 && (size_t)e.joint < names.size() ? names[e.joint] : std::to_string(e.joint);
    auto& a = pj[jn_][clip_reason_text(e.reason)];
    a[0] += e.n_ticks; a[1] = std::max(a[1], e.max_excess); a[2] = std::max(a[2], e.duration_s);
  }
  std::string s = "{\"ticks_total\":" + std::to_string(ticks_total) + ",\"ticks_with_clip\":" + std::to_string(ticks_with_clip) + ",\"n_episodes\":" + std::to_string(done_.size()) + ",\"per_joint\":{";
  bool f1 = true;
  for (auto& j : pj) {
    s += (f1 ? "" : ",") + jq(j.first) + ":{";
    f1 = false;
    bool f2 = true;
    for (auto& r : j.second) {
      s += (f2 ? "" : ",") + jq(r.first) + ":{\"ticks\":" + jn(r.second[0]) + ",\"max_excess\":" + jn(r.second[1]) + ",\"longest_s\":" + jn(r.second[2]) + "}";
      f2 = false;
    }
    s += "}";
  }
  s += "},\"episodes\":[";
  std::vector<const ClipEpisode*> v;
  for (auto& e : done_) v.push_back(&e);
  std::sort(v.begin(), v.end(), [](const ClipEpisode* a, const ClipEpisode* b) { return a->duration_s > b->duration_s; });
  for (size_t i = 0; i < v.size() && i < max_episodes; i++) {
    const ClipEpisode& e = *v[i];
    std::string jname = e.joint >= 0 && (size_t)e.joint < names.size() ? names[e.joint] : std::to_string(e.joint);
    s += std::string(i ? "," : "") + "{\"joint\":" + jq(jname) + ",\"reason\":" + jq(clip_reason_text(e.reason)) + ",\"first_tick\":" + std::to_string(e.first_tick) + ",\"last_tick\":" + std::to_string(e.last_tick) +
         ",\"n_ticks\":" + std::to_string(e.n_ticks) + ",\"t_start\":" + jn(e.t_start) + ",\"duration_s\":" + jn(e.duration_s) + ",\"max_excess\":" + jn(e.max_excess) + ",\"raw_at_max\":" + jn(e.raw_at_max) +
         ",\"clipped_at_max\":" + jn(e.clipped_at_max) + "}";
  }
  return s + "]}";
}

std::string log_stats_fields(const LogStats& s, const LowstateMonitor& mon) {
  auto u = [](uint64_t v) { return std::to_string((unsigned long long)v); };
  std::string o = "\"log_stats_ver\":2,\"ticks\":" + u(s.ticks) + ",\"dropped_ticks\":" + u(s.dropped_ticks) + ",\"dropped_states\":" + u(s.dropped_states) + ",\"dropped_frames\":" + u(s.dropped_frames) + ",\"dropped_clips\":" + u(s.dropped_clips) +
                  ",\"dropped_events\":" + u(s.dropped_events) + ",\"state_ring_evicted\":" + u(s.state_ring_evicted) + ",\"state_preroll_trimmed\":" + u(s.state_preroll_trimmed) + ",\"state_pushed\":" + u(s.state_pushed) + ",\"states_written\":" + u(s.states) +
                  ",\"preroll_written_s\":" + jn(s.preroll_written_s) + ",\"writer_max_lag_ms\":" + jn(s.max_queue_lag_ms) + ",\"write_errors\":" + u(s.write_errors) + ",\"max_tick_queue\":" + u(s.max_tick_queue) +
                  ",\"max_stream_state_queue\":" + u(s.max_stream_state_queue) + ",\"max_event_queue\":" + u(s.max_event_queue) + ",\"max_frame_queue\":" + u(s.max_frame_queue) + ",\"max_clip_queue\":" + u(s.max_clip_queue) +
                  ",\"state_cap\":" + u(s.state_cap) + ",\"tick_cap\":" + u(s.tick_cap) + ",\"event_cap\":" + u(s.event_cap) + ",\"frame_cap\":" + u(s.frame_cap) + ",\"clip_cap\":" + u(s.clip_cap) +
                  ",\"writer_pass_max_ms\":" + jn(s.writer_pass_max_ms) + ",\"fdatasync_max_ms\":" + jn(s.fdatasync_max_ms) + ",\"fdatasync_mean_ms\":" + jn(s.fdatasync_mean_ms) + ",\"fdatasync_n\":" + u(s.fdatasync_n);
  const LowstateMonitor::Snap sn = mon.snap();
  o += ",\"lowstate_rx\":" + u(sn.rx) + ",\"lowstate_lost_est\":" + u(sn.lost) + ",\"lowstate_dup_ticks\":" + u(mon.dup_ticks()) + ",\"lowstate_nominal_dtick\":" + u(mon.nominal_dtick()) + ",\"lowstate_max_gap_ms\":" + jn(sn.max_gap_ms) +
       ",\"lowstate_p99_gap_ms\":" + jn(sn.p99_gap_ms);
  return o;
}

// ------------------------------------------------------------------ run stats
Pctl percentiles(std::vector<double> v) {
  Pctl p;
  p.n = v.size();
  if (v.empty()) return p;
  std::sort(v.begin(), v.end());
  auto at = [&](double q) { return v[std::min(v.size() - 1, (size_t)(q * (double)(v.size() - 1) + 0.5))]; };
  p.p50 = at(0.5); p.p95 = at(0.95); p.p99 = at(0.99); p.max = v.back();
  double s = 0;
  for (double x : v) s += x;
  p.mean = s / (double)v.size();
  return p;
}
std::string RunStats::timing_json() const {
  auto col = [&](double TickLite::*m) { std::vector<double> v; for (auto& t : ticks) v.push_back(t.*m); return v; };
  auto pj = [&](const char* name, const Pctl& p) {
    return jq(name) + ":{\"n\":" + std::to_string(p.n) + ",\"mean\":" + jn(p.mean) + ",\"p50\":" + jn(p.p50) + ",\"p95\":" + jn(p.p95) + ",\"p99\":" + jn(p.p99) + ",\"max\":" + jn(p.max) + "}";
  };
  long late = 0, rep = 0, sync_to = 0, gap = 0, slow = 0, clip_ticks = 0;
  uint64_t lost = 0;
  double tilt_max = 0, gyro_max = 0, dq_max = 0, tau_max = 0, te_max = 0, re_max = 0;
  std::vector<double> te, re;
  for (auto& t : ticks) {
    late += (t.flags & TF_LATE) != 0; rep += (t.flags & TF_REPEAT_STATE) != 0; sync_to += (t.flags & TF_SYNC_TIMEOUT) != 0;
    gap += (t.flags & TF_TICK_GAP) != 0; slow += (t.flags & TF_SLOW_INFER) != 0; clip_ticks += t.clip_n > 0;
    lost = std::max(lost, t.ls_lost);
    tilt_max = std::max(tilt_max, t.tilt); gyro_max = std::max(gyro_max, t.gyro);
    dq_max = std::max(dq_max, t.max_dq_ratio); tau_max = std::max(tau_max, t.max_tau_ratio);
    te_max = std::max(te_max, t.track_err_max); re_max = std::max(re_max, t.ref_err_max);
    te.push_back(t.track_err_max); re.push_back(t.ref_err_max);
  }
  std::string s = "{\"ticks\":" + std::to_string(ticks.size()) + ",";
  s += pj("dt_ms", percentiles(col(&TickLite::dt_ms))) + "," + pj("infer_ms", percentiles(col(&TickLite::infer_ms))) + "," + pj("step_ms", percentiles(col(&TickLite::step_ms))) + "," +
       pj("state_age_ms", percentiles(col(&TickLite::state_age_ms))) + "," + pj("send_us", percentiles(col(&TickLite::send_us))) + "," + pj("dds_loopback_ms", percentiles(col(&TickLite::dds_loop_ms))) + "," +
       pj("cmd_tracking_err_rad", percentiles(te)) + "," + pj("ref_tracking_err_rad", percentiles(re)) + ",";
  s += "\"late_ticks\":" + std::to_string(late) + ",\"repeat_state_ticks\":" + std::to_string(rep) + ",\"sync_timeout_ticks\":" + std::to_string(sync_to) + ",\"lowstate_gap_ticks\":" + std::to_string(gap) +
       ",\"slow_infer_ticks\":" + std::to_string(slow) + ",\"ticks_with_clip\":" + std::to_string(clip_ticks) + ",\"lowstate_lost_est\":" + std::to_string((unsigned long long)lost) + ",\"max_tilt_rad\":" + jn(tilt_max) +
       ",\"max_gyro_rad_s\":" + jn(gyro_max) + ",\"max_dq_over_limit\":" + jn(dq_max) + ",\"max_tau_est_over_limit\":" + jn(tau_max) + ",\"max_q_minus_qcmd_rad\":" + jn(te_max) + ",\"max_q_minus_ref_rad\":" + jn(re_max) + "}";
  return s;
}

// ------------------------------------------------------------------ logger
struct RunLogger::Impl {
  int fd_ev = -1, fd_tr = -1, fd_obs = -1, fd_st = -1, fd_fr = -1, fd_cl = -1;
  BQ<EvMsg> evq;
  BQ<TickRec> tq;
  BQ<StateRec> sq;
  BQ<FrameRec> fq;
  BQ<ClipRow> cq;
  std::thread th;
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> passes{0};
  std::atomic<uint64_t> ev_seq{0};
  LogStats st;
  std::mutex st_m;
  uint64_t d_tick = 0, d_state = 0, d_frame = 0, d_clip = 0, d_event = 0;  // real drops
  uint64_t ev_state = 0, push_state = 0;  std::atomic<uint64_t> trim_state{0};                    // ring evictions (by design) / pre-roll trimmed at begin_stream / all pushed
  std::atomic<size_t> max_sq_stream{0}, max_eq{0}, max_fq{0}, max_cq{0};
  std::atomic<double> first_state_t{0}, pass_max_ms{0}, fsync_max_ms{0}, fsync_total_ms{0};
  std::atomic<uint64_t> fsync_n{0};
  std::atomic<uint64_t> bytes{0}, werr{0};
  std::atomic<double> max_lag_ms{0};
  double wall_off = 0;
  std::vector<std::string> names;
  int obs_dim = 0;
  std::map<std::string, std::string> paths;
  std::string ev_buf;
  std::atomic<double> stream_t0{0};
  std::atomic<bool> ever_streamed{false};
  double preroll_s = 5.0;
};

RunLogger::RunLogger() : I(new Impl) {}
RunLogger::~RunLogger() { close(); }

static void wr(void*, int fd, const std::string& s, std::atomic<uint64_t>& bytes, std::atomic<uint64_t>& werr) {
  if (fd < 0 || s.empty()) return;
  const char* p = s.data();
  size_t left = s.size();
  while (left) {
    ssize_t n = ::write(fd, p, left);
    if (n < 0) { if (errno == EINTR) continue; werr++; return; }
    p += n; left -= (size_t)n; bytes += (uint64_t)n;
  }
}

bool RunLogger::open(const LogOptions& o) {
  if (open_) return true;
  o_ = o;
  ::mkdir(o.dir.c_str(), 0775);
  I->names = o.joint_names;
  I->obs_dim = o.obs_dim;
  I->preroll_s = o.preroll_s;
  I->wall_off = wall() - mono();
  auto mk = [&](const char* name, int flags = O_WRONLY | O_CREAT | O_TRUNC) {
    std::string p = o.dir + "/" + name;
    int fd = ::open(p.c_str(), flags | O_CLOEXEC, 0664);
    I->paths[name] = p;
    return fd;
  };
  I->fd_ev = mk("events.jsonl", O_WRONLY | O_CREAT | O_APPEND);
  I->fd_tr = mk("trace.csv");
  I->fd_obs = o.obs_bin ? mk("obs.bin") : -1;
  I->fd_st = o.state_stream ? mk("state_stream.bin") : -1;
  I->fd_fr = mk("exit_frames.csv");
  I->fd_cl = mk("safety_clips.csv");
  if (I->fd_ev < 0 || I->fd_tr < 0) return false;
  I->evq.reset(o.event_cap); I->tq.reset(o.tick_cap); I->sq.reset(o.state_cap); I->fq.reset(o.frame_cap); I->cq.reset(o.clip_cap);
  wr(nullptr, I->fd_tr, trace_csv_header(o.joint_names), I->bytes, I->werr);
  wr(nullptr, I->fd_fr, frames_csv_header(o.joint_names), I->bytes, I->werr);
  wr(nullptr, I->fd_cl, clips_csv_header(), I->bytes, I->werr);
  if (I->fd_obs >= 0) {
    std::string js = "{\"format\":\"PMOB1\",\"obs_dim\":" + std::to_string(o.obs_dim) + ",\"record\":\"u32 n, f64 t_mono, f32[obs_dim]\",\"layout\":" + o.obs_layout_json + "}";
    std::string h(OBS_MAGIC);
    h.append(3, '\0');
    put_u32(h, (uint32_t)js.size());
    h += js;
    wr(nullptr, I->fd_obs, h, I->bytes, I->werr);
  }
  if (I->fd_st >= 0) {
    std::string js = "{\"format\":\"PMST1\",\"record_bytes\":" + std::to_string(STATE_REC_BYTES) +
                     ",\"record\":\"f64 t_mono, u64 seq, u32 tick, i32 mode_machine, f32 q[29], f32 dq[29], f32 tau_est[29], f32 quat_wxyz[4], f32 gyro[3], f32 rpy[3], u8 wireless_remote[40]\",\"n_joints\":" + std::to_string(N) + "}";
    std::string h(STATE_MAGIC);
    h.append(3, '\0');
    put_u32(h, (uint32_t)js.size());
    h += js;
    wr(nullptr, I->fd_st, h, I->bytes, I->werr);
  }
  open_ = true;
  Impl* im = I.get();
  const std::vector<std::string> names = o.joint_names;
  im->th = std::thread([this, im, names]() {
    std::unique_ptr<TickRec> tr(new TickRec());
    StateRec sr;
    FrameRec fr;
    ClipRow cr;
    EvMsg em;
    double t_sync = mono();
    double stall = o_.dev_stall_s;
    std::string b_tr, b_obs, b_st, b_fr, b_cl, b_ev;
    bool backlog_drained = false;
    for (;;) {
      bool stopping = im->stop.load();
      double now = mono();
      b_tr.clear(); b_obs.clear(); b_st.clear(); b_fr.clear(); b_cl.clear(); b_ev.clear();
      int guard = 0;
      { size_t q = im->evq.size(); if (q > im->max_eq.load()) im->max_eq = q; q = im->fq.size(); if (q > im->max_fq.load()) im->max_fq = q; q = im->cq.size(); if (q > im->max_cq.load()) im->max_cq = q; }
      while (guard < 4096 && im->evq.pop(em)) { guard++; b_ev += em.line; im->max_lag_ms = std::max(im->max_lag_ms.load(), (now - em.t) * 1e3); im->st.events++; }
      guard = 0;
      while (guard < 512 && im->tq.pop(*tr)) { guard++;
        im->max_lag_ms = std::max(im->max_lag_ms.load(), (now - tr->t_mono) * 1e3);
        trace_csv_row(*tr, b_tr);
        if (im->fd_obs >= 0) {
          put_u32(b_obs, tr->n);
          put_f64(b_obs, tr->t_mono);
          put_f32v(b_obs, tr->obs, im->obs_dim);
        }
        im->st.ticks++;
      }
      if (streaming_.load() || stopping) {
        guard = 0;
        const double keep_from = im->stream_t0.load() - im->preroll_s;
        const bool keep_any = im->ever_streamed.load();
        if (streaming_.load()) { size_t q = im->sq.size(); if (!backlog_drained && (q < 256 || now - im->stream_t0.load() > 1.0)) backlog_drained = true; if (backlog_drained && q > im->max_sq_stream.load()) im->max_sq_stream = q; }  // peak AFTER the pre-roll backlog was drained (the first passes write it out)
        while (guard < 4096 && im->sq.pop(sr)) { guard++;
          if (!keep_any || sr.t_mono < keep_from) { if (keep_any) im->trim_state++; continue; }  // older than the pre-roll window (discarded by design, counted apart) / run was never taken over: not recorded
          if (im->first_state_t.load() == 0) im->first_state_t = sr.t_mono;
          put_f64(b_st, sr.t_mono); put_u64(b_st, sr.seq); put_u32(b_st, sr.tick); put_u32(b_st, (uint32_t)sr.mode_machine);
          put_f32v(b_st, sr.q, N); put_f32v(b_st, sr.dq, N); put_f32v(b_st, sr.tau, N); put_f32v(b_st, sr.quat, 4); put_f32v(b_st, sr.gyro, 3); put_f32v(b_st, sr.rpy, 3);
          b_st.append((const char*)sr.wireless, 40);
          im->st.states++;
        }
      }
      guard = 0;
      while (guard < 1024 && im->fq.pop(fr)) { guard++;
        app(b_fr, "%.6f,%s,%llu,%.2f", fr.t_mono, frame_kind_name(fr.kind), (unsigned long long)fr.state_seq, fr.send_us);
        for (const float* a : {fr.q_meas, fr.q_cmd, fr.kp, fr.kd}) for (int i = 0; i < N; i++) app(b_fr, ",%.9g", a[i]);
        b_fr += '\n';
        im->st.frames++;
      }
      guard = 0;
      while (guard < 4096 && im->cq.pop(cr)) { guard++;  // V1.6: the guard is checked BEFORE the pop (V1.5.1 popped first and silently lost the item that hit the per-pass limit: 1 sample per pass with a >4096 backlog)
        app(b_cl, "%.6f,%.4f,%d,%s,%d,%s,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n", cr.t_mono, cr.t_rel, cr.tick, cr.phase == PH_RUN ? "RUN" : "EXIT", cr.joint,
            (size_t)cr.joint < names.size() ? names[cr.joint].c_str() : "?", clip_reason_text(cr.mask), cr.raw, cr.after_pos, cr.after_slew, cr.after_torque, cr.final_q, cr.final_q - cr.raw, cr.lo, cr.hi,
            cr.prev_target, cr.slew_step, cr.tau_pre_nm, cr.tau_lim_nm, cr.q_meas);
        im->st.clips++;
      }
      const double t_pass0 = now;
      wr(nullptr, im->fd_ev, b_ev, im->bytes, im->werr);
      wr(nullptr, im->fd_tr, b_tr, im->bytes, im->werr);
      wr(nullptr, im->fd_obs, b_obs, im->bytes, im->werr);
      wr(nullptr, im->fd_st, b_st, im->bytes, im->werr);
      wr(nullptr, im->fd_fr, b_fr, im->bytes, im->werr);
      wr(nullptr, im->fd_cl, b_cl, im->bytes, im->werr);
      { const double pm = (mono() - t_pass0) * 1e3; if (pm > im->pass_max_ms.load()) im->pass_max_ms = pm; }
      im->passes++;
      if (stall > 0 && im->st.ticks > 0) { logf("[runlog] DEV: writer stalled for %.1f s", stall); sleep_s(stall); stall = 0; }
      if (mono() - t_sync > 1.0) {
        t_sync = mono();
        const double t_fs = mono();
        for (int fd : {im->fd_ev, im->fd_tr, im->fd_obs, im->fd_st, im->fd_fr, im->fd_cl}) if (fd >= 0) ::fdatasync(fd);
        const double fm = (mono() - t_fs) * 1e3;
        if (fm > im->fsync_max_ms.load()) im->fsync_max_ms = fm;
        im->fsync_total_ms = im->fsync_total_ms.load() + fm; im->fsync_n++;
      }
      if (stopping && im->evq.size() == 0 && im->tq.size() == 0 && im->fq.size() == 0 && im->cq.size() == 0 && im->sq.size() == 0) break;
      if (!stopping) sleep_s(0.004);
    }
    for (int fd : {im->fd_ev, im->fd_tr, im->fd_obs, im->fd_st, im->fd_fr, im->fd_cl}) if (fd >= 0) { ::fdatasync(fd); }
  });
  {
    std::lock_guard<std::mutex> lk(g_reg_m);
    g_reg.push_back(this);
  }
  if (o.install_crash_handlers && !g_handlers_installed.exchange(true)) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = crash_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND;
    for (int s : {SIGSEGV, SIGABRT, SIGBUS, SIGFPE, SIGILL}) sigaction(s, &sa, nullptr);
    std::set_terminate(terminate_handler);
  }
  return true;
}

void RunLogger::event_at(double t_mono, const std::string& kind, const std::string& fields) {
  if (!open_) return;
  EvMsg m;
  char b[160];
  snprintf(b, sizeof b, "{\"seq\":%llu,\"t_mono\":%.6f,\"wall\":%.4f,\"kind\":\"", (unsigned long long)(I->ev_seq++), t_mono, t_mono + I->wall_off);
  m.line = b;
  m.line += json_escape(kind) + "\"";
  if (!fields.empty()) { m.line += ","; m.line += fields; }
  m.line += "}\n";
  m.t = mono();
  I->evq.push(m, false, I->d_event);
}
void RunLogger::event(const std::string& kind, const std::string& fields) { event_at(mono(), kind, fields); }
bool RunLogger::tick(const TickRec& r) { return open_ && I->tq.push(r, false, I->d_tick); }
bool RunLogger::frame(const FrameRec& f) { return open_ && I->fq.push(f, false, I->d_frame); }
bool RunLogger::clip(const ClipRow& c) { return open_ && I->cq.push(c, false, I->d_clip); }
void RunLogger::state(const LinkState& s) {
  mon_.feed(s);
  if (!open_ || I->fd_st < 0) return;
  StateRec r;
  r.t_mono = s.arrival; r.seq = s.seq; r.tick = s.tick; r.mode_machine = s.mode_machine;
  for (int i = 0; i < N; i++) { r.q[i] = (float)s.q[i]; r.dq[i] = (float)s.dq[i]; r.tau[i] = (float)s.tau[i]; }
  for (int i = 0; i < 4; i++) r.quat[i] = (float)s.quat[i];
  for (int i = 0; i < 3; i++) { r.gyro[i] = (float)s.gyro[i]; r.rpy[i] = (float)s.rpy[i]; }
  memcpy(r.wireless, s.wireless.data(), 40);
  I->push_state++;
  // before begin_stream(): ROLLING pre-roll ring - when full the oldest sample is overwritten = `state_ring_evicted` (by design, NOT a loss).
  // afterwards: the stream; a full queue loses the NEW sample = `dropped_states` (a real loss).
  I->sq.push(r, !streaming_.load(), I->d_state, &I->ev_state);
}
void RunLogger::begin_stream() {
  if (!open_ || streaming_) return;
  I->stream_t0 = mono();
  I->ever_streamed = true;
  // V1.6: the ring is normally FULL here (it rolled for as long as READY lasted) and nearly all of it is older than the pre-roll window: trim it now, otherwise the stream starts with a full queue
  // and the first samples after the take-over are lost until the writer has drained the backlog (seen on the robot: 4 samples at ARM after a 65 s wait).
  const double keep_from = I->stream_t0.load() - I->preroll_s;
  const size_t trimmed = I->sq.trim_front([keep_from](const StateRec& r) { return r.t_mono < keep_from; });
  // headroom for the first stream samples while the writer drains what is left: if the ring is (nearly) full of in-window samples (small ring / very high rate) the OLDEST pre-roll samples go (counted as trimmed,
  // the window gets shorter and that is VISIBLE as preroll_written_s) rather than the NEW stream samples being dropped
  const size_t cap = I->sq.capacity(), head = std::max<size_t>(64, cap / 16);
  const size_t cut = cap > head ? I->sq.trim_to_free(head) : 0;
  I->trim_state += trimmed + cut;
  streaming_ = true;
  event("state_stream_begin", "\"preroll_trimmed\":" + std::to_string(trimmed) + ",\"preroll_cut_for_headroom\":" + std::to_string(cut) + ",\"state_queue_after_trim\":" + std::to_string(I->sq.size()) + ",\"preroll_s\":" + jn(I->preroll_s) + ",\"ring_evicted_so_far\":" + std::to_string(I->ev_state));
}
void RunLogger::end_stream() {
  if (!open_ || !streaming_) return;
  event("state_stream_end");
  double t_end = mono() + 1.0;
  while (I->sq.size() > 0 && mono() < t_end) sleep_s(0.002);
  streaming_ = false;
}
bool RunLogger::flush(double timeout_s) {
  if (!open_) return true;
  double t_end = mono() + timeout_s;
  uint64_t p0 = I->passes.load();
  while (mono() < t_end) {
    bool empty = I->evq.size() == 0 && I->tq.size() == 0 && I->fq.size() == 0 && I->cq.size() == 0 && (!streaming_.load() || I->sq.size() == 0);
    if (empty && I->passes.load() >= p0 + 2) return true;
    sleep_s(0.002);
  }
  return false;
}
void RunLogger::emergency_drain(double max_s) { flush(max_s); }
void RunLogger::flush_all(double max_s) {
  std::vector<RunLogger*> v;
  {
    std::unique_lock<std::mutex> lk(g_reg_m, std::try_to_lock);
    if (!lk.owns_lock()) return;
    v = g_reg;
  }
  for (auto* l : v) if (l->open_) l->flush(max_s);
}
LogStats RunLogger::stats() const {
  LogStats s = I->st;
  s.dropped_ticks = I->d_tick; s.dropped_states = I->d_state; s.dropped_frames = I->d_frame; s.dropped_clips = I->d_clip; s.dropped_events = I->d_event;
  s.state_ring_evicted = I->ev_state; s.state_preroll_trimmed = I->trim_state; s.state_pushed = I->push_state;
  s.max_stream_state_queue = I->max_sq_stream.load(); s.max_event_queue = I->max_eq.load(); s.max_frame_queue = I->max_fq.load(); s.max_clip_queue = I->max_cq.load();
  s.state_cap = I->sq.capacity(); s.tick_cap = I->tq.capacity(); s.event_cap = I->evq.capacity(); s.frame_cap = I->fq.capacity(); s.clip_cap = I->cq.capacity();
  s.preroll_written_s = (I->ever_streamed.load() && I->first_state_t.load() > 0) ? I->stream_t0.load() - I->first_state_t.load() : -1.0;
  s.writer_pass_max_ms = I->pass_max_ms.load(); s.fdatasync_max_ms = I->fsync_max_ms.load(); s.fdatasync_n = I->fsync_n.load();
  s.fdatasync_mean_ms = s.fdatasync_n ? I->fsync_total_ms.load() / (double)s.fdatasync_n : 0.0;
  s.bytes = I->bytes.load(); s.write_errors = I->werr.load(); s.max_queue_lag_ms = I->max_lag_ms.load(); s.max_tick_queue = I->tq.max_size();
  return s;
}
std::vector<std::pair<std::string, uint64_t>> RunLogger::file_sizes() const {
  std::vector<std::pair<std::string, uint64_t>> v;
  for (auto& p : I->paths) {
    struct stat st;
    v.push_back({p.first, ::stat(p.second.c_str(), &st) == 0 ? (uint64_t)st.st_size : 0});
  }
  return v;
}
void RunLogger::close() {
  if (!open_.exchange(false)) return;
  {
    std::lock_guard<std::mutex> lk(g_reg_m);
    g_reg.erase(std::remove(g_reg.begin(), g_reg.end(), this), g_reg.end());
  }
  I->stop = true;
  if (I->th.joinable()) I->th.join();
  for (int* fd : {&I->fd_ev, &I->fd_tr, &I->fd_obs, &I->fd_st, &I->fd_fr, &I->fd_cl}) if (*fd >= 0) { ::close(*fd); *fd = -1; }
}

}  // namespace pm
