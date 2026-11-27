// P5.1 run logging: complete, asynchronous, crash-tolerant recording of one test run (see LOGGING.md for the file formats).
//
//   * The control thread only COPIES a plain struct into a pre-allocated bounded queue (microseconds, never blocks on disk, never allocates for ticks).
//     A single writer thread formats and writes every file with write(2) (no stdio buffering: a kill -9 loses at most the last ~10 ms), fdatasync once per second.
//   * On SIGSEGV/SIGABRT/SIGBUS/SIGFPE/SIGILL and on std::terminate the handler gives the writer thread a moment to drain the queues, writes a CRASH marker, then
//     re-raises. fast_exit() drains too (log_flush_all()).
//   * Files in <run_dir>/ (all times are mono() seconds = steady_clock; wall = t_mono + wall_offset from manifest.json):
//       trace.csv          one row per control tick: observation-side raw state, action, history action, reference, PD target, final command, timing
//       obs.bin            the exact float32 inputs of the ONNX model for every tick (bit-exact policy replay)
//       state_stream.bin   every rt/lowstate sample (full DDS rate): 5 s PRE-ROLL before take-over, the run, and a POST-exit window
//       exit_frames.csv    every command frame NOT in trace.csv: acquire stream, HOLD / DAMP frames of the exit
//       safety_clips.csv   one row per (tick, joint) in which the safety guard changed the command: reason, raw value, clipped value, limits
//       events.jsonl       timeline of everything else: phases, RPC request/response, FSM changes, remote-controller events, exit steps, clip episodes, faults
#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "pm/guard.hpp"
#include "pm/link.hpp"
#include "pm/policy.hpp"
#include "pm/util.hpp"

namespace pm {

constexpr int OBS_MAX = 1536;  // >= 10*93 + 10*8 + 12*29 + 12*7 = 1442 (P2/P4 policy); the real size is PolicyStack::obs_dim()

enum Phase : uint8_t { PH_RUN = 0, PH_EXIT = 1 };

// flags of TickRec.flags
enum TickFlag : uint32_t {
  TF_LATE = 1u << 0,         // control period > 1.5 x DT
  TF_REPEAT_STATE = 1u << 1, // no new lowstate sample since the previous tick (the tick re-used a state)
  TF_SYNC_TIMEOUT = 1u << 2, // waited the full 8 ms and no new sample arrived
  TF_CLIP = 1u << 3,         // the safety guard changed at least one joint of this command
  TF_TICK_GAP = 1u << 4,     // the robot-side lowstate tick jumped by more than expected (lost lowstate samples since the previous control tick)
  TF_SLOW_INFER = 1u << 5,   // inference > 15 ms
};

struct TickRec {  // everything one control tick needs to be analysed / replayed offline (plain data: memcpy-able)
  uint32_t n = 0;                 // control tick index since take-over (0-based)
  double t_mono = 0, t_rel = 0;   // tick start (mono) / seconds since take-over
  uint8_t phase = PH_RUN;
  int32_t ssb_frame = 0;
  double authority = 0;
  float dt_ms = 0, infer_ms = 0, step_ms = 0, state_age_ms = 0, sync_wait_ms = 0, send_us = 0, state_age_send_ms = 0, dds_loop_ms = 0, ls_gap_ms = 0;
  uint64_t seq = 0, ls_rx = 0, ls_lost = 0;
  uint32_t seq_delta = 0, ls_tick = 0, tick_delta = 0, flags = 0, clip_n = 0, clip_mask = 0, keys = 0;
  int32_t mode_machine = 0, ref_frame = 0;
  double arrival = 0;
  uint8_t wireless[40] = {0};
  float quat[4] = {1, 0, 0, 0}, gyro[3] = {0, 0, 0}, rpy[3] = {0, 0, 0};
  float roll = 0, pitch = 0, tilt = 0;
  float q[N], dq[N], tau[N];                 // measured state used by this tick (lowstate)
  float action[N], action_prev[N];           // raw policy output / previous action (history input)
  double q_policy[N], q_pd[N];               // default + action*scale (raw policy target) / q_hold + authority*(q_policy - q_hold) = target handed to the guard
  float q_cmd[N], kp[N], kd[N];              // what was actually transmitted (after the safety guard, float32 as on the wire)
  float tau_pd[N];                           // kp*(q_cmd - q) - kd*dq: the PD torque the drive evaluates at the sampled state
  float tau_ff[N], tau_train[N];             // P5.3: feed-forward torque actually sent (0 unless guard_torque_ff) / the torque the TRAINING actuator would apply for the unclipped q_policy (effort-saturated)
  float ref_jp[N], ref_jv[N], ref_rp[3], ref_rq[4];  // dance reference at the CURRENT tick = step 0 of the observation future slices (joint pos / vel, root pos / quat wxyz); indexed by ssb_frame (lead-in included), see dance_frame
  // V1.6: `ssb_frame` is the index into the REPLAYED clip = lead_in_frames of frozen first frame + the dance. dance_frame = ssb_frame - lead_in_frames (< 0 while the lead-in is frozen, 0 = first dance frame, the SSB first frame),
  // dance_t = dance_frame / fps (s of dance time), seg = 0 lead-in | 1 dance | 2 end-hold (last frame held, window over) | 3 paused/frozen reference (hold mode: no dance at all)
  int32_t dance_frame = 0;
  double dance_t = 0;
  uint8_t seg = 1;
  int32_t obs_n = 0;
  float obs[OBS_MAX];                        // proprio_hist | root_quat_hist | ref_joint_pos | ref_root (the 4 ONNX inputs, concatenated)
  TickRec() {
    memset(q, 0, sizeof q); memset(dq, 0, sizeof dq); memset(tau, 0, sizeof tau); memset(action, 0, sizeof action); memset(action_prev, 0, sizeof action_prev);
    memset(q_policy, 0, sizeof q_policy); memset(q_pd, 0, sizeof q_pd); memset(q_cmd, 0, sizeof q_cmd); memset(kp, 0, sizeof kp); memset(kd, 0, sizeof kd);
    memset(tau_pd, 0, sizeof tau_pd); memset(tau_ff, 0, sizeof tau_ff); memset(tau_train, 0, sizeof tau_train); memset(ref_jp, 0, sizeof ref_jp); memset(ref_jv, 0, sizeof ref_jv); memset(ref_rp, 0, sizeof ref_rp); memset(ref_rq, 0, sizeof ref_rq);
    ref_rq[0] = 1;
  }
};

struct StateRec {  // one rt/lowstate sample (binary record, 452 bytes)
  double t_mono = 0;
  uint64_t seq = 0;
  uint32_t tick = 0;
  int32_t mode_machine = 0;
  float q[N], dq[N], tau[N], quat[4], gyro[3], rpy[3];
  uint8_t wireless[40];
};
constexpr size_t STATE_REC_BYTES = 8 + 8 + 4 + 4 + 4 * N * 3 + 4 * 4 + 4 * 3 + 4 * 3 + 40;

enum FrameKind2 : uint8_t { FK_ACQUIRE = 0, FK_POLICY = 1, FK_HOLD = 2, FK_DAMP = 3 };
inline const char* frame_kind_name(uint8_t k) { return k == FK_ACQUIRE ? "ACQUIRE" : k == FK_POLICY ? "POLICY" : k == FK_HOLD ? "HOLD" : "DAMP"; }
struct FrameRec {  // a command frame outside trace.csv
  double t_mono = 0;
  uint8_t kind = FK_HOLD;
  uint64_t state_seq = 0;
  float send_us = 0;
  float q_meas[N], q_cmd[N], kp[N], kd[N];
  FrameRec() { memset(q_meas, 0, sizeof q_meas); memset(q_cmd, 0, sizeof q_cmd); memset(kp, 0, sizeof kp); memset(kd, 0, sizeof kd); }
};

struct ClipRow {  // one clipped joint of one command (safety_clips.csv)
  double t_mono = 0, t_rel = 0;
  int32_t tick = -1;      // control tick index, -1 = exit frame
  uint8_t phase = PH_RUN;
  int32_t joint = 0;
  uint8_t mask = 0;       // GuardClip bits
  double raw = 0, after_pos = 0, after_slew = 0, after_torque = 0, final_q = 0;
  double lo = 0, hi = 0, prev_target = 0, slew_step = 0, tau_pre_nm = 0, tau_lim_nm = 0, q_meas = 0;
};

// ---- clip episodes: consecutive ticks with the same (joint, reason) -> duration, worst excess (pure logic, used by the runner and unit-tested)
struct ClipEpisode {
  int joint = 0;
  uint8_t reason = 0;  // single GuardClip bit
  int first_tick = 0, last_tick = 0, n_ticks = 0;
  double t_start = 0, t_end = 0, duration_s = 0;  // duration = (t of the last tick - t of the first tick) + one control period
  double max_excess = 0, raw_at_max = 0, clipped_at_max = 0;
};
class ClipTracker {
 public:
  explicit ClipTracker(double dt = 0.02) : dt_(dt) {}
  // call once per control tick with every row of that tick (may be empty): closes the episodes that were not continued
  void tick(int n, double t, const std::vector<ClipRow>& rows);
  void finish();  // closes everything open
  const std::vector<ClipEpisode>& episodes() const { return done_; }
  long ticks_with_clip = 0, ticks_total = 0;
  std::string summary_json(const std::vector<std::string>& names, size_t max_episodes = 40) const;
 private:
  struct Open { ClipEpisode e; int last_seen; };
  double dt_;
  std::vector<Open> open_;
  std::vector<ClipEpisode> done_;
};

struct LogStats {
  uint64_t ticks = 0, states = 0, frames = 0, clips = 0, events = 0;
  uint64_t dropped_ticks = 0, dropped_states = 0, dropped_frames = 0, dropped_clips = 0, dropped_events = 0;  // REAL losses only (queue full while recording): any value > 0 is a log defect
  // V1.6: the pre-roll ring is a ROLLING WINDOW by design (oldest sample overwritten when full, before begin_stream): that is NOT a loss and is counted apart from dropped_states.
  uint64_t state_ring_evicted = 0;      // overwritten in the pre-roll ring (by design)
  uint64_t state_preroll_trimmed = 0;   // older than the pre-roll window when the stream began (discarded by design)
  uint64_t state_pushed = 0;            // every lowstate sample handed to the logger
  size_t max_stream_state_queue = 0, max_event_queue = 0, max_frame_queue = 0, max_clip_queue = 0;  // peak depth while recording (state: streaming phase only)
  size_t state_cap = 0, tick_cap = 0, event_cap = 0, frame_cap = 0, clip_cap = 0;
  double preroll_written_s = -1;        // span of the pre-roll that really reached state_stream.bin (< preroll_s only if the ring was too small / the standby ring too short)
  double writer_pass_max_ms = 0, fdatasync_max_ms = 0, fdatasync_mean_ms = 0;  // writer thread: slowest pass / slowest + mean fdatasync (a slow disk stalls the writer, never the control loop)
  uint64_t fdatasync_n = 0;
  uint64_t bytes = 0;
  double max_queue_lag_ms = 0;  // oldest queued item age when the writer picked it up
  size_t max_tick_queue = 0;
  uint64_t write_errors = 0;
};

// lowstate arrival statistics (DDS reader thread -> cheap atomics)
class LowstateMonitor {
 public:
  void feed(const LinkState& s);
  struct Snap { uint64_t rx = 0, lost = 0; double last_gap_ms = 0, max_gap_ms = 0, p50_gap_ms = 0, p99_gap_ms = 0; };
  Snap snap() const;
  uint32_t nominal_dtick() const { return nominal_dtick_.load(); }  // ticks per sample the monitor settled on (1 for 1 kHz / ms ticks, 2 for 500 Hz): `lost` counts MISSING tick steps against it
  uint64_t dup_ticks() const { return dup_.load(); }              // consecutive samples with the SAME tick (not a loss: counted apart)
  void reset_max() { max_gap_ms_ = 0; }
  void reset_window();  // V1.5.1: a pre-warmed (standby) runner starts the run statistics at the trigger, not at process start
 private:
  std::atomic<uint64_t> rx_{0}, lost_{0}, dup_{0};
  std::atomic<double> last_gap_ms_{0}, max_gap_ms_{0}, last_arrival_{0};
  std::atomic<uint32_t> last_tick_{0};
  std::atomic<uint32_t> nominal_dtick_{0};
  mutable std::mutex hm_;
  std::array<uint32_t, 200> hist_{};  // 0.5 ms bins
  uint32_t first_ticks_[16] = {0};
  int n_first_ = 0;
};

struct LogOptions {
  std::string dir;
  std::vector<std::string> joint_names;
  int obs_dim = 1442;
  std::string obs_layout_json = "[]";   // [{"name":"proprio","dim":930},...]
  double preroll_s = 5.0;
  bool obs_bin = true, state_stream = true;
  size_t tick_cap = 1024, state_cap = 8192, frame_cap = 2048, clip_cap = 8192, event_cap = 8192;
  bool install_crash_handlers = true;
  std::string tag = "runner";
  double dev_stall_s = 0;   // DEV/TEST ONLY (--sim): the writer thread sleeps this long once, right after the first tick record (proves the control loop never waits for the disk)
};

class RunLogger {
 public:
  RunLogger();
  ~RunLogger();
  bool open(const LogOptions& o);
  bool is_open() const { return open_; }
  const std::string& dir() const { return o_.dir; }
  // ---- producers (any thread; never block on disk)
  void event(const std::string& kind, const std::string& fields_json = "");  // fields_json = comma separated JSON members without braces (may be empty)
  void event_at(double t_mono, const std::string& kind, const std::string& fields_json = "");
  bool tick(const TickRec& r);
  void state(const LinkState& s);         // DDS reader thread
  bool frame(const FrameRec& f);
  bool clip(const ClipRow& c);
  // state stream: samples are kept in a ring (pre-roll) until begin_stream(); then written to state_stream.bin until end_stream()
  void begin_stream();
  void end_stream();
  bool streaming() const { return streaming_; }
  // wait until everything queued so far is on disk (bounded)
  bool flush(double timeout_s = 2.0);
  void close();
  LogStats stats() const;
  LowstateMonitor& lowstate() { return mon_; }
  // for the crash handler / fast_exit
  void emergency_drain(double max_s);
  static void flush_all(double max_s);  // every open logger (used by fast_exit)
  std::vector<std::pair<std::string, uint64_t>> file_sizes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> I;
  LogOptions o_;
  std::atomic<bool> open_{false}, streaming_{false};
  LowstateMonitor mon_;
};

// V1.6 log_stats event members (log_stats_ver 2): `dropped_*` = REAL losses; `state_ring_evicted` / `state_preroll_trimmed` = the pre-roll window rolling by design; queue peaks, caps, writer/fdatasync timing
std::string log_stats_fields(const LogStats& s, const LowstateMonitor& mon);
// ---- formatting helpers (shared with the offline tools)
std::string trace_csv_header(const std::vector<std::string>& names);
void trace_csv_row(const TickRec& r, std::string& out);
std::string clips_csv_header();
std::string frames_csv_header(const std::vector<std::string>& names);
constexpr const char* OBS_MAGIC = "PMOB1";
constexpr const char* STATE_MAGIC = "PMST1";

// ---- run statistics (feed with TickLite; used for run_report.json and the anomaly summary, by the runner AND by the offline log tool)
struct TickLite {
  double t_rel = 0, dt_ms = 0, infer_ms = 0, step_ms = 0, state_age_ms = 0, send_us = 0, dds_loop_ms = 0, tilt = 0, gyro = 0;
  uint32_t flags = 0, seq_delta = 1, tick_delta = 0;
  uint64_t ls_lost = 0;
  double max_dq_ratio = 0, max_tau_ratio = 0, track_err_max = 0, ref_err_max = 0;  // |q - q_cmd|_inf / |q - ref_jp|_inf
  int clip_n = 0;
};
struct Pctl { double p50 = 0, p95 = 0, p99 = 0, max = 0, mean = 0; size_t n = 0; };
Pctl percentiles(std::vector<double> v);
struct RunStats {
  std::vector<TickLite> ticks;
  void add(const TickLite& t) { ticks.push_back(t); }
  std::string timing_json() const;
};

}  // namespace pm
