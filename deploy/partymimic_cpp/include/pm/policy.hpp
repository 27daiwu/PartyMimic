// Frozen P2 SSB tracking policy: observation stack, ONNX inference, action mapping. Faithful C++ port of the Python deployment stack
// (sim2real Tracking + supermimic observations + BasePolicy action mapping), numerically checked against it (dev/parity_test).
#pragma once
#include <Eigen/Dense>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "pm/util.hpp"

namespace pm {

constexpr int N = 29;  // G1 29-dof, policy order == robot joint order (lowstate motor index)
using Vec = Eigen::Matrix<double, N, 1>;

struct Params {
  std::vector<std::string> names;
  Vec default_pos, action_scale, kp, kd, pos_lo, pos_hi, vel_lim, effort_lim;
  std::array<bool, N> vel_known{};
  int history = 10;
  std::vector<int> future;  // -1..10
  double rate_hz = 50.0;
  static bool load(const std::string& path, Params& out, std::string& err);
};

struct Motion {  // pre-processed SSB reference (float32 like the Python loader output)
  int T = 0;
  float fps = 50.f;
  std::vector<float> jp, jv, rp, rq;  // [T*29], [T*29], [T*3], [T*4]
  static bool load(const std::string& path, Motion& out, std::string& err);
  // P5.3: the clip robust_d was TRAINED on is ssb_padded = `n` frozen copies of frame 0 + the SSB (+ frozen tail, which the index clamp already gives). Returns that clip built in memory
  // (motion.bin on disk and its hash are untouched). n = 0 returns a copy.
  Motion with_lead_in(int n) const;
};

// V1.5.1: the planned timeline of ONE run, computed by the single function the runner uses for its window (and by selfcheck / the packager's checks).
//  running reference: [lead_in frozen first frame] + [the whole clip, motion_t 0..T-1] -> window over at the tick whose motion_t reaches seg_frames; then the LAST frame is held for end_hold_s.
//  paused reference : the first frame is held for run_s.
//  The runner's own fail-safe break (elapsed > lead_in + run_s + end_hold + 1 s) must stay AFTER the planned end (margin_s > 0), otherwise a slow loop could cut the dance.
struct RunTimeline {
  bool running = false;
  int lead_in_frames = 0, clip_frames = 0 /*motion incl. lead-in*/, dance_frames = 0 /*motion.bin*/, seg_frames = 0 /*window over at motion_t >= seg_frames*/, window_over_tick = 0;
  double fps = 50, dt = 0.02, lead_in_s = 0, dance_s = 0, run_s = 0, end_hold_s = 0;
  double planned_total_s = 0;   // take-over -> hand-back request (lead-in + dance + end hold)
  double failsafe_s = 0;        // runner break time (elapsed since take-over)
  double margin_s = 0;          // failsafe_s - planned_total_s
  bool covers_whole_dance = false;   // running: the window reaches the last frame (nothing cut)
  bool last_frame_held = false;      // the last frame is held for end_hold_s after the window
};
RunTimeline plan_timeline(const Motion& motion, bool running, double run_s, double lead_in_s, double end_hold_s, double dt = 0.02);

// robot sample in the dtype the Python stack used (float32 containers)
struct RobotSample {
  std::array<float, N> q{}, dq{};
  std::array<float, 4> quat{1, 0, 0, 0};  // wxyz
  std::array<float, 3> gyro{};
};

struct PolicyOut {
  std::array<float, N> action{};
  std::array<float, N> action_prev{};   // P5.1: the previous action fed to this tick's observation (last 29 values of the newest proprio frame)
  Vec q_target;
  int motion_t = 0;
  double infer_ms = 0;
  // P5.1: reference (dance clip, lead-in padded) at step 0 of this tick (for tracking-error analysis; NOT an extra model input)
  int ref_idx = 0;
  std::array<float, N> ref_jp{}, ref_jv{};
  std::array<float, 3> ref_rp{};
  std::array<float, 4> ref_rq{1, 0, 0, 0};
};
// P5.1: what a Sim2Sim / replay tool needs to rebuild the policy state of a run (written to events.jsonl as `policy_reset`)
struct PolicyResetInfo {
  std::array<double, 4> yaw_offset{1, 0, 0, 0};  // qmul(projected yaw of the reference root, conj(projected yaw of the robot root)) fixed at reset
  int motion_t = 0;
  bool paused = true;
  RobotSample sample;
};

class PolicyStack {
 public:
  // threads: ONNX Runtime intra-op threads
  PolicyStack(const Params& p, const Motion& m, const std::string& onnx_path, int threads);
  ~PolicyStack();
  // = Tracking.reset(): motion_t=0, observation buffers filled from `s`, previous action = 0. paused=false lets the SSB clock run.
  void reset(const RobotSample& s, bool paused);
  // = one Tracking.update() + prepare_obs + policy(): the same code path for warm-up, parity test and the live loop
  PolicyOut tick(const RobotSample& s);
  // last ONNX inputs (proprio, root_quat_hist, ref_joint_pos, ref_root) for dev diagnostics
  std::vector<std::vector<float>> last_inputs() const;
  // freeze the reference clock at the current frame (paused slice: zero reference velocity), used for the end-of-window settle
  void hold_reference();
  int motion_length() const { return m_.T; }
  const std::vector<std::string>& onnx_inputs() const;
  // P5.1 logging accessors: the 4 model inputs of the LAST tick() concatenated (proprio | root_quat_hist | ref_joint_pos | ref_root) into dst (obs_dim() floats)
  int obs_dim() const;
  void copy_obs(float* dst) const;
  std::vector<std::pair<std::string, int>> obs_layout() const;  // (input name, element count) in copy_obs order
  PolicyResetInfo reset_info() const;
  // P5.3 repeated-start check: everything a previous run could leave behind. A fresh stack and a reset one must be identical (verified by selfcheck 7j and logged per run).
  struct StateDigest { int motion_t = 0; bool paused = true; double prev_action_abs_max = 0; bool history_identical = false; bool quat_history_identical = false; int ref_slice_first = 0; };
  StateDigest state_digest() const;
  // read-only warm-up (n full ticks on `s`, returns the slowest in ms). Clobbers the episode state: call reset() afterwards.
  double warmup(const RobotSample& s, int n);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Params p_;
  const Motion& m_;
};

}  // namespace pm
