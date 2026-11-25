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
};

// robot sample in the dtype the Python stack used (float32 containers)
struct RobotSample {
  std::array<float, N> q{}, dq{};
  std::array<float, 4> quat{1, 0, 0, 0};  // wxyz
  std::array<float, 3> gyro{};
};

struct PolicyOut {
  std::array<float, N> action{};
  Vec q_target;
  int motion_t = 0;
  double infer_ms = 0;
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
  // read-only warm-up (n full ticks on `s`, returns the slowest in ms). Clobbers the episode state: call reset() afterwards.
  double warmup(const RobotSample& s, int n);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Params p_;
  const Motion& m_;
};

}  // namespace pm
