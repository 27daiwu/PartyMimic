#include "pm/policy.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>

namespace pm {

// ------------------------------------------------------------------ params / motion loading
bool Params::load(const std::string& path, Params& o, std::string& err) {
  KV kv;
  if (!kv.load(path)) { err = "cannot read " + path; return false; }
  auto vec = [&](const char* k, Vec& v) {
    auto a = kv.nums(k);
    if (a.size() != N + (std::string(k) == "joint_names" ? 1 : 0)) { err = std::string("bad length for ") + k; return false; }
    for (int i = 0; i < N; i++) v[i] = a[i];
    return true;
  };
  // "joint_names 29 n1 ... n29": first token is the count
  auto jn = kv.m.find("joint_names");
  if (jn == kv.m.end() || jn->second.size() != N + 1) { err = "joint_names"; return false; }
  o.names.assign(jn->second.begin() + 1, jn->second.end());
  auto vecc = [&](const char* k, Vec& v) {  // lines are "key 29 v1 ... v29"
    auto a = kv.nums(k);
    if (a.size() != N + 1 || (int)a[0] != N) { err = std::string("bad vector ") + k; return false; }
    for (int i = 0; i < N; i++) v[i] = a[i + 1];
    return true;
  };
  (void)vec;
  if (!(vecc("default_pos", o.default_pos) && vecc("action_scale", o.action_scale) && vecc("kp", o.kp) && vecc("kd", o.kd) &&
        vecc("pos_lo", o.pos_lo) && vecc("pos_hi", o.pos_hi) && vecc("vel_lim", o.vel_lim) && vecc("effort_lim", o.effort_lim)))
    return false;
  auto vk = kv.ints("vel_known");
  if (vk.size() != N + 1) { err = "vel_known"; return false; }
  for (int i = 0; i < N; i++) o.vel_known[i] = vk[i + 1] != 0;
  o.history = (int)kv.num("history_steps", 10);
  auto f = kv.ints("future_steps");
  if (f.empty() || f[0] != (int)f.size() - 1) { err = "future_steps"; return false; }
  o.future.assign(f.begin() + 1, f.end());
  o.rate_hz = kv.num("rl_rate_hz", 50.0);
  return true;
}

bool Motion::load(const std::string& path, Motion& o, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { err = "cannot read " + path; return false; }
  char magic[4];
  int32_t T, J;
  float fps;
  f.read(magic, 4);
  f.read((char*)&T, 4);
  f.read((char*)&J, 4);
  f.read((char*)&fps, 4);
  if (!f || memcmp(magic, "PMMO", 4) != 0 || J != N || T <= 1) { err = "bad motion.bin header"; return false; }
  o.T = T; o.fps = fps;
  o.jp.resize((size_t)T * N); o.jv.resize((size_t)T * N); o.rp.resize((size_t)T * 3); o.rq.resize((size_t)T * 4);
  for (auto* v : {&o.jp, &o.jv, &o.rp, &o.rq}) f.read((char*)v->data(), v->size() * sizeof(float));
  if (!f) { err = "truncated motion.bin"; return false; }
  return true;
}

Motion Motion::with_lead_in(int n) const {
  Motion o;
  o.fps = fps;
  if (n <= 0 || T <= 0) { o = *this; return o; }
  o.T = T + n;
  auto pad = [&](const std::vector<float>& v, size_t w) {
    std::vector<float> r;
    r.reserve(v.size() + (size_t)n * w);
    for (int i = 0; i < n; i++) r.insert(r.end(), v.begin(), v.begin() + (long)w);
    r.insert(r.end(), v.begin(), v.end());
    return r;
  };
  o.jp = pad(jp, N); o.jv = pad(jv, N); o.rp = pad(rp, 3); o.rq = pad(rq, 4);
  return o;
}

RunTimeline plan_timeline(const Motion& motion, bool running, double run_s, double lead_in_s, double end_hold_s, double dt) {
  RunTimeline t;
  t.running = running; t.fps = motion.fps; t.dt = dt; t.run_s = run_s; t.end_hold_s = end_hold_s;
  t.dance_frames = motion.T;
  t.dance_s = motion.fps > 0 ? motion.T / (double)motion.fps : 0;
  t.lead_in_frames = running ? (int)std::lround(lead_in_s * motion.fps) : 0;
  t.lead_in_s = t.lead_in_frames / (double)motion.fps;
  t.clip_frames = motion.T + t.lead_in_frames;
  if (running) {
    t.seg_frames = std::min(t.clip_frames - 1, t.lead_in_frames + (int)std::lround(run_s * motion.fps));
    t.window_over_tick = std::max(0, t.seg_frames - 1);  // PolicyStack::tick advances motion_t first: tick n sees motion_t = min(n + 1, T - 1)
    t.planned_total_s = t.window_over_tick * dt + end_hold_s;
    t.covers_whole_dance = t.seg_frames == t.clip_frames - 1;
    t.last_frame_held = t.covers_whole_dance && end_hold_s > 0;
  } else {
    t.seg_frames = std::min(t.clip_frames - 1, (int)std::lround(run_s * motion.fps));
    t.window_over_tick = (int)std::ceil(run_s / dt - 1e-9);
    t.planned_total_s = run_s;  // paused: the hand-back starts as soon as the window is over (no extra hold)
  }
  t.failsafe_s = t.lead_in_s + run_s + end_hold_s + 1.0;
  t.margin_s = t.failsafe_s - t.planned_total_s;
  return t;
}

// ------------------------------------------------------------------ quaternion helpers (wxyz), identical formulas to sim2real/utils/math.py
namespace {
template <typename T> using Q = std::array<T, 4>;

template <typename T> Q<T> qmul(const Q<T>& a, const Q<T>& b) {
  T w1 = a[0], x1 = a[1], y1 = a[2], z1 = a[3], w2 = b[0], x2 = b[1], y2 = b[2], z2 = b[3];
  T ww = (z1 + x1) * (x2 + y2), yy = (w1 - y1) * (w2 + z2), zz = (w1 + y1) * (w2 - z2), xx = ww + yy + zz;
  T qq = T(0.5) * (xx + (z1 - x1) * (x2 - y2));
  T w = qq - ww + (z1 - y1) * (y2 - z2), x = qq - xx + (x1 + w1) * (x2 + w2), y = qq - yy + (w1 - x1) * (y2 + z2), z = qq - zz + (z1 + y1) * (w2 - x2);
  return {w, x, y, z};
}
template <typename T> Q<T> qconj(const Q<T>& q) { return {q[0], -q[1], -q[2], -q[3]}; }

template <typename T> std::array<T, 3> rotate(const Q<T>& q, const std::array<T, 3>& v) {  // quat_rotate_numpy
  T w = q[0], qx = q[1], qy = q[2], qz = q[3], k = 2 * w * w - 1;
  std::array<T, 3> a = {v[0] * k, v[1] * k, v[2] * k};
  std::array<T, 3> cr = {qy * v[2] - qz * v[1], qz * v[0] - qx * v[2], qx * v[1] - qy * v[0]};
  T dot = qx * v[0] + qy * v[1] + qz * v[2];
  return {a[0] + cr[0] * w * 2 + qx * dot * 2, a[1] + cr[1] * w * 2 + qy * dot * 2, a[2] + cr[2] * w * 2 + qz * dot * 2};
}
template <typename T> std::array<T, 3> rotate_inverse(const Q<T>& q, const std::array<T, 3>& v) {  // quat_rotate_inverse_numpy
  T w = q[0], qx = q[1], qy = q[2], qz = q[3], k = 2 * w * w - 1;
  std::array<T, 3> a = {v[0] * k, v[1] * k, v[2] * k};
  std::array<T, 3> cr = {qy * v[2] - qz * v[1], qz * v[0] - qx * v[2], qx * v[1] - qy * v[0]};
  T dot = qx * v[0] + qy * v[1] + qz * v[2];
  return {a[0] - cr[0] * w * 2 + qx * dot * 2, a[1] - cr[1] * w * 2 + qy * dot * 2, a[2] - cr[2] * w * 2 + qz * dot * 2};
}
template <typename T> Q<T> projected_yaw_quat(const Q<T>& q) {
  const T thr = T(0.1);
  auto xw = rotate<T>(q, {T(1), T(0), T(0)});
  auto zw = rotate<T>(q, {T(0), T(0), T(1)});
  T xn = std::sqrt(xw[0] * xw[0] + xw[1] * xw[1]);
  std::array<T, 2> zh = xw[2] < 0 ? std::array<T, 2>{zw[0], zw[1]} : std::array<T, 2>{-zw[0], -zw[1]};
  std::array<T, 2> h = xn > thr ? std::array<T, 2>{xw[0], xw[1]} : zh;
  T yaw = std::atan2(h[1], h[0]);
  return {std::cos(yaw / 2), T(0), T(0), std::sin(yaw / 2)};
}
}  // namespace


// ---- float16 emulation. The frozen Python stack keeps the SSB reference in float16 (any4hdmi storage dtype), so numpy evaluates
// projected_yaw_quat(ref quaternion) with every intermediate rounded to half precision. The reset-time yaw offset depends on it at the
// 3e-4 level (action sensitivity ~1e-2), so the same rounding is reproduced here (round-to-nearest-even after each numpy op).
static float rh(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
  uint32_t m = x & 0x7fffffu;
  if (((x >> 23) & 0xff) == 0xff) return f;  // inf/nan
  uint16_t h;
  if (e >= 31) h = (uint16_t)(sign | 0x7c00u);
  else if (e <= 0) {
    if (e < -10) h = (uint16_t)sign;
    else {
      m |= 0x800000u;
      int shift = 14 - e;
      uint32_t hm = m >> shift, rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
      if (rem > half || (rem == half && (hm & 1))) hm++;
      h = (uint16_t)(sign | hm);
    }
  } else {
    uint32_t hm = m >> 13, rem = m & 0x1fffu;
    uint32_t v = ((uint32_t)e << 10) | hm;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) v++;
    h = (uint16_t)(sign | v);
  }
  // half -> float
  uint32_t hs = (h & 0x8000u) << 16, he = (h >> 10) & 0x1f, hm2 = h & 0x3ffu, out;
  if (he == 0) {
    if (hm2 == 0) out = hs;
    else { int sh = 0; while (!(hm2 & 0x400u)) { hm2 <<= 1; sh++; } hm2 &= 0x3ffu; out = hs | ((uint32_t)(127 - 15 - sh + 1) << 23) | (hm2 << 13); }
  } else if (he == 31) out = hs | 0x7f800000u | (hm2 << 13);
  else out = hs | ((he - 15 + 127) << 23) | (hm2 << 13);
  float r;
  std::memcpy(&r, &out, 4);
  return r;
}

static std::array<float, 3> rotate_f16(const Q<float>& q, const std::array<float, 3>& v) {  // quat_rotate_numpy on float16 arrays
  float w = q[0], qx = q[1], qy = q[2], qz = q[3];
  float k = rh(rh(2.f * rh(w * w)) - 1.f);
  float cr[3] = {rh(rh(qy * v[2]) - rh(qz * v[1])), rh(rh(qz * v[0]) - rh(qx * v[2])), rh(rh(qx * v[1]) - rh(qy * v[0]))};
  float p0 = rh(qx * v[0]), p1 = rh(qy * v[1]), p2 = rh(qz * v[2]);
  float dot = rh(p0 + (p1 + p2));
  float qv[3] = {qx, qy, qz};
  std::array<float, 3> o;
  for (int i = 0; i < 3; i++) {
    float a = rh(v[i] * k), b = rh(rh(cr[i] * w) * 2.f), c = rh(rh(qv[i] * dot) * 2.f);
    o[i] = rh(rh(a + b) + c);
  }
  return o;
}
static Q<float> projected_yaw_quat_f16(const Q<float>& q) {
  auto xw = rotate_f16(q, {1.f, 0.f, 0.f});
  auto zw = rotate_f16(q, {0.f, 0.f, 1.f});
  float xn = rh(std::sqrt(rh(rh(xw[0] * xw[0]) + rh(xw[1] * xw[1]))));
  std::array<float, 2> zh = xw[2] < 0 ? std::array<float, 2>{zw[0], zw[1]} : std::array<float, 2>{-zw[0], -zw[1]};
  std::array<float, 2> h = xn > 0.1f ? std::array<float, 2>{xw[0], xw[1]} : zh;
  float yaw = rh(std::atan2(h[1], h[0]));
  float hy = rh(yaw / 2.f);
  return {rh(std::cos(hy)), 0.f, 0.f, rh(std::sin(hy))};
}

// ------------------------------------------------------------------ policy stack
struct PolicyStack::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "partymimic"};
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> sess;
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<std::string> in_names;
  std::string out_name;

  int motion_t = 0;
  bool paused = true;
  std::array<float, N> prev_action{};
  std::vector<float> hist_proprio;  // [history][93]
  std::vector<float> hist_quat;     // [history][8]
  std::array<double, 4> offset{1, 0, 0, 0};
  std::vector<float> ref_jp, ref_root;  // current slice (12 steps)
  PolicyResetInfo rinfo;
};

static constexpr int PROPRIO_DIM = 2 * N + 6 + N;  // 93

PolicyStack::PolicyStack(const Params& p, const Motion& m, const std::string& onnx_path, int threads) : impl_(new Impl), p_(p), m_(m) {
  impl_->opts.SetIntraOpNumThreads(threads);
  impl_->opts.SetInterOpNumThreads(1);
  impl_->opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  impl_->sess.reset(new Ort::Session(impl_->env, onnx_path.c_str(), impl_->opts));
  Ort::AllocatorWithDefaultOptions al;
  for (size_t i = 0; i < impl_->sess->GetInputCount(); i++) impl_->in_names.push_back(impl_->sess->GetInputNameAllocated(i, al).get());
  impl_->out_name = impl_->sess->GetOutputNameAllocated(0, al).get();
  const std::vector<std::string> want = {"proprio", "root_quat_hist", "ref_joint_pos", "ref_root"};
  if (impl_->in_names != want || impl_->out_name != "action") throw std::runtime_error("unexpected ONNX interface (not a PartyMimic tracking policy)");
  impl_->hist_proprio.assign((size_t)p_.history * PROPRIO_DIM, 0.f);
  impl_->hist_quat.assign((size_t)p_.history * 8, 0.f);
}
PolicyStack::~PolicyStack() = default;
const std::vector<std::string>& PolicyStack::onnx_inputs() const { return impl_->in_names; }

namespace {
int clampi(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }
}  // namespace

// proprio frame: [q - default (float32 default), dq, gyro, projected gravity, previous action]
static void proprio_frame(const Params& p, const RobotSample& s, const std::array<float, N>& prev, float* out) {
  int k = 0;
  for (int i = 0; i < N; i++) out[k++] = (float)((double)s.q[i] - (double)(float)p.default_pos[i]);
  for (int i = 0; i < N; i++) out[k++] = s.dq[i];
  for (int i = 0; i < 3; i++) out[k++] = s.gyro[i];
  Q<double> q = {s.quat[0], s.quat[1], s.quat[2], s.quat[3]};
  auto g = rotate_inverse<double>(q, {0.0, 0.0, -1.0});
  for (int i = 0; i < 3; i++) out[k++] = (float)g[i];
  for (int i = 0; i < N; i++) out[k++] = prev[i];
}

// robot quaternion in the motion frame + reference root quaternion at step -1 (index of step -1 in the slice)
static void quat_frame(const Motion& m, const std::array<double, 4>& off, const RobotSample& s, int ref_idx, float* out) {
  Q<double> r = {s.quat[0], s.quat[1], s.quat[2], s.quat[3]};
  auto a = qmul<double>(off, r);
  for (int i = 0; i < 4; i++) out[i] = (float)a[i];
  for (int i = 0; i < 4; i++) out[4 + i] = m.rq[(size_t)ref_idx * 4 + i];
}

static void fill_slice(const Params& p, const Motion& m, int motion_t, bool paused, std::vector<float>& jp, std::vector<float>& root) {
  size_t S = p.future.size();
  jp.resize(S * N);
  root.resize(S * 7);
  for (size_t k = 0; k < S; k++) {
    int idx = clampi(motion_t + (paused ? 0 : p.future[k]), 0, m.T - 1);
    std::memcpy(&jp[k * N], &m.jp[(size_t)idx * N], N * sizeof(float));
    std::memcpy(&root[k * 7], &m.rp[(size_t)idx * 3], 3 * sizeof(float));
    std::memcpy(&root[k * 7 + 3], &m.rq[(size_t)idx * 4], 4 * sizeof(float));
  }
}

static int index_of_step(const Params& p, int step) {
  for (size_t k = 0; k < p.future.size(); k++) if (p.future[k] == step) return (int)k;
  throw std::runtime_error("future step missing");
}

void PolicyStack::reset(const RobotSample& s, bool paused) {
  Impl& I = *impl_;
  I.motion_t = 0;
  I.prev_action.fill(0.f);
  fill_slice(p_, m_, 0, true, I.ref_jp, I.ref_root);  // _reset_motion(): paused slice at frame 0
  // proprio_history.reset
  std::vector<float> f(PROPRIO_DIM);
  proprio_frame(p_, s, I.prev_action, f.data());
  for (int h = 0; h < p_.history; h++) std::memcpy(&I.hist_proprio[(size_t)h * PROPRIO_DIM], f.data(), PROPRIO_DIM * sizeof(float));
  // root_quat_history.reset: yaw offset between robot and motion root, fixed at reset
  const int cur = index_of_step(p_, -1), zero = index_of_step(p_, 0);
  Q<float> ref0 = {m_.rq[(size_t)clampi(0, 0, m_.T - 1) * 4 + 0], m_.rq[1], m_.rq[2], m_.rq[3]};
  (void)zero;
  Q<float> pr = projected_yaw_quat_f16(ref0);  // reference quaternion is float16 in the Python stack
  Q<double> prd = {pr[0], pr[1], pr[2], pr[3]};
  Q<double> rq = {s.quat[0], s.quat[1], s.quat[2], s.quat[3]};
  Q<double> pj = projected_yaw_quat<double>(rq);
  I.offset = qmul<double>(prd, qconj<double>(pj));
  (void)cur;
  int ref_idx = clampi(0, 0, m_.T - 1);  // paused slice: step -1 maps to frame 0 too
  float qf[8];
  quat_frame(m_, I.offset, s, ref_idx, qf);
  for (int h = 0; h < p_.history; h++) std::memcpy(&I.hist_quat[(size_t)h * 8], qf, 8 * sizeof(float));
  I.paused = paused;
  I.rinfo.yaw_offset = {I.offset[0], I.offset[1], I.offset[2], I.offset[3]};
  I.rinfo.motion_t = 0;
  I.rinfo.paused = paused;
  I.rinfo.sample = s;
}

int PolicyStack::obs_dim() const { return (int)(impl_->hist_proprio.size() + impl_->hist_quat.size() + impl_->ref_jp.size() + impl_->ref_root.size()); }
void PolicyStack::copy_obs(float* dst) const {
  const Impl& I = *impl_;
  for (const std::vector<float>* v : {&I.hist_proprio, &I.hist_quat, &I.ref_jp, &I.ref_root}) {
    std::memcpy(dst, v->data(), v->size() * sizeof(float));
    dst += v->size();
  }
}
std::vector<std::pair<std::string, int>> PolicyStack::obs_layout() const {
  const Impl& I = *impl_;
  return {{"proprio", (int)I.hist_proprio.size()}, {"root_quat_hist", (int)I.hist_quat.size()}, {"ref_joint_pos", (int)I.ref_jp.size()}, {"ref_root", (int)I.ref_root.size()}};
}
PolicyResetInfo PolicyStack::reset_info() const { return impl_->rinfo; }
PolicyStack::StateDigest PolicyStack::state_digest() const {
  const Impl& I = *impl_;
  StateDigest d;
  d.motion_t = I.motion_t; d.paused = I.paused;
  for (float v : I.prev_action) d.prev_action_abs_max = std::max(d.prev_action_abs_max, (double)std::fabs(v));
  const size_t P = PROPRIO_DIM;
  d.history_identical = true;
  for (int h = 1; h < p_.history; h++) d.history_identical = d.history_identical && std::memcmp(&I.hist_proprio[0], &I.hist_proprio[(size_t)h * P], P * sizeof(float)) == 0;
  d.quat_history_identical = true;
  for (int h = 1; h < p_.history; h++) d.quat_history_identical = d.quat_history_identical && std::memcmp(&I.hist_quat[0], &I.hist_quat[(size_t)h * 8], 8 * sizeof(float)) == 0;
  d.ref_slice_first = 0;
  return d;
}

PolicyOut PolicyStack::tick(const RobotSample& s) {
  Impl& I = *impl_;
  // ---- Tracking.update()
  bool paused = I.paused;
  if (!paused) {
    I.motion_t += 1;
    if (m_.T > 0 && I.motion_t >= m_.T) { I.motion_t = m_.T - 1; I.paused = true; }
  }
  fill_slice(p_, m_, I.motion_t, paused, I.ref_jp, I.ref_root);
  // proprio_history.update (previous action = last policy output)
  std::memmove(I.hist_proprio.data(), I.hist_proprio.data() + PROPRIO_DIM, (size_t)(p_.history - 1) * PROPRIO_DIM * sizeof(float));
  proprio_frame(p_, s, I.prev_action, &I.hist_proprio[(size_t)(p_.history - 1) * PROPRIO_DIM]);
  // root_quat_history.update: reference root quaternion at step -1 of the (possibly paused) slice
  std::memmove(I.hist_quat.data(), I.hist_quat.data() + 8, (size_t)(p_.history - 1) * 8 * sizeof(float));
  int ref_idx = clampi(I.motion_t + (paused ? 0 : -1), 0, m_.T - 1);
  quat_frame(m_, I.offset, s, ref_idx, &I.hist_quat[(size_t)(p_.history - 1) * 8]);

  PolicyOut r;
  r.action_prev = I.prev_action;
  {  // reference at step 0 of this tick (the paused slice and the running slice both have step 0 at motion_t)
    int idx = clampi(I.motion_t, 0, m_.T - 1);
    r.ref_idx = idx;
    std::memcpy(r.ref_jp.data(), &m_.jp[(size_t)idx * N], N * sizeof(float));
    std::memcpy(r.ref_jv.data(), &m_.jv[(size_t)idx * N], N * sizeof(float));
    std::memcpy(r.ref_rp.data(), &m_.rp[(size_t)idx * 3], 3 * sizeof(float));
    std::memcpy(r.ref_rq.data(), &m_.rq[(size_t)idx * 4], 4 * sizeof(float));
  }

  // ---- inference
  std::array<int64_t, 1> sh0 = {(int64_t)I.hist_proprio.size()}, sh1 = {(int64_t)I.hist_quat.size()}, sh2 = {(int64_t)I.ref_jp.size()}, sh3 = {(int64_t)I.ref_root.size()};
  std::vector<Ort::Value> in;
  in.push_back(Ort::Value::CreateTensor<float>(I.mem, I.hist_proprio.data(), I.hist_proprio.size(), sh0.data(), 1));
  in.push_back(Ort::Value::CreateTensor<float>(I.mem, I.hist_quat.data(), I.hist_quat.size(), sh1.data(), 1));
  in.push_back(Ort::Value::CreateTensor<float>(I.mem, I.ref_jp.data(), I.ref_jp.size(), sh2.data(), 1));
  in.push_back(Ort::Value::CreateTensor<float>(I.mem, I.ref_root.data(), I.ref_root.size(), sh3.data(), 1));
  const char* names[4] = {"proprio", "root_quat_hist", "ref_joint_pos", "ref_root"};
  const char* outn[1] = {"action"};
  double t0 = mono();
  auto out = I.sess->Run(Ort::RunOptions{nullptr}, names, in.data(), 4, outn, 1);
  r.infer_ms = 1e3 * (mono() - t0);
  const float* a = out[0].GetTensorData<float>();
  size_t na = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
  if (na != N) throw std::runtime_error("policy output size != 29");
  for (int i = 0; i < N; i++) {
    r.action[i] = a[i];
    r.q_target[i] = p_.default_pos[i] + (double)a[i] * p_.action_scale[i];  // q_target = default + action * scale (float64, like BasePolicy)
  }
  I.prev_action = r.action;
  r.motion_t = I.motion_t;
  return r;
}

void PolicyStack::hold_reference() { impl_->paused = true; }

std::vector<std::vector<float>> PolicyStack::last_inputs() const {
  return {impl_->hist_proprio, impl_->hist_quat, impl_->ref_jp, impl_->ref_root};
}

double PolicyStack::warmup(const RobotSample& s, int n) {
  reset(s, false);
  double mx = 0;
  for (int i = 0; i < n; i++) mx = std::max(mx, tick(s).infer_ms);
  return mx;
}

}  // namespace pm
