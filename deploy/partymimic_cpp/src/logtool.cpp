// Offline log tool (P5.1): `partymimic logtool finalize|check|replay <run_dir>`. Never opens DDS, never touches the robot.
//   finalize : adds timing / safety-clip / logging / anomaly sections to run_report.json (creating a salvage report if the runner died) and writes anomaly_summary.{json,txt}
//              (the runner calls the same function at the end of every run, the service calls it for runs whose runner crashed)
//   check    : integrity of a run directory (files, row counts, columns, time order, hashes vs the files on this machine)
//   replay   : feeds obs.bin through the ONNX model and compares with the logged actions (bit-exact policy replay)
#include <dirent.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

#include "pm/app.hpp"
#include "pm/exit_policy.hpp"
#include "pm/frozen.hpp"
#include "pm/logtool.hpp"
#include "pm/runlog.hpp"
#include "pm/version.hpp"

namespace pm {
namespace {

// ---------------------------------------------------------------- tiny JSON helpers (flat documents written by this program)
size_t find_key(const std::string& d, const std::string& key, size_t from = 0) {
  size_t k = d.find("\"" + key + "\":", from);
  return k == std::string::npos ? k : k + key.size() + 3;
}
bool jstring(const std::string& d, const std::string& key, std::string& out, size_t from = 0) {
  size_t k = find_key(d, key, from);
  if (k == std::string::npos) return false;
  while (k < d.size() && d[k] == ' ') k++;
  if (k >= d.size() || d[k] != '"') return false;
  std::string r;
  for (size_t i = k + 1; i < d.size(); i++) {
    if (d[i] == '\\' && i + 1 < d.size()) { r += d[++i]; continue; }
    if (d[i] == '"') { out = r; return true; }
    r += d[i];
  }
  return false;
}
bool jnumber(const std::string& d, const std::string& key, double& out, size_t from = 0) {
  size_t k = find_key(d, key, from);
  if (k == std::string::npos) return false;
  while (k < d.size() && d[k] == ' ') k++;
  char* e = nullptr;
  double v = strtod(d.c_str() + k, &e);
  if (e == d.c_str() + k) return false;
  out = v;
  return true;
}
bool jbool_(const std::string& d, const std::string& key, bool& out, size_t from = 0) {
  size_t k = find_key(d, key, from);
  if (k == std::string::npos) return false;
  if (d.compare(k, 4, "true") == 0) { out = true; return true; }
  if (d.compare(k, 5, "false") == 0) { out = false; return true; }
  return false;
}
std::vector<double> jarray(const std::string& d, const std::string& key) {
  std::vector<double> v;
  size_t k = find_key(d, key);
  if (k == std::string::npos || k >= d.size() || d[k] != '[') return v;
  const char* p = d.c_str() + k + 1;
  while (*p && *p != ']') {
    char* e = nullptr;
    double x = strtod(p, &e);
    if (e == p) { p++; continue; }
    v.push_back(x);
    p = e;
    while (*p == ',' || *p == ' ') p++;
  }
  return v;
}
std::string q(const std::string& s) { return "\"" + json_escape(s) + "\""; }
std::string f2(double v) { char b[32]; snprintf(b, sizeof b, "%.2f", v); return b; }
std::string num(double v) {
  if (!std::isfinite(v)) return "null";
  char b[48];
  snprintf(b, sizeof b, "%.9g", v);
  return b;
}
std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> o;
  size_t p = 0;
  for (;;) {
    size_t e = s.find(c, p);
    if (e == std::string::npos) { o.push_back(s.substr(p)); break; }
    o.push_back(s.substr(p, e - p));
    p = e + 1;
  }
  return o;
}
bool is_dir(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode); }
uint64_t file_size(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0 ? (uint64_t)st.st_size : 0; }

struct Csv {
  std::vector<std::string> hdr;
  std::map<std::string, int> idx;
  std::vector<std::vector<double>> rows;
  size_t bad_rows = 0;
  std::vector<std::string> phase;
  bool load(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    if (!std::getline(f, line)) return false;
    hdr = split(line, ',');
    for (size_t i = 0; i < hdr.size(); i++) idx[hdr[i]] = (int)i;
    int pc = idx.count("phase") ? idx["phase"] : -1;
    while (std::getline(f, line)) {
      if (line.empty()) continue;
      auto fl = split(line, ',');
      if (fl.size() != hdr.size()) { bad_rows++; continue; }
      std::vector<double> r(fl.size());
      for (size_t i = 0; i < fl.size(); i++) r[i] = (int)i == pc ? 0.0 : strtod(fl[i].c_str(), nullptr);
      if (pc >= 0) phase.push_back(fl[pc]);
      rows.push_back(std::move(r));
    }
    return true;
  }
  int col(const std::string& n) const { auto it = idx.find(n); return it == idx.end() ? -1 : it->second; }
};

struct Limits { std::vector<double> vel_lim, vel_known, effort_lim, kp, kd; std::vector<std::string> names; bool ok = false; };
Limits load_limits(const std::string& manifest) {
  Limits L;
  L.vel_lim = jarray(manifest, "vel_lim"); L.vel_known = jarray(manifest, "vel_known"); L.effort_lim = jarray(manifest, "effort_lim");
  L.kp = jarray(manifest, "kp"); L.kd = jarray(manifest, "kd");
  L.ok = L.vel_lim.size() == (size_t)N && L.effort_lim.size() == (size_t)N;
  return L;
}

}  // namespace

// ---------------------------------------------------------------- finalize
std::vector<std::string> read_event_lines(const std::string& dir) {
  std::vector<std::string> v;
  std::ifstream f(dir + "/events.jsonl");
  std::string l;
  while (std::getline(f, l)) if (l.size() > 2 && l.front() == '{' && l.back() == '}') v.push_back(l);
  return v;
}

struct Item { std::string level, code, msg; };


// ---------------------------------------------------------------- P5.3 statistics
namespace {
struct Rms { double s = 0, mx = 0; size_t n = 0; void add(double e) { s += e * e; mx = std::max(mx, std::fabs(e)); n++; } double rms() const { return n ? std::sqrt(s / (double)n) : 0; } };
}  // namespace

// Keeps the four quantities apart: RAW policy target (q_policy), the CLIPPED command that was sent (q_cmd), the ACTUAL joint position (q) and the REFERENCE (ref_jp).
// Also: how many commands were cut because the policy asked beyond the hard joint range, and the torque the TRAINING actuator would have applied vs what the command asks the drive for.
static std::string tracking_json(const Csv& tr, const std::vector<std::string>& jn, const Limits& L, const std::vector<double>& lo, const std::vector<double>& hi, const std::vector<double>& eff, int& pct_out_ticks_x10, double& raw_outside_frac) {
  int c_q = -1, c_dq = -1, c_qc = -1, c_ref = -1, c_qp = -1, c_kp = -1, c_kd = -1, c_tpd = -1, c_tff = -1, c_cn = tr.col("clip_n");
  for (size_t i = 0; i < tr.hdr.size(); i++) {
    const std::string& h = tr.hdr[i];
    if (c_q < 0 && h.rfind("q:", 0) == 0) c_q = (int)i;
    if (c_dq < 0 && h.rfind("dq:", 0) == 0) c_dq = (int)i;
    if (c_qc < 0 && h.rfind("q_cmd:", 0) == 0) c_qc = (int)i;
    if (c_ref < 0 && h.rfind("ref_jp:", 0) == 0) c_ref = (int)i;
    if (c_qp < 0 && h.rfind("q_policy:", 0) == 0) c_qp = (int)i;
    if (c_kp < 0 && h.rfind("kp:", 0) == 0) c_kp = (int)i;
    if (c_kd < 0 && h.rfind("kd:", 0) == 0) c_kd = (int)i;
    if (c_tpd < 0 && h.rfind("tau_pd:", 0) == 0) c_tpd = (int)i;
    if (c_tff < 0 && h.rfind("tau_ff:", 0) == 0) c_tff = (int)i;
  }
  pct_out_ticks_x10 = 0; raw_outside_frac = 0;
  if (c_q < 0 || c_qc < 0 || c_ref < 0 || c_qp < 0 || tr.rows.empty() || lo.size() != (size_t)N || hi.size() != (size_t)N) return "{\"available\":false}";
  Rms pol_ref[N], cmd_ref[N], act_ref[N], act_cmd[N], pol_cmd[N];
  size_t out_ticks = 0, any_clip = 0, n_cmd = 0;
  double out_cnt[N] = {0}, out_exc[N] = {0}, def_sum[N] = {0}, def_max[N] = {0}, def_n[N] = {0}, def_big[N] = {0}, tau_tr_sum[N] = {0}, tau_cmd_sum[N] = {0};
  Rms all_pol_ref, all_cmd_ref, all_act_ref, all_act_cmd, all_pol_cmd;
  for (const auto& r : tr.rows) {
    bool any_out = false;
    const bool has_cmd = std::isfinite(r[c_qc]);
    for (int j = 0; j < N; j++) {
      const double q = r[c_q + j], ref = r[c_ref + j], qp = r[c_qp + j], qc = r[c_qc + j];
      pol_ref[j].add(qp - ref); act_ref[j].add(q - ref); all_pol_ref.add(qp - ref); all_act_ref.add(q - ref);
      const bool out = qp < lo[j] || qp > hi[j];
      if (out) { out_cnt[j]++; out_exc[j] += std::max(lo[j] - qp, qp - hi[j]); any_out = true; }
      if (has_cmd) {
        cmd_ref[j].add(qc - ref); act_cmd[j].add(q - qc); pol_cmd[j].add(qp - qc); all_cmd_ref.add(qc - ref); all_act_cmd.add(q - qc); all_pol_cmd.add(qp - qc);
        if (c_kp >= 0 && c_kd >= 0 && c_dq >= 0 && eff.size() == (size_t)N) {  // training-equivalent torque (target unclipped, effort-saturated) vs the torque the sent command asks for (+ feed-forward)
          const double kp = r[c_kp + j], kd = r[c_kd + j], dq = r[c_dq + j];
          const double tt = std::max(-eff[j], std::min(eff[j], kp * (qp - q) - kd * dq));
          const double td = kp * (qc - q) - kd * dq + (c_tff >= 0 ? r[c_tff + j] : 0.0);
          const double def = std::fabs(tt) - std::fabs(td) * ((tt >= 0) == (td >= 0) ? 1.0 : -1.0);  // shortfall in the direction the training torque points
          const double sd = (tt >= 0 ? 1.0 : -1.0) * (tt - td);
          def_sum[j] += sd; def_max[j] = std::max(def_max[j], sd); def_n[j]++; def_big[j] += sd > 5.0; tau_tr_sum[j] += std::fabs(tt); tau_cmd_sum[j] += std::fabs(td);
          (void)def;
        }
      }
    }
    if (any_out) out_ticks++;
    if (has_cmd) { n_cmd++; if (c_cn >= 0 && r[c_cn] > 0) any_clip++; }
  }
  const size_t nt = tr.rows.size();
  raw_outside_frac = (double)out_ticks / (double)nt;
  pct_out_ticks_x10 = (int)std::lround(1000.0 * raw_outside_frac);
  auto jv = [&](int j) { return j < (int)jn.size() ? jn[j] : std::to_string(j); };
  std::vector<int> order(N);
  for (int j = 0; j < N; j++) order[j] = j;
  std::sort(order.begin(), order.end(), [&](int a, int b) { return act_ref[a].rms() > act_ref[b].rms(); });
  std::string s = "{\"available\":true,\"ticks\":" + std::to_string(nt) + ",\"definition\":\"policy = q_policy (raw policy target, default + action*scale); command = q_cmd (after the safety guard, as sent); actual = q (measured); reference = ref_jp (dance reference at the current tick; lead-in / end-hold ticks included in these overall numbers: see tracking_segments)\"";
  auto blk = [&](const char* k, const Rms& a) { return std::string(",\"") + k + "\":{\"rms_rad\":" + num(a.rms()) + ",\"max_rad\":" + num(a.mx) + "}"; };
  s += blk("policy_vs_reference", all_pol_ref) + blk("command_vs_reference", all_cmd_ref) + blk("actual_vs_reference", all_act_ref) + blk("actual_vs_command", all_act_cmd) + blk("guard_removed_policy_minus_command", all_pol_cmd);
  s += ",\"clip\":{\"ticks_with_any_clip\":" + std::to_string(any_clip) + ",\"ticks_with_command\":" + std::to_string(n_cmd) + ",\"clip_fraction\":" + num(n_cmd ? (double)any_clip / (double)n_cmd : 0) + ",\"ticks_raw_target_outside_hard_limits\":" + std::to_string(out_ticks) +
       ",\"raw_target_outside_fraction\":" + num(raw_outside_frac) + ",\"note\":\"the raw policy target (before the guard) is beyond the hard joint range in this share of ticks: that is a property of the policy output, the guard only cuts it to the range\"}";
  s += ",\"per_joint\":[";
  for (int i = 0; i < 12; i++) {
    int j = order[i];
    s += std::string(i ? "," : "") + "{\"joint\":" + q(jv(j)) + ",\"actual_vs_ref_rms\":" + num(act_ref[j].rms()) + ",\"policy_vs_ref_rms\":" + num(pol_ref[j].rms()) + ",\"command_vs_ref_rms\":" + num(cmd_ref[j].rms()) + ",\"actual_vs_command_rms\":" + num(act_cmd[j].rms()) +
         ",\"raw_outside_limit_fraction\":" + num(out_cnt[j] / (double)nt) + ",\"mean_excess_rad\":" + num(out_cnt[j] ? out_exc[j] / out_cnt[j] : 0) + "}";
  }
  s += "],\"torque_vs_training\":[";
  bool first = true;
  std::vector<int> o2(N);
  for (int j = 0; j < N; j++) o2[j] = j;
  std::sort(o2.begin(), o2.end(), [&](int a, int b) { return (def_n[a] ? def_sum[a] / def_n[a] : 0) > (def_n[b] ? def_sum[b] / def_n[b] : 0); });
  for (int i = 0; i < 8; i++) {
    int j = o2[i];
    if (!def_n[j]) continue;
    s += std::string(first ? "" : ",") + "{\"joint\":" + q(jv(j)) + ",\"training_torque_abs_mean_nm\":" + num(tau_tr_sum[j] / def_n[j]) + ",\"command_torque_abs_mean_nm\":" + num(tau_cmd_sum[j] / def_n[j]) + ",\"shortfall_mean_nm\":" + num(def_sum[j] / def_n[j]) +
         ",\"shortfall_max_nm\":" + num(def_max[j]) + ",\"fraction_ticks_shortfall_gt_5nm\":" + num(def_big[j] / def_n[j]) + "}";
    first = false;
  }
  s += "]}";
  return s;
}

// maximum attitude over EVERYTHING recorded between the take-over and the end of the exit: control ticks, the full-rate lowstate stream and the sample that triggered the exit
// V1.6: reference-aware attitude statistics. A dance whose reference leans 0.5 rad must not be judged against a fixed 0.40: the measured attitude is compared with the REFERENCE pelvis attitude of the control tick
// that was current at the sample time (gravity-direction error = angle between the measured and the reference "up" vector in the pelvis frame: yaw-free, exactly the quantity of the run-time judge, EXIT_POLICY.md).
// Limits come from the run itself (events.jsonl tilt_judge_config.judge.config / run_report tilt_judge.config, written by the V1.6 runner). Key names/defaults are the ones of ExitConfig (include/pm/exit_policy.hpp, config/site.conf, main.cpp) -
// reconciled with A1 in the V1.6 integration: ref_aware (site key ref_aware_unstable) 1, unstable_hard_tilt_rad 0.70, ref_dev_rad 0.35, ref_dev_min_tilt_rad 0.25, ref_dev_growth_rad_s 0.0 (-2 = off), ref_max_age_s 0.15, unstable_tilt_rad 0.40 (fixed
// limit); they are used only for logs that carry no tilt_judge_config (V1.5.x). Structure read: events.jsonl {"kind":"tilt_judge_config","judge":{"ref_aware":b,"config":{...}}} or run_report.json "tilt_judge":{"ref_aware":b,"config":{...},"stats":{...}}.
// The REF_DEV test mirrors StabilityMonitor::update (growth = d(gravity error)/dt over a <=40 ms window, needs >=15 ms history, history reset when the reference is not valid); the run-time debounce is NOT applied (offline = conservative).
struct RefTilt {
  std::vector<double> t;                    // tick t_mono
  std::vector<std::array<double, 3>> up;    // reference "up" vector in the pelvis frame
  std::vector<double> ref_tilt;
  std::vector<char> valid;                  // reference counted as valid for the judge: running dance segment (not lead-in / end-hold / paused)
  double hard = 0.70, dev = 0.35, dev_min = 0.25, growth = 0.0, fixed = 0.40, max_age = 0.15;
  bool ref_aware = true, from_log = false;
};
static std::array<double, 3> up_of(const double w, const double x, const double y, const double z) {
  const double n = std::sqrt(w * w + x * x + y * y + z * z);
  if (!(n > 1e-9)) return {0, 0, 1};
  const double a = w / n, b = x / n, c = y / n, d = z / n;
  return {2 * (b * d - a * c), 2 * (c * d + a * b), 1 - 2 * (b * b + c * c)};
}
static double ang3(const std::array<double, 3>& a, const std::array<double, 3>& b) {
  const double d = std::max(-1.0, std::min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]));
  return std::acos(d);
}
struct AttExtremes {
  double tilt_trace = 0, tilt_stream = 0, tilt_trigger = 0, gyro_trace = 0, gyro_stream = 0, tilt_overall = 0, gyro_overall = 0, t_peak = 0; size_t stream_samples = 0;
  // reference-aware part (stream samples + the trace)
  double ref_tilt_at_peak = -1, dev_at_peak = -1;                        // reference tilt / gravity error at the sample of the maximum measured tilt (-1 = no valid reference there)
  double max_dev = -1, tilt_at_max_dev = 0, ref_tilt_at_max_dev = 0, t_max_dev = 0, max_tilt_minus_ref = -9;   // over reference-valid samples
  size_t n_ref_valid = 0, n_ref_invalid = 0, viol_hard = 0, viol_dev = 0, viol_fixed = 0;
  double max_tilt_no_ref = 0, max_tilt_ref = 0;
};
static AttExtremes attitude_extremes(const std::string& dir, double t_take, double t_end, double tilt_trace, double gyro_trace, double tilt_trigger, const RefTilt* rt = nullptr) {
  AttExtremes a;
  a.tilt_trace = tilt_trace; a.gyro_trace = gyro_trace; a.tilt_trigger = tilt_trigger;
  std::deque<std::pair<double, double>> dhist;   // (t, gravity error) of consecutive reference-valid samples: growth criterion of the run-time judge
  std::ifstream g(dir + "/state_stream.bin", std::ios::binary);
  if (g && t_take > 0) {
    char mg[8] = {0};
    uint32_t jl = 0;
    g.read(mg, 8); g.read((char*)&jl, 4);
    g.seekg(jl, std::ios::cur);
    std::vector<char> rec(STATE_REC_BYTES);
    while (g.read(rec.data(), (std::streamsize)STATE_REC_BYTES)) {
      double t;
      memcpy(&t, rec.data(), 8);
      if (t < t_take || (t_end > 0 && t > t_end)) continue;
      float quat[4], gyro[3];
      memcpy(quat, rec.data() + 8 + 8 + 4 + 4 + 4 * N * 3, 16);
      memcpy(gyro, rec.data() + 8 + 8 + 4 + 4 + 4 * N * 3 + 16, 12);
      std::array<double, 4> qd = {quat[0], quat[1], quat[2], quat[3]};
      double tilt = tilt_of_quat(qd), gxy = std::sqrt((double)gyro[0] * gyro[0] + (double)gyro[1] * gyro[1] + (double)gyro[2] * gyro[2]);
      bool have_ref = false;
      double dev = -1, rtilt = -1;
      if (rt && !rt->t.empty() && std::isfinite(tilt)) {
        const size_t k = (size_t)(std::upper_bound(rt->t.begin(), rt->t.end(), t) - rt->t.begin());
        if (k > 0 && t - rt->t[k - 1] <= rt->max_age + 0.05 && rt->valid[k - 1] && rt->ref_aware) {
          have_ref = true; rtilt = rt->ref_tilt[k - 1];
          dev = ang3(up_of(quat[0], quat[1], quat[2], quat[3]), rt->up[k - 1]);
        }
        if (have_ref) {
          a.n_ref_valid++;
          if (dev > a.max_dev) { a.max_dev = dev; a.tilt_at_max_dev = tilt; a.ref_tilt_at_max_dev = rtilt; a.t_max_dev = t; }
          a.max_tilt_minus_ref = std::max(a.max_tilt_minus_ref, tilt - rtilt);
          a.max_tilt_ref = std::max(a.max_tilt_ref, tilt);
          dhist.push_back({t, dev});
          while (dhist.size() > 2 && t - dhist[1].first >= 0.040) dhist.pop_front();
          bool rate_ok = false; double rate = 0;
          if (dhist.size() >= 2 && t - dhist.front().first >= 0.015) { rate = (dev - dhist.front().second) / (t - dhist.front().first); rate_ok = true; }
          if (dev > rt->dev && tilt > rt->dev_min && (rt->growth <= -2.0 || (rate_ok && rate >= rt->growth))) a.viol_dev++;
        } else {
          dhist.clear();
          a.n_ref_invalid++;
          a.max_tilt_no_ref = std::max(a.max_tilt_no_ref, tilt);
          if (tilt > rt->fixed) a.viol_fixed++;
        }
        if (tilt > rt->hard) a.viol_hard++;
      }
      if (std::isfinite(tilt) && tilt > a.tilt_stream) { a.tilt_stream = tilt; a.t_peak = t; a.ref_tilt_at_peak = have_ref ? rtilt : -1; a.dev_at_peak = have_ref ? dev : -1; }
      a.gyro_stream = std::max(a.gyro_stream, gxy);
      a.stream_samples++;
    }
  }
  a.tilt_overall = std::max(std::max(a.tilt_trace, a.tilt_stream), a.tilt_trigger);
  a.gyro_overall = std::max(a.gyro_trace, a.gyro_stream);
  return a;
}

FinalizeResult finalize_run(const std::string& dir, const std::string& base_report_in, const std::string& note) {
  FinalizeResult R;
  std::string manifest, report = base_report_in;
  read_file(dir + "/manifest.json", manifest);
  if (report.empty()) read_file(dir + "/run_report.json", report);
  const bool had_report = !report.empty();
  size_t cut = std::min(report.find(",\"attitude_extremes\":"), report.find(",\"timing\":"));
  if (cut != std::string::npos) report = report.substr(0, cut) + "}";
  Limits L = load_limits(manifest);

  // ---- trace
  Csv tr;
  tr.load(dir + "/trace.csv");
  RunStats rs;
  ClipTracker ct(0.02);
  auto col = [&](const std::string& n) { return tr.col(n); };
  int c_t = col("t"), c_dt = col("dt_ms"), c_inf = col("infer_ms"), c_step = col("step_ms"), c_age = col("state_age_ms"), c_send = col("send_us"), c_dds = col("dds_loop_ms"), c_tilt = col("tilt"),
      c_flags = col("flags"), c_sd = col("seq_delta"), c_td = col("tick_delta"), c_lost = col("ls_lost"), c_clip = col("clip_n"), c_gx = col("gyro_x"), c_n = col("n");
  int c_q = -1, c_dq = -1, c_tau = -1, c_qc = -1, c_ref = -1;
  {
    // joint columns: the 6 groups appear in a fixed order; locate by prefix
    for (size_t i = 0; i < tr.hdr.size(); i++) {
      const std::string& h = tr.hdr[i];
      if (c_q < 0 && h.rfind("q:", 0) == 0) c_q = (int)i;
      if (c_dq < 0 && h.rfind("dq:", 0) == 0) c_dq = (int)i;
      if (c_tau < 0 && h.rfind("tau_est:", 0) == 0) c_tau = (int)i;
      if (c_qc < 0 && h.rfind("q_cmd:", 0) == 0) c_qc = (int)i;
      if (c_ref < 0 && h.rfind("ref_jp:", 0) == 0) c_ref = (int)i;
    }
  }
  std::vector<std::string> jnames;
  if (c_q >= 0) for (int j = 0; j < N && c_q + j < (int)tr.hdr.size(); j++) jnames.push_back(tr.hdr[c_q + j].substr(2));
  auto get = [&](const std::vector<double>& r, int c) { return c >= 0 ? r[c] : 0.0; };
  double first_t_mono = 0, last_t_mono = 0;
  int c_tm = col("t_mono");
  for (size_t i = 0; i < tr.rows.size(); i++) {
    const auto& r = tr.rows[i];
    TickLite t;
    t.t_rel = get(r, c_t); t.dt_ms = get(r, c_dt); t.infer_ms = get(r, c_inf); t.step_ms = get(r, c_step); t.state_age_ms = get(r, c_age); t.send_us = get(r, c_send); t.dds_loop_ms = get(r, c_dds);
    t.tilt = get(r, c_tilt); t.flags = (uint32_t)get(r, c_flags); t.seq_delta = (uint32_t)get(r, c_sd); t.tick_delta = (uint32_t)get(r, c_td); t.ls_lost = (uint64_t)get(r, c_lost); t.clip_n = (int)get(r, c_clip);
    if (c_gx >= 0) t.gyro = std::sqrt(r[c_gx] * r[c_gx] + r[c_gx + 1] * r[c_gx + 1] + r[c_gx + 2] * r[c_gx + 2]);
    if (c_q >= 0 && c_qc >= 0) for (int j = 0; j < N; j++) t.track_err_max = std::max(t.track_err_max, std::fabs(r[c_q + j] - r[c_qc + j]));
    if (c_q >= 0 && c_ref >= 0) for (int j = 0; j < N; j++) t.ref_err_max = std::max(t.ref_err_max, std::fabs(r[c_q + j] - r[c_ref + j]));
    if (L.ok && c_dq >= 0) for (int j = 0; j < N; j++) t.max_dq_ratio = std::max(t.max_dq_ratio, std::fabs(r[c_dq + j]) / (L.vel_known.size() > (size_t)j && L.vel_known[j] != 0 ? L.vel_lim[j] : 20.0));
    if (L.ok && c_tau >= 0) for (int j = 0; j < N; j++) t.max_tau_ratio = std::max(t.max_tau_ratio, std::fabs(r[c_tau + j]) / L.effort_lim[j]);
    if (c_tm >= 0) { if (i == 0) first_t_mono = r[c_tm]; last_t_mono = r[c_tm]; }
    rs.add(t);
  }
  // ---- safety clips (csv rows -> episodes)
  Csv cc;
  size_t clip_rows = 0;
  if (cc.load(dir + "/safety_clips.csv")) {
    clip_rows = cc.rows.size();
    int ctick = cc.col("tick"), ct_rel = cc.col("t_rel"), cj = cc.col("joint_idx"), craw = cc.col("raw"), cap = cc.col("after_pos"), cas = cc.col("after_slew"), cat = cc.col("after_torque"), cf = cc.col("final");
    int creason = cc.col("reason");
    (void)creason;
    // reasons are strings: re-read the column as text
    std::vector<std::string> reasons;
    {
      std::ifstream f(dir + "/safety_clips.csv");
      std::string line;
      std::getline(f, line);
      while (std::getline(f, line)) { auto fl = split(line, ','); if (fl.size() == cc.hdr.size()) reasons.push_back(fl[creason]); }
    }
    std::map<int, std::vector<ClipRow>> by_tick;
    std::map<int, double> t_of;
    for (size_t i = 0; i < cc.rows.size() && i < reasons.size(); i++) {
      const auto& r = cc.rows[i];
      int tk = (int)r[ctick];
      if (tk < 0) continue;  // exit-frame clips are listed in safety_clips.csv only
      ClipRow c;
      c.tick = tk; c.t_rel = r[ct_rel]; c.joint = (int)r[cj]; c.raw = r[craw]; c.after_pos = r[cap]; c.after_slew = r[cas]; c.after_torque = r[cat]; c.final_q = r[cf];
      for (auto& part : split(reasons[i], '|')) {
        if (part == "POS_LIMIT") c.mask |= CLIP_POS; else if (part == "SLEW") c.mask |= CLIP_SLEW; else if (part == "TORQUE") c.mask |= CLIP_TORQUE;
        else if (part == "POS_LIMIT_AFTER_TORQUE") c.mask |= CLIP_POS2; else if (part == "KP_CAP") c.mask |= CLIP_KP;
      }
      by_tick[tk].push_back(c);
      t_of[tk] = c.t_rel;
    }
    int last = -2;
    for (size_t i = 0; i < tr.rows.size(); i++) {
      int n = c_n >= 0 ? (int)tr.rows[i][c_n] : (int)i;
      auto it = by_tick.find(n);
      ct.tick(n, get(tr.rows[i], c_t), it == by_tick.end() ? std::vector<ClipRow>{} : it->second);
      last = n;
    }
    (void)last;
    ct.finish();
  }
  // ---- events
  auto evs = read_event_lines(dir);
  std::map<std::string, int> kinds;
  bool crash_marker = false;
  double log_dropped_total = 0, writer_lag = 0, werr = 0;
  double d_ticks = 0, d_states = 0, d_frames = 0, d_clips = 0, d_events = 0;
  int ls_ver = 0;
  double ls_evicted = 0, ls_trimmed = 0, ls_pushed = 0, ls_written = 0, ls_preroll_written = -1, ls_maxq_state = 0, ls_maxq_tick = 0, ls_maxq_event = 0, ls_cap_state = 0, ls_cap_tick = 0, ls_cap_event = 0;
  double ls_pass_max = 0, ls_fsync_max = 0, ls_fsync_mean = 0, ls_rx = 0, ls_lost_total = 0, ls_dup = 0, ls_nom = 0, ls_maxgap = 0, t_stream_begin = 0;
  std::string fault_detail, first_fault, exit_trigger_detail;
  int n_rpc = 0, n_fsm_changes = 0, n_remote = 0;
  double max_rpc_ms = 0, t_take = 0, t_trigger = 0, t_exit_res = 0;
  std::vector<std::string> timeline;
  bool exit_seen = false;
  for (auto& e : evs) {
    std::string kind;
    jstring(e, "kind", kind);
    kinds[kind]++;
    double tm = 0;
    jnumber(e, "t_mono", tm);
    if (kind == "CRASH") crash_marker = true;
    if (kind == "log_stats") {
      jnumber(e, "dropped_ticks", d_ticks); jnumber(e, "dropped_states", d_states); jnumber(e, "dropped_frames", d_frames); jnumber(e, "dropped_clips", d_clips); jnumber(e, "dropped_events", d_events);
      jnumber(e, "writer_max_lag_ms", writer_lag); jnumber(e, "write_errors", werr);
      double v = 0;  // V1.6 (log_stats_ver 2): ring evictions are counted apart from real drops; older logs have no such fields
      ls_ver = jnumber(e, "log_stats_ver", v) ? (int)v : 1;
      auto g = [&](const char* k, double& dst) { double x = 0; if (jnumber(e, k, x)) dst = x; };
      g("state_ring_evicted", ls_evicted); g("state_preroll_trimmed", ls_trimmed); g("state_pushed", ls_pushed); g("states_written", ls_written); g("preroll_written_s", ls_preroll_written);
      g("max_stream_state_queue", ls_maxq_state); g("max_tick_queue", ls_maxq_tick); g("max_event_queue", ls_maxq_event); g("state_cap", ls_cap_state); g("tick_cap", ls_cap_tick); g("event_cap", ls_cap_event);
      g("writer_pass_max_ms", ls_pass_max); g("fdatasync_max_ms", ls_fsync_max); g("fdatasync_mean_ms", ls_fsync_mean); g("lowstate_rx", ls_rx); g("lowstate_lost_est", ls_lost_total);
      g("lowstate_dup_ticks", ls_dup); g("lowstate_nominal_dtick", ls_nom); g("lowstate_max_gap_ms", ls_maxgap);
    }
    if (kind == "state_stream_begin" && t_stream_begin == 0) t_stream_begin = tm;
    if (kind == "rpc") { n_rpc++; double l = 0; if (jnumber(e, "latency_ms", l)) max_rpc_ms = std::max(max_rpc_ms, l); }
    if (kind == "fsm_change") n_fsm_changes++;
    if (kind.rfind("remote_", 0) == 0) n_remote++;
    if (kind == "user_control_confirmed" && t_take == 0) t_take = tm;
    if (kind == "exit_result") t_exit_res = tm;
    if (kind == "EXIT_TRIGGER" && t_trigger == 0) { t_trigger = tm; jstring(e, "fault", first_fault); jstring(e, "detail", exit_trigger_detail); }
    if (kind == "EXIT_TRIGGER" || kind.rfind("exit_", 0) == 0) exit_seen = true;
    static const std::set<std::string> keep = {"phase", "user_control_confirmed", "take_over", "policy_reset", "EXIT_TRIGGER", "exit_submit", "exit_rpc_attempt", "exit_rpc_result", "exit_fsm_matched", "exit_stable_confirmed",
                                               "exit_frames_stopped", "exit_escalate_passive", "exit_result", "exit_fsm_dropped_after_match", "fsm_change", "remote_start_double_click", "remote_l2b_emergency", "state_stream_begin",
                                               "state_stream_end", "SwitchToUserCtrl", "CRASH", "cmd_rejected", "post_exit_end", "released", "EXIT_UNCONFIRMED"};
    if (keep.count(kind) && timeline.size() < 120) {
      std::string brief;
      for (const char* k : {"phase", "fault", "mode", "result", "fsm", "to", "rc", "text"}) {
        std::string sv;
        double nv;
        if (jstring(e, k, sv)) brief += std::string(k) + "=" + sv + " ";
        else if (jnumber(e, k, nv)) brief += std::string(k) + "=" + num(nv) + " ";
      }
      timeline.push_back("{\"t_mono\":" + num(tm) + ",\"kind\":" + q(kind) + ",\"brief\":" + q(brief) + "}");
    }
  }
  (void)exit_seen;
  log_dropped_total = d_ticks + d_states + d_frames + d_clips + d_events;

  // ---- state stream / obs rows
  uint64_t obs_records = 0, st_records = 0;
  double st_first = 0, st_last = 0;
  struct StreamScan { bool ok = false; uint64_t seq_missing_pre = 0, seq_missing_stream = 0, seq_gaps_pre = 0, seq_gaps_stream = 0, dup = 0, tick1 = 0, run_dticks = 0, run_tick_sum = 0; uint32_t max_dtick = 0; double first_stream_gap_t = -1, run_gap_max_ms = 0, gap_max_ms = 0; } ss;
  int obs_dim = 0;
  {
    std::ifstream f(dir + "/obs.bin", std::ios::binary);
    if (f) {
      char mg[8] = {0};
      uint32_t jl = 0;
      f.read(mg, 8); f.read((char*)&jl, 4);
      std::string js(jl, ' ');
      f.read(&js[0], jl);
      double od = 0;
      jnumber(js, "obs_dim", od);
      obs_dim = (int)od;
      uint64_t sz = file_size(dir + "/obs.bin"), hdr = 12 + jl;
      uint64_t rec = 12 + (uint64_t)obs_dim * 4;
      obs_records = (obs_dim > 0 && sz >= hdr) ? (sz - hdr) / rec : 0;
    }
    std::ifstream g(dir + "/state_stream.bin", std::ios::binary);
    if (g) {
      char mg[8] = {0};
      uint32_t jl = 0;
      g.read(mg, 8); g.read((char*)&jl, 4);
      g.seekg(jl, std::ios::cur);
      uint64_t sz = file_size(dir + "/state_stream.bin"), hdr = 12 + jl;
      st_records = sz >= hdr ? (sz - hdr) / STATE_REC_BYTES : 0;
      if (st_records) {
        double t0 = 0, t1 = 0;
        g.seekg((std::streamoff)hdr, std::ios::beg); g.read((char*)&t0, 8);
        g.seekg((std::streamoff)(hdr + (st_records - 1) * STATE_REC_BYTES), std::ios::beg); g.read((char*)&t1, 8);
        st_first = t0; st_last = t1;
        // V1.6: reconcile the counters against the file itself. seq = the Link's per-callback counter (contiguous unless a callback's sample was NOT recorded), tick = the robot's own counter
        // (a jump = a sample the robot / DDS never delivered). The run window = take-over (trace first row) .. end of the trace.
        g.seekg((std::streamoff)hdr, std::ios::beg);
        std::vector<char> blk((size_t)STATE_REC_BYTES * 4096);
        uint64_t left = st_records, pseq = 0, ptick = 0; double pt = 0; bool have = false;
        const double tb = t_stream_begin > 0 ? t_stream_begin : first_t_mono;
        while (left) {
          const size_t n = (size_t)std::min<uint64_t>(left, 4096);
          g.read(blk.data(), (std::streamsize)(n * STATE_REC_BYTES));
          if (!g) break;
          for (size_t k = 0; k < n; k++) {
            const char* r = blk.data() + k * STATE_REC_BYTES;
            double t; uint64_t sq; uint32_t tk;
            memcpy(&t, r, 8); memcpy(&sq, r + 8, 8); memcpy(&tk, r + 16, 4);
            const bool in_stream = tb > 0 && t >= tb, in_run = first_t_mono > 0 && t >= first_t_mono && t <= last_t_mono;
            if (have) {
              if (sq > pseq + 1) { const uint64_t m = sq - pseq - 1; (in_stream ? ss.seq_missing_stream : ss.seq_missing_pre) += m; if (in_stream) ss.seq_gaps_stream++; else ss.seq_gaps_pre++; if (in_stream && ss.first_stream_gap_t < 0) ss.first_stream_gap_t = t - tb; }
              const uint32_t dtk = tk - (uint32_t)ptick;
              if (dtk == 0) ss.dup++;
              else if (dtk < 100000) { if (dtk == 1) ss.tick1++; if (in_run) { ss.run_dticks++; ss.run_tick_sum += dtk; } ss.max_dtick = std::max<uint32_t>(ss.max_dtick, dtk); }
              if (in_run) ss.run_gap_max_ms = std::max(ss.run_gap_max_ms, (t - pt) * 1e3);
              ss.gap_max_ms = std::max(ss.gap_max_ms, (t - pt) * 1e3);
            }
            have = true; pseq = sq; ptick = tk; pt = t;
          }
          left -= n;
        }
        ss.ok = true;
      }
    }
  }
  // ---- V1.6 counter reconciliation. V1.5.x counted the OVERWRITING of the rolling pre-roll ring as `dropped_states` (it grows with the READY wait: 48784 after a 56 s wait on the robot) -> a false LOG_DROPPED.
  // Version-2 logs separate `state_ring_evicted` (by design) from `dropped_states` (real). For a version-1 log the real loss is what the file itself shows: missing seq numbers AFTER state_stream_begin.
  const bool ls_v2 = ls_ver >= 2;
  double states_real = d_states, states_evicted = ls_evicted;  // v2: only the rolling-ring overwrites are reported (the pre-roll trim at begin_stream is routine, listed in logging.state_preroll_trimmed)
  if (!ls_v2 && ls_ver == 1 && ss.ok) { states_real = std::min(d_states, (double)ss.seq_missing_stream); states_evicted = d_states - states_real; }
  const bool legacy_reclassified = !ls_v2 && ls_ver == 1 && d_states > states_real;
  log_dropped_total = d_ticks + states_real + d_frames + d_clips + d_events;
  size_t frame_rows = 0;
  { Csv fr; if (fr.load(dir + "/exit_frames.csv")) frame_rows = fr.rows.size(); }

  // ---- values from the base report
  std::string result, fault_s, exit_reason, mode_s;
  jstring(report, "result", result); jstring(report, "fault", fault_s); jstring(report, "exit_reason", exit_reason); jstring(report, "mode", mode_s);
  double man_run_s = 0, man_auth = 0;
  if (!had_report) {  // salvage: identity of the run comes from the manifest
    jstring(manifest, "mode", mode_s); jnumber(manifest, "run_s", man_run_s); jnumber(manifest, "authority", man_auth);
    result = "CRASH_NO_REPORT"; exit_reason = "runner ended without run_report.json";
  }
  const bool guardian_log = note.find("guardian") != std::string::npos;
  double detect_to_rpc = -1, rpc_latency = -1, exit_code = 0, final_fsm = -1, t_stable = -1;
  size_t ed = report.find("\"exit_detail\":");
  if (ed != std::string::npos) { jnumber(report, "detect_to_rpc_sent_ms", detect_to_rpc, ed); jnumber(report, "rpc_latency_ms", rpc_latency, ed); jnumber(report, "t_stable_confirmed", t_stable, ed); }
  jnumber(report, "exit_code", exit_code); jnumber(report, "final_fsm", final_fsm);
  bool recovery_ok = false, took = false;
  jbool_(report, "recovery_ok", recovery_ok); jbool_(report, "took_control", took);

  // ---- anomaly rules
  std::vector<Item> items;
  auto add = [&](const char* lv, const char* code, const std::string& msg) { items.push_back({lv, code, msg}); };
  char b[1400];
  if (!had_report) add("FAIL", "NO_RUN_REPORT", "runner 没有写出 run_report.json（进程崩溃/被杀）：本报告由日志文件补写，退出结果以 guardian 记录为准");
  if (crash_marker) add("FAIL", "PROCESS_CRASH", "日志中有 CRASH 标记：进程收到致命信号（SIGSEGV/SIGABRT…），数据已尽量保全");
  if (!result.empty() && had_report) {
    if (result == "COMPLETED") add("INFO", "RESULT", "正常结束：WalkRun 恢复已确认");
    else if (result == "EXITED_WALKRUN") add(exit_reason.rfind("REMOTE_STOP", 0) == 0 || exit_reason.rfind("SIGNAL", 0) == 0 || exit_reason.find("operator") != std::string::npos ? "INFO" : "WARN", "RESULT", "提前退出并已恢复 WalkRun：" + exit_reason);
    else if (result == "CHECK_PASSED") add("INFO", "RESULT", "只读检查通过：机器人未被控制（check-only，没有下发任何指令）");
    else if (result == "ABORTED_BEFORE_ARM" || result == "READY_TIMEOUT") add("INFO", "RESULT", "接管前取消（机器人未被控制）：" + exit_reason);
    else add("FAIL", "RESULT", "运行结果 " + result + "：" + exit_reason);
  }
  const bool operator_stop = fault_s == "REMOTE_STOP" || fault_s == "SIGNAL";
  if (operator_stop) add("INFO", "OPERATOR_STOP", "操作员主动退出（" + fault_s + "）：" + exit_trigger_detail);
  else if (!fault_s.empty() && fault_s != "NONE") add("FAIL", "FAULT", "触发故障 " + fault_s + (exit_trigger_detail.empty() ? "" : "：" + exit_trigger_detail));
  if (took && !recovery_ok && result != "EMERGENCY_PASSIVE") add("FAIL", "RECOVERY", "WalkRun 恢复未确认（final_fsm=" + num(final_fsm) + "）");
  if (detect_to_rpc > 20) { snprintf(b, sizeof b, "检测→退出 RPC 发出 %.1f ms（>20 ms）", detect_to_rpc); add("WARN", "EXIT_LATENCY", b); }
  if (rpc_latency > 500) { snprintf(b, sizeof b, "SwitchToInternalCtrl RPC 往返 %.0f ms（>500 ms）", rpc_latency); add("WARN", "RPC_SLOW", b); }
  double trace_tilt_max = 0, trace_gyro_max = 0;
  const size_t nt = rs.ticks.size();
  if (nt) {
    std::vector<double> dt, inf, age, dds;
    for (auto& t : rs.ticks) { dt.push_back(t.dt_ms); inf.push_back(t.infer_ms); age.push_back(t.state_age_ms); dds.push_back(t.dds_loop_ms); }
    Pctl pdt = percentiles(dt), pinf = percentiles(inf), page = percentiles(age), pdds = percentiles(dds);
    long late = 0, rep = 0, gap = 0, sto = 0;
    double tilt = 0, gyro = 0, dq = 0, tau = 0, te = 0;
    uint64_t lost = 0;
    for (auto& t : rs.ticks) {
      late += (t.flags & TF_LATE) != 0; rep += (t.flags & TF_REPEAT_STATE) != 0; gap += (t.flags & TF_TICK_GAP) != 0; sto += (t.flags & TF_SYNC_TIMEOUT) != 0;
      tilt = std::max(tilt, t.tilt); gyro = std::max(gyro, t.gyro); dq = std::max(dq, t.max_dq_ratio); tau = std::max(tau, t.max_tau_ratio); te = std::max(te, t.track_err_max); lost = std::max(lost, t.ls_lost);
    }
    if (late) { snprintf(b, sizeof b, "控制周期超时 %ld/%zu 个 tick（>30 ms），最大步长 %.1f ms", late, nt, pdt.max); add(late * 20 > (long)nt ? "FAIL" : "WARN", "LATE_TICKS", b); }
    if (pinf.p99 > 15 || pinf.max > 30) { snprintf(b, sizeof b, "推理耗时 p99 %.1f ms / 最大 %.1f ms（控制周期 20 ms）", pinf.p99, pinf.max); add(pinf.max > 60 ? "FAIL" : "WARN", "INFER_SLOW", b); }
    if (page.max > 20) { snprintf(b, sizeof b, "取用状态时年龄最大 %.1f ms（lowstate 陈旧，>20 ms）", page.max); add(page.max > 50 ? "FAIL" : "WARN", "STATE_STALE", b); }
    if (rep) { snprintf(b, sizeof b, "%ld 个 tick 重复使用同一 lowstate 样本（丢帧/延迟）", rep); add("WARN", "REPEAT_STATE", b); }
    if (lost || gap) {
      // V1.6: `lost` (ls_lost) is cumulative since the run statistics began (the F1+F3 trigger), so it also counts the whole READY wait; the number that matters for the control window is the increase over the trace.
      // lowstate here is a ~1 kHz stream whose `tick` is a millisecond counter: one missing tick = one missing 1 ms sample, NOT a missed 20 ms control step. WARN only when the window really has a hole (arrival gap > one control period).
      const uint64_t lost_first = rs.ticks.empty() ? 0 : rs.ticks.front().ls_lost;
      const uint64_t lost_run = lost >= lost_first ? lost - lost_first : 0;
      const double gap_ms = ss.ok ? ss.run_gap_max_ms : 0;
      snprintf(b, sizeof b, "lowstate 缺失 tick：控制窗口内 %llu（自 F1+F3 触发累计 %llu），控制 tick 中带「tick 跳变」标志 %ld 个；控制窗口内最大到达间隔 %.1f ms（控制周期 20 ms）；%s重复 tick %llu 个（同一 tick 的重复样本，不算丢失）；lowstate 约 1 kHz，tick 是毫秒计数，缺 1 个 tick = 缺 1 ms 样本，不是漏了 20 ms 控制周期",
               (unsigned long long)lost_run, (unsigned long long)lost, gap, gap_ms, ls_nom > 0 ? (std::string("名义 tick 步长 ") + std::to_string((int)ls_nom) + "（1=1 kHz，2=500 Hz）；").c_str() : "", (unsigned long long)(ss.ok ? ss.dup : (uint64_t)ls_dup));
      add(gap_ms > 20.0 || (ss.ok ? (ss.run_dticks > 0 && (double)(ss.run_tick_sum - ss.run_dticks) > 0.01 * (double)ss.run_tick_sum) : (lost_run > 0 && lost_run * 100 > (uint64_t)(nt * 20))) ? "WARN" : "INFO", "LOWSTATE_LOSS", b);
    }
    if (sto) { snprintf(b, sizeof b, "%ld 个 tick 等待新状态超时（8 ms）", sto); add("WARN", "SYNC_TIMEOUT", b); }
    if (pdds.max > 5) { snprintf(b, sizeof b, "DDS 指令回环延迟 p99 %.2f ms / 最大 %.2f ms", pdds.p99, pdds.max); add("WARN", "DDS_LOOPBACK", b); }
    trace_tilt_max = tilt; trace_gyro_max = gyro;
    if (dq > 0.9) { snprintf(b, sizeof b, "关节速度达到硬件限速的 %.0f%%", dq * 100); add("WARN", "JOINT_SPEED", b); }
    if (tau > 1.0) { snprintf(b, sizeof b, "测得力矩达到额定峰值的 %.0f%%", tau * 100); add(tau > 1.2 ? "WARN" : "INFO", "TORQUE", b); }
    if (te > 0.5) { snprintf(b, sizeof b, "关节位置与下发指令最大偏差 %.2f rad", te); add("WARN", "CMD_NOT_FOLLOWED", b); }
  } else if (took && !guardian_log && had_report) add("WARN", "NO_TICKS", "已接管但 trace.csv 没有控制 tick 记录");
  if (ct.ticks_with_clip > 0) {
    struct Agg { double ex = 0; int n_ticks = 0; double longest = 0; uint8_t reasons = 0; };
    std::map<int, Agg> agg;
    bool torque = false;
    for (auto& e : ct.episodes()) {
      Agg& g = agg[e.joint];
      g.ex = std::max(g.ex, e.max_excess); g.n_ticks += e.n_ticks; g.longest = std::max(g.longest, e.duration_s); g.reasons |= e.reason;
      torque = torque || (e.reason & CLIP_TORQUE);
    }
    std::vector<std::pair<int, Agg>> v(agg.begin(), agg.end());
    std::sort(v.begin(), v.end(), [](const std::pair<int, Agg>& a, const std::pair<int, Agg>& b) { return a.second.ex > b.second.ex; });
    std::string worst;
    for (size_t i = 0; i < v.size() && i < 4; i++) {
      char w[200];
      snprintf(w, sizeof w, "%s%s(%s, 最大%.3frad, 共%d tick, 最长%.2fs)", i ? "；" : "", v[i].first < (int)jnames.size() ? jnames[v[i].first].c_str() : std::to_string(v[i].first).c_str(), clip_reason_text(v[i].second.reasons), v[i].second.ex, v[i].second.n_ticks, v[i].second.longest);
      worst += w;
    }
    snprintf(b, sizeof b, "安全门裁剪了 %ld/%ld 个 tick 的指令（%.1f%%），%zu 段，涉及 %zu 个关节。逐关节最严重：", ct.ticks_with_clip, ct.ticks_total, ct.ticks_total ? 100.0 * (double)ct.ticks_with_clip / (double)ct.ticks_total : 0.0, ct.episodes().size(), v.size());
    add(torque || ct.ticks_with_clip * 5 > ct.ticks_total ? "WARN" : "INFO", "SAFETY_CLIP", std::string(b) + worst + "（明细见 safety_clips.csv / run_report.json safety_clips）");
  }
  // ---- P5.3: attitude extremes over trace + full-rate lowstate + the sample that triggered the exit (the trigger frame is not a trace row)
  double tilt_trig = 0;
  { size_t ad = report.find("\"attitude_at_decision\""); if (ad != std::string::npos) jnumber(report, "tilt_rad", tilt_trig, ad); }
  // ---- V1.6: segments of the trace (lead-in / dance / end-hold / paused) and the reference attitude per tick
  double man_li = 0, man_mf = 0, man_fps = 50;
  jnumber(manifest, "lead_in_frames", man_li); jnumber(manifest, "motion_frames", man_mf); jnumber(manifest, "motion_fps", man_fps);
  const int LI = (int)man_li, MF = (int)man_mf;
  std::string man_ref_mode; jstring(manifest, "ref_mode", man_ref_mode);
  const int c_ssb = col("ssb_frame"), c_seg = col("seg"), c_dfr = col("dance_frame"), c_qw = col("quat_w"), c_rqw = col("ref_rq_w"), c_tmono = col("t_mono");
  std::vector<int> seg_of(tr.rows.size(), 1);   // 0 lead-in | 1 dance | 2 end-hold | 3 paused / frozen reference
  {
    bool last_seen = false;
    for (size_t i = 0; i < tr.rows.size(); i++) {
      if (c_seg >= 0) { seg_of[i] = (int)tr.rows[i][c_seg]; continue; }   // V1.6 log: written by the runner
      if (man_ref_mode == "paused") { seg_of[i] = 3; continue; }          // legacy log: derive from ssb_frame and the manifest
      const int f = c_ssb >= 0 ? (int)tr.rows[i][c_ssb] : 0;
      if (f < LI) seg_of[i] = 0;
      else if (MF > 0 && f >= LI + MF - 1) { seg_of[i] = last_seen ? 2 : 1; last_seen = true; }
      else seg_of[i] = 1;
    }
  }
  RefTilt rtl;
  {
    // limits in force: the V1.6 runner logs them (events tilt_judge_config); V1.5.x logs have none -> A1 defaults above (fixed 0.40 as before for non-reference samples)
    std::string cfg_ev;
    for (auto& e : evs) { std::string k; jstring(e, "kind", k); if (k == "tilt_judge_config") { cfg_ev = e; break; } }
    if (cfg_ev.empty()) { size_t tj = report.find("\"tilt_judge\":"); if (tj != std::string::npos) cfg_ev = report.substr(tj); }
    if (!cfg_ev.empty()) {
      double v;
      if (jnumber(cfg_ev, "unstable_hard_tilt_rad", v)) { rtl.hard = v; rtl.from_log = true; }
      if (jnumber(cfg_ev, "ref_dev_rad", v)) rtl.dev = v;
      if (jnumber(cfg_ev, "ref_dev_min_tilt_rad", v)) rtl.dev_min = v;
      if (jnumber(cfg_ev, "ref_dev_growth_rad_s", v)) rtl.growth = v;
      if (jnumber(cfg_ev, "ref_max_age_s", v)) rtl.max_age = v;
      if (jnumber(cfg_ev, "unstable_tilt_rad", v)) rtl.fixed = v;
      bool b; if (jbool_(cfg_ev, "ref_aware", b)) rtl.ref_aware = b;
    }
    if (c_rqw >= 0 && c_tmono >= 0) {
      for (size_t i = 0; i < tr.rows.size(); i++) {
        const auto& r = tr.rows[i];
        rtl.t.push_back(r[c_tmono]);
        rtl.up.push_back(up_of(r[c_rqw], r[c_rqw + 1], r[c_rqw + 2], r[c_rqw + 3]));
        rtl.ref_tilt.push_back(std::acos(std::max(-1.0, std::min(1.0, rtl.up.back()[2]))));
        rtl.valid.push_back(seg_of[i] == 1 && (c_ssb < 0 || (int)r[c_ssb] > LI));   // same rule as the run-time judge: a running dance segment only
      }
    }
  }
  std::string seg_json = "{\"available\":false}";
  if (c_q >= 0 && c_ref >= 0 && !tr.rows.empty()) {
    struct Sg { size_t n = 0, clip = 0; double sq_ref = 0, mx_ref = 0, sq_cmd = 0, mx_cmd = 0, tilt_max = 0, ref_tilt_max = 0, dev_max = 0, dev_sum = 0, tdiff_max = -9, t0 = 1e18, t1 = -1e18, d0 = 1e18, d1 = -1e18; size_t ncmd = 0, nd = 0; } sg[4], first;
    const int c_cn2 = col("clip_n");
    for (size_t i = 0; i < tr.rows.size(); i++) {
      const auto& r = tr.rows[i];
      const bool is_first = c_n >= 0 ? (int)r[c_n] == 0 : i == 0;
      Sg& g = is_first ? first : sg[std::max(0, std::min(3, seg_of[i]))];
      g.n++;
      double e2 = 0, emx = 0, c2 = 0, cmx = 0;
      const bool has_cmd = c_qc >= 0 && std::isfinite(r[c_qc]);
      for (int j = 0; j < N; j++) {
        const double e = r[c_q + j] - r[c_ref + j];
        e2 += e * e; emx = std::max(emx, std::fabs(e));
        if (has_cmd) { const double c = r[c_q + j] - r[c_qc + j]; c2 += c * c; cmx = std::max(cmx, std::fabs(c)); }
      }
      g.sq_ref += e2 / N; g.mx_ref = std::max(g.mx_ref, emx);
      if (has_cmd) { g.sq_cmd += c2 / N; g.mx_cmd = std::max(g.mx_cmd, cmx); g.ncmd++; }
      if (c_cn2 >= 0 && r[c_cn2] > 0) g.clip++;
      const double tl_ = c_tilt >= 0 ? r[c_tilt] : 0;
      g.tilt_max = std::max(g.tilt_max, tl_);
      if (c_qw >= 0 && c_rqw >= 0) {
        const auto um = up_of(r[c_qw], r[c_qw + 1], r[c_qw + 2], r[c_qw + 3]);
        const double dv = ang3(um, rtl.up[i]), rt_ = rtl.ref_tilt[i];
        g.ref_tilt_max = std::max(g.ref_tilt_max, rt_); g.dev_max = std::max(g.dev_max, dv); g.dev_sum += dv; g.nd++; g.tdiff_max = std::max(g.tdiff_max, tl_ - rt_);
      }
      g.t0 = std::min(g.t0, get(r, c_t)); g.t1 = std::max(g.t1, get(r, c_t));
      if (c_ssb >= 0) { const double df = (c_dfr >= 0 ? r[c_dfr] : r[c_ssb] - LI); g.d0 = std::min(g.d0, df); g.d1 = std::max(g.d1, df); }
    }
    auto one = [&](const char* name, const Sg& g) {
      if (!g.n) return std::string();
      return std::string(",\"") + name + "\":{\"ticks\":" + std::to_string(g.n) + ",\"t_rel_first\":" + num(g.t0) + ",\"t_rel_last\":" + num(g.t1) + ",\"dance_frame_first\":" + num(g.d0 < 1e17 ? g.d0 : 0) + ",\"dance_frame_last\":" + num(g.d1 > -1e17 ? g.d1 : 0) +
             ",\"actual_vs_ref_joint_rms_rad\":" + num(std::sqrt(g.sq_ref / (double)g.n)) + ",\"actual_vs_ref_joint_max_rad\":" + num(g.mx_ref) + ",\"actual_vs_command_rms_rad\":" + num(g.ncmd ? std::sqrt(g.sq_cmd / (double)g.ncmd) : 0) + ",\"max_tilt_rad\":" + num(g.tilt_max) +
             ",\"max_ref_tilt_rad\":" + num(g.ref_tilt_max) + ",\"max_gravity_err_vs_ref_rad\":" + num(g.dev_max) + ",\"mean_gravity_err_vs_ref_rad\":" + num(g.nd ? g.dev_sum / (double)g.nd : 0) + ",\"max_tilt_minus_ref_tilt_rad\":" + num(g.tdiff_max > -8 ? g.tdiff_max : 0) +
             ",\"ticks_with_clip\":" + std::to_string(g.clip) + "}";
    };
    seg_json = "{\"available\":true,\"lead_in_frames\":" + std::to_string(LI) + ",\"motion_frames\":" + std::to_string(MF) + ",\"fps\":" + num(man_fps) + ",\"trace_has_seg_column\":" + (c_seg >= 0 ? "true" : "false") +
               ",\"definition\":\"trace ssb_frame = index into the REPLAYED clip (lead_in_frames of frozen first frame + the dance); dance_frame = ssb_frame - lead_in_frames (negative in the lead-in); lead_in = frozen first frame, dance = reference moving, end_hold = last frame held after the window, takeover_first_tick = tick 0 (authority ramp starts, excluded from the segments)\"" +
               one("lead_in", sg[0]) + one("dance", sg[1]) + one("end_hold", sg[2]) + one("paused_reference", sg[3]) + one("takeover_first_tick", first) + "}";
  }
  AttExtremes ae = attitude_extremes(dir, first_t_mono > 0 ? first_t_mono - 0.0 : t_take, t_exit_res > 0 ? t_exit_res : 0, trace_tilt_max, trace_gyro_max, tilt_trig, rtl.t.empty() ? nullptr : &rtl);
  {
    // V1.6 reference-aware: report the MEASURED tilt, the REFERENCE tilt at the same time and the difference; FAIL only when the robot deviates from the reference (gravity-direction error above ref_dev_rad with tilt above
    // ref_dev_min_tilt_rad), exceeds the ABSOLUTE hard cap, or - where no valid reference exists (lead-in / end-hold / paused / V1.5.x log without reference) - the fixed limit. A lean that FOLLOWS the reference is not a failure.
    const bool have_ref = ae.n_ref_valid > 0;
    snprintf(b, sizeof b, "最大倾角 %.2f rad（trace %.2f / 全速率 lowstate %.2f / 触发帧 %.2f）；该时刻参考倾角 %s，偏差(重力方向误差) %s；有效参考样本内最大偏差 %.2f rad（此时倾角 %.2f / 参考 %.2f，倾角−参考 最大 %.2f）；判据: 偏差>%.2f 且倾角>%.2f，或 >%.2f 绝对上限，或无有效参考时 >%.2f%s",
             ae.tilt_overall, ae.tilt_trace, ae.tilt_stream, ae.tilt_trigger, ae.ref_tilt_at_peak >= 0 ? f2(ae.ref_tilt_at_peak).c_str() : "无有效参考", ae.dev_at_peak >= 0 ? f2(ae.dev_at_peak).c_str() : "-", ae.max_dev,
             ae.tilt_at_max_dev, ae.ref_tilt_at_max_dev, ae.max_tilt_minus_ref > -8 ? ae.max_tilt_minus_ref : 0.0, rtl.dev, rtl.dev_min, rtl.hard, rtl.fixed, rtl.from_log ? "" : "（旧版日志，无运行时判据记录：采用 V1.6 默认阈值）");
    if (!have_ref && ae.stream_samples == 0 && ae.tilt_overall > rtl.fixed) add("FAIL", "TILT", std::string("(无参考信息，沿用固定阈值 ") + num(rtl.fixed) + ") " + b);
    else if (ae.viol_hard > 0) add("FAIL", "TILT", std::string("超过绝对硬上限 ") + num(rtl.hard) + " rad：" + b);
    else if (ae.viol_dev > 0) add("FAIL", "TILT", std::string("偏离参考姿态：") + b);
    else if (ae.viol_fixed > 0) add("FAIL", "TILT", std::string("无有效参考的阶段(前导/末帧保持/暂停)倾角超过固定阈值：") + b);
    else if (ae.tilt_overall > 0.25 || ae.max_dev > 0.6 * rtl.dev) add(ae.max_dev > 0.8 * rtl.dev ? "WARN" : "INFO", "TILT", std::string("倾角未超判据（跟随参考）：") + b);
  }
  if (ae.gyro_overall > 8) { snprintf(b, sizeof b, "最大角速度 %.1f rad/s（>8；trace %.1f / lowstate %.1f）", ae.gyro_overall, ae.gyro_trace, ae.gyro_stream); add("WARN", "GYRO", b); }
  int pct_out_x10 = 0; double raw_out_frac = 0;
  std::string trk = tracking_json(tr, jnames, L, jarray(manifest, "pos_lo"), jarray(manifest, "pos_hi"), jarray(manifest, "effort_lim"), pct_out_x10, raw_out_frac);
  if (raw_out_frac > 0.5) {
    snprintf(b, sizeof b, "Policy 原始目标有 %.0f%% 的 tick 超出硬限位（这是 Policy 输出本身的性质：机器人踝等关节被要求力矩饱和，安全门只是把目标裁到限位内）；对照训练仿真 robust_d 约 56%%", 100 * raw_out_frac);
    add("INFO", "RAW_TARGET_OUTSIDE_LIMITS", b);
  }
  if (trk.find("\"torque_vs_training\":[{") != std::string::npos) {  // worst shortfall joint
    size_t k = trk.find("\"torque_vs_training\":[{"), j0 = trk.find("\"joint\":\"", k);
    std::string jname;
    jstring(trk, "joint", jname, j0);
    double sm = 0, sx = 0, sf = 0;
    jnumber(trk, "shortfall_mean_nm", sm, j0); jnumber(trk, "shortfall_max_nm", sx, j0); jnumber(trk, "fraction_ticks_shortfall_gt_5nm", sf, j0);
    if (sm > 5) { snprintf(b, sizeof b, "与训练执行器相比力矩缺口最大的关节 %s：平均少 %.1f Nm（最大 %.1f Nm，%.0f%% 的 tick 少 >5 Nm）——目标被裁到限位内后驱动器收到的力矩小于训练时的饱和力矩", jname.c_str(), sm, sx, 100 * sf); add("WARN", "TORQUE_SHORTFALL_VS_TRAINING", b); }
  }
  const bool rows_ok = obs_records == 0 || obs_records == tr.rows.size();
  if (tr.bad_rows) { snprintf(b, sizeof b, "trace.csv 有 %zu 行不完整（进程被杀时的最后一行）", tr.bad_rows); add("WARN", "TRACE_PARTIAL_ROW", b); }
  if (!rows_ok) { snprintf(b, sizeof b, "obs.bin 记录数 %llu 与 trace.csv 行数 %zu 不一致", (unsigned long long)obs_records, tr.rows.size()); add("FAIL", "OBS_ROWS", b); }
  if (log_dropped_total > 0 || werr > 0) {
    // V1.6: a REAL drop (queue full while recording) is FAIL. A V1.5.x log whose only loss is a few state samples right after the take-over (the stream began on a full ring) is a WARN: the data is intact apart from those samples.
    const bool legacy_only_states = !ls_v2 && d_ticks + d_frames + d_clips + d_events == 0 && werr == 0;
    snprintf(b, sizeof b, "日志队列真实丢弃 %.0f 条（tick %.0f / 状态 %.0f / 指令帧 %.0f / 裁剪 %.0f / 事件 %.0f），写入错误 %.0f%s", log_dropped_total, d_ticks, states_real, d_frames, d_clips, d_events, werr,
             legacy_only_states ? "。旧版(V1.5.x)日志：状态丢失来自 state_stream_begin 时环形缓冲已满（接管后头几毫秒的样本），数据其余部分完整" : "");
    add(legacy_only_states ? "WARN" : "FAIL", "LOG_DROPPED", b);
  }
  if (states_evicted > 0) {
    snprintf(b, sizeof b, "pre-roll 环形缓冲滚动淘汰 %.0f 条（设计内：READY 等待期间只保留最近 %.0f 秒窗口，不是丢日志；真实丢弃 dropped_states=%.0f）%s", states_evicted, 5.0, states_real,
             legacy_reclassified ? "。注意：旧版(V1.5.x)日志把这些淘汰计入 dropped_states，已按 state_stream.bin 的序号对账重新分类" : "");
    add("INFO", "LOG_PREROLL_ROLLING", b);
  }
  if (ls_v2 && ls_preroll_written >= 0 && took && ls_preroll_written < 0.9 * 5.0 && ls_pushed > 0) {
    snprintf(b, sizeof b, "pre-roll 窗口实际只有 %.2f s（期望 ~5 s）：环形缓冲或 standby 私有环过短（lowstate 速率过高/READY 太短）", ls_preroll_written);
    add(st_first > 0 && ls_preroll_written >= 0 && ls_pushed < 100 ? "INFO" : "WARN", "PREROLL_SHORT", b);
  }
  if (ls_v2 && ls_cap_state > 0 && ls_maxq_state > 0.5 * ls_cap_state) {
    snprintf(b, sizeof b, "流式阶段 state 队列峰值 %.0f / 容量 %.0f（>50%%：写盘变慢，接近真实丢弃）", ls_maxq_state, ls_cap_state);
    add("WARN", "LOG_QUEUE_HIGH", b);
  }
  if (ss.ok && ss.seq_missing_stream > 0 && ls_v2) { snprintf(b, sizeof b, "state_stream.bin 流式阶段缺 %llu 个序号（%llu 处），与计数器 dropped_states=%.0f 对账", (unsigned long long)ss.seq_missing_stream, (unsigned long long)ss.seq_gaps_stream, d_states); add(ss.seq_missing_stream > d_states ? "WARN" : "INFO", "STATE_STREAM_SEQ_GAP", b); }
  if (took && st_records == 0 && file_exists(dir + "/state_stream.bin")) add("WARN", "NO_STATE_STREAM", "已接管但 state_stream.bin 没有状态样本");
  if (kinds.count("log_stats") == 0 && had_report) add("WARN", "NO_LOG_STATS", "没有 log_stats 事件（日志未正常关闭）");

  int n_fail = 0, n_warn = 0;
  for (auto& i : items) { n_fail += i.level == "FAIL"; n_warn += i.level == "WARN"; }
  std::string level = n_fail ? "FAIL" : n_warn ? "WARN" : "OK";
  R.level = level; R.n_fail = n_fail; R.n_warn = n_warn;

  // ---- summary json / text
  std::string aj = "{\"level\":" + q(level) + ",\"n_fail\":" + std::to_string(n_fail) + ",\"n_warn\":" + std::to_string(n_warn) + ",\"items\":[";
  std::string at = "PartyMimic 运行异常摘要（" + std::string(PM_VERSION) + "）  目录: " + dir + "\n结论: " + (level == "OK" ? "未发现异常" : level == "WARN" ? "有警告，请查看" : "存在失败项，禁止进入下一测试阶段") + "\n";
  at += "结果: " + (result.empty() ? std::string("(无)") : result) + "  原因: " + exit_reason + "  mode: " + mode_s + "  final_fsm: " + num(final_fsm) + "\n";
  for (size_t i = 0; i < items.size(); i++) {
    aj += std::string(i ? "," : "") + "{\"level\":" + q(items[i].level) + ",\"code\":" + q(items[i].code) + ",\"msg\":" + q(items[i].msg) + "}";
    at += "[" + items[i].level + "] " + items[i].code + ": " + items[i].msg + "\n";
  }
  aj += "]}";
  if (!note.empty()) at += "备注: " + note + "\n";

  std::string logging = "{\"dir\":" + q(dir) + ",\"complete\":" + (rows_ok && log_dropped_total == 0 && !crash_marker ? "true" : "false") + ",\"rows\":{\"trace\":" + std::to_string(tr.rows.size()) + ",\"trace_partial\":" + std::to_string(tr.bad_rows) +
                        ",\"obs\":" + std::to_string(obs_records) + ",\"state_stream\":" + std::to_string(st_records) + ",\"exit_frames\":" + std::to_string(frame_rows) + ",\"safety_clips\":" + std::to_string(clip_rows) +
                        ",\"events\":" + std::to_string(evs.size()) + "},\"state_stream_t_first\":" + num(st_first) + ",\"state_stream_t_last\":" + num(st_last) + ",\"trace_t_first\":" + num(first_t_mono) + ",\"trace_t_last\":" + num(last_t_mono) +
                        ",\"state_preroll_s\":" + num(first_t_mono > 0 && st_first > 0 ? first_t_mono - st_first : 0) + ",\"state_post_exit_s\":" + num(last_t_mono > 0 && st_last > 0 ? st_last - last_t_mono : 0) +
                        ",\"log_stats_ver\":" + std::to_string(ls_ver) + ",\"state_ring_evicted\":" + num(states_evicted) + ",\"state_preroll_trimmed\":" + num(ls_trimmed) + ",\"legacy_reclassified\":" + (legacy_reclassified ? "true" : "false") +
                        ",\"state_stream_seq_missing_pre_roll\":" + std::to_string((unsigned long long)ss.seq_missing_pre) + ",\"state_stream_seq_missing_stream\":" + std::to_string((unsigned long long)ss.seq_missing_stream) + ",\"state_stream_tick_dup\":" + std::to_string((unsigned long long)ss.dup) +
                        ",\"run_window_max_arrival_gap_ms\":" + num(ss.run_gap_max_ms) + ",\"preroll_written_s\":" + num(ls_preroll_written) + ",\"queue_peaks\":{\"state_stream\":" + num(ls_maxq_state) + ",\"tick\":" + num(ls_maxq_tick) + ",\"event\":" + num(ls_maxq_event) + "},\"queue_caps\":{\"state\":" + num(ls_cap_state) + ",\"tick\":" + num(ls_cap_tick) + ",\"event\":" + num(ls_cap_event) +
                        "},\"writer_pass_max_ms\":" + num(ls_pass_max) + ",\"fdatasync_max_ms\":" + num(ls_fsync_max) + ",\"fdatasync_mean_ms\":" + num(ls_fsync_mean) + ",\"dropped\":{\"ticks\":" + num(d_ticks) + ",\"states\":" + num(states_real) + ",\"frames\":" + num(d_frames) + ",\"clips\":" + num(d_clips) + ",\"events\":" + num(d_events) + "},\"writer_max_lag_ms\":" + num(writer_lag) +
                        ",\"write_errors\":" + num(werr) + ",\"crash_marker\":" + (crash_marker ? "true" : "false") + ",\"n_rpc\":" + std::to_string(n_rpc) + ",\"max_rpc_ms\":" + num(max_rpc_ms) + ",\"n_fsm_changes\":" + std::to_string(n_fsm_changes) +
                        ",\"n_remote_events\":" + std::to_string(n_remote) + ",\"files\":[";
  {
    bool first = true;
    DIR* dp = opendir(dir.c_str());
    std::vector<std::string> names;
    if (dp) { while (dirent* de = readdir(dp)) if (de->d_name[0] != '.') names.push_back(de->d_name); closedir(dp); }
    std::sort(names.begin(), names.end());
    for (auto& n : names) { if (is_dir(dir + "/" + n)) continue; logging += std::string(first ? "" : ",") + "{\"name\":" + q(n) + ",\"bytes\":" + std::to_string(file_size(dir + "/" + n)) + "}"; first = false; }
  }
  logging += "]}";
  std::string tl = "[";
  for (size_t i = 0; i < timeline.size(); i++) tl += (i ? "," : "") + timeline[i];
  tl += "]";
  char ab[700];
  snprintf(ab, sizeof ab, "{\"max_tilt_rad_overall\":%.6g,\"max_tilt_rad_trace\":%.6g,\"max_tilt_rad_lowstate_stream\":%.6g,\"max_tilt_rad_at_exit_trigger\":%.6g,\"t_mono_of_peak_in_stream\":%.6f,\"max_gyro_rad_s_overall\":%.6g,\"max_gyro_rad_s_trace\":%.6g,\"max_gyro_rad_s_lowstate_stream\":%.6g,\"stream_samples_used\":%zu,"
                  "\"note\":\"overall = max(trace ticks, every lowstate sample between take-over and the end of the exit, the sample that triggered the exit). The trigger frame is not a trace row (the loop stops before logging it), so trace alone under-reports.\"}",
           ae.tilt_overall, ae.tilt_trace, ae.tilt_stream, ae.tilt_trigger, ae.t_peak, ae.gyro_overall, ae.gyro_trace, ae.gyro_stream, ae.stream_samples);
  std::string refaware = "{\"limits\":{\"unstable_hard_tilt_rad\":" + num(rtl.hard) + ",\"ref_dev_rad\":" + num(rtl.dev) + ",\"ref_dev_min_tilt_rad\":" + num(rtl.dev_min) + ",\"ref_dev_growth_rad_s\":" + num(rtl.growth) + ",\"ref_max_age_s\":" + num(rtl.max_age) + ",\"ref_aware\":" + (rtl.ref_aware ? "true" : "false") + ",\"fixed_unstable_tilt_rad\":" + num(rtl.fixed) + ",\"from_run_log\":" + (rtl.from_log ? "true" : "false") +
                         "},\"samples_with_valid_reference\":" + std::to_string(ae.n_ref_valid) + ",\"samples_without_valid_reference\":" + std::to_string(ae.n_ref_invalid) + ",\"max_tilt_with_reference_rad\":" + num(ae.max_tilt_ref) + ",\"max_tilt_without_reference_rad\":" + num(ae.max_tilt_no_ref) +
                         ",\"ref_tilt_at_peak_rad\":" + num(ae.ref_tilt_at_peak) + ",\"gravity_err_at_peak_rad\":" + num(ae.dev_at_peak) + ",\"max_gravity_err_vs_ref_rad\":" + num(ae.max_dev) + ",\"tilt_at_max_gravity_err_rad\":" + num(ae.tilt_at_max_dev) + ",\"ref_tilt_at_max_gravity_err_rad\":" + num(ae.ref_tilt_at_max_dev) +
                         ",\"max_tilt_minus_ref_tilt_rad\":" + num(ae.max_tilt_minus_ref > -8 ? ae.max_tilt_minus_ref : 0) + ",\"violations\":{\"hard_cap\":" + std::to_string(ae.viol_hard) + ",\"reference_deviation\":" + std::to_string(ae.viol_dev) + ",\"fixed_limit_no_reference\":" + std::to_string(ae.viol_fixed) + "}}";
  std::string tail = ",\"attitude_extremes\":" + std::string(ab) + ",\"attitude_vs_reference\":" + refaware + ",\"tracking\":" + trk + ",\"tracking_segments\":" + seg_json + ",\"timing\":" + rs.timing_json() + ",\"safety_clips\":" + ct.summary_json(jnames) + ",\"logging\":" + logging + ",\"timeline\":" + tl +
                     ",\"anomaly\":" + aj + ",\"finalized_wall\":" + num(wall()) + ",\"finalized_by\":" + q(note.empty() ? "runner" : note) + "}";
  std::string out;
  if (had_report) {
    size_t e = report.find_last_of('}');
    out = report.substr(0, e) + tail;
  } else {
    out = "{\"summary\":{\"result\":\"CRASH_NO_REPORT\",\"exit_reason\":\"runner ended without run_report.json\",\"took_control\":" + std::string(t_take > 0 ? "true" : "false") + "},\"result\":\"CRASH_NO_REPORT\",\"exit_reason\":\"runner ended without run_report.json\"," +
          "\"mode\":" + q(mode_s) + ",\"run_s\":" + num(man_run_s) + ",\"authority\":" + num(man_auth) + ",\"reconstructed_from_logs\":true" + tail;
  }
  write_atomic(dir + "/run_report.json", out);
  write_atomic(dir + "/anomaly_summary.json", aj + "\n");
  write_atomic(dir + "/anomaly_summary.txt", at);
  R.report = out;
  R.text = at;
  return R;
}

// ---------------------------------------------------------------- check
namespace {
struct Chk {
  int fail = 0, warn = 0, pass = 0;
  void rec(const char* lv, const std::string& name, const std::string& d) {
    printf("[%-4s] %s: %s\n", lv, name.c_str(), d.c_str());
    if (!strcmp(lv, "FAIL")) fail++; else if (!strcmp(lv, "WARN")) warn++; else pass++;
  }
  void ok(bool c, const std::string& name, const std::string& d) { rec(c ? "PASS" : "FAIL", name, d); }
};
}  // namespace

int logtool_check(const std::string& dir, bool quiet_pass) {
  (void)quiet_pass;
  Chk C;
  if (!is_dir(dir)) { printf("[FAIL] run_dir: %s is not a directory\n", dir.c_str()); return 1; }
  std::string manifest, report;
  bool has_m = read_file(dir + "/manifest.json", manifest), has_r = read_file(dir + "/run_report.json", report);
  C.ok(has_r, "run_report.json", has_r ? "present" : "missing");
  std::string mode_name;
  jstring(report, "mode", mode_name);
  bool took = false;
  jbool_(report, "took_control", took);
  if (!has_m) C.rec("WARN", "manifest.json", "missing (run ended before the check phase?)");
  else {
    std::string pol, par, mot, ver;
    size_t s = manifest.find("\"sha256\":");
    jstring(manifest, "policy_onnx", pol, s); jstring(manifest, "policy_params", par, s); jstring(manifest, "motion_bin", mot, s);
    jstring(manifest, "program_version", ver);
    bool a = pol == FROZEN_SHA256_POLICY, bb = par == FROZEN_SHA256_POLICY_PARAMS, c = mot == FROZEN_SHA256_MOTION;
    C.ok(a && bb && c, "manifest_frozen_hashes", std::string("policy ") + (a ? "OK" : "MISMATCH " + pol) + ", params " + (bb ? "OK" : "MISMATCH") + ", motion " + (c ? "OK" : "MISMATCH") + ", program " + ver);
  }
  Csv tr;
  bool ht = tr.load(dir + "/trace.csv");
  C.ok(ht, "trace.csv", ht ? std::to_string(tr.rows.size()) + " rows, " + std::to_string(tr.hdr.size()) + " columns, " + std::to_string(tr.bad_rows) + " partial rows" : "missing");
  if (ht && !tr.rows.empty()) {
    int c_tm = tr.col("t_mono"), c_n = tr.col("n");
    bool mono_ok = true, n_ok = true, fin = true;
    int c_q = -1;
    for (size_t i = 0; i < tr.hdr.size(); i++) if (tr.hdr[i].rfind("q:", 0) == 0) { c_q = (int)i; break; }
    for (size_t i = 1; i < tr.rows.size(); i++) {
      if (c_tm >= 0 && !(tr.rows[i][c_tm] > tr.rows[i - 1][c_tm])) mono_ok = false;
      if (c_n >= 0 && tr.rows[i][c_n] != tr.rows[i - 1][c_n] + 1) n_ok = false;
    }
    for (auto& r : tr.rows) if (c_q >= 0) for (int j = 0; j < N; j++) fin = fin && std::isfinite(r[c_q + j]);
    C.ok(mono_ok && n_ok, "trace_time_order", "t_mono strictly increasing, n contiguous");
    C.ok(fin, "trace_measured_finite", "q finite in every row");
    for (const char* need : {"q_policy:", "q_pd:", "q_cmd:", "action:", "action_prev:", "tau_pd:", "ref_jp:", "kp:", "kd:", "dq:", "tau_est:"}) {
      bool f = false;
      for (auto& h : tr.hdr) f = f || h.rfind(need, 0) == 0;
      if (!f) C.rec("FAIL", "trace_columns", std::string("group ") + need + " missing");
    }
  }
  uint64_t obs_sz = file_size(dir + "/obs.bin");
  if (obs_sz) {
    std::ifstream f(dir + "/obs.bin", std::ios::binary);
    char mg[8] = {0};
    uint32_t jl = 0;
    f.read(mg, 8); f.read((char*)&jl, 4);
    std::string js(jl, ' ');
    f.read(&js[0], jl);
    double od = 0;
    jnumber(js, "obs_dim", od);
    uint64_t rec = 12 + (uint64_t)od * 4, body = obs_sz - 12 - jl;
    C.ok(std::string(mg).rfind(OBS_MAGIC, 0) == 0 && body % rec == 0, "obs.bin", std::to_string(body / rec) + " records of " + num(od) + " floats");
    C.ok(body / rec == tr.rows.size(), "obs_rows_match_trace", std::to_string(body / rec) + " vs " + std::to_string(tr.rows.size()));
  } else C.rec(took ? "FAIL" : "WARN", "obs.bin", "empty");
  uint64_t st_sz = file_size(dir + "/state_stream.bin");
  if (st_sz > 12) {
    std::ifstream g(dir + "/state_stream.bin", std::ios::binary);
    char mg[8] = {0};
    uint32_t jl = 0;
    g.read(mg, 8); g.read((char*)&jl, 4);
    uint64_t body = st_sz - 12 - jl;
    C.ok(std::string(mg).rfind(STATE_MAGIC, 0) == 0 && body % STATE_REC_BYTES == 0, "state_stream.bin", std::to_string(body / STATE_REC_BYTES) + " lowstate samples");
    if (body >= STATE_REC_BYTES * 2) {
      g.seekg(12 + jl, std::ios::beg);
      double prev = -1;
      bool order = true;
      std::vector<char> rec(STATE_REC_BYTES);
      for (uint64_t i = 0; i < body / STATE_REC_BYTES; i++) {
        g.read(rec.data(), (std::streamsize)STATE_REC_BYTES);
        double t;
        memcpy(&t, rec.data(), 8);
        if (t < prev) order = false;
        prev = t;
      }
      C.ok(order, "state_stream_time_order", "t_mono non-decreasing");
    }
  } else C.rec(took ? "FAIL" : "WARN", "state_stream.bin", "no samples");
  auto evs = read_event_lines(dir);
  C.ok(!evs.empty(), "events.jsonl", std::to_string(evs.size()) + " events");
  bool crash = false, stats_ev = false;
  for (auto& e : evs) { std::string k; jstring(e, "kind", k); crash = crash || k == "CRASH"; stats_ev = stats_ev || k == "log_stats"; }
  C.ok(!crash, "no_crash_marker", crash ? "CRASH event present" : "none");
  C.rec(stats_ev ? "PASS" : "WARN", "log_stats_event", stats_ev ? "present (logger closed cleanly)" : "missing (process died before closing the logger)");
  if (has_r) {
    std::string tmp;
    for (const char* k : {"result", "exit_reason", "run_s", "authority"}) {
      bool f = report.find(std::string("\"") + k + "\":") != std::string::npos;
      if (!f) C.rec("FAIL", "run_report_key", k);
    }
    bool has_tail = report.find(",\"timing\":") != std::string::npos && report.find("\"anomaly\":") != std::string::npos;
    C.ok(has_tail, "run_report_finalized", has_tail ? "timing / safety_clips / logging / anomaly sections present" : "finalize has not run");
    C.ok(file_exists(dir + "/anomaly_summary.txt") && file_exists(dir + "/anomaly_summary.json"), "anomaly_summary", "anomaly_summary.{json,txt}");
  }
  printf("LOGCHECK %s: %d pass, %d warn, %d fail\n", C.fail ? "FAIL" : "PASS", C.pass, C.warn, C.fail);
  return C.fail ? 1 : 0;
}

// ---------------------------------------------------------------- replay
int logtool_replay(const std::string& dir, const std::string& onnx, double tol) {
  std::ifstream f(dir + "/obs.bin", std::ios::binary);
  if (!f) { printf("REPLAY FAIL: no obs.bin\n"); return 1; }
  char mg[8] = {0};
  uint32_t jl = 0;
  f.read(mg, 8); f.read((char*)&jl, 4);
  std::string js(jl, ' ');
  f.read(&js[0], jl);
  double od = 0;
  jnumber(js, "obs_dim", od);
  std::vector<int> dims;
  {
    size_t k = js.find("\"layout\":");
    while (k != std::string::npos) {
      k = js.find("\"dim\":", k);
      if (k == std::string::npos) break;
      dims.push_back(atoi(js.c_str() + k + 6));
      k += 6;
    }
  }
  if (dims.size() != 4) { printf("REPLAY FAIL: obs.bin layout has %zu inputs (need 4)\n", dims.size()); return 1; }
  Csv tr;
  if (!tr.load(dir + "/trace.csv")) { printf("REPLAY FAIL: no trace.csv\n"); return 1; }
  int c_a = -1, c_n = tr.col("n");
  for (size_t i = 0; i < tr.hdr.size(); i++) if (tr.hdr[i].rfind("action:", 0) == 0) { c_a = (int)i; break; }
  if (c_a < 0) { printf("REPLAY FAIL: no action columns\n"); return 1; }
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "pm_replay"};
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(2);
  Ort::Session sess(env, onnx.c_str(), so);
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const char* names[4] = {"proprio", "root_quat_hist", "ref_joint_pos", "ref_root"};
  const char* outn[1] = {"action"};
  size_t nrec = 0, bad = 0;
  double maxd = 0;
  std::vector<float> obs((size_t)od);
  size_t row = 0;
  for (;;) {
    uint32_t n;
    double t;
    f.read((char*)&n, 4);
    f.read((char*)&t, 8);
    f.read((char*)obs.data(), (std::streamsize)obs.size() * 4);
    if (!f) break;
    std::vector<Ort::Value> in;
    size_t off = 0;
    std::vector<std::array<int64_t, 1>> shapes(4);
    for (int i = 0; i < 4; i++) {
      shapes[i] = {(int64_t)dims[i]};
      in.push_back(Ort::Value::CreateTensor<float>(mem, obs.data() + off, (size_t)dims[i], shapes[i].data(), 1));
      off += (size_t)dims[i];
    }
    auto out = sess.Run(Ort::RunOptions{nullptr}, names, in.data(), 4, outn, 1);
    const float* a = out[0].GetTensorData<float>();
    while (row < tr.rows.size() && c_n >= 0 && (uint32_t)tr.rows[row][c_n] != n) row++;
    if (row >= tr.rows.size()) break;
    double d = 0;
    for (int j = 0; j < N; j++) d = std::max(d, std::fabs((double)a[j] - (double)(float)tr.rows[row][c_a + j]));  // the CSV holds the float32 action with 9 digits = exact round trip
    maxd = std::max(maxd, d);
    if (d > tol) bad++;
    nrec++;
    row++;
  }
  printf("REPLAY %s: %zu ticks replayed through %s, max |action_replay - action_logged| = %.3e (tolerance %.1e), %zu ticks above tolerance\n", (bad == 0 && nrec > 0) ? "PASS" : "FAIL", nrec, onnx.c_str(), maxd, tol, bad);
  return (bad == 0 && nrec > 0) ? 0 : 1;
}

int run_logtool(int argc, char** argv, const Paths& P) {
  if (argc < 2) { fprintf(stderr, "usage: partymimic logtool finalize|check|replay <run_dir> [--onnx FILE] [--tol 1e-5]\n"); return 2; }
  std::string cmd = argv[0], dir = argv[1];
  while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
  if (cmd == "finalize") {
    FinalizeResult r = finalize_run(dir, "", "logtool finalize");
    printf("%s", r.text.c_str());
    return r.n_fail ? 1 : 0;
  }
  if (cmd == "check") return logtool_check(dir, false);
  if (cmd == "replay") {
    std::string onnx = P.asset("policy.onnx");
    double tol = 1e-5;  // same CPU architecture: bit-exact (0); aarch64 robot log replayed on x86: float32 rounding <= ~3e-6
    for (int i = 2; i + 1 < argc; i += 2) { if (!strcmp(argv[i], "--onnx")) onnx = argv[i + 1]; else if (!strcmp(argv[i], "--tol")) tol = atof(argv[i + 1]); }
    return logtool_replay(dir, onnx, tol);
  }
  fprintf(stderr, "unknown logtool command %s\n", cmd.c_str());
  return 2;
}

}  // namespace pm
