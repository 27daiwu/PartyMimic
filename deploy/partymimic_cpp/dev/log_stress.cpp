// DEV ONLY (never shipped): run-log stress test (V1.6). Feeds the RunLogger with synthetic rt/lowstate samples at a real-time rate (1 kHz / 2 kHz) + 50 Hz tick records and checks the counters:
//   * the pre-roll ring ROLLS BY DESIGN: `state_ring_evicted` counts the overwritten samples, `dropped_states` stays 0 (the V1.5.1 false LOG_DROPPED after a long READY wait)
//   * the pre-roll window really written is ~preroll_s whenever the ring is large enough; shorter only if the ring is too small
//   * a writer stall (slow disk, `dev_stall_s`) shorter than the queue capacity loses nothing; a longer one loses samples and they ARE counted in `dropped_states`
//   * identity for every run: state_pushed == states_written + dropped_states + state_ring_evicted + state_preroll_trimmed
//   * state_stream.bin of the recorded run has contiguous seq numbers (no unseen loss)
//   usage: log_stress [scenario ...]    scenarios: normal long_wait high_rate stall_small stall_big small_ring
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "pm/runlog.hpp"
#include "pm/util.hpp"

using namespace pm;
namespace pm { const char* clip_reason_text(uint8_t) { return "stub"; } }  // the real one lives in guard.cpp (robot library); this tool links the policy library only

static int g_fail = 0, g_pass = 0;
static void check(bool ok, const std::string& what) { printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str()); (ok ? g_pass : g_fail)++; }

struct Scn { const char* name; double rate_hz, pre_s, run_s, stall_s; size_t state_cap; };

// reads state_stream.bin: record count, first/last t, number of missing seq numbers
static void scan_stream(const std::string& path, uint64_t& n, double& t0, double& t1, uint64_t& missing) {
  n = 0; missing = 0; t0 = t1 = 0;
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return;
  char mg[8]; uint32_t jl = 0;
  if (fread(mg, 1, 8, f) != 8 || fread(&jl, 4, 1, f) != 1) { fclose(f); return; }
  fseek(f, (long)jl, SEEK_CUR);
  std::vector<char> rec(STATE_REC_BYTES);
  uint64_t pseq = 0; bool have = false;
  while (fread(rec.data(), 1, STATE_REC_BYTES, f) == STATE_REC_BYTES) {
    double t; uint64_t sq;
    memcpy(&t, rec.data(), 8); memcpy(&sq, rec.data() + 8, 8);
    if (!n) t0 = t;
    t1 = t; n++;
    if (have && sq > pseq + 1) missing += sq - pseq - 1;
    pseq = sq; have = true;
  }
  fclose(f);
}

static void run(const Scn& S) {
  printf("== %s: %.0f Hz, pre-roll phase %.1f s, stream %.1f s, writer stall %.1f s, state queue cap %zu\n", S.name, S.rate_hz, S.pre_s, S.run_s, S.stall_s, S.state_cap);
  char dir[64];
  snprintf(dir, sizeof dir, "/tmp/pm_log_stress_%s", S.name);
  { std::string c = std::string("rm -rf ") + dir; if (system(c.c_str())) {} }
  RunLogger L;
  LogOptions o;
  o.dir = dir; o.joint_names = std::vector<std::string>(N, "j"); o.obs_dim = 8; o.obs_layout_json = "[{\"name\":\"x\",\"dim\":8}]"; o.preroll_s = 5.0;
  o.state_cap = S.state_cap; o.tick_cap = 1024; o.install_crash_handlers = false; o.dev_stall_s = S.stall_s;
  if (!L.open(o)) { check(false, "open"); return; }
  LinkState st;
  st.q.setZero(); st.dq.setZero(); st.tau.setZero();
  uint64_t seq = 0; uint32_t tick = 1000000;
  const double dt = 1.0 / S.rate_hz;
  double next = mono();
  auto feed_until = [&](double t_end, bool with_ticks) {
    double next_tick = mono();
    uint32_t n = 0;
    while (mono() < t_end) {
      st.arrival = mono(); st.seq = ++seq; st.tick = tick += (uint32_t)(1000.0 / S.rate_hz + 0.5 > 0 ? (S.rate_hz <= 1000 ? 1000 / (uint32_t)S.rate_hz : 1) : 1);
      L.state(st);
      if (with_ticks && mono() >= next_tick) {
        TickRec r; memset((void*)&r, 0, sizeof r);
        r.n = n++; r.t_mono = mono(); r.t_rel = n * 0.02; r.obs_n = 8;
        L.tick(r);
        next_tick += 0.02;
      }
      next += dt;
      double w = next - mono();
      if (w > 0) sleep_s(w); else if (w < -0.05) next = mono();
    }
  };
  feed_until(mono() + S.pre_s, false);            // READY: the pre-roll ring rolls
  const LogStats pre = L.stats();
  L.begin_stream();
  feed_until(mono() + S.run_s, true);             // stream (+ ticks: the dev stall starts after the first tick record)
  L.end_stream();
  sleep_s(0.3);
  const LogStats s = L.stats();
  const std::string fields = log_stats_fields(s, L.lowstate());
  L.close();
  uint64_t n = 0, miss = 0; double t0 = 0, t1 = 0;
  scan_stream(std::string(dir) + "/state_stream.bin", n, t0, t1, miss);
  printf("  [info] pushed %llu written %llu(file %llu) evicted %llu trimmed %llu dropped %llu; ring evicted before begin %llu; preroll_written %.2f s; peak stream queue %zu/%zu; fdatasync max %.1f ms; writer pass max %.1f ms; file seq missing %llu\n",
         (unsigned long long)s.state_pushed, (unsigned long long)s.states, (unsigned long long)n, (unsigned long long)s.state_ring_evicted, (unsigned long long)s.state_preroll_trimmed, (unsigned long long)s.dropped_states,
         (unsigned long long)pre.state_ring_evicted, s.preroll_written_s, s.max_stream_state_queue, s.state_cap, s.fdatasync_max_ms, s.writer_pass_max_ms, (unsigned long long)miss);
  check(s.state_pushed == s.states + s.dropped_states + s.state_ring_evicted + s.state_preroll_trimmed, "accounting identity: pushed == written + dropped + ring_evicted + preroll_trimmed");
  check(s.states == n, "states_written == records in state_stream.bin");
  check(fields.find("\"log_stats_ver\":2") != std::string::npos && fields.find("\"state_ring_evicted\"") != std::string::npos, "log_stats event carries the V1.6 fields");
  const std::string nm = S.name;
  const double expect_ring_evict = S.rate_hz * S.pre_s - (double)S.state_cap;
  if (nm == "normal") {
    check(s.dropped_states == 0 && s.state_ring_evicted == 0, "normal run: dropped_states == 0 and nothing evicted (ring not full)");
    check(s.preroll_written_s > 4.8 && s.preroll_written_s < 5.2, "pre-roll window written ~5 s");
    check(miss == 0, "state_stream.bin seq contiguous");
  } else if (nm == "long_wait" || nm == "high_rate") {
    check(pre.state_ring_evicted > 0.8 * expect_ring_evict && pre.state_ring_evicted > 100, "ring rolled: state_ring_evicted > 0 (~pushed - capacity)");
    check(s.dropped_states == 0, "dropped_states == 0 (V1.5.1 reported ~all of the evictions here as LOG_DROPPED FAIL)");
    check(s.preroll_written_s > 4.8 && s.preroll_written_s < 5.2, "pre-roll window written ~5 s after a full ring");
    check(miss == 0, "state_stream.bin seq contiguous across the begin_stream boundary (no loss at the take-over)");
  } else if (nm == "stall_small") {
    check(s.dropped_states == 0 && s.dropped_ticks == 0, "writer stall 2 s < queue capacity: nothing lost");
    check(s.max_stream_state_queue > 1000, "the stall is visible in the peak queue depth");
    check(miss == 0, "state_stream.bin seq contiguous");
  } else if (nm == "stall_big") {
    check(s.dropped_states > 300, "writer stall longer than the queue: the loss is COUNTED in dropped_states");
    check(miss > 0 && miss >= s.dropped_states - 4, "the missing seq numbers in state_stream.bin match the counter");
  } else if (nm == "small_ring") {
    check(s.state_ring_evicted > 0 && s.dropped_states == 0, "small ring: evictions counted apart, no drop");
    check(s.preroll_written_s < 4.0, "pre-roll shortened by the small ring is VISIBLE (preroll_written_s < preroll_s)");
  }
}

int main(int argc, char** argv) {
  const std::vector<Scn> all = {
      {"normal", 1000, 8, 3, 0, 16384},       {"long_wait", 1000, 20, 3, 0, 16384}, {"high_rate", 2000, 10, 3, 0, 16384},
      {"stall_small", 1000, 8, 5, 2, 16384},  {"stall_big", 1000, 3, 8, 6, 2500},   {"small_ring", 1000, 6, 2, 0, 2000}};
  std::vector<std::string> want;
  for (int i = 1; i < argc; i++) want.push_back(argv[i]);
  for (auto& s : all) {
    bool on = want.empty();
    for (auto& w : want) on = on || w == s.name;
    if (on) run(s);
  }
  printf("\nlog stress: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
