// Policy numerical parity: C++ stack vs the Python deployment stack on the 354 golden robot states (deploy/partymimic_g1 reference).
#include <cmath>
#include <cstdio>
#include <fstream>

#include "pm/policy.hpp"
using namespace pm;

int main(int argc, char** argv) {
  std::string root = argc > 1 ? argv[1] : ".";
  Params p; Motion m; std::string err;
  if (!Params::load(root + "/assets/policy_params.txt", p, err) || !Motion::load(argc > 3 ? argv[3] : root + "/assets/motion.bin", m, err)) { printf("FAIL load: %s\n", err.c_str()); return 1; }
  std::ifstream f(root + "/dev/fixtures/parity_ref.bin", std::ios::binary);
  int n; f.read((char*)&n, 4);
  auto rd = [&](int cols) { std::vector<double> v((size_t)n * cols); f.read((char*)v.data(), v.size() * 8); return v; };
  auto q = rd(29), dq = rd(29), quat = rd(4), gyro = rd(3), act = rd(29), qt = rd(29);
  PolicyStack ps(p, m, argc > 2 ? argv[2] : root + "/assets/policy.onnx", 2);
  auto sample = [&](int k) {
    RobotSample s;
    for (int i = 0; i < 29; i++) { s.q[i] = (float)q[(size_t)k * 29 + i]; s.dq[i] = (float)dq[(size_t)k * 29 + i]; }
    for (int i = 0; i < 4; i++) s.quat[i] = (float)quat[(size_t)k * 4 + i];
    for (int i = 0; i < 3; i++) s.gyro[i] = (float)gyro[(size_t)k * 3 + i];
    return s;
  };
  ps.reset(sample(0), false);
  double da = 0, dt = 0, ma = 0;
  std::vector<double> ms;
  for (int k = 0; k < n; k++) {
    PolicyOut o = ps.tick(sample(k));
    ms.push_back(o.infer_ms);
    if (getenv("PM_DUMP")) { static std::ofstream d(getenv("PM_DUMP"), std::ios::binary); for (auto& v : ps.last_inputs()) d.write((const char*)v.data(), v.size() * 4); }
    if (k < 3 || k % 100 == 0) { double e = 0; for (int i = 0; i < 29; i++) e = std::max(e, std::fabs((double)o.action[i] - act[(size_t)k * 29 + i])); printf("  frame %d diff %.3e\n", k, e); }
    for (int i = 0; i < 29; i++) {
      da = std::max(da, std::fabs((double)o.action[i] - act[(size_t)k * 29 + i]));
      dt = std::max(dt, std::fabs(o.q_target[i] - qt[(size_t)k * 29 + i]));
      ma = std::max(ma, std::fabs(act[(size_t)k * 29 + i]));
    }
  }
  std::sort(ms.begin(), ms.end());
  printf("frames %d  max|action diff| %.3e  max|q_target diff| %.3e  (max|action| %.3f)  infer p50 %.2f ms p99 %.2f ms\n", n, da, dt, ma, ms[n / 2], ms[(size_t)(n * 0.99)]);
  bool ok = da < 1e-3 && dt < 1e-3;
  printf("%s (tol 1e-3, same tolerance as the Python selfcheck)\n", ok ? "PARITY PASS" : "PARITY FAIL");
  return ok ? 0 : 1;
}
