// DEV PC ONLY: prints GOLDEN_ACTION for the 3 golden states of include/pm/selfcheck_golden.hpp for a given policy ONNX (used to rebase the on-robot
// selfcheck golden replay onto a new policy; same C++ stack + ORT as selfcheck). usage: gen_golden <pkg_root> <policy.onnx>
#include <cstdio>
#include "pm/policy.hpp"
#include "pm/selfcheck_golden.hpp"
using namespace pm;
int main(int argc, char** argv) {
  std::string root = argc > 1 ? argv[1] : ".", onnx = argc > 2 ? argv[2] : root + "/assets/policy.onnx";
  Params p; Motion m; std::string err;
  if (!Params::load(root + "/assets/policy_params.txt", p, err) || !Motion::load(root + "/assets/motion.bin", m, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  PolicyStack ps(p, m, onnx, 2);
  auto gs = [&](int k) {
    RobotSample s;
    for (int i = 0; i < N; i++) { s.q[i] = (float)GOLDEN_Q[k * N + i]; s.dq[i] = (float)GOLDEN_DQ[k * N + i]; }
    for (int i = 0; i < 4; i++) s.quat[i] = (float)GOLDEN_QUAT[k * 4 + i];
    for (int i = 0; i < 3; i++) s.gyro[i] = (float)GOLDEN_GYRO[k * 3 + i];
    return s;
  };
  ps.reset(gs(0), false);
  printf("constexpr double GOLDEN_ACTION[] = {");
  for (int k = 0; k < GOLDEN_N; k++) { PolicyOut o = ps.tick(gs(k)); for (int i = 0; i < N; i++) printf("%s%.17g", (k || i) ? ", " : "", (double)o.action[i]); }
  printf("};\n");
}
