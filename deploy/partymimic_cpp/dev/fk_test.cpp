// FK check: C++ chain evaluation vs MuJoCo xquat on 200 random states for 5 bodies.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "pm/fk.hpp"
#include "pm/policy.hpp"
using namespace pm;

int main(int argc, char** argv) {
  std::string root = argc > 1 ? argv[1] : ".";
  Params p; std::string err;
  if (!Params::load(root + "/assets/policy_params.txt", p, err)) { printf("FAIL %s\n", err.c_str()); return 1; }
  std::string chains; read_file(root + "/dev/fixtures/fk_chains.txt", chains);
  std::vector<std::string> bodies, texts;
  std::istringstream in(chains); std::string line;
  while (std::getline(in, line)) { if (line.rfind("body ", 0) == 0) { bodies.push_back(line.substr(5)); texts.push_back(""); } else if (!texts.empty()) texts.back() += line + "\n"; }
  std::ifstream f(root + "/dev/fixtures/fk_ref.bin", std::ios::binary); int n, nb; f.read((char*)&n, 4); f.read((char*)&nb, 4);
  std::vector<double> qs((size_t)n * 29), qu((size_t)n * 4), ref((size_t)n * nb * 4);
  f.read((char*)qs.data(), qs.size() * 8); f.read((char*)qu.data(), qu.size() * 8); f.read((char*)ref.data(), ref.size() * 8);
  double worst = 0;
  for (int b = 0; b < nb; b++) {
    FkChain c; if (!c.parse(texts[b], p.names, err)) { printf("FAIL %s\n", err.c_str()); return 1; }
    double w = 0;
    for (int i = 0; i < n; i++) {
      auto x = c.anchor_quat({qu[i * 4], qu[i * 4 + 1], qu[i * 4 + 2], qu[i * 4 + 3]}, &qs[(size_t)i * 29]);
      const double* r = &ref[((size_t)i * nb + b) * 4];
      double d = std::min(std::fabs(x[0] - r[0]) + std::fabs(x[1] - r[1]) + std::fabs(x[2] - r[2]) + std::fabs(x[3] - r[3]),
                          std::fabs(x[0] + r[0]) + std::fabs(x[1] + r[1]) + std::fabs(x[2] + r[2]) + std::fabs(x[3] + r[3]));
      w = std::max(w, d);
    }
    printf("%-24s links %zu  max quat err %.2e\n", bodies[b].c_str(), c.links().size(), w);
    worst = std::max(worst, w);
  }
  printf("%s\n", worst < 1e-9 ? "FK PASS" : "FK FAIL");
  return worst < 1e-9 ? 0 : 1;
}
