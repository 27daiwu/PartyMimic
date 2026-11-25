#include "pm/fk.hpp"

#include <cmath>
#include <sstream>

namespace pm {
namespace {
using Q = std::array<double, 4>;
Q mul(const Q& a, const Q& b) {
  return {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
          a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
}
}  // namespace

bool FkChain::parse(const std::string& text, const std::vector<std::string>& names, std::string& err) {
  links_.clear();
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string tag;
    if (!(ss >> tag) || tag[0] == '#' || tag != "link") continue;
    std::string jn;
    Link l;
    ss >> jn;
    for (auto& v : l.body_quat) ss >> v;
    for (auto& v : l.axis) ss >> v;
    if (!ss) { err = "bad fk link line: " + line; return false; }
    if (jn != "-") {
      l.joint = -1;
      for (size_t i = 0; i < names.size(); i++) if (names[i] == jn) l.joint = (int)i;
      if (l.joint < 0) { err = "fk joint not in policy order: " + jn; return false; }
    }
    links_.push_back(l);
  }
  return true;
}

std::array<double, 4> FkChain::anchor_quat(const std::array<double, 4>& root, const double* q) const {
  double n = std::sqrt(root[0] * root[0] + root[1] * root[1] + root[2] * root[2] + root[3] * root[3]);
  Q x = {root[0] / n, root[1] / n, root[2] / n, root[3] / n};  // MuJoCo normalises the free-joint quaternion
  for (const Link& l : links_) {
    x = mul(x, l.body_quat);
    if (l.joint >= 0) {
      double an = std::sqrt(l.axis[0] * l.axis[0] + l.axis[1] * l.axis[1] + l.axis[2] * l.axis[2]);
      double h = 0.5 * q[l.joint], s = std::sin(h) / an;
      x = mul(x, Q{std::cos(h), s * l.axis[0], s * l.axis[1], s * l.axis[2]});
    }
  }
  return x;
}

}  // namespace pm
