// Offline-exported kinematic chain (root floating base -> anchor body) evaluated in real time with plain quaternion algebra.
// The frozen P2 policy uses anchor_body_name=pelvis (the floating base), so its chain is empty and FK(root,q) == normalised root quaternion.
#pragma once
#include <array>
#include <string>
#include <vector>

namespace pm {

class FkChain {
 public:
  struct Link { int joint = -1; std::array<double, 4> body_quat{1, 0, 0, 0}; std::array<double, 3> axis{0, 0, 0}; };
  // text: lines "link <joint|-> qw qx qy qz ax ay az" (root->anchor order); joint names resolved against `names`
  bool parse(const std::string& text, const std::vector<std::string>& names, std::string& err);
  // orientation (wxyz) of the anchor body in the world frame
  std::array<double, 4> anchor_quat(const std::array<double, 4>& root_wxyz, const double* q29) const;
  const std::vector<Link>& links() const { return links_; }

 private:
  std::vector<Link> links_;
};

}  // namespace pm
