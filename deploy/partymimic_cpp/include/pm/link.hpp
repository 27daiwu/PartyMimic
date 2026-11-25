// Transport to the G1 through the OFFICIAL SDK2 user-control route (mirrors unitree_sdk2/example/g1/high_level/g1_userctrl_dds_example.cpp):
//   rt/lowstate (HG LowState_)  <- robot          read-only
//   rt/user_lowcmd (HG LowCmd_) -> robot          ONLY after SwitchToUserCtrl succeeded
//   "sport" RPC service: GetFsmId (7001), SwitchToUserCtrl (7110), SwitchToInternalCtrl (7111)
// It deliberately does NOT use MotionSwitcher.ReleaseMode / rt/lowcmd.
#pragma once
#include <array>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "pm/policy.hpp"

namespace unitree_hg { namespace msg { namespace dds_ { class LowCmd_; } } }

namespace pm {

// CRC32 of an HG LowCmd_ over its 1004-byte wire layout (every word except the trailing crc), computed field by field (no struct casting)
uint32_t lowcmd_crc(const unitree_hg::msg::dds_::LowCmd_& m);

struct LinkState {
  Vec q, dq, tau;
  std::array<double, 4> quat{1, 0, 0, 0};  // wxyz (IMU)
  std::array<double, 3> gyro{};
  std::array<double, 3> rpy{};
  uint32_t tick = 0;
  int mode_machine = 0;
  double arrival = 0;  // mono() at reception
  uint64_t seq = 0;
  std::array<uint8_t, 40> wireless{};  // rt/lowstate.wireless_remote (same bytes remote_start decodes)
};
using StatePtr = std::shared_ptr<const LinkState>;

struct LinkOpts {
  bool rpc = true;           // create the sport RPC client (GetFsmId / switch)
  bool observe_cmd = true;   // count samples on the command topic (foreign-writer detection)
  bool state = true;         // subscribe rt/lowstate
  double rpc_timeout_s = 2.0;
  std::string cmd_topic = "rt/user_lowcmd";
};

// DDS built-in discovery result: topic -> list of "participantGuid/endpointGuid"
using Writers = std::map<std::string, std::vector<std::string>>;

class Link {
 public:
  Link(const std::string& iface, int domain, const LinkOpts& o = LinkOpts());
  ~Link();
  StatePtr latest() const;
  void add_state_listener(std::function<void(const LinkState&)> fn);  // call before traffic matters; callbacks run in the DDS reader thread: cheap only
  void enable_write();                                                  // create the publisher (silent until send())
  void send(const Vec& q, const Vec& dq, const Vec& tau, const Vec& kp, const Vec& kd);
  std::pair<int, int> get_fsm();       // (rpc_code, fsm_id or -1)
  int switch_to_user();                // rpc code
  int switch_to_internal(const std::string& mode);  // "LAST" | "PASSIVE" | "WALKRUN"
  uint64_t cmd_rx_count() const { return cmd_rx_; }
  double last_active_rx() const { return active_rx_; }  // mono() of the last stiff (kp>0) frame seen on the command topic
  Writers discover_writers(double wait_s = 0.6);
  // samples seen on each command topic in a window (temporary subscribers): distinguishes IDLE writers from ACTIVE controllers
  std::map<std::string, uint64_t> measure_traffic(double window_s, const std::vector<std::string>& topics = {"rt/lowcmd", "rt/arm_sdk", "rt/user_lowcmd"});
  static uint32_t lowcmd_crc_packed(const std::array<uint32_t, 251>& words);
  int domain() const { return domain_; }
  const std::string& iface() const { return iface_; }
  double mode_machine_expected() const { return 5; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string iface_;
  int domain_;
  LinkOpts o_;
  mutable std::mutex mtx_;
  StatePtr latest_;
  uint64_t rx_seq_ = 0;
  std::atomic<uint64_t> cmd_rx_{0};
  std::atomic<double> active_rx_{0};
  std::vector<std::function<void(const LinkState&)>> listeners_;
  std::mutex wr_mtx_, lis_mtx_;
};

}  // namespace pm
