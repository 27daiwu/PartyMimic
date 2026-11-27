// Transport to the G1 through the OFFICIAL SDK2 user-control route (mirrors unitree_sdk2/example/g1/high_level/g1_userctrl_dds_example.cpp):
//   rt/lowstate (HG LowState_)  <- robot          read-only
//   rt/user_lowcmd (HG LowCmd_) -> robot          ONLY after SwitchToUserCtrl succeeded
//   "sport" RPC service: GetFsmId (7001), SwitchToUserCtrl (7110), SwitchToInternalCtrl (7111)
// It deliberately does NOT use MotionSwitcher.ReleaseMode / rt/lowcmd.
//
// V1.5.1 DDS ENTITY LIFECYCLE RULE (root cause of the field crash `EntityDelegate::prevent_callbacks(): Assertion 'false' failed`, SIGABRT / rc 134, CHECKING phase):
//   CycloneDDS-C++ 0.10.2 asserts when a DataReader that has a data_available listener is DESTROYED while a callback is in flight (the callback thread then holds the
//   last reference and runs the destructor itself). The old traffic scan created and destroyed three temporary readers (rt/lowcmd carries ~500 Hz from the factory
//   service) on every F1+F3 and every L1+A. Therefore: NO DDS entity of this process is ever destroyed while the process lives. Readers / writers / RPC client are created
//   once, are IMMORTAL (Link::~Link() neutralises the callbacks and parks the entities in a process-lifetime graveyard; every process ends with _exit), and all state the
//   callbacks touch lives in a shared Core that outlives the Link object. Traffic classification uses persistent monitor readers + a rolling 10 s history instead of temporary readers.
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
  bool monitor_traffic = false;  // V1.5.1: persistent monitor readers on rt/lowcmd, rt/arm_sdk, rt/user_lowcmd + discovery participant + 10 s rolling history (the runner; created ONCE at start-up)
  double rpc_timeout_s = 2.0;
  std::string cmd_topic = "rt/user_lowcmd";
};

// P5.1: every sport-service RPC (request and response) is reported to an optional observer (the run logger): api, argument, return code, fsm id read, times.
struct RpcEvent {
  std::string api;   // GetFsmId | SwitchToUserCtrl | SwitchToInternalCtrl
  std::string arg;   // "" | WALKRUN | PASSIVE | LAST
  int rc = -1, fsm = -1;
  double t_req = 0, t_resp = 0;  // mono()
  std::string exception;         // non-empty if the SDK call threw
};
using RpcObserver = std::function<void(const RpcEvent&)>;

// DDS built-in discovery result: topic -> list of "participantGuid/endpointGuid"
using Writers = std::map<std::string, std::vector<std::string>>;

// Statistics over the last `span_s` seconds of the rolling history (sampled every 10 ms by an internal thread; see Link::window). covered = the history is long enough
// (monitors matched + `want` seconds old); a window that is not covered must not be used as evidence.
struct WindowStats {
  bool covered = false;
  double span_s = 0;            // actual span of the samples used
  uint64_t ls_rx = 0;           // rt/lowstate samples received in the window
  double ls_hz = 0;             // ls_rx / span
  uint32_t distinct_ticks = 0;  // distinct lowstate.tick values seen
  uint64_t user_cmd = 0;        // samples on the observed command topic (o.cmd_topic) in the window
  uint64_t topic_rx[3] = {0, 0, 0};  // monitors: rt/lowcmd, rt/arm_sdk, rt/user_lowcmd
};

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
  uint64_t cmd_rx_count() const { return core_->cmd_rx; }
  double last_active_rx() const { return core_->active_rx; }  // mono() of the last stiff (kp>0) frame seen on the command topic
  // currently ALIVE writers of the three command topics (DDS built-in discovery). With monitor_traffic it reads the persistent discovery reader (instant); otherwise the
  // persistent reader is created on first use and `wait_s` is the discovery grace. Nothing is ever destroyed.
  Writers discover_writers(double wait_s = 0.6);
  // samples seen on each command topic during the LAST `window_s` seconds (persistent monitor readers; distinguishes IDLE writers from ACTIVE controllers). If the monitors
  // are younger than the window (cold start) it waits for the remainder; with a warm Link (start-up) it returns at once. Topics: rt/lowcmd, rt/arm_sdk, rt/user_lowcmd.
  std::map<std::string, uint64_t> measure_traffic(double window_s, const std::vector<std::string>& topics = {"rt/lowcmd", "rt/arm_sdk", "rt/user_lowcmd"});
  // rolling history (needs monitor_traffic or state): statistics over the last window_s seconds; covered=false if the history does not span it yet
  WindowStats window(double window_s) const;
  bool monitors_ready() const;  // monitor readers created and the discovery grace (0.5 s) elapsed
  double monitor_age_s() const;
  void close();                 // idempotent: neutralise the callbacks and park the DDS entities forever (never destroys one)
  static uint32_t lowcmd_crc_packed(const std::array<uint32_t, 251>& words);
  void set_rpc_observer(RpcObserver fn);
  // P5.1 latency instrumentation of the command path
  double last_send_us() const { return core_->last_send_us; }       // time spent inside Write() of the last command
  double last_loopback_ms() const { return core_->loop_ms; }       // publish -> own subscriber on the same host (DDS stack latency), last / worst since reset
  double max_loopback_ms() const { return core_->loop_max_ms; }
  void reset_loopback_max() { core_->loop_max_ms = 0; }
  uint64_t n_sent() const { return core_->n_sent; }
  int domain() const { return domain_; }
  const std::string& iface() const { return iface_; }
  double mode_machine_expected() const { return 5; }

 private:
  struct Impl;
  // everything the DDS callbacks / the sampler thread touch: shared_ptr-owned, so it stays valid after the Link object is gone (the entities are immortal, see top of file)
  struct Core {
    std::atomic<bool> closed{false};
    std::mutex mtx;
    StatePtr latest;
    uint64_t rx_seq = 0;
    std::mutex lis_mtx;
    std::vector<std::function<void(const LinkState&)>> listeners;
    std::atomic<uint64_t> cmd_rx{0};
    std::atomic<double> active_rx{0};
    std::atomic<double> last_send_us{0}, loop_ms{0}, loop_max_ms{0};
    std::atomic<uint64_t> n_sent{0}, rx_base{0};
    std::array<std::atomic<double>, 4096> send_t;
    std::atomic<uint64_t> topic_rx[3];
    std::atomic<double> monitors_t0{0};     // mono() when the monitor readers were created (0 = none)
    // rolling history (10 ms samples, 12 s)
    struct Snap { double t; uint64_t ls_seq, cmd_rx, topic[3]; uint32_t ls_tick; };
    std::mutex hist_mtx;
    std::vector<Snap> hist;                 // ring, size HIST_N
    size_t hist_head = 0, hist_n = 0;
    static constexpr size_t HIST_N = 1200;
    std::atomic<bool> sampler_started{false};
    Core() { for (auto& a : send_t) a.store(0.0); for (auto& a : topic_rx) a.store(0); hist.resize(HIST_N); }
  };
  std::shared_ptr<Core> core_;
  std::shared_ptr<Impl> impl_;
  std::string iface_;
  int domain_;
  LinkOpts o_;
  std::mutex rpc_mtx_;
  RpcObserver rpc_obs_;
  void report_rpc(const RpcEvent& e);
  void ensure_monitors();
  void start_sampler();
  std::mutex wr_mtx_;
  std::atomic<bool> closed_{false};
};

}  // namespace pm
