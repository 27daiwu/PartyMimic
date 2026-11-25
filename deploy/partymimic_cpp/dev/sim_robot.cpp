// DEV ONLY (never shipped): loopback test robot for the control-exit-path checks. Domain 77 on `lo`.
//  * publishes rt/lowstate (500 Hz) with a standing G1; joints follow rt/user_lowcmd while the fsm is 1000 (user control), else hold
//  * "sport" RPC service: GetFsmId 7001, SwitchToUserCtrl 7110, SwitchToInternalCtrl 7111 (fsm: WalkRun 500/505/801 -> 1000 -> 500 | 1)
//  * "voice" service stub: logs every TtsMaker text (verifies the voice announcements)
//  * remote keys: hex mask in <dir>/keys (same bit order as gamepad.hpp)
//  * OEM model for the exit paths (all test hooks are command-line options):
//      --tilt-ramp D,R,T      after D s in user control the roll ramps 0 -> R rad over T s (gyro = ramp rate); stays there while fsm 1000/1
//      --oem-fall-tilt X      SwitchToInternalCtrl(WALKRUN) arriving (after --internal-delay) with tilt > X rad drops the OEM into fsm 1 (default 0.45: the two field falls)
//      --internal-delay S     latency of SwitchToInternalCtrl only (the RPC is slow while the robot keeps tilting)
//      --walkrun-fsm1         WALKRUN request: rpc returns 0 but the fsm becomes 1 (the field-log failure)
//      --ignore-walkrun       WALKRUN request: rpc returns 0, fsm unchanged (PASSIVE still honoured)
//      --user-reject          SwitchToUserCtrl is rejected (3203): ARM fails while control may be half-taken
//      --ignore-internal      every SwitchToInternalCtrl: rpc 0, fsm unchanged
//      --walkrun-drop-s S     WALKRUN reaches the entry fsm, then drops into fsm 1 S seconds later
//      /<dir>/mute            existing file = rt/lowstate muted (communication loss) while it exists
//      /<dir>/tilt            "roll_rad gyro_x" hook file: forces the published attitude (removed file = off)
//  * validates user_lowcmd frames (CRC, mode=1, mode_machine=5) and logs events to <dir>/sim_events.log
#include <signal.h>
#include <unistd.h>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/server/server.hpp>

#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>

#include "gamepad.hpp"
#include "pm/link.hpp"
#include "pm/policy.hpp"

using namespace pm;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;

static std::atomic<bool> g_stop{false};
static std::string g_dir = "/tmp/pm_sim";
static std::mutex g_log_m;
static double t0w = 0;
static void ev(const std::string& s) {
  std::lock_guard<std::mutex> lk(g_log_m);
  FILE* f = fopen((g_dir + "/sim_events.log").c_str(), "a");
  if (!f) return;
  fprintf(f, "%.3f %s\n", wall() - t0w, s.c_str());
  fclose(f);
}

static std::atomic<int> g_fsm{802};
static int g_entry_fsm = 802;  // the field robot reads 802 under the factory WalkRun
static std::atomic<int> g_user_since{0};
static std::atomic<long> g_stiff{0}, g_damp{0}, g_badcrc{0}, g_frames{0};
static std::atomic<double> g_last_cmd_t{0};
static Vec g_cmd_q = Vec::Zero();
static std::mutex g_cmd_m;
static std::atomic<bool> g_cmd_stiff{false};
static double g_rpc_delay = 0, g_internal_delay = 0, g_oem_fall_tilt = 0.45, g_walkrun_drop_s = -1;
static bool g_user_reject = false, g_ignore_internal = false, g_ignore_walkrun = false, g_walkrun_fsm1 = false;
static std::atomic<double> g_tilt{0};        // current published roll
static std::atomic<double> g_drop_at{0};
static std::atomic<long> g_exit_req_n{0};

static int parse_data(const std::string& p) {
  size_t k = p.find("\"data\"");
  if (k == std::string::npos) return -1;
  k = p.find(':', k);
  return k == std::string::npos ? -1 : atoi(p.c_str() + k + 1);
}

class SimSport : public unitree::robot::Server {
 public:
  SimSport() : Server("sport") {}
  void Init() override {
    SetApiVersion("1.0.0.0");
    RegistHandler(7001, [this](const std::string&, std::string& data) { data = "{\"data\":" + std::to_string(g_fsm.load()) + "}"; return 0; });
    RegistHandler(7110, [this](const std::string&, std::string&) {
      if (g_rpc_delay > 0) pm::sleep_s(g_rpc_delay);
      int f = g_fsm.load();
      if (g_user_reject) { ev("SwitchToUserCtrl REJECTED (--user-reject)"); return 3203; }
      if (f != 500 && f != 505 && f != 801 && f != 802) { ev("SwitchToUserCtrl REJECTED fsm=" + std::to_string(f)); return 3203; }
      g_fsm = 1000;
      g_user_since = 1;
      ev("SwitchToUserCtrl OK fsm 1000");
      return 0;
    });
    RegistHandler(7111, [this](const std::string& param, std::string&) {
      int m = parse_data(param);
      char b[200];
      snprintf(b, sizeof b, "REQUEST mode=%d tilt=%.3f stiff=%ld damp=%ld", m, g_tilt.load(), g_stiff.load(), g_damp.load());
      ev(std::string("SwitchToInternalCtrl ") + b);
      g_exit_req_n++;
      long st0 = g_stiff.load(), dm0 = g_damp.load();
      if (g_rpc_delay > 0) pm::sleep_s(g_rpc_delay);
      if (g_internal_delay > 0) pm::sleep_s(g_internal_delay);
      double tilt = g_tilt.load();
      int newfsm;
      std::string how;
      if (g_ignore_internal || (g_ignore_walkrun && m != 1)) { ev("SwitchToInternalCtrl IGNORED mode=" + std::to_string(m) + " (rpc 0, fsm unchanged)"); return 0; }
      if (m == 1) newfsm = 1;
      else if (g_walkrun_fsm1) { newfsm = 1; how = " [walkrun-fsm1: rpc 0 but OEM lands in fsm 1]"; }
      else if (tilt > g_oem_fall_tilt) { newfsm = 1; how = " [OEM cannot recover tilt " + std::to_string(tilt) + " > " + std::to_string(g_oem_fall_tilt) + ": falls into fsm 1]"; }
      else { newfsm = g_entry_fsm; if (g_walkrun_drop_s >= 0) g_drop_at = mono() + g_walkrun_drop_s; }
      g_fsm = newfsm;
      snprintf(b, sizeof b, " frames during the rpc: stiff +%ld damp +%ld", g_stiff.load() - st0, g_damp.load() - dm0);
      ev("SwitchToInternalCtrl mode=" + std::to_string(m) + (m == 0 ? "(LAST)" : m == 1 ? "(PASSIVE)" : "(WALKRUN)") + " -> fsm " + std::to_string(g_fsm.load()) + how + b);
      return 0;
    });
  }
};
class SimVoice : public unitree::robot::Server {
 public:
  SimVoice() : Server("voice") {}
  void Init() override {
    SetApiVersion("1.0.0.0");
    RegistHandler(1001, [](const std::string& p, std::string&) {
      size_t k = p.find("\"text\"");
      std::string t = k == std::string::npos ? p : p.substr(k);
      ev("TTS " + t);
      return 0;
    });
    RegistHandler(1006, [](const std::string& p, std::string&) { ev("SetVolume " + p); return 0; });
  }
};

int main(int argc, char** argv) {
  std::string iface = "lo";
  int domain = 77;
  double flip_after_user_s = -1, hz = 500;
  std::string rival;
  double ramp_d = -1, ramp_r = 0, ramp_t = 1;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto nx = [&]() { return std::string(argv[++i]); };
    if (a == "--dir") g_dir = nx();
    else if (a == "--fsm") g_entry_fsm = atoi(nx().c_str());
    else if (a == "--rpc-delay") g_rpc_delay = atof(nx().c_str());
    else if (a == "--ignore-internal") g_ignore_internal = true;
    else if (a == "--user-reject") g_user_reject = true;
    else if (a == "--ignore-walkrun") g_ignore_walkrun = true;
    else if (a == "--walkrun-fsm1") g_walkrun_fsm1 = true;
    else if (a == "--walkrun-drop-s") g_walkrun_drop_s = atof(nx().c_str());
    else if (a == "--internal-delay") g_internal_delay = atof(nx().c_str());
    else if (a == "--oem-fall-tilt") g_oem_fall_tilt = atof(nx().c_str());
    else if (a == "--tilt-ramp") sscanf(nx().c_str(), "%lf,%lf,%lf", &ramp_d, &ramp_r, &ramp_t);
    else if (a == "--flip-after-user-s") flip_after_user_s = atof(nx().c_str());
    else if (a == "--hz") hz = atof(nx().c_str());
    else if (a == "--rival") rival = nx();   // user_lowcmd | arm_sdk: a competing controller ACTIVELY writing
  }
  g_fsm = g_entry_fsm;
  t0w = wall();
  if (system(("mkdir -p " + g_dir).c_str()) != 0) return 1;
  signal(SIGTERM, [](int) { g_stop = true; });
  signal(SIGINT, [](int) { g_stop = true; });
  unitree::robot::ChannelFactory::Instance()->Init(domain, iface);
  Params p;
  std::string err;
  const char* root = getenv("PM_ROOT");
  Params::load(std::string(root ? root : ".") + "/assets/policy_params.txt", p, err);
  unitree::robot::ChannelPublisher<LowState> pub("rt/lowstate");
  pub.InitChannel();
  // factory-like DDS writers seen on the real robot: an ACTIVE rt/lowcmd motion service plus IDLE writers on rt/arm_sdk and rt/user_lowcmd
  unitree::robot::ChannelPublisher<LowCmd> p_low("rt/lowcmd"), p_arm("rt/arm_sdk"), p_user("rt/user_lowcmd");
  p_low.InitChannel(); p_arm.InitChannel(); p_user.InitChannel();
  LowCmd zero_cmd, rival_cmd;
  for (int i = 0; i < 29; i++) { rival_cmd.motor_cmd()[i].mode() = 1; rival_cmd.motor_cmd()[i].kp() = 10.f; }
  rival_cmd.mode_machine() = 5;
  rival_cmd.crc() = lowcmd_crc(rival_cmd);
  long iter = 0;
  SimSport sport; sport.Init(); sport.Start();
  SimVoice voice; voice.Init(); voice.Start();
  unitree::robot::ChannelSubscriber<LowCmd> sub("rt/user_lowcmd");
  sub.InitChannel([&](const void* m) {
    const LowCmd& c = *static_cast<const LowCmd*>(m);
    g_frames++;
    if (c.crc() != lowcmd_crc(c)) { if (g_badcrc++ == 0) ev("BAD CRC on user_lowcmd"); }
    bool stiff = false;
    Vec q;
    for (int i = 0; i < 29; i++) { if (c.motor_cmd()[i].kp() > 0) stiff = true; q[i] = c.motor_cmd()[i].q(); if (c.motor_cmd()[i].mode() != 1) ev("BAD mode"); }
    if (c.mode_machine() != 5) ev("BAD mode_machine");
    if (stiff) { if (g_stiff++ == 0) ev("first stiff user_lowcmd frame"); } else { if (g_damp++ == 0) ev("first damping user_lowcmd frame (kp=0)"); }
    {
      std::lock_guard<std::mutex> lk(g_cmd_m);
      g_cmd_q = q;
    }
    g_cmd_stiff = stiff;
    g_last_cmd_t = mono();
  }, 0);

  Vec q = p.default_pos, dq = Vec::Zero();
  double dt = 1.0 / hz, next = mono(), user_t0 = 0, last_keys_t = 0;
  uint16_t keys = 0;
  uint32_t tick = 0;
  LowState msg;
  int last_fsm = -1;
  double flip_t = 0, ramp_last = 0, hook_roll = -1, hook_gyro = 0;
  bool ramp_logged = false, hook_logged = false;
  ev("sim robot up (entry fsm " + std::to_string(g_entry_fsm) + ")");
  while (!g_stop) {
    double now = mono();
    if (now - last_keys_t > 0.02) {
      last_keys_t = now;
      std::ifstream f(g_dir + "/keys");
      std::string s;
      if (f && std::getline(f, s)) keys = (uint16_t)strtoul(s.c_str(), nullptr, 0);
      std::ifstream th(g_dir + "/tilt");
      if (th && std::getline(th, s)) { double r = 0, g = 0; if (sscanf(s.c_str(), "%lf %lf", &r, &g) >= 1) { hook_roll = r; hook_gyro = g; } } else hook_roll = -1;
      std::ifstream ff(g_dir + "/fsm");  // test hook: the factory/operator changes the mode
      if (ff && std::getline(ff, s) && !s.empty()) { g_fsm = atoi(s.c_str()); ::remove((g_dir + "/fsm").c_str()); }
    }
    int fsm = g_fsm.load();
    if (fsm != last_fsm) { ev("fsm -> " + std::to_string(fsm)); if (fsm == 1000) user_t0 = now; last_fsm = fsm; }
    // joint motion: follow the user command (stiff) while in user control
    bool follow = fsm == 1000 && g_cmd_stiff.load() && now - g_last_cmd_t.load() < 0.1;
    Vec target;
    { std::lock_guard<std::mutex> lk(g_cmd_m); target = g_cmd_q; }
    Vec nq = q;
    if (follow) nq = q + (target - q) * std::min(1.0, 12.0 * dt);
    dq = (nq - q) / dt;
    q = nq;
    // attitude: legacy instant flip, ramp, or hook file; after a successful WalkRun the OEM stands the robot up again
    std::array<float, 4> quat = {1, 0, 0, 0};
    double roll = 0, gyro_x = 0;
    if (flip_after_user_s >= 0 && fsm == 1000 && now - user_t0 > flip_after_user_s) {
      if (flip_t == 0) { flip_t = now; ev("FAULT INJECTION: robot tips over (roll 1.3 rad)"); }
      roll = 1.3;
    }
    if (flip_t > 0 && fsm == 1) roll = 1.3;
    if (ramp_d >= 0 && (fsm == 1000 || fsm == 1) && user_t0 > 0 && now - user_t0 > ramp_d) {
      double x = std::min(1.0, (now - user_t0 - ramp_d) / std::max(1e-3, ramp_t));
      if (fsm == 1000) { roll = std::max(roll, ramp_r * x); gyro_x = (x < 1.0) ? ramp_r / ramp_t : 0.0; ramp_last = roll; }
      else roll = std::max(roll, ramp_last);                 // passive: it stays where it fell
      if (!ramp_logged) { ramp_logged = true; ev("FAULT INJECTION: tilt ramp starts (final roll " + std::to_string(ramp_r) + ")"); }
    }
    if (hook_roll >= 0) { roll = hook_roll; gyro_x = hook_gyro; if (!hook_logged) { hook_logged = true; ev("FAULT INJECTION: tilt hook published (roll " + std::to_string(roll) + ", gyro " + std::to_string(gyro_x) + ")"); } } else hook_logged = false;
    if (g_drop_at.load() > 0 && now > g_drop_at.load()) { g_drop_at = 0; g_fsm = 1; ev("OEM drops into fsm 1 after the walkrun-drop delay"); }
    g_tilt = roll;
    if (roll != 0) quat = {(float)std::cos(roll / 2), (float)std::sin(roll / 2), 0, 0};
    msg.tick() = ++tick;
    msg.mode_machine() = 5;
    for (int i = 0; i < 29; i++) { msg.motor_state()[i].q() = (float)q[i]; msg.motor_state()[i].dq() = (float)dq[i]; msg.motor_state()[i].tau_est() = 0.f; }
    for (int i = 0; i < 4; i++) msg.imu_state().quaternion()[i] = quat[i];
    for (int i = 0; i < 3; i++) msg.imu_state().gyroscope()[i] = i == 0 ? (float)gyro_x : 0.f;
    unitree::common::REMOTE_DATA_RX rx;
    memset(rx.buff, 0, 40);
    rx.RF_RX.btn.value = keys;
    for (int i = 0; i < 40; i++) msg.wireless_remote()[i] = rx.buff[i];
    {  // test hook <dir>/mute: communication loss - rt/lowstate stops being published while the file exists (the RPC service keeps answering)
      static bool mute_logged = false; bool muted = access((g_dir + "/mute").c_str(), F_OK) == 0;
      if (muted != mute_logged) { mute_logged = muted; ev(muted ? "FAULT INJECTION: rt/lowstate muted (communication loss)" : "rt/lowstate resumed"); }
      if (!muted) pub.Write(msg);
    }
    p_low.Write(zero_cmd);
    if (!rival.empty() && (++iter % 10) == 0) { if (rival == "user_lowcmd") p_user.Write(rival_cmd); else p_arm.Write(rival_cmd); }
    next += dt;
    double d = next - mono();
    if (d > 0) pm::sleep_s(d); else next = mono();
  }
  char b[256];
  snprintf(b, sizeof b, "user_lowcmd frames=%ld stiff=%ld damp=%ld bad_crc=%ld final_fsm=%d", g_frames.load(), g_stiff.load(), g_damp.load(), g_badcrc.load(), g_fsm.load());
  ev(b);
  fflush(stdout);
  _exit(0);
}
