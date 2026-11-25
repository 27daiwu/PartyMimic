#include "pm/link.hpp"

#include <dds/dds.h>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/client/client.hpp>

#include <cstring>

namespace pm {

using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;

namespace {
// Own sport-service client: needs only unitree::robot::Client (present in every SDK2 release), not the SDK's LocoClient version.
class SportClient : public unitree::robot::Client {
 public:
  SportClient() : Client("sport", false) {}
  void Init() {
    SetApiVersion("1.0.0.0");
    UT_ROBOT_CLIENT_REG_API_NO_PROI(7001);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(7110);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(7111);
  }
  int32_t GetFsmId(int& fsm) {
    std::string data;
    int32_t r = Call(7001, "", data);
    fsm = -1;
    if (r == 0) {
      size_t k = data.find("\"data\"");
      if (k != std::string::npos) {
        k = data.find(':', k);
        if (k != std::string::npos) fsm = atoi(data.c_str() + k + 1);
      }
    }
    return r;
  }
  int32_t SwitchToUser() { std::string d; return Call(7110, "{\"data\":false}", d); }
  int32_t SwitchToInternal(int mode) { std::string d; return Call(7111, "{\"data\":" + std::to_string(mode) + "}", d); }
};
}  // namespace

struct Link::Impl {
  unitree::robot::ChannelSubscriberPtr<LowState> sub_state;
  unitree::robot::ChannelSubscriberPtr<LowCmd> sub_cmd;
  unitree::robot::ChannelPublisherPtr<LowCmd> pub;
  std::unique_ptr<SportClient> sport;
  LowCmd msg;
  std::mutex pub_mtx;
};

uint32_t lowcmd_crc(const LowCmd& m) {
  std::array<uint32_t, 251> w{};
  size_t k = 0;
  w[k++] = (uint32_t)m.mode_pr() | ((uint32_t)m.mode_machine() << 8);
  for (int i = 0; i < 35; i++) {
    const auto& c = m.motor_cmd()[i];
    auto f2u = [](float f) { uint32_t u; memcpy(&u, &f, 4); return u; };
    w[k++] = (uint32_t)c.mode();
    w[k++] = f2u(c.q()); w[k++] = f2u(c.dq()); w[k++] = f2u(c.tau()); w[k++] = f2u(c.kp()); w[k++] = f2u(c.kd());
    w[k++] = c.reserve();
  }
  for (int i = 0; i < 4; i++) w[k++] = m.reserve()[i];
  return crc32_words(w.data(), 250);
}

uint32_t Link::lowcmd_crc_packed(const std::array<uint32_t, 251>& w) { return crc32_words(w.data(), 250); }

Link::Link(const std::string& iface, int domain, const LinkOpts& o) : impl_(new Impl), iface_(iface), domain_(domain), o_(o) {
  unitree::robot::ChannelFactory::Instance()->Init(domain, iface);
  if (o.state) {
    impl_->sub_state.reset(new unitree::robot::ChannelSubscriber<LowState>("rt/lowstate"));
    impl_->sub_state->InitChannel(
        [this](const void* m) {
          const LowState& s = *static_cast<const LowState*>(m);
          auto st = std::make_shared<LinkState>();
          for (int i = 0; i < N; i++) {
            const auto& ms = s.motor_state()[i];
            st->q[i] = ms.q(); st->dq[i] = ms.dq(); st->tau[i] = ms.tau_est();
          }
          for (int i = 0; i < 4; i++) st->quat[i] = s.imu_state().quaternion()[i];
          for (int i = 0; i < 3; i++) { st->gyro[i] = s.imu_state().gyroscope()[i]; st->rpy[i] = s.imu_state().rpy()[i]; }
          st->tick = s.tick();
          st->mode_machine = s.mode_machine();
          for (int i = 0; i < 40; i++) st->wireless[i] = s.wireless_remote()[i];
          st->arrival = mono();
          {
            std::lock_guard<std::mutex> lk(mtx_);
            st->seq = ++rx_seq_;
            latest_ = st;
          }
          {
            std::lock_guard<std::mutex> lk(lis_mtx_);
            for (auto& fn : listeners_) fn(*st);
          }
        },
        1);
  }
  if (o.observe_cmd) {
    impl_->sub_cmd.reset(new unitree::robot::ChannelSubscriber<LowCmd>(o.cmd_topic));
    impl_->sub_cmd->InitChannel(
        [this](const void* m) {
          const LowCmd& c = *static_cast<const LowCmd*>(m);
          cmd_rx_++;
          for (int i = 0; i < N; i++)
            if (c.motor_cmd()[i].kp() > 0.0f) { active_rx_ = mono(); break; }
        },
        0);
  }
  if (o.rpc) {
    impl_->sport.reset(new SportClient());
    impl_->sport->SetTimeout((float)o.rpc_timeout_s);
    impl_->sport->Init();
  }
}
Link::~Link() = default;

StatePtr Link::latest() const {
  std::lock_guard<std::mutex> lk(mtx_);
  return latest_;
}
void Link::add_state_listener(std::function<void(const LinkState&)> fn) {
  std::lock_guard<std::mutex> lk(lis_mtx_);
  listeners_.push_back(std::move(fn));
}

void Link::enable_write() {
  std::lock_guard<std::mutex> lk(impl_->pub_mtx);
  if (impl_->pub) return;
  impl_->pub.reset(new unitree::robot::ChannelPublisher<LowCmd>(o_.cmd_topic));
  impl_->pub->InitChannel();
}

void Link::send(const Vec& q, const Vec& dq, const Vec& tau, const Vec& kp, const Vec& kd) {
  std::lock_guard<std::mutex> lk(impl_->pub_mtx);
  if (!impl_->pub) throw std::runtime_error("link: write not enabled");
  LowCmd& m = impl_->msg;
  for (int i = 0; i < N; i++) {
    auto& c = m.motor_cmd()[i];
    c.mode() = 1;
    c.q() = (float)q[i]; c.dq() = (float)dq[i]; c.tau() = (float)tau[i]; c.kp() = (float)kp[i]; c.kd() = (float)kd[i];
  }
  // slots 29..34 stay at their zero defaults (demo: mode 0, kp=kd=0)
  m.mode_pr() = 0;
  m.mode_machine() = 5;  // 29-dof G1 (demo main.cpp)
  m.crc() = lowcmd_crc(m);
  impl_->pub->Write(m);
}

std::pair<int, int> Link::get_fsm() {
  if (!impl_->sport) return {-1, -1};
  int fsm = -1;
  try {
    int r = impl_->sport->GetFsmId(fsm);
    return {r, r == 0 ? fsm : -1};
  } catch (...) { return {-1, -1}; }
}
int Link::switch_to_user() {
  if (!impl_->sport) return -1;
  try { return impl_->sport->SwitchToUser(); } catch (...) { return -1; }
}
int Link::switch_to_internal(const std::string& mode) {
  if (!impl_->sport) return -1;
  int m = mode == "LAST" ? 0 : mode == "PASSIVE" ? 1 : 2;
  try { return impl_->sport->SwitchToInternal(m); } catch (...) { return -1; }
}

std::map<std::string, uint64_t> Link::measure_traffic(double window_s, const std::vector<std::string>& topics) {
  std::map<std::string, uint64_t> out;
  std::vector<std::unique_ptr<std::atomic<uint64_t>>> counters;
  std::vector<unitree::robot::ChannelSubscriberPtr<LowCmd>> subs;
  for (auto& t : topics) {
    counters.emplace_back(new std::atomic<uint64_t>(0));
    std::atomic<uint64_t>* c = counters.back().get();
    auto sub = std::make_shared<unitree::robot::ChannelSubscriber<LowCmd>>(t);
    sub->InitChannel([c](const void*) { (*c)++; }, 0);
    subs.push_back(sub);
  }
  sleep_s(window_s);
  for (size_t i = 0; i < topics.size(); i++) out[topics[i]] = counters[i]->load();
  for (auto& s : subs) s->CloseChannel();
  return out;
}

Writers Link::discover_writers(double wait_s) {
  std::lock_guard<std::mutex> lk(wr_mtx_);
  Writers w;
  for (const char* t : {"rt/lowcmd", "rt/arm_sdk", "rt/user_lowcmd"}) w[t];
  dds_entity_t pp = dds_create_participant((dds_domainid_t)domain_, nullptr, nullptr);
  if (pp < 0) throw std::runtime_error("writer discovery: cannot create participant");
  dds_entity_t rd = dds_create_reader(pp, DDS_BUILTIN_TOPIC_DCPSPUBLICATION, nullptr, nullptr);
  if (rd < 0) { dds_delete(pp); throw std::runtime_error("writer discovery: cannot create builtin reader"); }
  sleep_s(wait_s);
  auto hex = [](const dds_guid_t& g) {
    char b[40];
    for (int i = 0; i < 16; i++) snprintf(b + 2 * i, 3, "%02x", g.v[i]);
    return std::string(b);
  };
  void* samples[64] = {nullptr};
  dds_sample_info_t info[64];
  for (;;) {
    int n = dds_take(rd, samples, info, 64, 64);
    if (n <= 0) break;
    for (int i = 0; i < n; i++) {
      if (!info[i].valid_data || info[i].instance_state != DDS_ALIVE_INSTANCE_STATE) continue;
      auto* ep = static_cast<dds_builtintopic_endpoint_t*>(samples[i]);
      auto it = w.find(ep->topic_name);
      if (it != w.end()) it->second.push_back(hex(ep->participant_key) + "/" + hex(ep->key));
    }
    dds_return_loan(rd, samples, n);
  }
  dds_delete(pp);
  return w;
}

}  // namespace pm
