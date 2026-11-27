#include "pm/link.hpp"

#include <dds/dds.h>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/client/client.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

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
  unitree::robot::ChannelSubscriberPtr<LowCmd> sub_mon[3];  // V1.5.1 persistent traffic monitors: rt/lowcmd, rt/arm_sdk, rt/user_lowcmd
  dds_entity_t disc_pp = 0, disc_rd = 0;                    // V1.5.1 persistent built-in discovery reader (C API)
  unitree::robot::ChannelPublisherPtr<LowCmd> pub;
  std::unique_ptr<SportClient> sport;
  LowCmd msg;
  std::mutex pub_mtx;
};

namespace {
const char* const MON_TOPICS[3] = {"rt/lowcmd", "rt/arm_sdk", "rt/user_lowcmd"};
// Process-lifetime graveyard of DDS entities. Nothing parked here is ever destroyed (see the lifecycle rule in link.hpp); every process of this program ends with _exit().
std::mutex g_grave_m;
std::vector<std::shared_ptr<void>>& graveyard() { static auto* g = new std::vector<std::shared_ptr<void>>(); return *g; }
}  // namespace

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

void Link::set_rpc_observer(RpcObserver fn) {
  std::lock_guard<std::mutex> lk(rpc_mtx_);
  rpc_obs_ = std::move(fn);
}
void Link::report_rpc(const RpcEvent& e) {
  RpcObserver f;
  { std::lock_guard<std::mutex> lk(rpc_mtx_); f = rpc_obs_; }
  if (f) { try { f(e); } catch (...) {} }
}

Link::Link(const std::string& iface, int domain, const LinkOpts& o) : core_(std::make_shared<Core>()), impl_(std::make_shared<Impl>()), iface_(iface), domain_(domain), o_(o) {
  unitree::robot::ChannelFactory::Instance()->Init(domain, iface);
  std::shared_ptr<Core> core = core_;  // the callbacks own the Core, never the Link (the Link object may be gone while a callback is still running)
  if (o.state) {
    impl_->sub_state.reset(new unitree::robot::ChannelSubscriber<LowState>("rt/lowstate"));
    impl_->sub_state->InitChannel(
        [core](const void* m) {
          if (core->closed.load(std::memory_order_relaxed)) return;
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
            std::lock_guard<std::mutex> lk(core->mtx);
            st->seq = ++core->rx_seq;
            core->latest = st;
          }
          {
            std::lock_guard<std::mutex> lk(core->lis_mtx);
            for (auto& fn : core->listeners) fn(*st);
          }
        },
        1);
  }
  if (o.observe_cmd) {
    impl_->sub_cmd.reset(new unitree::robot::ChannelSubscriber<LowCmd>(o.cmd_topic));
    impl_->sub_cmd->InitChannel(
        [core](const void* m) {
          if (core->closed.load(std::memory_order_relaxed)) return;
          const LowCmd& c = *static_cast<const LowCmd*>(m);
          uint64_t k = ++core->cmd_rx;  // the k-th sample seen on the command topic; our own samples are echoed to our reader: latency = rx time - send time of the (k - base)-th send
          {
            uint64_t base = core->rx_base.load(), ns = core->n_sent.load();
            if (k > base && k - base <= ns && ns - (k - base) < core->send_t.size()) {
              double lat = (mono() - core->send_t[(k - base - 1) % core->send_t.size()].load()) * 1e3;
              if (lat >= 0 && lat < 1000) { core->loop_ms = lat; if (lat > core->loop_max_ms.load()) core->loop_max_ms = lat; }
            }
          }
          for (int i = 0; i < N; i++)
            if (c.motor_cmd()[i].kp() > 0.0f) { core->active_rx = mono(); break; }
        },
        0);
  }
  if (o.rpc) {
    impl_->sport.reset(new SportClient());
    impl_->sport->SetTimeout((float)o.rpc_timeout_s);
    impl_->sport->Init();
  }
  if (o.monitor_traffic) ensure_monitors();  // at start-up: the readers match the factory writers while the model loads
}

// V1.5.1: end of life. The callbacks are neutralised and EVERY DDS entity (readers, writer, RPC client, discovery reader) is parked in a process-lifetime graveyard instead of
// being destroyed: destroying a reader with a data_available listener while a callback is in flight aborts the process inside CycloneDDS-C++ (see link.hpp).
void Link::close() {
  if (closed_.exchange(true)) return;
  core_->closed = true;
  std::lock_guard<std::mutex> lk(g_grave_m);
  graveyard().push_back(std::shared_ptr<void>(impl_, impl_.get()));
}
Link::~Link() { close(); }

StatePtr Link::latest() const {
  std::lock_guard<std::mutex> lk(core_->mtx);
  return core_->latest;
}
void Link::add_state_listener(std::function<void(const LinkState&)> fn) {
  std::lock_guard<std::mutex> lk(core_->lis_mtx);
  core_->listeners.push_back(std::move(fn));
}

void Link::enable_write() {
  std::lock_guard<std::mutex> lk(impl_->pub_mtx);
  if (impl_->pub) return;
  impl_->pub.reset(new unitree::robot::ChannelPublisher<LowCmd>(o_.cmd_topic));
  impl_->pub->InitChannel();
  core_->rx_base = core_->cmd_rx.load();  // samples seen before our first write are not ours
  core_->n_sent = 0;
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
  double t0 = mono();
  uint64_t idx = core_->n_sent.load();
  core_->send_t[idx % core_->send_t.size()].store(t0);
  core_->n_sent = idx + 1;
  impl_->pub->Write(m);
  core_->last_send_us = (mono() - t0) * 1e6;
}

std::pair<int, int> Link::get_fsm() {
  if (!impl_->sport) return {-1, -1};
  int fsm = -1;
  RpcEvent e;
  e.api = "GetFsmId"; e.t_req = mono();
  try {
    int r = impl_->sport->GetFsmId(fsm);
    e.t_resp = mono(); e.rc = r; e.fsm = r == 0 ? fsm : -1;
    report_rpc(e);
    return {r, r == 0 ? fsm : -1};
  } catch (const std::exception& x) { e.t_resp = mono(); e.exception = x.what(); report_rpc(e); return {-1, -1}; }
  catch (...) { e.t_resp = mono(); e.exception = "unknown"; report_rpc(e); return {-1, -1}; }
}
int Link::switch_to_user() {
  if (!impl_->sport) return -1;
  RpcEvent e;
  e.api = "SwitchToUserCtrl"; e.t_req = mono();
  try { int r = impl_->sport->SwitchToUser(); e.t_resp = mono(); e.rc = r; report_rpc(e); return r; }
  catch (const std::exception& x) { e.t_resp = mono(); e.exception = x.what(); report_rpc(e); return -1; }
  catch (...) { e.t_resp = mono(); e.exception = "unknown"; report_rpc(e); return -1; }
}
int Link::switch_to_internal(const std::string& mode) {
  if (!impl_->sport) return -1;
  int m = mode == "LAST" ? 0 : mode == "PASSIVE" ? 1 : 2;
  RpcEvent e;
  e.api = "SwitchToInternalCtrl"; e.arg = mode; e.t_req = mono();
  try { int r = impl_->sport->SwitchToInternal(m); e.t_resp = mono(); e.rc = r; report_rpc(e); return r; }
  catch (const std::exception& x) { e.t_resp = mono(); e.exception = x.what(); report_rpc(e); return -1; }
  catch (...) { e.t_resp = mono(); e.exception = "unknown"; report_rpc(e); return -1; }
}

// V1.5.1: persistent monitors. Created once (Link constructor with monitor_traffic, or on first use), never destroyed. The reader callbacks only bump an atomic counter.
void Link::ensure_monitors() {
  std::lock_guard<std::mutex> lk(wr_mtx_);
  if (core_->monitors_t0.load() != 0) return;
  std::shared_ptr<Core> core = core_;
  for (int i = 0; i < 3; i++) {
    impl_->sub_mon[i].reset(new unitree::robot::ChannelSubscriber<LowCmd>(MON_TOPICS[i]));
    impl_->sub_mon[i]->InitChannel([core, i](const void*) { if (!core->closed.load(std::memory_order_relaxed)) core->topic_rx[i].fetch_add(1, std::memory_order_relaxed); }, 0);
  }
  impl_->disc_pp = dds_create_participant((dds_domainid_t)domain_, nullptr, nullptr);
  if (impl_->disc_pp < 0) { impl_->disc_pp = 0; throw std::runtime_error("writer discovery: cannot create participant"); }
  impl_->disc_rd = dds_create_reader(impl_->disc_pp, DDS_BUILTIN_TOPIC_DCPSPUBLICATION, nullptr, nullptr);
  if (impl_->disc_rd < 0) { impl_->disc_rd = 0; throw std::runtime_error("writer discovery: cannot create builtin reader"); }
  core->monitors_t0 = mono();
  start_sampler();
}

void Link::start_sampler() {
  if (core_->sampler_started.exchange(true)) return;
  std::shared_ptr<Core> core = core_;
  std::thread([core]() {  // detached; owns the Core; stops when the Link is closed (or the process ends)
    while (!core->closed.load()) {
      Core::Snap sn{};
      sn.t = mono();
      {
        std::lock_guard<std::mutex> lk(core->mtx);
        if (core->latest) { sn.ls_seq = core->latest->seq; sn.ls_tick = core->latest->tick; }
      }
      sn.cmd_rx = core->cmd_rx.load();
      for (int i = 0; i < 3; i++) sn.topic[i] = core->topic_rx[i].load();
      {
        std::lock_guard<std::mutex> lk(core->hist_mtx);
        core->hist[core->hist_head] = sn;
        core->hist_head = (core->hist_head + 1) % Core::HIST_N;
        core->hist_n = std::min(core->hist_n + 1, Core::HIST_N);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }).detach();
}

bool Link::monitors_ready() const {
  double t0 = core_->monitors_t0.load();
  return t0 != 0 && mono() - t0 >= 0.5;
}
double Link::monitor_age_s() const {
  double t0 = core_->monitors_t0.load();
  return t0 == 0 ? -1 : mono() - t0;
}

WindowStats Link::window(double w) const {
  WindowStats r;
  const double now = mono(), from = now - w;
  const double t0 = core_->monitors_t0.load();
  std::lock_guard<std::mutex> lk(core_->hist_mtx);
  if (core_->hist_n < 2) return r;
  auto at = [&](size_t k) -> const Core::Snap& { return core_->hist[(core_->hist_head + Core::HIST_N - 1 - k) % Core::HIST_N]; };  // k = 0: newest
  const Core::Snap& newest = at(0);
  size_t kb = 0;
  while (kb + 1 < core_->hist_n && at(kb + 1).t >= from) kb++;
  const Core::Snap& base = at(kb);
  const bool hist_long_enough = base.t - from <= 0.03 && (kb + 1 < core_->hist_n || base.t - from <= 0.012);
  r.covered = t0 != 0 && hist_long_enough && from >= t0 + 0.5 - 0.03;  // monitors must have been matched for the whole window
  r.span_s = newest.t - base.t;
  r.ls_rx = newest.ls_seq - base.ls_seq;
  r.ls_hz = r.span_s > 0 ? (double)r.ls_rx / r.span_s : 0;
  r.user_cmd = newest.cmd_rx - base.cmd_rx;
  for (int i = 0; i < 3; i++) r.topic_rx[i] = newest.topic[i] - base.topic[i];
  uint32_t last = base.ls_tick, nd = 1;
  for (size_t k = kb; k-- > 0;) { if (at(k).ls_tick != last) { nd++; last = at(k).ls_tick; } }
  r.distinct_ticks = nd;
  return r;
}

std::map<std::string, uint64_t> Link::measure_traffic(double window_s, const std::vector<std::string>&) {
  ensure_monitors();
  const double ready_t = core_->monitors_t0.load() + 0.5 + window_s + 0.02;  // monitors matched + one full window of history
  const double wait = ready_t - mono();
  if (wait > 0) sleep_s(wait);
  WindowStats w = window(window_s);
  std::map<std::string, uint64_t> out;
  for (int i = 0; i < 3; i++) out[MON_TOPICS[i]] = w.topic_rx[i];
  return out;
}

Writers Link::discover_writers(double wait_s) {
  Writers w;
  for (const char* t : MON_TOPICS) w[t];
  const bool fresh = core_->monitors_t0.load() == 0;
  if (fresh) {  // first use without start-up monitors: create (persistent) and give discovery its grace period
    ensure_monitors();
    sleep_s(wait_s);
  }
  auto hex = [](const dds_guid_t& g) {
    char b[40];
    for (int i = 0; i < 16; i++) snprintf(b + 2 * i, 3, "%02x", g.v[i]);
    return std::string(b);
  };
  constexpr int MAXS = 1024;
  static thread_local void* samples[MAXS];
  static thread_local dds_sample_info_t info[MAXS];
  samples[0] = nullptr;  // loan
  int n = dds_read(impl_->disc_rd, samples, info, MAXS, MAXS);  // READ, not take: the built-in reader keeps describing every currently alive endpoint
  for (int i = 0; i < n; i++) {
    if (!info[i].valid_data || info[i].instance_state != DDS_ALIVE_INSTANCE_STATE) continue;
    auto* ep = static_cast<dds_builtintopic_endpoint_t*>(samples[i]);
    auto it = w.find(ep->topic_name);
    if (it != w.end()) it->second.push_back(hex(ep->participant_key) + "/" + hex(ep->key));
  }
  if (n > 0) dds_return_loan(impl_->disc_rd, samples, n);
  return w;
}

}  // namespace pm
