// DEV ONLY (never shipped): DDS entity lifecycle stress test for pm::Link (V1.5.1).
// Reproduces the CHECKING-phase crash of the first real-robot trigger: CycloneDDS-C++ EntityDelegate::prevent_callbacks() `assert(false)` (SIGABRT, rc 134) fires when a
// DataReader that has a data_available listener is destroyed while a callback is in flight (the callback thread then holds the LAST reference and runs the destructor itself).
// usage: dds_stress [--iters N] [--window S] [--domain 77] [--iface lo] [--link-cycles N] [--mechanism US]      needs dev/sim_robot running on the same domain (it is the hot rt/lowcmd + rt/lowstate publisher)
// Exercises the SAME calls the runner makes during CHECKING / GO re-check: Link::measure_traffic + Link::discover_writers (+ whole Link create/destroy cycles with --link-cycles).
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "pm/link.hpp"
#include "pm/util.hpp"

int main(int argc, char** argv) {
  int iters = 200, domain = 77;
  double window = 0.05;
  std::string iface = "lo";
  int link_cycles = 0, mech = 0;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto nx = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--iters") iters = atoi(nx().c_str());
    else if (a == "--window") window = atof(nx().c_str());
    else if (a == "--domain") domain = atoi(nx().c_str());
    else if (a == "--iface") iface = nx();
    else if (a == "--link-cycles") link_cycles = atoi(nx().c_str());
    else if (a == "--mechanism") mech = atoi(nx().c_str());
  }
  if (mech > 0) {  // ROOT-CAUSE DEMONSTRATION (not what the runner does any more): create + destroy a subscriber with a listener while data_available callbacks are in flight.
    // A callback that takes ~`mech` us (a loaded robot computer) makes the race window large; with the default fast callback it is hit rarely (that is why it passed on the bench before).
    unitree::robot::ChannelFactory::Instance()->Init(domain, iface);
    for (int k = 0; k < iters; k++) {
      auto sub = std::make_shared<unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowCmd_>>("rt/lowcmd");
      sub->InitChannel([mech](const void*) { std::this_thread::sleep_for(std::chrono::microseconds(mech)); }, 0);
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
      sub->CloseChannel();  // == the old Link::measure_traffic() teardown
      if (k % 20 == 0) { printf("mechanism iter %d\n", k); fflush(stdout); }
    }
    printf("MECHANISM: no crash in %d create/destroy cycles\n", iters);
    _exit(0);
  }
  pm::LinkOpts lo;
  lo.rpc = false; lo.observe_cmd = true; lo.state = true; lo.monitor_traffic = true;  // state=true: a hot rt/lowstate listener exists next to the traffic monitors (like the runner)
  if (link_cycles > 0) {  // whole-Link create / close / destroy cycles against hot traffic (Link::~Link parks the entities, it must never abort)
    for (int c = 0; c < link_cycles; c++) {
      pm::Link l(iface, domain, lo);
      pm::sleep_s(0.15 + 0.01 * (c % 7));
      auto t = l.measure_traffic(0.1);
      auto wr = l.discover_writers();
      if (c % 10 == 0) { printf("link cycle %d: lowcmd %llu writers %zu\n", c, (unsigned long long)t["rt/lowcmd"], wr["rt/lowcmd"].size()); fflush(stdout); }
    }
    printf("LINK CYCLES OK: %d\n", link_cycles);
    fflush(stdout);
  }
  pm::Link link(iface, domain, lo);
  uint64_t tot_low = 0;
  double t0 = pm::mono();
  for (int k = 0; k < iters; k++) {
    std::map<std::string, uint64_t> rx;
    pm::Writers w;
    std::thread t([&] { rx = link.measure_traffic(window); });  // the runner scans in a helper thread while the main thread runs the other checks
    w = link.discover_writers(std::min(0.05, window));
    t.join();
    tot_low += rx["rt/lowcmd"];
    if (k % 50 == 0) { printf("iter %d: rt/lowcmd rx=%llu writers(lowcmd)=%zu\n", k, (unsigned long long)rx["rt/lowcmd"], w["rt/lowcmd"].size()); fflush(stdout); }
  }
  printf("STRESS OK: %d scans in %.1f s, total rt/lowcmd samples seen %llu\n", iters, pm::mono() - t0, (unsigned long long)tot_low);
  pm::WindowStats ws = link.window(1.0);
  printf("rolling window(1.0 s): covered=%d span=%.3f lowstate %.0f Hz distinct_ticks=%u lowcmd=%llu arm=%llu user=%llu\n", (int)ws.covered, ws.span_s, ws.ls_hz, ws.distinct_ticks,
         (unsigned long long)ws.topic_rx[0], (unsigned long long)ws.topic_rx[1], (unsigned long long)ws.topic_rx[2]);
  fflush(stdout);
  _exit(0);
}
